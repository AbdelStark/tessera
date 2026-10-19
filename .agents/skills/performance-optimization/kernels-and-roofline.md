# Tessera Hot Kernels & Roofline Reference

Hardware and algebraic reference for optimizing Tessera's six critical kernel families ([08-performance-budget.md](../../../docs/spec/08-performance-budget.md), [RFC-0002](../../../docs/rfcs/RFC-0002-field-arithmetic.md), [RFC-0011](../../../docs/rfcs/RFC-0011-advice-commitment.md), [RFC-0010](../../../docs/rfcs/RFC-0010-proof-schedule.md), [RFC-0008](../../../docs/rfcs/RFC-0008-logup-and-tables.md), [RFC-0018](../../../docs/rfcs/RFC-0018-gpu-prover-runtime.md)).

## 1. Goldilocks Field (`p = 2^64 - 2^32 + 1`) & Quartic Extension `Fp4`

### Algebraic Structure (`RFC-0002`)
- **Prime**: $p = 2^{64} - 2^{32} + 1 = \texttt{0xFFFF\_FFFF\_0000\_0001}$.
- **Key reduction identity**: $2^{64} \equiv 2^{32} - 1 \pmod p$, and $2^{96} \equiv -1 \pmod p$.
- **128-bit reduction (`reduce128`)**: decompose a 128-bit product $x = x_{\text{hi}} \cdot 2^{64} + x_{\text{lo}}$ into three 32-bit/64-bit parts:
  - $x_{\text{lo}} \in [0, 2^{64})$, $x_{\text{mid}} = x_{\text{hi}} \bmod 2^{32}$, $x_{\text{top}} = x_{\text{hi}} \gg 32$.
  - Because $2^{64} \equiv 2^{32} - 1 \pmod p$ and $2^{96} \equiv -1 \pmod p$:
    $$x \equiv x_{\text{lo}} + (x_{\text{mid}} \ll 32) - x_{\text{mid}} - x_{\text{top}} \pmod p.$$
- **Canonical representation invariant (`INV-FIELD-1`)**: every public `Goldilocks` value must be in canonical form $[0, p)$ when serialized, compared for equality, or converted via `.to_canonical_u64()`. Internal lazy representations in registers `[0, 2^64)` are allowed inside tight loops provided canonicalization happens before observation.
- **Quartic extension `Fp4`**: $\mathbb{F}_p[X]/(X^4 - 7)$ (`W = 7`). Multiplication in `Fp4` uses Karatsuba / Toom-Cook over quadratic towers $(a_0 + a_2 X^2) + X(a_1 + a_3 X^2)$ with $(X^2)^2 = 7$, reducing 16 base-field multiplications to 9 base-field multiplications.

### Common Pitfalls
- **Underflow/overflow in `reduce128`**: subtracting $x_{\text{top}}$ or adding $(x_{\text{mid}} \ll 32) - x_{\text{mid}}$ can wrap around $2^{64}$. On overflow, add $2^{64} \equiv 2^{32} - 1 = \texttt{0xFFFF\_FFFF}$; on underflow, subtract $\texttt{0xFFFF\_FFFF}$. Always test `reduce128((p - 1) as u128 * (p - 1) as u128)` and `reduce128(u64::MAX as u128 * u64::MAX as u128)`.
- **Non-canonical `0xFFFF_FFFF_0000_0001..=0xFFFF_FFFF_FFFF_FFFF`**: a lazy representation in $[p, 2^{64})$ that escapes into `Eq`, `Hash`, or TCE encoding violates `INV-FIELD-1` and `INV-TCE-1`.

## 2. Sumcheck Round Prover (`tessera-sumcheck`, `RFC-0010`)

### Roofline Profile
- **Operation**: given $k$ multilinear tables $A_1, \dots, A_k$ of size $2^{v}$, evaluate the univariate round polynomial $s_i(t) = \sum_{b \in \{0,1\}^{v-1}} g(A_1(t, b), \dots, A_k(t, b))$ at $t \in \{0, 1, \dots, d\}$, then fold each table in place: $A_j'(b) = A_j(0, b) + r_i \cdot (A_j(1, b) - A_j(0, b))$.
- **Arithmetic intensity**: low ($O(d)$ field ops per 16 bytes read). **Memory-bandwidth bound** for $2^v$ exceeding L3 cache; **L1/L2 cache bound** in later rounds ($v \le 14$).

### Optimization Patterns
1. **Fused fold-and-evaluate**: after receiving challenge $r_{i-1}$, fold $A_j$ for round $i-1$ and immediately accumulate the round-$i$ evaluations $s_i(t)$ in a single pass over memory, cutting memory traffic in half.
2. **Difference-step evaluation**: for a linear interpolation $A(t, b) = A(0, b) + t \cdot \Delta(b)$ where $\Delta(b) = A(1, b) - A_j(0, b)$, compute $A(2, b) = A(1, b) + \Delta(b)$, $A(3, b) = A(2, b) + \Delta(b)$ via additions instead of multiplications, and evaluate the asymptotic point $\Delta(b)$ (`t = \infty` / leading coefficient) directly.
3. **Base-field vs extension-field split**: when the witness table $A_j$ holds base-field `Goldilocks` values and the challenge $r_1 \in \mathbb{F}_{p^4}$ enters in round 1:
   - Round 1 evaluates products of base-field elements (1x bandwidth, 1x mul cost) and only lifts the final univariate sums to $\mathbb{F}_{p^4}$.
   - After folding with $r_1 \in \mathbb{F}_{p^4}$, the table becomes $\mathbb{F}_{p^4}$ (4x bandwidth, 9x mul cost), and its length halves to $2^{v-1}$.

## 3. LogUp & Fractional GKR (`tessera-logup`, `RFC-0008`)

### Roofline Profile
- **Operation**: initialize leaves $p_i / q_i$ where $q_i = \gamma - \text{key}_i$ (or batched across tags), then propagate the binary tree of projective fractions:
  $$P_{\text{parent}} = P_L \cdot Q_R + P_R \cdot Q_L, \qquad Q_{\text{parent}} = Q_L \cdot Q_R.$$
- **Hotspots**:
  1. Leaf initialization and witness generation (computing all layer numerators and denominators).
  2. Per-layer GKR sumcheck over the degree-3 relation $\sum_{x} \widetilde{eq}(r, x) \cdot (\text{num}(x) + \lambda \cdot \text{den}(x))$.

### Optimization Patterns
1. **Zero-multiplicity skipping in leaf generation**: table entries with multiplicity $m_i = 0$ have $P_i = 0$, simplifying $P_{\text{parent}} = P_R \cdot Q_L$ when $P_L = 0$.
2. **Contiguous layer arena**: allocate a single flat buffer of size $2 \cdot 2^L$ for all tree layers rather than $L$ separate `Vec` allocations per column.
3. **Factored `eq(r, x)` tables**: when evaluating $\widetilde{eq}(r, x)$ over $x \in \{0, 1\}^\ell$, precompute the hypercube table via doubling ($2^\ell$ additions + $2^\ell$ multiplications total) rather than $O(\ell \cdot 2^\ell)$ point-wise products.

## 4. Ligero PCS & NTT (`tessera-pcs`, `RFC-0011`, `RFC-0012`)

### Roofline Profile
- **Operation**: arrange witness of size $N = m \times k$ into an $m$-row by $k$-column matrix, pad with $b$ blinding columns (`k' = k + b`), encode each row with a Reed–Solomon code of length $n = \rho^{-1} k'$ via radix-2 NTT over Goldilocks, transpose or hash columns into $n$ Blake3 Merkle leaves, and answer linear row combinations + $t$ opened columns.
- **Hotspots**:
  1. $m$ independent length-$n$ NTTs (compute + cache-strided butterfly access).
  2. Column hashing into Blake3 leaves (memory-transpose + hash compute bound).

### Optimization Patterns
1. **Six-step / four-step cache-oblivious NTT**: for row length $n$ exceeding L1 cache, process multiple rows together in cache-resident tiles, or use Stockham / Gentleman-Sande butterflies with precomputed twiddle tables in bit-reversed order so twiddles are read sequentially per stage.
2. **Tiled matrix transpose before column hashing**: never read column $j$ with stride $n$ across $m$ rows directly into Blake3 when $m \times n$ exceeds L2 cache. Transpose $64 \times 64$ (or $8 \times 8$ cache-line) tiles into contiguous column buffers and hash 4–8 columns in parallel.
3. **Precomputed twiddle powers**: cache the roots of unity $\omega^{0}, \omega^{1}, \dots, \omega^{n/2 - 1}$ once per domain size $\log_2 n$.

## 5. Poseidon2 (`tessera-hash`, `RFC-0004`)

### Roofline Profile
- **Parameters**: state width $t = 12$, rate $r = 8$, capacity $c = 4$, S-box $x \mapsto x^7$, $R_F = 8$ external rounds (4 initial + 4 terminal), $R_P = 22$ internal rounds.
- **Hotspot**: the 22 internal rounds, which apply the S-box only to `state[0]` and multiply by the internal diffusion matrix $M_I = \mathbf{1}\mathbf{1}^T + \text{diag}(V)$.

### Optimization Patterns
1. **S-box addition chain ($x^7$)**: compute $x^2 = x \cdot x$, $x^3 = x^2 \cdot x$, $x^4 = x^2 \cdot x^2$, $x^7 = x^4 \cdot x^3$ (4 multiplications, depth 3, with $x^3$ and $x^4$ independent in superscalar pipelines).
2. **Internal round sum reuse**: compute $\Sigma = \sum_{i=0}^{11} \text{state}[i]$ once per internal round, and update $\text{state}[i] \leftarrow \Sigma + V_i \cdot \text{state}[i]$. When $V_i$ is a small power of two or shift (per `RFC-0004` constants), replace general 64x64 multiplication with shifts and additions where specified.
3. **Keep all 12 state words in general-purpose registers**: unroll the internal round loop so `state[0..12]` stay in 12 CPU registers (`x0..x11` on `aarch64`, `r8..r15, rax..rdx` on `x86_64`) with zero stack spills.

## 6. GPU Kernels (`tessera-gpu`, `RFC-0018`)

- **Coalesced global memory access**: threads in a 32-thread warp (`threadIdx.x % 32`) must read and write consecutive 8-byte `Goldilocks` elements (`u64`) or 16-byte `ulong2` pairs.
- **Shared memory bank conflicts**: when staging NTT butterflies or matrix transposes in `__shared__` memory (32 banks of 4 bytes or 8 bytes), pad the inner dimension by `+1` element (`shared[TILE][TILE + 1]`) so column strides are coprime to 32.
- **CPU fallback parity**: every `tessera-gpu` entry point must have an identical signature and bit-exact result to the CPU implementation so tests pass on machines without CUDA hardware.
