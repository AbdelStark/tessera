# RFC-0008: LogUp-GKR Lookups and the Deterministic Table Registry

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.1 (argument), v0.2 (registry)

## Summary

Every non-arithmetic relation in Tessera reduces to lookups `(key, value) ∈ T` proven with LogUp-GKR: one fraction tree per table instance, batched across all layers, heads and tokens, with challenges in `Fp2`. Keys are virtual, values and multiplicities are committed. Tables are public, defined by integer-only generators whose outputs are bit-identical across the prover, the verifier and the Python reference, and pinned by digests in the claim list. Large tables (up to 2^32 rows) are cheap for the verifier because every v1 table is a piecewise function whose multilinear extension is evaluated in time proportional to its non-constant core, at most 2^24 entries.

## Motivation

Lookups are half the cost of a proof (source §5: 40 ps per lookup against 30 ps per committed element) and every uniqueness argument leans on them: the exponential lookup proves `c_q ≥ x_qi` (source §7, "Softmax by max certificate"); requantization tables absorb rescales (source §6, principle 4). The source specifies LogUp-GKR with one instance per table batched across layers and degree-2 extension challenges, bringing the LogUp error from VerInf's 2^-28 to about 2^-90 (source §7, "Lookups everywhere are LogUp-GKR"). Tables must also be generated deterministically: a table built with a platform's floating-point `exp` would differ across machines, and the verifier, which regenerates every table, would reject honest proofs.

## Goals

- A LogUp-GKR prover and an independent verifier with soundness error `(N + T)/|Fp2|` plus sumcheck terms.
- Batching of all lookups into a table, across every layer, into one instance.
- A registry of table kinds sufficient for `D`, scoring and the anchor circuits.
- Integer-only generators with pinned digests.
- Verifier table evaluation in `O(n · core)` time.

## Non-Goals

- One-hot or decomposable-table arguments (Lasso, Shout).
- LogUp micro-architecture optimizations beyond the v1 kernels (NG-6).
- Tables whose contents are committed rather than public. `D`'s embedding uses a one-hot matmul instead (RFC-0006).

## Proposed Design

### Relation

For a table instance with rows `t_0 … t_{T−1}` and lookups `a_0 … a_{N−1}`, each a tuple compressed to one `Fp2` element with challenge `β` (`a = key + β·value`, `t = key_t + β·value_t`), the prover commits multiplicities `m_j` and proves

```math
\sum_{i<N} \frac{1}{\alpha - a_i} \;-\; \sum_{j<T} \frac{m_j}{\alpha - t_j} \;=\; 0
```

for a challenge `α ∈ Fp2` drawn after the values and multiplicities are committed. If some `a_i` is not a row, the identity holds for at most `N + T − 1` values of `α`, so the error is at most `(N + T)/|Fp2|`.

### Fraction tree

- Leaves: lookups contribute `(1, α − a_i)`; table rows contribute `(−m_j, α − t_j)`. Both sides share one tree (combined), padded to a power of two by lookups that repeat row 0 with `m_0` increased by the padding count.
- Internal nodes: `(n, d) = (n_L d_R + n_R d_L, d_L d_R)`.
- Claim: the root has `n = 0` and `d ≠ 0`. The verifier checks `d_root ≠ 0`.
- Each tree level is proven by a sumcheck of degree 3 per variable (Papini and Haböck), reducing a claim on level `ℓ` to claims on level `ℓ + 1`.
- Final claims land on the leaf polynomials: the lookup-side key polynomial (virtual, handed to the stage that produces the key, RFC-0010), the value polynomial (committed advice), the multiplicity polynomial (committed advice), and the table polynomial `t̃`, which the verifier evaluates itself.

### Batching

- All lookups into one table definition, across every layer, head, token and channel, form one instance.
- All instances run their GKR layers in one batched sumcheck per level with a random combination; trees of different heights join the batch at the level where their height matches (as in Stwo's LogUp-GKR batching).
- `α` and `β` are shared by all instances and derived from the transcript immediately after Round 1 (RFC-0014, label `logup/alpha`, `logup/beta`).

### Table kinds

| Kind | Key | Value | Rows | Used by |
|---|---|---|---|---|
| `Range { bits }` | `x ∈ [0, 2^bits)` | none | `2^bits` | Split remainders, limbs, brackets |
| `Requant { b, classes, r_by_class, lo, hi }` | `(class, k)`, `k ∈ [−2^{b−1}, 2^{b−1})` | `clamp(floor((k + 2^{r−1}) / 2^r), lo, hi)` | `|classes| · 2^b` | `Requant1` (RFC-0009) |
| `Silu { b, classes, r_by_class, out_exp }` | `(class, k)` | `clamp(round(silu(h · σ_h) / σ_out), −128, 127)`, `h = k >> r` | `|classes| · 2^b` | MLP |
| `Exp { key_bits, sigma, scale }` | `k ∈ [0, 2^{key_bits})` | `round(2^σ · exp(−k · scale))` | `2^{key_bits}` | Softmax, vocabulary softmax |
| `Log2Frac { bits, frac_bits }` | `x ∈ [0, 2^bits)` | one-sided bound on `log2` (RFC-0017) | `2^bits` | Scoring |
| `Bitwise8 { op }` | `(a, b)`, 8-bit each | `a ⊕ b`, `a ∧ b` | 2^16 | SHA-256 |
| `AesSbox`, `Gf256Mul { c }` | `x`, 8-bit | S-box, `c · x` in GF(2^8) | 2^8 | AES-128 |
| `Rope { theta, scaling, bits, max_context, d_head }` | `(position, frequency)` | `(C, S)`, 14-bit | `max_context · d_head / 2` | Public constants for `Rotate`, not a lookup table |

Signed keys are offset-encoded (`k + 2^{b−1}`); the compressed key adds `2^b · class` so tuples pack into one integer below `2^{B_max}`.

### Structured evaluation

The verifier needs `t̃(ρ)` at the random point where the table side of the fraction tree ends. With `t = key + β·value`:

- `key` is an affine function of the row index bits (offset encoding and class packing), so `kẽy(ρ)` is a closed-form `O(n)` expression.
- For `Requant`, `Silu` and `Exp`, `value(x)` depends only on the high bits `h = x >> r`, so `valũe(ρ) = f̃(ρ_high)`: the low variables sum out because `Σ_{x_low} eq(ρ_low, x_low) = 1`.
- Every `f` is piecewise: constant below `a`, a dense core on `[a, c)` of at most 2^24 explicit values, and constant or affine on `[c, 2^{n_h})`. Then `f̃(ρ) = A · LT(ρ, a) + Σ_{h=a}^{c−1} f(h) eq(ρ, h) + Σ_{h ≥ c} eq(ρ, h)(u·h + w)`, where `LT(ρ, a) = Σ_{h<a} eq(ρ, h)` and the weighted suffix sums have `O(n)` bit-recursive forms. The dense middle term costs `O(core)` with Gray-code updates of `eq`.
- `Range`, `Bitwise8`, `AesSbox` and `Gf256Mul` are small enough to evaluate by enumeration.

Verifier cost per table: `O(n · core)`, at most a few times 2^24 field operations; independent of the 2^32 row count the prover enumerates.

### Generators and digests

- Generators use integers only: no floating point, no platform math library, no big-float dependency.
- `Exp`, `Silu` and `Rope` use fixed-point arithmetic with 128-bit intermediates: range reduction by a pinned fixed-point `ln 2` (respectively `2π`), and a fixed-degree Taylor polynomial with pinned fixed-point coefficients evaluated by Horner's rule with floor division at each step. The constants (`ln 2`, `log2 e`, `2π`, Taylor coefficients) are computed once at 256-bit precision, stored in `spec/vectors/tables/constants.json`, and written as literals in every implementation.
- **The generator is the definition.** A table's correctness for the protocol is its exact output, not its closeness to the real function. Accuracy (at most one output unit from the true value for every key) is a fidelity property, tested separately against a 256-bit reference.
- `TableSpec.digest = H("table", TCE(spec without digest) ‖ TCE(piece parameters) ‖ TCE(core values))`. Hashing the defining data rather than all rows keeps digest computation `O(core)`.
- Three implementations exist: `tessera-tables` (prover), `tv-tables` (verifier), `tessera_quant.tables` (Python reference). All are checked against pinned digests.

### Interfaces

```rust
pub trait Table {
    fn spec(&self) -> &TableSpec;
    fn log2_rows(&self) -> u8;
    fn row(&self, index: u64) -> (Fp, Fp);                 // (key, value) as field elements
    fn eval_mle(&self, rho: &[Fp2], beta: Fp2) -> Fp2;     // structured evaluation of key + beta*value
    fn lookup(&self, key: i64, class: u8) -> Option<i64>;  // witness-side integer lookup
}

pub struct LogUpInstance<'a> { pub table: &'a dyn Table, pub keys: VirtualPoly, pub values: CommittedPoly,
                               pub multiplicities: CommittedPoly }
pub fn prove_logup_batch(instances: &[LogUpInstance], tr: &mut Transcript) -> LogUpProof;
```

### Failure modes

| Failure | Response |
|---|---|
| Honest key outside the table | `ERR-INTERNAL` (range analysis bug, INV-IR-2) |
| `d_root = 0` | `Reject(Lookup)` |
| Final table evaluation mismatch | `Reject(Lookup)` |
| Generator divergence between implementations | Digest mismatch; `Reject(StatementMismatch)` at claim-list validation; CI conformance failure |
| Multiplicity counter overflow | Impossible: counts are `< N < 2^{48}` and multiplicities are field elements |

## Alternatives Considered

1. **One-hot lookups (Lasso/Shout, as in Jolt Atlas).** Considered for huge decomposable tables. Rejected: with hash-based commitments each lookup commits several one-hot chunks, so per-lookup cost rises, and the separation result cited by the source shows the activation function's structure cannot be exploited to lower it (source §3). The piecewise structured evaluation already makes large tables cheap for the verifier.
2. **Univariate lookups (Plookup, Caulk, cq).** Rejected: they assume univariate polynomial commitments, often pairing-based, incompatible with the multilinear hash-based design.
3. **Base-field LogUp challenges (VerInf).** Rejected: about 2^-28 error at frontier volume.
4. **Separate lookup and table trees.** Rejected: two roots to compare and two GKR instances where one combined tree suffices.
5. **Shipping table contents with the claim list.** Rejected: a 2^32-row table is 64 GB, and the verifier would trust prover-supplied data instead of regenerating it.
6. **Floating-point or big-float generators (libm, MPFR).** Rejected: libm differs across platforms; MPFR is a C dependency under a copyleft license, outside the verifier allowlist.

## Drawbacks

- Fixed-point generators must be implemented three times with identical rounding.
- The combined tree mixes lookup counts and table sizes, so a small table used rarely still pays its full row count per proof; tables are therefore shared across layers wherever definitions coincide.

## Migration / Rollout

The LogUp-GKR prover and verifier land in v0.1 (with dense test tables) for the unit-cost measurement. The registry, generators and structured evaluation land in v0.2.

## Testing Strategy

- Completeness: random dense tables (2^8 to 2^16 rows) and lookups (2^4 to 2^20) verify; batched instances of different heights verify.
- Soundness: one lookup outside the table, one wrong value, one multiplicity off by one, each rejected; over the `F65537` test field, the empirical false-accept rate of a single random-`α` trial matches `(N + T)/|F|` within a 99% binomial interval.
- Padding: instance sizes `2^k − 1`, `2^k`, `2^k + 1` verify.
- Structured evaluation: for each table kind at small parameters (`b ≤ 16`), `eval_mle` equals brute-force `Σ_x eq(ρ, x) t(x)` at 100 random points.
- Generators: cross-implementation digests agree; accuracy against a 256-bit reference is within one output unit for every key at test parameters and on 10^6 sampled keys at production parameters.
- Performance: CPU throughput per lookup in `criterion`; GPU throughput is the v0.1 unit-cost measurement (RFC-0020).

## Open Questions

None.

## References

- Source §3 (separation result), §5 (lookup unit cost), §6 principles 4 and 6, §7 ("Lookups everywhere are LogUp-GKR"), §9 problem 1.
- Haböck, "Multivariate lookups based on logarithmic derivatives", 2022.
- Papini and Haböck, "Improving logarithmic derivative lookups using GKR", 2023.
- Setty and Thaler, "Twist and Shout", 2025 (one-hot alternative).
