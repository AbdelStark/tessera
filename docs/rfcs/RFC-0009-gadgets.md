# RFC-0009: Uniqueness Gadgets: Requantization, Max-Certificate Softmax, Brackets, and Sampling Certificates

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.3
- Amended by: [RFC-0022](RFC-0022-saturating-residual.md) ('RsqrtRow': the root's range check), [RFC-0024](RFC-0024-wide-requantization.md) ('RequantWide'), [RFC-0028](RFC-0028-balanced-split.md) ('RequantSplit': the split point), [RFC-0030](RFC-0030-masked-exponentials.md) ('SoftmaxRow': every position looked up, the leaf claims' reduction, Lemma 5), [RFC-0035](RFC-0035-padded-rows.md) ('RsqrtRow', 'SoftmaxRow', 'OneHot', 'InverseCdf': the row selector `ψ` on padded rows), [RFC-0050](RFC-0050-gadget-advice-columns.md) ('BigIntCompare' and 'RsqrtRow': canonical auxiliary columns, range requests and exact counts), [RFC-0058](RFC-0058-inverse-cdf-columns.md) ('InverseCdf': the partition's bound, the integers' limbs, the committed values, range requests and exact counts), [RFC-0071](RFC-0071-request-slots.md) ('SoftmaxRow': in a batch, keys within the query's request)

## Summary

A gadget is a constraint pattern that pins a committed value to the unique result of a non-arithmetic operation. This RFC specifies every gadget used upstream of the logits and in scoring: `Requant1`, `RequantSplit`, `RequantWide`, `NonNeg`, `BigIntCompare`, `SoftmaxRow` (max certificate and reciprocal bracket), `RsqrtRow`, `OneHot`, `Argmax` and `InverseCdf`. For each: committed values, lookups, sumcheck relations, cost, and a uniqueness lemma with its proof. Uniqueness upstream of the logits is the soundness rule the whole statement rests on (source §6, principle 5).

## Motivation

The source keeps VerInf's rule verbatim: "no prover freedom before the logits; downstream freedom must only raise `Û`" (source §6, principle 5). Each rescale, normalization and softmax is a place where a prover could choose among several values that all satisfy loose constraints, which zkLLM's tolerance band permits (source §3 table). The source introduces the max certificate, which halves VerInf's per-score cost (source §7, "Softmax by max certificate"). A contributor implementing a gadget needs its exact constraints and the argument that they admit one witness.

## Goals

- Every committed value upstream of the logits has a gadget with a uniqueness lemma.
- Gadgets compose with the batched schedule: their relations are sumchecks, lookups or product trees.
- Costs stated per instance so the compiler's report is exact.

## Non-Goals

- Anchor circuits (AES-128, SHA-256), specified in RFC-0015.
- The scoring bound gadget (`OneSidedLog2`), specified in RFC-0017.
- Zero-knowledge masking, specified in RFC-0013.

## Proposed Design

Notation: `L = 16` (limb width, `GadgetPolicy.limb_bits`); `B_max = 32`; "range check `v ∈ [0, 2^k)`" means one lookup into `Range { bits: k }` for `k ≤ B_max`; all named values are per instance; "virtual" values are expressions whose claims reduce through sumchecks (RFC-0010). A **true integer** has an exact range inside `(−(P−1)/2, (P−1)/2)` ([RFC-0006](RFC-0006-compiler-ir.md#range-analysis)).

**Lemma 0 (small congruence).** If integers `a, b` satisfy `a ≡ b (mod P)` and `|a|, |b| < (P−1)/2`, then `a = b`. The lemmas below reduce to it by keeping every compared quantity below 2^62, except Lemma 2. Its two sides lie in `[0, n · 2^b)`, and two integers in `[0, P)` that are congruent modulo `P` are equal, so it needs `n · 2^b < P`.

### Requant1

Inputs: virtual key `z` (true integer, `b ≤ B_max` signed bits), hidden channel class `c` (committed with the weights, RFC-0012). Output: committed `y`.

The requantization gadgets take `c` to be an integer in `[0, n)` for the site's `n` classes, its domain in [RFC-0006](RFC-0006-compiler-ir.md#range-analysis) 'Range analysis'. Lemmas 1, 2 and 11 rely on it for their value half only: for any field `z` and `c` at most one witness passes, and for `c ∈ [0, n)` its `y` is `rq(z, r_c)` ([RFC-0048](RFC-0048-weight-trust.md#the-lemmas-two-halves)). `R_W`'s publisher vouches for the domain. The lookups alone do not establish it: a class outside `[0, n)`, or a field element that is no integer, can reach the row of another key or class and pass with its output.

- Lookup: `(pack(c, z), y) ∈ Requant { b, r_by_class, lo, hi }` with `pack(c, z) = (z + 2^{b−1}) + 2^b · c` ([RFC-0008](RFC-0008-logup-and-tables.md#table-specifications)). The class term is present only when the table has several classes.
- Cost: 1 commitment, 1 lookup.
- Lemma 1. Given `z` and `c`, exactly one `y` passes. Proof: the table is a function of its key.
- The same gadget with a `Silu` table implements the SiLU activation.

### RequantSplit

Inputs as `Requant1` with `B_max < b ≤ 2·B_max − 2` and shift `r`. Lemma 2 holds for any split point `s` with `b − B_max ≤ s ≤ min(r − 1, B_max)`, when `n · 2^b < P` for the table's `n` classes (both checked at claim-list validation); [RFC-0028](RFC-0028-balanced-split.md) fixes the one that balances the two tables, `s = clamp(⌈b/2⌉, b − B_max, min(r − 1, B_max))` for one class. Of keys of at most 62 bits, `n · 2^b < P` excludes only four classes at `b = 62`, which take `RequantWide` ([RFC-0006](RFC-0006-compiler-ir.md#gadget-selection)).

- Committed: `z_lo`, `y`. Virtual: `z_off = z + 2^{b−1}`; `z_hi = (z_off − z_lo) · (2^s)^{−1}`.
- Lookups: `z_lo ∈ [0, 2^s)`; `(pack(c, ⌊z / 2^s⌋), y) ∈ Requant { b − s, r_by_class − s, lo, hi }`. `z_hi = ⌊z_off / 2^s⌋` is already the offset encoding of `⌊z / 2^s⌋` in `b − s` bits, since `2^s` divides `2^{b−1}`, so the key is `z_hi + 2^{b−s} · c`: the packing adds only the class.
- Cost: 2 commitments, 2 lookups.
- Lemma 2. Let `n · 2^b < P`, and let `c` be an integer in `[0, n)`, of shift `r`. Then `z_lo = z_off mod 2^s` is the only value passing both lookups, the high key is the row of `z_hi = floor(z_off / 2^s)` in class `c`, and `y = clamp(floor((z + 2^{r−1}) / 2^r), lo, hi)`. Proof: suppose `z_lo' ∈ [0, 2^s)` and the high key `(z_off − z_lo')·2^{−s} + 2^{b−s} · c` is the row `j ∈ [0, n · 2^{b−s})`. Then `z_lo' + 2^s · j ≡ z_off + 2^b · c (mod P)`. Both sides are integers in `[0, n · 2^b − 1]`, so they differ by less than `n · 2^b`, and `n · 2^b < P` makes them equal. Since `2^s` divides `2^b`, `z_lo' ≡ z_off (mod 2^s)`, hence `z_lo' = z_lo` and `j = z_hi + 2^{b−s} · c`. For the output: since `2^{r−1}` is a multiple of `2^s` when `s ≤ r − 1`, `floor((z_off + 2^{r−1}) / 2^r) = floor((z_hi + 2^{r−1−s}) / 2^{r−s})` by nested floor division, which is what the reduced table computes.
  - *Why `n · 2^b < P`.* Four classes at `b = 62` break it, `4 · 2^62 = 2^64 > P`, and `2^32 · (2^32 − 1) = P − 1`. At the split `s = 32`, in class 0 at `z = −2^61 + d` with `0 ≤ d ≤ 2^32 − 2`, `z_lo' = d + 1` makes the high key `2^32 − 1`: the table's last row, class 3's largest key. It passes with that row's output: with shifts 33 to 36 and INT8 bounds, `127` where the true one is `−128`. Such a key takes `RequantWide`, whose class term reads only values its lookups pin (Lemma 11).

### RequantWide

Inputs as `Requant1`, for a key that `RequantSplit` cannot take ([RFC-0024](RFC-0024-wide-requantization.md)): `b ≤ 62`, shifts `r_c` from `r_min` to `r_max`, and `M = max(|lo|, |hi|, 1)`. The key has three parts: `s = min(r_min − 1, B_max)` low bits, `t = (r_max − s) + ⌈log2 M⌉` middle bits and `k = b − s − t` top bits. The preconditions `t + 2 + ⌈log2 n⌉ ≤ B_max` and `k + 1 ≤ B_max` are checked at claim-list validation.

- Committed: `z_lo` (when `s ≥ 1`), `h0`, `g`, `y`.
- Virtual:
  - `z_off = z + 2^{b−1}`;
  - `w_off = (z_off − z_lo) · (2^s)^{−1}`;
  - `h1_off = (w_off − h0) · (2^t)^{−1}`, and `h1 = h1_off − 2^{k−1}`;
  - `w' = h0 + 2^t · g`.
- Lookups:
  - `z_lo ∈ [0, 2^s)`;
  - `h0 ∈ [0, 2^t)`;
  - `h1_off ∈ [0, 2^k)`;
  - `(pack(0, 2·h1), g) ∈ Requant { k + 1, [1], −2, 1 }`, so `g = clamp(h1, −2, 1)`;
  - `(pack(c, w'), y) ∈ Requant { t + 2, r_by_class − s, lo, hi }`.
- Cost: 4 commitments, 5 lookups; 3 and 4 when `s = 0`.
- Lemma 11. The digits of `z_off` in the radix `(2^s, 2^t)` are the only `z_lo`, `h0`, `h1_off` passing the three `Range` lookups, and then `y = clamp(floor((z + 2^{r_c−1}) / 2^{r_c}), lo, hi)`.
  - *Uniqueness.* The lookups make the three values integers below `2^s`, `2^t` and `2^k`. Then `2^{s+t} h1_off + 2^s h0 + z_lo ≡ z_off`, with both sides in `[0, 2^b)` and `b ≤ 62`, so Lemma 0 gives equality and the digits follow. `g` and `y` are functions of their keys.
  - *The value.* Let `w = floor(z / 2^s) = 2^t h1 + h0` and `r' = r_c − s`. Since `s ≤ r_c − 1`, nested floor division reduces the requantization to `q(w) = floor((w + 2^{r'−1}) / 2^{r'})`, as in Lemma 2.
    - For `−2 ≤ h1 ≤ 1`, `w' = w`.
    - For `h1 ≥ 2`, both `w` and `w' = h0 + 2^t` are at least `2^t`, where `q ≥ 2^{t−r'} ≥ M ≥ hi`.
    - For `h1 ≤ −3`, both are at most `−2^t − 1`, where `q ≤ −2^{t−r'} ≤ −M ≤ lo`.
  - *Why the top part's range check.* Without it, the half-integer `h1_off = m + 1/2` reaches the saturation table's odd key `2m + 1`, and `h0 − 2^{t−1}` passes with another `y`.

### NonNeg

Proves a virtual true integer `v ∈ [0, 2^k)` for `k ≤ 62` with `n = ceil(k / L)` limbs.

**The limb box.** Every decomposition of a field element into range-checked limbs, in any gadget, has its box below `P`: the limbs' widths add up to at most 62 bits, the top limb checked to its own width. Otherwise a value `x` also decomposes as the integer `x + P`, and a comparison on the limbs reads the wrong integer ([#377](https://github.com/AbdelStark/tessera/issues/377)). `NonNeg`'s top limb is checked to `k − (n − 1)·L` bits for this reason. A gadget that decomposes a field element into limbs uses a `NonNeg`, or states its box.

- Committed: limbs `v_0 … v_{n−2}`. Virtual: `v_{n−1} = (v − Σ_{j<n−1} v_j 2^{jL}) · 2^{−(n−1)L}`.
- Lookups: `v_j ∈ [0, 2^L)` for `j < n − 1`; `v_{n−1} ∈ [0, 2^{k − (n−1)L})`.
- Cost: `n − 1` commitments, `n` lookups.
- Lemma 3. Passes iff `0 ≤ v < 2^k`, with unique limbs. Proof: as Lemma 2, applied limb by limb, except that `v` may be negative: the limbs' sum, in `[0, 2^k)`, and `v`, a true integer, compare by Lemma 0.

### BigIntCompare

[RFC-0050](RFC-0050-gadget-advice-columns.md) fixes the six grouped committed arrays, their range requests and the public carry boxes. Its Rsqrt specialization fixes the complete product limb count; constant sides and final zero carries are retained.

Proves `X ≤ Y` for non-negative integers given in base `2^L` limbs that may exceed `P` as whole numbers (used where products such as `ρ^2 · u` reach `2^{2w+55}`, with `w` the bit length of `RsqrtRow`'s largest root: `2^{91}` at `w = 18`).

- Inputs: limb vectors `x_j`, `y_j` (each a committed or virtual true integer in `[0, 2^L)` for limbs, or an exact limb-product sum bounded below 2^40).
- Committed: difference limbs `d_j ∈ [0, 2^L)` and borrows `β_j ∈ {0, 1}` such that `y_j − x_j − β_{j−1} = d_j − 2^L β_j` for every `j`, with `β_{−1} = 0` and final borrow `β_last = 0`.
- Lookups: each `d_j` in `Range(L)`; each `β_j` in `Range(1)`; limb inputs that are sums of products are first normalized by a carry chain with committed carries range-checked to the bound the range analysis gives.
  - Normalization of either input `a`: commit digits `ā_j ∈ [0, 2^L)` and carries `κ_j ≥ 0`, and check `a_j + κ_{j−1} = ā_j + 2^L κ_j`, with `κ_{−1} = 0`. The public limb count includes enough high zero limbs to hold the whole integer; check `κ_last = 0` on **both** sides. Dropping this final check compares truncated integers. Already normalized inputs may use the same chain with zero honest carries.
  - Carry bounds are derived from the input ranges, not declared by the prover. For inclusive upper bounds `A_j`, the honest maximum obeys `K_{−1} = 0`, `K_j = floor((A_j + K_{j−1}) / 2^L)`. A carry lookup may use `Range(max(1, bit_length(K_j)))`. For Lemma 0, bound the equation with the **entire lookup box**, not just `K_j`: if its largest allowed carry is `C_j`, the two sides are at most `A_j + C_{j−1}` and `(2^L − 1) + 2^L C_j`, with `C_{−1} = 0`. These boxes must stay below `2^41` and inside the centered field range. The bound of each input sum must itself be established upstream (SP-7); a field equation does not range-check an unconstrained input.
- Cost: `O(limbs)` commitments and lookups per instance; used per row, never per element.
- Lemma 4. With every limb, borrow and carry range-checked, each limb equation involves integers below 2^41, so it holds over the integers (Lemma 0); the chain then computes `Y − X` exactly, and `β_last = 0` holds iff `Y ≥ X`.

### SoftmaxRow

For query row `q` of a sequence of `n` positions, with valid key positions `i ∈ V_q` (causal mask: `C(q, i) = [i ≤ q < n]`, public, [RFC-0030](RFC-0030-masked-exponentials.md#the-causal-selector); in a batch, keys within `q`'s request, [RFC-0071](RFC-0071-request-slots.md)) and virtual scores `x_qi` (true integers, `|x| ≤ 2^21`). Every other position of the padded score tensor is masked: padded rows and columns, and, where the tensor is stacked, every position of a padded layer or head.

- Committed: row maximum `c_q`; exponentials `e_qi` at every position, 0 at masked ones; reciprocal `R_q`; limbs of two `NonNeg` instances.
- Virtual: keys `k_qi = c_q − x_qi`; lookup keys `κ_qi = C(q, i)·k_qi + (1 − C(q, i))·K_0`, with `K_0 = 2^{key_bits} − 1` the table's last key; row sum `s_q = Σ_i e_qi`, which equals `Σ_{i ∈ V_q} e_qi` because the masked `e` are 0; bracket slacks `a_q = ψ_q·2^{2σ} − R_q s_q` and `b_q = s_q − ψ_q − a_q`, with `ψ_q` the row selector, 1 on a real row and 0 on a padded one ([RFC-0035](RFC-0035-padded-rows.md#the-row-selector)).
- Relations:
  1. Lookup `(κ_qi, e_qi) ∈ Exp { key_bits, σ, scale }` at every position of the padded score tensor: the key `k_qi` at a valid position, `K_0` at a masked one. The table maps `K_0` to 0: `key_bits` is the narrowest width, from the spread's up, whose last key lies in the zero tail ([RFC-0030](RFC-0030-masked-exponentials.md#relation-1-every-position-looked-up)).
  2. Product tree: `Π_{i ∈ V_q} k_qi = 0`, one tree per row. The rows' trees form one forest ([RFC-0010](RFC-0010-proof-schedule.md#tree-levels) 'Tree levels') of one height `h`, so row `q`'s tree has `2^h` leaves. Leaf `i` is `L_qi = C(q, i)·k_qi + 1 − C(q, i)`, with `C` the public causal selector. Every masked position's leaf is the constant 1, so the tree's product is exactly `Π_{i ∈ V_q} k_qi`. A padded row's tree has every leaf 1, and its public root value is 1.
  3. `R_q ∈ [0, 2^{σ+1})`; `NonNeg(a_q, σ + 18)`; `NonNeg(b_q, σ + 18)`. On a padded row `s_q = 0`, so both slacks are 0, and a padding zero-check pins `c_q` and `R_q` to 0 ([RFC-0035](RFC-0035-padded-rows.md)). The width `σ + 18` holds every honest slack, below `s_q ≤ 2^{σ+17}`, because [RFC-0007](RFC-0007-claim-list.md#verifier-validation) bounds `max_context` by `2^{17}` (Lemma 6).
- The forest's leaf claim `L̃(r)` and the lookup's key claim `κ̃(ρ)` go to the stage that produces the keys. It proves each as the sum it stands for, an `Elementwise` relation of degree 3 ([RFC-0030](RFC-0030-masked-exponentials.md#relation-2-the-trees)):
  - `Σ_x eq(r, x)·C(x)·k(x) = L̃(r) − 1 + C̃(r)`;
  - `Σ_x eq(ρ, x)·C(x)·k(x) = κ̃(ρ) − K_0·(1 − C̃(ρ))`.

  The verifier evaluates the causal selector's multilinear extension `C̃` itself, in `O(log S)` time, at the points of the claims and at the stage's final point.
- Cost per valid score: 1 lookup, 1 commitment, 1 product-tree leaf. A masked position's lookup is the constant tuple `(K_0, 0)`, and its leaf the constant 1 ([#321](https://github.com/AbdelStark/tessera/issues/321)). Per row: 2 commitments (`c_q`, `R_q`), about 4 limb commitments and 6 lookups.
- Lemma 5 (max certificate). Relations 1 and 2 hold iff, for every row `q < n`, `c_q = max_{i ∈ V_q} x_qi` and every valid `e_qi = Exp[c_q − x_qi]`, and every masked `e_qi = 0`. Proof: at a masked position, relation 1 puts `(K_0, e_qi)` in `Exp`, whose keys are distinct, so `e_qi = Exp[K_0] = 0`. At a valid position, relation 1 forces each `k_qi` to be an integer in `[0, 2^{key_bits})`, so `c_q ≥ x_qi` for all `i ∈ V_q`. The keys are integers below `P`, and a field has no zero divisors, so relation 2 forces some `k_qi ≡ 0`, hence `k_qi = 0` and `c_q = x_qi` for that `i`. So `c_q` is the maximum; values follow from relation 1. The proof needs the masked leaves to be non-zero and public, and the constant 1 is that.
  - **A masked leaf of 0**, or a selector that multiplies the leaf, would make every row's product vanish. Then `c_q` could be any value at or above the maximum.
  - **A masked leaf of the real key `c_q − x_qi`** of a future position would let `c_q` equal any future score above the causal maximum.
- Lemma 6 (reciprocal). Let `(2·max_context − 1)·2^{2σ} < P`, which claim-list validation checks ([RFC-0007](RFC-0007-claim-list.md#verifier-validation) step 2). On a real row, `ψ_q = 1`, relation 3 holds iff `R_q = floor(2^{2σ} / s_q)`. Proof:
  - *The row sum.* By Lemma 5, `s_q` is the integer `Σ_{i ≤ q} e_qi`: masked positions, padded ones included, add 0 ([RFC-0030](RFC-0030-masked-exponentials.md#what-reads-e)), and each valid `e_qi` lies in `[0, 2^σ]` ([RFC-0008](RFC-0008-logup-and-tables.md#fixed-point-generators)). The maximal score has key 0, and `e(0) = 2^σ`. A sequence holds at most `max_context` positions ([RFC-0006](RFC-0006-compiler-ir.md#llama-lowering)), so `2^σ ≤ s_q ≤ max_context·2^σ`.
  - *The slacks.* Each `NonNeg`'s lookups make its limbs' sum an integer in `[0, 2^{σ+18})` congruent to its input, whether or not that input is a true integer: `α ≡ a_q` and `β ≡ b_q`. `a_q` need not be one, since `R_q` is any value its range check admits: `R_q·s_q` ranges below `max_context·2^{2σ+1}`, which is `2^{64}` at `max_context = 2^{17}` and `σ = 23`.
  - *`b_q` bounds `α`.* `β ≡ s_q − 1 − α`, both sides integers below `2^{62}` in absolute value, so Lemma 0 makes them equal: `0 ≤ α ≤ s_q − 1`.
  - *`a_q` pins `R_q`.* `T = R_q·s_q + α ≡ 2^{2σ}`, and `0 ≤ T ≤ 2^{σ+1}·s_q − 1 < P + 2^{2σ}`, by `s_q ≤ max_context·2^σ` and the precondition. So `T − 2^{2σ}` is a multiple of `P` in `[−2^{2σ}, P)`, and `2^{2σ} < P` makes it 0. Then `R_q·s_q + α = 2^{2σ}` with `0 ≤ α < s_q`, so `R_q = floor(2^{2σ} / s_q)` and `α = 2^{2σ} mod s_q`.
  - *Completeness.* The true `R_q ≤ 2^σ` passes its range check, and its slacks lie in `[0, s_q) ⊆ [0, 2^{σ+17})`, inside `NonNeg`'s `σ + 18` bits. That width rests on RFC-0007's `max_context ≤ 2^{17}`; a protocol that raises the limit widens it.
  - *Why the precondition.* When `s_q·2^{σ+1} > 2^{2σ} + P`, `R' = floor((2^{2σ} + P) / s_q)` is below `2^{σ+1}` and passes too: `R'·s_q` wraps past `P` once, and its slack `2^{2σ} + P − R'·s_q` lies in `[0, s_q)`. At `σ = 24`, a row of 33,280 positions at the maximal score, `s_q = 33,280·2^{24}`, passes with `R = 504` and with `R' = 33,038,714` ([#376](https://github.com/AbdelStark/tessera/issues/376)). The largest row sum public data allows, `max_context·2^σ`, keeps `s_q·2^{σ+1} ≤ 2^{2σ} + P` exactly when `(2·max_context − 1)·2^{2σ} ≤ P`, and no such product equals the prime `P`. So the precondition is the weakest bound that serves every such row sum. For every `max_context` below `2^{32}`, it reads `2σ + ⌈log2 max_context⌉ ≤ 63`: `σ ≤ 23` at `max_context = 2^{17}`, 8B's context, and `σ ≤ 31` at the least. Counting the context's padding, `2^{⌈log2 max_context⌉}` positions, gives the same rule.
- Relational fact emitted: `R_q · s_q ≤ 2^{2σ}`, used by range analysis to bound the attention output.

### RsqrtRow

[RFC-0050](RFC-0050-gadget-advice-columns.md) fixes the exact decomposition, role order and lookup sequence; those choices are no longer implementation-defined.

For a residual row with virtual `u = ψ·(Σ_i res_i^2 + e)` (true integer, `u < 2^55`), dimension `d`, precision `κ`, constant `N = d · 2^{2κ}`, and row selector `ψ`, 1 on a real row and 0 on a padded one ([RFC-0035](RFC-0035-padded-rows.md)):

- Committed: `ρ`; the limbs of `NonNeg(u, 55)`: three 16-bit limbs and a top limb in `Range{7}`, a box of `2^{55} < P`. The box, not the bound `u < 2^55` alone, makes the recomposition exact: four full 16-bit limbs would also represent `u + P` for every `u ≤ 2^{32} − 2`, and admit a second root; `BigIntCompare` witnesses for `ρ^2 u ≤ ψ·N` and `ψ·(N + 1) ≤ (ρ + 1)^2 u`. On a padded row `u = 0`, both comparisons are `0 ≤ 0`, and a padding zero-check pins `ρ` to 0.
- Lookups: `ρ ∈ [0, 2^w)`, with `w = max(1, bitlen(⌊√(N / e)⌋))`, at most 32 ([RFC-0022](RFC-0022-saturating-residual.md#complete-norms)); limb and carry range checks.
- Products enter `BigIntCompare` as bounded limb-product sums, not whole products reduced modulo the field. The canonical byte-root decomposition of RFC-0050 is tied to `ρ` in a box below `P` (the limb-box rule above); `(ρ + 1)` retains its carry when it reaches `2^w`. Both comparisons include enough high zero limbs for the full product and check their final normalization carries (Lemma 4).
- Cost: `4+r+12m` commitments and `6+r+12m` range lookups per row, with `r=ceil(w/8)-1`, `m=ceil((2w+55)/8)` (RFC-0050). One row per token per norm; at `w=32` these are 187 and 189. Historical approximate counts are superseded.
- Lemma 7. The constraints hold iff `ρ = max{ρ : ρ^2 u ≤ N}`. Proof: Lemma 4 makes both inequalities exact over the integers; they bracket `ρ` uniquely.
- Completeness. `u ≥ e` gives the honest `ρ = ⌊√(N / u)⌋ ≤ ⌊√(N / e)⌋ < 2^w`, so every row's root passes the range check. The `BigIntCompare` witnesses' products stay below `2^{2w+55}`.

### OneHot

For a token `t ∈ [0, vocab)` (committed in the token stream, RFC-0015):

- Committed: `h ∈ {0,1}^{vocab}`.
- Relations: zero-check `Σ_x eq(r, x) h(x)(1 − h(x)) = 0` (degree 3); zero-check `Σ_x eq(r, x)·(1 − LT(x_v, vocab))·h(x) = 0` over the padded vocabulary axis, which pins `h_v = 0` for `v ≥ vocab` with no commitment and no lookup, `LT` public; `Σ_v h_v = ψ_t`, with `ψ_t` the row selector, 1 on a real token and 0 on a padded one ([RFC-0035](RFC-0035-padded-rows.md)); `Σ_v v · h_v = t`. A padded row's `h` and `t` are therefore 0.
- Cost: `vocab` commitments per token (0.75% of the linear-layer count at 405B).
- Lemma 8. The relations hold iff `t ∈ [0, vocab)` and `h` is the indicator of `t`, except with probability at most `3 log2(vocab) / |Fp2|` over the zero-check challenge. Proof: booleanity plus sum one leaves exactly one non-zero entry, the vocabulary zero-check puts it below `vocab`, and the weighted sum fixes its index. Without that zero-check a token in `[vocab, V_pad)` passes and reads a padded column of `Embed` ([#378](https://github.com/AbdelStark/tessera/issues/378)).
- The one-hot vector of the token at position `p + 1` is reused by `Argmax` and `InverseCdf` at position `p` (teacher forcing).
- Relational fact emitted: `h` holds exactly one 1, so `W · h` is the column `t` of `W`, and range analysis gives it `W`'s range.

### Argmax (greedy decoding)

Logits `ℓ_v` (virtual true integers, `|ℓ_v| < 2^28`), output token `y` with one-hot `h`:

- Committed: maximum `m`.
- Virtual: `lt_v = Σ_{w > v} h_w` (1 if `v < y`), keys `g_v = m − ℓ_v − lt_v`.
- Relations: `[v < vocab]·g_v ∈ [0, 2^{29})` for every `v` of the padded vocabulary axis, the public selector masking the padded positions to the key 0; `Σ_v h_v ℓ_v = m`. `OneHot`'s vocabulary zero-check keeps `y` below `vocab`, so the checks over `[0, vocab)` cover every position the one-hot can mark.
- Vocabulary padding does not join the honest maximum: a zero padded logit must not raise an all-negative real maximum. The selected-logit sum can include padding because OneHot pins its indicator entries to zero.
- Suffix reduction: a claim on `lt` at `ρ` is `Σ_v h_v·Σ_{i<v} eq(ρ,i)`. At a sumcheck's final point `r` the public factor is `Σ_{i<v} eq(ρ,i)eq(r,v)`, evaluable in `O(log vocab)` by the highest differing bit. This binds the suffix to the same next-token one-hot.
- Cost per output token: `vocab` lookups, 1 commitment.
- Lemma 9. Holds iff `y = min{v : ℓ_v = max ℓ}`. Proof: `g_v ≥ 0` gives `ℓ_v ≤ m` for `v ≥ y` and `ℓ_v ≤ m − 1` for `v < y`; the inner product gives `ℓ_y = m`.

### InverseCdf (temperature sampling)

Vocabulary exponentials `e_v = Exp_T[c − ℓ_v]` from a `SoftmaxRow`-style max certificate over the vocabulary (temperature folded into the table scale), partition `Z = Σ_v e_v ≤ vocab·2^σ ≤ 2^{σ+18}` ([RFC-0058](RFC-0058-inverse-cdf-columns.md) corrects the earlier strict bound, which a vocabulary of `2^18` equal logits reaches), PRF output `u ∈ [0, 2^{32})` (committed, produced by the AES gadget of RFC-0017), sampled token `ŷ` with one-hot `ĥ`:

- Virtual: `C_≤ = Σ_v e_v Σ_{w ≥ v} ĥ_w` (prefix sum through `ŷ`), `C_< = C_≤ − Σ_v ĥ_v e_v`.
- Relations: `BigIntCompare(C_< · 2^{32} ≤ u · Z)` and `BigIntCompare(u · Z + ψ ≤ C_≤ · 2^{32})`, with `ψ` the row selector ([RFC-0035](RFC-0035-padded-rows.md)). On a padded row the vocabulary certificate masks every position as `SoftmaxRow` masks a padded query, so `e`, `Z` and `C` are 0 and both comparisons are `0 ≤ 0`; a padding zero-check pins the certificate's maximum to 0, and [RFC-0058](RFC-0058-inverse-cdf-columns.md#padded-rows)'s pins `Z`.
- Cost per output token: the vocabulary max certificate (`vocab` lookups, commitments and product leaves), one one-hot vector if `ŷ ≠ y` is possible (bounded mode), and two comparators. [RFC-0058](RFC-0058-inverse-cdf-columns.md) fixes the comparators' limbs, the committed values, their order and their exact counts.
- Lemma 10. Holds iff `ŷ = min{v : C(v) · 2^{32} > u · Z}`, the inverse-CDF sample. Proof: Lemma 8 fixes `ĥ`; Lemma 4 makes the comparisons exact; the prefix sums are monotone, so the bracket identifies one index.

### Summary of costs

| Gadget | Per | Commitments | Lookups | Product leaves |
|---|---|---|---|---|
| `Requant1` | element | 1 | 1 | 0 |
| `RequantSplit` | element | 2 | 2 | 0 |
| `RequantWide` | element | 4 | 5 | 0 |
| `SoftmaxRow` | score | 1 | 1 | 1 |
| `SoftmaxRow` | row | about 6 | about 6 | 0 |
| `RsqrtRow` | row | `4+r+12m` (RFC-0050) | `6+r+12m` (RFC-0050) | 0 |
| `OneHot` | token | `vocab` | 0 | 0 |
| `Argmax` | output token | 1 | `vocab` | 0 |
| `InverseCdf` | output token | the vocabulary certificate's, about `vocab`, plus `1 + 2(k − 1) + 12(k + 2)` with `k = ⌈bit_length(vocab·2^σ)/16⌉`, plus `pad(vocab)` in bounded mode ([RFC-0058](RFC-0058-inverse-cdf-columns.md)) | the certificate's `vocab`, plus `2 + 2k + 12(k + 2)` ([RFC-0058](RFC-0058-inverse-cdf-columns.md)) | `vocab` |

## Alternatives Considered

1. **VerInf's two-table softmax bracket.** Kept by the source as a valid design (source §4 table, "Keep") but superseded by the max certificate, which needs one lookup and one commitment per score instead of two of each (source §6, principle 6). Rejected for v1 on cost.
2. **zkLLM's tolerance-band softmax.** Rejected: leaves witness slack upstream of the logits.
3. **Row maximum by committed one-hot index per row.** Considered: proves "exists" with a one-hot vector instead of a product tree. Rejected: a one-hot entry costs a commitment (30 ps) where a product-tree leaf costs 12 ps, and the one-hot needs its own booleanity check.
4. **rsqrt by table lookup.** Rejected: `u` reaches 2^55, far beyond any table.
5. **In-circuit Newton iteration for rsqrt and reciprocal.** Rejected: iterates are prover-chosen unless pinned by a bracket, and the bracket alone suffices.
6. **Embedding by lookup into a committed table.** Considered. Rejected: LogUp over a committed 2^31-row table requires opening the table's commitment at the fraction tree's final point; the one-hot matmul reuses the weight-commitment fold already present.

## Drawbacks

- `BigIntCompare` adds limb bookkeeping that is easy to get subtly wrong; it is confined to per-row and per-output-token gadgets.
- `RequantSplit` doubles the cost of wide rescales (RISK-2), and `RequantWide` quadruples it where a key is too wide for its shift.
- Bounded-mode sampling needs a second one-hot vector per output token.

## Migration / Rollout

`Requant1`, `RequantSplit`, `RequantWide`, `NonNeg`, `SoftmaxRow`, `RsqrtRow` and `OneHot` land in v0.3 (exact mode with greedy decoding needs `Argmax` too). `InverseCdf` lands in v0.4 with the sampler PRF.

## Testing Strategy

For each gadget, all on random instances at small and production widths:

- Completeness: honest witnesses pass for 10^4 random inputs, including boundary inputs (all scores equal; a single valid position; `u = 0`; `u = 2^55 − 1`; `z` at both ends of its range; ties in `Argmax`).
- Alternative witnesses, each rejected: a `SoftmaxRow` forest whose masked leaves are 0, or the keys of the masked positions' real scores, at the stage that consumes the leaf claim; `c_q` replaced by the second-largest score or by `max + 1`; `R_q ± 1`; `ρ ± 1`; `z_lo + 2^s` and `z_lo − 1`; a one-hot vector with two ones, or one one at the wrong index; `y` replaced by a later tie in `Argmax`; `ŷ ± 1` in `InverseCdf`.
- Lemma checks by exhaustion at toy sizes: for `b ≤ 12`, enumerate all `z_lo'` and confirm only one passes (`RequantSplit`); for `b ≤ 14` over `F65537`, enumerate all `(z_lo', h0')` and confirm only the digits pass (`RequantWide`); for 3-element rows with scores in `[−8, 8]`, enumerate all `c` (`SoftmaxRow`).
- Soundness over `F65537`: wrap-based attacks that succeed when a precondition is violated (for example `s > r − 1`, `n · 2^b ≥ P` in `RequantSplit`, limb bounds too wide, `RequantWide` without its top part's range check, or `(2·max_context − 1)·2^{2σ} ≥ P` in `SoftmaxRow`'s bracket, which Goldilocks shows too at `σ = 24`) are demonstrated to succeed there, confirming the preconditions are necessary, and to fail when the preconditions hold.

## Open Questions

None.

## References

- Source §3 (zkLLM slack), §4 (VerInf bracket, unique witness rule), §6 principles 5 and 6, §7 ("Softmax by max certificate", "Integer model D", RMSNorm carry-chain bracket from VerInf appendix B.4).
- [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0017](RFC-0017-scoring.md).
