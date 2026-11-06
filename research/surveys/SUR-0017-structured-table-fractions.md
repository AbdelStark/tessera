# SUR-0017: Structured evaluation of generated table fraction trees

| | |
|---|---|
| Status | `current (2026-11-06)` |
| Agenda topic | `lookups` (table side of the per-lookup cost), specialized to [#691](https://github.com/AbdelStark/tessera/issues/691) |
| Questions | Bounded addendum for #691: what the bounded CPU prover currently does per generated table tree, which primary techniques evaluate structured fraction-tree levels exactly in sublinear work, and the limits on applying them without changing `tessera-protocol/1` messages. It does not repeat SUR-0002's lookup-argument comparison. |
| Period covered | Primary sources 2007–2026 checked 2026-11-06; local source at `be4e2c65221e51d7238abf7ee588bea71da5d99c` |
| Produced by | `autoresearch survey lookups`, backlog #691, 2026-11-06 |

## Method

The current-source finding comes from reading the production path at the pinned commit. A delegated reading traced `prove_logup_bounded` and `prove_table_batch_bounded` (`crates/tessera-logup/src/streaming.rs`) through `PreparedTreeBatch` and `FractionTreeRows` (`crates/tessera-sumcheck/src/trees/bounded.rs`, `fraction_rows.rs`, `window.rs`) to the CPU table plan (`crates/tessera-prover/src/cpu/table_plan.rs`). The two load-bearing claims were re-read by hand: four generated operands per level, each calling `read_level` (`trees/bounded.rs:184-231`), and `cache_depth` (`table_plan.rs:14-22`). Real geometry comes from EXP-0030's committed public counts (`results/counts.json.gz`, real candidate, B = 30, two positions).

Primary sources: [PH23] and [STW23] (already in SUR-0002) for the fraction tree and sparse-dense sumcheck; [Stwo26] for the production LogUp-GKR layer kinds; [BGS07] for evaluating a polynomial on an arithmetic progression; [vzGG97] for Taylor shifts. [BGS07]'s landing record was checked, but its PDF text could not be extracted on this host. Its theorem statements are therefore cited at abstract level, and the experiment implements and counts its own arithmetic-progression arithmetic instead of crediting the paper's bounds. [vzGG97] is cited from its bibliographic record (ACM DL returned HTTP 403).

## Findings

| Source | What it is | Mechanism (one sentence) | Numbers | Term | Applicability to Tessera | Caveats |
|---|---|---|---|---|---|---|
| Tessera `be4e2c6` bounded table path | Current production source | Each generated tree is regenerated from its leaves: one scan for the root (or the cache), then for every level `t ≥ c` four operand reads of level `t + 1` per pass, `1 + c_t` passes. | `1 + Σ_{t=c}^{h−1} 4·(1 + c_t)` full leaf scans per tree, `c = min(h − 4, L ≤ 20)` (E0, code reading). At least 41 scans of 2^30 leaves for `Range{30}`, which has 66 lookups. | `regen`, `c_ℓ` (table side) | direct: this is the baseline | `c_t` (streamed rounds) depends on the allowance and is not derived here; `c_t = 0` gives a lower bound |
| EXP-0030 real 1B counts | Committed public geometry | Table kinds, `log2` rows and lookup entries per instance. | 5,536,481,870 used table rows against about 2.26·10⁶ lookup entries. By rows: `Silu` 43.6%, `Range` 33.7% (21 tables, 1,866,181,582 rows), `Requant` 22.6%, `Exp` 0.1% (E0, primary) | `c_ℓ` | direct: sets the Amdahl limit of a Range-only result at 33.7% of table rows | Two-position NI-100 case only. Larger contexts change lookup entries, not table rows. |
| [PH23] | LogUp-GKR | A fraction tree `(n, d) = (n_L d_R + n_R d_L, d_L d_R)` proven level by level with a degree-3 sumcheck. | (E0, primary) | — | direct: RFC-0008's relation, unchanged | Says nothing about structured tables |
| [Stwo26] | Production LogUp-GKR | Layer kinds `LogUpSingles`, `LogUpMultiplicities`, `LogUpGeneric`, all dense over the layer. | Per packed pair: 3–5 extension multiplications (E0, code reading) | — | adapt: confirms the dense baseline has no structured table kind | No public table-specialized layer |
| [STW23] | Lasso, sparse-dense sumcheck | A sumcheck over a sparse operand times a structured dense operand costs time in the sparse support. The dense factor must decompose into prefix and suffix functions. | `3cm + cn^{1/c}` committed elements (E0, primary) | `c_ℓ` | adapt: the fraction tree's numerators are linear in the lookups (`N_{t+1}[e] = −Σ_{a_i ∈ e} Π_{s∈e, s≠a_i}(α − s)`), so their contribution is a sum over the fixed lookup capacity | A block denominator `P_B(α − B·y)` is not a prefix/suffix product. Here it is handled as a polynomial in the node index instead. |
| [BGS07] | Shifted evaluation on arithmetic progressions | Given a polynomial's values on an arithmetic progression, its values on a shifted progression follow from one convolution. Products of consecutive linear terms then cost `Õ(√n)` by baby-step giant-step. | Abstract-level `claim` here: cost statements not read in full | `regen` | adapt: Range denominators at a level are `P_B(α − B·e)` on the progression `e = 0, 1, …`, which is the setting of the paper | Needs inverses of small integers (public constants in Goldilocks). Fast versions need an NTT, which Goldilocks has to 2^32. |
| [vzGG97] | Taylor shift algorithms | Computing `f(x + a)` from `f`'s coefficients: classical `O(n²)`, convolution-based `O(M(n))`. | (E0, primary record) | `regen` | direct: binding a sumcheck variable maps a node polynomial `H(u)` to `(1 − ρ)·H(2u) + ρ·H(2u + 1)`, a scaling plus a Taylor shift | Fast variants need division by small integers. The classical variant is division-free. |

## Synthesis

- **Range tables are structured.** A Range table's tree has public denominators. At level `t + 1` of a height-`h` tree, entry `e` covers the leaves `[eB, (e + 1)B)` with `B = 2^{h−t−1}`, so `D_{t+1}[e] = P_B(α − B·e)` with `P_B(z) = Π_{s<B}(z − s)`. That is a polynomial of degree `B` in `e`.
- **Folding keeps the structure.** A sumcheck that binds the lowest variable maps a polynomial in the remaining index to another polynomial of the same degree ([vzGG97]). Each round's dense sum over the eq-weighted suffix is therefore a dot product of the product polynomial's coefficients with the suffix moments `Σ_u eq(r, u)·u^k`. Those follow from Pascal's rule, without division.
- **Numerators are linear in the lookups.** A fixed-capacity loop over the lookup column's `2^{h_a}` entries, padding included, reproduces them exactly ([STW23]'s sparse-dense setting). Duplicate keys add without grouping, and a zero denominator (α a table key) needs no exception because `Π_{s≠a}` is computed without division.
- **The messages are unchanged.** Exact field values fix the messages, so any such evaluator gives byte-identical round messages, evaluations and roots. No relation, transcript or parameter changes.
- **What the current source pays.** It scans the whole leaf domain at least `1 + 4(h − c)` times per tree, and four times per level in each pass even when the four operands read the same level. This redundancy alone costs 4× for every generated table, valued tables included.
- **Applicability limits.**
  - The polynomial structure needs `t_j = j` on the leaf range, which holds for `Range`.
  - Valued tables (`Requant`, `Silu`, `Exp`) have `t_j = key + β·value(key)`. They are piecewise structured: the value is constant on aligned runs (`2^r` for `Requant`, `2^{r_c}` and saturated tails for `Silu`, the zero tail for `Exp`). On such a run the denominator is `(α − β·v) − key`. That is a separate idea and earns no Range credit (#691).
  - Accumulating per-lookup numerators into a level's dense node table at secret node indices is a secret-dependent address that INV-ARCH-2 (01 §6) does not list. Only the variant that leaves the upper levels to the existing public-index scan is admissible without an amendment.

## Candidate ideas

- **C0** (→ IDEA-0043): exact structured bottom levels for Range trees. Keep the cached upper levels and replace the levels `t ≥ c` with polynomial denominators and fixed-capacity numerator entries. This moves `regen`, at least `4(h − c)` leaf scans per tree to about one. Killed by any message mismatch or by a counted cost above the baseline.
- **C1**: the same with the cached level built from shifted evaluation ([BGS07]) and block-accumulated numerators, making the whole tree sublinear in `T`. Killed by INV-ARCH-2 unless amended.
- **O4**: one `read_level` per level pass serving all four operands, for every generated table, valued ones included. 4× fewer scans on the uncached levels, implementation only.
- **V1**: piecewise-constant valued tables (`Requant`, `Silu`, `Exp`), with the C0 evaluator per constant-value run. Needs its own registration (#691).

## Staleness triggers

- A production change to `FractionTreeRows`, `PreparedTreeBatch` or `cache_depth` (re-derive the scan count).
- A real-model count run at another context or policy (re-derive the Range share).
- A readable full text of [BGS07] (replace the abstract-level citation).

## References

[PH23], [STW23], [Stwo26], [BGS07], [vzGG97], resolved in [bibliography.md](../bibliography.md).
