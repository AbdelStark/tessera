# EXP-0032: Exact structured bottom levels for Range table fraction trees

| | |
|---|---|
| Status | `done: validated` |
| Idea | [IDEA-0043](../../ideas/IDEA-0043-structured-range-fractions.md), [#691](https://github.com/AbdelStark/tessera/issues/691) |
| Evidence level | E0: exact Goldilocks `Fp2` arithmetic and counted field multiplications; no timing |
| Hardware | Apple M1 Max, 64 GiB (E0 only; the machine does not affect a result) |
| Pre-registered on | 2026-11-06 |

## Hypothesis

For a `Range{h}` table tree, the levels `t ≥ c` can be computed from:

- two denominator polynomials of degree `B = 2^{h−t−1}` in the node index;
- suffix moments of the eq weights;
- one fixed-capacity numerator entry per lookup slot.

Computed this way, every level's round polynomials, its four evaluations and the tree root are exactly those of the dense level-by-level computation of RFC-0008 'Fraction tree' and RFC-0010 'Tree levels'. Counted in field multiplications, this costs at most a quarter of the current bounded path on the real 1B Range instances.

## Success criterion

`validated` requires all four items below. It is scoped to finite E0 algebra and operation counts. It means no production implementation, no wall-clock speed-up, no real-model acceptance and no protocol/2 coverage.

1. **Exactness.** All registered cases are exactly equal: the candidate and the dense reference agree on every level `t` from `0` to `h − 1`, every round's four values `s(0..3)`, the four level evaluations and the root `(n, d)`, with zero mismatches. Case grid, over `Fp2 = Fp[X]/(X² − 7)`, `p = 2^64 − 2^32 + 1`:
   - heights `h = 1…10` and cache depths `c ∈ {0, …, h}` (the structured route applies to `t ≥ c`, the dense route to `t < c`);
   - lookup heights `h_a ∈ {0, 1, 3, h, h + 2}`, with keys from SplitMix64 seed `0x0691`;
   - for each `(h, h_a)` four key patterns: uniform, every key 0 (duplicates, row-0 padding), every key `2^h − 1`, a single key repeated;
   - α in two families: random `Fp2`, and a base-field value in `[0, 2^h)` (a zero leaf denominator);
   - `r`, `ρ`, `λ` random in `Fp2`, with the lookup column padded to `2^{h_a}` with key 0 as RFC-0008 requires.
2. **Independent check.** A separate checker that reads only the produced messages accepts every honest transcript. It checks `s(0) + s(1)` against the running claim, the final value against `eq·g(evals)`, and the root against the cross-multiplied identity with the lookup tree's root `Σ 1/(α − a_i)`. It also rejects each transcript with one round value incremented.
3. **Faults.** Each of the seven registered faults below makes at least one case mismatch the reference, while the unmodified candidate passes:
   - F1, block offset `α − B(2y + 1 − b)`;
   - F2, duplicate keys deduplicated;
   - F3, the term `N_1·D_0` dropped;
   - F4, falling factorial over `s ∈ [1, B]`;
   - F5, `ρ` and `1 − ρ` swapped in the polynomial fold;
   - F6, an entry's weight computed as `D/(α − a)` with an inverse, which must fail (exception or mismatch) on the zero-denominator family;
   - F7, the lookup padding entries omitted.
4. **Counts.** The harness's operation counter is checked first: its closed-form counts (`Fp2` multiplications per level, per strategy) must equal the instrumented counts on every case with `h ≤ 10`. On the 21 Range instances of EXP-0030's real record (B = 30, two positions), the summed closed-form candidate count, with the upper levels kept dense from the existing cache, is at most 0.25× the summed conservative baseline. The conservative baseline:
   - takes `c = min(h − 4, 20)` (`cache_depth` with `L = 20`), and `c_t = 0` (no streamed rounds: the fewest scans);
   - charges every leaf scan `2^h` leaf generations plus `3·(2^h − 2^{depth})` multiplications for subtree recombination;
   - adds the dense level sumchecks' multiplications.

**Verdicts.**

- `refuted`: any exactness or independent-check failure that is not a harness bug fixed before results are recorded (each such fix is logged), or a count ratio above 0.5.
- `inconclusive`: any other failure of 3 or 4, including a ratio in `(0.25, 0.5]`.

## Method

`run.py`, a single Python 3.10+ program run with `python3 -I`, using arbitrary-precision integers only.

1. **Field.** `Fp2` arithmetic is checked against `spec/vectors/field/goldilocks_ext2.json`, all cases. That is the baseline reproducing a known value.
2. **Reference.** Dense leaves, every level built bottom-up by RFC-0008's rule, and a dense level sumcheck binding the lowest variable first.
3. **Candidate.** RFC-0008's cache depth `c`, dense levels `t < c`, and the structured route of IDEA-0043 for `t ≥ c`. Polynomial arithmetic is classical and division-free, moments use Pascal's rule, and entry selects are arithmetic. The numerator weights at level `t + 1` are prefix/suffix products within the block.
4. **Checker.** A separate function that sees only the transcript, as described in criterion 2.
5. **Faults.** Each fault is a flag on the candidate. Every flag is run over the full grid and must give at least one mismatch.
6. **Counts.** A multiplication counter in the field class, plus the closed forms. The closed forms are compared with the counter on the grid, then evaluated on `results/real-range.json`, the 21 Range instances read from EXP-0030's `counts.json.gz` by `run.py`.

## Reproduction

```sh
python3 -I research/experiments/EXP-0032-structured-range-fractions/run.py \
  --out research/experiments/EXP-0032-structured-range-fractions/results
```

## Results

Run on 2026-11-06 (UTC 09:4x–10:3x): Apple M1 Max, macOS 15.6.1, Python 3.15.2, `python3 -I`, the three `--only` modes in parallel. The results are E0, so they do not depend on timing. Raw output in `results/`:

- `report-exact.json`, `report-faults.json`, `report-counts.json`, `real-range.json`;
- `run-exact.log`, `run-faults.log`, `run-counts.log`.

The field check reproduced every case of `spec/vectors/field/goldilocks_ext2.json`, products and inverses.

| Criterion | Registered requirement | Result |
|---|---|---|
| 1 Exactness | zero mismatches on the `h ≤ 10` grid | 2,536 cases, 8,736 structured levels compared (all levels `t ≥ c`). 0 mismatches in round values, level evaluations or roots. |
| 2 Independent check | honest accepted, one-value tamper rejected, root identity | 8,736 honest level transcripts accepted, 8,736 tampered rejected, 2,536 root identities hold: 0 problems |
| 3 Faults | each of F1–F7 caught | F1 2,160 cases · F2 1,552 · F3 1,203 · F4 2,160 · F5 2,136 · F6 108 (the zero-denominator family, as registered) · F7 1,839. The unmodified candidate passes criterion 1. |
| 4a Count closed forms | equal to instrumented counts on every `h ≤ 10` case | 8,736 structured calls in the grid and 432 count-check rows: 0 mismatches. The counts do not vary with key values, the pattern or α's family. |
| 4b Real Range ratio | summed candidate / conservative baseline ≤ 0.25 | 2.7454·10¹⁰ / 2.8971·10¹¹ = **0.0948** |

Per table (`real-range.json`; counted `Fp2` and scalar multiplications, leaf generations at 1 each; conservative baseline with `c_t = 0`):

| Range bits | Lookup height | Entries | `c` | Baseline | Candidate | Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 7 | 66 | 20 | 1.796·10¹¹ | 4.688·10⁹ | 0.026 |
| 29 | 17 | 81,920 | 20 | 8.140·10¹⁰ | 9.920·10⁹ | 0.122 |
| 27 | 10 | 528 | 20 | 1.630·10¹⁰ | 8.938·10⁸ | 0.055 |
| 26 | 18 | 131,336 | 20 | 7.245·10⁹ | 4.019·10⁹ | 0.555 |
| 25 | 18 | 131,336 | 20 | 3.252·10⁹ | 2.770·10⁹ | 0.852 |
| 24 | 9 | 264 | 20 | 1.524·10⁹ | 4.057·10⁸ | 0.266 |
| 21 … 1 (15 tables) | 7–19 | 66–278,660 | 0–17 | 3.9·10⁸ in total | 4.75·10⁹ in total | 4.9–4,360 each |

**Harness fixes before the registered run.** All were found on scratch trials (`h ≤ 4`, `h ≤ 7`) whose output was discarded:

- `eq1_small` rewritten as `X·r + (1 − X)(1 − r)` by additions;
- dead code removed from the prefix update;
- `eq_table` simplified to the plain doubling;
- the closed-form message term corrected to three multiplications per `X`;
- the shared cache-read term charged for the level that reads the cache directly.

Two changes widened coverage to match the registration: counts are now compared on every grid call, and the count check covers `h ≤ 10` instead of 8. None changed the candidate's algebra.

## Analysis

**What is settled at E0.** Every message of a `Range` tree's levels `t ≥ c` can be produced from three objects, never touching the `2^{t+1}`-entry level and never re-scanning the leaves:

- two degree-`B` denominator polynomials, folded by a division-free Taylor shift;
- eq suffix moments, by Pascal's rule;
- one numerator entry per lookup slot.

The messages are identical in every registered case, including:

- α equal to a table key (a zero leaf denominator), where the division-based variant F6 fails as registered;
- duplicate keys and all-padding columns;
- empty lookup columns (`h_a = 0`).

**The cost limit.** The registered summed ratio, 0.0948, is carried by the six tables of 2^24 rows or more, which hold 99.8% of the Range rows (1,862,270,976 of 1,866,181,582). On the fifteen tables of at most 2^21 rows the structured route loses, by up to 4,360×: its per-entry work grows like `Σ_t t·2^{h_a}·B`, while a dense pass over a small table is cheap. A production route must therefore be chosen per tree by comparing the two closed forms. Both are public (`h`, `h_a`, `c`). Taking the cheaper one per tree gives 2.3037·10¹⁰, a ratio of **0.0795**. This last figure is descriptive, outside the registered criterion.

**Threats to validity.**

- *Counts against time.* Multiplications are counted, not timed. A scalar (`Fp × Fp2`) product is counted like a full `Fp2` product, which disfavours the candidate. Leaf generation is charged one unit, though it is a subtraction plus a multiplicity read, which favours the baseline. Additions are not counted: the classical Taylor shift adds `O(B²)` additions per round, and at the largest bottom-level `B = 512` that is about 1.3·10⁵ per round. The cache-read and dense terms are identical in both arms.
- *The baseline is conservative.* `c_t = 0` is its cheapest streaming case. A real allowance that streams rounds multiplies its scans by `1 + c_t`.
- *Geometry.* Two positions of NI-100 at B = 30. Longer contexts raise lookup heights, which raises the candidate's entry work and leaves the baseline unchanged.
- *Scope.* Protocol/1 only. Masked levels (RFC-0032) add mask terms the candidate does not compute. Valued tables are excluded. The Rust implementation, its memory charge and wall time are not measured.

## Verdict

**validated**, in the registered finite E0 scope: exactness, the independent checker, all seven faults and both count criteria pass. This is not a production implementation, a wall-clock speed-up, a real-model proof or a protocol/2 result.

## Follow-ups

- Rust implementation behind the existing generic path, under #641's admission and its allocation controls, with the per-tree public route choice above and its scratch charge:
  - per-entry state of `2^{h_a}` × (position, weight, one eq accumulator);
  - two polynomials of `B + 1` coefficients and moments of `2B + 1` entries per bit at the cache boundary.

  Validate against the generic path on full proofs, then measure (E1) on the synthetic table-batch bench.
- O4, one level read for the four operands, filed as [#694](https://github.com/AbdelStark/tessera/issues/694).
- V1 (piecewise-constant valued tables).
- C1 (an INV-ARCH-2 amendment for a sublinear cached level).
- Masked levels for protocol/2.
