# EXP-0032: Exact structured bottom levels for Range table fraction trees

| | |
|---|---|
| Status | `planned` |
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

Not run.

## Analysis

Not run.

## Verdict

Not run.

## Follow-ups

- Rust implementation behind the existing generic path, under #641's admission and its allocation controls.
- O4 (one level read for the four operands).
- V1 (piecewise-constant valued tables).
- C1 (an INV-ARCH-2 amendment for a sublinear cached level).
- Masked levels for protocol/2.
