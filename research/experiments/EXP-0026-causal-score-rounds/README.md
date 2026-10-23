# EXP-0026: Causal score rounds without the score tensor

| | |
|---|---|
| Status | `done: validated` |
| Idea | [IDEA-0039](../../ideas/IDEA-0039-causal-score-rounds.md) |
| Evidence level | E0 exact Goldilocks-extension arithmetic and operation counts |
| Hardware | none for the analytic result; synthetic CPU execution only |
| Pre-registered on | 2026-10-23 |

## Hypothesis

For `SoftmaxRow`'s key and forest sums over a padded `[layers, heads, queries, keys]` score tensor with grouped key heads, a prover that answers the rounds over the key bits from the queries, prefix sums of the folded keys and the causal mask's three-case form, and the remaining rounds from row tables, sends every round polynomial and final evaluation a literal sumcheck prover over the materialized tensor sends. Its multiplications over the key-bit rounds grow as `O(H·n·d·log n)` per layer, not `O(H·n²)`.

## Success criterion

- **`validated`** only if all three hold:
  - **Exact equality.** On every registered domain and challenge set, every round polynomial's evaluations at 0, 1, 2 and 3, and the final evaluations of the score and the row maximum at the final point, are equal in `Fp2` between the candidate and the literal prover.
  - **Faults detected.** At least eight distinct planted faults in the candidate each produce a mismatch on at least one registered domain. An unchanged control matches before and after.
  - **The counts.** With `H = 2`, `H_kv = 1`, `d = 4`, `L = 1` and `n = 2^6` to `2^10`, the candidate's counted `Fp2` multiplications over the key-bit rounds, divided by the literal prover's, fall at every doubling of `n` and lie below 0.25 at `n = 2^10`.
- **`refuted`.** Any mismatch on a registered domain, or a failed count criterion.
- **`inconclusive`.** An interrupted or incomplete run.
- **What does not follow.** No CPU speed, GPU, memory, soundness or zero-knowledge result follows: the experiment checks the equality of the messages and counts operations.

These criteria and the method are frozen before experiment code.

## Method

One Python program holds an independent literal prover and the candidate.
- **The field.** Goldilocks `Fp2 = Fp[X]/(X^2 − 7)` with arbitrary-precision integers.
- **The inputs.** Seeded with SplitMix64 `0x45800260001`: INT8 queries and keys in `[−128, 127]`, row maxima in `[−2^20, 2^20]`, and claim points and challenges uniform in `Fp2`. No model weights, tokens or private data.

**The relation.**
- **The sum.** `Σ_x eq(z, x)·M(x)·(c(row(x)) − s(x))` over `x = (ℓ, h, q, i)`, the key index `i` lowest, padded to `n_pad = 2^⌈log2 n⌉` positions, `H_pad` heads and `L_pad` layers.
- **The mask.** `M(x) = [i ≤ q < n]·[h < H]·[ℓ < L]`.
- **The score.** `s = Q_{ℓ,h,q}·K_{ℓ,g(h),i}`, with `g(h) = ⌊h·H_kv/H⌋` the key head of query head `h`.
- **The literal prover.** It materializes `eq`, `M`, the broadcast `c` and `s` as tables over the hypercube, and runs RFC-0010's rounds: lowest variable first, each table folded `f[2k] + r·(f[2k+1] − f[2k])`.

**The candidate.**
- **Its rounds.** It computes the same rounds from the queries, the keys folded over their bound bits, prefix sums over the key blocks, and `LT`'s closed form, as IDEA-0039 states. After the key bits, it switches to row tables over `(ℓ, h, q)`.
- **Its inputs.** It never reads a score-sized table.

**Registered domains.**
- **The shapes.** `n ∈ {1, 2, 3, 5, 8, 9, 16}`; `(H, H_kv) ∈ {(1, 1), (2, 1), (3, 1), (4, 2)}`; `L ∈ {1, 3}`; `d ∈ {1, 3}`.
- **The draws.** Three challenge sets each, one with `z` and every challenge drawn at random, 336 runs in all.

**Faults.** Each fault is planted in the candidate alone, one at a time:
- a prefix sum that includes `i_hi = q_hi`;
- `LT(·, q_lo)` in place of `LT(·, q_lo + 1)`;
- query rows `q ≥ n` left unmasked;
- the key head `h mod H_kv` in place of `g(h)`;
- the keys folded with `r` and `1 − r` swapped;
- the high key bits' `eq` factor dropped;
- padded heads left unmasked;
- the row maximum's mask taken over `i_hi = q_hi` alone;
- the row tables' mask `LT(r_i, q)` in place of `LT(r_i, q + 1)`.

A neutral control rewrites one sum without changing it.

**Counts.** The arithmetic helpers count `Fp2` multiplications, a base-field operand counting as one. The literal prover's count covers its first `log2 n_pad` rounds, the candidate's its key-bit rounds.

## Reproduction

```sh
python3 research/experiments/EXP-0026-causal-score-rounds/main.py all research/experiments/EXP-0026-causal-score-rounds/results/run.json
```

Environment: recorded in `results/run.json`.

## Results

E0, exact `Fp2` arithmetic on one CPU core: Python 3.15.0, macOS-26.6.2-arm64-arm-64bit-Mach-O. The run started at 2026-10-23T16:49:54Z and finished at 2026-10-23T16:51:08Z. Raw output: [results/run.json](results/run.json).

**Equality.**
- **The controls.** The unchanged candidate matched the literal prover on all 336 of 336 registered runs before the faults, and on all 336 after. The match covers every round polynomial at 0, 1, 2 and 3, and both final evaluations. The neutral control, one dot product summed in reverse, matched on all 336.
- **The faults.** Each was detected, by a mismatch on at least one registered run:

| Fault | Result |
|---|---|
| prefix includes q_hi | detected |
| LT at q_lo | detected |
| padded rows unmasked | detected |
| key head mod | detected |
| fold swapped | detected |
| high eq dropped | detected |
| padded heads unmasked | detected |
| row maximum on the edge only | detected |
| row mask at q | detected |

**Counts.** `Fp2` multiplications over the key-bit rounds, `H = 2`, `H_kv = 1`, `d = 4`, `L = 1`:

| Positions | Literal | Candidate | Ratio |
|---|---:|---:|---:|
| `n = 64` | 225,792 | 51,600 | 0.2285 |
| `n = 128` | 910,336 | 125,227 | 0.1376 |
| `n = 256` | 3,655,680 | 297,670 | 0.0814 |
| `n = 512` | 14,651,392 | 696,033 | 0.0475 |
| `n = 1024` | 58,662,912 | 1,605,884 | 0.0274 |

The literal count grows four times per doubling of `n`; the candidate's by 2.43, 2.38, 2.34 and 2.31, as `n·log n` does.

## Analysis

- **Every criterion holds.** Exact equality on all 336 runs, 9 of 9 faults detected, and a ratio that falls at every doubling, to 0.027 at `n = 2^10`, below the registered 0.25.
- **What holds by construction.**
  - After the key bits, the literal prover's folded tables are exactly the candidate's row tables: `eq(z_i, r_i)·eq_rest`, `LT(r_i, q + 1)` on a real row, `c`, and `Q·K̃(r_i)`.
  - Before them, each round polynomial is the same sum regrouped. Exact equality over arbitrary padding values shows that no term was lost: the padded positions carried random queries, keys and maxima that only the mask removes.
- **Threats to validity.**
  - **The scale.** The domains are small, 16 positions and four heads at most. The counted ratio extrapolates as `d·log n/n` by construction, but large shapes run only in the count.
  - **What a count is not.** Counts are not time. The candidate's dot products multiply INT8 queries by `Fp2` vectors, cheaper than the `Fp2`-by-`Fp2` products counted here as one each.
  - **Masking.** RFC-0013's masking polynomial and the stage's other instances are outside the experiment. The candidate computes the unmasked share of one instance, which RFC-0010 combines with `γ^i` like any other.
  - **Development runs.** A four-shape smoke test exercised the harness under development before the registered run, on inputs of another seed. The registered run is the only one reported.
- **The relation.** The instance is RFC-0009's sum with a generic row maximum. The key claim's correction `K_0·(1 − M̃)` and the forest's `1 − M̃` are public, so they do not enter the rounds.

## Verdict

`validated`. IDEA-0039 becomes `validated`, and the CPU prover's `SoftmaxRow` key and forest instances can drop their score-sized tables with no change to the messages. Promotion: RFC-0056 (#458).

## Follow-ups

- **The prover.** The production prover in `tessera-prover`'s native relations, with the same literal prover as the oracle on the conformance models (#458's implementation).
- **The grid.** The window grid of RFC-0055 for the instances that stay elementwise, and its arithmetic on H100 (OQ-0055-1).
