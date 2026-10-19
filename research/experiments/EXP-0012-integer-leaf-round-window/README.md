# EXP-0012: Integer leaf round window

| | |
|---|---|
| Status | `planned` |
| Idea | [IDEA-0022](../../ideas/IDEA-0022-integer-leaf-level-round-window.md), hosting IDEA-0017's in-memory window and IDEA-0010 mechanism 2 |
| Evidence level | E0 operation counts; E1 CPU wall clock; no E2 GPU evidence |
| Hardware | Apple M4 Max, macOS; exact environment saved with the run |
| Pre-registered on | 2026-10-19 |

## Hypothesis

A leaf-level integer grid over the first v rounds, v in {1,3,4,5}, preserves the production transcript and independent verifier acceptance, and its best fixed v in {3,4,5} saves at least 20% of whole-forest time against the EXP-0008 split-eq Gruen kernel on all three 2^22-leaf shapes.

## Success criterion

The judged baseline is EXP-0008's split-eq Gruen forest kernel ported to the current production transcript schedule; current production prove_forests is a separately timed context arm. For each v, sigma = 1 - median(candidate)/median(split). The judged shapes are 2^8 trees of 2^14 leaves, 2^10 trees of 2^12 leaves and one tree of 2^22 leaves. Context shapes have 2^20 and 2^21 leaves with height 14. Every arm includes level construction, root claims, all level proofs and leaf claims.

`validated` requires one fixed v in {3,4,5} with sigma >= 0.20 on every judged shape, bit-identical round messages, final evaluations, claims and transcript states to split and production, independent tv-core acceptance, and CV <= 3% for every arm used. `refuted` requires every v in {3,4,5} to have sigma < 0.15 on every judged shape and all correctness/CV checks passing. Everything else is `inconclusive`, including a remaining CV flag after one rerun. v=1 hosts IDEA-0010 mechanism 2 as an observation. The standalone full leaf-level kernel at 2^21 pairs is reported for all v as the note's kill-criterion diagnostic; it does not replace the whole-forest criterion. A differing digest is a harness bug to fix before judgment.

## Method

Port the EXP-0008 harness to current crates without production changes. Only the bottom integer leaf level uses a window; all higher levels use the same split kernel. Grid points {0,1,2}^v evaluate the two multilinear leaf halves as signed integers. Products are accumulated with decomposed eq weights over the remaining variables, then the grid collapses by quadratic Lagrange interpolation after each challenge. The current variable's eq factor and previous eq scale form the cubic message; all messages/challenges remain in production order. The original leaf tables bind once to the v challenges, and the split kernel finishes the remaining rounds. Window work includes grid formation, integer accumulation and the once-bound tables; no preprocessing is excluded.

Overflow cadence is fixed before code. Keys are at most K=2^22-1. For v<=5, abs(V)<=K*3^v, abs(product)<=K^2*9^v, and scaling by a canonical Goldilocks component is below 2^64*K^2*9^v. Accumulate at most B(v) blocks before signed-i128 reduction, with B(1)=32, B(3)=32, B(4)=32 and B(5)=8. The exact check B(v)*(2^64-1)*K^2*9^v < 2^127 is asserted independently before proving. (The note's coarse v=4 bound leaves no slack at 32 blocks; the exact integer inequality is authoritative.) Reduction handles signed sums exactly. A split-eq implementation may reduce at the end of each inner block group, never later than B(v); the actual cadence and extra reductions are counted. Fixed loops/cadence depend only on shapes, not key values.

Synthetic inputs use EXP-0008 SplitMix64 seeds: leaves 0x7E55_E7A0_0058_0001, isolated level 0x7E55_E7A0_0058_0002, correctness 0x7E55_E7A0_0058_0005, one seeded zero per tree, all other keys below 2^22. Check hand root 210 for [2,3,5,7], wrong-root rejection, small sizes/window boundaries, two forests with front padding, maximum integer leaves, and eq/challenge degeneracies (zero, one and prior eq scale zero). For every timed forest compare all proof bytes/digests, claims and transcript states to production and run the independent tv-core tree verifier from roots computed in tv-field. No verifier code changes.

RFC-0020 timing: one thread, 3 warmups, 10 measured samples, median/minimum/maximum/CV, each measured sample >=20 ms by repetition. Rerun each arm once if CV>3%, preserving both raw records. Save raw samples, run environment (CPU/OS/toolchain/date/git commit), and JSON. Time grid formation/accumulation separately as an observation so the i128 component's contribution is visible; isolating integer accumulation is a microkernel observation, not an additive attribution. count.py records exact structural counts and an explicitly assumed H100 int32 translation, never GPU measurements. Convert any leaf_ps scenario only using the existing counted cost-model tool.

## Reproduction

From the repository root:

```sh
cargo run --release --manifest-path research/experiments/EXP-0012-integer-leaf-round-window/Cargo.toml --locked -- check
cargo run --release --manifest-path research/experiments/EXP-0012-integer-leaf-round-window/Cargo.toml --locked -- all research/experiments/EXP-0012-integer-leaf-round-window/results/run.json
python3 -I research/experiments/EXP-0012-integer-leaf-round-window/count.py
```

The first build creates the independent Cargo.lock; no root workspace lockfile changes. Preregistration text is frozen in results/README.preregistered.txt and its SHA-256/time in results/preregistration.txt before harness code.

## Results

Pending.

## Analysis

Threats: CPU instruction costs do not establish H100 i128 emulation cost or HBM behavior; structural counts are E0; cache effects and synthetic key distribution may change ratios. The in-memory harness does not establish the future streamed leaf layout or pass savings of IDEA-0017. Historical EXP-0008 absolute timings are context only; all comparisons use current same-harness measurements.

## Verdict

Pending the pre-registered comparisons.

## Follow-ups

An H100 run must implement the same grid with int32 limb accumulation, retain byte-equivalence checks, and establish compute/HBM limits and tile layout before any E2 claim. No scheduler or production change is part of this experiment.

## Amendment 2026-10-19: hosted fraction-window case

This adds IDEA-0017's distinct in-memory fraction-leaf proxy before its code, without changing the primary integer-forest criterion. Its separate case compares a per-round split-eq fraction leaf kernel against the same kernel with a first window v in {2,3,4,5}, on 2^20, 2^21, 2^22, 2^23 and 2^24 synthetic leaves, with SplitMix64 seed 0x7E55_E7A0_0058_0003, key <2^32 and value <2^16. Denominators are alpha-key-beta*value and unit numerators; the three sumcheck instances are [eq,d_even,d_odd], [eq,d_even], [eq,d_odd]. A field-valued grid is permitted because denominators contain full-field challenges. Window messages and final evaluations must equal production prove_batch and the split kernel, and tv-core verify_batch must accept from independently computed hypercube sums. Current transcript messages, challenges and gamma weights remain unchanged. Existing zero/one/prior-eq-zero checks also cover this fraction window.

The separate IDEA-0017 case is `validated` if v=4 candidate time / split time <=1.5 on every size and all correctness checks and CV<=3% pass. It is `refuted` if the ratio exceeds v for every v>=2 on every size; otherwise `inconclusive`. Times per leaf per original round and maximum window widths are reported. Counting the input grid-read pass against v repeated original-input reads gives an E0 pass proxy of v, while the once-binding input read remains a separate pass. The harness keeps inputs resident and does not simulate GEMM regeneration, so this proxy establishes neither actual streamed-pass count nor regen latency. RFC-0020's same 3 warmups, 10 samples >=20ms, median/min/max/CV and one rerun apply. These observations cannot promote IDEA-0017's streamed claim to E2; its status/score remain conditional on planner and pass accounting.
