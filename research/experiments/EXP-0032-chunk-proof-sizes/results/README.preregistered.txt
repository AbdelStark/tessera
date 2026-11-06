# EXP-0032: Exact advice rows of 405B requests, and what each chunk of a long request opens

| | |
|---|---|
| Status | `planned` |
| Idea | [IDEA-0043](../../ideas/IDEA-0043-context-commitments.md) |
| Evidence level | E1 for the rows (exact compiler layouts on CPU); E0 for the bytes (RFC-0057's byte model) |
| Hardware | Apple M1 Max-class CPU, macOS; no GPU needed |
| Pre-registered on | 2026-11-06 |

## Hypothesis

The non-zero advice rows of a whole Llama-3.1-405B `servable` request are `a·n + b·n²` within 2% from 4K to 64K tokens, and under 08 §5's 4 GiB gate a 128K request splits into 25 to 35 binary-aligned chunk proofs.

## Success criterion

Measured: `rows(n)`, the `ModelAdviceLayout` non-zero rows (`AdviceLayout::rows`) of `lower(405B servable, n)` with the selection, policy (`B_max = 32`, two classes) and `ni-100/1` layout of `crates/tessera-bench/tests/commitment_report.rs`, at `n` = 4096, 8192, 12288, 16384, 20480, 24576, 32768, 49152, 65536.

- **Control.** `rows(16384) = 953,580`, `rows(24576) = 1,819,563` and `rows(32768) = 2,889,129` exactly (RFC-0057's table). A mismatch makes the run invalid, not a result.
- **Validated** when both hold: (1) the least-squares fit `a·n + b·n²` over all nine sizes has every relative residual within ±2%; (2) the plan below gives 25 to 35 proofs at 128K, each of at most 4 GiB.
- **Refuted** when the fit's largest residual exceeds 5%, or the plan needs more than 50 proofs or finds a 128K offset where no aligned chunk of at least 1,024 tokens fits the gate (IDEA-0043's kill criterion).
- **Inconclusive** otherwise, or when a size cannot be lowered on this machine (the size is then reported and dropped, and the criterion is judged on the rest if at least six remain).

The plan, fixed now: a chunk of `C` tokens at offset `P` has `rows(P + C) − rows(P)` rows (the rows its positions add to the whole request, its scores against all earlier keys included), with `rows` the fit above 64K and the measured value at measured sizes. Its proof is `1,240 · rows` bytes of opened columns, plus 2.02 GB (the largest non-column part of RFC-0057's three rows), plus 0.07 GB of linking and 1 MB per earlier chunk (IDEA-0043, E0). From offset 0, each chunk is the largest power of two at most 65,536 that divides `P`, does not pass `S` and fits `4 · 2^30` bytes.

## Method

- A standalone binary in this directory, depending on `tessera-compiler` and `tessera-ir` by path. It copies the 405B `servable` fixture of `commitment_report.rs` (`llama_405b(Profile::Servable, 2)`) and builds the claim list and `ModelAdviceLayout` as that test does.
- Inputs are public shapes only: no seed, no weights.
- One run per size: the rows are exact integers, so there is nothing to average. Lowering time and peak RSS are reported for information only.
- Correctness check: the three control rows above.
- The planner is the script `plan.py`, reading the binary's JSON output.

## Reproduction

```sh
cd research/experiments/EXP-0032-chunk-proof-sizes
cargo run --release > results/rows.json
python3 -I plan.py results/rows.json > results/plan.txt
```

Environment: to be recorded with the results.

## Results

Not run.

## Analysis

Not run.

## Verdict

Not run.

## Follow-ups

Exact per-chunk layouts need a lowering of a chunk at an offset with cached keys, which the compiler does not have; it comes with the RFC that IDEA-0043 promotes into. Proving time per chunk needs the H100 gate of RFC-0020.
