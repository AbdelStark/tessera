# EXP-0032: Exact advice rows of 405B requests, and what each chunk of a long request opens

| | |
|---|---|
| Status | `done: refuted` |
| Idea | [IDEA-0043](../../ideas/IDEA-0043-context-commitments.md) |
| Evidence level | E1 for the rows (exact compiler layouts on CPU); E0 for the bytes (RFC-0057's byte model) |
| Hardware | Apple M4 Max CPU, macOS; no GPU needed |
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

Environment: Apple M4 Max, macOS 26 (Darwin 25.6.0), `rustc 1.95.0`, Python 3, 2026-11-06; source `main` at `79b0b1f` plus this directory. One run, 31.1 s, 554 MB peak RSS (`/usr/bin/time -l`), alongside an eight-worker fuzz campaign (timing only, rows are exact).

## Results

Controls: `rows(16384) = 953,580`, `rows(24576) = 1,819,563`, `rows(32768) = 2,889,129`, exactly RFC-0057's (asserted in the binary). Raw output: [results/rows.json](results/rows.json), [results/run.log](results/run.log), [results/plan.txt](results/plan.txt).

| `n` | Exact rows (E1) | Fit `a·n + b·n²` | Residual |
|---:|---:|---:|---:|
| 4,096 | 179,295 | 146,668 | −18.20% |
| 8,192 | 372,878 | 355,940 | −4.54% |
| 12,288 | 644,589 | 627,819 | −2.60% |
| 16,384 | 953,580 | 962,302 | +0.91% |
| 20,480 | 1,363,393 | 1,359,390 | −0.29% |
| 24,576 | 1,819,563 | 1,819,084 | −0.03% |
| 32,768 | 2,889,129 | 2,926,287 | +1.29% |
| 49,152 | 5,911,335 | 5,891,956 | −0.33% |
| 65,536 | 9,856,803 | 9,859,309 | +0.03% |

Fit: `a = 28.165` rows per token, `b = 1.8658×10^{-3}` rows per token². The plan (E0 bytes on E1 rows):

| Request | Proofs | Total | Largest | Chunks |
|---|---:|---:|---:|---|
| 16K | 1 | 3.3 GB | 3.048 GiB | 16K |
| 32K | 3 | 9.9 GB | 3.184 GiB | 16K, 2 × 8K |
| 64K | 9 | 31.1 GB | 3.830 GiB | 16K, 4 × 8K, 4 × 4K |
| 128K | 31 | 109.6 GB | 3.940 GiB | 16K, 4 × 8K, 14 × 4K, 12 × 2K |

## Analysis

Criterion (1) fails: the largest residual is 18.20% at 4K, beyond the 5% refutation bound. Criterion (2) holds: 31 proofs at 128K, each under 4 GiB, and no offset lacks an aligned chunk of at least 1,024 tokens.

The functional form was wrong, not the scale. A post-hoc fit with a constant, `c + a·n + b·n²`, gives `c = 36,826` rows, `a = 25.677`, `b = 1.8971×10^{-3}`, with every residual within −3.05% to +1.38%. This fit is not the pre-registered one and decides nothing; it explains the miss. The constant is rows every proof carries whatever its length, its table side at least. The registered chunk model `rows(P + C) − rows(P)` cancels it, so it under-charges each chunk by about 36,800 rows, 0.046 GB (0.043 GiB) of opened columns. Added to the largest 128K chunk, that gives about 3.98 GiB, still under the gate, so the plan's count does not move here. It would at a tighter gate.

Threats to validity: the bytes are RFC-0057's E0 model, and its 2.02 GB non-column part is the maximum of three whole requests and does not shrink with a chunk. The chunk model charges a chunk the rows its positions add to a whole request; an exact chunk layout needs a lowering at an offset with cached keys, which the compiler does not have. No proving time is measured.

## Verdict

`refuted` as registered: the two-term fit misses by 18% at 4K. IDEA-0043's kill criterion is not met (every 128K offset has a fitting chunk of at least 1,024 tokens, 31 proofs in all), so the idea stays `triaged`. Its expected-gain table and any RFC must use a per-proof constant (about 37K rows) besides the per-token and per-pair terms.

## Follow-ups

- Re-register the chunk model as `c + a·C + 2b·(C·P + C²/2)` with `c` charged per chunk, and judge it against an exact chunk lowering once one exists.
- Exact per-chunk layouts need a lowering of a chunk at an offset with cached keys, which the compiler does not have; it comes with the RFC that IDEA-0043 promotes into. Proving time per chunk needs the H100 gate of RFC-0020.
