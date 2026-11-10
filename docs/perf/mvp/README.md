# MVP measurements

Evidence for the MVP plan ([MVP.md](../../roadmap/MVP.md), epic [#722](https://github.com/AbdelStark/tessera/issues/722)). Every figure here is measured on rented hardware through [`scripts/gpu-run`](../../../scripts/gpu-run) and [`scripts/mvp-model.sh`](../../../scripts/mvp-model.sh), with the commit that produced it.

| File | What |
|---|---|
| [`benchmark.md`](benchmark.md) | Tessera's numbers in the observatory's units beside FlashProve and SASH, with caveats (#743) |
| `verdict-8b-h200.json`, `verdict-8b-h200-v2.json`, `verdict-8b-h200-wrong-token.json`, `verdict-1b-h200.json`, `verdict-1b-h200-p4.json` | Verifier verdicts of the accepted real-model proofs, and the `REJ-TOKENS` control |
| `commitment-report-8b.json`, `commitment-report-1b.json` | The compiler's per-model counts: commitments and lookups per token, table rows |
| `serving-{8b,1b}-h200-{8x8,c1024}.json` | Serving baselines on the idle H200 NVL the proofs ran on: vLLM 0.32.0, batch 1, at the proofs' 8 + 8 shape and the workload's 1,024 + 64, median of 8 requests after 2 warm-ups, `gpu_memory_utilization` 0.9 ([`scripts/serving_baseline.py`](../../../scripts/serving_baseline.py)) |
| `serving-8b-h100.json`, `serving-1b-h100.json` | Production-serving baselines (#742): vLLM 0.32.0, one H100 SXM, batch 1, 1024 prompt and 64 greedy output tokens, median of 8 requests after 2 warm-ups ([`scripts/serving_baseline.py`](../../../scripts/serving_baseline.py)). Measured while the host CPU ran quantization, at `gpu_memory_utilization` 0.5 and 0.3 so that the quantizer kept GPU memory; batch 1 does not use the KV capacity this removes. |

## Units

The observatory reports per token covered and per Gbit of active weights: active parameters per token times weight bits. Llama 3.1 8B at 8-bit weights is 64.24 Gbit; Llama 3.2 1B at 8-bit (with its materialized head) is 11.99 Gbit. The overhead is GPU-seconds per token of the prover over GPU-seconds per token of production serving at the same load.

## Llama-3.2-1B on one H200: the first accepted real-model proof (2026-11-10)

Tessera's integer model of [NousResearch/Llama-3.2-1B](https://huggingface.co/NousResearch/Llama-3.2-1B/tree/a9745ffc3556f145a830ac0c203509ba860582a3) (servable, affine scales, migration strength 1/2, 64 calibration windows), `ni-100/1`, `tessera-protocol/1`, no zero knowledge. One request: an 8-token prompt and its 8-token exact-greedy continuation (16 positions). `tessera prove --backend gpu` on one H200 NVL (141 GB) with 82 host CPUs and 1 TB of RAM, `--memory-budget 300`. `tessera-verify verify --tokens` (public prompt and continuation) returns **Accept** ([verdict](verdict-1b-h200.json)).

| Phase | Wall time |
|---|---|
| Weight recommit and witness | 1 min 23 s |
| P1 advice commitment (GPU encode and hash) | 1 min 37 s |
| P2 leaf stage: lookup and table trees (GPU level rounds) | 11 min 16 s |
| P3 reduction stages | 5 min 43 s |
| P4 openings (combined rows) | 29 min 10 s |
| P5 column answers (GPU re-encode) | 3 min 32 s |
| **Prove, total** | **54 min 36 s**, peak host RSS 190 GB |
| Verify | 2 min 8 s, 8.7 GB |
| Proof | 4.48 GB |

The run shared the host with another proof for its first 25 minutes. The table side is fixed per model (59 tables, 5.5 G rows at `B_max = 32`, [commitment report](commitment-report-1b.json)), so 16 positions measure mostly fixed cost; per-token figures need the workload's 1,088 positions. Before this run, every servable model failed `REJ-PCS-EVAL`: the prover opened RsqrtRow's root bytes before its input limbs, against RFC-0037's order (fixed in this branch).

## Llama-3.1-8B on one H200: accepted (2026-11-10)

An 8 + 8 request of Tessera's integer model of Llama-3.1-8B-Instruct, proven in 1 h 30 min at 386 GiB peak host memory with GPU P1, P2, P4 and P5, verified with the public tokens in 2 min 15 s: **Accept**. The same day's profiled fixes (GPU weight encoding, parallel stage loads) bring the proof to 48 min 22 s, also accepted. Phases and the comparison: [benchmark.md](benchmark.md).
