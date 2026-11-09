# MVP measurements

Evidence for the MVP plan ([MVP.md](../../roadmap/MVP.md), epic [#722](https://github.com/AbdelStark/tessera/issues/722)). Every figure here is measured on rented hardware through [`scripts/gpu-run`](../../../scripts/gpu-run) and [`scripts/mvp-model.sh`](../../../scripts/mvp-model.sh), with the commit that produced it.

| File | What |
|---|---|
| `serving-8b-h100.json`, `serving-1b-h100.json` | Production-serving baselines (#742): vLLM 0.32.0, one H100 SXM, batch 1, 1024 prompt and 64 greedy output tokens, median of 8 requests after 2 warm-ups ([`scripts/serving_baseline.py`](../../../scripts/serving_baseline.py)). Measured while the host CPU ran quantization, at `gpu_memory_utilization` 0.5 and 0.3 so that the quantizer kept GPU memory; batch 1 does not use the KV capacity this removes. |

## Units

The observatory reports per token covered and per Gbit of active weights: active parameters per token times weight bits. Llama 3.1 8B at 8-bit weights is 64.24 Gbit; Llama 3.2 1B at 8-bit (with its materialized head) is 11.99 Gbit. The overhead is GPU-seconds per token of the prover over GPU-seconds per token of production serving at the same load.
