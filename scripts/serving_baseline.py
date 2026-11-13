# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Production-serving baseline for the MVP benchmark (#742): vLLM serving one request at a time
(batch 1) of PROMPT prompt tokens and NEW greedy output tokens, as the inference-proofs
observatory's ref-serve does for l31-8b-c1024, timed per request after warm-up. Prints one JSON
line: seconds per request and GPU-seconds per token covered (prompt plus output, one GPU).

    python serving_baseline.py MODEL_DIR [PROMPT=1024] [NEW=64] [REQUESTS=8] [MEM_FRACTION=0.9]
"""

import json
import statistics
import sys
import time

from vllm import LLM, SamplingParams


def main() -> None:
    model = sys.argv[1]
    prompt, new, requests = (int(a) for a in (sys.argv[2:5] + ["1024", "64", "8"][len(sys.argv[2:5]):]))
    fraction = float(sys.argv[5]) if len(sys.argv) > 5 else 0.9
    llm = LLM(model=model, max_model_len=prompt + new, max_num_seqs=1,
              gpu_memory_utilization=fraction, dtype="bfloat16")
    params = SamplingParams(temperature=0.0, max_tokens=new, ignore_eos=True)
    # Deterministic public prompts: token ids spread over the vocabulary, one set per request.
    prompts = [{"prompt_token_ids": [128000] + [(7919 * (r + 1) * (i + 1)) % 120000 + 1000
                                               for i in range(prompt - 1)]}
               for r in range(requests + 2)]
    for p in prompts[:2]:
        llm.generate([p], params, use_tqdm=False)  # warm-up
    times = []
    for p in prompts[2:]:
        start = time.perf_counter()
        out = llm.generate([p], params, use_tqdm=False)
        times.append(time.perf_counter() - start)
        assert len(out[0].outputs[0].token_ids) == new
    per_request = statistics.median(times)
    print(json.dumps({
        "schema": "tessera.serving-baseline/1", "engine": "vllm", "model": model, "prompt_tokens": prompt, "output_tokens": new,
        "batch_size": 1, "requests": requests, "gpu_memory_utilization": fraction,
        "seconds_per_request_median": per_request,
        "seconds_per_request_min": min(times), "seconds_per_request_max": max(times),
        "gpu_seconds_per_token_covered": per_request / (prompt + new),
        "decode_tokens_per_second": new / per_request,
    }))


if __name__ == "__main__":
    main()
