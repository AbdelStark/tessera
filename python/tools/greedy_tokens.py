# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Write the TokenFile of D's exact-greedy continuation of a prompt (MVP workload, #727).

    python tools/greedy_tokens.py MODEL.d CHECKPOINT_DIR OUT.ttk --prompt TEXT --new N
    python tools/greedy_tokens.py MODEL.d CHECKPOINT_DIR OUT.ttk --prompt-len P --new N
    python tools/greedy_tokens.py MODEL.d - OUT.ttk --prompt-ids '[128000, 791, …]' --new N

The prompt is the tokenizer's encoding of TEXT after BOS, or with --prompt-len the first P
tokens of a fixed public text. Each continuation token is the first index of the largest logit
of D's integer reference at the last position (RFC-0009 Lemma 9: the earliest tie). Prints the
prompt length, then the continuation as a JSON list. The tokens are the prover's secret input;
`--prompt-ids` takes them as given (an inference-proofs request) and needs no checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tessera_quant.checkpoint import import_checkpoint, read_tokenizer
from tessera_quant.integer_model import INTEGER_MODEL
from tessera_quant.reference import run_reference
from tessera_quant.tce import (
    INTEGER_MODEL_LIMITS,
    MESSAGE_LIMITS,
    TOKEN_FILE,
    TokenFile,
    read_artifact,
    write_artifact,
)

TEXT = (
    "The history of mathematics is the story of how people learned to count, to measure and "
    "to reason about quantity and shape. Early civilizations kept records of trade, harvests "
    "and the movements of the stars, and from these needs arithmetic and geometry grew. "
) * 64


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt")
    g.add_argument("--prompt-len", type=int)
    g.add_argument("--prompt-ids", help="a JSON list of token ids")
    p.add_argument("--new", type=int, required=True)
    a = p.parse_args()
    d = read_artifact(a.model, INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    if a.prompt_ids is not None:
        ids = [int(t) for t in json.loads(a.prompt_ids)]
    else:
        tok = read_tokenizer(a.checkpoint, import_checkpoint(a.checkpoint).arch)
        ids = [tok.bos, *tok.encode(a.prompt if a.prompt is not None else TEXT)]
    if a.prompt_len is not None:
        if a.prompt_len > len(ids):
            sys.exit("prompt-len exceeds the fixed text")
        ids = ids[: a.prompt_len]
    prompt = len(ids)
    for _ in range(a.new):
        logits = run_reference(d, np.asarray(ids, dtype=np.uint32))
        ids.append(int(np.argmax(logits[-1])))
    seq = np.asarray(ids, dtype=np.uint32)
    write_artifact(a.out, TOKEN_FILE, TokenFile([seq]), limits=MESSAGE_LIMITS)
    print(prompt)
    print(json.dumps(ids[prompt:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
