#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 OR MIT
# Prove one MVP run directory (model.d, compiled/, weights.twc, tokens.ttk) with the GPU backend,
# timed, with JSON debug events in prove.err for the per-phase profile (#737, #744).
#   scripts/prove-gpu.sh RUN_DIR PROMPT_LEN [extra prove flags]
set -euo pipefail
run=$1 prompt=$2; shift 2
bin=${TESSERA_GPU_BIN:-/root/target-gpu/release/tessera}
cd "$run"
rm -f proof.tnp
/usr/bin/time -f '%e %M' -o prove.time "$bin" --json --log-level debug --log-format json \
  prove --backend gpu --model model.d --plan compiled/plan.tep --weights weights.twc \
  --reference-tokens tokens.ttk --prompt-len "$prompt" --request-id 1 \
  --request-session "$(printf '11%.0s' {1..16})" --manifest-digest "$(printf '13%.0s' {1..32})" \
  --memory-budget "${MEMORY_GIB:-300}" --out proof.tnp "$@" > prove.out 2> prove.err
