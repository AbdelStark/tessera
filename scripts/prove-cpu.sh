#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 OR MIT
# As prove-gpu.sh, on the CPU backend with THREADS threads, into RUN_DIR/cpu/ (#730).
set -euo pipefail
run=$1 prompt=$2; shift 2
bin=${TESSERA_BIN:-/root/tessera/target/release/tessera}
mkdir -p "$run/cpu"; cd "$run"
/usr/bin/time -f '%e %M' -o cpu/prove.time "$bin" --json --log-level debug --log-format json \
  prove --threads "${THREADS:-$(nproc)}" --model model.d --plan compiled/plan.tep \
  --weights weights.twc --reference-tokens tokens.ttk --prompt-len "$prompt" --request-id 1 \
  --request-session "$(printf '11%.0s' {1..16})" --manifest-digest "$(printf '13%.0s' {1..32})" \
  --memory-budget "${MEMORY_GIB:-600}" --out cpu/proof.tnp "$@" > cpu/prove.out 2> cpu/prove.err
