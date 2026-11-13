#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 OR MIT
# Prove the tiny conformance model at ni-100 (real advice geometry: 2^22-wide rows, several
# segments) with CPU 1 thread, CPU many threads and the GPU backend, and verify each
# independently with public tokens: isolates a real-scale rejection (#730, #737).
set -euo pipefail
T=${TESSERA_BIN:-$PWD/target/release/tessera}; V=$PWD/verifier/target/release/tessera-verify
PIN=$PWD/verifier/target/release/examples/statement_pin
v=$PWD/spec/vectors/exact-cli; dir=${DIR:-/root/runs/tiny-ni100}; mkdir -p "$dir"; cd "$dir"
[ -e compiled/plan.tep ] || "$T" compile --model "$v/tiny-3-reference.d" --params ni-100 \
  --out-dir compiled --max-log2-rows 18 > /dev/null
[ -e weights.twc ] || "$T" commit-weights --model "$v/tiny-3-reference.d" --plan compiled/plan.tep \
  --out weights.twc > /dev/null
echo "1 2 2 6 0" > tokens.txt
pin=$("$PIN" compiled/claims.tcl weights.twr 2 3)
for mode in ${MODES:-cpu1 cpu64 gpu}; do
  case $mode in
    cpu1) flags=(--threads 1);; cpu64) flags=(--threads 64);; gpu) flags=(--backend gpu);;
  esac
  start=$(date +%s)
  "$T" --json prove "${flags[@]}" --model "$v/tiny-3-reference.d" --plan compiled/plan.tep \
    --weights weights.twc --reference-tokens "$v/recorded.ttk" --prompt-len 2 --request-id 1 \
    --request-session "$(printf '11%.0s' {1..16})" --manifest-digest "$(printf '13%.0s' {1..32})" \
    --memory-budget 600 --out "$mode.tnp" > /dev/null 2> "$mode.err"
  set +e
  "$V" verify --json --claims compiled/claims.tcl --weights weights.twr --weights-root "${pin#* }" \
    --statement-digest "${pin% *}" --bound 0 --tokens tokens.txt --proof "$mode.tnp" \
    --max-artifact-bytes 8000000000 --max-memory-bytes 400000000000 > "$mode.verdict"
  code=$?; set -e
  echo "$mode: prove $(( $(date +%s) - start )) s, $(stat -c %s "$mode.tnp") bytes, verify exit $code $(grep -o '"verdict":{[^}]*}[^}]*}' "$mode.verdict")"
done
