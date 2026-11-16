#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 OR MIT
# Prove one model with the prover binary $T and verify it (test-params debug verifier, public
# tokens): MODEL TOKENS_TTK TOKENS_TXT PROMPT OUTPUT DIR [prove flags]. For bisecting (#730).
set -euo pipefail
model=$1 ttk=$2 txt=$3 prompt=$4 output=$5 dir=$6; shift 6
root=/root/tessera; T=${T:-$root/target/release/tessera}
V=$root/verifier/target/debug/tessera-verify; PIN=$root/verifier/target/debug/examples/statement_pin
mkdir -p "$dir"; cd "$dir"
"$T" compile --model "$model" --params test-small --out-dir c --max-log2-rows "${MAX_LOG2_ROWS:-30}" > /dev/null
"$T" commit-weights --model "$model" --plan c/plan.tep --out w.twc > /dev/null
start=$(date +%s)
"$T" prove --model "$model" --plan c/plan.tep --weights w.twc --reference-tokens "$ttk" \
  --prompt-len "$prompt" --request-id 1 --request-session "$(printf '11%.0s' {1..16})" \
  --manifest-digest "$(printf '13%.0s' {1..32})" --memory-budget 600 --out p.tnp "$@" > /dev/null 2> prove.err
pin=$("$PIN" c/claims.tcl w.twr "$prompt" "$output")
set +e
"$V" verify --json --claims c/claims.tcl --weights w.twr --weights-root "${pin#* }" \
  --statement-digest "${pin% *}" --bound 0 --tokens "$txt" --proof p.tnp > v.json
echo "$dir: prove $(( $(date +%s) - start )) s, verify exit $? $(grep -o '"verdict":{[^}]*}[^}]*}' v.json)"
