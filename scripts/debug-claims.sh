#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 OR MIT
# Debug-only: one opening group per claim in a scratch prover and verifier, to name the claim
# that a REJ-PCS-EVAL rejects (#730). Never part of a release.
set -euo pipefail
export TESSERA_DEBUG_ONE_GROUP_PER_CLAIM=1
m=/root/tessera/spec/vectors/models/tiny-2-servable.d; r=/root/runs/tiny-models/tiny-2-servable
d=/root/runs/debug-claims; mkdir -p "$d"; cd "$d"
T=/root/target-dbg/release/tessera
"$T" compile --model "$m" --params test-small --out-dir c --max-log2-rows 30 > /dev/null
"$T" commit-weights --model "$m" --plan c/plan.tep --out w.twc > /dev/null
"$T" prove --threads 64 --memory-budget 600 --model "$m" --plan c/plan.tep --weights w.twc \
  --reference-tokens "$r/t.ttk" --prompt-len 3 --request-id 1 --request-session "$(printf '11%.0s' {1..16})" \
  --manifest-digest "$(printf '13%.0s' {1..32})" --out p.tnp > /dev/null 2> prove.err
pin=$(/root/tessera/verifier/target/debug/examples/statement_pin c/claims.tcl w.twr 3 3)
/root/target-vdbg/debug/tessera-verify verify --json --claims c/claims.tcl --weights w.twr \
  --weights-root "${pin#* }" --statement-digest "${pin% *}" --bound 0 --tokens "$r/t.txt" \
  --proof p.tnp > v.json 2> verify.err || true
grep -o '"verdict":{[^}]*}[^}]*}' v.json; cat verify.err | head -20
