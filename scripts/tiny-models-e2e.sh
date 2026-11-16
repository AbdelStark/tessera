#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 OR MIT
# Prove and independently verify (with public tokens) each tiny conformance model at test-small:
# a fast reproducer for real-model rejections (#730). Debug builds; certifies nothing.
set -euo pipefail
root=$PWD
cargo build --release --locked -p tessera-cli --example generate -q
cargo build --release --locked -p tessera-cli -q
(cd verifier && cargo build --locked -p tv-cli --features tv-core/test-params -q \
  && cargo build --locked -p tv-cli --example statement_pin -q)
T=$root/target/release/tessera; G=$root/target/release/examples/generate
V=$root/verifier/target/debug/tessera-verify; PIN=$root/verifier/target/debug/examples/statement_pin
dir=${DIR:-$(mktemp -d)}
for m in ${MODELS:-tiny-2-servable tiny-2-minimal tiny-2-affine tiny-4-servable}; do
  d=$root/spec/vectors/models/$m.d; w=$dir/$m; mkdir -p "$w"
  "$T" compile --model "$d" --params test-small --out-dir "$w/c" --max-log2-rows "${MAX_LOG2_ROWS:-32}" >/dev/null 2>"$w/compile.err" || { echo "$m: compile failed: $(tail -1 "$w/compile.err")"; continue; }
  "$T" commit-weights --model "$d" --plan "$w/c/plan.tep" --out "$w/w.twc" >/dev/null 2>&1
  out=$("$G" "$d" '[1, 2, 3]' 3 "$w/t.ttk" 2>/dev/null)
  tokens="1 2 3 $(echo "$out" | tail -1 | tr -d '[],')"; echo "$tokens" > "$w/t.txt"
  "$T" prove --threads "${THREADS:-8}" --memory-budget "${MEMORY_GIB:-64}" --model "$d" --plan "$w/c/plan.tep" --weights "$w/w.twc" \
    --reference-tokens "$w/t.ttk" --prompt-len 3 --request-id 1 \
    --request-session "$(printf '11%.0s' {1..16})" --manifest-digest "$(printf '13%.0s' {1..32})" \
    --out "$w/p.tnp" >/dev/null 2>"$w/prove.err" || { echo "$m: prove failed: $(tail -1 "$w/prove.err")"; continue; }
  pin=$("$PIN" "$w/c/claims.tcl" "$w/w.twr" 3 3)
  set +e
  "$V" verify --json --claims "$w/c/claims.tcl" --weights "$w/w.twr" --weights-root "${pin#* }" \
    --statement-digest "${pin% *}" --bound 0 --tokens "$w/t.txt" --proof "$w/p.tnp" > "$w/v.json"
  echo "$m: tokens [$tokens] verify exit $? $(grep -o '"verdict":{[^}]*}[^}]*}' "$w/v.json")"
  set -e
done
