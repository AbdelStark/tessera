#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 OR MIT
# The MVP's GPU rule (#722 principle 3): on the tiny conformance models, a GPU proof equals the
# CPU proof byte for byte. Seeded masks need the test-only `test-rng` feature, so this is a dev
# build; it certifies nothing and only compares. Run from the repository root on a GPU host.
set -euo pipefail
cargo build --locked -p tessera-cli --features gpu,test-rng -q
T=$PWD/target/debug/tessera
dir=$(mktemp -d)
for set in exact-cli:tiny-3-reference.d:2 canary:canary-reference.d:3; do
  IFS=: read -r name model prompt <<< "$set"
  v=spec/vectors/$name
  "$T" compile --model "$v/$model" --params test-small --out-dir "$dir/$name" \
    --max-log2-rows 18 --query-block 2 > /dev/null
  "$T" commit-weights --model "$v/$model" --plan "$dir/$name/plan.tep" \
    --out "$dir/$name/weights.twc" > /dev/null
  for backend in cpu gpu; do
    "$T" --log-level debug prove --backend "$backend" --test-mask-seed 7 --model "$v/$model" \
      --plan "$dir/$name/plan.tep" --weights "$dir/$name/weights.twc" \
      --reference-tokens "$v/recorded.ttk" --prompt-len "$prompt" --request-id 1 \
      --request-session "$(printf '11%.0s' {1..16})" \
      --manifest-digest "$(printf '13%.0s' {1..32})" --out "$dir/$name/$backend.tnp" 2>&1 >/dev/null | grep -E "p1_backend|P1 advice" || true
  done
  cmp "$dir/$name/cpu.tnp" "$dir/$name/gpu.tnp"
  echo "$name: GPU proof equals CPU proof ($(stat -c %s "$dir/$name/gpu.tnp") bytes)"
done
