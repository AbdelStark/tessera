#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 OR MIT
# The MVP pipeline for one real Llama model on a rented GPU host (#722, #730, #737):
# download, quantize on CUDA, greedy tokens, compile, commit, prove, verify, with each step's
# wall time and peak RSS in $RUN/steps.tsv. Steps whose output exists are skipped, so the
# script resumes. ALPHA fixes the migration strength (tessera-quant --alpha), 1/2 unless set: an
# empty ALPHA searches strengths on the held-out windows, about 40 CPU minutes per window and
# strength for the 8B model. PROVER_FEATURES=gpu with `--backend gpu` among the prove flags proves
# on the GPU. RECORD names a run manifest to compare the proof's per-stage profile with (#744):
# the script then exits 1 when a phase, pass or stage regressed by more than 5%. Run from the
# repository root:
#
#   scripts/mvp-model.sh 1b|8b PROMPT_LEN NEW_TOKENS [extra `tessera prove` flags]
set -euo pipefail
which=${1:?1b or 8b}; prompt=${2:-8}; new=${3:-8}; shift $(( $# < 3 ? $# : 3 ))
repo=$PWD
models=${TESSERA_MODEL_DIR:-/root/models}
run=${RUN:-/root/runs/$which-p$prompt-n$new}
max_log2_rows=${MAX_LOG2_ROWS:-32}
ALPHA=${ALPHA-1/2}
mkdir -p "$models" "$run"
case $which in
  1b) repo_id=NousResearch/Llama-3.2-1B; rev=a9745ffc3556f145a830ac0c203509ba860582a3
      ckpt=$models/llama-3.2-1b-prepared ;;
  # The inference-proofs closed workload l31-8b-c1024 pins this checkpoint (#741).
  8b) repo_id=unsloth/Llama-3.1-8B-Instruct; rev=4699cc75b550f9c6f3173fb80f4703b62d946aa5
      ckpt=$models/llama-3.1-8b-instruct ;;
  *) echo "usage: $0 1b|8b PROMPT_LEN NEW_TOKENS" >&2; exit 2 ;;
esac
src=$models/$(basename "$ckpt" -prepared)

step() { # step NAME OUTPUT CMD...: run unless OUTPUT exists; record time and RSS.
  local name=$1 out=$2; shift 2
  if [ -e "$out" ]; then echo "== $name: done ($out)"; return; fi
  echo "== $name"
  /usr/bin/time -f "%e %M" -o "$run/$name.time" "$@" > "$run/$name.out" 2> >(tee "$run/$name.err" >&2)
  read -r secs kib < "$run/$name.time"
  printf '%s\t%s\t%s\n' "$name" "$secs" "$((kib / 1024))" >> "$run/steps.tsv"
  echo "   $name: ${secs}s, $((kib / 1024)) MiB peak"
}

cd "$repo/python"
uv sync --locked --extra cu130 -q
py="env PYTHONPATH=. uv run --no-sync python"  # a command, so /usr/bin/time can run it

step download "$src/config.json" $py - "$src" "$repo_id" "$rev" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[2], revision=sys.argv[3], local_dir=sys.argv[1],
                  allow_patterns=["*.json", "*.safetensors", "LICENSE*", "USE_POLICY.md"])
PY
if [ "$which" = 1b ]; then
  step tie-head "$ckpt/config.json" $py tools/materialize_tied_head.py prepare "$src" "$ckpt"
fi
cat > "$run/calibration.json" <<EOF
{"schema": "tessera.calibration-spec/1", "source": "c4", "sequences": ${CALIB_SEQ:-128},
 "tokens_per_sequence": 2048, "held_out": ${CALIB_HELD:-32}}
EOF
step quantize "$run/model.d" $py -m tessera_quant compile --json --checkpoint "$ckpt" \
  --calibration "$run/calibration.json" --profile servable --scales affine \
  --out "$run/model.d" --device cuda ${ALPHA:+--alpha $ALPHA}
step tokens "$run/tokens.ttk" $py tools/greedy_tokens.py "$run/model.d" "$ckpt" \
  "$run/tokens.ttk" --prompt-len "$prompt" --new "$new"

cd "$repo"
cargo build --release --locked -p tessera-cli ${PROVER_FEATURES:+--features $PROVER_FEATURES} -q
(cd verifier && cargo build --release --locked -p tv-cli --example statement_pin -q && cargo build --release --locked -p tv-cli -q)
T=$repo/target/release/tessera; V=$repo/verifier/target/release/tessera-verify
PIN=$repo/verifier/target/release/examples/statement_pin
step compile "$run/compiled/plan.tep" "$T" --json compile --model "$run/model.d" \
  --params ni-100 --out-dir "$run/compiled" --max-log2-rows "$max_log2_rows"
step commit "$run/weights.twc" "$T" --json commit-weights --model "$run/model.d" \
  --plan "$run/compiled/plan.tep" --out "$run/weights.twc"
step prove "$run/proof.tnp" "$T" --json --log-level debug prove --model "$run/model.d" \
  --plan "$run/compiled/plan.tep" --weights "$run/weights.twc" \
  --reference-tokens "$run/tokens.ttk" --prompt-len "$prompt" --request-id 1 \
  --request-session "$(printf '11%.0s' {1..16})" --manifest-digest "$(printf '13%.0s' {1..32})" \
  --out "$run/proof.tnp" "$@"
# The verifier takes its statement pin from public inputs only (07 §6).
pin=$("$PIN" "$run/compiled/claims.tcl" "$run/weights.twr" "$prompt" "$new")
step verify "$run/verdict.json" "$V" verify --json --claims "$run/compiled/claims.tcl" \
  --weights "$run/weights.twr" --weights-root "${pin#* }" --statement-digest "${pin% *}" \
  --bound 0 --out "$run/verdict.json" --proof "$run/proof.tnp"
ls -l "$run/proof.tnp"; cat "$run/steps.tsv"
# The prove step's manifest, beside the weights, against the record (#744).
if [ -n "${RECORD:-}" ]; then scripts/perf-regress.py "$RECORD" "$run/run-manifest.json"; fi
