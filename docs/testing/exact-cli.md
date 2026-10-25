# Exact-mode CLI suite

`cargo test -p tessera-e2e --test exact_cli --locked` runs actual `tessera` and
separately built `tessera-verify` processes. The suite compiles the model,
commits weights, produces NI and directory Live proofs, and checks independent
`Accept` verdicts with `Û = 0`. It also checks the exact mismatch error, trusted
pins, no-overwrite output, session reuse, and canonical but malicious packets
at each Live round. A rejected Round1 receives no seed reveal; a rejected
Round2 receives no queries. Child processes have watchdogs and are reaped.

The deterministic synthetic fixture has three layers, width two, and five
tokens (two prompt and three continuation). Both the time and layer axes need
padding. Its nonzero integer logits and greedy continuation come from the
[independent Python oracle](../../spec/vectors/exact-cli/generate.py), not a
producer capture. Regenerate with `python3 spec/vectors/exact-cli/generate.py`;
`cargo xtask gen-vectors --check` checks byte parity. This is a test model,
not a pretrained language model or a fidelity benchmark.

For repeated local runs, explicitly build once and supply both binaries:

```sh
cargo build -p tessera-cli --locked
(cd verifier && cargo build -p tv-cli --features tv-core/test-params --locked)
TESSERA_BIN="$PWD/target/debug/tessera" \
TESSERA_VERIFY_BIN="$PWD/verifier/target/debug/tessera-verify" \
  cargo test -p tessera-e2e --test exact_cli --locked
```

Without these variables the suite builds both binaries into separate directories
under `target/exact-cli-binaries`, using each workspace's own lockfile. It never
silently skips acceptance tests. Required CI builds the pair before the normal
integration step in `rust (prover)`; its eight-minute job deadline enforces the
under-ten-minute requirement. The ordinary verifier build rejects test-small.
Test-small acceptance certifies nothing.

To exercise the producer manually, choose a fresh output directory:

```sh
MODEL=spec/vectors/exact-cli/tiny-3-reference.d
TOKENS=spec/vectors/exact-cli/recorded.ttk
WORK=$(mktemp -d)
target/debug/tessera compile --model "$MODEL" --params test-small \
  --max-log2-rows 18 --query-block 2 --out-dir "$WORK/compiled"
target/debug/tessera commit-weights --model "$MODEL" \
  --plan "$WORK/compiled/plan.tep" --out "$WORK/weights.twc"
target/debug/tessera prove --model "$MODEL" --plan "$WORK/compiled/plan.tep" \
  --weights "$WORK/weights.twc" --reference-tokens "$TOKENS" \
  --prompt-len 2 --request-id 23 \
  --request-session 11111111111111111111111111111111 \
  --manifest-digest 1313131313131313131313131313131313131313131313131313131313131313 \
  --out "$WORK/proof.tnp" --json
```

The compile flags select the existing gadget policy. Defaults stay at 32 bits
and 512 query positions; those defaults can exceed the resident CPU backend's
conservative memory allowance even for a small model. The fixture also bounds
the complete reciprocal-square-root carry tables, not only its root table.
No memory or range check is relaxed. Streaming pass planning remains #132.

For Live, create a fresh directory and replace `--out` with `--live DIR`.
Run the [Live verifier](../verifier/reference-cli.md#directory-live) concurrently.
Both roles write the existing seven RFC-0014 TCE packets atomically, without
replacement. The application must independently construct its whole expected
statement and weight-root pins from its own claim list, public epoch artifact
and request metadata **before** either proof is used. The producer's
[`tessera.proof-run/1`](../../spec/schemas/proof-run-v1.schema.json) receipt is
informational and cannot supply an independent trust pin.

The reference path handles one exact-greedy request on CPU and is non-hiding
protocol/1. It provides directory exchange; authenticated TLS (#181), anchors
and ledger integration (#137/#163), complete resource hardening (#182), GPU
and zero knowledge remain separate work.

The sampled route, `tessera prove --sample`, has its own suite:
`cargo test -p tessera-cli --test prove_sample --locked` plays one session
through `tessera recorder ref`, draws and batches its sample, proves the batch
in NI and directory Live mode and checks `tessera-verify`'s verdicts, then the
refusals of 02 §2.2. It reads `TESSERA_VERIFY_BIN` too, or builds the verifier
into the same directory.
