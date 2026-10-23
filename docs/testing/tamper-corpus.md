# Tamper corpus

The SC-4 tamper corpus ([07 §4](../spec/07-testing-strategy.md#4-soundness-and-uniqueness-tests)) mutates a valid tiny proof once per verifier check. Every mutation must yield the verdict the specification predicts, at the step [RFC-0019 'Check order'](../rfcs/RFC-0019-verifier.md#check-order) assigns it. It lives in [`spec/vectors/tamper/`](../../spec/vectors/tamper/README.md), which lists its files, its case format and its families.

## Run it

```sh
# The independent verifier judges every case, non-interactive and recorded live.
(cd verifier && cargo test -p tv-core --features test-params --lib driver::corpus --locked)
# The base proofs regenerate byte for byte; every NI case through tv-core's public API.
cargo test -p tessera-e2e --features tamper --test tamper_corpus --example tamper-base --locked
# The oracle reproduces protocol-1.json from the committed bases.
python3 spec/vectors/tamper/generate.py --check
```

The `tamper corpus` job of required CI runs all three; the aggregate `test` check fails without it. Both test features, `tamper` and `test-params`, enable test-only features that release builds refuse ([RFC-0001 'Cargo features'](../rfcs/RFC-0001-workspace-and-verifier-independence.md#cargo-features)). Acceptance at `test-small` certifies nothing.

The verifier's test reports every case whose verdict differs, not only the first. A verdict that differs is a finding: either the verifier departs from the specification, or the oracle's prediction does. Fix whichever is wrong. Never copy a verifier's output into `protocol-1.json`.

## Who writes what

- **The bases.** The `tessera-e2e` example `tamper-base` proves the [exact-cli fixture](exact-cli.md) once non-interactively and once in a live session. The live session has a fixed verifier share and fixed queries. Masks and the weight commitment's seed are seeded, so the files regenerate byte for byte. These files are producer output, inputs to the corpus and not evidence of its verdicts.
- **The mutations and verdicts.** [`generate.py`](../../spec/vectors/tamper/generate.py), the independent oracle, parses the bases from RFC-0003's encoding. It chooses each mutation and predicts its verdict from RFC-0008, RFC-0010, RFC-0011, RFC-0019, RFC-0033, RFC-0037, RFC-0043 and RFC-0045. It runs no Rust.
- **The consumers.** `tv-core`'s `driver::corpus` test applies every case and runs both sessions. A recorded live session replays its seed and queries through the driver's internal session, because the public `LiveVerifier` draws its own. `tessera-e2e`'s `tamper_corpus` test runs the non-interactive cases through `verify_non_interactive_bytes`. Each builds the expected statement from the claim list, the weight artifact and the request metadata, never from a proof.

## Regenerate it

When the protocol version or a message layout changes, or the fixture does:

```sh
cargo run -p tessera-e2e --features tamper --example tamper-base
python3 spec/vectors/tamper/generate.py
```

Then rerun the consumers. A new protocol version takes a new `protocol-N.json` and new bases, and keeps the old ones while their version is supported.

## Beyond the corpus

`tv-core` also holds an ignored sweep, `every_value_of_the_recorded_live_session_is_bound`. It changes each value of the recorded live session in turn: every field element, digest and bound in its three prover messages, and Round1's session id and advice root. It requires that none of them is accepted, and predicts no verdicts. Optimized, its 57,968 changes take about 15 minutes:

```sh
cd verifier
CARGO_PROFILE_TEST_OPT_LEVEL=3 cargo test -p tv-core --features test-params --lib --locked \
  -- --ignored every_value --nocapture
```

The corpus does not reach checks that `tessera-protocol/1` lacks: anchors, the ledger and the weight reservation (v0.4), and the LogUp identity sumcheck of `tessera-protocol/2`. It does not reach the LogUp leaf checks or the gadget final checks either: a single change trips an earlier check first. The [alternative-witness tests](../spec/07-testing-strategy.md#4-soundness-and-uniqueness-tests) reach those with a consistent cheating prover. Fuzzing is #185.
