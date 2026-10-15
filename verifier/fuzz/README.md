# Verifier fuzz targets

`cargo-fuzz` targets for the verifier's decoders ([RFC-0003](../../docs/rfcs/RFC-0003-canonical-encoding.md#testing-strategy), [RFC-0019](../../docs/rfcs/RFC-0019-verifier.md#robustness)). This directory is a workspace of its own, so `libfuzzer-sys` never enters the verifier's dependency graph, lockfile or allowlist.

| Target | Input | It fails when |
|---|---|---|
| `decode` | A type selector byte (`TYPES` in `src/lib.rs`), then a whole input for that type's decoder | a decoder panics or allocates beyond libFuzzer's limit; an accepted input is not the canonical encoding of its value (INV-TCE-2); a proper prefix of an accepted input is not `ERR-CODEC-TRUNCATED`; an error points past the input |
| `envelope` | A reader configuration byte (expected schema, limits, whether to reseal), a payload type selector, then an envelope | opening or decoding panics; an opened envelope's payload is not its declared length; a decoded payload is not canonical; a payload error points outside the payload |

## Running

From this directory, where `rust-toolchain.toml` pins the nightly toolchain CI uses, with `cargo-fuzz` 0.13.2:

```sh
python3 seeds.py seeds    # seed corpora from spec/vectors/codec/
cargo fuzz run decode corpus/decode seeds/decode -- -max_total_time=600
cargo fuzz run envelope corpus/envelope seeds/envelope -- -max_total_time=600
```

libFuzzer adds new inputs to the first directory and only reads the others. A failing input is saved under `artifacts/<target>/`; `cargo fuzz run <target> <file>` replays it.

On macOS 26, AddressSanitizer in the pinned nightly deadlocks before the first input runs. Fuzz there with `--sanitizer none`, or with a newer nightly (`cargo +nightly fuzz run …`).

## In CI

`.github/workflows/fuzz.yml` checks this crate's format and lints, then fuzzes each target of `fuzz_targets/` for 10 minutes under AddressSanitizer, seeded as above. It runs on pull requests and pushes to `main` that touch `tv-codec`, `tv-field`, `tv-testkit`, this directory or the codec vectors, and on demand with a chosen duration. A new target needs only its file in `fuzz_targets/` and its `[[bin]]` entry, of the same name, in `Cargo.toml`.
