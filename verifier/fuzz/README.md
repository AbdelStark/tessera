# Verifier fuzz targets

`cargo-fuzz` targets for the verifier's decoders ([RFC-0003](../../docs/rfcs/RFC-0003-canonical-encoding.md#testing-strategy), [RFC-0019](../../docs/rfcs/RFC-0019-verifier.md#robustness)), and the tests that need a global allocator. This directory is a workspace of its own, so `libfuzzer-sys` and the allocator crate never enter the verifier's dependency graph, lockfile or allowlist.

| Target | Input | It fails when |
|---|---|---|
| `decode` | A type selector byte (`TYPES` in `src/lib.rs`), a limits byte, then a whole input for that type's decoder | a decoder panics; a rejection is a fault of the reader (`ERR-RES-MEMORY`, `ERR-INTERNAL`) or points past the input; an accepted input is not the canonical encoding of its value (INV-TCE-2), or its value is past a limit; a proper prefix of an accepted input is not `ERR-CODEC-TRUNCATED` |
| `envelope` | A reader configuration byte (expected schema, limits, whether to reseal), a payload type selector, then an envelope | opening or decoding panics; a rejection is a fault of the reader, or points outside the input (the payload, for a payload error); an opened envelope is not, byte for byte, the one a writer of the version found emits for its payload; a decoded payload is not canonical, or is past a limit |

The oracles judge an accepted value without trusting the decoder that accepted it. `src/lib.rs` re-encodes each value from RFC-0003's encoding table, with rationals in lowest terms and map entries sorted by key, and measures its depth and longest sequence; it panics on what no value of the type can be: a zero denominator, text outside `Ascii`, two equal keys, an unknown tag. A decoder that skips a check thus fails the target on the first input that check should have rejected. The offsets of rejections are pinned by the vectors and by `tv-codec`'s own tests, not here.

`tests/allocation.rs` measures and caps memory with the `cap` allocator: a hostile length is rejected before any allocation; thirty nested sequences, each declaring as many elements as 16 MiB allow, stay within 4 MiB; and memory too short for honest values is `ERR-RES-MEMORY`, not a fault of the input.

## Running

From this directory, where `rust-toolchain.toml` pins the nightly toolchain CI uses, with `cargo-fuzz` 0.13.2:

```sh
cargo test                # allocation tests
python3 seeds.py seeds    # seed corpora from spec/vectors/codec/
cargo fuzz run decode corpus/decode seeds/decode -- -max_total_time=600
cargo fuzz run envelope corpus/envelope seeds/envelope -- -max_total_time=600
```

libFuzzer adds new inputs to the first directory and only reads the others. A failing input is saved under `artifacts/<target>/`; `cargo fuzz run <target> <file>` replays it.

On macOS 26, AddressSanitizer in the pinned nightly deadlocks before the first input runs. Fuzz there with `--sanitizer none`, or with a newer nightly (`cargo +nightly fuzz run …`).

## In CI

`.github/workflows/fuzz.yml` checks this crate's format and lints and runs its tests, then fuzzes each target of `fuzz_targets/` for 10 minutes under AddressSanitizer, seeded as above, with a limit of 10 seconds per input. It runs on pull requests and pushes to `main` that touch `verifier/Cargo.toml`, `tv-codec`, `tv-field`, `tv-testkit`, this directory or the codec vectors, and on demand with a chosen duration. A new target needs only its file in `fuzz_targets/` and its `[[bin]]` entry, of the same name, in `Cargo.toml`; it runs unseeded until `seeds.py` learns its input format.
