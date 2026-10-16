# Verifier fuzz targets

`cargo-fuzz` targets for the verifier's decoders and proof checks ([RFC-0003](../../docs/rfcs/RFC-0003-canonical-encoding.md#testing-strategy), [RFC-0004](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#testing-strategy), [RFC-0010](../../docs/rfcs/RFC-0010-proof-schedule.md#testing-strategy), [RFC-0019](../../docs/rfcs/RFC-0019-verifier.md#robustness)), and the tests that need a global allocator. This directory is a workspace of its own, so `libfuzzer-sys` and the allocator crate never enter the verifier's dependency graph, lockfile or allowlist.

| Target | Input | It fails when |
|---|---|---|
| `decode` | A type selector byte (`TYPES` in `src/lib.rs`), a limits byte, then a whole input for that type's decoder | a decoder panics; a rejection is a fault of the reader (`ERR-RES-MEMORY`, `ERR-INTERNAL`) or points past the input; an accepted input is not the canonical encoding of its value (INV-TCE-2), or its value is past a limit; a proper prefix of an accepted input is not `ERR-CODEC-TRUNCATED` |
| `envelope` | A reader configuration byte (expected schema, limits, whether to reseal), a payload type selector, then an envelope | opening or decoding panics; a rejection is a fault of the reader, or points outside the input (the payload, for a payload error); an opened envelope is not, byte for byte, the one a writer of the version found emits for its payload; a decoded payload is not canonical, or is past a limit |
| `multiproof` | A tree of 2^0 to 2^10 leaves, the indices to open, then a script of changes to the canonical multiproof (`multiproof_case` in `src/proofs.rs`) | `verify_multiproof` panics, or its result is not exactly the oracle's: an accepted proof that is not the canonical multiproof of its leaves under its root, or a rejection by another check than the first that fails |
| `inclusion` | A log of 1 to 260 entries, a leaf index, then a script of changes to the RFC 6962 inclusion proof (`inclusion_case`) | `verify_inclusion` panics, or its result is not exactly the oracle's: an accepted proof that does not lead from the leaf to the root, or a rejection by another step of RFC 9162 §2.1.3.2 than the first that fails |
| `sumcheck` | A stage of 1 to 4 front-padded products of degree 1 to 7, or a batch of 1 to 4 product trees of heights 1 to 5, some with a zero leaf, with a seed for its tables; then a script of changes to its honest proof (`case` in `src/sumcheck.rs`) | `tv-core`'s sumcheck checks panic, or their result is not the oracle's: an unchanged proof rejected, or verified at other challenges; a changed proof verified with true claims; a structural defect rejected by another check than the first that fails; changed rounds that pass the final check |

The decoder targets' oracles judge an accepted value without trusting the decoder that accepted it. `src/lib.rs` re-encodes each value from RFC-0003's encoding table, with rationals in lowest terms and map entries sorted by key, and measures its depth and longest sequence; it panics on what no value of the type can be: a zero denominator, text outside `Ascii`, two equal keys, an unknown tag. A decoder that skips a check thus fails the target on the first input that check should have rejected. The offsets of rejections are pinned by the vectors and by `tv-codec`'s own tests, not here.

The proof targets' oracles, in `src/proofs.rs`, build honest proofs from the specifications with the test kit's reference BLAKE3 and recompute the root of every changed proof in their own way: level by level for a multiproof, where `tv-hash` climbs from leaf to leaf, and by RFC 6962's recursive `PATH` read backwards for an inclusion proof, where `tv-hash` runs RFC 9162's iterative check. They also predict which check rejects a proof, so a check that is skipped or reordered fails the target even when a later one still rejects.

The `sumcheck` target's proofs come from the test kit's reference prover (`tv_testkit::sumcheck`), which shares no code with `tv-core`, `tv-field` or `tv-hash`. Its oracle, in `src/sumcheck.rs`, predicts the first structural defect of a changed proof in `tv-core`'s order of checks, and the final check of a stage whose evaluations alone changed from the prover's `γ` and last claim. A changed proof whose shape is right and whose rounds changed must fail the final check, which the honest evaluations at another point pass only with probability `(n·d + m)/|Fp2|`. A stage accepted with a changed evaluation, such as two evaluations of one product exchanged, holds a false claim on a table, and the oracle checks that it does. A tree level cannot tell `u_i` from `w_i`, so a batch of trees runs whole, as its caller would join trees of different heights, and its leaf claims are checked against the leaves: a changed batch must be rejected by a level or leave a false leaf claim.

`tests/allocation.rs` measures and caps memory with the `cap` allocator: a hostile length is rejected before any allocation; thirty nested sequences, each declaring as many elements as 16 MiB allow, stay within 4 MiB; and memory too short for honest values is `ERR-RES-MEMORY`, not a fault of the input. `tests/hash_allocation.rs` counts what `tv-hash` allocates: checking multiproofs and inclusion proofs of every shape the scripts build, hashing, chaining columns and replaying a transcript allocate nothing, and `challenge_indices` exactly its indices. `tests/sumcheck_allocation.rs` counts what `tv-core`'s sumcheck checks allocate: a round of a million values, 131,072 rounds past the last, or a million final evaluations make a check allocate no more than the honest proof of the same batch. `tests/opening_allocation.rs` does the same for the opening checks, on the honest openings of `spec/vectors/pcs/`: the evaluation check allocates nothing, whatever the proof holds; the column check allocates the sorted queries, the leaves, one combined row's coefficients and codeword, the NTT's twiddles and the tables of `eq`, all sized by the draws, the layout and the parameters, so that 10,000 extra columns, a column of a million symbols or 65,536 extra siblings allocate no more than the honest answer; and memory too short for the codeword is `ERR-RES-MEMORY`, never a verdict on the proof. `tests/sumcheck_oracle.rs` runs the `sumcheck` target's cases and oracle on seeded inputs, so that `cargo test` covers them without the fuzzer.

## Running

From this directory, where `rust-toolchain.toml` pins the nightly toolchain CI uses, with `cargo-fuzz` 0.13.2:

```sh
cargo test                # allocation tests
python3 seeds.py seeds    # seed corpora from spec/vectors/
cargo fuzz run decode corpus/decode seeds/decode -- -max_total_time=600
cargo fuzz run envelope corpus/envelope seeds/envelope -- -max_total_time=600
cargo fuzz run multiproof corpus/multiproof seeds/multiproof -- -max_total_time=600
cargo fuzz run inclusion corpus/inclusion seeds/inclusion -- -max_total_time=600
cargo fuzz run sumcheck corpus/sumcheck seeds/sumcheck -- -max_total_time=600
```

libFuzzer adds new inputs to the first directory and only reads the others. A failing input is saved under `artifacts/<target>/`; `cargo fuzz run <target> <file>` replays it.

On macOS 26, AddressSanitizer in the pinned nightly deadlocks before the first input runs. Fuzz there with `cargo fuzz run --sanitizer none <target> …`, the flag before the target name (after `--` it would go to libFuzzer), or with a newer nightly (`cargo +nightly fuzz run …`).

## In CI

`.github/workflows/fuzz.yml` checks this crate's format and lints and runs its tests, then fuzzes each target of `fuzz_targets/` for 10 minutes under AddressSanitizer, seeded as above, with a limit of 10 seconds per input. It runs on pull requests and pushes to `main` that touch `verifier/Cargo.toml`, `tv-codec`, `tv-core`, `tv-field`, `tv-hash`, `tv-testkit`, this directory, or the codec, hash, RFC 6962 and sumcheck vectors, and on demand with a chosen duration. A new target needs only its file in `fuzz_targets/` and its `[[bin]]` entry, of the same name, in `Cargo.toml`; it runs unseeded until `seeds.py` learns its input format.
