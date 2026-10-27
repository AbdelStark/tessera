# Verifier fuzz targets

`cargo-fuzz` targets for the verifier's decoders and proof checks ([RFC-0003](../../docs/rfcs/RFC-0003-canonical-encoding.md#testing-strategy), [RFC-0004](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#testing-strategy), [RFC-0010](../../docs/rfcs/RFC-0010-proof-schedule.md#testing-strategy), [RFC-0019](../../docs/rfcs/RFC-0019-verifier.md#robustness)), and the tests that need a global allocator. This directory is a workspace of its own, so `libfuzzer-sys` and the allocator crate never enter the verifier's dependency graph, lockfile or allowlist.

| Target | Input | It fails when |
|---|---|---|
| `protocol` | Packet selector (the eight `TYPES` in `src/protocol.rs`), limit configuration and envelope/payload bytes | A packet decoder panics, reports an input error outside the input, or accepts bytes different from its decoded fields' encoding |
| `decode` | A type selector byte (`TYPES` in `src/lib.rs`), a limits byte, then a whole input for that type's decoder | a decoder panics; a rejection is a fault of the reader (`ERR-RES-MEMORY`, `ERR-INTERNAL`) or points past the input; an accepted input is not the canonical encoding of its value (INV-TCE-2), or its value is past a limit; a proper prefix of an accepted input is not `ERR-CODEC-TRUNCATED` |
| `envelope` | A reader configuration byte (expected schema, limits, whether to reseal), a payload type selector, then an envelope | opening or decoding panics; a rejection is a fault of the reader, or points outside the input (the payload, for a payload error); an opened envelope is not, byte for byte, the one a writer of the version found emits for its payload; a decoded payload is not canonical, or is past a limit |
| `multiproof` | A tree of 2^0 to 2^10 leaves, the indices to open, then a script of changes to the canonical multiproof (`multiproof_case` in `src/proofs.rs`) | `verify_multiproof` panics, or its result is not exactly the oracle's: an accepted proof that is not the canonical multiproof of its leaves under its root, or a rejection by another check than the first that fails |
| `inclusion` | A log of 1 to 260 entries, a leaf index, then a script of changes to the RFC 6962 inclusion proof (`inclusion_case`) | `verify_inclusion` panics, or its result is not exactly the oracle's: an accepted proof that does not lead from the leaf to the root, or a rejection by another step of RFC 9162 §2.1.3.2 than the first that fails |
| `sumcheck` | A stage of 1 to 4 front-padded products of degree 1 to 7, or a batch of 1 to 4 product trees of heights 1 to 5, some with a zero leaf, with a seed for its tables; then a script of changes to its honest proof (`case` in `src/sumcheck.rs`) | `tv-core`'s sumcheck checks panic, or their result is not the oracle's: an unchanged proof rejected, or verified at other challenges; a changed proof verified with true claims; a structural defect rejected by another check than the first that fails; changed rounds that pass the final check |
| `audit` | A reader selector byte (`AuditSample`, `VerifierLedger` or `VerifierToss`, payload or envelope), a limits byte, then the input (`check` in `src/audit.rs`) | a reader panics; a rejection is a fault of the reader (`ERR-RES-MEMORY`, `ERR-INTERNAL`) or points past the input; an accepted payload does not re-encode, or an accepted envelope reseal, to the input (INV-TCE-2); a proper prefix of an accepted payload is not `ERR-CODEC-TRUNCATED`; an accepted value breaks a rule of RFC-0016 or RFC-0019 'Ledger' read from its accessors, or a sample's report with an empty ledger leaves a request not unproven; a toss is accepted whose share does not open its commitment under the test kit's BLAKE3, or refused with `ERR-PROTO-COINTOSS` when it does (RFC-0067 'The pending toss') |

The decoder targets' oracles judge an accepted value without trusting the decoder that accepted it. `src/lib.rs` re-encodes each value from RFC-0003's encoding table, with rationals in lowest terms and map entries sorted by key, and measures its depth and longest sequence; it panics on what no value of the type can be: a zero denominator, text outside `Ascii`, two equal keys, an unknown tag. A decoder that skips a check thus fails the target on the first input that check should have rejected. The offsets of rejections are pinned by the vectors and by `tv-codec`'s own tests, not here.

The proof targets' oracles, in `src/proofs.rs`, build honest proofs from the specifications with the test kit's reference BLAKE3 and recompute the root of every changed proof in their own way: level by level for a multiproof, where `tv-hash` climbs from leaf to leaf, and by RFC 6962's recursive `PATH` read backwards for an inclusion proof, where `tv-hash` runs RFC 9162's iterative check. They also predict which check rejects a proof, so a check that is skipped or reordered fails the target even when a later one still rejects.

The `sumcheck` stage script uses header bit 1 to select protocol/2 and degrees through ten; header zero retains protocol/1 and degrees through seven. `seeds.py` includes degrees eight, nine and ten with every mutation, and `tests/sumcheck_oracle.rs` exercises them without libFuzzer. Tree scripts retain their existing format and bound.

The `sumcheck` target's proofs come from the test kit's reference prover (`tv_testkit::sumcheck`), which shares no code with `tv-core`, `tv-field` or `tv-hash`. Its oracle, in `src/sumcheck.rs`, predicts the first structural defect of a changed proof in `tv-core`'s order of checks, and the final check of a stage whose evaluations alone changed from the prover's `γ` and last claim. A changed proof whose shape is right and whose rounds changed must fail the final check, which the honest evaluations at another point pass only with probability `(n·d + m)/|Fp2|`. A stage accepted with a changed evaluation, such as two evaluations of one product exchanged, holds a false claim on a table, and the oracle checks that it does. A tree level cannot tell `u_i` from `w_i`, so a batch of trees runs whole, as its caller would join trees of different heights, and its leaf claims are checked against the leaves: a changed batch must be rejected by a level or leave a false leaf claim.

Every test that caps memory, `tests/allocation.rs` and the `tests/*_allocation.rs` files, is a `harness = false` target that runs its one test from its own `main`. The cap counts every thread's allocations, and libtest's harness keeps a thread of its own, whose allocations inside a capped window would abort the process (#501). `cargo test` runs them as it runs the others, but a test-name filter does not select within them.

`tests/allocation.rs` measures and caps memory with the `cap` allocator: a hostile length is rejected before any allocation; thirty nested sequences, each declaring as many elements as 16 MiB allow, stay within 4 MiB; and memory too short for honest values is `ERR-RES-MEMORY`, not a fault of the input. `tests/hash_allocation.rs` counts what `tv-hash` allocates: checking multiproofs and inclusion proofs of every shape the scripts build, hashing, chaining columns and replaying a transcript allocate nothing, and `challenge_indices` exactly its indices. `tests/sumcheck_allocation.rs` counts what `tv-core`'s sumcheck checks allocate: a round of a million values, 131,072 rounds past the last, or a million final evaluations make a check allocate no more than the honest proof of the same batch. `tests/logup_allocation.rs` does the same for the LogUp checks, on a leaf stage of the test kit's reference prover: a roots message, a level's evaluations or a values message of a million values allocates no more than the honest message. `tests/opening_allocation.rs` does the same for the opening checks, on the honest openings of `spec/vectors/pcs/`: the evaluation check allocates nothing, whatever the proof holds; the column check allocates the sorted queries, the leaves, one combined row's coefficients and codeword, the NTT's twiddles and the tables of `eq`, all sized by the draws, the layout and the parameters, so that 10,000 extra columns, a column of a million symbols or 65,536 extra siblings allocate no more than the honest answer; and memory too short for the codeword, equality tables or NTT twiddles is `ERR-RES-MEMORY`, never a verdict on the proof. `tests/sumcheck_oracle.rs` runs the `sumcheck` target's cases and oracle on seeded inputs, so that `cargo test` covers them without the fuzzer; `tests/audit_oracle.rs` does the same for the `audit` target, on the samples of `spec/vectors/audit/`, ledgers built from them and their changed bytes. `tests/weights_oracle.rs` opens each `.twr` artifact of `spec/vectors/weights/` under the `envelope` target's `0x0301` reader and checks that the `WeightCommitmentPublic` echo of its payload is the payload, so that the seeds `seeds.py` draws from those artifacts start the `decode` and `envelope` targets on accepted inputs.

`tests/field_allocation.rs` caps the scratch of all four NTT transforms over `Fp` and `Fp2`. Shortages return `NttError::Memory` without changing the input; exact scratch budgets succeed, singletons allocate nothing, and invalid lengths and zero coset shifts retain their error precedence under a zero allocation budget. The cancellable transforms also stop before allocating under a zero allocator budget.

`tests/runtime_allocation.rs` caps allocations for a request with an appended greedy Range table. Table regeneration and catalog, advice, leaf, split and mixed-native metadata return memory errors under sampled allocation caps; Range/Requant evaluation allocates no heap memory; the measured honest allocation bound remains sufficient. These checks do not establish a complete runtime proof.

`protocol` exercises all eight RFC-0036 packet types as raw payloads and sealed envelopes,
including nested `NiProof` messages. `seeds.py` includes the independent protocol fixtures;
`tests/protocol_oracle.rs` replays the target and byte mutations without libFuzzer.
`tests/protocol_allocation.rs` caps actual decoder/encoder allocations: impossible lengths
allocate nothing, nested impossible columns reserve only the bounded outer vector, and
insufficient memory for honest stages is a reader failure (`ERR-RES-MEMORY`).
These checks establish packet syntax, not proof acceptance.

`tests/driver_deadline_allocation.rs`, also behind `--features test-params`, expires the clock at the opening boundary and measures that phase's additional work. Under the resulting allocator cap, an expired run must return `ERR-RES-TIME`, while an active run must fail with `ERR-RES-MEMORY`; accepting controls surround them. This independent control prevents a driver that ignores NTT cancellation from calibrating its own passing cap. Each call is process-isolated. The test covers one recorded fixture, not a production time or memory bound.

`tests/driver_work_allocation.rs` runs the complete recorded `test-small` NI proof,
including arithmetic and openings, with `--features test-params`. Each allocation
cap runs in its own process: an infallible allocation abort fails the controller,
and a worker that exceeds 30 seconds is killed and fails the test. It checks both
the verifier's memory policy and simulated allocator shortages independently, at powers of two,
the measured peak's neighbours and 256 seeded allocator budgets. Every shortage
must return `Inconclusive(ERR-RES-MEMORY)`; uncapped controls must accept before
and after. The allocator measures additional verification work; caller-owned
decoded inputs and setup are outside that budget. Historical setup peaks are
excluded from the measurement. Policy cases leave the allocator 64 MiB of spare
capacity and assert that the measured work peak stays within the smaller policy;
allocator cases keep the verifier's default policy and limit available allocation.
These are one recorded proof's regressions, not
a production-size memory bound or a deadline test. The feature inherits
`tv-core`'s release-build refusal, and test-small acceptance certifies nothing.
The CI check runs this target explicitly, in addition to the ordinary tests:

```sh
cargo test --features test-params --test driver_work_allocation --locked
```

`sumcheck` also reserves the `GDG` prefix for the ten families in `src/gadgets.rs`.
Its arithmetic oracles enumerate cube equality/public masks, strict triangular sums,
integer comparison prefixes and literal root-digit triples. Model affine cases use
the independent `spec/vectors/gadget-finals/` fixtures, perturb one field or length,
and require the exact gadget/own-input error. Passing evaluations retain every
source and lookup obligation; they are not treated as complete proof acceptance.
`tests/gadgets_oracle.rs` replays all families, short inputs and affine changes.
`seeds.py` adds valid, malformed and nonzero-operand gadget cases to the existing
sumcheck corpus; unprefixed sumcheck/tree scripts keep their format.

## Running

From this directory, where `rust-toolchain.toml` pins the nightly toolchain CI uses, with `cargo-fuzz` 0.13.2:

```sh
cargo test                # allocation tests
python3 seeds.py seeds    # seed corpora from spec/vectors/
cargo fuzz run decode corpus/decode seeds/decode -- -max_total_time=600
cargo fuzz run envelope corpus/envelope seeds/envelope -- -max_total_time=600
cargo fuzz run protocol corpus/protocol seeds/protocol -- -max_total_time=600
cargo fuzz run multiproof corpus/multiproof seeds/multiproof -- -max_total_time=600
cargo fuzz run inclusion corpus/inclusion seeds/inclusion -- -max_total_time=600
cargo fuzz run sumcheck corpus/sumcheck seeds/sumcheck -- -max_total_time=600
```

libFuzzer adds new inputs to the first directory and only reads the others. A failing input is saved under `artifacts/<target>/`; `cargo fuzz run <target> <file>` replays it.

On macOS 26, AddressSanitizer in the pinned nightly deadlocks before the first input runs. Fuzz there with `cargo fuzz run --sanitizer none <target> …`, the flag before the target name (after `--` it would go to libFuzzer), or with a newer nightly (`cargo +nightly fuzz run …`).

## In CI

`.github/workflows/fuzz.yml` checks this crate's format and lints and runs its tests, then fuzzes each target of `fuzz_targets/` for 10 minutes under AddressSanitizer, seeded as above, with a limit of 10 seconds per input. It runs every night from `.github/workflows/nightly.yml`, and on demand with a chosen duration. `ci.yml` runs its check alone, the format, lints and tests without fuzzing, on every pull request that changes code, since no other job compiles this crate (#481). A new target needs only its file in `fuzz_targets/` and its `[[bin]]` entry, of the same name, in `Cargo.toml`; it runs unseeded until `seeds.py` learns its input format.

The greedy boundary gadget family (9) uses the independent literal
`greedy-boundary` fixtures, dense public-coefficient sums and original carry
endpoints. It predicts malformed target/point/value errors before allocating
source obligations; arithmetic success remains subject to graph and PCS checks.
