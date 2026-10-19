# Conformance vectors

Data that the prover workspace, the independent verifier and the Python package each check with their own code ([RFC-0001 'Shared artifacts that are not code'](../../docs/rfcs/RFC-0001-workspace-and-verifier-independence.md#shared-artifacts-that-are-not-code)). Agreement on these files is agreement through the specification. A file here changes only with the code or oracle that writes it, and the verifier's tests must pass on it unchanged, or its code changes in the same pull request with a second reviewer.

| Set | What it pins | Written by |
|---|---|---|
| [`blake3/`](blake3/README.md) | BLAKE3's official test vectors | copied unmodified from the BLAKE3 repository |
| `claims/` | The conformance models' claim lists for `test-small/1`, RFC-0007 and RFC-0026: each sealed envelope and its digest | `cargo xtask gen-vectors`, from `tessera-compiler`'s emission, after the models' oracle |
| [`codec/`](codec/README.md) | The canonical encoding, RFC-0003: golden values, non-canonical inputs, envelopes; the conformance models' execution plans, RFC-0041; the weight commitment's artifacts, RFC-0040 | `cargo xtask gen-vectors`, from the cases of `tessera-codec` and `tessera-field`, and the plans from `tessera-compiler`'s derivation, after the models' oracle |
| [`cost-model/`](cost-model/README.md) | The source's `μ` projections and the cost model on them, RFC-0020 | its oracle |
| [`encoding/`](encoding/README.md) | The advice commitment's encoding conventions, RFC-0011: its transforms, coset and mask | its oracle |
| `errors/` | The error codes of 04 §3, with their classes and exit codes | kept by hand; a test keeps it, the specification's table and the error enum equal |
| `field/` | Goldilocks and its quadratic extension, RFC-0002 | its oracle |
| `hash/` | Hash domains, Merkle multiproofs and column digests, RFC-0004; `commitment_id` of commitment descriptors, RFC-0031 | its oracle |
| [`lowering/`](lowering/README.md) | The Llama lowering, RFC-0044: every input, node, constant, shift-class set, weight and table of the conformance models' graphs and of two variants', at several lengths | its oracle, which rebuilds the models with theirs |
| [`model/`](model/README.md) | The encoding of `IntegerModel`, 03 §2: two one-layer models | `cargo xtask gen-vectors`, from `tessera-ir`'s types |
| [`models/`](models/README.md) | Tiny conformance models with the residuals and logits `D` gives on their sequences, 07 §5 | its oracle |
| [`pcs/`](pcs/README.md) | Openings of the advice commitment, RFC-0011 | its oracle |
| [`poly/`](poly/README.md) | Multilinear extensions and their conventions, RFC-0010 | its oracle |
| [`protocol/`](protocol/README.md) | The protocol messages of `tessera-protocol/1`, RFC-0036: encodings, envelopes, the coin toss and refused inputs | its oracle |
| [`rfc6962/`](rfc6962/README.md) | RFC 6962 inclusion proofs, for RFC-0015's logs | copied unmodified from transparency-dev/merkle |
| [`schedule/`](schedule/README.md) | The stage schedules of the conformance models, RFC-0010 and RFC-0037: each stage's instances, relations, variable counts and points, the leaf stage's claims and the claim pool | `cargo xtask gen-vectors`, from `tessera-compiler`'s derivation, after the models' oracle |
| [`sha256/`](sha256/README.md) | NIST's SHA-256 test vectors (CAVP) | copied unmodified from NIST |
| [`soundness/`](soundness/README.md) | The soundness calculator, RFC-0014: every row's bound, bits, target and decision for the parameter sets at the 405B statement, at statements that miss one target each, and at random sizes | its oracle |
| [`sumcheck/`](sumcheck/README.md) | Honest sumcheck proofs, RFC-0010 | its oracle |
| `tables/` | RFC-0008's table kinds: specifications, digests, rows and lookups at the edges of their keys | its oracle |
| `transcript/` | Fiat–Shamir transcript sessions, RFC-0004 | its oracle |
| [`transcript-schedule/`](transcript-schedule/README.md) | The transcript schedule of `tessera-protocol/1`, RFC-0014: every label of a proof unit in both modes, with the challenges, seeds, indices and nonces drawn at each, for synthetic sizes and the conformance models | its oracle |
| [`weights/`](weights/README.md) | The weight commitment of the conformance models at `test-small/1`, RFC-0040: each matrix's layout, `commitment_id` and root, `R_W` and the `.twr` artifacts; rows encoded at rate 1/4 | its oracle, after the models' |
| [`weights-opening/`](weights-opening/README.md) | Unmasked weight openings, RFC-0012/0040/0049: sparse structural rows versus dense proximity positions, folds, direct column roots, authentication and two-row failure ordering | its independent oracle, written before the weight checker |

An oracle is the set's `generate.py`: an independent implementation in standard-library Python, written from the specification, sharing no code with `crates/`, `verifier/` or `python/`. Oracles may import one another, as `models/` takes its tables from `tables/` and its BLAKE3 from `hash/`.

- `cargo xtask gen-vectors` rewrites every set it writes and runs every oracle.
- `cargo xtask gen-vectors --check` does both on a copy and fails on any file that differs. `cargo test` runs the same comparison.
- The oracles need `python3`. They run isolated (`-I -B`), from the repository root.
