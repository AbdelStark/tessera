# Conformance vectors

Data that the prover workspace, the independent verifier and the Python package each check with their own code ([RFC-0001 'Shared artifacts that are not code'](../../docs/rfcs/RFC-0001-workspace-and-verifier-independence.md#shared-artifacts-that-are-not-code)). Agreement on these files is agreement through the specification. A file here changes only with the code or oracle that writes it, and the verifier's tests must pass on it unchanged, or its code changes in the same pull request with a second reviewer.

| Set | What it pins | Written by |
|---|---|---|
| [`anchor/`](anchor/README.md) | The anchor's frames, RFC-0015 with RFC-0059: commitments, session-open frames, 41-byte headers, plaintexts, AES-128-CTR ciphertexts, frame digests and digest-log entries | independent Python AES-128 (FIPS-197), CTR checked against SP 800-38A F.5.1, and `hashlib`'s SHA-256 |
| [`anchor-graphs/`](anchor-graphs/README.md) | RFC-0066's SHA-256 circuit node by node, with RFC-0072's public tensors and sizes, for the key commitment at `M = 1` and `M = 3`, one and three public messages and two blocks with committed bytes; its public inputs from bytes for FIPS 180-4's examples, CAVP's messages and `anchor/`'s key commitments and frames | independent Python from RFC-0066 and RFC-0072, its SHA-256 checked against `hashlib`, CAVP and FIPS 180-4, and an interpreter of the graphs |
| [`anchor-relations/`](anchor-relations/README.md) | RFC-0072's public factors of the anchor relations: `ψ̃` and `LT`, `ConstAdd`'s shift `ĉ`, the selection kernels `S̃` and `K̃`, recomposition's weights and zero-extensions, the `BitFn` polynomials, and recomposition's identity and final value on honest and false witnesses | independent Python from RFC-0072 and RFC-0066, each closed form checked against the direct extension of its table |
| [`audit/`](audit/README.md) | Audit sampling, RFC-0016 with RFC-0063: a period's candidates and lengths from two periods' logs, rate thresholds, sample seeds, selection values, samples with their encodings and refusals, and first-fit decreasing batches | independent Python, its BLAKE3 `hash/`'s |
| [`blake3/`](blake3/README.md) | BLAKE3's official test vectors | copied unmodified from the BLAKE3 repository |
| [`canary/`](canary/README.md) | The canary model and tokens of the OBS-3 redaction scan, 05 §3: runs of weights whose bytes are distinctive 64-bit values, distinctive token ids and the model's greedy continuation | its oracle, on the models' |
| `claims/` | The conformance models' claim lists for `test-small/1`, RFC-0007 and RFC-0026: each sealed envelope and its digest | `cargo xtask gen-vectors`, from `tessera-compiler`'s emission, after the models' oracle |
| [`codec/`](codec/README.md) | The canonical encoding, RFC-0003: golden values, non-canonical inputs, envelopes; the conformance models' execution plans, RFC-0041; the weight commitment's artifacts, RFC-0040 | `cargo xtask gen-vectors`, from the cases of `tessera-codec` and `tessera-field`, and the plans from `tessera-compiler`'s derivation, after the models' oracle |
| [`constraint-digest/`](constraint-digest/README.md) | CI-only constraint tree framing, exact byte lengths, scalar types and diagnostic hashes (RFC-0007) | independent Python framing and BLAKE3 oracles |
| [`cost-model/`](cost-model/README.md) | The source's `μ` projections and the cost model on them, RFC-0020 | its oracle |
| [`ed25519/`](ed25519/README.md) | Strict Ed25519, RFC 8032 with RFC-0060: crafted keys and signatures for each rule, small-order, mixed-order and non-canonical points among them, each with the first rule it fails; and RFC 8032's Section 7.1 | independent Python from RFC 8032 and RFC-0060, `hashlib`'s SHA-512; the section copied unmodified from the RFC |
| [`encoding/`](encoding/README.md) | The advice commitment's encoding conventions, RFC-0011: its transforms, coset and mask | its oracle |
| `errors/` | The error codes of 04 §3, with their classes and exit codes | kept by hand; a test keeps it, the specification's table and the error enum equal |
| [`gadget-finals/`](gadget-finals/README.md) | Gadget final relations and full comparison arrays, including constants beyond Goldilocks (RFC-0009/0050) | independent integer and direct-cube oracle |
| [`greedy-runtime/`](greedy-runtime/README.md) | Complete per-request catalogs, advice tails, leaf counts and native descriptors (draft RFC-0053) | independent Python public-model/range/advice oracles |
| `field/` | Goldilocks and its quadratic extension, RFC-0002 | its oracle |
| `hash/` | Hash domains, Merkle multiproofs and column digests, RFC-0004; `commitment_id` of commitment descriptors, RFC-0031 | its oracle |
| [`lowering/`](lowering/README.md) | The Llama lowering, RFC-0044: every input, node, constant, shift-class set, weight and table of the conformance models' graphs and of two variants', at several lengths | its oracle, which rebuilds the models with theirs |
| [`manifest/`](manifest/README.md) | Log manifests, RFC-0015 with RFC-0060: a log's chain of three, each period's entries, leaves, root and inclusion proofs, signed message, Ed25519 signature, encoding, envelope and identity | independent Python: RFC 8032's reference Ed25519, `hashlib`, and the BLAKE3 of `hash/` |
| [`model/`](model/README.md) | The encoding of `IntegerModel`, 03 §2: two one-layer models | `cargo xtask gen-vectors`, from `tessera-ir`'s types |
| [`models/`](models/README.md) | Tiny conformance models with the residuals and logits `D` gives on their sequences, 07 §5 | its oracle |
| [`pcs/`](pcs/README.md) | Openings of the advice commitment, RFC-0011 | its oracle |
| [`pcs-streaming/`](pcs-streaming/README.md) | Canonical segmented advice commitments, exact row sets, masked codewords, chain digests and roots, RFC-0011/0046/0047 | its independent oracle |
| [`poly/`](poly/README.md) | Multilinear extensions and their conventions, RFC-0010 | its oracle |
| [`protocol/`](protocol/README.md) | The protocol messages of `tessera-protocol/1`, RFC-0036: encodings, envelopes, the coin toss and refused inputs | its oracle |
| [`ranges/`](ranges/README.md) | RFC-0006 range analysis: one interval per tensor of the lowering's graphs and of `anchor-graphs/`'s, and small graphs that each break one of RFC-0066's and #527's range rules | independent Python, on the `lowering/` and `anchor-graphs/` oracles' graphs |
| [`rfc6962/`](rfc6962/README.md) | RFC 6962 inclusion proofs, for RFC-0015's logs | copied unmodified from transparency-dev/merkle |
| [`runtime-budget/`](runtime-budget/README.md) | Shared tree-level parent populations, raw final fields and instance counts (RFC-0010/0014/0053) | independent literal Python oracle |
| [`scoring/`](scoring/README.md) | The sampler PRF of RFC-0017, `(key, request, position) → u_t`, and the inverse-CDF token of RFC-0009 Lemma 10 at the edges of its bracket | independent Python AES-128 from FIPS-197, checked against its examples |
| [`schedule/`](schedule/README.md) | The stage schedules of the conformance models, RFC-0010 and RFC-0037: each stage's instances, relations, variable counts and points, the leaf stage's claims and the claim pool | `cargo xtask gen-vectors`, from `tessera-compiler`'s derivation, after the models' oracle |
| [`sha256/`](sha256/README.md) | NIST's SHA-256 test vectors (CAVP) | copied unmodified from NIST |
| [`sha512/`](sha512/README.md) | NIST's SHA-512 test vectors (CAVP), for Ed25519 | copied unmodified from NIST |
| [`soundness/`](soundness/README.md) | The soundness calculator, RFC-0014: every row's bound, bits, target and decision for the parameter sets at the 405B statement, at statements that miss one target each, and at random sizes | its oracle |
| [`sumcheck/`](sumcheck/README.md) | Honest sumcheck proofs, RFC-0010 | its oracle |
| [`tamper/`](tamper/README.md) | The SC-4 tamper corpus of `tessera-protocol/1`, 07 §4: mutations of a valid tiny NI proof and recorded live session, one per verifier check, each with its RFC-0019 verdict | its oracle, on base proofs from the `tessera-e2e` example `tamper-base` |
| `tables/` | RFC-0008's table kinds: specifications, digests, rows and lookups at the edges of their keys | its oracle |
| `transcript/` | Fiat–Shamir transcript sessions, RFC-0004 | its oracle |
| [`transcript-schedule/`](transcript-schedule/README.md) | The transcript schedule of `tessera-protocol/1`, RFC-0014: every label of a proof unit in both modes, with the challenges, seeds, indices and nonces drawn at each, for synthetic sizes and the conformance models | its oracle |
| [`weights/`](weights/README.md) | The weight commitment of the conformance models at `test-small/1`, RFC-0040: each matrix's layout, `commitment_id` and root, `R_W` and the `.twr` artifacts; rows encoded at rate 1/4 | its oracle, after the models' |
| [`rsqrt/`](rsqrt/README.md) | Reciprocal-square-root brackets, 55-bit inputs, 32-bit roots and padded rows, RFC-0009 Lemma 7 | independent Python integer oracle |
| [`weights-opening/`](weights-opening/README.md) | Unmasked weight openings, RFC-0012/0040/0049: sparse structural rows versus dense proximity positions, folds, direct column roots, authentication and two-row failure ordering | its independent oracle, written before the weight checker |
| [`advice-opening/`](advice-opening/README.md) | Canonical advice counts and descriptors, original combination-claim grouping/powers, weighted images, and complete micro-model openings (RFC-0011/0031/0037) | its independent oracle, written before the model opening checker |

An oracle is the set's `generate.py`: an independent implementation in standard-library Python, written from the specification, sharing no code with `crates/`, `verifier/` or `python/`. Oracles may import one another, as `models/` takes its tables from `tables/` and its BLAKE3 from `hash/`.

- `cargo xtask gen-vectors` rewrites every set it writes and runs every oracle.
- `cargo xtask gen-vectors --check` does both on a copy and fails on any file that differs. `cargo test` runs the same comparison.
- The oracles need `python3`. They run isolated (`-I -B`), from the repository root.
