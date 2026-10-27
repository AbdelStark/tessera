# 06 — Security

- Status: Accepted for v1 planning
- Governs: threat model, trust boundaries, security properties, secrets handling, disclosure policy
- Related: [RFC-0013](../rfcs/RFC-0013-zero-knowledge.md), [RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md), [RFC-0015](../rfcs/RFC-0015-transcript-anchor.md), [RFC-0019](../rfcs/RFC-0019-verifier.md)

## 1. Assets

| Asset | Owner | Property needed |
|---|---|---|
| Model weights | Prover | Confidentiality against the verifier |
| Prompts, outputs, activations, logits | Prover (and its users) | Confidentiality against the verifier |
| The certified statement | Both | Integrity: soundness of every accepted proof |
| Recorder digest log | Both | Integrity and non-repudiation |
| Verifier trusted base | Verifier | Integrity of code and dependencies |
| Sampler keys, transcript keys, masking randomness, per-proof seeds | Prover | Confidentiality; unpredictability of masks |

## 2. Adversaries

- **ADV-P: malicious prover (OC5, source §2).** Runs arbitrary hardware and software, deviates anywhere, grinds Fiat–Shamir with very large budgets. Cannot break 128-bit collision resistance of the hash, cannot forge recorder signatures, cannot predict randomness the verifier draws live.
- **ADV-V: malicious verifier.** Follows or deviates from the protocol to learn weights, prompts or outputs: chooses its seed share and column queries adversarially, aborts selectively, replays or reorders sessions, requests proofs repeatedly to exhaust the weight-commitment hiding budget.
- **ADV-N: network attacker** on the live channel between the parties.
- **ADV-S: supply-chain attacker** targeting either party's build, especially the verifier's dependencies.

Out of scope: an adversary that subverts the recorder itself (the recorder is a trust anchor of the statement), and physical attacks on the verifier host.

## 3. Trust boundaries

| ID | Boundary | What crosses | Control |
|---|---|---|---|
| TB-1 | Prover node ↔ verifier | Protocol messages, and when they are sent | Zero knowledge; canonical encoding; mutually authenticated TLS 1.3 for live sessions; the release schedule (INV-ARCH-2) |
| TB-2 | Serving engine ↔ recorder | AES-128-CTR encrypted transcript frames | Key commitment recorded at session open |
| TB-3 | Recorder ↔ both parties | Signed `LogManifest`, digest entries | Ed25519 signatures; manifest frozen before sampling |
| TB-4 | Verifier build ↔ upstream code | Dependencies | Allowlist, `cargo-vet` audits, vendored sources, reproducible build |
| TB-5 | Quantization front-end ↔ checkpoint files | Untrusted files | `safetensors` only; no pickle; `torch.load(weights_only=True)` never used on untrusted input |

## 4. Security properties

| ID | Property | Mechanism | Specified in |
|---|---|---|---|
| SP-1 | Soundness ≥ 100 bits for every component made non-interactive; ≥ 64 bits proximity per live audit | 128-bit extension challenges, proof-of-work grinding, live queries after all commitments | RFC-0014 |
| SP-2 | Statistical zero knowledge against ADV-V; leakage limited to the public statement | Coin-tossed seeds, masked sumchecks and extensions, hiding commitments, column and blinding-row budgets | RFC-0013, RFC-0012 |
| SP-3 | Uniqueness upstream of the logits: for a valid statement exactly one witness exists for every value upstream of the logits | Max certificates, brackets, range-checked decompositions, true-integer keys | RFC-0009, RFC-0006 |
| SP-4 | Anchor binding: proven token streams equal recorded streams | AES-CTR and SHA-256 circuits, key commitments, signed manifests | RFC-0015 |
| SP-5 | Sample unpredictability: the prover cannot know which requests will be audited before the manifest is frozen | Coin toss or beacon after freeze | RFC-0016 |
| SP-6 | Verifier independence: no prover code in the verifier's trusted base | Separate workspace, CI isolation gate | RFC-0001, RFC-0019 |
| SP-7 | Claim-list soundness: the verifier rejects any claim list whose lookup keys are not provably true integers in range | Independent range re-derivation in the verifier | RFC-0007, RFC-0019 |
| SP-8 | Bounded verifier resources on hostile input | `ResourceLimits` checked before allocation | RFC-0019 |

## 5. Threats and mitigations

| Threat | Adversary | Mitigation |
|---|---|---|
| Grinding Fiat–Shamir challenges | ADV-P | Challenges in `Fp2` (`\|F\| ≈ 2^128`); in non-interactive mode the seed and query phases each require 20 bits of proof-of-work; in live mode the seed comes from a coin toss |
| Grinding the LogUp challenge at `(N + T)/\|F\| ≈ 2^-90` | ADV-P | Derived from the live seed (one shot) or behind 20-bit proof-of-work (RISK-5, [00 §9](00-overview.md#9-risks)) |
| Committing a non-codeword | ADV-P | Proximity test with `τ` columns drawn after all commitments; unique-decoding analysis |
| Exploiting a wrapped lookup key to admit a false value | ADV-P | Compiler proves each key a true integer; the verifier independently re-derives ranges from the claim list (SP-7) |
| Committing weights, classes, multipliers or padding outside their domains | ADV-P | Uniqueness holds for any committed values. Fidelity rests on `R_W`'s publisher, and `commit-weights` refuses an out-of-domain `D` ([RFC-0048](../rfcs/RFC-0048-weight-trust.md)) |
| Substituting tables | ADV-P | The verifier regenerates every table from its definition with its own code and compares digests |
| Malleable encodings to fork the transcript | ADV-P | Canonical decoding; the transcript absorbs canonical bytes only |
| Proving a different stream than the one served | ADV-P | Transcript anchor; per-session key commitment in the signed log |
| Choosing which requests to prove | ADV-P | Every sampled request must be proven; the sample derives from randomness drawn after the freeze; an unproven sampled request is reported as such |
| Steering outputs to carry a covert channel | ADV-P | Exact mode: `Û = 0`, and the sampling stream's only per-request input is a keyed class of `b` public bits, under a key whose commitment the verifier pins ahead of its window ([RFC-0061](../rfcs/RFC-0061-sampling-randomness.md)). That caps the choice of stream at `b` bits per logged request under one pin, and the verifier's default `b = 0` removes the per-request choice. The key, derived with a nonce its issuing verifier draws and used for one fixed-length window, is neither the prover's nor that verifier's to choose. The prompt, the output length, request boundaries, the temperature, which windows carry temperature traffic, the pin near a rotation and the pins a held prompt spans remain the serving stack's ([RFC-0061 'What remains the prover's'](../rfcs/RFC-0061-sampling-randomness.md#what-remains-the-provers)). Bounded mode: `Û` bounds the channel capacity |
| Learning the sampling stream from outputs, to read which slice of `D`'s distribution each output fell in | ADV-V | At `b = 0` every request under a key shares one stream, which an observer holding `D`'s distributions or a close proxy learns from a few responses. Larger `b` spreads requests over `2^b` keyed streams, at `b` bits of choice for the prover ([RFC-0061 'Drawbacks'](../rfcs/RFC-0061-sampling-randomness.md#drawbacks)) |
| Reseeding a `Round1` to collect two transcripts over the same commitments | ADV-V | The prover answers at most one seed per `Round1`; aborts force fresh commitments and masks (INV-PROTO-1) |
| Adaptive column queries to reconstruct weights | ADV-V | Padding columns make any `≤ P_W` opened columns perfectly hiding; the budget state machine refuses to exceed it, except after a rollback of the epoch record with every file that would refute it, or from a copy of the model entry, which the prover cannot detect (draft [RFC-0073](../rfcs/RFC-0073-weight-mask-expansion.md#what-the-prover-cannot-detect)); refresh with a linking proof |
| Reading weights from combined rows | ADV-V | One fresh `Fp2` blinding row, two committed rows, per revealed combination ([RFC-0013](../rfcs/RFC-0013-zero-knowledge.md#hiding-openings)) |
| Timing side channel | ADV-V | Control flow depends only on public data, except the rejection of a stream of RFC-0004's words, whose pattern is, in the random-oracle model, independent of every value any stream yields and of every other secret (INV-ARCH-2's one exception, [RFC-0004](../rfcs/RFC-0004-hashing-and-transcript.md#hash-function-and-domains), draft [RFC-0073](../rfcs/RFC-0073-weight-mask-expansion.md#the-stream)). Every message a verifier can time is released on a public schedule, which hides the data-dependent memory accesses of multiplicity counting and table reads. An overrun reveals at most which slot a message took. Phase timings stay on the prover's side (INV-ARCH-2, [RFC-0018 'Release schedule'](../rfcs/RFC-0018-gpu-prover-runtime.md#release-schedule)). An attacker sharing the prover's hardware is outside this model. |
| Session hijack or replay | ADV-N | Mutually authenticated TLS 1.3; session ids bound into the transcript; manifest and batch digests in the statement |
| Verifier denial of service | ADV-P | Size and memory limits checked before allocation; fuzzed decoders |
| Malicious checkpoint | ADV-S | Safetensors-only loading; shape and dtype validation against `ArchSpec` |
| Dependency compromise | ADV-S | Lockfiles, `cargo-deny` advisories and licenses, `cargo-vet` for the verifier, reproducible verifier builds |

## 6. Cryptographic assumptions

- BLAKE3 with 256-bit output: collision resistance at 128 bits; modeled as a random oracle for Fiat–Shamir.
- AES-128 as a pseudorandom permutation (transcript encryption, sampler PRF); SHA-256 collision resistance (recorder digests).
- SHA-256's compression function as a PRF keyed by its chaining value, and keyed by a message block whose other bytes are fixed (the dual-PRF assumption of HMAC's analysis): the sampler key's derivation and the stream class. A key used under several labels, or by SHA-256 and AES, is taken to be independent in each use, as it is when the compression function is modeled as ideal ([RFC-0061](../rfcs/RFC-0061-sampling-randomness.md#security-argument)).
- Ed25519 (RFC 8032) existential unforgeability for recorder signatures.
- Schwartz–Zippel over `Fp2` for all polynomial identity tests.

The proof system itself relies only on hash functions and information-theoretic arguments, which are believed to resist quantum adversaries at reduced margins. Recorder signatures are not post-quantum (OQ-SEC-1).

## 7. Secrets handling

- Randomness comes from the operating system CSPRNG (`getrandom`). Test builds may pin masking randomness only through the `test-rng` feature, which release builds reject.
- Secret values live in `Secret<T>` wrappers, zeroized on drop ([05 §3](05-observability.md#3-data-classification-and-redaction)). What the proving engines derive from them, such as bound tables, partial products, contractions and tree levels, lives in `SecretBuf`, which clears every allocation it releases, when it drops and when it grows (OBS-6). An engine reads a secret table only through an entry point that takes it by reference to its `Secret`, in a file `secret-access.toml` lists (OBS-1).
- Secret artifacts (`IntegerModel`, `WeightCommitmentState`, the weights' codeword store ([RFC-0012](../rfcs/RFC-0012-weight-commitment.md#column-answers)), per-proof seed files, and under draft [RFC-0073](../rfcs/RFC-0073-weight-mask-expansion.md#the-model-entry) the weights' refresh key, kept apart from every state file) are written with mode 0600 through atomic rename. Encryption at rest is the operator's responsibility (encrypted volumes); v1 has no built-in key-management integration.
- The per-proof secret seed is persisted before `Round1` is sent so that crash recovery reproduces identical messages ([RFC-0018](../rfcs/RFC-0018-gpu-prover-runtime.md#crash-consistency)), and deleted after `Round3Response` is sent.

## 8. Disclosure policy

`SECURITY.md` at the repository root states the policy:

- Report privately through GitHub security advisories on this repository. Do not open public issues for vulnerabilities.
- Acknowledge within 3 business days; triage within 10 business days.
- Severity: any soundness or zero-knowledge break, or any verifier acceptance of an invalid proof, is critical. Resource exhaustion in the verifier is high. Prover-side crashes on valid input are medium.
- Fixes for critical issues ship as a patch release with a new protocol version if the fix changes verification.
- Reporters are credited unless they ask otherwise.

## 9. Security gates

| Gate | When | Requirement |
|---|---|---|
| Threat-model review | Each milestone | This document updated; new trust boundaries listed |
| Simulator argument | Before v0.4 implementation | Written and reviewed ([RFC-0013](../rfcs/RFC-0013-zero-knowledge.md#simulator-argument)) |
| Verifier fuzzing | v1.0 | Every decoder and check fuzzed for 24 CPU-hours without findings |
| External review | Before v1.0 | Protocol and verifier reviewed by a party outside the implementation team |

## 10. Open questions

- **OQ-SEC-1.** Should the digest log require a post-quantum signature scheme (ML-DSA-65, FIPS 204) in addition to Ed25519? Owner: @AbdelStark. Resolve by v1.0 after surveying recorder hardware support. Until then the log format reserves a scheme tag for it ([RFC-0015](../rfcs/RFC-0015-transcript-anchor.md#signatures)).
