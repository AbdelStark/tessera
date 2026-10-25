# Tessera Specification (v1)

- Status: Accepted for v1 planning
- Date: 2026-10-15
- Source of intent: [`proofs-of-inference-sota-path-forward.md`](proofs-of-inference-sota-path-forward.md)
- Change process: RFCs ([09 §6](docs/spec/09-release-and-versioning.md#6-contribution-workflow))

## Executive summary

Tessera proves that a committed integer language model `D` produced a recorded token stream. The proof is sound against a prover that controls every piece of hardware and software, hides weights, activations, prompts and outputs from a counterparty verifier, and costs a small multiple of the inference it certifies. It is built for verification between mutually distrustful parties, such as states auditing each other's AI inference, and for anyone who needs that statement with the same rigor.

The design is a sumcheck/GKR proof over the Goldilocks field in which matmuls and attention products are never committed. Only lookup outputs, per-row scalars and weights are. Every layer, head and query block is proven in a constant number of batched stages under one seed. Softmax is pinned by a max certificate at one lookup, one commitment and one product-tree leaf per score. A persistent weight commitment is opened by the same pass that folds the weights for the sumcheck. An independent verifier rebuilds every constraint from a public claim list and shares no code with the prover.

The source projects a proof multiplier of 3.2 at 1K tokens and 18.8 at 16K for Llama-3.1-405B on one 8×H100 node. This corpus finds four costs the source's model leaves out: rescale decompositions (RISK-2), witness regeneration (RISK-9), stages larger than memory (RISK-10), and the table side of LogUp. It restates the projection as about 13.8 to 15.7 at 1K and 29 to 31 at 16K. With the unmeasured column answers and the memory plans' arithmetic at their upper estimates, it is up to 19.1 and 40.5. The leaf stage is proven in token groups with one merge ([RFC-0054](docs/rfcs/RFC-0054-token-groups.md)); streaming it in one batch would have added 53 to 235. The reduction stages larger than memory answer their first rounds from a window ([RFC-0055](docs/rfcs/RFC-0055-windowed-stages.md)), one more regeneration per window where streaming round by round would have cost 14 to 25. Stage 1's score instances, which span the score tensor, are proven without building it ([RFC-0056](docs/rfcs/RFC-0056-native-working-sets.md)): 45 GB at 16K of context instead of 141 TB. Both remain four orders of magnitude below the best published frontier-scale proof. Every figure is a projection until milestone v0.1 measures the unit costs on H100 and records a go/no-go decision against a rule fixed in advance ([RFC-0020](docs/rfcs/RFC-0020-cost-model.md#decision-rule)).

Sub-10% overhead is an amortized property, reached by recording every request and proving an unpredictable sample in full. It is not a per-request cost, and this corpus never claims it is.

## Reading order

1. [00 Overview](docs/spec/00-overview.md): thesis, statement, goals, non-goals, success criteria, risks.
2. [01 Architecture](docs/spec/01-architecture.md): components, dependency rules, lifecycle.
3. The RFC for the component you are building (index below).
4. [03 Data model](docs/spec/03-data-model.md), [04 Error model](docs/spec/04-error-model.md) and [07 Testing strategy](docs/spec/07-testing-strategy.md) for the contracts every change must respect.

## Specification sections

| Section | Content |
|---|---|
| [00 Overview](docs/spec/00-overview.md) | Thesis, parties, statement, modes, goals, non-goals, success criteria, hardest problems, risk register |
| [01 Architecture](docs/spec/01-architecture.md) | Four lanes, components, dependency rules, artifacts, prover phases, determinism invariants, deployment |
| [02 Public API](docs/spec/02-public-api.md) | Stability tiers, CLIs, exit codes, Rust and Python interfaces, parameter sets, compatibility rules |
| [03 Data model](docs/spec/03-data-model.md) | Scalar, model, circuit and statement types; artifact catalog; invariants; schema versioning |
| [04 Error model](docs/spec/04-error-model.md) | Verdicts, reject reasons, stable error codes, failure modes and responses, retry rules |
| [05 Observability](docs/spec/05-observability.md) | Logging, data classification and redaction, metrics, tracing, run manifests, verifier audit trail |
| [06 Security](docs/spec/06-security.md) | Assets, adversaries, trust boundaries, security properties, threats and mitigations, disclosure policy |
| [07 Testing strategy](docs/spec/07-testing-strategy.md) | Test pyramid, soundness and uniqueness tests, differential tests, ZK and ML tests, CI gates |
| [08 Performance budget](docs/spec/08-performance-budget.md) | Cost model, unit-cost targets, end-to-end targets, memory and proof-size budgets, regression gates |
| [09 Release and versioning](docs/spec/09-release-and-versioning.md) | Release trains, protocol and schema versions, deprecation, changelog, contribution, licensing, platforms |
| [10 Glossary](docs/spec/10-glossary.md) | Canonical terms |

## RFC index

| RFC | Title | Milestone |
|---|---|---|
| [RFC-0001](docs/rfcs/RFC-0001-workspace-and-verifier-independence.md) | Workspace layout, language boundaries, and verifier independence | v0.1 |
| [RFC-0002](docs/rfcs/RFC-0002-field-arithmetic.md) | Goldilocks field and quadratic extension | v0.1 |
| [RFC-0003](docs/rfcs/RFC-0003-canonical-encoding.md) | Canonical encoding and schema-versioned artifacts | v0.1 |
| [RFC-0004](docs/rfcs/RFC-0004-hashing-and-transcript.md) | Hashing, Merkle trees, and the Fiat–Shamir transcript | v0.1 |
| [RFC-0005](docs/rfcs/RFC-0005-integer-model.md) | Integer model D: W8A8 quantization, calibration, and fidelity evaluation | v0.2 |
| [RFC-0006](docs/rfcs/RFC-0006-compiler-ir.md) | Compiler IR: typed integer dataflow, range analysis, and gadget selection | v0.2 |
| [RFC-0007](docs/rfcs/RFC-0007-claim-list.md) | Public claim list and statement schema | v0.2 |
| [RFC-0008](docs/rfcs/RFC-0008-logup-and-tables.md) | LogUp-GKR lookups and the deterministic table registry | v0.1–v0.2 |
| [RFC-0009](docs/rfcs/RFC-0009-gadgets.md) | Uniqueness gadgets | v0.3 |
| [RFC-0010](docs/rfcs/RFC-0010-proof-schedule.md) | Batched constant-depth sumcheck/GKR proof schedule | v0.3 |
| [RFC-0011](docs/rfcs/RFC-0011-advice-commitment.md) | Streaming Ligero-family commitment for per-proof advice | v0.1–v0.3 |
| [RFC-0012](docs/rfcs/RFC-0012-weight-commitment.md) | Persistent weight commitment opened by the folding pass | v0.3–v0.4 |
| [RFC-0013](docs/rfcs/RFC-0013-zero-knowledge.md) | Zero knowledge: masked sumchecks, masked extensions, and hiding openings | v0.4 |
| [RFC-0014](docs/rfcs/RFC-0014-interaction-and-soundness.md) | Interaction model, soundness budget, and parameter selection | v0.3–v1.0 |
| [RFC-0015](docs/rfcs/RFC-0015-transcript-anchor.md) | Transcript anchor and recorder digest log | v0.4 |
| [RFC-0016](docs/rfcs/RFC-0016-audit-sampling.md) | Commit-then-sample audit protocol | v0.4 |
| [RFC-0017](docs/rfcs/RFC-0017-scoring.md) | Scoring and the unexplained-information bound Û | v0.3–v0.4 |
| [RFC-0018](docs/rfcs/RFC-0018-gpu-prover-runtime.md) | Witness engine and GPU prover runtime | v0.2–v1.0 |
| [RFC-0019](docs/rfcs/RFC-0019-verifier.md) | Independent verifier | v0.3–v1.0 |
| [RFC-0020](docs/rfcs/RFC-0020-cost-model.md) | Cost model and unit-cost measurement gate | v0.1 |
| [RFC-0022](docs/rfcs/RFC-0022-saturating-residual.md) | A saturating residual stream and complete norms (amends RFC-0005, RFC-0006, RFC-0009) | v0.2 |
| [RFC-0023](docs/rfcs/RFC-0023-norm-precision.md) | The norm's precision: choosing κ per model (amends RFC-0005) | v0.2 |
| [RFC-0024](docs/rfcs/RFC-0024-wide-requantization.md) | Wide requantization: keys too wide for their shift (amends RFC-0005, RFC-0006, RFC-0009) | v0.2 |
| [RFC-0025](docs/rfcs/RFC-0025-affine-sites.md) | Affine rescale sites in the claim list (amends RFC-0005, RFC-0006, RFC-0007, RFC-0010) | v0.2 |
| [RFC-0026](docs/rfcs/RFC-0026-claim-list-layout.md) | The claim list's commitment layout, scoring and anchor (amends RFC-0007, RFC-0012) | v0.2 |
| [RFC-0027](docs/rfcs/RFC-0027-logits-exponent.md) | The scale of `D`'s logits (amends RFC-0007, RFC-0026) | v0.2 |
| [RFC-0028](docs/rfcs/RFC-0028-balanced-split.md) | A balanced split point for `RequantSplit` (amends RFC-0006, RFC-0009) | v0.2 |
| [RFC-0029](docs/rfcs/RFC-0029-table-set.md) | The claim list is the table set: `TableSetManifest` retired (amends RFC-0006) | v0.2 |
| [RFC-0030](docs/rfcs/RFC-0030-masked-exponentials.md) | `SoftmaxRow` pins its masked exponentials to zero (amends RFC-0006, RFC-0007, RFC-0009, RFC-0010) | v0.3 |
| [RFC-0031](docs/rfcs/RFC-0031-commitment-id.md) | `commitment_id`, where every column chain starts (amends RFC-0004, RFC-0011, RFC-0012, RFC-0019) | v0.3 |
| [RFC-0032](docs/rfcs/RFC-0032-zero-knowledge-tree-levels.md) | Zero knowledge of the tree levels and the LogUp roots (amends RFC-0008, RFC-0010, RFC-0013, RFC-0014) | v0.4 |
| [RFC-0033](docs/rfcs/RFC-0033-tree-level-rejections.md) | Rejection reasons for the leaf stage's sumchecks (amends 04, RFC-0010, RFC-0019) | v0.3 |
| [RFC-0034](docs/rfcs/RFC-0034-causal-blocks.md) | Causal blocks: the score tensors without their masked half (amends RFC-0006, RFC-0011, RFC-0018, RFC-0020, RFC-0030; resolves RISK-11) | v0.3 |
| [RFC-0035](docs/rfcs/RFC-0035-padded-rows.md) | Padded rows satisfy every row gadget (amends RFC-0009, RFC-0010, RFC-0011, RFC-0014, RFC-0017) | v0.3 |
| [RFC-0036](docs/rfcs/RFC-0036-protocol-messages.md) | Protocol messages and session state machines (amends RFC-0014, 01, 02, 03, 04) | v0.3 |
| [RFC-0037](docs/rfcs/RFC-0037-stage-schedule.md) | Stage schedule: stacked layers, split claims and the claim pool (amends RFC-0006, RFC-0010, RFC-0011, RFC-0014, RFC-0036) | v0.3 |
| [RFC-0038](docs/rfcs/RFC-0038-uniform-layers.md) | One template for every layer: every layer's sites rescaled alike (amends RFC-0007, RFC-0022, RFC-0037) | v0.3 |
| [RFC-0039](docs/rfcs/RFC-0039-statement-fields.md) | Statement fields: the payload of `Reject(StatementMismatch)`, one variant per check (amends 04, RFC-0007) | v0.3 |
| [RFC-0040](docs/rfcs/RFC-0040-weight-commitment-artifacts.md) | Weight commitment artifacts: `WeightCommitmentPublic` and `WeightCommitmentState`, every weight's position, the verifier's matrix roots (amends RFC-0012, RFC-0019, RFC-0025, RFC-0026, 02, 03) | v0.3 |
| [RFC-0041](docs/rfcs/RFC-0041-execution-plan.md) | Tiles and the execution plan: what each tile holds, the production order, the plan as claim list and tiles (amends RFC-0006, RFC-0011, RFC-0029, 03) | v0.3 |
| [RFC-0042](docs/rfcs/RFC-0042-silu-exponents.md) | SiLU exponents in the claim list: each layer's `Silu` table determined by public data (amends RFC-0007, RFC-0026, 03) | v0.3 |
| [RFC-0043](docs/rfcs/RFC-0043-rejection-payloads.md) | Rejection payloads: `ArtifactKind`, `PowPhase`, `AnchorFailure` and `ParamSetId`; the anchor's and the audit sample's rejections (amends 04, RFC-0015, RFC-0016, RFC-0019, RFC-0039) | v0.3 |
| [RFC-0044](docs/rfcs/RFC-0044-lowering-nodes.md) | The Llama lowering, node by node: every input, node, constant, shift-class set and table by id (amends RFC-0006) | v0.3 |
| [RFC-0045](docs/rfcs/RFC-0045-leaf-layout.md) | The leaf layout: each LogUp instance's blocks, its lookup tree's leaf order, one point per block size, and the leaf stage's split values (amends RFC-0008, RFC-0037) | v0.3 |
| [RFC-0046](docs/rfcs/RFC-0046-advice-segments.md) | Advice segments and short query blocks: the layers and rows each segment holds, an attention tile's `Q' = min(Q, N)` queries, and the refusal of rows too short for a key (amends RFC-0007, RFC-0011, RFC-0034) | v0.3 |
| [RFC-0047](docs/rfcs/RFC-0047-advice-order.md) | The advice layout's order, size and sequence length: padded axes first, `ν`, at most `2^{32}` rows, heads grouped by a power of two, `n = prompt_len + output_len` (amends RFC-0007, RFC-0011, RFC-0034, 03) | v0.3 |
| [RFC-0049](docs/rfcs/RFC-0049-weight-proximity.md) | The weights' proximity point before the stages: `pcs/weights/{T}/xi` right after the seed in protocol/1, so one weight pass yields folds and combined rows; coefficients over committed rows; the calculator's protocol/1 counts (amends RFC-0012, RFC-0014) | v0.3 |
| [RFC-0048](docs/rfcs/RFC-0048-weight-trust.md) | What the weights root vouches for: uniqueness holds for any committed values, while fidelity and totality rest on `R_W`'s publisher (amends 00, 06, RFC-0006, RFC-0009, RFC-0040) | v0.3 |
| [RFC-0050](docs/rfcs/RFC-0050-gadget-advice-columns.md) | Canonical gadget advice columns: public roles/domains, exact Rsqrt products, comparison arrays, complete range requests and explicit migration (amends RFC-0006, RFC-0009, RFC-0022, RFC-0041) | v0.3 |
| [RFC-0051](docs/rfcs/RFC-0051-soundness-target-exit-code.md) | Soundness target exit code: exit 10 when `tessera params check` or `tessera-verify params` misses a target while still printing the report (amends 02, RFC-0014) | v0.3 |
| [RFC-0052](docs/rfcs/RFC-0052-channel-params-claims.md) | Opening several claims on `ChannelParams`: one evaluation row per reading site in pool order, folded in a single pass (amends RFC-0012, RFC-0036, RFC-0037, RFC-0045, RFC-0049) | v0.3 |
| [RFC-0053](docs/rfcs/RFC-0053-greedy-runtime-boundary.md) | Original-source exact-greedy runtime boundary, public-span range and canonical maximum/native obligations | v0.3 |
| [RFC-0054](docs/rfcs/RFC-0054-token-groups.md) | Token groups: a leaf stage larger than memory proven group by group in memory, the table trees streamed without regeneration, and one merge sumcheck back to one point per tensor; two regenerations at any size (amends RFC-0008, RFC-0010, RFC-0014, RFC-0018, RFC-0020, RFC-0036) | v0.3 |
| [RFC-0055](docs/rfcs/RFC-0055-windowed-stages.md) | Windowed rounds: a reduction stage larger than memory answers its first rounds from a grid accumulated in one pass and folds its tables in a second; two regenerations at any size, messages unchanged; the commitment report's stage tables and the cost model's stage plan (amends RFC-0006, RFC-0018, RFC-0020, RFC-0054) | v0.3 |
| [RFC-0056](docs/rfcs/RFC-0056-native-working-sets.md) | Native relations in the working sets: the compiler's report counts the prover's native gadget relations; `SoftmaxRow`'s score instances proven from the INT8 queries and keys and the causal mask's closed form, never the score tensor; identical instances share their tables; a window per eight own variables; messages unchanged (amends RFC-0006, RFC-0018, RFC-0020, RFC-0055) | v0.3 |
| [RFC-0057](docs/rfcs/RFC-0057-one-proof-unit.md) | One proof unit per proof: at the reference statement no budget needs layer-range units after token groups, windows and native working sets, and each unit only adds combined rows and a table side, a set of requests also a weight fold; a batch too large for one proof is several proofs, which nothing links; whole 405B requests fit one proof up to 64K tokens in memory at best and about 24K within the proof-size gate, longer ones and the 128K projections await #470 (amends RFC-0018, RFC-0031, RFC-0049, RFC-0054, RFC-0056, 08 §3 and §5) | v0.3 |
| [RFC-0058](docs/rfcs/RFC-0058-inverse-cdf-columns.md) | InverseCdf columns: the vocabulary sums `Z`, `C_<` and `e_ŷ`, radix-`2^16` limbs of the draw, the partition and the prefix, two comparisons of `k + 2` limbs, the committed values' and range requests' order and exact counts, the partition's bound `vocab·2^σ`, and the gadget's own padding zero-check and bounded-mode OneHot relations (amends RFC-0009, RFC-0035) | v0.4 |
| [RFC-0059](docs/rfcs/RFC-0059-frame-headers-and-nonces.md) | Frame headers, nonces and the session counter: the 41-byte header carries `frame_seq`, which the frame's log entry copies, so a verifier can build a sampled frame's AES-128-CTR counter blocks; big-endian SP 800-38A counter blocks; `TokenFrame`'s encoding, with its 22-byte prefix a public input; the commitments' exact inputs with 16-byte salts, and the sampler commitment opened in a circuit; random session identifiers; INV-ANC-3 by construction (amends RFC-0015, RFC-0043) | v0.4 |
| [RFC-0060](docs/rfcs/RFC-0060-log-manifests.md) | Log manifests, their chain and inclusion proofs: a manifest's identity is its envelope digest, which `prev_manifest` and a statement's `manifest_digest` hold; consecutive manifests continue the log's entries and periods; RFC 9162's tree, empty periods included, and its inclusion proofs; RFC 8032 Ed25519 verified strictly, with scheme 1 refused (amends RFC-0015) | v0.4 |
| [RFC-0061](docs/rfcs/RFC-0061-sampling-randomness.md) | Sampling randomness under a pinned key: a keyed stream class of `b` public bits replaces `request_id`, so the serving stack's choice of stream is at most `b` bits per logged request under one pin, whatever it does to the prompt; the reference and verifier default `b = 0` leaves no per-request input, at the cost of one stream per key; the PRF's key is derived from the prover's committed base key, a nonce the issuing verifier draws when it pins it, `b` and the start of a fixed-length window, so neither the prover nor that verifier chooses it; a request falls within one pin's window, up to a grace, and a batch within one pin; the stack still chooses which windows carry temperature traffic; the session-open frame keeps its key commitment only; `AnchorFailure::SamplerKey` (amends RFC-0015, RFC-0016, RFC-0017, RFC-0019, RFC-0043, RFC-0059, 04 §2, 06 §5–6) | v0.4 |
| [RFC-0062](docs/rfcs/RFC-0062-period-log-sessions.md) | The period log and its sessions: the verifier checks the period's whole log against the manifest's root, count and `seq` range once, before it derives the sample, keeps that copy, and reads every request's entries from it; the recorder carries open sessions into each period with a `SessionCarry` entry; a session used in a period has exactly one open or carry before its other entries, one frame per position and no counter used twice; the statement's lengths count all the request's frames in the period; a statement needs no chain of manifests, while the audit links them to find unaudited periods; `AnchorFailure::Session`, `Inclusion` retired (amends RFC-0015, RFC-0016, RFC-0019, RFC-0043, RFC-0059, RFC-0060, RFC-0061, 04 §2–3) | v0.4 |
| [RFC-0063](docs/rfcs/RFC-0063-request-boundaries.md) | Request boundaries in the log: the serving stack ends each request with a cleartext 28-byte request-end frame, which the recorder logs as `RequestEnd`; every request with an end in a period is a candidate of that period, and one with a frame and no end in time is an unterminated candidate, so every logged frame lies in a request the sample can draw; a request's frames are all its frames in its end's period and the one before, whose verified log the verifier holds through `prev_manifest`, with one end, every frame before it, each position once, the session's rules in both periods, and its inbound frames before its outbound ones, which the recorder's ingress sequencing guarantees for honest requests (amends RFC-0015, RFC-0016, RFC-0019, RFC-0043, RFC-0061, RFC-0062) | v0.4 |
| [RFC-0064](docs/rfcs/RFC-0064-weight-linking-proof.md) | The weight linking proof opens the difference: one combined row per matrix over both epochs' commitments, coefficients `eq(ρ_T, ·)` on the old rows and their negatives on the new, a blinding pair from each epoch, checked against the public claim 0 at the same columns of both, with each commitment's own proximity row; it reveals no evaluation of the weights (R3 of the simulator argument); each epoch's seed derived from the model's refresh key `K`, so a link is the same bytes however often it is made; the transcript (`weights-link`, `link/{T}/…`), `weights-link/1`, `τ_link = 122` and `g_link = 20` from the weights' layout, `WeightLinkProof`, the verifier's checks with the claim list, a `link` transition whose commit point is closing the old epoch, one successor of each root in the ledger, and the four rows at each end of an epoch (supersedes RFC-0012 'Refresh and linking proof' and 'Failure modes' in part, and RFC-0014's weight-linking row; amends RFC-0004, RFC-0012 'Budget state machine', RFC-0013 'Cost', RFC-0014 'Grinding', 'Transcript schedule' and 'Parameter sets', RFC-0019 'Ledger', RFC-0040, RFC-0043 and 03 §5) | v0.4 |
| [RFC-0065](docs/rfcs/RFC-0065-masked-channel-params.md) | Masked `ChannelParams` claims in `tessera-protocol/2`. Each claim gets its own mask row, with `Z` over its type tensor's coordinates, which is its evaluation row's designated block, and each matrix one blinding pair for its blinding claims. A proof reserves `2·e_T + 4` rows of a matrix with `e_T` claims: six as before, and 18 for `ChannelParams` at `e = 7`. `ChannelParams` has `B_CP = B_W + 2·(e_CP − 1)·⌊(B_W − 4)/6⌋` blinding rows, 12,280 at 405B, so its epoch lasts as long as the others'. Stages reveal type tensors only at points whose coordinates other than the lowest are challenges, and `tessera-protocol/2` admitted only affine `ChannelParams` sites until RFC-0068's leaf stage (supersedes RFC-0052 'Non-Goals' for `tessera-protocol/2`, RFC-0012's six-row `reserve` and RFC-0013 'Cost'; amends RFC-0012 'Layout' and 'Column answers', RFC-0013 'Masked extensions', 'Hiding openings', 'Simulator argument' and INV-ZK-2, RFC-0007 step 2.4, RFC-0014 'Messages' and the field term, RFC-0019 step 5, RFC-0026, RFC-0031, RFC-0036, RFC-0039 step 2.4, RFC-0040, RFC-0049, RFC-0052 'Opening by the single-pass fold' and RFC-0064) | v0.4 |
| [RFC-0066](docs/rfcs/RFC-0066-bits-and-sha256.md) | Bits in the IR and the SHA-256 circuit: `BitDecompose { low, high }` commits a true integer's bits, booleanity and recomposition as zero-checks, the bits its input's unique binary representation and its range check; `BitFn` is XOR, AND, `Ch` and `Maj` on bits with range `[0, 1]`; `AssertEqual` is an equality zero-check. SHA-256 is FIPS 180-4's compression over bit-major words, its additions decomposed with their carries, its digest bound by `AssertEqual`: 6,376 committed bits per block, about 430 per token (amends RFC-0006 'Operation set' and 'Range analysis', RFC-0010 'Relations', RFC-0015 'Circuits' steps 3 and 4 and RFC-0035 'Zero-check points') | v0.4 |
| [RFC-0067](docs/rfcs/RFC-0067-verifier-ledger-encoding.md) | The verifier ledger's encoding and the audit report: `BitSet` is canonical `Bytes`, each record of `proven` keeps its accepted requests' `Û` and records sort by sample; the ledger records the samples the verifier derives itself, and its accounting and report refuse any other; the ledger is written atomically under a lock, its directory synced, its path a regular file of one link; `tessera-verify audit report` states `Accepted` or `Unproven(DeadlineMissed)` from the ledger alone and exits 10 unless every request is accepted; the verifier's audit plan without a cost model gives detection only (amends RFC-0019 'Ledger', RFC-0016 'Audit report' and 'Calculators') | v0.4 |
| [RFC-0068](docs/rfcs/RFC-0068-leaf-stage-zero-knowledge.md) | The leaf stage in `tessera-protocol/2`: each LogUp instance's key and value columns take an `Fp2` mask row of their own, and a masked layer sumcheck reduces every instance's leaf claims to one value per site and tensor at a random layer point, where the tensors' masks act, folding the column masks' terms in. No per-layer value is sent: 614 split values at 405B where there were 53,489, for 36 mask rows and one more advice opening group. Stage 1 stacks at the sumcheck's point, and `ChannelParams` sites of two classes or more are admitted (amends RFC-0007 and RFC-0039 step 2.4, RFC-0014 'Transcript schedule' rows 5 and 6 and 'Soundness calculator', RFC-0013 'Masked extensions' and 'Masked sumcheck', RFC-0011's group count, RFC-0032 'Leaf checks', RFC-0036 schema version 2, RFC-0037 'The layer point', 'Splits' and 'Messages and transcript', RFC-0045 'The leaf point' and 'The split', and RFC-0065 'What the stages owe') | v0.4 |
| [RFC-0069](docs/rfcs/RFC-0069-one-layer-claims.md) | Claims at one layer in `tessera-protocol/2`: a claim with the layer weights of one layer `ℓ` on a stacked committed kind, such as the epilogue's on the last checkpoint, is a claim on the layer's tensor in its own variables, masked by an `Fp2` mask row of its own, since the stacked kind's `Z` vanishes at a Boolean layer. One mask row at 405B, for 33 claims at two points, no group or round added (amends RFC-0013 'Masked extensions' and 'What the verifier sees', RFC-0037 'Layers and kinds', 'Stages and their instances' and 'The claim pool', and RFC-0068 'The layer sumcheck' step 1, 'Zero knowledge' and 'Non-Goals') | v0.4 |
| [RFC-0070](docs/rfcs/RFC-0070-mask-batching.md) | Opening every mask claim in one batching sumcheck in `tessera-protocol/2`: every masked sumcheck's `G(r)`, every tree level's mask claim and the `SoftmaxRow` leaf mask's are linear functionals of one masks' block `M`; a challenge `α` combines them, and a sumcheck on `M + ρ·S`, with a pad `S` laid out like `M` and committed in Round 1, reduces them to one value in one more opening group. Lemma 4a: zero knowledge needs `S` to cover `M`'s layout, not a rank condition. At 405B, 2,668 claims for +65.85 MiB, against 38.25 GiB for a group per column point (amends RFC-0010, RFC-0011, RFC-0013, RFC-0014, RFC-0019, RFC-0032, RFC-0036, RFC-0037, RFC-0057, RFC-0068, 04 §2, 08 §5 and `docs/zk/simulator.md`: R11 met) | v0.4 |
| [RFC-0071](docs/rfcs/RFC-0071-request-slots.md) | A batch of requests in one proof unit: each request of a batch of two or more in a slot of `S = 2^{⌈log2 max n_r⌉}` rows of one token axis, its positions first. Per-token relations and the graph's nodes are unchanged; attention reads keys and values within the slot; the row selector, causal selector, root claim and RoPE's positions become sums over the requests; one greedy maximum serves the batch, with a key relation per request with outputs; the audit packs requests by slot class. A batch of one request is today's proof. At 405B a 16K batch of 16 requests of 1K is about 2.07 GB of `tessera-protocol/1` proof, against 25.4 GB as 16 proofs (amends RFC-0006, RFC-0010, RFC-0015, RFC-0016, RFC-0017, RFC-0019, RFC-0030, RFC-0034, RFC-0035, RFC-0037, RFC-0041, RFC-0044, RFC-0046, RFC-0047, RFC-0053, RFC-0054, RFC-0056, 02 §2.2 and 03 §4) | v0.4 |

All RFCs listed are Accepted; RFC-0021 is reserved for a draft. A decision changes only through a new RFC that supersedes it.

## Release plan

| Milestone | Scope | Exit criteria |
|---|---|---|
| **v0.1** Primitives and unit-cost gate | Workspaces and CI; field, codec, hashing and transcript (prover and verifier sides); multilinear polynomials and sumcheck; LogUp-GKR and product trees; reference advice commitment; GPU microkernels; executable cost model | SC-0: unit costs measured, decision recorded per RFC-0020; conformance vectors pass in both workspaces |
| **v0.2** Integer model and compiler | Python front-end for Llama-3.1 (import, calibration, W8A8, export, integer reference, fidelity); table registry; IR, range analysis, gadget selection, lowering, claim list, commitment report; CPU witness engine | SC-7 bit-exactness; fidelity gates for 8B; commitment report for 8B and 405B shapes resolves RISK-2's count |
| **v0.3** End-to-end reference proof | Gadgets, stage schedule, advice layout and streaming commitment, binding weight commitment, openings, live and non-interactive protocol, soundness calculator, verifier driver, CLIs | Exact-mode greedy proofs of tiny models verify in both modes; tamper corpus rejected in full (SC-4); `tessera-protocol/1` |
| **v0.4** Full statement | Zero knowledge (after the simulator-argument gate), weight hiding budget and refresh, transcript anchor and reference recorder, audit sampling, sampler PRF and mixture scoring, verifier ledger | SC-5 and SC-6; `tessera-protocol/2` |
| **v0.5** GPU prover | CUDA kernels, GPU witness engine, layer-range partitioning, token groups, pass planner; 8B and 70B proofs | Measured `μ` for 8B and 70B; model re-fit; RISK-9 and RISK-10 measured |
| **v1.0** Frontier | Llama-3.1-405B at 1K, 4K and 16K on one 8×H100 node; verifier hardening, fuzzing and written specification; live transport; reproducible builds; external review; documentation | SC-1, SC-2, SC-3; protocol frozen |

The implementation tracker, generated from this corpus, is [docs/roadmap/IMPLEMENTATION.md](docs/roadmap/IMPLEMENTATION.md).

## Risk register

Full statements and resolution plans are in [00 §9](docs/spec/00-overview.md#9-risks).

| Risk | Summary | Resolved by |
|---|---|---|
| RISK-1 | Unit costs are assumptions | v0.1 measurement and decision rule |
| RISK-2 | Wide-key rescales need two commitments, or four where the key is too wide for its shift, and a classed residual site two steps; up to +108% linear-layer commitments at 405B in `minimal` and +174% in `servable` | Resolved by the v0.2 commitment reports ([`docs/perf/v0.2/`](docs/perf/v0.2/README.md)) |
| RISK-3 | The source's commitment-minimal `D` is not INT8-servable in exact mode | Profiles `servable` and `minimal` |
| RISK-4 | `τ = 100` gives 41.5 bits, not 64 | `τ_A = 155`, `τ_W = 97` in `live-64` |
| RISK-5 | LogUp error about 2^-90 at frontier volume | Live seed; 20-bit grinding in non-interactive mode |
| RISK-6 | Zero-knowledge composition across stages | Written simulator argument before v0.4 work |
| RISK-7 | W8A8 fidelity and bounded-mode `Û` at 405B unmeasured | v0.2 and v0.4 fidelity studies |
| RISK-8 | BLAKE3 throughput within the committed-element budget | v0.1 measurement |
| RISK-9 | Witness regeneration absent from the source cost model | Explicit `n_regen` term; measured at v0.3 and v0.5 |
| RISK-10 | The leaf stage and the reduction stages exceed device memory | Token groups and one merge (RFC-0054); windowed rounds for the reduction stages (RFC-0055); score instances without the score tensor (RFC-0056); measured at v0.5 |

## Open questions

| ID | Question | Owner | Resolve by |
|---|---|---|---|
| OQ-0001-1 | Move the verifier to its own repository? | @AbdelStark | v1.0 release planning |
| OQ-0004-1 | Reduce hashed volume if BLAKE3 alone exceeds 20 ps per element? | @AbdelStark | v0.1 gate |
| OQ-0005-1 | Deployment-specific calibration for bounded mode? | @AbdelStark | v0.4 fidelity study |
| OQ-0012-1 | Larger `B_W` and `P_W` for longer epochs? | @AbdelStark | v1.0 |
| OQ-0013-1 | Pack mask rows of small tensors? | @AbdelStark | v0.4, with the simulator argument |
| OQ-0014-1 | Live-session deadlines and retry policy | @AbdelStark | v1.0 |
| OQ-0015-1 | Recorder placement when clients use TLS | @AbdelStark | Before the first bilateral pilot (v1.0) |
| OQ-0017-1 | CDF-window predictor for bounded mode? | @AbdelStark | v0.4 fidelity study |
| OQ-0019-1 | Route to mechanized verifier proofs | @AbdelStark | v1.1 planning |
| OQ-0057-1 | Requests longer than one proof holds: chunks linked by their cached keys and values, or restated 128K targets ([#470](https://github.com/AbdelStark/tessera/issues/470)) | @AbdelStark | Before v1.0's 128K target |
| OQ-0020-1 | B200 as reference hardware for v1.0 claims? | @AbdelStark | v0.5 |
| OQ-SEC-1 | Post-quantum recorder signatures (ML-DSA-65)? | @AbdelStark | v1.0 |

## Conventions

- **Normative language.** In Proposed Design sections and in invariant and requirement tables, "must", "must not" and "may" are normative. Everything else is explanatory.
- **Identifiers.** Goals `G-n`, non-goals `NG-n`, success criteria `SC-n`, invariants `INV-<AREA>-n`, error codes `ERR-*`, reject reasons `REJ-*`, risks `RISK-n`, open questions `OQ-<RFC>-n`, dependency rules `DEP-n`, compatibility rules `COMPAT-n`, observability rules `OBS-n`, security properties `SP-n`, trust boundaries `TB-n`, adversaries `ADV-*`. Identifiers are stable; cite them in code comments, tests and issues.
- **Projections.** Every number derived from the source's cost model is labeled a projection until a measured value with a run manifest replaces it.
- **Source citations.** "Source §n" refers to sections of the source document, [`proofs-of-inference-sota-path-forward.md`](proofs-of-inference-sota-path-forward.md) at the repository root.
