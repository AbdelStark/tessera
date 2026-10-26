# RFC-0073: Expanding an Epoch's Mask Seed into the Weights' Masks and Blinding Rows

- Status: Draft. The amendments below are written into the documents they amend in the change that proposes this RFC, and take effect when it is accepted.
- Authors: @AbdelStark
- Created: 2026-10-26
- Target milestone: v0.4
- Amends:
  - [RFC-0040](RFC-0040-weight-commitment-artifacts.md#weightcommitmentstate) '`WeightCommitmentState`' (`masks_seed`) and 'Encoding, leaves and roots' (the codewords): the expansion of `masks_seed` into a `tessera-protocol/2` commitment's masks and blinding rows, normative and versioned, which RFC-0040 left to #147;
  - [RFC-0004](RFC-0004-hashing-and-transcript.md#hash-function-and-domains) 'Hash function and domains': the domains `weights-mask` and `weights-blinding`, read as streams;
  - [RFC-0064](RFC-0064-weight-linking-proof.md#the-inputs) 'The inputs' and 'One link per epoch': the expansion a link is remade with;
  - [RFC-0012](RFC-0012-weight-commitment.md#layout) 'Layout' and 'Column answers': where each row's mask and each blinding row's values come from;
  - [01 §6](../spec/01-architecture.md#6-concurrency-and-determinism) INV-ARCH-2: which words a secret stream skips is not a secret value.
- Origin: [#147](https://github.com/AbdelStark/tessera/issues/147), from RFC-0064's requirement ([#485](https://github.com/AbdelStark/tessera/issues/485))

## Summary

In `tessera-protocol/2` each weight matrix commits `B_T` blinding rows after its `m_T` data rows, and every committed row carries a vanishing mask of `P_W` coefficients (RFC-0012, RFC-0013, RFC-0065). Both come from the epoch's `masks_seed`, which RFC-0064 derives from the model's refresh key `K`, so that a link remade after a crash, from a lost record or on a second node is the same bytes. That holds only while every build expands a seed into the same values. RFC-0040 left the expansion to #147.

This RFC fixes it. Each committed row's mask and each blinding row's values come from a stream of their own: BLAKE3's extended output under a domain of RFC-0004, keyed by the seed and addressed by the matrix's `commitment_id` and the row. The stream is read as 64-bit little-endian words, and a word that is not below `p` is skipped, so every value is exactly uniform on `Fp`. A masked commitment is then a function of the weights, the claim list, the row block, the epoch and `K`, and so is a link. Since a commitment can then be made again, a state rebuilt for an epoch that had one serves only that epoch's link: its rows were burned by the record it replaces. Nothing the verifier receives or checks changes.

## Motivation

- **RFC-0064 needs it.** 'The inputs' makes a link a deterministic function of `K`, the epoch and the weights, and adds: "Until [#147 specifies the expansion], a link may be remade only by the build that made it." A second build that expanded a seed differently would commit other masks under the same epoch, and its link would open other columns with other pairs: the second link that 'One link per epoch' forbids.
- **RFC-0040 asks for it.** "The expansion of `masks_seed` into the masks and blinding rows is normative and versioned, with a conformance vector; #147 specifies it."
- **Other parts need a row's values alone**, without replaying the rows before it:
  - the codeword store's fallback, which regenerates the rows it answers from the seed (RFC-0012 'Column answers');
  - each proof, which regenerates its reservation's blinding rows and every row's mask for its combined masks;
  - a link, which regenerates the four rows at each end of an epoch (RFC-0064 'The budget');
  - a commitment made in parallel by several devices, and the byte-identical CPU and GPU commitments of [#174](https://github.com/AbdelStark/tessera/issues/174).
- **RFC-0004 has no stream.** `H` gives 32 bytes, and the transcript is public and sequential; INV-ZK-3 forbids drawing masks from it anyway.

## Goals

- Every value a masked weight commitment draws is a function of `masks_seed` and public data, the same in every implementation and on every device.
- Each committed row's mask and each blinding row's values are addressable alone.
- The values are exactly uniform on `Fp`, and independent across rows, matrices, commitments and epochs, under the random-oracle model the simulator argument takes for hashes ([`docs/zk/simulator.md`](../zk/simulator.md#2-model) §2).
- One primitive, RFC-0004's BLAKE3, at one compression per 64 bytes of output.
- A conformance vector from an independent oracle, and a version that any change must bump.

## Non-Goals

- **The advice's per-proof mask stream** (RFC-0011 'Streaming procedure', [RFC-0018](RFC-0018-gpu-prover-runtime.md#crash-consistency) 'Crash consistency', [#156](https://github.com/AbdelStark/tessera/issues/156)). It may take 'The stream' below under a domain of its own.
- **How the masked rows are opened, reserved and budgeted:** RFC-0012, RFC-0013, RFC-0064 and RFC-0065.
- **The verifier.** It never holds a seed and never expands one. Soundness does not depend on the masks, which are the prover's own choice (RFC-0064 'Zero knowledge').
- **`tessera-protocol/1`,** which masks no weight row and commits no blinding row (RFC-0040).

## Proposed Design

### The seeds

As RFC-0040 and RFC-0064 give them:
- **`K`**, the model's refresh key, is 32 bytes from the operating system's CSPRNG, drawn when epoch 0 is committed and kept apart from every state file.
- **`masks_seed_e = H("weights-refresh", K ‖ u32(e))`** is epoch `e`'s seed, and epoch `e`'s `WeightCommitmentState` holds it.
- **Nothing else seeds the expansion:** no per-proof value and no transcript value (INV-ZK-3).

### The stream

For a domain `d` of RFC-0004 and an input `x`:

```text
S(d, x)  = BLAKE3.derive_key(context = "tessera/v1/" + d, key_material = x), extended output
w_t      = u64_le(S(d, x)[8t .. 8t + 8]),                         t = 0, 1, 2, …
v_i      = the i-th word w_t < p, in increasing t,                  i = 0, 1, 2, …
```

- **The extended output** is BLAKE3's root output blocks `0, 1, 2, …` (the BLAKE3 specification, §2.6). Its first 32 bytes are RFC-0004's `H(d, x)`.
- **Words** are read in order, eight per 64-byte output block.
- **Values.** A word below `p` is the next value. A word of `p` or more is skipped, never reduced. Skipping one has probability `(2^32 − 1)/2^64 < 2^{-32}`.
- **Uniform.** Each value is exactly uniform on `Fp` and independent of the others, given uniform words. Reducing the word modulo `p` would give the values below `2^32 − 1` twice the probability of the others, which RFC-0004 refuses for challenges and this RFC refuses for masks.
- **No bound.** The prover's loop has none, as RFC-0004's challenges have none on the prover's side.
- **Skipping is not a leak.** It branches on secret randomness, but which words are skipped is independent of every value the stream yields, so the branch's timing says nothing about them. INV-ARCH-2, amended, allows it.
- **Cost.** An input of 72 bytes is two BLAKE3 blocks: one compression for the first, then one per output block of 64 bytes.

### The masks and the blinding rows

Let `T` be a weight matrix with `m_T` committed data rows and `B_T` blinding rows (`B_W`, or `ChannelParams`' `B_CP`; [RFC-0065](RFC-0065-masked-channel-params.md#channelparams-blinding-rows)), message width `k_T`, and `cid_T` its `commitment_id` in `tessera-protocol/2`, whose `rows` is `m_T + B_T` ([RFC-0040](RFC-0040-weight-commitment-artifacts.md#encoding-leaves-and-roots), [RFC-0031](RFC-0031-commitment-id.md#the-descriptor)). Committed row `j` counts from 0 over the committed rows in RFC-0040's order: for `j < m_T`, the `j`-th committed data row in increasing structural row `r` (RFC-0040 'The matrices'), never `r` itself, which counts the skipped rows; then blinding row `i = j − m_T` (RFC-0064 'Positions'). Blinding row `i` counts from 0 over the `B_T` blinding rows, the numbering of RFC-0065's reservation. Then:

| Domain | Input, 72 bytes | Values |
|---|---|---|
| `weights-mask` | `masks_seed: [u8; 32] ‖ cid_T: Digest ‖ u64(j)`, for each committed row `j < m_T + B_T` | `v_0 … v_{P_W − 1}`: the row's mask coefficients `m_0 … m_{P_W − 1}`, `m_0` first |
| `weights-blinding` | `masks_seed: [u8; 32] ‖ cid_T: Digest ‖ u64(i)`, for each blinding row `i < B_T` | `v_0 … v_{k_T − 1}`: the row's values in natural order, `v_x` at position `x`, the evaluation at `ω_{k_T}^x` |

- **Encoding.** Committed row `j` is encoded as [RFC-0011](RFC-0011-advice-commitment.md#encoding) 'Encoding' says, at the weights' rate: `g_j = f_j + Z_H·m_j`, with `m_j(X) = Σ_{s < P_W} m_s·X^s` from its `weights-mask` stream, and `f_j` the polynomial whose evaluations on `H` are committed data row `j`'s values (RFC-0040 'The matrices') or blinding row `j − m_T`'s. Every other step of RFC-0040 'Encoding, leaves and roots' is unchanged.
- **Every committed row is masked,** data and blinding rows alike, each from its own stream. No value is read twice: a stream gives exactly the values above, and nothing reads its later words.
- **Roles are not drawn.** Every blinding row is drawn the same way. A proof's reservation gives each its role, the evaluation rows' pair `β_T`, a claim's mask row or the proximity pair, after the commitment and before the seed ([RFC-0065](RFC-0065-masked-channel-params.md#the-reservation)).
- **The input is injective.** Its parts have fixed widths, and each domain has one input format, so two rows, two purposes or two commitments never share a stream.
- **Why `cid_T`.** It binds the layout, the matrix, its width, the row block, the rows and the epoch (RFC-0031). Two commitments made from one seed, such as one under another row block, therefore draw independent masks. The rule "one key, one model entry" of RFC-0064 does not have to cover them.
- **A matrix with no committed row**, `ChannelParams` when no weight has one, has `B_CP = 0` and draws nothing.
- **A matrix narrower than its mask.** RFC-0011's encoding takes `P_W ≤ k_T`. A `tessera-protocol/2` commitment with a committed matrix of `k_T < P_W` is not made: `commit-weights` refuses it with `ERR-USAGE`, before drawing anything. What the verifier does with such a claim list is OQ-0073-2.

### What it fixes

- **A commitment.** A masked commitment is a function of `D`'s weights, the claim list (its layout and shapes, RFC-0040), the row block `B`, the epoch and `masks_seed_e`, hence `K`: its codewords, column chains, roots and `R_W`.
- **A link.** It is a function of `K`, the epoch, the weights and the claim list, as RFC-0064 'One link per epoch' requires: both commitments are, its transcript is a function of their roots and rows, and its nonce is the least.
- **Any row, alone.** Every row's mask and every blinding row's values can be drawn without the rows before them: by devices committing in parallel, by a proof for its combined rows, by a link for both epochs' rows, and by the store's fallback for the rows it answers.

### Regenerated commitments and rebuilt states

The expansion makes an epoch's commitment reproducible. That serves links, and it would reset the budget if a reproduced state could prove again.

- **Regenerating a commitment.** A prover that regenerates an epoch's commitment from a seed, whole or in part, sends nothing that reads it until the regenerated `WeightCommitmentPublic` equals the epoch's: the `.twr` file, or the root `R_W` the model's verifiers hold. A difference is `ERR-USAGE`: the key, the state, the weights, the claim list or the build's expansion is not the epoch's.
- **Each combined row, before it is sent.** A prover that does not regenerate the whole commitment for a proof, but draws the masks and blinding values its combined rows need, checks each combined row's column check at one column of its codeword store, read and checked against the matrix root as RFC-0012 'Column answers' reads columns. The column is not sent, and nothing about it is revealed. A failure is `ERR-USAGE`. A wrong mask or blinding value changes the combined codeword at all but a few positions, so one column catches it.
- **A rebuilt state serves no proof.** A state written for an epoch whose state existed before is a rebuilt state: one regenerated from `K` after the state was lost, or one written again by a remade link. It holds the same commitment, so its rows are the rows the lost record had burned. A rebuilt state therefore has every matrix's `next_blinding_row` at `B_T − 4`, where `reserve` always fails and the link's rows begin. It serves only the link from its epoch, which opens what any link from that epoch opens (RFC-0064 'One link per epoch').
- **Epoch 0 is committed once per key.** `commit-weights` commits epoch 0 only under a key it draws itself, and never under an existing one.
- **A link's successor.** A link writes its successor's state from row 4 (RFC-0064 'The budget') only the first time it is made. That is when the epoch it links from is the model's latest, its successor never committed. Every other successor state the link writes is rebuilt. After a lost state, the prover rebuilds the latest epoch it holds a public artifact of, links it, and proves from the successor.

### Versions

- **This is version 1** of the expansion. Its domains are `weights-mask` and `weights-blinding`.
- **What a new version is.** Any change to what this RFC fixes: the primitive, a domain, an input, the order, the reduction or the rows that draw. A new version takes new domains, so that two versions never give related values from one seed.
- **The protocol version names it.** `tessera-protocol/2` expands by version 1, and `tessera-protocol/1` expands nothing (RFC-0040). A new version is a new protocol version, as any change that alters a tier-A commitment's bytes is ([02 §1](../spec/02-public-api.md#1-stability-tiers)). The claim list states its protocol version (RFC-0007), so a rebuilt epoch's expansion is known from public data. `WeightCommitmentState`'s schema does not change.
- **A build that expands by another version** fails the regeneration check, and refuses.

### What is revealed, and why it stays hidden

The expansion sends nothing itself. Every revealed value that depends on it is hidden by a value it draws, under the PRG hybrid of the simulator argument (§2):

| Revealed | Hidden by | What it needs from the expansion |
|---|---|---|
| Opened column symbols, at most `P_W` distinct per matrix and epoch, the link's included | Each row's mask (Lemma 3) | Each row's `m_j` uniform, independent of every other row's and of the weights |
| Combined rows and their combined masks: evaluation, proximity, the link's difference | Each row's designated blinding pair or mask row, and that pair's own masks (Lemma 5, RFC-0013 'Masks of combined masks') | Each blinding row's values uniform, independent of every other row's values and of every mask |
| Blinding claims `q` | One functional of a fresh pair (RFC-0013 'Hiding openings') | The same |
| Masked evaluations `W'(p)` and `ChannelParams`' `v_i` | A mask row `R`, two blinding rows (RFC-0013, RFC-0065) | The same |
| Two epochs' columns and the link | Each epoch's own masks | Independent seeds: `H("weights-refresh", K ‖ u32(e))` per epoch |

- **A malicious verifier** controls no input of the expansion. The seed is secret, the addresses are public and fixed before any session, and the transcript plays no part.
- **`ε_PRG`.** In the random-oracle model, a verifier learns anything about a stream only by querying the oracle at an input that holds its 256-bit seed, with probability at most `Q_H·2^{-256}` for `Q_H` queries. It counts once per epoch a view spans, and once more for `K` (RFC-0064 'Zero knowledge').
- **Leaks.** A leaked state gives its own epoch's seed and so every mask and blinding row of that epoch, which with the epoch's combined rows and columns reveals folds of the weights. A leaked `K` gives every epoch's. Both are secret-class ([05 §3](../spec/05-observability.md#3-data-classification-and-redaction)), and RFC-0064's forward secrecy among epochs holds.
- **A malicious prover** gains nothing. The verifier never checks the expansion, and a prover that expands otherwise risks only its own secrecy and the determinism of its links.

### Cost at 405B

At RFC-0012's 461,310 committed rows, 49,144 of them blinding rows (826 fewer with `ChannelParams`' 56 data rows, RFC-0065):
- **Per epoch.** The commitment draws 461,310 mask streams of `2^{16}` values and 49,144 blinding streams of `2^{20}`: 3.02·10^10 and 5.15·10^10 values, 654 GB of output and about 1.02·10^10 compressions. A counter mode over `H`, at 32 bytes a compression, would take twice as many.
- **Per proof.** A combined row's mask is `Σ_j c_j·m_j` over every row its coefficients reach, and its values `Σ_j c_j·A_j` likewise. The evaluation rows reach every data row, and the proximity row, under RFC-0012's generator, every data and blinding row. So a proof needs every row's mask and every blinding row's values: the whole 654 GB, about 1.2 GPU-s at RFC-0004's 530 GB/s per H100, unless the prover keeps them. If #394 keeps the blinding rows out of the proximity combination, a proof needs every mask and only its reservation's rows: 242 GB and 0.6 GB, about 0.46 GPU-s.
- **Per link.** Its rows reach both epochs' rows: about 1.3 TB, once per epoch.

### The conformance vector

`spec/vectors/weight-masks/` holds `masks.json`, written by `generate.py`: standard-library Python, written from this RFC, which takes its BLAKE3 from `hash/`'s oracle (checked there against the official vectors at extended length) and its encoding and layout from `encoding/` and `weights/`. It holds:
- **`refresh`.** A fixed `K` and `masks_seed_e` for `e` in 0, 1 and `2^32 − 2`.
- **`streams`.** For both domains, inputs with row 0, the last row (`m_T + B_T − 1` for `weights-mask`, `B_T − 1` for `weights-blinding`) and `2^{64} − 1`, under two `cid_T`: the first 24 words, three output blocks, and the values they give.
- **`rejection`.** An input whose stream has a word of `p` or more among its first eight, found once by a search outside the oracle and pinned by its bytes. The oracle checks that word and that the values skip it.
- **`models`.** `tiny-2-minimal`'s commitment at `test-small/1` in `tessera-protocol/2`, epochs 0 and 1 from the fixed `K`. Its `ChannelParams` has claims, so `B_CP > B_W`, and commits structural rows 2 to 5, 8 to 11 and 14 to 15, so a build that indexes by `r` fails it. Per matrix:
  - `m_T`, `B_T` and `cid_T`;
  - one SHA-256 (`hashlib`, independent of the BLAKE3 under test) of all its mask coefficients, rows in committed order, each `u64` little-endian, and one of all its blinding values;
  - the first and last values of each;
  - the root.

  Then `R_W`.

`cargo xtask gen-vectors` writes it, and `tessera-pcs` checks it. The verifier reads none of it, since it never expands a seed.

## Alternatives Considered

1. **A counter mode over `H`:** `H("weights-mask", seed ‖ cid ‖ u64(j) ‖ u64(block))`, 32 bytes each. Every implementation already has `H`. Rejected for twice the compressions: 2.0·10^10 against 1.0·10^10 per 405B epoch, and twice the cost of every proof's combined masks. The extended output is BLAKE3's own, and `hash/`'s oracle and `tv-hash` already check it against the official vectors at extended length.
2. **One stream per matrix or per epoch**, as the CPU prover's per-proof advice masks are drawn today. Rejected: a row's values would need every row before it. The fallback, the reservation's rows, the link's rows and parallel commitment all read rows alone.
3. **A stream per row, values and mask together.** It saves 49,144 stream starts per 405B epoch. Rejected: a reader of masks alone would draw `k_T` values first for each blinding row. Such readers are a proof under #394's exclusion of the blinding rows from the proximity combination, and a fallback that answers data rows only.
4. **AES-128-CTR or ChaCha20.** Rejected: a second primitive on the commitment path, which RFC-0004 keeps to BLAKE3. On a GPU, BLAKE3's 7 rounds per 64 bytes cost less than ChaCha20's 20.
5. **128-bit words reduced modulo `p`.** A bias near `2^{-64}` per value, and twice the stream. Rejected for exactness, as RFC-0004 rejects it for challenges.
6. **The matrix id in place of `cid_T`.** It saves 31 bytes and one compression per stream. Rejected: two commitments from one seed would share every mask.
7. **Drawing from the transcript.** Rejected: INV-ZK-3. Its values are public.

## Drawbacks

- **Positions move.** A value's word depends on how many words before it were skipped, so a GPU kernel that draws words in parallel must compact them. The fast path, no skipped word, holds for a row with probability about `1 − 2^{-12}` at `2^{20}` values. The conformance vector pins a skip.
- **Per-proof draws.** Under RFC-0012's proximity generator a 405B proof draws the whole 654 GB, about 1.2 GPU-s, or reads a store of it. RFC-0012 counts that only for its fallback, and 08 §2 still says the weight column answers add no GPU time, under a weight-pass budget of 0.24 GPU-s that alerts above 0.5. OQ-0073-5 asks how to pay for it.
- **One more normative format** that every prover implementation must follow, which no verifier test can check.

## Migration / Rollout

- **#147** implements the domains, the expansion, the masked commitment and the conformance vector, with RFC-0012's budget and RFC-0065's reservation. Both workspaces' domain lists and `spec/vectors/hash/`'s oracle follow RFC-0004's table (RFC-0031), `weights-link` and `weights-refresh` included, which neither has yet.
- **#153** remakes links with it, and checks a regenerated epoch before linking.
- **#156** may draw the per-proof advice masks from 'The stream' under a domain of its own.
- **#174** compares CPU and GPU commitments on it.
- **`tessera-protocol/1`** is unchanged.

## Testing Strategy

- **The vector.** `tessera-pcs` reproduces `masks.json`: every stream, the rejection, the seeds, and each model's digests, roots and `R_W`.
- **The skip.** A stub word source drives the skip branch: the words `p`, `2^{64} − 1` and `p − 1` give one value, `p − 1`.
- **Independence.** Changing the seed, `cid_T`, the row or the domain changes the first output block. Two epochs from one `K` give different roots for the same weights.
- **Determinism.** Regenerating a commitment from one `K` and epoch gives the same bytes, rows drawn in any order or in parallel.
- **The regeneration check.** A state whose seed, epoch or weights are not its public artifact's is refused with `ERR-USAGE` before any message. A build whose expansion differs fails the check of a combined row against a store column.
- **Rebuilt states.** A state rebuilt from `K` refuses `reserve` with `ERR-WITNESS-BUDGET` and serves its link. `commit-weights` refuses an existing key. A remade link leaves its successor's state as it is, or writes it rebuilt.
- **Hiding.** #147's distinguisher tells apart two weight sets under an expansion that ignores the row, giving two rows one mask, and under one that ignores the epoch. It does not under this one.

## Open Questions

- **OQ-0073-1.** Where `K` is kept. RFC-0064 says "in the operator's key store", and `commit-weights` must draw it at epoch 0. Recommended: a secret artifact `WeightRefreshKey` (`0x0304`, `.twk`), `{ key: [u8; 32], model: Digest }` with `model` the envelope digest of `D`, written once with mode 0600 by atomic rename.
  - `commit-weights --refresh-key` creates it and refuses an existing file ('Regenerated commitments and rebuilt states').
  - `weights refresh --refresh-key` reads it and refuses it for another `D`, so that one key never serves two models.
- **OQ-0073-2.** A claim list with a committed matrix narrower than its mask, `k_T < P_W`.
  - **Where it arises.** Nothing in RFC-0012 or RFC-0040 excludes it. At `live-64`, any weight type with `out_pad·in_pad < 2^{16}` has it: `d_model = 128` gives `k_Q = 2^{14}`. So does a small `ChannelParams`: at `test-small`, a one-layer model with `C = 2` has `k_CP = 32 < 64`.
  - **Why it matters for soundness.** Once `k_T + P_W ≥ n_T`, that is `k_T ≤ P_W/3` at rate 1/4, every word is a masked codeword, and the matrix's proximity test proves nothing. A malicious prover's claim list need not respect the honest prover's refusal.
  - **Recommended.** In `tessera-protocol/2` the verifier refuses such a claim list at validation (RFC-0007), `Reject(StatementMismatch { field: Layout })`, from the layout alone, and the compiler refuses to emit it. That is a change to an acceptance check, for its own adversarial reading. A soundness calculator that takes each matrix's `k_T` already shows the loss (RFC-0014).
- **OQ-0073-3.** Where `pcs/weights/{T}/xi` goes in `tessera-protocol/2`, which RFC-0049 leaves to #147. RFC-0014 still lists it at row 7, after the stages, so the fold pass could not form the proximity row in its one read (INV-WC-4). Recommended: row 3, right after the seed, as in `tessera-protocol/1`, with `⌈log2(m_T + B_T)⌉` coordinates. The proximity pair is fixed by `Round1`'s reservation before the seed. The pair's coefficients themselves are [#394](https://github.com/AbdelStark/tessera/issues/394)'s (R1, R2).
- **OQ-0073-4.** `WeightReservation`'s bytes, which RFC-0065 defines in content only. Recommended: `WeightReservation { matrices: Vec<MatrixReservation> }`, `MatrixReservation { matrix: u8, first_row: u32 }`, placed after `advice` in `Round1`'s schema version 2, as RFC-0014's sketch orders it, and absorbed with `Round1` under `round1`. Ids out of order are `ERR-CODEC-NONCANONICAL` at the id's byte, as RFC-0036 'Matrix ids' rules.
- **OQ-0073-5.** Paying for the per-proof draws ('Cost at 405B'): about 1.2 GPU-s per 405B proof under RFC-0012's proximity generator, 0.46 if #394 excludes the blinding rows. Options: regenerate on the GPUs and amend 08's weight-pass budget; keep the masks and blinding values in a secret store of 654 GB per epoch beside the codewords, read in full by every proof; or exclude the blinding rows from the proximity combination (#394). Recommended: a cost-model issue against RFC-0020 and 08, decided with #394.

## References

- [RFC-0004](RFC-0004-hashing-and-transcript.md), [RFC-0011](RFC-0011-advice-commitment.md) 'Encoding', [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0031](RFC-0031-commitment-id.md), [RFC-0040](RFC-0040-weight-commitment-artifacts.md), [RFC-0064](RFC-0064-weight-linking-proof.md), [RFC-0065](RFC-0065-masked-channel-params.md).
- [`docs/zk/simulator.md`](../zk/simulator.md) §2 (the PRG hybrid), §3 (Lemmas 3 and 5), §5 and §6.
- O'Connor, Aumasson, Neves, Wilcox-O'Hearn, 'BLAKE3: one function, fast everywhere', 2020, §2.6 (extendable output).
