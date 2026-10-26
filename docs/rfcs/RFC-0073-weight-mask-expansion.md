# RFC-0073: Expanding an Epoch's Mask Seed into the Weights' Masks and Blinding Rows

- Status: Draft. The documents it amends carry its amendments, each marked as a draft's, in the change that proposes it. They take effect, and lose the mark, when it is accepted.
- Authors: @AbdelStark
- Created: 2026-10-26
- Target milestone: v0.4
- Amends:
  - [RFC-0040](RFC-0040-weight-commitment-artifacts.md#weightcommitmentstate) '`WeightCommitmentState`' (`masks_seed`) and 'Encoding, leaves and roots' (the codewords): the expansion of `masks_seed` into a `tessera-protocol/2` commitment's masks and blinding rows, normative and versioned, which RFC-0040 left to #147;
  - [RFC-0004](RFC-0004-hashing-and-transcript.md#hash-function-and-domains) 'Hash function and domains': the domains `weights-mask` and `weights-blinding`, read as streams;
  - [RFC-0012](RFC-0012-weight-commitment.md#budget-state-machine) 'Layout' and 'Column answers': where each row's mask and each blinding row's values come from; 'Budget state machine': `reserve` only at the record's latest epoch, and a `link` that writes its successor fresh only past it; 'Failure modes': a lost state;
  - [RFC-0064](RFC-0064-weight-linking-proof.md#the-inputs) 'The inputs', 'The budget' and 'One link per epoch': the expansion a link is remade with, the epoch record, and the successor's state;
  - [RFC-0018](RFC-0018-gpu-prover-runtime.md#crash-consistency) 'Crash consistency': the per-proof seed draws the advice's masks, not the weights';
  - [01 §6](../spec/01-architecture.md#6-concurrency-and-determinism) INV-ARCH-2: a stream's skipped words, as a narrow exception;
  - [02 §1](../spec/02-public-api.md#1-stability-tiers): the expansion is tier A;
  - [05 §3](../spec/05-observability.md#3-data-classification-and-redaction) and OBS-2, and [06 §7](../spec/06-security.md#7-secrets-handling): `K`, `masks_seed` and a stream's state are key material;
  - [`docs/zk/simulator.md`](../zk/simulator.md#5-aborted-sessions) §5 and §9, R6: a rebuilt state reserves nothing.
- Origin: [#147](https://github.com/AbdelStark/tessera/issues/147), from RFC-0064's requirement ([#485](https://github.com/AbdelStark/tessera/issues/485))
- Related: [#564](https://github.com/AbdelStark/tessera/issues/564) (the verifier's refusal of narrow weight matrices)

## Summary

In `tessera-protocol/2` each weight matrix commits `B_T` blinding rows after its `m_T` data rows, and every committed row carries a vanishing mask of `P_W` coefficients (RFC-0012, RFC-0013, RFC-0065). Both come from the epoch's `masks_seed`, which RFC-0064 derives from the model's refresh key `K`, so that a link remade after a crash, from a lost budget record or on a second node is the same bytes. That holds only while every build expands a seed into the same values. RFC-0040 left the expansion to #147.

This RFC fixes it. Each committed row's mask and each blinding row's values come from a stream of their own: BLAKE3's extended output under a domain of RFC-0004, keyed by the seed and addressed by the matrix's `commitment_id` and the row. The stream is read as 64-bit little-endian words, and a word that is not below `p` is skipped, so every value is exactly uniform on `Fp`. A masked commitment is then a function of the weights, the claim list, the row block, the epoch and `K`, and so is a link.

A commitment that can be made again can also be proven from again, with rows its earlier proofs revealed. So the key comes with a durable record of the model's epochs. A state is fresh, with rows to reserve, only when it is written for an epoch past the record's latest. Every other state is rebuilt and serves only its epoch's link, and `reserve` runs only at the record's latest epoch. Nothing the verifier receives or checks changes.

## Motivation

- **RFC-0064 needs it.** 'The inputs' makes a link a deterministic function of `K`, the epoch and the weights, and adds: "Until [#147 specifies the expansion], a link may be remade only by the build that made it." A second build that expanded a seed differently would commit other masks under the same epoch, and its link would open other columns with other pairs: the second link that 'One link per epoch' forbids.
- **RFC-0040 asks for it.** "The expansion of `masks_seed` into the masks and blinding rows is normative and versioned, with a conformance vector; #147 specifies it."
- **Other parts need a row's values alone**, without replaying the rows before it:
  - the codeword store's fallback, which regenerates the rows it answers from the seed (RFC-0012 'Column answers');
  - each proof, which draws the masks and blinding values its combined rows reach;
  - a link, which reaches both epochs' rows (RFC-0064);
  - a commitment made in parallel by several devices, and the byte-identical CPU and GPU commitments of [#174](https://github.com/AbdelStark/tessera/issues/174).
- **RFC-0004 has no stream.** `H` gives 32 bytes, and the transcript is public and sequential; INV-ZK-3 forbids drawing masks from it anyway.
- **Determinism has a cost.** RFC-0064 kept a state's budget as the only record of the rows an epoch has burned. Once an epoch's commitment can be made again, a lost or copied state, or a remade link, can write a fresh budget over burned rows ([`docs/zk/simulator.md`](../zk/simulator.md#5-aborted-sessions) §5, R6). Telling a fresh state from a rebuilt one needs a record the state files do not hold.

## Goals

- Every value a masked weight commitment draws is a function of `masks_seed` and public data, the same in every implementation and on every device.
- Each committed row's mask and each blinding row's values are addressable alone.
- The values are exactly uniform on `Fp`, and independent across rows, matrices, commitments and epochs, under the random-oracle model the simulator argument takes for hashes ([`docs/zk/simulator.md`](../zk/simulator.md#2-model) §2).
- One primitive, RFC-0004's BLAKE3, at one compression per 64 bytes of output.
- No row reserved twice, decided from what the prover holds, after any loss of state files.
- A conformance vector from an independent oracle, and a version that any change must bump.

## Non-Goals

- **The advice's per-proof mask stream** (RFC-0011 'Streaming procedure', RFC-0018 'Crash consistency', [#156](https://github.com/AbdelStark/tessera/issues/156)). It may take 'The stream' below under a domain of its own.
- **How the masked rows are opened, reserved and budgeted** within an epoch: RFC-0012, RFC-0013, RFC-0064 and RFC-0065.
- **The verifier.** It never holds a seed and never expands one. Soundness does not depend on the masks, which are the prover's own choice (RFC-0064 'Zero knowledge'). Its refusal of narrow weight matrices is [#564](https://github.com/AbdelStark/tessera/issues/564)'s.
- **`tessera-protocol/1`,** which masks no weight row and commits no blinding row (RFC-0040).

## Proposed Design

### The seeds

As RFC-0040 and RFC-0064 give them:
- **`K`**, the model's refresh key, is 32 bytes from the operating system's CSPRNG, drawn when epoch 0 is committed and kept apart from every state file, in its own artifact ('The key and the epoch record').
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
- **Values.** A word below `p` is the next value. A word of `p` or more is skipped, never reduced. Skipping one has probability `(2^32 − 1)/2^64 < 2^{-32}`: about 19 words in a whole 405B epoch.
- **Uniform.** Each value is exactly uniform on `Fp` and independent of the others, given uniform words. Reducing the word modulo `p` would give the values below `2^32 − 1` twice the probability of the others, which RFC-0004 refuses for challenges and this RFC refuses for masks.
- **No bound.** The prover's loop has none, as RFC-0004's challenges have none on the prover's side.
- **The one secret-dependent branch allowed.** The acceptance test is public and fixed, `w_t < p`. It reads the candidate word alone, and a rejected word is discarded whole.
  - **What follows.** Which words are skipped is independent of every value any stream yields and of every other secret. So neither the branch's timing nor the addresses a parallel kernel compacts the values to reveal them. INV-ARCH-2, amended, admits exactly this.
  - **The skip pattern stays secret-class.** It is a function of the seed (05 §3, OBS-6). No log, metric or report carries a stream's skips or their count.
- **Cost.** An input of 72 bytes is two BLAKE3 blocks: one compression for the first, then one per output block of 64 bytes.

### The masks and the blinding rows

Let `T` be a weight matrix with:
- `m_T` committed data rows;
- `B_T` blinding rows: `B_W`, or `ChannelParams`' `B_CP` ([RFC-0065](RFC-0065-masked-channel-params.md#channelparams-blinding-rows));
- message width `k_T`;
- `cid_T`, its `commitment_id` in `tessera-protocol/2`, whose `rows` is `m_T + B_T` ([RFC-0040](RFC-0040-weight-commitment-artifacts.md#encoding-leaves-and-roots), [RFC-0031](RFC-0031-commitment-id.md#the-descriptor)).

Rows are numbered two ways:
- **Committed row `j`** counts from 0 over the committed rows in RFC-0040's order. For `j < m_T` it is the `j`-th committed data row in increasing structural row `r` (RFC-0040 'The matrices'), never `r` itself, which counts the skipped rows. Then comes blinding row `i = j − m_T` (RFC-0064 'Positions').
- **Blinding row `i`** counts from 0 over the `B_T` blinding rows: the numbering of RFC-0065's reservation.

Then:

| Domain | Input, 72 bytes | Values |
|---|---|---|
| `weights-mask` | `masks_seed: [u8; 32] ‖ cid_T: Digest ‖ u64(j)`, for each committed row `j < m_T + B_T` | `v_0 … v_{P_W − 1}`: the row's mask coefficients `m_0 … m_{P_W − 1}`, `m_0` first |
| `weights-blinding` | `masks_seed: [u8; 32] ‖ cid_T: Digest ‖ u64(i)`, for each blinding row `i < B_T` | `v_0 … v_{k_T − 1}`: the row's values in natural order, `v_x` at position `x`, the evaluation at `ω_{k_T}^x` |

- **Encoding.** Committed row `j` is encoded as [RFC-0011](RFC-0011-advice-commitment.md#encoding) 'Encoding' says, at the weights' rate.
  - The codeword is `g_j = f_j + Z_H·m_j`, with `m_j(X) = Σ_{s < P_W} m_s·X^s` from its `weights-mask` stream.
  - `f_j` is the polynomial whose evaluations on `H` are committed data row `j`'s values (RFC-0040 'The matrices'), or blinding row `j − m_T`'s.
  - Every other step of RFC-0040 'Encoding, leaves and roots' is unchanged.
- **Every committed row is masked,** data and blinding rows alike, each from its own stream. No value is read twice: a stream gives exactly the values above, and nothing reads its later words.
- **Roles are not drawn.** Every blinding row is drawn the same way. A proof's reservation gives each its role: the evaluation rows' pair `β_T`, a claim's mask row, or the proximity pair. It does so after the commitment and before the seed ([RFC-0065](RFC-0065-masked-channel-params.md#the-reservation)).
- **The input is injective.** Its parts have fixed widths, and each domain has one input format, so two rows, two purposes or two commitments never share a stream.
- **Why `cid_T`.** It binds the layout, the matrix, its width, the row block, the rows and the epoch (RFC-0031). Two commitments made under one epoch number, such as one under another row block, therefore draw independent masks. The budget, the record and the rebuilt-state rule below are per commitment, `R_W`: only the commitment the record lists is the epoch's.
- **A matrix with no committed row**, `ChannelParams` when no weight has one, has `B_CP = 0` and draws nothing.
- **Narrow matrices: until #564, `k_T ≥ k_W`.** A `tessera-protocol/2` commitment is made only when every committed matrix has `k_T ≥ k_W = 2^{layout.log2_k}`. Otherwise `commit-weights` refuses it with `ERR-USAGE`, before drawing anything.
  - **Why `k_W`.** The parameter set's `τ_W` is computed for `k_W`'s rate, `(k_W + P_W)/n_W` (RFC-0012 'Layout', RFC-0014), and a narrower matrix's rate is higher. At `live-64`, `k_T = P_W` gives 40.3 bits of proximity instead of 64.
  - **No bits at all.** Once `k_T + P_W ≥ n_T`, the row `u + (X^{n_T} − 7^{n_T})·h`, `deg h < k_T + P_W − n_T`, passes every column check with any values on `H`. `d_model = 128` at `live-64` is such a case.
  - **Not RFC-0011's bound.** RFC-0011's `P ≤ k` is the advice's decoding bound at rate 1/2. It does not bound the weights' soundness.
  - **The rest is #564's.** A prover's refusal does not bind a malicious prover. The verifier's refusal of such a claim list, and whether narrow matrices are refused or padded to `k_W`, are [#564](https://github.com/AbdelStark/tessera/issues/564)'s. If #564 pads them, this rule follows its padding.

### What it fixes

- **A commitment.** A masked commitment is a function of `D`'s weights, the claim list (its layout and shapes, RFC-0040), the row block `B`, the epoch and `masks_seed_e`, hence `K`: its codewords, column chains, roots and `R_W`.
- **A link.** It is a function of `K`, the epoch, the weights and the claim list, as RFC-0064 'One link per epoch' requires: both commitments are, its transcript is a function of their roots and rows, and its nonce is the least.
- **Any row, alone.** Every row's mask and every blinding row's values can be drawn without the rows before them. That serves devices committing in parallel, a proof's combined rows, a link's rows in both epochs, and the store's fallback for the rows it answers.

### The key and the epoch record

```rust
pub struct WeightRefreshKey {            // schema 0x0304, version 1, secret, .twk
    pub model: Digest,                   // the envelope digest of D's IntegerModel artifact
    pub key: [u8; 32],                   // K
}
pub struct WeightEpochRecord {           // schema 0x0305, version 1, private, .twe
    pub model: Digest,                   // the same digest
    pub roots: Vec<Digest>,              // R_W of epochs 0, 1, …, latest, in order
}
```

- **TCE.** The fields in order. `roots` lists at least epoch 0's root. A reader refuses an empty list with `ERR-CODEC-LIMIT`, as a bound the schema states (04 §3).
- **The latest epoch** is `roots.len() − 1`, and its root is the last.
- **Where they live.** Both live in the operator's key store, apart from every state file. The key is written once, with mode 0600, by atomic rename. The record is written by atomic rename too, and changes only by appending a root.
- **One writer.** The key store has one writer at a time, as RFC-0012 assumes of a state. A copy of the key store on a second node is a second writer and is not allowed. Two nodes that prove for one model need one owner of the record.
- **Proofs never read `K`.** A proof reads the record, the state and the store. Only `commit-weights` and `weights refresh` read the key.
- **One model.** Both artifacts name `D`. A command refuses a key or a record of another `D` with `ERR-USAGE`. That enforces RFC-0064's "one model, one key": with deterministic masks, an old key used for a changed `D` would reuse masks at the columns both commitments open.

### Fresh and rebuilt states

- **A fresh state** is written only by the transition that appends its epoch to the record, right after the append:
  - `commit-weights`, for epoch 0. It draws `K`, commits epoch 0, writes the key and a record of one root, then epoch 0's state, from row 0, and the `.twr`. It refuses an existing key or record path with `ERR-USAGE`. Epoch 0 is therefore committed once per key.
  - The first link from the record's latest epoch `e`, for `e + 1`. It appends `R_W'` before it writes `e + 1`'s state, from row 4 (RFC-0064 'The budget').
- **Every other state is rebuilt.** That covers a state regenerated from `K` after a loss, and one written by a link remade from an epoch that is no longer the latest.
  - In a rebuilt state, every matrix with committed rows has `next_blinding_row = B_T − 4`, where `reserve` always fails and the link's rows begin, and every `opened_columns` is empty.
  - It serves only the link from its epoch, which opens what any link from that epoch opens (RFC-0064 'One link per epoch').
- **`reserve`** runs only on a state of the record's latest epoch whose public part hashes to the record's last root, besides RFC-0012's preconditions. A state of an earlier epoch is refused with `ERR-USAGE`. That covers a restored backup of a closed epoch, and every epoch a later link left behind.
- **Crashes fail safe.** A crash between the record's append and the fresh state's write leaves no state. The state written on restart is then rebuilt, since its epoch is no longer past the latest. That costs one refresh, never a row.
- **After a lost state.** The prover rebuilds the record's latest epoch from `K` and links it. That link appends the successor, which is fresh. Older epochs it rebuilds serve their links only.
- **A lost or stale record fails closed.** Without the record, fresh and rebuilt cannot be told apart, so the prover refuses every `reserve`, link and commitment for the model with `ERR-USAGE`. This applies when the record is lost, unreadable or fails its digest. The operator then starts a new model entry under a new key, as RFC-0064 does when `K` is lost.
  - **A stale record is refused the same way.** A record is stale when the prover holds a state or a `.twr` of an epoch it does not list, or of a root other than the one it lists.
  - **The residual risk.** A key store restored from a backup older than every artifact the prover holds cannot be detected. Keeping none is the operator's duty, as RFC-0064 makes a restored state record the operator's.
- **The verifiers' roots are not the reference.** The record is the prover's own. A root a verifier supplies is never used to decide whether a state is fresh: a malicious verifier could hide a later epoch.

### The regeneration checks

- **A regenerated commitment.** A prover that regenerates an epoch's commitment from a seed, whole or in part, checks that its `WeightCommitmentPublic` hashes to the record's root for that epoch before anything else. A difference is `ERR-USAGE`: the key, the state, the weights, the claim list or the build's expansion is not the epoch's.
  - **Before any write or send.** The check comes before any state, record entry or link is written from the regenerated commitment, and before anything is sent. A link persisted from a wrong commitment would be kept and resent on every restart (RFC-0064 'One link per epoch').
- **Each combined row, before it is sent.** A prover that draws the masks and blinding values its combined rows need, rather than holding the whole commitment, checks each combined row at one column of its own choosing before sending it. The column is never sent, and nothing about it is revealed. A failure is `ERR-USAGE`.
  - **With the codeword store**, the column is read from the store and checked against the matrix root, as RFC-0012 'Column answers' reads columns.
  - **Without the store**, the prover regenerates the column's symbol in every row, then the column's chain and leaf. It checks the leaf against the root through the matrix's Merkle tree, which such a node keeps: `2n_T` digests per matrix.
  - **Why one column suffices.** A wrong mask coefficient or blinding value adds to the combined codeword a polynomial of degree below `k_T + P_W`. One wrong value adds a polynomial that is non-zero at every point of the coset, since neither `Z_H` nor a Lagrange factor vanishes off `H`. A wrong expansion's pseudorandom error vanishes at a given point with probability about `1/p`.

### Versions

- **This is version 1** of the expansion. Its domains are `weights-mask` and `weights-blinding`.
- **What a new version is.** Any change to what this RFC fixes: the primitive, a domain, an input, the order, the reduction or the rows that draw. A new version takes new domains, so that two versions never give related values from one seed.
- **The protocol version names it.** `tessera-protocol/2` expands by version 1, and `tessera-protocol/1` expands nothing (RFC-0040). The expansion is a tier-A surface (02 §1), so a new version is a new protocol version. The claim list states its protocol version (RFC-0007), so a rebuilt epoch's expansion is known from public data. `WeightCommitmentState`'s schema does not change.
- **The other inputs of `cid_T`** are frozen with it: RFC-0040's layout and RFC-0065's `B_CP`. A change to either within a protocol version would break every rebuild, so it is a protocol-version change too.
- **A build that expands by another version** fails the regeneration check, and refuses.

### Secrets

| Value | Class | Held as |
|---|---|---|
| `K` | SECRET key material (05 §3, OBS-2) | `Secret<HeapKey<32>>`, from the CSPRNG or the key artifact; read only by `commit-weights` and `weights refresh` |
| `masks_seed_e` | SECRET key material | On the heap inside the state's `Secret` (RFC-0040). `H("weights-refresh", K ‖ u32(e))` is written straight into that buffer, never into a stack copy |
| A stream's state: the BLAKE3 chaining value after `masks_seed ‖ cid_T`, and the output reader | SECRET key material: the chaining value draws every row of one matrix in one domain | `Secret<Box<…>>`, zeroized on drop with `blake3`'s `zeroize` feature, never `Clone` or `Debug`. A cached chaining value is held the same way |
| Mask coefficients, blinding values, a stream's skip pattern and its count | SECRET | `SecretBuf` and the commitment's `Secret` rows (OBS-6) |
| The epoch record | PRIVATE | the `.twe` file; its roots are public, its contents are the operator's |

- **Never through the public hasher.** `tessera_hash::Hasher` derives `Clone` and `Debug` and does not clear its state, so a stream is never read through it. The reader follows the CPU prover's advice stream, `Secret<Heap<blake3::OutputReader>>` (`crates/tessera-prover/src/cpu/random.rs`). Its file is a reader in `secret-access.toml`.
- **Nothing derived from them is logged.** No log, metric, error or report carries a seed, a stream's words or its skips (OBS-1).

### What is revealed, and why it stays hidden

The expansion sends nothing itself. Every revealed value that depends on it is hidden by a value it draws, under the PRG hybrid of the simulator argument (§2):

| Revealed | Hidden by | What it needs from the expansion |
|---|---|---|
| Opened column symbols, at most `P_W` distinct per matrix and epoch, the link's included | Each row's mask (Lemma 3) | Each row's `m_j` uniform, independent of every other row's and of the weights |
| Combined rows and their combined masks: evaluation, proximity, the link's difference | Each row's designated blinding pair or mask row, and that pair's own masks (Lemma 5, RFC-0013 'Masks of combined masks') | Each blinding row's values uniform, independent of every other row's values and of every mask |
| Blinding claims `q` | One functional of a fresh pair (RFC-0013 'Hiding openings') | The same |
| Masked evaluations `W'(p)` and `ChannelParams`' `v_i` | A mask row `R`, two blinding rows (RFC-0013, RFC-0065) | The same |
| Two epochs' columns and the link | Each epoch's own masks | Independent seeds: `H("weights-refresh", K ‖ u32(e))` per epoch |
| Every one of these, across the proofs of an epoch | Each pair and mask row used by one proof (R6) | No row reserved twice: 'Fresh and rebuilt states' |

- **A malicious verifier** controls no input of the expansion. The seed is secret, the addresses are public and fixed before any session, and the transcript plays no part. It cannot make the prover reserve a row twice either: a root it supplies decides nothing.
- **`ε_PRG`.** In the random-oracle model, a verifier learns anything about a stream only by querying the oracle at an input that holds its 256-bit seed. That has probability at most `Q_H·2^{-256}` for `Q_H` queries. It counts once per epoch a view spans, and once more for `K` (RFC-0064 'Zero knowledge').
- **Leaks.** A leaked state gives its own epoch's seed, and so every mask and blinding row of that epoch, which with the epoch's combined rows and columns reveals folds of the weights. A leaked `K` gives every epoch's. RFC-0064's forward secrecy among epochs holds, and proofs never read `K`.
- **A malicious prover** gains nothing. The verifier never checks the expansion, and a prover that expands otherwise risks only its own secrecy and the determinism of its links.

### Cost at 405B

At RFC-0012's 461,310 committed rows, 49,144 of them blinding rows (826 fewer with `ChannelParams`' 56 data rows, RFC-0065):
- **Per epoch.** The commitment draws 461,310 mask streams of `2^{16}` values and 49,144 blinding streams of `2^{20}`: 3.02·10^10 and 5.15·10^10 values, 654 GB of output and about 1.02·10^10 compressions. A counter mode over `H`, at 32 bytes a compression, would take twice as many.
- **Per proof.** A combined row's mask is `Σ_j c_j·m_j` over every row its coefficients reach, and its values `Σ_j c_j·A_j` likewise.
  - **Under RFC-0012's generator.** The evaluation rows reach every data row, and the proximity row every data and blinding row. So a proof draws every mask and every blinding value, the whole 654 GB: about 1.2 GPU-s at RFC-0004's 530 GB/s per H100, unless the prover keeps them.
  - **If #394 keeps the blinding rows out of the proximity combination,** a proof draws the data rows' masks and its reservation's rows: 412,166 rows × `2^{16}` × 8 bytes, 216 GB, about 0.41 GPU-s.
- **Per link.** Its rows reach both epochs' rows: about 1.3 TB, once per epoch.

### The conformance vector

`spec/vectors/weight-masks/` holds `masks.json`, written by `generate.py`. That oracle is standard-library Python, written from this RFC. It takes its BLAKE3 from `hash/`'s oracle, which checks it against the official vectors at extended length, and its encoding and layout from `encoding/` and `weights/`. The file holds:
- **`refresh`.** A fixed `K` and `masks_seed_e` for `e` in 0, 1 and `2^32 − 2`.
- **`streams`.** For both domains, under two `cid_T`:
  - the inputs: row 0, the last row (`m_T + B_T − 1` for `weights-mask`, `B_T − 1` for `weights-blinding`) and `2^{64} − 1`;
  - for each, the first 24 words, three output blocks, and the values they give.
- **`rejection`.** One input in each domain whose stream has a word of `p` or more among its first eight. Each is found once by a search outside the oracle, about `2^{29}` BLAKE3 calls, and pinned by its domain and bytes. The oracle checks the word and that the values skip it.
- **`models`.**
  - **`tiny-2-minimal`'s commitment** at `test-small/1` in `tessera-protocol/2`, epochs 0 and 1 from the fixed `K`. Its `ChannelParams` has claims, so `B_CP > B_W`, and it commits structural rows 2 to 5, 8 to 11 and 14 to 15, so a build that indexes by `r` fails it.
  - **`synthetic`** (`spec/vectors/weights/`). At `test-small/1` it is refused before anything is drawn: its `K` and `V` have `k_T = 128 < k_W`. Under the test layout `log2_k = 7`, rate 1/4, `P_W = 32`, `B_W = 32` it is committed, and its `ChannelParams`, of no committed row and `B_CP = 0`, draws nothing.

  Per committed matrix:
  - `m_T`, `B_T` and `cid_T`;
  - one SHA-256 (`hashlib`, independent of the BLAKE3 under test) of all its mask coefficients, rows in committed order, each `u64` little-endian, and one of all its blinding values;
  - the first and last values of each;
  - the root.

  Then `R_W`.

`cargo xtask gen-vectors` writes it, and `tessera-pcs` checks it. The verifier reads none of it, since it never expands a seed.

## Alternatives Considered

1. **A counter mode over `H`:** `H("weights-mask", seed ‖ cid ‖ u64(j) ‖ u64(block))`, 32 bytes each. Every implementation already has `H`. Rejected for twice the compressions: 2.0·10^10 against 1.0·10^10 per 405B epoch, and twice the cost of every proof's draws. The extended output is BLAKE3's own, and `hash/`'s oracle and `tv-hash` already check it against the official vectors at extended length.
2. **One stream per matrix or per epoch**, as the CPU prover's per-proof advice masks are drawn today. Rejected: a row's values would need every row before it. The fallback, the reservation's rows, the link's rows and parallel commitment all read rows alone.
3. **A stream per row, values and mask together.** It saves 49,144 stream starts per 405B epoch. Rejected: a reader of masks alone would draw `k_T` values first for each blinding row. Such readers are a proof under #394's exclusion of the blinding rows from the proximity combination, and a fallback that answers data rows only.
4. **AES-128-CTR or ChaCha20.** Rejected: a second primitive on the commitment path, which RFC-0004 keeps to BLAKE3. On a GPU, BLAKE3's 7 rounds per 64 bytes cost less than ChaCha20's 20.
5. **128-bit words reduced modulo `p`.** A bias near `2^{-64}` per value, and twice the stream. Rejected for exactness, as RFC-0004 rejects it for challenges.
6. **The matrix id in place of `cid_T`.** It saves 31 bytes and one compression per stream. Rejected: two commitments from one seed would share every mask.
7. **Drawing from the transcript.** Rejected: INV-ZK-3. Its values are public.
8. **Fresh masks per commitment, from the CSPRNG**, with no expansion to fix. Rejected: RFC-0064's links would not be remade as the same bytes, which is what 'One link per epoch' rests on.
9. **Telling a rebuilt state from the state itself.** Rejected: an epoch-0 state exhausted by proofs stops at exactly `B_T − 4`, byte for byte a rebuilt state's budget. A lost epoch also leaves no trace in the files that survive. The record kept with the key decides it.
10. **The record inside the key artifact.** Rejected: every `reserve` reads the record, and a proof would then read `K`, which RFC-0064 keeps out of proving for forward secrecy.

## Drawbacks

- **Positions move.** A value's word depends on how many words before it were skipped, so a GPU kernel that draws words in parallel must compact them. The fast path, no skipped word, holds for a row with probability about `1 − 2^{-12}` at `2^{20}` values. The conformance vector pins a skip in each domain.
- **Per-proof draws.** Under RFC-0012's proximity generator a 405B proof draws the whole 654 GB, about 1.2 GPU-s, or reads a store of it. RFC-0012 counts that only for its fallback, and 08 §2 still says the weight column answers add no GPU time, under a weight-pass budget of 0.24 GPU-s that alerts above 0.5. OQ-0073-5 asks how to pay for it.
- **Two more artifacts and one more writer to keep single.** The key store becomes state the operator must back up without ever restoring an older copy over a newer one.
- **A lost record ends the model entry,** as a lost key does.
- **Small models leave `tessera-protocol/2`** until #564 decides whether to pad them: every committed matrix needs `k_T ≥ k_W`.
- **One more normative format** that every prover implementation must follow, which no verifier test can check.

## Migration / Rollout

- **#147** implements:
  - the domains;
  - the expansion and the conformance vector;
  - the masked commitment;
  - the key and the record, with `commit-weights`' flags;
  - `reserve`'s rule on the latest epoch;
  - the regeneration checks.

  Both workspaces' domain lists and `spec/vectors/hash/`'s oracle follow RFC-0004's table (RFC-0031), `weights-link` and `weights-refresh` included, which neither has yet. 03 §5's catalog gains `0x0304` and `0x0305`, and 02 §2.2 its flags, with the code that reads them: `tessera-codec`'s registry test reads the catalog.
- **#153** remakes links with the expansion and writes successors fresh or rebuilt by the record.
- **#156** may draw the per-proof advice masks from 'The stream' under a domain of its own. Its resume path never writes a state: a resumed session uses the state its reservation was persisted in.
- **#564** gives the verifier's refusal of narrow matrices, before any `tessera-protocol/2` verifier ships.
- **#174** compares CPU and GPU commitments on it.
- **`tessera-protocol/1`** is unchanged.

## Testing Strategy

- **The vector.** `tessera-pcs` reproduces `masks.json`: every stream, both rejections, the seeds, and each model's digests, roots and `R_W`. It also reproduces the synthetic model's refusal at `test-small/1`.
- **The skip.** A stub word source drives the skip branch: the words `p`, `2^{64} − 1` and `p − 1` give one value, `p − 1`.
- **Independence.** Changing the seed, `cid_T`, the row or the domain changes the first output block. Two epochs from one `K` give different roots for the same weights.
- **Determinism.** Regenerating a commitment from one `K` and epoch gives the same bytes, rows drawn in any order or in parallel.
- **Narrow matrices.** `commit-weights` refuses `k_T = k_W/2` with `ERR-USAGE` before drawing, and accepts `k_T = k_W`.
- **The regeneration checks.**
  - A state whose seed, epoch or weights are not the record's root's is refused with `ERR-USAGE` before any write or message.
  - A build whose expansion differs fails the check of a combined row at one column, with the store and without it.
- **Fresh and rebuilt states.** Each case here reserves no row twice:
  - **The latest directory lost.** With epoch `e + 1`'s directory removed and epoch `e`'s kept, a remade link writes `e + 1` rebuilt, and the next fresh state is `e + 2`'s.
  - **A stale restore.** A restored state of a closed epoch refuses `reserve` with `ERR-USAGE`.
  - **A crash** between the record's append and the successor's state leaves a rebuilt successor.
  - **A lost or stale record** refuses every transition.
  - **A key or record of another `D`** is refused.
  - **An exhausted epoch-0 state and a rebuilt epoch-0 state** both refuse `reserve`; a link from either writes a fresh successor exactly when epoch 0 is the record's latest.
- **Secrets.** A stream's state has neither `Clone` nor `Debug` (`compile_fail`), and is cleared on drop. No event carries a skip count.
- **Hiding.** #147's distinguisher separates two weight sets in three negative controls: an expansion that ignores the row, giving two rows one mask; one that ignores the epoch; and a successor written fresh over a lost epoch. It does not separate them under this RFC.

## Open Questions

- **OQ-0073-1.** Settled by 'The key and the epoch record': `K` lives in its own artifact, bound to `D`, beside the epoch record.
- **OQ-0073-2.** Settled for the prover by 'Narrow matrices: until #564, `k_T ≥ k_W`'. The verifier's refusal is #564's.
- **OQ-0073-3.** Where `pcs/weights/{T}/xi` goes in `tessera-protocol/2`, which RFC-0049 leaves to #147.
  - RFC-0014 still lists it at row 7, after the stages, so the fold pass could not form the proximity row in its one read (INV-WC-4).
  - Recommended: row 3, right after the seed, as in `tessera-protocol/1`, with `⌈log2(m_T + B_T)⌉` coordinates. The proximity pair is fixed by `Round1`'s reservation before the seed.
  - The pair's coefficients themselves are [#394](https://github.com/AbdelStark/tessera/issues/394)'s (R1, R2).
- **OQ-0073-4.** `WeightReservation`'s bytes, which RFC-0065 defines in content only. Recommended:
  - `WeightReservation { matrices: Vec<MatrixReservation> }`, with `MatrixReservation { matrix: u8, first_row: u32 }`;
  - placed after `advice` in `Round1`'s schema version 2, as RFC-0014's sketch orders it, and absorbed with `Round1` under `round1`;
  - ids out of order are `ERR-CODEC-NONCANONICAL` at the id's byte, as RFC-0036 'Matrix ids' rules.
- **OQ-0073-5.** Paying for the per-proof draws ('Cost at 405B'): about 1.2 GPU-s per 405B proof under RFC-0012's proximity generator, 0.41 if #394 excludes the blinding rows. Options:
  - regenerate on the GPUs, and amend 08's weight-pass budget;
  - keep the masks and blinding values in a secret store of 654 GB per epoch beside the codewords, read in full by every proof;
  - exclude the blinding rows from the proximity combination (#394).

  Recommended: a cost-model issue against RFC-0020 and 08, decided with #394.

## References

- [RFC-0004](RFC-0004-hashing-and-transcript.md), [RFC-0011](RFC-0011-advice-commitment.md) 'Encoding', [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0014](RFC-0014-interaction-and-soundness.md) 'Soundness budget', [RFC-0031](RFC-0031-commitment-id.md), [RFC-0040](RFC-0040-weight-commitment-artifacts.md), [RFC-0064](RFC-0064-weight-linking-proof.md), [RFC-0065](RFC-0065-masked-channel-params.md).
- [`docs/zk/simulator.md`](../zk/simulator.md) §2 (the PRG hybrid), §3 (Lemmas 3 and 5), §5 (R6) and §6.
- O'Connor, Aumasson, Neves, Wilcox-O'Hearn, 'BLAKE3: one function, fast everywhere', 2020, §2.6 (extendable output).
