# RFC-0073: Expanding an Epoch's Mask Seed into the Weights' Masks and Blinding Rows

- Status: Draft. The documents it amends carry its amendments, each marked as a draft's, in the change that proposes it. They take effect, and lose the mark, when it is accepted.
- Authors: @AbdelStark
- Created: 2026-10-26
- Target milestone: v0.4
- Amends:
  - [RFC-0040](RFC-0040-weight-commitment-artifacts.md#weightcommitmentstate) '`WeightCommitmentState`' (`masks_seed`) and 'Encoding, leaves and roots' (the codewords): the expansion of `masks_seed` into a `tessera-protocol/2` commitment's masks and blinding rows, normative and versioned, which RFC-0040 left to #147;
  - [RFC-0004](RFC-0004-hashing-and-transcript.md#hash-function-and-domains) 'Hash function and domains': the domains `weights-mask` and `weights-blinding`, read as streams;
  - [RFC-0012](RFC-0012-weight-commitment.md#budget-state-machine):
    - 'Layout' and 'Column answers': where each row's mask and each blinding row's values come from, the narrow-matrix refusal, and evaluated column answers checked against the root;
    - 'Budget state machine': `reserve` at the latest epoch only, `record_queries` raising the record's column count first, and a `link` that writes its successor fresh only right after its append;
    - 'Failure modes': a lost state;
  - [RFC-0040](RFC-0040-weight-commitment-artifacts.md#command-line) 'Command line': in `tessera-protocol/2`, `commit-weights` writes a model entry, not a free state path;
  - [RFC-0064](RFC-0064-weight-linking-proof.md#the-inputs) 'The inputs', 'The budget' and 'One link per epoch': the expansion a link is remade with, the model entry, the epoch record, the successor's state, the lock in place of the pending-reservation precondition, and the link's error codes;
  - [RFC-0018](RFC-0018-gpu-prover-runtime.md#crash-consistency) 'Crash consistency' and 'Determinism': the per-proof seed draws the advice's masks, not the weights', and a kernel may compact a stream's words;
  - [RFC-0002](RFC-0002-field-arithmetic.md#timing) 'Timing': the stream's rejection;
  - [01 §6](../spec/01-architecture.md#6-concurrency-and-determinism) INV-ARCH-2: a stream's skipped words, as a narrow exception;
  - [02 §1](../spec/02-public-api.md#1-stability-tiers): the expansion is tier A; [02 §2.2](../spec/02-public-api.md#22-tessera-rust-prover-side): a note that `commit-weights`, `weights refresh` and, in `tessera-protocol/2`, `prove` take a model entry's directory (`--entry`) and presented artifacts (`--present`), whose flags join the listing with their code;
  - [03 §5](../spec/03-data-model.md#5-artifact-catalog): `WeightRefreshKey` (`0x0304`) and `WeightEpochRecord` (`0x0305`), whose rows join the catalog with the code that reads them, since `tessera-codec`'s registry test reads it;
  - [05 §3](../spec/05-observability.md#3-data-classification-and-redaction) and OBS-2: `K`, `masks_seed` and a stream's state are key material;
  - [06 §5](../spec/06-security.md#5-threats-and-mitigations) (timing) and [06 §7](../spec/06-security.md#7-secrets-handling): the exception, and the key artifact;
  - [`docs/zk/simulator.md`](../zk/simulator.md#5-aborted-sessions) §5 and §9, R6: a rebuilt state reserves nothing.
- Origin: [#147](https://github.com/AbdelStark/tessera/issues/147), from RFC-0064's requirement ([#485](https://github.com/AbdelStark/tessera/issues/485))
- Related: [#564](https://github.com/AbdelStark/tessera/issues/564) (the verifier's refusal of narrow weight matrices)

## Summary

In `tessera-protocol/2` each weight matrix commits `B_T` blinding rows after its `m_T` data rows, and every committed row carries a vanishing mask of `P_W` coefficients (RFC-0012, RFC-0013, RFC-0065). Both come from the epoch's `masks_seed`, which RFC-0064 derives from the model's refresh key `K`, so that a link remade after a crash, from a lost budget state or on a second node is the same bytes. That holds only while every build expands a seed into the same values. RFC-0040 left the expansion to #147.

This RFC fixes it. Each committed row's mask and each blinding row's values come from a stream of their own: BLAKE3's extended output under a domain of RFC-0004, keyed by the seed and addressed by the matrix's `commitment_id` and the row. The stream is read as 64-bit little-endian words, and a word that is not below `P` is skipped, so every value is exactly uniform on `Fp`. A masked commitment is then a function of the weights, the claim list, the row block, the epoch and `K`, and so is a link.

A commitment that can be made again can also be proven from again, with rows its earlier proofs revealed. So each model entry lives in one directory under fixed names, locked as a whole, beside a durable record of its epochs. The record holds each epoch's root and the latest epoch's high-water marks.
- A state is fresh, with rows to reserve, only when written right after the append that makes its epoch the record's latest. Every other state is rebuilt and serves only its epoch's link.
- `reserve` runs only at the latest epoch, from the larger of the state's cursor and the record's marks.

What the prover cannot detect from its own files is stated as a rule on files, with its bound and with what an operator can present to catch it. That is a rollback of the record together with every file that would refute it, or a copy of the entry. An exhaustive model of these rules checks the statement over every combination of lost and restored entry files. Nothing the verifier receives or checks changes.

## Motivation

- **RFC-0064 needs it.** 'The inputs' makes a link a deterministic function of `K`, the epoch and the weights, and adds: "Until [#147 specifies the expansion], a link may be remade only by the build that made it." A second build that expanded a seed differently would commit other masks under the same epoch, and its link would open other columns with other pairs: the second link that 'One link per epoch' forbids.
- **RFC-0040 asks for it.** "The expansion of `masks_seed` into the masks and blinding rows is normative and versioned, with a conformance vector; #147 specifies it."
- **Other parts need a row's values alone**, without replaying the rows before it:
  - the codeword store's fallback, which regenerates the rows it answers from the seed (RFC-0012 'Column answers');
  - each proof, which draws the masks and blinding values its combined rows reach;
  - a link, which reaches both epochs' rows (RFC-0064);
  - a commitment made in parallel by several devices, and the byte-identical CPU and GPU commitments of [#174](https://github.com/AbdelStark/tessera/issues/174).
- **RFC-0004 has no stream.** `H` gives 32 bytes, and the transcript is public and sequential; INV-ZK-3 forbids drawing masks from it anyway.
- **Determinism has a cost.** RFC-0064 kept a state's budget as the only record of the rows an epoch has burned. Once an epoch's commitment can be made again, a lost, restored or copied state, or a remade link, can write a fresh budget over burned rows ([`docs/zk/simulator.md`](../zk/simulator.md#5-aborted-sessions) §5, R6). Telling a fresh state from a rebuilt one needs a record the state files do not hold, and a fixed place to look for them.

## Goals

- Every value a masked weight commitment draws is a function of `masks_seed` and public data, the same in every implementation and on every device.
- Each committed row's mask and each blinding row's values are addressable alone.
- The values are exactly uniform on `Fp`, and independent across rows, matrices, commitments and epochs, under the random-oracle model the simulator argument takes for hashes ([`docs/zk/simulator.md`](../zk/simulator.md#2-model) §2).
- One primitive, RFC-0004's BLAKE3, at one compression per 64 bytes of output.
- **No row reserved twice and no column opened past `P_W`**, decided from the model entry's directory and from the artifacts the operator presents. Every loss, crash, restore that leaves a refuting file, and concurrent process is covered. What remains is stated as a rule on files and checked against an exhaustive model of the rules.
- A conformance vector from an independent oracle, buildable from this text, and a version that any change must bump.

## Non-Goals

- **The advice's per-proof mask stream** (RFC-0011 'Streaming procedure', RFC-0018 'Crash consistency', [#156](https://github.com/AbdelStark/tessera/issues/156)). It may take 'The stream' below under a domain of its own.
- **How the masked rows are opened, reserved and budgeted** within an epoch: RFC-0012, RFC-0013, RFC-0064 and RFC-0065.
- **The verifier.** It never holds a seed and never expands one. Soundness does not depend on the masks, which are the prover's own choice (RFC-0064 'Zero knowledge'). Its refusal of narrow weight matrices is [#564](https://github.com/AbdelStark/tessera/issues/564)'s.
- **`tessera-protocol/1`,** which masks no weight row and commits no blinding row (RFC-0040).

## Proposed Design

### The seeds

As RFC-0040 and RFC-0064 give them:
- **`K`**, the model's refresh key, is 32 bytes from the operating system's CSPRNG, drawn when epoch 0 is committed and kept apart from every state file, in its own artifact ('The model entry').
- **`masks_seed_e = H("weights-refresh", K ‖ u32(e))`** is epoch `e`'s seed, and epoch `e`'s `WeightCommitmentState` holds it.
- **Nothing else seeds the expansion:** no per-proof value and no transcript value (INV-ZK-3).

### The stream

For a domain `d` of RFC-0004 and an input `x`:

```text
S(d, x)  = BLAKE3.derive_key(context = "tessera/v1/" + d, key_material = x), extended output
w_t      = u64_le(S(d, x)[8t .. 8t + 8]),                         t = 0, 1, 2, …
v_i      = the i-th word w_t < P, in increasing t,                  i = 0, 1, 2, …
```

- **The extended output** is BLAKE3's root output blocks `0, 1, 2, …` (the BLAKE3 specification, §2.6). Its first 32 bytes are RFC-0004's `H(d, x)`.
- **Words** are read in order, eight per 64-byte output block.
- **Values.** A word below `P` is the next value. A word of `P` or more is skipped, never reduced. Skipping one has probability `(2^32 − 1)/2^64 < 2^{-32}`: about 19 words in a whole 405B epoch.
- **Uniform.** Each value is exactly uniform on `Fp` and independent of the others, given uniform words. Reducing the word modulo `P` would give the values below `2^32 − 1` twice the probability of the others, which RFC-0004 refuses for challenges and this RFC refuses for masks.
- **No bound.** The prover's loop has none, as RFC-0004's challenges have none on the prover's side.
- **The one secret-dependent branch allowed.** The acceptance test is public and fixed, `w_t < P`. It reads the candidate word alone, and a rejected word is discarded whole.
  - **What follows.** In the random-oracle model, which words are skipped is independent of every value any stream yields and of every other secret. So neither the branch's timing nor the addresses a parallel kernel compacts the values to reveal them. INV-ARCH-2, amended, admits exactly this, for the streams of RFC-0004.
  - **The skip pattern stays secret-class.** It is a function of the seed (05 §3, OBS-6). No log, metric, report or `Debug` form carries a stream's skips, its position or their count.
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
  - **A full-rate code.** Once `k_T + P_W ≥ n_T`, the masked code is all of `Fp^{n_T}`. A row that differs from the true combination passes each queried column with probability `(n_T − 1)/n_T` or more, and nothing is left of the proximity bound.
  - **An exact forgery.** Once `k_T + P_W > n_T`, the row `u + (X^{n_T} − 7^{n_T})·h`, with `deg h < k_T + P_W − n_T`, passes every column check. Its values on `H` range over `k_T + P_W − n_T` dimensions, enough to meet any one evaluation claim, and any values at all once `P_W ≥ n_T`. `d_model = 128` at `live-64` is such a case: `k_Q = 2^{14}` and `P_W = n_Q = 2^{16}`.
  - **Not RFC-0011's bound.** RFC-0011's `P ≤ k` is the advice's decoding bound at rate 1/2. It does not bound the weights' soundness.
  - **The rest is #564's.** A prover's refusal does not bind a malicious prover. The verifier's refusal of such a claim list, and whether narrow matrices are refused or padded to `k_W`, are [#564](https://github.com/AbdelStark/tessera/issues/564)'s. If #564 pads them, this rule follows its padding.

### What it fixes

- **A commitment.** A masked commitment is a function of `D`'s weights, the claim list (its layout and shapes, RFC-0040), the row block `B`, the epoch and `masks_seed_e`, hence `K`: its codewords, column chains, roots and `R_W`.
- **A link.** It is a function of `K`, the epoch, the weights and the claim list, as RFC-0064 'One link per epoch' requires: both commitments are, its transcript is a function of their roots and rows, and its nonce is the least.
- **Any row, alone.** Every row's mask and every blinding row's values can be drawn without the rows before them. That serves devices committing in parallel, a proof's combined rows, a link's rows in both epochs, and the store's fallback for the rows it answers.

### The model entry

```rust
pub struct WeightRefreshKey {            // schema 0x0304, version 1, secret, .twk, in the key store
    pub root_0: Digest,                  // R_W of epoch 0: the model entry it serves
    pub key: [u8; 32],                   // K
}
pub struct WeightEpochRecord {           // schema 0x0305, version 1, private, epochs.twe in the entry
    pub roots: Vec<Digest>,              // R_W of epochs 0, 1, …, latest, in order
    pub marks: [BudgetMark; 10],         // the latest epoch's, in WeightMatrix order
}
pub struct BudgetMark {
    pub next_blinding_row: u32,          // the first blinding row not burned
    pub opened_columns: u32,             // never below the number of distinct columns opened
}
```

- **TCE.** The fields in order. `roots` lists at least epoch 0's root. A decoder refuses an empty list with `ERR-CODEC-NONCANONICAL` at its length, as RFC-0067 refuses a required list that is empty.
- **The latest epoch** is `roots.len() − 1`, and its root is the last.
- **The marks.** Within an epoch, the marks only grow. An append replaces them with the new epoch's start: row 4, and the link's distinct columns. `opened_columns` counts conservatively: it may exceed the distinct columns opened, never fall below.
- **One directory per model entry**, under fixed names:
  - `epochs.twe`, the record;
  - `epoch-<e>.twc` and `epoch-<e>.twr`, epoch `e`'s state and public commitment;
  - `link-<e>.twl`, the link from epoch `e` to `e + 1`.

  `<e>` is decimal, with no leading zero.
- **What else the directory may hold.** It may hold the atomic writer's leftover temporary files (`.<name>.<16 hex digits>.tmp`, RFC-0003's `write_file`) and `run-manifest.json` (05 §6), which every check ignores. Any other file, a name such as `epoch-01.twc` included, makes the entry check refuse.
- **Commands take the entry's directory** with `--entry DIR`, and write a state, the record or a link nowhere else. That covers `commit-weights`, `weights refresh`, and `prove` in `tessera-protocol/2`. A published `.twr` or `.twl` may be copied elsewhere, but only the entry's own files decide anything.
- **The key** lives in the operator's key store, apart from the entry and every state file. It is written once, with mode 0600, by atomic rename. Only `commit-weights` and `weights refresh` read it, never a proof. A command that reads it refuses a key whose `root_0` is not the record's first root.
- **The record** is written by atomic rename, as every file of the entry is. It holds nothing derived from a secret: the roots are public, and the marks are the budget's cursor (05 §3 PRIVATE). So a run manifest names it with its envelope digest (05 §6).
- **One model.** A key, a record or a state of another model entry fails the root checks below, so no field names `D`.

### One proof at a time per entry: the lock

The lock is RFC-0067's lock for the verifier's ledger, applied to the entry directory:
- **On the directory itself.** A writer takes an exclusive advisory lock on the entry directory, through `std::fs::File::try_lock` (`flock` on Linux and macOS), resolved once when it opens the entry.
  - A second writer does not wait: its `try_lock` fails, and it is refused with `ERR-RES-IO`, as RFC-0067 refuses one. It tries again later.
  - Removing or renaming a file in the directory releases nothing, and neither does renaming the directory, which keeps its inode.
  - The system releases the lock when its process ends, so no operator step clears it.
- **Before each rename** a writer checks, by device and inode, that the file it replaces is still the one it last read or wrote, and that the entry's path still names the directory it locked. Otherwise it writes nothing, `ERR-RES-IO`.
- **Local file systems only.** The entry directory is on a local file system. The prover cannot decide whether a shared file system honours `flock`, so none is supported, as RFC-0067 rules for the ledger.
- **The span.** A writer holds the lock for its whole transition:
  - `commit-weights`, throughout;
  - a proof, from its entry check through its `record_queries` or `abort`;
  - a link, from its entry check through closing epoch `e`;
  - a rebuild.
- **One proof at a time per entry.** The span is a whole proof because `reserve`'s column condition counts only recorded queries. Two reservations open at once would both pass it, and together could open up to `τ_W` columns past it. In live mode the verifier's pace holds the lock, up to `ERR-PROTO-TIMEOUT`.
- **One node per entry.** The record does not partition the budget between nodes.
- **Commands that only read**, such as `tessera inspect`, take no lock and decide nothing.

### The entry check

Every command that writes runs it first, under the lock, and refuses on the first failure:
1. **The record** is present: else `ERR-USAGE`, and the model entry has failed closed.
   - A record that cannot be read is `ERR-RES-IO`.
   - One that does not decode carries its decoder's code, `ERR-CODEC-*`.
2. **The directory holds only its fixed names** and the files 'The model entry' lets every check ignore: else `ERR-USAGE`.
3. **No file of the entry is past the record or differs from it.** Each failure here marks the record stale, `ERR-USAGE`.
   - No `epoch-<e>.twc` or `epoch-<e>.twr` exists for `e` past the latest epoch, and none has a root other than the record's for its epoch.
   - No `link-<e>.twl` exists for `e + 1` past the latest, and none names roots other than the record's.
   - **The latest epoch's state, if present, is not closed:** no matrix's `next_blinding_row` is past `B_T − 4`. A link always appends before it closes its epoch, so a closed latest state means an append was lost.
4. **Presented artifacts.** The operator may present artifacts with `--present`, repeatable. Each can only refuse or burn, never grant a reservation:
   - **A later epoch's root:** a published `.twr` or `.twl`, or the root a verifier's ledger holds as current, naming an epoch the record does not list or a root other than the record's.
     - **Authenticated, it marks the record stale,** `ERR-USAGE`. That is a `.twl` whose old root the record lists and that passes RFC-0064's checks against the claim list, or a root that `weights refresh` regenerates from `K` for its epoch.
     - **Not authenticated, it refuses this command only,** `ERR-USAGE`, and marks nothing stale. So an artifact a malicious verifier forged cannot retire the entry.
   - **A reservation of the latest epoch:** the rows a verifier's ledger holds for it (RFC-0019 'Ledger'), or the `WeightReservation` of a `Round1` the operator kept outside the entry. It raises the marks to cover it:
     - the cursor goes past its last row;
     - the column count grows by `τ_W` for every `r_T` rows it adds, at the largest `τ_W` of the layout's parameter sets.

     It is never authenticated. A false one only burns rows and columns, at most the rest of the epoch, which costs a refresh.

A record that fails item 3, or an authenticated item 4, is stale. The prover refuses every `reserve`, link and commitment for the entry, which fails closed. The operator then starts a new model entry under a new key, as RFC-0064 does when `K` is lost.

### Fresh and rebuilt states

- **A fresh state** is written only by the transition that appends its epoch to the record, right after the append:
  - **`commit-weights`, for epoch 0.**
    1. It refuses an entry directory that holds any file but those every check ignores, and an existing key path, with `ERR-USAGE`.
    2. It draws `K` and commits epoch 0, refusing a narrow matrix.
    3. It writes the key, then a record of one root, whose marks are row 0 and no column.
    4. It writes epoch 0's state, from row 0, then the `.twr`.

    Epoch 0 is therefore committed once per key. A key written with no record beside it is unused: the operator may move it aside and commit again.
  - **The first link from the record's latest epoch `e`, for `e + 1`.**
    1. In one write of the record it appends `R_W'` and sets the marks to `e + 1`'s start: row 4, and the link's distinct columns.
    2. Only then does it write `e + 1`'s state, from row 4 (RFC-0064 'The budget'), then `e + 1`'s `.twr` and the link.
    3. Then it closes epoch `e`.

    Every file of epoch `e + 1` is written after the append, so no crash leaves a file past the record.
- **Every other state is rebuilt.** That covers a state regenerated from `K` after a loss, and one written by a link remade from an epoch that is no longer the latest.
  - In a rebuilt state, every matrix with committed rows has `next_blinding_row = B_T − 4`, where `reserve` always fails and the link's rows begin, and every `opened_columns` is empty.
  - It serves only the link from its epoch, which opens what any link from that epoch opens (RFC-0064 'One link per epoch').
- **`reserve`** works only on the state at `epoch-<latest>.twc`, whose public part hashes to the record's last root. Every other state is `ERR-USAGE`. Its conditions:
  - **The latest state exists.** If it is lost, `ERR-WITNESS-BUDGET`: `weights refresh` rebuilds it and links it.
  - **The previous epoch is closed:** for a latest epoch `e ≥ 1`, `epoch-<e−1>.twc` is closed (RFC-0064 'Not yet usable'). Else `ERR-WITNESS-BUDGET`: `weights refresh` completes the unfinished link first, rebuilding `e − 1`'s state if it is lost, then closing it.
  - **The cursor** is the larger of the state's `next_blinding_row`, the record's mark and a presented reservation's.
  - **The column count** is the larger of `|opened_columns|` and the mark's count, raised by a presented reservation.
  - **Rows and columns.** RFC-0012's preconditions hold on those, else `ERR-WITNESS-BUDGET`.
  - **The order.** The record's marks are written first, then the state, then `Round1` is built.
- **`record_queries`** adds to each mark's count the indices not in the state's `opened_columns`, at least the truly new ones. It writes the record, then the state.
- **A latest-epoch state behind the record's marks** has its cursor raised to them, and its rows below the marks count as burned. That covers a restored state, or a crash between the two writes. A record behind the latest epoch's state is raised to it.
- **`weights refresh`**, in order:
  1. completes every unfinished link below the latest epoch: one whose `link-<e>.twl` or closed `epoch-<e>.twc` is missing;
  2. rebuilds a lost latest state;
  3. links from the latest epoch when its `reserve` would fail for rows or columns.
- **No pending reservation decides a link.** RFC-0064's precondition, that every reservation of epoch `e` has recorded its queries or aborted, is decided by the lock. A link holds it, and a proof holds it from its entry check to its `record_queries` or `abort`.
  - **Without #156,** finding the lock free means every earlier holder has died and its reservation is aborted. Its rows stay burned, and its columns were never revealed.
  - **#156's resume** holds the lock and resumes only while the record's marks are its reservation's, which an append replaces.
- **Crashes fail safe.**
  - A crash between a link's append and the fresh state's write leaves no state. The state written on restart is then rebuilt, since its epoch's append is no longer the last: one refresh lost, never a row.
  - A crash between the record and the state of a `reserve` or a `record_queries` leaves the state behind the marks, which raise it.
- **After a lost state.** The prover rebuilds the record's latest epoch at its canonical path and links it. That link appends the successor, which is fresh. Older epochs it rebuilds serve their links only.

### What the prover cannot detect

- **Detected,** and refused or made safe by the rules above:
  - a crash at any point;
  - a lost state, record, key or directory;
  - a second process on the entry, the lock file or any other file removed included;
  - a restore that leaves a refuting file in place (below), whatever else it rolls back.
- **The refuting files.** A rollback of the record is refuted by any one of these, current:
  - within the record's latest epoch, that epoch's state;
  - across a link, any state, `.twr` or link of a later epoch, or the record's latest epoch's state, which the link closed.
- **The residual risk.** Two cases cannot be detected from the entry's own files:
  - **A rollback of the record** in which every refuting file is also rolled back or lost. It may come by a restore, or by a loss then a restore, from one backup or several, in place or under another path.
  - **A copy of the entry** that proves beside it, under another path on the same node or on another node.
  - **The bound.** In either case, the proofs made after the rollback, or from the copy, reserve again every row and open again every column revealed since the backup, in every epoch the rollback spans. That is up to `B_T − 8` rows per matrix per epoch: 4,088 at 405B, and 12,272 for `ChannelParams`. It is also up to `P_W − 2τ_link` columns per matrix per epoch, 65,292 at `live-64`, past `P_W` in all. Past `P_W`, Lemma 3 no longer hides the opened rows.
- **What a presented artifact catches.** Presenting the right artifact to every command catches both cases, in the exhaustive model of these rules:
  - **Across a link,** the latest epoch's published `.twr`, its link, or a ledger's current root.
  - **Within an epoch, and a copy,** the rows of a verifier's ledger that has seen every proof since the backup, or the reservations of `Round1`s kept outside the entry. Every proof's `Round1` publishes its reservation (RFC-0065), and every verifier that checked it keeps the rows (RFC-0019 'Ledger').
- **What the verifiers catch.**
  - **Rows.** A verifier whose ledger holds a reused row refuses the proof at RFC-0019 step 5, `Reject(Budget)` ([#158](https://github.com/AbdelStark/tessera/issues/158)).
  - **A superseded root.** A verifier that accepted a later link refuses a proof against the earlier root at step 4.
  - **In live mode,** both checks run before the verifier reveals its seed share, so the prover sends it no combined row.
  - **What they miss.** A non-interactive proof is whole when it is sent, so a verifier that refuses it has already received the reused pairs. A verifier whose ledger never saw the earlier proofs accepts. Verifiers that pool their views see both.
  - **Columns.** No verifier catches reused columns: `VerifierLedger` records no opened columns (RFC-0019 'Ledger'), though step 13 says it does.
- **The operator's duty,** as RFC-0064 already makes it for a state:
  - never restore the record or a state;
  - never copy an entry to prove beside it;
  - after any restore, present the latest published `.twr` and a ledger's rows for the latest epoch to every command.

### The regeneration checks

- **A regenerated commitment.** A prover that regenerates an epoch's whole commitment from a seed checks that its `WeightCommitmentPublic` hashes to the record's root for that epoch, before anything else. A difference is `ERR-USAGE`: the key, the state, the weights, the claim list or the build's expansion is not the epoch's.
  - **Before any write or send.** The check comes before any state, record entry or link is written from the regenerated commitment, and before anything is sent. A link persisted from a wrong commitment would be kept and resent on every restart (RFC-0064 'One link per epoch').
  - **A part of a commitment** cannot be hashed to `R_W`. What a proof draws in part is checked by the next item.
- **Each combined row, before it is sent.** A prover that draws the masks and blinding values its combined rows need, rather than holding the whole commitment, checks each combined row at one column before sending it. The column is drawn from its CSPRNG, so that no structured error can hide at a fixed column. It is never sent, and nothing about it is revealed. A failure is `ERR-USAGE`.
  - **With the codeword store**, the column is read from the store and checked against the matrix root, as RFC-0012 'Column answers' reads columns.
  - **Without the store**, the prover regenerates the column's symbol in every row, then the column's chain and leaf. It checks the leaf against the root through the matrix's Merkle tree, which a node that answers columns keeps: `2n_T` digests per matrix.
  - **Why one column suffices.** A wrong mask coefficient or blinding value adds to the combined codeword a polynomial of degree below `k_T + P_W`. One wrong value adds a polynomial that is non-zero at every point of the coset, since neither `Z_H` nor a Lagrange factor vanishes off `H`. A wrong expansion's pseudorandom error vanishes at a given point with probability about `1/P`.
- **Evaluated column answers.** A column answer evaluated rather than read from the store is checked against the matrix root before it is sent, as a stored one is: its leaf and the multiproof the answer carries anyway (RFC-0012 'Column answers').

### Errors

One code per failure. `ERR-WITNESS-BUDGET`'s remedy is `weights refresh` (04 §5), and every failure it marks below is one that command repairs.

| Failure | Code |
|---|---|
| The entry's lock is held, or a file or the directory changed under a writer (device and inode) | `ERR-RES-IO` |
| The record or the key cannot be read | `ERR-RES-IO` |
| The record or the key does not decode, an empty `roots` included | the decoder's `ERR-CODEC-*` (`ERR-CODEC-NONCANONICAL` for an empty `roots`) |
| The record is absent | `ERR-USAGE` |
| The record is stale: a file past it, a closed latest state, or an authenticated later artifact | `ERR-USAGE` |
| A file the entry may not hold | `ERR-USAGE` |
| A presented artifact that differs from the record and is not authenticated | `ERR-USAGE`, this command only |
| The key is absent when `weights refresh` needs it, or its `root_0` is not the record's first root | `ERR-USAGE` |
| A regeneration check or a combined row's column check fails | `ERR-USAGE` |
| `reserve` on a state other than the latest epoch's, or of another root | `ERR-USAGE` |
| `reserve` with the latest state lost, or the previous epoch not closed | `ERR-WITNESS-BUDGET` |
| `reserve` with too few rows or columns left at the latest epoch, a rebuilt state included | `ERR-WITNESS-BUDGET` |
| `commit-weights` with a narrow matrix, a non-empty entry or an existing key path | `ERR-USAGE` |

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
| `masks_seed_e` | SECRET key material | On the heap inside the state's `Secret` (RFC-0040), filled straight from the derivation's output reader, never returned by value |
| A hasher with a secret input: `weights-refresh`'s, over `K`, and a stream's, over `masks_seed ‖ cid_T`, whose chaining value draws every row of one matrix in one domain | SECRET key material | `Secret<Box<blake3::Hasher>>`, cleared when it drops |
| A stream's output reader | SECRET key material | `Secret<Box<blake3::OutputReader>>`, constructed in its box, or with any stack copy cleared, as [#567](https://github.com/AbdelStark/tessera/issues/567) requires |
| Mask coefficients, blinding values, a stream's skip pattern, position and count | SECRET | `SecretBuf` and the commitment's `Secret` rows (OBS-6) |
| The epoch record | PRIVATE | `epochs.twe` in the entry: public roots and the budget's cursor |

- **Never through the public hasher.** `tessera_hash::Hasher` and `tessera_hash::hash` derive `Clone` and `Debug`, hold the state by value, and never clear it. `blake3`'s `zeroize` feature implements `Zeroize` but clears nothing on drop. So no secret-class input goes through `tessera_hash`, `K` included: `weights-refresh`, `weights-mask` and `weights-blinding` each take a `Secret<Box<blake3::Hasher>>`. The file that does it is a reader in `secret-access.toml`. The CPU prover's advice stream (`crates/tessera-prover/src/cpu/random.rs`) is no model: it moves its reader through the stack, which #567 fixes.
- **Never `Debug`, readers included.** `blake3::OutputReader`'s `Debug` prints its position: the words a stream has drawn, and so its skips. The rule against `Debug` is load-bearing for readers, not cosmetic.
- **Nothing derived from them is logged.** No log, metric, error or report carries a seed, a stream's words, its position or its skips (OBS-1).

### What is revealed, and why it stays hidden

The expansion sends nothing itself. Every revealed value that depends on it is hidden by a value it draws, under the PRG hybrid of the simulator argument (§2):

| Revealed | Hidden by | What it needs from the expansion |
|---|---|---|
| Opened column symbols, at most `P_W` distinct per matrix and epoch, the link's included | Each row's mask (Lemma 3) | Each row's `m_j` uniform, independent of every other row's and of the weights |
| Combined rows and their combined masks: evaluation, proximity, the link's difference | Each row's designated blinding pair or mask row, and that pair's own masks (Lemma 5, RFC-0013 'Masks of combined masks') | Each blinding row's values uniform, independent of every other row's values and of every mask |
| Blinding claims `q` | One functional of a fresh pair (RFC-0013 'Hiding openings') | The same |
| Masked evaluations `W'(q)` and `ChannelParams`' `v_i` | A mask row `R`, two blinding rows (RFC-0013, RFC-0065) | The same |
| Two epochs' columns and the link | Each epoch's own masks | Independent seeds: `H("weights-refresh", K ‖ u32(e))` per epoch |
| Every one of these, across the proofs of an epoch | Each pair and mask row used by one proof (R6) | No row reserved twice: 'The model entry' to 'What the prover cannot detect' |

- **A malicious verifier** controls no input of the expansion. The seed is secret, the addresses are public and fixed before any session, and the transcript plays no part. A root it supplies can only refuse, and a reservation it supplies only burn: neither grants a reservation. A forged later root that the prover cannot authenticate refuses one command and retires nothing.
- **`ε_PRG`.** In the random-oracle model, a verifier learns anything about a stream only by querying the oracle at an input that holds its 256-bit seed. That has probability at most `Q_H·2^{-256}` for `Q_H` queries. It counts once per epoch a view spans, and once more for `K` (RFC-0064 'Zero knowledge').
- **Leaks.** A leaked state gives its own epoch's seed, and so every mask and blinding row of that epoch, which with the epoch's combined rows and columns reveals folds of the weights. A leaked `K` gives every epoch's. RFC-0064's forward secrecy among epochs holds, and proofs never read `K`.
- **A malicious prover** gains nothing. The verifier never checks the expansion, and a prover that expands otherwise risks only its own secrecy and the determinism of its links.

### Cost at 405B

At RFC-0012's 461,310 committed rows, 49,144 of them blinding rows (826 fewer with `ChannelParams`' 56 data rows, RFC-0065):
- **Per epoch.** The commitment draws 461,310 mask streams of `2^{16}` values and 49,144 blinding streams of `2^{20}`: 3.02·10^10 and 5.15·10^10 values, 654 GB of output and about 1.02·10^10 compressions. A counter mode over `H` would take twice as many with the chaining value of its 64-byte prefix cached, and four times as many without.
- **Per proof.** A combined row's mask is `Σ_j c_j·m_j` over every row its coefficients reach, and its values `Σ_j c_j·A_j` likewise.
  - **Under RFC-0012's generator.** The evaluation rows reach every data row, and the proximity row every data and blinding row. So a proof draws every mask and every blinding value, the whole 654 GB: about 1.2 GPU-s at RFC-0004's 530 GB/s per H100, unless the prover keeps them.
  - **If #394 keeps the blinding rows out of the proximity combination,** a proof draws the data rows' masks and its reservation's rows: 412,166 rows × `2^{16}` × 8 bytes, 216 GB, about 0.41 GPU-s.
- **Per link.** Its rows reach both epochs' rows: about 1.3 TB, once per epoch.

### The conformance vector

`spec/vectors/weight-masks/` holds `masks.json`, written by `generate.py`. That oracle is standard-library Python, written from this RFC. It takes its BLAKE3 from `hash/`'s oracle, which checks it against the official vectors at extended length, and its encoding and layout from `encoding/` and `weights/`. The file holds:
- **`refresh`.** A fixed `K` and `masks_seed_e` for `e` in 0, 1 and `2^32 − 2`, the last epoch a link can reach: RFC-0064 refuses a link from epoch `2^32 − 1`.
- **`streams`.** For both domains, under the `cid_T` of `tiny-2-minimal`'s `Embed` and `ChannelParams` at epoch 0 (below), with epoch 0's seed:
  - the inputs: row 0, the matrix's last row (`m_T + B_T − 1` for `weights-mask`, `B_T − 1` for `weights-blinding`) and `2^{64} − 1`;
  - for each, the first 24 words, three output blocks, and the values they give.
- **`rejection`.** One input in each domain whose stream has a word of `P` or more among its first eight. Each is found once by a search outside the oracle, about `2^{29}` BLAKE3 calls, and pinned by its domain and bytes. The oracle checks the word and that the values skip it.
- **`models`.** Both are committed in `tessera-protocol/2` with the row block `B = 4`, at epochs 0 and 1 from the fixed `K`.
  - **`tiny-2-minimal`** at `test-small/1`. Its `PublicQuantSpec`, the one its `test-small/1` claim list in `spec/vectors/claims/` holds, gives a `ChannelParams` to five weight types: Q, K, O, Gate and Down. So `e_CP = 5`, `B_CP = 32 + 2·4·4 = 64`, and `r_CP = 14`. Its `ChannelParams` commits structural rows 2 to 5, 8 to 11 and 14 to 15, so a build that indexes by `r` fails it.
  - **`synthetic`** (`spec/vectors/weights/`). At `test-small/1` it is refused before anything is drawn: its `K` and `V` have `k_T = 128 < k_W`. It is committed under the test layout `WeightLayoutParams { log2_k: 7, log2_inv_rate: 2, mask_coeffs: 32, blinding_rows: 32 }`. There every committed matrix has `k_T ≥ 128`, and its `ChannelParams`, with no claim, no committed row and `B_CP = 0`, draws nothing.

  Per committed matrix:
  - `m_T`, `e_T`, `B_T` and `cid_T`;
  - one SHA-256 (`hashlib`, independent of the BLAKE3 under test) of all its mask coefficients, rows in committed order, each `u64` little-endian, and one of all its blinding values;
  - the first and last values of each;
  - the root.

  Then `R_W`.

`cargo xtask gen-vectors` writes it, and `tessera-pcs` checks it. The verifier reads none of it, since it never expands a seed.

## Alternatives Considered

1. **A counter mode over `H`:** `H("weights-mask", seed ‖ cid ‖ u64(j) ‖ u64(block))`, 32 bytes each. Every implementation already has `H`. Rejected: twice the compressions per 405B epoch, 2.0·10^10 against 1.0·10^10, even with the 64-byte prefix's chaining value cached, and four times without it; and twice the cost of every proof's draws. The extended output is BLAKE3's own, and `hash/`'s oracle and `tv-hash` already check it against the official vectors at extended length.
2. **One stream per matrix or per epoch**, as the CPU prover's per-proof advice masks are drawn today. Rejected: a row's values would need every row before it. The fallback, the reservation's rows, the link's rows and parallel commitment all read rows alone.
3. **A stream per row, values and mask together.** It saves 49,144 stream starts per 405B epoch. Rejected: a reader of masks alone would draw `k_T` values first for each blinding row. Such readers are a proof under #394's exclusion of the blinding rows from the proximity combination, and a fallback that answers data rows only.
4. **AES-128-CTR or ChaCha20.** Rejected: a second primitive on the commitment path, which RFC-0004 keeps to BLAKE3. On a GPU, BLAKE3's 7 rounds per 64 bytes cost less than ChaCha20's 20.
5. **128-bit words reduced modulo `P`.** A bias near `2^{-64}` per value, and twice the stream. Rejected for exactness, as RFC-0004 rejects it for challenges.
6. **The matrix id in place of `cid_T`.** It saves 31 bytes and one compression per stream. Rejected: two commitments from one seed would share every mask.
7. **Drawing from the transcript.** Rejected: INV-ZK-3. Its values are public.
8. **Fresh masks per commitment, from the CSPRNG**, with no expansion to fix. Rejected: RFC-0064's links would not be remade as the same bytes, which is what 'One link per epoch' rests on.
9. **Telling a rebuilt state from the state itself.** Rejected: a fresh state written a second time, by a remade link or after a loss, is byte for byte the first. Only a record of what was written decides it.
10. **The record in the key store, beside `K`.** Rejected. Every `reserve` reads the record, which would bring `K`'s store into every proof. A record restored with the key store alone would also be judged against no state, while beside the states its staleness shows.
11. **A free path for every state and link.** Rejected: a stale record could not find the later states that refute it.
12. **Partitioning the budget between nodes in the record.** Rejected for now: one node proves per entry. Several would need the record to arbitrate between them, a shared counter with compare-and-swap, which no local file gives.

## Drawbacks

- **Positions move.** A value's word depends on how many words before it were skipped, so a GPU kernel that draws words in parallel must compact them. The fast path, no skipped word, holds for a row with probability about `1 − 2^{-12}` at `2^{20}` values. The conformance vector pins a skip in each domain.
- **Per-proof draws.** Under RFC-0012's proximity generator a 405B proof draws the whole 654 GB, about 1.2 GPU-s, or reads a store of it. RFC-0012 counts that only for its fallback, and 08 §2 still says the weight column answers add no GPU time, under a weight-pass budget of 0.24 GPU-s that alerts above 0.5. OQ-0073-5 points to the cost-model issue, [#568](https://github.com/AbdelStark/tessera/issues/568).
- **More state to keep.** Two more artifacts, a fixed directory and a lock per model entry. The operator must never restore the record or a state, and must present the latest published artifacts after any restore.
- **One proof at a time per model entry.** The lock spans a whole proof, and in live mode the verifier's pace holds it. Throughput per model is one proof's until the budget can be partitioned.
- **A lost record ends the model entry,** as a lost key does.
- **Small models leave `tessera-protocol/2`** until #564 decides whether to pad them: every committed matrix needs `k_T ≥ k_W`.
- **One more normative format** that every prover implementation must follow, which no verifier test can check.

## Migration / Rollout

- **#147** implements:
  - the domains;
  - the expansion and the conformance vector;
  - the masked commitment;
  - the model entry: its directory, its lock, the key and the record, and 02 §2.2's `--entry`, `--refresh-key` and `--present` flags for `commit-weights`, `weights refresh` and `prove`, with RFC-0040 'Command line';
  - `reserve`'s rules on the latest epoch and the marks;
  - the regeneration checks.

  Both workspaces' domain lists and `spec/vectors/hash/`'s oracle follow RFC-0004's table (RFC-0031), `weights-link` and `weights-refresh` included, which neither has yet. 03 §5's catalog gains `0x0304` and `0x0305`, and 02 §2.2 its flags, with the code that reads them: `tessera-codec`'s registry test reads the catalog.
- **#153** remakes links with the expansion, in the entry directory, and writes successors fresh or rebuilt by the record.
- **#156** may draw the per-proof advice masks from 'The stream' under a domain of RFC-0004 of its own. Its resume path holds the entry's lock and never writes a state: a resumed session uses the state its reservation was persisted in, and resumes only while the record's marks are that reservation's.
- **#564** gives the verifier's refusal of narrow matrices, before any `tessera-protocol/2` verifier ships.
- **#174** compares CPU and GPU commitments on it.
- **`tessera-protocol/1`** is unchanged.

## Testing Strategy

- **The vector.** `tessera-pcs` reproduces `masks.json`: every stream, both rejections, the seeds, and each model's digests, roots and `R_W`. It also reproduces `synthetic`'s refusal at `test-small/1`.
- **The skip.** A stub word source drives the skip branch: the words `P`, `2^{64} − 1` and `P − 1` give one value, `P − 1`.
- **Independence.** Changing the seed, `cid_T`, the row or the domain changes the first output block. Two epochs from one `K` give different roots for the same weights.
- **Determinism.** Regenerating a commitment from one `K` and epoch gives the same bytes, rows drawn in any order or in parallel.
- **Narrow matrices.** `commit-weights` refuses `k_T = k_W/2` with `ERR-USAGE` before drawing, and accepts `k_T = k_W`.
- **The regeneration checks.**
  - A state whose seed, epoch or weights are not the record's root's is refused with `ERR-USAGE` before any write or message.
  - A build whose expansion differs fails the check of a combined row at one column, with the store and without it.
  - An evaluated column answer with one wrong symbol is caught against the root before it is sent.
- **The model entry: the gate.** An executable model of these rules, a test in S4, replays every combination of the six entry files: the record, epochs 0's and 1's states and `.twr`, and the link. Each file is current, restored from one of three backups or lost. The model also replays a crash after every write of every command, two processes, a removed file, and a copy.
  - Every outcome that reuses a row or opens a column past `P_W` is in 'What the prover cannot detect''s residual class, with nothing presented.
  - None remains once the latest `.twr` and a ledger's rows are presented.
  - These are refused or made safe, 0 rows reused:
    - every crash;
    - a second process, the lock file removed included;
    - a restored latest state alone, raised to the marks;
    - a restored record with a refuting file present;
    - a closed latest state under a rolled-back record;
    - a lost latest state.
  - **The negative control.** RFC-0012's `record_queries` row as it read before this RFC, raising no mark, over-opens columns under a restored state.
- **The model entry, case by case.** Each of these is a test:
  - `commit-weights` refuses a non-empty entry or an existing key;
  - a key whose `root_0` is not the record's is refused;
  - a lost or stale record refuses every transition;
  - a file of a non-canonical name is refused;
  - a presented later root that is not authenticated refuses this command and marks nothing stale;
  - `reserve` before the previous epoch is closed is `ERR-WITNESS-BUDGET`, and `weights refresh` completes the link;
  - an exhausted epoch-0 state and a rebuilt one both refuse `reserve`, and a link from either writes a fresh successor exactly when epoch 0 is the record's latest.
- **Secrets.** A hasher with a secret input and a stream's reader have neither `Clone` nor `Debug` (`compile_fail`), and are cleared on drop. No event carries a skip, a position or a count.
- **Hiding.** #147's distinguisher separates two weight sets in three negative controls: an expansion that ignores the row, giving two rows one mask; one that ignores the epoch; and a successor written fresh over a lost epoch. It does not separate them under this RFC.

## Open Questions

- **OQ-0073-1.** Settled by 'The model entry': `K` lives in its own artifact in the key store, naming the entry by its epoch-0 root.
- **OQ-0073-2.** Settled for the prover by 'Narrow matrices: until #564, `k_T ≥ k_W`'. The verifier's refusal is #564's.
- **OQ-0073-3.** Where `pcs/weights/{T}/xi` goes in `tessera-protocol/2`, which RFC-0049 leaves to #147. RFC-0014's transcript schedule and RFC-0049's schedule must decide it, by amendment, before #147's opening slice.
  - RFC-0014 still lists it at row 7, after the stages, so the fold pass could not form the proximity row in its one read (INV-WC-4).
  - Recommended: row 3, right after the seed, as in `tessera-protocol/1`, with `⌈log2(m_T + B_T)⌉` coordinates. The proximity pair is fixed by `Round1`'s reservation before the seed.
  - The pair's coefficients themselves are [#394](https://github.com/AbdelStark/tessera/issues/394)'s (R1, R2).
- **OQ-0073-4.** `WeightReservation`'s bytes, which RFC-0065 defines in content only. RFC-0036, which owns `Round1`'s schema version 2, and RFC-0065 must decide them, by amendment, before #147's `Round1` slice. Recommended:
  - `WeightReservation { matrices: Vec<MatrixReservation> }`, with `MatrixReservation { matrix: u8, first_row: u32 }`;
  - placed after `advice` in `Round1`'s schema version 2, as RFC-0014's sketch orders it, and absorbed with `Round1` under `round1`;
  - ids out of order are `ERR-CODEC-NONCANONICAL` at the id's byte, as RFC-0036 'Matrix ids' rules.
- **OQ-0073-5.** Paying for the per-proof draws ('Cost at 405B'): about 1.2 GPU-s per 405B proof under RFC-0012's proximity generator, 0.41 if #394 excludes the blinding rows. That is a cost-model issue against RFC-0020 and 08, [#568](https://github.com/AbdelStark/tessera/issues/568), decided with #394. Its options:
  - regenerate on the GPUs, and amend 08's weight-pass budget;
  - keep the masks and blinding values in a secret store of 654 GB per epoch beside the codewords, read in full by every proof;
  - exclude the blinding rows from the proximity combination (#394).

## References

- [RFC-0004](RFC-0004-hashing-and-transcript.md), [RFC-0011](RFC-0011-advice-commitment.md) 'Encoding', [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0014](RFC-0014-interaction-and-soundness.md) 'Soundness budget', [RFC-0019](RFC-0019-verifier.md) 'Check order', [RFC-0031](RFC-0031-commitment-id.md), [RFC-0040](RFC-0040-weight-commitment-artifacts.md), [RFC-0064](RFC-0064-weight-linking-proof.md), [RFC-0065](RFC-0065-masked-channel-params.md), [RFC-0067](RFC-0067-verifier-ledger-encoding.md).
- [`docs/zk/simulator.md`](../zk/simulator.md) §2 (the PRG hybrid), §3 (Lemmas 3 and 5), §5 (R6) and §6.
- O'Connor, Aumasson, Neves, Wilcox-O'Hearn, 'BLAKE3: one function, fast everywhere', 2020, §2.6 (extendable output).
