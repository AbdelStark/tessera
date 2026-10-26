# RFC-0040: Weight Commitment Artifacts

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-18
- Target milestone: v0.3
- Supersedes, on acceptance:
  - [RFC-0012](RFC-0012-weight-commitment.md#layout) 'Layout': the message width of a matrix whose input is wider than `k_W`, the position of every entry, the committed rows, an empty `ChannelParams`, the leaves and the first epoch;
  - [RFC-0012](RFC-0012-weight-commitment.md#budget-state-machine) 'Budget state machine': the fields of `WeightCommitmentState` and `TypeBudget`;
  - [RFC-0026](RFC-0026-claim-list-layout.md#schema) 'Schema': the weights' message width;
  - [RFC-0031](RFC-0031-commitment-id.md#the-descriptor) 'The descriptor': a weight matrix's `rows`;
  - [RFC-0019](RFC-0019-verifier.md#inputs) 'Inputs' and 'Check order' step 4: the verifier reads the matrix roots from `WeightCommitmentPublic`;
  - [RFC-0039](RFC-0039-statement-fields.md#the-checks) 'The checks': the weight layout's check, at RFC-0019 step 4;
  - [02 §2](../spec/02-public-api.md#2-command-line-interface): `commit-weights` writes both artifacts, and `tessera-verify` reads the public one;
  - [03 §5](../spec/03-data-model.md#5-artifact-catalog): both artifacts are defined here.
- Amended by: [RFC-0064](RFC-0064-weight-linking-proof.md) ('`WeightCommitmentState`': each epoch's `masks_seed` from the model's refresh key `K`), [RFC-0065](RFC-0065-masked-channel-params.md) ('The matrices' and 'Encoding, leaves and roots': `ChannelParams`' blinding rows `B_CP`, `rows = m_T + B_T`, and the leaves over the blinding rows), draft [RFC-0073](RFC-0073-weight-mask-expansion.md) ('Encoding, leaves and roots' and '`WeightCommitmentState`': the expansion of `masks_seed` into a `tessera-protocol/2` commitment's masks and blinding rows, `K`'s artifact and the model entry; 'Command line': `commit-weights` writes a model entry in `tessera-protocol/2`; in effect when it is accepted)
- Origin: [#102](https://github.com/AbdelStark/tessera/issues/102)

## Summary

RFC-0012 commits the weights of `D` in ten matrices under one root `R_W`. The artifact catalog names two artifacts for that commitment: `WeightCommitmentPublic` (`0x0301`, `.twr`) and `WeightCommitmentState` (`0x0302`, `.twc`). Neither has a field layout in any document. RFC-0012's layout also leaves a few cases open, and no document says how the verifier obtains the ten matrix roots that its column checks need. This RFC defines both artifacts, fixes the open cases of the layout, and gives the verifier the public artifact as an input. It changes nothing in the commitment's soundness or in the protocol's messages.

## Motivation

- **Two undefined tier-A schemas** ([02 §1](../spec/02-public-api.md#1-stability-tiers)).
  - RFC-0012 sketches `WeightCommitmentState` with a field TCE cannot encode, `BitSet`. It holds the public part only as the comment "R_W, epoch, layout", and two of its budget fields repeat `WeightLayoutParams`.
  - `WeightCommitmentPublic` has no sketch at all.
- **The verifier has no matrix roots.** RFC-0019 step 11 checks every opened weight column against its matrix's root, and step 4 checks `R_W`. But `Round1` carries the statement and the advice commitment only ([RFC-0036](RFC-0036-protocol-messages.md#round1)), `VerifyInputs` has no weight commitment, and `tessera-verify` takes `--weights-root ROOT` alone. The ten roots behind `R_W` reach the verifier nowhere.
- **A message narrower than its input.** The width rule `k_T = min(2^{log2_k}, out_pad · in_pad)` assumes `in_pad ≤ k_T`, since a row holds `2^{a_T}` whole input rows. `test-small` has `k_W = 2^8`. The conformance models' embedding is 512 wide, and so is `tiny-4-servable`'s down projection, so `a_T` would be negative.
- **Open cases.**
  - The committed rows are "padded output rows … skipped", with no rule for padded layers.
  - `ChannelParams`' width `C` is undefined for a model with no `ChannelParams`.
  - The first epoch's number is not stated.
  - RFC-0004 gives a matrix's leaf as its chain's last digest, RFC-0011 merges the advice's segment digests, and nothing says which rule a weight matrix follows.

## Goals

- Field layouts for both artifacts, each value with exactly one encoding (RFC-0003).
- One `R_W` for given weights, layout and epoch, which the prover and the verifier compute independently.
- Every weight and channel parameter at a position computable from public data.
- The matrix roots in the verifier's hands, bound by the root it trusts.

## Non-Goals

- Masks, blinding rows, the budget's transitions and refresh, which belong to v0.4 (#147, #158). `tessera-protocol/1` applies none of them (RFC-0012 'Migration / Rollout', [RFC-0036](RFC-0036-protocol-messages.md#round2response)). This RFC defines the state's fields so that v0.4 changes behaviour, not the schema.
- The codeword store (RFC-0012 'Column answers'), which v0.5 brings.

## Proposed Design

### The matrices

Each `WeightMatrix` ([RFC-0031](RFC-0031-commitment-id.md#the-descriptor)) `T` is a matrix over `Fp`. Its entries are given below for every position. The weights of 03 §2 enter as true integers in the centered representation ([RFC-0002](RFC-0002-field-arithmetic.md)), and every padding position is 0.

**The nine weight types.** A weight of `out × in` (03 §2: rows are output channels, and `Embed` is `d_model × vocab`, transposed) has:

- **Padded sizes.** `out_pad`, `in_pad` and, for the seven per-layer types, `L_pad` are the next powers of two of `out`, `in` and `n_layers`. `Embed` and `LmHead` have one layer, `L_pad = 1`.
- **Message width.** `k_T = max(in_pad, min(k_W, out_pad · in_pad))`, with `k_W = 2^{layout.log2_k}`, and `a_T = log2(k_T / in_pad)`. The `max` makes a row hold at least one input row. It changes no width when `k_W ≥ in_pad`, as at `live-64` and `ni-100` for every Llama-3.1 shape: v1 bounds `vocab` by `2^{18}`, 405B's widest other input pads to `2^{16}`, and `k_W` is `2^{20}`.
- **Position.** Entry `W_ℓ[o][i]` lies at row `r = ℓ · 2^{h_T} + ⌊o / 2^{a_T}⌋` and column `(o mod 2^{a_T}) · in_pad + i`, where `h_T = log2 out_pad − a_T`. The row's bits are `(ℓ, o_hi)` and the column's `(o_lo, i)`, as RFC-0012 lays them out.
- **Committed rows.** A row is structurally zero when `ℓ ≥ n_layers`, or when `⌊o / 2^{a_T}⌋ · 2^{a_T} ≥ out`, which makes every output it holds padding. The others are committed, in increasing `r`: `m_T` rows. A committed row may still contain padding positions, and a weight that happens to be zero is committed like any other.

**`ChannelParams`.** The tensor `[16, L_pad, C]` of RFC-0012 'Layout' is flattened row-major: index `f = (t · L_pad + ℓ) · C + c`, with `t` the weight type's declaration index in 03 §2 (`Embed` 0 to `LmHead` 8).

- **Liveness.** A position is live when the weight `(t, ℓ)` has a `ChannelParams` and `c` is below its output width. Its entry is then the channel's class index, or its multiplier at an affine site. Every other entry is 0.
- **Width `C`.** `C` is the largest padded output width among the weights that have a `ChannelParams`. When no weight has one, `C` is 1, so that the tensor is defined and holds only zeros.
- **Message width.** `k_CP = min(k_W, 16 · L_pad · C)`. Entry `f` lies at row `⌊f / k_CP⌋` and column `f mod k_CP`.
- **Committed rows.** A row is committed when it holds a live position. Which weights have a `ChannelParams` is public, from the claim list (03 §2), so the committed rows are too. A model with no `ChannelParams` commits no row of this matrix.

### Encoding, leaves and roots

- **Codewords.** Each committed row's `k_T` values are the evaluations on the subgroup of order `k_T` of a polynomial of degree below `k_T`. Its codeword is that polynomial on the coset `7·L_T` of the subgroup of order `n_T = 2^{layout.log2_inv_rate} · k_T`: rate 1/4 at every parameter set (RFC-0012). In `tessera-protocol/1` the codeword carries no mask, and the matrix has no blinding rows. In `tessera-protocol/2` every committed row, data and blinding alike, carries a mask of `P_W` coefficients, and the `B_T` blinding rows follow the data rows; both are drawn from `masks_seed` as draft [RFC-0073](RFC-0073-weight-mask-expansion.md#the-masks-and-the-blinding-rows) says.
- **`commitment_id_T`.** It is `H("commitment-id", TCE(CommitmentDescriptor::WeightMatrix { … }))` ([RFC-0031](RFC-0031-commitment-id.md#the-descriptor)) with:
  - the claim list's `WeightLayoutParams` as `layout`;
  - `matrix` `T` and `log2_k` its own `log2 k_T`;
  - `row_block` the parameter set's `B`;
  - `rows = m_T`, plus `B_T` from the version that adds blinding rows: `B_W`, and `ChannelParams`' `B_CP` ([RFC-0065](RFC-0065-masked-channel-params.md#channelparams-blinding-rows));
  - `epoch`.
- **Leaves.** Each column is one chain over the matrix's committed rows: `leaf_j = d_j[⌈m_T / B⌉]` ([RFC-0004](RFC-0004-hashing-and-transcript.md#column-digests)), and `d_j[⌈(m_T + B_T)/B⌉]` over the blinding rows too, from the version that adds them ([RFC-0065](RFC-0065-masked-channel-params.md#channelparams-blinding-rows)). A weight matrix has no segments and no `column-merge`. A matrix of no committed rows has the leaves `d_j[0]`.
- **Roots.** `root_T` is the root of the Merkle tree over its `n_T` leaves ([RFC-0004](RFC-0004-hashing-and-transcript.md#merkle-trees)).

### `WeightCommitmentPublic`

```rust
pub struct WeightCommitmentPublic {          // schema 0x0301, version 1, public, .twr
    pub layout: WeightLayoutParams,          // the claim list's (RFC-0026)
    pub roots: [Digest; 10],                 // root_T, in WeightMatrix order
    pub epoch: u32,                          // 0 at the first commitment; refresh gives e + 1
}
```

`R_W = H("weights-root", TCE(WeightCommitmentPublic))`. This is RFC-0012's `H("weights-root", TCE(WeightLayoutParams) ‖ root_Embed ‖ … ‖ root_ChannelParams ‖ u32(epoch))` byte for byte, since TCE writes a struct's fields in order and a fixed array without a length. The artifact therefore holds exactly what `R_W` binds (INV-WC-3), and nothing that a reader could take from it unbound.

### `WeightCommitmentState`

```rust
pub struct WeightCommitmentState {           // schema 0x0302, version 1, secret, .twc
    pub public: WeightCommitmentPublic,
    pub masks_seed: [u8; 32],                // secret-class: decoded into a Secret that holds the 32 bytes on the heap
    pub budget: [TypeBudget; 10],            // in WeightMatrix order
}

pub struct TypeBudget {
    pub opened_columns: Vec<u32>,            // distinct columns revealed this epoch, increasing as integers
    pub next_blinding_row: u32,              // blinding rows below it are burned
}
```

- **The budget's limits** are `WeightLayoutParams`' `P_W` and each matrix's `B_T`: `B_W`, and `ChannelParams`' `B_CP` (RFC-0065). RFC-0012's `max_opened_columns` and `blinding_rows` are dropped, since they repeated them.
- **`opened_columns`** is a set: its indices are strictly increasing as integers, and a decoder rejects an unsorted or repeated index (`ERR-CODEC-NONCANONICAL`). The order is numeric, not the byte order of a map's keys (RFC-0003 'Composite types').
- **`masks_seed`.** Each epoch's is derived from the model's refresh key `K`, `H("weights-refresh", K ‖ u32(e))`. `K` is 32 bytes the committer draws from the operating system's CSPRNG for epoch 0 and keeps apart from every state file. A weight link is then the same bytes however often it is made ([RFC-0064](RFC-0064-weight-linking-proof.md#one-link-per-epoch)). The expansion of `masks_seed` into the masks and blinding rows is normative and versioned, with a conformance vector. Draft [RFC-0073](RFC-0073-weight-mask-expansion.md) specifies it, one stream per committed row's mask and per blinding row, version 1 in `tessera-protocol/2`. It also keeps `K` in an artifact of its own, and each model entry in one directory with a record of its epochs. A state is fresh only when written right after the append that makes its epoch the record's latest. Every other state is rebuilt and serves only that epoch's link (draft RFC-0073 'Fresh and rebuilt states'). A reader listed in `secret-access.toml` decodes it into a `Secret` that holds the 32 bytes on the heap, so that no move copies them, and encodes it back only to write the state (05 §3, OBS-2).
- **In `tessera-protocol/1`** the seed regenerates nothing, every `opened_columns` is empty, and every `next_blinding_row` is 0. The prover recomputes the commitment from `D` and the state, as RFC-0012 'Migration / Rollout' plans for v0.3: no masks, and codewords held in memory.
- **Writing.** The state is written with mode 0600, by atomic rename: a new file in the target's directory, written, flushed to disk, renamed over the target, and the directory flushed ([06 §7](../spec/06-security.md#7-secrets-handling)). The public artifact is written by atomic rename too, with the default mode.

### The verifier's weight inputs

- `VerifyInputs` gains `weights: WeightCommitmentPublic`, the commitment of the epoch the proof names ([RFC-0019](RFC-0019-verifier.md#inputs)).
- **At RFC-0019 step 4**, before the statement's fields are compared:
  - `H("weights-root", TCE(weights))` must be the root the verifier trusts for the model, and `weights.epoch` its epoch: the ledger's current root, or the `--weights-root` given to the command line. A difference is `ERR-USAGE`, since the verifier's own inputs disagree and nothing has been evaluated.
  - The statement's `weights_root` and `weights_epoch` are then compared with them ([RFC-0036](RFC-0036-protocol-messages.md#round1)).
  - After them, and before `layout_digest`, `weights.layout` must equal the claim list's `layout.weights`, else `Reject(StatementMismatch { field: Layout })` ([RFC-0039](RFC-0039-statement-fields.md#the-checks)). The trusted root binds the commitment's layout, so a claim list with another layout describes another commitment.
- **At steps 9 to 11** the verifier reads `root_T` from `weights.roots`, and derives each matrix's `k_T`, `m_T` and committed rows from the architecture and the layout as above.

### Command line

- `tessera commit-weights --model MODEL.d --plan plan.tep --out WEIGHTS.twc` writes the state, and the public artifact beside it as `WEIGHTS.twr` (the same name with the extension `.twr`). It prints `R_W`. In `tessera-protocol/2`, under draft [RFC-0073](RFC-0073-weight-mask-expansion.md#the-model-entry), it writes a model entry instead: `--entry DIR` and `--refresh-key KEY.twk` in place of `--out`, with every epoch's files under fixed names.
- `tessera-verify verify` and `tessera-verify live` take `--weights WEIGHTS.twr` beside `--weights-root ROOT`.

## Alternatives Considered

1. **The roots in `Round1`.** Ten digests, 320 bytes per proof, would spare the verifier an input. But the roots are the epoch's, not the proof's, and `Round1`'s layout is fixed (RFC-0036). The public artifact exists in the catalog for this purpose, and parties exchange it once per epoch.
2. **`R_W` in the public artifact.** It would be a field that a reader must check against the others, and a second encoding of one fact. Hashing the artifact gives it.
3. **A bitmap for `opened_columns`.** `n_T / 8` bytes per matrix, 512 KiB at 405B, against at most `4·P_W`, 256 KiB, for the set. It is also not canonical without a rule for its trailing bits.
4. **A larger `k_W` for `test-small`.** `log2_k = 9` would fit the conformance models. But it is a new parameter set version, and every claim-list vector would change, for a case the `max` covers in one word.
5. **Splitting a wide input across rows.** Rows of `(ℓ, o, i_hi)` would keep `k_T ≤ k_W`, but the combined row would then fold input bits, which the matmul sumcheck does not use (RFC-0012 'Alternatives Considered' 2).

## Drawbacks

- The verifier needs one more input file per epoch.
- `test-small`'s wide matrices have rows wider than its `k_W`. The soundness of its 8 queries per matrix does not depend on the width, only on the rate.

## Migration / Rollout

v0.3, for #102:

1. this RFC and its amendments;
2. `tessera-pcs` takes the inverse rate as a parameter of its row encoding (rate 1/2 for advice, 1/4 for weights), and encodes a weight row without a mask in `tessera-protocol/1`;
3. the weight layout, the roots, `R_W` and both artifacts, with an independent oracle in `spec/vectors/weights/` and golden vectors for `0x0301` and `0x0302` (03 §7);
4. the verifier's input and checks with #109 and #128, and the command line with #111 and #131.

`WeightLayoutParams` may move from `tessera-ir` to `tessera-pcs`, which `tessera-ir` may depend on ([01 §3](../spec/01-architecture.md#3-dependency-rules)), so that the commitment crate names the layout it encodes. Its bytes do not change.

## Testing Strategy

- **Positions.** For random small `D`, every weight and channel parameter is read back from its matrix at its documented position, padding positions are 0, and exactly the structurally zero rows are skipped (RFC-0012 'Testing Strategy').
- **Widths.** `k_T` and `a_T` for every conformance model at `test-small`, `tiny-4-servable`'s 512-wide down projection among them, and for 405B's shapes at `live-64`: `a = 6`, 4 and 3.
- **Roots.** An independent oracle in Python's integers computes every `root_T` and `R_W` of each conformance model at `test-small`. `tessera-pcs` reproduces them from the weights. The verifier, which never sees weights, recomputes `R_W` from each vector's public artifact.
- **Binding** (INV-WC-3). Changing one weight, one byte of the layout, the epoch, or one root changes `R_W`.
- **Artifacts.**
  - Golden vectors cover both schemas (03 §7). The prover's decoder reads both, and the verifier's reads `0x0301`, since it never reads the secret state. An unsorted or repeated `opened_columns` is rejected.
  - The state file is written with mode 0600, and an interrupted write leaves the previous file whole.
- **Verifier.**
  - A `.twr` that does not hash to the trusted root is `ERR-USAGE`. The trusted root vouches for `D`'s domains ([RFC-0048](RFC-0048-weight-trust.md#the-assumption)).
  - A claim list whose weight layout differs is `Reject(StatementMismatch)` (#109, #128).

## Open Questions

None.

## References

- [RFC-0003](RFC-0003-canonical-encoding.md), [RFC-0004](RFC-0004-hashing-and-transcript.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0019](RFC-0019-verifier.md), [RFC-0026](RFC-0026-claim-list-layout.md), [RFC-0031](RFC-0031-commitment-id.md), [RFC-0036](RFC-0036-protocol-messages.md).
