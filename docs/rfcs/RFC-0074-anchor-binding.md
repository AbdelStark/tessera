# RFC-0074: The Anchor in a Batch's Proof: One Graph and the Token-Byte Binding

- Status: Draft. The documents it amends carry its amendments, each marked as a draft's, in the change that proposes it. They take effect, and lose the mark, when it is accepted.
- Authors: @AbdelStark
- Created: 2026-10-26
- Target milestone: v0.4
- Amends:
  - [RFC-0015](RFC-0015-transcript-anchor.md#circuits) 'Circuits' steps 1 to 5: the circuits a batch holds, their inputs, the AES-128-CTR graph to the node, the plaintext committed per session and range-checked, and the binding's relations; 'Verifier checks' 2 and 4 and 'Prover pre-check' step 3: the anchor's size bound, and the tail zeros as public inputs; 'Testing Strategy': which reordering is rejected;
  - [RFC-0059](RFC-0059-frame-headers-and-nonces.md#the-circuits) 'The plaintext': at most two frames of no token per request; 'The circuits': the tail zeros of a frame's last block are public too;
  - [RFC-0063](RFC-0063-request-boundaries.md#the-requests-end) 'Why a frame of its own': empty frames within that limit;
  - [RFC-0062](RFC-0062-period-log-sessions.md#the-session-rules) 'Who is named': the size bound among a request's frame checks;
  - [RFC-0043](RFC-0043-rejection-payloads.md#anchor) 'Anchor': `Frames` covers a request past the size bound;
  - [RFC-0045](RFC-0045-leaf-layout.md#the-verifiers-derivation) 'The verifier's derivation': the anchor's blocks come from the logs;
  - [RFC-0072](RFC-0072-anchor-relations.md#padded-positions) 'Real positions' (the batch refusal 'Refused until #146'), 'Graphs without layers' ('In a model's graph'), 'Non-Goals' (token groups) and 'Migration' item 3;
  - [RFC-0054](RFC-0054-token-groups.md#groups) 'Groups', the token axis: `GatherPublic` over the token axis, and the anchor's operations;
  - [RFC-0011](RFC-0011-advice-commitment.md#canonical-advice-layout) 'Canonical advice layout' (its inputs, rule 4) and INV-PCS-1, with its restatements in [RFC-0031](RFC-0031-commitment-id.md#the-descriptor) 'The descriptor', [RFC-0041](RFC-0041-execution-plan.md#motivation) 'Motivation' and 'Goals', [RFC-0046](RFC-0046-advice-segments.md#motivation) and [RFC-0071](RFC-0071-request-slots.md#the-plan-and-the-layout) 'The plan and the layout', 'The maximum layout';
  - [RFC-0041](RFC-0041-execution-plan.md#production-order) 'Production order' and 'The plan': the anchor's tiles, a runtime expansion of the plan;
  - [RFC-0006](RFC-0006-compiler-ir.md#gadget-selection) 'Gadget selection': `B_max` bounds the model's tables, not the anchor's four;
  - [RFC-0026](RFC-0026-claim-list-layout.md#scoring-and-the-anchor) 'Scoring and the anchor' and its quote of step 7, [RFC-0007](RFC-0007-claim-list.md#schema) 'Schema' and 'Verifier validation' step 7, and [RFC-0039](RFC-0039-statement-fields.md#the-checks) 'The checks' step 7: the anchor joins no claim-list field;
  - [RFC-0066](RFC-0066-bits-and-sha256.md#the-sha-256-circuit) 'The message': one frame per circuit, its committed bytes a `Slice` of the ciphertext;
  - [RFC-0053](RFC-0053-greedy-runtime-boundary.md#public-request-and-sources) 'Public request and sources': the certificate ids count every node of the graph, the anchor's included;
  - [RFC-0019](RFC-0019-verifier.md#check-order) 'Check order' step 1 (the manifest's signature before its log), step 2 (no `tessera-protocol/2` parameter set yet) and step 4 (the interim cap, the resource decision, the size bound among the frames' checks, the composed layout's maximum);
  - [02 §3.2](../spec/02-public-api.md#32-verifier): `ResourceLimits.max_anchor_blocks`;
  - [07 §6](../spec/07-testing-strategy.md#6-end-to-end-tests): `e2e_anchor_mismatch`'s two cases and their verdicts;
  - [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) 'Committed tensors', 'Hiding', R10's gap list, item 10: the anchor in a batch's graph, and the token binding's selection claim on the tokens.
- Origin: [#146](https://github.com/AbdelStark/tessera/issues/146)

## Summary

RFC-0072 proves AES-128-CTR and SHA-256 as graphs of their own. #146 puts them into a batch's proof, after the model's nodes, and binds them to the tokens the forward pass reads. Nothing new is proven: every relation is RFC-0066's, RFC-0072's or RFC-0010's. This RFC fixes what the graph holds, in which order, and from which public data, so that the prover and an independent verifier derive one graph:

- **One graph.** The model's lowering, then, per session of the batch, its key commitment, its AES-128-CTR, its plaintext's binding and each frame's SHA-256. Model node, gadget and table ids do not move. AES-128-CTR's graph is fixed to the node, as RFC-0066 fixes SHA-256's.
- **The token-byte binding.** Each session's plaintext is one committed tensor, the bytes AES encrypts. `BitDecompose { 8, 0 }` proves each byte. Two `AssertEqual`s bind it: its public bytes to their values from the log, and its token bytes, recomposed as `Σ_b 2^{8b}·byte_b` by one combined selection, to the token axis's rows, which `OneHot` reads.
- **Inputs from the log.** The graph's shapes and public inputs are a function of the statement's batch and the two verified logs of its window. No claim-list or statement field is added.
- **Size.** A request's frames hold at most `2·n_r + 4` SHA-256 blocks, a public bound past which the request is refused. Until OQ-0074-1 and OQ-0072-3 land, an interim cap of 4,096 blocks at that bound per batch keeps every proof within the decode limit.
- **Batches, token groups and the layout.** The anchor reads the token axis only through a gather of real rows, which lifts RFC-0072's batch refusal for it. Its operations have no token axis, so token groups apply. A tensor non-zero at padding has every row committed.

For a tiny model, a request of 2 prompt and 2 output tokens in two frames takes 14 stages and 17,555 instances. The model alone takes 4 stages and 221 instances. The binding itself, the two `AssertEqual`s and their gathers, is 2 zero-checks, 3 selections and 3 pool claims per session; the plaintext's range check is counted with AES.

## Motivation

- **RFC-0072 'Migration' item 3** leaves the anchor's wiring to #146: the circuits in a batch's graph, their inputs from the log and the session, and the statement's sizes.
- **RFC-0015 step 1 is not a relation yet.** "Each token's four little-endian bytes are committed advice, range-checked to 8 bits, with `token = Σ_b byte_b 2^{8b}`" names no tensor, no operation and no link to `OneHot`'s input. The AES builder takes its plaintext as a committed input that nothing range-checks (RFC-0006 'Committed graph inputs': AES-128-CTR alone is a fragment).
- **Several rules depend on #146:**
  - RFC-0072 refuses a `ConstAdd` or `AssertEqual` downstream of the tokens in a batch of several requests;
  - RFC-0054's token axis refuses `Bitwise` and the bit operations, so a graph that holds the circuits runs its leaf stage ungrouped;
  - RFC-0026 took `AnchorSpec` out of the claim list until the circuits define what it states.
- **The code.** The CPU prover proves the circuits alone, with an in-test verifier and no commitment (#526). No advice layout exists for their tensors, and #526's review found its precondition: a row of padded positions that are not 0 must not be skipped.

## Goals

- One graph, schedule, leaf layout, advice layout and opening order per batch, derived by both workspaces from public data alone.
- INV-ANC-1 as a consequence of named relations, each with a test that breaks exactly it.
- The model's part unchanged: its node, gadget, tensor and table ids, its sites' order and its schedule's instances.
- No new kind of relation, message, label or opening group.

## Non-Goals

- **The sampler key** (RFC-0015 step 6). A greedy batch has no pin. A batch that decodes with temperature adds its circuit with #152; until then the prover refuses it (`ERR-USAGE`), and the verifier returns `Inconclusive(ERR-USAGE)` at step 4, where it binds the predictor, as for every predictor it does not support (RFC-0019 'Current protocol/1 composition').
- **`tessera-protocol/2`.** The anchor's tensors need #145's masks, and R10's check does not cover them yet (#488). Until both do, no `tessera-protocol/2` proof exists, since every proof is anchored: the prover refuses one (`ERR-USAGE`), and the verifier allows no `tessera-protocol/2` parameter set, `Reject(Params)` at RFC-0019 step 2.
- **Fewer opening groups and rows** (OQ-0072-3), and grouped messages (OQ-0074-1).
- **The verifier's mirror** (#150, #159).

## Proposed Design

### The anchor's inputs

The verifier derives them at RFC-0019 step 4 from `statement.batch` and the verified logs of the period and the one before it ([RFC-0063](RFC-0063-request-boundaries.md#the-previous-periods-log)), once RFC-0015's checks 1 and 2 pass. The prover derives them from the same logs after its pre-check passes.

- **A request's key.** Its key commitment is that of its session's open or carry in the periods of its window that hold its frames, one across them (RFC-0063 rule 4).
- **Sessions.** An anchor session is a pair of a `session_id` and a key commitment, one per distinct pair among the batch's requests, in order of first appearance in `statement.batch`. A `session_id` closed in the previous period and opened again under another key is two anchor sessions: each request's frames are encrypted under the key its own periods log (INV-ANC-2).
- **A session's frames.** For each of its requests, in batch order: the request's `In` frames by `frame_index`, then its `Out` frames by `frame_index`, read from its window. This is the order of the request's tokens: its prompt, then its outputs.
- **A frame's data.** From its entry:
  - its header, 41 bytes ([RFC-0059](RFC-0059-frame-headers-and-nonces.md#the-header));
  - its length `ℓ = 22 + 4n`, with `n` its tokens;
  - its counter `frame_seq`;
  - its digest.
- **Rows.** The `k`-th token of request `r`'s frames, in that order, is position `k` of `r`, row `r·S + k` of the token axis ([RFC-0071](RFC-0071-request-slots.md#the-slots)); with one request, row `k`. Check 2 makes the `In` frames' tokens `prompt_len_r` and the `Out` frames' `output_len_r` ([RFC-0062](RFC-0062-period-log-sessions.md#the-periods-log) 'Its lengths', RFC-0063), so these are each real row of the request, once, prompt rows from the `In` frames.
- **Blocks.** A session's frames take consecutive blocks of 16 bytes, frame `f` its `⌈ℓ_f/16⌉` blocks from block `o_f`. `B_s` is their total. Block `o_f + i` holds frame `f`'s plaintext bytes `16i` to `16i + 15`, and 0 past `ℓ_f`, and its counter is `session_id[0..8] ‖ u32_be(frame_seq_f) ‖ u32_be(i)` ([RFC-0059](RFC-0059-frame-headers-and-nonces.md#the-keystream)).

Every frame takes two blocks at least, since `ℓ ≥ 22`. Every session holds a token, since `prompt_len ≥ 1`.

- **The size bound.** A request's frames hold at most `2·n_r + 4` SHA-256 blocks:

  ```text
  Σ_f ⌈(ℓ_f + 50)/64⌉ ≤ 2·n_r + 4,   n_r = prompt_len_r + output_len_r
  ```

  - **Why these numbers.** Frame `f` hashes `41 + ℓ_f` bytes, which FIPS 180-4 pads to `⌈(ℓ_f + 50)/64⌉` blocks. A frame of `n ≥ 1` tokens takes `⌈(72 + 4n)/64⌉ ≤ 2n` blocks, so one token per frame, the most a token can cost, takes `2·n_r`. The 4 admit two frames of no token.
  - **The stack's duty.** A request has at most two frames of no token ([RFC-0059](RFC-0059-frame-headers-and-nonces.md#the-plaintext) 'The plaintext', as amended); the reference cipher refuses a third. The bound refuses no log that keeps it.
  - **Its rule.** It is a rule of RFC-0015's check 2, checked after the lengths, from the entries alone and before anything of the anchor is derived. A request past it is `Reject(Anchor { request_id: Some(r), kind: Frames })`; the prover's pre-check refuses it first, `ERR-WITNESS-ANCHOR`.
  - **Why it is needed.** Without it, frames of no token, which change no token count and no length bucket, would grow a request's anchor at the serving stack's choice, past any verifier's resources.
  - **What follows.** A frame's AES blocks, `⌈ℓ_f/16⌉ ≤ 4·⌈(ℓ_f + 50)/64⌉`, are then at most `8·n_r + 16` per request. Each anchor session adds one key-commitment block and one key schedule, and there are at most `R` sessions. So a batch's anchor is at most

    ```text
    A_b = Σ_r (2·n_r + 5)  SHA-256 blocks,   Σ_r (8·n_r + 16) AES blocks,   R key schedules
    ```

    a function of the statement's lengths alone.
- **The interim cap.** Until OQ-0074-1 and OQ-0072-3 are resolved, a batch's `A_b` is at most `A_max = 4,096`.
  - **Its check.** The verifier checks it with the batch's other checks at RFC-0019 step 4, from the statement alone and before it reads any entry: past it is `Reject(StatementMismatch { field: Batch })`.
  - **The packer.** It splits a batch past the cap as RFC-0071 splits one past the maximum layout. A request with `2·n_r + 5 > 4,096`, more than 2,045 tokens, is unprovable meanwhile: it is left out of its batch and named as unproven (RFC-0071 'An unprovable request').
  - **Why this number.** At the cap an anchor costs at most about 1.5 M advice rows, 2.3 GB of opened columns at `τ_A = 193`, 0.36 GB of evaluations and 1.6 GB of extra opening groups ('Drawbacks'). That keeps a 405B proof under RFC-0003's 8 GiB decode limit for either production parameter set, whatever the framing.

### The composed graph

The batch's graph is the model's lowering ([RFC-0071](RFC-0071-request-slots.md#the-lowering)), then the anchor's nodes, outside the layers ([RFC-0072](RFC-0072-anchor-relations.md#graphs-without-layers) 'In a model's graph').

For each session `s`, in session order:

1. **The key commitment.** RFC-0066's circuit over `"tessera-anchor-key/v1" ‖ k_s ‖ salt_k`, one message of one block. Its committed input `[32, 1]` holds `k_s` then `salt_k`, at offsets 21 to 52. Its digest is bound to the session's `key_commitment`.
2. **AES-128-CTR** ('The AES-128-CTR circuit') over `B_s` blocks:
   - its key is `Reshape([16])(Slice { axis: 0, start: 0, len: 16 })` of the key commitment's committed input;
   - its round constants and counter blocks are public;
   - its plaintext `p_s`, `[16, B_s]`, is committed advice, declared `u8` in `[0, 255]`, then `BitDecompose { 8, 0 }(p_s)`, before the circuit's nodes.
3. **The plaintext's binding** (below).
4. **Each frame's SHA-256**, in frame order. RFC-0066's circuit over `header_f ‖ ciphertext_f`, one message. Its committed bytes are the frame's ciphertext, `Reshape([ℓ_f, 1])(Slice { axis: 0, start: 16·o_f, len: ℓ_f })` of `flat(c_s)`, with `c_s` AES's ciphertext. They are at offsets 41 to `40 + ℓ_f`. Its digest is bound to the frame entry's.

Here `flat(x) = Reshape([16·B_s])(Transpose { perm: [1, 0] }(x))`, whose element `16i + j` is byte `j` of block `i`. A `Reshape` of `[B_s, 16]` maps coordinates, since 16 is a power of two (RFC-0037 'Head layouts'). A `Reshape` of `[16, B_s]` would not when `B_s` is not one.

- **Order.** Per session:
  1. the key commitment's inputs in RFC-0066's order ('The SHA-256 circuit', 'The graph'), its nodes, its digest and its binding;
  2. the key's `Slice` and `Reshape`, AES's round constants, counters and plaintext, the plaintext's `BitDecompose`, then AES's nodes;
  3. the binding's nodes ('The token-byte binding');
  4. `flat(c_s)`, then per frame its `Slice` and `Reshape`, its SHA-256 inputs in RFC-0066's order but the committed bytes, which are that `Reshape`, its nodes, its digest and its binding.

  Each constant is declared just before its node, as RFC-0066's are. Node, tensor and constant ids follow this order, after the lowering's.
- **Stage 1.** The model's zero-checks come first, the native relations' among them, then the anchor's bit zero-checks in node order (RFC-0072 'The bit zero-checks'). The anchor declares no committed output of a linear node, so it adds no equality zero-check.
- **Gadgets.** They are numbered in node order (RFC-0006 'Gadget selection'), so the model's ids do not move. The exact-greedy certificate's id is the whole graph's node count plus its ordinal ([RFC-0053](RFC-0053-greedy-runtime-boundary.md#public-request-and-sources)), which keeps it above every gadget's.
- **Tables.** The model's keep their ids: the claim list's, then the exact-greedy certificate's `Range` table when it adds one (RFC-0053). The anchor's four follow, numbered in order of first use: `AesSbox`, `Bitwise8 { Xor }`, `Gf256Mul { 2 }`, `Gf256Mul { 3 }`. They are not the claim list's ('The claim list and the statement'). The verifier generates them, as it does the certificate's table. `GadgetPolicy.max_log2_rows` bounds the model's gadget selection (RFC-0006); these four are fixed, of at most `2^{16}` rows, and outside it.
- **Ranges.** Range analysis runs on the whole graph, and every committed advice input must be proven (RFC-0006 'Committed graph inputs'):
  - the tokens by `OneHot`;
  - each `[32, 1]` committed input by the key commitment's `BitDecompose { 8, 0 }`;
  - each `p_s` by its own.

  The profile's constraints S1 to S3 read the model's nodes alone (RFC-0006 'Llama lowering': no anchor node is part of `D`).

### The AES-128-CTR circuit

RFC-0015 step 2 describes the lowering. This section fixes the graph to the node, as RFC-0066 'The SHA-256 circuit' fixes SHA-256's, so that a verifier builds it from this text alone. `tessera_compiler::anchor::aes128_ctr` builds exactly this graph.

- **Layout.** A state is `[16, B]`: row `j` is byte `j` of each block in FIPS-197's input order, byte `r + 4c` the state's row `r` and column `c`, and column `i` is block `i`. A round key is `[16]` in the same order.
- **Inputs, in order:** the key `[16]`; the round constants, public `[10, 4]`, row `r − 1` FIPS-197's word `[rc_r, 0, 0, 0]` with `rc_r` = `01, 02, 04, 08, 10, 20, 40, 80, 1b, 36`; the counter blocks, public `[16, B]`; the plaintext `[16, B]`. In a batch's graph the key is the `Reshape` of 'The composed graph' step 2, and the other three follow it.
- **Operations.** Every tensor is declared `u8` in `[0, 255]`, and range analysis settles it.
  - `S(x)`, `M2(x)` and `M3(x)` are `TableMap` into `AesSbox`, `Gf256Mul { 2 }` and `Gf256Mul { 3 }`. `x ⊕ y` is `Bitwise` into `Bitwise8 { Xor }`, `x` its first operand. Each output is a lookup output.
  - `G_I(x)` is `GatherPublic` of `x`'s outermost axis at the index list `I`, virtual. Its constant is declared just before it and never shared.
  - `E(x)` is `Expand { axis: 1, len: B }`, virtual.
- **The key schedule.** `rk_0` is the key. For `r` from 1 to 10, with `D = rk_{r−1}`, in this order:
  1. `rot = G_[13, 14, 15, 12](D)`, `[4]`, then `sub = S(rot)`;
  2. `row = G_[r − 1](rcon)`, `[1, 4]`, then `rc = Reshape(row)`, `[4]`, then `t = sub ⊕ rc`;
  3. `t16 = G_I(t)` and then `d_0 = G_I(D)`, both with `I[j] = j mod 4` for `j < 16`, then `acc = t16 ⊕ d_0`;
  4. for `c` from 1 to 3: `d_c = G_I(D)` with `I[j] = 4c + (j mod 4)`, then the constant `[⌊j/4⌋ ≥ c]` for `j < 16` and the `ConstMul` of `d_c` by it, virtual, then `acc = acc ⊕` that product;
  5. `rk_r = acc`.

  Column `c` of `rk_r` is `t ⊕ D_0 ⊕ … ⊕ D_c`, FIPS-197's word `w[4r + c]`. A round is 17 nodes and 72 lookups.
- **The keystream** of the counter blocks `X`:
  1. `E(rk_0)`, then `s = X ⊕` it;
  2. for `r` from 1 to 10: `sub = S(s)`, then `sh = G_SR(sub)`, ShiftRows, with `SR[j] = (j mod 4) + 4·((⌊j/4⌋ + j mod 4) mod 4)`;
     - for `r < 10`, MixColumns: `a = M2(sh)`, `b = M3(sh)`, `b_1 = G_{D_1}(b)`, `s_2 = G_{D_2}(sh)`, `s_3 = G_{D_3}(sh)`, then `m = ((a ⊕ b_1) ⊕ s_2) ⊕ s_3`, with `D_k[j] = ((j mod 4 + k) mod 4) + 4·⌊j/4⌋`;
     - for `r = 10`, `m = sh`;
     - then `E(rk_r)`, and `s = m ⊕` it.
- **The ciphertext** is `p ⊕ s`, the plaintext first.
- **Size.** 170 nodes for the key schedule and 115 for the keystream and the ciphertext: 285 whatever `B`. 720 lookups for the key and 1,072 per block.

### The token-byte binding

For session `s`, with `T_s` tokens:

- **The flat plaintext.** `q = flat(p_s)`, `[16·B_s]`.
- **Public bytes.** `P_s` lists the offsets of every frame's first 22 bytes and of its last block's bytes past `ℓ_f`, frame by frame, ascending. `v_s`, public `u8 [|P_s|]`, holds the plaintext prefix at the first 22 ([RFC-0059](RFC-0059-frame-headers-and-nonces.md#the-plaintext)), `u64_le(request_id) ‖ u16_le(direction) ‖ u32_le(frame_index) ‖ u64_le(n)` from the entry, and 0 past `ℓ_f`. The check is `AssertEqual(GatherPublic { P_s }(q), v_s)`.
- **Token bytes.** Token `k` of frame `f` is at offset `16·o_f + 22 + 4k + b` for its byte `b`. `D_b` lists those offsets for the session's tokens in order, and `R_s` their rows. The check is `AssertEqual(u, w)`, with:
  - `u = GatherPublic { R_s }(t)`, `t` the token axis's tokens, `OneHot`'s input;
  - `w = ((g_0 + 2^8·g_1) + 2^{16}·g_2) + 2^{24}·g_3`, with `g_b = GatherPublic { D_b }(q)`, built as `Add` and scalar `ConstMul`.
- **Node order.** `flat(p_s)`, the public gather, `v_s`, the public `AssertEqual`; then `u`; then `g_0`, and for `b` from 1 to 3 `g_b`, its `ConstMul` and the `Add`; then the token `AssertEqual`.

**What is committed.**

| Tensor | Kind | Proven by |
|---|---|---|
| `t`, the tokens | committed advice | `OneHot` (RFC-0009 Lemma 8): `[0, vocab)`, 0 on padded rows |
| `p_s`, the plaintext bytes, the limbs among them | committed advice | `BitDecompose { 8, 0 }` (RFC-0066 Lemma 1's converse), its padded blocks pinned to 0 by RFC-0066's input padding check |
| `p_s`'s bits, `[16, B_s, 8]` | committed advice | booleanity and recomposition |
| `q`, `g_b`, `w`, `u`, the gathered public bytes | virtual | selections and RFC-0010's linear substitution |
| `v_s`, the counters, the round constants | public | the verifier, from the log and RFC-0059 |

**The relations.**
- The four `g_b` read one tensor along one axis at the token check's point. So they are one combined selection whose kernel is `Σ_b 2^{8b}·S_b` (RFC-0072 'Combined selections'): one sumcheck of `⌈log2 16B_s⌉` rounds, one claim on `q`.
- `u` is a selection of `t`, which leaves one claim on `t` at a random point. That is RFC-0071's sum over the whole token axis with public coefficients, never a slice at a Boolean request coordinate.
- Both `AssertEqual`s are stage 1's zero-checks, their selector `ψ` over their own shapes.

**Every limb pinned** (SP-3). Given `t`, `p_s` has one passing value:
- each element lies in `[0, 256)`, and is 0 at padded blocks (`BitDecompose`);
- each public position holds its value (the public `AssertEqual`);
- at each token, `Σ_b 2^{8b}·p_b` and `t` lie in `[0, 2^{32})` and `[0, vocab)`, both below `P`, so the token `AssertEqual`'s equality in `Fp` is equality of integers. Base 256 has one representation.

Its bits are then unique by Lemma 1. AES's lookup outputs are unique for their keys, and the ciphertext's bits by Lemma 1 again.

**Where the limbs come from.** They are the bytes AES encrypts: RFC-0015 step 2 XORs the plaintext with the keystream, and step 3 hashes the ciphertext, the lookups' outputs. No committed copy of the ciphertext exists, and the verifier never holds one: the log carries only its digest.

### INV-ANC-1

Take a batch whose proof passes. For every request `r` and position `p < n_r`, `t[r·S + p]` is then the `p`-th token of `r`'s recorded frames, unless one of the following happens:
- one of the proof's checks errs, with the probability RFC-0014's calculator bounds;
- the prover finds a SHA-256 collision.

The chain:
1. **Coverage.** Every request lies in one anchor session, and `R_s` lists each of its real rows once, its prompt rows from its `In` frames and its output rows from its `Out` frames (RFC-0015 check 2's lengths, per direction; 'The anchor's inputs').
2. **The token check.** `t` at those rows equals the recomposed plaintext bytes, which are bytes ('The token-byte binding').
3. **The key.** `k_s` is the key commitment's committed input, hashed to the `key_commitment` logged in the request's own periods: the logged key, or a SHA-256 collision.
4. **The ciphertext.** It is the plaintext XOR AES-128-CTR's keystream under `k_s` and the public counters: every lookup is LogUp's, every byte a byte.
5. **The digest.** Each frame's ciphertext, after its public header, hashes to the logged digest: the recorded ciphertext, or a SHA-256 collision.
6. **The plaintext.** CTR mode is a bijection of the plaintext under a key and counters. So the plaintext is the recorded one, and its token bytes are the recorded tokens.

`OneHot`'s native token binding (RFC-0053) makes these tokens the ones the forward pass reads.

### Batches of several requests

RFC-0072 'Real positions' refuses a `ConstAdd` or `AssertEqual` downstream of the tokens or positions in a graph of more than one request. Its `ψ` would count the padded rows between slots as real. This RFC states the rule for the anchor:
- **The cut.** Downstream is reachability from the tokens and the positions through any node but a `GatherPublic` that reads one of them directly. Such a gather is admitted only when each of its indices is a real row, `τ mod S < n_{⌊τ/S⌋}`, and refused otherwise (`ERR-INTERNAL`).
- **Why it holds.** Each position inside its output's shape is then a real token, and its padded positions are 0, by the selection relation. So `ψ` over its shape selects exactly the real tokens, as RFC-0071's row factor would.
- **The decision.** It reads the graph and the batch's lengths alone.
- **What stays refused.** The token binding is the anchor's one read of the tokens. Every other path stays refused until a selector with RFC-0071's row factor is stated.
- **One request.** A graph of one request has no padded row inside its shape: RFC-0006 already refuses an index past `n`.

### The leaf stage and token groups

- **Sites.** The model's sites in node order, then the anchor's lookup sites in node order, then the logits' site, last, when the schedule keeps one: a generation request's defers it to RFC-0053's native relations ([RFC-0037](RFC-0037-stage-schedule.md#the-layer-point) 'The layer point'). The anchor's lookups are RFC-0072's: outside the layers, public reads sending no split value ([RFC-0045](RFC-0045-leaf-layout.md#the-split)).
- **The token axis** ([RFC-0054](RFC-0054-token-groups.md#groups) 'Groups'), as amended:
  - `GatherPublic` over the token axis gives an output with none: its rows are a frame's tokens, in the frame's order, not the batch's token index. Over another axis it keeps its input's;
  - `Bitwise`, `BitDecompose`, `BitFn` and `AssertEqual` are admitted when no input has a token axis, and their outputs have none;
  - `Argmax` and `InverseCdf` stay refused.

  Every anchor tensor then has none: its committed inputs have none, and the gather cuts the tokens' axis.
- **Groups.** An anchor lookup block has no token axis and lies outside the layers. It belongs to token subset 0, whole, and to layer subset 0 (RFC-0054 'No token axis'). Grouping changes no verdict. A batch that holds the anchor is grouped as any other: its token axes always derive, so no grouping within RFC-0054's bounds is refused because of the anchor.

### The advice layout and the openings

- **Its inputs.** The layout is a function of the claim list, the batch descriptor, and the anchor's inputs read from the logs the statement's `manifest_digest` names (INV-PCS-1, as amended). The statement digest still binds them, through the manifest, its `entries_root` and its chain to the previous period ([RFC-0031](RFC-0031-commitment-id.md#the-descriptor)).
- **Production order** ([RFC-0041](RFC-0041-execution-plan.md#production-order)):
  1. the model's tiles, but its tables';
  2. the anchor's committed graph inputs, in declaration order;
  3. the anchor's nodes' committed outputs, node by node in node order, each node's in output order: a `BitDecompose`'s bits, a lookup's output;
  4. the greedy certificate's maximum ([RFC-0053](RFC-0053-greedy-runtime-boundary.md#public-request-and-sources));
  5. the tables' multiplicities in table order, the anchor's last.

  Each anchor tensor smaller than a row takes a row of its own (RFC-0011 rule 2). Every anchor tile follows the last layer's, so the last segment holds it ([RFC-0046](RFC-0046-advice-segments.md#segments)).
- **Zero rows.** RFC-0011 rule 4 skips a row only when public data shows it zero. A tensor is non-zero at padding when a lookup node non-zero at padding produces it or lies upstream of it in the graph ([RFC-0072](RFC-0072-anchor-relations.md#padded-positions)), as `schedule::tensors_nonzero_at_padding` computes it. Every row of such a tensor is committed, whatever its shape. The prover commits there what its operation computes, such as a padded block's `AesSbox(0) = 0x63`, never a zero it assumed. The decision reads the graph, its tables and the shapes, never committed values.
- **Openings.** [RFC-0037](RFC-0037-stage-schedule.md#the-claim-pool) 'The claim pool' orders them, the anchor's claims in their places:
  - its pool claims among the pool's, in pool order;
  - its lookup outputs among the fixed non-native leaf claims, in node order, after the model's;
  - its tables' multiplicities after the model's, in table order.

### The claim list and the statement

- **No claim-list field.** RFC-0026 deferred `AnchorSpec` until the circuits define what a claim list states. They state nothing a model or deployment chooses:
  - the circuits are fixed by RFC-0015, RFC-0059, RFC-0066, RFC-0072 and this RFC;
  - their shapes and public inputs come from the batch's log entries;
  - their four tables are fixed;
  - the frame format is the log entry's schema version (`0x0401`, version 1), and a new format is a new schema version and protocol.

  Every statement names a manifest whose log holds each of its requests (INV-DM-7), so every proof is anchored, and no field says whether it is. RFC-0007 'Verifier validation' step 7 applies to scoring alone.
- **No statement field.** The anchor's sizes are the frames' lengths and counts, which the verified logs give. The statement's `prompt_len` and `output_len` must equal their token counts (check 2).
- **The verifier's order** (RFC-0019 step 4):
  1. the batch's checks, the interim cap and the resource decision among them, from the statement;
  2. the log and the session rules;
  3. per request, its own frames, the size bound after its lengths, then its sampler key, in RFC-0043's order;
  4. the anchor's part of the graph, then the composed layout against the maximum layout (`StatementMismatch { field: Batch }`) and `layout_digest` against it.
- **Its preconditions.** The anchor's part passes range analysis, RFC-0006 'Committed graph inputs' and every gadget precondition for any statement and logs that pass RFC-0015's checks 1 and 2: its ranges do not depend on them. A failure is the verifier's own, `Inconclusive(ERR-INTERNAL)` at step 4.
- **Its resources.** The verifier provisions the anchor from `A_b` ('The size bound'), never from the prover's frames.
  - **The cost of a block.** A block of `A_b` stands for at most 6,100 nodes, 370 committed tensors and 3,440 stage instances: a SHA-256 block's 6,032 nodes and 368 tensors with its share of its frame's other nodes, or a key commitment's 6,066 nodes and 370 tensors. A session adds at most 450 nodes and 130 tensors: AES's 285 nodes and 129 tensors, the key's two moves, the plaintext's `BitDecompose`, the binding and the flattened ciphertext.
  - **The decision.** `ResourceLimits.max_anchor_blocks` (02 §3.2), 4,096 by default, bounds it. A batch whose `A_b` exceeds it is `Inconclusive(ERR-RES-MEMORY)` at step 4, right after the interim cap and before any entry is read.
  - **Consequences.** Two verifiers with the same limits decide alike. With the default, no batch within the cap is refused.
  - **The proof's own fault.** A request past its size bound is `Reject(Anchor { kind: Frames })`.
- **Rollout.** An unanchored proof checked by a verifier that derives the anchor, or the converse, fails at step 4: its `layout_digest` is another layout's (`Reject(StatementMismatch { field: LayoutDigest })`). No field marks a proof anchored, since no verifier accepts both.

### Rejections

Each kind of mismatch has one verdict:
- **The log and the manifest** ([RFC-0043](RFC-0043-rejection-payloads.md#anchor)):
  - a log altered after its manifest was signed is `Inconclusive(ERR-CODEC-DIGEST)` at step 1 (RFC-0062);
  - a manifest under a key that is not pinned is `Reject(Anchor { request_id: None, kind: ManifestSignature })`, at step 1: its signature is checked before its log (RFC-0062 'The period's log', 'When'), so a manifest that fails both takes this verdict;
  - a request's frames not contiguous from 0, an inbound frame logged after an outbound one, or frames whose tokens are not the statement's lengths are `Reject(Anchor { request_id: Some(r), kind: Frames })` (RFC-0063), and so is a session that breaks RFC-0062's rule 2 or 3; rule 1, or two keys across the window, is `Reject(Anchor { request_id: Some(r), kind: Session })` (RFC-0062, RFC-0063 'The kinds'). Entries of one direction logged out of `frame_index` order break no rule: tokens are taken by `frame_index`.
- **The proof.** Tokens that are not the recorded frames' fail the circuits, below. RFC-0015's check 4 cannot fail on its own: the verifier supplies every public input.
- **The prover.** A stored transcript that does not re-seal to the log is `ERR-WITNESS-ANCHOR` before `Round1`.

In the proof, the verifier checks the leaf stage's lookups (RFC-0019 step 7) before the stages (step 8):

| Proof | First failure |
|---|---|
| A token on the axis other than its frame's bytes' value, every other lookup valid | the token `AssertEqual`: `Reject(Sumcheck { stage: 1, round: n_1 })` |
| A plaintext byte `(c ⊕ (k + 1)) + 256` with `k < 255` its keystream byte and `c` the recorded ciphertext byte: the `Bitwise8` key reads the row `(c ⊕ (k + 1), k + 1)`, whose output is `c`, so AES and the digest still pass, while the token recomposes to another value | `p_s`'s `BitDecompose`: `Reject(Sumcheck { stage: 1, round: n_1 })` |
| A recorded frame whose plaintext prefix disagrees with its header; or a witness whose tail byte is not 0 | the public `AssertEqual`: `Reject(Sumcheck { stage: 1, round: n_1 })` |
| A stored frame other than the one the signed log records, proven without the pre-check | the frame's digest `AssertEqual`: `Reject(Sumcheck { stage: 1, round: n_1 })` |
| A session key whose commitment is not the logged one | the key commitment's `AssertEqual`: `Reject(Sumcheck { stage: 1, round: n_1 })` |
| A lookup output off its table, such as an S-box's | `Reject(Lookup { table })` (RFC-0008) |
| Request `r`'s tokens bound to another request's frames | the token `AssertEqual`: `Reject(Sumcheck { stage: 1, round: n_1 })` |

- **A lie in a round** fails a later check, as RFC-0072 'Messages, rejections and stages' says. `n_1` is stage 1's rounds.
- **Another log.** A proof built for another log names another manifest, since the manifest binds its log: `Reject(StatementMismatch { field: ManifestDigest })` at step 4.
- **End to end.** 07 §6's `e2e_anchor_mismatch` takes two cases: a log signed without a request's middle frame, the statement's lengths those of all its frames, `Reject(Anchor { request_id: Some(r), kind: Frames })`; and altered tokens proven with the pre-check bypassed, `Reject(Sumcheck { stage: 1, round: n_1 })`, in a request without outputs or at a prompt token whose change keeps every greedy output. On a generation request whose greedy output the change moves, the greedy certificate fails first (below).
- **The model's checks come first** when a cheat also breaks them: a changed token that changes a greedy output fails the greedy certificate first. Each row above assumes a witness that keeps every other check.

### Soundness

- **No new relation.** The binding uses RFC-0066's `BitDecompose` and `AssertEqual`, RFC-0072's selection relation, and RFC-0010's linear substitution. Each instance counts in its stage's `m`, and each zero-check in RFC-0014's `zero_checks`.
- **Per session, the binding adds** two `AssertEqual` zero-checks, of `⌈log2 |P_s|⌉` and `⌈log2 T_s⌉` variables, and three selections. `p_s`'s range check, counted with AES in 'Sizes', adds its booleanity, its recomposition and, when `B_s` is not a power of two, its input padding check.
- **Uniqueness** is 'Every limb pinned', then RFC-0066's and RFC-0072's for the circuits.
- **Computational.** INV-ANC-1 also rests on SHA-256's collision resistance ('INV-ANC-1').
- **The budget.** The calculator reads the composed schedule (RFC-0053 'Schedule, transcript and budget'). RFC-0072 puts 500 blocks at about `2^{−106.5}` in the union row of the zero-checks. A batch of 16K tokens, all outputs streamed one token per frame, would hold about 32K blocks: about `2^{−100.5}` from the anchor's zero-checks alone, at the edge of a 100-bit target before the model's terms. Prompts in long frames cost about one block per 16 tokens. Within the interim cap a batch holds at most 4,096 blocks, about `2^{−103.5}` by the same figure. Neither calculator has been run on the composed schedule at 405B (OQ-0074-2).

### Sizes

From the compiler's schedule on `tiny-2-minimal` (2 layers, `d_model` 64, vocabulary 512) with a generation request, one session, the anchor built as above. The schedule is `derive_composed_schedule` with `model_natives`. The opening points are counted on the anchor's part alone, proven by the CPU stage prover with its tokens public.

| Batch | Frames | Stages | Instances | Evaluations | Split values | Pool claims | Zero-checks | Anchor tensors | Opening points |
|---|---|---|---|---|---|---|---|---|---|
| The model alone, `n = 4` | 0 | 4 | 221 | 330 | 0 | 236 | 4 | 0 | — |
| Prompt 2, outputs 2 | 2 | 14 | 17,555 | 24,246 | 3,437 | 13,020 | 3,438 | 1,973 | 27 |
| Prompt 2, outputs 1 + 1 | 3 | 14 | 24,424 | 33,751 | 4,816 | 18,055 | 4,809 | 2,710 | 27 |
| Prompt 3, outputs 5 × 1, `n = 8` | 6 | 14 | 45,028 | 62,263 | 8,950 | 33,157 | 8,919 | 4,921 | 30 |

- **The columns.**
  - Instances and evaluations are every stage's, the leaf stage's none.
  - Split values are the schedule's split parts, `stage/{k}/split`, stage 0's atom splits included (1 in the second and third batches), the leaf stage's 332 lookup values of AES excluded.
  - Zero-checks are stage 1's graph zero-check instances. RFC-0053's native zero relations are not counted; with them the four rows hold 14, 3,448, 4,819 and 8,929.
  - Anchor tensors are its committed advice tensors, lookup outputs included.
  - Opening points are the distinct points of the anchor's opening targets.
  - The model alone's figures are the same at `n = 8`.
- **The binding.** Without it, the first batch takes 17,550 instances, 24,240 evaluations, 13,017 pool claims and 3,436 zero-checks.
- **The rest.** Each further frame of one token adds about 6,900 instances, its SHA-256's two blocks.
- **Opening points.** They barely grow with the frames: circuits of one shape share their stages' challenges. In the composed schedule, the pool's advice claims lie at 45 distinct points, the model alone's at 21, counting each leaf site at the leaf point.
- **Rows.** Each anchor tensor is one row at `k = 2^{22}`, and at `2^{10}` in the first two batches. In the third, `p_s`'s bits `[16, 13, 8]` pad to 2,048 entries, two rows at `2^{10}`. The anchor's four tables add 67 multiplicity rows at `2^{10}` and 4 at `2^{22}`.

## Alternatives Considered

1. **A committed limb tensor `[T, 4]` and a virtual plaintext.** The plaintext would be its prefix plus the placed limbs. Rejected: range analysis bounds the sum of two byte tensors by `[0, 510]`, outside `Bitwise8`'s operands. Assembling it at the bit level, as RFC-0066 assembles a message, costs a `BitDecompose` of every token and three value levels per byte.
2. **Tokens computed from the limbs.** The token input would be virtual, `Σ_b 2^{8b}·limb_b`. Rejected: RFC-0006 admits `OneHot`'s range proof only on a committed input read directly. It would also change the model's graph.
3. **A committed ciphertext**, bound to AES's by an `AssertEqual`. Rejected: it commits every ciphertext byte twice, and AES's lookup outputs already are those bytes.
4. **One SHA-256 circuit for a session's frames of one length**, `M` of them. It cuts a streamed request's instances by about `M` times. Deferred (OQ-0074-1): `Reshape([K·M])` to `[K, M]` maps coordinates only when `M` is a power of two, so it needs a builder that gathers `[K, M]` otherwise.
5. **Frame lengths in `RequestRef`**, so that INV-PCS-1 stays as it was. Rejected: it copies the log into the statement and needs a check that the two agree, while the verifier holds the verified log at step 4 already.
6. **`AnchorSpec { frame_format }` in the claim list.** Rejected: the log entry's schema version already fixes the frame format. A claim list describes `D`, and the anchor is the deployment's.

## Drawbacks

- **Stages.** A batch takes 14 stages, against Llama's 3 or 4. Stages 4 to 13 hold the anchor alone; in stages 1 to 3 each instance is front-padded to its stage's widest, as any is.
- **Streamed outputs.** A frame per token costs two SHA-256 blocks per token: 12,100 nodes (the compiler's 12,098 for its SHA-256, plus its `Slice` and `Reshape`) and 6,869 instances per one-token frame. A batch of 16K tokens streamed one per frame holds 32,768 blocks, about `2.0·10^8` nodes and `1.1·10^8` instances, and spends the soundness budget ('Soundness', OQ-0074-2).
- **Proof size.** The binding cost is rows and bytes. Each SHA-256 block commits 368 tensors, one advice row each (RFC-0072), and a row opens `τ_A × 8` bytes, 1,240 at `τ_A = 155` (RFC-0072's 228 MB per 184,000 rows). The anchor's opening points add about 24 groups of `2^{22} × 16` bytes ('Sizes'), 1.6 GB. For a batch of 16K tokens, against 405B's own 953,580 rows and about 2.8 GB (RFC-0011, 08 §5):

  | Framing | SHA-256 blocks | Advice rows | Opened columns, `τ_A` = 155 / 193 | Evaluations and splits | With the groups |
  |---|---|---|---|---|---|
  | One token per frame | 32,768 | 12.08 M | 15.0 / 18.6 GB | 2.85 GB | about 19.5 GB |
  | Frames of 64 tokens | 1,536 | 0.57 M | 0.70 / 0.87 GB | 0.13 GB | about 2.4 GB |
  | Frames of 1,024 tokens | 1,056 | 0.39 M | 0.48 / 0.60 GB | 0.09 GB | about 2.2 GB |

  - **What breaks.** Every framing breaks 08's gate of 4 GiB for the total proof: the groups alone take a 405B proof past it. One token per frame also breaks RFC-0003's 8 GiB decode limit, so its honest proof would be `Reject(Malformed { ERR-CODEC-LIMIT })` at a default verifier.
  - **Why the cap.** A verifier must provision for `A_b`, the bound, whatever the actual framing. So OQ-0074-1 and OQ-0072-3 are needed before production verifies any batch of the largest size, not only before it proves streamed outputs. The interim cap ('The size bound') keeps every admissible proof under the decode limit meanwhile, at the price of batches of at most about 2,000 tokens and of requests of at most 2,045.
- **The 4 GiB gate.** It stays the model's until OQ-0072-3. An anchored 405B proof exceeds it by its opening groups even within the cap.
- **A gather per session over the token axis.** The verifier evaluates its kernel in `O(T_s·(log T_s + log N))`.
- **Graph size.** A SHA-256 block is 6,032 nodes (RFC-0066). A batch of 16K tokens holds about 1,000 blocks in long frames, 1,536 in frames of 64 tokens and 32,768 streamed one per frame: 6 million, 9 million and `2.0·10^8` nodes. Within the interim cap, an anchor holds at most about 25 million nodes, which the verifier provisions for ('Its resources', [RFC-0019](RFC-0019-verifier.md#robustness)).

## Migration / Rollout

1. **This RFC.**
2. **#146, in slices.** The prover workspace builds the composed graph, its leaf stage, its layout and openings, and the CPU prover's anchored proof. Its tests verify the stages through the in-test verifier written from RFC-0072 and this RFC, as #526's did.
3. **#150 and #159** mirror the derivation and the relations in `tv-core`.
4. **#163** runs the anchor end to end. `prove --sample` then emits anchored proofs and `tessera-verify` requires them. Until both workspaces hold the anchor, `tessera-protocol/1` acceptance makes no anchor claim (02 §3).
   - **`prove --reference-tokens`** (02 §2.2) has no transcripts and no logs, so it cannot anchor its statement, and no verifier that derives the anchor accepts its proofs (step 4, `layout_digest`). From #163 on it is a test path only, under the `test-params` feature, for unanchored proofs of `test-small/1`, whose acceptance certifies nothing (02 §3). #163 amends 02 §2.2.
5. **#551** builds batches of several requests, whose cut and refusal ('Batches of several requests') land with it.
6. **The interim cap** ('The size bound') holds until OQ-0074-1 and OQ-0072-3 bring a batch at its bound within RFC-0003's decode limit and 08's gate. Until then:
   - every verifier provisions for `A_max = 4,096` blocks;
   - production audits prove requests of at most 2,045 tokens, in batches the packer splits by the cap;
   - longer sampled requests are reported unproven (RFC-0016).

   Lifting the cap is a change of this RFC, or of its successor, with the figures recomputed.

## Testing Strategy

- **INV-ANC-1 on tiny models.** For a tiny conformance model, a request of two frames and one of three, each case is a proof of the composed graph:
  - a token unequal to its frame's bytes, in a request without outputs or at a prompt token whose change keeps every greedy output;
  - a plaintext byte `(c ⊕ (k + 1)) + 256` at a keystream byte `k < 255` of a token below 256, the token at its row the recomposed value, AES and the digest unchanged, in a model whose vocabulary holds that value (`tiny-2-minimal`'s 512 does) and a request without outputs or a prompt token whose change keeps every greedy output;
  - a recorded frame whose plaintext prefix disagrees with its header;
  - a tail byte set to 1;
  - a stored frame other than the recorded one, proven without the pre-check;
  - request 1's tokens bound to request 0's frames, once #551 builds two requests.

  Each is rejected at `Reject(Sumcheck { stage: 1, round: n_1 })`. Each but the stored frame's is accepted with exactly its check removed. A tampered stored transcript is `ERR-WITNESS-ANCHOR` before `Round1`.
- **Unique witnesses.** The honest plaintext is the only one that passes, given the tokens. Each alternative fails an identity on the witness: a byte borrowed into its neighbour, a padded block's byte set, a tail byte set.
- **The model's ids.** In the composed graph, every model node, tensor, gadget and table keeps its id, and the model's sites keep their order. The certificate's id exceeds every gadget's.
- **AES-128-CTR's graph.** Built from 'The AES-128-CTR circuit' alone, it gives FIPS-197's and NIST SP 800-38A F.5.1's vectors, 285 nodes, and RFC-0072's AES row of sizes.
- **Derivations.** An oracle written from this RFC gives the anchor's inputs, block offsets, `P_s`, `D_b` and `R_s` for windows across two periods, sessions shared by two requests, a `session_id` reopened under another key, and frames of no token. The derivation matches it. The oracle also asserts, over every case it enumerates (soundness-review 'Anchors'):
  - `P_s` and the four `D_b` partition the real bytes of the session's plaintext, `[0, 16·B_s)`, exactly;
  - `R_s`, over the batch's sessions, lists each real row of each request exactly once.
- **The size bound.** A request of `2·n_r + 4` SHA-256 blocks is admitted and one of `2·n_r + 5` is `Reject(Anchor { request_id: Some(r), kind: Frames })`, by frames of no token added to an honest log; every log of one token per frame passes, with two frames of no token.
- **The cap and the resources.**
  - A batch with `A_b = 4,096` is admitted, and one with `4,097` is `Reject(StatementMismatch { field: Batch })` before any entry is read.
  - A verifier whose `max_anchor_blocks` is below a batch's `A_b` returns `Inconclusive(ERR-RES-MEMORY)` before it builds the anchor. In the same batch, a request past its own size bound is `Reject` once the limit allows the batch.
  - A batch whose requests each take their own session (`S = R`) stays within the `+ 5` per request.
- **Degenerate shapes.** Each proves and verifies through the in-test verifier:
  - a session of one token, whose token `AssertEqual` has 0 variables;
  - a session whose token count is not a power of two;
  - a plaintext of `B_s` blocks not a power of two.
- **Token axes.** In the composed graph, every anchor tensor has none. A grouped leaf stage and the ungrouped one give the same merged claims.
- **Zero rows.** A tensor non-zero at padding whose row holds only padded positions is committed, with its computed values, and opens. Skipping that row fails the opening.
- **Refusals** (`ERR-INTERNAL`):
  - a gather of the token axis with an index at a padded row, in a graph of two requests;
  - a `ConstAdd` reached from the tokens other than through the gather;
  - in `tessera-protocol/2`, a batch, by `ERR-USAGE`;
  - a batch of temperature, by `ERR-USAGE`.
- **Sizes.** The 'Sizes' table, recomputed from the compiler.

## Open Questions

- **OQ-0074-1. Fewer SHA-256 circuits.** Should a session's frames of one length share one circuit of `M` messages? A request that streams `O` one-token frames would then take one frame circuit, not `O`. It needs a builder that gathers `[K, M]` for any `M`, or messages padded to a power of two. It also keeps the rows from growing with `M`: a `[32, M]` tensor stays one row while it fits one. With OQ-0072-3, which brings the anchor's opening groups back toward the model's, it decides when the interim cap can be lifted ('Drawbacks').
- **OQ-0074-2. The soundness budget at production sizes.** Streamed outputs would bring a 16K-token batch to about 32K SHA-256 blocks, near a 100-bit target ('Soundness'); the interim cap holds a batch to 4,096 meanwhile. The calculators should be run on the composed schedule at 405B with a stated prompt and output mix. A schedule under its target is `Reject(Params { name })` today (RFC-0053), so an honest streamed batch could be refused, a completeness gap. If a batch misses its target: should the calculator bound stage 1's zero-checks by their largest instance (OQ-0072-1), should RFC-0016's packer split a batch by blocks as RFC-0071 splits one past the maximum layout, or should OQ-0074-1's grouping, which keeps the instances from growing with the frames, settle it? The answer is needed before the anchor proves production batches.

## References

- [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0007](RFC-0007-claim-list.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0015](RFC-0015-transcript-anchor.md), [RFC-0019](RFC-0019-verifier.md), [RFC-0026](RFC-0026-claim-list-layout.md), [RFC-0037](RFC-0037-stage-schedule.md), [RFC-0041](RFC-0041-execution-plan.md), [RFC-0045](RFC-0045-leaf-layout.md), [RFC-0053](RFC-0053-greedy-runtime-boundary.md), [RFC-0054](RFC-0054-token-groups.md), [RFC-0059](RFC-0059-frame-headers-and-nonces.md), [RFC-0063](RFC-0063-request-boundaries.md), [RFC-0066](RFC-0066-bits-and-sha256.md), [RFC-0071](RFC-0071-request-slots.md), [RFC-0072](RFC-0072-anchor-relations.md).
- [#146](https://github.com/AbdelStark/tessera/issues/146), [#150](https://github.com/AbdelStark/tessera/issues/150), [#159](https://github.com/AbdelStark/tessera/issues/159), [#163](https://github.com/AbdelStark/tessera/issues/163), [#526](https://github.com/AbdelStark/tessera/issues/526), [#551](https://github.com/AbdelStark/tessera/issues/551).
