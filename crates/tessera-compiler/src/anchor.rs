// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The AES-128-CTR circuit of the transcript anchor (RFC-0015 'Circuits' step 2, RFC-0059
//! 'The circuits'), lowered into the IR: the key schedule once per key, ten rounds per counter
//! block, and the keystream's XOR with the plaintext.
//!
//! Every byte operation is a lookup the IR already has (RFC-0006 'Operation set'), and every
//! rearrangement of bytes a public linear map:
//!
//! - **SubBytes** is `TableMap` into `AesSbox`;
//! - **MixColumns** is `TableMap` into `Gf256Mul { 2 }` and `Gf256Mul { 3 }`, then three XORs;
//! - **AddRoundKey**, the round keys' words and the ciphertext are `Bitwise` into
//!   `Bitwise8 { xor }`, whose key is `a + 2^8 · b`;
//! - **ShiftRows** and MixColumns' reads of a column's other bytes are `GatherPublic` over the
//!   byte axis; a round key reaches every block through `Expand`.
//!
//! A state is a `[16, B]` tensor, its outer axis the 16 bytes in FIPS-197's input order, byte
//! `r + 4c` the state's row `r` and column `c` (FIPS-197 §3.4), and its inner axis the `B`
//! blocks, so that the byte rearrangements are gathers of the outermost axis.
//!
//! **What its inputs must be.** The key's and the plaintext's bytes enter `Bitwise` lookups as
//! operands, and a key `a + 2^8 · b` determines `a` and `b` only if each lies in `[0, 2^8)`. Every
//! lookup output does: it is a table value. The inputs are range-checked by their producers, as
//! RFC-0015 assigns them: the key `k_s` by the SHA-256 circuit's committed bits of
//! `"tessera-anchor-key/v1" ‖ k_s ‖ salt_k` (step 3, #143), the plaintext's token bytes by their
//! own range checks against the token stream (step 1, #146), and its 22-byte prefix and the
//! counter blocks are public (RFC-0059). A graph that uses this circuit without those producers
//! does not bind its inputs to bytes.
//!
//! **Cost.** Per 16-byte block: 16 lookups for the first AddRoundKey, 112 for each of rounds 1
//! to 9 (16 S-box, 32 `Gf256Mul`, 64 XOR), 32 for round 10, and 16 for the ciphertext: 1,072, so
//! 67 per plaintext byte and 268 per four-byte token. Each lookup commits its output, one byte.
//! The key schedule costs 72 lookups per round, 720 per key, once per session.

use tessera_ir::{
    CommitmentClass, GraphBuilder, IntRange, IntType, IrError, NodeKind, Op, TableId, TensorDecl,
    TensorId,
};

/// The four tables the circuit reads (RFC-0015 'Circuits' step 2, RFC-0008 'Table kinds').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AesTables {
    /// `AesSbox`, FIPS-197 §5.1.1.
    pub sbox: TableId,
    /// `Gf256Mul { c: 2 }`.
    pub mul2: TableId,
    /// `Gf256Mul { c: 3 }`.
    pub mul3: TableId,
    /// `Bitwise8 { op: Xor }`.
    pub xor: TableId,
}

/// The round constants, one row per round from 1 to 10, as the word `[rc_r, 0, 0, 0]` the key
/// schedule XORs into its first word (FIPS-197 §5.2): the public input `rcon` of
/// [`aes128_key_schedule`] holds these values.
pub const RCON: [[i64; 4]; 10] = [
    [0x01, 0, 0, 0],
    [0x02, 0, 0, 0],
    [0x04, 0, 0, 0],
    [0x08, 0, 0, 0],
    [0x10, 0, 0, 0],
    [0x20, 0, 0, 0],
    [0x40, 0, 0, 0],
    [0x80, 0, 0, 0],
    [0x1b, 0, 0, 0],
    [0x36, 0, 0, 0],
];

/// The lookups of one block's encryption and XOR with its plaintext.
pub const LOOKUPS_PER_BLOCK: u64 = 16 + 9 * 112 + 32 + 16;

/// The lookups of one key's schedule.
pub const LOOKUPS_PER_KEY: u64 = 10 * 72;

/// The tensors of an AES-128-CTR circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AesCtr {
    /// The round keys `rk_0` (the key itself) to `rk_10`, each `[16]`.
    pub round_keys: [TensorId; 11],
    /// The keystream, `[16, B]`: each counter block encrypted.
    pub keystream: TensorId,
    /// The ciphertext, `[16, B]`: the plaintext XOR the keystream.
    pub ciphertext: TensorId,
}

/// An unsigned byte.
const BYTE: IntType = IntType {
    bits: 8,
    signed: false,
};

/// `[0, 255]`.
const BYTES: IntRange = IntRange { lo: 0, hi: 255 };

fn decl(shape: &[u32], kind: NodeKind) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: BYTE,
        range: BYTES,
        kind,
    }
}

/// The declarations of the circuit's inputs, for `blocks` counter blocks: the key, committed
/// advice, `[16]`; the round constants, public, `[10, 4]`, holding [`RCON`]; the counter blocks,
/// public, `[16, B]`; and the plaintext, committed advice, `[16, B]`.
pub fn aes128_ctr_inputs(blocks: u32) -> [TensorDecl; 4] {
    let advice = NodeKind::Committed {
        commitment: CommitmentClass::Advice,
    };
    [
        decl(&[16], advice),
        decl(&[10, 4], NodeKind::Public),
        decl(&[16, blocks], NodeKind::Public),
        decl(&[16, blocks], advice),
    ]
}

/// One node of one output.
fn one(
    g: &mut GraphBuilder,
    op: Op,
    inputs: &[TensorId],
    output: TensorDecl,
) -> Result<TensorId, IrError> {
    let ids = g.node(op, inputs, vec![output])?;
    Ok(ids[0])
}

/// The rows of `x`'s outermost axis at `index`, in order.
fn gather(
    g: &mut GraphBuilder,
    x: TensorId,
    index: Vec<i64>,
    inner: &[u32],
) -> Result<TensorId, IrError> {
    let mut shape = vec![u32::try_from(index.len()).unwrap_or(u32::MAX)];
    shape.extend_from_slice(inner);
    let index = g.constant(index)?;
    one(
        g,
        Op::GatherPublic { index },
        &[x],
        decl(&shape, NodeKind::Virtual),
    )
}

/// `a ⊕ b`, elementwise.
fn xor(
    g: &mut GraphBuilder,
    tables: &AesTables,
    a: TensorId,
    b: TensorId,
    shape: &[u32],
) -> Result<TensorId, IrError> {
    one(
        g,
        Op::Bitwise { table: tables.xor },
        &[a, b],
        decl(shape, NodeKind::LookupOutput { table: tables.xor }),
    )
}

/// `table(x)`, elementwise.
fn map(
    g: &mut GraphBuilder,
    table: TableId,
    x: TensorId,
    shape: &[u32],
) -> Result<TensorId, IrError> {
    one(
        g,
        Op::TableMap { table },
        &[x],
        decl(shape, NodeKind::LookupOutput { table }),
    )
}

/// Byte `j` of column `c`'s word repeated in every column: the indices `4c, …, 4c + 3`, four
/// times.
fn column_everywhere(c: i64) -> Vec<i64> {
    (0..16).map(|j| 4 * c + j % 4).collect()
}

/// ShiftRows: byte `r + 4c` of the output is byte `r + 4((c + r) mod 4)` of the input
/// (FIPS-197 §5.1.2).
fn shift_rows() -> Vec<i64> {
    (0..16)
        .map(|j| (j % 4) + 4 * ((j / 4 + j % 4) % 4))
        .collect()
}

/// Byte `r + 4c` of the output is byte `((r + k) mod 4) + 4c` of the input: the column's byte
/// `k` rows down, as MixColumns reads it (FIPS-197 §5.1.3).
fn down(k: i64) -> Vec<i64> {
    (0..16).map(|j| (j % 4 + k) % 4 + 4 * (j / 4)).collect()
}

/// The key schedule of FIPS-197 §5.2: `rk_0 = key`, and for `r` from 1 to 10, with `D = rk_{r−1}`
/// and `t = SubWord(RotWord(D_3)) ⊕ rcon_r`, the word `c` of `rk_r` is
/// `t ⊕ D_0 ⊕ … ⊕ D_c`.
///
/// The IR has no concatenation, and its range analysis bounds a sum of byte tensors only by the
/// sum of their ranges. So each round builds `rk_r` as a `[16]` tensor of four XORs: `t` and
/// `D_0` into every column, then `D_1`, `D_2` and `D_3` into the columns at and after theirs,
/// each masked to zero before its column by a per-byte `ConstMul`. XOR with 0 is a lookup like
/// any other.
///
/// # Errors
///
/// An [`IrError`] of the builder: inputs that are not a `[16]` key and the `[10, 4]` round
/// constants.
pub fn aes128_key_schedule(
    g: &mut GraphBuilder,
    tables: &AesTables,
    key: TensorId,
    rcon: TensorId,
) -> Result<[TensorId; 11], IrError> {
    let mut keys = [key; 11];
    for r in 1..11 {
        let d = keys[r - 1];
        let rot = gather(g, d, vec![13, 14, 15, 12], &[])?;
        let sub = map(g, tables.sbox, rot, &[4])?;
        let row = gather(g, rcon, vec![i64::try_from(r - 1).unwrap_or(0)], &[4])?;
        let rc = one(g, Op::Reshape, &[row], decl(&[4], NodeKind::Virtual))?;
        let t = xor(g, tables, sub, rc, &[4])?;
        let t16 = gather(g, t, (0..16).map(|j| j % 4).collect(), &[])?;
        let d0 = gather(g, d, column_everywhere(0), &[])?;
        let mut acc = xor(g, tables, t16, d0, &[16])?;
        for c in 1..4i64 {
            let dc = gather(g, d, column_everywhere(c), &[])?;
            let mask = g.constant((0..16).map(|j| i64::from(j / 4 >= c)).collect())?;
            let masked = one(
                g,
                Op::ConstMul { c: mask },
                &[dc],
                decl(&[16], NodeKind::Virtual),
            )?;
            acc = xor(g, tables, acc, masked, &[16])?;
        }
        keys[r] = acc;
    }
    Ok(keys)
}

/// AES-128 encryption of `B` blocks, `[16, B]`, under the eleven round keys (FIPS-197 §5.1).
///
/// # Errors
///
/// An [`IrError`] of the builder: blocks that are not `[16, B]`, or round keys that are not
/// `[16]`.
pub fn aes128_encrypt(
    g: &mut GraphBuilder,
    tables: &AesTables,
    round_keys: &[TensorId; 11],
    blocks: TensorId,
    b: u32,
) -> Result<TensorId, IrError> {
    let state_shape = [16, b];
    let key_plane = |g: &mut GraphBuilder, rk: TensorId| {
        one(
            g,
            Op::Expand { axis: 1, len: b },
            &[rk],
            decl(&state_shape, NodeKind::Virtual),
        )
    };
    let k0 = key_plane(g, round_keys[0])?;
    let mut state = xor(g, tables, blocks, k0, &state_shape)?;
    for (r, &rk) in round_keys.iter().enumerate().skip(1) {
        let sub = map(g, tables.sbox, state, &state_shape)?;
        let shifted = gather(g, sub, shift_rows(), &[b])?;
        let mixed = if r < 10 {
            let twice = map(g, tables.mul2, shifted, &state_shape)?;
            let thrice = map(g, tables.mul3, shifted, &state_shape)?;
            let thrice_below = gather(g, thrice, down(1), &[b])?;
            let two_below = gather(g, shifted, down(2), &[b])?;
            let three_below = gather(g, shifted, down(3), &[b])?;
            let x = xor(g, tables, twice, thrice_below, &state_shape)?;
            let x = xor(g, tables, x, two_below, &state_shape)?;
            xor(g, tables, x, three_below, &state_shape)?
        } else {
            shifted
        };
        let plane = key_plane(g, rk)?;
        state = xor(g, tables, mixed, plane, &state_shape)?;
    }
    Ok(state)
}

/// AES-128-CTR over `B` counter blocks (NIST SP 800-38A §6.5): the key schedule, each counter
/// block encrypted, and the keystream XOR the plaintext. The inputs are those of
/// [`aes128_ctr_inputs`], in its order.
///
/// # Errors
///
/// An [`IrError`] of the builder, for inputs of other shapes.
pub fn aes128_ctr(
    g: &mut GraphBuilder,
    tables: &AesTables,
    [key, rcon, counters, plaintext]: [TensorId; 4],
    b: u32,
) -> Result<AesCtr, IrError> {
    let round_keys = aes128_key_schedule(g, tables, key, rcon)?;
    let keystream = aes128_encrypt(g, tables, &round_keys, counters, b)?;
    let ciphertext = xor(g, tables, plaintext, keystream, &[16, b])?;
    Ok(AesCtr {
        round_keys,
        keystream,
        ciphertext,
    })
}
