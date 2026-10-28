// SPDX-License-Identifier: Apache-2.0 OR MIT

//! In-circuit Sampler PRF lowering (RFC-0017 'Sampler PRF', RFC-0061 'The circuits').
//!
//! Lowers all four steps of RFC-0061 'The circuits' into the IR so they pass full range analysis
//! ([`crate::analyze`]), stage scheduling ([`crate::derive_graph_schedule`]), and exact execution
//! (`tessera_witness::Engine`):
//!
//! 1. **Base-key opening**: SHA-256 over `"tessera-sampler-key/v1" ‖ K_p ‖ salt` (`1` block,
//!    `K_p` and `salt` committed in one `[32, 1]` byte input) bound via [`sha256_bind`] to the
//!    batch's pinned `sampler_commitment`.
//! 2. **Pin key derivation**: SHA-256 over
//!    `"tessera-sampler-derive/v1" ‖ K_p ‖ 0^23 ‖ nonce ‖ u8(b) ‖ u64_le(from_ns)` (`2` blocks,
//!    slicing the same committed `K_p`), taking the first 16 digest bytes as `K` in `[0, 255]`.
//! 3. **Stream class (`b ≥ 1`)**: 1-block compression of `"tessera-stream-class/v1" ‖ K ‖ 0^25`
//!    once per pin to produce the 256-bit midstate, followed per request by `⌈(17 + 4n)/64⌉`
//!    compressions over `u64_le(n) ‖ u32_le(x_0) ‖ … ‖ u32_le(x_{n−1})`, extracting the top `b`
//!    bits of `u64_le(h[0..8])` into `u64_le(class)`. At `b = 0`, `class = 0` with no class hash.
//! 4. **PRF stream**: AES-128 under `K` over `u64_le(class) ‖ u32_le(t) ‖ 0^4`, combining the low
//!    four output bytes in a two-level binary tree so range analysis bounds every draw `u_t`
//!    tightly in `[0, 2^32 − 1]`.

use tessera_ir::{
    BoolFn, CommitmentClass, GraphBuilder, IntRange, IntType, IrError, NodeKind, Op, TensorDecl,
    TensorId,
};

use super::{
    AesTables, RCON, SHA256_IV, SHA256_MASKS, Sha256Layout, aes128_encrypt, aes128_key_schedule,
    sha256, sha256_bind, sha256_bits, sha256_digest_input, sha256_inputs, sha256_pad,
};

/// Domain-separation prefix of `sampler_commitment` (22 bytes, RFC-0059 / RFC-0061).
pub const SAMPLER_KEY_LABEL: &[u8] = b"tessera-sampler-key/v1";

/// Domain-separation prefix of pin key derivation (25 bytes, RFC-0061).
pub const SAMPLER_DERIVE_LABEL: &[u8] = b"tessera-sampler-derive/v1";

/// Domain-separation prefix of the stream-class key block (23 bytes, RFC-0061).
pub const SAMPLER_CLASS_LABEL: &[u8] = b"tessera-stream-class/v1";

const BIT: IntType = IntType {
    bits: 1,
    signed: false,
};

const BYTE: IntType = IntType {
    bits: 8,
    signed: false,
};

const U32_TY: IntType = IntType {
    bits: 32,
    signed: false,
};

const ADVICE: NodeKind = NodeKind::Committed {
    commitment: CommitmentClass::Advice,
};

fn bit_decl(shape: &[u32], kind: NodeKind) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: BIT,
        range: IntRange { lo: 0, hi: 1 },
        kind,
    }
}

fn byte_decl(shape: &[u32], kind: NodeKind) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: BYTE,
        range: IntRange { lo: 0, hi: 255 },
        kind,
    }
}

fn virtual_decl(shape: &[u32]) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: IntType {
            bits: 64,
            signed: true,
        },
        range: IntRange { lo: 0, hi: 0 },
        kind: NodeKind::Virtual,
    }
}

fn one(
    g: &mut GraphBuilder,
    op: Op,
    inputs: &[TensorId],
    output: TensorDecl,
) -> Result<TensorId, IrError> {
    Ok(g.node(op, inputs, vec![output])?[0])
}

/// Public pin parameters supplied by the verifier for a temperature batch (RFC-0061).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplerPinPublic {
    /// Pinned `SHA-256("tessera-sampler-key/v1" ‖ K_p ‖ salt)`.
    pub sampler_commitment: [u8; 32],
    /// Verifier-issued 32-byte randomness for this pin window.
    pub nonce: [u8; 32],
    /// Stream-class width `b ≤ 64`.
    pub class_bits: u8,
    /// Pin window start in nanoseconds.
    pub from_ns: u64,
}

/// Layout of `"tessera-sampler-key/v1" ‖ K_p ‖ salt` (`54` bytes -> `1` SHA-256 block).
#[must_use]
pub fn sampler_commit_layout() -> Sha256Layout {
    Sha256Layout {
        blocks: 1,
        messages: 1,
        committed: (22..54).collect(),
    }
}

/// Layout of `"tessera-sampler-derive/v1" ‖ K_p ‖ 0^23 ‖ nonce ‖ u8(b) ‖ u64_le(from_ns)` (`105`
/// bytes -> `2` SHA-256 blocks).
#[must_use]
pub fn sampler_derive_layout() -> Sha256Layout {
    Sha256Layout {
        blocks: 2,
        messages: 1,
        committed: (25..41).collect(),
    }
}

/// Layout of the stream-class key block `"tessera-stream-class/v1" ‖ K ‖ 0^25` (`64` raw bytes ->
/// `1` compression block without SHA-256 length padding).
#[must_use]
pub fn sampler_key_block_layout() -> Sha256Layout {
    Sha256Layout {
        blocks: 1,
        messages: 1,
        committed: (23..39).collect(),
    }
}

/// Layout of a request's stream-class suffix `u64_le(n) ‖ u32_le(x_0) ‖ … ‖ padding`, compressed
/// from the key-block midstate.
#[must_use]
pub fn sampler_class_layout(prompt_len: u32) -> Sha256Layout {
    let token_bytes = 4u32.saturating_mul(prompt_len);
    let rest = 8u64 + u64::from(token_bytes);
    let blocks = u32::try_from((rest + 9).div_ceil(64)).unwrap_or(u32::MAX);
    Sha256Layout {
        blocks,
        messages: 1,
        committed: (8..8 + token_bytes).collect(),
    }
}

/// Public message bits (`[512, 1]`) for [`sampler_commit_layout`], with secret `K_p` and `salt`
/// zeroed.
#[must_use]
pub fn sampler_commit_public_bits() -> Vec<i64> {
    let mut msg = vec![0u8; 54];
    msg[..SAMPLER_KEY_LABEL.len()].copy_from_slice(SAMPLER_KEY_LABEL);
    let padded = sha256_pad(&msg);
    sampler_commit_layout().public_bits(&[padded.as_slice()])
}

/// Public message bits (`[1024, 1]`) for [`sampler_derive_layout`], with secret `K_p` zeroed.
#[must_use]
pub fn sampler_derive_public_bits(nonce: &[u8; 32], class_bits: u8, from_ns: u64) -> Vec<i64> {
    let mut msg = vec![0u8; 105];
    msg[..SAMPLER_DERIVE_LABEL.len()].copy_from_slice(SAMPLER_DERIVE_LABEL);
    msg[64..96].copy_from_slice(nonce);
    msg[96] = class_bits;
    msg[97..105].copy_from_slice(&from_ns.to_le_bytes());
    let padded = sha256_pad(&msg);
    sampler_derive_layout().public_bits(&[padded.as_slice()])
}

/// Public message bits (`[512, 1]`) for [`sampler_key_block_layout`], with secret `K` zeroed.
#[must_use]
pub fn sampler_key_block_public_bits() -> Vec<i64> {
    let mut block = [0u8; 64];
    block[..SAMPLER_CLASS_LABEL.len()].copy_from_slice(SAMPLER_CLASS_LABEL);
    sampler_key_block_layout().public_bits(&[&block[..]])
}

/// Public message bits (`[512·blocks, 1]`) for [`sampler_class_layout`], with secret prompt tokens
/// zeroed and final bit-length `8 · (64 + 8 + 4 · prompt_len)`.
#[must_use]
pub fn sampler_class_public_bits(prompt_len: u32) -> Vec<i64> {
    let rest = 8usize + 4 * prompt_len as usize;
    let total_len = 64 + rest;
    let padded_len = (rest + 9).div_ceil(64) * 64;
    let mut padded = vec![0u8; padded_len];
    padded[..8].copy_from_slice(&(prompt_len as u64).to_le_bytes());
    padded[rest] = 0x80;
    padded[padded_len - 8..].copy_from_slice(&(8 * total_len as u64).to_be_bytes());
    sampler_class_layout(prompt_len).public_bits(&[padded.as_slice()])
}

/// Public `[256, 1]` SHA-256 IV bits.
#[must_use]
pub fn sha256_iv_bits() -> Vec<i64> {
    let iv: Vec<u8> = SHA256_IV.iter().flat_map(|w| w.to_be_bytes()).collect();
    sha256_bits(&[iv.as_slice()])
}

/// Public `[2, 32]` SHA-256 shift-mask bits.
#[must_use]
pub fn sha256_mask_bits() -> Vec<i64> {
    SHA256_MASKS.iter().flatten().copied().collect()
}

/// Public `[10, 4]` AES-128 round-constant values.
#[must_use]
pub fn aes_rcon_values() -> Vec<i64> {
    RCON.iter().flatten().copied().collect()
}

/// Public `[16, positions.len()]` AES counter-suffix blocks `0^8 ‖ u32_le(t) ‖ 0^4` in `[16, B]`
/// state-major layout (row `byte`, column `index`).
#[must_use]
pub fn sampler_position_blocks(positions: &[u32]) -> Vec<i64> {
    let b = positions.len();
    let mut out = vec![0i64; 16 * b];
    for (col, &t) in positions.iter().enumerate() {
        let le = t.to_le_bytes();
        for (k, &byte) in le.iter().enumerate() {
            out[(8 + k) * b + col] = i64::from(byte);
        }
    }
    out
}

/// Combines `[8, N]` bits (row `bit` of weight `2^{bit}`) into `[N]` bytes in `[0, 255]` via a
/// three-level binary tree so range analysis bounds every level tightly (`[0, 3]`, `[0, 15]`,
/// `[0, 255]`).
fn bits_to_bytes(g: &mut GraphBuilder, mut rows: TensorId, n: u32) -> Result<TensorId, IrError> {
    let mut count = 8i64;
    let mut weight = 2i64;
    while count > 1 {
        let even_idx = g.constant((0..count).step_by(2).collect())?;
        let odd_idx = g.constant((1..count).step_by(2).collect())?;
        count /= 2;
        let shape = [u32::try_from(count).unwrap_or(u32::MAX), n];
        let even = one(
            g,
            Op::GatherPublic { index: even_idx },
            &[rows],
            virtual_decl(&shape),
        )?;
        let odd = one(
            g,
            Op::GatherPublic { index: odd_idx },
            &[rows],
            virtual_decl(&shape),
        )?;
        let c = g.constant(vec![weight])?;
        let scaled = one(g, Op::ConstMul { c }, &[odd], virtual_decl(&shape))?;
        rows = one(g, Op::Add, &[even, scaled], virtual_decl(&shape))?;
        weight *= weight;
    }
    one(g, Op::Reshape, &[rows], virtual_decl(&[n]))
}

/// Packs `words` (`K` `[32, 1]` bit tensors in `[0, 1]`, `1 ≤ K ≤ 8`) into a single `[32·K]` bit
/// tensor in `[0, 1]` using masked `GatherPublic` + `ConstMul` and `Xor3`/`Xor` (`Range(0, 1)`).
fn pack_words(g: &mut GraphBuilder, words: &[TensorId]) -> Result<TensorId, IrError> {
    let total = u32::try_from(32 * words.len()).unwrap_or(u32::MAX);
    let mut placed = Vec::with_capacity(words.len());
    for (w_idx, &word) in words.iter().enumerate() {
        let flat = one(g, Op::Reshape, &[word], bit_decl(&[32], NodeKind::Virtual))?;
        let start = 32 * w_idx;
        let end = start + 32;
        let gather_idx = g.constant(
            (0..total as usize)
                .map(|r| {
                    if (start..end).contains(&r) {
                        (r - start) as i64
                    } else {
                        0
                    }
                })
                .collect(),
        )?;
        let gathered = one(
            g,
            Op::GatherPublic { index: gather_idx },
            &[flat],
            virtual_decl(&[total]),
        )?;
        let mask = g.constant(
            (0..total as usize)
                .map(|r| i64::from((start..end).contains(&r)))
                .collect(),
        )?;
        placed.push(one(
            g,
            Op::ConstMul { c: mask },
            &[gathered],
            virtual_decl(&[total]),
        )?);
    }
    let mut iter = placed.into_iter();
    let mut acc = iter.next().ok_or(IrError::Declaration {
        rule: "at least one word",
    })?;
    while let Some(next1) = iter.next() {
        if let Some(next2) = iter.next() {
            acc = one(
                g,
                Op::BitFn { f: BoolFn::Xor3 },
                &[acc, next1, next2],
                bit_decl(&[total], NodeKind::Virtual),
            )?;
        } else {
            acc = one(
                g,
                Op::BitFn { f: BoolFn::Xor },
                &[acc, next1],
                bit_decl(&[total], NodeKind::Virtual),
            )?;
        }
    }
    Ok(acc)
}

/// Converts the first four SHA-256 chaining words (`[32, 1]` each) into `[16]` big-endian digest
/// bytes with analyzed range `[0, 255]`.
pub fn sha256_first_16_bytes(
    g: &mut GraphBuilder,
    words: &[TensorId; 8],
) -> Result<TensorId, IrError> {
    let packed = pack_words(g, &words[..4])?;
    // Row `bit` (0..8), col `byte` (0..16) -> flat index `16 * bit + byte`.
    // In SHA-256 word `byte / 4`, big-endian byte `byte % 4` holds bits `8*(3 - byte%4) + bit`.
    let perm = g.constant(
        (0..8usize)
            .flat_map(|bit| {
                (0..16usize).map(move |byte| {
                    let w = byte / 4;
                    let k = byte % 4;
                    (32 * w + 8 * (3 - k) + bit) as i64
                })
            })
            .collect(),
    )?;
    let ordered = one(
        g,
        Op::GatherPublic { index: perm },
        &[packed],
        virtual_decl(&[128]),
    )?;
    let bit_planes = one(g, Op::Reshape, &[ordered], virtual_decl(&[8, 16]))?;
    bits_to_bytes(g, bit_planes, 16)
}

/// Inputs and outputs of the base-key commitment opening and pin key derivation subgraph
/// (RFC-0061 'The circuits' steps 1–2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplerDeriveSubgraph {
    /// Public `[512, 1]` message bits of `"tessera-sampler-key/v1" ‖ 0^32 ‖ padding`.
    pub commit_message: TensorId,
    /// Public `[1024, 1]` message bits of `"tessera-sampler-derive/v1" ‖ 0^16 ‖ 0^23 ‖ …`.
    pub derive_message: TensorId,
    /// Public `[256, 1]` SHA-256 IV bits.
    pub iv: TensorId,
    /// Public `[2, 32]` SHA-256 shift-mask bits.
    pub masks: TensorId,
    /// Public `[512]` committed-bit placement for the key commitment.
    pub commit_place: TensorId,
    /// Public `[1024]` committed-bit placement for the derivation.
    pub derive_place: TensorId,
    /// Public `[256, 1]` pinned `sampler_commitment` digest bits.
    pub commitment_digest: TensorId,
    /// Committed `[32, 1]` advice bytes `K_p ‖ salt`.
    pub base_and_salt: TensorId,
    /// Virtual `[16]` derived pin key `K` with analyzed range `[0, 255]`.
    pub derived_key: TensorId,
}

/// Appends the base-key opening and pin-key derivation circuits to `g` (RFC-0061 'The circuits').
///
/// # Errors
/// Propagates any [`IrError`] from `g`.
pub fn build_sampler_derive(g: &mut GraphBuilder) -> Result<SamplerDeriveSubgraph, IrError> {
    let commit_layout = sampler_commit_layout();
    let derive_layout = sampler_derive_layout();
    let commit_decls = sha256_inputs(&commit_layout);
    let derive_decls = sha256_inputs(&derive_layout);

    let commit_message = g.input(commit_decls[0].clone())?;
    let derive_message = g.input(derive_decls[0].clone())?;
    let iv = g.input(commit_decls[1].clone())?;
    let masks = g.input(commit_decls[2].clone())?;
    let commit_place = g.input(commit_decls[4].clone())?;
    let derive_place = g.input(derive_decls[4].clone())?;
    let commitment_digest = g.input(sha256_digest_input(1))?;
    let base_and_salt = g.input(commit_decls[3].clone())?;

    let commit_sha = sha256(
        g,
        &commit_layout,
        &[commit_message, iv, masks, base_and_salt, commit_place],
    )?;
    let commit_words = commit_sha.chaining.last().ok_or(IrError::Declaration {
        rule: "one commit block",
    })?;
    sha256_bind(g, commit_words, commitment_digest, 1)?;

    let base_key = one(
        g,
        Op::Slice {
            axis: 0,
            start: 0,
            len: 16,
        },
        &[base_and_salt],
        byte_decl(&[16, 1], NodeKind::Virtual),
    )?;
    let derive_sha = sha256(
        g,
        &derive_layout,
        &[derive_message, iv, masks, base_key, derive_place],
    )?;
    let derive_words = derive_sha.chaining.last().ok_or(IrError::Declaration {
        rule: "two derive blocks",
    })?;
    let derived_key = sha256_first_16_bytes(g, derive_words)?;

    Ok(SamplerDeriveSubgraph {
        commit_message,
        derive_message,
        iv,
        masks,
        commit_place,
        derive_place,
        commitment_digest,
        base_and_salt,
        derived_key,
    })
}

/// Compresses `"tessera-stream-class/v1" ‖ K ‖ 0^25` once per pin into a `[256, 1]` bit midstate
/// (`Range(0, 1)`), ready to pass as `start` to [`build_sampler_stream_class`].
///
/// # Errors
/// Propagates any [`IrError`] from `g`.
pub fn build_sampler_key_block(
    g: &mut GraphBuilder,
    derived_key: TensorId,
    iv: TensorId,
    masks: TensorId,
) -> Result<(TensorId, TensorId, TensorId), IrError> {
    let layout = sampler_key_block_layout();
    let decls = sha256_inputs(&layout);
    let key_block_message = g.input(decls[0].clone())?;
    let key_block_place = g.input(decls[4].clone())?;
    let key_col = one(
        g,
        Op::Reshape,
        &[derived_key],
        byte_decl(&[16, 1], NodeKind::Virtual),
    )?;
    let sha = sha256(
        g,
        &layout,
        &[key_block_message, iv, masks, key_col, key_block_place],
    )?;
    let words = sha.chaining.last().ok_or(IrError::Declaration {
        rule: "one key block",
    })?;
    let packed = pack_words(g, words)?;
    let midstate = one(
        g,
        Op::Reshape,
        &[packed],
        bit_decl(&[256, 1], NodeKind::Virtual),
    )?;
    Ok((key_block_message, key_block_place, midstate))
}

/// Inputs of one request's stream-class suffix hash (`1 ≤ class_bits ≤ 64`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamClassInputs {
    /// Public `[512·blocks, 1]` message bits from [`sampler_class_public_bits`].
    pub message: TensorId,
    /// Public `[512·blocks]` placement bits from [`sampler_class_layout`], present iff
    /// `prompt_len > 0`.
    pub place: Option<TensorId>,
    /// Virtual `[8]` little-endian `u64_le(class)` bytes in `[0, 255]`.
    pub class_bytes: TensorId,
}

/// Compresses `u64_le(n) ‖ u32_le(x_0) ‖ … ‖ padding` from `midstate` (`[256, 1]`) and extracts
/// the top `class_bits` (`1..=64`) of `u64_le(h[0..8])` into `u64_le(class)` (`[8]` bytes in
/// `[0, 255]`).
///
/// `prompt_tokens` (`Some` iff `prompt_len > 0`) has shape `[prompt_len]` with values in
/// `[0, 2^32)`: either a standalone `[prompt_len]` committed `u32` input (which this function's
/// `BitDecompose { 32, 0 }` range-checks) or a slice of a model's `tokens` tensor.
///
/// # Errors
/// [`IrError::Declaration`] if `class_bits` is not in `1..=64` or `prompt_tokens.is_some()`
/// disagrees with `prompt_len > 0`; propagates builder [`IrError`]s.
pub fn build_sampler_stream_class(
    g: &mut GraphBuilder,
    midstate: TensorId,
    masks: TensorId,
    class_bits: u8,
    prompt_len: u32,
    prompt_tokens: Option<TensorId>,
    detach_committed_source: bool,
) -> Result<StreamClassInputs, IrError> {
    if !(1..=64).contains(&class_bits) || prompt_tokens.is_some() != (prompt_len > 0) {
        return Err(IrError::Declaration {
            rule: "stream class hash takes 1 ≤ b ≤ 64 and prompt_tokens iff prompt_len > 0",
        });
    }
    let layout = sampler_class_layout(prompt_len);
    let decls = sha256_inputs(&layout);
    let message = g.input(decls[0].clone())?;
    let (sha, place) = if let Some(mut tokens) = prompt_tokens {
        if detach_committed_source {
            let one_const = g.constant(vec![1])?;
            tokens = one(
                g,
                Op::ConstMul { c: one_const },
                &[tokens],
                virtual_decl(&[prompt_len]),
            )?;
        }
        let token_bits = g.node(
            Op::BitDecompose { low: 32, high: 0 },
            &[tokens],
            vec![bit_decl(&[prompt_len, 32], ADVICE)],
        )?[0];
        let byte_bits = one(
            g,
            Op::Reshape,
            &[token_bits],
            bit_decl(&[4 * prompt_len, 8], NodeKind::Virtual),
        )?;
        let planes = one(
            g,
            Op::Transpose { perm: vec![1, 0] },
            &[byte_bits],
            bit_decl(&[8, 4 * prompt_len], NodeKind::Virtual),
        )?;
        let flat_bytes = bits_to_bytes(g, planes, 4 * prompt_len)?;
        let byte_col = one(
            g,
            Op::Reshape,
            &[flat_bytes],
            byte_decl(&[4 * prompt_len, 1], NodeKind::Virtual),
        )?;
        let place_in = g.input(decls[4].clone())?;
        let sha = sha256(g, &layout, &[message, midstate, masks, byte_col, place_in])?;
        (sha, Some(place_in))
    } else {
        let sha = sha256(g, &layout, &[message, midstate, masks])?;
        (sha, None)
    };

    let words = sha.chaining.last().ok_or(IrError::Declaration {
        rule: "at least one class block",
    })?;
    let packed_two = pack_words(g, &words[..2])?;
    // Map packed_two (`[64]` in SHA-256 word layout for words 0 and 1) to `val_bits` (`[64]` where
    // index `p` is bit `p` of `u64_le(h[0..8])`), shifted right by `64 - b`, arranged as `[8, 8]`
    // (`row = out_bit`, `col = out_byte` of `u64_le(class)`).
    let shift = usize::from(64 - class_bits);
    let b_usize = usize::from(class_bits);
    let gather_idx = g.constant(
        (0..8usize)
            .flat_map(|out_bit| {
                (0..8usize).map(move |out_byte| {
                    let q = 8 * out_byte + out_bit;
                    let p = (q + shift).min(63);
                    let byte_idx = p / 8;
                    let bit_in_byte = p % 8;
                    let src = if byte_idx < 4 {
                        8 * (3 - byte_idx) + bit_in_byte
                    } else {
                        32 + 8 * (7 - byte_idx) + bit_in_byte
                    };
                    src as i64
                })
            })
            .collect(),
    )?;
    let shifted = one(
        g,
        Op::GatherPublic { index: gather_idx },
        &[packed_two],
        virtual_decl(&[64]),
    )?;
    let mask = g.constant(
        (0..8usize)
            .flat_map(|out_bit| {
                (0..8usize).map(move |out_byte| i64::from(8 * out_byte + out_bit < b_usize))
            })
            .collect(),
    )?;
    let masked = one(g, Op::ConstMul { c: mask }, &[shifted], virtual_decl(&[64]))?;
    let bit_planes = one(g, Op::Reshape, &[masked], virtual_decl(&[8, 8]))?;
    let class_bytes = bits_to_bytes(g, bit_planes, 8)?;
    Ok(StreamClassInputs {
        message,
        place,
        class_bytes,
    })
}

/// Encrypts `num_draws` blocks `u64_le(class) ‖ u32_le(t) ‖ 0^4` under `round_keys` (`class_bytes`
/// `None` when `b = 0`, `Some([8])` when `b ≥ 1`) and returns `(position_blocks, draws)` where
/// `draws` is `[num_draws]` virtual `u_t` with exact analyzed range `[0, 2^32 − 1]`.
///
/// # Errors
/// [`IrError::Declaration`] if `num_draws == 0`; propagates builder [`IrError`]s.
pub fn build_sampler_draws(
    g: &mut GraphBuilder,
    tables: &AesTables,
    round_keys: &[TensorId; 11],
    class_bytes: Option<TensorId>,
    num_draws: u32,
) -> Result<(TensorId, TensorId), IrError> {
    if num_draws == 0 {
        return Err(IrError::Declaration {
            rule: "sampler_draws needs at least one output position",
        });
    }
    let position_blocks = g.input(byte_decl(&[16, num_draws], NodeKind::Public))?;
    let blocks = if let Some(class_bytes) = class_bytes {
        let pad_idx = g.constant((0..16).map(|r| if r < 8 { r } else { 0 }).collect())?;
        let gathered = one(
            g,
            Op::GatherPublic { index: pad_idx },
            &[class_bytes],
            virtual_decl(&[16]),
        )?;
        let mask = g.constant((0..16).map(|r| i64::from(r < 8)).collect())?;
        let padded_class = one(
            g,
            Op::ConstMul { c: mask },
            &[gathered],
            virtual_decl(&[16]),
        )?;
        let class_plane = one(
            g,
            Op::Expand {
                axis: 1,
                len: num_draws,
            },
            &[padded_class],
            virtual_decl(&[16, num_draws]),
        )?;
        one(
            g,
            Op::Bitwise { table: tables.xor },
            &[class_plane, position_blocks],
            byte_decl(
                &[16, num_draws],
                NodeKind::LookupOutput { table: tables.xor },
            ),
        )?
    } else {
        position_blocks
    };

    let cipher = aes128_encrypt(g, tables, round_keys, blocks, num_draws)?;
    let low4 = one(
        g,
        Op::Slice {
            axis: 0,
            start: 0,
            len: 4,
        },
        &[cipher],
        virtual_decl(&[4, num_draws]),
    )?;
    let even_idx = g.constant(vec![0, 2])?;
    let odd_idx = g.constant(vec![1, 3])?;
    let even = one(
        g,
        Op::GatherPublic { index: even_idx },
        &[low4],
        virtual_decl(&[2, num_draws]),
    )?;
    let odd = one(
        g,
        Op::GatherPublic { index: odd_idx },
        &[low4],
        virtual_decl(&[2, num_draws]),
    )?;
    let c256 = g.constant(vec![256])?;
    let odd_scaled = one(
        g,
        Op::ConstMul { c: c256 },
        &[odd],
        virtual_decl(&[2, num_draws]),
    )?;
    let u16_pairs = one(
        g,
        Op::Add,
        &[even, odd_scaled],
        virtual_decl(&[2, num_draws]),
    )?;
    let lo16 = one(
        g,
        Op::Slice {
            axis: 0,
            start: 0,
            len: 1,
        },
        &[u16_pairs],
        virtual_decl(&[1, num_draws]),
    )?;
    let hi16 = one(
        g,
        Op::Slice {
            axis: 0,
            start: 1,
            len: 1,
        },
        &[u16_pairs],
        virtual_decl(&[1, num_draws]),
    )?;
    let c65536 = g.constant(vec![65_536])?;
    let hi16_scaled = one(
        g,
        Op::ConstMul { c: c65536 },
        &[hi16],
        virtual_decl(&[1, num_draws]),
    )?;
    let u32_row = one(
        g,
        Op::Add,
        &[lo16, hi16_scaled],
        virtual_decl(&[1, num_draws]),
    )?;
    let draws = one(g, Op::Reshape, &[u32_row], virtual_decl(&[num_draws]))?;
    Ok((position_blocks, draws))
}

/// Complete standalone PRF circuit from a committed 16-byte key `K` (range-checked by
/// `BitDecompose { 8, 0 }`), optional prompt tokens (`b ≥ 1`), and `num_draws` positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandaloneSamplerPrf {
    /// Committed `[16]` advice key `K`.
    pub key: TensorId,
    /// Public `[10, 4]` AES round constants.
    pub rcon: TensorId,
    /// Optional SHA-256 inputs when `class_bits ≥ 1`: `(iv, masks, key_block_msg, key_block_place,
    /// class_inputs, prompt_tokens)`.
    pub class_stage: Option<(
        TensorId,
        TensorId,
        TensorId,
        TensorId,
        StreamClassInputs,
        Option<TensorId>,
    )>,
    /// Public `[16, num_draws]` AES counter-suffix blocks.
    pub position_blocks: TensorId,
    /// Virtual `[num_draws]` output draws `u_t ∈ [0, 2^32 − 1]`.
    pub draws: TensorId,
}

/// Builds a complete standalone Sampler PRF graph from committed `K`, passing [`crate::analyze`]
/// and [`crate::derive_graph_schedule`].
///
/// # Errors
/// [`IrError::Declaration`] if `class_bits > 64` or `num_draws == 0`; propagates builder
/// [`IrError`]s.
pub fn build_standalone_sampler_prf(
    g: &mut GraphBuilder,
    tables: &AesTables,
    class_bits: u8,
    prompt_len: u32,
    num_draws: u32,
) -> Result<StandaloneSamplerPrf, IrError> {
    if class_bits > 64 {
        return Err(IrError::Declaration {
            rule: "class_bits is at most 64 (RFC-0061)",
        });
    }
    let key = g.input(byte_decl(&[16], ADVICE))?;
    // Range-check `key` (`[16]` committed bytes) in `[0, 255]` so `analyze` proves all 16 bytes.
    let _ = g.node(
        Op::BitDecompose { low: 8, high: 0 },
        &[key],
        vec![bit_decl(&[16, 8], ADVICE)],
    )?;
    let rcon = g.input(byte_decl(&[10, 4], NodeKind::Public))?;
    let round_keys = aes128_key_schedule(g, tables, key, rcon)?;

    let (class_stage, class_bytes) = if class_bits == 0 {
        (None, None)
    } else {
        let iv = g.input(bit_decl(&[256, 1], NodeKind::Public))?;
        let masks = g.input(bit_decl(&[2, 32], NodeKind::Public))?;
        let (kb_msg, kb_place, midstate) = build_sampler_key_block(g, key, iv, masks)?;
        let prompt_tokens = (prompt_len > 0)
            .then(|| {
                g.input(TensorDecl {
                    shape: vec![prompt_len],
                    ty: U32_TY,
                    range: IntRange {
                        lo: 0,
                        hi: (1 << 32) - 1,
                    },
                    kind: ADVICE,
                })
            })
            .transpose()?;
        let class_in = build_sampler_stream_class(
            g,
            midstate,
            masks,
            class_bits,
            prompt_len,
            prompt_tokens,
            false,
        )?;
        let cb = class_in.class_bytes;
        (
            Some((iv, masks, kb_msg, kb_place, class_in, prompt_tokens)),
            Some(cb),
        )
    };

    let (position_blocks, draws) =
        build_sampler_draws(g, tables, &round_keys, class_bytes, num_draws)?;
    Ok(StandaloneSamplerPrf {
        key,
        rcon,
        class_stage,
        position_blocks,
        draws,
    })
}

/// Complete end-to-end pinned Sampler PRF circuit: base-key commitment opening + pin-key
/// derivation + optional stream-class hash (`b ≥ 1`) + AES-128 draws.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PinnedSamplerPrf {
    /// Commitment opening and pin-key derivation subgraph.
    pub derive: SamplerDeriveSubgraph,
    /// Public `[10, 4]` AES round constants.
    pub rcon: TensorId,
    /// Optional stream-class hash inputs when `class_bits ≥ 1`: `(key_block_msg, key_block_place,
    /// class_inputs, prompt_tokens)`.
    pub class_stage: Option<(TensorId, TensorId, StreamClassInputs, Option<TensorId>)>,
    /// Public `[16, num_draws]` AES counter-suffix blocks.
    pub position_blocks: TensorId,
    /// Virtual `[num_draws]` output draws `u_t ∈ [0, 2^32 − 1]`.
    pub draws: TensorId,
}

/// Builds the complete end-to-end pinned Sampler PRF circuit (`RFC-0061` 'The circuits' steps 1–4),
/// passing [`crate::analyze`] and [`crate::derive_graph_schedule`].
///
/// # Errors
/// [`IrError::Declaration`] if `class_bits > 64` or `num_draws == 0`; propagates builder
/// [`IrError`]s.
pub fn build_pinned_sampler_prf(
    g: &mut GraphBuilder,
    tables: &AesTables,
    class_bits: u8,
    prompt_len: u32,
    num_draws: u32,
) -> Result<PinnedSamplerPrf, IrError> {
    if class_bits > 64 {
        return Err(IrError::Declaration {
            rule: "class_bits is at most 64 (RFC-0061)",
        });
    }
    let derive = build_sampler_derive(g)?;
    let rcon = g.input(byte_decl(&[10, 4], NodeKind::Public))?;
    let round_keys = aes128_key_schedule(g, tables, derive.derived_key, rcon)?;

    let (class_stage, class_bytes) = if class_bits == 0 {
        (None, None)
    } else {
        let (kb_msg, kb_place, midstate) =
            build_sampler_key_block(g, derive.derived_key, derive.iv, derive.masks)?;
        let prompt_tokens = (prompt_len > 0)
            .then(|| {
                g.input(TensorDecl {
                    shape: vec![prompt_len],
                    ty: U32_TY,
                    range: IntRange {
                        lo: 0,
                        hi: (1 << 32) - 1,
                    },
                    kind: ADVICE,
                })
            })
            .transpose()?;
        let class_in = build_sampler_stream_class(
            g,
            midstate,
            derive.masks,
            class_bits,
            prompt_len,
            prompt_tokens,
            false,
        )?;
        let cb = class_in.class_bytes;
        (Some((kb_msg, kb_place, class_in, prompt_tokens)), Some(cb))
    };

    let (position_blocks, draws) =
        build_sampler_draws(g, tables, &round_keys, class_bytes, num_draws)?;
    Ok(PinnedSamplerPrf {
        derive,
        rcon,
        class_stage,
        position_blocks,
        draws,
    })
}
