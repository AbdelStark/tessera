// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The weight commitment's public artifact, `WeightCommitmentPublic`, schema `0x0301`
//! (RFC-0040), and the weight root `R_W` it hashes to.
//!
//! # Specification
//!
//! Written from RFC-0040 'WeightCommitmentPublic' and RFC-0003 before the code:
//!
//! - The payload is a struct of three fields, in this order: `layout`, the claim list's
//!   `WeightLayoutParams` (RFC-0026), itself a struct of `log2_k: u8`, `log2_inv_rate: u8`,
//!   `mask_coeffs: u32` and `blinding_rows: u32`; `roots: [Digest; 10]`, one per matrix in
//!   `WeightMatrix` order; `epoch: u32`. That is 334 bytes, and every 334 bytes are one value.
//! - Depth: the payload's struct is one level, `layout` a second, `roots` a second and each
//!   digest a third, since an array is a composite (RFC-0003 'Limits').
//! - Errors, each at the offset of the first byte its check reads: a field cut short is
//!   `ERR-CODEC-TRUNCATED` at the field's first byte, `roots` checked as a whole, so at offset 10
//!   for any cut inside it; a nesting deeper than `max_depth` is `ERR-CODEC-LIMIT` at the
//!   composite's first byte; bytes after `epoch` are `ERR-CODEC-NONCANONICAL` at the first of them.
//! - `R_W = H("weights-root", TCE(WeightCommitmentPublic))`, BLAKE3 `derive_key` with the context
//!   [`WEIGHTS_ROOT_CONTEXT`] over the value's canonical encoding (RFC-0040, RFC-0004). The
//!   artifact holds exactly what `R_W` binds: the layout, the ten roots and the epoch (INV-WC-3).
//! - The verifier reads this artifact, never `WeightCommitmentState` (`0x0302`), which is secret.

use crate::{Cursor, DecodeError, ErrorCode, Expected, size};

/// The BLAKE3 `derive_key` context of `R_W`: the `weights-root` domain of RFC-0004.
pub const WEIGHTS_ROOT_CONTEXT: &str = "tessera/v1/weights-root";

/// The weight commitment's layout parameters (RFC-0026 'Schema'): the parameter set's values,
/// which the claim list states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightLayoutParams {
    /// `log2 k_W`.
    pub log2_k: u8,
    /// `log2` of the inverse rate: 2, rate 1/4.
    pub log2_inv_rate: u8,
    /// `P_W`, each row's mask coefficients.
    pub mask_coeffs: u32,
    /// `B_W`, each matrix's blinding rows.
    pub blinding_rows: u32,
}

/// The weight commitment's public artifact (RFC-0040): what `R_W` binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightCommitmentPublic {
    /// The claim list's weight layout.
    pub layout: WeightLayoutParams,
    /// Each matrix's root, `Embed`, `Q`, `K`, `V`, `O`, `Gate`, `Up`, `Down`, `LmHead`,
    /// `ChannelParams`.
    pub roots: [[u8; 32]; 10],
    /// The commitment's epoch: 0 for the first.
    pub epoch: u32,
}

/// The encoding's length: 1 + 1 + 4 + 4 bytes of layout, ten roots of 32 bytes, a 4-byte
/// epoch.
const ENCODED_LEN: usize = 334;

impl WeightCommitmentPublic {
    /// What its reader expects: schema `0x0301`, whose writers emit version 1 (03 §5).
    pub const EXPECTED: Expected = Expected {
        schema_id: 0x0301,
        version: 1,
    };

    /// Decodes one value at `cursor`, as the specification above lays it out.
    ///
    /// # Errors
    ///
    /// `ERR-CODEC-TRUNCATED` for a field cut short and `ERR-CODEC-LIMIT` for a nesting beyond the
    /// limits, at the offsets the specification gives.
    pub fn decode(cursor: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        cursor.nested(|c| {
            let layout = c.nested(|c| {
                Ok(WeightLayoutParams {
                    log2_k: c.u8()?,
                    log2_inv_rate: c.u8()?,
                    mask_coeffs: c.u32()?,
                    blinding_rows: c.u32()?,
                })
            })?;
            let at = c.offset();
            let roots: [[u8; 32]; 10] = c
                .array(10, size::DIGEST, Cursor::digest)?
                .try_into()
                .map_err(|_| DecodeError::new(ErrorCode::Internal, at))?;
            let epoch = c.u32()?;
            Ok(Self {
                layout,
                roots,
                epoch,
            })
        })
    }

    /// The canonical encoding: the fields in order, little-endian, the roots as their bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(ENCODED_LEN);
        out.push(self.layout.log2_k);
        out.push(self.layout.log2_inv_rate);
        out.extend_from_slice(&self.layout.mask_coeffs.to_le_bytes());
        out.extend_from_slice(&self.layout.blinding_rows.to_le_bytes());
        for root in &self.roots {
            out.extend_from_slice(root);
        }
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out
    }

    /// `R_W`: `derive_key` hashes the bytes it is given, the value's encoding, with the context
    /// [`WEIGHTS_ROOT_CONTEXT`]; the verifier's hashing crate supplies it.
    pub fn weights_root(&self, derive_key: impl FnOnce(&[u8]) -> [u8; 32]) -> [u8; 32] {
        derive_key(&self.encode())
    }
}
