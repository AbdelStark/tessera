// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `TableSpec` and its parts (RFC-0008 'Table specifications'), decoded through `tv-codec`'s
//! cursor and encoded for the digests (RFC-0003). Their TCE layout is tier A: a kind's tag is
//! its declaration index, so kinds are appended, never reordered.

use tv_codec::{Cursor, DecodeError, ErrorCode, Rational, size};
use tv_hash::Digest;

/// A table's identifier: a dense index the compiler assigns (INV-DM-5), and the order of the
/// claim list's tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableId(pub u32);

/// A table as the claim list names it: its definition, its row count and its digest.
///
/// Every field is the prover's word until [`Table::from_spec`](crate::Table::from_spec)
/// regenerates the table from `id` and `kind` and compares the other two.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TableSpec {
    /// The table's identifier.
    pub id: TableId,
    /// The table's kind and parameters.
    pub kind: TableKind,
    /// `⌈log2 T⌉` for the kind's `T` rows, at most 32.
    pub log2_rows: u8,
    /// `H("table", TCE(spec without digest) ‖ TCE(pieces) ‖ TCE(core))`.
    pub digest: Digest,
}

/// A table kind and its parameters (RFC-0008 'Table specifications'), in declaration order,
/// which is the order of the TCE tags.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TableKind {
    /// Tag 0: the keys `[0, 2^bits)`, no value.
    Range {
        /// The key width.
        bits: u8,
    },
    /// Tag 1: `rq(k, r_c, lo, hi)` of RFC-0005 for each class `c` and key
    /// `k ∈ [−2^{b−1}, 2^{b−1})`.
    Requant {
        /// The key width.
        b: u8,
        /// The shift `r_c` of each class.
        r_by_class: Vec<u8>,
        /// The clamp's lower bound.
        lo: i32,
        /// The clamp's upper bound.
        hi: i32,
    },
    /// Tag 2: `2^i` for `i ∈ [0, max_exp)`.
    Pow2 {
        /// The number of rows.
        max_exp: u8,
    },
    /// Tag 3: a bitwise operation on two bytes.
    Bitwise8 {
        /// The operation.
        op: BitOp,
    },
    /// Tag 4: the AES S-box (FIPS-197 §5.1.1).
    AesSbox,
    /// Tag 5: multiplication by `c` in GF(2^8) (FIPS-197 §4.2).
    Gf256Mul {
        /// The constant factor.
        c: u8,
    },
    /// Tag 6: `round(2^σ · e^{−k·scale})` for `k ∈ [0, 2^key_bits)`.
    Exp {
        /// The key width.
        key_bits: u8,
        /// `σ`, the output's fractional bits.
        sigma: u8,
        /// The scale of a key.
        scale: Rational,
    },
    /// Tag 7: `⌊2^out_bits · 2^{f / 2^frac_bits}⌋` from below, for `f ∈ [0, 2^frac_bits)`.
    Exp2Frac {
        /// The key width.
        frac_bits: u8,
        /// The output's fractional bits.
        out_bits: u8,
    },
    /// Tag 8: `clamp(round(silu(h · 2^{in_exp}) / 2^{out_exp}), −128, 127)` with `h = k >> r_c`,
    /// for each class `c` and key `k ∈ [−2^{b−1}, 2^{b−1})`.
    Silu {
        /// The key width.
        b: u8,
        /// The shift `r_c` of each class.
        r_by_class: Vec<u8>,
        /// The input's exponent.
        in_exp: i8,
        /// The output's exponent.
        out_exp: i8,
    },
    /// Tag 9: the RoPE constants `(C, S)` of every position and frequency, Llama-3 scaled; not a
    /// lookup table.
    Rope {
        /// The base `θ`.
        theta: u32,
        /// Llama-3's `factor`.
        factor: u16,
        /// Llama-3's `low_freq_factor`.
        low_freq_factor: u16,
        /// Llama-3's `high_freq_factor`.
        high_freq_factor: u16,
        /// Llama-3's `original_max_position_embeddings`, `L`.
        original_max: u32,
        /// The constants' fractional bits.
        bits: u8,
        /// The number of positions.
        max_context: u32,
        /// The head dimension; it has `d_head / 2` frequencies.
        d_head: u16,
    },
}

/// The operation of a [`TableKind::Bitwise8`] table, in declaration order: `Xor` is tag 0 and
/// `And` tag 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BitOp {
    /// `a ⊕ b`.
    Xor,
    /// `a ∧ b`.
    And,
}

/// The number of [`TableKind`] variants: tags from 0 to 9.
const KINDS: u16 = 10;

/// The number of [`BitOp`] variants.
const BIT_OPS: u16 = 2;

impl TableSpec {
    /// The smallest encoding, `min_size(TableSpec)` in RFC-0003 'Limits': the id, the tag of a
    /// kind without fields, `log2_rows` and the digest, 39 bytes.
    pub const MIN_SIZE: u64 = size::sum(&[size::U32, TableKind::MIN_SIZE, size::U8, size::DIGEST]);

    /// Decodes a `TableSpec`, a struct one level deeper, with the cursor's errors: a kind or
    /// operation tag that names no variant, and a scale that is not in lowest terms, are
    /// `ERR-CODEC-NONCANONICAL`. The classes of `Requant` and `Silu` are copied into memory
    /// reserved fallibly: memory that cannot be reserved is `ERR-RES-MEMORY` at their length.
    /// Nothing here checks the parameters, `log2_rows` or the digest:
    /// [`Table::from_spec`](crate::Table::from_spec) does.
    pub fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        c.nested(|c| {
            let id = TableId(c.u32()?);
            let kind = TableKind::decode(c)?;
            let log2_rows = c.u8()?;
            let digest = c.digest()?;
            Ok(Self {
                id,
                kind,
                log2_rows,
                digest,
            })
        })
    }

    /// `TCE(self)`: the fields in declaration order (RFC-0003 'Composite types').
    ///
    /// The encoding is as long as the specification, in memory that is not reserved fallibly.
    /// For a specification that [`Table::from_spec`](crate::Table::from_spec) accepts, with at
    /// most four classes, that is at most 60 bytes; a specification decoded from a claim list
    /// may hold classes up to the claim list's limits until `from_spec` has refused it, so encode
    /// one only once `from_spec` has accepted it.
    pub fn to_tce(&self) -> Vec<u8> {
        let mut out = Vec::new();
        encode_head(self.id, &self.kind, self.log2_rows, &mut out);
        out.extend_from_slice(&self.digest);
        out
    }
}

impl TableKind {
    /// The smallest encoding of a kind: the tag of `AesSbox`, which has no field.
    pub const MIN_SIZE: u64 = size::TAG;

    /// Decodes a kind: a `u16` tag, then the variant's fields.
    pub fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        let at = c.offset();
        c.enumeration(KINDS, |c, tag| match tag {
            0 => Ok(TableKind::Range { bits: c.u8()? }),
            1 => {
                let b = c.u8()?;
                let r_by_class = classes(c)?;
                let lo = c.i32()?;
                let hi = c.i32()?;
                Ok(TableKind::Requant {
                    b,
                    r_by_class,
                    lo,
                    hi,
                })
            }
            2 => Ok(TableKind::Pow2 { max_exp: c.u8()? }),
            3 => Ok(TableKind::Bitwise8 {
                op: BitOp::decode(c)?,
            }),
            4 => Ok(TableKind::AesSbox),
            5 => Ok(TableKind::Gf256Mul { c: c.u8()? }),
            6 => {
                let key_bits = c.u8()?;
                let sigma = c.u8()?;
                let scale = c.rational()?;
                Ok(TableKind::Exp {
                    key_bits,
                    sigma,
                    scale,
                })
            }
            7 => {
                let frac_bits = c.u8()?;
                let out_bits = c.u8()?;
                Ok(TableKind::Exp2Frac {
                    frac_bits,
                    out_bits,
                })
            }
            8 => {
                let b = c.u8()?;
                let r_by_class = classes(c)?;
                let in_exp = c.i8()?;
                let out_exp = c.i8()?;
                Ok(TableKind::Silu {
                    b,
                    r_by_class,
                    in_exp,
                    out_exp,
                })
            }
            9 => {
                let theta = c.u32()?;
                let factor = c.u16()?;
                let low_freq_factor = c.u16()?;
                let high_freq_factor = c.u16()?;
                let original_max = c.u32()?;
                let bits = c.u8()?;
                let max_context = c.u32()?;
                let d_head = c.u16()?;
                Ok(TableKind::Rope {
                    theta,
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max,
                    bits,
                    max_context,
                    d_head,
                })
            }
            // `enumeration` has refused every tag of `KINDS` or more.
            _ => Err(DecodeError::new(ErrorCode::Internal, at)),
        })
    }

    /// The variant's TCE tag, its declaration index.
    pub fn tag(&self) -> u16 {
        match self {
            TableKind::Range { .. } => 0,
            TableKind::Requant { .. } => 1,
            TableKind::Pow2 { .. } => 2,
            TableKind::Bitwise8 { .. } => 3,
            TableKind::AesSbox => 4,
            TableKind::Gf256Mul { .. } => 5,
            TableKind::Exp { .. } => 6,
            TableKind::Exp2Frac { .. } => 7,
            TableKind::Silu { .. } => 8,
            TableKind::Rope { .. } => 9,
        }
    }

    /// Appends `TCE(self)`: the tag, then the variant's fields in declaration order. Its length
    /// and memory are those of [`TableSpec::to_tce`]: encode a kind from a claim list only once
    /// its specification has been accepted.
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.tag().to_le_bytes());
        match self {
            TableKind::Range { bits } => out.push(*bits),
            TableKind::Requant {
                b,
                r_by_class,
                lo,
                hi,
            } => {
                out.push(*b);
                encode_bytes(r_by_class, out);
                out.extend_from_slice(&lo.to_le_bytes());
                out.extend_from_slice(&hi.to_le_bytes());
            }
            TableKind::Pow2 { max_exp } => out.push(*max_exp),
            TableKind::Bitwise8 { op } => out.extend_from_slice(&op.tag().to_le_bytes()),
            TableKind::AesSbox => {}
            TableKind::Gf256Mul { c } => out.push(*c),
            TableKind::Exp {
                key_bits,
                sigma,
                scale,
            } => {
                out.push(*key_bits);
                out.push(*sigma);
                out.extend_from_slice(&scale.num().to_le_bytes());
                out.extend_from_slice(&scale.den().to_le_bytes());
            }
            TableKind::Exp2Frac {
                frac_bits,
                out_bits,
            } => {
                out.push(*frac_bits);
                out.push(*out_bits);
            }
            TableKind::Silu {
                b,
                r_by_class,
                in_exp,
                out_exp,
            } => {
                out.push(*b);
                encode_bytes(r_by_class, out);
                out.extend_from_slice(&in_exp.to_le_bytes());
                out.extend_from_slice(&out_exp.to_le_bytes());
            }
            TableKind::Rope {
                theta,
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max,
                bits,
                max_context,
                d_head,
            } => {
                out.extend_from_slice(&theta.to_le_bytes());
                out.extend_from_slice(&factor.to_le_bytes());
                out.extend_from_slice(&low_freq_factor.to_le_bytes());
                out.extend_from_slice(&high_freq_factor.to_le_bytes());
                out.extend_from_slice(&original_max.to_le_bytes());
                out.push(*bits);
                out.extend_from_slice(&max_context.to_le_bytes());
                out.extend_from_slice(&d_head.to_le_bytes());
            }
        }
    }
}

impl BitOp {
    /// Decodes an operation: a `u16` tag, 0 or 1.
    pub fn decode(c: &mut Cursor<'_>) -> Result<Self, DecodeError> {
        let at = c.offset();
        c.enumeration(BIT_OPS, |_, tag| match tag {
            0 => Ok(BitOp::Xor),
            1 => Ok(BitOp::And),
            // `enumeration` has refused every tag of `BIT_OPS` or more.
            _ => Err(DecodeError::new(ErrorCode::Internal, at)),
        })
    }

    /// The variant's TCE tag, its declaration index.
    pub fn tag(self) -> u16 {
        match self {
            BitOp::Xor => 0,
            BitOp::And => 1,
        }
    }
}

/// Appends `TCE(id) ‖ TCE(kind) ‖ TCE(log2_rows)`: a specification without its digest, the
/// first part of what the digest hashes (RFC-0008 'Generators and digests').
pub(crate) fn encode_head(id: TableId, kind: &TableKind, log2_rows: u8, out: &mut Vec<u8>) {
    out.extend_from_slice(&id.0.to_le_bytes());
    kind.encode(out);
    out.push(log2_rows);
}

/// Decodes `r_by_class`, a `Vec<u8>`, and copies it out of the input. The cursor checks its length
/// against the limits and the bytes left before anything is read (RFC-0003 'Limits'); the copy's
/// memory is then reserved fallibly, and memory that cannot be reserved is `ERR-RES-MEMORY` at the
/// length, as for every sequence `tv-codec` decodes (RFC-0019 'Robustness'). A hostile claim list
/// may hold classes up to its limits: [`Table::from_spec`](crate::Table::from_spec) reads their
/// number first, from the length alone, and refuses more than four before reading any class.
fn classes(c: &mut Cursor<'_>) -> Result<Vec<u8>, DecodeError> {
    let at = c.offset();
    let bytes = c.bytes()?;
    let mut copy = Vec::new();
    copy.try_reserve_exact(bytes.len())
        .map_err(|_| DecodeError::new(ErrorCode::Memory, at))?;
    copy.extend_from_slice(bytes);
    Ok(copy)
}

/// Appends `TCE(Vec<u8>)`: the length as a `u64`, then the bytes.
fn encode_bytes(bytes: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&length(bytes.len()).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// A length as `u64`. Every supported target has pointers of at most 64 bits, so this is exact.
pub(crate) fn length(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}
