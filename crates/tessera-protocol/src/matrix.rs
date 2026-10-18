// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Weight matrix ids (RFC-0036 'Matrix ids'), and the decoding of the sequences they key.
//!
//! A message names a weight matrix by a `u8`, the declaration index of RFC-0031's
//! `WeightMatrix`. Three sequences are keyed by it: `Round2Response.weight_combined`, whose
//! `matrix` strictly increases, and `ColumnQueries.weights` and `Round3Response.weight_columns`,
//! which follow its order (RFC-0036 'Matrix ids'). Like a map's keys (RFC-0003 'Composite
//! types'), each entry's id is read and checked before the rest of the entry: an id that names
//! no matrix, or that is not above the previous entry's, is `ERR-CODEC-NONCANONICAL` at the id's
//! byte, the entry's first.
//!
//! The entry types, `WeightCombinedRows`, `MatrixQueries` and `MatrixColumns`, have no `Decode`
//! of their own, so `Vec<T>`'s generic decoder, which would skip the order, cannot take them:
//! an entry decodes only inside its message, through [`decode_by_matrix`].

use core::fmt;

use tessera_codec::{Decode, DecodeError, Reader};

/// The weight matrices, in the declaration order of RFC-0031's `WeightMatrix`, which is
/// RFC-0012's table order (RFC-0036 'Matrix ids').
///
/// A message names a matrix by its declaration index, [`id`](Self::id), and a transcript label
/// by its name, [`name`](Self::name): `pcs/weights/Q/gamma`, `pcs/weights/ChannelParams/xi`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WeightMatrix {
    /// The transposed embedding.
    Embed,
    /// The query projection.
    Q,
    /// The key projection.
    K,
    /// The value projection.
    V,
    /// The output projection.
    O,
    /// The MLP gate projection.
    Gate,
    /// The MLP up projection.
    Up,
    /// The MLP down projection.
    Down,
    /// The vocabulary head.
    LmHead,
    /// The channel parameters: hidden shift classes and affine multipliers (RFC-0012 'Layout').
    ChannelParams,
}

impl WeightMatrix {
    /// Every matrix in declaration order: `ALL[id]` is the matrix of id `id`.
    pub const ALL: [Self; 10] = [
        Self::Embed,
        Self::Q,
        Self::K,
        Self::V,
        Self::O,
        Self::Gate,
        Self::Up,
        Self::Down,
        Self::LmHead,
        Self::ChannelParams,
    ];

    /// The id a message names the matrix by: its declaration index.
    pub const fn id(self) -> u8 {
        self as u8
    }

    /// The matrix of id `id`, or `None` when no matrix has it.
    pub const fn from_id(id: u8) -> Option<Self> {
        if (id as usize) < Self::ALL.len() {
            Some(Self::ALL[id as usize])
        } else {
            None
        }
    }

    /// The variant's name as RFC-0031 writes it, a label's `{T}`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embed => "Embed",
            Self::Q => "Q",
            Self::K => "K",
            Self::V => "V",
            Self::O => "O",
            Self::Gate => "Gate",
            Self::Up => "Up",
            Self::Down => "Down",
            Self::LmHead => "LmHead",
            Self::ChannelParams => "ChannelParams",
        }
    }
}

/// The name, as a label spells it.
impl fmt::Display for WeightMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// An entry of a sequence keyed by matrix id, which starts with the id.
pub(crate) trait ByMatrix: Sized {
    /// The fewest bytes an entry encodes to, which the sequence's length check multiplies.
    const MIN_SIZE: usize;

    /// Decodes one entry whose id must be above `previous`.
    fn decode_after(reader: &mut Reader<'_>, previous: Option<u8>) -> Result<Self, DecodeError>;

    /// The entry's matrix id.
    fn matrix(&self) -> u8;
}

/// Reads a matrix id: `ERR-CODEC-NONCANONICAL` at its byte unless it names a [`WeightMatrix`]
/// and is above `previous`.
pub(crate) fn read_matrix(
    reader: &mut Reader<'_>,
    previous: Option<u8>,
) -> Result<u8, DecodeError> {
    let at = reader.position();
    let id = u8::decode_from(reader)?;
    if WeightMatrix::from_id(id).is_none() || previous.is_some_and(|p| id <= p) {
        return Err(reader.noncanonical(at));
    }
    Ok(id)
}

/// Decodes a `Vec<T>` of entries keyed by matrix id, the ids strictly increasing. The length is
/// checked as any sequence's (RFC-0003 'Limits'), and each entry's id before the rest of it.
pub(crate) fn decode_by_matrix<T: ByMatrix>(
    reader: &mut Reader<'_>,
) -> Result<Vec<T>, DecodeError> {
    reader.enter()?;
    let n = reader.sequence_len(<T as ByMatrix>::MIN_SIZE)?;
    // Never more memory than the input could fill, as the codec's own sequences reserve.
    let mut entries = Vec::with_capacity(n.min(reader.remaining() / size_of::<T>().max(1)));
    let mut previous = None;
    for _ in 0..n {
        let entry = T::decode_after(reader, previous)?;
        previous = Some(entry.matrix());
        entries.push(entry);
    }
    reader.leave();
    Ok(entries)
}
