// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `WeightCommitmentPublic`, schema `0x0301` (RFC-0040 'WeightCommitmentPublic'): what `R_W`
//! binds, and nothing else.

use std::io;
use std::path::Path;

use tessera_codec::{Decode, Encode, encode, envelope, schema};
use tessera_hash::{Digest, Domain, hash};

use crate::{WeightLayoutParams, WeightMatrix};

/// The weight commitment's public artifact: schema `0x0301`, version 1, public, `.twr`
/// (RFC-0040, 03 §5).
///
/// `R_W = H("weights-root", TCE(WeightCommitmentPublic))`. TCE writes a struct's fields in order
/// and a fixed array without a length, so this is RFC-0012's
/// `H("weights-root", TCE(WeightLayoutParams) ‖ root_Embed ‖ … ‖ root_ChannelParams ‖
/// u32(epoch))` byte for byte: the artifact holds exactly what `R_W` binds (INV-WC-3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct WeightCommitmentPublic {
    /// The claim list's weight layout (RFC-0026).
    pub layout: WeightLayoutParams,
    /// `root_T` of each matrix, in [`WeightMatrix`] order.
    pub roots: [Digest; 10],
    /// The epoch: 0 at the first commitment, `e + 1` after a refresh.
    pub epoch: u32,
}

impl WeightCommitmentPublic {
    /// `R_W = H("weights-root", TCE(self))` (RFC-0040, INV-WC-3).
    pub fn weights_root(&self) -> Digest {
        hash(Domain::WeightsRoot, &encode(self))
    }

    /// `root_T` of `matrix`.
    pub fn root(&self, matrix: WeightMatrix) -> Digest {
        self.roots[matrix.index()]
    }

    /// Writes the artifact to `path`, a `.twr` file, by atomic rename and with the default mode
    /// (RFC-0040 'Writing'), and returns its envelope digest.
    ///
    /// # Errors
    ///
    /// Those of [`envelope::write_file`]; `path` then holds what it held before.
    pub fn write(&self, path: &Path) -> io::Result<envelope::Digest> {
        envelope::write_file(path, &schema::WEIGHT_COMMITMENT_PUBLIC, self)
    }
}
