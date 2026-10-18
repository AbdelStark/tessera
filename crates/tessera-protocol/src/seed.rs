// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The coin toss of a live session (RFC-0036 'Seed messages', RFC-0014 'Session order').
//!
//! The verifier commits to a fresh 32-byte share `s_V` before it sees anything of the prover's,
//! the prover sends a fresh share `s_P` with `Round1`, and the verifier then reveals `s_V`. The
//! seed `H("coin-toss", s_V ‖ s_P)` is uniform if either party is honest, and fixed only after
//! every commitment (INV-PROTO-2). Each message names its session, so neither side can take one
//! session's share for another's.

use tessera_codec::{Decode, Encode};
use tessera_hash::{Digest, Domain, Hasher, hash};

/// Step 0: the verifier's commitment to its share (schema `0x0501`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct SeedCommit {
    /// The session, which the verifier draws fresh from its CSPRNG for every session
    /// (INV-PROTO-4).
    pub session_id: [u8; 16],
    /// `c_V = H("coin-commit", s_V)`: [`coin_commitment`] of the share the verifier reveals.
    pub commitment: Digest,
}

/// Step 1: the prover's share, sent beside `Round1` (schema `0x0502`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct ProverSeedShare {
    /// The session, `SeedCommit`'s.
    pub session_id: [u8; 16],
    /// `s_P`, which the prover draws fresh for every `Round1`.
    pub share: [u8; 32],
}

/// Step 2: the verifier's share, revealed once `Round1` and the prover's share are in
/// (schema `0x0503`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct SeedReveal {
    /// The session, `SeedCommit`'s.
    pub session_id: [u8; 16],
    /// `s_V`, which opens `SeedCommit.commitment`.
    pub share: [u8; 32],
}

/// `c_V = H("coin-commit", s_V)`, the verifier's commitment to its share (RFC-0004's
/// `coin-commit` domain, over the share's 32 bytes).
pub fn coin_commitment(verifier_share: &[u8; 32]) -> Digest {
    hash(Domain::CoinCommit, verifier_share)
}

/// The live seed, `H("coin-toss", s_V ‖ s_P)` (RFC-0004's `coin-toss` domain, over the two
/// shares' 64 bytes). Transcript row 3 absorbs its 32 bytes under `seed` (RFC-0036 'What the
/// transcript absorbs').
pub fn coin_toss(verifier_share: &[u8; 32], prover_share: &[u8; 32]) -> Digest {
    Hasher::new(Domain::CoinToss)
        .update(verifier_share)
        .update(prover_share)
        .finalize()
}
