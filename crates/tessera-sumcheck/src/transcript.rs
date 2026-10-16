// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What a sumcheck needs from the Fiat–Shamir transcript (RFC-0004 'Transcript').

use tessera_codec::{Encode, Sink};
use tessera_field::GoldilocksExt2;
use tessera_hash::{Transcript, TranscriptHash};

/// The two transcript operations a sumcheck prover performs, over its field `F`: absorbing a
/// message of field elements and squeezing a challenge, each under a label.
///
/// [`Transcript`] implements it over `GoldilocksExt2`, the field of every challenge. The
/// soundness-calibration tests implement it over `F65537Ext2`, where cheating succeeds often
/// enough to be measured (07 §4).
pub trait SumcheckTranscript<F> {
    /// Absorbs `message` under `label` as the TCE encoding of a `Vec<F>`: its length as a
    /// little-endian `u64`, then each element's encoding.
    fn absorb_message(&mut self, label: &str, message: &[F]);

    /// A challenge exactly uniform in `F`, squeezed under `label`.
    fn challenge(&mut self, label: &str) -> F;
}

/// RFC-0004's transcript, with challenges in `Fp2` ([`Transcript::challenge_fp2`]).
impl<H: TranscriptHash> SumcheckTranscript<GoldilocksExt2> for Transcript<H> {
    fn absorb_message(&mut self, label: &str, message: &[GoldilocksExt2]) {
        self.absorb(label, &Elements(message));
    }

    fn challenge(&mut self, label: &str) -> GoldilocksExt2 {
        self.challenge_fp2(label)
    }
}

/// A slice encoded as the `Vec` it would be: the `u64` length, then the elements. It lets a
/// message be absorbed without copying it into a `Vec`.
struct Elements<'a, T>(&'a [T]);

impl<T: Encode> Encode for Elements<'_, T> {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        (self.0.len() as u64).encode_to(out);
        T::encode_slice(self.0, out);
    }
}
