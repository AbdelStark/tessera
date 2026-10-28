// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Transcript replay (RFC-0004 'Transcript'): the verifier's side of the Fiat–Shamir transcript,
//! with the proof-of-work check of RFC-0014 'Grinding'.
//!
//! The state is a digest and a squeeze counter. The session context starts it; each prover
//! message is absorbed as TCE bytes under a label; each challenge is squeezed from it, and every
//! squeeze ratchets it forward:
//!
//! ```text
//! new:     state = H("transcript-init", TCE(SessionContext)),   counter = 0
//! absorb:  state = H("transcript-absorb", state ‖ Ascii(label) ‖ TCE(value))
//! squeeze: out   = H("transcript-squeeze", state ‖ Ascii(label) ‖ u64(counter)),
//!          counter += 1,  state = H("transcript-ratchet", state ‖ out)
//! pow:     H("pow", state ‖ Ascii(label) ‖ u64(nonce)) has `bits` leading zero bits
//! ```
//!
//! A label is TCE `Ascii`, its `u64` length then its bytes, so no two label and value pairs give
//! the same input. Labels come from the tier-A patterns of RFC-0014 'Transcript schedule'; this
//! module checks only that they are printable ASCII, when an [`Ascii`] is made.
//!
//! ```
//! use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
//!
//! let context = SessionContext {
//!     protocol: Ascii::new("tessera-protocol/1")?,
//!     mode: Mode::NonInteractive,
//!     params: ParamSetId(Ascii::new("ni-100/1")?),
//!     statement_digest: [0x5a; 32],
//!     session_id: [7; 16],
//! };
//! let mut transcript = Transcript::new(&context);
//! transcript.absorb(Ascii::new("round1")?, &0u64.to_le_bytes());
//! // A nonce with 0 leading zero bits required always passes, and is absorbed.
//! let mut transcript = transcript.check_grind(Ascii::new("seed-pow")?, 0, 0)?;
//! let seed = transcript.squeeze(Ascii::new("seed")?);
//! assert_eq!(transcript.counter(), 1);
//! # let _ = seed;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use core::fmt;

use tv_field::{Fp, Fp2};

use crate::Digest;
use crate::domain::{Domain, DomainHasher, hash_pair};

/// The most squeezes one challenge may take before [`TranscriptError::Exhausted`].
///
/// RFC-0004 repeats a squeeze until its words fall below `P`, and RFC-0019 admits no unbounded
/// loop, so RFC-0004 'Failure modes' stops the verifier after 16. A word is at or above `P` with
/// probability below `2^-32`, so `challenge_fp2` needs a seventeenth squeeze with probability
/// below `2^-496`, and `challenge_fp` below `2^-2048`; no prover reaches that without breaking
/// BLAKE3.
pub const MAX_CHALLENGE_SQUEEZES: u32 = 16;

/// Printable ASCII text, TCE `Ascii` (RFC-0003 'Primitive types'): every byte in `0x20..=0x7E`.
/// Labels and the text fields of [`SessionContext`] are `Ascii`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ascii<'a>(&'a str);

impl<'a> Ascii<'a> {
    /// `text` as `Ascii`, or the position of its first byte outside `0x20..=0x7E`.
    pub const fn new(text: &'a str) -> Result<Self, NotAscii> {
        let mut rest = text.as_bytes();
        let mut position: usize = 0;
        while let [byte, tail @ ..] = rest {
            if *byte < 0x20 || *byte > 0x7E {
                return Err(NotAscii { position });
            }
            position = position.saturating_add(1);
            rest = tail;
        }
        Ok(Self(text))
    }

    /// The text.
    pub const fn as_str(self) -> &'a str {
        self.0
    }

    /// Writes the TCE encoding: the `u64` length, little-endian, then the bytes.
    fn encode(self, sink: &mut impl Sink) {
        let length = u64::try_from(self.0.len()).unwrap_or(u64::MAX);
        sink.put(&length.to_le_bytes());
        sink.put(self.0.as_bytes());
    }
}

/// Text that is not printable ASCII: the position of its first byte outside `0x20..=0x7E`.
///
/// Labels and session-context text are built by the verifier from public data, so this is a
/// fault of the verifier, not of a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NotAscii {
    position: usize,
}

impl NotAscii {
    /// The position of the first byte outside `0x20..=0x7E`.
    pub const fn position(self) -> usize {
        self.position
    }
}

impl fmt::Display for NotAscii {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "byte {} is not printable ASCII", self.position)
    }
}

impl std::error::Error for NotAscii {}

/// How the session runs (RFC-0004 'Transcript', RFC-0014). TCE encodes it as a `u16` tag, the
/// variant's position: `Live` is 0, `NonInteractive` 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// A bilateral session: the seed comes from a coin toss and the verifier draws the column
    /// queries.
    Live,
    /// A transferable proof: the seed and the queries are squeezed behind proofs of work.
    NonInteractive,
}

impl Mode {
    /// The TCE tag.
    pub const fn tag(self) -> u16 {
        match self {
            Mode::Live => 0,
            Mode::NonInteractive => 1,
        }
    }
}

/// A parameter set's name and version, such as `ni-100/1` (RFC-0014 'Parameter sets'). TCE
/// encodes it as its one field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParamSetId<'a>(pub Ascii<'a>);

/// What every transcript of a session is bound to (RFC-0004 'Transcript').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SessionContext<'a> {
    /// The protocol name and version, such as `tessera-protocol/1`.
    pub protocol: Ascii<'a>,
    /// Live or non-interactive.
    pub mode: Mode,
    /// The parameter set.
    pub params: ParamSetId<'a>,
    /// `H("statement", TCE(PublicStatement))`.
    pub statement_digest: Digest,
    /// The session's identifier.
    pub session_id: [u8; 16],
}

impl SessionContext<'_> {
    /// The TCE encoding: the fields in declaration order, `protocol` and `params` as `Ascii`,
    /// `mode` as its `u16` tag, then the 32 bytes of the digest and the 16 of the session id.
    pub fn to_tce(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }

    fn encode(&self, sink: &mut impl Sink) {
        self.protocol.encode(sink);
        sink.put(&self.mode.tag().to_le_bytes());
        self.params.0.encode(sink);
        sink.put(&self.statement_digest);
        sink.put(&self.session_id);
    }
}

/// Where TCE bytes go: a hash input, or a buffer.
trait Sink {
    fn put(&mut self, bytes: &[u8]);
}

impl Sink for DomainHasher {
    fn put(&mut self, bytes: &[u8]) {
        self.update(bytes);
    }
}

impl Sink for Vec<u8> {
    fn put(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

/// A challenge the transcript could not give.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TranscriptError {
    /// `challenge_indices` was asked for indices below `2^log_n` with `log_n` above 64.
    LogN {
        /// The `log_n` asked for.
        log_n: u32,
    },
    /// A challenge took [`MAX_CHALLENGE_SQUEEZES`] squeezes without one whose words fall below
    /// `P` as RFC-0004 requires. It does not happen without a break of BLAKE3; a verifier
    /// reports it as `Inconclusive(ERR-INTERNAL)`, never as a verdict on the proof (RFC-0004
    /// 'Failure modes'), and does not use the transcript again: it is those squeezes ahead.
    Exhausted,
    /// Memory for the indices could not be reserved (`ERR-RES-MEMORY`).
    Memory,
}

impl fmt::Display for TranscriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranscriptError::LogN { log_n } => {
                write!(f, "indices below 2^{log_n} asked for; 64 bits is the most")
            }
            TranscriptError::Exhausted => write!(
                f,
                "no challenge after {MAX_CHALLENGE_SQUEEZES} squeezes of the transcript"
            ),
            TranscriptError::Memory => f.write_str("no memory for the challenge indices"),
        }
    }
}

impl std::error::Error for TranscriptError {}

/// A proof-of-work nonce without enough leading zero bits (`Reject(ProofOfWork)`, RFC-0004).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProofOfWorkFailed {
    bits: u8,
    zeros: u32,
}

impl ProofOfWorkFailed {
    /// The leading zero bits the proof of work required.
    pub const fn required(self) -> u8 {
        self.bits
    }

    /// The leading zero bits the nonce's hash had.
    pub const fn found(self) -> u32 {
        self.zeros
    }
}

impl fmt::Display for ProofOfWorkFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the proof-of-work hash has {} leading zero bits, {} are required",
            self.zeros, self.bits
        )
    }
}

impl std::error::Error for ProofOfWorkFailed {}

/// The verifier's copy of the Fiat–Shamir transcript.
///
/// It is not `Clone`: a transcript is replayed once, in protocol order, and a failed proof of
/// work ends it ([`check_grind`](Self::check_grind) takes it by value).
#[derive(Debug, PartialEq, Eq)]
pub struct Transcript {
    state: Digest,
    counter: u64,
}

impl Transcript {
    /// The transcript of a session: `state = H("transcript-init", TCE(context))`, no squeeze yet.
    pub fn new(context: &SessionContext<'_>) -> Self {
        let mut hasher = DomainHasher::new(Domain::TranscriptInit);
        context.encode(&mut hasher);
        Self {
            state: hasher.finalize(),
            counter: 0,
        }
    }

    /// The current state, a public digest.
    pub fn state(&self) -> &Digest {
        &self.state
    }

    /// The number of squeezes so far.
    pub fn counter(&self) -> u64 {
        self.counter
    }

    /// The hash of `state ‖ Ascii(label) ‖ tail` in `domain`.
    fn hash_with(&self, domain: Domain, label: Ascii<'_>, tail: &[u8]) -> Digest {
        let mut hasher = DomainHasher::new(domain);
        hasher.update(&self.state);
        label.encode(&mut hasher);
        hasher.update(tail);
        hasher.finalize()
    }

    /// Absorbs a prover message, given as its TCE bytes `tce`, under `label`:
    /// `state = H("transcript-absorb", state ‖ Ascii(label) ‖ tce)`.
    pub fn absorb(&mut self, label: Ascii<'_>, tce: &[u8]) {
        match self.absorb_with_cancel(label, tce, &mut || Ok::<(), core::convert::Infallible>(())) {
            Ok(()) => {}
            Err(never) => match never {},
        }
    }

    /// Absorbs the same single message with cooperative cancellation (RFC-0019, SP-8).
    /// Checks before work, before each 64 KiB label/message chunk and before committing
    /// the new state. The counter is unchanged, including on cancellation (INV-FS-1..3).
    ///
    /// # Errors
    /// Returns the callback's error unchanged. On error, state and counter are unchanged;
    /// no challenge may depend on the unabsorbed message.
    pub fn absorb_with_cancel<E>(
        &mut self,
        label: Ascii<'_>,
        tce: &[u8],
        check: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        check()?;
        let mut hasher = DomainHasher::new(Domain::TranscriptAbsorb);
        hasher.update(&self.state);
        let length = u64::try_from(label.as_str().len()).unwrap_or(u64::MAX);
        hasher.update(&length.to_le_bytes());
        for bytes in [label.as_str().as_bytes(), tce] {
            for block in bytes.chunks(65536) {
                check()?;
                hasher.update(block);
            }
        }
        let state = hasher.finalize();
        check()?;
        self.state = state;
        Ok(())
    }

    /// Squeezes 32 bytes under `label`:
    /// `out = H("transcript-squeeze", state ‖ Ascii(label) ‖ u64(counter))`, then the counter
    /// moves on and `state = H("transcript-ratchet", state ‖ out)`.
    pub fn squeeze(&mut self, label: Ascii<'_>) -> Digest {
        let out = self.hash_with(
            Domain::TranscriptSqueeze,
            label,
            &self.counter.to_le_bytes(),
        );
        // A transcript of 2^64 squeezes is out of reach.
        self.counter = self.counter.wrapping_add(1);
        self.state = hash_pair(Domain::TranscriptRatchet, &self.state, &out);
        out
    }

    /// A challenge exactly uniform in `Fp`: squeezes under `label` until a squeeze has a word
    /// below `P` among its four 8-byte little-endian words, and returns the first such word;
    /// [`TranscriptError::Exhausted`] after [`MAX_CHALLENGE_SQUEEZES`] squeezes without one.
    pub fn challenge_fp(&mut self, label: Ascii<'_>) -> Result<Fp, TranscriptError> {
        draw(|| self.squeeze(label), first_canonical_word)
    }

    /// A challenge exactly uniform in `Fp2`: squeezes under `label` until the first two words of
    /// a squeeze are both below `P`, and returns them as `(c0, c1)`;
    /// [`TranscriptError::Exhausted`] after [`MAX_CHALLENGE_SQUEEZES`] squeezes without them.
    pub fn challenge_fp2(&mut self, label: Ascii<'_>) -> Result<Fp2, TranscriptError> {
        draw(|| self.squeeze(label), leading_pair)
    }

    /// `count` indices exactly uniform below `2^log_n`, drawn with replacement: each squeezed
    /// word, in order, four per squeeze, masked to its low `log_n` bits; the words of the last
    /// squeeze that are not needed are dropped, and `count = 0` squeezes nothing.
    ///
    /// `log_n` above 64 is [`TranscriptError::LogN`] and memory that cannot be reserved for the
    /// indices [`TranscriptError::Memory`], both before any squeeze.
    pub fn challenge_indices(
        &mut self,
        label: Ascii<'_>,
        log_n: u32,
        count: u32,
    ) -> Result<Vec<u64>, TranscriptError> {
        let mask = match log_n {
            64 => u64::MAX,
            0..64 => (1u64 << log_n).wrapping_sub(1),
            _ => return Err(TranscriptError::LogN { log_n }),
        };
        let count = usize::try_from(count).map_err(|_| TranscriptError::Memory)?;
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(count)
            .map_err(|_| TranscriptError::Memory)?;
        while indices.len() < count {
            let need = count.saturating_sub(indices.len());
            let out = self.squeeze(label);
            indices.extend(words(&out).into_iter().take(need).map(|word| word & mask));
        }
        Ok(indices)
    }

    /// Checks a proof-of-work nonce under `label`: `H("pow", state ‖ Ascii(label) ‖ u64(nonce))`
    /// must have at least `bits` leading zero bits, the most significant bit of byte 0 first.
    ///
    /// A valid nonce is absorbed under `label`, as a `u64`, and the transcript continues; any
    /// valid nonce is accepted, not only the smallest. An invalid one ends the transcript: it is
    /// consumed, and the verifier rejects the proof (`Reject(ProofOfWork)`) without using it
    /// again.
    pub fn check_grind(
        mut self,
        label: Ascii<'_>,
        bits: u8,
        nonce: u64,
    ) -> Result<Self, ProofOfWorkFailed> {
        let attempt = self.hash_with(Domain::Pow, label, &nonce.to_le_bytes());
        let zeros = leading_zero_bits(&attempt);
        if zeros < u32::from(bits) {
            return Err(ProofOfWorkFailed { bits, zeros });
        }
        self.absorb(label, &nonce.to_le_bytes());
        Ok(self)
    }
}

/// The first value `pick` finds in the squeezes `squeeze` makes, one after another, or
/// [`TranscriptError::Exhausted`] after [`MAX_CHALLENGE_SQUEEZES`] of them.
fn draw<T>(
    mut squeeze: impl FnMut() -> Digest,
    pick: impl Fn(&Digest) -> Option<T>,
) -> Result<T, TranscriptError> {
    for _ in 0..MAX_CHALLENGE_SQUEEZES {
        if let Some(value) = pick(&squeeze()) {
            return Ok(value);
        }
    }
    Err(TranscriptError::Exhausted)
}

/// The four 8-byte little-endian words of a squeeze.
fn words(out: &Digest) -> [u64; 4] {
    let mut words = [0u64; 4];
    for (word, bytes) in words.iter_mut().zip(out.chunks_exact(8)) {
        let mut le = [0u8; 8];
        for (slot, byte) in le.iter_mut().zip(bytes) {
            *slot = *byte;
        }
        *word = u64::from_le_bytes(le);
    }
    words
}

/// The first word of a squeeze below `P`, if any.
fn first_canonical_word(out: &Digest) -> Option<Fp> {
    words(out).into_iter().find_map(Fp::from_canonical)
}

/// The first two words of a squeeze as `(c0, c1)`, if both are below `P`.
fn leading_pair(out: &Digest) -> Option<Fp2> {
    let [c0, c1, _, _] = words(out);
    Some(Fp2::new(Fp::from_canonical(c0)?, Fp::from_canonical(c1)?))
}

/// The zero bits before the first 1 bit, the most significant bit of byte 0 first.
fn leading_zero_bits(digest: &Digest) -> u32 {
    let mut zeros: u32 = 0;
    for byte in digest {
        zeros = zeros.saturating_add(byte.leading_zeros());
        if *byte != 0 {
            break;
        }
    }
    zeros
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = Fp::MODULUS;

    fn digest_of(words: [u64; 4]) -> Digest {
        let mut out = [0u8; 32];
        for (slot, word) in out.chunks_exact_mut(8).zip(words) {
            slot.copy_from_slice(&word.to_le_bytes());
        }
        out
    }

    fn started_transcript() -> Transcript {
        let mut transcript = Transcript::new(&SessionContext {
            protocol: Ascii::new("tessera-protocol/1").unwrap(),
            mode: Mode::Live,
            params: ParamSetId(Ascii::new("live-64/1").unwrap()),
            statement_digest: [5; 32],
            session_id: [7; 16],
        });
        transcript.squeeze(Ascii::new("before").unwrap());
        transcript
    }

    #[test]
    fn cancellable_absorption_preserves_framing_state_and_challenges() {
        for label_len in [0usize, 1, 65535, 65536, 65537, 3 * 65536] {
            let text = "x".repeat(label_len);
            let label = Ascii::new(&text).unwrap();
            for value_len in [0usize, 1, 65535, 65536, 65537, 3 * 65536] {
                let value = vec![17; value_len];
                let mut actual = started_transcript();
                // The unchanged whole-input hash spells RFC-0004's framing independently
                // of the chunked absorption method and its compatibility wrapper.
                let mut expected = Transcript {
                    state: actual.hash_with(Domain::TranscriptAbsorb, label, &value),
                    counter: actual.counter,
                };
                let mut polls = 0;
                actual
                    .absorb_with_cancel(label, &value, &mut || {
                        polls += 1;
                        Ok::<(), u32>(())
                    })
                    .unwrap();
                assert_eq!(
                    polls,
                    2 + label_len.div_ceil(65536) + value_len.div_ceil(65536)
                );
                assert_eq!(actual, expected);
                let challenge = Ascii::new("after").unwrap();
                assert_eq!(
                    actual.challenge_fp2(challenge),
                    expected.challenge_fp2(challenge)
                );
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn absorption_cancellation_at_every_poll_is_atomic_and_preserves_the_error() {
        for (label_len, value_len) in [(0usize, 0usize), (7, 3 * 65536), (3 * 65536, 7)] {
            let text = "x".repeat(label_len);
            let label = Ascii::new(&text).unwrap();
            let value = vec![17; value_len];
            let checks = 2 + label_len.div_ceil(65536) + value_len.div_ceil(65536);
            for stop in 1..=checks {
                let mut actual = started_transcript();
                let before_state = actual.state;
                let before_counter = actual.counter;
                let mut polls = 0;
                assert_eq!(
                    actual.absorb_with_cancel(label, &value, &mut || {
                        polls += 1;
                        if polls == stop { Err(stop) } else { Ok(()) }
                    }),
                    Err(stop)
                );
                assert_eq!(polls, stop);
                assert_eq!(actual.state, before_state);
                assert_eq!(actual.counter, before_counter);

                let expected = actual.hash_with(Domain::TranscriptAbsorb, label, &value);
                actual
                    .absorb_with_cancel(label, &value, &mut || Ok::<(), usize>(()))
                    .unwrap();
                assert_eq!(actual.state, expected);
                assert_eq!(actual.counter, before_counter);
            }
        }
    }

    /// The selection rules on crafted squeezes: RFC-0004 'Testing Strategy' asks for words at or
    /// above `P` in every position, which BLAKE3 outputs reach only with probability 2^-32.
    #[test]
    fn challenge_words_skip_every_value_at_or_above_p() {
        let canonical = |x: u64| Fp::from_canonical(x).unwrap();
        assert_eq!(
            first_canonical_word(&digest_of([5, 6, 7, 8])),
            Some(canonical(5))
        );
        assert_eq!(
            first_canonical_word(&digest_of([P, u64::MAX, P - 1, 0])),
            Some(canonical(P - 1))
        );
        assert_eq!(
            first_canonical_word(&digest_of([P, P + 1, u64::MAX, 0])),
            Some(canonical(0))
        );
        assert_eq!(first_canonical_word(&digest_of([P, P, P, u64::MAX])), None);

        assert_eq!(
            leading_pair(&digest_of([P - 1, 0, P, P])),
            Some(Fp2::new(canonical(P - 1), canonical(0)))
        );
        assert_eq!(leading_pair(&digest_of([P, 0, 1, 2])), None);
        assert_eq!(leading_pair(&digest_of([0, P, 1, 2])), None);
        assert_eq!(leading_pair(&digest_of([u64::MAX, u64::MAX, 1, 2])), None);
    }

    /// `draw` on `rejected` squeezes then `accepted` ones: its result, and how many squeezes it
    /// made.
    fn drawn<T>(
        rejected: usize,
        reject: Digest,
        accept: Digest,
        pick: impl Fn(&Digest) -> Option<T>,
    ) -> (Result<T, TranscriptError>, usize) {
        let mut made = 0;
        let outcome = draw(
            || {
                made += 1;
                if made <= rejected { reject } else { accept }
            },
            pick,
        );
        (outcome, made)
    }

    /// RFC-0004 'Testing Strategy' and 'Failure modes': a stub squeeze source drives both
    /// challenges to the bound, which BLAKE3 reaches with probability below 2^-496. Fifteen
    /// rejected squeezes still give the sixteenth's value; sixteen are `Exhausted`, after exactly
    /// sixteen squeezes.
    #[test]
    fn a_challenge_takes_at_most_sixteen_squeezes() {
        let canonical = |x: u64| Fp::from_canonical(x).unwrap();
        assert_eq!(MAX_CHALLENGE_SQUEEZES, 16);
        let fp_accept = digest_of([P, u64::MAX, 9, 0]);
        let fp_reject = digest_of([P, P + 1, u64::MAX, P]);
        for (rejected, expected) in [
            (0, (Ok(canonical(9)), 1)),
            (1, (Ok(canonical(9)), 2)),
            (15, (Ok(canonical(9)), 16)),
            (16, (Err(TranscriptError::Exhausted), 16)),
            (1000, (Err(TranscriptError::Exhausted), 16)),
        ] {
            assert_eq!(
                drawn(rejected, fp_reject, fp_accept, first_canonical_word),
                expected,
                "Fp, {rejected} rejected"
            );
        }
        let fp2_accept = digest_of([1, P - 1, P, P]);
        let pair = Fp2::new(canonical(1), canonical(P - 1));
        for fp2_reject in [digest_of([P, 0, 0, 0]), digest_of([0, P, 0, 0])] {
            for (rejected, expected) in [
                (0, (Ok(pair), 1)),
                (15, (Ok(pair), 16)),
                (16, (Err(TranscriptError::Exhausted), 16)),
            ] {
                assert_eq!(
                    drawn(rejected, fp2_reject, fp2_accept, leading_pair),
                    expected,
                    "Fp2, {rejected} rejected"
                );
            }
        }
    }

    #[test]
    fn leading_zero_bits_count_from_the_top_of_byte_zero() {
        let mut digest = [0u8; 32];
        assert_eq!(leading_zero_bits(&digest), 256);
        digest[31] = 1;
        assert_eq!(leading_zero_bits(&digest), 255);
        digest[1] = 0x10;
        assert_eq!(leading_zero_bits(&digest), 11);
        digest[0] = 0x80;
        assert_eq!(leading_zero_bits(&digest), 0);
    }

    #[test]
    fn ascii_admits_printable_bytes_only() {
        assert!(Ascii::new("").is_ok());
        assert!(Ascii::new(" ~tree/0/r/12").is_ok());
        assert_eq!(Ascii::new("a\u{7f}b").unwrap_err().position(), 1);
        assert_eq!(Ascii::new("ab\n").unwrap_err().position(), 2);
        assert_eq!(Ascii::new("\u{1f}").unwrap_err().position(), 0);
        assert_eq!(Ascii::new("é").unwrap_err().position(), 0);
    }
}
