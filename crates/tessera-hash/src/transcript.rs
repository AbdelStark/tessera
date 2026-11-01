// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The Fiat–Shamir transcript (RFC-0004 'Transcript'): a chaining value that absorbs labeled
//! prover messages as TCE bytes and squeezes challenges exactly uniform in `Fp`, `Fp2` and
//! power-of-two index ranges, with proof-of-work grinding.
//!
//! Every step hashes the current state, the step's label as TCE `Ascii` (its length, then its
//! bytes), and the step's own data, each in its own domain:
//!
//! | Step | Input | New state |
//! |---|---|---|
//! | [`Transcript::new`] | `TCE(SessionContext)` | the digest |
//! | [`absorb`](Transcript::absorb) | `state ‖ Ascii(label) ‖ TCE(value)` | the digest |
//! | [`squeeze`](Transcript::squeeze) | `state ‖ Ascii(label) ‖ counter: u64`, giving `out` | `H("transcript-ratchet", state ‖ out)` |
//! | proof of work | `state ‖ Ascii(label) ‖ nonce: u64` | unchanged; the nonce is then absorbed |

use tessera_codec::{Ascii, Decode, Encode, Sink};
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};

use crate::{Digest, Domain, Hasher};

/// Whether the parties interact (RFC-0014 'Session order'): in a live session the verifier
/// contributes the seed; in a non-interactive proof the prover grinds for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub enum Mode {
    /// The verifier answers each round.
    Live,
    /// The prover derives every challenge from the transcript alone.
    NonInteractive,
}

/// A parameter set's identity: its name and version as RFC-0014 'Parameter sets' writes them,
/// such as `ni-100/1`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct ParamSetId(pub Ascii);

/// What a transcript binds before its first message (RFC-0004 'Transcript'): two sessions that
/// differ in any field draw independent challenges.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct SessionContext {
    /// The protocol version string, `tessera-protocol/1`.
    pub protocol: Ascii,
    /// Live or non-interactive.
    pub mode: Mode,
    /// The parameter set the proof is checked under.
    pub params: ParamSetId,
    /// The digest of the public statement (RFC-0007).
    pub statement_digest: Digest,
    /// Fresh per session (RFC-0014, INV-PROTO-4).
    pub session_id: [u8; 16],
}

/// The hash a transcript draws on: `H(domain, ·)` of RFC-0004, fed incrementally.
///
/// Production transcripts use [`Blake3`]. The trait lets a test substitute outputs, for example
/// words above the field modulus, to drive every rejection branch of the challenge samplers,
/// which BLAKE3 reaches with probability about 2^-32 per word.
pub trait TranscriptHash {
    /// An input being hashed.
    type Input: Sink;

    /// An empty input in `domain`.
    fn input(&self, domain: Domain) -> Self::Input;

    /// The digest of `input`.
    fn digest(&self, input: Self::Input) -> Digest;
}

/// BLAKE3 in key-derivation mode, the hash of every other domain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Blake3;

impl TranscriptHash for Blake3 {
    type Input = Hasher;

    fn input(&self, domain: Domain) -> Hasher {
        Hasher::new(domain)
    }

    fn digest(&self, input: Hasher) -> Digest {
        input.finalize()
    }
}

/// The Fiat–Shamir transcript: a 32-byte state and a squeeze counter.
///
/// It is deliberately not `Clone`: a copy that later diverges from the original is how state
/// restoration is attempted, and an honest party has one transcript per session.
///
/// Labels are the printable ASCII strings of RFC-0014 'Transcript schedule'. Prover and verifier
/// must absorb the same messages under the same labels, in the same order (INV-FS-1, INV-FS-4).
pub struct Transcript<H: TranscriptHash = Blake3> {
    hash: H,
    state: Digest,
    counter: u64,
}

/// Shows the state and the counter, which derive from public data only.
impl<H: TranscriptHash> core::fmt::Debug for Transcript<H> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Transcript")
            .field("state", &self.state)
            .field("counter", &self.counter)
            .finish_non_exhaustive()
    }
}

impl Transcript {
    /// Maximum simultaneous heap requests made by the production transcript's
    /// absorption or proof-of-work buffer (RFC-0004/0018), for labels no longer
    /// than `max_pow_label_bytes`. Borrowed messages/context and caller-created
    /// labels are separate. Does not allocate; unaddressable lengths return `None`.
    pub fn work_memory_bytes(max_pow_label_bytes: u64) -> Option<u64> {
        let pow = 32u64
            .checked_add(8)?
            .checked_add(max_pow_label_bytes)?
            .checked_add(8)?;
        if usize::try_from(pow).is_err() || pow > isize::MAX as u64 {
            return None;
        }
        // Each staged append needs fewer than STAGE bytes. Rust1.95 Vec growth
        // doubles at most: old + replacement requests remain below 3 * STAGE.
        Some(pow.max(3 * STAGE as u64))
    }

    /// A transcript over BLAKE3: `state = H("transcript-init", TCE(context))`.
    pub fn new(context: &SessionContext) -> Self {
        Self::with_hash(context, Blake3)
    }
}

impl<H: TranscriptHash> Transcript<H> {
    /// A transcript over `hash`: `state = H("transcript-init", TCE(context))`.
    pub fn with_hash(context: &SessionContext, hash: H) -> Self {
        let mut input = hash.input(Domain::TranscriptInit);
        context.encode_to(&mut input);
        let state = hash.digest(input);
        Self {
            hash,
            state,
            counter: 0,
        }
    }

    /// The current state.
    pub fn state(&self) -> Digest {
        self.state
    }

    /// How many squeezes the transcript has made.
    pub fn counter(&self) -> u64 {
        self.counter
    }

    /// An input in `domain` that starts `state ‖ Ascii(label)`.
    fn begin(&self, domain: Domain, label: &str) -> H::Input {
        let mut input = self.hash.input(domain);
        input.put(&self.state);
        input.put(&label_prefix(label));
        input.put(label.as_bytes());
        input
    }

    /// Absorbs `value`'s TCE encoding under `label`:
    /// `state = H("transcript-absorb", state ‖ Ascii(label) ‖ TCE(value))`. The encoding streams
    /// into the hash, a long message's small writes collected into large updates, so that no
    /// message is held whole and the hash works on several chunks at once.
    pub fn absorb<T: Encode + ?Sized>(&mut self, label: &str, value: &T) {
        let mut input = self.begin(Domain::TranscriptAbsorb, label);
        let mut staged = Staged::new(&mut input);
        value.encode_to(&mut staged);
        staged.finish();
        self.state = self.hash.digest(input);
    }

    /// 32 bytes under `label`: `out = H("transcript-squeeze", state ‖ Ascii(label) ‖ counter)`,
    /// then `counter += 1` and `state = H("transcript-ratchet", state ‖ out)`.
    pub fn squeeze(&mut self, label: &str) -> Digest {
        let mut input = self.begin(Domain::TranscriptSqueeze, label);
        self.counter.encode_to(&mut input);
        let out = self.hash.digest(input);
        self.counter = self
            .counter
            .checked_add(1)
            .expect("a transcript makes fewer than 2^64 squeezes");
        let mut ratchet = self.hash.input(Domain::TranscriptRatchet);
        ratchet.put(&self.state);
        ratchet.put(&out);
        self.state = self.hash.digest(ratchet);
        out
    }

    /// A challenge exactly uniform in `Fp`: the first of a squeeze's four little-endian words
    /// below `P`, squeezing again if none is. A word is rejected with probability below 2^-32.
    pub fn challenge_fp(&mut self, label: &str) -> Goldilocks {
        loop {
            let found = words(&self.squeeze(label))
                .into_iter()
                .find_map(Goldilocks::from_canonical_u64);
            if let Some(x) = found {
                return x;
            }
        }
    }

    /// A challenge exactly uniform in `Fp2`: a squeeze's first two words as `(c0, c1)`, squeezing
    /// again until both are below `P`.
    pub fn challenge_fp2(&mut self, label: &str) -> GoldilocksExt2 {
        loop {
            let [c0, c1, ..] = words(&self.squeeze(label));
            if let (Some(c0), Some(c1)) = (
                Goldilocks::from_canonical_u64(c0),
                Goldilocks::from_canonical_u64(c1),
            ) {
                return GoldilocksExt2::new(c0, c1);
            }
        }
    }

    /// `count` indices exactly uniform in `0..2^log_n`, drawn with replacement: squeezed words
    /// masked to their low `log_n` bits, four per squeeze, in order. No squeeze is made for
    /// `count = 0`.
    ///
    /// # Panics
    ///
    /// If `log_n > 64`: indices are `u64`, so no domain is larger than `2^64`.
    pub fn challenge_indices(&mut self, label: &str, log_n: u32, count: u32) -> Vec<u64> {
        assert!(log_n <= 64, "an index domain of 2^{log_n} exceeds u64");
        let mask = u64::MAX.checked_shr(64 - log_n).unwrap_or(0);
        let count = usize::try_from(count).expect("a u32 count fits a usize");
        let mut indices = Vec::with_capacity(count);
        while indices.len() < count {
            let still = count - indices.len();
            let out = self.squeeze(label);
            indices.extend(words(&out).into_iter().take(still).map(|word| word & mask));
        }
        indices
    }

    /// The prover's proof of work: the smallest `nonce` whose `H("pow", state ‖ Ascii(label) ‖
    /// nonce)` starts with `bits` zero bits (most significant bit of byte 0 first), which it then
    /// absorbs under `label`. It takes about `2^bits` hashes.
    pub fn grind(&mut self, label: &str, bits: u8) -> u64 {
        let mut attempt = self.pow_input(label);
        let nonce = (0..=u64::MAX)
            .find(|&nonce| self.pow_meets(&mut attempt, bits, nonce))
            .expect("a nonce below 2^64 meets any requirement the parameter sets make");
        self.absorb(label, &nonce);
        nonce
    }

    /// The verifier's proof-of-work check: whether `nonce` meets the `bits` requirement under
    /// `label`, with one hash. Any such nonce is accepted, not only the smallest: soundness
    /// already charges the prover for every attempt.
    ///
    /// A valid nonce is absorbed as [`grind`](Self::grind) absorbs it; an invalid one is not
    /// absorbed. On `false` the verifier rejects the proof (`Reject(ProofOfWork)`) and the
    /// transcript is not used again: a prover that skipped the work would otherwise simply
    /// continue from the unchanged state.
    #[must_use]
    pub fn check_grind(&mut self, label: &str, bits: u8, nonce: u64) -> bool {
        let valid = self.pow_meets(&mut self.pow_input(label), bits, nonce);
        if valid {
            self.absorb(label, &nonce);
        }
        valid
    }

    /// The proof-of-work input `state ‖ Ascii(label) ‖ nonce`, with room for the nonce last, so
    /// that each attempt rewrites eight bytes and hashes the input in one piece.
    fn pow_input(&self, label: &str) -> Vec<u8> {
        [
            &self.state[..],
            &label_prefix(label),
            label.as_bytes(),
            &[0; 8],
        ]
        .concat()
    }

    /// Whether `H("pow", attempt)`, with `nonce` written into `attempt`'s last eight bytes, starts
    /// with `bits` zero bits.
    fn pow_meets(&self, attempt: &mut [u8], bits: u8, nonce: u64) -> bool {
        let at = attempt.len() - 8;
        attempt[at..].copy_from_slice(&nonce.to_le_bytes());
        let mut input = self.hash.input(Domain::Pow);
        input.put(attempt);
        leading_zero_bits(&self.hash.digest(input)) >= u32::from(bits)
    }
}

/// The length that starts `Ascii(label)`, after checking that `label` is printable ASCII, as
/// every label of RFC-0014 is: another label would break the `Ascii` encoding both sides hash.
fn label_prefix(label: &str) -> [u8; 8] {
    assert!(
        label.bytes().all(|b| (0x20..=0x7E).contains(&b)),
        "a transcript label is printable ASCII (RFC-0014): {label:?}"
    );
    (label.len() as u64).to_le_bytes()
}

/// The first bytes of a message, which go straight to the hash: most messages are a few round
/// coefficients, for which collecting them would cost more than it saves.
const DIRECT: usize = 1 << 10;

/// The bytes a [`Staged`] buffer holds before it passes them on.
const STAGE: usize = 64 << 10;

/// A sink that collects the small writes of a long message, such as one field element at a
/// time, into updates of up to [`STAGE`] bytes, which BLAKE3 hashes several chunks at a time.
/// The first [`DIRECT`] bytes pass straight through, so a short message allocates nothing.
struct Staged<'s, S: Sink> {
    sink: &'s mut S,
    direct: usize,
    buffer: Vec<u8>,
}

impl<'s, S: Sink> Staged<'s, S> {
    fn new(sink: &'s mut S) -> Self {
        Self {
            sink,
            direct: 0,
            buffer: Vec::new(),
        }
    }

    /// Passes on what is left.
    fn finish(mut self) {
        self.flush();
    }

    fn flush(&mut self) {
        if !self.buffer.is_empty() {
            self.sink.put(&self.buffer);
            self.buffer.clear();
        }
    }
}

impl<S: Sink> Sink for Staged<'_, S> {
    fn put(&mut self, bytes: &[u8]) {
        if self.direct < DIRECT {
            self.direct += bytes.len();
            self.sink.put(bytes);
            return;
        }
        if bytes.len() > STAGE - self.buffer.len() {
            self.flush();
        }
        if bytes.len() >= STAGE {
            self.sink.put(bytes);
        } else {
            self.buffer.extend_from_slice(bytes);
        }
    }
}

/// The four little-endian 64-bit words of a digest, in order.
fn words(digest: &Digest) -> [u64; 4] {
    core::array::from_fn(|i| {
        let mut word = [0; 8];
        word.copy_from_slice(&digest[8 * i..8 * i + 8]);
        u64::from_le_bytes(word)
    })
}

/// The number of zero bits before the first one, most significant bit of byte 0 first.
fn leading_zero_bits(digest: &Digest) -> u32 {
    let mut zeros = 0;
    for &byte in digest {
        if byte != 0 {
            return zeros + byte.leading_zeros();
        }
        zeros += 8;
    }
    zeros
}
