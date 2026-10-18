// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `WeightCommitmentState`, schema `0x0302` (RFC-0040 'WeightCommitmentState'): the public part,
//! the masks' seed and each matrix's hiding budget.
//!
//! # Secrets
//!
//! `masks_seed` is secret-class masking randomness (05 §3). A state gets it in one of three ways,
//! and no other:
//!
//! - drawn from the operating system's CSPRNG ([`WeightCommitmentState::new`]);
//! - decoded from the artifact ([`WeightCommitmentState::open`], [`WeightCommitmentState::read`]);
//! - given, in tests only (`WeightCommitmentState::with_seed`): in this crate's own test build,
//!   and for other crates under the `test-rng` feature, which release builds reject.
//!
//! The seed is written straight into a heap allocation inside its `Secret`, so moving the state
//! moves a pointer and leaves no copy of the bytes behind, and the allocation is cleared when the
//! state drops (OBS-2). The field is private, and the state has no `Encode` or `Decode`: no code
//! outside this file can copy the seed out or encode the state (AGENTS.md invariant 6).
//!
//! This file is one of the readers `secret-access.toml` lists. The seed leaves its `Secret` only
//! into the TCE of the secret artifact `0x0302`, through a private wrapper:
//! [`WeightCommitmentState::write`] streams it to a file of mode 0600 through the envelope
//! writer's zeroizing buffer, and `WeightCommitmentState::sealed`, test-only as `with_seed` is,
//! returns it in a `SecretBuf`. The state's `Debug` form shows its public part and budget, never
//! the seed.
//!
//! The test-only items are not behind a dev-dependency of this crate on itself with `test-rng`:
//! that would enable the feature in this crate's benchmarks too, which build without debug
//! assertions and so stop at the feature's guard. This crate's own test build has them through
//! `cfg(test)` instead, which no other crate's build ever sees.
//!
//! The state cannot be encoded by name either:
//!
//! ```compile_fail
//! fn leak(state: &tessera_pcs::WeightCommitmentState) -> Vec<u8> {
//!     tessera_codec::encode(state)
//! }
//! ```
//!
//! ```compile_fail
//! fn leak(state: &tessera_pcs::WeightCommitmentState) {
//!     let _seed = &state.masks_seed;
//! }
//! ```

use core::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use tessera_codec::envelope::{self, ReadError};
use tessera_codec::{
    Decode, DecodeError, DecodeLimits, Encode, Reader, Secret, SecretBuf, Sink, Zeroize, schema,
};

use crate::{PcsError, WeightCommitmentPublic};

/// The limits the state is read under: RFC-0003's defaults for messages.
const LIMITS: DecodeLimits = DecodeLimits::MESSAGE;

/// The distinct columns of one matrix revealed in an epoch: a set, kept strictly increasing as
/// integers (RFC-0040 'WeightCommitmentState').
///
/// It encodes as `Vec<u32>`. The indices increase strictly, so a set has one encoding, and a
/// decoder rejects an unsorted or repeated index with `ERR-CODEC-NONCANONICAL` at that index's
/// first byte. The order is numeric, not the byte order of a map's keys (RFC-0003 'Composite
/// types'): 1 comes before 256.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct OpenedColumns(Vec<u32>);

impl OpenedColumns {
    /// The set of `columns`, which must increase strictly.
    ///
    /// # Errors
    ///
    /// [`PcsError::ColumnOrder`] at the first column not above the one before it.
    pub fn new(columns: Vec<u32>) -> Result<Self, PcsError> {
        if let Some(before) = columns.windows(2).position(|pair| pair[1] <= pair[0]) {
            return Err(PcsError::ColumnOrder {
                position: before + 1,
            });
        }
        Ok(Self(columns))
    }

    /// The columns, in increasing order.
    pub fn columns(&self) -> &[u32] {
        &self.0
    }
}

/// As `Vec<u32>`: a `u64` count, then the indices.
impl Encode for OpenedColumns {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        self.0.encode_to(out);
    }
}

/// As `Vec<u32>`, one level deep, with each index read whole and then checked above the one
/// before it (RFC-0003 'Limits', as for a map's keys).
impl Decode for OpenedColumns {
    const MIN_SIZE: usize = 8;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        // The count was checked against the bytes left, at 4 bytes an index, so the vector's
        // room is no more than the input holds (INV-TCE-3).
        let count = reader.sequence_len(<u32 as Decode>::MIN_SIZE)?;
        let mut columns = Vec::with_capacity(count);
        for _ in 0..count {
            let at = reader.position();
            let column = u32::decode_from(reader)?;
            if columns.last().is_some_and(|&before| column <= before) {
                return Err(reader.noncanonical(at));
            }
            columns.push(column);
        }
        reader.leave();
        Ok(Self(columns))
    }
}

/// One matrix's hiding budget in an epoch (RFC-0040). Its limits are the layout's `P_W` and
/// `B_W`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Encode, Decode)]
pub struct TypeBudget {
    /// The distinct columns revealed this epoch.
    pub opened_columns: OpenedColumns,
    /// The first blinding row not burned: the rows below it are.
    pub next_blinding_row: u32,
}

/// The masks seed's 32 bytes on the heap. The locked `zeroize` clears boxed slices but has no
/// `Zeroize` for a boxed array, so this clears the array where it lives.
struct Seed(Box<[u8; 32]>);

impl Seed {
    /// A zeroed seed, its bytes on the heap before any secret is written into them.
    fn zeroed() -> Self {
        Self(Box::new([0; 32]))
    }
}

impl Zeroize for Seed {
    fn zeroize(&mut self) {
        self.0.as_mut().zeroize();
    }
}

/// The weight commitment's state: schema `0x0302`, version 1, secret, `.twc` (RFC-0040, 03 §5).
///
/// In `tessera-protocol/1` the seed regenerates nothing, every `opened_columns` is empty and
/// every `next_blinding_row` is 0: the prover recomputes the commitment from `D` and the state.
pub struct WeightCommitmentState {
    /// The commitment's public part.
    pub public: WeightCommitmentPublic,
    /// The seed of the masks and blinding rows, secret-class.
    masks_seed: Secret<Seed>,
    /// Each matrix's budget, in [`WeightMatrix`](crate::WeightMatrix) order.
    pub budget: [TypeBudget; 10],
}

/// Shows the public part and the budget; the seed is secret.
impl fmt::Debug for WeightCommitmentState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightCommitmentState")
            .field("public", &self.public)
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

impl WeightCommitmentState {
    /// The state of a new commitment `public` (RFC-0040): a seed drawn from the operating
    /// system's CSPRNG straight into its `Secret` (06 §7), no column opened and no blinding row
    /// burned.
    ///
    /// # Errors
    ///
    /// The operating system's, when its CSPRNG cannot be read.
    #[allow(clippy::disallowed_methods)]
    pub fn new(public: WeightCommitmentPublic) -> io::Result<Self> {
        let mut masks_seed = Secret::new(Seed::zeroed());
        getrandom::fill(&mut masks_seed.expose_secret_mut().0[..])?;
        Ok(Self {
            public,
            masks_seed,
            budget: core::array::from_fn(|_| TypeBudget::default()),
        })
    }

    /// The state an envelope of schema `0x0302` holds, checked in the order of RFC-0003
    /// 'Envelope' and decoded under the message limits; the seed is read straight into its
    /// `Secret`. A reader of version 1 accepts version 1 only.
    ///
    /// # Errors
    ///
    /// The envelope's and the payload's [`DecodeError`]s, at their offsets from the envelope's
    /// first byte; among them `ERR-CODEC-NONCANONICAL` at an opened column not above the one
    /// before it.
    pub fn open(bytes: &[u8]) -> Result<Self, DecodeError> {
        envelope::open::<Decoded>(bytes, &schema::WEIGHT_COMMITMENT_STATE, LIMITS)
            .map(|decoded| decoded.0)
    }

    /// The state in the file at `path`, a `.twc` file: its envelope is checked as it streams
    /// past ([`envelope::verify`]), then the file is read into a buffer that is cleared when it
    /// is freed, and opened ([`open`](Self::open)), which checks the digest again.
    ///
    /// # Errors
    ///
    /// [`ReadError::Io`] when the file cannot be read, [`ReadError::Invalid`] when its envelope or
    /// payload breaks RFC-0003 or RFC-0040.
    pub fn read(path: &Path) -> Result<Self, ReadError> {
        let header = envelope::verify(
            BufReader::new(File::open(path)?),
            &schema::WEIGHT_COMMITMENT_STATE,
            LIMITS,
        )?;
        // The header bounded the whole envelope by the limits, 8 GiB.
        let len = (envelope::HEADER_LEN + envelope::DIGEST_LEN) as u64 + header.payload_len;
        let len = usize::try_from(len).map_err(|_| io::Error::other("an envelope past memory"))?;
        let mut bytes = SecretBuf::filled(0u8, len);
        let mut file = File::open(path)?;
        file.read_exact(&mut bytes)?;
        if file.read(&mut [0u8; 1])? != 0 {
            return Err(io::Error::other("the file grew while it was read").into());
        }
        Ok(Self::open(&bytes)?)
    }

    /// The state of `public` with the seed `seed` and the budgets `budget`, for tests: it exists
    /// in this crate's own test build, and for other crates under the `test-rng` feature, which
    /// pins masking randomness and which release builds reject (RFC-0001 'Cargo features').
    #[cfg(any(test, feature = "test-rng"))]
    #[allow(clippy::disallowed_methods)]
    pub fn with_seed(
        public: WeightCommitmentPublic,
        seed: [u8; 32],
        budget: [TypeBudget; 10],
    ) -> Self {
        let mut masks_seed = Secret::new(Seed::zeroed());
        masks_seed.expose_secret_mut().0.copy_from_slice(&seed);
        Self {
            public,
            masks_seed,
            budget,
        }
    }

    /// Writes the state to `path`, a `.twc` file, by atomic rename and with mode 0600 (RFC-0040
    /// 'Writing', 06 §7), and returns its envelope digest.
    ///
    /// # Errors
    ///
    /// Those of [`envelope::write_file`]; `path` then holds what it held before.
    pub fn write(&self, path: &Path) -> io::Result<envelope::Digest> {
        envelope::write_file(path, &schema::WEIGHT_COMMITMENT_STATE, &Tce(self))
    }

    /// The state's envelope, as [`write`](Self::write) writes it, in a buffer that is cleared
    /// when it is freed: for tests only, as [`with_seed`](Self::with_seed) is, so that they can
    /// compare the encoder's bytes with the golden vectors (RFC-0001 'Cargo features').
    #[cfg(any(test, feature = "test-rng"))]
    pub fn sealed(&self) -> SecretBuf<u8> {
        let mut out = Collect(SecretBuf::new());
        envelope::write(&mut out, &schema::WEIGHT_COMMITMENT_STATE, &Tce(self))
            .expect("writing to memory cannot fail");
        out.0
    }
}

/// The state's TCE, which only this file builds: the fields in order, the public part, the
/// seed's 32 bytes, the ten budgets.
struct Tce<'a>(&'a WeightCommitmentState);

impl Encode for Tce<'_> {
    #[allow(clippy::disallowed_methods)]
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        let state = self.0;
        state.public.encode_to(out);
        <[u8; 32] as Encode>::encode_to(&state.masks_seed.expose_secret().0, out);
        state.budget.encode_to(out);
    }
}

/// A state decoded from its TCE: a struct of three fields, one level deep, the public part, the
/// seed as `[u8; 32]`, read straight into its `Secret` and one level deep itself, and
/// `[TypeBudget; 10]`.
struct Decoded(WeightCommitmentState);

impl Decode for Decoded {
    const MIN_SIZE: usize =
        <WeightCommitmentPublic as Decode>::MIN_SIZE + 32 + 10 * <TypeBudget as Decode>::MIN_SIZE;

    #[allow(clippy::disallowed_methods)]
    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        reader.enter()?;
        let public = WeightCommitmentPublic::decode_from(reader)?;
        let mut masks_seed = Secret::new(Seed::zeroed());
        reader.enter()?;
        masks_seed
            .expose_secret_mut()
            .0
            .copy_from_slice(reader.take(32)?);
        reader.leave();
        let budget = <[TypeBudget; 10]>::decode_from(reader)?;
        reader.leave();
        Ok(Self(WeightCommitmentState {
            public,
            masks_seed,
            budget,
        }))
    }
}

/// Collects an envelope into a `SecretBuf`, which clears each allocation it leaves as it grows.
#[cfg(any(test, feature = "test-rng"))]
struct Collect(SecretBuf<u8>);

#[cfg(any(test, feature = "test-rng"))]
impl io::Write for Collect {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.extend(bytes.iter().copied());
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod vectors;
