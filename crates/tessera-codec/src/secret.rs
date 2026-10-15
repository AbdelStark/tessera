// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Redaction by type (05 §3 'Data classification and redaction').
//!
//! [`Secret`] holds a secret-class value: weights, activations, prompt and output tokens,
//! logits, masking randomness, keys and seeds. It offers no way to print, clone or TCE-encode
//! the value, so the value cannot reach a log, an error, a report or a protocol message by
//! accident. The value comes out only through [`Secret::expose_secret`] and
//! [`Secret::expose_secret_mut`], and only in the files `secret-access.toml` lists: the witness
//! engine, gadget witness generation, commitment encoding and key handling (OBS-1). Elsewhere,
//! clippy rejects the call in the code it builds, and `cargo xtask lint` rejects any file that
//! names an accessor, under any `cfg`, or switches clippy's rule off.
//!
//! The value is zeroized when its wrapper drops (OBS-2). [`Public`] is the explicit opposite: a
//! value cleared for telemetry and errors.

use core::fmt;

use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{Decode, DecodeError, Encode, Reader, Sink};

/// A secret-class value. It has no `Debug`, `Display`, `Clone` or TCE `Encode`, and it is
/// zeroized on drop.
///
/// Zeroization clears the value where it lives when the wrapper drops, and nothing else:
/// - moving a `Secret` may copy its bytes and leave the old ones in place, so a large or
///   long-lived value belongs on the heap (`Secret<Vec<_>>`, `Secret<Box<_>>`), where a move
///   copies only the pointer;
/// - a vector that grows reallocates and leaves its old buffer uncleared, so size it up front;
/// - a leaked secret (`core::mem::forget`, `Box::leak`, a reference cycle) is never dropped and
///   never cleared;
/// - `T`'s own [`Zeroize`] decides what is cleared.
pub struct Secret<T: Zeroize>(T);

impl<T: Zeroize> Secret<T> {
    /// Takes ownership of `value`.
    pub fn new(value: T) -> Self {
        Self(value)
    }

    /// The value. Allowed only in the modules `secret-access.toml` lists (05 §3 OBS-1).
    pub fn expose_secret(&self) -> &T {
        &self.0
    }

    /// The value, for changing in place. Allowed only where [`Secret::expose_secret`] is.
    pub fn expose_secret_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: Zeroize> Drop for Secret<T> {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl<T: Zeroize> ZeroizeOnDrop for Secret<T> {}

/// A value cleared for telemetry and errors (05 §3 OBS-1). It prints and encodes exactly as the
/// value it wraps: the wrapper only records the decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Public<T>(pub T);

impl<T: fmt::Display> fmt::Display for Public<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: Encode> Encode for Public<T> {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        self.0.encode_to(out);
    }
}

impl<T: Decode> Decode for Public<T> {
    const MIN_SIZE: usize = T::MIN_SIZE;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        T::decode_from(reader).map(Public)
    }
}
