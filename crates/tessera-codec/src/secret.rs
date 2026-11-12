// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Redaction by type (05 §3 'Data classification and redaction').
//!
//! [`Secret`] holds a secret-class value: weights, activations, prompt and output tokens,
//! logits, masking randomness, keys and seeds. It offers no way to print, clone or TCE-encode
//! the value, so the value cannot reach a log, an error, a report or a protocol message by
//! accident. The value comes out only through [`Secret::expose_secret`] and
//! [`Secret::expose_secret_mut`], and only in the readers `secret-access.toml` lists: the witness
//! engine, gadget witness generation, commitment encoding, key handling and the proving engines'
//! entry points (OBS-1). Elsewhere, clippy rejects the call in the code it builds, and `cargo xtask
//! lint` rejects any file that names an accessor, under any `cfg`, or switches clippy's rule off.
//!
//! The value is zeroized when its wrapper drops (OBS-2). [`Public`] is the explicit opposite: a
//! value cleared for telemetry and errors.
//!
//! [`SecretBuf`] holds what code derives from secret-class values, such as a proving engine's
//! bound tables and partial products (OBS-6). Like `Secret`, it cannot be printed, cloned or
//! encoded, and it clears its memory: every allocation it releases, when it drops and when it
//! grows. Unlike `Secret`, it lends its values to the code that holds it, as a slice.

use core::fmt;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use std::collections::TryReserveError;

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

/// `N` secret bytes on the heap: an AES key or a salt (16, the default) or an Ed25519 seed (32).
/// Moving it moves a pointer and leaves no copy of the bytes behind (OBS-2), which a key that
/// lives as long as a session needs. Hold it in a [`Secret`]; it is also cleared when it drops
/// outside one. It has no `Debug`, `Clone` or TCE `Encode`.
pub struct HeapKey<const N: usize = 16>(Box<[u8; N]>);

impl<const N: usize> HeapKey<N> {
    /// `N` zero bytes, to fill in place.
    pub fn zeroed() -> Self {
        Self(Box::new([0; N]))
    }

    /// The key in `bytes`, already on the heap: the box moves, and no byte is copied.
    pub fn from_box(bytes: Box<[u8; N]>) -> Self {
        Self(bytes)
    }

    /// `N` bytes from the operating system's generator, in a [`Secret`].
    ///
    /// # Errors
    /// The generator's failure, as an I/O error (`ERR-RES-IO`).
    pub fn generate() -> Result<Secret<Self>, std::io::Error> {
        let mut key = Secret::new(Self::zeroed());
        getrandom::fill(&mut key.0.0[..]).map_err(std::io::Error::from)?;
        Ok(key)
    }
}

impl<const N: usize> Zeroize for HeapKey<N> {
    fn zeroize(&mut self) {
        self.0.as_mut().zeroize();
    }
}

impl<const N: usize> Drop for HeapKey<N> {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl<const N: usize> core::borrow::Borrow<[u8; N]> for HeapKey<N> {
    fn borrow(&self) -> &[u8; N] {
        &self.0
    }
}

impl<const N: usize> core::borrow::BorrowMut<[u8; N]> for HeapKey<N> {
    fn borrow_mut(&mut self) -> &mut [u8; N] {
        &mut self.0
    }
}

/// Values derived from secret-class data, in a vector that clears every allocation it releases
/// (05 §3 OBS-6): a proving engine's bound tables, partial products, contractions and tree levels.
///
/// The whole allocation, spare capacity included, is overwritten with zeros when the buffer drops
/// and when it grows: growing moves the values to an allocation at least twice the size and clears
/// the old one before freeing it, where a `Vec` would free it as it is. The zeros are a plain fill,
/// as fast as `memset`, followed by [`zeroize::optimization_barrier`], which the compiler must
/// assume reads them, so it cannot drop the fill as dead before the free. (`Vec`'s own `Zeroize`
/// clears spare capacity one volatile byte at a time, about 4 GB/s, and an engine's tables are
/// mostly spare capacity once binding has halved them.)
///
/// Like [`Secret`], it has no `Debug`, `Display`, `Clone` or TCE `Encode`. Unlike `Secret`, it
/// lends its values to whoever holds it, as a slice: it is working memory for code that computes
/// on secret-class values, which `secret-access.toml` lists, and that code keeps it to itself.
///
/// What it cannot clear:
/// - a copy taken out of it, such as `to_vec()` or a value read into a local;
/// - registers and stack slots that held its values, which the compiler places;
/// - a buffer that never drops (`core::mem::forget`, `Box::leak`, a reference cycle).
pub struct SecretBuf<T: Copy>(Vec<T>);

impl<T: Copy> SecretBuf<T> {
    /// An empty buffer, which allocates nothing.
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    /// An empty buffer with room for exactly `capacity` values.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// [`with_capacity`](Self::with_capacity), or the allocator's refusal.
    ///
    /// # Errors
    ///
    /// The [`TryReserveError`] of a capacity that overflows or that the allocator refuses.
    pub fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        let mut values = Vec::new();
        values.try_reserve_exact(capacity)?;
        Ok(Self(values))
    }

    /// `len` copies of `value`, with room for exactly `len`. Built by `vec!`, which allocates
    /// exactly `len` and takes memory the allocator has already zeroed when `value` is a zero
    /// integer.
    pub fn filled(value: T, len: usize) -> Self {
        Self(vec![value; len])
    }

    /// The number of values it holds without growing.
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// Makes room for `additional` more values. When the allocation is too small, the values move
    /// to a new one of at least twice its size, and the old one is cleared before it is freed.
    pub fn reserve(&mut self, additional: usize) {
        let required = self.0.len().saturating_add(additional);
        if required <= self.0.capacity() {
            return;
        }
        let mut grown = Vec::with_capacity(required.max(self.0.capacity().saturating_mul(2)));
        grown.extend_from_slice(&self.0);
        clear(&mut core::mem::replace(&mut self.0, grown));
    }

    /// Appends `value`, growing as [`reserve`](Self::reserve) does.
    pub fn push(&mut self, value: T) {
        self.reserve(1);
        self.0.push(value);
    }

    /// Shortens it to `len` values, or lengthens it with copies of `value`, growing as
    /// [`reserve`](Self::reserve) does.
    pub fn resize(&mut self, len: usize, value: T) {
        self.reserve(len.saturating_sub(self.0.len()));
        self.0.resize(len, value);
    }

    /// Keeps the first `len` values. The allocation keeps the others until it is cleared.
    pub fn truncate(&mut self, len: usize) {
        self.0.truncate(len);
    }

    /// Removes every value. The allocation keeps them until it is cleared.
    pub fn clear(&mut self) {
        self.0.clear();
    }
}

/// Empties `values` and overwrites its whole allocation with zeros that the compiler keeps.
fn clear<T: Copy>(values: &mut Vec<T>) {
    values.clear();
    let allocation = values.spare_capacity_mut();
    allocation.fill(MaybeUninit::zeroed());
    zeroize::optimization_barrier(allocation);
}

impl<T: Copy> Drop for SecretBuf<T> {
    fn drop(&mut self) {
        clear(&mut self.0);
    }
}

impl<T: Copy> ZeroizeOnDrop for SecretBuf<T> {}

impl<T: Copy> Default for SecretBuf<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy> Deref for SecretBuf<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T: Copy> DerefMut for SecretBuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T: Copy> AsRef<[T]> for SecretBuf<T> {
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

/// Appends the values, growing as [`SecretBuf::reserve`] does: once by the iterator's lower
/// bound, then one value at a time past it.
impl<T: Copy> Extend<T> for SecretBuf<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, values: I) {
        let mut values = values.into_iter();
        self.reserve(values.size_hint().0);
        // No more than the spare capacity, so that the vector's own extension never reallocates.
        let spare = self.0.capacity() - self.0.len();
        self.0.extend(values.by_ref().take(spare));
        for value in values {
            self.push(value);
        }
    }
}

impl<T: Copy> FromIterator<T> for SecretBuf<T> {
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        let mut buffer = Self::new();
        buffer.extend(values);
        buffer
    }
}

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
