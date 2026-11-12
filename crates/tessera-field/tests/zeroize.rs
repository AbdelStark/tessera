// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Field elements in secret-class containers leave nothing behind when their memory is freed
//! (05 §3 OBS-2, OBS-6): a `Secret<Vec<_>>` clears its whole capacity when it drops, and a
//! `SecretBuf` clears every allocation it releases, when it drops and when it grows.
//!
//! A recording allocator counts the nonzero bytes of a block as it is freed. Every slot of the
//! vectors under test, spare capacity included, holds nonzero bytes only, so each count is exact.
//! Two controls show what the recorder sees without clearing: every byte after a plain drop, and
//! the spare capacity after clearing the elements alone.

#![deny(unsafe_code)]

use std::fmt::Debug;
use std::sync::atomic::Ordering;

use tessera_codec::{Secret, SecretBuf, Zeroize};
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};

/// Forwards to `System` and, at the first free of the watched block, counts its nonzero bytes.
#[allow(unsafe_code)]
mod recorder {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::ptr;
    use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

    /// The block whose next free is inspected; null when none is.
    pub static WATCHED: AtomicPtr<u8> = AtomicPtr::new(ptr::null_mut());
    /// The nonzero bytes the watched block held when it was freed; `usize::MAX` until then.
    pub static NONZERO: AtomicUsize = AtomicUsize::new(usize::MAX);

    pub struct Recorder;

    // SAFETY: both methods forward to `System` with their own arguments, so its guarantees carry
    // over. `dealloc` first reads the block it frees: the block is live until the forwarded call,
    // `layout.size()` bytes long, and the tests initialize every byte of each block they watch.
    unsafe impl GlobalAlloc for Recorder {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: the caller's contract, forwarded unchanged.
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
            let watched = WATCHED.compare_exchange(
                block,
                ptr::null_mut(),
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
            if watched.is_ok() {
                // SAFETY: as stated on the impl: live, `layout.size()` bytes, initialized.
                let bytes = unsafe { std::slice::from_raw_parts(block, layout.size()) };
                NONZERO.store(bytes.iter().filter(|&&b| b != 0).count(), Ordering::SeqCst);
            }
            // SAFETY: the caller's contract, forwarded unchanged.
            unsafe { System.dealloc(block, layout) }
        }
    }
}

#[global_allocator]
static ALLOCATOR: recorder::Recorder = recorder::Recorder;

const CAPACITY: usize = 1024;
const LENGTH: usize = 600;

/// The byte `k` in all eight places: below `P` for `k ≤ 254`, so the element holds it as is.
fn repeated(k: u64) -> Goldilocks {
    Goldilocks::from_u64_reduced(k * 0x0101_0101_0101_0101)
}

/// `CAPACITY` values made of nonzero bytes only: `element(k)` repeats the byte `k`.
fn nonzero_values<T>(element: fn(u64) -> T) -> impl Iterator<Item = T> {
    (0..CAPACITY as u64).map(move |i| element(1 + i % 254))
}

/// A vector of capacity `CAPACITY` and length `LENGTH` whose every slot, spare capacity
/// included, holds written values.
fn filled<T>(element: fn(u64) -> T) -> Vec<T> {
    let mut values = Vec::with_capacity(CAPACITY);
    values.extend(nonzero_values(element));
    values.truncate(LENGTH);
    assert_eq!(values.capacity(), CAPACITY);
    values
}

/// Runs `free` with `block` watched, and returns the nonzero bytes `block` held when freed.
fn nonzero_bytes_when_freed<T>(block: *const T, free: impl FnOnce()) -> usize {
    recorder::NONZERO.store(usize::MAX, Ordering::SeqCst);
    recorder::WATCHED.store(block.cast::<u8>().cast_mut(), Ordering::SeqCst);
    free();
    recorder::NONZERO.load(Ordering::SeqCst)
}

fn check<T: Copy + Zeroize + PartialEq + Debug>(element: fn(u64) -> T) {
    let size = size_of::<T>();

    let plain = filled(element);
    let freed = nonzero_bytes_when_freed(plain.as_ptr(), || drop(plain));
    assert_eq!(
        freed,
        CAPACITY * size,
        "a plain drop frees every byte as written"
    );
    let mut elements = filled(element);
    let freed = nonzero_bytes_when_freed(elements.as_ptr(), || {
        elements.iter_mut().for_each(Zeroize::zeroize);
        drop(elements);
    });
    assert_eq!(
        freed,
        (CAPACITY - LENGTH) * size,
        "clearing each element clears all its bytes and leaves the spare capacity"
    );

    let secret = filled(element);
    let block = secret.as_ptr();
    let freed = nonzero_bytes_when_freed(block, || drop(Secret::new(secret)));
    assert_eq!(
        freed, 0,
        "a Secret<Vec<_>> frees zeros only, spare capacity included"
    );

    let mut buffer = SecretBuf::with_capacity(CAPACITY);
    buffer.extend(nonzero_values(element));
    buffer.truncate(LENGTH);
    assert_eq!(buffer.capacity(), CAPACITY);
    let freed = nonzero_bytes_when_freed(buffer.as_ptr(), || drop(buffer));
    assert_eq!(
        freed, 0,
        "a SecretBuf frees zeros only, spare capacity included"
    );

    let mut buffer: SecretBuf<T> = SecretBuf::with_capacity(CAPACITY);
    buffer.extend(nonzero_values(element));
    let next = element(77);
    let freed = nonzero_bytes_when_freed(buffer.as_ptr(), || buffer.push(next));
    assert_eq!(
        freed, 0,
        "a SecretBuf that grows frees its old allocation as zeros"
    );
    let expected: Vec<T> = nonzero_values(element).chain([next]).collect();
    assert!(buffer[..] == expected[..], "growing keeps the values");
    assert!(buffer.capacity() >= 2 * CAPACITY);
}

#[test]
fn secret_containers_free_only_zeros() {
    // One test, so that no other test of this binary frees memory while a block is watched.
    check(repeated);
    check(|k| GoldilocksExt2::new(repeated(k), repeated(255 - k)));
}
