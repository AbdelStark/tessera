// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Requested-byte geometry shared by public constraint builders.

use crate::{Affine, Committed, Lookup, Term, Virtual};
use tessera_field::Goldilocks as F;

pub(crate) fn array<T>(n: u64) -> Option<u64> {
    let bytes = n.checked_mul(size_of::<T>() as u64)?;
    (bytes <= isize::MAX as u64).then_some(bytes)
}
pub(crate) fn sum<const N: usize>(values: [Option<u64>; N]) -> Option<u64> {
    values.into_iter().try_fold(0u64, |s, v| s.checked_add(v?))
}
pub(crate) fn times(n: u64, value: Option<u64>) -> Option<u64> {
    n.checked_mul(value?)
}

// The pinned Rust Vec grows by doubling (or the requested size, if larger).
// Its minimum nonzero capacity is 8/4/1 for these element sizes. Total requests
// through a growth history are <= 4*max(min_capacity, maximum_length) elements.
// maximum_length includes lengths before deduplication; explicit reservations
// here never exceed that planned maximum.
// Exact vectors are charged with array instead. No payload is allocated here.
pub(crate) fn growing<T>(n: u64) -> Option<u64> {
    if n == 0 || size_of::<T>() == 0 {
        return Some(0);
    }
    let minimum = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    array::<T>(4u64.checked_mul(n.max(minimum))?)
}

pub(crate) fn nonneg(limbs: u64) -> Option<u64> {
    let committed = limbs.checked_sub(1)?;
    // Construction terms, their affine copy, the lookup clone and the virtual.
    sum([
        array::<Committed>(committed),
        array::<(Virtual, Affine<F>)>(1),
        growing::<Lookup<F>>(limbs),
        growing::<(F, Term)>(limbs),
        times(2, array::<(F, Term)>(limbs)),
        times(committed, array::<(F, Term)>(1)),
    ])
}
