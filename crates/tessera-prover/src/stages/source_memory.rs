// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Canonical relation-source heap geometry (RFC-0018/0055).
use super::StageError;

/// Source descriptors and canonical original resident input requests only.
/// Relation contractions, sumcheck engines, wrappers/messages, constructor
/// transients and this public planner's own allocations remain additional.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageSourceMemory {
    pub(super) metadata: u64,
    pub(super) inputs: u64,
}
impl StageSourceMemory {
    /// Retained canonical arena/plan/point/index/factor metadata upper bound.
    pub fn metadata_bytes(self) -> u64 {
        self.metadata
    }
    /// Original resident prime-field columns, including their Vec growth requests.
    /// Streaming recipes do not allocate these arrays; their folded/grid storage is separate.
    pub fn resident_input_bytes(self) -> u64 {
        self.inputs
    }
}
pub(super) fn add(a: u64, b: u64) -> Result<u64, StageError> {
    a.checked_add(b).ok_or(StageError::Memory)
}
pub(super) fn mul(a: u64, b: u64) -> Result<u64, StageError> {
    a.checked_mul(b).ok_or(StageError::Memory)
}
pub(super) fn array<T>(n: usize) -> Result<u64, StageError> {
    let bytes = mul(n as u64, size_of::<T>() as u64)?;
    if bytes > isize::MAX as u64 {
        return Err(StageError::Memory);
    }
    Ok(bytes)
}
pub(super) fn growing<T>(n: usize) -> Result<u64, StageError> {
    if n == 0 {
        return Ok(0);
    }
    let minimum = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    let cap = n.max(minimum).checked_mul(4).ok_or(StageError::Memory)?;
    array::<T>(cap)
}
pub(super) fn arc<T>() -> Result<u64, StageError> {
    add(
        add(size_of::<T>() as u64, 2 * size_of::<usize>() as u64)?,
        2 * (align_of::<T>() as u64 - 1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_allocation_geometry_fails_closed_without_allocating() {
        let measured = allocation_counter::measure(|| {
            assert_eq!(
                array::<u64>(isize::MAX as usize / 8 + 1),
                Err(StageError::Memory)
            );
            assert_eq!(add(u64::MAX, 1), Err(StageError::Memory));
            assert_eq!(mul(u64::MAX, 2), Err(StageError::Memory));
            assert_eq!(growing::<u64>(usize::MAX), Err(StageError::Memory));
            assert_eq!(growing::<u8>(usize::MAX / 4 + 1), Err(StageError::Memory));
            assert_eq!(growing::<u64>(0), Ok(0));
        });
        assert_eq!(measured.count_total, 0);
    }
    #[test]
    fn resident_generated_vector_requests_fit_the_input_envelope() {
        use tessera_field::{Field, Goldilocks};
        for n in [0usize, 1, 2, 3, 7, 513, 4097] {
            let bound = growing::<Goldilocks>(n).unwrap();
            let measured = allocation_counter::measure(|| {
                let out =
                    super::super::columns::Column::generated(false, n, 0, |_| Ok(Goldilocks::ZERO))
                        .unwrap();
                assert_eq!(out.len(), n);
            });
            assert!(measured.bytes_max <= bound, "{n}: {measured:?}, {bound}");
        }
    }
}
