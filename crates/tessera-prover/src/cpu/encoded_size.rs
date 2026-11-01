// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public proof-byte metrics without retaining a temporary TCE encoding (05 §4, RFC-0018).

use tessera_codec::{Encode, Sink};

/// Metrics only: saturation is not an allocation bound or a memory-admission decision.
#[derive(Default)]
pub(super) struct EncodedSize(u64);

impl EncodedSize {
    pub(super) fn of<T: Encode + ?Sized>(value: &T) -> u64 {
        let mut size = Self::default();
        size.add(value);
        size.finish()
    }

    /// Append another complete encoding, retaining the previous metric's sum of part sizes.
    pub(super) fn add<T: Encode + ?Sized>(&mut self, value: &T) {
        value.encode_to(self);
    }

    pub(super) fn finish(self) -> u64 {
        self.0
    }
}

impl Sink for EncodedSize {
    fn put(&mut self, bytes: &[u8]) {
        self.0 = self.0.saturating_add(bytes.len() as u64);
    }
}

/// Admission sizing must fail on overflow; metrics deliberately saturate.
#[cfg(test)]
pub(super) struct CheckedEncodedSize(Option<u64>);
#[cfg(test)]
impl CheckedEncodedSize {
    pub(super) fn of<T: Encode + ?Sized>(value: &T) -> Option<u64> {
        let mut size = Self(Some(0));
        value.encode_to(&mut size);
        size.0
    }
}
#[cfg(test)]
impl Sink for CheckedEncodedSize {
    fn put(&mut self, bytes: &[u8]) {
        self.0 = self.0.and_then(|n| n.checked_add(bytes.len() as u64));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_public_sizes_need_no_output_allocation() {
        let payload = vec![0xA5u8; 1 << 20];
        let nested = vec![Some(vec![1u64, 2, 3]), None, Some(Vec::new())];
        let measured = allocation_counter::measure(|| {
            // RFC-0003: Vec length is u64; Option tags are u16; integers are fixed width.
            assert_eq!(EncodedSize::of(&payload), 8 + (1 << 20));
            assert_eq!(EncodedSize::of(&nested), 8 + (2 + 8 + 3 * 8) + 2 + (2 + 8));
            let mut joined = EncodedSize::default();
            joined.add(&payload);
            joined.add(&nested);
            joined.add(&[0u16; 3]);
            assert_eq!(joined.finish(), 8 + (1 << 20) + 8 + 34 + 2 + 10 + 6);
        });
        assert_eq!(measured.bytes_total, 0);
        assert_eq!(measured.bytes_max, 0);
    }

    #[test]
    fn admission_encoding_size_rejects_overflow_without_allocating() {
        let value = vec![1u64, 2, 3];
        let observed = allocation_counter::measure(|| {
            assert_eq!(CheckedEncodedSize::of(&value), Some(8 + 3 * 8));
            let mut size = CheckedEncodedSize(Some(u64::MAX - 1));
            size.put(&[0, 0]);
            assert_eq!(size.0, None);
            size.put(&[0]);
            assert_eq!(size.0, None);
        });
        assert_eq!(observed.bytes_total, 0);
    }

    #[test]
    fn unrepresentable_metric_saturates_without_overflow() {
        let mut size = EncodedSize(u64::MAX - 1);
        size.put(&[0, 0]);
        size.add(&0u64);
        assert_eq!(size.finish(), u64::MAX);
    }
}
