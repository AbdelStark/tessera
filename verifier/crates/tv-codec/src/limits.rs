// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Decode limits (RFC-0003 'Limits').

/// The resource limits every decoder takes explicitly (RFC-0003 'Limits').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecodeLimits {
    /// The most bytes a whole input may hold; for an envelope, its header, payload and digest.
    pub max_total_bytes: u64,
    /// The largest length any single sequence may declare.
    pub max_vec_len: u64,
    /// The deepest nesting of composite values. A limit above
    /// [`DEPTH_CEILING`](Self::DEPTH_CEILING) counts as the ceiling.
    pub max_depth: u16,
}

impl DecodeLimits {
    /// The deepest nesting any limit admits. Decoding recurses once per level, and deeper
    /// recursion could exhaust a thread's stack.
    pub const DEPTH_CEILING: u16 = 256;

    /// Verifier messages: 8 GiB in all, sequences of up to 2^34 elements, depth 32.
    pub const MESSAGE: Self = Self {
        max_total_bytes: 8_589_934_592,
        max_vec_len: 17_179_869_184,
        max_depth: 32,
    };

    /// `IntegerModel` artifacts: 1 TiB in all, otherwise as [`MESSAGE`](Self::MESSAGE).
    pub const INTEGER_MODEL: Self = Self {
        max_total_bytes: 1_099_511_627_776,
        ..Self::MESSAGE
    };

    /// The depth decoding enforces: `max_depth`, or the ceiling when it is lower.
    pub const fn effective_max_depth(&self) -> u16 {
        if self.max_depth < Self::DEPTH_CEILING {
            self.max_depth
        } else {
            Self::DEPTH_CEILING
        }
    }
}
