// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's pinned sampler keys and its policy, checked at load, and the pin a request's
//! frames fall within (RFC-0061 'The sampler key and its pins', 'Which pin applies').

use core::fmt;

use super::Failure;
use crate::verdict::{ErrorCode, VerifierError};

/// A sampler key the verifier pinned on the prover's announcement (RFC-0061).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PinnedSamplerKey {
    /// `SHA-256("tessera-sampler-key/v1" ‖ K_p ‖ salt)`, the prover's base key's commitment.
    pub commitment: [u8; 32],
    /// `b`, the stream class's bits.
    pub class_bits: u8,
    /// From when it applies, on the recorders' clock.
    pub from_ns: u64,
    /// `W`, the window's length: the pin applies in `[from_ns, from_ns + W)`.
    pub window_ns: u64,
    /// `G`, the grace past the window's end within which a request's later frames may fall.
    pub grace_ns: u64,
    /// When the verifier received the announcement, on its own clock.
    pub received_ns: u64,
    /// The nonce the verifier drew on receipt.
    pub nonce: [u8; 32],
}

/// The verifier's sampler policy (RFC-0061).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplerPolicy {
    /// The `W` a new pin must have.
    pub window_ns: u64,
    /// The largest `G` a pin may have to apply for this verifier.
    pub max_grace_ns: u64,
    /// `b_max`, the largest `b` a pin may have to apply for this verifier.
    pub max_class_bits: u8,
}

impl SamplerPolicy {
    /// The defaults for an audit period of `period_ns` (RFC-0061): `W` the period, `G` at most
    /// `min(600 s, W/2)`, and `b_max = 0`.
    pub const fn default_for(period_ns: u64) -> Self {
        let half = period_ns / 2;
        Self {
            window_ns: period_ns,
            max_grace_ns: if half < MAX_DEFAULT_GRACE_NS {
                half
            } else {
                MAX_DEFAULT_GRACE_NS
            },
            max_class_bits: 0,
        }
    }

    /// Whether `pin` applies for this verifier: its `G` and `b` within the policy's. A pin
    /// outside them is treated as absent (RFC-0061 'The policy applies to every verifier').
    pub const fn admits(&self, pin: &PinnedSamplerKey) -> bool {
        pin.grace_ns <= self.max_grace_ns && pin.class_bits <= self.max_class_bits
    }
}

/// 600 seconds, the default grace's cap.
const MAX_DEFAULT_GRACE_NS: u64 = 600_000_000_000;

/// The most class bits any pin or policy may name.
const MAX_CLASS_BITS: u8 = 64;

/// A pin list and policy that passed RFC-0061's load checks: the only form in which the anchor
/// checks read pins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplerPins {
    pins: Vec<PinnedSamplerKey>,
    policy: SamplerPolicy,
}

impl SamplerPins {
    /// Loads the verifier's pins, in their list order, and its policy, before step 1 of RFC-0019
    /// (RFC-0061 'Checked at load'). The policy is checked first, then each pin in order, its own
    /// values before its relation to the pin before it:
    ///
    /// 1. the policy: `window_ns ≠ 0`, `2·max_grace_ns ≤ window_ns`, `max_class_bits ≤ 64`;
    /// 2. each pin: `W ≠ 0`, `2·G ≤ W`, `b ≤ 64`; `from_ns ≥ received_ns`, never retroactive; and
    ///    `from_ns + W + G` without overflow;
    /// 3. each pin after the first: `from_ns ≥` the previous pin's `from_ns + W`, and `W ≥ 2·G`
    ///    of the previous pin.
    ///
    /// The list's rules read each pin's own `W` and `G`, never the policy's, so a change of
    /// policy invalidates no pin held. Sums and doublings are checked; an overflow fails the
    /// load as the rule it computes for.
    ///
    /// # Errors
    ///
    /// The first rule broken ([`PinError`]): each is `Inconclusive(ERR-USAGE)`, a fault of the
    /// verifier's configuration whatever the statement.
    pub fn load(pins: Vec<PinnedSamplerKey>, policy: SamplerPolicy) -> Result<Self, PinError> {
        if policy.window_ns == 0 {
            return Err(PinError::PolicyWindow);
        }
        if policy
            .max_grace_ns
            .checked_mul(2)
            .is_none_or(|twice| twice > policy.window_ns)
        {
            return Err(PinError::PolicyGrace);
        }
        if policy.max_class_bits > MAX_CLASS_BITS {
            return Err(PinError::PolicyClassBits);
        }
        let mut previous: Option<&PinnedSamplerKey> = None;
        for (index, pin) in pins.iter().enumerate() {
            let fail = |rule| PinError::Pin { index, rule };
            if pin.window_ns == 0 {
                return Err(fail(PinRule::Window));
            }
            if pin
                .grace_ns
                .checked_mul(2)
                .is_none_or(|twice| twice > pin.window_ns)
            {
                return Err(fail(PinRule::Grace));
            }
            if pin.class_bits > MAX_CLASS_BITS {
                return Err(fail(PinRule::ClassBits));
            }
            if pin.from_ns < pin.received_ns {
                return Err(fail(PinRule::Retroactive));
            }
            if pin.limit().is_none() {
                return Err(fail(PinRule::Overflow));
            }
            if let Some(before) = previous {
                let end = before
                    .from_ns
                    .checked_add(before.window_ns)
                    .ok_or(fail(PinRule::Overflow))?;
                if pin.from_ns < end {
                    return Err(fail(PinRule::Overlap));
                }
                // `2·G ≤ W` held for the previous pin, so the doubling cannot overflow.
                let twice = before
                    .grace_ns
                    .checked_mul(2)
                    .ok_or(fail(PinRule::Overflow))?;
                if pin.window_ns < twice {
                    return Err(fail(PinRule::ShortWindow));
                }
            }
            previous = Some(pin);
        }
        Ok(Self { pins, policy })
    }

    /// The pins, in list order.
    pub fn pins(&self) -> &[PinnedSamplerKey] {
        &self.pins
    }

    /// The policy.
    pub fn policy(&self) -> &SamplerPolicy {
        &self.policy
    }

    /// The pin whose window contains `t_ns`, `from_ns ≤ t_ns < from_ns + W`, among those the
    /// policy admits, with its place in the list, which tells pins apart (RFC-0061 'Which pin
    /// applies'). The load rules make windows disjoint, so at most one does.
    pub fn containing(&self, t_ns: u64) -> Option<(usize, &PinnedSamplerKey)> {
        self.pins.iter().enumerate().find(|(_, pin)| {
            self.policy.admits(pin)
                && pin.from_ns <= t_ns
                && pin
                    .from_ns
                    .checked_add(pin.window_ns)
                    .is_some_and(|end| t_ns < end)
        })
    }
}

impl PinnedSamplerKey {
    /// `from_ns + W + G`, the first time no frame of a request under this pin may have; `None`
    /// on overflow, which the load refuses.
    pub const fn limit(&self) -> Option<u64> {
        match self.from_ns.checked_add(self.window_ns) {
            Some(end) => end.checked_add(self.grace_ns),
            None => None,
        }
    }
}

/// Why the verifier's pins or policy failed to load (RFC-0061 'Checked at load'). Each is
/// `Inconclusive(ERR-USAGE)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PinError {
    /// The policy's `window_ns` is 0.
    PolicyWindow,
    /// The policy's `2·max_grace_ns` exceeds its `window_ns`, or overflows.
    PolicyGrace,
    /// The policy's `max_class_bits` exceeds 64.
    PolicyClassBits,
    /// The pin at `index` of the list breaks `rule`.
    Pin {
        /// The pin's place in the list.
        index: usize,
        /// The rule it breaks.
        rule: PinRule,
    },
}

/// The rule of RFC-0061 'Checked at load' a pin breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PinRule {
    /// `W = 0`.
    Window,
    /// `2·G > W`, or the doubling overflows.
    Grace,
    /// `b > 64`.
    ClassBits,
    /// `from_ns < received_ns`: a pin is never retroactive.
    Retroactive,
    /// `from_ns + W + G`, or the previous pin's `from_ns + W`, overflows.
    Overflow,
    /// It starts before the previous pin's window ends.
    Overlap,
    /// Its `W` is below twice the previous pin's `G`.
    ShortWindow,
}

impl PinError {
    /// The verdict: `Inconclusive(ERR-USAGE)`.
    pub const fn failure(self) -> Failure {
        Failure::Inconclusive(VerifierError {
            code: ErrorCode::Usage,
        })
    }
}

impl fmt::Display for PinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PolicyWindow => f.write_str("the sampler policy's window is 0"),
            Self::PolicyGrace => f.write_str("the sampler policy's grace exceeds half its window"),
            Self::PolicyClassBits => {
                f.write_str("the sampler policy allows more than 64 class bits")
            }
            Self::Pin { index, rule } => write!(f, "sampler pin {index} breaks {rule:?}"),
        }
    }
}

impl std::error::Error for PinError {}
