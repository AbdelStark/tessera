// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's one wall clock for the `max_wall_time` budget (RFC-0019, resource limits).
//!
//! Every deadline read goes through [`now`]. A release build reads `Instant::now()` and nothing
//! else. A `test-params` build adds a per-thread offset that [`advance`] moves forward, so a test
//! can expire a session budget at an exact check instead of racing a sleep against a loaded host
//! (#631). The verifier does its work on the calling thread, so the offset reaches every read.

use std::time::{Duration, Instant};

#[cfg(feature = "test-params")]
thread_local! {
    static OFFSET: std::cell::Cell<Duration> = const { std::cell::Cell::new(Duration::ZERO) };
}

/// The current instant, as the session budget measures it.
#[must_use]
pub(crate) fn now() -> Instant {
    #[cfg(feature = "test-params")]
    {
        let offset = OFFSET.with(std::cell::Cell::get);
        // An offset the platform's `Instant` cannot represent reads as the unshifted instant,
        // which only makes a test's budget expire later and its expiry assertion fail.
        Instant::now()
            .checked_add(offset)
            .unwrap_or_else(Instant::now)
    }
    #[cfg(not(feature = "test-params"))]
    Instant::now()
}

/// Time measured from `start` on this clock; zero when `start` lies ahead of it.
#[must_use]
pub(crate) fn elapsed(start: Instant) -> Duration {
    now().saturating_duration_since(start)
}

/// Move this thread's verifier clock forward by `by`. Test-only: release builds do not have it.
#[cfg(feature = "test-params")]
pub fn advance(by: Duration) {
    OFFSET.with(|offset| offset.set(offset.get().saturating_add(by)));
}
