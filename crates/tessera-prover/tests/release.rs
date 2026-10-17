// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The release schedule of RFC-0018 (INV-ARCH-2). A message leaves at its budget when its work
//! meets it, otherwise at the first doubling the work meets, and the session aborts past the last
//! slot. Every duration that meets a slot leaves at the same instant, so a verifier that times
//! the message learns the slot and nothing else.

use core::time::Duration;

use tessera_prover::{Release, ReleaseSchedule, ScheduleError};

const NS: Duration = Duration::from_nanos(1);

#[test]
fn every_duration_within_a_slot_leaves_at_the_same_instant() {
    let budget = Duration::from_millis(1500);
    let schedule = ReleaseSchedule::new(budget, 3).expect("a schedule");
    // Work done by the budget, however fast, leaves at the budget.
    for elapsed in [Duration::ZERO, NS, budget / 3, budget - NS, budget] {
        assert_eq!(
            schedule.release(elapsed),
            Some(Release {
                after: budget,
                slot: 0
            }),
            "{elapsed:?}"
        );
    }
    // Each later slot holds everything from just past the slot before it to itself.
    for slot in 1..=3u32 {
        let (before, at) = (budget * (1 << (slot - 1)), budget * (1 << slot));
        for elapsed in [before + NS, (before + at) / 2, at] {
            assert_eq!(
                schedule.release(elapsed),
                Some(Release { after: at, slot }),
                "{elapsed:?}"
            );
        }
    }
    assert_eq!(schedule.last_slot(), budget * 8);
    assert_eq!(
        schedule.release(budget * 8 + NS),
        None,
        "past the last slot"
    );
}

/// Over many durations, the release is the least slot the work meets: never before the work is
/// done, and never a slot later than needed.
#[test]
fn the_release_is_the_first_slot_the_work_meets() {
    let schedule = ReleaseSchedule::new(Duration::from_micros(777), 5).expect("a schedule");
    let last = schedule.last_slot().as_nanos() as u64;
    let mut state = 0x0065_0257_u64;
    for _ in 0..10_000 {
        // SplitMix64, over durations up to a little past the last slot.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let elapsed = Duration::from_nanos((z ^ (z >> 31)) % (last + last / 8));
        match schedule.release(elapsed) {
            Some(Release { after, slot }) => {
                assert_eq!(after, schedule.budget() * (1 << slot), "{elapsed:?}");
                assert!(elapsed <= after, "{elapsed:?} leaves before it is done");
                assert!(
                    slot == 0 || elapsed > after / 2,
                    "{elapsed:?} waits a slot longer than it needs"
                );
            }
            None => assert!(elapsed > schedule.last_slot(), "{elapsed:?} aborted"),
        }
    }
}

#[test]
fn a_schedule_without_doublings_has_its_budget_only() {
    let budget = Duration::from_secs(2);
    let schedule = ReleaseSchedule::new(budget, 0).expect("a schedule");
    assert_eq!(schedule.last_slot(), budget);
    assert_eq!(
        schedule.release(budget),
        Some(Release {
            after: budget,
            slot: 0
        })
    );
    assert_eq!(schedule.release(budget + NS), None);
}

#[test]
fn schedules_that_cannot_hold_are_refused() {
    assert_eq!(
        ReleaseSchedule::new(Duration::ZERO, 3),
        Err(ScheduleError::ZeroBudget)
    );
    assert_eq!(
        ReleaseSchedule::new(Duration::MAX, 1),
        Err(ScheduleError::LastSlotOverflows)
    );
    assert_eq!(
        ReleaseSchedule::new(NS, 32),
        Err(ScheduleError::LastSlotOverflows),
        "2^32 does not fit the factor"
    );
    // The largest budget whose last slot still fits.
    let budget = Duration::MAX / 4;
    let schedule = ReleaseSchedule::new(budget, 2).expect("a schedule");
    assert_eq!(schedule.last_slot(), budget * 4);
    assert_eq!(
        ReleaseSchedule::new(budget + NS, 2),
        Err(ScheduleError::LastSlotOverflows)
    );
}
