// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public-size parallel traversal of the unchanged radix-2 tree (RFC-0011 Encoding).
//! Workers borrow disjoint slices and immutable twiddles. Control allocations
//! happen on the caller thread; each child runs an allocation-free scalar subtree.

use super::*;
use core::num::NonZeroUsize;
use std::{sync::Mutex, thread};

// Small whole-proof controls exercise the same fork tree with test-rng.
// Only public scheduling changes; test-rng cannot compile in release (OBS-4).
#[cfg(feature = "test-rng")]
const PARALLEL_MIN: usize = 1 << 9;
#[cfg(not(feature = "test-rng"))]
const PARALLEL_MIN: usize = 1 << 16;

pub(super) const STACK_BYTES: usize = 2 << 20;
// Pinned Rust 1.95 scope/packet Arcs, entry/closure boxes, Task/handle vectors,
// parent/child Thread handles and native-handle slack. See cpu-row-ntt.md.
const CONTROL_BYTES: u64 = 8192;

pub(crate) fn memory(n: usize, threads: NonZeroUsize) -> Option<u64> {
    log2_len(n).ok()?;
    (workers(n, threads.get()) as u64)
        .checked_sub(1)?
        .checked_mul((STACK_BYTES as u64).checked_add(CONTROL_BYTES)?)
}

fn workers(n: usize, requested: usize) -> usize {
    if n < PARALLEL_MIN {
        return 1;
    }
    let cap = requested.max(1).min(n / (PARALLEL_MIN / 2));
    // A public power-of-two frontier gives equal disjoint subtrees, at most
    // the requested count, with no copied rows or per-child queue allocations.
    1usize << (usize::BITS - 1 - cap.leading_zeros())
}

fn level<S: Shift, const GS: bool>(v: &mut [Fp], t: &[Fp], shift: &S, groups: usize) {
    let len = v.len() / groups;
    let half = len / 2;
    for (g, chunk) in v.chunks_exact_mut(len).enumerate() {
        let (lo, hi) = chunk.split_at_mut(half);
        if GS {
            if S::UNIT_FIRST && g == 0 {
                gs_unit_butterflies(lo, hi);
            } else {
                gs_butterflies(lo, hi, shift.twiddle(t[g], half.trailing_zeros()));
            }
        } else if S::UNIT_FIRST && g == 0 {
            ct_unit_butterflies(lo, hi);
        } else {
            ct_butterflies(lo, hi, shift.twiddle(t[g], half.trailing_zeros()));
        }
    }
}

type Task<'a> = Mutex<Option<&'a mut [Fp]>>;
type Scalar<S> = fn(&mut [Fp], &[Fp], &S, usize);

// These public-layout caps are part of the pinned control-memory inventory.
const _: () = assert!(size_of::<Task<'static>>() <= 64);
const _: () = assert!(size_of::<thread::ScopedJoinHandle<'static, ()>>() <= 64);

fn run_task<S: Shift>(slot: &Task<'_>, t: &[Fp], s: &S, group: usize, scalar: Scalar<S>) {
    let data = slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
        .expect("one owner of each NTT subtree");
    #[cfg(test)]
    {
        let mut result = None;
        let allocation = allocation_counter::measure(|| {
            result = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || scalar(data, t, s, group),
            )));
        });
        match result.expect("worker measurement ran") {
            Ok(()) => assert_eq!(
                allocation.bytes_total, 0,
                "scalar worker must allocate no field or control buffer"
            ),
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
    #[cfg(not(test))]
    scalar(data, t, s, group);
}

fn execute<S: Shift + Sync>(v: &mut [Fp], t: &[Fp], s: &S, groups: usize, scalar: Scalar<S>) {
    #[cfg(feature = "test-rng")]
    let log = v.len().trailing_zeros() as usize;
    let len = v.len() / groups;
    let mut tasks = Vec::with_capacity(groups);
    for chunk in v.chunks_exact_mut(len) {
        tasks.push(Mutex::new(Some(chunk)));
    }
    // Some platforms lazily allocate native Mutex state. Initialize it on the
    // caller, so child callbacks remain allocation-free and parent metering sees it.
    for slot in &tasks {
        drop(
            slot.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
    }
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(groups - 1);
        let parent = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for (g, slot) in tasks.iter().enumerate().skip(1) {
                match spawn(
                    thread::Builder::new().stack_size(STACK_BYTES),
                    scope,
                    move || run_task(slot, t, s, g, scalar),
                ) {
                    Ok(handle) => {
                        #[cfg(feature = "test-rng")]
                        SPAWNS.with(|c| {
                            let mut counts = c.get();
                            counts[log] = counts[log].saturating_add(1);
                            c.set(counts);
                        });
                        handles.push(handle);
                    }
                    Err(_) => run_task(slot, t, s, g, scalar),
                }
            }
            run_task(&tasks[0], t, s, 0, scalar);
        }));
        // Join every native thread, even after a parent or child panic, before
        // dropping the task borrows or resuming unwinding. No partial proof escapes.
        let mut failure = parent.err();
        for handle in handles {
            if let Err(payload) = handle.join()
                && failure.is_none()
            {
                failure = Some(payload);
            }
        }
        if let Some(payload) = failure {
            std::panic::resume_unwind(payload);
        }
    });
}

pub(crate) fn ct_parallel<S: Shift + Sync>(v: &mut [Fp], t: &[Fp], s: &S, requested: usize) {
    let count = workers(v.len(), requested);
    if count == 1 {
        ct_group(v, t, s, 0);
        return;
    }
    let mut groups = 1;
    while groups < count {
        level::<S, false>(v, t, s, groups);
        groups *= 2;
    }
    execute(v, t, s, count, ct_group);
}

pub(crate) fn gs_parallel<S: Shift + Sync>(v: &mut [Fp], t: &[Fp], s: &S, requested: usize) {
    let count = workers(v.len(), requested);
    if count == 1 {
        gs_group(v, t, s, 0);
        return;
    }
    execute(v, t, s, count, gs_group);
    let mut groups = count / 2;
    while groups > 0 {
        level::<S, true>(v, t, s, groups);
        groups /= 2;
    }
}

fn spawn<'scope, 'env, F>(
    builder: thread::Builder,
    scope: &'scope thread::Scope<'scope, 'env>,
    f: F,
) -> std::io::Result<thread::ScopedJoinHandle<'scope, ()>>
where
    F: FnOnce() + Send + 'scope,
{
    #[cfg(test)]
    if FAIL_AFTER.with(|state| match state.get() {
        Some(0) => true,
        Some(n) => {
            state.set(Some(n - 1));
            false
        }
        None => false,
    }) {
        return Err(std::io::ErrorKind::WouldBlock.into());
    }
    // The boxed entry wrapper's remaining fields fit the separately pinned cap.
    const {
        assert!(size_of::<F>() <= 128);
    }
    builder.spawn_scoped(scope, f)
}

#[cfg(test)]
thread_local! { static FAIL_AFTER: core::cell::Cell<Option<usize>> = const { core::cell::Cell::new(None) }; }

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroUsize;

    fn input(n: usize, boundary: bool) -> Vec<Fp> {
        let edges = [
            0,
            1,
            Fp::MODULUS - 1,
            (1 << 32) - 1,
            1 << 32,
            Fp::MODULUS - (1 << 32),
            u64::MAX,
        ];
        let mut state = 0x7e55_e7a0_0659_0002u64;
        (0..n)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                Fp::from_u64_reduced(if boundary {
                    edges[i % edges.len()]
                } else {
                    state
                })
            })
            .collect()
    }

    fn compare<S: Shift + Sync>(v: &[Fp], t: &[Fp], shift: &S, count: usize) {
        let mut expected = v.to_vec();
        let mut actual = v.to_vec();
        cooley_tukey(&mut expected, t, shift);
        ct_parallel(&mut actual, t, shift, count);
        assert_eq!(actual, expected, "CT n={} workers={count}", v.len());
        expected.copy_from_slice(v);
        actual.copy_from_slice(v);
        gentleman_sande(&mut expected, t, shift);
        gs_parallel(&mut actual, t, shift, count);
        assert_eq!(actual, expected, "GS n={} workers={count}", v.len());
    }

    #[test]
    fn parallel_subgroup_coset_and_inverse_tables_match_scalar() {
        for log in [0, 1, 9, 12, 16, 17] {
            for inverse in [false, true] {
                let t = twiddles(log, inverse);
                for boundary in [false, true] {
                    let v = input(1 << log, boundary);
                    for count in [1, 2, 3, 4, 8] {
                        compare(&v, &t, &Subgroup, count);
                        compare(&v, &t, &Coset::new(Fp::from_u64_reduced(7)), count);
                    }
                }
            }
        }
    }

    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            FAIL_AFTER.with(|s| s.set(None));
        }
    }

    #[test]
    fn spawn_failures_finish_unclaimed_work_and_join_successful_siblings() {
        let v = input(1 << 17, false);
        let t = twiddles(17, false);
        for after in [0, 1, 2] {
            FAIL_AFTER.with(|s| s.set(Some(after)));
            let _reset = Reset;
            compare(&v, &t, &Subgroup, 8);
            compare(&v, &t, &Coset::new(Fp::from_u64_reduced(7)), 8);
        }
    }

    struct Completion<'a> {
        ready: &'a std::sync::atomic::AtomicUsize,
        finished: &'a std::sync::atomic::AtomicUsize,
        panic_at: usize,
    }
    impl Shift for Completion<'_> {
        const UNIT_FIRST: bool = false;
        fn twiddle(&self, z: Fp, _: u32) -> Fp {
            z
        }
    }
    fn observed(v: &mut [Fp], _: &[Fp], s: &Completion<'_>, g: usize) {
        use std::sync::atomic::Ordering::SeqCst;
        s.ready.fetch_add(1, SeqCst);
        assert!(g != s.panic_at, "controlled worker panic");
        v.fill(Fp::ONE);
        s.finished.fetch_add(1, SeqCst);
    }
    #[test]
    fn parent_and_worker_panics_join_all_siblings_and_propagate() {
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
        for panic_at in [0, 1] {
            let ready = AtomicUsize::new(0);
            let finished = AtomicUsize::new(0);
            let shift = Completion {
                ready: &ready,
                finished: &finished,
                panic_at,
            };
            let mut values = vec![Fp::ZERO; 16];
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                execute(&mut values, &[], &shift, 4, observed)
            }));
            assert_eq!(ready.load(SeqCst), 4, "every task entered");
            assert!(
                result.is_err(),
                "never publish a partial transform after a panic"
            );
            assert_eq!(
                finished.load(SeqCst),
                3,
                "all non-panicking siblings joined"
            );
        }
    }

    #[test]
    fn thread_charge_is_zero_for_scalar_and_bounds_public_forks() {
        assert_eq!(memory(1 << 22, NonZeroUsize::MIN), Some(0));
        assert_eq!(memory(1, NonZeroUsize::new(8).unwrap()), Some(0));
        assert_eq!(memory(3, NonZeroUsize::MIN), None);
        assert_eq!(memory(0, NonZeroUsize::MIN), None);
        let per = STACK_BYTES as u64 + CONTROL_BYTES;
        assert_eq!(
            memory(1 << 22, NonZeroUsize::new(4).unwrap()),
            Some(3 * per)
        );
        assert_eq!(workers(PARALLEL_MIN, usize::MAX), 2);
    }
}

#[cfg(feature = "test-rng")]
thread_local! { static SPAWNS: core::cell::Cell<[usize;33]> = const { core::cell::Cell::new([0;33]) }; }

#[cfg(feature = "test-rng")]
pub(crate) fn test_spawns(n: usize) -> usize {
    log2_len(n).map_or(0, |log| SPAWNS.with(|c| c.get()[log as usize]))
}
