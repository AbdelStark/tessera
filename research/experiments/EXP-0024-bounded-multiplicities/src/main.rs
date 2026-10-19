// SPDX-License-Identifier: Apache-2.0 OR MIT

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::mem::size_of;

use allocation_counter::{AllocationInfo, measure};
use serde_json::{Value, json};
use tessera_codec::SecretBuf;

const P: u64 = 0xffff_ffff_0000_0001;
const SEED: u64 = 0x0112_0024_0050_0001;

#[derive(Clone, Copy)]
struct Update {
    row: u32,
    amount: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Error {
    Shape,
    Key,
    Capacity,
    Overflow,
    Field,
    Memory,
}

enum Storage {
    Dense(SecretBuf<u64>),
    Sorted(SecretBuf<Update>),
}

struct Counter {
    actual: u64,
    padded: u64,
    limit: usize,
    calls: usize,
    total: u64,
    storage: Storage,
}

impl Counter {
    fn new(actual: u64, limit: usize, dense: bool) -> Result<Self, Error> {
        if actual == 0 || actual > 1 << 32 {
            return Err(Error::Shape);
        }
        let padded = actual.next_power_of_two();
        let storage = if dense {
            let len = usize::try_from(padded).map_err(|_| Error::Memory)?;
            let mut values = SecretBuf::try_with_capacity(len).map_err(|_| Error::Memory)?;
            values.extend(std::iter::repeat_n(0, len));
            Storage::Dense(values)
        } else {
            limit
                .checked_mul(size_of::<Update>())
                .ok_or(Error::Memory)?;
            Storage::Sorted(SecretBuf::try_with_capacity(limit).map_err(|_| Error::Memory)?)
        };
        Ok(Self {
            actual,
            padded,
            limit,
            calls: 0,
            total: 0,
            storage,
        })
    }

    fn add(&mut self, update: Update) -> Result<(), Error> {
        if u64::from(update.row) >= self.actual {
            return Err(Error::Key);
        }
        if self.calls == self.limit {
            return Err(Error::Capacity);
        }
        let total = self
            .total
            .checked_add(update.amount)
            .ok_or(Error::Overflow)?;
        if total >= P {
            return Err(Error::Field);
        }
        match &mut self.storage {
            Storage::Dense(values) => values[update.row as usize] += update.amount,
            Storage::Sorted(values) => values.push(update),
        }
        self.calls += 1;
        self.total = total;
        Ok(())
    }

    fn finish(&mut self) {
        if let Storage::Sorted(values) = &mut self.storage {
            values.sort_unstable_by_key(|update| update.row);
        }
    }

    fn get(&self, row: u64) -> Option<u64> {
        if row >= self.padded {
            return None;
        }
        Some(match &self.storage {
            Storage::Dense(values) => values[row as usize],
            Storage::Sorted(values) => {
                let start = values.partition_point(|update| u64::from(update.row) < row);
                values[start..]
                    .iter()
                    .take_while(|update| u64::from(update.row) == row)
                    .map(|update| update.amount)
                    .sum()
            }
        })
    }

    fn metadata(&self) -> (bool, u64, usize, usize) {
        match &self.storage {
            Storage::Dense(values) => (
                true,
                self.padded,
                self.limit,
                values.capacity() * size_of::<u64>(),
            ),
            Storage::Sorted(values) => (
                false,
                self.padded,
                self.limit,
                values.capacity() * size_of::<Update>(),
            ),
        }
    }
}

fn dense_mode(actual: u64, limit: usize) -> bool {
    u128::from(actual.next_power_of_two()) * 8 <= (limit as u128) * size_of::<Update>() as u128
}

fn built(actual: u64, limit: usize, updates: &[Update], dense: bool) -> Counter {
    let mut out = Counter::new(actual, limit, dense).expect("public shape fits");
    for &update in updates {
        out.add(update).expect("valid synthetic update");
    }
    out.finish();
    out
}

fn window(
    actual: u64,
    start: u64,
    width: usize,
    updates: &[Update],
    limit: usize,
) -> Result<(SecretBuf<u64>, u64), Error> {
    if actual == 0
        || actual > 1 << 32
        || width == 0
        || start
            .checked_add(width as u64)
            .is_none_or(|end| end > actual.next_power_of_two())
    {
        return Err(Error::Shape);
    }
    if updates.len() > limit {
        return Err(Error::Capacity);
    }
    let mut out = SecretBuf::filled(0u64, width);
    let mut visits = 0;
    let mut total = 0u64;
    for update in updates {
        if u64::from(update.row) >= actual {
            return Err(Error::Key);
        }
        total = total.checked_add(update.amount).ok_or(Error::Overflow)?;
        if total >= P {
            return Err(Error::Field);
        }
        visits += 1;
        if let Some(offset) = u64::from(update.row)
            .checked_sub(start)
            .filter(|&n| n < width as u64)
        {
            out[offset as usize] = out[offset as usize]
                .checked_add(update.amount)
                .expect("bounded sum");
            assert!(out[offset as usize] < P);
        }
    }
    Ok((out, visits))
}

fn reference(updates: &[Update]) -> BTreeMap<u64, u128> {
    let mut out = BTreeMap::new();
    for update in updates {
        *out.entry(u64::from(update.row)).or_default() += u128::from(update.amount);
    }
    out
}

fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn inputs(actual: u64, limit: usize, pattern: usize, state: &mut u64) -> Vec<Update> {
    let mut out = Vec::new();
    let mut count = 0u64;
    for i in 0..limit.saturating_sub(1) {
        let row = match pattern {
            0 => 0,
            1 => actual - 1,
            2 => (i as u64 % 2) * (actual - 1),
            3 => i as u64 % actual,
            _ => random(state) % actual,
        };
        let amount = if i % 11 == 0 { 0 } else { 1 << (i % 4) };
        out.push(Update {
            row: row as u32,
            amount,
        });
        count += amount;
    }
    if limit > 0 {
        let tail = if count == 0 {
            0
        } else {
            count.next_power_of_two() - count
        };
        out.push(Update {
            row: 0,
            amount: tail,
        });
    }
    out
}

fn functional() -> Value {
    let hand = [
        Update { row: 0, amount: 2 },
        Update { row: 2, amount: 3 },
        Update { row: 2, amount: 4 },
        Update { row: 0, amount: 7 },
    ];
    for dense in [false, true] {
        let counter = built(3, hand.len(), &hand, dense);
        assert_eq!(
            (0..4)
                .map(|row| counter.get(row).unwrap())
                .collect::<Vec<_>>(),
            [9, 0, 7, 0]
        );
    }
    let mut state = SEED;
    let mut checks = 0u64;
    let mut cases = 0u64;
    let mut replay_visits = 0u64;
    for actual in [1u64, 2, 3, 128, 256, 768, 65_536, 1_048_576, 1 << 32] {
        for limit in [0usize, 1, 17, 513, 4_096] {
            let mut metadata = None;
            for pattern in 0..5 {
                let updates = inputs(actual, limit, pattern, &mut state);
                if let Some(tail) = updates.last() {
                    let unpadded: u128 = updates[..updates.len() - 1]
                        .iter()
                        .map(|update| u128::from(update.amount))
                        .sum();
                    let complete: u128 =
                        updates.iter().map(|update| u128::from(update.amount)).sum();
                    assert_eq!(tail.row, 0, "column tail repeats row zero");
                    assert_eq!(
                        complete,
                        if unpadded == 0 {
                            0
                        } else {
                            unpadded.next_power_of_two()
                        }
                    );
                }
                let expected = reference(&updates);
                let sorted = built(actual, limit, &updates, false);
                let mode = dense_mode(actual, limit);
                let chosen = if mode {
                    built(actual, limit, &updates, true).metadata()
                } else {
                    sorted.metadata()
                };
                assert_eq!(metadata.get_or_insert(chosen), &chosen);
                assert_eq!(sorted.get(sorted.padded), None);
                let row = |at: u64| -> u64 {
                    u64::try_from(expected.get(&at).copied().unwrap_or(0)).unwrap()
                };
                if actual <= 1_048_576 {
                    let dense = built(actual, limit, &updates, true);
                    let width = 1_024usize.min(sorted.padded as usize);
                    for start in (0..sorted.padded).step_by(width) {
                        let (values, visits) =
                            window(actual, start, width, &updates, limit).unwrap();
                        replay_visits += visits;
                        for (offset, &count) in values.iter().enumerate() {
                            let at = start + offset as u64;
                            assert_eq!(dense.get(at), Some(row(at)));
                            assert_eq!(sorted.get(at), Some(row(at)));
                            assert_eq!(count, row(at));
                            checks += 3;
                        }
                    }
                } else {
                    for &key in expected.keys() {
                        for at in key.saturating_sub(1)..=(key + 1).min(sorted.padded - 1) {
                            assert_eq!(sorted.get(at), Some(row(at)));
                            checks += 1;
                        }
                    }
                    for _ in 0..64 {
                        let start = (random(&mut state) % (sorted.padded / 64)) * 64;
                        let (values, visits) = window(actual, start, 64, &updates, limit).unwrap();
                        replay_visits += visits;
                        for (offset, &count) in values.iter().enumerate() {
                            let at = start + offset as u64;
                            assert_eq!(sorted.get(at), Some(row(at)));
                            assert_eq!(count, row(at));
                            checks += 2;
                        }
                    }
                }
                cases += 1;
            }
        }
    }
    json!({"cases": cases, "exact_checks": checks, "window_replay_visits": replay_visits})
}

fn rejections() {
    let bad_key = [Update { row: 3, amount: 1 }];
    assert!(matches!(window(3, 0, 4, &bad_key, 1), Err(Error::Key)));
    assert!(matches!(window(3, 0, 1, &bad_key, 1), Err(Error::Key)));
    let modulus = [
        Update {
            row: 0,
            amount: P - 1,
        },
        Update { row: 1, amount: 1 },
    ];
    assert!(matches!(window(3, 0, 1, &modulus, 2), Err(Error::Field)));
    let overflow = [
        Update {
            row: 0,
            amount: P - 1,
        },
        Update {
            row: 1,
            amount: P - 1,
        },
    ];
    assert!(matches!(
        window(3, 0, 1, &overflow, 2),
        Err(Error::Overflow)
    ));
    assert!(matches!(window(3, 0, 4, &modulus, 1), Err(Error::Capacity)));
    assert!(matches!(window(3, 4, 1, &[], 0), Err(Error::Shape)));
    assert!(matches!(window(0, 0, 1, &[], 0), Err(Error::Shape)));
    for dense in [false, true] {
        let mut counter = Counter::new(3, 4, dense).unwrap();
        assert_eq!(counter.add(Update { row: 3, amount: 1 }), Err(Error::Key));
        assert_eq!(counter.add(Update { row: 4, amount: 1 }), Err(Error::Key));
        assert_eq!(
            counter.add(Update {
                row: 0,
                amount: u64::MAX
            }),
            Err(Error::Field)
        );
        counter
            .add(Update {
                row: 2,
                amount: P - 1,
            })
            .unwrap();
        assert_eq!(counter.add(Update { row: 2, amount: 1 }), Err(Error::Field));
        assert_eq!(
            counter.add(Update {
                row: 2,
                amount: P - 1
            }),
            Err(Error::Overflow)
        );
        counter.finish();
        assert_eq!(counter.total, P - 1);
        assert_eq!(counter.get(2), Some(P - 1));
        assert_eq!(
            Counter::new(1, 0, dense)
                .unwrap()
                .add(Update { row: 0, amount: 0 }),
            Err(Error::Capacity)
        );
    }
    assert!(matches!(Counter::new(0, 0, false), Err(Error::Shape)));
    assert!(matches!(
        Counter::new((1 << 32) + 1, 0, false),
        Err(Error::Shape)
    ));
    assert!(matches!(
        Counter::new(1, usize::MAX, false),
        Err(Error::Memory)
    ));
}

fn allocation_sample(actual: u64, limit: usize, updates: &[Update], dense: bool) -> AllocationInfo {
    let sample = measure(|| {
        let mut counter = Counter::new(actual, limit, dense).unwrap();
        let add_sort = measure(|| {
            for &update in updates {
                counter.add(update).unwrap();
            }
            counter.finish();
        });
        assert_eq!(
            add_sort.count_total, 0,
            "insertion/sort makes no heap allocation"
        );
        assert_eq!(add_sort.bytes_total, 0);
        assert_eq!(
            counter.metadata().3 as u64,
            if dense {
                actual.next_power_of_two() * 8
            } else {
                (limit * size_of::<Update>()) as u64
            }
        );
        std::hint::black_box(counter.get(actual - 1));
    });
    assert_eq!(sample.bytes_current, 0);
    assert_eq!(sample.count_current, 0);
    sample
}

fn allocation_report(actual: u64, updates: &[Update], dense: bool) -> Value {
    for _ in 0..3 {
        allocation_sample(actual, updates.len(), updates, dense);
    }
    let samples: Vec<_> = (0..10)
        .map(|_| allocation_sample(actual, updates.len(), updates, dense))
        .collect();
    let mut bytes: Vec<_> = samples.iter().map(|s| s.bytes_max).collect();
    let mut counts: Vec<_> = samples.iter().map(|s| s.count_total).collect();
    bytes.sort_unstable();
    counts.sort_unstable();
    // Allocation sizes/counts must be deterministic; timing is not a criterion.
    assert!(bytes.iter().all(|&n| n == bytes[0]));
    assert!(counts.iter().all(|&n| n == counts[0]));
    if actual == 1 << 32 {
        assert!(bytes[9] <= 128 * 1_024);
    }
    json!({"actual_rows": actual, "updates": updates.len(), "dense": dense,
        "bytes_min": bytes[0], "bytes_median": bytes[5], "bytes_max": bytes[9],
        "allocations_min": counts[0], "allocations_median": counts[5], "allocations_max": counts[9],
        "cv": 0, "retained_bytes": 0, "retained_allocations": 0})
}

fn window_allocation_report(updates: &[Update]) -> Value {
    let sample = || {
        let info = measure(|| {
            let (values, visits) = window(65_536, 0, 1_024, updates, updates.len()).unwrap();
            assert_eq!(visits as usize, updates.len());
            std::hint::black_box(values);
        });
        assert_eq!(info.bytes_current, 0);
        assert_eq!(info.count_current, 0);
        info
    };
    for _ in 0..3 {
        sample();
    }
    let samples: Vec<_> = (0..10).map(|_| sample()).collect();
    assert!(samples.iter().all(|s| s.bytes_max == samples[0].bytes_max && s.count_total == samples[0].count_total));
    json!({"arm":"window", "actual_rows":65_536, "width":1_024,
        "updates":updates.len(), "bytes_min":samples[0].bytes_max,
        "bytes_median":samples[5].bytes_max, "bytes_max":samples[9].bytes_max,
        "allocations_min":samples[0].count_total, "allocations_median":samples[5].count_total,
        "allocations_max":samples[9].count_total, "cv":0,
        "retained_bytes":0, "retained_allocations":0})
}

fn main() {
    rejections();
    let correctness = functional();
    let mut state = SEED;
    let small = inputs(65_536, 4_096, 4, &mut state);
    let wide = inputs(1 << 32, 4_096, 4, &mut state);
    let reports = [
        allocation_report(65_536, &small, true),
        allocation_report(65_536, &small, false),
        allocation_report(1 << 32, &wide, false),
        window_allocation_report(&small),
    ];
    println!(
        "{}",
        json!({"schema":"tessera.research.bounded-multiplicities/1", "seed":SEED,
        "os":std::env::consts::OS, "arch":std::env::consts::ARCH,
        "update_bytes":size_of::<Update>(), "correctness":correctness,
        "allocation_reports":reports, "wide_dense_bytes_e0":(1u64 << 32) * 8,
        "scope":"E0 equivalence and E1 counter allocations only; no production proof, timing ZK or GPU claim"})
    );
}
