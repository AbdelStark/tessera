// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public storage bounds for the ordinary batch engine (RFC-0010/0018).
use super::*;
use crate::WindowError;
use crate::terms::Group;

/// Public upper geometry for one ordinary batched instance, before challenges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchShape {
    /// Own variables, without front padding.
    pub vars: usize,
    /// Literal source tables, including public and unused ones; zero for computed rounds.
    pub tables: usize,
    /// Sum-of-products terms; zero denotes the product of all literal tables.
    pub terms: usize,
    /// Every sent final evaluation, without assuming relation/point coincidences.
    pub evaluations: usize,
    /// Whether the instance delegates to a Computed implementation.
    pub computed: bool,
}
/// Requested-byte bounds for prove_batch's storage, excluding borrowed input
/// instances/tables and their construction, transcript callback/encoding work,
/// delegated ComputedState payloads, stack/runtime and allocator overhead.
/// Counts must cover each instance, table and term before any deduplication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchMemory {
    peak: u64,
    output: u64,
}
impl BatchMemory {
    /// Complete owned batch storage plus constructor/round/final-evaluation scratch.
    /// Delegated computed payloads and the other documented exclusions are additional.
    pub fn peak_bytes(self) -> u64 {
        self.peak
    }
    /// Retained ProvenBatch heap, including round/evaluation output and index/point metadata.
    pub fn output_bytes(self) -> u64 {
        self.output
    }

    /// Derive storage from matching public geometry without allocating or reading
    /// field/table values. Every table/term/instance is counted, even if runtime
    /// deduplication or equal term weights would reduce actual allocations.
    ///
    /// # Errors
    /// WindowError::Shape for inconsistent geometry and WindowError::Allocation
    /// for overflowing or unaddressable requests. This neither validates a proof
    /// nor admits a CPU route. Protocol masks/committed source storage are separate.
    pub fn new<F: PolyField>(degree: usize, shapes: &[BatchShape]) -> Result<Self, WindowError> {
        if shapes.is_empty() || degree == 0 || degree > MAX_DEGREE {
            return Err(WindowError::Shape(
                "a nonempty batch needs a supported degree",
            ));
        }
        let instances = shapes.len() as u64;
        let (mut tables, mut evaluations, mut rounds, mut max_tables, mut max_terms) =
            (0, 0, 0, 0, 0);
        let (mut payload, mut plans, mut evaluation_scratch, mut states) = (0, 0, 0, 0);
        for s in shapes {
            if s.vars >= usize::BITS as usize
                || s.computed && (s.tables != 0 || s.terms != 0 || s.vars > 62)
                || !s.computed
                    && (s.tables == 0
                        || s.evaluations > s.tables
                        || s.terms == 0 && s.tables > degree)
            {
                return Err(WindowError::Shape(
                    "batch counts cover the canonical instance",
                ));
            }
            rounds = rounds.max(s.vars as u64);
            evaluations = add(evaluations, s.evaluations as u64)?;
            if s.computed {
                payload = add(payload, array::<F>(2)?)?; // BatchState::Computed's front-padding sums
                continue;
            }
            let t = s.tables as u64;
            let terms = s.terms as u64;
            tables = add(tables, t)?;
            max_tables = max_tables.max(t);
            max_terms = max_terms.max(terms);
            let n = power(s.vars)?;
            // Read bindings retain half a table; unread final evaluations retain one value.
            payload = add(payload, mul(t, array::<F>((n / 2).max(1))?)?)?;
            let low = s.vars.min(10); // pinned tessera-poly blocked evaluator
            evaluation_scratch =
                evaluation_scratch.max(array::<F>(add(power(low)?, power(s.vars - low)?)?)?);
            // State's bound indices, temporary read flags, and literal/compiled table views.
            states = add(states, add(array::<usize>(t)?, array::<bool>(t)?)?)?;
            states = add(
                states,
                if terms == 0 {
                    mul(2, grow::<&[F]>(t)?)?
                } else {
                    array::<Entries<'_, F>>(t)?
                },
            )?;
            if terms != 0 {
                // Plan::new's counts/common/rests/read/lines/slot/common-slots.
                let counts = add(array::<Vec<usize>>(terms)?, mul(terms, array::<usize>(t)?)?)?;
                let rests = add(
                    array::<Vec<usize>>(terms)?,
                    mul(2, separate::<usize>(mul(terms, degree as u64)?, terms)?)?,
                )?;
                let columns = add(
                    mul(2, array::<usize>(t)?)?,
                    add(array::<bool>(t)?, grow::<usize>(t)?)?,
                )?;
                let groups = add(
                    grow::<Group<F>>(terms)?,
                    separate::<Vec<usize>>(terms, terms)?,
                )?;
                plans = add(
                    plans,
                    add(
                        add(counts, rests)?,
                        add(add(columns, grow::<usize>(degree as u64)?)?, groups)?,
                    )?,
                )?;
            }
        }
        let output = add(
            add(
                array::<Vec<F>>(rounds)?,
                array::<F>(mul(rounds, (degree + 1) as u64)?)?,
            )?,
            add(
                add(array::<F>(rounds)?, grow::<F>(evaluations)?)?,
                add(grow::<usize>(instances)?, array::<Range<usize>>(instances)?)?,
            )?,
        )?;
        let batch = add(
            array::<BatchState<'_, F>>(instances)?,
            add(
                array::<usize>(instances)?,
                add(
                    grow::<usize>(instances)?,
                    add(
                        grow::<F>(instances)?,
                        add(
                            btree::<(u8, usize, usize), Vec<usize>>(instances)?,
                            separate::<usize>(instances, instances)?,
                        )?,
                    )?,
                )?,
            )?,
        )?;
        let bindings = add(
            grow::<Binding<'_, F>>(tables)?,
            btree::<(u8, usize, usize), usize>(tables)?,
        )?;
        let mut scratch = add(
            array::<[F; NODES]>(kernel::BLOCK as u64)?,
            array::<[F::Base; NODES]>(kernel::BLOCK as u64)?,
        )?;
        if max_terms != 0 {
            let row = mul(kernel::TERM_BLOCK as u64, NODES as u64)?;
            scratch = add(
                scratch,
                add(
                    grow::<F>(mul(max_tables, row)?)?,
                    add(mul(3, grow::<F>(row)?)?, grow::<[F; NODES]>(max_terms)?)?,
                )?,
            )?;
        }
        // At most one instance's bound views and one computed-round value vector at a time.
        let transient = add(
            add(
                grow::<&[F]>(max_tables)?,
                grow::<Entries<'_, F>>(max_tables)?,
            )?,
            array::<F>(NODES as u64)?,
        )?;
        let prefix = "group/".len() + 10 + "/tree/".len() + 10;
        let label = grow::<u8>((prefix + "/challenge/".len() + 20) as u64)?;
        let peak = add(
            output,
            add(
                batch,
                add(
                    bindings,
                    add(
                        states,
                        add(
                            plans,
                            add(
                                payload,
                                add(scratch, add(evaluation_scratch, add(transient, label)?)?)?,
                            )?,
                        )?,
                    )?,
                )?,
            )?,
        )?;
        Ok(Self { peak, output })
    }
}
fn add(a: u64, b: u64) -> Result<u64, WindowError> {
    a.checked_add(b).ok_or(WindowError::Allocation)
}
fn mul(a: u64, b: u64) -> Result<u64, WindowError> {
    a.checked_mul(b).ok_or(WindowError::Allocation)
}
fn power(bits: usize) -> Result<u64, WindowError> {
    u32::try_from(bits)
        .ok()
        .and_then(|b| 1u64.checked_shl(b))
        .ok_or(WindowError::Allocation)
}
fn array<T>(n: u64) -> Result<u64, WindowError> {
    let bytes = mul(n, size_of::<T>() as u64)?;
    if usize::try_from(n).is_err() || bytes > isize::MAX as u64 {
        return Err(WindowError::Allocation);
    }
    Ok(bytes)
}
fn grow<T>(n: u64) -> Result<u64, WindowError> {
    if n == 0 {
        return Ok(0);
    }
    let min = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    array::<T>(mul(4, n.max(min))?)
}
fn separate<T>(n: u64, vectors: u64) -> Result<u64, WindowError> {
    array::<T>(mul(4, add(n, mul(4, vectors)?)?)?)
}
fn btree<K, V>(n: u64) -> Result<u64, WindowError> {
    if n == 0 {
        return Ok(0);
    }
    // Rust1.95 alloc/btree/node.rs: B=6, eleven keys/values, twelve edges,
    // parent pointer + two u16s. Seven alignment gaps bound both repr(C) levels.
    // https://github.com/rust-lang/rust/blob/1.95.0/library/alloc/src/collections/btree/node.rs
    let align = align_of::<K>()
        .max(align_of::<V>())
        .max(align_of::<usize>()) as u64;
    let node = add(
        add(
            size_of::<usize>() as u64 + 4,
            mul(11, (size_of::<K>() + size_of::<V>()) as u64)?,
        )?,
        add(array::<usize>(12)?, mul(7, align - 1)?)?,
    )?;
    // Each occupied node has a key; cover an additional transient root/split node.
    mul(mul(2, add(n, 1)?)?, node)
}

#[cfg(test)]
mod tests {
    use super::*;
    use allocation_counter::measure;
    use tessera_field::GoldilocksExt2 as F;

    #[test]
    fn batch_storage_refuses_overflow_and_invalid_geometry_without_allocations() {
        let valid = BatchShape {
            vars: 3,
            tables: 2,
            terms: 1,
            evaluations: 1,
            computed: false,
        };
        let measured = measure(|| {
            assert_eq!(
                array::<F>(isize::MAX as u64 / size_of::<F>() as u64 + 1),
                Err(WindowError::Allocation)
            );
            assert_eq!(
                array::<F>(u64::MAX / size_of::<F>() as u64 + 1),
                Err(WindowError::Allocation)
            );
            assert_eq!(grow::<F>(u64::MAX), Err(WindowError::Allocation));
            assert_eq!(
                btree::<(u8, usize, usize), Vec<usize>>(u64::MAX),
                Err(WindowError::Allocation)
            );
            assert!(matches!(
                BatchMemory::new::<F>(0, &[valid]),
                Err(WindowError::Shape(_))
            ));
            assert!(matches!(
                BatchMemory::new::<F>(3, &[]),
                Err(WindowError::Shape(_))
            ));
            assert!(matches!(
                BatchMemory::new::<F>(
                    3,
                    &[BatchShape {
                        vars: usize::MAX,
                        ..valid
                    }]
                ),
                Err(WindowError::Shape(_))
            ));
            assert!(matches!(
                BatchMemory::new::<F>(
                    3,
                    &[BatchShape {
                        computed: true,
                        ..valid
                    }]
                ),
                Err(WindowError::Shape(_))
            ));
            assert!(matches!(
                BatchMemory::new::<F>(
                    3,
                    &[BatchShape {
                        evaluations: 3,
                        ..valid
                    }]
                ),
                Err(WindowError::Shape(_))
            ));
            assert_eq!(
                BatchMemory::new::<F>(
                    3,
                    &[BatchShape {
                        tables: usize::MAX,
                        evaluations: 0,
                        ..valid
                    }]
                ),
                Err(WindowError::Allocation)
            );
            assert!(BatchMemory::new::<F>(3, &[valid]).is_ok());
        });
        assert_eq!(measured.count_total, 0);
    }

    #[test]
    fn pinned_btree_nodes_fit_both_actual_batch_key_value_types() {
        for n in [0usize, 1, 2, 5, 6, 11, 12, 13, 31, 64, 129, 513, 4097] {
            let mut ids = BTreeMap::new();
            let measured = measure(|| {
                for j in 0..n {
                    ids.insert((0u8, j, 1usize), j);
                }
            });
            assert!(
                measured.bytes_total <= btree::<(u8, usize, usize), usize>(n as u64).unwrap(),
                "index {n} {measured:?}"
            );
            let values = (0..n).map(|j| vec![j]).collect::<Vec<_>>();
            let mut buckets = BTreeMap::new();
            let measured = measure(|| {
                for (j, value) in values.into_iter().enumerate() {
                    buckets.insert((0u8, j, 1usize), value);
                }
            });
            assert!(
                measured.bytes_total <= btree::<(u8, usize, usize), Vec<usize>>(n as u64).unwrap(),
                "buckets {n} {measured:?}"
            );
        }
    }
}
