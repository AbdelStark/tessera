// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public constructor/preparation bounds for owned native storage (RFC-0018/0056).
use super::*;

/// Requested-byte footprint upper bounds for native construction, one initial
/// prepare, and the metadata of one instance per native tag. Borrowed plan,
/// lowering, leaf/model witnesses and canonical sources remain separate, as do
/// sumcheck/computed round state, windows, claims/openings, stack/runtime and
/// allocator overhead. This neither authenticates metadata nor admits a route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeMemory {
    construction: u64,
    prepared: u64,
    instances: u64,
}
impl NativeMemory {
    /// Upper bound on native-owned storage plus constructor temporaries.
    pub fn construction_peak_bytes(self) -> u64 {
        self.construction
    }
    /// Upper bound during the first prepare, including constructed native storage.
    pub fn prepared_peak_bytes(self) -> u64 {
        self.prepared
    }
    /// Additional metadata requests for one instance of every native tag.
    /// Instances borrow native tables; round/window state is separate.
    pub fn instance_bytes(self) -> u64 {
        self.instances
    }
}

pub(super) fn add(a: u64, b: u64) -> Result<u64, StageError> {
    a.checked_add(b).ok_or(StageError::Memory)
}
pub(super) fn mul(a: u64, b: u64) -> Result<u64, StageError> {
    a.checked_mul(b).ok_or(StageError::Memory)
}
pub(super) fn array<T>(n: u64) -> Result<u64, StageError> {
    let bytes = mul(n, size_of::<T>() as u64)?;
    if usize::try_from(n).is_err() || bytes > isize::MAX as u64 {
        return Err(StageError::Memory);
    }
    Ok(bytes)
}
pub(super) fn growing<T>(n: u64) -> Result<u64, StageError> {
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
fn power(n: u32) -> Result<u64, StageError> {
    1u64.checked_shl(n).ok_or(StageError::Memory)
}
fn width(shape: &[u32]) -> Result<u64, StageError> {
    shape.iter().try_fold(1, |n, &d| {
        if d == 0 {
            return Err(bad());
        }
        mul(n, u64::from(d).next_power_of_two())
    })
}
fn arc<T>() -> Result<u64, StageError> {
    // Pinned std ArcInner: two atomic usize counters, T, and alignment padding
    // before/after T. Extra padding is conservative for the supported targets.
    add(
        add(size_of::<T>() as u64, 2 * size_of::<usize>() as u64)?,
        2 * (align_of::<T>() as u64 - 1),
    )
}
fn evaluation_scratch(vars: u32) -> Result<u64, StageError> {
    let integer = add(power(vars / 2)?, power(vars - vars / 2)?)?;
    // tessera-poly's blocked evaluator retains its <=2^10 equality entries
    // and one partial per high-index block; integer evaluation uses two halves.
    let low = vars.min(10);
    let blocked = add(power(low)?, power(vars - low)?)?;
    array::<F>(integer.max(blocked))
}

fn equality_cache(n: u64) -> Result<u64, StageError> {
    if n == 0 {
        return Ok(0);
    }
    // Pinned std/hashbrown: <=7/8 load, power-of-two buckets (minimum 4).
    // Sum bucket requests across doubling is <2*last; each request adds <=16
    // control bytes plus <=15 alignment bytes. No point collisions are assumed.
    let buckets = add(mul(n, 8)?, 6)?
        .checked_div(7)
        .ok_or(StageError::Memory)?
        .max(4)
        .checked_next_power_of_two()
        .ok_or(StageError::Memory)?;
    let entries = mul(2, buckets)?;
    add(
        add(
            array::<(Vec<F>, Arc<Vec<F>>)>(entries)?,
            array::<u8>(entries)?,
        )?,
        mul(31, u64::from(buckets.trailing_zeros()))?,
    )
}
#[derive(Default)]
struct Requests(u64);
impl Requests {
    fn add(&mut self, value: u64) -> Result<(), StageError> {
        self.0 = add(self.0, value)?;
        Ok(())
    }
    fn array<T>(&mut self, n: u64) -> Result<(), StageError> {
        self.add(array::<T>(n)?)
    }
    fn grow<T>(&mut self, n: u64) -> Result<(), StageError> {
        self.add(growing::<T>(n)?)
    }
    fn repeat(&mut self, n: u64, bytes: u64) -> Result<(), StageError> {
        self.add(mul(n, bytes)?)
    }
    fn terms(&mut self, n: u64, vectors: u64) -> Result<(), StageError> {
        // Sum minimum-capacity slack over separately constructed term vectors.
        self.array::<Term<F>>(mul(4, add(n, mul(4, vectors)?)?)?)?;
        self.repeat(n, array::<usize>(4)?)
    }
}
fn computed(plan: &ModelNativePlan, family: &Family, relation: Relation, literal: bool) -> bool {
    matches!(family.kind, Kind::Softmax { .. })
        && matches!(relation, Relation::Key | Relation::Forest)
        && !literal
        && plan.batch.is_none()
}

impl ModelNativePlan {
    /// Plan native construction and one initial prepare from matching public
    /// lowering metadata without allocating or reading witness values.
    /// `literal_scores=false` covers `ModelNativeProver::new`, `true` its literal
    /// oracle; slotted batches always retain the literal representation.
    /// Canonical leaf-point/source callbacks are assumed: arbitrary callback
    /// allocations are excluded. The remaining exclusions are on [`NativeMemory`].
    ///
    /// # Errors
    /// `Internal` for inconsistent geometry, `Memory` for overflowing/unaddressable
    /// requests. This does not replace source/leaf validation or CPU admission.
    pub fn memory(
        &self,
        lowering: &Lowering,
        literal_scores: bool,
    ) -> Result<NativeMemory, StageError> {
        if self.native.len() != self.slots.len() {
            return Err(bad());
        }
        let mut build = Requests::default();
        let mut prepare = Requests::default();
        let mut instances = Requests::default();
        let count = self.native.len() as u64;
        build.grow::<Data>(count)?;
        // Canonical token-axis analysis: one result and <=max(1, outputs) per node.
        build.array::<Option<u8>>(lowering.graph.tensors.len() as u64)?;
        for node in &lowering.graph.nodes {
            build.array::<Option<u8>>(node.outputs.len().max(1) as u64)?;
        }
        prepare.add(equality_cache(count)?)?;
        let mut tables = 0u64;
        let mut scratch = 0u64;
        if !literal_scores {
            build.add(self.score_input_memory(lowering)?.peak_bytes())?;
        }
        for family in &self.families {
            let layers = power(family.lambda)?;
            if family.nodes.is_empty() || family.nodes.len() as u64 > layers {
                return Err(bad());
            }
            let domain = mul(layers, width(&family.shape)?)?;
            match family.kind {
                Kind::Root {
                    digits,
                    coefficients,
                    residual,
                    ..
                } => {
                    if !(1..=4).contains(&digits) || !(1..=15).contains(&coefficients) {
                        return Err(bad());
                    }
                    let d = digits as u64;
                    let c = coefficients as u64;
                    let row_tables = d + 6;
                    build.repeat(row_tables, array::<G>(domain)?)?;
                    let shape = &lowering
                        .graph
                        .tensors
                        .get(residual.0 as usize)
                        .ok_or_else(bad)?
                        .shape;
                    build.array::<G>(mul(layers, width(shape)?)?)?;
                    tables = add(tables, row_tables + 1)?;
                    build.grow::<usize>(d + 4)?; // temporary shared raw-id vector
                    // Across all coefficients: 4*d*d lower products and at most
                    // four upper terms per product, plus one zero fallback each.
                    let terms = add(mul(20, mul(d, d)?)?, mul(2, c)?)?;
                    build.terms(terms, 2 * c)?;
                    instances.terms(terms, 2 * c)?;
                }
                Kind::Softmax { .. } => {
                    let &[heads, queries, keys] = family.shape.as_slice() else {
                        return Err(bad());
                    };
                    let rows = mul(layers, width(&[heads, queries])?)?;
                    build.repeat(3, array::<G>(rows)?)?; // max, reciprocal, average
                    tables = add(tables, 3)?;
                    if literal_scores || self.batch.is_some() {
                        build.repeat(2, array::<G>(domain)?)?; // broadcast max and scores
                        tables = add(tables, 2)?;
                        // The causal mask is cloned for key/forest data.
                        build.repeat(2, array::<G>(domain)?)?;
                        let _ = keys;
                    } else {
                        build.add(arc::<score::Inputs>()?)?;
                    }
                    if let Some(batch) = &self.batch {
                        build.array::<u64>(batch.lengths.len() as u64)?;
                    }
                }
                Kind::Runtime(kind) => {
                    let &[n, v] = family.shape.as_slice() else {
                        return Err(bad());
                    };
                    let tp = width(&[n])?;
                    let vp = width(&[v])?;
                    let nt = tp.trailing_zeros();
                    let raw = match kind {
                        Runtime::Tokens => 2,
                        Runtime::Observe => 1,
                        _ => u64::from(nt) + 3,
                    };
                    build.repeat(raw, array::<G>(domain)?)?;
                    tables = add(tables, raw)?;
                    if matches!(kind, Runtime::Tokens) {
                        build.array::<G>(tp)?;
                    }
                    if let Runtime::Key(_, column) = kind {
                        build.array::<G>(power(column.variables)?)?;
                    }
                    if matches!(kind, Runtime::Key(..) | Runtime::Selected(_)) {
                        build.array::<G>(domain)?;
                    } // temporary H
                    prepare.array::<F>(add(mul(2, tp)?, vp)?)?;
                }
            }
        }
        build.grow::<SecretBuf<G>>(tables)?;
        for (index, (meta, slot)) in self.native.iter().zip(&self.slots).enumerate() {
            let family = self.families.get(slot.family).ok_or_else(bad)?;
            let n = power(meta.vars)?;
            let layers = family.nodes.len() as u64;
            build.grow::<usize>(meta.evaluations as u64)?;
            build.repeat(4, growing::<F>(layers)?)?; // fixed result and transient fixed() vectors
            build.repeat(2, growing::<F>(u64::from(meta.vars))?)?;
            build.repeat(2, growing::<usize>(u64::from(meta.vars))?)?; // grouped point maps
            build.add(arc::<Vec<F>>()?)?; // Data::add's initial empty equality
            // Integer/fixed evaluations are sequential: retain only the largest
            // pair of equality halves or blocked-evaluation scratch buffers.
            scratch = scratch.max(evaluation_scratch(meta.vars)?);
            if matches!(
                slot.relation,
                Relation::RootPad
                    | Relation::ResidualPad
                    | Relation::MaxPad
                    | Relation::ReciprocalPad
            ) {
                build.array::<G>(n)?;
            }
            let score = computed(self, family, slot.relation, literal_scores);
            if !matches!(slot.relation, Relation::Product) && !score {
                let terms = meta.evaluations as u64 + 2;
                build.terms(terms, 1)?;
                instances.terms(terms, 1)?;
            }
            let public = self
                .window_shape(u32::try_from(index + 1).map_err(|_| StageError::Memory)?)?
                .tables
                .checked_sub(meta.evaluations)
                .ok_or_else(bad)? as u64;
            if matches!(family.kind, Kind::Runtime(_)) {
                // Observe builds then clears two coefficient arrays as well.
                let factors = public.max(2);
                prepare.repeat(factors, array::<F>(n)?)?;
                prepare.grow::<Vec<F>>(factors)?;
                prepare.terms(meta.evaluations as u64 + 2, 1)?;
                prepare.add(arc::<Vec<F>>()?)?;
            } else {
                prepare.grow::<F>(u64::from(meta.vars))?;
                if !score {
                    prepare.array::<F>(n)?;
                    prepare.add(arc::<Vec<F>>()?)?;
                    prepare.array::<F>(power(family.lambda)?)?;
                }
            }
            if !score {
                instances.grow::<Table<'_, F>>(add(meta.evaluations as u64, public)?)?;
            }
        }
        let construction = add(build.0, scratch)?;
        Ok(NativeMemory {
            construction,
            prepared: add(construction, prepare.0)?,
            instances: instances.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use allocation_counter::measure;

    #[test]
    fn native_allocation_sizes_refuse_address_and_product_overflow() {
        let measured = measure(|| {
            assert_eq!(
                array::<F>(isize::MAX as u64 / size_of::<F>() as u64 + 1),
                Err(StageError::Memory)
            );
            assert_eq!(array::<F>(u64::MAX), Err(StageError::Memory));
            assert_eq!(
                array::<F>(u64::MAX / size_of::<F>() as u64 + 1),
                Err(StageError::Memory)
            );
            assert_eq!(growing::<F>(u64::MAX / 4 + 1), Err(StageError::Memory));
            assert_eq!(equality_cache(u64::MAX), Err(StageError::Memory));
            assert_eq!(power(64), Err(StageError::Memory));
        });
        assert_eq!(measured.count_total, 0);
    }

    #[test]
    fn pinned_equality_cache_and_arc_requests_fit_public_bounds() {
        for n in [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 17, 63, 64, 65, 129, 513] {
            let pairs = (0..n)
                .map(|j| (vec![fp(j as u64)], Arc::new(Vec::<F>::new())))
                .collect::<Vec<_>>();
            let mut cache = HashMap::new();
            let measured = measure(|| {
                for (key, value) in pairs {
                    cache.insert(key, value);
                }
            });
            assert!(
                measured.bytes_total <= equality_cache(n as u64).unwrap(),
                "{n} {measured:?}"
            );
        }
        let mut value = None;
        let measured = measure(|| value = Some(Arc::new(Vec::<F>::new())));
        assert!(measured.bytes_total <= arc::<Vec<F>>().unwrap());
    }
}
