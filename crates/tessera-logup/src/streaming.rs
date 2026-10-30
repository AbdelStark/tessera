// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054 generated table fractions, bounded across the complete canonical batch.

use core::{fmt, num::NonZeroUsize};
use tessera_codec::SecretBuf;
use tessera_field::{Field, PrimeField};
use tessera_poly::PolyField;
use tessera_sumcheck::{
    BoundedTree, Claim, Forest, FractionTreeRows, FractionTreeSource, LeafNumerators, LevelClaim,
    PreparedTreeBatch, SumcheckTranscript, Tree, TreeScope, WindowError, check_forests,
};

use crate::instance::ceil_log2;
use crate::prover::{check_schedule, compress, tree, value_at};
use crate::{Column, InstanceClaims, LogUpError, ProvenLogUp, ProvenTables, TreeId, Tuples};

/// Per-tree wrapper/input-vector reservation during bounded batch construction.
///
/// This charge is additional to caller-owned backing storage in
/// [`TableBatchBudget::source_bytes`], retained tree/output metadata and read scratch.
/// Public CPU planners reserve it separately from their native source descriptors
/// (RFC-0018). It changes no protocol message or transcript schedule.
pub const TABLE_BATCH_CONSTRUCTION_BYTES_PER_TREE: u64 = 4096;

/// Immutable public table rows and their already committed canonical multiplicities.
///
/// A read covers the padded `2^ceil(log2(rows))` domain: rows after `rows()` repeat
/// row 0's key/value, while counts are the exact committed counts at those positions.
/// Sources must reproduce identical data on every pass. They may generate rows directly
/// and read fixed-capacity count storage; no table-sized conversion is required.
pub trait TableSource<B: PrimeField> {
    /// Actual table rows, before row-0 padding; at least one.
    fn rows(&self) -> usize;
    /// Whether this table has values. `Range` has none.
    fn has_values(&self) -> bool;
    /// Extra temporary allocation bound for a read of at most `rows` entries.
    /// Backing storage is charged separately by [`TableBatchBudget::source_bytes`].
    fn read_scratch_bytes(&self, _rows: usize) -> Option<u64> {
        Some(0)
    }
    /// Fill three equally sized consecutive chunks of padded keys, values and counts.
    /// For a table without values, fill `values` with zero. The buffers are clearing
    /// storage lent for this read; the source retains no destination or derived copy.
    ///
    /// # Errors
    /// A public shape, allocation or availability error. Failure discards this proof pass.
    fn read(
        &self,
        first: usize,
        keys: &mut [B],
        values: &mut [B],
        counts: &mut [B],
    ) -> Result<(), WindowError>;
}

/// Complete batch allowance and the bounded chunk policy chosen from public geometry.
#[derive(Clone, Copy, Debug)]
pub struct TableBatchBudget {
    /// Batch working-set allowance, including sources, resident lookup trees and output.
    pub allowance: u64,
    /// Coexisting caller-owned source, forest-leaf and retained merge storage, including
    /// its capacities and metadata. The owner supplies its checked allocation plan.
    pub source_bytes: u64,
    /// Maximum table leaves per source read, independent of values and sparsity.
    pub chunk_rows: NonZeroUsize,
}

/// A lookup column and its immutable generated table/count source.
pub struct StreamingInstance<'a, B: PrimeField> {
    /// Public generated rows and the previously committed multiplicities.
    pub table: &'a dyn TableSource<B>,
    /// Secret-class lookup tuples in canonical leaf order, with original row-0 padding.
    pub lookups: Tuples<'a, B>,
}

/// A shape error from ordinary LogUp validation or a fatal bounded-source failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundedLogUpError {
    /// The same public shape/schedule errors as the resident engine.
    LogUp(LogUpError),
    /// Public geometry, insufficient allowance, allocation or source failure.
    Window(WindowError),
}
impl From<LogUpError> for BoundedLogUpError {
    fn from(error: LogUpError) -> Self {
        Self::LogUp(error)
    }
}
impl From<WindowError> for BoundedLogUpError {
    fn from(error: WindowError) -> Self {
        Self::Window(error)
    }
}
impl fmt::Display for BoundedLogUpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LogUp(e) => e.fmt(f),
            Self::Window(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for BoundedLogUpError {}

fn buffer<B: Field>(len: usize) -> Result<SecretBuf<B>, WindowError> {
    let mut out = SecretBuf::try_with_capacity(len).map_err(|_| WindowError::Allocation)?;
    out.resize(len, B::ZERO);
    Ok(out)
}

struct Fractions<'a, F: PolyField> {
    table: &'a dyn TableSource<F::Base>,
    height: usize,
    alpha: F,
    beta: F,
}
impl<F: PolyField> FractionTreeSource<F> for Fractions<'_, F> {
    fn height(&self) -> usize {
        self.height
    }
    fn read_scratch_bytes(&self, rows: usize) -> Option<u64> {
        (rows as u64)
            .checked_mul(3 * size_of::<F::Base>() as u64)?
            .checked_add(self.table.read_scratch_bytes(rows)?)
    }
    fn read(&self, first: usize, n: &mut [F], d: &mut [F]) -> Result<(), WindowError> {
        if n.len() != d.len()
            || first
                .checked_add(n.len())
                .is_none_or(|end| end > 1usize << self.height)
        {
            return Err(WindowError::Source);
        }
        let mut keys = buffer(n.len())?;
        let mut values = buffer(n.len())?;
        let mut counts = buffer(n.len())?;
        self.table
            .read(first, &mut keys, &mut values, &mut counts)?;
        for j in 0..n.len() {
            n[j] = -F::from(counts[j]);
            d[j] = self.alpha
                - compress(
                    self.beta,
                    keys[j],
                    self.table.has_values().then_some(values[j]),
                );
        }
        Ok(())
    }
}

fn check_table<B: PrimeField>(
    table: &dyn TableSource<B>,
    at: usize,
) -> Result<usize, BoundedLogUpError> {
    if table.rows() == 0 {
        return Err(LogUpError::NoRows { instance: at }.into());
    }
    let height = ceil_log2(table.rows());
    if 1usize.checked_shl(height as u32).is_none() {
        return Err(WindowError::Shape("fraction leaves fit usize").into());
    }
    Ok(height)
}

// RFC-0054/0018: charge construction before allocating resident lookup trees or
// batch-sized descriptor vectors. PreparedTreeBatch checks the subsequent levels.
// This pass uses public geometry only and allocates/reads no source data.
fn check_construction<'a, F: PolyField>(
    instances: impl Iterator<Item = (&'a dyn TableSource<F::Base>, Option<usize>)>,
    forests: &[Forest<'_, F>],
    budget: TableBatchBudget,
) -> Result<(), WindowError> {
    fn add(a: u64, b: u64) -> Result<u64, WindowError> {
        a.checked_add(b).ok_or(WindowError::Allocation)
    }
    fn mul(a: u64, b: u64) -> Result<u64, WindowError> {
        a.checked_mul(b).ok_or(WindowError::Allocation)
    }
    let e = size_of::<F>() as u64;
    let mut trees = forests.len() as u64;
    let mut max_vars = 0;
    let mut owned = 0;
    let mut scratch = 0;
    for (table, lookup_rows) in instances {
        let height = ceil_log2(table.rows());
        max_vars = max_vars.max(height);
        trees = add(trees, if lookup_rows.is_some() { 2 } else { 1 })?;
        let source = Fractions {
            table,
            height,
            alpha: F::ZERO,
            beta: F::ZERO,
        };
        let rows = FractionTreeRows::new(&source, budget.chunk_rows, budget.allowance)?;
        scratch = scratch.max(add(
            rows.scratch_bytes(),
            mul(rows.chunk_rows() as u64, mul(4, e)?)?,
        )?);
        if let Some(count) = lookup_rows {
            let height = ceil_log2(count);
            let rows = count
                .checked_next_power_of_two()
                .ok_or(WindowError::Allocation)? as u64;
            max_vars = max_vars.max(height);
            // Unit-numerator leaf denominators plus both extension parent columns.
            owned = add(
                owned,
                add(mul(mul(3, e)?, rows)?, mul(512, (height + 1) as u64)?)?,
            )?;
        }
    }
    for forest in forests {
        max_vars = max_vars.max(ceil_log2(forest.len()));
        owned = add(
            owned,
            add(
                mul(mul(2, size_of::<F::Base>() as u64)?, forest.len() as u64)?,
                mul(128, (forest.height() + 1) as u64)?,
            )?,
        )?;
    }
    // Same retained metadata model as PreparedTreeBatch, plus wrapper/input vectors.
    let metadata = add(
        65_536,
        mul(
            trees,
            mul(
                mul((max_vars + 1) as u64, (max_vars + 8) as u64)?,
                add(256, mul(8, e)?)?,
            )?,
        )?,
    )?;
    let fixed = add(
        budget.source_bytes,
        add(
            owned,
            add(
                metadata,
                add(
                    scratch,
                    mul(TABLE_BATCH_CONSTRUCTION_BYTES_PER_TREE, trees)?,
                )?,
            )?,
        )?,
    )?;
    if fixed > budget.allowance {
        return Err(WindowError::Allocation);
    }
    Ok(())
}

/// RFC-0054 `g=0`: keep lookup trees and forests resident, regenerate the table trees
/// inside the same level batch, with the original schedule, labels and final claims.
///
/// Multiplicities and lookup values must already be committed. Roots do not discharge
/// the root-denominator, identity, table-evaluation or opening checks owed by the verifier.
/// `budget` includes the complete batch, not an allowance per table.
///
/// # Errors
/// Shape/schedule errors before challenges; fatal source/allocation errors discard the
/// session. The complete tree batch is planned before computing its table roots.
pub fn prove_logup_bounded<F, T>(
    transcript: &mut T,
    instances: &[StreamingInstance<'_, F::Base>],
    forests: &[Forest<'_, F>],
    schedule: &[TreeId],
    budget: TableBatchBudget,
) -> Result<ProvenLogUp<F>, BoundedLogUpError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    for (k, instance) in instances.iter().enumerate() {
        check_table(instance.table, k)?;
        if instance.lookups.is_empty() {
            return Err(LogUpError::NoLookups { instance: k }.into());
        }
        if instance.table.has_values() != instance.lookups.has_values() {
            return Err(LogUpError::ValuesMismatch { instance: k }.into());
        }
        if let Some(values) = instance.lookups.values
            && values.len() != instance.lookups.len()
        {
            return Err(LogUpError::ColumnLength {
                instance: k,
                column: Column::LookupValues,
                len: values.len(),
                expected: instance.lookups.len(),
            }
            .into());
        }
    }
    check_forests(forests).map_err(LogUpError::from)?;
    check_construction::<F>(
        instances.iter().map(|i| (i.table, Some(i.lookups.len()))),
        forests,
        budget,
    )?;
    let positions = check_schedule(schedule, instances.len(), forests.len())?;
    let alpha = transcript.challenge("logup/alpha");
    let beta = transcript.challenge("logup/beta");
    let sources: Vec<_> = instances
        .iter()
        .map(|instance| Fractions {
            table: instance.table,
            height: ceil_log2(instance.table.rows()),
            alpha,
            beta,
        })
        .collect();
    let mut padding_values = buffer(instances.len())?;
    let mut lookups = Vec::with_capacity(instances.len());
    for (k, instance) in instances.iter().enumerate() {
        let mut key = buffer(1)?;
        let mut value = buffer(1)?;
        let mut count = buffer(1)?;
        instance.table.read(0, &mut key, &mut value, &mut count)?;
        padding_values[k] = value[0];
        let padding = alpha
            - compress(
                beta,
                key[0],
                instance.table.has_values().then_some(value[0]),
            );
        lookups.push(Some(tree(
            instance.lookups,
            ceil_log2(instance.lookups.len()),
            LeafNumerators::Unit,
            alpha,
            beta,
            padding,
        )?));
    }
    let mut table_rows = sources
        .iter()
        .map(|source| FractionTreeRows::new(source, budget.chunk_rows, budget.allowance).map(Some))
        .collect::<Result<Vec<_>, _>>()?;
    let trees = schedule
        .iter()
        .map(|id| {
            match *id {
                TreeId::Lookup(k) => lookups[k]
                    .take()
                    .map(|tree| BoundedTree::Resident(Tree::Fraction(tree))),
                TreeId::Table(k) => table_rows[k].take().map(BoundedTree::Generated),
                TreeId::Forest(k) => Some(BoundedTree::Resident(Tree::Forest(forests[k]))),
            }
            .ok_or(WindowError::Source)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let batch = PreparedTreeBatch::new(
        TreeScope::Stage,
        trees,
        budget.allowance,
        budget.source_bytes,
    )?;
    let mut roots = Vec::with_capacity(4 * instances.len());
    for k in 0..instances.len() {
        let a = batch
            .fraction_root(positions.lookup[k])
            .ok_or(WindowError::Source)?;
        let t = batch
            .fraction_root(positions.table[k])
            .ok_or(WindowError::Source)?;
        roots.extend([a.0, a.1, t.0, t.1]);
    }
    transcript.absorb_message("logup/roots", &roots);
    let (levels, leaf_claims) = batch.prove(transcript)?.into_levels();
    let fraction = |at| match &leaf_claims[at] {
        LevelClaim::Fraction(c) => Ok(c),
        _ => Err(WindowError::Source),
    };
    let mut values = Vec::new();
    let mut claims = Vec::with_capacity(instances.len());
    for (k, instance) in instances.iter().enumerate() {
        let lookup = fraction(positions.lookup[k])?;
        let table = fraction(positions.table[k])?;
        let value = instance
            .lookups
            .values
            .map(|v| value_at(v, padding_values[k], &lookup.point));
        if let Some(value) = value {
            values.push(value);
        }
        claims.push(InstanceClaims {
            key: Claim {
                point: lookup.point.clone(),
                value: alpha - lookup.denominator - beta * value.unwrap_or(F::ZERO),
            },
            value: value.map(|value| Claim {
                point: lookup.point.clone(),
                value,
            }),
            multiplicity: Claim {
                point: table.point.clone(),
                value: -table.numerator,
            },
        });
    }
    transcript.absorb_message("logup/values", &values);
    let forest_claims = positions
        .forest
        .iter()
        .map(|&at| match &leaf_claims[at] {
            LevelClaim::Product(c) => Ok(c.clone()),
            _ => Err(WindowError::Source),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ProvenLogUp {
        alpha,
        beta,
        roots,
        levels,
        values,
        leaf_claims,
        claims,
        forest_claims,
    })
}

/// RFC-0054 grouped table batch, with all roots/levels under one allowance and no
/// full table columns, negated-count copies or retained fraction levels.
///
/// `alpha` and `beta` are the original `begin_groups` challenges. The caller still owes
/// the grouped identity, all multiplicity openings and the leaf merge.
///
/// # Errors
/// Shape errors before transcript movement; source/allocation errors abort this session.
pub fn prove_table_batch_bounded<F, T>(
    transcript: &mut T,
    alpha: F,
    beta: F,
    tables: &[&dyn TableSource<F::Base>],
    budget: TableBatchBudget,
) -> Result<ProvenTables<F>, BoundedLogUpError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    for (at, table) in tables.iter().enumerate() {
        check_table(*table, at)?;
    }
    check_construction::<F>(tables.iter().map(|&table| (table, None)), &[], budget)?;
    let sources: Vec<_> = tables
        .iter()
        .map(|&table| Fractions {
            table,
            height: ceil_log2(table.rows()),
            alpha,
            beta,
        })
        .collect();
    let trees = sources
        .iter()
        .map(|source| {
            FractionTreeRows::new(source, budget.chunk_rows, budget.allowance)
                .map(BoundedTree::Generated)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let batch = PreparedTreeBatch::new(
        TreeScope::Tables,
        trees,
        budget.allowance,
        budget.source_bytes,
    )?;
    let mut roots = Vec::with_capacity(2 * tables.len());
    for at in 0..tables.len() {
        let (n, d) = batch.fraction_root(at).ok_or(WindowError::Source)?;
        roots.extend([n, d]);
    }
    transcript.absorb_message("tables/logup/roots", &roots);
    let (levels, leaves) = batch.prove(transcript)?.into_levels();
    let multiplicities = leaves
        .into_iter()
        .map(|claim| match claim {
            LevelClaim::Fraction(c) => Ok(Claim {
                point: c.point,
                value: -c.numerator,
            }),
            _ => Err(WindowError::Source),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ProvenTables {
        roots,
        levels,
        multiplicities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LogUpInstance, prove_logup, prove_table_batch};
    use core::cell::Cell;
    use tessera_codec::Ascii;
    use tessera_field::{Goldilocks as B, GoldilocksExt2 as F};
    use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};

    struct Source {
        keys: Vec<B>,
        values: Option<Vec<B>>,
        counts: Vec<B>,
        largest: Cell<usize>,
        reads: Cell<usize>,
        fail: Cell<bool>,
    }
    impl TableSource<B> for Source {
        fn rows(&self) -> usize {
            self.keys.len()
        }
        fn has_values(&self) -> bool {
            self.values.is_some()
        }
        fn read(
            &self,
            first: usize,
            keys: &mut [B],
            values: &mut [B],
            counts: &mut [B],
        ) -> Result<(), WindowError> {
            self.reads.set(self.reads.get() + 1);
            self.largest.set(self.largest.get().max(keys.len()));
            if self.fail.get() {
                return Err(WindowError::Source);
            }
            if keys.len() != values.len()
                || keys.len() != counts.len()
                || first
                    .checked_add(keys.len())
                    .is_none_or(|end| end > self.counts.len())
            {
                return Err(WindowError::Source);
            }
            for j in 0..keys.len() {
                let at = if first + j < self.rows() {
                    first + j
                } else {
                    0
                };
                keys[j] = self.keys[at];
                values[j] = self.values.as_ref().map_or(B::ZERO, |v| v[at]);
                counts[j] = self.counts[first + j];
            }
            Ok(())
        }
    }
    struct Input {
        table: Source,
        keys: Vec<B>,
        values: Option<Vec<B>>,
    }
    impl Input {
        fn new(rows: usize, valued: bool) -> Self {
            let keys: Vec<_> = (0..rows).map(|j| B::from_u64_reduced(j as u64)).collect();
            let values: Option<Vec<_>> =
                valued.then(|| (0..rows).map(|j| B::from_i64(3 * j as i64 - 7)).collect());
            let lookups = [0, rows - 1, rows / 2, rows - 1, 0];
            let mut counts = vec![B::ZERO; rows.next_power_of_two()];
            for &row in &lookups {
                counts[row] += B::ONE;
            }
            counts[0] += B::from_u64_reduced(3);
            Self {
                keys: lookups.iter().map(|&j| keys[j]).collect(),
                values: values
                    .as_ref()
                    .map(|v| lookups.iter().map(|&j| v[j]).collect()),
                table: Source {
                    keys,
                    values,
                    counts,
                    largest: Cell::new(0),
                    reads: Cell::new(0),
                    fail: Cell::new(false),
                },
            }
        }
        fn literal(&self) -> LogUpInstance<'_, B> {
            LogUpInstance::new(
                Tuples::new(&self.table.keys, self.table.values.as_deref()),
                Tuples::new(&self.keys, self.values.as_deref()),
                &self.table.counts,
            )
        }
        fn streaming(&self) -> StreamingInstance<'_, B> {
            StreamingInstance {
                table: &self.table,
                lookups: Tuples::new(&self.keys, self.values.as_deref()),
            }
        }
        fn bytes(&self) -> u64 {
            ((self.table.keys.capacity()
                + self.table.counts.capacity()
                + self.keys.capacity()
                + self.table.values.as_ref().map_or(0, Vec::capacity)
                + self.values.as_ref().map_or(0, Vec::capacity))
                * size_of::<B>()
                + size_of::<Self>()) as u64
        }
    }
    fn transcript(mode: Mode) -> Transcript {
        Transcript::new(&SessionContext {
            protocol: Ascii::new("tessera-protocol/1").unwrap(),
            mode,
            params: ParamSetId(Ascii::new("test-small/1").unwrap()),
            statement_digest: [89; 32],
            session_id: [90; 16],
        })
    }
    fn budget(bytes: u64, chunk: usize, source_bytes: u64) -> TableBatchBudget {
        TableBatchBudget {
            allowance: bytes,
            source_bytes,
            chunk_rows: NonZeroUsize::new(chunk).unwrap(),
        }
    }
    fn assert_levels(
        a: &[tessera_sumcheck::ProvenTreeLevel<F>],
        b: &[tessera_sumcheck::ProvenTreeLevel<F>],
    ) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert_eq!(a.proof(), b.proof());
            assert_eq!(a.point(), b.point());
            assert_eq!(a.next_claims(), b.next_claims());
            assert_eq!(
                (a.gamma(), a.lambda(), a.rho()),
                (b.gamma(), b.lambda(), b.rho())
            );
        }
    }
    #[test]
    fn generated_tables_match_resident_logup_in_both_scopes() {
        let inputs = [
            Input::new(1, true),
            Input::new(3, false),
            Input::new(5, true),
            Input::new(31, true),
        ];
        // A shared table remains a separate protocol instance on every use.
        let order = [0, 1, 2, 3, 2];
        let literal: Vec<_> = order.iter().map(|&i| inputs[i].literal()).collect();
        let streamed: Vec<_> = order.iter().map(|&i| inputs[i].streaming()).collect();
        let sources: Vec<&dyn TableSource<B>> = order
            .iter()
            .map(|&i| &inputs[i].table as &dyn TableSource<B>)
            .collect();
        let forest: Vec<_> = (0..64).map(|j| B::from_u64_reduced(j + 1)).collect();
        let forests = [Forest::new(&forest, 3)];
        let mut schedule = vec![TreeId::Forest(0)];
        schedule.extend(TreeId::paired(literal.len()).into_iter().rev());
        for chunk in [1, 3, 17, 256] {
            for mode in [Mode::Live, Mode::NonInteractive] {
                for input in &inputs {
                    input.table.largest.set(0);
                }
                let mut before = transcript(mode);
                let expected =
                    prove_logup::<F, _>(&mut before, &literal, &forests, &schedule).unwrap();
                let mut after = transcript(mode);
                let actual = prove_logup_bounded::<F, _>(
                    &mut after,
                    &streamed,
                    &forests,
                    &schedule,
                    budget(u64::MAX, chunk, 0),
                )
                .unwrap();
                assert_eq!(actual.roots(), expected.roots());
                assert_eq!(actual.values(), expected.values());
                assert_eq!(actual.claims(), expected.claims());
                assert_eq!(actual.forest_claims(), expected.forest_claims());
                assert_eq!(actual.leaf_claims(), expected.leaf_claims());
                assert_levels(actual.levels(), expected.levels());
                assert_eq!(after.challenge_fp2("next"), before.challenge_fp2("next"));
                assert!(inputs.iter().all(|i| i.table.largest.get() <= chunk));
                let alpha = F::from(B::from_u64_reduced(1_000_003));
                let beta = F::from(B::from_u64_reduced(53));
                let mut before = transcript(mode);
                let expected = prove_table_batch(&mut before, alpha, beta, &literal).unwrap();
                let mut after = transcript(mode);
                let actual = prove_table_batch_bounded(
                    &mut after,
                    alpha,
                    beta,
                    &sources,
                    budget(u64::MAX, chunk, 0),
                )
                .unwrap();
                assert_eq!(actual.roots(), expected.roots());
                assert_eq!(actual.multiplicities(), expected.multiplicities());
                assert_levels(actual.levels(), expected.levels());
                assert_eq!(after.challenge_fp2("next"), before.challenge_fp2("next"));
                assert!(inputs.iter().all(|i| i.table.largest.get() <= chunk));
            }
        }
    }
    #[test]
    fn complete_batch_peak_includes_sources_and_fits_streaming_allowance() {
        let inputs = [
            Input::new(1, true),
            Input::new(3, false),
            Input::new(5, true),
            Input::new((1 << 14) - 3, true),
        ];
        let literal: Vec<_> = inputs.iter().map(Input::literal).collect();
        let streamed: Vec<_> = inputs.iter().map(Input::streaming).collect();
        let source_bytes = inputs.iter().map(Input::bytes).sum::<u64>()
            + (literal.capacity() * size_of::<LogUpInstance<'_, B>>()
                + streamed.capacity() * size_of::<StreamingInstance<'_, B>>()) as u64;
        let schedule = TreeId::paired(inputs.len());
        let mut expected = None;
        let resident = allocation_counter::measure(|| {
            expected = Some(
                prove_logup::<F, _>(
                    &mut transcript(Mode::NonInteractive),
                    &literal,
                    &[],
                    &schedule,
                )
                .unwrap(),
            );
        });
        let mut actual = None;
        let allowance = 1_700_000;
        let bounded = allocation_counter::measure(|| {
            actual = Some(
                prove_logup_bounded::<F, _>(
                    &mut transcript(Mode::NonInteractive),
                    &streamed,
                    &[],
                    &schedule,
                    budget(allowance, 17, source_bytes),
                )
                .unwrap(),
            );
        });
        let actual = actual.unwrap();
        let expected = expected.unwrap();
        assert_eq!(actual.roots(), expected.roots());
        assert_levels(actual.levels(), expected.levels());
        assert!(
            bounded.bytes_max as u64 + source_bytes <= allowance,
            "source={source_bytes}, bounded={bounded:?}"
        );
        assert!(
            bounded.bytes_max * 2 < resident.bytes_max,
            "resident={resident:?}, bounded={bounded:?}"
        );
        assert!(inputs.iter().all(|i| i.table.largest.get() <= 17));
    }
    #[test]
    fn insufficient_batch_allowance_refuses_before_lookup_tree_construction() {
        let mut input = Input::new(1, true);
        let rows = 1usize << 17;
        input.keys = vec![input.table.keys[0]; rows];
        input.values = Some(vec![input.table.values.as_ref().unwrap()[0]; rows]);
        input.table.counts[0] = B::from_u64_reduced(rows as u64);
        let source_bytes = input.bytes();
        let allowance = source_bytes + 131_072;
        let schedule = TreeId::paired(1);
        let mut tr = transcript(Mode::NonInteractive);
        let mut result = None;
        let measured = allocation_counter::measure(|| {
            result = Some(prove_logup_bounded::<F, _>(
                &mut tr,
                &[input.streaming()],
                &[],
                &schedule,
                budget(allowance, 17, source_bytes),
            ));
        });
        assert!(matches!(
            result.unwrap(),
            Err(BoundedLogUpError::Window(WindowError::Allocation))
        ));
        assert!(
            measured.bytes_max as u64 + source_bytes <= allowance,
            "source={source_bytes}, allowance={allowance}, measured={measured:?}"
        );
        assert_eq!(input.table.reads.get(), 0);
        assert_eq!(
            tr.challenge_fp2("after"),
            transcript(Mode::NonInteractive).challenge_fp2("after")
        );
    }
    #[test]
    fn insufficient_table_only_allowance_refuses_before_descriptor_allocation() {
        let input = Input::new(1, true);
        let tables: Vec<&dyn TableSource<B>> = vec![&input.table; 1 << 14];
        let source_bytes =
            input.bytes() + (tables.capacity() * size_of::<&dyn TableSource<B>>()) as u64;
        let allowance = source_bytes + 131_072;
        let mut tr = transcript(Mode::NonInteractive);
        let mut result = None;
        let measured = allocation_counter::measure(|| {
            result = Some(prove_table_batch_bounded(
                &mut tr,
                F::ONE,
                F::ONE,
                &tables,
                budget(allowance, 17, source_bytes),
            ));
        });
        assert!(matches!(
            result.unwrap(),
            Err(BoundedLogUpError::Window(WindowError::Allocation))
        ));
        assert!(
            measured.bytes_max as u64 + source_bytes <= allowance,
            "source={source_bytes}, allowance={allowance}, measured={measured:?}"
        );
        assert_eq!(input.table.reads.get(), 0);
        assert_eq!(
            tr.challenge_fp2("after"),
            transcript(Mode::NonInteractive).challenge_fp2("after")
        );
    }
    #[test]
    fn invalid_shapes_budgets_and_failed_sources_abort_without_substitution() {
        let input = Input::new(5, true);
        let empty = Source {
            keys: vec![],
            values: None,
            counts: vec![],
            largest: Cell::new(0),
            reads: Cell::new(0),
            fail: Cell::new(false),
        };
        let config = budget(u64::MAX, 3, 0);
        for (instance, error) in [
            (
                StreamingInstance {
                    table: &input.table,
                    lookups: Tuples::new(&input.keys, Some(&input.keys[..1])),
                },
                LogUpError::ColumnLength {
                    instance: 0,
                    column: Column::LookupValues,
                    len: 1,
                    expected: input.keys.len(),
                },
            ),
            (
                StreamingInstance {
                    table: &empty,
                    lookups: Tuples::new(&input.keys, None),
                },
                LogUpError::NoRows { instance: 0 },
            ),
            (
                StreamingInstance {
                    table: &input.table,
                    lookups: Tuples::new(&[], Some(&[])),
                },
                LogUpError::NoLookups { instance: 0 },
            ),
            (
                StreamingInstance {
                    table: &input.table,
                    lookups: Tuples::new(&input.keys, None),
                },
                LogUpError::ValuesMismatch { instance: 0 },
            ),
        ] {
            let mut tr = transcript(Mode::Live);
            assert_eq!(
                prove_logup_bounded::<F, _>(&mut tr, &[instance], &[], &TreeId::paired(1), config)
                    .unwrap_err(),
                BoundedLogUpError::LogUp(error)
            );
            assert_eq!(
                tr.challenge_fp2("after"),
                transcript(Mode::Live).challenge_fp2("after")
            );
        }
        let mut tr = transcript(Mode::Live);
        assert_eq!(
            prove_logup_bounded::<F, _>(
                &mut tr,
                &[input.streaming()],
                &[],
                &TreeId::paired(1),
                budget(0, 3, 0)
            )
            .unwrap_err(),
            BoundedLogUpError::Window(WindowError::Allocation)
        );
        assert_eq!(input.table.reads.get(), 0);
        input.table.fail.set(true);
        assert_eq!(
            prove_logup_bounded::<F, _>(
                &mut tr,
                &[input.streaming()],
                &[],
                &TreeId::paired(1),
                config
            )
            .unwrap_err(),
            BoundedLogUpError::Window(WindowError::Source)
        );
        assert_eq!(
            prove_table_batch_bounded(&mut tr, F::ONE, F::ONE, &[&input.table], config)
                .unwrap_err(),
            BoundedLogUpError::Window(WindowError::Source)
        );
    }
}
