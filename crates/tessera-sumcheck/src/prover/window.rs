// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0055/0056: factored-EQ grids and bounded windows of at most eight own variables.
//!
//! Preparation has no transcript access: its true sums can be sent in the previous stage's
//! splits. Only `prove` squeezes gamma. Secret-derived grids and folded tables are clearing
//! buffers, including temporary multilinear-to-grid transforms.

mod memory;
pub use memory::{WindowMemory, WindowMetadataMemory, WindowShape};

use core::{fmt, num::NonZeroUsize};

use tessera_codec::SecretBuf;
use tessera_field::PrimeField;
use tessera_poly::PolyField;

use super::{Labels, Prefix, ProvenBatch, Table, Values, padding_scale};
use crate::{Protocol, SumcheckProof, SumcheckTranscript, Term, parallelism};

/// Below this many entries per thread, one thread runs a resident round.
const PARALLEL_MIN_ENTRIES: usize = 1 << 14;

/// `0..len` in at most `parallelism()` contiguous ranges of at least `PARALLEL_MIN_ENTRIES`.
fn ranges(len: usize) -> Vec<core::ops::Range<usize>> {
    let threads = parallelism().min(len / PARALLEL_MIN_ENTRIES).max(1);
    let per = len.div_ceil(threads).max(1);
    (0..len)
        .step_by(per)
        .map(|start| start..len.min(start + per))
        .collect()
}

/// Runs `f` on each range of `ranges(len)`, on scoped threads when there are several, and
/// returns the results in range order. Results are combined by exact field addition by the
/// callers, so the thread count never changes a value (RFC-0018 'CPU backend').
fn par_map<T: Send>(
    len: usize,
    f: impl Fn(core::ops::Range<usize>) -> Result<T, WindowError> + Sync,
) -> Result<Vec<T>, WindowError> {
    let ranges = ranges(len);
    if ranges.len() <= 1 {
        return ranges.into_iter().map(&f).collect();
    }
    std::thread::scope(|scope| {
        let workers: Vec<_> = ranges
            .into_iter()
            .map(|range| {
                let f = &f;
                scope.spawn(move || f(range))
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().unwrap_or(Err(WindowError::Allocation)))
            .collect()
    })
}

/// RFC-0056's maximum number of own variables in one grid.
pub const MAX_WINDOW_VARS: usize = 8;

/// A failure described entirely by public geometry or an allocation failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowError {
    /// An invalid public shape or an unsupported protocol.
    Shape(&'static str),
    /// An allocation was refused.
    Allocation,
    /// The source could not regenerate or supply its public table geometry.
    Source,
}

impl fmt::Display for WindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(rule) => write!(f, "windowed sumcheck: {rule}"),
            Self::Allocation => f.write_str("windowed sumcheck allocation refused"),
            Self::Source => f.write_str("windowed sumcheck source unavailable"),
        }
    }
}

impl std::error::Error for WindowError {}

/// The pass whose stage tensors a source must regenerate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowPass {
    /// P3: before gamma and the previous stage's splits.
    Prepare,
    /// P3b: after the given number of global rounds. Also builds the next grid, if any.
    Fold(usize),
}

/// Regenerable tables. IDs and indices are public. Implementations keep secret tensors in
/// clearing storage and write derived entries directly into the supplied clearing destination.
/// The source must reproduce the same table on every pass (RFC-0055, P3/P3b).
pub trait WindowSource<F: PolyField> {
    /// Regenerate the stage tensors for this pass.
    fn regenerate(&mut self, pass: WindowPass) -> Result<(), WindowError>;
    /// An already resident operand, lent as an opaque table. The engine reads it without
    /// exposing its entries to the caller; native relations and folded matmuls may use this.
    fn resident(&self, _table: usize) -> Option<Table<'_, F>> {
        None
    }
    /// End the grid/fold pass after all its arithmetic, for truthful backend pass metrics.
    fn finish_pass(&mut self, _pass: WindowPass) -> Result<(), WindowError> {
        Ok(())
    }
    /// Read consecutive entries of the table named by `table`.
    fn read(&self, table: usize, start: usize, out: &mut [F]) -> Result<(), WindowError>;
    /// The most entries one regeneration of this source serves, when it has such a unit.
    /// The engine then reads resident tables range by range, every table's range before the
    /// next, in at most this many entries, so tables that share one regeneration (a fraction
    /// level's four operands, #694) can be served from it. The values read are unchanged.
    fn read_granule(&self) -> Option<NonZeroUsize> {
        None
    }
}

/// One operand, named by its stable public source ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowTable {
    /// Source table ID.
    pub id: usize,
    /// Public factors are omitted from final evaluations, as in [`super::Table`].
    pub public: bool,
}

/// A relation `eq(z, x_E) * sum_t weight_t * product_k table_k(x)`.
/// `eq[j] = None` means coordinate `j` is absent from E. The factored EQ is public and is
/// omitted from final evaluations. Other public factors remain ordinary tables.
#[derive(Clone)]
pub struct WindowInstance<F> {
    /// Own variables, lowest first; every source table has exactly `2^vars` entries.
    pub vars: usize,
    /// Operands, in final-evaluation order.
    pub tables: Vec<WindowTable>,
    /// Nonempty weighted products of operands; repeated factors are permitted.
    pub terms: Vec<Term<F>>,
    /// One optional public EQ coordinate per own variable.
    pub eq: Vec<Option<F>>,
    /// Whether this instance uses grids; resident instances may share the batch.
    pub windowed: bool,
}

// No Debug implementation: terms' weights and EQ points need not be exposed in logs.

fn buffer<F: PolyField>(len: usize) -> Result<SecretBuf<F>, WindowError> {
    let mut out = SecretBuf::try_with_capacity(len).map_err(|_| WindowError::Allocation)?;
    out.resize(len, F::ZERO);
    Ok(out)
}

fn read_source<F: PolyField, S: WindowSource<F> + ?Sized>(
    source: &S,
    id: usize,
    start: usize,
    out: &mut [F],
) -> Result<(), WindowError> {
    if let Some(table) = source.resident(id) {
        let end = start.checked_add(out.len()).ok_or(WindowError::Source)?;
        match table.values {
            Values::Base(values) => {
                let values = values.get(start..end).ok_or(WindowError::Source)?;
                for (slot, &value) in out.iter_mut().zip(values) {
                    *slot = F::from(value);
                }
            }
            Values::Ext(values) => {
                out.copy_from_slice(values.get(start..end).ok_or(WindowError::Source)?)
            }
        }
        Ok(())
    } else {
        source.read(id, start, out)
    }
}

fn integer<F: PolyField>(n: usize) -> F {
    F::from(F::Base::from_u64_reduced(n as u64))
}

fn eq_at<F: PolyField>(z: Option<F>, x: F) -> F {
    z.map_or(F::ONE, |z| (F::ONE - z) * (F::ONE - x) + z * x)
}

fn boolean_eq<F: PolyField>(z: &[Option<F>], index: usize) -> F {
    z.iter()
        .enumerate()
        .fold(F::ONE, |v, (j, &z)| v * eq_at(z, integer((index >> j) & 1)))
}

/// The two polynomials of tree levels, evaluated with fewer products than the generic terms:
/// a fraction level's `N0·D1 + N1·D0 + λ·D0·D1` over columns `[N0, N1, D0, D1]`, and a product
/// level's `V0·V1`. Lines at `x = 2, 3` are sums of the pair's slope, and every product is the
/// same field element the generic evaluation forms (RFC-0010 'Tree levels').
#[derive(Clone, Copy)]
enum Shape<F> {
    Fraction { lambda: F },
    Product,
}

impl<F: PolyField> Shape<F> {
    /// The shape of `terms` over `columns` tables, when it is one and the slots are `0, 2, 3`.
    fn of(terms: &[Term<F>], columns: usize, xs: &[F]) -> Option<Self> {
        let standard = xs.len() == 3
            && xs[0] == F::ZERO
            && xs[1] == integer::<F>(2)
            && xs[2] == integer::<F>(3);
        if !standard {
            return None;
        }
        match (columns, terms) {
            (4, [a, b, c])
                if *a.weight() == F::ONE
                    && a.factors() == [0, 3]
                    && *b.weight() == F::ONE
                    && b.factors() == [1, 2]
                    && c.factors() == [2, 3] =>
            {
                Some(Self::Fraction {
                    lambda: *c.weight(),
                })
            }
            (2, [a]) if *a.weight() == F::ONE && a.factors() == [0, 1] => Some(Self::Product),
            _ => None,
        }
    }

    /// `Σ_pairs suffix(pair)·P(line(x))` at `x = 0, 2, 3` over `pairs`.
    fn sums(
        self,
        folded: &[SecretBuf<F>],
        suffix_eq: &Weights<F>,
        pairs: core::ops::Range<usize>,
        degree: usize,
    ) -> Result<SecretBuf<F>, WindowError> {
        let mut part = buffer::<F>(degree)?;
        let mut sums = [F::ZERO; 3];
        // Each column at x = 0, 2, 3: a, a + 2(b − a), a + 3(b − a).
        let at = |t: &SecretBuf<F>, pair: usize| {
            let (a, b) = (t[2 * pair], t[2 * pair + 1]);
            let d = b - a;
            let two = b + d;
            [a, two, two + d]
        };
        match self {
            Self::Fraction { lambda } => {
                for pair in pairs {
                    let w = suffix_eq.at(pair);
                    let (n0, n1, d0, d1) = (
                        at(&folded[0], pair),
                        at(&folded[1], pair),
                        at(&folded[2], pair),
                        at(&folded[3], pair),
                    );
                    for z in 0..3 {
                        let p = n0[z] * d1[z] + n1[z] * d0[z] + lambda * (d0[z] * d1[z]);
                        sums[z] += w * p;
                    }
                }
            }
            Self::Product => {
                for pair in pairs {
                    let w = suffix_eq.at(pair);
                    let (v0, v1) = (at(&folded[0], pair), at(&folded[1], pair));
                    for z in 0..3 {
                        sums[z] += w * (v0[z] * v1[z]);
                    }
                }
            }
        }
        part.copy_from_slice(&sums[..degree]);
        Ok(part)
    }
}

/// Public split EQ weights: no allocation of the full `2^bound` prefix.
struct Weights<F> {
    low: Vec<F>,
    high: Vec<F>,
    low_bits: usize,
}

impl<F: PolyField> Weights<F> {
    fn new(point: &[F]) -> Result<Self, WindowError> {
        Self::from_eq(&point.iter().copied().map(Some).collect::<Vec<_>>())
    }

    fn from_eq(point: &[Option<F>]) -> Result<Self, WindowError> {
        fn half<F: PolyField>(point: &[Option<F>]) -> Result<Vec<F>, WindowError> {
            // Reserve the final public half once. Incremental growth would briefly retain
            // both old and new allocations, exceeding RFC-0055's split-EQ memory charge.
            let capacity = 1usize
                .checked_shl(point.len() as u32)
                .ok_or(WindowError::Allocation)?;
            let mut weights = Vec::new();
            weights
                .try_reserve_exact(capacity)
                .map_err(|_| WindowError::Allocation)?;
            weights.push(F::ONE);
            for &r in point {
                let len = weights.len();
                weights.resize(2 * len, F::ZERO);
                for i in 0..len {
                    let value = weights[i];
                    weights[i + len] = value * eq_at(r, F::ONE);
                    weights[i] = value * eq_at(r, F::ZERO);
                }
            }
            Ok(weights)
        }
        let low_bits = point.len() / 2;
        Ok(Self {
            low: half(&point[..low_bits])?,
            high: half(&point[low_bits..])?,
            low_bits,
        })
    }

    fn at(&self, index: usize) -> F {
        self.low[index & (self.low.len() - 1)] * self.high[index >> self.low_bits]
    }
}

fn radix_index(mut boolean: usize, radix: usize, vars: usize) -> usize {
    let mut out = 0;
    let mut stride = 1;
    for _ in 0..vars {
        out += (boolean & 1) * stride;
        boolean >>= 1;
        stride *= radix;
    }
    out
}

/// Extend Boolean corners to `{0,..,delta}^v` in place, one multilinear axis at a time.
fn extend<F: PolyField>(values: &mut [F], radix: usize, vars: usize) {
    let mut stride = 1;
    for axis in 0..vars {
        for high in 0..1 << (vars - axis - 1) {
            let base = radix_index(high, radix, vars - axis - 1) * stride * radix;
            for low in 0..stride {
                let at = base + low;
                let a = values[at];
                let delta = values[at + stride] - a;
                for node in 2..radix {
                    values[at + node * stride] = a + integer::<F>(node) * delta;
                }
            }
        }
        stride *= radix;
    }
}

/// Tensor Lagrange interpolation at integer nodes. No division by `r-node`: challenges 0/1
/// (and every other grid node) are ordinary inputs.
fn basis<F: PolyField>(radix: usize, r: F) -> Result<Vec<F>, WindowError> {
    (0..radix)
        .map(|j| {
            let mut numerator = F::ONE;
            let mut denominator = F::ONE;
            for k in 0..radix {
                if k != j {
                    numerator *= r - integer::<F>(k);
                    denominator *= integer::<F>(j) - integer::<F>(k);
                }
            }
            denominator
                .inverse()
                .map(|inverse| numerator * inverse)
                .ok_or(WindowError::Shape(
                    "grid nodes must be distinct in the field",
                ))
        })
        .collect()
}

struct Grid<F: PolyField> {
    values: SecretBuf<F>,
    radix: usize,
    /// Own variables covered by this grid, including those already contracted.
    end: usize,
}

struct State<F: PolyField> {
    spec: WindowInstance<F>,
    grid: Option<Grid<F>>,
    folded: Vec<SecretBuf<F>>,
    /// The folded tables on a device, when a registered offload took them (#734).
    device: Option<Box<dyn crate::DeviceLevel>>,
    bound: usize,
    /// True initial sum, not a prover-supplied claim. Cleared on drop with the other scalars.
    sum: SecretBuf<F>,
}

impl<F: PolyField> State<F> {
    fn eq_prefix(&self, point: &[F]) -> F {
        self.spec
            .eq
            .iter()
            .zip(point)
            .fold(F::ONE, |v, (&z, &r)| v * eq_at(z, r))
    }

    fn polynomial(&self, columns: &[SecretBuf<F>], at: usize) -> F {
        self.spec.terms.iter().fold(F::ZERO, |sum, term| {
            sum + term
                .factors()
                .iter()
                .fold(*term.weight(), |v, &k| v * columns[k][at])
        })
    }

    /// Contract already bound coordinates directly from a regenerated source. A single source
    /// entry lives in a clearing buffer; public split weights never depend on witness values.
    fn entry<S: WindowSource<F> + ?Sized>(
        source: &S,
        id: usize,
        remaining: usize,
        bound: usize,
        weights: &Weights<F>,
        one: &mut [F],
    ) -> Result<F, WindowError> {
        let mut value = F::ZERO;
        let count = 1usize << bound;
        let chunk = count.min(one.len());
        for prefix in (0..count).step_by(chunk) {
            read_source(source, id, (remaining << bound) | prefix, &mut one[..chunk])?;
            for (offset, &entry) in one[..chunk].iter().enumerate() {
                value += weights.at(prefix + offset) * entry;
            }
        }
        Ok(value)
    }

    fn prepare<S: WindowSource<F> + ?Sized>(
        &mut self,
        source: &S,
        point: &[F],
        end: usize,
        initial: bool,
    ) -> Result<(), WindowError> {
        let bound = point.len();
        let remaining = self.spec.vars - bound;
        let window = end - bound;
        // Release the consumed grid before allocating prefix EQ weights or its replacement.
        self.grid = None;
        let weights = Weights::new(point)?;
        let mut one = buffer::<F>(1 << window.max(bound.min(MAX_WINDOW_VARS)))?;
        if !self.spec.windowed || window == 0 && remaining == 0 {
            // Resident instances are loaded only on P3, then bound in place. A windowed
            // instance is materialized only after its last window's challenges are known.
            let len = 1usize << remaining;
            // Range-major: every table reads the same entries before the next range (#694).
            let mut folded = self
                .spec
                .tables
                .iter()
                .map(|_| buffer::<F>(len))
                .collect::<Result<Vec<_>, WindowError>>()?;
            if bound == 0 {
                let step = source.read_granule().map_or(len, |g| g.get().min(len));
                for start in (0..len).step_by(step) {
                    let end = len.min(start + step);
                    for (out, table) in folded.iter_mut().zip(&self.spec.tables) {
                        read_source(source, table.id, start, &mut out[start..end])?;
                    }
                }
            } else {
                for index in 0..len {
                    for (out, table) in folded.iter_mut().zip(&self.spec.tables) {
                        out[index] =
                            Self::entry(source, table.id, index, bound, &weights, &mut one)?;
                    }
                }
            }
            self.folded = folded;
            self.grid = None;
            if initial {
                let eq = Weights::from_eq(&self.spec.eq)?;
                let parts = par_map(len, |range| {
                    Ok(range.fold(F::ZERO, |s, index| {
                        s + eq.at(index) * self.polynomial(&self.folded, index)
                    }))
                })?;
                self.sum[0] = parts.into_iter().fold(F::ZERO, |s, p| s + p);
            }
            self.offload()?;
            return Ok(());
        }
        let delta = self
            .spec
            .terms
            .iter()
            .map(|term| term.factors().len())
            .max()
            .unwrap_or(1);
        let radix = delta + 1;
        let size = radix
            .checked_pow(window as u32)
            .ok_or(WindowError::Allocation)?;
        let mut grid = buffer::<F>(size)?;
        let mut columns = self
            .spec
            .tables
            .iter()
            .map(|_| buffer::<F>(size))
            .collect::<Result<Vec<_>, _>>()?;
        let suffix_eq = Weights::from_eq(&self.spec.eq[end..])?;
        for suffix in 0..1 << (remaining - window) {
            if bound == 0 {
                for (column, table) in columns.iter_mut().zip(&self.spec.tables) {
                    read_source(source, table.id, suffix << window, &mut one[..1 << window])?;
                    for vertex in 0..1 << window {
                        column[radix_index(vertex, radix, window)] = one[vertex];
                    }
                }
            } else {
                // Vertex-major, so tables sharing a regeneration read each entry in turn (#694).
                for vertex in 0..1 << window {
                    let at = radix_index(vertex, radix, window);
                    for (column, table) in columns.iter_mut().zip(&self.spec.tables) {
                        column[at] = Self::entry(
                            source,
                            table.id,
                            (suffix << window) | vertex,
                            bound,
                            &weights,
                            &mut one,
                        )?;
                    }
                }
            }
            for column in &mut columns {
                extend(column, radix, window);
            }
            let eq = suffix_eq.at(suffix);
            for (at, value) in grid.iter_mut().enumerate() {
                *value += eq * self.polynomial(&columns, at);
            }
        }
        if initial {
            self.sum[0] = (0..1 << window).fold(F::ZERO, |sum, vertex| {
                sum + boolean_eq(&self.spec.eq[..end], vertex)
                    * grid[radix_index(vertex, radix, window)]
            });
        }
        self.grid = Some(Grid {
            values: grid,
            radix,
            end,
        });
        self.bound = bound;
        Ok(())
    }

    /// Moves the folded tables to the registered offload's device, when it takes this shape
    /// over `Fp2` (#734). The host copies clear as they drop.
    fn offload(&mut self) -> Result<(), WindowError> {
        use core::any::Any;
        use tessera_field::GoldilocksExt2 as Fp2;
        let Some(offload) = crate::offload::level_offload() else {
            return Ok(());
        };
        if self.spec.vars == self.bound || self.degree_slots().is_none() {
            return Ok(());
        }
        let xs = [integer::<F>(0), integer::<F>(2), integer::<F>(3)];
        let shape = match Shape::of(&self.spec.terms, self.folded.len(), &xs) {
            Some(Shape::Fraction { lambda }) => {
                let Some(&lambda) = (&lambda as &dyn Any).downcast_ref::<Fp2>() else {
                    return Ok(());
                };
                crate::LevelShape::Fraction { lambda }
            }
            Some(Shape::Product) => crate::LevelShape::Product,
            None => return Ok(()),
        };
        let tables: Option<Vec<&[Fp2]>> = self
            .folded
            .iter()
            .map(|t| {
                (t as &dyn Any)
                    .downcast_ref::<SecretBuf<Fp2>>()
                    .map(|t| &t[..])
            })
            .collect();
        let Some(tables) = tables else {
            return Ok(());
        };
        if let Some(device) = offload.load(shape, &tables)? {
            self.device = Some(device);
            self.folded = Vec::new();
        }
        Ok(())
    }

    /// `Some` when the instance's polynomial has degree 3, the only one the offload serves.
    fn degree_slots(&self) -> Option<()> {
        (self.spec.terms.iter().map(|t| t.factors().len()).max() == Some(2)).then_some(())
    }

    fn message(&self, degree: usize, point: &[F]) -> Result<SecretBuf<F>, WindowError> {
        let mut out = buffer::<F>(degree)?;
        let prefix = self.eq_prefix(point);
        if let Some(device) = &self.device {
            use core::any::Any;
            use tessera_field::GoldilocksExt2 as Fp2;
            let suffix_eq = Weights::from_eq(&self.spec.eq[self.bound + 1..])?;
            let (low, high) = (
                (&suffix_eq.low as &dyn Any).downcast_ref::<Vec<Fp2>>(),
                (&suffix_eq.high as &dyn Any).downcast_ref::<Vec<Fp2>>(),
            );
            let (Some(low), Some(high)) = (low, high) else {
                return Err(WindowError::Source);
            };
            let sums = device.sums(low, high, suffix_eq.low_bits)?;
            for (slot, value) in out.iter_mut().enumerate() {
                let sum = *(&sums[slot] as &dyn Any)
                    .downcast_ref::<F>()
                    .ok_or(WindowError::Source)?;
                let x = integer::<F>(if slot == 0 { 0 } else { slot + 1 });
                *value = sum * (prefix * eq_at(self.spec.eq[self.bound], x));
            }
            return Ok(out);
        }
        if let Some(grid) = &self.grid {
            let rest = grid.end - self.bound - 1;
            let mut line = buffer::<F>(grid.radix)?;
            for vertex in 0..1 << rest {
                let offset = radix_index(vertex, grid.radix, rest) * grid.radix;
                let weight = boolean_eq(&self.spec.eq[self.bound + 1..grid.end], vertex);
                for (node, value) in line.iter_mut().enumerate() {
                    *value += weight * grid.values[offset + node];
                }
            }
            for (slot, value) in out.iter_mut().enumerate() {
                let node = if slot == 0 { 0 } else { slot + 1 };
                let x = integer::<F>(node);
                let weights = basis(grid.radix, x)?;
                *value = prefix
                    * eq_at(self.spec.eq[self.bound], x)
                    * line
                        .iter()
                        .zip(weights)
                        .fold(F::ZERO, |s, (&v, w)| s + v * w);
            }
        } else {
            let len = 1 << (self.spec.vars - self.bound - 1);
            let suffix_eq = Weights::from_eq(&self.spec.eq[self.bound + 1..])?;
            // Slot `s` evaluates at x = 0, 2, 3, …: the round's first, third, … values.
            let xs: Vec<F> = (0..degree)
                .map(|slot| integer::<F>(if slot == 0 { 0 } else { slot + 1 }))
                .collect();
            let columns = self.folded.len();
            let shape = Shape::of(&self.spec.terms, columns, &xs);
            // Σ_pairs suffix·P(line(x)) per slot; the public factor prefix·eq(z, x) is applied
            // once after the sum, which distributivity makes the same field element.
            let parts = par_map(len, |pairs| {
                if let Some(shape) = shape {
                    return shape.sums(&self.folded, &suffix_eq, pairs, degree);
                }
                let mut part = buffer::<F>(degree)?;
                let mut base = buffer::<F>(columns)?;
                let mut slope = buffer::<F>(columns)?;
                let mut line = buffer::<F>(columns)?;
                for pair in pairs {
                    let suffix = suffix_eq.at(pair);
                    for (k, table) in self.folded.iter().enumerate() {
                        base[k] = table[2 * pair];
                        slope[k] = table[2 * pair + 1] - base[k];
                    }
                    for (value, &x) in part.iter_mut().zip(&xs) {
                        for k in 0..columns {
                            line[k] = base[k] + x * slope[k];
                        }
                        let polynomial = self.spec.terms.iter().fold(F::ZERO, |s, term| {
                            s + term
                                .factors()
                                .iter()
                                .fold(*term.weight(), |v, &k| v * line[k])
                        });
                        *value += suffix * polynomial;
                    }
                }
                Ok(part)
            })?;
            for part in parts {
                for (value, p) in out.iter_mut().zip(part.iter()) {
                    *value += *p;
                }
            }
            for (value, &x) in out.iter_mut().zip(&xs) {
                *value *= prefix * eq_at(self.spec.eq[self.bound], x);
            }
        }
        Ok(out)
    }

    fn bind(&mut self, r: F) -> Result<(), WindowError> {
        if let Some(device) = &mut self.device {
            use core::any::Any;
            let r = *(&r as &dyn Any)
                .downcast_ref::<tessera_field::GoldilocksExt2>()
                .ok_or(WindowError::Source)?;
            device.bind(r)?;
            self.bound += 1;
            return Ok(());
        }
        if let Some(grid) = &mut self.grid {
            let weights = basis(grid.radix, r)?;
            let len = grid.values.len() / grid.radix;
            for index in 0..len {
                grid.values[index] = (0..grid.radix).fold(F::ZERO, |s, j| {
                    s + grid.values[index * grid.radix + j] * weights[j]
                });
            }
            grid.values.truncate(len);
        } else {
            for table in &mut self.folded {
                let len = table.len() / 2;
                if ranges(len).len() <= 1 {
                    for index in 0..len {
                        table[index] =
                            table[2 * index] + r * (table[2 * index + 1] - table[2 * index]);
                    }
                    table.truncate(len);
                    continue;
                }
                // Out of place across threads: in place, one thread's writes would overlap
                // another's reads.
                let source: &SecretBuf<F> = table;
                let halves = par_map(len, |range| {
                    let mut out = buffer::<F>(range.len())?;
                    for (o, index) in out.iter_mut().zip(range) {
                        *o = source[2 * index] + r * (source[2 * index + 1] - source[2 * index]);
                    }
                    Ok(out)
                })?;
                let mut bound =
                    SecretBuf::try_with_capacity(len).map_err(|_| WindowError::Allocation)?;
                for half in &halves {
                    bound.extend(half.iter().copied());
                }
                *table = bound;
            }
        }
        self.bound += 1;
        Ok(())
    }
}

/// A prepared mixed batch, whose true sums are available before gamma/splits. No transcript
/// input is consumed on a shape or source failure during preparation.
#[must_use]
pub struct PreparedWindow<F: PolyField> {
    states: Vec<State<F>>,
    degree: usize,
    rounds: usize,
    cut: usize,
    window_vars: usize,
}

impl<F: PolyField> PreparedWindow<F> {
    /// Prepare P3. `cut` is the number of global rounds answered using grids; it must leave
    /// the final global round resident. Grids cover at most eight own variables per pass.
    /// Protocol Two is refused until masking's earlier activation is covered by the planner.
    pub fn new<S: WindowSource<F> + ?Sized>(
        source: &mut S,
        protocol: Protocol,
        degree: usize,
        instances: Vec<WindowInstance<F>>,
        cut: usize,
    ) -> Result<Self, WindowError> {
        Self::with_window_vars(source, protocol, degree, instances, cut, MAX_WINDOW_VARS)
    }

    /// Prepare grids with between one and eight own variables per pass.
    /// One-variable grids implement RFC-0054's round-by-round table streaming;
    /// the ordinary reduction-stage constructor retains eight-variable windows.
    /// Geometry and source failures have the same semantics as [`Self::new`].
    pub fn with_window_vars<S: WindowSource<F> + ?Sized>(
        source: &mut S,
        protocol: Protocol,
        degree: usize,
        instances: Vec<WindowInstance<F>>,
        cut: usize,
        window_vars: usize,
    ) -> Result<Self, WindowError> {
        if !(1..=MAX_WINDOW_VARS).contains(&window_vars) {
            return Err(WindowError::Shape(
                "a grid covers one through eight own variables",
            ));
        }
        if protocol != Protocol::One {
            return Err(WindowError::Shape("masked windowing is not implemented"));
        }
        if degree == 0 || degree > 7 || instances.is_empty() {
            return Err(WindowError::Shape(
                "a nonempty batch needs degree 1 through 7",
            ));
        }
        let rounds = instances.iter().map(|s| s.vars).max().unwrap_or(0);
        if rounds >= usize::BITS as usize
            || (rounds != 0 && cut >= rounds)
            || (rounds == 0 && cut != 0)
        {
            return Err(WindowError::Shape(
                "the cut must leave the final round resident",
            ));
        }
        for spec in &instances {
            if spec.eq.len() != spec.vars || spec.tables.is_empty() || spec.terms.is_empty() {
                return Err(WindowError::Shape(
                    "a relation needs tables, terms and one EQ option per variable",
                ));
            }
            let eq_degree = usize::from(spec.eq.iter().any(Option::is_some));
            for term in &spec.terms {
                if term.factors().is_empty()
                    || term.factors().len() + eq_degree > degree
                    || term.factors().iter().any(|&k| k >= spec.tables.len())
                {
                    return Err(WindowError::Shape(
                        "a term must fit the table list and degree",
                    ));
                }
            }
        }
        let first_padding = instances
            .iter()
            .filter(|s| s.windowed && cut > rounds - s.vars)
            .map(|s| rounds - s.vars)
            .min();
        let first_boundary = first_padding.map_or(0, |p| cut.min(p + window_vars));
        source.regenerate(WindowPass::Prepare)?;
        let mut states = Vec::with_capacity(instances.len());
        for mut spec in instances {
            spec.windowed &= cut > rounds - spec.vars;
            let end = first_boundary.saturating_sub(rounds - spec.vars);
            let mut state = State {
                spec,
                grid: None,
                folded: Vec::new(),
                device: None,
                bound: 0,
                sum: buffer(1)?,
            };
            state.prepare(source, &[], if state.spec.windowed { end } else { 0 }, true)?;
            states.push(state);
        }
        source.finish_pass(WindowPass::Prepare)?;
        Ok(Self {
            states,
            degree,
            rounds,
            cut,
            window_vars,
        })
    }

    /// True unpadded sum of instance `i`, for the previous stage's split value.
    pub fn sum(&self, instance: usize) -> Option<F> {
        self.states.get(instance).map(|state| state.sum[0])
    }

    /// Emit the unchanged RFC-0010 round/final-evaluation messages. The source is regenerated
    /// once after each window, and the next grid is accumulated in that same P3b pass.
    pub fn prove<T, S>(
        self,
        transcript: &mut T,
        prefix: Prefix,
        source: &mut S,
    ) -> Result<ProvenBatch<F>, WindowError>
    where
        T: SumcheckTranscript<F> + ?Sized,
        S: WindowSource<F> + ?Sized,
    {
        let mut labels = Labels::new(prefix);
        let gamma = transcript.challenge(labels.gamma());
        self.prove_with_gamma(transcript, prefix, source, gamma)
    }

    /// Prove using the batching challenge already drawn by the caller.
    ///
    /// The caller must draw gamma at `<prefix>/gamma` before constructing dependent
    /// relations. Tree levels then draw lambda, as RFC-0014 requires. This method
    /// never squeezes gamma again; round/challenge/evaluation labels are unchanged.
    pub fn prove_with_gamma<T, S>(
        mut self,
        transcript: &mut T,
        prefix: Prefix,
        source: &mut S,
        gamma: F,
    ) -> Result<ProvenBatch<F>, WindowError>
    where
        T: SumcheckTranscript<F> + ?Sized,
        S: WindowSource<F> + ?Sized,
    {
        let mut labels = Labels::new(prefix);
        let mut rounds = Vec::with_capacity(self.rounds);
        let mut point = Vec::with_capacity(self.rounds);
        let mut boundary = self
            .states
            .iter()
            .filter(|s| s.spec.windowed)
            .map(|s| self.rounds - s.spec.vars)
            .min()
            .map_or(0, |p| self.cut.min(p + self.window_vars));
        for j in 0..self.rounds {
            let mut message = vec![F::ZERO; self.degree];
            let mut weight = F::ONE;
            for state in &self.states {
                let padding = self.rounds - state.spec.vars;
                if j < padding {
                    let value =
                        weight * state.sum[0] * F::from(padding_scale::<F::Base>(padding - j - 1));
                    for out in &mut message {
                        *out += value;
                    }
                } else {
                    let values = state.message(self.degree, &point[padding..])?;
                    for (out, value) in message.iter_mut().zip(values.iter().copied()) {
                        *out += weight * value;
                    }
                }
                weight *= gamma;
            }
            transcript.absorb_message(labels.round(j), &message);
            let r = transcript.challenge(labels.challenge(j));
            for state in &mut self.states {
                if j >= self.rounds - state.spec.vars {
                    state.bind(r)?;
                }
            }
            rounds.push(message);
            point.push(r);
            if boundary != 0 && j + 1 == boundary {
                source.regenerate(WindowPass::Fold(boundary))?;
                let next = (boundary + self.window_vars).min(self.cut);
                for state in &mut self.states {
                    if state.spec.windowed {
                        let padding = self.rounds - state.spec.vars;
                        let bound = boundary.saturating_sub(padding);
                        let end = next.saturating_sub(padding);
                        // At the final boundary, materialize the remaining folded tables.
                        if boundary == self.cut {
                            state.spec.windowed = false;
                        }
                        state.prepare(
                            source,
                            &point[padding.min(point.len())..],
                            end.max(bound),
                            false,
                        )?;
                    }
                }
                source.finish_pass(WindowPass::Fold(boundary))?;
                boundary = if boundary == self.cut { 0 } else { next };
            }
        }
        let mut final_evals = Vec::new();
        let mut evals = Vec::with_capacity(self.states.len());
        let vars = self.states.iter().map(|s| s.spec.vars).collect();
        for state in &self.states {
            let start = final_evals.len();
            if let Some(device) = &state.device {
                use core::any::Any;
                for (table, value) in state.spec.tables.iter().zip(device.firsts()?) {
                    if !table.public {
                        final_evals.push(
                            *(&value as &dyn Any)
                                .downcast_ref::<F>()
                                .ok_or(WindowError::Source)?,
                        );
                    }
                }
            } else {
                for (table, values) in state.spec.tables.iter().zip(&state.folded) {
                    if !table.public {
                        final_evals.push(values[0]);
                    }
                }
            }
            evals.push(start..final_evals.len());
        }
        transcript.absorb_message(labels.evals(), &final_evals);
        Ok(ProvenBatch::from_parts(
            SumcheckProof {
                rounds,
                final_evals,
            },
            gamma,
            point,
            vars,
            evals,
        ))
    }
}

#[cfg(test)]
mod tests;
