// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The leaf stage's secret side (RFC-0045 'The column', 'The leaves', 'The split'): each
//! instance's lookup column in leaf order with its multiplicities, the LogUp proof, the leaf point
//! `P`, and the split values at it.
//!
//! The tensors a site's lookups read are secret-class (05 §3): the caller lends them, by
//! reference and for the call, in [`SiteTensors`]. What is computed from them, the columns and the
//! multiplicities, lives in `SecretBuf`s that clear their memory (OBS-6), and the rows the
//! multiplicities count in a `Secret`. Control flow depends on the layout, which is public, and
//! on the lengths (INV-ARCH-2): a key that names no row, the one outcome that depends on the
//! values, is reported after every key of its instance is converted.

use core::fmt;

use tessera_codec::{Secret, SecretBuf};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField};
use tessera_gadgets::{TableColumns, Term};
use tessera_logup::{LogUpInstance, ProvenLogUp, TreeId, Tuples, multiplicities, prove_logup};
use tessera_poly::evaluate_base;
use tessera_sumcheck::{Forest, SumcheckTranscript};
use tessera_tables::TableId;

use super::layout::{Block, InstanceLayout, LeafLayout, Read, leaf_index};
use super::{LeafError, internal};

/// The label of the leaf stage's split values, stage 0's (RFC-0037 'Splits', RFC-0045 'What is
/// sent').
pub const SPLIT_LABEL: &str = "stage/0/split";

/// The tensors one site's lookups read in one layer, each as its gadget's term with its values:
/// over the tensor's own axes, padded per dimension with what the node computes there (RFC-0011
/// rule 1, RFC-0035), row-major, `2^{|π_t|}` entries ([`Read::coords`]).
///
/// The values are secret-class (05 §3), lent for the call. The debug form shows the terms and the
/// lengths, never a value.
pub struct SiteTensors<'a, S = Term> {
    terms: Vec<(S, &'a [i64])>,
}

impl<'a, S: Copy + Eq> SiteTensors<'a, S> {
    /// The tensors `terms`, each term once.
    pub fn new(terms: Vec<(S, &'a [i64])>) -> Self {
        Self { terms }
    }

    /// The values of `read`'s tensor, after checking that it is given once with `2^{|π_t|}`
    /// entries.
    pub(super) fn of(&self, read: &Read<S>) -> Result<&'a [i64], LeafError> {
        let mut found = self.terms.iter().filter(|(term, _)| *term == read.term);
        let (Some(&(_, values)), None) = (found.next(), found.next()) else {
            return internal("the witness gives each tensor a site's lookups read, once");
        };
        let expected = u32::try_from(read.coords.len())
            .ok()
            .and_then(|vars| 1usize.checked_shl(vars));
        match Some(values.len()) == expected {
            true => Ok(values),
            false => internal("a tensor holds 2^k entries, its axes padded per dimension"),
        }
    }
}

/// Shows the terms and the lengths, never a value (05 §3).
impl<S: fmt::Debug> fmt::Debug for SiteTensors<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.terms.iter().map(|(term, values)| (term, values.len())))
            .finish()
    }
}

/// One instance's lookup column in leaf order, and its multiplicities in row order (RFC-0045 'The
/// leaves', 'The column').
///
/// The columns and the multiplicities are secret-class (05 §3), in `SecretBuf`s, and lent to the
/// code that holds them. The debug form shows the shapes, never an entry.
pub struct InstanceColumns {
    table: TableId,
    keys: SecretBuf<Goldilocks>,
    values: Option<SecretBuf<Goldilocks>>,
    multiplicities: SecretBuf<Goldilocks>,
}

impl InstanceColumns {
    /// The table.
    pub fn table(&self) -> TableId {
        self.table
    }

    /// The `2^h` lookups' keys in leaf order: leaf `i` holds column entry `σ_h(i)`, the padding
    /// included.
    ///
    /// The keys are secret-class (05 §3), lent to the code that computes on them, such as the
    /// LogUp prover. Their holder puts no key, and nothing derived from one, in a log, an error, a
    /// metric or a report; hands them only to code that `secret-access.toml` lists (OBS-1); and
    /// copies none out, since the clearing cannot reach a copy (OBS-6).
    pub fn keys(&self) -> &[Goldilocks] {
        &self.keys
    }

    /// The lookups' values in leaf order, `None` for a table without values. Secret-class and lent
    /// as the [`keys`](Self::keys) are, under the same rules.
    pub fn values(&self) -> Option<&[Goldilocks]> {
        self.values.as_deref()
    }

    /// The multiplicities an honest prover commits, one per row of the table tree: each lookup
    /// counts its row, the padding entries row 0. Secret-class and lent as the
    /// [`keys`](Self::keys) are, under the same rules.
    pub fn multiplicities(&self) -> &[Goldilocks] {
        &self.multiplicities
    }
}

/// Shows the shapes, never an entry (05 §3).
impl fmt::Debug for InstanceColumns {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InstanceColumns")
            .field("table", &self.table)
            .field("keys", &self.keys.len())
            .field("values", &self.values.as_ref().map(|v| v.len()))
            .field("multiplicities", &self.multiplicities.len())
            .finish()
    }
}

/// Every instance's lookup column in leaf order, with its multiplicities, in table order: the
/// lookups the LogUp prover takes (RFC-0045 'The leaves').
#[derive(Debug)]
pub struct LeafColumns {
    instances: Vec<InstanceColumns>,
}

impl LeafColumns {
    /// The columns of `layout`'s instances from `witness`, which holds each site's tensors in each
    /// of its layers, `witness[s][l]` for site `s` and layer `l` (RFC-0045 'The column', 'The
    /// leaves'):
    ///
    /// - each block's entries are its lookup's key and value forms, evaluated element by element
    ///   over its domain from the tensors, each read through its axes;
    /// - column entry `q` is leaf `σ_h(q)`, and entries `m` to `2^h − 1` hold row 0's tuple, so
    ///   the instance has `2^h` explicit lookups and `tessera-logup` pads none;
    /// - the multiplicities count each lookup's row, its key, in the table's row order.
    ///
    /// `tables` holds the columns of every table an instance reads, in any order.
    ///
    /// # Errors
    ///
    /// [`LeafError::Internal`] (`ERR-INTERNAL`) for the first rule broken: a witness without one
    /// entry per site and real layer; then instance by instance in table order, an instance's
    /// table without columns among `tables` or without a row, and block by block a lookup with
    /// values into a table without them, or the other way round, and a tensor its gadget reads
    /// missing, given twice or of another length; then a key that names no row of its table,
    /// which an honest witness never has (INV-IR-2). [`LeafError::Memory`] when a column cannot
    /// be allocated.
    pub fn new<S: Copy + Eq>(
        layout: &LeafLayout<S>,
        tables: &[TableColumns<Goldilocks>],
        witness: &[Vec<SiteTensors<'_, S>>],
    ) -> Result<Self, LeafError> {
        check_witness(layout, witness)?;
        let mut instances = Vec::with_capacity(layout.instances().len());
        for instance in layout.instances() {
            let rows = columns(tables, instance.table)?.tuples();
            let (keys, values) = lookup_columns(layout, instance, rows, witness, |_, x| Some(x))?;
            let multiplicities = count(rows, &keys)?;
            instances.push(InstanceColumns {
                table: instance.table,
                keys,
                values,
                multiplicities,
            });
        }
        Ok(Self { instances })
    }

    /// The instances' columns, in table order.
    pub fn columns(&self) -> &[InstanceColumns] {
        &self.instances
    }

    /// The LogUp instances, in table order: each table's rows from `tables`, its column's `2^h`
    /// lookups and its multiplicities.
    ///
    /// # Errors
    ///
    /// [`LeafError::Internal`] for an instance's table without columns among `tables`.
    pub fn instances<'a>(
        &'a self,
        tables: &'a [TableColumns<Goldilocks>],
    ) -> Result<Vec<LogUpInstance<'a, Goldilocks>>, LeafError> {
        self.instances
            .iter()
            .map(|instance| {
                let table = columns(tables, instance.table)?;
                Ok(LogUpInstance::new(
                    table.tuples(),
                    Tuples::new(&instance.keys, instance.values.as_deref()),
                    &instance.multiplicities,
                ))
            })
            .collect()
    }
}

/// One instance's lookups in leaf order (RFC-0045 'The column', 'The leaves'), keys then values:
/// each block's key and value forms, evaluated entry by entry from the tensors of `stage`'s
/// `witness`, each read through its axes, and row 0's tuple in the padding. Entry `x` of block `b`
/// is the stage block's entry `source(b, x)`: `x` itself for the stage's own instances, a group's
/// share of it for a group's (RFC-0054 'What a group holds'); `None` is refused.
pub(super) fn lookup_columns<S: Copy + Eq>(
    stage: &LeafLayout<S>,
    instance: &InstanceLayout,
    rows: Tuples<'_, Goldilocks>,
    witness: &[Vec<SiteTensors<'_, S>>],
    source: impl Fn(&Block, u64) -> Option<u64>,
) -> Result<(SecretBuf<Goldilocks>, Option<SecretBuf<Goldilocks>>), LeafError> {
    let key_0 = *rows.keys.first().ok_or(LeafError::Internal {
        rule: "an instance's table has a row",
    })?;
    let value_0 = rows.values.and_then(|v| v.first().copied());
    let leaves = 1usize
        .checked_shl(instance.height)
        .ok_or(LeafError::Memory)?;
    // Every entry starts as row 0's tuple: the padding stays it, the blocks overwrite the rest.
    let mut keys = filled(key_0, leaves)?;
    let mut values = match value_0 {
        Some(value) => Some(filled(value, leaves)?),
        None => None,
    };
    for block in &instance.blocks {
        let gadget = &stage.sites()[block.site].layers[block.layer];
        let lookup = &gadget.lookups[block.lookup].lookup;
        if lookup.value.is_some() != values.is_some() {
            return internal("a lookup has values exactly when its table does");
        }
        let reads = stage
            .reads(block.site, block.layer)
            .ok_or(LeafError::Internal {
                rule: "the layout reads every site's tensors",
            })?;
        let tensors = &witness[block.site][block.layer];
        let terms: Vec<(S, &[i64], &[u32])> = reads
            .iter()
            .map(|read| Ok((read.term, tensors.of(read)?, read.coords.as_slice())))
            .collect::<Result<_, LeafError>>()?;
        // The value of `term` at the domain's element `x`: its tensor's entry at the bits of `x`
        // that its axes take. Every term of the gadget's forms is among its reads, which the
        // layout derives from those forms, so `find` never falls back.
        let at = |term: S, x: u64| -> Goldilocks {
            terms.iter().find(|(t, _, _)| *t == term).map_or(
                Goldilocks::ZERO,
                |&(_, values, coords)| {
                    let index = coords
                        .iter()
                        .enumerate()
                        .fold(0usize, |i, (j, &c)| i | ((((x >> c) & 1) as usize) << j));
                    Goldilocks::from_i64(values[index])
                },
            )
        };
        for j in 0..block.entries() {
            let x = source(block, j).ok_or(LeafError::Internal {
                rule: "a block's entries are its stage block's",
            })?;
            let leaf = leaf_index(instance.height, block.offset + j) as usize;
            keys[leaf] = lookup.key.evaluate::<Goldilocks>(|term| at(term, x));
            if let (Some(column), Some(form)) = (&mut values, &lookup.value) {
                column[leaf] = form.evaluate::<Goldilocks>(|term| at(term, x));
            }
        }
    }
    Ok((keys, values))
}

/// The leaf point `P = (ρ_{H−1}, r_{n−1}, …, r_0)` of a proven leaf stage (RFC-0045 'The leaf
/// point'): the last tree level's line challenge, then its round challenges from the last squeezed
/// to the first. Empty when no tree has a level, `H = 0`.
///
/// An instance of height `h` has its key and value claims, read in column order, at `P[0..h]`, and
/// a block of `2^k` entries its claims at `P[0..k]`, in every instance.
pub fn leaf_point<F: Copy>(proven: &ProvenLogUp<F>) -> Vec<F> {
    super::grouped::level_point(proven.levels())
}

/// The split values (RFC-0045 'What is sent'): for each site in node order, each of its real
/// layers in order and each tensor its lookups read, in the order they first read them, the
/// tensor's extension at its point `π_t(P[0..k])`, `point` being `P`.
///
/// # Errors
///
/// [`LeafError::Internal`] for a witness without one gadget per site and layer, a tensor missing,
/// given twice or of another length, or a point shorter than a tensor's coordinates;
/// [`LeafError::Memory`] when a table cannot be allocated.
pub fn split_values<S: Copy + Eq>(
    layout: &LeafLayout<S>,
    point: &[GoldilocksExt2],
    witness: &[Vec<SiteTensors<'_, S>>],
) -> Result<Vec<GoldilocksExt2>, LeafError> {
    check_witness(layout, witness)?;
    let mut values = Vec::with_capacity(layout.split_len());
    for (s, site) in layout.sites().iter().enumerate() {
        for (l, tensors) in witness[s].iter().enumerate().take(site.layers.len()) {
            let reads = layout.reads(s, l).ok_or(LeafError::Internal {
                rule: "the layout reads every site's tensors",
            })?;
            for read in reads {
                let entries = tensors.of(read)?;
                let at = read
                    .coords
                    .iter()
                    .map(|&c| point.get(c as usize).copied())
                    .collect::<Option<Vec<_>>>()
                    .ok_or(LeafError::Internal {
                        rule: "the leaf point holds every tensor's coordinates",
                    })?;
                let mut table =
                    SecretBuf::try_with_capacity(entries.len()).map_err(|_| LeafError::Memory)?;
                table.extend(entries.iter().map(|&v| Goldilocks::from_i64(v)));
                let value = evaluate_base::<GoldilocksExt2>(&table, &at).map_err(|_| {
                    LeafError::Internal {
                        rule: "a tensor's point has its variables",
                    }
                })?;
                values.push(value);
            }
        }
    }
    Ok(values)
}

/// A proven leaf stage's LogUp part and split (RFC-0045): the LogUp proof of the instances, the
/// leaf point `P`, and the split values, absorbed under `stage/0/split`.
///
/// The messages, in transcript order: the LogUp proof's roots, levels and values, then the
/// [`splits`](Self::splits). What the checks leave, each split value as a claim on its tensor at
/// its point and each instance's multiplicity claim, later stages and the openings owe; a caller
/// that drops them accepts what nothing checked.
#[must_use]
pub struct ProvenLeaf {
    logup: ProvenLogUp<GoldilocksExt2>,
    point: Vec<GoldilocksExt2>,
    splits: Vec<GoldilocksExt2>,
}

impl ProvenLeaf {
    /// The LogUp proof, in table order. Its instances' key and value claims are at the lookup
    /// trees' leaf points in leaf order, which are `P[0..h]` in column order.
    pub fn logup(&self) -> &ProvenLogUp<GoldilocksExt2> {
        &self.logup
    }

    /// The leaf point `P` ([`leaf_point`]).
    pub fn point(&self) -> &[GoldilocksExt2] {
        &self.point
    }

    /// The `stage/0/split` message: one value per tensor, site and real layer
    /// ([`split_values`]), followed by any graph atom split values supplied at finish.
    pub fn splits(&self) -> &[GoldilocksExt2] {
        &self.splits
    }

    /// The LogUp proof, the leaf point and the split values, by value.
    pub fn into_parts(
        self,
    ) -> (
        ProvenLogUp<GoldilocksExt2>,
        Vec<GoldilocksExt2>,
        Vec<GoldilocksExt2>,
    ) {
        (self.logup, self.point, self.splits)
    }
}

/// Shows the shapes: the messages and challenges are protocol-class data (05 §3).
impl fmt::Debug for ProvenLeaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProvenLeaf")
            .field("logup", &self.logup)
            .field("point", &self.point.len())
            .field("splits", &self.splits.len())
            .finish()
    }
}

/// The shared tree levels and lookup split values, before `stage/0/split` is absorbed.
///
/// Use the leaf point and forest claims to compute the graph's atom splits, then call
/// [`finish`](Self::finish) before drawing any further challenge (INV-FS-1). Dropping this
/// value leaves the transcript unfinished; it certifies nothing. This value borrows the
/// transcript exclusively so no challenge can be drawn until it is consumed or dropped.
///
/// ```compile_fail,E0499
/// use tessera_prover::leaf::{prepare_leaf, LeafLayout, LeafError, SiteTensors};
/// use tessera_field::{Goldilocks, GoldilocksExt2};
/// use tessera_gadgets::TableColumns;
/// use tessera_sumcheck::SumcheckTranscript;
/// fn interleave(
///     tr: &mut impl SumcheckTranscript<GoldilocksExt2>,
///     layout: &LeafLayout,
///     tables: &[TableColumns<Goldilocks>],
///     witness: &[Vec<SiteTensors<'_>>],
/// ) -> Result<(), LeafError> {
///     let pending = prepare_leaf(tr, layout, tables, witness, &[])?;
///     tr.challenge("stage/1/layer/0"); // split values are not bound yet
///     let _proven = pending.finish(&[])?;
///     Ok(())
/// }
/// ```
#[must_use = "finish the single leaf split message before drawing another challenge"]
pub struct PreparedLeaf<'tr, T: SumcheckTranscript<GoldilocksExt2> + ?Sized> {
    inner: ProvenLeaf,
    transcript: &'tr mut T,
}

impl<T: SumcheckTranscript<GoldilocksExt2> + ?Sized> fmt::Debug for PreparedLeaf<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PreparedLeaf").field(&self.inner).finish()
    }
}

impl<T: SumcheckTranscript<GoldilocksExt2> + ?Sized> PreparedLeaf<'_, T> {
    /// The shared LogUp/forest proof and its outstanding leaf claims.
    pub fn logup(&self) -> &ProvenLogUp<GoldilocksExt2> {
        &self.inner.logup
    }

    /// The leaf point `P` from the last shared tree level.
    pub fn point(&self) -> &[GoldilocksExt2] {
        &self.inner.point
    }

    /// Append the graph's atom splits after the lookup tensor values and absorb exactly one
    /// TCE vector under `stage/0/split` (RFC-0037/0045). `atom_splits` is in the graph
    /// schedule's claim, real-layer and part order, as computed by
    /// [`StageProver::leaf_split_values`](crate::StageProver::leaf_split_values).
    ///
    /// # Errors
    /// [`LeafError::Memory`] if the combined message cannot be allocated; the transcript has
    /// not absorbed a split message on that error.
    pub fn finish(mut self, atom_splits: &[GoldilocksExt2]) -> Result<ProvenLeaf, LeafError> {
        self.inner
            .splits
            .try_reserve(atom_splits.len())
            .map_err(|_| LeafError::Memory)?;
        self.inner.splits.extend_from_slice(atom_splits);
        self.transcript
            .absorb_message(SPLIT_LABEL, &self.inner.splits);
        Ok(self.inner)
    }
}

/// Prove the lookup trees and product forests together, and compute the lookup tensor values
/// at their shared leaf point, stopping before `stage/0/split` (RFC-0010/0037/0045).
///
/// The schedule takes each lookup instance's lookup tree then table tree, in table order,
/// followed by `forests` in their supplied public schedule order. The forest roots must be
/// the public roots of the model's relations; all returned multiplicity, forest and tensor
/// claims still require stages or commitment openings. Nothing here checks them.
///
/// After computing the graph's atom split values, call [`PreparedLeaf::finish`] before
/// drawing any further challenge. [`prove_leaf`] is the standalone lookup-only case.
///
/// # Errors
/// As [`LeafColumns::new`], then [`LeafError::LogUp`] for invalid tree shapes before the
/// transcript moves; [`LeafError::Memory`] if a split table cannot be allocated after LogUp.
pub fn prepare_leaf<'tr, S: Copy + Eq, T: SumcheckTranscript<GoldilocksExt2> + ?Sized>(
    transcript: &'tr mut T,
    layout: &LeafLayout<S>,
    tables: &[TableColumns<Goldilocks>],
    witness: &[Vec<SiteTensors<'_, S>>],
    forests: &[Forest<'_, GoldilocksExt2>],
) -> Result<PreparedLeaf<'tr, T>, LeafError> {
    let columns = LeafColumns::new(layout, tables, witness)?;
    let instances = columns.instances(tables)?;
    let mut schedule = layout.trees();
    schedule.extend((0..forests.len()).map(TreeId::Forest));
    let logup = prove_logup(transcript, &instances, forests, &schedule)?;
    let point = leaf_point(&logup);
    let splits = split_values(layout, &point, witness)?;
    Ok(PreparedLeaf {
        inner: ProvenLeaf {
            logup,
            point,
            splits,
        },
        transcript,
    })
}

/// Proves the LogUp part of the leaf stage over `layout` and its split, on `transcript`
/// (RFC-0045, RFC-0008 'Fraction tree', RFC-0037 'Splits'):
///
/// 1. each instance's column in leaf order, with its multiplicities ([`LeafColumns::new`]);
/// 2. the LogUp instances in table order, each lookup tree followed by its table tree
///    ([`prove_logup`]): `logup/alpha`, `logup/beta`, `logup/roots`, the tree levels and
///    `logup/values`;
/// 3. the leaf point `P` from the last level ([`leaf_point`]);
/// 4. the split values at `P` ([`split_values`]), absorbed as one TCE `Vec<Fp2>` under
///    `stage/0/split`, before any later challenge.
///
/// For product forests and graph atom splits in the same leaf stage, use [`prepare_leaf`]
/// followed by [`PreparedLeaf::finish`]. Here the levels hold the instances' trees alone.
///
/// # Errors
///
/// As [`LeafColumns::new`], and [`LeafError::LogUp`] when the LogUp prover refuses the
/// instances, both before the transcript moves; [`LeafError::Memory`] when a split value's table
/// cannot be allocated, after the LogUp proof.
pub fn prove_leaf<S: Copy + Eq, T>(
    transcript: &mut T,
    layout: &LeafLayout<S>,
    tables: &[TableColumns<Goldilocks>],
    witness: &[Vec<SiteTensors<'_, S>>],
) -> Result<ProvenLeaf, LeafError>
where
    T: SumcheckTranscript<GoldilocksExt2> + ?Sized,
{
    prepare_leaf(transcript, layout, tables, witness, &[])?.finish(&[])
}

/// Checks that `witness` holds one entry per site and real layer of `layout`.
pub(super) fn check_witness<S: Copy + Eq>(
    layout: &LeafLayout<S>,
    witness: &[Vec<SiteTensors<'_, S>>],
) -> Result<(), LeafError> {
    let shaped = witness.len() == layout.sites().len()
        && layout
            .sites()
            .iter()
            .zip(witness)
            .all(|(site, layers)| layers.len() == site.layers.len());
    match shaped {
        true => Ok(()),
        false => internal("the witness gives each site's tensors in each of its layers"),
    }
}

/// The columns of `table` among `tables`.
pub(super) fn columns(
    tables: &[TableColumns<Goldilocks>],
    table: TableId,
) -> Result<&TableColumns<Goldilocks>, LeafError> {
    tables
        .iter()
        .find(|columns| columns.table() == table)
        .ok_or(LeafError::Internal {
            rule: "the caller holds the columns of every instance's table",
        })
}

/// `len` copies of `value`, in a buffer that clears its memory.
fn filled(value: Goldilocks, len: usize) -> Result<SecretBuf<Goldilocks>, LeafError> {
    let mut buffer = SecretBuf::try_with_capacity(len).map_err(|_| LeafError::Memory)?;
    buffer.resize(len, value);
    Ok(buffer)
}

/// The multiplicities of `keys` into `rows`: each key names its row, key `j` being row `j` of
/// every lookup table (RFC-0008 'Table specifications'). Which lookup names which row is
/// secret-class: the rows are counted in their `Secret`, and whether every key fits a row index
/// is folded over the keys without branching (INV-ARCH-2).
pub(super) fn count(
    rows: Tuples<'_, Goldilocks>,
    keys: &[Goldilocks],
) -> Result<SecretBuf<Goldilocks>, LeafError> {
    let mut named = Vec::with_capacity(keys.len());
    let mut fits = true;
    for key in keys {
        let row = u32::try_from(key.to_canonical_u64());
        fits &= row.is_ok();
        named.push(row.unwrap_or(0));
    }
    let named = Secret::new(named);
    let unnamed = LeafError::Internal {
        rule: "every lookup's key names a row of its table (INV-IR-2)",
    };
    match fits {
        // `multiplicities` refuses only a row past `T`: a key that names no row.
        true => multiplicities(rows, &named).map_err(|_| unnamed),
        false => Err(unnamed),
    }
}
