// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A LogUp instance: a table, the lookups into it and their multiplicities (RFC-0008 'Relation',
//! 'Interfaces').

use core::fmt;

use tessera_codec::{Secret, SecretBuf, Zeroize};
use tessera_field::PrimeField;

use crate::{Column, LogUpError};

/// Pairs `(key, value)` over the prime subfield, as two columns: a table's rows or an instance's
/// lookups (RFC-0008 'Relation'). A table without values, such as `Range`, has none, and so have
/// its lookups; the challenge `β` compresses a pair to `key + β·value`, and a key alone to itself.
///
/// The debug form shows the lengths, never an entry: lookups and their values are secret-class
/// values (05 §3).
pub struct Tuples<'a, B> {
    /// The keys, one per pair.
    pub keys: &'a [B],
    /// The values, one per pair, or `None` for a table without values.
    pub values: Option<&'a [B]>,
}

impl<B> Clone for Tuples<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for Tuples<'_, B> {}

impl<'a, B> Tuples<'a, B> {
    /// Keys and values, pair by pair: a public table, or test data.
    pub fn new(keys: &'a [B], values: Option<&'a [B]>) -> Self {
        Self { keys, values }
    }

    /// Secret-class keys and values, such as an instance's lookups, each column in its
    /// [`Secret`]: the prover reads them to build its trees, which clear their levels as they drop
    /// (05 §3 OBS-1, OBS-6).
    #[allow(clippy::disallowed_methods)]
    pub fn secret(keys: &'a Secret<Vec<B>>, values: Option<&'a Secret<Vec<B>>>) -> Self
    where
        B: Zeroize,
    {
        Self::new(
            keys.expose_secret(),
            values.map(Secret::expose_secret).map(Vec::as_slice),
        )
    }

    /// The number of pairs: the number of keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether there is no pair.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Whether the pairs have values.
    pub fn has_values(&self) -> bool {
        self.values.is_some()
    }
}

/// Shows the lengths, never an entry (05 §3).
impl<B> fmt::Debug for Tuples<'_, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tuples")
            .field("keys", &self.keys.len())
            .field("values", &self.values.map(<[B]>::len))
            .finish()
    }
}

/// One LogUp instance (RFC-0008 'Relation', 'Interfaces'): the `N ≥ 1` lookups
/// `a_0, …, a_{N−1}` into the public table `t_0, …, t_{T−1}`, `T ≥ 1`, with the committed
/// multiplicities `m_j`, which it proves to satisfy
/// `Σ_{i<N} 1/(α − a_i) − Σ_j m_j/(α − t_j) = 0` for the challenge `α`.
///
/// - **The lookups** are the virtual keys and the committed values of the lookup side, in leaf
///   order: lookup `i` is leaf `i` of the lookup tree, of `2^{h_a}` leaves with `h_a = ⌈log2 N⌉`.
///   They have values exactly when the table does.
/// - **The multiplicities** are one per leaf of the table tree, `2^{h_t}` with `h_t = ⌈log2 T⌉`,
///   committed before `α`: `m_j` counts the lookups of row `j`, and `m_0` also the lookup tree's
///   `2^{h_a} − N` padding leaves, which repeat row 0. The padding rows `T, …, 2^{h_t} − 1` repeat
///   row 0 too, and an honest prover gives them 0. [`multiplicities`] counts them from each
///   lookup's row.
///
/// The prover takes these columns as they are: a lookup that is no row, a wrong value or a wrong
/// multiplicity yields honest trees over its leaves, whose roots the verifier rejects with
/// probability at least `1 − (N + T)/|F|` (RFC-0008 'Relation').
///
/// The debug form shows the shapes, never an entry (05 §3).
pub struct LogUpInstance<'a, B> {
    /// The table's rows: its keys, and its values when it has them.
    pub table: Tuples<'a, B>,
    /// The lookups, in leaf order.
    pub lookups: Tuples<'a, B>,
    /// `m_j` for each of the `2^{h_t}` leaves of the table tree.
    pub multiplicities: &'a [B],
}

impl<B> Clone for LogUpInstance<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for LogUpInstance<'_, B> {}

impl<'a, B> LogUpInstance<'a, B> {
    /// The instance of `lookups` into `table` with the committed `multiplicities`.
    pub fn new(table: Tuples<'a, B>, lookups: Tuples<'a, B>, multiplicities: &'a [B]) -> Self {
        Self {
            table,
            lookups,
            multiplicities,
        }
    }

    /// `h_a = ⌈log2 N⌉`: the lookup tree's height.
    pub fn lookup_height(&self) -> usize {
        ceil_log2(self.lookups.len())
    }

    /// `h_t = ⌈log2 T⌉`: the table tree's height.
    pub fn table_height(&self) -> usize {
        ceil_log2(self.table.len())
    }

    /// The first shape check the instance fails, in the order of [`LogUpError`].
    pub(crate) fn check(&self, instance: usize) -> Result<(), LogUpError> {
        let rows = self.table.len();
        if rows == 0 {
            return Err(LogUpError::NoRows { instance });
        }
        if let Some(values) = self.table.values {
            length(instance, Column::TableValues, values.len(), rows)?;
        }
        let lookups = self.lookups.len();
        if lookups == 0 {
            return Err(LogUpError::NoLookups { instance });
        }
        match (self.table.values, self.lookups.values) {
            (Some(_), Some(values)) => {
                length(instance, Column::LookupValues, values.len(), lookups)?;
            }
            (None, None) => {}
            _ => return Err(LogUpError::ValuesMismatch { instance }),
        }
        length(
            instance,
            Column::Multiplicities,
            self.multiplicities.len(),
            1 << self.table_height(),
        )
    }
}

/// Shows the shapes, never an entry (05 §3).
impl<B> fmt::Debug for LogUpInstance<'_, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogUpInstance")
            .field("table", &self.table)
            .field("lookups", &self.lookups)
            .field("multiplicities", &self.multiplicities.len())
            .finish()
    }
}

fn length(instance: usize, column: Column, len: usize, expected: usize) -> Result<(), LogUpError> {
    match len == expected {
        true => Ok(()),
        false => Err(LogUpError::ColumnLength {
            instance,
            column,
            len,
            expected,
        }),
    }
}

/// `⌈log2 n⌉`, 0 for `n ≤ 1`. A slice's length is below `2^63`, so `2^⌈log2 n⌉` is a `usize`.
pub(crate) fn ceil_log2(n: usize) -> usize {
    match n {
        0 | 1 => 0,
        _ => (usize::BITS - (n - 1).leading_zeros()) as usize,
    }
}

/// The multiplicities an honest prover commits for lookups into `table`, each lookup given by its
/// row: one per leaf of the table tree, `2^{h_t}` entries with `h_t = ⌈log2 T⌉` (RFC-0008
/// 'Fraction tree').
///
/// - `m_j`, for a row `j < T`, counts the lookups of row `j`.
/// - `m_0` also counts the lookup tree's `2^{h_a} − N` padding leaves, `N` the lookups and
///   `h_a = ⌈log2 N⌉`: each repeats row 0, `(1, α − t_0)`.
/// - The padding rows `T, …, 2^{h_t} − 1` repeat row 0, and get 0.
///
/// A table of no row has no leaf and so no multiplicity, and no lookup no padding leaf.
/// Multiplicities are field elements: a count stays below `2^{h_a} < 2^{61}` and is reduced
/// modulo the field's characteristic.
///
/// Counting increments the counter of each lookup's row, so the memory it touches depends on the
/// rows. INV-ARCH-2 allows that and nothing more: control flow follows the lengths only, and a
/// verifier times the prover only through the release schedule of RFC-0018, which public data
/// fixes.
///
/// The rows, which lookup hits which row, are secret-class: they come in their [`Secret`], and
/// the counts and the multiplicities, a histogram of them, are [`SecretBuf`]s, cleared when they
/// drop (05 §3 OBS-1, OBS-6).
///
/// # Errors
///
/// [`LogUpError::LookupRow`] for the first lookup whose row is not below `T`.
#[allow(clippy::disallowed_methods)]
pub fn multiplicities<B: PrimeField>(
    table: Tuples<'_, B>,
    lookups: &Secret<Vec<u32>>,
) -> Result<SecretBuf<B>, LogUpError> {
    let lookups = lookups.expose_secret();
    let rows = table.len();
    let leaves = match rows {
        0 => 0,
        _ => 1 << ceil_log2(rows),
    };
    let mut counts = SecretBuf::filled(0u64, leaves);
    for (lookup, &row) in lookups.iter().enumerate() {
        let row = row as usize;
        match counts.get_mut(row) {
            Some(count) if row < rows => *count += 1,
            _ => return Err(LogUpError::LookupRow { lookup, rows }),
        }
    }
    if let Some(first) = counts.first_mut()
        && !lookups.is_empty()
    {
        let padding = (1usize << ceil_log2(lookups.len())) - lookups.len();
        *first += padding as u64;
    }
    Ok(counts
        .iter()
        .map(|&count| B::from_u64_reduced(count))
        .collect())
}
