// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A gadget's LogUp instances (RFC-0008 'Relation', 'Batching'): its lookups' key and value
//! columns as field elements, the multiplicities an honest prover commits, and the instances the
//! LogUp prover takes.
//!
//! All lookups into one table definition form one instance (RFC-0008 'Batching'). Within one
//! gadget, lookups that read the same table, such as `NonNeg`'s limbs, form one instance whose
//! columns are theirs one after another, in lookup order: lookup `i` of a table's `m` holds
//! entries `i·N` to `(i + 1)·N − 1` of its columns, `N` the gadget's elements. With `N` a power of
//! two, each lookup's block is a subcube of the instance's lookup tree.
//!
//! The columns are computed from the gadget's [`Constraints`], element by element, so that the
//! description that later stages read is the one the instances were built from. The input, the
//! classes, the committed tensors and every column computed from them are secret-class (05 §3):
//! the columns and the multiplicities live in `SecretBuf`s, and the rows counted for the
//! multiplicities in a `Secret`, both cleared when they drop (OBS-6). Control flow depends on the
//! shapes only (INV-ARCH-2).

use core::fmt;

use tessera_codec::{Secret, SecretBuf};
use tessera_field::PrimeField;
use tessera_logup::{LogUpInstance, Tuples, multiplicities};
use tessera_poly::PolyField;
use tessera_tables::{GeneratedTable, Table, TableId, TableKind};

use crate::{Column, Committed, Constraints, GadgetError, Term, Witness};

/// A table's rows as the LogUp prover's columns: `T` keys, and `T` values for a table with values.
/// Every table a gadget reads has key `j` at row `j` (RFC-0008 'Table specifications').
///
/// The rows are public, and dense: `tessera-logup`'s prover takes a table as its rows' columns
/// (RFC-0008 'Interfaces'), so this holds `T` field elements per column, up to `2^32`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableColumns<B> {
    table: TableId,
    keys: Vec<B>,
    values: Option<Vec<B>>,
}

impl<B: PrimeField> TableColumns<B> {
    /// The rows of `table`, its values mapped to `B` in the centered representation (RFC-0002).
    /// A `Range` table has no values.
    ///
    /// # Errors
    ///
    /// [`GadgetError::Table`] when the table has more rows than `B` has elements: its keys would
    /// not be distinct field elements. That holds for no table over Goldilocks, and for tables of
    /// more than `2^16` rows over `F65537`.
    pub fn new(table: &GeneratedTable) -> Result<Self, GadgetError> {
        let id = table.spec().id;
        let rows = table.row_count();
        if rows > B::MODULUS {
            return Err(GadgetError::Table {
                table: id,
                rule: "a LogUp table has no more rows than its field has elements",
            });
        }
        let keys = (0..rows).map(B::from_u64_reduced).collect();
        let values = match table.spec().kind {
            TableKind::Range { .. } => None,
            _ => Some(
                (0..rows)
                    .map(|j| {
                        // A row below `T` exists, and its value is a true integer.
                        let value = table.row(j).map_or(0, |(_, v)| v.to_centered_i128());
                        B::from_i64(i64::try_from(value).unwrap_or(0))
                    })
                    .collect(),
            ),
        };
        Ok(Self {
            table: id,
            keys,
            values,
        })
    }

    /// The table's id.
    pub fn table(&self) -> TableId {
        self.table
    }

    /// The rows, as a LogUp instance's table.
    pub fn tuples(&self) -> Tuples<'_, B> {
        Tuples::new(&self.keys, self.values.as_deref())
    }
}

/// One LogUp instance of a gadget: the lookups that read one table, their columns one after
/// another.
///
/// The columns are secret-class (05 §3), in `SecretBuf`s, and the instance lends them to the code
/// that holds it. The debug form shows the shapes, never an entry.
pub struct Group<B: Copy> {
    table: TableId,
    lookups: Vec<usize>,
    lengths: Vec<usize>,
    keys: SecretBuf<B>,
    values: Option<SecretBuf<B>>,
}

impl<B: Copy> Group<B> {
    /// The table the lookups read.
    pub fn table(&self) -> TableId {
        self.table
    }

    /// The lookups, as indices into the constraints' `lookups`, in the order of the columns.
    pub fn lookups(&self) -> &[usize] {
        &self.lookups
    }

    /// Each lookup's unpadded column length, in [`lookups`](Self::lookups) order.
    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    /// The key column: each lookup's keys, one lookup after another.
    ///
    /// The keys are secret-class (05 §3), lent to the code that computes on them, such as the
    /// LogUp prover through [`Lookups::instances`]. Their holder puts no key, and nothing derived
    /// from one, in a log, an error, a metric or a report; hands them only to code that
    /// `secret-access.toml` lists (OBS-1); and copies none out, since the clearing cannot reach a
    /// copy (OBS-6).
    pub fn keys(&self) -> &[B] {
        &self.keys
    }

    /// The value column, `None` for a table without values.
    ///
    /// The values are secret-class (05 §3), lent as the [`keys`](Self::keys) are, and their
    /// holder keeps the same rules: no log, error, metric or report; only to code that
    /// `secret-access.toml` lists (OBS-1); no copy (OBS-6).
    pub fn values(&self) -> Option<&[B]> {
        self.values.as_deref()
    }
}

/// Shows the shapes, never an entry (05 §3).
impl<B: Copy> fmt::Debug for Group<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Group")
            .field("table", &self.table)
            .field("lookups", &self.lookups)
            .field("keys", &self.keys.len())
            .field("values", &self.values.as_ref().map(|v| v.len()))
            .finish()
    }
}

/// A gadget's lookups as LogUp columns, one [`Group`] per table it reads, in order of first use.
pub struct Lookups<B: Copy> {
    groups: Vec<Group<B>>,
}

/// Shows the shapes, never an entry (05 §3).
impl<B: Copy> fmt::Debug for Lookups<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lookups")
            .field("groups", &self.groups)
            .finish()
    }
}

/// Over a prime field `B`, in which the forms are evaluated element by element.
impl<B: PrimeField + PolyField<Base = B>> Lookups<B> {
    /// The columns of `constraints`' lookups, each form evaluated element by element on `input`,
    /// `classes` and the tensors of `witness`, all of one entry per element: integers mapped to
    /// `B` in the centered representation.
    ///
    /// # Errors
    ///
    /// [`GadgetError::Missing`] for a column a lookup reads that is not given, and
    /// [`GadgetError::Length`] for one of another length than `input`: the classes first, when a
    /// lookup reads them, then each committed tensor in the order the lookups first read it.
    /// Then [`GadgetError::Shape`] for lookups into one table of which some have values and some
    /// not.
    pub fn new(
        constraints: &Constraints<B>,
        input: &[i64],
        classes: Option<&[u8]>,
        witness: &Witness,
    ) -> Result<Self, GadgetError> {
        let len = input.len();
        // The columns the lookups read, checked once.
        let classes = match constraints.reads(Term::Class) {
            true => Some(checked(Column::Classes, classes, len)?),
            false => None,
        };
        let mut tensors: Vec<(Committed, &[i64])> = Vec::new();
        for lookup in &constraints.lookups {
            let forms = core::iter::once(&lookup.key).chain(&lookup.value);
            for &(_, term) in forms.flat_map(|form| &form.terms) {
                if let Term::Committed(tensor) = term
                    && tensors.iter().all(|(held, _)| *held != tensor)
                {
                    let column = checked(Column::Tensor(tensor), witness.tensor(tensor), len)?;
                    tensors.push((tensor, column));
                }
            }
        }
        // The term's entry at element `i`.
        let value = |term: Term, i: usize| -> B {
            match term {
                Term::Input => B::from_i64(input[i]),
                Term::Class => B::from_u64_reduced(classes.map_or(0, |c| u64::from(c[i]))),
                Term::Committed(tensor) => tensors
                    .iter()
                    .find(|(held, _)| *held == tensor)
                    .map_or(B::ZERO, |(_, column)| B::from_i64(column[i])),
            }
        };
        // The lookups by table, in order of first use.
        let mut tables: Vec<(TableId, Vec<usize>)> = Vec::new();
        for (index, lookup) in constraints.lookups.iter().enumerate() {
            match tables.iter_mut().find(|(table, _)| *table == lookup.table) {
                Some((_, lookups)) => lookups.push(index),
                None => tables.push((lookup.table, vec![index])),
            }
        }
        let mut groups = Vec::with_capacity(tables.len());
        for (table, indices) in tables {
            let with_values = constraints.lookups[indices[0]].value.is_some();
            if indices
                .iter()
                .any(|&i| constraints.lookups[i].value.is_some() != with_values)
            {
                return Err(GadgetError::Shape {
                    rule: "the lookups into one table all have values, or none has",
                });
            }
            let mut keys = SecretBuf::with_capacity(indices.len() * len);
            let mut values = with_values.then(|| SecretBuf::with_capacity(indices.len() * len));
            for &index in &indices {
                let lookup = &constraints.lookups[index];
                for i in 0..len {
                    keys.push(lookup.key.evaluate(|term| value(term, i)));
                }
                if let (Some(column), Some(form)) = (&mut values, &lookup.value) {
                    for i in 0..len {
                        column.push(form.evaluate(|term| value(term, i)));
                    }
                }
            }
            groups.push(Group {
                table,
                lengths: vec![len; indices.len()],
                lookups: indices,
                keys,
                values,
            });
        }
        Ok(Self { groups })
    }

    /// Group already evaluated columns, including row and position lookups of different lengths.
    /// Columns and every intermediate buffer remain clearing allocations (OBS-6).
    pub(crate) fn from_columns(
        columns: Vec<(TableId, SecretBuf<B>, Option<SecretBuf<B>>)>,
    ) -> Result<Self, GadgetError> {
        let mut groups: Vec<Group<B>> = Vec::new();
        for (index, (table, keys, values)) in columns.into_iter().enumerate() {
            if values.as_ref().is_some_and(|v| v.len() != keys.len()) {
                return Err(GadgetError::Shape {
                    rule: "a lookup's key and value columns have equal lengths",
                });
            }
            let position = groups.iter().position(|g| g.table == table);
            match position {
                None => groups.push(Group {
                    table,
                    lookups: vec![index],
                    lengths: vec![keys.len()],
                    keys,
                    values,
                }),
                Some(i) => {
                    let group = &mut groups[i];
                    if group.values.is_some() != values.is_some() {
                        return Err(GadgetError::Shape {
                            rule: "the lookups into one table all have values, or none has",
                        });
                    }
                    group.lookups.push(index);
                    group.lengths.push(keys.len());
                    group.keys.extend(keys.iter().copied());
                    if let (Some(dst), Some(src)) = (&mut group.values, values) {
                        dst.extend(src.iter().copied());
                    }
                }
            }
        }
        Ok(Self { groups })
    }

    /// The instances' columns, one group per table, in order of first use.
    pub fn groups(&self) -> &[Group<B>] {
        &self.groups
    }

    /// The multiplicities an honest prover commits for each group, in order: each lookup counts
    /// the row its key names, and row 0 also the lookup tree's padding
    /// ([`tessera_logup::multiplicities`]). `tables` holds the columns of every table a group
    /// reads, in any order.
    ///
    /// Which lookup names which row is secret-class: the rows are counted in their `Secret`, and
    /// whether every key fits a row index is folded over the keys without branching
    /// (INV-ARCH-2).
    ///
    /// # Errors
    ///
    /// [`GadgetError::Table`], group by group, for a group whose table has no columns among
    /// `tables`, then for a group with a key that names no row of its table: an honest witness's
    /// keys always do (INV-IR-2).
    pub fn multiplicities(
        &self,
        tables: &[TableColumns<B>],
    ) -> Result<Vec<SecretBuf<B>>, GadgetError> {
        self.groups
            .iter()
            .map(|group| {
                let table = columns(tables, group.table)?;
                // Row `j`'s key is `j`, so a lookup's row is its key. A key past `u32` names no
                // row, since `T ≤ 2^32`; `multiplicities` checks the others against `T`.
                let mut rows = Vec::with_capacity(group.keys.len());
                let mut fits = true;
                for key in group.keys.iter() {
                    let row = u32::try_from(key.to_canonical_u64());
                    fits &= row.is_ok();
                    rows.push(row.unwrap_or(0));
                }
                let rows = Secret::new(rows);
                let unnamed = GadgetError::Table {
                    table: group.table,
                    rule: "every lookup's key names a row of its table",
                };
                match fits {
                    // `multiplicities` refuses only a row past `T`: a key that names no row.
                    true => multiplicities(table.tuples(), &rows).map_err(|_| unnamed),
                    false => Err(unnamed),
                }
            })
            .collect()
    }

    /// The LogUp instances, one per group in order, with the committed `multiplicities`, one
    /// buffer per group: those of [`multiplicities`](Self::multiplicities) for an honest prover.
    /// `tables` holds the columns of every table a group reads, in any order.
    ///
    /// # Errors
    ///
    /// [`GadgetError::Length`] unless `multiplicities` holds one buffer per group, then
    /// [`GadgetError::Table`] for a group whose table has no columns among `tables`. The LogUp
    /// prover checks each instance's own shapes.
    pub fn instances<'a>(
        &'a self,
        tables: &'a [TableColumns<B>],
        multiplicities: &'a [SecretBuf<B>],
    ) -> Result<Vec<LogUpInstance<'a, B>>, GadgetError> {
        if multiplicities.len() != self.groups.len() {
            return Err(GadgetError::Length {
                column: Column::Multiplicities,
                len: multiplicities.len(),
                expected: self.groups.len(),
            });
        }
        self.groups
            .iter()
            .zip(multiplicities)
            .map(|(group, m)| {
                let table = columns(tables, group.table)?;
                Ok(LogUpInstance::new(
                    table.tuples(),
                    Tuples::new(&group.keys, group.values.as_deref()),
                    m,
                ))
            })
            .collect()
    }
}

/// `column`, after checking that it is given with `len` entries.
fn checked<T>(column: Column, values: Option<&[T]>, len: usize) -> Result<&[T], GadgetError> {
    let values = values.ok_or(GadgetError::Missing { column })?;
    match values.len() == len {
        true => Ok(values),
        false => Err(GadgetError::Length {
            column,
            len: values.len(),
            expected: len,
        }),
    }
}

/// The columns of `table` among `tables`.
fn columns<B>(tables: &[TableColumns<B>], table: TableId) -> Result<&TableColumns<B>, GadgetError> {
    tables
        .iter()
        .find(|columns| columns.table == table)
        .ok_or(GadgetError::Table {
            table,
            rule: "the caller holds the columns of every table the lookups read",
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_field::Goldilocks as B;

    #[test]
    fn evaluated_columns_preserve_tuple_alignment_and_reject_mixed_schemas() {
        let buf = |xs: &[u64]| {
            xs.iter()
                .copied()
                .map(B::from_u64_reduced)
                .collect::<SecretBuf<_>>()
        };
        let groups = Lookups::from_columns(vec![
            (TableId(2), buf(&[1]), Some(buf(&[10]))),
            (TableId(3), buf(&[9]), None),
            (TableId(2), buf(&[2, 3]), Some(buf(&[20, 30]))),
        ])
        .unwrap();
        assert_eq!(groups.groups()[0].table(), TableId(2));
        assert_eq!(groups.groups()[0].lookups(), [0, 2]);
        assert_eq!(groups.groups()[0].lengths(), [1, 2]);
        assert_eq!(groups.groups()[0].keys(), &buf(&[1, 2, 3])[..]);
        assert_eq!(
            groups.groups()[0].values().unwrap(),
            &buf(&[10, 20, 30])[..]
        );
        assert_eq!(groups.groups()[1].keys(), [B::from_u64_reduced(9)]);
        assert!(
            Lookups::<B>::from_columns(vec![])
                .unwrap()
                .groups()
                .is_empty()
        );
        let error =
            Lookups::from_columns(vec![(TableId(2), buf(&[1]), Some(buf(&[])))]).unwrap_err();
        assert_eq!(
            error,
            GadgetError::Shape {
                rule: "a lookup's key and value columns have equal lengths"
            }
        );
        for columns in [
            vec![
                (TableId(2), buf(&[1]), Some(buf(&[1]))),
                (TableId(2), buf(&[2]), None),
            ],
            vec![
                (TableId(2), buf(&[1]), None),
                (TableId(2), buf(&[2]), Some(buf(&[2]))),
            ],
        ] {
            assert_eq!(
                Lookups::from_columns(columns).unwrap_err(),
                GadgetError::Shape {
                    rule: "the lookups into one table all have values, or none has"
                }
            );
        }
    }
}
