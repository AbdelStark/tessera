// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `NonNeg` (RFC-0009): a virtual true integer `v` lies in `[0, 2^k)`, for `k ≤ 62`, by
//! `n = ⌈k / L⌉` limbs of `L` bits.
//!
//! The limbs `v_0 … v_{n−2}` are committed, each looked up in `Range { L }`, and the top limb
//! `v_{n−1} = (v − Σ_{j<n−1} v_j·2^{jL})·2^{−(n−1)L}` is virtual, looked up in
//! `Range { k − (n−1)L }`. Lemma 3: the lookups pass iff `0 ≤ v < 2^k`, and then the limbs are
//! `v`'s digits. A key of `k ≤ L` bits has no committed limb: `v` itself is the top limb.

use tessera_field::PrimeField;
use tessera_tables::{GeneratedTable, Table, TableKind, TableSpec};

use crate::constraints::{affine, inv_pow2, tensor};
use crate::{Committed, Constraints, GadgetError, Lookup, Term, Virtual};

/// The bound of RFC-0009 Lemma 3: `v` and its limbs stay true integers.
const MAX_BITS: u8 = 62;

/// `NonNeg { k, L }` (RFC-0009), with its `Range` tables regenerated from their specifications.
///
/// [`new`](Self::new) finds the tables by kind, since tables are shared by definition (RFC-0006
/// 'Gadget selection'); [`constraints`](Self::constraints) describes its lookups, and
/// [`witness`](Self::witness) computes its limbs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NonNeg {
    /// `k`.
    pub(crate) bits: u32,
    /// `L`.
    pub(crate) limb_bits: u32,
    /// `Range { L }`, for the committed limbs; `None` when `n = 1`.
    pub(crate) limb: Option<GeneratedTable>,
    /// `Range { k − (n−1)L }`, for the top limb.
    pub(crate) top: GeneratedTable,
}

impl NonNeg {
    /// `NonNeg` for `v ∈ [0, 2^bits)` with limbs of `limb_bits` bits (`GadgetPolicy.limb_bits`,
    /// RFC-0007), its tables `Range { L }` (when `n ≥ 2`) and `Range { k − (n−1)L }` among
    /// `tables`.
    ///
    /// # Errors
    ///
    /// In this order: [`GadgetError::Shape`] unless `1 ≤ k ≤ 62` and `L ≥ 1`, or when a table
    /// is not among `tables`; [`GadgetError::Spec`] for one that does not regenerate.
    pub fn new(bits: u8, limb_bits: u8, tables: &[TableSpec]) -> Result<Self, GadgetError> {
        if !(1..=MAX_BITS).contains(&bits) {
            return Err(GadgetError::Shape {
                rule: "NonNeg proves v ∈ [0, 2^k) for 1 ≤ k ≤ 62 (RFC-0009 Lemma 3)",
            });
        }
        if limb_bits == 0 {
            return Err(GadgetError::Shape {
                rule: "a NonNeg limb has at least one bit",
            });
        }
        let (k, l) = (u32::from(bits), u32::from(limb_bits));
        let n = k.div_ceil(l);
        let range = |bits: u32| -> Result<GeneratedTable, GadgetError> {
            let kind = TableKind::Range {
                bits: u8::try_from(bits).unwrap_or(u8::MAX),
            };
            let spec = tables
                .iter()
                .find(|spec| spec.kind == kind)
                .ok_or(GadgetError::Shape {
                    rule: "the specifications hold NonNeg's Range { L } and Range { k − (n−1)L }",
                })?;
            GeneratedTable::from_spec(spec).map_err(|error| GadgetError::Spec {
                table: spec.id,
                error,
            })
        };
        let limb = match n {
            1 => None,
            _ => Some(range(l)?),
        };
        let top = range(k - (n - 1) * l)?;
        Ok(Self {
            bits: k,
            limb_bits: l,
            limb,
            top,
        })
    }

    /// `n = ⌈k / L⌉`, the limbs: `n − 1` committed, and the top one virtual.
    pub fn limbs(&self) -> u32 {
        self.bits.div_ceil(self.limb_bits)
    }

    /// The tables its lookups read, each once: `Range { L }`, when it has committed limbs, then
    /// `Range { k − (n−1)L }`, which is `Range { L }` itself when `L` divides `k`.
    pub fn tables(&self) -> Vec<&GeneratedTable> {
        match &self.limb {
            Some(limb) if limb.spec().id != self.top.spec().id => vec![limb, &self.top],
            _ => vec![&self.top],
        }
    }

    /// Its constraints over `F` (RFC-0009 'NonNeg'): the committed limbs `v_0 … v_{n−2}`, the
    /// virtual top limb `v_{n−1} = 2^{−(n−1)L}·v − Σ_{j<n−1} 2^{−(n−1−j)L}·v_j`, and the lookups
    /// `v_j ∈ Range { L }` for `j < n − 1`, then `v_{n−1} ∈ Range { k − (n−1)L }`.
    pub fn constraints<F: PrimeField>(&self) -> Constraints<F> {
        let n = self.limbs();
        let shift = (n - 1) * self.limb_bits;
        let committed: Vec<Committed> = (0..n - 1).map(|j| Committed::Limb(limb(j))).collect();
        let mut terms = vec![(inv_pow2::<F>(shift), Term::Input)];
        for j in 0..n - 1 {
            terms.push((
                -inv_pow2::<F>(shift - j * self.limb_bits),
                Term::Committed(Committed::Limb(limb(j))),
            ));
        }
        let top = affine(F::ZERO, &terms);
        let mut lookups: Vec<Lookup<F>> = match &self.limb {
            Some(table) => committed
                .iter()
                .map(|&which| Lookup {
                    table: table.spec().id,
                    key: tensor(which),
                    value: None,
                })
                .collect(),
            None => Vec::new(),
        };
        lookups.push(Lookup {
            table: self.top.spec().id,
            key: top.clone(),
            value: None,
        });
        Constraints {
            committed,
            virtuals: vec![(Virtual::TopLimb, top)],
            lookups,
        }
    }
}

/// Limb `j < n ≤ 62` as `Committed::Limb`'s index.
pub(crate) fn limb(j: u32) -> u8 {
    u8::try_from(j).unwrap_or(u8::MAX)
}
