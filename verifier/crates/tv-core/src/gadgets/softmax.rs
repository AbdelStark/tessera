// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0009/0030/0035 causal maximum, reciprocal and padding relations.
//! Each incoming key and forest-leaf claim has its own reduction. The returned
//! source and lookup obligations still need graph, LogUp and opening checks.

use super::decomposition::{Decomposition, Values};
use super::requant::Lookup;
use super::{Claim, GadgetError, Input, Plan};
use crate::advice::{Auxiliary, Source};
use crate::lowering::{NodeId, Op, TensorId};
use crate::selection::Pattern;
use crate::sumcheck::{Instance, eq};
use crate::validation::GadgetId;
use tv_field::{Fp, Fp2};
use tv_tables::{TableId, TableKind};

/// A Softmax site derived only from public graph and table metadata.
#[derive(Clone, Copy, Debug)]
pub struct Softmax<'p> {
    gadget: GadgetId,
    node: NodeId,
    pub(super) slots: Option<&'p crate::lowering::Slots>,
    scores: TensorId,
    exponentials: TensorId,
    reciprocal: TensorId,
    exp: TableId,
    ranges: &'p [TableId],
    decomposition: Decomposition,
    limb_bits: u8,
    pub(super) n: u32,
    pub(super) heads: u32,
    key_vars: usize,
    token_vars: usize,
    row_vars: usize,
    numerator: Fp2,
    row_scale: Fp2,
    last_key: Fp2,
}

/// The committed values left free on a padded row (RFC-0035).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Padding {
    /// The causal maximum.
    Maximum,
    /// The reciprocal of the exponential row sum.
    Reciprocal,
}

/// One incoming masked-key or maximum-forest leaf claim.
#[must_use = "reduce this key claim and bind its lookup/value obligations when present"]
pub struct Key<'p, 'a> {
    site: Softmax<'p>,
    target: &'a [Fp2],
    claim: Fp2,
    lookup: Option<(Fp2, Fp2)>,
}

/// Final key relation, retaining both the maximum and virtual score claims.
#[must_use = "check the batched relation and bind every canonical source claim"]
pub struct KeyFinal<'a> {
    value: Fp2,
    claims: [Claim<'a>; 2],
}

/// Affine range bindings and the remaining nonlinear reciprocal obligation.
#[must_use = "bind all source/range claims, reduce the reciprocal product, and check both padding relations"]
pub struct Checked<'p, 'a> {
    site: Softmax<'p>,
    point: &'a [Fp2],
    sum_point: Vec<Fp2>,
    sources: &'a [Fp2],
    keys: &'a [Fp2],
    decompositions: [super::decomposition::Checked<'a>; 2],
    product: Fp2,
}

/// Final reciprocal product with its reciprocal and exponential opening debts.
#[must_use = "check the batched product and bind both canonical source claims"]
pub struct ReciprocalFinal<'p, 'a> {
    site: Softmax<'p>,
    point: &'a [Fp2],
    sum_point: Vec<Fp2>,
    value: Fp2,
    reciprocal: Fp2,
    exponential: Fp2,
}

/// A padding zero-check and its canonical source-opening obligation.
#[must_use = "check the batched padding relation and open its canonical source"]
pub struct PaddingFinal<'a> {
    value: Fp2,
    claim: Claim<'a>,
}

fn field(value: u64) -> Fp2 {
    Fp2::from(Fp::reduce(value))
}
fn bits(value: u32) -> Result<usize, GadgetError> {
    usize::try_from(u32::BITS.saturating_sub(value.saturating_sub(1).leading_zeros()))
        .map_err(|_| GadgetError::Internal)
}

impl Plan<'_> {
    /// Softmax sites in node order, retaining their global gadget ids.
    /// An item is `Internal` for inconsistent verifier-derived geometry.
    pub fn softmaxes(&self) -> impl Iterator<Item = Result<Softmax<'_>, GadgetError>> {
        self.selected
            .choices
            .iter()
            .enumerate()
            .filter(|(_, c)| c.pattern == Pattern::SoftmaxRow)
            .map(|(index, c)| {
                let internal = GadgetError::Internal;
                let node = self.graph.node(c.node).ok_or(internal)?;
                let Op::SoftmaxRow { sigma, .. } = *node.op() else {
                    return Err(internal);
                };
                let &scores = node.inputs().first().ok_or(internal)?;
                let &exponentials = node.outputs().first().ok_or(internal)?;
                let &reciprocal = node.outputs().get(1).ok_or(internal)?;
                let shape = self.graph.tensor(scores).ok_or(internal)?.shape;
                let &[heads, n, keys] = shape.dims() else {
                    return Err(internal);
                };
                if n != self.graph.length()
                    || keys != self.graph.key_length()
                    || heads == 0
                    || n == 0
                    || keys == 0
                    || self.graph.tensor(exponentials).ok_or(internal)?.shape != shape
                    || self.graph.tensor(reciprocal).ok_or(internal)?.shape.dims() != [heads, n]
                {
                    return Err(internal);
                }
                let table = |id: TableId| self.selected.tables.get(usize::try_from(id.0).ok()?);
                let &exp = c.tables.first().ok_or(internal)?;
                let TableKind::Exp {
                    key_bits,
                    sigma: table_sigma,
                    ..
                } = table(exp).ok_or(internal)?.kind
                else {
                    return Err(internal);
                };
                if sigma != table_sigma {
                    return Err(internal);
                }
                let ranges = c.tables.get(1..).ok_or(internal)?;
                let TableKind::Range { bits: limb_bits } =
                    table(*ranges.get(1).ok_or(internal)?).ok_or(internal)?.kind
                else {
                    return Err(internal);
                };
                let width = sigma.checked_add(18).ok_or(internal)?;
                let decomposition = Decomposition::derive(width, limb_bits)?;
                let limbs = decomposition.limbs();
                if ranges.len()
                    != limbs
                        .checked_mul(2)
                        .and_then(|n| n.checked_add(1))
                        .ok_or(internal)?
                {
                    return Err(internal);
                }
                for (i, &id) in ranges.iter().enumerate() {
                    let expected = if i == 0 {
                        sigma.checked_add(1).ok_or(internal)?
                    } else {
                        let j = i
                            .checked_sub(1)
                            .and_then(|n| n.checked_rem(limbs))
                            .ok_or(internal)?;
                        width
                            .checked_sub(
                                u8::try_from(
                                    j.checked_mul(usize::from(limb_bits)).ok_or(internal)?,
                                )
                                .map_err(|_| internal)?,
                            )
                            .ok_or(internal)?
                            .min(limb_bits)
                    };
                    if table(id).ok_or(internal)?.kind != (TableKind::Range { bits: expected }) {
                        return Err(internal);
                    }
                }
                let token_vars = bits(n)?;
                let key_vars = match self.graph.attention_key_bits(scores) {
                    Some(kb) => usize::try_from(kb).map_err(|_| internal)?,
                    None => bits(keys)?,
                };
                Ok(Softmax {
                    gadget: GadgetId(u32::try_from(index).map_err(|_| internal)?),
                    node: c.node,
                    slots: self.graph.slots(),
                    scores,
                    exponentials,
                    reciprocal,
                    exp,
                    ranges,
                    decomposition,
                    limb_bits,
                    n,
                    heads,
                    key_vars,
                    token_vars,
                    row_vars: token_vars.checked_add(bits(heads)?).ok_or(internal)?,
                    numerator: field(
                        1u64.checked_shl(u32::from(sigma).checked_mul(2).ok_or(internal)?)
                            .ok_or(internal)?,
                    ),
                    row_scale: field(
                        1u64.checked_shl(u32::try_from(key_vars).map_err(|_| internal)?)
                            .ok_or(internal)?,
                    ),
                    last_key: field(
                        1u64.checked_shl(u32::from(key_bits))
                            .and_then(|v| v.checked_sub(1))
                            .ok_or(internal)?,
                    ),
                })
            })
    }
}

impl<'p> Softmax<'p> {
    /// The global gadget id derived by the verifier.
    pub const fn gadget(self) -> GadgetId {
        self.gadget
    }
    /// The canonical owning graph node.
    pub const fn node(self) -> NodeId {
        self.node
    }
    /// Variables of the score key axis (`b = slot_bits` for batched attention).
    pub const fn key_variables(self) -> usize {
        self.key_vars
    }
    /// Variables of a score tensor, natural order key/query/head.
    pub const fn variables(self) -> usize {
        self.key_vars.saturating_add(self.row_vars)
    }
    /// Variables of a row tensor, natural order query/head.
    pub const fn row_variables(self) -> usize {
        self.row_vars
    }
    /// All Range requests after the Exp request, including repeated table ids.
    pub const fn range_tables(self) -> &'p [TableId] {
        self.ranges
    }
    /// Canonical source order of the affine bracket: R, E at half-key, A lows, B lows.
    pub fn source(self, index: usize) -> Option<Source> {
        let low = self.decomposition.limbs().checked_sub(1)?;
        if index == 0 {
            Some(Source::Tensor(self.reciprocal))
        } else if index == 1 {
            Some(Source::Tensor(self.exponentials))
        } else {
            let i = index.checked_sub(2)?;
            let aux = if i < low {
                Auxiliary::SoftmaxA(u32::try_from(i).ok()?)
            } else if i < low.checked_mul(2)? {
                Auxiliary::SoftmaxB(u32::try_from(i.checked_sub(low)?).ok()?)
            } else {
                return None;
            };
            Some(Source::Auxiliary(self.node, aux))
        }
    }
    pub(super) fn psi(self, point: &[Fp2]) -> Result<Fp2, GadgetError> {
        if point.len() != self.row_vars {
            return Err(GadgetError::Internal);
        }
        let (row, head) = point.split_at(self.token_vars);
        match self.slots {
            Some(s) if s.lengths().len() >= 2 => Ok(crate::selectors::slot_selector(
                row,
                usize::try_from(s.slot_bits()).map_err(|_| GadgetError::Internal)?,
                s.lengths(),
            )
            .ok_or(GadgetError::Internal)?
                * crate::selectors::lt(head, u64::from(self.heads))
                    .ok_or(GadgetError::Internal)?),
            _ => crate::selectors::row_selector(
                row,
                u64::from(self.n),
                Some((head, u64::from(self.heads))),
                None,
            )
            .ok_or(GadgetError::Internal),
        }
    }
    pub(super) fn causal(self, point: &[Fp2]) -> Result<Fp2, GadgetError> {
        if point.len() != self.variables() {
            return Err(GadgetError::Internal);
        }
        let (key, row) = point.split_at(self.key_vars);
        let (query, head) = row.split_at(self.token_vars);
        let c = match self.slots {
            Some(s) if s.lengths().len() >= 2 => crate::selectors::causal_slots(
                query,
                key,
                usize::try_from(s.slot_bits()).map_err(|_| GadgetError::Internal)?,
                s.lengths(),
            )
            .ok_or(GadgetError::Internal)?,
            _ => crate::selectors::causal(query, key, u64::from(self.n))
                .ok_or(GadgetError::Internal)?,
        };
        Ok(c * crate::selectors::lt(head, u64::from(self.heads)).ok_or(GadgetError::Internal)?)
    }
    /// Public maximum-forest root: zero on real rows, one on padding.
    /// Returns `Internal` for the verifier's wrong row-point geometry.
    pub fn forest_root(self, point: &[Fp2]) -> Result<Fp2, GadgetError> {
        Ok(Fp2::ONE - self.psi(point)?)
    }
    /// The canonical exponential point for a row sum: half on each key coordinate.
    /// Returns `Internal` for wrong own geometry, or `Memory` on reservation failure.
    pub fn row_sum_point(self, row: &[Fp2]) -> Result<Vec<Fp2>, GadgetError> {
        if row.len() != self.row_vars {
            return Err(GadgetError::Internal);
        }
        let mut out = Vec::new();
        out.try_reserve_exact(self.variables())
            .map_err(|_| GadgetError::Memory)?;
        out.resize(
            self.key_vars,
            Fp2::from(Fp::reduce(2).inverse().ok_or(GadgetError::Internal)?),
        );
        out.extend_from_slice(row);
        Ok(out)
    }
    /// Bind an Exp lookup key at its own incoming point, retaining value and lookup debts.
    /// Returns `Internal` for wrong verifier-owned point geometry.
    pub fn lookup<'a>(
        self,
        point: &'a [Fp2],
        key: Fp2,
        value: Fp2,
    ) -> Result<Key<'p, 'a>, GadgetError> {
        Ok(Key {
            site: self,
            target: point,
            claim: key - self.last_key * (Fp2::ONE - self.causal(point)?),
            lookup: Some((key, value)),
        })
    }
    /// Bind a maximum-forest leaf at its own incoming point.
    /// Returns `Internal` for wrong verifier-owned point geometry.
    pub fn tree<'a>(self, point: &'a [Fp2], leaf: Fp2) -> Result<Key<'p, 'a>, GadgetError> {
        Ok(Key {
            site: self,
            target: point,
            claim: leaf - Fp2::ONE + self.causal(point)?,
            lookup: None,
        })
    }
    /// Check the reciprocal Range key, every NonNeg limb and the linear row sum.
    /// `sources` follows [`Self::source`]; key/value slices follow [`Self::range_tables`].
    /// The row-sum source is E at [`Self::row_sum_point`], not E at the row point.
    ///
    /// # Errors
    /// Own geometry is `Internal`; proof counts/affine failures are `Rejected` with
    /// this gadget id; a point reservation failure is `Memory`.
    pub fn check<'a>(
        self,
        point: &'a [Fp2],
        sources: &'a [Fp2],
        keys: &'a [Fp2],
        values: &[Fp2],
    ) -> Result<Checked<'p, 'a>, GadgetError> {
        let psi = self.psi(point)?;
        let reject = GadgetError::Rejected {
            gadget: self.gadget,
        };
        let internal = GadgetError::Internal;
        let m = self.decomposition.limbs();
        let low = m.checked_sub(1).ok_or(internal)?;
        if sources.len()
            != 2usize
                .checked_add(low.checked_mul(2).ok_or(internal)?)
                .ok_or(internal)?
            || keys.len() != self.ranges.len()
            || values.len() != keys.len()
            || values.iter().any(|&v| v != Fp2::ZERO)
            || keys.first() != sources.first()
        {
            return Err(reject);
        }
        let akeys = keys
            .get(1..m.checked_add(1).ok_or(internal)?)
            .ok_or(internal)?;
        let bkeys = keys
            .get(m.checked_add(1).ok_or(internal)?..)
            .ok_or(internal)?;
        let radix = field(
            1u64.checked_shl(u32::from(self.limb_bits))
                .ok_or(internal)?,
        );
        let alpha = akeys.iter().rev().fold(Fp2::ZERO, |a, &v| a * radix + v);
        let sum = self.row_scale * sources.get(1).copied().ok_or(internal)?;
        let beta = sum - psi - alpha;
        let decomposition = |value, keys, start: usize| {
            self.decomposition.check(
                self.gadget,
                Values {
                    point,
                    value,
                    low: sources
                        .get(start..start.checked_add(low).ok_or(internal)?)
                        .ok_or(internal)?,
                    keys,
                },
            )
        };
        let decompositions = [
            decomposition(alpha, akeys, 2)?,
            decomposition(beta, bkeys, 2usize.checked_add(low).ok_or(internal)?)?,
        ];
        Ok(Checked {
            site: self,
            point,
            sum_point: self.row_sum_point(point)?,
            sources,
            keys,
            decompositions,
            product: self.numerator * psi - alpha,
        })
    }
    /// Mandatory padded-maximum and padded-reciprocal relation geometry, degree three.
    pub const fn padding_instance(self) -> Instance {
        Instance {
            claim: Fp2::ZERO,
            vars: self.row_vars,
            evals: 1,
        }
    }
    /// Evaluate one mandatory padding relation and retain its source opening.
    /// Own point errors are `Internal`; a wrong proof count is `Rejected` with its gadget id.
    pub fn padding<'a>(
        self,
        kind: Padding,
        zero: &[Fp2],
        point: &'a [Fp2],
        values: &[Fp2],
    ) -> Result<PaddingFinal<'a>, GadgetError> {
        let psi = self.psi(point)?;
        if zero.len() != self.row_vars {
            return Err(GadgetError::Internal);
        }
        if values.len() != 1 {
            return Err(GadgetError::Rejected {
                gadget: self.gadget,
            });
        }
        let value = values.first().copied().ok_or(GadgetError::Internal)?;
        Ok(PaddingFinal {
            value: eq(zero, point).ok_or(GadgetError::Internal)? * (Fp2::ONE - psi) * value,
            claim: Claim {
                input: Input {
                    source: match kind {
                        Padding::Maximum => Source::Auxiliary(self.node, Auxiliary::SoftmaxMaximum),
                        Padding::Reciprocal => Source::Tensor(self.reciprocal),
                    },
                    point,
                },
                value,
            },
        })
    }
}
impl Key<'_, '_> {
    /// Degree-three Elementwise metadata for this incoming claim.
    pub const fn instance(&self) -> Instance {
        Instance {
            claim: self.claim,
            vars: self.site.variables(),
            evals: 2,
        }
    }
    /// The original Exp lookup claim, when this is a lookup-key reduction.
    pub fn lookup(&self) -> Option<Lookup<'_>> {
        self.lookup.map(|(key, value)| Lookup {
            table: self.site.exp,
            point: self.target,
            key,
            value,
        })
    }
    /// The incoming Exp value's canonical committed source, when present.
    pub fn input_claim(&self) -> Option<Claim<'_>> {
        self.lookup.map(|(_, value)| Claim {
            input: Input {
                source: Source::Tensor(self.site.exponentials),
                point: self.target,
            },
            value,
        })
    }
    /// Evaluate the masked-key polynomial and retain maximum/score source claims.
    /// Own geometry is `Internal`; a wrong count is `Rejected` with the owning id.
    pub fn evaluate<'a>(
        &self,
        point: &'a [Fp2],
        values: &[Fp2],
    ) -> Result<KeyFinal<'a>, GadgetError> {
        let causal = self.site.causal(point)?;
        if values.len() != 2 {
            return Err(GadgetError::Rejected {
                gadget: self.site.gadget,
            });
        }
        let c = values.first().copied().ok_or(GadgetError::Internal)?;
        let x = values.get(1).copied().ok_or(GadgetError::Internal)?;
        Ok(KeyFinal {
            value: eq(self.target, point).ok_or(GadgetError::Internal)? * causal * (c - x),
            claims: [
                Claim {
                    input: Input {
                        source: Source::Auxiliary(self.site.node, Auxiliary::SoftmaxMaximum),
                        point: point
                            .get(self.site.key_vars..)
                            .ok_or(GadgetError::Internal)?,
                    },
                    value: c,
                },
                Claim {
                    input: Input {
                        source: Source::Tensor(self.site.scores),
                        point,
                    },
                    value: x,
                },
            ],
        })
    }
}
impl KeyFinal<'_> {
    /// The value to compare against the batched sumcheck's final claim.
    pub const fn value(&self) -> Fp2 {
        self.value
    }
    /// The canonical maximum and virtual-score claims, in operand order.
    pub fn claims(&self) -> impl Iterator<Item = Claim<'_>> {
        self.claims.iter().copied()
    }
}
impl Checked<'_, '_> {
    /// Source debts in canonical order, with the row-sum projection applied.
    pub fn claims(&self) -> impl Iterator<Item = Claim<'_>> {
        let primary = self
            .sources
            .iter()
            .take(2)
            .enumerate()
            .filter_map(|(i, &value)| {
                self.site.source(i).map(|source| Claim {
                    input: Input {
                        source,
                        point: if i == 1 { &self.sum_point } else { self.point },
                    },
                    value,
                })
            });
        let low = self.site.decomposition.limbs().saturating_sub(1);
        primary.chain(
            self.decompositions
                .iter()
                .enumerate()
                .flat_map(move |(side, checked)| {
                    checked.claims().filter_map(move |claim| {
                        let j = claim.limb?;
                        if j >= low {
                            return None;
                        }
                        let index = 2usize.checked_add(side.checked_mul(low)?)?.checked_add(j)?;
                        Some(Claim {
                            input: Input {
                                source: self.site.source(index)?,
                                point: claim.point,
                            },
                            value: claim.value,
                        })
                    })
                }),
        )
    }
    /// Every prescribed Range request, including virtual tops and repeated tables.
    pub fn lookups(&self) -> impl Iterator<Item = Lookup<'_>> {
        self.site
            .ranges
            .iter()
            .zip(self.keys)
            .map(|(&table, &key)| Lookup {
                table,
                point: self.point,
                key,
                value: Fp2::ZERO,
            })
    }
    /// The remaining degree-three reciprocal product; its claim is derived, not sent.
    pub const fn instance(&self) -> Instance {
        Instance {
            claim: self.product,
            vars: self.site.row_vars,
            evals: 2,
        }
    }
    /// Evaluate `eq(target,r)*R(r)*sum_i E(r,i)` at the stage's final row point.
    /// Own geometry is `Internal`, proof count `Rejected`, point reservation `Memory`.
    pub fn evaluate<'a>(
        &self,
        point: &'a [Fp2],
        values: &[Fp2],
    ) -> Result<ReciprocalFinal<'_, 'a>, GadgetError> {
        let sum_point = self.site.row_sum_point(point)?;
        if values.len() != 2 {
            return Err(GadgetError::Rejected {
                gadget: self.site.gadget,
            });
        }
        let reciprocal = values.first().copied().ok_or(GadgetError::Internal)?;
        let exponential = values.get(1).copied().ok_or(GadgetError::Internal)?;
        Ok(ReciprocalFinal {
            site: self.site,
            point,
            sum_point,
            value: eq(self.point, point).ok_or(GadgetError::Internal)?
                * reciprocal
                * self.site.row_scale
                * exponential,
            reciprocal,
            exponential,
        })
    }
}
impl ReciprocalFinal<'_, '_> {
    /// The value to compare against the batched final claim.
    pub const fn value(&self) -> Fp2 {
        self.value
    }
    /// The reciprocal and exponential-at-half-key claims, in operand order.
    pub fn claims(&self) -> impl Iterator<Item = Claim<'_>> {
        [
            Claim {
                input: Input {
                    source: Source::Tensor(self.site.reciprocal),
                    point: self.point,
                },
                value: self.reciprocal,
            },
            Claim {
                input: Input {
                    source: Source::Tensor(self.site.exponentials),
                    point: &self.sum_point,
                },
                value: self.exponential,
            },
        ]
        .into_iter()
    }
}
impl PaddingFinal<'_> {
    /// The value to compare against the batched final claim.
    pub const fn value(&self) -> Fp2 {
        self.value
    }
    /// The canonical maximum or reciprocal opening debt.
    pub const fn claim(&self) -> Claim<'_> {
        self.claim
    }
}
