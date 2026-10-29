// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0009 OneHot's four relations, with canonical graph source bindings.
//! These evaluators join a degree-three stage batch; they do not squeeze their
//! own challenges or accept a proof without checking every returned source claim.

use super::{Claim, GadgetError, Input, Plan};
use crate::advice::Source;
use crate::lowering::{NodeId, Op, TensorId};
use crate::selection::Pattern;
use crate::sumcheck::{Instance, eq};
use crate::validation::GadgetId;
use tv_field::{Fp, Fp2};

/// OneHot's constraints in canonical relation order (RFC-0009, Lemma 8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    /// Every indicator is zero or one, including all padded entries.
    Booleanity,
    /// Indicators beyond the real vocabulary are zero.
    Vocabulary,
    /// Indicators sum to one on real token rows and zero on padded rows.
    RowSum,
    /// The indicator's weighted index equals the committed token.
    Index,
}

impl Relation {
    /// The degree per variable before batching.
    pub const fn degree(self) -> usize {
        match self {
            Self::Booleanity | Self::Vocabulary => 3,
            Self::RowSum | Self::Index => 2,
        }
    }

    /// Each relation leaves one evaluation of the canonical indicator tensor.
    pub const fn evaluations(self) -> usize {
        1
    }
}

/// A OneHot site derived solely from the verifier's public graph.
#[derive(Clone, Copy, Debug)]
pub struct OneHot {
    gadget: GadgetId,
    node: NodeId,
    tokens: TensorId,
    indicators: TensorId,
    n: u32,
    vocab: u32,
    vocab_vars: usize,
    variables: usize,
}

/// One relation's final value, with every source evaluation still to open.
#[must_use = "check the batched relation and discharge every canonical source claim"]
pub struct Final<'a> {
    value: Fp2,
    source: TensorId,
    point: &'a [Fp2],
    indicator: Fp2,
}

/// A site's row reductions bound to the token claim left by the leaf stage.
#[must_use = "bind the leaf token claim and include all four OneHot relations in the stage"]
pub struct At<'a> {
    site: OneHot,
    row_point: &'a [Fp2],
    token: Fp2,
    count: Fp2,
}

impl Plan<'_> {
    /// OneHot sites in graph order, retaining their global gadget ids.
    /// An item is `Internal` only if the verifier's own metadata conflicts.
    pub fn one_hots(&self) -> impl Iterator<Item = Result<OneHot, GadgetError>> + '_ {
        self.selected
            .choices
            .iter()
            .enumerate()
            .filter(|(_, choice)| choice.pattern == Pattern::OneHot)
            .map(|(id, choice)| {
                let internal = GadgetError::Internal;
                let node = self.graph.node(choice.node).ok_or(internal)?;
                let Op::OneHot { vocab } = *node.op() else {
                    return Err(internal);
                };
                let &tokens = node.inputs().first().ok_or(internal)?;
                let &indicators = node.outputs().first().ok_or(internal)?;
                let n = self.graph.length();
                if self.graph.tensor(tokens).ok_or(internal)?.shape.dims() != [n]
                    || self.graph.tensor(indicators).ok_or(internal)?.shape.dims() != [n, vocab]
                {
                    return Err(internal);
                }
                let bits = |value: u32| {
                    usize::try_from(
                        u32::BITS.saturating_sub(value.saturating_sub(1).leading_zeros()),
                    )
                    .map_err(|_| internal)
                };
                let vocab_vars = bits(vocab)?;
                Ok(OneHot {
                    gadget: GadgetId(u32::try_from(id).map_err(|_| internal)?),
                    node: choice.node,
                    tokens,
                    indicators,
                    n,
                    vocab,
                    vocab_vars,
                    variables: vocab_vars.checked_add(bits(n)?).ok_or(internal)?,
                })
            })
    }
}

impl OneHot {
    /// The canonical gadget id, never selected by the proof.
    pub const fn gadget(self) -> GadgetId {
        self.gadget
    }

    /// The graph node that owns this site.
    pub const fn node(self) -> NodeId {
        self.node
    }

    /// All four required relations in node-local order. Omitting any of them
    /// leaves a false token or a second witness (RFC-0009, Lemma 8).
    pub fn relations(self) -> impl ExactSizeIterator<Item = Relation> {
        [
            Relation::Booleanity,
            Relation::Vocabulary,
            Relation::RowSum,
            Relation::Index,
        ]
        .into_iter()
    }

    /// Bind to this site's incoming token claim. `row_point` comes from the leaf
    /// stage's canonical point; `token` must already be fixed by its transcript.
    ///
    /// # Errors
    /// `Internal` for a wrong verifier-owned row-point dimension.
    pub fn at(self, row_point: &[Fp2], token: Fp2) -> Result<At<'_>, GadgetError> {
        if row_point.len().checked_add(self.vocab_vars) != Some(self.variables) {
            return Err(GadgetError::Internal);
        }
        Ok(At {
            site: self,
            row_point,
            token,
            count: crate::selectors::lt(row_point, u64::from(self.n))
                .ok_or(GadgetError::Internal)?,
        })
    }

    /// Bind to this site's incoming token claim on a slotted batch graph (RFC-0071).
    /// Reduces to [`Self::at`] for a single-request batch (`R = 1`).
    ///
    /// # Errors
    /// `Internal` for a wrong verifier-owned row-point dimension or mismatched slot geometry.
    pub fn at_slots<'a>(
        self,
        slots: &crate::lowering::Slots,
        row_point: &'a [Fp2],
        token: Fp2,
    ) -> Result<At<'a>, GadgetError> {
        if slots.token_len() != self.n {
            return Err(GadgetError::Internal);
        }
        if slots.lengths().len() < 2 {
            return self.at(row_point, token);
        }
        if row_point.len().checked_add(self.vocab_vars) != Some(self.variables) {
            return Err(GadgetError::Internal);
        }
        let slot_bits = usize::try_from(slots.slot_bits()).map_err(|_| GadgetError::Internal)?;
        Ok(At {
            site: self,
            row_point,
            token,
            count: crate::selectors::slot_selector(row_point, slot_bits, slots.lengths())
                .ok_or(GadgetError::Internal)?,
        })
    }
}

impl At<'_> {
    /// Public instance metadata: zero for the two zero checks, the independently
    /// evaluated row selector for `RowSum`, and the incoming token claim for `Index`.
    pub const fn instance(&self, relation: Relation) -> Instance {
        Instance {
            claim: match relation {
                Relation::Booleanity | Relation::Vocabulary => Fp2::ZERO,
                Relation::RowSum => self.count,
                Relation::Index => self.token,
            },
            vars: self.site.variables,
            evals: relation.evaluations(),
        }
    }

    /// Evaluate a final relation at the stage point and retain its source claims.
    /// `zero` is the common stage-one zero-check point, fixed before `gamma`.
    /// The row reductions use `eq(incoming_row_point, final_row_point)`. Their
    /// sums equal the selector and incoming token claim; the latter is not a
    /// new proof-chosen token evaluation at the final point.
    ///
    /// # Errors
    /// `Rejected { gadget }` for a wrong proof evaluation count; `Internal` for
    /// a wrong verifier-owned point dimension or inconsistent derived metadata.
    pub fn evaluate<'a>(
        &self,
        relation: Relation,
        zero: &[Fp2],
        point: &'a [Fp2],
        evaluations: &'a [Fp2],
    ) -> Result<Final<'a>, GadgetError> {
        let internal = GadgetError::Internal;
        if zero.len() != self.site.variables || point.len() != self.site.variables {
            return Err(internal);
        }
        let reject = GadgetError::Rejected {
            gadget: self.site.gadget,
        };
        if evaluations.len() != relation.evaluations() {
            return Err(reject);
        }
        let &h = evaluations.first().ok_or(reject)?;
        let (v, row) = point
            .split_at_checked(self.site.vocab_vars)
            .ok_or(internal)?;
        let value = match relation {
            Relation::Booleanity => eq(zero, point).ok_or(internal)? * h * (Fp2::ONE - h),
            Relation::Vocabulary => {
                eq(zero, point).ok_or(internal)?
                    * (Fp2::ONE
                        - crate::selectors::lt(v, u64::from(self.site.vocab)).ok_or(internal)?)
                    * h
            }
            Relation::RowSum | Relation::Index => {
                let term = if relation == Relation::RowSum {
                    h
                } else {
                    let (mut index, mut scale) = (Fp2::ZERO, Fp2::ONE);
                    let two = Fp2::from(Fp::reduce(2));
                    for &coordinate in v {
                        index += scale * coordinate;
                        scale *= two;
                    }
                    index * h
                };
                eq(self.row_point, row).ok_or(internal)? * term
            }
        };
        Ok(Final {
            value,
            source: self.site.indicators,
            point,
            indicator: h,
        })
    }

    /// The canonical incoming token claim. Its value must be fixed before this
    /// stage's batching challenge and must reach the commitment-opening pool.
    pub fn input_claim(&self) -> Claim<'_> {
        Claim {
            input: Input {
                source: Source::Tensor(self.site.tokens),
                point: self.row_point,
            },
            value: self.token,
        }
    }
}

impl<'a> Final<'a> {
    /// The relation value that the batched sumcheck must check.
    pub const fn value(&self) -> Fp2 {
        self.value
    }

    /// Every canonical source claim, in relation-input order. Repeated indicator
    /// claims from different relations must not be silently dropped.
    pub fn claims(&self) -> impl Iterator<Item = Claim<'a>> + '_ {
        core::iter::once(Claim {
            input: Input {
                source: Source::Tensor(self.source),
                point: self.point,
            },
            value: self.indicator,
        })
    }
}
