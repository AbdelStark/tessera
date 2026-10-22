// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bind RFC-0017 output-certificate roles to RFC-0044's actual model tensors.
//! Output u reads logits[prompt-1+u] and OneHot[prompt+u] (RFC-0009).
//! Row slicing is a linear reduction, not an affine transformation of an MLE point.

use super::argmax::{Argmax, Claim as RoleClaim, Role};
use super::{Claim, GadgetError, Input, Plan};
use crate::advice::Source;
use crate::claims::WeightType;
use crate::lowering::{Op, TensorId};
use crate::sumcheck::Instance;
use crate::validation::GadgetId;
use tv_field::Fp2;

/// Model-derived output domain and teacher-forcing source identities.
#[derive(Clone, Copy, Debug)]
pub struct Outputs {
    site: Argmax,
    prompt: u32,
    source_vars: usize,
    logits: TensorId,
    indicators: TensorId,
}
/// A canonical certificate maximum still owed to its advice commitment.
/// The model graph ends at logits; this is not a fabricated graph auxiliary.
#[must_use = "open this certificate's maximum at its exact output-row point"]
pub struct Maximum<'a> {
    /// The verifier-owned certificate id.
    pub gadget: GadgetId,
    /// The output-row point.
    pub point: &'a [Fp2],
    /// The claimed maximum extension.
    pub value: Fp2,
}
/// Every Argmax source debt remains explicit until its binding is discharged.
#[must_use = "reduce the graph view or open the canonical certificate maximum before acceptance"]
pub enum Binding<'a> {
    /// A linear view of the model logits or existing next-token OneHot.
    Graph(View<'a>),
    /// The certificate maximum, which the scoring advice layout must locate.
    Maximum(Maximum<'a>),
}
/// One output-view claim to reduce to its canonical model tensor.
#[must_use = "prove this row-view reduction and discharge its canonical graph source claim"]
pub struct View<'a> {
    context: Outputs,
    source: TensorId,
    offset: u32,
    target: &'a [Fp2],
    claim: Fp2,
}
/// A final view relation and its remaining graph/opening obligation.
#[must_use = "check the batched relation and discharge the canonical graph tensor claim"]
pub struct Final {
    value: Fp2,
    source: TensorId,
    point: Vec<Fp2>,
    evaluation: Fp2,
}

impl Plan<'_> {
    /// Derive a certificate's output views from the actual lowered graph.
    /// `prompt` and `gadget` come from the verifier's validated request/certificate
    /// context, never proof-selected source ids. Outputs are `graph.length()-prompt`.
    /// The certificate owner still supplies its canonical maximum advice and Range29.
    ///
    /// # Errors
    /// `Internal` for no outputs, an empty prompt, or inconsistent own graph geometry.
    pub fn argmax_outputs(&self, gadget: GadgetId, prompt: u32) -> Result<Outputs, GadgetError> {
        let internal = GadgetError::Internal;
        let n = self.graph.length();
        if prompt == 0 || prompt >= n {
            return Err(internal);
        }
        let first = self.graph.nodes().first().ok_or(internal)?;
        let Op::OneHot { vocab } = *first.op() else {
            return Err(internal);
        };
        let &indicators = first.outputs().first().ok_or(internal)?;
        let last = self.graph.nodes().last().ok_or(internal)?;
        if !matches!(last.op(),Op::MatMulWeight{weight} if weight.ty==WeightType::LmHead) {
            return Err(internal);
        }
        let &logits = last.outputs().first().ok_or(internal)?;
        if self.graph.tensor(indicators).ok_or(internal)?.shape.dims() != [n, vocab]
            || self.graph.tensor(logits).ok_or(internal)?.shape.dims() != [n, vocab]
        {
            return Err(internal);
        }
        let rows = n.checked_sub(prompt).ok_or(internal)?;
        let source_vars =
            usize::try_from(u32::BITS.saturating_sub(n.saturating_sub(1).leading_zeros()))
                .map_err(|_| internal)?;
        Ok(Outputs {
            site: Argmax::derive(gadget, vocab, rows)?,
            prompt,
            source_vars,
            logits,
            indicators,
        })
    }
}
impl Outputs {
    /// The final-relation geometry derived from the model vocabulary and output rows.
    pub const fn argmax(self) -> Argmax {
        self.site
    }
    /// Bind a role to the actual model or retain the maximum commitment debt.
    /// A virtual suffix must first pass its suffix reduction; it is not a graph tensor.
    ///
    /// # Errors
    /// `Internal` for a suffix role or wrong verifier-owned point geometry.
    pub fn bind(self, claim: RoleClaim<'_>) -> Result<Binding<'_>, GadgetError> {
        let offset = match claim.role {
            Role::Maximum => {
                if claim.point.len() != self.site.row_variables() {
                    return Err(GadgetError::Internal);
                }
                return Ok(Binding::Maximum(Maximum {
                    gadget: self.site.gadget(),
                    point: claim.point,
                    value: claim.value,
                }));
            }
            Role::Indicator => self.prompt,
            Role::Logits => self.prompt.checked_sub(1).ok_or(GadgetError::Internal)?,
            Role::Suffix => return Err(GadgetError::Internal),
        };
        if claim.point.len() != self.site.variables() {
            return Err(GadgetError::Internal);
        }
        Ok(Binding::Graph(View {
            context: self,
            source: if claim.role == Role::Indicator {
                self.indicators
            } else {
                self.logits
            },
            offset,
            target: claim.point,
            claim: claim.value,
        }))
    }
}

// LSB addition by offset, with the higher bit overriding u's comparison with rows.
// The context guarantees 0<rows<sequence<=2^17 and offset+rows<=sequence.
pub(super) fn kernel(target: &[Fp2], point: &[Fp2], offset: u32, rows: u32) -> Option<Fp2> {
    let mut state = [[Fp2::ZERO; 3]; 2];
    *state.get_mut(0)?.get_mut(1)? = Fp2::ONE;
    for (i, &r) in point.iter().enumerate() {
        let shift = u32::try_from(i).ok()?;
        let a = usize::try_from(offset.checked_shr(shift).unwrap_or(0) & 1).ok()?;
        let b = usize::try_from(rows.checked_shr(shift).unwrap_or(0) & 1).ok()?;
        let mut next = [[Fp2::ZERO; 3]; 2];
        for (carry, comparisons) in state.iter().enumerate() {
            for (cmp, &weight) in comparisons.iter().enumerate() {
                for u in 0..=usize::from(target.get(i).is_some()) {
                    let total = u.checked_add(a)?.checked_add(carry)?;
                    let y = total.checked_rem(2)?;
                    let carry = total.checked_div(2)?;
                    let cmp = if u < b {
                        0
                    } else if u > b {
                        2
                    } else {
                        cmp
                    };
                    let input = target
                        .get(i)
                        .map_or(Fp2::ONE, |&x| if u == 0 { Fp2::ONE - x } else { x });
                    let output = if y == 0 { Fp2::ONE - r } else { r };
                    *next.get_mut(carry)?.get_mut(cmp)? += weight * input * output;
                }
            }
        }
        state = next;
    }
    state.first()?.first().copied()
}
impl View<'_> {
    /// Degree-two reduction over the original model's time domain.
    pub const fn instance(&self) -> Instance {
        Instance {
            claim: self.claim,
            vars: self.context.source_vars,
            evals: 1,
        }
    }
    /// Evaluate the literal row-copy relation and retain its canonical model claim.
    /// The sole final evaluation is that model tensor at fixed vocabulary coordinates
    /// followed by this reduction's final source-row coordinates.
    ///
    /// # Errors
    /// Own point geometry is `Internal`, proof count is the owning gadget's
    /// `Rejected`, and allocation failure is `Memory`.
    pub fn evaluate(&self, point: &[Fp2], values: &[Fp2]) -> Result<Final, GadgetError> {
        let internal = GadgetError::Internal;
        if point.len() != self.context.source_vars {
            return Err(internal);
        }
        if values.len() != 1 {
            return Err(GadgetError::Rejected {
                gadget: self.context.site.gadget(),
            });
        }
        let vb = self
            .context
            .site
            .variables()
            .checked_sub(self.context.site.row_variables())
            .ok_or(internal)?;
        let (v, row) = self.target.split_at_checked(vb).ok_or(internal)?;
        let evaluation = values.first().copied().ok_or(internal)?;
        let value =
            kernel(row, point, self.offset, self.context.site.rows()).ok_or(internal)? * evaluation;
        let mut full = Vec::new();
        full.try_reserve_exact(v.len().checked_add(point.len()).ok_or(internal)?)
            .map_err(|_| GadgetError::Memory)?;
        full.extend_from_slice(v);
        full.extend_from_slice(point);
        Ok(Final {
            value,
            source: self.source,
            point: full,
            evaluation,
        })
    }
}
impl Final {
    /// The value to compare with the batched sumcheck final claim.
    pub const fn value(&self) -> Fp2 {
        self.value
    }
    /// The actual model tensor and point, still owed to graph reduction or opening.
    pub fn claim(&self) -> Claim<'_> {
        Claim {
            input: Input {
                source: Source::Tensor(self.source),
                point: &self.point,
            },
            value: self.evaluation,
        }
    }
}
