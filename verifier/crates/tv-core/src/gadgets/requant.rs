// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0009/0024 affine lookup bindings for Requant1, Split, Wide and SiLU.
//! Membership is checked by LogUp, not by treating an extension-field evaluation
//! as a table row. Every source evaluation remains an explicit obligation.

use super::{GadgetError, Plan};
use crate::advice::{Auxiliary, Source};
use crate::lowering::NodeId;
use crate::selection::{Choice, Pattern};
use crate::validation::GadgetId;
use tv_field::{Fp, Fp2};
use tv_tables::TableId;

/// One requantization site, constructible only from a verifier-owned [`Plan`].
#[derive(Clone, Copy, Debug)]
pub struct Requant<'p> {
    choice: &'p Choice,
    gadget: GadgetId,
    bits: u32,
    final_bits: u32,
    variables: usize,
    channel_variables: usize,
    sources: [Source; 6],
    count: usize,
    class: Option<usize>,
    low: Option<usize>,
    middle: Option<usize>,
    saturation: Option<usize>,
}

/// A canonically identified source at its prescribed point.
#[derive(Clone, Copy)]
pub struct Input<'a> {
    /// A graph tensor or the same node's canonical auxiliary. Virtual tensors
    /// need graph reduction; committed tensors and auxiliaries need openings.
    pub source: Source,
    /// Natural LSB-first coordinates; class vectors use only channel coordinates.
    pub point: &'a [Fp2],
}

/// A source evaluation still owed to graph reduction or commitment opening.
#[derive(Clone, Copy)]
pub struct Claim<'a> {
    /// The source and its prescribed point.
    pub input: Input<'a>,
    /// Its claimed multilinear extension, not an integer at a general point.
    pub value: Fp2,
}

/// A prescribed lookup evaluation still owed to the earlier LogUp reduction.
#[derive(Clone, Copy)]
pub struct Lookup<'a> {
    /// The independently derived table id; repeated ids remain distinct requests.
    pub table: TableId,
    /// The common output-domain point.
    pub point: &'a [Fp2],
    /// The key extension at that point.
    pub key: Fp2,
    /// The value extension, zero for a Range table.
    pub value: Fp2,
}

/// Affine relations that match, with all source and lookup debts retained.
#[must_use = "bind every source to its graph reduction/opening and every lookup to LogUp"]
pub struct Checked<'p, 'a> {
    site: Requant<'p>,
    point: &'a [Fp2],
    channel: &'a [Fp2],
    values: &'a [Fp2],
    lookups: &'a [(Fp2, Fp2)],
}

impl Plan<'_> {
    /// Requantization and TableMap sites in graph order. Gadget ids count every
    /// gadget family, including the root, softmax and one-hot sites between them.
    ///
    /// An item is `Internal` only if the verifier's own derived metadata conflicts.
    pub fn requantizations(&self) -> impl Iterator<Item = Result<Requant<'_>, GadgetError>> {
        self.selected
            .choices
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                matches!(
                    c.pattern,
                    Pattern::Requant1
                        | Pattern::RequantSplit { .. }
                        | Pattern::RequantWide { .. }
                        | Pattern::Lookup
                )
            })
            .map(|(id, choice)| Requant::derive(self, id, choice))
    }
}

fn power(bits: u32) -> Result<Fp2, GadgetError> {
    Ok(Fp2::from(Fp::reduce(
        1u64.checked_shl(bits).ok_or(GadgetError::Internal)?,
    )))
}

impl<'p> Requant<'p> {
    fn derive(plan: &'p Plan<'_>, id: usize, choice: &'p Choice) -> Result<Self, GadgetError> {
        let internal = GadgetError::Internal;
        let node = plan.graph.node(choice.node).ok_or(internal)?;
        let &z = node.inputs().first().ok_or(internal)?;
        let &y = node.outputs().first().ok_or(internal)?;
        let table = choice.tables.last().ok_or(internal)?;
        let table = plan
            .selected
            .tables
            .get(usize::try_from(table.0).map_err(|_| internal)?)
            .ok_or(internal)?;
        let (b, classes) = match &table.kind {
            tv_tables::TableKind::Requant { b, r_by_class, .. }
            | tv_tables::TableKind::Silu { b, r_by_class, .. } => (*b, r_by_class.len()),
            _ => return Err(internal),
        };
        let bits = if choice.pattern == Pattern::Lookup {
            u32::from(b)
        } else {
            crate::range_checks::signed_width(
                *plan
                    .selected
                    .ranges
                    .get(usize::try_from(z.0).map_err(|_| internal)?)
                    .ok_or(internal)?,
            )
            .ok_or(internal)?
        };
        let shape = plan.graph.tensor(y).ok_or(internal)?.shape;
        let log =
            |d: u32| usize::try_from(u32::BITS.saturating_sub(d.saturating_sub(1).leading_zeros()));
        let variables = shape.dims().iter().try_fold(0usize, |n, &d| {
            n.checked_add(log(d).map_err(|_| internal)?).ok_or(internal)
        })?;
        let channel_variables = log(*shape.dims().last().ok_or(internal)?).map_err(|_| internal)?;
        let mut out = Self {
            choice,
            gadget: GadgetId(u32::try_from(id).map_err(|_| internal)?),
            bits,
            final_bits: u32::from(b),
            variables,
            channel_variables,
            sources: [Source::Tensor(z); 6],
            count: 0,
            class: None,
            low: None,
            middle: None,
            saturation: None,
        };
        out.push(Source::Tensor(z))?;
        out.push(Source::Tensor(y))?;
        if classes > 1 {
            let &c = node.inputs().get(1).ok_or(internal)?;
            let class_shape = plan.graph.tensor(c).ok_or(internal)?.shape;
            if class_shape.dims()
                != shape
                    .dims()
                    .last()
                    .map(core::slice::from_ref)
                    .ok_or(internal)?
            {
                return Err(internal);
            }
            out.class = Some(out.push(Source::Tensor(c))?);
        }
        match choice.pattern {
            Pattern::RequantSplit { .. } => {
                out.low = Some(out.auxiliary(Auxiliary::RequantLow)?);
            }
            Pattern::RequantWide { split, .. } => {
                if split > 0 {
                    out.low = Some(out.auxiliary(Auxiliary::RequantLow)?);
                }
                out.middle = Some(out.auxiliary(Auxiliary::RequantMiddle)?);
                out.saturation = Some(out.auxiliary(Auxiliary::RequantSaturation)?);
            }
            Pattern::Requant1 | Pattern::Lookup => {}
            _ => return Err(internal),
        }
        Ok(out)
    }

    fn push(&mut self, source: Source) -> Result<usize, GadgetError> {
        let slot = self.count;
        *self.sources.get_mut(slot).ok_or(GadgetError::Internal)? = source;
        self.count = self.count.checked_add(1).ok_or(GadgetError::Internal)?;
        Ok(slot)
    }

    fn auxiliary(&mut self, role: Auxiliary) -> Result<usize, GadgetError> {
        self.push(Source::Auxiliary(self.choice.node, role))
    }

    /// The canonical `REJ-GADGET` payload.
    pub const fn gadget(self) -> GadgetId {
        self.gadget
    }

    /// The graph node that owns this gadget and all its auxiliary roles.
    pub fn node(self) -> NodeId {
        self.choice.node
    }

    /// The independently selected constraint pattern.
    pub fn pattern(self) -> Pattern {
        self.choice.pattern
    }

    /// Lookup table ids in canonical request order, including repetitions.
    pub fn tables(self) -> &'p [TableId] {
        &self.choice.tables
    }

    /// The fixed source order: key, output, class when multi-class, then low,
    /// middle and saturation columns when the selected pattern commits them.
    ///
    /// # Errors
    /// `Internal` if the caller's verifier-owned point has the wrong dimension.
    pub fn inputs<'a>(
        self,
        point: &'a [Fp2],
    ) -> Result<impl Iterator<Item = Input<'a>>, GadgetError> {
        if point.len() != self.variables {
            return Err(GadgetError::Internal);
        }
        let channel = point
            .get(..self.channel_variables)
            .ok_or(GadgetError::Internal)?;
        Ok(self
            .sources
            .into_iter()
            .take(self.count)
            .enumerate()
            .map(move |(i, source)| Input {
                source,
                point: if self.class == Some(i) {
                    channel
                } else {
                    point
                },
            }))
    }

    /// Check every lookup key/value against the bound source evaluations. All
    /// evaluations must already be fixed at the appropriate transcript step.
    ///
    /// # Errors
    /// `Rejected { gadget }` for wrong counts or any wrong affine key/value;
    /// `Internal` for a wrong verifier-owned point or inconsistent derived metadata.
    pub fn check<'a>(
        self,
        point: &'a [Fp2],
        values: &'a [Fp2],
        lookups: &'a [(Fp2, Fp2)],
    ) -> Result<Checked<'p, 'a>, GadgetError> {
        let reject = GadgetError::Rejected {
            gadget: self.gadget,
        };
        if point.len() != self.variables {
            return Err(GadgetError::Internal);
        }
        let channel = point
            .get(..self.channel_variables)
            .ok_or(GadgetError::Internal)?;
        if values.len() != self.count || lookups.len() != self.choice.tables.len() {
            return Err(reject);
        }
        let z = *values.first().ok_or(reject)?;
        let y = *values.get(1).ok_or(reject)?;
        let optional = |slot: Option<usize>| match slot {
            Some(i) => values.get(i).copied().ok_or(GadgetError::Internal),
            None => Ok(Fp2::ZERO),
        };
        let c = optional(self.class)?;
        let class = c * power(self.final_bits)?;
        let zoff = z + power(self.bits.checked_sub(1).ok_or(GadgetError::Internal)?)?;
        let low = optional(self.low)?;
        let mut requests = lookups.iter();
        let mut matches = |key, value| {
            if requests.next().copied() == Some((key, value)) {
                Ok(())
            } else {
                Err(reject)
            }
        };
        match self.choice.pattern {
            Pattern::Requant1 | Pattern::Lookup => matches(zoff + class, y)?,
            Pattern::RequantSplit { split } => {
                let inv = power(u32::from(split))?
                    .inverse()
                    .ok_or(GadgetError::Internal)?;
                matches(low, Fp2::ZERO)?;
                matches((zoff - low) * inv + class, y)?;
            }
            Pattern::RequantWide { split, middle } => {
                let h0 = optional(self.middle)?;
                let g = optional(self.saturation)?;
                let inv = power(
                    u32::from(split)
                        .checked_add(u32::from(middle))
                        .ok_or(GadgetError::Internal)?,
                )?
                .inverse()
                .ok_or(GadgetError::Internal)?;
                let h1off = (zoff - low - power(u32::from(split))? * h0) * inv;
                if split > 0 {
                    matches(low, Fp2::ZERO)?;
                }
                matches(h0, Fp2::ZERO)?;
                matches(h1off, Fp2::ZERO)?;
                // RFC-0024: pack(0, 2*h1) = 2*h1_off, including at field points.
                matches(Fp2::from(Fp::reduce(2)) * h1off, g)?;
                matches(
                    h0 + power(u32::from(middle))? * g + power(u32::from(middle) + 1)? + class,
                    y,
                )?;
            }
            _ => return Err(GadgetError::Internal),
        }
        if requests.next().is_some() {
            return Err(GadgetError::Internal);
        }
        Ok(Checked {
            site: self,
            point,
            channel,
            values,
            lookups,
        })
    }
}

impl<'a> Checked<'_, 'a> {
    /// All canonical source claims. Virtual graph tensors must be reduced before
    /// forwarding their committed sources to the opening verifier.
    pub fn claims(&self) -> impl Iterator<Item = Claim<'a>> + '_ {
        self.site
            .sources
            .iter()
            .take(self.site.count)
            .enumerate()
            .zip(self.values)
            .map(|((i, &source), &value)| Claim {
                input: Input {
                    source,
                    point: if self.site.class == Some(i) {
                        self.channel
                    } else {
                        self.point
                    },
                },
                value,
            })
    }

    /// Every key/value claim in the same order as the canonical LogUp requests.
    pub fn lookups(&self) -> impl Iterator<Item = Lookup<'a>> + '_ {
        self.site
            .choice
            .tables
            .iter()
            .zip(self.lookups)
            .map(|(&table, &(key, value))| Lookup {
                table,
                point: self.point,
                key,
                value,
            })
    }
}
