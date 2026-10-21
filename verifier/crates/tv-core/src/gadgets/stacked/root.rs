// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{Family, Final, Layer, layer_eq, reserve};
use crate::gadgets::{GadgetError, Plan, root};
use crate::lowering::NodeId;
use crate::sumcheck::{Instance, eq};
use tv_field::Fp2;

/// One root template position, or one root outside the layer template.
pub struct RootStack<'p> {
    family: Family,
    sites: Vec<root::Root<'p>>,
}
/// Per-real-layer affine checks retained alongside their stacked product targets.
#[must_use = "bind every layer's original source/lookup claims and prove all products and padding"]
pub struct RootChecked<'r, 'a> {
    stack: &'r RootStack<'r>,
    checked: Vec<root::Checked<'a>>,
    layer: &'a [Fp2],
}
/// One complete byte coefficient of a stacked root product.
#[derive(Clone, Copy)]
pub struct Product<'r, 'a, 'c> {
    checked: &'c RootChecked<'r, 'a>,
    index: usize,
}
impl Plan<'_> {
    /// Derive one root family from a first-layer or non-template node.
    /// All nodes and product geometry come from the independent public lowering.
    /// `Internal` means a wrong own node/template/geometry; failed reserves give `Memory`.
    pub fn root_stack(&self, node: NodeId) -> Result<RootStack<'_>, GadgetError> {
        let family = Family::derive(self.graph, node)?;
        let mut sites = reserve(family.nodes.len())?;
        for id in &family.nodes {
            let site = self
                .roots()
                .find_map(|site| match site {
                    Ok(site) if site.node() == *id => Some(Ok(site)),
                    Err(e) => Some(Err(e)),
                    _ => None,
                })
                .ok_or(GadgetError::Internal)??;
            sites.push(site);
        }
        let first = sites.first().ok_or(GadgetError::Internal)?;
        if sites.iter().any(|s| {
            s.padding_instance() != first.padding_instance()
                || s.residual_padding_instance() != first.residual_padding_instance()
                || s.sources().len() != first.sources().len()
                || s.n != first.n
                || s.dim != first.dim
                || s.epsilon != first.epsilon
        }) {
            return Err(GadgetError::Internal);
        }
        Ok(RootStack { family, sites })
    }
}
impl<'p> RootStack<'p> {
    /// Actual root sites in real-layer order; their individual table bindings stay distinct.
    pub fn sites(&self) -> &[root::Root<'p>] {
        &self.sites
    }
    /// Highest coordinates of this family's layer axis.
    pub fn layer_variables(&self) -> usize {
        self.family.vars
    }
    /// Check every real layer at its fixed leaf point, before stage gamma.
    /// Keep and bind every entry of [`RootChecked::layers`]. No new lookup is formed by stacking.
    /// Existing per-node count/affine errors retain their gadget ids; own layer geometry is
    /// `Internal`, failed reserves `Memory`.
    pub fn check<'a>(
        &'a self,
        point: &'a [Fp2],
        layer: &'a [Fp2],
        values: &[Layer<'a>],
    ) -> Result<RootChecked<'a, 'a>, GadgetError> {
        if values.len() != self.sites.len() || layer.len() != self.family.vars {
            return Err(GadgetError::Internal);
        }
        let mut checked = reserve(values.len())?;
        for (site, value) in self.sites.iter().zip(values) {
            checked.push(site.check(point, value.sources, value.keys, value.values)?);
        }
        let count = checked
            .first()
            .ok_or(GadgetError::Internal)?
            .products()
            .count();
        if checked.iter().any(|c| c.products().count() != count) {
            return Err(GadgetError::Internal);
        }
        for index in 0..count {
            let first = checked
                .first()
                .and_then(|c| c.products().nth(index))
                .ok_or(GadgetError::Internal)?
                .instance();
            if checked.iter().any(|c| {
                c.products().nth(index).is_none_or(|p| {
                    p.instance().vars != first.vars || p.instance().evals != first.evals
                })
            }) {
                return Err(GadgetError::Internal);
            }
        }
        Ok(RootChecked {
            stack: self,
            checked,
            layer,
        })
    }
    /// Mandatory root or residual padding metadata, with layer variables highest, degree three.
    /// Wrong own metadata is `Internal`.
    pub fn padding_instance(&self, residual: bool) -> Result<Instance, GadgetError> {
        let site = self.sites.first().ok_or(GadgetError::Internal)?;
        let mut out = if residual {
            site.residual_padding_instance()
        } else {
            site.padding_instance()
        };
        out.vars = out
            .vars
            .checked_add(self.family.vars)
            .ok_or(GadgetError::Internal)?;
        Ok(out)
    }
    /// Root or residual padding at the common stage zero point, retaining the source family.
    /// Residual padding covers feature, time AND layer axes; it is a premise of `u=S+epsilon*psi`.
    /// Existing count errors retain the representative gadget; own point errors are `Internal`.
    pub fn padding(
        &self,
        residual: bool,
        zero: &[Fp2],
        point: &[Fp2],
        values: &[Fp2],
    ) -> Result<Final, GadgetError> {
        let site = self.sites.first().ok_or(GadgetError::Internal)?;
        let local_vars = if residual {
            site.residual_padding_instance().vars
        } else {
            site.padding_instance().vars
        };
        let (local, layer) = self.family.point(point, local_vars)?;
        let (z, _) = self.family.point(zero, local_vars)?;
        let mut rows = reserve(self.sites.len())?;
        for site in &self.sites {
            let result = if residual {
                site.residual_padding(z, local, values)?
            } else {
                site.padding(z, local, values)?
            };
            let mut row = reserve(1)?;
            row.push(result.claim());
            rows.push(row);
        }
        let first = rows
            .first()
            .and_then(|r| r.first())
            .ok_or(GadgetError::Internal)?;
        let psi = if residual {
            let (feature, time) = local
                .split_at_checked(site.features)
                .ok_or(GadgetError::Internal)?;
            crate::selectors::lt(feature, u64::from(site.dim)).ok_or(GadgetError::Internal)?
                * crate::selectors::lt(time, u64::from(site.n)).ok_or(GadgetError::Internal)?
        } else {
            crate::selectors::lt(local, u64::from(site.n)).ok_or(GadgetError::Internal)?
        };
        Ok(Final {
            value: eq(zero, point).ok_or(GadgetError::Internal)?
                * (Fp2::ONE - psi * self.family.real(layer)?)
                * first.value,
            claims: self.family.claims(layer, &rows)?,
        })
    }
}
impl<'r, 'a> RootChecked<'r, 'a> {
    /// Every original per-layer source and lookup binding, at its original point/table.
    pub fn layers(&self) -> &[root::Checked<'a>] {
        &self.checked
    }
    /// Both FULL product vectors, lower then upper, in byte order. No high coefficient omitted.
    pub fn products(&self) -> impl Iterator<Item = Product<'r, 'a, '_>> {
        let n = self
            .checked
            .first()
            .map(|c| c.products().count())
            .unwrap_or(0);
        (0..n).map(move |index| Product {
            checked: self,
            index,
        })
    }
}
impl Product<'_, '_, '_> {
    /// Degree-four fixed incoming target and variable/evaluation counts for the ONE stage batch.
    /// An inconsistent own coefficient/point is `Internal`.
    pub fn instance(&self) -> Result<Instance, GadgetError> {
        let c = self.checked;
        let first = c
            .checked
            .first()
            .and_then(|c| c.products().nth(self.index))
            .ok_or(GadgetError::Internal)?
            .instance();
        let mut claims = reserve(c.checked.len())?;
        for checked in &c.checked {
            claims.push(
                checked
                    .products()
                    .nth(self.index)
                    .ok_or(GadgetError::Internal)?
                    .instance()
                    .claim,
            );
        }
        Ok(Instance {
            claim: c.stack.family.fold(c.layer, claims)?,
            vars: first
                .vars
                .checked_add(c.stack.family.vars)
                .ok_or(GadgetError::Internal)?,
            evals: first.evals,
        })
    }
    /// Final raw coefficient and original stacked operand debts. Front-padding must already be
    /// removed by `verify_batch`; its returned layer coordinates are kept on every source.
    /// Wrong count retains the component rejection; own geometry `Internal`, reserves `Memory`.
    pub fn evaluate(&self, point: &[Fp2], values: &[Fp2]) -> Result<Final, GadgetError> {
        let c = self.checked;
        let p = c
            .checked
            .first()
            .and_then(|c| c.products().nth(self.index))
            .ok_or(GadgetError::Internal)?;
        let (local, layer) = c.stack.family.point(point, p.instance().vars)?;
        let mut rows = reserve(c.checked.len())?;
        let mut value = None;
        for checked in &c.checked {
            let final_value = checked
                .products()
                .nth(self.index)
                .ok_or(GadgetError::Internal)?
                .evaluate(local, values)?;
            if value.is_some_and(|v| v != final_value.value()) {
                return Err(GadgetError::Internal);
            }
            value = Some(final_value.value());
            let mut row = reserve(values.len().checked_add(2).ok_or(GadgetError::Internal)?)?;
            row.extend(final_value.claims());
            rows.push(row);
        }
        let mut claims = c.stack.family.claims(layer, &rows)?;
        let site = c.stack.sites.first().ok_or(GadgetError::Internal)?;
        let psi = crate::selectors::lt(local, u64::from(site.n)).ok_or(GadgetError::Internal)?;
        // RFC-0050's graph S is zero padded over time AND layers.
        let squares = claims.first_mut().ok_or(GadgetError::Internal)?;
        squares.value += site.epsilon * psi * (Fp2::ONE - c.stack.family.real(layer)?);
        Ok(Final {
            value: value.ok_or(GadgetError::Internal)? * layer_eq(c.layer, layer)?,
            claims,
        })
    }
}
