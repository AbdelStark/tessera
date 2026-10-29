// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{Family, Final, Layer, layer_eq, reserve};
use crate::gadgets::{GadgetError, Plan, softmax};
use crate::lowering::NodeId;
use crate::sumcheck::{Instance, eq};
use tv_field::Fp2;

/// One Softmax template position, retaining every real layer's original table and sources.
pub struct SoftmaxStack<'p> {
    family: Family,
    sites: Vec<softmax::Softmax<'p>>,
}
/// A masked-key or forest-leaf native target at its prescribed point.
#[must_use = "check the native batch and all original lookup/source bindings"]
pub struct Key<'s, 'a> {
    stack: &'s SoftmaxStack<'s>,
    point: &'a [Fp2],
    layer: &'a [Fp2],
    claim: Fp2,
    lookups: Vec<softmax::Key<'s, 'a>>,
}
/// Original per-layer reciprocal/slack checks and their one stacked product target.
#[must_use = "bind every original per-layer source/lookup claim and check product and padding"]
pub struct SoftmaxChecked<'s, 'a> {
    stack: &'s SoftmaxStack<'s>,
    checked: Vec<softmax::Checked<'s, 'a>>,
    layer: &'a [Fp2],
}
impl Plan<'_> {
    /// Derive a Softmax family from a first-layer or non-template node.
    /// `Internal` for wrong own node/template/geometry; failed reserves give `Memory`.
    pub fn softmax_stack(&self, node: NodeId) -> Result<SoftmaxStack<'_>, GadgetError> {
        let family = Family::derive(self.graph, node)?;
        let mut sites = reserve(family.nodes.len())?;
        for id in &family.nodes {
            sites.push(
                self.softmaxes()
                    .find_map(|s| match s {
                        Ok(s) if s.node() == *id => Some(Ok(s)),
                        Err(e) => Some(Err(e)),
                        _ => None,
                    })
                    .ok_or(GadgetError::Internal)??,
            );
        }
        let first = sites.first().ok_or(GadgetError::Internal)?;
        if sites.iter().any(|s| {
            s.variables() != first.variables()
                || s.key_variables() != first.key_variables()
                || s.row_variables() != first.row_variables()
                || s.range_tables().len() != first.range_tables().len()
                || s.n != first.n
                || s.slots != first.slots
                || s.heads != first.heads
        }) {
            return Err(GadgetError::Internal);
        }
        Ok(SoftmaxStack { family, sites })
    }
}
impl<'p> SoftmaxStack<'p> {
    /// Actual sites, preserving all original per-layer table ids and sources.
    pub fn sites(&self) -> &[softmax::Softmax<'p>] {
        &self.sites
    }
    /// Highest layer coordinates.
    pub fn layer_variables(&self) -> usize {
        self.family.vars
    }
    /// Bind each real layer's fixed Exp key/value at the shared lookup point. Public
    /// `K0_l*(1-C_local)` is subtracted BEFORE folding, so different layer tables are supported.
    /// No late per-layer evaluations may replace these fixed targets.
    /// Own counts/points give `Internal`; reserves give `Memory`.
    pub fn lookup<'a>(
        &'a self,
        point: &'a [Fp2],
        layer: &'a [Fp2],
        values: &[(Fp2, Fp2)],
    ) -> Result<Key<'a, 'a>, GadgetError> {
        if values.len() != self.sites.len() || layer.len() != self.family.vars {
            return Err(GadgetError::Internal);
        }
        let mut lookups = reserve(values.len())?;
        let mut claims = reserve(values.len())?;
        for (&site, &(key, value)) in self.sites.iter().zip(values) {
            let bound = site.lookup(point, key, value)?;
            claims.push(bound.instance().claim);
            lookups.push(bound);
        }
        Ok(Key {
            stack: self,
            point,
            layer,
            claim: self.family.fold(layer, claims)?,
            lookups,
        })
    }
    /// Bind a previously checked forest leaf at ITS OWN full key/query/head/layer point.
    /// For a scalar key domain the driver supplies the public root `1-psi` as this leaf.
    /// Wrong own geometry is `Internal`; reserves give `Memory`.
    pub fn forest<'a>(&'a self, point: &'a [Fp2], leaf: Fp2) -> Result<Key<'a, 'a>, GadgetError> {
        let site = self.sites.first().ok_or(GadgetError::Internal)?;
        let (point, layer) = self.family.point(point, site.variables())?;
        Ok(Key {
            stack: self,
            point,
            layer,
            claim: leaf - Fp2::ONE + site.causal(point)? * self.family.real(layer)?,
            lookups: Vec::new(),
        })
    }
    /// Check each real layer's original reciprocal/slack bindings at the fixed row point.
    /// Every returned per-layer binding remains owed. Count/affine errors retain the gadget id;
    /// own layer geometry is `Internal`, failed reserves `Memory`.
    pub fn check<'a>(
        &'a self,
        point: &'a [Fp2],
        layer: &'a [Fp2],
        values: &[Layer<'a>],
    ) -> Result<SoftmaxChecked<'a, 'a>, GadgetError> {
        if values.len() != self.sites.len() || layer.len() != self.family.vars {
            return Err(GadgetError::Internal);
        }
        let mut checked = reserve(values.len())?;
        for (&site, value) in self.sites.iter().zip(values) {
            checked.push(site.check(point, value.sources, value.keys, value.values)?);
        }
        Ok(SoftmaxChecked {
            stack: self,
            checked,
            layer,
        })
    }
    /// Maximum/reciprocal padding metadata, with layer variables highest, degree three.
    /// Inconsistent own metadata is `Internal`.
    pub fn padding_instance(&self) -> Result<Instance, GadgetError> {
        let mut out = self
            .sites
            .first()
            .ok_or(GadgetError::Internal)?
            .padding_instance();
        out.vars = out
            .vars
            .checked_add(self.family.vars)
            .ok_or(GadgetError::Internal)?;
        Ok(out)
    }
    /// Mandatory maximum or reciprocal zero-check including padded layers.
    /// Wrong proof count retains the representative gadget; own geometry is `Internal`.
    pub fn padding(
        &self,
        kind: softmax::Padding,
        zero: &[Fp2],
        point: &[Fp2],
        values: &[Fp2],
    ) -> Result<Final, GadgetError> {
        let site = self.sites.first().ok_or(GadgetError::Internal)?;
        let (local, layer) = self.family.point(point, site.row_variables())?;
        let (z, _) = self.family.point(zero, site.row_variables())?;
        let mut finals = reserve(self.sites.len())?;
        for &site in &self.sites {
            finals.push(site.padding(kind, z, local, values)?);
        }
        let mut rows = reserve(finals.len())?;
        for checked in &finals {
            let mut row = reserve(1)?;
            row.push(checked.claim());
            rows.push(row);
        }
        let value = rows
            .first()
            .and_then(|r| r.first())
            .ok_or(GadgetError::Internal)?
            .value;
        Ok(Final {
            value: eq(zero, point).ok_or(GadgetError::Internal)?
                * (Fp2::ONE - site.psi(local)? * self.family.real(layer)?)
                * value,
            claims: self.family.claims(layer, &rows)?,
        })
    }
}
impl Key<'_, '_> {
    /// Every original per-layer Exp lookup and exponential source claim; empty for a forest.
    pub fn lookups(&self) -> &[softmax::Key<'_, '_>] {
        &self.lookups
    }
    /// Degree-three key relation metadata for the ONE stage batch.
    /// Wrong own sizes give `Internal`.
    pub fn instance(&self) -> Result<Instance, GadgetError> {
        let site = self.stack.sites.first().ok_or(GadgetError::Internal)?;
        Ok(Instance {
            claim: self.claim,
            vars: site
                .variables()
                .checked_add(self.stack.family.vars)
                .ok_or(GadgetError::Internal)?,
            evals: 2,
        })
    }
    /// Evaluate the key relation and retain maximum/score families, dropping only maximum's key
    /// coordinates. Count errors retain the representative gadget; own geometry is `Internal`.
    pub fn evaluate(&self, point: &[Fp2], values: &[Fp2]) -> Result<Final, GadgetError> {
        let site = self.stack.sites.first().ok_or(GadgetError::Internal)?;
        let (local, layer) = self.stack.family.point(point, site.variables())?;
        let mut finals = reserve(self.stack.sites.len())?;
        let mut value = None;
        for &site in &self.stack.sites {
            // Only the final polynomial is used here; the fixed incoming target is self.claim.
            let key = site.tree(self.point, Fp2::ZERO)?;
            let checked = key.evaluate(local, values)?;
            value = Some(checked.value());
            finals.push(checked);
        }
        let mut rows = reserve(finals.len())?;
        for checked in &finals {
            let mut row = reserve(2)?;
            row.extend(checked.claims());
            rows.push(row);
        }
        Ok(Final {
            value: value.ok_or(GadgetError::Internal)?
                * layer_eq(self.layer, layer)?
                * self.stack.family.real(layer)?,
            claims: self.stack.family.claims(layer, &rows)?,
        })
    }
}
impl SoftmaxChecked<'_, '_> {
    /// Original per-layer affine source/Range bindings, unchanged and still owed.
    pub fn layers(&self) -> &[softmax::Checked<'_, '_>] {
        &self.checked
    }
    /// Fixed stacked reciprocal product target for the ONE stage batch, degree three.
    /// Inconsistent own geometry is `Internal`; failed reserves give `Memory`.
    pub fn instance(&self) -> Result<Instance, GadgetError> {
        let first = self
            .checked
            .first()
            .ok_or(GadgetError::Internal)?
            .instance();
        Ok(Instance {
            claim: self
                .stack
                .family
                .fold(self.layer, self.checked.iter().map(|c| c.instance().claim))?,
            vars: first
                .vars
                .checked_add(self.stack.family.vars)
                .ok_or(GadgetError::Internal)?,
            evals: first.evals,
        })
    }
    /// Final reciprocal product, retaining E at half-key with all final layer coordinates.
    /// Wrong count retains the representative gadget; own geometry is `Internal`.
    pub fn evaluate(&self, point: &[Fp2], values: &[Fp2]) -> Result<Final, GadgetError> {
        let site = self.stack.sites.first().ok_or(GadgetError::Internal)?;
        let (local, layer) = self.stack.family.point(point, site.row_variables())?;
        let mut finals = reserve(self.checked.len())?;
        let mut value = None;
        for bound in &self.checked {
            let checked = bound.evaluate(local, values)?;
            value = Some(checked.value());
            finals.push(checked);
        }
        let mut rows = reserve(finals.len())?;
        for checked in &finals {
            let mut row = reserve(2)?;
            row.extend(checked.claims());
            rows.push(row);
        }
        Ok(Final {
            value: value.ok_or(GadgetError::Internal)? * layer_eq(self.layer, layer)?,
            claims: self.stack.family.claims(layer, &rows)?,
        })
    }
}
