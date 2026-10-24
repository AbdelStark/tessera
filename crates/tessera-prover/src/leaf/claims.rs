// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Where the leaf stage leaves its claims, for the stages and the openings that take them
//! (RFC-0045 'The leaf point', RFC-0054 'The stage's outputs').
//!
//! - **Ungrouped.** Every claim on a tensor of `n` variables lies at `P[0..n]`, a split value at
//!   `π_t(P[0..k])`; the forests' leaf claims and the multiplicities' are the LogUp proof's.
//! - **Grouped.** A claim on a tensor of `n` variables with group variables `S`, the low `g_T`
//!   variables of its token axis, lies at the merged point `r(n, S)`: the merge's last `n`
//!   challenges, assigned to the variables outside `S` lowest first, then to those in `S`. A
//!   split value's claim is its merged polynomial's; the forests' leaf claims are their merged
//!   claims, and the multiplicities' the table batch's.
//!
//! [`LeafClaims`] gives both alike, so the native relations ([`ModelNativeProver`]) and the
//! openings ([`ModelOpenings`]) read them without knowing how the stage was proven.
//!
//! [`ModelNativeProver`]: crate::native::ModelNativeProver
//! [`ModelOpenings`]: crate::ModelOpenings

use tessera_field::GoldilocksExt2;
use tessera_sumcheck::Claim;

use super::layout::axis_vars;
use super::{GroupedLayout, LeafLayout, ModelColumn, ProvenGroupedLeaf, ProvenLeaf};

/// The claims a proven leaf stage leaves, at their points.
pub trait LeafClaims {
    /// The point of `vars` variables with group variables `group_vars` (ascending positions):
    /// `P[0..vars]` for an ungrouped stage, which has none, and `r(vars, S)` for a grouped one.
    /// `None` past the stage's point.
    fn point(&self, vars: usize, group_vars: &[usize]) -> Option<Vec<GoldilocksExt2>>;

    /// The token groups' bits on the token axis, `g_T`; 0 for an ungrouped stage.
    fn token_bits(&self) -> u32;

    /// The point of the stage's claim on a tensor of `shape` whose token axis is `token`
    /// (RFC-0054 'The stage's outputs'): its group variables are the token axis's low `g_T`
    /// variables, the axes after it taking the lower ones.
    fn tensor_point(&self, shape: &[u32], token: Option<usize>) -> Option<Vec<GoldilocksExt2>> {
        let vars: u32 = shape.iter().map(|&len| axis_vars(len)).sum();
        let group_vars: Vec<usize> = match token {
            Some(a) => {
                let low: u32 = shape.get(a + 1..)?.iter().map(|&len| axis_vars(len)).sum();
                (low..low + self.token_bits()).map(|v| v as usize).collect()
            }
            None => Vec::new(),
        };
        self.point(vars as usize, &group_vars)
    }

    /// The point of split value `read` at site `site` and layer `layer` of `layout`, the stage's
    /// layout: its tensor's claim.
    fn read_point(
        &self,
        layout: &LeafLayout<ModelColumn>,
        site: usize,
        layer: usize,
        read: usize,
    ) -> Option<Vec<GoldilocksExt2>>;

    /// Forest `k`'s leaf claim, on its stacked leaf table.
    fn forest_claim(&self, k: usize) -> Option<&Claim<GoldilocksExt2>>;

    /// Instance `k`'s claim on its multiplicities, in table order.
    fn multiplicity_claim(&self, k: usize) -> Option<&Claim<GoldilocksExt2>>;

    /// The instances' count: the stage's tables.
    fn instances(&self) -> usize;
}

impl LeafClaims for ProvenLeaf {
    fn point(&self, vars: usize, group_vars: &[usize]) -> Option<Vec<GoldilocksExt2>> {
        match group_vars.is_empty() {
            true => self.point().get(..vars).map(<[_]>::to_vec),
            false => None,
        }
    }

    fn token_bits(&self) -> u32 {
        0
    }

    fn read_point(
        &self,
        layout: &LeafLayout<ModelColumn>,
        site: usize,
        layer: usize,
        read: usize,
    ) -> Option<Vec<GoldilocksExt2>> {
        layout
            .reads(site, layer)?
            .get(read)?
            .coords
            .iter()
            .map(|&c| self.point().get(c as usize).copied())
            .collect()
    }

    fn forest_claim(&self, k: usize) -> Option<&Claim<GoldilocksExt2>> {
        self.logup().forest_claims().get(k)
    }

    fn multiplicity_claim(&self, k: usize) -> Option<&Claim<GoldilocksExt2>> {
        self.logup().claims().get(k).map(|c| &c.multiplicity)
    }

    fn instances(&self) -> usize {
        self.logup().claims().len()
    }
}

/// A grouped stage's claims: its proof and its layouts.
#[derive(Clone, Copy, Debug)]
pub struct GroupedClaims<'a> {
    proven: &'a ProvenGroupedLeaf,
    grouped: &'a GroupedLayout,
}

impl<'a> GroupedClaims<'a> {
    /// The claims `proven` leaves under `grouped`, the layouts it was proven over.
    pub fn new(proven: &'a ProvenGroupedLeaf, grouped: &'a GroupedLayout) -> Self {
        Self { proven, grouped }
    }
}

impl LeafClaims for GroupedClaims<'_> {
    fn point(&self, vars: usize, group_vars: &[usize]) -> Option<Vec<GoldilocksExt2>> {
        self.proven.merged_point(vars, group_vars)
    }

    fn token_bits(&self) -> u32 {
        u32::from(self.grouped.groups().token_bits)
    }

    fn read_point(
        &self,
        layout: &LeafLayout<ModelColumn>,
        site: usize,
        layer: usize,
        read: usize,
    ) -> Option<Vec<GoldilocksExt2>> {
        // The merged polynomial is the read's index in the ungrouped split.
        let mut polynomial = 0usize;
        for (s, entry) in layout.sites().iter().enumerate() {
            for l in 0..entry.layers.len() {
                let reads = layout.reads(s, l)?.len();
                if (s, l) == (site, layer) {
                    return (read < reads)
                        .then(|| self.proven.merged().get(polynomial + read))
                        .flatten()
                        .map(|claim| claim.point.clone());
                }
                polynomial += reads;
            }
        }
        None
    }

    fn forest_claim(&self, k: usize) -> Option<&Claim<GoldilocksExt2>> {
        self.proven
            .merged()
            .get(self.grouped.polynomials().len().checked_add(k)?)
    }

    fn multiplicity_claim(&self, k: usize) -> Option<&Claim<GoldilocksExt2>> {
        self.proven.tables().multiplicities().get(k)
    }

    fn instances(&self) -> usize {
        self.proven.tables().multiplicities().len()
    }
}
