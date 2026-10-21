// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0037 native instances stacked over real layers, with explicit source families.
//!
//! First check each layer's fixed affine/lookup values. Subtract public constants
//! there, then stack the resulting native targets, before the stage batching
//! challenge. In particular Softmax stacks `kappa_l-K0_l*(1-C_local)`, not an
//! assumed zero-padded kappa: tables may have different `K0_l`. A forest instead
//! already names its full layer point, and retains that point.
//!
//! These instances join the ONE mixed `sumcheck::verify_batch` for their stage.
//! Returned claims and every original per-layer binding remain debts: this module
//! neither routes graph atoms nor checks commitment openings or accepts a model.

mod root;
mod softmax;
pub use root::{Product, RootChecked, RootStack};
pub use softmax::{Key, SoftmaxChecked, SoftmaxStack};

use super::GadgetError;
use crate::advice::Source;
use crate::lowering::{Graph, NodeId};
use crate::sumcheck::eq;
use tv_field::Fp2;

/// Fixed evaluations of one real layer, in the per-node component's source/request order.
#[derive(Clone, Copy)]
pub struct Layer<'a> {
    /// Original source values, including virtual inputs already bound to their graph equations.
    pub sources: &'a [Fp2],
    /// Lookup key evaluations at the prescribed leaf point.
    pub keys: &'a [Fp2],
    /// Lookup value evaluations (zero for Range).
    pub values: &'a [Fp2],
}

/// A zero-padded stacked original source, still owed to graph reduction or opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    /// Actual canonical sources, one per real layer in order; one outside the template.
    pub sources: Vec<Source>,
    /// Natural source coordinates, excluding the highest layer coordinates.
    pub point: Vec<Fp2>,
    /// Highest coordinates; empty for a source outside the template or a single real layer.
    pub layer_point: Vec<Fp2>,
    /// `sum_real eq(layer_point,l)*source_l(point)`; padded layers contribute zero.
    pub value: Fp2,
}

/// A native callback result. Its value checks the batch; its claims remain explicit debts.
#[must_use = "check the batch and discharge every returned original-source claim"]
pub struct Final {
    /// Final polynomial value for the existing stage batch.
    pub value: Fp2,
    /// Original source families, at their prescribed points.
    pub claims: Vec<Claim>,
}

struct Family {
    nodes: Vec<NodeId>,
    vars: usize,
}
fn reserve<T>(n: usize) -> Result<Vec<T>, GadgetError> {
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| GadgetError::Memory)?;
    Ok(out)
}
fn copy<T: Copy>(v: &[T]) -> Result<Vec<T>, GadgetError> {
    let mut out = reserve(v.len())?;
    out.extend_from_slice(v);
    Ok(out)
}
fn bits(n: usize) -> Result<usize, GadgetError> {
    if n == 0 {
        return Err(GadgetError::Internal);
    }
    usize::try_from(usize::BITS.saturating_sub(n.saturating_sub(1).leading_zeros()))
        .map_err(|_| GadgetError::Internal)
}
fn basis(point: &[Fp2], index: usize) -> Fp2 {
    point.iter().enumerate().fold(Fp2::ONE, |out, (bit, &x)| {
        out * if (index >> bit) & 1 == 0 {
            Fp2::ONE - x
        } else {
            x
        }
    })
}
impl Family {
    fn derive(graph: &Graph, node: NodeId) -> Result<Self, GadgetError> {
        let count = usize::try_from(crate::schedule::template_layers(graph).map_err(
            |e| match e {
                crate::schedule::ScheduleError::Memory => GadgetError::Memory,
                _ => GadgetError::Internal,
            },
        )?)
        .map_err(|_| GadgetError::Internal)?;
        let producer = |t| {
            graph
                .tensor(t)
                .and_then(|t| t.producer)
                .ok_or(GadgetError::Internal)
        };
        let start = producer(*graph.checkpoints().first().ok_or(GadgetError::Internal)?)?
            .0
            .checked_add(1)
            .ok_or(GadgetError::Internal)?;
        let end = producer(*graph.checkpoints().last().ok_or(GadgetError::Internal)?)?
            .0
            .checked_add(1)
            .ok_or(GadgetError::Internal)?;
        graph.node(node).ok_or(GadgetError::Internal)?;
        let width = end
            .checked_sub(start)
            .and_then(|n| n.checked_div(u32::try_from(count).ok()?))
            .filter(|&n| n > 0)
            .ok_or(GadgetError::Internal)?;
        let stacked = node.0 >= start && node.0 < end;
        if stacked && node.0 >= start.checked_add(width).ok_or(GadgetError::Internal)? {
            return Err(GadgetError::Internal);
        }
        let n = if stacked { count } else { 1 };
        let mut nodes = reserve(n)?;
        for layer in 0..n {
            let offset = width
                .checked_mul(u32::try_from(layer).map_err(|_| GadgetError::Internal)?)
                .and_then(|n| node.0.checked_add(n))
                .ok_or(GadgetError::Internal)?;
            nodes.push(NodeId(if stacked { offset } else { node.0 }));
        }
        Ok(Self {
            nodes,
            vars: bits(n)?,
        })
    }
    fn point<'a>(
        &self,
        point: &'a [Fp2],
        local: usize,
    ) -> Result<(&'a [Fp2], &'a [Fp2]), GadgetError> {
        if point.len() != local.checked_add(self.vars).ok_or(GadgetError::Internal)? {
            return Err(GadgetError::Internal);
        }
        point.split_at_checked(local).ok_or(GadgetError::Internal)
    }
    fn fold(
        &self,
        point: &[Fp2],
        values: impl IntoIterator<Item = Fp2>,
    ) -> Result<Fp2, GadgetError> {
        if point.len() != self.vars {
            return Err(GadgetError::Internal);
        }
        let mut out = Fp2::ZERO;
        let mut n = 0usize;
        for value in values {
            if n >= self.nodes.len() {
                return Err(GadgetError::Internal);
            }
            out += basis(point, n) * value;
            n = n.checked_add(1).ok_or(GadgetError::Internal)?;
        }
        if n != self.nodes.len() {
            return Err(GadgetError::Internal);
        }
        Ok(out)
    }
    fn real(&self, point: &[Fp2]) -> Result<Fp2, GadgetError> {
        if self.vars != point.len() {
            return Err(GadgetError::Internal);
        }
        let layers = u64::try_from(self.nodes.len()).map_err(|_| GadgetError::Internal)?;
        crate::selectors::lt(point, layers).ok_or(GadgetError::Internal)
    }
    fn claims(
        &self,
        layer: &[Fp2],
        rows: &[Vec<super::Claim<'_>>],
    ) -> Result<Vec<Claim>, GadgetError> {
        if layer.len() != self.vars || rows.len() != self.nodes.len() {
            return Err(GadgetError::Internal);
        }
        let first = rows.first().ok_or(GadgetError::Internal)?;
        if rows.iter().any(|r| r.len() != first.len()) {
            return Err(GadgetError::Internal);
        }
        let mut out = reserve(first.len())?;
        for (index, original) in first.iter().enumerate() {
            let mut sources = reserve(rows.len())?;
            for row in rows {
                let claim = row.get(index).ok_or(GadgetError::Internal)?;
                if claim.input.point != original.input.point || claim.value != original.value {
                    return Err(GadgetError::Internal);
                }
                sources.push(claim.input.source);
            }
            out.push(Claim {
                sources,
                point: copy(original.input.point)?,
                layer_point: copy(layer)?,
                value: original.value,
            });
        }
        Ok(out)
    }
}
fn layer_eq(a: &[Fp2], b: &[Fp2]) -> Result<Fp2, GadgetError> {
    eq(a, b).ok_or(GadgetError::Internal)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::gadgets::{Claim as Original, Input};
    use crate::lowering::TensorId;
    use tv_field::Fp;
    fn f(n: u64) -> Fp2 {
        Fp2::from(Fp::reduce(n))
    }
    #[test]
    fn ragged_fold_keeps_exact_counts_and_zero_padding() {
        let family = Family {
            nodes: vec![NodeId(2), NodeId(7), NodeId(12)],
            vars: 2,
        };
        let p = [f(2), f(3)];
        assert_eq!(family.fold(&p, [f(1), f(2), f(4)]), Ok(-f(18)));
        assert_eq!(family.real(&p), Ok(-f(5)));
        assert_eq!(family.real(&p[..1]), Err(GadgetError::Internal));
        for values in [vec![], vec![f(1), f(2)], vec![f(1), f(2), f(4), f(8)]] {
            assert_eq!(family.fold(&p, values), Err(GadgetError::Internal));
        }
        assert_eq!(
            family.fold(&p[..1], [f(1), f(2), f(4)]),
            Err(GadgetError::Internal)
        );
        assert_eq!(family.point(&p, 1), Err(GadgetError::Internal));
        let mut consumed = 0;
        let hostile = core::iter::from_fn(|| {
            consumed += 1;
            assert!(consumed <= 4, "overlong own iterator was not bounded");
            Some(Fp2::ONE)
        });
        assert_eq!(family.fold(&p, hostile), Err(GadgetError::Internal));
        let scalar = Family {
            nodes: vec![NodeId(4)],
            vars: 0,
        };
        assert_eq!(scalar.fold(&[], [f(9)]), Ok(f(9)));
        assert_eq!(bits(0), Err(GadgetError::Internal));
    }
    #[test]
    fn source_families_retain_order_points_and_single_fixed_value() {
        let family = Family {
            nodes: vec![NodeId(2), NodeId(7), NodeId(12)],
            vars: 2,
        };
        let p = [f(3)];
        let other = [f(5)];
        let layers = [f(2), f(7)];
        let row = |id| {
            vec![Original {
                input: Input {
                    source: Source::Tensor(TensorId(id)),
                    point: &p,
                },
                value: f(11),
            }]
        };
        let good = vec![row(17), row(3), row(29)];
        let claims = family.claims(&layers, &good).unwrap();
        assert_eq!(
            claims[0],
            Claim {
                sources: vec![
                    Source::Tensor(TensorId(17)),
                    Source::Tensor(TensorId(3)),
                    Source::Tensor(TensorId(29))
                ],
                point: p.to_vec(),
                layer_point: layers.to_vec(),
                value: f(11)
            }
        );
        assert_eq!(
            family.claims(&layers[..1], &good),
            Err(GadgetError::Internal)
        );
        assert_eq!(
            family.claims(&layers, &good[..2]),
            Err(GadgetError::Internal)
        );
        let mut bad = good.clone();
        bad[1].clear();
        assert_eq!(family.claims(&layers, &bad), Err(GadgetError::Internal));
        let mut extra = good.clone();
        let duplicate = extra[1][0];
        extra[1].push(duplicate);
        assert_eq!(family.claims(&layers, &extra), Err(GadgetError::Internal));
        let mut bad = good.clone();
        bad[1][0].input.point = &other;
        assert_eq!(family.claims(&layers, &bad), Err(GadgetError::Internal));
        let mut bad = good.clone();
        bad[1][0].value += Fp2::ONE;
        assert_eq!(family.claims(&layers, &bad), Err(GadgetError::Internal));
    }
}
