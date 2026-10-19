// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integer intervals derived from the verifier's public graph (RFC-0006, SP-7).

use crate::lowering::{Broadcast, Graph, Node, Op, TensorId};
use tv_tables::TableKind;

/// An inclusive i128 interval, or unbounded when exact endpoint arithmetic overflows.
///
/// This is derived from public model metadata, never observed weight or activation values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval(Option<(i128, i128)>);

impl Interval {
    #[cfg(test)]
    pub(crate) fn test_bounds(lo: i128, hi: i128) -> Self {
        Self(Some((lo, hi)))
    }

    /// Inclusive endpoints, or `None` for an unbounded interval.
    pub fn bounds(&self) -> Option<(i128, i128)> {
        self.0
    }
}

/// A failure of the verifier's own analysis, distinct from an invalid gadget domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeError {
    /// Storage for public intervals could not be reserved.
    Memory,
    /// The verifier's derived graph breaks its internal representation rules.
    Internal,
}

/// Derive every tensor's integer interval, in tensor-id order (RFC-0006 'Range analysis').
///
/// `graph` must be the verifier's own lowering of validated public architecture/quantization
/// data. A bounded interval is not by itself a proof that an integer-requiring gadget input is
/// admissible: the validator must also check its domain and centered-field bounds (RFC-0007 5.2).
///
/// # Errors
/// [`RangeError::Memory`] for unavailable storage; [`RangeError::Internal`] for a graph whose
/// operands or parameters break the lowering's representation. Integer overflow makes the
/// affected interval unbounded instead of wrapping.
pub fn analyze(graph: &Graph) -> Result<Vec<Interval>, RangeError> {
    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(graph.tensors().len())
        .map_err(|_| RangeError::Memory)?;
    ranges.resize(graph.tensors().len(), Interval(None));
    for input in graph.inputs() {
        if input.lo > input.hi {
            return Err(RangeError::Internal);
        }
        put(
            &mut ranges,
            input.tensor,
            Interval(Some((i128::from(input.lo), i128::from(input.hi)))),
        )?;
    }
    for node in graph.nodes() {
        let first = operand(node, 0)?;
        let a = get(&ranges, first)?;
        let output = match *node.op() {
            Op::OneHot { .. } => Interval(Some((0, 1))),
            Op::MatMulWeight { .. } => {
                let is_one_hot =
                    producer(graph, first).is_some_and(|p| matches!(p.op(), Op::OneHot { .. }));
                if is_one_hot {
                    Interval(Some((-127, 127)))
                } else {
                    let weight = graph
                        .tensor(operand(node, 1)?)
                        .ok_or(RangeError::Internal)?;
                    let fan_in =
                        i128::from(*weight.shape.dims().last().ok_or(RangeError::Internal)?);
                    symmetric(
                        a.max_abs()
                            .and_then(|m| m.checked_mul(127)?.checked_mul(fan_in)),
                    )
                }
            }
            Op::ConstMul { c } => {
                let constant = graph
                    .constants()
                    .get(usize::try_from(c).map_err(|_| RangeError::Internal)?)
                    .ok_or(RangeError::Internal)?;
                a.product(Interval(Some((
                    i128::from(*constant),
                    i128::from(*constant),
                ))))
            }
            Op::Add => a.add(get(&ranges, operand(node, 1)?)?),
            Op::SumSquares { axis } => {
                let shape = graph.tensor(first).ok_or(RangeError::Internal)?.shape;
                let count = *shape
                    .dims()
                    .get(usize::from(axis))
                    .ok_or(RangeError::Internal)?;
                let squared = a.0.and_then(|(lo, hi)| {
                    let (x, y) = (lo.checked_mul(lo)?, hi.checked_mul(hi)?);
                    Some((if lo <= 0 && hi >= 0 { 0 } else { x.min(y) }, x.max(y)))
                });
                Interval(squared).scale(i128::from(count))
            }
            Op::RsqrtRow {
                kappa,
                eps_int,
                dim,
            } => {
                let power = 1u128
                    .checked_shl(u32::from(kappa).saturating_mul(2))
                    .ok_or(RangeError::Internal)?;
                let ratio = u128::from(dim)
                    .checked_mul(power)
                    .and_then(|n| n.checked_div(u128::from(eps_int)))
                    .ok_or(RangeError::Internal)?;
                let root = ratio.isqrt();
                let width = u128::BITS.saturating_sub(root.leading_zeros()).max(1);
                let last = 1u128
                    .checked_shl(width)
                    .and_then(|v| v.checked_sub(1))
                    .ok_or(RangeError::Internal)?;
                Interval(Some((
                    0,
                    i128::try_from(last).map_err(|_| RangeError::Internal)?,
                )))
            }
            Op::Mul { broadcast } => {
                let ordinary = a.product(get(&ranges, operand(node, 1)?)?);
                match broadcast {
                    Broadcast::Trailing => mass_bound(graph, node, &ranges)?.unwrap_or(ordinary),
                    Broadcast::Leading => ordinary,
                }
            }
            Op::Requant { lo, hi, .. } => Interval(Some((i128::from(lo), i128::from(hi)))),
            Op::Reshape | Op::Transpose { .. } | Op::Expand { .. } => a,
            Op::Rotate { rope } => {
                let TableKind::Rope { bits, .. } =
                    graph.table(rope).ok_or(RangeError::Internal)?.kind
                else {
                    return Err(RangeError::Internal);
                };
                let maximum = a.max_abs().and_then(|m| {
                    m.checked_mul(
                        1i128
                            .checked_shl(u32::from(bits).saturating_add(1))
                            .filter(|&v| v > 0)?,
                    )
                });
                symmetric(maximum)
            }
            Op::MatMulAct { transpose_rhs } => {
                let right = operand(node, 1)?;
                let dims = graph.tensor(right).ok_or(RangeError::Internal)?.shape;
                let from_end = if transpose_rhs { 1 } else { 2 };
                let axis = dims
                    .dims()
                    .len()
                    .checked_sub(from_end)
                    .ok_or(RangeError::Internal)?;
                let count = *dims.dims().get(axis).ok_or(RangeError::Internal)?;
                a.product(get(&ranges, right)?).scale(i128::from(count))
            }
            Op::SoftmaxRow { sigma, .. } => {
                let maximum = 1i128
                    .checked_shl(u32::from(sigma))
                    .filter(|&v| v > 0)
                    .ok_or(RangeError::Internal)?;
                Interval(Some((0, maximum)))
            }
            Op::TableMap { table } => match graph.table(table).ok_or(RangeError::Internal)?.kind {
                TableKind::Silu { .. } => Interval(Some((-128, 127))),
                _ => return Err(RangeError::Internal),
            },
        };
        let expected = if matches!(node.op(), Op::SoftmaxRow { .. }) {
            2
        } else {
            1
        };
        if node.outputs().len() != expected {
            return Err(RangeError::Internal);
        }
        for (index, &id) in node.outputs().iter().enumerate() {
            let value = match (node.op(), index) {
                (Op::SoftmaxRow { sigma, .. }, 1) => {
                    let exclusive = 1i128
                        .checked_shl(u32::from(*sigma).saturating_add(1))
                        .filter(|&v| v > 0)
                        .ok_or(RangeError::Internal)?;
                    Interval(Some((
                        0,
                        exclusive.checked_sub(1).ok_or(RangeError::Internal)?,
                    )))
                }
                _ => output,
            };
            put(&mut ranges, id, value)?;
        }
    }
    Ok(ranges)
}

impl Interval {
    fn max_abs(self) -> Option<i128> {
        let (lo, hi) = self.0?;
        Some(lo.checked_abs()?.max(hi.checked_abs()?))
    }

    fn add(self, other: Self) -> Self {
        Self(
            self.0
                .zip(other.0)
                .and_then(|((a, b), (c, d))| Some((a.checked_add(c)?, b.checked_add(d)?))),
        )
    }

    fn product(self, other: Self) -> Self {
        let Some(((a, b), (c, d))) = self.0.zip(other.0) else {
            return Self(None);
        };
        let ends = [
            a.checked_mul(c),
            a.checked_mul(d),
            b.checked_mul(c),
            b.checked_mul(d),
        ];
        let mut lo = i128::MAX;
        let mut hi = i128::MIN;
        for value in ends {
            let Some(value) = value else {
                return Self(None);
            };
            lo = lo.min(value);
            hi = hi.max(value);
        }
        Self(Some((lo, hi)))
    }

    fn scale(self, by: i128) -> Self {
        self.product(Self(Some((by, by))))
    }
}

fn symmetric(maximum: Option<i128>) -> Interval {
    Interval(maximum.and_then(|m| Some((m.checked_neg()?, m))))
}

fn operand(node: &Node, index: usize) -> Result<TensorId, RangeError> {
    node.inputs()
        .get(index)
        .copied()
        .ok_or(RangeError::Internal)
}

fn get(ranges: &[Interval], id: TensorId) -> Result<Interval, RangeError> {
    ranges
        .get(usize::try_from(id.0).map_err(|_| RangeError::Internal)?)
        .copied()
        .ok_or(RangeError::Internal)
}

fn put(ranges: &mut [Interval], id: TensorId, value: Interval) -> Result<(), RangeError> {
    *ranges
        .get_mut(usize::try_from(id.0).map_err(|_| RangeError::Internal)?)
        .ok_or(RangeError::Internal)? = value;
    Ok(())
}

fn producer(graph: &Graph, id: TensorId) -> Option<&Node> {
    graph.node(graph.tensor(id)?.producer?)
}

/// RFC-0006: only a row-broadcast reciprocal from the same SoftmaxRow as the untransposed
/// contraction's exponentials supplies the mass certificate. No shape coincidence is a fact.
fn mass_bound(
    graph: &Graph,
    node: &Node,
    ranges: &[Interval],
) -> Result<Option<Interval>, RangeError> {
    let Some(matmul) = producer(graph, operand(node, 0)?) else {
        return Ok(None);
    };
    let reciprocal = operand(node, 1)?;
    let Some(softmax) = producer(graph, reciprocal) else {
        return Ok(None);
    };
    let Op::SoftmaxRow { sigma, .. } = softmax.op() else {
        return Ok(None);
    };
    if !matches!(
        matmul.op(),
        Op::MatMulAct {
            transpose_rhs: false
        }
    ) || softmax.outputs().first().copied() != matmul.inputs().first().copied()
        || softmax.outputs().get(1).copied() != Some(reciprocal)
    {
        return Ok(None);
    }
    let values = get(ranges, operand(matmul, 1)?)?;
    let mass = 1i128
        .checked_shl(u32::from(*sigma).saturating_mul(2))
        .filter(|&v| v > 0);
    let result = values
        .0
        .zip(mass)
        .and_then(|((lo, hi), m)| Some((lo.min(0).checked_mul(m)?, hi.max(0).checked_mul(m)?)));
    Ok(Some(Interval(result)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering::NodeId;

    #[test]
    fn intervals_match_exhaustive_integer_arithmetic() {
        for lo in -4..=4 {
            for hi in lo..=4 {
                for low in -4..=4 {
                    for high in low..=4 {
                        let a = Interval(Some((lo, hi)));
                        let b = Interval(Some((low, high)));
                        let products: Vec<_> = (lo..=hi)
                            .flat_map(|x| (low..=high).map(move |y| x * y))
                            .collect();
                        assert_eq!(
                            a.product(b).bounds(),
                            Some((
                                *products.iter().min().unwrap(),
                                *products.iter().max().unwrap()
                            ))
                        );
                        assert_eq!(a.add(b).bounds(), Some((lo + low, hi + high)));
                    }
                }
            }
        }
    }

    #[test]
    fn endpoint_overflow_is_unbounded_without_wrapping() {
        let point = |x| Interval(Some((x, x)));
        assert_eq!(point(i128::MAX).add(point(1)), Interval(None));
        assert_eq!(point(i128::MIN).add(point(-1)), Interval(None));
        assert_eq!(point(i128::MIN).product(point(-1)), Interval(None));
        assert_eq!(point(i128::MAX).product(point(2)), Interval(None));
        assert_eq!(point(i128::MIN).product(point(1)), point(i128::MIN));
        assert_eq!(point(i128::MAX).product(point(1)), point(i128::MAX));
        assert_eq!(Interval(None).product(point(1)), Interval(None));
        assert_eq!(symmetric(point(i128::MIN).max_abs()), Interval(None));
        // Isolate each product corner: three products fit, exactly the fourth overflows.
        for (a, b) in [
            ((i128::MIN, 0), (-1, 0)),
            ((i128::MIN, 0), (0, 2)),
            ((0, i128::MAX), (-2, 0)),
            ((0, i128::MAX), (0, 2)),
        ] {
            assert_eq!(Interval(Some(a)).product(Interval(Some(b))), Interval(None));
        }
        assert_eq!(
            Interval(Some((i128::MIN, 0))).add(Interval(Some((-1, 0)))),
            Interval(None)
        );
        assert_eq!(
            Interval(Some((0, i128::MAX))).add(Interval(Some((0, 1)))),
            Interval(None)
        );
    }

    fn graph() -> Graph {
        let file = tv_testkit::read_vectors("claims/claim-lists.json");
        let case = file
            .get("cases")
            .unwrap()
            .as_array()
            .unwrap()
            .first()
            .unwrap();
        let bytes = tv_testkit::hex(case.get("hex").unwrap().as_str().unwrap());
        let claims =
            crate::claims::ClaimList::open(&bytes, &tv_codec::DecodeLimits::MESSAGE).unwrap();
        let public = crate::validation::check_public(&claims).unwrap();
        crate::lowering::lower(&public, 13).unwrap()
    }

    #[test]
    fn the_mass_fact_requires_every_certificate_link() {
        let original = graph();
        let ranges = analyze(&original).unwrap();
        let (index, node) = original
            .nodes()
            .iter()
            .enumerate()
            .find(|(_, node)| {
                matches!(
                    node.op(),
                    Op::Mul {
                        broadcast: Broadcast::Trailing
                    }
                ) && mass_bound(&original, node, &ranges).unwrap().is_some()
            })
            .unwrap();
        let id = NodeId(index as u32);
        let contraction_id = original.tensor(node.inputs()[0]).unwrap().producer.unwrap();
        let reciprocal = node.inputs()[1];
        let softmax_id = original.tensor(reciprocal).unwrap().producer.unwrap();
        let exponential = original.node(softmax_id).unwrap().outputs()[0];
        for op in [
            Op::MatMulAct {
                transpose_rhs: true,
            },
            Op::Add,
        ] {
            let mut edited = original.clone();
            edited.set_op(contraction_id, op);
            assert_eq!(
                mass_bound(&edited, edited.node(id).unwrap(), &ranges),
                Ok(None)
            );
        }
        let mut edited = original.clone();
        edited.set_input(contraction_id, 0, reciprocal);
        assert_eq!(
            mass_bound(&edited, edited.node(id).unwrap(), &ranges),
            Ok(None)
        );
        let mut edited = original.clone();
        edited.set_input(id, 1, exponential);
        assert_eq!(
            mass_bound(&edited, edited.node(id).unwrap(), &ranges),
            Ok(None)
        );
        let mut edited = original.clone();
        edited.set_op(
            softmax_id,
            Op::MatMulAct {
                transpose_rhs: false,
            },
        );
        assert_eq!(
            mass_bound(&edited, edited.node(id).unwrap(), &ranges),
            Ok(None)
        );
        let mut edited = original.clone();
        edited.set_op(
            id,
            Op::Mul {
                broadcast: Broadcast::Leading,
            },
        );
        let ordinary = get(&ranges, node.inputs()[0])
            .unwrap()
            .product(get(&ranges, reciprocal).unwrap());
        let out = node.outputs()[0];
        assert_eq!(get(&analyze(&edited).unwrap(), out), Ok(ordinary));
    }
}
