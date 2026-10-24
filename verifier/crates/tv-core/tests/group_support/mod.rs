// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Oracles for RFC-0054's token axes and group layouts, written from RFC-0006's operation table
//! and RFC-0045's index order, not from the verifier's derivation.
//!
//! - [`tags`] follows every element of every tensor: an output element belongs to the token whose
//!   row it is computed from (RFC-0006: `MatMulWeight` maps rows, `MatMulAct` and `SoftmaxRow`
//!   keep their first operand's rows, the elementwise operations keep each element's). A
//!   tensor's token axis is then the axis whose index equals every element's token.
//! - [`domain`] names the tensor a site's request runs over, from the gadget alone.

#![allow(dead_code)]

use tv_core::claims::{ClaimList, GadgetPolicy};
use tv_core::lowering::{Broadcast, Graph, NodeId, Op, Role, TensorId, lower};
use tv_core::validation::check_public;
use tv_field::{Fp, Fp2};
use tv_testkit::SplitMix64;

// The test crate's root declares `claim_support` beside this module.
pub use crate::claim_support::{golden, with_layers};

/// A conformance model's graph at `n` tokens, padded to `layers` layers when given.
pub fn model(name: &str, layers: Option<u32>, n: u32) -> (Graph, GadgetPolicy) {
    let mut claims: ClaimList = golden(name);
    if let Some(layers) = layers {
        claims = with_layers(&claims, layers);
    }
    let public = check_public(&claims).unwrap();
    (lower(&public, n).unwrap(), claims.gadget_policy)
}

/// `⌈log2 n⌉`.
pub fn bits(n: u32) -> u32 {
    u32::BITS - (n - 1).leading_zeros()
}

fn strides(dims: &[u32]) -> Vec<usize> {
    let mut out = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        out[i] = out[i + 1] * dims[i + 1] as usize;
    }
    out
}
fn unravel(mut flat: usize, dims: &[u32]) -> Vec<usize> {
    let mut out = vec![0; dims.len()];
    for i in (0..dims.len()).rev() {
        out[i] = flat % dims[i] as usize;
        flat /= dims[i] as usize;
    }
    out
}
fn ravel(index: &[usize], dims: &[u32]) -> usize {
    index.iter().zip(strides(dims)).map(|(&i, s)| i * s).sum()
}
fn size(dims: &[u32]) -> usize {
    dims.iter().map(|&d| d as usize).product()
}

/// The one token every element of a row shares, `None` for a row of no token.
fn row_tag(row: &[Option<u32>]) -> Option<u32> {
    let first = row[0];
    assert!(row.iter().all(|&t| t == first), "a row mixes tokens");
    first
}

/// Every element's token, by tensor id and row-major element index.
pub fn tags(graph: &Graph) -> Vec<Vec<Option<u32>>> {
    let dims = |t: TensorId| graph.tensor(t).unwrap().shape.dims().to_vec();
    let mut out: Vec<Option<Vec<Option<u32>>>> = vec![None; graph.tensors().len()];
    for input in graph.inputs() {
        let d = dims(input.tensor);
        out[input.tensor.0 as usize] = Some(match input.role {
            Role::Tokens | Role::Positions => (0..d[0]).map(Some).collect(),
            _ => vec![None; size(&d)],
        });
    }
    for node in graph.nodes() {
        let get = |t: TensorId| out[t.0 as usize].clone().expect("derived before use");
        let x = node.inputs()[0];
        let (xd, xt) = (dims(x), get(x));
        let y = node.outputs()[0];
        let yd = dims(y);
        let width = |d: &[u32]| *d.last().unwrap() as usize;
        let result: Vec<Option<u32>> = match *node.op() {
            Op::OneHot { .. } => (0..size(&yd)).map(|f| xt[f / width(&yd)]).collect(),
            Op::MatMulWeight { .. } => {
                let w = width(&xd);
                let rows: Vec<_> = xt.chunks(w).map(row_tag).collect();
                assert!(get(node.inputs()[1]).iter().all(Option::is_none));
                (0..size(&yd)).map(|f| rows[f / width(&yd)]).collect()
            }
            Op::ConstMul { .. }
            | Op::Requant { .. }
            | Op::TableMap { .. }
            | Op::RsqrtRow { .. }
            | Op::Rotate { .. }
            | Op::Reshape => xt.clone(),
            Op::Add => {
                assert_eq!(xt, get(node.inputs()[1]), "Add of two tokens");
                xt.clone()
            }
            Op::Mul { broadcast } => {
                let b = get(node.inputs()[1]);
                match broadcast {
                    Broadcast::Trailing => {
                        let per = size(&xd) / b.len();
                        for (f, &t) in xt.iter().enumerate() {
                            assert_eq!(b[f / per], t, "Mul's rows disagree");
                        }
                    }
                    Broadcast::Leading => assert!(b.iter().all(Option::is_none)),
                }
                xt.clone()
            }
            Op::Transpose { perm } => (0..size(&yd))
                .map(|f| {
                    let yi = unravel(f, &yd);
                    let mut xi = vec![0; xd.len()];
                    for (i, &p) in perm.iter().enumerate() {
                        xi[p as usize] = yi[i];
                    }
                    xt[ravel(&xi, &xd)]
                })
                .collect(),
            Op::Expand { axis, .. } => (0..size(&yd))
                .map(|f| {
                    let mut yi = unravel(f, &yd);
                    yi.remove(axis as usize);
                    xt[ravel(&yi, &xd)]
                })
                .collect(),
            Op::SumSquares { axis } => (0..size(&yd))
                .map(|f| {
                    let yi = unravel(f, &yd);
                    let row: Vec<_> = (0..xd[axis as usize] as usize)
                        .map(|j| {
                            let mut xi = yi.clone();
                            xi.insert(axis as usize, j);
                            xt[ravel(&xi, &xd)]
                        })
                        .collect();
                    row_tag(&row)
                })
                .collect(),
            Op::MatMulAct { .. } => {
                let rows: Vec<_> = xt.chunks(width(&xd)).map(row_tag).collect();
                (0..size(&yd)).map(|f| rows[f / width(&yd)]).collect()
            }
            Op::SoftmaxRow { .. } => {
                let rows: Vec<_> = xt.chunks(width(&xd)).map(row_tag).collect();
                let r = node.outputs()[1];
                out[r.0 as usize] = Some(rows);
                xt.clone()
            }
        };
        out[y.0 as usize] = Some(result);
    }
    out.into_iter().map(|t| t.expect("every tensor")).collect()
}

/// The axes whose index equals each element's token: empty when the tensor holds no token.
pub fn candidates(graph: &Graph, tags: &[Vec<Option<u32>>], tensor: TensorId) -> Vec<u8> {
    let dims = graph.tensor(tensor).unwrap().shape.dims().to_vec();
    let tags = &tags[tensor.0 as usize];
    if tags.iter().all(Option::is_none) {
        return vec![];
    }
    let found: Vec<u8> = (0..dims.len())
        .filter(|&a| {
            tags.iter()
                .enumerate()
                .all(|(f, &t)| t == Some(unravel(f, &dims)[a] as u32))
        })
        .map(|a| a as u8)
        .collect();
    assert!(
        !found.is_empty(),
        "tensor {tensor:?} holds tokens on no axis"
    );
    found
}

/// The tensor request `lookup` of the gadget at `node` runs over (RFC-0009, RFC-0045 'Blocks'):
/// a `Requant`'s or `TableMap`'s input, a `SoftmaxRow`'s scores for its `Exp` lookup and its
/// reciprocals for the others, an `RsqrtRow`'s root.
pub fn domain(graph: &Graph, node: NodeId, lookup: usize) -> TensorId {
    let n = graph.node(node).unwrap();
    match n.op() {
        Op::Requant { .. } | Op::TableMap { .. } => n.inputs()[0],
        Op::SoftmaxRow { .. } if lookup == 0 => n.inputs()[0],
        Op::SoftmaxRow { .. } => n.outputs()[1],
        Op::RsqrtRow { .. } => n.outputs()[0],
        other => panic!("no lookup domain at {other:?}"),
    }
}

/// A block entry's index along each axis of `widths` variables: per-dimension padded, the last
/// axis lowest (RFC-0011 rule 1, RFC-0045 'Blocks').
pub fn entry_axes(entry: u64, widths: &[u32]) -> Vec<u64> {
    let mut out = vec![0; widths.len()];
    let mut shift = 0;
    for i in (0..widths.len()).rev() {
        out[i] = (entry >> shift) & ((1u64 << widths[i]) - 1);
        shift += widths[i];
    }
    out
}
/// The inverse of [`entry_axes`] over per-axis widths `widths`.
pub fn entry_of(axes: &[u64], widths: &[u32]) -> u64 {
    let mut entry = 0;
    let mut shift = 0;
    for i in (0..widths.len()).rev() {
        entry |= axes[i] << shift;
        shift += widths[i];
    }
    entry
}

pub fn f(x: u64) -> Fp {
    Fp::reduce(x)
}
pub fn e(x: u64) -> Fp2 {
    Fp2::from(f(x))
}
pub fn random(rng: &mut SplitMix64) -> Fp2 {
    Fp2::new(f(rng.next_u64()), f(rng.next_u64()))
}
pub fn eq(point: &[Fp2], vertex: u64) -> Fp2 {
    point
        .iter()
        .enumerate()
        .fold(Fp2::ONE, |weight, (bit, &r)| {
            weight
                * if vertex >> bit & 1 == 0 {
                    Fp2::ONE - r
                } else {
                    r
                }
        })
}
/// `f̃(point)` for a table of `2^{point.len()}` values, its lowest variable the index's lowest bit.
pub fn mle(values: &[Fp2], point: &[Fp2]) -> Fp2 {
    assert_eq!(values.len(), 1 << point.len());
    values
        .iter()
        .enumerate()
        .fold(Fp2::ZERO, |sum, (row, &v)| sum + v * eq(point, row as u64))
}
