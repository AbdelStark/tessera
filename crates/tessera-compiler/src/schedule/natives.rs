// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The native relations the CPU prover composes into stage 1 (RFC-0050, RFC-0009, RFC-0053):
//! each root's coefficient products and padding checks, each `SoftmaxRow`'s key, forest and
//! reciprocal relations and padding checks, and a request's token and greedy relations. They are
//! public geometry, derived here from the lowering alone so that the commitment report can size
//! them (RFC-0056 'Working sets'). The prover's `ModelNativePlan` derives the same descriptors;
//! its tests hold the two equal.

use tessera_ir::{Op, TensorId};

use super::{Deferred, LayerMap, Native, NativeOutput};
use crate::{CompileError, Lowering};

/// What a native relation proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeKind {
    /// One coefficient product of a root's bracket polynomial, of degree 4.
    Product,
    /// A root's padding zero-check.
    RootPad,
    /// A root's residual padding zero-check.
    ResidualPad,
    /// `SoftmaxRow`'s lookup key claim, a sum over the score tensor.
    Key,
    /// `SoftmaxRow`'s forest leaf claim, a sum over the score tensor.
    Forest,
    /// `SoftmaxRow`'s reciprocal.
    Reciprocal,
    /// `SoftmaxRow`'s row-maximum padding zero-check.
    MaxPad,
    /// `SoftmaxRow`'s reciprocal padding zero-check.
    ReciprocalPad,
    /// A request's tokens against the embedding's one-hot vectors (RFC-0053).
    Tokens,
    /// A request's greedy keys over the logits, when it has outputs (RFC-0053).
    GreedyKey,
    /// A request's selected logits, when it has outputs (RFC-0053).
    GreedySelected,
    /// The logits of a request without outputs, observed (RFC-0053).
    Observe,
}

/// Which relations of a request join the model's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Request {
    /// The model's relations alone, as `ModelNativePlan::derive` composes them.
    None,
    /// A request without outputs: its tokens, and its logits observed.
    Prefill,
    /// A request with outputs: its tokens, and the greedy certificate over its logits.
    Generate,
}

/// A lowering's native relations, in tag order, with what each proves and the graph sources
/// they replace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelNatives {
    /// The descriptors, tags from 1.
    pub native: Vec<Native>,
    /// What each descriptor proves, in the same order: each one's [`Native::kind`].
    pub kinds: Vec<NativeKind>,
    /// The graph sources the descriptors replace.
    pub deferred: Vec<Deferred>,
}

/// The variables of a tensor of `shape`, each axis padded to a power of two.
fn vars(shape: &[u32]) -> u32 {
    shape.iter().map(|&n| super::axis_vars(n)).sum()
}

/// The native relations of `lowering`'s roots and softmax rows, in the order of their sites:
/// node order, a site in the layers at its first layer's node; then `request`'s, as
/// `ModelNativePlan::derive_runtime` appends them.
///
/// # Errors
///
/// [`CompileError::Schedule`] for a root whose input is not a sum of squares or whose range is
/// unbounded, logits that are not `[positions, vocabulary]`, or a graph the layer map refuses.
pub fn model_natives(lowering: &Lowering, request: Request) -> Result<ModelNatives, CompileError> {
    let active = match request {
        Request::None => None,
        Request::Prefill => Some(0),
        Request::Generate => Some(1),
    };
    model_natives_inner(lowering, active)
}

/// The native relations of `lowering` and a multi-request batch with `active_output_requests`
/// requests whose `output_len ≥ 1` (RFC-0071 'The greedy certificate and scoring'): token binding,
/// then `active_output_requests` greedy key relations followed by one `GreedySelected`, or token
/// binding then `Observe` when `active_output_requests == 0`.
///
/// # Errors
///
/// As [`model_natives`].
pub fn model_natives_batch(
    lowering: &Lowering,
    active_output_requests: usize,
) -> Result<ModelNatives, CompileError> {
    model_natives_inner(lowering, Some(active_output_requests))
}

fn model_natives_inner(
    lowering: &Lowering,
    active_output_requests: Option<usize>,
) -> Result<ModelNatives, CompileError> {
    let bad = |rule| CompileError::Schedule { node: None, rule };
    let layers = LayerMap::new(lowering)?;
    let graph = &lowering.graph;
    let layer0 = layers.first..layers.first + layers.width;
    let in_layers = layers.first..layers.first + layers.width * layers.count;
    let shape = |t: TensorId| graph.tensors[t.0 as usize].shape.as_slice();
    let mut out = ModelNatives {
        native: Vec::new(),
        kinds: Vec::new(),
        deferred: Vec::new(),
    };
    for node in &graph.nodes {
        let id = node.id.0;
        // A site in the layers is its first layer's node; later layers repeat it.
        let lambda = match (layer0.contains(&id), in_layers.contains(&id)) {
            (true, _) => layers.vars,
            (false, true) => continue,
            (false, false) => 0,
        };
        let mut add = |kind, local: u32, degree, evaluations, zero, outputs: Vec<NativeOutput>| {
            let tag = u32::try_from(out.native.len() + 1).unwrap_or(u32::MAX);
            out.native.push(Native {
                tag,
                kind,
                node: node.id,
                vars: local + lambda,
                degree,
                evaluations,
                zero,
                outputs,
            });
            out.kinds.push(kind);
        };
        let output = |tensor, vars| NativeOutput {
            tensor,
            from: 0,
            vars,
        };
        match node.op {
            Op::RsqrtRow {
                dim,
                kappa,
                eps_int,
            } => {
                let width = crate::range::root_bits(kappa, eps_int, dim)
                    .ok_or(bad("a root's range is bounded"))?;
                let width = usize::try_from(width).unwrap_or(usize::MAX);
                let (digits, coefficients) = (width.div_ceil(8), (2 * width + 55).div_ceil(8));
                let sum = graph
                    .producer(node.inputs[0])
                    .map(|p| &graph.nodes[p.0 as usize])
                    .filter(|p| matches!(p.op, Op::SumSquares { .. }))
                    .ok_or(bad("a root reads a sum of squares"))?;
                let local = vars(shape(node.outputs[0]));
                for _ in 0..2 * coefficients {
                    add(
                        NativeKind::Product,
                        local,
                        4,
                        digits + 4,
                        false,
                        vec![
                            output(node.inputs[0], local),
                            output(node.outputs[0], local),
                        ],
                    );
                }
                add(
                    NativeKind::RootPad,
                    local,
                    3,
                    1,
                    true,
                    vec![output(node.outputs[0], local)],
                );
                let residual = sum.inputs[0];
                let rvars = vars(shape(residual));
                add(
                    NativeKind::ResidualPad,
                    rvars,
                    3,
                    1,
                    true,
                    vec![output(residual, rvars)],
                );
            }
            Op::SoftmaxRow { .. } => {
                let local = super::tensor_vars(graph, node.inputs[0]);
                for kind in [NativeKind::Key, NativeKind::Forest] {
                    add(
                        kind,
                        local,
                        3,
                        2,
                        false,
                        vec![output(node.inputs[0], local)],
                    );
                }
                let row = vars(shape(node.outputs[1]));
                add(
                    NativeKind::Reciprocal,
                    row,
                    3,
                    2,
                    false,
                    vec![output(node.outputs[1], row)],
                );
                add(NativeKind::MaxPad, row, 3, 1, true, Vec::new());
                add(
                    NativeKind::ReciprocalPad,
                    row,
                    3,
                    1,
                    true,
                    vec![output(node.outputs[1], row)],
                );
                out.deferred.push((node.id, Some(0)));
            }
            _ => {}
        }
    }
    if let Some(active) = active_output_requests {
        let h = graph
            .producer(lowering.one_hot)
            .ok_or(bad("the one-hot vectors have a producer"))?;
        let l = graph
            .producer(lowering.logits)
            .ok_or(bad("the logits have a producer"))?;
        let &[n, v] = shape(lowering.logits) else {
            return Err(bad("the logits are positions by vocabulary"));
        };
        let (nt, nv) = (vars(&[n]), vars(&[v]));
        let mut add = |kind, node, degree, evaluations, zero, outputs: Vec<NativeOutput>| {
            let tag = u32::try_from(out.native.len() + 1).unwrap_or(u32::MAX);
            out.native.push(Native {
                tag,
                kind,
                node,
                vars: nt + nv,
                degree,
                evaluations,
                zero,
                outputs,
            });
            out.kinds.push(kind);
        };
        add(
            NativeKind::Tokens,
            h,
            2,
            2,
            true,
            vec![
                NativeOutput {
                    tensor: lowering.one_hot,
                    from: 0,
                    vars: nv + nt,
                },
                NativeOutput {
                    tensor: lowering.tokens,
                    from: nv,
                    vars: nt,
                },
            ],
        );
        out.deferred.push((h, Some(0)));
        let logits = vec![NativeOutput {
            tensor: lowering.logits,
            from: 0,
            vars: nv + nt,
        }];
        let greedy = nt as usize + 3;
        if active == 0 {
            add(NativeKind::Observe, l, 1, 1, true, logits);
        } else {
            for _ in 0..active {
                add(NativeKind::GreedyKey, l, 2, greedy, false, logits.clone());
            }
            add(NativeKind::GreedySelected, l, 3, greedy, true, logits);
        }
        out.deferred.push((l, None));
    }
    Ok(out)
}
