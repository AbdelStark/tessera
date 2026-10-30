// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0072 'Padded positions' in the stage tables (#526's fourth slice): every tensor's padded
//! positions hold what its operation computes from its inputs' padded positions, which
//! [`PaddedWitness`] computes over the engine's witness.
//!
//! Each case runs one operation on the S-box `s = AesSbox(x)` of committed bytes `x` of `[4, 3]`,
//! whose padded column holds `AesSbox(0) = 0x63`, and requantizes the result at a leaf site. The
//! stage prover proves it over the padded witness, and the verifier of [`super`] accepts it, each
//! public factor its own: a `ConstAdd`'s shift `ĉ`, a selection's kernel, an `Expand`'s `LT`, a
//! per-channel constant 0 at padded channels, RoPE's kernel 0 on padded rows. Its pool is the
//! committed tensors' extensions over their padded tables. Every witness here differs from the
//! zero-padded one at padded positions, so a public factor that matters only there, such as
//! `ψ̃` or `LT`, is pinned (the soundness-review skill's 'Evidence', #561): a padded table that
//! the operation does not compute there breaks the relation that reads it.
//!
//! The padded tables are the witness's at real positions. A sum over a padded axis that the
//! S-box reaches, which RFC-0072 'Summing relations' refuses, breaks that, and so does a
//! `Reshape` that does not map the padded cube as the identity; a `OneHot` that reads the S-box
//! is refused, its row relations taking a padded row's input as 0 (RFC-0035).

use super::*;
use tessera_ir::{BoolFn, Broadcast};

/// The tables: the S-box (0), `Gf256Mul { 2 }` (1), `Bitwise8 { Xor }` (2), a requantization
/// table non-zero at padding, `Requant { b: 9, [1], 1, 255 }` (3), with `rq(0) = 1`, and RoPE (4).
fn tables() -> Vec<TableSpec> {
    let lookup = |id, kind| {
        GeneratedTable::new(TableId(id), kind)
            .expect("a table")
            .spec()
            .clone()
    };
    let rope = RopeTable::new(
        TableId(4),
        TableKind::Rope {
            theta: 10_000,
            factor: 1,
            low_freq_factor: 1,
            high_freq_factor: 4,
            original_max: 64,
            bits: 14,
            max_context: 16,
            d_head: 4,
        },
    )
    .expect("a RoPE table");
    vec![
        lookup(0, TableKind::AesSbox),
        lookup(1, TableKind::Gf256Mul { c: 2 }),
        lookup(
            2,
            TableKind::Bitwise8 {
                op: tessera_tables::BitOp::Xor,
            },
        ),
        lookup(
            3,
            TableKind::Requant {
                b: 9,
                r_by_class: vec![1],
                lo: 1,
                hi: 255,
            },
        ),
        rope.spec().clone(),
    ]
}

/// Committed bytes of `shape`.
fn bytes(g: &mut GraphBuilder, shape: &[u32]) -> TensorId {
    let mut input = decl(shape, ADVICE);
    input.range = IntRange { lo: 0, hi: 255 };
    g.input(input).expect("bytes")
}

/// `table(x)` elementwise, a lookup output of `shape`.
fn map(g: &mut GraphBuilder, table: u32, x: TensorId, shape: &[u32]) -> TensorId {
    let output = decl(
        shape,
        NodeKind::LookupOutput {
            table: TableId(table),
        },
    );
    g.node(
        Op::TableMap {
            table: TableId(table),
        },
        &[x],
        vec![output],
    )
    .expect("a lookup")[0]
}

fn virt(shape: &[u32]) -> TensorDecl {
    decl(shape, NodeKind::Virtual)
}

/// A case: what it builds on the S-box `s` of `[4, 3]` and the other inputs it adds, returning
/// the tensor requantized at the leaf site and its shape.
type Build = fn(&mut GraphBuilder, TensorId, &mut Vec<TensorId>) -> (TensorId, Vec<u32>);

/// The graph of a case: `x` committed bytes `[4, 3]`, `s = AesSbox(x)`, the case's tensor `y`,
/// and `requant(y)`; with its random inputs.
fn case(build: Build) -> (Graph, Vec<TensorId>) {
    let mut g = GraphBuilder::new();
    let x = bytes(&mut g, &[4, 3]);
    let s = map(&mut g, 0, x, &[4, 3]);
    let mut inputs = vec![x];
    let (y, shape) = build(&mut g, s, &mut inputs);
    requant(&mut g, y, &shape, 1 << 28);
    (g.finish(), inputs)
}

/// `tensor`'s committed table over its padded shape: the padded witness's, or its values padded
/// with zeros.
fn table_of(
    graph: &Graph,
    padded: &PaddedWitness<'_, Witness>,
    tensor: TensorId,
) -> Vec<Goldilocks> {
    match padded.padded(tensor) {
        Some(values) => values.iter().map(|&v| Goldilocks::from_i64(v)).collect(),
        None => {
            let shape = &graph.tensors[tensor.0 as usize].shape;
            let vars: u32 = shape.iter().map(|&d| axis_vars(d)).sum();
            let mut table = vec![Goldilocks::ZERO; 1 << vars];
            let values = padded.values(tensor).expect("a tensor");
            for (flat, &v) in values.iter().enumerate() {
                let (mut rest, mut index, mut shift) = (flat, 0usize, 0);
                for &len in shape.iter().rev() {
                    index |= (rest % len as usize) << shift;
                    rest /= len as usize;
                    shift += axis_vars(len);
                }
                table[index] = Goldilocks::from_i64(v);
            }
            table
        }
    }
}

/// The verifier's public data (RFC-0072 'Public tensors'): the public graph inputs it holds, and
/// every other public tensor computed from them alone, through the nodes that are not
/// gadget-backed, by this test's own evaluator over the real shapes: never from a prover's
/// witness, nor by the CPU engine, which the prover shares. It holds no other tensor.
pub(super) struct PublicOf {
    tensors: Vec<Option<Vec<i64>>>,
}

impl PublicOf {
    /// The public tensors of `graph` from the values `inputs` gives its public inputs; the values
    /// of any other input among `inputs` are not read.
    pub(super) fn new(graph: &Graph, inputs: &[(TensorId, Vec<i64>)]) -> Self {
        let public = public_tensors(graph);
        let mut tensors: Vec<Option<Vec<i64>>> = vec![None; graph.tensors.len()];
        for (t, values) in inputs {
            if public[t.0 as usize] {
                tensors[t.0 as usize] = Some(values.clone());
            }
        }
        for node in &graph.nodes {
            if !node.outputs.iter().all(|t| public[t.0 as usize]) || node.outputs.is_empty() {
                continue;
            }
            let shape = |t: TensorId| graph.tensors[t.0 as usize].shape.clone();
            let ins: Vec<&[i64]> = node
                .inputs
                .iter()
                .map(|t| tensors[t.0 as usize].as_deref().expect("a public input"))
                .collect();
            let y = public_node(
                graph,
                node,
                &ins,
                &shape(node.inputs[0]),
                &shape(node.outputs[0]),
            );
            tensors[node.outputs[0].0 as usize] = Some(y);
        }
        Self { tensors }
    }
}

impl WitnessSource for PublicOf {
    fn values(&self, tensor: TensorId) -> Option<&[i64]> {
        self.tensors.get(tensor.0 as usize)?.as_deref()
    }
}

/// The coordinates of row-major index `flat` of `shape`.
fn coordinates(shape: &[u32], flat: usize) -> Vec<usize> {
    let mut rest = flat;
    let mut out = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        out[axis] = rest % shape[axis] as usize;
        rest /= shape[axis] as usize;
    }
    out
}

/// The row-major index of `coords` in `shape`.
fn flat_of(shape: &[u32], coords: &[usize]) -> usize {
    coords
        .iter()
        .zip(shape)
        .fold(0, |index, (&c, &len)| index * len as usize + c)
}

/// A public node's output over its real shape `y`, from its inputs' values, the first of shape
/// `x`: RFC-0006's semantics of the nodes that are not gadget-backed, written here.
fn public_node(
    graph: &Graph,
    node: &tessera_ir::Node,
    ins: &[&[i64]],
    x: &[u32],
    y: &[u32],
) -> Vec<i64> {
    let count: usize = y.iter().map(|&d| d as usize).product();
    let constant = |c: tessera_ir::ConstRef| graph.constants[c.0 as usize].clone();
    (0..count)
        .map(|i| {
            let at = coordinates(y, i);
            match &node.op {
                Op::Reshape => ins[0][i],
                Op::Add => ins[0][i] + ins[1][i],
                Op::Sub => ins[0][i] - ins[1][i],
                Op::Mul { broadcast } => {
                    let b = ins[1];
                    ins[0][i]
                        * match broadcast {
                            Broadcast::Trailing => b[i / (count / b.len())],
                            Broadcast::Leading => b[i % b.len()],
                        }
                }
                Op::ConstMul { c } | Op::ConstAdd { c } => {
                    let c = constant(*c);
                    let k = if c.len() == 1 {
                        c[0]
                    } else {
                        c[at[at.len() - 1]]
                    };
                    match node.op {
                        Op::ConstMul { .. } => ins[0][i] * k,
                        _ => ins[0][i] + k,
                    }
                }
                Op::Transpose { perm } => {
                    let mut from = vec![0; x.len()];
                    for (k, &axis) in perm.iter().enumerate() {
                        from[usize::from(axis)] = at[k];
                    }
                    ins[0][flat_of(x, &from)]
                }
                Op::Slice { axis, start, .. } => {
                    let mut from = at.clone();
                    from[usize::from(*axis)] += *start as usize;
                    ins[0][flat_of(x, &from)]
                }
                Op::GatherPublic { index } => {
                    let mut from = at.clone();
                    from[0] = usize::try_from(constant(*index)[at[0]]).expect("a row");
                    ins[0][flat_of(x, &from)]
                }
                Op::Expand { axis, .. } => {
                    let mut from = at.clone();
                    from.remove(usize::from(*axis));
                    ins[0][flat_of(x, &from)]
                }
                Op::Sum { axis } | Op::SumSquares { axis } => {
                    let axis = usize::from(*axis);
                    (0..x[axis] as usize)
                        .map(|k| {
                            let mut from = at.clone();
                            from.insert(axis, k);
                            let v = ins[0][flat_of(x, &from)];
                            if matches!(node.op, Op::SumSquares { .. }) {
                                v * v
                            } else {
                                v
                            }
                        })
                        .sum()
                }
                Op::BitFn { f } => {
                    let bits: Vec<i64> = ins.iter().map(|v| v[i]).collect();
                    f.eval(&bits)
                }
                op => panic!("this test's public evaluator knows no {}", op.name()),
            }
        })
        .collect()
}

/// The verifier's verdict on a case's honest proof over the padded witness, and whether its pool
/// is the committed tensors' extensions over their padded tables. The inputs `inputs` take random
/// values in their ranges, those of `given` the values given.
fn proves(
    name: &str,
    graph: &Graph,
    inputs: &[TensorId],
    given: &[(TensorId, Vec<i64>)],
    seed: u64,
) {
    let tables = tables();
    let schedule = derive_graph_schedule(graph, &tables, &[], None)
        .unwrap_or_else(|e| panic!("{name}: a schedule, not {e:?}"));
    let mut values = random_inputs(graph, inputs, seed);
    values.extend(
        given
            .iter()
            .map(|(t, v)| (*t, v.iter().copied().collect::<SecretBuf<i64>>())),
    );
    let witness = Engine::for_graph(graph, &tables)
        .expect("an engine")
        .evaluate(values)
        .unwrap_or_else(|e| panic!("{name}: a witness, not {e:?}"));
    let padded = PaddedWitness::new(graph, &tables, &witness)
        .unwrap_or_else(|e| panic!("{name}: the padded tables, not {e:?}"));
    // The S-box's padded column holds 0x63: the witness differs from the zero-padded one there.
    let s = graph.nodes[0].outputs[0];
    assert_eq!(
        padded.padded(s).expect("the S-box's table")[3],
        0x63,
        "{name}"
    );
    let points = leaf_points(&schedule, seed);
    let (proof, prover_pool) = prove_graph(graph, &tables, &schedule, &padded, &points);
    let statement = PublicOf::new(graph, given);
    let publics = Publics {
        values: &statement,
        shift: Shift::Extension,
        kernel: Kernel::Closed,
    };
    let pool = verify_public(graph, &tables, &schedule, &proof, &points, publics)
        .unwrap_or_else(|r| panic!("{name}: the honest proof verifies, not {r:?}"));
    let prover_pool: Vec<F> = prover_pool
        .into_iter()
        .map(|v| v.expect("a value"))
        .collect();
    assert_eq!(pool, prover_pool, "{name}: the prover's pool");
    let coords = coords_after(&schedule, &proof, &points);
    let opened = pool_of_tables(&coords, graph, &schedule, &|t| table_of(graph, &padded, t));
    assert_eq!(
        pool, opened,
        "{name}: the pool is the committed padded tables' extensions"
    );
}

/// RFC-0072 'Padded positions': each operation downstream of the S-box holds what it computes
/// at its padded positions, and proves and verifies over the padded witness: `Add`, `Sub`, `Mul`
/// broadcast both ways, `ConstMul` and `ConstAdd` by a scalar and per channel, `Reshape`,
/// `Transpose` of two axes and of three by a cycle, `Slice` on both axes and from the first row,
/// `GatherPublic`, `Expand`, `Sum` and `SumSquares` over the axis of 4, both matmuls, `Rotate`, a
/// `BitDecompose` and a `BitFn` of its bits, an `AssertEqual` whose sides differ at padded
/// positions alone, and the lookups `TableMap`, `Bitwise` and `Requant`, of one class and of
/// two, below it.
#[test]
fn every_operation_holds_what_it_computes_at_padded_positions() {
    let shape = |s: &[u32]| s.to_vec();
    let cases: Vec<(&str, Build)> = vec![
        ("Add", |g, s, inputs| {
            let x = bytes(g, &[4, 3]);
            inputs.push(x);
            let t = map(g, 1, x, &[4, 3]);
            (one(g, Op::Add, &[s, t], virt(&[4, 3])), vec![4, 3])
        }),
        ("Sub", |g, s, inputs| {
            let x = bytes(g, &[4, 3]);
            inputs.push(x);
            let t = map(g, 1, x, &[4, 3]);
            (one(g, Op::Sub, &[s, t], virt(&[4, 3])), vec![4, 3])
        }),
        ("Mul by rows", |g, s, inputs| {
            let w = bytes(g, &[4]);
            inputs.push(w);
            let op = Op::Mul {
                broadcast: Broadcast::Trailing,
            };
            (one(g, op, &[s, w], virt(&[4, 3])), vec![4, 3])
        }),
        ("Mul by channels", |g, s, inputs| {
            let c = bytes(g, &[3]);
            inputs.push(c);
            let op = Op::Mul {
                broadcast: Broadcast::Leading,
            };
            (one(g, op, &[s, c], virt(&[4, 3])), vec![4, 3])
        }),
        ("ConstMul", |g, s, _| {
            let c = g.constant(vec![3]).expect("a constant");
            (one(g, Op::ConstMul { c }, &[s], virt(&[4, 3])), vec![4, 3])
        }),
        ("ConstMul per channel", |g, s, _| {
            let c = g.constant(vec![1, 2, 3]).expect("a constant");
            (one(g, Op::ConstMul { c }, &[s], virt(&[4, 3])), vec![4, 3])
        }),
        ("ConstAdd", |g, s, _| {
            let c = g.constant(vec![5]).expect("a constant");
            (one(g, Op::ConstAdd { c }, &[s], virt(&[4, 3])), vec![4, 3])
        }),
        ("ConstAdd per channel", |g, s, _| {
            let c = g.constant(vec![7, -2, 9]).expect("a constant");
            (one(g, Op::ConstAdd { c }, &[s], virt(&[4, 3])), vec![4, 3])
        }),
        ("Reshape", |g, s, _| {
            (one(g, Op::Reshape, &[s], virt(&[2, 2, 3])), vec![2, 2, 3])
        }),
        ("Transpose", |g, s, _| {
            let op = Op::Transpose { perm: vec![1, 0] };
            (one(g, op, &[s], virt(&[3, 4])), vec![3, 4])
        }),
        ("Transpose of three axes", |g, s, _| {
            // A 3-cycle, not its own inverse: `y[i, j, k] = x[k, i, j]`, `x` padded on two axes.
            let wide = one(g, Op::Expand { axis: 0, len: 3 }, &[s], virt(&[3, 4, 3]));
            let op = Op::Transpose {
                perm: vec![1, 2, 0],
            };
            (one(g, op, &[wide], virt(&[4, 3, 3])), vec![4, 3, 3])
        }),
        ("Slice of rows", |g, s, _| {
            let op = Op::Slice {
                axis: 0,
                start: 1,
                len: 3,
            };
            (one(g, op, &[s], virt(&[3, 3])), vec![3, 3])
        }),
        ("Slice of rows from the first", |g, s, _| {
            // Its padded row would select `s`'s last row, were it not 0.
            let op = Op::Slice {
                axis: 0,
                start: 0,
                len: 3,
            };
            (one(g, op, &[s], virt(&[3, 3])), vec![3, 3])
        }),
        ("Slice of channels", |g, s, _| {
            let op = Op::Slice {
                axis: 1,
                start: 1,
                len: 2,
            };
            (one(g, op, &[s], virt(&[4, 2])), vec![4, 2])
        }),
        ("GatherPublic", |g, s, _| {
            let index = g.constant(vec![3, 0, 2]).expect("rows");
            (
                one(g, Op::GatherPublic { index }, &[s], virt(&[3, 3])),
                vec![3, 3],
            )
        }),
        ("Expand", |g, s, _| {
            let op = Op::Expand { axis: 0, len: 3 };
            (one(g, op, &[s], virt(&[3, 4, 3])), vec![3, 4, 3])
        }),
        ("Sum", |g, s, _| {
            (one(g, Op::Sum { axis: 0 }, &[s], virt(&[3])), vec![3])
        }),
        ("SumSquares", |g, s, _| {
            // Over the last axis, the rows' 4 once turned.
            let turned = one(g, Op::Transpose { perm: vec![1, 0] }, &[s], virt(&[3, 4]));
            (
                one(g, Op::SumSquares { axis: 1 }, &[turned], virt(&[3])),
                vec![3],
            )
        }),
        ("MatMulAct", |g, s, inputs| {
            let turned = one(g, Op::Transpose { perm: vec![1, 0] }, &[s], virt(&[3, 4]));
            let a = one(g, Op::Reshape, &[turned], virt(&[1, 3, 4]));
            let b = bytes(g, &[1, 4, 2]);
            inputs.push(b);
            let op = Op::MatMulAct {
                transpose_rhs: false,
            };
            (one(g, op, &[a, b], virt(&[1, 3, 2])), vec![1, 3, 2])
        }),
        ("MatMulAct transposed", |g, s, inputs| {
            let turned = one(g, Op::Transpose { perm: vec![1, 0] }, &[s], virt(&[3, 4]));
            let a = one(g, Op::Reshape, &[turned], virt(&[1, 3, 4]));
            let b = bytes(g, &[1, 2, 4]);
            inputs.push(b);
            let op = Op::MatMulAct {
                transpose_rhs: true,
            };
            (one(g, op, &[a, b], virt(&[1, 3, 2])), vec![1, 3, 2])
        }),
        ("MatMulWeight", |g, s, inputs| {
            let weight = WeightRef {
                ty: WeightType::Q,
                layer: 0,
            };
            let w = g.weight(weight, 3, 3).expect("a weight");
            inputs.push(w);
            let op = Op::MatMulWeight { weight };
            (one(g, op, &[s, w], virt(&[4, 3])), vec![4, 3])
        }),
        ("BitDecompose and BitFn", |g, s, inputs| {
            let x = bytes(g, &[4, 3]);
            inputs.push(x);
            let t = map(g, 1, x, &[4, 3]);
            let [a, b] = [s, t].map(|v| {
                let op = Op::BitDecompose { low: 8, high: 0 };
                g.node(op, &[v], vec![decl(&[4, 3, 8], ADVICE)])
                    .expect("the bits")[0]
            });
            let op = Op::BitFn { f: BoolFn::Xor };
            (one(g, op, &[a, b], virt(&[4, 3, 8])), vec![4, 3, 8])
        }),
        ("TableMap", |g, s, _| {
            let t = map(g, 1, s, &[4, 3]);
            (one(g, Op::Add, &[t, s], virt(&[4, 3])), vec![4, 3])
        }),
        ("Bitwise", |g, s, inputs| {
            let x = bytes(g, &[4, 3]);
            inputs.push(x);
            let output = decl(&[4, 3], NodeKind::LookupOutput { table: TableId(2) });
            let op = Op::Bitwise { table: TableId(2) };
            let t = g.node(op, &[s, x], vec![output]).expect("a lookup")[0];
            (one(g, Op::Sub, &[t, s], virt(&[4, 3])), vec![4, 3])
        }),
        (
            "Requant below and a TableMap non-zero at padding",
            |g, s, _| {
                let shift = g.shift_classes(vec![2]).expect("a shift");
                let mut output = decl(&[4, 3], ADVICE);
                output.range = IntRange { lo: 0, hi: 255 };
                let op = Op::Requant {
                    shift,
                    lo: 0,
                    hi: 255,
                };
                let r = g.node(op, &[s], vec![output]).expect("a requantization")[0];
                let q = map(g, 3, r, &[4, 3]);
                (one(g, Op::Add, &[q, r], virt(&[4, 3])), vec![4, 3])
            },
        ),
    ];
    for (k, (name, build)) in cases.into_iter().enumerate() {
        let (graph, inputs) = case(build);
        proves(name, &graph, &inputs, &[], 0x0526_4A00 + k as u64);
    }
    // RoPE: `Rotate` of `[3, 1, 4]`, the S-box's rows turned, its padded row 0 by its kernel.
    let mut g = GraphBuilder::new();
    let x = bytes(&mut g, &[4, 3]);
    let s = map(&mut g, 0, x, &[4, 3]);
    let turned = one(
        &mut g,
        Op::Transpose { perm: vec![1, 0] },
        &[s],
        virt(&[3, 4]),
    );
    let rows = one(&mut g, Op::Reshape, &[turned], virt(&[3, 1, 4]));
    let mut positions = decl(&[3], NodeKind::Public);
    positions.range = IntRange { lo: 0, hi: 15 };
    let positions = g.input(positions).expect("positions");
    let op = Op::Rotate { rope: TableId(4) };
    let rotated = one(&mut g, op, &[rows, positions], virt(&[3, 1, 4]));
    requant(&mut g, rotated, &shape(&[3, 1, 4]), 1 << 28);
    let graph = g.finish();
    proves(
        "Rotate",
        &graph,
        &[x],
        &[(positions, vec![0, 1, 2])],
        0x0526_4A80,
    );
    // `AssertEqual` of the S-box and committed bytes equal to it at its real positions, 0 at its
    // padded column, where the S-box holds 0x63: its `ψ̃` weighs that column out.
    let sbox = GeneratedTable::new(TableId(0), TableKind::AesSbox).expect("the S-box");
    let bytes_of: Vec<i64> = (0..12).map(|i| (37 * i + 11) % 256).collect();
    let boxed: Vec<i64> = bytes_of
        .iter()
        .map(|&b| sbox.lookup(b, 0).expect("a byte"))
        .collect();
    let mut g = GraphBuilder::new();
    let x = bytes(&mut g, &[4, 3]);
    let s = map(&mut g, 0, x, &[4, 3]);
    let y = bytes(&mut g, &[4, 3]);
    g.node(Op::AssertEqual, &[s, y], Vec::new())
        .expect("an equality");
    requant(&mut g, s, &[4, 3], 1 << 28);
    proves(
        "AssertEqual",
        &g.finish(),
        &[],
        &[(x, bytes_of), (y, boxed)],
        0x0526_4A81,
    );
    // A requantization of two classes, class 0 at the padded channel: `rq(0x63)` by its shift 1,
    // 50, where class 1's shift 2 would give 25.
    let mut g = GraphBuilder::new();
    let x = bytes(&mut g, &[4, 3]);
    let s = map(&mut g, 0, x, &[4, 3]);
    let mut classes = decl(
        &[3],
        NodeKind::Committed {
            commitment: CommitmentClass::Weight,
        },
    );
    classes.range = IntRange { lo: 0, hi: 1 };
    let c = g.input(classes).expect("the classes");
    let shift = g.shift_classes(vec![1, 2]).expect("two shifts");
    let mut output = decl(&[4, 3], ADVICE);
    output.range = IntRange { lo: 0, hi: 255 };
    let op = Op::Requant {
        shift,
        lo: 0,
        hi: 255,
    };
    let r = g.node(op, &[s, c], vec![output]).expect("a requantization")[0];
    requant(&mut g, r, &[4, 3], 1 << 28);
    let graph = g.finish();
    let tables = tables();
    let values = vec![
        (x, (0..12).map(|i| (91 * i + 5) % 256).collect()),
        (c, vec![1, 0, 1].into_iter().collect()),
    ];
    let witness = Engine::for_graph(&graph, &tables)
        .expect("an engine")
        .evaluate(values)
        .expect("a witness");
    let padded = PaddedWitness::new(&graph, &tables, &witness).expect("padded");
    let table = padded.padded(r).expect("the requantization's table");
    assert!(
        (0..4).all(|row| table[4 * row + 3] == 50),
        "class 0 at the padded channel"
    );
    proves(
        "Requant of two classes",
        &graph,
        &[],
        &[
            (x, (0..12).map(|i| (91 * i + 5) % 256).collect()),
            (c, vec![1, 0, 1]),
        ],
        0x0526_4A82,
    );
}

/// The padded tables are the witness's at real positions, so a graph that sums the S-box over
/// its padded axis, which RFC-0072 'Summing relations' refuses, is refused here too; so is a
/// `Reshape` that does not map the padded cube as the identity, and a `OneHot` that reads the
/// S-box, whose row relations take a padded row's token as 0 (RFC-0035).
#[test]
fn the_padded_tables_refuse_what_rfc_0072_does_not_admit() {
    let refused = |graph: Graph, inputs: &[TensorId]| {
        let tables = tables();
        let witness = Engine::for_graph(&graph, &tables)
            .expect("an engine")
            .evaluate(random_inputs(&graph, inputs, 0x0526_4A90))
            .expect("a witness");
        match PaddedWitness::new(&graph, &tables, &witness) {
            Err(StageError::Internal { rule }) => rule,
            other => panic!("refused, not {other:?}"),
        }
    };
    let summed = |axis: u8, out: &[u32]| {
        let mut g = GraphBuilder::new();
        let x = bytes(&mut g, &[4, 3]);
        let s = map(&mut g, 0, x, &[4, 3]);
        one(&mut g, Op::Sum { axis }, &[s], virt(out));
        (g.finish(), x)
    };
    let (graph, x) = summed(1, &[4]);
    assert_eq!(
        refused(graph, &[x]),
        "a padded table is the witness's values at its real positions (RFC-0072 'Summing \
         relations')"
    );
    let (graph, x) = summed(0, &[3]);
    let tables = tables();
    let witness = Engine::for_graph(&graph, &tables)
        .expect("an engine")
        .evaluate(random_inputs(&graph, &[x], 1))
        .expect("a witness");
    PaddedWitness::new(&graph, &tables, &witness).expect("a sum over the axis of 4");
    let mut g = GraphBuilder::new();
    let x = bytes(&mut g, &[4, 3]);
    let s = map(&mut g, 0, x, &[4, 3]);
    one(&mut g, Op::Reshape, &[s], virt(&[12]));
    assert_eq!(
        refused(g.finish(), &[x]),
        "a Reshape of a tensor non-zero at padding maps its padded cube as the identity"
    );
    let mut g = GraphBuilder::new();
    let x = bytes(&mut g, &[3]);
    let s = map(&mut g, 0, x, &[3]);
    let shift = g.shift_classes(vec![9]).expect("a shift");
    let mut tokens = decl(&[3], ADVICE);
    tokens.range = IntRange { lo: 0, hi: 0 };
    let op = Op::Requant {
        shift,
        lo: 0,
        hi: 0,
    };
    let token = g.node(op, &[s], vec![tokens]).expect("a token")[0];
    g.node(
        Op::OneHot { vocab: 4 },
        &[token],
        vec![decl(&[3, 4], ADVICE)],
    )
    .expect("a one-hot");
    assert_eq!(
        refused(g.finish(), &[x]),
        "no gadget but a lookup, BitDecompose or AssertEqual reads a tensor non-zero at its \
         padded positions: RFC-0035's row relations take a padded row's input as 0"
    );
}

/// A ragged INT8 weight must still pad correctly when another operand contains AesSbox(0).
#[test]
fn borrowed_int8_weights_preserve_nonzero_lookup_padding_and_stage_messages() {
    struct Borrowed<'a> {
        witness: &'a Witness,
        tensor: TensorId,
        weight: SecretBuf<i8>,
    }
    impl WitnessSource for Borrowed<'_> {
        fn values(&self, tensor: TensorId) -> Option<&[i64]> {
            (tensor != self.tensor)
                .then(|| self.witness.tensor(tensor))
                .flatten()
                .map(|v| &v[..])
        }
        fn int8_values(&self, tensor: TensorId) -> Option<&[i8]> {
            (tensor == self.tensor).then_some(&self.weight[..])
        }
    }
    let (graph, inputs) = case(|g, s, inputs| {
        let weight = WeightRef {
            ty: WeightType::Q,
            layer: 0,
        };
        let w = g.weight(weight, 3, 3).unwrap();
        inputs.push(w);
        (
            one(g, Op::MatMulWeight { weight }, &[s, w], virt(&[4, 3])),
            vec![4, 3],
        )
    });
    let tables = tables();
    let witness = Engine::for_graph(&graph, &tables)
        .unwrap()
        .evaluate(random_inputs(&graph, &inputs, 0x0641_8A55))
        .unwrap();
    let tensor = graph.weights[0].1;
    let borrowed = Borrowed {
        witness: &witness,
        tensor,
        weight: witness
            .tensor(tensor)
            .unwrap()
            .iter()
            .map(|&v| i8::try_from(v).unwrap())
            .collect(),
    };
    assert!(borrowed.weight.iter().any(|&v| v < 0));
    let original = PaddedWitness::new(&graph, &tables, &witness).unwrap();
    let padded = PaddedWitness::new(&graph, &tables, &borrowed).unwrap();
    let sbox = graph.nodes[0].outputs[0];
    assert_eq!(padded.padded(sbox).unwrap()[3], 0x63);
    for spec in &graph.tensors {
        assert_eq!(padded.padded(spec.id), original.padded(spec.id));
    }
    let schedule = derive_graph_schedule(&graph, &tables, &[], None).unwrap();
    let points = leaf_points(&schedule, 0x0641_8A55);
    let (reference, original_pool) = prove_graph(&graph, &tables, &schedule, &original, &points);
    let (proof, pool) = prove_graph(&graph, &tables, &schedule, &padded, &points);
    assert_eq!(proof.leaf, reference.leaf);
    assert_eq!(proof.leaf_splits, reference.leaf_splits);
    assert_eq!(proof.stages, reference.stages);
    assert_eq!(pool, original_pool);
    let statement = PublicOf::new(&graph, &[]);
    let verified = verify_public(
        &graph,
        &tables,
        &schedule,
        &proof,
        &points,
        Publics {
            values: &statement,
            shift: Shift::Extension,
            kernel: Kernel::Closed,
        },
    )
    .unwrap();
    assert_eq!(
        verified,
        pool.iter().map(|v| v.unwrap()).collect::<Vec<_>>()
    );
}
