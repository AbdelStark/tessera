// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Range analysis (RFC-0006 'Range analysis') against an exact interpreter of the IR.
//!
//! - **Soundness, operation by operation.** Every operation, in graphs of one node with random
//!   input intervals, keeps every value of random and corner inputs inside the range the analysis
//!   gives it.
//! - **Soundness on a layer.** An attention and MLP block shaped as RFC-0005's servable layer
//!   (norm, projections, RoPE, softmax, the attention output, requantizations, residuals, SiLU)
//!   keeps every tensor inside its range on 10^4 random inputs and 10^3 adversarial ones: weights
//!   of extreme sign, activations at their bounds, a single dominant score.
//! - **The relational fact.** With `R` and `e` from one `SoftmaxRow`, the attention output's
//!   range is RFC-0005's `[−2^39, 127 · 2^32]`, and its lower end is attained; with `R` from
//!   another row's softmax, the fact does not apply and the range is the product bound.
//! - **True integers.** Each rule at a gadget input refuses a graph that breaks only it, with
//!   `ERR-COMPILE-WRAP` for no bound and `ERR-COMPILE-RANGE` for a bound outside what the gadget
//!   or its table accepts.

use proptest::prelude::*;
use tessera_codec::Rational;
use tessera_codec::error::ErrorCode;
use tessera_compiler::{CompileError, HALF_FIELD, Range, analyze};
use tessera_ir::{
    CommitmentClass, Graph, GraphBuilder, IntRange, IntType, NodeId, NodeKind, Op, TableId,
    TensorDecl, TensorId, WeightRef, WeightType,
};
use tessera_tables::{GeneratedTable, RopeTable, Table, TableKind, TableSpec};

// ----- Building graphs -------------------------------------------------------------------------

const ADVICE: NodeKind = NodeKind::Committed {
    commitment: CommitmentClass::Advice,
};

fn decl(shape: &[u32], kind: NodeKind, lo: i128, hi: i128) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: IntType::I32,
        range: IntRange { lo, hi },
        kind,
    }
}

/// An output: its declared range is the analysis's to compute, so any placeholder will do.
fn out(shape: &[u32], kind: NodeKind) -> TensorDecl {
    decl(shape, kind, 0, 0)
}

fn virt(shape: &[u32]) -> TensorDecl {
    out(shape, NodeKind::Virtual)
}

fn lookup(shape: &[u32], table: u32) -> TensorDecl {
    out(
        shape,
        NodeKind::LookupOutput {
            table: TableId(table),
        },
    )
}

fn one(builder: &mut GraphBuilder, op: Op, inputs: &[TensorId], output: TensorDecl) -> TensorId {
    builder
        .node(op, inputs, vec![output])
        .expect("a valid node")[0]
}

// ----- An exact interpreter of the IR ---------------------------------------------------------

/// The honest values of every tensor, flat and row-major, from the inputs' values.
struct Run<'a> {
    graph: &'a Graph,
    tables: &'a [Box<dyn Lookup>],
    values: Vec<Option<Vec<i128>>>,
}

/// What the interpreter needs of a table: the value of a key, and RoPE's pairs.
trait Lookup {
    fn id(&self) -> TableId;
    fn value(&self, key: i128) -> i128;
    fn rope(&self, _position: u32, _frequency: u32) -> (i128, i128) {
        unreachable!("not a RoPE table")
    }
}

impl Lookup for GeneratedTable {
    fn id(&self) -> TableId {
        self.spec().id
    }
    fn value(&self, key: i128) -> i128 {
        let key = i64::try_from(key).expect("a key in i64");
        i128::from(self.lookup(key, 0).expect("a key inside the table"))
    }
}

impl Lookup for RopeTable {
    fn id(&self) -> TableId {
        self.spec().id
    }
    fn value(&self, _key: i128) -> i128 {
        unreachable!("RoPE is no lookup table")
    }
    fn rope(&self, position: u32, frequency: u32) -> (i128, i128) {
        let (c, s) = self.constants().at(position, frequency).expect("a pair");
        (i128::from(c), i128::from(s))
    }
}

fn elements(shape: &[u32]) -> usize {
    shape.iter().map(|&d| d as usize).product()
}

fn rq(x: i128, r: u8, lo: i64, hi: i64) -> i128 {
    ((x + (1 << (r - 1))) >> r).clamp(i128::from(lo), i128::from(hi))
}

fn isqrt(n: i128) -> i128 {
    let mut r = (n as f64).sqrt() as i128;
    while r * r > n {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= n {
        r += 1;
    }
    r
}

impl Run<'_> {
    fn get(&self, t: TensorId) -> &[i128] {
        self.values[t.0 as usize].as_deref().expect("computed")
    }

    fn shape(&self, t: TensorId) -> &[u32] {
        &self.graph.tensors[t.0 as usize].shape
    }

    fn table(&self, id: TableId) -> &dyn Lookup {
        self.tables
            .iter()
            .find(|t| t.id() == id)
            .expect("a table")
            .as_ref()
    }

    fn run(&mut self) {
        for node in &self.graph.nodes {
            let outputs = self.node(node);
            for (&t, v) in node.outputs.iter().zip(outputs) {
                self.values[t.0 as usize] = Some(v);
            }
        }
    }

    fn node(&self, node: &tessera_ir::Node) -> Vec<Vec<i128>> {
        let x = |k: usize| self.get(node.inputs[k]);
        let shape = |k: usize| self.shape(node.inputs[k]);
        let last = |k: usize| *shape(k).last().expect("an axis") as usize;
        match &node.op {
            Op::MatMulWeight { .. } => {
                let w = self.shape(node.inputs[1]);
                let (rows, cols) = (w[0] as usize, w[1] as usize);
                let (input, weight) = (x(0), x(1));
                let mut y = Vec::new();
                for row in input.chunks(cols) {
                    for o in 0..rows {
                        y.push((0..cols).map(|i| weight[o * cols + i] * row[i]).sum());
                    }
                }
                vec![y]
            }
            Op::Add => vec![x(0).iter().zip(x(1)).map(|(a, b)| a + b).collect()],
            Op::Sub => vec![x(0).iter().zip(x(1)).map(|(a, b)| a - b).collect()],
            Op::ConstMul { c } | Op::ConstAdd { c } => {
                let values = &self.graph.constants[c.0 as usize];
                let n = last(0);
                let y = x(0).iter().enumerate().map(|(i, &v)| {
                    let c = i128::from(if values.len() == 1 {
                        values[0]
                    } else {
                        values[i % n]
                    });
                    if matches!(node.op, Op::ConstMul { .. }) {
                        c * v
                    } else {
                        c + v
                    }
                });
                vec![y.collect()]
            }
            Op::Rotate { rope } => {
                let table = self.table(*rope);
                let (n, width) = (shape(0)[0] as usize, last(0));
                let per_row = elements(shape(0)) / n;
                let positions = x(1);
                let mut y = x(0).to_vec();
                for (pair, chunk) in y.chunks_mut(2).enumerate() {
                    let row = pair * 2 / per_row;
                    let frequency = ((pair * 2) % width) / 2;
                    let p = u32::try_from(positions[row]).expect("a position");
                    let (c, s) = table.rope(p, u32::try_from(frequency).expect("small"));
                    let (a, b) = (chunk[0], chunk[1]);
                    chunk[0] = a * c - b * s;
                    chunk[1] = a * s + b * c;
                }
                vec![y]
            }
            Op::Sum { axis } | Op::SumSquares { axis } => {
                let s = shape(0);
                let axis = usize::from(*axis);
                let inner: usize = s[axis + 1..].iter().map(|&d| d as usize).product();
                let len = s[axis] as usize;
                let outer = elements(s) / (inner * len);
                let square = matches!(node.op, Op::SumSquares { .. });
                let mut y = Vec::new();
                for o in 0..outer {
                    for i in 0..inner {
                        y.push(
                            (0..len)
                                .map(|k| x(0)[(o * len + k) * inner + i])
                                .map(|v| if square { v * v } else { v })
                                .sum(),
                        );
                    }
                }
                vec![y]
            }
            Op::Reshape => vec![x(0).to_vec()],
            Op::Slice { axis, start, len } => {
                let s = shape(0);
                let axis = usize::from(*axis);
                let inner: usize = s[axis + 1..].iter().map(|&d| d as usize).product();
                let full = s[axis] as usize;
                let outer = elements(s) / (inner * full);
                let mut y = Vec::new();
                for o in 0..outer {
                    for k in *start as usize..(*start + *len) as usize {
                        y.extend_from_slice(&x(0)[(o * full + k) * inner..][..inner]);
                    }
                }
                vec![y]
            }
            Op::GatherPublic { index } => {
                let row = elements(shape(0)) / shape(0)[0] as usize;
                let rows = &self.graph.constants[index.0 as usize];
                vec![
                    rows.iter()
                        .flat_map(|&r| x(0)[r as usize * row..][..row].to_vec())
                        .collect(),
                ]
            }
            Op::MatMulAct { transpose_rhs } => {
                let (a, b) = (shape(0), shape(1));
                let r = a.len();
                let (m, k) = (a[r - 2] as usize, a[r - 1] as usize);
                let n = if *transpose_rhs { b[r - 2] } else { b[r - 1] } as usize;
                let batches = elements(a) / (m * k);
                let mut y = Vec::new();
                for batch in 0..batches {
                    let (lhs, rhs) = (&x(0)[batch * m * k..], &x(1)[batch * k * n..]);
                    for i in 0..m {
                        for j in 0..n {
                            y.push(
                                (0..k)
                                    .map(|t| {
                                        let right = if *transpose_rhs {
                                            rhs[j * k + t]
                                        } else {
                                            rhs[t * n + j]
                                        };
                                        lhs[i * k + t] * right
                                    })
                                    .sum(),
                            );
                        }
                    }
                }
                vec![y]
            }
            Op::Mul => {
                let trailing = elements(shape(0)) / elements(shape(1));
                vec![
                    x(0).iter()
                        .enumerate()
                        .map(|(i, &a)| a * x(1)[i / trailing])
                        .collect(),
                ]
            }
            Op::OneHot { vocab } => {
                let v = *vocab as usize;
                vec![
                    x(0).iter()
                        .flat_map(|&t| (0..v).map(move |j| i128::from(j as i128 == t)))
                        .collect(),
                ]
            }
            Op::Requant { shift, lo, hi } => {
                let classes = &self.graph.shift_classes[shift.0 as usize];
                let n = last(0);
                let y = x(0).iter().enumerate().map(|(i, &v)| {
                    let class = if node.inputs.len() > 1 {
                        x(1)[i % n] as usize
                    } else {
                        0
                    };
                    rq(v, classes[class], *lo, *hi)
                });
                vec![y.collect()]
            }
            Op::TableMap { table } => {
                let t = self.table(*table);
                vec![x(0).iter().map(|&k| t.value(k)).collect()]
            }
            Op::SoftmaxRow { exp_table, sigma } => {
                let t = self.table(*exp_table);
                let n = last(0);
                let (mut e, mut r) = (Vec::new(), Vec::new());
                for row in x(0).chunks(n) {
                    let max = *row.iter().max().expect("a score");
                    let exps: Vec<i128> = row.iter().map(|&s| t.value(max - s)).collect();
                    let sum: i128 = exps.iter().sum();
                    r.push((1i128 << (2 * u32::from(*sigma))) / sum);
                    e.extend(exps);
                }
                vec![e, r]
            }
            Op::RsqrtRow { kappa, eps_int } => {
                // d is the row length the sums of squares ran over.
                let d = self
                    .graph
                    .nodes
                    .iter()
                    .find(|n| n.outputs.contains(&node.inputs[0]))
                    .map_or(1, |p| match p.op {
                        Op::SumSquares { axis } => {
                            i128::from(self.shape(p.inputs[0])[usize::from(axis)])
                        }
                        _ => 1,
                    });
                let big = d << (2 * u32::from(*kappa));
                vec![
                    x(0).iter()
                        .map(|&u| isqrt(big / (u + i128::from(*eps_int))))
                        .collect(),
                ]
            }
            Op::Bitwise { table } => {
                let t = self.table(*table);
                vec![
                    x(0).iter()
                        .zip(x(1))
                        .map(|(&a, &b)| t.value(a + 256 * b))
                        .collect(),
                ]
            }
            Op::Argmax | Op::InverseCdf { .. } => Vec::new(),
        }
    }
}

/// Runs `graph` on the inputs' values and checks every tensor against its analyzed range.
fn check_run(
    graph: &Graph,
    tables: &[Box<dyn Lookup>],
    specs: &[TableSpec],
    inputs: Vec<(TensorId, Vec<i128>)>,
) {
    let ranges = analyze(graph, specs).expect("the graph analyzes");
    let mut run = Run {
        graph,
        tables,
        values: vec![None; graph.tensors.len()],
    };
    for (t, v) in inputs {
        assert_eq!(v.len(), elements(&graph.tensors[t.0 as usize].shape));
        run.values[t.0 as usize] = Some(v);
    }
    run.run();
    for (k, values) in run.values.iter().enumerate() {
        let Some(values) = values else { continue };
        let range = ranges.all()[k];
        if let Range::Bounded(r) = range {
            for &v in values {
                assert!(
                    r.lo <= v && v <= r.hi,
                    "tensor {k}: {v} outside [{}, {}]",
                    r.lo,
                    r.hi
                );
            }
        }
    }
}

// ----- Soundness, operation by operation -------------------------------------------------------

/// A random interval of magnitude below `2^bits`, possibly a single point.
fn interval(bits: u32) -> impl Strategy<Value = (i128, i128)> {
    let bound = 1i64 << bits;
    (-bound..bound, -bound..bound).prop_map(|(a, b)| (i128::from(a.min(b)), i128::from(a.max(b))))
}

/// Values for a tensor in `[lo, hi]`: a corner, its value nearest 0, alternating corners, or
/// random values.
fn fill(n: usize, (lo, hi): (i128, i128), seed: u64, corner: u8) -> Vec<i128> {
    let mut state = seed;
    (0..n)
        .map(|i| match corner {
            0 => lo,
            1 => hi,
            2 => {
                if i % 2 == 0 {
                    lo
                } else {
                    hi
                }
            }
            4 => 0i128.clamp(lo, hi),
            _ => {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let span = (hi - lo + 1) as u128;
                lo + ((u128::from(state >> 11) * 0x9E37_79B9) % span) as i128
            }
        })
        .collect()
}

fn single_op(op: Op, a: (i128, i128), b: (i128, i128), seed: u64) {
    let mut g = GraphBuilder::new();
    let shape = [2, 3, 4];
    let x = g.input(decl(&shape, ADVICE, a.0, a.1)).expect("an input");
    let y = g.input(decl(&shape, ADVICE, b.0, b.1)).expect("an input");
    let c = g.constant(vec![-3, 7, 0, 5]).expect("a constant");
    let inputs: Vec<TensorId>;
    let output = match op {
        Op::Add | Op::Sub | Op::Mul => {
            inputs = vec![x, y];
            virt(&shape)
        }
        Op::ConstMul { .. } | Op::ConstAdd { .. } => {
            inputs = vec![x];
            virt(&shape)
        }
        Op::Sum { axis } | Op::SumSquares { axis } => {
            inputs = vec![x];
            let mut s = shape.to_vec();
            s.remove(usize::from(axis));
            virt(&s)
        }
        Op::MatMulAct {
            transpose_rhs: true,
        } => {
            inputs = vec![x, y];
            virt(&[2, 3, 3])
        }
        _ => unreachable!(),
    };
    let op = match op {
        Op::ConstMul { .. } => Op::ConstMul { c },
        Op::ConstAdd { .. } => Op::ConstAdd { c },
        op => op,
    };
    one(&mut g, op, &inputs, output);
    let graph = g.finish();
    for corner in 0..5u8 {
        for other in 0..5u8 {
            let values = vec![
                (x, fill(24, a, seed, corner)),
                (y, fill(24, b, seed ^ 0x55, other)),
            ];
            check_run(&graph, &[], &[], values);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn linear_and_degree_2_operations_keep_their_values_in_range(
        a in interval(40), b in interval(40), seed in any::<u64>(), which in 0..8usize,
    ) {
        let op = [
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::ConstMul { c: tessera_ir::ConstRef(0) },
            Op::ConstAdd { c: tessera_ir::ConstRef(0) },
            Op::Sum { axis: 1 },
            Op::SumSquares { axis: 2 },
            Op::MatMulAct { transpose_rhs: true },
        ][which].clone();
        single_op(op, a, b, seed);
    }

    #[test]
    fn a_weight_product_stays_within_its_public_bound(
        a in interval(20), seed in any::<u64>(), signs in any::<u64>(),
    ) {
        let mut g = GraphBuilder::new();
        let x = g.input(decl(&[3, 5], ADVICE, a.0, a.1)).expect("an input");
        let weight = WeightRef { ty: WeightType::Up, layer: 0 };
        let w = g.weight(weight, 4, 5).expect("a weight");
        one(&mut g, Op::MatMulWeight { weight }, &[x, w], virt(&[3, 4]));
        let graph = g.finish();
        // Weights of ±127, the extremes the public bound allows.
        let weights: Vec<i128> = (0..20).map(|i| if (signs >> i) & 1 == 1 { 127 } else { -127 }).collect();
        for corner in 0..5u8 {
            check_run(&graph, &[], &[], vec![(x, fill(15, a, seed, corner)), (w, weights.clone())]);
        }
    }
}

#[test]
fn rotation_one_hot_vectors_and_byte_operations_keep_their_values_in_range() {
    let (tables, specs) = layer_tables();
    // RoPE at every position and frequency, with pairs at opposite corners: a·C − b·S reaches
    // (|C| + |S|) · m, above 2^bits · m wherever C and S are both large.
    let mut g = GraphBuilder::new();
    let m = 1000;
    let x = g.input(decl(&[16, 4], ADVICE, -m, m)).expect("x");
    let positions = g
        .input(decl(&[16], NodeKind::Public, 0, 15))
        .expect("positions");
    one(
        &mut g,
        Op::Rotate {
            rope: TableId(ROPE),
        },
        &[x, positions],
        virt(&[16, 4]),
    );
    let graph = g.finish();
    for corner in 0..5u8 {
        let values = vec![
            (x, fill(64, (-m, m), 7, corner)),
            (positions, (0..16).collect()),
        ];
        check_run(&graph, &tables, &specs, values);
    }
    // One-hot vectors of every token.
    let mut g = GraphBuilder::new();
    let t = g.input(decl(&[10], ADVICE, 0, 9)).expect("tokens");
    one(
        &mut g,
        Op::OneHot { vocab: 10 },
        &[t],
        out(&[10, 10], ADVICE),
    );
    check_run(&g.finish(), &tables, &specs, vec![(t, (0..10).collect())]);
    // Every byte pair through Bitwise8.
    let xor = GeneratedTable::new(
        TableId(5),
        TableKind::Bitwise8 {
            op: tessera_tables::BitOp::Xor,
        },
    )
    .expect("a Bitwise8 table");
    let mut g = GraphBuilder::new();
    let a = g.input(decl(&[256 * 256], ADVICE, 0, 255)).expect("a");
    let b = g.input(decl(&[256 * 256], ADVICE, 0, 255)).expect("b");
    one(
        &mut g,
        Op::Bitwise { table: TableId(5) },
        &[a, b],
        lookup(&[256 * 256], 5),
    );
    let inputs = vec![
        (a, (0..65536).map(|i| i % 256).collect()),
        (b, (0..65536).map(|i| i / 256).collect()),
    ];
    check_run(
        &g.finish(),
        &[Box::new(xor.clone())],
        &[xor.spec().clone()],
        inputs,
    );
}

// ----- A layer: attention and MLP -----------------------------------------------------------

const SIGMA: u8 = 16;
const EXP: u32 = 1;
const SILU: u32 = 2;
const ROPE: u32 = 3;

/// The tables of the layer: `Exp` (σ = 16), `Silu` and RoPE at test sizes.
fn layer_tables() -> (Vec<Box<dyn Lookup>>, Vec<TableSpec>) {
    let exp = GeneratedTable::new(
        TableId(EXP),
        TableKind::Exp {
            key_bits: 22,
            sigma: SIGMA,
            scale: Rational::new(1, 2896).expect("a scale"),
        },
    )
    .expect("an Exp table");
    let silu = GeneratedTable::new(
        TableId(SILU),
        TableKind::Silu {
            b: 24,
            r_by_class: vec![10],
            in_exp: -6,
            out_exp: -4,
        },
    )
    .expect("a Silu table");
    let rope = RopeTable::new(
        TableId(ROPE),
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
    let specs = vec![exp.spec().clone(), silu.spec().clone(), rope.spec().clone()];
    (vec![Box::new(exp), Box::new(silu), Box::new(rope)], specs)
}

struct Layer {
    graph: Graph,
    residual: TensorId,
    positions: TensorId,
    weights: Vec<TensorId>,
    attention: TensorId,
}

/// One servable layer of width 8, with 2 query heads and 1 key and value head of width 4, over
/// `n` positions, every position visible (no masked term): RFC-0005 'Integer semantics' in
/// small.
fn layer(n: u32) -> Layer {
    let mut g = GraphBuilder::new();
    let (d, dh, ff) = (8u32, 4u32, 16u32);
    let residual = g
        .input(decl(&[n, d], ADVICE, -(1 << 19), (1 << 19) - 1))
        .expect("res");
    let positions = g
        .input(decl(&[n], NodeKind::Public, 0, i128::from(n) - 1))
        .expect("pos");
    let r8 = g.shift_classes(vec![8]).expect("shifts");
    let r12 = g.shift_classes(vec![12]).expect("shifts");
    let int8 = |g: &mut GraphBuilder, x: TensorId, shift, shape: &[u32]| {
        one(
            g,
            Op::Requant {
                shift,
                lo: -128,
                hi: 127,
            },
            &[x],
            out(shape, ADVICE),
        )
    };
    let mut weights = Vec::new();
    let mut weight = |g: &mut GraphBuilder, ty, rows, cols| {
        let w = WeightRef { ty, layer: 0 };
        let t = g.weight(w, rows, cols).expect("a weight");
        weights.push(t);
        (w, t)
    };
    // Attention norm: ρ = rsqrt of the row's sum of squares, y = rq(res · ρ).
    let squares = one(&mut g, Op::SumSquares { axis: 1 }, &[residual], virt(&[n]));
    let rho = one(
        &mut g,
        Op::RsqrtRow {
            kappa: 16,
            eps_int: 3,
        },
        &[squares],
        out(&[n], ADVICE),
    );
    let scaled = one(&mut g, Op::Mul, &[residual, rho], virt(&[n, d]));
    let y = int8(&mut g, scaled, r12, &[n, d]);
    // Q, K, V, and RoPE on Q and K.
    let (wq, q_w) = weight(&mut g, WeightType::Q, 2 * dh, d);
    let (wk, k_w) = weight(&mut g, WeightType::K, dh, d);
    let (wv, v_w) = weight(&mut g, WeightType::V, dh, d);
    let q = one(
        &mut g,
        Op::MatMulWeight { weight: wq },
        &[y, q_w],
        virt(&[n, 2 * dh]),
    );
    let k = one(
        &mut g,
        Op::MatMulWeight { weight: wk },
        &[y, k_w],
        virt(&[n, dh]),
    );
    let v = one(
        &mut g,
        Op::MatMulWeight { weight: wv },
        &[y, v_w],
        virt(&[n, dh]),
    );
    let q = one(&mut g, Op::Reshape, &[q], virt(&[n, 2, dh]));
    let q = one(
        &mut g,
        Op::Rotate {
            rope: TableId(ROPE),
        },
        &[q, positions],
        virt(&[n, 2, dh]),
    );
    let k = one(
        &mut g,
        Op::Rotate {
            rope: TableId(ROPE),
        },
        &[k, positions],
        virt(&[n, dh]),
    );
    let q8 = int8(&mut g, q, r12, &[n, 2, dh]);
    let k8 = int8(&mut g, k, r12, &[n, dh]);
    let v8 = int8(&mut g, v, r8, &[n, dh]);
    // Scores of query head 0 against the shared key head, the softmax, and R · PV.
    let q0 = one(
        &mut g,
        Op::Slice {
            axis: 1,
            start: 0,
            len: 1,
        },
        &[q8],
        virt(&[n, 1, dh]),
    );
    let q0 = one(&mut g, Op::Reshape, &[q0], virt(&[n, dh]));
    let scores = one(
        &mut g,
        Op::MatMulAct {
            transpose_rhs: true,
        },
        &[q0, k8],
        virt(&[n, n]),
    );
    let softmax = g
        .node(
            Op::SoftmaxRow {
                exp_table: TableId(EXP),
                sigma: SIGMA,
            },
            &[scores],
            vec![lookup(&[n, n], EXP), out(&[n], ADVICE)],
        )
        .expect("a softmax");
    let pv = one(
        &mut g,
        Op::MatMulAct {
            transpose_rhs: false,
        },
        &[softmax[0], v8],
        virt(&[n, dh]),
    );
    let attention = one(&mut g, Op::Mul, &[pv, softmax[1]], virt(&[n, dh]));
    let o8 = int8(&mut g, attention, r12, &[n, dh]);
    // O projection back to the residual, then the MLP: gate and up, SiLU, product, down.
    let (wo, o_w) = weight(&mut g, WeightType::O, d, dh);
    let acc = one(
        &mut g,
        Op::MatMulWeight { weight: wo },
        &[o8, o_w],
        virt(&[n, d]),
    );
    let update = one(
        &mut g,
        Op::Requant {
            shift: r8,
            lo: -(1 << 19),
            hi: (1 << 19) - 1,
        },
        &[acc],
        out(&[n, d], ADVICE),
    );
    let res = one(&mut g, Op::Add, &[residual, update], virt(&[n, d]));
    let (wg, g_w) = weight(&mut g, WeightType::Gate, ff, d);
    let (wu, u_w) = weight(&mut g, WeightType::Up, ff, d);
    let res8 = int8(&mut g, res, r12, &[n, d]);
    let gate = one(
        &mut g,
        Op::MatMulWeight { weight: wg },
        &[res8, g_w],
        virt(&[n, ff]),
    );
    let up = one(
        &mut g,
        Op::MatMulWeight { weight: wu },
        &[res8, u_w],
        virt(&[n, ff]),
    );
    let a = one(
        &mut g,
        Op::TableMap {
            table: TableId(SILU),
        },
        &[gate],
        lookup(&[n, ff], SILU),
    );
    let u8 = int8(&mut g, up, r8, &[n, ff]);
    let product = one(&mut g, Op::Mul, &[a, u8], virt(&[n, ff]));
    let h = int8(&mut g, product, r8, &[n, ff]);
    let (wd, d_w) = weight(&mut g, WeightType::Down, d, ff);
    let down = one(
        &mut g,
        Op::MatMulWeight { weight: wd },
        &[h, d_w],
        virt(&[n, d]),
    );
    one(
        &mut g,
        Op::Requant {
            shift: r8,
            lo: -(1 << 19),
            hi: (1 << 19) - 1,
        },
        &[down],
        out(&[n, d], ADVICE),
    );
    Layer {
        graph: g.finish(),
        residual,
        positions,
        weights,
        attention,
    }
}

#[test]
fn a_layer_keeps_every_tensor_in_range_on_random_and_adversarial_inputs() {
    let (tables, specs) = layer_tables();
    let n = 6;
    let layer = layer(n);
    // Every tensor of the layer has a bound, so every value below is checked, and the attention
    // output's is the softmax fact's.
    let ranges = analyze(&layer.graph, &specs).expect("analyzes");
    assert!(ranges.all().iter().all(|r| matches!(r, Range::Bounded(_))));
    assert_eq!(
        ranges.of(layer.attention),
        Some(Range::new(-(1 << 39), 127 << 32))
    );
    let positions: Vec<i128> = (0..i128::from(n)).collect();
    let mut state = 0x7E55_E7A0_0077_0001u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 11
    };
    for trial in 0..11_000 {
        let adversarial = trial >= 10_000;
        let pick = |lo: i128, hi: i128, r: u64| {
            lo + i128::from(r % u64::try_from(hi - lo + 1).unwrap_or(1))
        };
        // Adversarial: activations at a bound, weights all of one sign, or alternating.
        let residual: Vec<i128> = (0..48)
            .map(|i| {
                let r = next();
                match (adversarial, trial % 3) {
                    (true, 0) => (1 << 19) - 1,
                    (true, 1) => -(1 << 19),
                    (true, _) => {
                        if (r + i) % 2 == 0 {
                            (1 << 19) - 1
                        } else {
                            -(1 << 19)
                        }
                    }
                    (false, _) => pick(-(1 << 19), (1 << 19) - 1, r),
                }
            })
            .collect();
        let mut inputs = vec![
            (layer.residual, residual),
            (layer.positions, positions.clone()),
        ];
        for &w in &layer.weights {
            let count = elements(&layer.graph.tensors[w.0 as usize].shape);
            let values = (0..count)
                .map(|i| match (adversarial, trial % 4) {
                    (true, 0) => 127,
                    (true, 1) => -127,
                    (true, _) => {
                        if i % 2 == 0 {
                            127
                        } else {
                            -127
                        }
                    }
                    (false, _) => pick(-127, 127, next()),
                })
                .collect();
            inputs.push((w, values));
        }
        check_run(&layer.graph, &tables, &specs, inputs);
    }
}

// ----- The relational fact --------------------------------------------------------------------

/// `Mul(MatMulAct(e, v), R)` over `n` positions, `v` in INT8, with `R` from the same softmax as
/// `e`, or from a second softmax of other scores when `same` is false.
fn attention(n: u32, same: bool) -> (Graph, TensorId, TensorId, TensorId) {
    let mut g = GraphBuilder::new();
    let scores = g
        .input(decl(&[n, n], ADVICE, -(1 << 20), 1 << 20))
        .expect("scores");
    let others = g
        .input(decl(&[n, n], ADVICE, -(1 << 20), 1 << 20))
        .expect("scores");
    let v = g.input(decl(&[n, 4], ADVICE, -128, 127)).expect("values");
    let softmax = |g: &mut GraphBuilder, s| {
        g.node(
            Op::SoftmaxRow {
                exp_table: TableId(EXP),
                sigma: SIGMA,
            },
            &[s],
            vec![lookup(&[n, n], EXP), out(&[n], ADVICE)],
        )
        .expect("a softmax")
    };
    let first = softmax(&mut g, scores);
    let second = softmax(&mut g, others);
    let pv = one(
        &mut g,
        Op::MatMulAct {
            transpose_rhs: false,
        },
        &[first[0], v],
        virt(&[n, 4]),
    );
    let r = if same { first[1] } else { second[1] };
    let y = one(&mut g, Op::Mul, &[pv, r], virt(&[n, 4]));
    (g.finish(), scores, v, y)
}

#[test]
fn the_softmax_fact_bounds_the_attention_output_as_rfc_0005_states() {
    let (tables, specs) = layer_tables();
    let n = 32;
    let (graph, scores, v, y) = attention(n, true);
    let ranges = analyze(&graph, &specs).expect("analyzes");
    // 2^{2σ} · [−128, 127] = [−2^39, 127 · 2^32].
    assert_eq!(ranges.of(y), Some(Range::new(-(1 << 39), 127 << 32)));

    // Its lower end is attained: one dominant score, every value −128.
    let mut s = vec![-(1 << 20); (n * n) as usize];
    for row in 0..n as usize {
        s[row * n as usize] = 1 << 20;
    }
    let values = vec![(scores, s), (v, vec![-128; (n * 4) as usize])];
    let mut run = Run {
        graph: &graph,
        tables: &tables,
        values: vec![None; graph.tensors.len()],
    };
    for (t, x) in values.clone() {
        run.values[t.0 as usize] = Some(x);
    }
    let others = TensorId(1);
    run.values[others.0 as usize] = Some(vec![0; (n * n) as usize]);
    run.run();
    assert_eq!(run.get(y).iter().min(), Some(&-(1i128 << 39)));

    // With R from another softmax, only the operands' bounds apply: R ≤ 2^{σ+1} − 1, and PV is
    // n terms of e ≤ 2^σ times v in [−128, 127].
    let (unrelated, ..) = attention(n, false);
    let product = analyze(&unrelated, &specs).expect("analyzes");
    let r_max: i128 = (1 << (SIGMA + 1)) - 1;
    let pv = (
        i128::from(n) * (1 << SIGMA) * -128,
        i128::from(n) * (1 << SIGMA) * 127,
    );
    assert_eq!(product.of(y), Some(Range::new(pv.0 * r_max, pv.1 * r_max)));
    // n · (2^{σ+1} − 1) / 2^σ, about 64 times wider at n = 32.
    let (Some(Range::Bounded(fact)), Some(Range::Bounded(bound))) = (ranges.of(y), product.of(y))
    else {
        unreachable!()
    };
    assert!(bound.hi / fact.hi >= 63);
}

// ----- True integers ----------------------------------------------------------------------------

fn refused(graph: &Graph, specs: &[TableSpec]) -> CompileError {
    analyze(graph, specs).expect_err("refused")
}

#[test]
fn a_gadget_input_without_a_bound_is_refused_as_a_possible_wrap() {
    // Products of 2^62-wide values leave i128; the requantization's key has no bound.
    let mut g = GraphBuilder::new();
    let x = g
        .input(decl(&[2, 2], ADVICE, -(1 << 62), 1 << 62))
        .expect("x");
    let mut t = x;
    for _ in 0..2 {
        t = one(
            &mut g,
            Op::MatMulAct {
                transpose_rhs: false,
            },
            &[t, t],
            virt(&[2, 2]),
        );
    }
    let shift = g.shift_classes(vec![8]).expect("shifts");
    one(
        &mut g,
        Op::Requant {
            shift,
            lo: -128,
            hi: 127,
        },
        &[t],
        out(&[2, 2], ADVICE),
    );
    let graph = g.finish();
    let ranges = analyze(&graph, &[]);
    let error = ranges.expect_err("refused");
    assert_eq!(
        error,
        CompileError::Wrap {
            node: NodeId(2),
            input: 0
        }
    );
    assert_eq!(error.code(), ErrorCode::CompileWrap);
}

#[test]
fn each_true_integer_rule_refuses_a_graph_that_breaks_only_it() {
    let (_, specs) = layer_tables();
    let range = |node: u32, input: usize, rule: &'static str| CompileError::Range {
        node: NodeId(node),
        input,
        rule,
    };
    // A key just outside the centered field range.
    let mut g = GraphBuilder::new();
    let x = g.input(decl(&[2], ADVICE, 0, HALF_FIELD)).expect("x");
    let shift = g.shift_classes(vec![8]).expect("shifts");
    one(
        &mut g,
        Op::Requant {
            shift,
            lo: -128,
            hi: 127,
        },
        &[x],
        out(&[2], ADVICE),
    );
    assert_eq!(
        refused(&g.finish(), &specs),
        range(0, 0, "a true integer lies in (−(P−1)/2, (P−1)/2)")
    );

    // A token one past the vocabulary.
    let mut g = GraphBuilder::new();
    let t = g.input(decl(&[3], ADVICE, 0, 50)).expect("tokens");
    one(
        &mut g,
        Op::OneHot { vocab: 50 },
        &[t],
        out(&[3, 50], ADVICE),
    );
    assert_eq!(
        refused(&g.finish(), &specs),
        range(0, 0, "a token lies in [0, vocab)")
    );

    // A class index past the shift classes.
    let mut g = GraphBuilder::new();
    let x = g.input(decl(&[2, 3], ADVICE, -100, 100)).expect("x");
    let classes = g
        .input(decl(
            &[3],
            NodeKind::Committed {
                commitment: CommitmentClass::Weight,
            },
            0,
            2,
        ))
        .expect("classes");
    let shift = g.shift_classes(vec![8, 9]).expect("shifts");
    one(
        &mut g,
        Op::Requant {
            shift,
            lo: -128,
            hi: 127,
        },
        &[x, classes],
        out(&[2, 3], ADVICE),
    );
    assert_eq!(
        refused(&g.finish(), &specs),
        range(0, 1, "a channel class indexes the shift classes")
    );

    // A SiLU key outside its table's 24-bit keys.
    let mut g = GraphBuilder::new();
    let x = g.input(decl(&[4], ADVICE, -(1 << 23), 1 << 23)).expect("x");
    one(
        &mut g,
        Op::TableMap {
            table: TableId(SILU),
        },
        &[x],
        lookup(&[4], SILU),
    );
    assert_eq!(
        refused(&g.finish(), &specs),
        range(0, 0, "a key lies in its table's keys")
    );

    // Scores at 2^21; and scores of spread 2^16, one key past a 16-bit Exp table's.
    let narrow = GeneratedTable::new(
        TableId(4),
        TableKind::Exp {
            key_bits: 16,
            sigma: SIGMA,
            scale: Rational::new(1, 64).expect("a scale"),
        },
    )
    .expect("an Exp table");
    let mut with_narrow = specs.clone();
    with_narrow.push(narrow.spec().clone());
    for (table, lo, hi, rule) in [
        (EXP, 0, 1 << 21, "a score lies in (−2^21, 2^21)"),
        (
            4,
            -(1 << 15),
            1 << 15,
            "the scores' spread fits the Exp table's keys",
        ),
    ] {
        let mut g = GraphBuilder::new();
        let s = g.input(decl(&[2, 2], ADVICE, lo, hi)).expect("scores");
        g.node(
            Op::SoftmaxRow {
                exp_table: TableId(table),
                sigma: SIGMA,
            },
            &[s],
            vec![lookup(&[2, 2], table), out(&[2], ADVICE)],
        )
        .expect("a softmax");
        assert_eq!(refused(&g.finish(), &with_narrow), range(0, 0, rule));
    }

    // A sum of squares whose epsilon reaches 2^55.
    let mut g = GraphBuilder::new();
    let u = g.input(decl(&[2], ADVICE, 0, (1 << 55) - 3)).expect("u");
    one(
        &mut g,
        Op::RsqrtRow {
            kappa: 16,
            eps_int: 3,
        },
        &[u],
        out(&[2], ADVICE),
    );
    assert_eq!(
        refused(&g.finish(), &specs),
        range(0, 0, "u + e lies in [0, 2^55)")
    );

    // A table the specifications lack is a bug, not a range.
    let mut g = GraphBuilder::new();
    let x = g.input(decl(&[2], ADVICE, 0, 1)).expect("x");
    one(
        &mut g,
        Op::TableMap { table: TableId(99) },
        &[x],
        lookup(&[2], 99),
    );
    let error = refused(&g.finish(), &specs);
    assert_eq!(error.code(), ErrorCode::Internal);
}
