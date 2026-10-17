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

mod common;

use common::{ADVICE, Lookup, WEIGHT, check_run, decl, elements, evaluate, lookup, one, out, virt};
use proptest::prelude::*;
use tessera_codec::Rational;
use tessera_codec::error::ErrorCode;
use tessera_compiler::{CompileError, HALF_FIELD, Range, analyze};
use tessera_ir::{
    Broadcast, Graph, GraphBuilder, NodeId, NodeKind, Op, TableId, TensorId, WeightRef, WeightType,
};
use tessera_tables::{GeneratedTable, RopeTable, Table, TableKind, TableSpec};

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
        Op::Add | Op::Sub | Op::Mul { .. } => {
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
            Op::Mul { broadcast: Broadcast::Trailing },
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
            dim: d,
        },
        &[squares],
        out(&[n], ADVICE),
    );
    let scaled = one(
        &mut g,
        Op::Mul {
            broadcast: Broadcast::Trailing,
        },
        &[residual, rho],
        virt(&[n, d]),
    );
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
    let attention = one(
        &mut g,
        Op::Mul {
            broadcast: Broadcast::Trailing,
        },
        &[pv, softmax[1]],
        virt(&[n, dh]),
    );
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
    let product = one(
        &mut g,
        Op::Mul {
            broadcast: Broadcast::Trailing,
        },
        &[a, u8],
        virt(&[n, ff]),
    );
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
    let y = one(
        &mut g,
        Op::Mul {
            broadcast: Broadcast::Trailing,
        },
        &[pv, r],
        virt(&[n, 4]),
    );
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
    let others = TensorId(1);
    let values = evaluate(
        &graph,
        &tables,
        vec![
            (scores, s),
            (others, vec![0; (n * n) as usize]),
            (v, vec![-128; (n * 4) as usize]),
        ],
    );
    assert_eq!(values[y.0 as usize].iter().min(), Some(&-(1i128 << 39)));

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
    let classes = g.input(decl(&[3], WEIGHT, 0, 2)).expect("classes");
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

    // Scores one past 2^21; and scores of spread 2^16, one key past a 16-bit Exp table's.
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
        (EXP, 0, (1 << 21) + 1, "a score lies in [−2^21, 2^21]"),
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

    // A sum of squares whose epsilon reaches 2^55; then roots with no bound (e = 0), or past one
    // lookup: d · 2^{2κ} / e of 2^94 has a root of 2^47, and one of 2^64 a root of 2^32, 33 bits.
    let roots = "the roots' range check covers every row: e ≥ 1, and w ≤ 32";
    for (hi, kappa, eps, dim, rule) in [
        ((1 << 55) - 3, 16, 3, 16, "u + e lies in [0, 2^55)"),
        (1 << 20, 16, 0, 16, roots),
        (1 << 20, 31, 1, u32::MAX, roots),
        (1 << 20, 17, 1, 1 << 30, roots),
    ] {
        let mut g = GraphBuilder::new();
        let u = g.input(decl(&[2], ADVICE, 0, hi)).expect("u");
        one(
            &mut g,
            Op::RsqrtRow {
                kappa,
                eps_int: eps,
                dim,
            },
            &[u],
            out(&[2], ADVICE),
        );
        assert_eq!(refused(&g.finish(), &specs), range(0, 0, rule));
    }

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

// ----- Classed tables, the one-hot fact, and the bounds at their edges --------------------------

#[test]
fn a_classed_table_takes_the_channel_classes_and_only_it() {
    let silu = GeneratedTable::new(
        TableId(6),
        TableKind::Silu {
            b: 12,
            r_by_class: vec![3, 5],
            in_exp: -4,
            out_exp: -3,
        },
    )
    .expect("a Silu table of two classes");
    let (mut tables, mut specs) = layer_tables();
    specs.push(silu.spec().clone());
    tables.push(Box::new(silu));
    let build = |table: u32, with_classes: bool, last_class: i128| {
        let mut g = GraphBuilder::new();
        let x = g.input(decl(&[2, 3], ADVICE, -2048, 2047)).expect("keys");
        let mut inputs = vec![x];
        if with_classes {
            inputs.push(g.input(decl(&[3], WEIGHT, 0, last_class)).expect("classes"));
        }
        one(
            &mut g,
            Op::TableMap {
                table: TableId(table),
            },
            &inputs,
            lookup(&[2, 3], table),
        );
        (g.finish(), inputs)
    };
    // Every key of each class through the interpreter: values inside the table's.
    let (graph, inputs) = build(6, true, 1);
    for classes in [[0, 1, 1], [1, 0, 0]] {
        let keys = (0..4096i128).step_by(683).map(|k| k - 2048);
        for start in keys {
            let values = vec![
                (
                    inputs[0],
                    (0..6).map(|i| (start + 701 * i) % 2048).collect(),
                ),
                (inputs[1], classes.to_vec()),
            ];
            check_run(&graph, &tables, &specs, values);
        }
    }
    // A class past the table's two.
    let (graph, _) = build(6, true, 2);
    assert_eq!(
        refused(&graph, &specs),
        CompileError::Range {
            node: NodeId(0),
            input: 1,
            rule: "a channel class indexes the table's classes",
        }
    );
    // The classes missing for a classed table, or given to a table of one class: a lowering bug.
    for (table, with_classes) in [(6, false), (SILU, true)] {
        let (graph, _) = build(table, with_classes, 0);
        let error = refused(&graph, &specs);
        assert_eq!(
            error,
            CompileError::Table {
                node: NodeId(0),
                table: TableId(table),
                rule: "the channel classes come with a table of several classes, and only with one",
            }
        );
        assert_eq!(error.code(), ErrorCode::Internal);
    }
}

#[test]
fn a_one_hot_vector_selects_a_column_within_the_weights_range() {
    let vocab = 300u32;
    let mut g = GraphBuilder::new();
    let t = g
        .input(decl(&[4], ADVICE, 0, i128::from(vocab) - 1))
        .expect("tokens");
    let h = one(&mut g, Op::OneHot { vocab }, &[t], out(&[4, vocab], ADVICE));
    let embed = WeightRef {
        ty: WeightType::Embed,
        layer: 0,
    };
    let e = g.weight(embed, 8, vocab).expect("the embedding");
    let x = one(
        &mut g,
        Op::MatMulWeight { weight: embed },
        &[h, e],
        virt(&[4, 8]),
    );
    // The same product over an advice vector of 0s and 1s: nothing makes it one-hot.
    let bits = g.input(decl(&[4, vocab], ADVICE, 0, 1)).expect("bits");
    let y = one(
        &mut g,
        Op::MatMulWeight { weight: embed },
        &[bits, e],
        virt(&[4, 8]),
    );
    let graph = g.finish();
    let weights: Vec<i128> = (0..8 * vocab)
        .map(|i| if i % 3 == 0 { -127 } else { 127 })
        .collect();
    let tokens = vec![0, 299, 150, 7];
    let (ranges, values) = check_run(
        &graph,
        &[],
        &[],
        vec![
            (t, tokens.clone()),
            (e, weights.clone()),
            (bits, vec![1; (4 * vocab) as usize]),
        ],
    );
    assert_eq!(ranges.of(x), Some(Range::new(-127, 127)));
    let reach = 127 * i128::from(vocab);
    assert_eq!(ranges.of(y), Some(Range::new(-reach, reach)));
    // Each row is the token's column of the embedding.
    for (row, &token) in tokens.iter().enumerate() {
        for c in 0..8usize {
            let column = weights[c * vocab as usize + token as usize];
            assert_eq!(values[x.0 as usize][row * 8 + c], column);
        }
    }
}

#[test]
fn int8_heads_of_width_128_reach_2_21_and_their_scores_pass() {
    // q = k = −128 in every coordinate gives 2^21 exactly: the bound is inclusive.
    let (tables, specs) = layer_tables();
    let n = 3u32;
    let mut g = GraphBuilder::new();
    let q = g
        .input(decl(&[n, 128], ADVICE, -128, 127))
        .expect("queries");
    let k = g.input(decl(&[n, 128], ADVICE, -128, 127)).expect("keys");
    let scores = one(
        &mut g,
        Op::MatMulAct {
            transpose_rhs: true,
        },
        &[q, k],
        virt(&[n, n]),
    );
    g.node(
        Op::SoftmaxRow {
            exp_table: TableId(EXP),
            sigma: SIGMA,
        },
        &[scores],
        vec![lookup(&[n, n], EXP), out(&[n], ADVICE)],
    )
    .expect("a softmax");
    let graph = g.finish();
    let corner = vec![-128; (n * 128) as usize];
    let (ranges, values) = check_run(
        &graph,
        &tables,
        &specs,
        vec![(q, corner.clone()), (k, corner)],
    );
    assert_eq!(
        ranges.of(scores),
        Some(Range::new(-(1 << 21) + (1 << 14), 1 << 21))
    );
    assert!(values[scores.0 as usize].iter().all(|&x| x == 1 << 21));
}

#[test]
fn a_root_of_32_bits_is_one_lookup() {
    // d · 2^{2κ} / e = 2^62: the largest root is 2^31, and its check the widest one lookup takes.
    let mut g = GraphBuilder::new();
    let u = g.input(decl(&[1], ADVICE, 0, 1 << 40)).expect("u");
    let rho = one(
        &mut g,
        Op::RsqrtRow {
            kappa: 16,
            eps_int: 1,
            dim: 1 << 30,
        },
        &[u],
        out(&[1], ADVICE),
    );
    let (ranges, values) = check_run(&g.finish(), &[], &[], vec![(u, vec![0])]);
    assert_eq!(values[rho.0 as usize], [1 << 31]);
    assert_eq!(ranges.of(rho), Some(Range::new(0, (1 << 32) - 1)));
}

#[test]
fn the_root_range_is_the_bit_length_of_the_largest_honest_root() {
    // RFC-0022: a zero sum gives the largest root, ⌊√(d · 2^{2κ} / e)⌋, and the analysis's range
    // is exactly its bit length, so every honest root passes the check and no wider check is
    // asked for. A sum of 4 · d, the root's other extreme here, stays inside too.
    for dim in [1u32, 4, 16, 17, 4096, 16384, 1 << 20] {
        for eps in [1u64, 2, 3, 1023, 1024, 1025, 4096, 4097, 1 << 30] {
            let mut g = GraphBuilder::new();
            let u = g.input(decl(&[2], ADVICE, 0, 1 << 40)).expect("u");
            let rho = one(
                &mut g,
                Op::RsqrtRow {
                    kappa: 16,
                    eps_int: eps,
                    dim,
                },
                &[u],
                out(&[2], ADVICE),
            );
            let graph = g.finish();
            let sums = vec![0, 4 * i128::from(dim)];
            let (ranges, values) = check_run(&graph, &[], &[], vec![(u, sums)]);
            let largest = values[rho.0 as usize][0];
            let Some(Range::Bounded(range)) = ranges.of(rho) else {
                unreachable!("a root's range is bounded")
            };
            // [0, 2^w − 1] with w the bit length of the largest root.
            assert_eq!(range.lo, 0);
            let w = 128 - largest.leading_zeros();
            assert_eq!(range.hi, (1 << w) - 1, "d = {dim}, e = {eps}");
        }
    }
}
