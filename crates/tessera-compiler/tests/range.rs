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
use tessera_compiler::anchor::{
    SHA256_IV, SHA256_MASKS, Sha256Layout, sha256, sha256_bits, sha256_inputs, sha256_pad,
};
use tessera_compiler::{
    CompileError, HALF_FIELD, Range, analyze, analyze_fragment, check_profile, select,
};
use tessera_ir::{
    BoolFn, Broadcast, GadgetPolicy, Graph, GraphBuilder, NodeId, NodeKind, Op, Profile, TableId,
    TensorId, WeightRef, WeightType,
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
    let ranges = analyze_fragment(&layer.graph, &specs).expect("analyzes");
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
    let ranges = analyze_fragment(&graph, &specs).expect("analyzes");
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
    let product = analyze_fragment(&unrelated, &specs).expect("analyzes");
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
    analyze_fragment(graph, specs).expect_err("refused")
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
    let ranges = analyze_fragment(&graph, &[]);
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

// ----- Bits (RFC-0066) ----------------------------------------------------------------------------

/// `x`, `[2, 2]`, read directly (`mover` 0) or moved unchanged by `Reshape`, `Slice`,
/// `Transpose`, `GatherPublic` or `Expand` (1 to 5), with the moved shape.
fn moved_by(g: &mut GraphBuilder, x: TensorId, mover: u8) -> (TensorId, Vec<u32>) {
    let (op, shape) = match mover {
        0 => return (x, vec![2, 2]),
        1 => (Op::Reshape, vec![4]),
        2 => (
            Op::Slice {
                axis: 1,
                start: 1,
                len: 1,
            },
            vec![2, 1],
        ),
        3 => (Op::Transpose { perm: vec![1, 0] }, vec![2, 2]),
        4 => (
            Op::GatherPublic {
                index: g.constant(vec![1, 0]).expect("rows"),
            },
            vec![2, 2],
        ),
        _ => (Op::Expand { axis: 0, len: 1 }, vec![1, 2, 2]),
    };
    (one(g, op, &[x], virt(&shape)), shape)
}

/// `x + y` of two advice inputs declared `[lo, hi]` each, `[2]`.
fn sum_of(g: &mut GraphBuilder, lo: i128, hi: i128) -> TensorId {
    sum_of_kind(g, ADVICE, lo, hi)
}

/// `x + y` of two inputs of `kind` declared `[lo, hi]` each, `[2]`.
fn sum_of_kind(g: &mut GraphBuilder, kind: NodeKind, lo: i128, hi: i128) -> TensorId {
    let x = g.input(decl(&[2], kind, lo, hi)).expect("x");
    let y = g.input(decl(&[2], kind, lo, hi)).expect("y");
    one(g, Op::Add, &[x, y], virt(&[2]))
}

#[test]
fn each_bit_rule_refuses_a_graph_that_breaks_only_it() {
    let range = |node: u32, input: usize, rule: &'static str| CompileError::Range {
        node: NodeId(node),
        input,
        rule,
    };
    let width = "the integer lies in [0, 2^{low + high})";
    // BitDecompose: a virtual integer one past its width, or one below 0, against one inside.
    for (lo, hi, low, high, refused_at) in [
        (0, 128, 8, 0, Some(1)),
        (0, 127, 8, 0, None),
        (-1, 127, 8, 0, Some(1)),
        (0, 1 << 34, 32, 3, Some(1)),
        (0, (1 << 34) - 1, 32, 3, None),
        (0, 1 << 61, 61, 1, Some(1)),
        (0, (1 << 61) - 1, 32, 30, None),
    ] {
        let mut g = GraphBuilder::new();
        let x = sum_of(&mut g, lo, hi);
        let mut outputs = vec![out(&[2, u32::from(low)], ADVICE)];
        if high > 0 {
            outputs.push(out(&[2, u32::from(high)], ADVICE));
        }
        g.node(Op::BitDecompose { low, high }, &[x], outputs)
            .expect("a decomposition");
        let result = analyze_fragment(&g.finish(), &[]);
        match refused_at {
            Some(node) => assert_eq!(result.expect_err("refused"), range(node, 0, width)),
            None => {
                let ranges = result.expect("inside the width");
                assert!(ranges.all()[3..].iter().all(|r| *r == Range::new(0, 1)));
            }
        }
    }
    // A virtual integer with no bound is a possible wrap; a committed input of any range is its
    // own bits' to check (Lemma 1's converse), but a public one is not.
    let mut g = GraphBuilder::new();
    let x = g
        .input(decl(&[2], ADVICE, -(1 << 100), 1 << 100))
        .expect("x");
    let square = one(
        &mut g,
        Op::Mul {
            broadcast: Broadcast::Trailing,
        },
        &[x, x],
        virt(&[2]),
    );
    let cube = one(
        &mut g,
        Op::Mul {
            broadcast: Broadcast::Trailing,
        },
        &[square, x],
        virt(&[2]),
    );
    g.node(
        Op::BitDecompose { low: 8, high: 0 },
        &[cube],
        vec![out(&[2, 8], ADVICE)],
    )
    .expect("a decomposition");
    assert_eq!(
        refused(&g.finish(), &[]),
        CompileError::Wrap {
            node: NodeId(2),
            input: 0
        }
    );
    for (kind, refused_here) in [(ADVICE, false), (WEIGHT, false), (NodeKind::Public, true)] {
        let mut g = GraphBuilder::new();
        let x = g.input(decl(&[2], kind, -(1 << 40), 1 << 40)).expect("x");
        g.node(
            Op::BitDecompose { low: 8, high: 0 },
            &[x],
            vec![out(&[2, 8], ADVICE)],
        )
        .expect("a decomposition");
        let result = analyze_fragment(&g.finish(), &[]);
        assert_eq!(result.is_err(), refused_here, "{kind:?}");
        if refused_here {
            assert_eq!(result.expect_err("refused"), range(0, 0, width));
        }
    }
    // A committed byte declared [0, 255] that nine bits would admit up to 511, past what its
    // other readers trust: a key 256 + k' would pass as a Bitwise8 operand's low byte.
    let declared = "a committed integer's declared range holds [0, 2^{low + high})";
    for (lo, low, high, refused_here) in [
        (0, 8, 1, true),
        (0, 9, 0, true),
        (1, 8, 0, true),
        (0, 8, 0, false),
        (-3, 8, 0, false),
    ] {
        let mut g = GraphBuilder::new();
        let x = g.input(decl(&[2], ADVICE, lo, 255)).expect("a byte");
        let mut outputs = vec![out(&[2, u32::from(low)], ADVICE)];
        if high > 0 {
            outputs.push(out(&[2, u32::from(high)], ADVICE));
        }
        g.node(Op::BitDecompose { low, high }, &[x], outputs)
            .expect("a decomposition");
        let result = analyze_fragment(&g.finish(), &[]);
        if refused_here {
            assert_eq!(
                result.expect_err("refused"),
                range(0, 0, declared),
                "{low}+{high}"
            );
        } else {
            result.expect("the bits prove the declared range");
        }
    }
    // BitFn: each input, alone, a sum of two bits, [0, 2].
    let bit = "a Boolean function's input is a bit";
    for f in [
        BoolFn::Xor,
        BoolFn::Xor3,
        BoolFn::And,
        BoolFn::Ch,
        BoolFn::Maj,
    ] {
        for k in 0..f.arity() {
            let mut g = GraphBuilder::new();
            let b = g.input(decl(&[2], NodeKind::Public, 0, 1)).expect("a bit");
            let wide = sum_of_kind(&mut g, NodeKind::Public, 0, 1);
            let inputs: Vec<TensorId> = (0..f.arity())
                .map(|i| if i == k { wide } else { b })
                .collect();
            one(&mut g, Op::BitFn { f }, &inputs, virt(&[2]));
            assert_eq!(
                refused(&g.finish(), &[]),
                range(1, k, bit),
                "{f:?} input {k}"
            );
        }
        // Public bits, and bits a decomposition proves, pass.
        let mut g = GraphBuilder::new();
        let b = g.input(decl(&[2], NodeKind::Public, 0, 1)).expect("a bit");
        let y = one(&mut g, Op::BitFn { f }, &vec![b; f.arity()], virt(&[2]));
        let ranges = analyze_fragment(&g.finish(), &[]).expect("bits");
        assert_eq!(ranges.of(y), Some(Range::new(0, 1)));
        let mut g = GraphBuilder::new();
        let byte = g.input(decl(&[2], ADVICE, 0, 255)).expect("a byte");
        let bits = g
            .node(
                Op::BitDecompose { low: 8, high: 0 },
                &[byte],
                vec![out(&[2, 8], ADVICE)],
            )
            .expect("bits")[0];
        one(
            &mut g,
            Op::BitFn { f },
            &vec![bits; f.arity()],
            virt(&[2, 8]),
        );
        analyze_fragment(&g.finish(), &[]).expect("proven bits");
    }
    // A committed input declared [0, 1], read directly or moved by any of the five moves, is no
    // proven bit; and a byte declared [0, 255], read or moved the same ways, under nine bits is
    // refused, while eight bits prove it.
    let proven = "a Boolean function's input is a bit some operation proves";
    let declared = "a committed integer's declared range holds [0, 2^{low + high})";
    for mover in 0..6 {
        for (lo, hi, op) in [(0, 1, None), (0, 255, Some(1u8)), (0, 255, Some(0))] {
            let mut g = GraphBuilder::new();
            let c = g.input(decl(&[2, 2], ADVICE, lo, hi)).expect("committed");
            let (moved, shape) = moved_by(&mut g, c, mover);
            let node = u32::from(mover > 0);
            let result = match op {
                None => {
                    let p = g
                        .input(decl(&shape, NodeKind::Public, 0, 1))
                        .expect("a bit");
                    one(
                        &mut g,
                        Op::BitFn { f: BoolFn::Xor },
                        &[p, moved],
                        virt(&shape),
                    );
                    analyze_fragment(&g.finish(), &[])
                        .map(|_| ())
                        .map_err(|e| (e, range(node, 1, proven)))
                }
                Some(high) => {
                    let with = |bits: u32| {
                        let mut s = shape.clone();
                        s.push(bits);
                        out(&s, ADVICE)
                    };
                    let mut outputs = vec![with(8)];
                    if high > 0 {
                        outputs.push(with(1));
                    }
                    g.node(Op::BitDecompose { low: 8, high }, &[moved], outputs)
                        .expect("a decomposition");
                    let result = analyze_fragment(&g.finish(), &[]);
                    if high == 0 {
                        result.expect("eight bits prove the declared byte");
                        continue;
                    }
                    result
                        .map(|_| ())
                        .map_err(|e| (e, range(node, 0, declared)))
                }
            };
            let (error, expected) = result.expect_err("refused");
            assert_eq!(error, expected, "mover {mover}, {op:?}");
        }
    }
    // AssertEqual: an input outside the centered field range, or with no bound.
    let integer = "a true integer lies in (−(P−1)/2, (P−1)/2)";
    for k in 0..2 {
        let mut g = GraphBuilder::new();
        let fine = g.input(decl(&[2], ADVICE, 0, 1)).expect("a bit");
        let wide = g.input(decl(&[2], ADVICE, 0, HALF_FIELD)).expect("wide");
        let inputs = if k == 0 { [wide, fine] } else { [fine, wide] };
        g.node(Op::AssertEqual, &inputs, Vec::new())
            .expect("an equality");
        assert_eq!(refused(&g.finish(), &[]), range(0, k, integer));
    }
}

#[test]
fn the_sha256_circuit_keeps_every_tensor_in_range() {
    // FIPS 180-4's "abc" (Appendix B.1), its first two bytes committed: every value the exact
    // interpreter computes lies in the range the analysis gives it, and the chaining value is
    // the digest.
    let layout = Sha256Layout {
        blocks: 1,
        messages: 2,
        committed: vec![0, 1],
    };
    let mut g = GraphBuilder::new();
    let inputs: Vec<TensorId> = sha256_inputs(&layout)
        .into_iter()
        .map(|d| g.input(d).expect("an input"))
        .collect();
    let sha = sha256(&mut g, &layout, &inputs).expect("the circuit");
    let graph = g.finish();
    let padded = [sha256_pad(b"abc"), sha256_pad(b"abd")];
    let rows: Vec<&[u8]> = padded.iter().map(Vec::as_slice).collect();
    let iv: Vec<u8> = SHA256_IV.iter().flat_map(|w| w.to_be_bytes()).collect();
    let wide = |v: Vec<i64>| v.into_iter().map(i128::from).collect::<Vec<_>>();
    let values = vec![
        (inputs[0], wide(layout.public_bits(&rows))),
        (inputs[1], wide(sha256_bits(&[&iv, &iv]))),
        (
            inputs[2],
            wide(SHA256_MASKS.iter().flatten().copied().collect()),
        ),
        (
            inputs[3],
            wide(
                layout
                    .committed
                    .iter()
                    .flat_map(|&p| rows.iter().map(move |m| i64::from(m[p as usize])))
                    .collect(),
            ),
        ),
        (inputs[4], wide(layout.place())),
    ];
    let (ranges, values) = check_run(&graph, &[], &[], values);
    // `BitDecompose { 8, 0 }` reads the committed bytes directly, so it proves them: the circuit
    // is a proof's graph, not a fragment (RFC-0006 'Committed graph inputs').
    analyze(&graph, &[]).expect("the bytes' decomposition proves them");
    // Until the anchor's relations are scheduled (#526), the compiler takes the circuit to no
    // proof, and it is no part of a model.
    assert_eq!(
        select(&graph, &ranges, &[], &GadgetPolicy::default()).err(),
        Some(CompileError::Model {
            rule: "RFC-0066's bit operations have no proof schedule yet"
        })
    );
    let profile = check_profile(&graph, &ranges, &[], Profile::Minimal).expect_err("not a model");
    let CompileError::Profile { node, rule } = profile else {
        panic!("{profile:?}");
    };
    assert_eq!(rule, "S3: a forward pass holds only RFC-0005's operations");
    assert!(matches!(
        graph.nodes[node.0 as usize].op,
        Op::BitDecompose { .. } | Op::BitFn { .. } | Op::AssertEqual
    ));
    let digest: Vec<u8> = sha.chaining[0]
        .iter()
        .flat_map(|w| {
            let bits = &values[w.0 as usize];
            let word = (0..32).fold(0u32, |acc, i| {
                acc | (u32::try_from(bits[2 * i]).expect("a bit") << i)
            });
            word.to_be_bytes()
        })
        .collect();
    let expected = [
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22,
        0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00,
        0x15, 0xad,
    ];
    assert_eq!(digest, expected);
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

// ----- Committed graph inputs (RFC-0006 'Committed graph inputs', #527) -------------------------

fn unproven(tensor: u32) -> CompileError {
    CompileError::Unproven {
        tensor: TensorId(tensor),
        rule: "every element of a committed graph input is read by an operation that proves its \
               declared range",
    }
}

/// `BitDecompose { 8, 0 }` of `x`, of `shape`.
fn bytes_of(g: &mut GraphBuilder, x: TensorId, shape: &[u32]) {
    let mut bits = shape.to_vec();
    bits.push(8);
    g.node(
        Op::BitDecompose { low: 8, high: 0 },
        &[x],
        vec![out(&bits, ADVICE)],
    )
    .expect("a decomposition");
}

#[test]
fn a_committed_input_that_no_operation_proves_is_refused() {
    // A committed byte under `ConstAdd(0)`, then `BitDecompose { 8, 1 }`: the sum's [0, 255] is
    // inside nine bits, but those prove only x < 512, so the byte's [0, 255] is unproven.
    let mut g = GraphBuilder::new();
    let x = g.input(decl(&[2], ADVICE, 0, 255)).expect("a byte");
    let zero = g.constant(vec![0]).expect("0");
    let y = one(&mut g, Op::ConstAdd { c: zero }, &[x], virt(&[2]));
    g.node(
        Op::BitDecompose { low: 8, high: 1 },
        &[y],
        vec![out(&[2, 8], ADVICE), out(&[2, 1], ADVICE)],
    )
    .expect("bits");
    let graph = g.finish();
    let error = analyze(&graph, &[]).expect_err("refused");
    assert_eq!(error, unproven(0));
    assert_eq!(error.code(), ErrorCode::CompileRange);
    analyze_fragment(&graph, &[]).expect("a fragment takes its committed inputs as given");

    // A committed input no operation reads proves nothing, whatever its declared range; a weight
    // input takes the domain its publisher vouches for (RFC-0048), and a public one is known.
    for (kind, refused_here) in [(ADVICE, true), (WEIGHT, false), (NodeKind::Public, false)] {
        let mut g = GraphBuilder::new();
        let p = g.input(decl(&[2], NodeKind::Public, 0, 255)).expect("p");
        let x = g.input(decl(&[2], kind, 0, 255)).expect("x");
        bytes_of(&mut g, p, &[2]);
        let _ = x;
        let result = analyze(&g.finish(), &[]);
        if refused_here {
            assert_eq!(result.expect_err("refused"), unproven(1), "{kind:?}");
        } else {
            result.expect("needs no reader");
        }
    }

    // A `Bitwise8` lookup proves neither operand alone: its key a + 2^8·b binds a only once b is
    // a byte. A decomposed operand is proven, and the other is not.
    let specs = [GeneratedTable::new(
        TableId(0),
        TableKind::Bitwise8 {
            op: tessera_tables::BitOp::Xor,
        },
    )
    .expect("a table")
    .spec()
    .clone()];
    let mut g = GraphBuilder::new();
    let a = g.input(decl(&[2], ADVICE, 0, 255)).expect("a");
    let b = g.input(decl(&[2], ADVICE, 0, 255)).expect("b");
    one(
        &mut g,
        Op::Bitwise { table: TableId(0) },
        &[a, b],
        out(&[2], NodeKind::LookupOutput { table: TableId(0) }),
    );
    bytes_of(&mut g, b, &[2]);
    assert_eq!(
        analyze(&g.finish(), &specs).expect_err("refused"),
        unproven(0)
    );
}

#[test]
fn a_one_hot_proves_a_declared_range_that_holds_its_vocabulary() {
    // `OneHot { vocab }`'s relations prove [0, vocab): tokens declared [0, vocab − 1] are proven.
    // Declared narrower at either end, some other reader would trust what they do not prove.
    // (Wider is `ERR-COMPILE-RANGE` at the node.) Through a move the tokens are not proven: the
    // row selector pins the moved tensor's padded rows, not the input's.
    for (lo, hi, moved, proven) in [
        (0, 9, false, true),
        (0, 8, false, false),
        (1, 9, false, false),
        (0, 9, true, false),
    ] {
        let mut g = GraphBuilder::new();
        let shape: &[u32] = if moved { &[2, 3] } else { &[6] };
        let tokens = g.input(decl(shape, ADVICE, lo, hi)).expect("tokens");
        let read = if moved {
            one(&mut g, Op::Reshape, &[tokens], virt(&[6]))
        } else {
            tokens
        };
        one(
            &mut g,
            Op::OneHot { vocab: 10 },
            &[read],
            out(&[6, 10], ADVICE),
        );
        let result = analyze(&g.finish(), &[]);
        if proven {
            result.expect("the relations prove the tokens");
        } else {
            assert_eq!(
                result.expect_err("refused"),
                unproven(0),
                "[{lo}, {hi}], {moved}"
            );
        }
    }
}

#[test]
fn each_move_carries_the_proof_to_the_elements_it_keeps() {
    // Read through one move, by `moved_by`'s numbering: directly, Reshape, Slice of column 1,
    // Transpose, GatherPublic of rows [1, 0], Expand. All but the slice keep every element.
    for mover in 0..6 {
        let mut g = GraphBuilder::new();
        let x = g.input(decl(&[2, 2], ADVICE, 0, 255)).expect("bytes");
        let (moved, shape) = moved_by(&mut g, x, mover);
        bytes_of(&mut g, moved, &shape);
        let result = analyze(&g.finish(), &[]);
        if mover == 2 {
            assert_eq!(result.expect_err("column 0 is unproven"), unproven(0));
        } else {
            result.expect("every element proven");
        }
    }
    // Two slices that keep both columns between them prove the input; two that keep column 1
    // twice do not. A gather of row 1 twice leaves row 0 unproven.
    for (starts, proven) in [([0, 1], true), ([1, 1], false)] {
        let mut g = GraphBuilder::new();
        let x = g.input(decl(&[2, 2], ADVICE, 0, 255)).expect("bytes");
        for start in starts {
            let column = one(
                &mut g,
                Op::Slice {
                    axis: 1,
                    start,
                    len: 1,
                },
                &[x],
                virt(&[2, 1]),
            );
            bytes_of(&mut g, column, &[2, 1]);
        }
        assert_eq!(analyze(&g.finish(), &[]).is_ok(), proven, "{starts:?}");
    }
    let mut g = GraphBuilder::new();
    let x = g.input(decl(&[2, 2], ADVICE, 0, 255)).expect("bytes");
    let index = g.constant(vec![1, 1]).expect("rows");
    let rows = one(&mut g, Op::GatherPublic { index }, &[x], virt(&[2, 2]));
    bytes_of(&mut g, rows, &[2, 2]);
    assert_eq!(analyze(&g.finish(), &[]).expect_err("row 0"), unproven(0));
}

/// A random move of a tensor of `shape`: its op and output shape.
fn random_move(g: &mut GraphBuilder, rng: &mut common::Seeded, shape: &[u32]) -> (Op, Vec<u32>) {
    let rank = shape.len();
    loop {
        match rng.next() % 5 {
            0 => return (Op::Reshape, vec![shape.iter().product()]),
            1 if rank >= 2 => {
                let mut perm: Vec<u8> = (0..rank as u8).collect();
                for i in (1..rank).rev() {
                    perm.swap(i, (rng.next() % (i as u64 + 1)) as usize);
                }
                let to = perm.iter().map(|&p| shape[usize::from(p)]).collect();
                return (Op::Transpose { perm }, to);
            }
            2 => {
                let axis = (rng.next() % rank as u64) as usize;
                let start = (rng.next() % u64::from(shape[axis])) as u32;
                let len = 1 + (rng.next() % u64::from(shape[axis] - start)) as u32;
                let mut to = shape.to_vec();
                to[axis] = len;
                let axis = axis as u8;
                return (Op::Slice { axis, start, len }, to);
            }
            3 => {
                let rows = 1 + rng.next() % 3;
                let index = (0..rows)
                    .map(|_| (rng.next() % u64::from(shape[0])) as i64)
                    .collect();
                let mut to = shape.to_vec();
                to[0] = rows as u32;
                let index = g.constant(index).expect("rows");
                return (Op::GatherPublic { index }, to);
            }
            4 if rank < 4 => {
                let axis = (rng.next() % (rank as u64 + 1)) as u8;
                let len = 1 + (rng.next() % 2) as u32;
                let mut to = shape.to_vec();
                to.insert(usize::from(axis), len);
                return (Op::Expand { axis, len }, to);
            }
            _ => {}
        }
    }
}

#[test]
fn the_analysis_accepts_exactly_the_graphs_whose_readers_reach_every_element() {
    // Random chains of moves from a committed [2, 3] input to one or two decompositions. The
    // interpreter, run on six distinct bytes, shows which elements reach a decomposition: the
    // analysis must accept exactly when all six do.
    let (mut accepted, mut refused) = (0, 0);
    for seed in 0..3000u64 {
        let mut rng = common::Seeded(seed);
        let mut g = GraphBuilder::new();
        let x = g.input(decl(&[2, 3], ADVICE, 0, 255)).expect("bytes");
        let mut readers = Vec::new();
        for _ in 0..1 + rng.next() % 2 {
            let (mut tensor, mut shape) = (x, vec![2u32, 3]);
            for _ in 0..rng.next() % 5 {
                let (op, to) = random_move(&mut g, &mut rng, &shape);
                tensor = one(&mut g, op, &[tensor], virt(&to));
                shape = to;
            }
            bytes_of(&mut g, tensor, &shape);
            readers.push(tensor);
        }
        let graph = g.finish();
        let values = evaluate(&graph, &[], vec![(x, (10..16).collect())]);
        let reached: std::collections::BTreeSet<i128> = readers
            .iter()
            .flat_map(|r| values[r.0 as usize].iter().copied())
            .collect();
        let every = reached.len() == 6;
        let result = analyze(&graph, &[]);
        assert_eq!(result.is_ok(), every, "seed {seed}: {result:?}");
        if every {
            accepted += 1;
        } else {
            assert_eq!(result.expect_err("refused"), unproven(0), "seed {seed}");
            refused += 1;
        }
    }
    // Both outcomes are exercised.
    assert!(
        accepted > 300 && refused > 300,
        "{accepted} accepted, {refused} refused"
    );
}

#[test]
fn the_aes_circuit_needs_its_key_and_plaintext_proven() {
    use tessera_compiler::anchor::{AesTables, aes128_ctr, aes128_ctr_inputs};
    let tables = AesTables {
        sbox: TableId(0),
        mul2: TableId(1),
        mul3: TableId(2),
        xor: TableId(3),
    };
    let specs: Vec<TableSpec> = [
        (tables.sbox, TableKind::AesSbox),
        (tables.mul2, TableKind::Gf256Mul { c: 2 }),
        (tables.mul3, TableKind::Gf256Mul { c: 3 }),
        (
            tables.xor,
            TableKind::Bitwise8 {
                op: tessera_tables::BitOp::Xor,
            },
        ),
    ]
    .into_iter()
    .map(|(id, kind)| {
        GeneratedTable::new(id, kind)
            .expect("a table")
            .spec()
            .clone()
    })
    .collect();
    // Alone, the key (input 0) and the plaintext (input 3) are `Bitwise` operands that nothing
    // proves bytes: a fragment. Decomposing them, as the SHA-256 key commitment does the key's,
    // proves them.
    for decomposed in [vec![], vec![0], vec![0, 3]] {
        let mut g = GraphBuilder::new();
        let inputs = aes128_ctr_inputs(2).map(|d| g.input(d).expect("an input"));
        aes128_ctr(&mut g, &tables, inputs, 2).expect("the circuit");
        for &k in &decomposed {
            let shape = if k == 0 { vec![16] } else { vec![16, 2] };
            bytes_of(&mut g, inputs[k], &shape);
        }
        let graph = g.finish();
        analyze_fragment(&graph, &specs).expect("the circuit's lookups take bytes");
        let result = analyze(&graph, &specs);
        match decomposed.len() {
            0 => assert_eq!(result.expect_err("the key"), unproven(inputs[0].0)),
            1 => assert_eq!(result.expect_err("the plaintext"), unproven(inputs[3].0)),
            _ => {
                result.expect("both proven");
            }
        }
    }
}
