// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gadget selection and table sizing (RFC-0006 'Gadget selection', RFC-0009).
//!
//! - **Widths.** Keys of `B_max`, `B_max + 1`, `2·B_max − 2` and `2·B_max − 1` bits take
//!   `Requant1`, `RequantSplit`, `RequantSplit` and `ERR-COMPILE-RANGE`. A table of several
//!   classes spends their index bits first, the split needs `s ≤ r − 1` for every class, and its
//!   low part must fit a table too.
//! - **The split computes the requantization.** The low bits pass their `Range` table, and the
//!   reduced table returns `rq(z, r)` on random keys of every width.
//! - **Sizing.** A lookup's table holds at most `2^{B_max}` rows, and `B_max` lies in `[1, 32]`.
//! - **Sharing.** Llama-3.1 8B's 32 layers read exactly the tables its first two do. `minimal`
//!   8B is refused where its down projection's key is too wide for its shift (#292).

mod common;

use common::{ADVICE, WEIGHT, decl, llama, lookup, one, out, virt};
use proptest::prelude::*;
use tessera_codec::Rational;
use tessera_codec::error::ErrorCode;
use tessera_compiler::{CompileError, Gadget, PublicModel, Selection, analyze, lower, select};
use tessera_ir::{
    GadgetId, GadgetPolicy, Graph, GraphBuilder, NodeId, Op, Profile, TableId, TensorId,
};
use tessera_tables::{GeneratedTable, Table, TableKind, TableSpec};

fn policy(max_log2_rows: u8) -> GadgetPolicy {
    GadgetPolicy {
        max_log2_rows,
        ..GadgetPolicy::default()
    }
}

/// One `Requant` of a key whose signed width is exactly `b`, with shift classes `shifts`.
fn requant(b: u32, shifts: &[u8]) -> Graph {
    let mut g = GraphBuilder::new();
    let half = 1i128 << (b - 1);
    let x = g
        .input(decl(&[2, 3], ADVICE, -half, half - 1))
        .expect("a key");
    let mut inputs = vec![x];
    if shifts.len() > 1 {
        let last = i128::try_from(shifts.len()).expect("small") - 1;
        inputs.push(g.input(decl(&[3], WEIGHT, 0, last)).expect("classes"));
    }
    let shift = g.shift_classes(shifts.to_vec()).expect("shifts");
    one(
        &mut g,
        Op::Requant {
            shift,
            lo: -128,
            hi: 127,
        },
        &inputs,
        out(&[2, 3], ADVICE),
    );
    g.finish()
}

fn selected(
    graph: &Graph,
    specs: &[TableSpec],
    policy: &GadgetPolicy,
) -> Result<Selection, CompileError> {
    let ranges = analyze(graph, specs).expect("the graph analyzes");
    select(graph, &ranges, specs, policy)
}

fn kind(selection: &Selection, table: TableId) -> &TableKind {
    &selection
        .tables
        .iter()
        .find(|t| t.id == table)
        .expect("a table")
        .kind
}

fn refused(rule: &'static str) -> CompileError {
    CompileError::Range {
        node: NodeId(0),
        input: 0,
        rule,
    }
}

#[test]
fn requant_keys_take_their_gadget_by_width_at_each_boundary() {
    let default = GadgetPolicy::default();
    // b = B_max: one lookup of the whole key.
    let selection = selected(&requant(32, &[8]), &[], &default).expect("selects");
    let Gadget::Requant1 { table } = selection.gadgets[0].1 else {
        panic!("Requant1 expected");
    };
    assert_eq!(
        kind(&selection, table),
        &TableKind::Requant {
            b: 32,
            r_by_class: vec![8],
            lo: -128,
            hi: 127
        }
    );
    // b = B_max + 1 and 2·B_max − 2: split, the low part `b − B_max` bits.
    for (b, r, split) in [(33, 8, 1), (62, 31, 30)] {
        let selection = selected(&requant(b, &[r]), &[], &default).expect("selects");
        let Gadget::RequantSplit {
            split: s,
            low,
            high,
        } = selection.gadgets[0].1
        else {
            panic!("RequantSplit expected at b = {b}");
        };
        assert_eq!(s, split);
        assert_eq!(kind(&selection, low), &TableKind::Range { bits: split });
        assert_eq!(
            kind(&selection, high),
            &TableKind::Requant {
                b: 32,
                r_by_class: vec![r - split],
                lo: -128,
                hi: 127
            }
        );
    }
    // b = 2·B_max − 1: past what a split keeps as true integers.
    let error = selected(&requant(63, &[40]), &[], &default).expect_err("refused");
    assert_eq!(error, refused("a requantization key has at most 62 bits"));
    assert_eq!(error.code(), ErrorCode::CompileRange);
}

#[test]
fn a_tables_classes_take_key_bits_before_the_split() {
    let default = GadgetPolicy::default();
    // Two classes take one bit, three or four take two.
    for (shifts, whole) in [
        (&[8u8, 9][..], 31),
        (&[8, 9, 10][..], 30),
        (&[8, 9, 10, 11][..], 30),
    ] {
        let selection = selected(&requant(whole, shifts), &[], &default).expect("selects");
        assert!(
            matches!(selection.gadgets[0].1, Gadget::Requant1 { .. }),
            "{shifts:?}"
        );
        let selection = selected(&requant(whole + 1, shifts), &[], &default).expect("selects");
        let Gadget::RequantSplit { split, high, .. } = selection.gadgets[0].1 else {
            panic!("RequantSplit expected for {shifts:?}");
        };
        assert_eq!(split, 1);
        let reduced: Vec<u8> = shifts.iter().map(|r| r - 1).collect();
        assert!(matches!(
            kind(&selection, high),
            TableKind::Requant { b, r_by_class, .. } if *b == whole as u8 && *r_by_class == reduced
        ));
    }
}

#[test]
fn a_split_keeps_its_rounding_only_below_every_classs_shift() {
    // b = 40 splits 8 bits: a shift of 9 keeps the rounding bit above them, one of 8 does not.
    let default = GadgetPolicy::default();
    let rule = "a split key keeps its rounding: s ≤ r − 1 for every class";
    assert!(matches!(
        selected(&requant(40, &[9]), &[], &default)
            .expect("selects")
            .gadgets[0]
            .1,
        Gadget::RequantSplit { split: 8, .. }
    ));
    assert_eq!(
        selected(&requant(40, &[8]), &[], &default),
        Err(refused(rule))
    );
    // The least shift decides: 39 bits and a class bit split 8.
    assert_eq!(
        selected(&requant(39, &[12, 8]), &[], &default),
        Err(refused(rule))
    );
    assert!(selected(&requant(39, &[12, 9]), &[], &default).is_ok());
}

#[test]
fn a_split_keys_low_part_fits_a_table_of_b_max() {
    // B_max = 20: a key of 40 bits splits 20 low bits, one of 41 would need 21.
    let small = policy(20);
    assert!(matches!(
        selected(&requant(40, &[30]), &[], &small)
            .expect("selects")
            .gadgets[0]
            .1,
        Gadget::RequantSplit { split: 20, .. }
    ));
    assert_eq!(
        selected(&requant(41, &[30]), &[], &small),
        Err(refused(
            "a split key's low part fits a table of 2^{B_max} rows"
        ))
    );
}

#[test]
fn b_max_lies_between_1_and_32() {
    let graph = requant(8, &[4]);
    for bad in [0, 33] {
        let error = selected(&graph, &[], &policy(bad)).expect_err("refused");
        assert_eq!(
            error,
            CompileError::Usage {
                rule: "B_max lies in [1, 32], the most a Range table holds"
            }
        );
        assert_eq!(error.code(), ErrorCode::Usage);
    }
    assert!(selected(&graph, &[], &policy(8)).is_ok());
}

#[test]
fn a_lookups_table_holds_at_most_2_b_max_rows() {
    let exp = GeneratedTable::new(
        TableId(0),
        TableKind::Exp {
            key_bits: 22,
            sigma: 16,
            scale: Rational::new(1, 2896).expect("a scale"),
        },
    )
    .expect("an Exp table");
    let specs = [exp.spec().clone()];
    let mut g = GraphBuilder::new();
    let scores = g.input(decl(&[4, 4], ADVICE, -1000, 1000)).expect("scores");
    g.node(
        Op::SoftmaxRow {
            exp_table: TableId(0),
            sigma: 16,
        },
        &[scores],
        vec![lookup(&[4, 4], 0), out(&[4], ADVICE)],
    )
    .expect("a softmax");
    let keys = g.input(decl(&[4], ADVICE, 0, 1000)).expect("keys");
    one(
        &mut g,
        Op::TableMap { table: TableId(0) },
        &[keys],
        lookup(&[4], 0),
    );
    let graph = g.finish();
    let selection = selected(&graph, &specs, &policy(22)).expect("22 bits fit");
    assert_eq!(
        selection.gadgets,
        [
            (NodeId(0), Gadget::SoftmaxRow { exp: TableId(0) }),
            (NodeId(1), Gadget::Lookup { table: TableId(0) }),
        ]
    );
    assert_eq!(
        selected(&graph, &specs, &policy(21)),
        Err(refused("a lookup's table has at most 2^{B_max} rows"))
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn the_split_tables_compute_the_requantization(
        b in 33u32..=62, extra in 1u8..=8, seed in any::<u64>(), lo in -200i32..=0, span in 0i32..=400,
    ) {
        // The least shift the split allows, a few bits more: the low part passes its Range table
        // and the reduced table returns rq(z, r, lo, hi) on keys across the width.
        let split = b - 32;
        let r = u8::try_from(split).expect("small") + extra;
        let hi = lo + span;
        let mut g = GraphBuilder::new();
        let half = 1i128 << (b - 1);
        let x = g.input(decl(&[1], ADVICE, -half, half - 1)).expect("a key");
        let shift = g.shift_classes(vec![r]).expect("shifts");
        one(&mut g, Op::Requant { shift, lo: i64::from(lo), hi: i64::from(hi) }, &[x], out(&[1], ADVICE));
        let selection = selected(&g.finish(), &[], &GadgetPolicy::default()).expect("selects");
        let Gadget::RequantSplit { split: s, low, high } = selection.gadgets[0].1 else {
            panic!("RequantSplit expected");
        };
        prop_assert_eq!(u32::from(s), split);
        let table = |id: TableId| {
            let spec = selection.tables.iter().find(|t| t.id == id).expect("a table");
            GeneratedTable::from_spec(spec).expect("the table regenerates")
        };
        let (low, high) = (table(low), table(high));
        let mut state = seed;
        for _ in 0..64 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let z = i128::from(state >> 1) % half * if state & 1 == 0 { 1 } else { -1 };
            let offset = z + half;
            let low_bits = offset % (1 << split);
            prop_assert!(low.lookup(i64::try_from(low_bits).expect("small"), 0).is_some());
            let reduced = i64::try_from(z >> split).expect("a reduced key");
            let expected = ((z + (1 << (r - 1))) >> r).clamp(i128::from(lo), i128::from(hi));
            prop_assert_eq!(i128::from(high.lookup(reduced, 0).expect("in the table")), expected);
        }
    }
}

#[test]
fn llama_8b_shares_every_table_across_its_layers() {
    let select_8b = |layers: u32| {
        let (arch, quant, silu) = llama(Profile::Servable, layers, 4096, 32, 8, 14_336);
        let model = PublicModel {
            arch: &arch,
            profile: Profile::Servable,
            quant: &quant,
            silu_in_exponents: &silu,
        };
        let lowering = lower(model, 16).expect("8B's shape lowers");
        let ranges = analyze(&lowering.graph, &lowering.tables).expect("analyzes");
        select(
            &lowering.graph,
            &ranges,
            &lowering.tables,
            &GadgetPolicy::default(),
        )
        .expect("every gadget fits")
    };
    let two = select_8b(2);
    let all = select_8b(32);
    // The same tables whatever the depth: RoPE, Exp and Silu from the lowering, then the
    // requantizations. The norms, RoPE and the attention output split; the rest take one lookup.
    assert_eq!(all.tables, two.tables);
    let splits = |s: &Selection| {
        s.gadgets
            .iter()
            .filter(|(_, g)| matches!(g, Gadget::RequantSplit { .. }))
            .count()
    };
    // Per layer: the two norms, RoPE's queries and keys, the attention output; then the final norm.
    assert_eq!(splits(&all), 5 * 32 + 1);
    assert_eq!(splits(&two), 5 * 2 + 1);
    // Gadget ids are positions in node order.
    let (node, gadget) = all.gadgets[3];
    assert_eq!(all.of(node), Some((GadgetId(3), gadget)));
    assert_eq!(all.of(NodeId(u32::MAX)), None);
    // A second selection of the same graph is the same.
    assert_eq!(select_8b(2), two);
}

#[test]
fn minimal_8b_is_refused_where_its_down_projection_needs_a_wider_split() {
    // The down projection reads a · u unrequantized: its residual key reaches about 2^55, a split
    // of 23 bits, against a shift of 17 (#292).
    let (arch, quant, silu) = llama(Profile::Minimal, 1, 4096, 32, 8, 14_336);
    let model = PublicModel {
        arch: &arch,
        profile: Profile::Minimal,
        quant: &quant,
        silu_in_exponents: &silu,
    };
    let lowering = lower(model, 16).expect("the shape lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("analyzes");
    let error = select(
        &lowering.graph,
        &ranges,
        &lowering.tables,
        &GadgetPolicy::default(),
    )
    .expect_err("refused");
    let CompileError::Range { node, rule, .. } = error else {
        panic!("a range refusal");
    };
    assert_eq!(
        rule,
        "a split key keeps its rounding: s ≤ r − 1 for every class"
    );
    // The refused node is the layer's last requantization, the down projection's residual.
    let last = lowering
        .graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, Op::Requant { .. }))
        .nth_back(1)
        .expect("the residual's requantization");
    assert_eq!(node, last.id);
}

#[test]
fn nodes_without_a_gadget_are_left_out() {
    let mut g = GraphBuilder::new();
    let x = g.input(decl(&[2], ADVICE, 0, 9)).expect("x");
    let y = one(&mut g, Op::Add, &[x, x], virt(&[2]));
    let _: TensorId = y;
    let selection = selected(&g.finish(), &[], &GadgetPolicy::default()).expect("selects");
    assert!(selection.gadgets.is_empty());
    assert!(selection.tables.is_empty());
}
