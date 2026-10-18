// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gadget selection and table sizing (RFC-0006 'Gadget selection', RFC-0009, RFC-0024).
//!
//! - **Widths.** Keys of `B_max`, `B_max + 1`, `2·B_max − 2` and `2·B_max − 1` bits take
//!   `Requant1`, `RequantSplit`, `RequantSplit` and `ERR-COMPILE-RANGE`. A table of several
//!   classes spends their index bits first.
//! - **Too wide for the split.** A key whose split would pass a class's rounding bit, or whose
//!   low part would not fit a table, takes `RequantWide`, whose five tables must fit `B_max`. So
//!   does a key of `n` classes with `n·2^b ≥ P`, four classes at 62 bits (RFC-0009 Lemma 2).
//! - **The tables compute the requantization.** The split's two lookups and the wide gadget's five
//!   return `rq(z, r_c, lo, hi)` on random keys of every width and class, ends included.
//! - **Sizing.** A lookup's table holds at most `2^{B_max}` rows, and `B_max` lies in `[1, 32]`.
//! - **Llama.** 8B's 32 layers read exactly the tables its first two do. `minimal` 8B and 405B
//!   compile: their down projections' residual keys take `RequantWide`, with one class or four
//!   (#292).

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
    // b = B_max + 1 to 2·B_max − 2: split. RFC-0028 balances the two tables at `s = ⌈b/2⌉`,
    // inside Lemma 2's bounds: at least `b − B_max`, so the high table fits, and below the
    // rounding bit `r`.
    for (b, r, split) in [(33, 40, 17), (33, 8, 7), (48, 40, 24), (62, 31, 30)] {
        let selection = selected(&requant(b, &[r]), &[], &default).expect("selects");
        let Gadget::RequantSplit {
            split: s,
            low,
            high,
        } = selection.gadgets[0].1
        else {
            panic!("RequantSplit expected at b = {b}");
        };
        assert_eq!(s, split, "b = {b}, r = {r}");
        assert_eq!(kind(&selection, low), &TableKind::Range { bits: split });
        assert_eq!(
            kind(&selection, high),
            &TableKind::Requant {
                b: u8::try_from(b).expect("small") - split,
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
    // Two classes take one bit, three or four take two: the key of `whole` bits fits one table,
    // and one more bit splits.
    for (classes, whole) in [(2u8, 31u8), (3, 30), (4, 30)] {
        let class_bits = 32 - whole;
        // From a shift of 8, the split stops below the rounding bit; from 40, it balances the
        // tables at `⌈(b + ⌈log2 n⌉)/2⌉` (RFC-0028).
        for (base, split) in [(8u8, 7u8), (40, (whole + 1 + class_bits).div_ceil(2))] {
            let shifts: Vec<u8> = (base..base + classes).collect();
            let selection =
                selected(&requant(u32::from(whole), &shifts), &[], &default).expect("selects");
            assert!(
                matches!(selection.gadgets[0].1, Gadget::Requant1 { .. }),
                "{shifts:?}"
            );
            let selection =
                selected(&requant(u32::from(whole) + 1, &shifts), &[], &default).expect("selects");
            let Gadget::RequantSplit { split: s, high, .. } = selection.gadgets[0].1 else {
                panic!("RequantSplit expected for {shifts:?}");
            };
            assert_eq!(s, split, "{shifts:?}");
            let reduced: Vec<u8> = shifts.iter().map(|r| r - split).collect();
            assert!(matches!(
                kind(&selection, high),
                TableKind::Requant { b, r_by_class, .. }
                    if *b == whole + 1 - split && *r_by_class == reduced
            ));
        }
    }
}

/// The tables of a `RequantWide`: `Range { s }` or none, `Range { t }`, `Range { k }`, the
/// saturation's and the reduced requantization's.
fn wide_tables(selection: &Selection, gadget: Gadget) -> (u8, u8, Vec<TableKind>) {
    let Gadget::RequantWide {
        split,
        middle,
        low,
        mid,
        top,
        sign,
        high,
    } = gadget
    else {
        panic!("RequantWide expected, found {gadget:?}");
    };
    let mut kinds: Vec<TableKind> = low.iter().map(|&t| kind(selection, t).clone()).collect();
    for table in [mid, top, sign, high] {
        kinds.push(kind(selection, table).clone());
    }
    (split, middle, kinds)
}

fn range(bits: u8) -> TableKind {
    TableKind::Range { bits }
}

fn rq(b: u8, r_by_class: &[u8], lo: i32, hi: i32) -> TableKind {
    TableKind::Requant {
        b,
        r_by_class: r_by_class.to_vec(),
        lo,
        hi,
    }
}

#[test]
fn a_key_too_wide_for_its_shift_takes_requant_wide() {
    // b = 40 splits 8 bits: a shift of 9 keeps the rounding bit above them, one of 8 does not.
    let default = GadgetPolicy::default();
    assert!(matches!(
        selected(&requant(40, &[9]), &[], &default)
            .expect("selects")
            .gadgets[0]
            .1,
        Gadget::RequantSplit { split: 8, .. }
    ));
    // At 8, the low part stops below the rounding bit: s = 7. Past |w| ≥ 2^8 an INT8 output
    // saturates (2^{8 − 1} ≥ 128), so the top part takes the other 25 bits.
    let selection = selected(&requant(40, &[8]), &[], &default).expect("selects");
    assert_eq!(
        wide_tables(&selection, selection.gadgets[0].1),
        (
            7,
            8,
            vec![
                range(7),
                range(8),
                range(25),
                rq(26, &[1], -2, 1),
                rq(10, &[1], -128, 127)
            ]
        )
    );
    // The least shift decides the low part, the largest the middle: 39 bits and a class bit
    // split 8, past the shift of 8; the middle holds 12 − 7 + 7 bits.
    let selection = selected(&requant(39, &[12, 8]), &[], &default).expect("selects");
    assert_eq!(
        wide_tables(&selection, selection.gadgets[0].1),
        (
            7,
            12,
            vec![
                range(7),
                range(12),
                range(20),
                rq(21, &[1], -2, 1),
                rq(14, &[5, 1], -128, 127)
            ]
        )
    );
    assert!(matches!(
        selected(&requant(39, &[12, 9]), &[], &default)
            .expect("selects")
            .gadgets[0]
            .1,
        Gadget::RequantSplit { split: 8, .. }
    ));
}

#[test]
fn a_split_keys_low_part_past_b_max_takes_requant_wide() {
    // B_max = 20: a key of 40 bits splits 20 low bits; one of 41 would need 21, so the wide
    // gadget takes it with a low part of 20.
    let small = policy(20);
    assert!(matches!(
        selected(&requant(40, &[30]), &[], &small)
            .expect("selects")
            .gadgets[0]
            .1,
        Gadget::RequantSplit { split: 20, .. }
    ));
    let selection = selected(&requant(41, &[30]), &[], &small).expect("selects");
    assert_eq!(
        wide_tables(&selection, selection.gadgets[0].1),
        (
            20,
            17,
            vec![
                range(20),
                range(17),
                range(4),
                rq(5, &[1], -2, 1),
                rq(19, &[10], -128, 127)
            ]
        )
    );
}

#[test]
fn a_split_whose_packed_keys_reach_p_takes_requant_wide() {
    // RFC-0009 Lemma 2 needs n·2^b < P. Four classes at 62 bits, 4·2^62 = 2^64 > P, would split
    // at s = 32 inside Lemma 2's bounds on s, but take RequantWide (RFC-0024):
    // s = min(33 − 1, 32) = 32, t = (36 − 32) + ⌈log2 128⌉ = 11, k = 62 − 32 − 11 = 19.
    let default = GadgetPolicy::default();
    let selection = selected(&requant(62, &[33, 34, 35, 36]), &[], &default).expect("selects");
    assert_eq!(
        wide_tables(&selection, selection.gadgets[0].1),
        (
            32,
            11,
            vec![
                range(32),
                range(11),
                range(19),
                rq(20, &[1], -2, 1),
                rq(13, &[1, 2, 3, 4], -128, 127)
            ]
        )
    );
    // Below P, the split keeps RFC-0028's balanced point, ⌈(b + 2)/2⌉ = 32 clamped to
    // [b + 2 − 32, 32]: three classes at 62 bits, 3·2^62, and four at 61, 2^63.
    for (b, shifts) in [(62u32, &[33u8, 34, 35][..]), (61, &[33, 34, 35, 36][..])] {
        let selection = selected(&requant(b, shifts), &[], &default).expect("selects");
        let Gadget::RequantSplit {
            split: 32,
            low,
            high,
        } = selection.gadgets[0].1
        else {
            panic!("RequantSplit at s = 32 expected for b = {b}, shifts {shifts:?}");
        };
        let reduced: Vec<u8> = shifts.iter().map(|r| r - 32).collect();
        assert_eq!(kind(&selection, low), &range(32));
        assert_eq!(
            kind(&selection, high),
            &rq(u8::try_from(b - 32).expect("small"), &reduced, -128, 127)
        );
    }
}

#[test]
fn a_wide_keys_tables_fit_b_max() {
    // The reduced table: t = 4 + 7 bits, two more for the saturation and two for four classes.
    let classes = requant(30, &[8, 9, 10, 11]);
    let selection = selected(&classes, &[], &policy(15)).expect("15 bits fit");
    assert!(matches!(
        selection.gadgets[0].1,
        Gadget::RequantWide { middle: 11, .. }
    ));
    assert_eq!(
        selected(&classes, &[], &policy(14)),
        Err(refused(
            "a wide key's reduced table fits 2^{B_max} rows: t + 2 + ⌈log2 n⌉ ≤ B_max"
        ))
    );
    // The top part: a shift of 1 leaves no low part, and the doubled top key of k + 1 bits fits
    // 32 up to b = 39.
    let default = GadgetPolicy::default();
    let selection = selected(&requant(39, &[1]), &[], &default).expect("selects");
    assert_eq!(
        wide_tables(&selection, selection.gadgets[0].1),
        (
            0,
            8,
            vec![
                range(8),
                range(31),
                rq(32, &[1], -2, 1),
                rq(10, &[1], -128, 127)
            ]
        )
    );
    for b in [40, 62] {
        assert_eq!(
            selected(&requant(b, &[1]), &[], &default),
            Err(refused(
                "a wide key's top part fits a table of 2^{B_max} rows: k + 1 ≤ B_max"
            ))
        );
    }
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
        b in 33u32..=62, extra in 1u8..=31, seed in any::<u64>(), lo in -200i32..=0, span in 0i32..=400,
    ) {
        // A shift from just above the least split to past the balanced point: the split is
        // RFC-0028's, the balanced `⌈b/2⌉` inside Lemma 2's bounds; the low part passes its Range
        // table and the reduced table returns rq(z, r, lo, hi) on keys across the width.
        let r = u8::try_from(b - 32).expect("small") + extra;
        let split = b.div_ceil(2).clamp(b - 32, (u32::from(r) - 1).min(32));
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn the_wide_tables_compute_the_requantization(
        least in 1u8..=30, n in 1u8..=4, below in 0i64..=1 << 20, above in 0i64..=1 << 20, seed in any::<u64>(),
    ) {
        // Keys too wide for the split, of every width the wide gadget takes: its low part, middle,
        // top part and saturation pass their tables, and the reduced table returns rq(z, r_c,
        // lo, hi) for every class, at both ends of the key's range and inside it. The bounds are
        // drawn apart, so that the wider of the two sets the saturation.
        let shifts: Vec<u8> = (0..n).map(|c| least + c).collect();
        let class_bits = u32::from(n.next_power_of_two().trailing_zeros() as u8);
        let (lo, hi) = (-below, above);
        let bound = u64::try_from(hi.max(-lo).max(1)).expect("positive");
        let saturates = bound.next_power_of_two().trailing_zeros();
        let r_max = u32::from(least + n - 1);
        // The split fails from b = 32 − ⌈log2 n⌉ + least; the top table fits until k + 1 = 32.
        let first = (32 - class_bits + u32::from(least)).max(33);
        let last = (31 + r_max + saturates).min(62);
        prop_assume!(first <= last);
        let b = first + u32::try_from(seed % u64::from(last - first + 1)).expect("small");
        let mut g = GraphBuilder::new();
        let half = 1i128 << (b - 1);
        let x = g.input(decl(&[1], ADVICE, -half, half - 1)).expect("a key");
        let mut inputs = vec![x];
        if n > 1 {
            inputs.push(g.input(decl(&[1], WEIGHT, 0, i128::from(n) - 1)).expect("classes"));
        }
        let shift = g.shift_classes(shifts.clone()).expect("shifts");
        one(&mut g, Op::Requant { shift, lo, hi }, &inputs, out(&[1], ADVICE));
        let selection = selected(&g.finish(), &[], &GadgetPolicy::default()).expect("selects");
        let Gadget::RequantWide { split, middle, low, mid, top, sign, high } = selection.gadgets[0].1 else {
            panic!("RequantWide expected at b = {b}, shifts {shifts:?}");
        };
        let table = |id: TableId| {
            let spec = selection.tables.iter().find(|t| t.id == id).expect("a table");
            GeneratedTable::from_spec(spec).expect("the table regenerates")
        };
        let (s, t) = (u32::from(split), u32::from(middle));
        let k = b - s - t;
        let low = low.map(table);
        let (mid, top, sign, high) = (table(mid), table(top), table(sign), table(high));
        let mut state = seed;
        let mut keys = vec![-half, half - 1, -1, 0, 1];
        for _ in 0..32 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            // Magnitudes spread over every width up to the key's, so that each case of the top
            // part comes up.
            let bits = (state >> 58) as u32 % b;
            let magnitude = i128::from(state >> 2) % (1i128 << bits);
            keys.push(if state & 1 == 0 { magnitude } else { -magnitude - 1 });
        }
        for z in keys {
            let z_lo = (z + half) % (1 << s);
            if let Some(low) = &low {
                prop_assert!(low.lookup(i64::try_from(z_lo).expect("small"), 0).is_some());
            }
            let w = z >> s;
            let (h0, h1) = (w.rem_euclid(1 << t), w >> t);
            prop_assert!(mid.lookup(i64::try_from(h0).expect("small"), 0).is_some());
            let h1_off = i64::try_from(h1 + (1 << (k - 1))).expect("small");
            prop_assert!(top.lookup(h1_off, 0).is_some());
            let g = sign.lookup(i64::try_from(2 * h1).expect("small"), 0).expect("in the table");
            let key = i64::try_from(h0).expect("small") + (1i64 << t) * g;
            for (class, &r) in shifts.iter().enumerate() {
                let found = high.lookup(key, u8::try_from(class).expect("small")).expect("in the table");
                let expected = ((z + (1i128 << (r - 1))) >> r).clamp(i128::from(lo), i128::from(hi));
                prop_assert_eq!(i128::from(found), expected, "z = {}, r = {}", z, r);
            }
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
fn minimal_takes_requant_wide_where_its_down_projection_is_too_wide_for_a_split() {
    // The down projection reads a · u unrequantized: its residual key reaches about 2^54 at 8B
    // and 2^58 at 405B, a split of 23 or more bits against shifts near 17 (#292). With one class
    // the fused key saturates at INT20; with four, the update u = rq(acc, r_c, ±(2^20 − 1)) does.
    // The low part keeps 16 bits, the middle 20 or 24, and the top part the rest.
    let int20 = (-(1 << 19), (1 << 19) - 1);
    let update = (-((1 << 20) - 1), (1 << 20) - 1);
    let cases = [
        (4096, 32, 14_336, vec![17], 20, 55, int20),
        (4096, 32, 14_336, vec![17, 18, 19, 20], 24, 55, update),
        (16_384, 128, 53_248, vec![17], 20, 59, int20),
        (16_384, 128, 53_248, vec![17, 18, 19, 20], 24, 59, update),
    ];
    for (d, heads, d_ff, shifts, middle, b, (lo, hi)) in cases {
        let (arch, mut quant, silu) = llama(Profile::Minimal, 1, d, heads, 8, d_ff);
        let site = tessera_ir::QuantPoint::Layer {
            layer: 0,
            site: tessera_ir::Site::MlpResidual,
        };
        for (point, set) in &mut quant.shift_class_sets {
            if *point == site {
                set.clone_from(&shifts.iter().map(|&r| r as i8).collect());
            }
        }
        let model = PublicModel {
            arch: &arch,
            profile: Profile::Minimal,
            quant: &quant,
            silu_in_exponents: &silu,
        };
        let lowering = lower(model, 16).expect("the shape lowers");
        let ranges = analyze(&lowering.graph, &lowering.tables).expect("analyzes");
        let selection = select(
            &lowering.graph,
            &ranges,
            &lowering.tables,
            &GadgetPolicy::default(),
        )
        .expect("every gadget fits");
        let wide: Vec<_> = selection
            .gadgets
            .iter()
            .filter(|(_, g)| matches!(g, Gadget::RequantWide { .. }))
            .collect();
        let [(node, gadget)] = wide[..] else {
            panic!("one RequantWide expected, found {}", wide.len());
        };
        let Op::Requant { shift, .. } = lowering.graph.nodes[node.0 as usize].op else {
            panic!("a requantization");
        };
        assert_eq!(lowering.graph.shift_classes[shift.0 as usize], shifts);
        let reduced: Vec<u8> = shifts.iter().map(|r| r - 16).collect();
        let top = b - 16 - middle;
        assert_eq!(
            wide_tables(&selection, *gadget),
            (
                16,
                middle,
                vec![
                    range(16),
                    range(middle),
                    range(top),
                    rq(top + 1, &[1], -2, 1),
                    rq(middle + 2, &reduced, lo, hi)
                ]
            ),
            "d = {d}, shifts {shifts:?}"
        );
    }
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

/// `quant` with the site at `point` affine, of shift `r` (RFC-0025): out of `shift_class_sets`,
/// into `affine_sites` in order.
fn make_affine(quant: &mut tessera_ir::PublicQuantSpec, point: tessera_ir::QuantPoint, r: i8) {
    quant.shift_class_sets.retain(|&(p, _)| p != point);
    let at = quant.affine_sites.partition_point(|&(p, _)| p < point);
    quant.affine_sites.insert(at, (point, r));
}

/// RFC-0025 at 405B's shape: each affine site's key is its `pot` key times a 16-bit multiplier,
/// within the bound of RFC-0025's table and above `2^{15}` times the `pot` one. Every such key
/// is wider than a table, so it takes `RequantSplit` or `RequantWide`, as its width and shift
/// allow, never one lookup.
#[test]
fn affine_keys_are_16_bits_wider_and_take_the_split_or_the_wide_gadget() {
    use tessera_ir::{QuantPoint, Site};
    let (arch, mut quant, silu) = llama(Profile::Servable, 1, 16_384, 128, 8, 53_248);
    // The pot shifts of `llama`, 16 bits higher.
    let affine = [
        (Site::RopeQ, 38),
        (Site::RopeK, 38),
        (Site::V, 24),
        (Site::AttnResidual, 20),
        (Site::Up, 24),
        (Site::MlpResidual, 20),
    ];
    for (site, r) in affine {
        make_affine(&mut quant, QuantPoint::Layer { layer: 0, site }, r);
    }
    let model = PublicModel {
        arch: &arch,
        profile: Profile::Servable,
        quant: &quant,
        silu_in_exponents: &silu,
    };
    let lowering = lower(model, 16).expect("405B's shape lowers with affine sites");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("analyzes");
    let selection = select(
        &lowering.graph,
        &ranges,
        &lowering.tables,
        &GadgetPolicy::default(),
    )
    .expect("every gadget fits");
    assert_eq!(lowering.multipliers.len(), affine.len());
    let pot = 127i128 * 128;
    for (site, r) in affine {
        let point = QuantPoint::Layer { layer: 0, site };
        let &(_, output) = lowering
            .sites
            .iter()
            .find(|(p, _)| *p == point)
            .expect("the site");
        let node =
            &lowering.graph.nodes[lowering.graph.producer(output).expect("a node").0 as usize];
        let key = lowering.graph.tensors[node.inputs[0].0 as usize].range;
        let bound = key.lo.abs().max(key.hi);
        let residual = 1i128 << (19 + r);
        // RFC-0025's table, and the pot key's bound times 2^{15} below it.
        let (above, below) = match site {
            Site::V | Site::Up => ((pot * 16_384) << 15, 1 << 44),
            Site::RopeQ | Site::RopeK => ((pot * 16_384) << 29, 1 << 60),
            Site::AttnResidual => (((pot * 16_384) << 15) + residual, (1 << 44) + residual),
            _ => (((pot * 53_248) << 15) + residual, (1 << 46) + residual),
        };
        assert!(above < bound && bound < below, "{site:?}: {bound}");
        let gadget = selection.of(node.id).map(|(_, g)| g);
        assert!(
            matches!(
                gadget,
                Some(Gadget::RequantSplit { .. } | Gadget::RequantWide { .. })
            ),
            "{site:?}: {gadget:?}"
        );
    }
}

/// In `minimal`, the O and down projections' accumulators are not requantized: with a multiplier
/// their keys pass 62 bits, and an affine site there is refused with `ERR-COMPILE-RANGE`
/// (RFC-0025).
#[test]
fn minimal_refuses_affine_o_and_down_projections() {
    use tessera_ir::{QuantPoint, Site};
    for (site, r) in [(Site::AttnResidual, 53), (Site::MlpResidual, 33)] {
        let (arch, mut quant, silu) = llama(Profile::Minimal, 1, 16_384, 128, 8, 53_248);
        make_affine(&mut quant, QuantPoint::Layer { layer: 0, site }, r);
        let model = PublicModel {
            arch: &arch,
            profile: Profile::Minimal,
            quant: &quant,
            silu_in_exponents: &silu,
        };
        let error = lower(model, 16)
            .and_then(|lowering| {
                let ranges = analyze(&lowering.graph, &lowering.tables)?;
                select(
                    &lowering.graph,
                    &ranges,
                    &lowering.tables,
                    &GadgetPolicy::default(),
                )
            })
            .expect_err("refused");
        assert_eq!(error.code(), ErrorCode::CompileRange, "{site:?}: {error}");
    }
}
