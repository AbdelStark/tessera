// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's lowering against `spec/vectors/lowering/lowering.json`, written by an
//! independent oracle from RFC-0044 alone (`spec/vectors/lowering/README.md`): for every
//! conformance model, and for the two variants that take the forms the models miss, at every
//! length, the graph lowered from the claim list's fields alone equals the oracle's, field by
//! field: every graph input with its role, shape, kind, type and range, every node with its
//! operation, parameters, inputs and outputs, every constant, shift-class set, weight and table,
//! each by its id in creation order, and the checkpoints, residual routes and tensor count.
//!
//! A variant is a public part and nothing more: its edits are made to the golden claim list's
//! fields, which is all the lowering reads (RFC-0006 'Llama lowering'). The vectors' `at` and
//! `forms` describe the graph for people and are not part of it.

mod claim_support;

use std::fmt::Write as _;

use claim_support::{
    at, goldens, integer, items, list, make_affine, make_pot, member, string, text,
};
use tv_codec::{DecodeLimits, Rational};
use tv_core::claims::{ClaimList, Profile, Site, WeightType};
use tv_core::lowering::{
    Broadcast, Graph, IntType, Kind, LowerError, LoweredTable, Op, Role, Route, TensorId,
    WeightRef, lower,
};
use tv_core::ranges::analyze;
use tv_core::validation::{ClaimsError, StatementField, check_public};
use tv_tables::{TableId, TableKind};
use tv_testkit::{Json, read_vectors};

/// The differences found so far, reported together at the end.
#[derive(Default)]
struct Differences {
    report: String,
    count: usize,
}

impl Differences {
    /// Records that `what` of `case` is `found` where the oracle has `expected`.
    fn check<T: PartialEq + std::fmt::Debug>(
        &mut self,
        case: &str,
        what: &str,
        expected: T,
        found: T,
    ) {
        if expected != found {
            self.count += 1;
            if self.count <= 40 {
                let _ = writeln!(
                    self.report,
                    "{case}: {what}: expected {expected:?}, found {found:?}"
                );
            }
        }
    }
}

fn number<T: TryFrom<i128>>(value: &Json) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(integer(value)).expect("in range")
}

fn site_named(name: &str) -> Site {
    match name {
        "AttnNorm" => Site::AttnNorm,
        "RopeQ" => Site::RopeQ,
        "RopeK" => Site::RopeK,
        "V" => Site::V,
        "AttnOut" => Site::AttnOut,
        "AttnResidual" => Site::AttnResidual,
        "MlpNorm" => Site::MlpNorm,
        "Silu" => Site::Silu,
        "Up" => Site::Up,
        "Product" => Site::Product,
        "MlpResidual" => Site::MlpResidual,
        other => panic!("unknown site {other}"),
    }
}

fn weight_type_named(name: &str) -> WeightType {
    match name {
        "Embed" => WeightType::Embed,
        "Q" => WeightType::Q,
        "K" => WeightType::K,
        "V" => WeightType::V,
        "O" => WeightType::O,
        "Gate" => WeightType::Gate,
        "Up" => WeightType::Up,
        "Down" => WeightType::Down,
        "LmHead" => WeightType::LmHead,
        other => panic!("unknown weight type {other}"),
    }
}

/// `[type, layer]`, a `WeightRef`.
fn weight_ref(value: &Json) -> WeightRef {
    let [ty, layer] = items(value) else {
        panic!("a weight is [type, layer]");
    };
    WeightRef {
        ty: weight_type_named(string(ty)),
        layer: number(layer),
    }
}

/// A kind as RFC-0044 'Notation' writes it: `A`, `W`, `P`, `v` or `L(t)`.
fn kind_named(name: &str) -> Kind {
    match name {
        "A" => Kind::Advice,
        "W" => Kind::Weight,
        "P" => Kind::Public,
        "v" => Kind::Virtual,
        lookup => {
            let table = lookup
                .strip_prefix("L(")
                .and_then(|rest| rest.strip_suffix(')'))
                .unwrap_or_else(|| panic!("unknown kind {lookup}"));
            Kind::Lookup {
                table: TableId(table.parse().expect("a table id")),
            }
        }
    }
}

/// A type as 03 §1 writes it: `u` or `i` and its bits.
fn type_named(name: &str) -> IntType {
    let (signed, bits) = match name.split_at(1) {
        ("u", bits) => (false, bits),
        ("i", bits) => (true, bits),
        _ => panic!("unknown type {name}"),
    };
    IntType {
        bits: bits.parse().expect("a bit count"),
        signed,
    }
}

/// A graph input's role: `tokens`, `positions`, or `weight`, `classes` or `multipliers` followed
/// by the weight type and the layer.
fn role_named(name: &str) -> Role {
    let words: Vec<&str> = name.split(' ').collect();
    match words.as_slice() {
        ["tokens"] => Role::Tokens,
        ["positions"] => Role::Positions,
        [what, ty, layer] => {
            let weight = WeightRef {
                ty: weight_type_named(ty),
                layer: layer.parse().expect("a layer"),
            };
            match *what {
                "weight" => Role::Weight(weight),
                "classes" => Role::Classes(weight),
                "multipliers" => Role::Multipliers(weight),
                other => panic!("unknown role {other}"),
            }
        }
        _ => panic!("unknown role {name}"),
    }
}

/// A table kind as the vectors write it, with RFC-0008's field names.
fn table_kind(value: &Json) -> TableKind {
    let Json::Object(members) = value else {
        panic!("{value:?} is not a kind");
    };
    let [(name, fields)] = members.as_slice() else {
        panic!("{value:?} names more than one kind");
    };
    let field = |key: &str| member(fields, key);
    match name.as_str() {
        "Rope" => TableKind::Rope {
            theta: number(field("theta")),
            factor: number(field("factor")),
            low_freq_factor: number(field("low_freq_factor")),
            high_freq_factor: number(field("high_freq_factor")),
            original_max: number(field("original_max")),
            bits: number(field("bits")),
            max_context: number(field("max_context")),
            d_head: number(field("d_head")),
        },
        "Exp" => {
            let [num, den] = items(field("scale")) else {
                panic!("a scale is [num, den]");
            };
            TableKind::Exp {
                key_bits: number(field("key_bits")),
                sigma: number(field("sigma")),
                scale: Rational::new(number(num), number(den)).expect("in lowest terms"),
            }
        }
        "Silu" => TableKind::Silu {
            b: number(field("b")),
            r_by_class: items(field("r_by_class")).iter().map(number).collect(),
            in_exp: number(field("in_exp")),
            out_exp: number(field("out_exp")),
        },
        other => panic!("the lowering derives no {other} table"),
    }
}

/// The operation a node of the vectors names, with its parameters.
fn operation(name: &str, params: &Json) -> Op {
    let field = |key: &str| member(params, key);
    let table = |key: &str| TableId(number(field(key)));
    match name {
        "OneHot" => Op::OneHot {
            vocab: number(field("vocab")),
        },
        "MatMulWeight" => Op::MatMulWeight {
            weight: weight_ref(field("weight")),
        },
        "ConstMul" => Op::ConstMul {
            c: number(field("c")),
        },
        "Add" => Op::Add,
        "Reshape" => Op::Reshape,
        "SumSquares" => Op::SumSquares {
            axis: number(field("axis")),
        },
        "RsqrtRow" => Op::RsqrtRow {
            kappa: number(field("kappa")),
            eps_int: number(field("eps_int")),
            dim: number(field("dim")),
        },
        "Mul" => Op::Mul {
            broadcast: match string(field("broadcast")) {
                "Trailing" => Broadcast::Trailing,
                "Leading" => Broadcast::Leading,
                other => panic!("unknown broadcast {other}"),
            },
        },
        "Requant" => Op::Requant {
            shift: number(field("shift")),
            lo: number(field("lo")),
            hi: number(field("hi")),
        },
        "Rotate" => Op::Rotate {
            rope: table("rope"),
        },
        "Transpose" => {
            let perm: Vec<u8> = items(field("perm")).iter().map(number).collect();
            Op::Transpose {
                perm: perm.try_into().expect("three axes"),
            }
        }
        "Expand" => Op::Expand {
            axis: number(field("axis")),
            len: number(field("len")),
        },
        "MatMulAct" => Op::MatMulAct {
            transpose_rhs: matches!(field("transpose_rhs"), Json::Bool(true)),
        },
        "SoftmaxRow" => Op::SoftmaxRow {
            exp_table: table("exp_table"),
            sigma: number(field("sigma")),
        },
        "TableMap" => Op::TableMap {
            table: table("table"),
        },
        other => panic!("the lowering creates no {other} node"),
    }
}

/// `[id, shape, kind]`: a tensor of the vectors.
fn tensor(value: &Json) -> (u32, Vec<u32>, Kind) {
    let [id, shape, kind] = items(value) else {
        panic!("an output is [id, shape, kind]");
    };
    (
        number(id),
        items(shape).iter().map(number).collect(),
        kind_named(string(kind)),
    )
}

/// The golden claim list of the model artifact `file`.
fn claims_of(file: &str) -> ClaimList {
    let golden = goldens()
        .into_iter()
        .find(|g| g.model == file)
        .unwrap_or_else(|| panic!("no golden claim list for {file}"));
    ClaimList::open(&golden.envelope, &DecodeLimits::MESSAGE).expect("a golden claim list")
}

/// Makes a variant's edits to the claim list's public part, each in every layer it names.
fn apply_variant(variant: &Json, claims: &mut ClaimList) {
    if let Some(heads) = variant.get("n_kv_heads") {
        claims.arch.n_kv_heads = number(heads);
    }
    if let Some(rescales) = variant.get("rescale") {
        for edit in items(rescales) {
            let [layer, site, mode, value] = items(edit) else {
                panic!("a rescale is [layer, site, mode, value]");
            };
            let point = at(number(layer), site_named(string(site)));
            match string(mode) {
                "pot" => {
                    let shifts: Vec<i8> = items(value).iter().map(number).collect();
                    make_pot(claims, point, &shifts);
                }
                "affine" => make_affine(claims, point, number(value)),
                other => panic!("unknown mode {other}"),
            }
        }
    }
    if let Some(scales) = variant.get("score_scales") {
        for edit in items(scales) {
            let [layer, num, den] = items(edit) else {
                panic!("a score scale is [layer, num, den]");
            };
            let layer: u32 = number(layer);
            let entry = claims
                .quant
                .score_scales
                .iter_mut()
                .find(|(l, _)| *l == layer)
                .expect("the layer's scale");
            entry.1 = Rational::new(number(num), number(den)).expect("in lowest terms");
        }
    }
}

/// Compares `graph` with the vectors' `case`, field by field.
fn compare(case: &Json, name: &str, graph: &Graph, diffs: &mut Differences) {
    let tensors = graph.tensors();
    let shape_of = |id: TensorId| -> Option<Vec<u32>> {
        let index = usize::try_from(id.0).ok()?;
        tensors.get(index).map(|t| t.shape.dims().to_vec())
    };
    let kind_of = |id: TensorId| -> Option<Kind> {
        let index = usize::try_from(id.0).ok()?;
        tensors.get(index).map(|t| t.kind)
    };
    diffs.check(
        name,
        "the number of tensors",
        number::<usize>(member(case, "tensors")),
        tensors.len(),
    );
    let checkpoints: Vec<TensorId> = list(case, "checkpoints")
        .iter()
        .map(|c| TensorId(number(c)))
        .collect();
    diffs.check(
        name,
        "the checkpoints",
        checkpoints,
        graph.checkpoints().to_vec(),
    );
    let routes = member(case, "routes");
    for site in [Site::AttnResidual, Site::MlpResidual] {
        let expected = match text(routes, &format!("{site:?}")) {
            "fused key" => Route::FusedKey,
            "two steps" => Route::TwoSteps,
            other => panic!("unknown route {other}"),
        };
        diffs.check(
            name,
            &format!("{site:?}'s route"),
            Some(expected),
            graph.route(site),
        );
    }

    // Inputs.
    let inputs = list(case, "inputs");
    diffs.check(
        name,
        "the number of inputs",
        inputs.len(),
        graph.inputs().len(),
    );
    for (index, (expected, found)) in inputs.iter().zip(graph.inputs()).enumerate() {
        let [id, role, shape, kind, ty, lo, hi] = items(expected) else {
            panic!("an input is [id, role, shape, kind, type, lo, hi]");
        };
        let what = |field: &str| format!("input {index} ({}): {field}", string(role));
        diffs.check(name, &what("tensor"), TensorId(number(id)), found.tensor);
        diffs.check(name, &what("role"), role_named(string(role)), found.role);
        let shape: Vec<u32> = items(shape).iter().map(number).collect();
        diffs.check(name, &what("shape"), Some(shape), shape_of(found.tensor));
        diffs.check(
            name,
            &what("kind"),
            Some(kind_named(string(kind))),
            kind_of(found.tensor),
        );
        diffs.check(name, &what("type"), type_named(string(ty)), found.ty);
        diffs.check(name, &what("lo"), number::<i64>(lo), found.lo);
        diffs.check(name, &what("hi"), number::<i64>(hi), found.hi);
    }

    // Nodes.
    let nodes = list(case, "nodes");
    diffs.check(
        name,
        "the number of nodes",
        nodes.len(),
        graph.nodes().len(),
    );
    for (index, (expected, found)) in nodes.iter().zip(graph.nodes()).enumerate() {
        let [id, op, params, node_inputs, outputs, _at] = items(expected) else {
            panic!("a node is [id, operation, parameters, inputs, outputs, at]");
        };
        let what = |field: &str| format!("node {index} ({}): {field}", string(op));
        diffs.check(name, &what("id"), number::<usize>(id), index);
        diffs.check(
            name,
            &what("operation"),
            &operation(string(op), params),
            found.op(),
        );
        let node_inputs: Vec<TensorId> = items(node_inputs)
            .iter()
            .map(|t| TensorId(number(t)))
            .collect();
        diffs.check(
            name,
            &what("inputs"),
            node_inputs.as_slice(),
            found.inputs(),
        );
        let outputs: Vec<(u32, Vec<u32>, Kind)> = items(outputs).iter().map(tensor).collect();
        diffs.check(
            name,
            &what("the number of outputs"),
            outputs.len(),
            found.outputs().len(),
        );
        for (o, ((id, shape, kind), &out)) in outputs.iter().zip(found.outputs()).enumerate() {
            diffs.check(name, &what(&format!("output {o}")), TensorId(*id), out);
            diffs.check(
                name,
                &what(&format!("output {o}'s shape")),
                Some(shape.clone()),
                shape_of(out),
            );
            diffs.check(
                name,
                &what(&format!("output {o}'s kind")),
                Some(*kind),
                kind_of(out),
            );
        }
    }

    // Constants and shift-class sets.
    let constants: Vec<Vec<i64>> = list(case, "constants")
        .iter()
        .map(|c| items(c).iter().map(number).collect())
        .collect();
    let found: Vec<Vec<i64>> = graph.constants().iter().map(|&c| vec![c]).collect();
    diffs.check(name, "the constants", constants, found);
    let shifts: Vec<Vec<u8>> = list(case, "shift_classes")
        .iter()
        .map(|s| items(s).iter().map(number).collect())
        .collect();
    let found: Vec<Vec<u8>> = graph
        .shift_classes()
        .iter()
        .map(|s| s.as_slice().to_vec())
        .collect();
    diffs.check(name, "the shift-class sets", shifts, found);

    // Weights.
    let weights: Vec<(WeightRef, TensorId)> = list(case, "weights")
        .iter()
        .map(|w| {
            let [ty, layer, id] = items(w) else {
                panic!("a weight is [type, layer, tensor]");
            };
            (
                WeightRef {
                    ty: weight_type_named(string(ty)),
                    layer: number(layer),
                },
                TensorId(number(id)),
            )
        })
        .collect();
    diffs.check(name, "the weights", weights, graph.weights().to_vec());

    // Tables.
    let tables = list(case, "tables");
    diffs.check(
        name,
        "the number of tables",
        tables.len(),
        graph.tables().len(),
    );
    for (index, (expected, found)) in tables.iter().zip(graph.tables()).enumerate() {
        let [id, kind, log2_rows] = items(expected) else {
            panic!("a table is [id, kind, log2_rows]");
        };
        let what = |field: &str| format!("table {index}: {field}");
        diffs.check(name, &what("id"), TableId(number(id)), found.id);
        diffs.check(name, &what("kind"), &table_kind(kind), &found.kind);
        diffs.check(
            name,
            &what("log2_rows"),
            number::<u8>(log2_rows),
            found.log2_rows,
        );
    }
}

#[test]
fn every_case_of_the_lowering_vectors_is_reproduced() {
    let file = read_vectors("lowering/lowering.json");
    let cases = list(&file, "cases");
    assert_eq!(cases.len(), 24, "six models and variants at four lengths");
    let mut diffs = Differences::default();
    for case in cases {
        let n: u32 = number(member(case, "n"));
        let name = format!("{} at n = {n}", text(case, "model"));
        let mut claims = claims_of(text(case, "file"));
        if let Some(variant) = case.get("variant") {
            apply_variant(variant, &mut claims);
        }
        let profile = match text(case, "profile") {
            "Servable" => Profile::Servable,
            "Minimal" => Profile::Minimal,
            other => panic!("unknown profile {other}"),
        };
        assert_eq!(claims.profile, profile, "{name}");
        let public = check_public(&claims).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let graph = lower(&public, n).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        compare(case, &name, &graph, &mut diffs);
        // This oracle predates the verifier range implementation and uses only RFC-0006,
        // independent lowering metadata and Python integers.
        let range_file = read_vectors("ranges/ranges.json");
        let expected = list(&range_file, "cases")
            .iter()
            .find(|c| {
                text(c, "model") == text(case, "model") && member(c, "n") == member(case, "n")
            })
            .expect("a range vector for each lowering case");
        let wanted: Vec<Option<(i128, i128)>> = list(expected, "ranges")
            .iter()
            .map(|range| match range {
                Json::Null => None,
                _ => {
                    let [lo, hi] = items(range) else {
                        panic!("an inclusive interval")
                    };
                    Some((integer(lo), integer(hi)))
                }
            })
            .collect();
        let found: Vec<_> = analyze(&graph)
            .expect("range analysis")
            .iter()
            .map(|r| r.bounds())
            .collect();
        diffs.check(
            &name,
            "every tensor's independently derived range",
            wanted,
            found,
        );
    }
    assert!(
        diffs.count == 0,
        "{} differences from the oracle's graphs; the first:\n{}",
        diffs.count,
        diffs.report
    );
}

#[test]
fn the_node_and_tensor_ids_do_not_depend_on_the_length() {
    // RFC-0041 'Sequence length': only shapes do.
    for golden in goldens() {
        let claims = ClaimList::open(&golden.envelope, &DecodeLimits::MESSAGE).unwrap();
        let public = check_public(&claims).unwrap();
        let one = lower(&public, 1).unwrap();
        let longest = lower(&public, claims.arch.max_context).unwrap();
        assert_eq!(one.nodes(), longest.nodes(), "{}", golden.name);
        assert_eq!(
            one.tensors().len(),
            longest.tensors().len(),
            "{}",
            golden.name
        );
        assert_eq!(one.tables(), longest.tables(), "{}", golden.name);
        assert_eq!(one.constants(), longest.constants(), "{}", golden.name);
        assert_eq!(one.weights(), longest.weights(), "{}", golden.name);
    }
}

#[test]
fn a_length_outside_the_context_is_refused() {
    let claims = claims_of("tiny-2-servable.d");
    let public = check_public(&claims).unwrap();
    for n in [0, claims.arch.max_context + 1, u32::MAX] {
        assert_eq!(
            lower(&public, n).unwrap_err(),
            LowerError::Length { n },
            "n = {n}"
        );
    }
}

/// The `Exp` table each layer's `SoftmaxRow` reads, in layer order.
fn exp_tables(graph: &Graph) -> Vec<LoweredTable> {
    graph
        .nodes()
        .iter()
        .filter_map(|node| match *node.op() {
            Op::SoftmaxRow { exp_table, .. } => graph.table(exp_table).cloned(),
            _ => None,
        })
        .collect()
}

/// The key width of an `Exp` table.
fn key_bits(table: &LoweredTable) -> u8 {
    match table.kind {
        TableKind::Exp { key_bits, .. } => key_bits,
        ref other => panic!("{other:?} is not an Exp table"),
    }
}

#[test]
fn the_exp_search_ends_at_the_first_width_and_rfc_0008_allows_its_table() {
    // RFC-0044 'The tables', as amended: the search ends at the first width whose table maps its
    // last key to 0, and RFC-0007 step 2 guarantees that width exists and that RFC-0008 allows
    // its table. At `σ = 16` the scale `2^-20` widens layer 1's table from the spread's 19 bits
    // to 24.
    let mut claims = claims_of("tiny-2-servable.d");
    claims.quant.score_scales[1].1 = Rational::new(1, 1 << 20).unwrap();
    let graph = lower(&check_public(&claims).unwrap(), 4).unwrap();
    let tables = exp_tables(&graph);
    assert_eq!(tables.iter().map(key_bits).collect::<Vec<_>>(), [19, 24]);
    // Heads of 2,048: the search ends at the spread's own width, 26 bits, and both layers, at
    // one scale, read one table.
    let mut wide = claims_of("tiny-2-servable.d");
    wide.arch.d_head = 2_048;
    wide.arch.d_model = 4 * 2_048;
    for entry in &mut wide.quant.score_scales {
        entry.1 = Rational::new(1, 2896).unwrap();
    }
    let wide_graph = lower(&check_public(&wide).unwrap(), 4).unwrap();
    let wide_tables = exp_tables(&wide_graph);
    assert_eq!(
        wide_tables.iter().map(key_bits).collect::<Vec<_>>(),
        [26, 26]
    );
    assert_eq!(wide_tables[0].id, wide_tables[1].id);
    // RFC-0008 allows every table the search ended at, with the row count the lowering states.
    for table in tables.iter().chain(&wide_tables) {
        assert_eq!(
            table.kind.check_parameters(),
            Ok(table.log2_rows),
            "{table:?}"
        );
    }
    // One bit more, `2^-21`, widens the table to 25 bits, whose zero tail starts past key 2^24:
    // step 2 refuses the claim list, and the lowering never sees it.
    claims.quant.score_scales[1].1 = Rational::new(1, 1 << 21).unwrap();
    assert_eq!(
        check_public(&claims).unwrap_err(),
        ClaimsError::StatementMismatch {
            field: StatementField::Quantization
        }
    );
}
