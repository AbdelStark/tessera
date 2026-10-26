// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The public factors of the anchor relations that the prover exposes, against the independent
//! oracle `spec/vectors/anchor-relations/relations.json` (#150, slice S1), written from RFC-0072
//! and RFC-0066 by an author who read neither implementation:
//!
//! - **`LT`**, `tessera_poly::less_than`, which the prover's shift and selectors evaluate, at
//!   each case's point and bounds;
//! - **`ĉ`**, a `ConstAdd`'s shift through [`public_value`] at each case's point, scalar and per
//!   channel; a per-channel case's `c̃(r_c)·ψ̃(r)` differs from it exactly where the oracle says;
//! - **`ψ̃`**, as the shift of a `ConstAdd` of the constant 1, which RFC-0072 makes `1·ψ̃(r)`, at
//!   each `psi` case's and each `const_add` case's point;
//! - **the `BitFn` truth tables**, through `BoolFn::eval`, the witness engine's function.
//!
//! The selection kernels `S̃` and `K̃`, recomposition's weights `W̃_b`, `W̃_h` and its
//! zero-extensions, the `BitFn` polynomials over `Fp2` and recomposition's final value are the
//! stage prover's own, behind its relations, and no public function returns them. The stage
//! tests (`stages.rs`'s `selections` and `bits`) prove each with the prover and check the proof
//! with a verifier whose closed forms they check against the extensions of the tables.

use serde_json::Value;
use std::path::Path;
use tessera_compiler::schedule::PublicSource;
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_ir::{
    BoolFn, Graph, GraphBuilder, IntRange, IntType, NodeId, NodeKind, Op, TensorDecl, TensorId,
};
use tessera_poly::less_than;
use tessera_prover::{PublicTensors, WitnessSource, public_value};

type F = GoldilocksExt2;

fn relations() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/anchor-relations/relations.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// An `Fp` element of the vectors, `"0x"` and 16 hex digits, canonical.
fn base(v: &Value) -> Goldilocks {
    let s = v.as_str().expect("a field element");
    let digits = s.strip_prefix("0x").expect("0x");
    let x = u64::from_str_radix(digits, 16).expect("hex");
    Goldilocks::from_canonical_u64(x).unwrap_or_else(|| panic!("{s} is canonical"))
}

/// An `Fp2` element of the vectors, `[c0, c1]`, `c0 + c1·X` with `X^2 = 7`.
fn ext(v: &Value) -> F {
    F::new(base(&v[0]), base(&v[1]))
}

fn point(v: &Value) -> Vec<F> {
    v.as_array().expect("a point").iter().map(ext).collect()
}

fn uint(v: &Value) -> u32 {
    u32::try_from(v.as_u64().expect("an integer")).expect("a u32")
}

/// A witness of no tensor: a shift reads only its node's constant and shape.
struct NoValues;

impl WitnessSource for NoValues {
    fn values(&self, _tensor: TensorId) -> Option<&[i64]> {
        None
    }
}

/// A graph of one `ConstAdd` of `constant` on a public input of `shape`, and its node.
fn const_add(shape: &[u32], constant: Vec<i64>) -> (Graph, NodeId) {
    let decl = |kind| TensorDecl {
        shape: shape.to_vec(),
        ty: IntType {
            bits: 64,
            signed: true,
        },
        range: IntRange { lo: 0, hi: 0 },
        kind,
    };
    let mut g = GraphBuilder::new();
    let x = g.input(decl(NodeKind::Public)).expect("an input");
    let c = g.constant(constant).expect("a constant");
    g.node(Op::ConstAdd { c }, &[x], vec![decl(NodeKind::Virtual)])
        .expect("a ConstAdd");
    let graph = g.finish();
    let node = graph.nodes[0].id;
    (graph, node)
}

/// `ĉ(r)` of a `ConstAdd` of `constant` on `shape`, through the prover's [`public_value`].
fn shift(shape: &[u32], constant: Vec<i64>, r: &[F]) -> F {
    let (graph, node) = const_add(shape, constant);
    let public = PublicTensors::of(&graph);
    public_value(&graph, &public, PublicSource::Shift(node), &NoValues, r).expect("the shift")
}

fn shape(v: &Value) -> Vec<u32> {
    v.as_array().expect("a shape").iter().map(uint).collect()
}

/// RFC-0072 'Real positions': `LT(r, a) = Σ_{h < a} eq(r, h)` at each case's point, `a` up to
/// `2^n`, is the oracle's.
#[test]
fn less_than_is_the_oracles() {
    let doc = relations();
    let cases = doc["lt"].as_array().expect("lt");
    let mut checked = 0;
    for case in cases {
        let r = point(&case["point"]);
        assert_eq!(r.len(), uint(&case["n"]) as usize, "n coordinates");
        for pair in case["values"].as_array().expect("values") {
            let a = pair[0].as_u64().expect("a bound");
            let got = less_than(&r, a).expect("a bound within the cube");
            assert_eq!(got, ext(&pair[1]), "LT(r, {a}) on {} coordinates", r.len());
            checked += 1;
        }
    }
    assert!(checked >= cases.len(), "every case has values");
}

/// RFC-0072 'Real positions', 'Linear operations that are not coordinate maps': `ψ̃(r) =
/// Π_a LT(r_a, n_a)` is the shift of a `ConstAdd` of 1, at each `psi` case's point and at each
/// `const_add` case's, whose `psi` the oracle states.
#[test]
fn the_selector_is_the_oracles() {
    let doc = relations();
    for case in doc["psi"].as_array().expect("psi") {
        let s = shape(&case["shape"]);
        let r = point(&case["point"]);
        assert_eq!(shift(&s, vec![1], &r), ext(&case["value"]), "ψ̃ of {s:?}");
    }
    for case in doc["const_add"].as_array().expect("const_add") {
        let s = shape(&case["shape"]);
        let r = point(&case["point"]);
        assert_eq!(shift(&s, vec![1], &r), ext(&case["psi"]), "ψ̃ of {s:?}");
    }
}

/// RFC-0072 'Linear operations that are not coordinate maps': the prover's `ĉ(r)` is the
/// oracle's for a constant of one value, `c·ψ̃(r)`, and for one per channel,
/// `c̃(r_c)·Π_{a ≠ last} LT(r_a, n_a)`; it differs from `c̃(r_c)·ψ̃(r)` exactly when the last axis's
/// length is not a power of two, which the oracle's `wrong_differs` records.
#[test]
fn the_const_add_shift_is_the_oracles() {
    let doc = relations();
    let cases = doc["const_add"].as_array().expect("const_add");
    let mut per_channel = 0;
    for case in cases {
        let s = shape(&case["shape"]);
        let constant: Vec<i64> = case["constant"]
            .as_array()
            .expect("a constant")
            .iter()
            .map(|v| v.as_i64().expect("an i64"))
            .collect();
        let r = point(&case["point"]);
        let got = shift(&s, constant.clone(), &r);
        assert_eq!(got, ext(&case["value"]), "ĉ of {constant:?} on {s:?}");
        if let Some(wrong) = case.get("wrong") {
            per_channel += 1;
            let differs = case["wrong_differs"].as_bool().expect("a flag");
            assert_eq!(
                got != ext(wrong),
                differs,
                "c̃(r_c)·ψ̃(r) of {constant:?} on {s:?} differs from ĉ exactly as the oracle says"
            );
            let last = *s.last().expect("an axis");
            assert_eq!(differs, !last.is_power_of_two(), "the last axis {last}");
        }
    }
    assert!(per_channel > 0, "constants per channel");
}

/// RFC-0066 '`BitFn`': each Boolean function's truth table is the oracle's, the first input the
/// most significant, through `BoolFn::eval`, which the witness engine computes `BitFn` with.
#[test]
fn the_bit_functions_truth_tables_are_the_oracles() {
    let doc = relations();
    let cases = doc["bitfn"].as_array().expect("bitfn");
    let mut names = Vec::new();
    for case in cases {
        let name = case["f"].as_str().expect("a name");
        let f = match name {
            "Xor" => BoolFn::Xor,
            "Xor3" => BoolFn::Xor3,
            "And" => BoolFn::And,
            "Ch" => BoolFn::Ch,
            "Maj" => BoolFn::Maj,
            other => panic!("a Boolean function: {other}"),
        };
        assert_eq!(f.arity(), uint(&case["inputs"]) as usize, "{name}'s arity");
        let rows = case["truth_table"].as_array().expect("a truth table");
        assert_eq!(rows.len(), 1 << f.arity(), "{name}: every input");
        for (k, row) in rows.iter().enumerate() {
            let row: Vec<i64> = row
                .as_array()
                .expect("a row")
                .iter()
                .map(|v| v.as_i64().expect("a bit"))
                .collect();
            let (inputs, output) = row.split_at(f.arity());
            let bits: Vec<i64> = (0..f.arity())
                .map(|i| i64::from((k >> (f.arity() - 1 - i)) & 1 == 1))
                .collect();
            assert_eq!(
                inputs, bits,
                "{name}: row {k}, the first input most significant"
            );
            assert_eq!(f.eval(inputs), output[0], "{name}{inputs:?}");
        }
        names.push(name);
    }
    names.sort_unstable();
    assert_eq!(
        names,
        ["And", "Ch", "Maj", "Xor", "Xor3"],
        "the five functions"
    );
}
