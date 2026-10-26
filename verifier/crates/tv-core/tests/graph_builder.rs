// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The graph builder of `tv_core::lowering` against RFC-0006's rules ('Operation set' and its
//! table of shapes, as RFC-0066 and RFC-0072 amend them), and the refusals of the anchor
//! circuits' operations by every check that reads a graph after the lowering.
//!
//! - Every rule is broken alone, each in a call whose honest twin passes, and the call is
//!   refused with exactly that rule, the graph unchanged.
//! - Two anchor-shaped fragments are built from RFC-0066's and RFC-0072's text and read back: a
//!   SHA-256 block's pieces (the masks, the message's assembly, a function, a sum's value levels
//!   and its decomposition, the digest's binding) and AES's (a masked round key expanded over the
//!   blocks, `AddRoundKey`, `SubBytes`, `ShiftRows`, `MixColumns`' products and the range checks of
//!   the key and plaintext). They are fragments in the shape of the circuits, not the circuits.
//! - Building the same graph twice gives equal graphs.
//! - Each operation the Llama lowering never creates, appended to a lowered graph, is refused by
//!   range analysis, gadget selection, the schedule, the token axes, the advice layout, the
//!   gadget plan and the leaf stage's derivations with each one's typed error, where the same
//!   graph with a lowering's operation appended in its place passes.

mod claim_support;

use claim_support::claims_of;
use tv_core::advice::{self, Config};
use tv_core::claims::{ClaimList, WeightType};
use tv_core::leaf::groups::{AxisError, TokenAxes};
use tv_core::leaf::{ModelSchedule, split};
use tv_core::lowering::{
    BoolFn, Broadcast, BuildError, Capacity, Graph, GraphBuilder, IntType, Kind, Node, NodeId, Op,
    Perm, Role, Route, Rule, TensorId, WeightRef, lower,
};
use tv_core::ranges::{self, RangeError};
use tv_core::schedule::{self, ScheduleError};
use tv_core::validation::{ClaimsError, check_public};
use tv_core::{gadgets, selection};
use tv_tables::{BitOp, TableId, TableKind};

const BIT: IntType = IntType {
    bits: 1,
    signed: false,
};
const BYTE: IntType = IntType {
    bits: 8,
    signed: false,
};
const INT8: IntType = IntType {
    bits: 8,
    signed: true,
};
const WORD: IntType = IntType {
    bits: 40,
    signed: false,
};

fn perm(axes: &[u8]) -> Perm {
    Perm::new(axes).expect("at most four axes")
}

fn roomy() -> Capacity {
    Capacity {
        tensors: 4_096,
        nodes: 4_096,
        constant_values: 65_536,
    }
}

/// The tiny servable model's claim list and its graph at `n = 4`.
fn lowered() -> (ClaimList, Graph) {
    let claims = claims_of("tiny-2-servable.d");
    let graph = lower(&check_public(&claims).expect("the golden validates"), 4).expect("lowers");
    (claims, graph)
}

/// `graph`'s sizes and room for a few more of each.
fn room_after(graph: &Graph) -> Capacity {
    let more = |len: usize, extra: u32| u32::try_from(len).unwrap() + extra;
    Capacity {
        tensors: more(graph.tensors().len(), 64),
        nodes: more(graph.nodes().len(), 64),
        constant_values: more(graph.constants().len(), 256),
    }
}

fn next_node(b: &GraphBuilder) -> NodeId {
    NodeId(u32::try_from(b.graph().nodes().len()).unwrap())
}

// ----- Declarations ------------------------------------------------------------------------------

/// A declaration refused with `rule`.
fn refused<T>(rule: Rule) -> Result<T, BuildError> {
    Err(BuildError::Declaration { rule })
}

#[test]
fn each_declaration_rule_broken_alone_is_refused_and_adds_nothing() {
    let mut b = GraphBuilder::new(roomy()).unwrap();
    let before = b.graph().clone();

    // Shapes: no axis, an axis of 0, five axes.
    for dims in [&[][..], &[2, 0][..], &[1, 1, 1, 1, 1][..]] {
        assert_eq!(b.input(dims, Kind::Public, BIT, 0, 1), refused(Rule::Shape));
    }
    // Kinds: only public and committed advice are graph inputs; weights are declared as such.
    for kind in [
        Kind::Virtual,
        Kind::Weight,
        Kind::Lookup { table: TableId(0) },
    ] {
        assert_eq!(b.input(&[2], kind, BIT, 0, 1), refused(Rule::InputKind));
    }
    // Ranges: the type's bits, the order of the bounds, and each bound inside the type.
    let types = [
        (
            IntType {
                bits: 0,
                signed: false,
            },
            0,
            0,
        ),
        (
            IntType {
                bits: 65,
                signed: false,
            },
            0,
            1,
        ),
        (BIT, 1, 0),
        (BIT, 0, 2),
        (BIT, -1, 1),
        (INT8, -129, 0),
        (INT8, 0, 128),
    ];
    for (ty, lo, hi) in types {
        assert_eq!(
            b.input(&[2], Kind::Public, ty, lo, hi),
            refused(Rule::Range)
        );
    }
    // Weight-committed inputs: a role that is not the weight commitment's.
    let q = WeightRef {
        ty: WeightType::Q,
        layer: 0,
    };
    for role in [Role::Tokens, Role::Positions, Role::Input(0)] {
        assert_eq!(
            b.weight(role, &[4, 4], INT8, -127, 127),
            refused(Rule::InputKind)
        );
    }
    // Constants, shift-class sets, tables, checkpoints.
    assert_eq!(b.constant(&[]), refused(Rule::Constant));
    for shifts in [
        &[][..],
        &[1, 2, 3, 4, 5][..],
        &[0][..],
        &[64][..],
        &[3, 0][..],
    ] {
        assert_eq!(b.shift_classes(shifts), refused(Rule::ShiftClasses));
    }
    for kind in [
        TableKind::Range { bits: 8 },
        TableKind::Pow2 { max_exp: 8 },
        TableKind::Exp2Frac {
            frac_bits: 4,
            out_bits: 8,
        },
    ] {
        assert_eq!(b.table(kind), refused(Rule::Table));
    }
    assert_eq!(b.checkpoint(TensorId(0)), refused(Rule::Checkpoint));
    assert_eq!(&before, b.graph(), "a refused declaration adds nothing");

    // Their honest twins pass.
    let x = b.input(&[2], Kind::Public, BIT, 0, 1).unwrap();
    assert_eq!(x, TensorId(0));
    assert!(b.input(&[1, 1, 1, 1], Kind::Advice, BYTE, 0, 255).is_ok());
    assert!(b.input(&[2], Kind::Public, INT8, -128, 127).is_ok());
    assert!(
        b.input(
            &[2],
            Kind::Public,
            IntType {
                bits: 64,
                signed: false
            },
            0,
            i64::MAX
        )
        .is_ok()
    );
    assert!(b.weight(Role::Weight(q), &[4, 4], INT8, -127, 127).is_ok());
    assert_eq!(
        b.weight(Role::Weight(q), &[4, 4], INT8, -127, 127),
        refused(Rule::Weight),
        "a weight declared twice"
    );
    assert!(b.weight(Role::Classes(q), &[4], BIT, 0, 1).is_ok());
    assert_eq!(
        b.weight(Role::Classes(q), &[4], BIT, 0, 1),
        refused(Rule::Weight)
    );
    assert_eq!(b.constant(&[7]), Ok(0));
    assert_eq!(b.constant(&[1, 2, 3]), Ok(1));
    assert_eq!(b.shift_classes(&[1, 63]), Ok(0));
    assert_eq!(b.checkpoint(x), Ok(()));
}

#[test]
fn tables_are_the_anchors_numbered_in_declaration_order_and_equal_kinds_shared() {
    let mut b = GraphBuilder::new(roomy()).unwrap();
    let xor = TableKind::Bitwise8 { op: BitOp::Xor };
    assert_eq!(b.table(xor.clone()), Ok(TableId(0)));
    assert_eq!(b.table(TableKind::AesSbox), Ok(TableId(1)));
    assert_eq!(b.table(TableKind::Gf256Mul { c: 2 }), Ok(TableId(2)));
    assert_eq!(b.table(TableKind::Gf256Mul { c: 3 }), Ok(TableId(3)));
    assert_eq!(
        b.table(xor),
        Ok(TableId(0)),
        "an equal kind is the earlier table"
    );
    assert_eq!(
        b.table(TableKind::Bitwise8 { op: BitOp::And }),
        Ok(TableId(4))
    );
    let g = b.finish();
    let rows: Vec<(u32, u8)> = g.tables().iter().map(|t| (t.id.0, t.log2_rows)).collect();
    // RFC-0008 'Table kinds': `2^16` rows for `Bitwise8`, `2^8` for the others.
    assert_eq!(rows, [(0, 16), (1, 8), (2, 8), (3, 8), (4, 16)]);
}

#[test]
fn the_capacity_bounds_every_collection_before_it_grows() {
    let tiny = Capacity {
        tensors: 2,
        nodes: 1,
        constant_values: 3,
    };
    let mut b = GraphBuilder::new(tiny).unwrap();
    let x = b.input(&[2], Kind::Public, BIT, 0, 1).unwrap();
    assert_eq!(b.constant(&[1, 2, 3, 4]), Err(BuildError::Capacity));
    assert_eq!(b.constant(&[1, 2, 3]), Ok(0));
    assert_eq!(b.constant(&[1]), Err(BuildError::Capacity));
    assert_eq!(b.shift_classes(&[1]), Ok(0));
    assert_eq!(
        b.shift_classes(&[1]),
        Err(BuildError::Capacity),
        "sets: at most `nodes`"
    );
    assert_eq!(b.table(TableKind::AesSbox), Ok(TableId(0)));
    assert_eq!(
        b.table(TableKind::Gf256Mul { c: 2 }),
        Err(BuildError::Capacity),
        "tables: at most `nodes`"
    );
    assert_eq!(
        b.table(TableKind::AesSbox),
        Ok(TableId(0)),
        "an earlier table costs nothing"
    );
    let before = b.graph().clone();
    assert_eq!(
        b.node(
            Op::SoftmaxRow {
                exp_table: TableId(0),
                sigma: 8
            },
            &[x],
            &[(&[2], Kind::Advice), (&[2], Kind::Advice)]
        ),
        Err(BuildError::Capacity),
        "two outputs where one tensor is left"
    );
    assert_eq!(&before, b.graph());
    let y = b
        .node1(Op::Sum { axis: 0 }, &[x], &[1], Kind::Virtual)
        .unwrap();
    assert_eq!(
        b.node1(Op::Sum { axis: 0 }, &[y], &[1], Kind::Virtual),
        Err(BuildError::Capacity)
    );
    assert_eq!(
        b.input(&[1], Kind::Public, BIT, 0, 1),
        Err(BuildError::Capacity)
    );
    assert_eq!(b.checkpoint(y), Ok(()));
    assert_eq!(b.checkpoint(y), Ok(()));
    assert_eq!(
        b.checkpoint(y),
        Err(BuildError::Capacity),
        "checkpoints: at most `tensors`"
    );
    let graph = b.finish();
    assert_eq!(
        GraphBuilder::extend(graph.clone(), Capacity { tensors: 1, ..tiny }).err(),
        Some(BuildError::Capacity),
        "a graph already past the capacity"
    );
    assert!(GraphBuilder::extend(graph, tiny).is_ok());
    assert_eq!(BuildError::Capacity.code(), "ERR-INTERNAL");
    assert_eq!(BuildError::Memory.code(), "ERR-RES-MEMORY");
    assert_eq!(
        BuildError::Node {
            node: NodeId(0),
            rule: Rule::Equal
        }
        .code(),
        "ERR-INTERNAL"
    );
}

// ----- Nodes -------------------------------------------------------------------------------------

/// A lowered graph extended with the builder's inputs, constants, sets and tables, which each
/// case reads.
struct Fixture {
    b: GraphBuilder,
    /// Public bits `[2, 4]`, three of them, and `[3, 4]`.
    a: TensorId,
    a2: TensorId,
    a3: TensorId,
    other: TensorId,
    /// Committed bytes `[4]`, two of them, and values `[2]` below `2^35`.
    bytes: TensorId,
    bytes2: TensorId,
    values: TensorId,
    /// Public `[2, 3, 4]`, `[2, 4, 5]` and `[2, 5, 4]`.
    x3: TensorId,
    y3: TensorId,
    z3: TensorId,
    /// Public `[1, 1, 1, 2]`, of four axes.
    x4: TensorId,
    /// Public `[2]` and `[4]`.
    rows: TensorId,
    channels: TensorId,
    /// A weight `[5, 4]` of `w`, its classes `[4]` and its multipliers `[3]`.
    w: WeightRef,
    matrix: TensorId,
    classes: TensorId,
    multipliers: TensorId,
    /// The lowering's positions `[4]`, and a public `[4, 2, 6]` to rotate.
    positions: TensorId,
    rotated: TensorId,
    /// Public scores `[2, 3, 3]`.
    scores: TensorId,
    /// Constants: `[3]`, one per channel of 4, of 3, the indices `[1, 0, 1]`, `[2]` and `[-1]`.
    one: u32,
    per4: u32,
    per3: u32,
    gather: u32,
    past: u32,
    negative: u32,
    /// Shift-class sets of one class and of two.
    set1: u32,
    set2: u32,
    /// The lowering's `Rope` and `Exp` tables; the builder's `Bitwise8 { Xor }` and `AesSbox`.
    rope: TableId,
    exp: TableId,
    xor: TableId,
    sbox: TableId,
}

fn fixture() -> Fixture {
    let (_, graph) = lowered();
    let positions = graph
        .inputs()
        .iter()
        .find(|i| i.role == Role::Positions)
        .unwrap()
        .tensor;
    let exp = graph
        .tables()
        .iter()
        .find(|t| matches!(t.kind, TableKind::Exp { .. }))
        .unwrap()
        .id;
    let rope = graph
        .tables()
        .iter()
        .find(|t| matches!(t.kind, TableKind::Rope { .. }))
        .unwrap()
        .id;
    let mut b = GraphBuilder::extend(graph.clone(), room_after(&graph)).unwrap();
    let public =
        |b: &mut GraphBuilder, dims: &[u32]| b.input(dims, Kind::Public, INT8, -128, 127).unwrap();
    let a = b.input(&[2, 4], Kind::Public, BIT, 0, 1).unwrap();
    let a2 = b.input(&[2, 4], Kind::Public, BIT, 0, 1).unwrap();
    let a3 = b.input(&[2, 4], Kind::Public, BIT, 0, 1).unwrap();
    let other = b.input(&[3, 4], Kind::Public, BIT, 0, 1).unwrap();
    let bytes = b.input(&[4], Kind::Advice, BYTE, 0, 255).unwrap();
    let bytes2 = b.input(&[4], Kind::Advice, BYTE, 0, 255).unwrap();
    let values = b.input(&[2], Kind::Advice, WORD, 0, (1 << 35) - 1).unwrap();
    let x3 = public(&mut b, &[2, 3, 4]);
    let y3 = public(&mut b, &[2, 4, 5]);
    let z3 = public(&mut b, &[2, 5, 4]);
    let x4 = public(&mut b, &[1, 1, 1, 2]);
    let rows = public(&mut b, &[2]);
    let channels = public(&mut b, &[4]);
    let w = WeightRef {
        ty: WeightType::Q,
        layer: 9,
    };
    let matrix = b.weight(Role::Weight(w), &[5, 4], INT8, -127, 127).unwrap();
    let classes = b.weight(Role::Classes(w), &[4], BIT, 0, 1).unwrap();
    let multipliers = b.weight(Role::Multipliers(w), &[3], BIT, 0, 1).unwrap();
    let rotated = public(&mut b, &[4, 2, 6]);
    let scores = public(&mut b, &[2, 3, 3]);
    let one = b.constant(&[3]).unwrap();
    let per4 = b.constant(&[1, 2, 3, 4]).unwrap();
    let per3 = b.constant(&[1, 2, 3]).unwrap();
    let gather = b.constant(&[1, 0, 1]).unwrap();
    let past = b.constant(&[2]).unwrap();
    let negative = b.constant(&[-1]).unwrap();
    let set1 = b.shift_classes(&[8]).unwrap();
    let set2 = b.shift_classes(&[8, 9]).unwrap();
    let xor = b.table(TableKind::Bitwise8 { op: BitOp::Xor }).unwrap();
    let sbox = b.table(TableKind::AesSbox).unwrap();
    Fixture {
        b,
        a,
        a2,
        a3,
        other,
        bytes,
        bytes2,
        values,
        x3,
        y3,
        z3,
        x4,
        rows,
        channels,
        w,
        matrix,
        classes,
        multipliers,
        positions,
        rotated,
        scores,
        one,
        per4,
        per3,
        gather,
        past,
        negative,
        set1,
        set2,
        rope,
        exp,
        xor,
        sbox,
    }
}

/// One node call: what it builds, and the rule it breaks, `None` for an honest call.
type Call = fn(&mut Fixture) -> Result<Node, BuildError>;

/// Runs `call` on a fresh fixture: an honest call adds exactly one node; a refused one is
/// `BuildError::Node` with `rule` at the next node's id, and adds nothing.
fn run(name: &str, rule: Option<Rule>, call: Call) {
    let mut f = fixture();
    let next = next_node(&f.b);
    let before = f.b.graph().clone();
    let result = call(&mut f);
    match rule {
        None => {
            let node =
                result.unwrap_or_else(|e| panic!("{name}: the honest call is refused: {e:?}"));
            assert_eq!(
                f.b.graph().nodes().len(),
                before.nodes().len() + 1,
                "{name}"
            );
            assert_eq!(f.b.graph().node(next), Some(&node), "{name}");
            for &out in node.outputs() {
                assert_eq!(
                    f.b.graph().tensor(out).unwrap().producer,
                    Some(next),
                    "{name}"
                );
            }
        }
        Some(rule) => {
            assert_eq!(result, Err(BuildError::Node { node: next, rule }), "{name}");
            assert_eq!(&before, f.b.graph(), "{name}: a refused node adds nothing");
        }
    }
}

fn one(
    f: &mut Fixture,
    op: Op,
    inputs: &[TensorId],
    dims: &[u32],
    kind: Kind,
) -> Result<Node, BuildError> {
    f.b.node(op, inputs, &[(dims, kind)])
}

#[test]
fn the_general_rules_broken_alone_are_refused() {
    let cases: &[(&str, Option<Rule>, Call)] = &[
        ("an honest Add", None, |f| {
            one(f, Op::Add, &[f.a, f.a2], &[2, 4], Kind::Virtual)
        }),
        ("a committed Add", None, |f| {
            one(f, Op::Add, &[f.a, f.a2], &[2, 4], Kind::Advice)
        }),
        ("one input to Add", Some(Rule::Inputs), |f| {
            one(f, Op::Add, &[f.a], &[2, 4], Kind::Virtual)
        }),
        ("three inputs to Add", Some(Rule::Inputs), |f| {
            one(f, Op::Add, &[f.a, f.a2, f.a3], &[2, 4], Kind::Virtual)
        }),
        ("two outputs of Add", Some(Rule::Outputs), |f| {
            f.b.node(
                Op::Add,
                &[f.a, f.a2],
                &[(&[2, 4], Kind::Virtual), (&[2, 4], Kind::Virtual)],
            )
        }),
        ("a dangling input", Some(Rule::Dangling), |f| {
            one(
                f,
                Op::Add,
                &[f.a, TensorId(1_000_000)],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("an output of no axis", Some(Rule::Shape), |f| {
            one(f, Op::Add, &[f.a, f.a2], &[], Kind::Virtual)
        }),
        ("an output axis of 0", Some(Rule::Shape), |f| {
            one(f, Op::Add, &[f.a, f.a2], &[2, 0], Kind::Virtual)
        }),
        ("an output of five axes", Some(Rule::Shape), |f| {
            one(f, Op::Add, &[f.a, f.a2], &[1, 1, 1, 2, 4], Kind::Virtual)
        }),
        ("a public output", Some(Rule::OutputKind), |f| {
            one(f, Op::Add, &[f.a, f.a2], &[2, 4], Kind::Public)
        }),
        ("a weight output", Some(Rule::OutputKind), |f| {
            one(f, Op::Add, &[f.a, f.a2], &[2, 4], Kind::Weight)
        }),
        ("a lookup output of Add", Some(Rule::OutputKind), |f| {
            let t = f.sbox;
            one(f, Op::Add, &[f.a, f.a2], &[2, 4], Kind::Lookup { table: t })
        }),
    ];
    for &(name, rule, call) in cases {
        run(name, rule, call);
    }
}

#[test]
fn the_linear_operations_rules_broken_alone_are_refused() {
    let cases: &[(&str, Option<Rule>, Call)] = &[
        // Add and Sub: equal shapes.
        ("Sub", None, |f| {
            one(f, Op::Sub, &[f.a, f.a2], &[2, 4], Kind::Virtual)
        }),
        ("Sub of two shapes", Some(Rule::Equal), |f| {
            one(f, Op::Sub, &[f.a, f.other], &[2, 4], Kind::Virtual)
        }),
        ("Add of two shapes", Some(Rule::Equal), |f| {
            one(f, Op::Add, &[f.other, f.a], &[3, 4], Kind::Virtual)
        }),
        ("Sub's output", Some(Rule::Output), |f| {
            one(f, Op::Sub, &[f.a, f.a2], &[3, 4], Kind::Virtual)
        }),
        // ConstMul and ConstAdd: a scalar or one value per channel.
        ("ConstMul by a scalar", None, |f| {
            one(f, Op::ConstMul { c: f.one }, &[f.a], &[2, 4], Kind::Virtual)
        }),
        ("ConstMul per channel", None, |f| {
            one(
                f,
                Op::ConstMul { c: f.per4 },
                &[f.a],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("ConstAdd per channel, committed", None, |f| {
            one(f, Op::ConstAdd { c: f.per4 }, &[f.a], &[2, 4], Kind::Advice)
        }),
        ("ConstAdd by a scalar on one channel", None, |f| {
            one(
                f,
                Op::ConstAdd { c: f.one },
                &[f.values],
                &[2],
                Kind::Virtual,
            )
        }),
        ("ConstMul's missing constant", Some(Rule::Constant), |f| {
            one(f, Op::ConstMul { c: 99 }, &[f.a], &[2, 4], Kind::Virtual)
        }),
        ("ConstAdd's missing constant", Some(Rule::Constant), |f| {
            one(f, Op::ConstAdd { c: 99 }, &[f.a], &[2, 4], Kind::Virtual)
        }),
        (
            "ConstMul of 3 values on 4 channels",
            Some(Rule::ConstantLength),
            |f| {
                one(
                    f,
                    Op::ConstMul { c: f.per3 },
                    &[f.a],
                    &[2, 4],
                    Kind::Virtual,
                )
            },
        ),
        (
            "ConstAdd of 4 values on 2 channels",
            Some(Rule::ConstantLength),
            |f| {
                one(
                    f,
                    Op::ConstAdd { c: f.per4 },
                    &[f.values],
                    &[2],
                    Kind::Virtual,
                )
            },
        ),
        ("ConstAdd's output", Some(Rule::Output), |f| {
            one(f, Op::ConstAdd { c: f.one }, &[f.a], &[4, 2], Kind::Virtual)
        }),
        // Sum and SumSquares.
        ("Sum", None, |f| {
            one(f, Op::Sum { axis: 0 }, &[f.a], &[4], Kind::Virtual)
        }),
        ("Sum of the only axis", None, |f| {
            one(f, Op::Sum { axis: 0 }, &[f.values], &[1], Kind::Virtual)
        }),
        ("SumSquares", None, |f| {
            one(f, Op::SumSquares { axis: 1 }, &[f.a], &[2], Kind::Virtual)
        }),
        ("Sum past the axes", Some(Rule::Axis), |f| {
            one(f, Op::Sum { axis: 2 }, &[f.a], &[2], Kind::Virtual)
        }),
        ("SumSquares past the axes", Some(Rule::Axis), |f| {
            one(f, Op::SumSquares { axis: 2 }, &[f.a], &[2], Kind::Virtual)
        }),
        ("Sum's output", Some(Rule::Output), |f| {
            one(f, Op::Sum { axis: 0 }, &[f.a], &[2], Kind::Virtual)
        }),
        // Reshape.
        ("Reshape", None, |f| {
            one(f, Op::Reshape, &[f.a], &[8], Kind::Virtual)
        }),
        ("Reshape of another count", Some(Rule::Elements), |f| {
            one(f, Op::Reshape, &[f.a], &[9], Kind::Virtual)
        }),
        // Slice.
        ("Slice", None, |f| {
            one(
                f,
                Op::Slice {
                    axis: 1,
                    start: 1,
                    len: 3,
                },
                &[f.a],
                &[2, 3],
                Kind::Virtual,
            )
        }),
        ("Slice past the axes", Some(Rule::Axis), |f| {
            one(
                f,
                Op::Slice {
                    axis: 2,
                    start: 0,
                    len: 1,
                },
                &[f.a],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("Slice of no row", Some(Rule::Selection), |f| {
            one(
                f,
                Op::Slice {
                    axis: 1,
                    start: 0,
                    len: 0,
                },
                &[f.a],
                &[2, 1],
                Kind::Virtual,
            )
        }),
        ("Slice past its axis", Some(Rule::Selection), |f| {
            one(
                f,
                Op::Slice {
                    axis: 1,
                    start: 2,
                    len: 3,
                },
                &[f.a],
                &[2, 3],
                Kind::Virtual,
            )
        }),
        ("Slice whose end overflows", Some(Rule::Selection), |f| {
            one(
                f,
                Op::Slice {
                    axis: 1,
                    start: u32::MAX,
                    len: 1,
                },
                &[f.a],
                &[2, 1],
                Kind::Virtual,
            )
        }),
        ("Slice's output", Some(Rule::Output), |f| {
            one(
                f,
                Op::Slice {
                    axis: 1,
                    start: 1,
                    len: 3,
                },
                &[f.a],
                &[2, 2],
                Kind::Virtual,
            )
        }),
        // GatherPublic.
        ("GatherPublic", None, |f| {
            one(
                f,
                Op::GatherPublic { index: f.gather },
                &[f.a],
                &[3, 4],
                Kind::Virtual,
            )
        }),
        (
            "GatherPublic's missing constant",
            Some(Rule::Constant),
            |f| {
                one(
                    f,
                    Op::GatherPublic { index: 99 },
                    &[f.a],
                    &[3, 4],
                    Kind::Virtual,
                )
            },
        ),
        ("a gather index past the rows", Some(Rule::Selection), |f| {
            one(
                f,
                Op::GatherPublic { index: f.past },
                &[f.a],
                &[1, 4],
                Kind::Virtual,
            )
        }),
        ("a negative gather index", Some(Rule::Selection), |f| {
            one(
                f,
                Op::GatherPublic { index: f.negative },
                &[f.a],
                &[1, 4],
                Kind::Virtual,
            )
        }),
        ("GatherPublic's output", Some(Rule::Output), |f| {
            one(
                f,
                Op::GatherPublic { index: f.gather },
                &[f.a],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        // Transpose.
        ("Transpose of three axes", None, |f| {
            one(
                f,
                Op::Transpose {
                    perm: perm(&[2, 0, 1]),
                },
                &[f.x3],
                &[4, 2, 3],
                Kind::Virtual,
            )
        }),
        ("Transpose of two axes", None, |f| {
            one(
                f,
                Op::Transpose {
                    perm: perm(&[1, 0]),
                },
                &[f.a],
                &[4, 2],
                Kind::Virtual,
            )
        }),
        (
            "Transpose repeating an axis",
            Some(Rule::Permutation),
            |f| {
                one(
                    f,
                    Op::Transpose {
                        perm: perm(&[0, 0, 1]),
                    },
                    &[f.x3],
                    &[2, 2, 3],
                    Kind::Virtual,
                )
            },
        ),
        ("Transpose of another rank", Some(Rule::Permutation), |f| {
            one(
                f,
                Op::Transpose {
                    perm: perm(&[1, 0]),
                },
                &[f.x3],
                &[3, 2],
                Kind::Virtual,
            )
        }),
        ("Transpose past the axes", Some(Rule::Permutation), |f| {
            one(
                f,
                Op::Transpose {
                    perm: perm(&[0, 1, 3]),
                },
                &[f.x3],
                &[2, 3, 4],
                Kind::Virtual,
            )
        }),
        ("Transpose's output", Some(Rule::Output), |f| {
            one(
                f,
                Op::Transpose {
                    perm: perm(&[2, 0, 1]),
                },
                &[f.x3],
                &[2, 3, 4],
                Kind::Virtual,
            )
        }),
        // Expand.
        ("Expand", None, |f| {
            one(
                f,
                Op::Expand { axis: 1, len: 3 },
                &[f.a],
                &[2, 3, 4],
                Kind::Virtual,
            )
        }),
        ("Expand after the last axis", None, |f| {
            one(
                f,
                Op::Expand { axis: 2, len: 3 },
                &[f.a],
                &[2, 4, 3],
                Kind::Virtual,
            )
        }),
        ("Expand past the rank", Some(Rule::Axis), |f| {
            one(
                f,
                Op::Expand { axis: 3, len: 3 },
                &[f.a],
                &[2, 4, 3],
                Kind::Virtual,
            )
        }),
        ("Expand of no row", Some(Rule::Parameter), |f| {
            one(
                f,
                Op::Expand { axis: 1, len: 0 },
                &[f.a],
                &[2, 1, 4],
                Kind::Virtual,
            )
        }),
        ("Expand's output", Some(Rule::Output), |f| {
            one(
                f,
                Op::Expand { axis: 1, len: 3 },
                &[f.a],
                &[3, 2, 4],
                Kind::Virtual,
            )
        }),
        ("Expand to five axes", Some(Rule::Output), |f| {
            one(
                f,
                Op::Expand { axis: 0, len: 1 },
                &[f.x4],
                &[1, 1, 1, 2],
                Kind::Virtual,
            )
        }),
    ];
    for &(name, rule, call) in cases {
        run(name, rule, call);
    }
}

#[test]
fn the_products_rules_broken_alone_are_refused() {
    let cases: &[(&str, Option<Rule>, Call)] = &[
        // MatMulWeight: the weight its reference names; W [out, in], x […, in].
        ("MatMulWeight", None, |f| {
            one(
                f,
                Op::MatMulWeight { weight: f.w },
                &[f.a, f.matrix],
                &[2, 5],
                Kind::Virtual,
            )
        }),
        (
            "MatMulWeight of another reference",
            Some(Rule::Weight),
            |f| {
                let w = WeightRef { layer: 8, ..f.w };
                one(
                    f,
                    Op::MatMulWeight { weight: w },
                    &[f.a, f.matrix],
                    &[2, 5],
                    Kind::Virtual,
                )
            },
        ),
        (
            "MatMulWeight of a tensor not the weight",
            Some(Rule::Weight),
            |f| {
                one(
                    f,
                    Op::MatMulWeight { weight: f.w },
                    &[f.a, f.a2],
                    &[2, 5],
                    Kind::Virtual,
                )
            },
        ),
        (
            "MatMulWeight of an inner axis that differs",
            Some(Rule::MatMul),
            |f| {
                one(
                    f,
                    Op::MatMulWeight { weight: f.w },
                    &[f.y3, f.matrix],
                    &[2, 4, 5],
                    Kind::Virtual,
                )
            },
        ),
        ("MatMulWeight's output", Some(Rule::Output), |f| {
            one(
                f,
                Op::MatMulWeight { weight: f.w },
                &[f.a, f.matrix],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("MatMulWeight committed", Some(Rule::OutputKind), |f| {
            one(
                f,
                Op::MatMulWeight { weight: f.w },
                &[f.a, f.matrix],
                &[2, 5],
                Kind::Advice,
            )
        }),
        // MatMulAct.
        ("MatMulAct", None, |f| {
            one(
                f,
                Op::MatMulAct {
                    transpose_rhs: false,
                },
                &[f.x3, f.y3],
                &[2, 3, 5],
                Kind::Virtual,
            )
        }),
        ("MatMulAct transposed", None, |f| {
            one(
                f,
                Op::MatMulAct {
                    transpose_rhs: true,
                },
                &[f.x3, f.z3],
                &[2, 3, 5],
                Kind::Virtual,
            )
        }),
        (
            "MatMulAct of inner axes that differ",
            Some(Rule::MatMul),
            |f| {
                one(
                    f,
                    Op::MatMulAct {
                        transpose_rhs: false,
                    },
                    &[f.x3, f.z3],
                    &[2, 3, 4],
                    Kind::Virtual,
                )
            },
        ),
        (
            "MatMulAct of leading axes that differ",
            Some(Rule::MatMul),
            |f| {
                one(
                    f,
                    Op::MatMulAct {
                        transpose_rhs: false,
                    },
                    &[f.x3, f.rotated],
                    &[2, 3, 6],
                    Kind::Virtual,
                )
            },
        ),
        ("MatMulAct of ranks that differ", Some(Rule::MatMul), |f| {
            one(
                f,
                Op::MatMulAct {
                    transpose_rhs: false,
                },
                &[f.x3, f.a],
                &[2, 3, 4],
                Kind::Virtual,
            )
        }),
        ("MatMulAct of one axis", Some(Rule::MatMul), |f| {
            one(
                f,
                Op::MatMulAct {
                    transpose_rhs: false,
                },
                &[f.channels, f.channels],
                &[1],
                Kind::Virtual,
            )
        }),
        ("MatMulAct's output", Some(Rule::Output), |f| {
            one(
                f,
                Op::MatMulAct {
                    transpose_rhs: false,
                },
                &[f.x3, f.y3],
                &[2, 3, 4],
                Kind::Virtual,
            )
        }),
        ("MatMulAct committed", Some(Rule::OutputKind), |f| {
            one(
                f,
                Op::MatMulAct {
                    transpose_rhs: false,
                },
                &[f.x3, f.y3],
                &[2, 3, 5],
                Kind::Advice,
            )
        }),
        // Mul.
        ("Mul Trailing by rows", None, |f| {
            one(
                f,
                Op::Mul {
                    broadcast: Broadcast::Trailing,
                },
                &[f.a, f.rows],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("Mul Trailing of one shape", None, |f| {
            one(
                f,
                Op::Mul {
                    broadcast: Broadcast::Trailing,
                },
                &[f.a, f.a2],
                &[2, 4],
                Kind::Advice,
            )
        }),
        ("Mul Leading by channels", None, |f| {
            one(
                f,
                Op::Mul {
                    broadcast: Broadcast::Leading,
                },
                &[f.a, f.channels],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("Mul Trailing by channels", Some(Rule::Broadcast), |f| {
            one(
                f,
                Op::Mul {
                    broadcast: Broadcast::Trailing,
                },
                &[f.a, f.channels],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("Mul Leading by rows", Some(Rule::Broadcast), |f| {
            one(
                f,
                Op::Mul {
                    broadcast: Broadcast::Leading,
                },
                &[f.a, f.rows],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("Mul's output", Some(Rule::Output), |f| {
            one(
                f,
                Op::Mul {
                    broadcast: Broadcast::Trailing,
                },
                &[f.a, f.rows],
                &[2],
                Kind::Virtual,
            )
        }),
        // Rotate.
        ("Rotate", None, |f| {
            one(
                f,
                Op::Rotate { rope: f.rope },
                &[f.rotated, f.positions],
                &[4, 2, 6],
                Kind::Virtual,
            )
        }),
        ("Rotate by another table", Some(Rule::Table), |f| {
            one(
                f,
                Op::Rotate { rope: f.sbox },
                &[f.rotated, f.positions],
                &[4, 2, 6],
                Kind::Virtual,
            )
        }),
        (
            "Rotate by committed positions",
            Some(Rule::InputKind),
            |f| {
                one(
                    f,
                    Op::Rotate { rope: f.rope },
                    &[f.rotated, f.bytes],
                    &[4, 2, 6],
                    Kind::Virtual,
                )
            },
        ),
        ("Rotate of an odd width", Some(Rule::Rotate), |f| {
            one(
                f,
                Op::Rotate { rope: f.rope },
                &[f.x3, f.positions],
                &[2, 3, 4],
                Kind::Virtual,
            )
        }),
        ("Rotate of one axis", Some(Rule::Rotate), |f| {
            one(
                f,
                Op::Rotate { rope: f.rope },
                &[f.channels, f.positions],
                &[4],
                Kind::Virtual,
            )
        }),
        ("Rotate's output", Some(Rule::Output), |f| {
            one(
                f,
                Op::Rotate { rope: f.rope },
                &[f.rotated, f.positions],
                &[4, 6, 2],
                Kind::Virtual,
            )
        }),
    ];
    for &(name, rule, call) in cases {
        run(name, rule, call);
    }
}

#[test]
fn the_gadgets_rules_broken_alone_are_refused() {
    let cases: &[(&str, Option<Rule>, Call)] = &[
        // OneHot.
        ("OneHot", None, |f| {
            one(
                f,
                Op::OneHot { vocab: 5 },
                &[f.bytes],
                &[4, 5],
                Kind::Advice,
            )
        }),
        ("OneHot of no vocabulary", Some(Rule::Parameter), |f| {
            one(
                f,
                Op::OneHot { vocab: 0 },
                &[f.bytes],
                &[4, 1],
                Kind::Advice,
            )
        }),
        ("OneHot's output", Some(Rule::Output), |f| {
            one(
                f,
                Op::OneHot { vocab: 5 },
                &[f.bytes],
                &[5, 4],
                Kind::Advice,
            )
        }),
        ("OneHot virtual", Some(Rule::OutputKind), |f| {
            one(
                f,
                Op::OneHot { vocab: 5 },
                &[f.bytes],
                &[4, 5],
                Kind::Virtual,
            )
        }),
        // Requant.
        ("Requant of one class", None, |f| {
            one(
                f,
                Op::Requant {
                    shift: f.set1,
                    lo: -128,
                    hi: 127,
                },
                &[f.a],
                &[2, 4],
                Kind::Advice,
            )
        }),
        ("Requant of two classes", None, |f| {
            one(
                f,
                Op::Requant {
                    shift: f.set2,
                    lo: -128,
                    hi: 127,
                },
                &[f.a, f.classes],
                &[2, 4],
                Kind::Advice,
            )
        }),
        ("Requant's missing set", Some(Rule::ShiftClasses), |f| {
            one(
                f,
                Op::Requant {
                    shift: 99,
                    lo: -128,
                    hi: 127,
                },
                &[f.a],
                &[2, 4],
                Kind::Advice,
            )
        }),
        (
            "Requant's bounds out of order",
            Some(Rule::Parameter),
            |f| {
                one(
                    f,
                    Op::Requant {
                        shift: f.set1,
                        lo: 1,
                        hi: 0,
                    },
                    &[f.a],
                    &[2, 4],
                    Kind::Advice,
                )
            },
        ),
        ("classes of one class", Some(Rule::Classes), |f| {
            one(
                f,
                Op::Requant {
                    shift: f.set1,
                    lo: -128,
                    hi: 127,
                },
                &[f.a, f.classes],
                &[2, 4],
                Kind::Advice,
            )
        }),
        ("no classes for two", Some(Rule::Classes), |f| {
            one(
                f,
                Op::Requant {
                    shift: f.set2,
                    lo: -128,
                    hi: 127,
                },
                &[f.a],
                &[2, 4],
                Kind::Advice,
            )
        }),
        ("classes of another length", Some(Rule::Classes), |f| {
            one(
                f,
                Op::Requant {
                    shift: f.set2,
                    lo: -128,
                    hi: 127,
                },
                &[f.a, f.multipliers],
                &[2, 4],
                Kind::Advice,
            )
        }),
        ("classes not weight-committed", Some(Rule::InputKind), |f| {
            one(
                f,
                Op::Requant {
                    shift: f.set2,
                    lo: -128,
                    hi: 127,
                },
                &[f.a, f.channels],
                &[2, 4],
                Kind::Advice,
            )
        }),
        ("Requant virtual", Some(Rule::OutputKind), |f| {
            one(
                f,
                Op::Requant {
                    shift: f.set1,
                    lo: -128,
                    hi: 127,
                },
                &[f.a],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("Requant's output", Some(Rule::Output), |f| {
            one(
                f,
                Op::Requant {
                    shift: f.set1,
                    lo: -128,
                    hi: 127,
                },
                &[f.a],
                &[4, 2],
                Kind::Advice,
            )
        }),
        // TableMap into the anchor's tables.
        ("TableMap into AesSbox", None, |f| {
            let t = f.sbox;
            one(
                f,
                Op::TableMap { table: t },
                &[f.bytes],
                &[4],
                Kind::Lookup { table: t },
            )
        }),
        ("TableMap into Gf256Mul", None, |f| {
            let t = f.b.table(TableKind::Gf256Mul { c: 3 })?;
            one(
                f,
                Op::TableMap { table: t },
                &[f.bytes],
                &[4],
                Kind::Lookup { table: t },
            )
        }),
        ("TableMap into Bitwise8", Some(Rule::Table), |f| {
            let t = f.xor;
            one(
                f,
                Op::TableMap { table: t },
                &[f.bytes],
                &[4],
                Kind::Lookup { table: t },
            )
        }),
        ("TableMap into an Exp table", Some(Rule::Table), |f| {
            let t = f.exp;
            one(
                f,
                Op::TableMap { table: t },
                &[f.bytes],
                &[4],
                Kind::Lookup { table: t },
            )
        }),
        ("TableMap into no table", Some(Rule::Table), |f| {
            let t = TableId(999);
            one(
                f,
                Op::TableMap { table: t },
                &[f.bytes],
                &[4],
                Kind::Lookup { table: t },
            )
        }),
        ("classes for an S-box", Some(Rule::Classes), |f| {
            let t = f.sbox;
            one(
                f,
                Op::TableMap { table: t },
                &[f.bytes, f.classes],
                &[4],
                Kind::Lookup { table: t },
            )
        }),
        (
            "TableMap into another table's output",
            Some(Rule::OutputKind),
            |f| {
                let (t, other) = (f.sbox, f.xor);
                one(
                    f,
                    Op::TableMap { table: t },
                    &[f.bytes],
                    &[4],
                    Kind::Lookup { table: other },
                )
            },
        ),
        ("TableMap advice", Some(Rule::OutputKind), |f| {
            let t = f.sbox;
            one(f, Op::TableMap { table: t }, &[f.bytes], &[4], Kind::Advice)
        }),
        ("TableMap's output", Some(Rule::Output), |f| {
            let t = f.sbox;
            one(
                f,
                Op::TableMap { table: t },
                &[f.bytes],
                &[2, 2],
                Kind::Lookup { table: t },
            )
        }),
        // Bitwise.
        ("Bitwise", None, |f| {
            let t = f.xor;
            one(
                f,
                Op::Bitwise { table: t },
                &[f.bytes, f.bytes2],
                &[4],
                Kind::Lookup { table: t },
            )
        }),
        ("Bitwise into an S-box", Some(Rule::Table), |f| {
            let t = f.sbox;
            one(
                f,
                Op::Bitwise { table: t },
                &[f.bytes, f.bytes2],
                &[4],
                Kind::Lookup { table: t },
            )
        }),
        ("Bitwise of two shapes", Some(Rule::Equal), |f| {
            let t = f.xor;
            one(
                f,
                Op::Bitwise { table: t },
                &[f.bytes, f.rows],
                &[4],
                Kind::Lookup { table: t },
            )
        }),
        ("Bitwise of one operand", Some(Rule::Inputs), |f| {
            let t = f.xor;
            one(
                f,
                Op::Bitwise { table: t },
                &[f.bytes],
                &[4],
                Kind::Lookup { table: t },
            )
        }),
        ("Bitwise virtual", Some(Rule::OutputKind), |f| {
            let t = f.xor;
            one(
                f,
                Op::Bitwise { table: t },
                &[f.bytes, f.bytes2],
                &[4],
                Kind::Virtual,
            )
        }),
        ("Bitwise's output", Some(Rule::Output), |f| {
            let t = f.xor;
            one(
                f,
                Op::Bitwise { table: t },
                &[f.bytes, f.bytes2],
                &[2, 2],
                Kind::Lookup { table: t },
            )
        }),
        // SoftmaxRow.
        ("SoftmaxRow", None, |f| {
            let t = f.exp;
            f.b.node(
                Op::SoftmaxRow {
                    exp_table: t,
                    sigma: 16,
                },
                &[f.scores],
                &[
                    (&[2, 3, 3], Kind::Lookup { table: t }),
                    (&[2, 3], Kind::Advice),
                ],
            )
        }),
        ("SoftmaxRow by another table", Some(Rule::Table), |f| {
            let t = f.sbox;
            f.b.node(
                Op::SoftmaxRow {
                    exp_table: t,
                    sigma: 16,
                },
                &[f.scores],
                &[
                    (&[2, 3, 3], Kind::Lookup { table: t }),
                    (&[2, 3], Kind::Advice),
                ],
            )
        }),
        (
            "SoftmaxRow of keys unlike queries",
            Some(Rule::Softmax),
            |f| {
                let t = f.exp;
                f.b.node(
                    Op::SoftmaxRow {
                        exp_table: t,
                        sigma: 16,
                    },
                    &[f.x3],
                    &[
                        (&[2, 3, 4], Kind::Lookup { table: t }),
                        (&[2, 3], Kind::Advice),
                    ],
                )
            },
        ),
        ("SoftmaxRow's reciprocals", Some(Rule::Output), |f| {
            let t = f.exp;
            f.b.node(
                Op::SoftmaxRow {
                    exp_table: t,
                    sigma: 16,
                },
                &[f.scores],
                &[
                    (&[2, 3, 3], Kind::Lookup { table: t }),
                    (&[2, 3, 3], Kind::Advice),
                ],
            )
        }),
        ("SoftmaxRow of one output", Some(Rule::Outputs), |f| {
            let t = f.exp;
            one(
                f,
                Op::SoftmaxRow {
                    exp_table: t,
                    sigma: 16,
                },
                &[f.scores],
                &[2, 3, 3],
                Kind::Lookup { table: t },
            )
        }),
        // RsqrtRow.
        ("RsqrtRow", None, |f| {
            one(
                f,
                Op::RsqrtRow {
                    kappa: 4,
                    eps_int: 1,
                    dim: 4,
                },
                &[f.channels],
                &[4],
                Kind::Advice,
            )
        }),
        ("RsqrtRow of no width", Some(Rule::Parameter), |f| {
            one(
                f,
                Op::RsqrtRow {
                    kappa: 4,
                    eps_int: 1,
                    dim: 0,
                },
                &[f.channels],
                &[4],
                Kind::Advice,
            )
        }),
        ("RsqrtRow of κ 32", Some(Rule::Parameter), |f| {
            one(
                f,
                Op::RsqrtRow {
                    kappa: 32,
                    eps_int: 1,
                    dim: 4,
                },
                &[f.channels],
                &[4],
                Kind::Advice,
            )
        }),
        ("RsqrtRow's output", Some(Rule::Output), |f| {
            one(
                f,
                Op::RsqrtRow {
                    kappa: 4,
                    eps_int: 1,
                    dim: 4,
                },
                &[f.channels],
                &[2],
                Kind::Advice,
            )
        }),
    ];
    for &(name, rule, call) in cases {
        run(name, rule, call);
    }
}

#[test]
fn the_bit_operations_rules_broken_alone_are_refused() {
    let cases: &[(&str, Option<Rule>, Call)] = &[
        // BitDecompose { low, high } (RFC-0066): the bit axes innermost.
        ("BitDecompose { 8, 0 }", None, |f| {
            one(
                f,
                Op::BitDecompose { low: 8, high: 0 },
                &[f.bytes],
                &[4, 8],
                Kind::Advice,
            )
        }),
        ("BitDecompose { 32, 3 }", None, |f| {
            f.b.node(
                Op::BitDecompose { low: 32, high: 3 },
                &[f.values],
                &[(&[2, 32], Kind::Advice), (&[2, 3], Kind::Advice)],
            )
        }),
        ("BitDecompose { 1, 61 }", None, |f| {
            f.b.node(
                Op::BitDecompose { low: 1, high: 61 },
                &[f.values],
                &[(&[2, 1], Kind::Advice), (&[2, 61], Kind::Advice)],
            )
        }),
        (
            "BitDecompose of no result bit",
            Some(Rule::Parameter),
            |f| {
                f.b.node(
                    Op::BitDecompose { low: 0, high: 3 },
                    &[f.values],
                    &[(&[2, 1], Kind::Advice), (&[2, 3], Kind::Advice)],
                )
            },
        ),
        ("BitDecompose of 63 bits", Some(Rule::Parameter), |f| {
            f.b.node(
                Op::BitDecompose { low: 32, high: 31 },
                &[f.values],
                &[(&[2, 32], Kind::Advice), (&[2, 31], Kind::Advice)],
            )
        }),
        ("BitDecompose of 510 bits", Some(Rule::Parameter), |f| {
            f.b.node(
                Op::BitDecompose {
                    low: 255,
                    high: 255,
                },
                &[f.values],
                &[(&[2, 255], Kind::Advice), (&[2, 255], Kind::Advice)],
            )
        }),
        ("a carry output with no carry", Some(Rule::Outputs), |f| {
            f.b.node(
                Op::BitDecompose { low: 8, high: 0 },
                &[f.bytes],
                &[(&[4, 8], Kind::Advice), (&[4, 1], Kind::Advice)],
            )
        }),
        ("no carry output", Some(Rule::Outputs), |f| {
            one(
                f,
                Op::BitDecompose { low: 32, high: 3 },
                &[f.values],
                &[2, 32],
                Kind::Advice,
            )
        }),
        ("the result bits outermost", Some(Rule::Output), |f| {
            one(
                f,
                Op::BitDecompose { low: 8, high: 0 },
                &[f.bytes],
                &[8, 4],
                Kind::Advice,
            )
        }),
        ("the carry's width", Some(Rule::Output), |f| {
            f.b.node(
                Op::BitDecompose { low: 32, high: 3 },
                &[f.values],
                &[(&[2, 32], Kind::Advice), (&[2, 4], Kind::Advice)],
            )
        }),
        ("virtual bits", Some(Rule::OutputKind), |f| {
            one(
                f,
                Op::BitDecompose { low: 8, high: 0 },
                &[f.bytes],
                &[4, 8],
                Kind::Virtual,
            )
        }),
        (
            "bits declared lookup outputs",
            Some(Rule::OutputKind),
            |f| {
                let t = f.sbox;
                one(
                    f,
                    Op::BitDecompose { low: 8, high: 0 },
                    &[f.bytes],
                    &[4, 8],
                    Kind::Lookup { table: t },
                )
            },
        ),
        // BitFn { f }: two or three inputs, one shape, a virtual output.
        ("Xor", None, |f| {
            one(
                f,
                Op::BitFn { f: BoolFn::Xor },
                &[f.a, f.a2],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("And", None, |f| {
            one(
                f,
                Op::BitFn { f: BoolFn::And },
                &[f.a, f.a2],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("Xor3", None, |f| {
            one(
                f,
                Op::BitFn { f: BoolFn::Xor3 },
                &[f.a, f.a2, f.a3],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("Ch", None, |f| {
            one(
                f,
                Op::BitFn { f: BoolFn::Ch },
                &[f.a, f.a2, f.a3],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("Maj", None, |f| {
            one(
                f,
                Op::BitFn { f: BoolFn::Maj },
                &[f.a, f.a2, f.a3],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("Xor of three", Some(Rule::Inputs), |f| {
            one(
                f,
                Op::BitFn { f: BoolFn::Xor },
                &[f.a, f.a2, f.a3],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("And of one", Some(Rule::Inputs), |f| {
            one(
                f,
                Op::BitFn { f: BoolFn::And },
                &[f.a],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("Ch of two", Some(Rule::Inputs), |f| {
            one(
                f,
                Op::BitFn { f: BoolFn::Ch },
                &[f.a, f.a2],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("Maj of two shapes", Some(Rule::Equal), |f| {
            one(
                f,
                Op::BitFn { f: BoolFn::Maj },
                &[f.a, f.a2, f.other],
                &[2, 4],
                Kind::Virtual,
            )
        }),
        ("Xor of two shapes", Some(Rule::Equal), |f| {
            one(
                f,
                Op::BitFn { f: BoolFn::Xor },
                &[f.other, f.a],
                &[3, 4],
                Kind::Virtual,
            )
        }),
        ("BitFn's output", Some(Rule::Output), |f| {
            one(
                f,
                Op::BitFn { f: BoolFn::Xor3 },
                &[f.a, f.a2, f.a3],
                &[4, 2],
                Kind::Virtual,
            )
        }),
        ("a committed BitFn", Some(Rule::OutputKind), |f| {
            one(
                f,
                Op::BitFn { f: BoolFn::Xor },
                &[f.a, f.a2],
                &[2, 4],
                Kind::Advice,
            )
        }),
        // AssertEqual: two tensors of one shape, no output.
        ("AssertEqual", None, |f| {
            f.b.node(Op::AssertEqual, &[f.a, f.a2], &[])
        }),
        ("AssertEqual of two shapes", Some(Rule::Equal), |f| {
            f.b.node(Op::AssertEqual, &[f.a, f.other], &[])
        }),
        ("AssertEqual of one tensor", Some(Rule::Inputs), |f| {
            f.b.node(Op::AssertEqual, &[f.a], &[])
        }),
        ("AssertEqual with an output", Some(Rule::Outputs), |f| {
            one(f, Op::AssertEqual, &[f.a, f.a2], &[2, 4], Kind::Virtual)
        }),
    ];
    for &(name, rule, call) in cases {
        run(name, rule, call);
    }
}

#[test]
fn rules_are_checked_in_the_documented_order() {
    // Each call breaks two rules; the earlier in the module documentation's order is the error.
    let cases: &[(&str, Option<Rule>, Call)] = &[
        ("arity before a dangling input", Some(Rule::Inputs), |f| {
            one(f, Op::Add, &[TensorId(1_000_000)], &[2, 4], Kind::Virtual)
        }),
        (
            "a dangling input before the output's shape",
            Some(Rule::Dangling),
            |f| one(f, Op::Add, &[f.a, TensorId(1_000_000)], &[], Kind::Virtual),
        ),
        (
            "the output's shape before its kind",
            Some(Rule::Shape),
            |f| one(f, Op::Add, &[f.a, f.a2], &[0], Kind::Public),
        ),
        (
            "the output's kind before the operation's rules",
            Some(Rule::OutputKind),
            |f| one(f, Op::Add, &[f.a, f.other], &[2, 4], Kind::Public),
        ),
        ("the table before the shapes", Some(Rule::Table), |f| {
            let t = f.sbox;
            one(
                f,
                Op::Bitwise { table: t },
                &[f.bytes, f.rows],
                &[9],
                Kind::Lookup { table: t },
            )
        }),
        (
            "the parameter before the output",
            Some(Rule::Parameter),
            |f| {
                one(
                    f,
                    Op::BitDecompose { low: 0, high: 0 },
                    &[f.bytes],
                    &[9],
                    Kind::Advice,
                )
            },
        ),
        ("the axis before the selection", Some(Rule::Axis), |f| {
            one(
                f,
                Op::Slice {
                    axis: 5,
                    start: 9,
                    len: 0,
                },
                &[f.a],
                &[2, 4],
                Kind::Virtual,
            )
        }),
    ];
    for &(name, rule, call) in cases {
        run(name, rule, call);
    }
}

// ----- Anchor-shaped fragments -------------------------------------------------------------------

/// The gather of `ROTR^r` on a word `[32, M]`: row `i` from row `(i + r) mod 32` (RFC-0066
/// 'Layout').
fn rotation(r: i64) -> Vec<i64> {
    (0..32).map(|i| (i + r) % 32).collect()
}

/// A SHA-256-shaped fragment over `M = 2` messages of one block and `K = 2` committed bytes each,
/// in RFC-0066 'The graph''s order: the inputs; the masks of `SHR^3` and `SHR^{10}`; the
/// message's bits from the committed bytes; an initial word; two message words; `σ0` of one;
/// `Ch` and `Maj` of three words; a sum of two words, its five value levels, its round constant
/// and its decomposition with a carry of 2; then the digest and the binding of one word.
fn sha256_fragment() -> Graph {
    let mut b = GraphBuilder::new(roomy()).unwrap();
    let m = 2u32;
    // The inputs, in RFC-0066's order.
    let message = b.input(&[512, m], Kind::Public, BIT, 0, 1).unwrap();
    let start = b.input(&[256, m], Kind::Public, BIT, 0, 1).unwrap();
    let masks = b.input(&[2, 32], Kind::Public, BIT, 0, 1).unwrap();
    let committed = b.input(&[2, m], Kind::Advice, BYTE, 0, 255).unwrap();
    let place = b.input(&[512], Kind::Public, BIT, 0, 1).unwrap();
    let v = Kind::Virtual;
    let gather = |b: &mut GraphBuilder, x: TensorId, rows: &[i64], dims: &[u32]| {
        let index = b.constant(rows).unwrap();
        b.node1(Op::GatherPublic { index }, &[x], dims, v).unwrap()
    };
    // 1. The masks: a gather of each row of `masks`, then a reshape to `[32]`.
    let mut shr = Vec::new();
    for row in 0..2 {
        let picked = gather(&mut b, masks, &[row], &[1, 32]);
        shr.push(b.node1(Op::Reshape, &[picked], &[32], v).unwrap());
    }
    // 2. The message's bits: the committed bytes' bits, moved to rows `8k + i`, gathered to their
    // word positions, placed, and XORed into the public bits.
    let bits = b
        .node1(
            Op::BitDecompose { low: 8, high: 0 },
            &[committed],
            &[2, m, 8],
            Kind::Advice,
        )
        .unwrap();
    let moved = b
        .node1(
            Op::Transpose {
                perm: perm(&[0, 2, 1]),
            },
            &[bits],
            &[2, 8, m],
            v,
        )
        .unwrap();
    let rows = b.node1(Op::Reshape, &[moved], &[16, m], v).unwrap();
    // Committed bytes 0 and 1 at offsets 4 and 5: word 1, bits 24..31 and 16..23 (big-endian).
    let placed: Vec<i64> = (0..512)
        .map(|row| match row {
            24..=31 => row - 24,
            16..=23 => 8 + row - 16,
            _ => 0,
        })
        .collect();
    let spread = gather(&mut b, rows, &placed, &[512, m]);
    let masked = b
        .node1(
            Op::Mul {
                broadcast: Broadcast::Trailing,
            },
            &[spread, place],
            &[512, m],
            v,
        )
        .unwrap();
    let assembled = b
        .node1(
            Op::BitFn { f: BoolFn::Xor },
            &[message, masked],
            &[512, m],
            v,
        )
        .unwrap();
    // 3. An initial word, a `Slice` of `start`.
    let h0 = b
        .node1(
            Op::Slice {
                axis: 0,
                start: 0,
                len: 32,
            },
            &[start],
            &[32, m],
            v,
        )
        .unwrap();
    // 4. Two message words.
    let w0 = b
        .node1(
            Op::Slice {
                axis: 0,
                start: 0,
                len: 32,
            },
            &[assembled],
            &[32, m],
            v,
        )
        .unwrap();
    let w1 = b
        .node1(
            Op::Slice {
                axis: 0,
                start: 32,
                len: 32,
            },
            &[assembled],
            &[32, m],
            v,
        )
        .unwrap();
    // σ0(w1) = ROTR^7 ⊕ ROTR^18 ⊕ SHR^3: two rotations, a shift's gather and its mask.
    let r7 = gather(&mut b, w1, &rotation(7), &[32, m]);
    let r18 = gather(&mut b, w1, &rotation(18), &[32, m]);
    let s3 = gather(&mut b, w1, &rotation(3), &[32, m]);
    let shifted = b
        .node1(
            Op::Mul {
                broadcast: Broadcast::Trailing,
            },
            &[s3, shr[0]],
            &[32, m],
            v,
        )
        .unwrap();
    let sigma = b
        .node1(
            Op::BitFn { f: BoolFn::Xor3 },
            &[r7, r18, shifted],
            &[32, m],
            v,
        )
        .unwrap();
    let ch = b
        .node1(Op::BitFn { f: BoolFn::Ch }, &[h0, w0, w1], &[32, m], v)
        .unwrap();
    let maj = b
        .node1(Op::BitFn { f: BoolFn::Maj }, &[h0, w0, sigma], &[32, m], v)
        .unwrap();
    // A sum: its rows, its value in five levels, its constant, its decomposition.
    let sum = b.node1(Op::Add, &[ch, maj], &[32, m], v).unwrap();
    let mut level = sum;
    let mut len = 32u32;
    for l in 0..5 {
        len /= 2;
        let even: Vec<i64> = (0..i64::from(len)).map(|j| 2 * j).collect();
        let odd: Vec<i64> = (0..i64::from(len)).map(|j| 2 * j + 1).collect();
        let e = gather(&mut b, level, &even, &[len, m]);
        let o = gather(&mut b, level, &odd, &[len, m]);
        let c = b.constant(&[1i64 << (1 << l)]).unwrap();
        let scaled = b.node1(Op::ConstMul { c }, &[o], &[len, m], v).unwrap();
        level = b.node1(Op::Add, &[e, scaled], &[len, m], v).unwrap();
    }
    let value = b.node1(Op::Reshape, &[level], &[m], v).unwrap();
    let k = b.constant(&[0x428a_2f98]).unwrap();
    let shifted_value = b.node1(Op::ConstAdd { c: k }, &[value], &[m], v).unwrap();
    let decomposed = b
        .node(
            Op::BitDecompose { low: 32, high: 2 },
            &[shifted_value],
            &[(&[m, 32], Kind::Advice), (&[m, 2], Kind::Advice)],
        )
        .unwrap();
    let word = b
        .node1(
            Op::Transpose {
                perm: perm(&[1, 0]),
            },
            &[decomposed.outputs()[0]],
            &[32, m],
            v,
        )
        .unwrap();
    // 5. The digest, after the circuit's nodes, and one word's binding.
    let digest = b.input(&[256, m], Kind::Public, BIT, 0, 1).unwrap();
    let expected = b
        .node1(
            Op::Slice {
                axis: 0,
                start: 0,
                len: 32,
            },
            &[digest],
            &[32, m],
            v,
        )
        .unwrap();
    b.node(Op::AssertEqual, &[word, expected], &[]).unwrap();
    b.finish()
}

#[test]
fn a_sha256_shaped_fragment_builds_and_reads_back() {
    let g = sha256_fragment();
    // No sequence, no layer, no route (RFC-0072 'Graphs without layers').
    assert_eq!(g.length(), 0);
    assert!(g.checkpoints().is_empty());
    assert_eq!(g.route(tv_core::claims::Site::AttnResidual), None);
    assert!(g.tables().is_empty() && g.weights().is_empty() && g.shift_classes().is_empty());
    // The inputs: their roles, kinds, types and shapes, the digest last.
    let inputs: Vec<(Role, Kind, u8, Vec<u32>, i64, i64)> = g
        .inputs()
        .iter()
        .map(|i| {
            let t = g.tensor(i.tensor).unwrap();
            assert_eq!(t.producer, None);
            (
                i.role,
                t.kind,
                i.ty.bits,
                t.shape.dims().to_vec(),
                i.lo,
                i.hi,
            )
        })
        .collect();
    assert_eq!(
        inputs,
        [
            (Role::Input(0), Kind::Public, 1, vec![512, 2], 0, 1),
            (Role::Input(1), Kind::Public, 1, vec![256, 2], 0, 1),
            (Role::Input(2), Kind::Public, 1, vec![2, 32], 0, 1),
            (Role::Input(3), Kind::Advice, 8, vec![2, 2], 0, 255),
            (Role::Input(4), Kind::Public, 1, vec![512], 0, 1),
            (Role::Input(5), Kind::Public, 1, vec![256, 2], 0, 1),
        ]
    );
    // The digest is declared after every node but the binding's two.
    let digest = g.inputs()[5].tensor;
    let binding = g.nodes().len() - 2;
    assert!(
        g.tensors()[..digest.0 as usize]
            .iter()
            .filter(|t| t.producer.is_some())
            .count()
            > 0
    );
    assert_eq!(
        g.tensor(TensorId(digest.0 + 1)).unwrap().producer,
        Some(NodeId(binding as u32))
    );
    // The operations, in order.
    let names: Vec<String> = g
        .nodes()
        .iter()
        .map(|n| match n.op() {
            Op::GatherPublic { .. } => "G".into(),
            Op::Reshape => "R".into(),
            Op::BitDecompose { low, high } => format!("D{low}.{high}"),
            Op::Transpose { perm } => format!("T{:?}", perm.as_slice()),
            Op::Mul { .. } => "M".into(),
            Op::BitFn { f } => format!("{f:?}"),
            Op::Slice { start, .. } => format!("S{start}"),
            Op::ConstMul { .. } => "C".into(),
            Op::Add => "A".into(),
            Op::ConstAdd { .. } => "K".into(),
            Op::AssertEqual => "E".into(),
            other => panic!("{other:?}"),
        })
        .collect();
    let levels = ["G", "G", "C", "A"].repeat(5);
    let mut expected: Vec<&str> = vec![
        "G",
        "R",
        "G",
        "R",
        "D8.0",
        "T[0, 2, 1]",
        "R",
        "G",
        "M",
        "Xor",
        "S0",
        "S0",
        "S32",
        "G",
        "G",
        "G",
        "M",
        "Xor3",
        "Ch",
        "Maj",
        "A",
    ];
    expected.extend(levels.iter().copied());
    expected.extend(["R", "K", "D32.2", "T[1, 0]", "S0", "E"]);
    assert_eq!(names, expected);
    // The three-input nodes keep their order; `AssertEqual` has no output.
    let xor3 = g
        .nodes()
        .iter()
        .find(|n| *n.op() == Op::BitFn { f: BoolFn::Xor3 })
        .unwrap();
    assert_eq!(xor3.inputs().len(), 3);
    let last = g.nodes().last().unwrap();
    assert_eq!((last.inputs().len(), last.outputs().len()), (2, 0));
    // The decomposition: committed bits and carries, their bit axes innermost.
    let d = g
        .nodes()
        .iter()
        .find(|n| *n.op() == Op::BitDecompose { low: 32, high: 2 })
        .unwrap();
    let shapes: Vec<(Vec<u32>, Kind)> = d
        .outputs()
        .iter()
        .map(|&t| {
            let t = g.tensor(t).unwrap();
            (t.shape.dims().to_vec(), t.kind)
        })
        .collect();
    assert_eq!(
        shapes,
        [(vec![2, 32], Kind::Advice), (vec![2, 2], Kind::Advice)]
    );
    // Constants: each node's own, in node order; the gathers' are vectors.
    assert_eq!(g.constant(0), Some(&[0][..]));
    assert_eq!(g.constant(1), Some(&[1][..]));
    let rotation7 = g.constant(3).unwrap();
    assert_eq!(rotation7, rotation(7).as_slice());
    let value_level0 = g.constant(8).unwrap();
    assert_eq!(
        value_level0,
        &[2][..],
        "the first level's odd rows times 2^1"
    );
    assert_eq!(
        g.constant(g.constant_count() as u32 - 1),
        Some(&[0x428a_2f98][..])
    );
    assert_eq!(g.constant(g.constant_count() as u32), None);
    let flat: usize = (0..g.constant_count() as u32)
        .map(|c| g.constant(c).unwrap().len())
        .sum();
    assert_eq!(
        flat,
        g.constants().len(),
        "the flat list is every constant's values"
    );
    // Ids are dense in creation order, each output after its node's inputs.
    for (i, node) in g.nodes().iter().enumerate() {
        for &out in node.outputs() {
            assert_eq!(g.tensor(out).unwrap().producer, Some(NodeId(i as u32)));
            assert!(node.inputs().iter().all(|&x| x < out));
        }
    }
}

/// An AES-shaped fragment over `B = 3` blocks: a committed key masked to its columns by a
/// `ConstMul` per channel and expanded over the blocks, `AddRoundKey` on public counters,
/// `SubBytes`, `ShiftRows`, `MixColumns`' two products, the ciphertext, and the range checks of
/// the key and the plaintext (RFC-0072 'AES's graphs', 'AES's lookups', 'Linear operations that
/// are not coordinate maps'). RFC-0072 OQ-1 leaves the circuit itself unspecified.
fn aes_fragment() -> Graph {
    let mut b = GraphBuilder::new(roomy()).unwrap();
    let blocks = 3u32;
    let counters = b.input(&[16, blocks], Kind::Public, BYTE, 0, 255).unwrap();
    let key = b.input(&[16], Kind::Advice, BYTE, 0, 255).unwrap();
    let plaintext = b.input(&[16, blocks], Kind::Advice, BYTE, 0, 255).unwrap();
    let v = Kind::Virtual;
    // The range checks of the committed key and plaintext.
    b.node1(
        Op::BitDecompose { low: 8, high: 0 },
        &[key],
        &[16, 8],
        Kind::Advice,
    )
    .unwrap();
    b.node1(
        Op::BitDecompose { low: 8, high: 0 },
        &[plaintext],
        &[16, blocks, 8],
        Kind::Advice,
    )
    .unwrap();
    // The round key masked to three of its four columns, then repeated over the blocks.
    let columns = b.node1(Op::Reshape, &[key], &[4, 4], v).unwrap();
    let mask = b.constant(&[1, 1, 1, 0]).unwrap();
    let masked = b
        .node1(Op::ConstMul { c: mask }, &[columns], &[4, 4], v)
        .unwrap();
    let flat = b.node1(Op::Reshape, &[masked], &[16], v).unwrap();
    let round_key = b
        .node1(
            Op::Expand {
                axis: 1,
                len: blocks,
            },
            &[flat],
            &[16, blocks],
            v,
        )
        .unwrap();
    // AddRoundKey, SubBytes, ShiftRows.
    let xor = b.table(TableKind::Bitwise8 { op: BitOp::Xor }).unwrap();
    let state = b
        .node1(
            Op::Bitwise { table: xor },
            &[counters, round_key],
            &[16, blocks],
            Kind::Lookup { table: xor },
        )
        .unwrap();
    let sbox = b.table(TableKind::AesSbox).unwrap();
    let subbed = b
        .node1(
            Op::TableMap { table: sbox },
            &[state],
            &[16, blocks],
            Kind::Lookup { table: sbox },
        )
        .unwrap();
    let shift_rows: Vec<i64> = (0..16).map(|i| (i + 4 * (i % 4)) % 16).collect();
    let index = b.constant(&shift_rows).unwrap();
    let shifted = b
        .node1(Op::GatherPublic { index }, &[subbed], &[16, blocks], v)
        .unwrap();
    // MixColumns' products by 2 and by 3, both of the shifted state, and their sum's first term.
    let times2 = b.table(TableKind::Gf256Mul { c: 2 }).unwrap();
    let times3 = b.table(TableKind::Gf256Mul { c: 3 }).unwrap();
    let doubled = b
        .node1(
            Op::TableMap { table: times2 },
            &[shifted],
            &[16, blocks],
            Kind::Lookup { table: times2 },
        )
        .unwrap();
    let tripled = b
        .node1(
            Op::TableMap { table: times3 },
            &[shifted],
            &[16, blocks],
            Kind::Lookup { table: times3 },
        )
        .unwrap();
    let mixed = b
        .node1(
            Op::Bitwise { table: xor },
            &[doubled, tripled],
            &[16, blocks],
            Kind::Lookup { table: xor },
        )
        .unwrap();
    // The ciphertext: the keystream XOR the plaintext.
    assert_eq!(b.table(TableKind::Bitwise8 { op: BitOp::Xor }), Ok(xor));
    b.node1(
        Op::Bitwise { table: xor },
        &[mixed, plaintext],
        &[16, blocks],
        Kind::Lookup { table: xor },
    )
    .unwrap();
    b.finish()
}

#[test]
fn an_aes_shaped_fragment_builds_and_reads_back() {
    let g = aes_fragment();
    assert_eq!(g.length(), 0);
    assert!(g.checkpoints().is_empty());
    // The tables, in order of first use, each kind once.
    let tables: Vec<(TableId, TableKind, u8)> = g
        .tables()
        .iter()
        .map(|t| (t.id, t.kind.clone(), t.log2_rows))
        .collect();
    assert_eq!(
        tables,
        [
            (TableId(0), TableKind::Bitwise8 { op: BitOp::Xor }, 16),
            (TableId(1), TableKind::AesSbox, 8),
            (TableId(2), TableKind::Gf256Mul { c: 2 }, 8),
            (TableId(3), TableKind::Gf256Mul { c: 3 }, 8),
        ]
    );
    // The lookups: each output is its own table's.
    let lookups: Vec<(Op, Kind)> = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), Op::TableMap { .. } | Op::Bitwise { .. }))
        .map(|n| (*n.op(), g.tensor(n.outputs()[0]).unwrap().kind))
        .collect();
    let lookup = |table| Kind::Lookup {
        table: TableId(table),
    };
    assert_eq!(
        lookups,
        [
            (Op::Bitwise { table: TableId(0) }, lookup(0)),
            (Op::TableMap { table: TableId(1) }, lookup(1)),
            (Op::TableMap { table: TableId(2) }, lookup(2)),
            (Op::TableMap { table: TableId(3) }, lookup(3)),
            (Op::Bitwise { table: TableId(0) }, lookup(0)),
            (Op::Bitwise { table: TableId(0) }, lookup(0)),
        ]
    );
    // The per-channel mask and the ShiftRows indices are vector constants.
    let Op::ConstMul { c } = *g
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), Op::ConstMul { .. }))
        .unwrap()
        .op()
    else {
        unreachable!()
    };
    assert_eq!(g.constant(c), Some(&[1, 1, 1, 0][..]));
    let Op::GatherPublic { index } = *g
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), Op::GatherPublic { .. }))
        .unwrap()
        .op()
    else {
        unreachable!()
    };
    assert_eq!(
        g.constant(index),
        Some(&[0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11][..])
    );
    // The round key's Expand inserts the block axis.
    let expand = g
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), Op::Expand { .. }))
        .unwrap();
    assert_eq!(
        g.tensor(expand.outputs()[0]).unwrap().shape.dims(),
        &[16, 3]
    );
}

#[test]
fn building_the_same_graph_twice_gives_identical_graphs() {
    for build in [sha256_fragment as fn() -> Graph, aes_fragment] {
        let (first, second) = (build(), build());
        assert_eq!(first, second);
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
    }
    // And an extended lowering, twice.
    let extended = || {
        let (_, graph) = lowered();
        let mut b = GraphBuilder::extend(graph.clone(), room_after(&graph)).unwrap();
        let x = b.input(&[4], Kind::Advice, BYTE, 0, 255).unwrap();
        b.node1(
            Op::BitDecompose { low: 8, high: 0 },
            &[x],
            &[4, 8],
            Kind::Advice,
        )
        .unwrap();
        b.finish()
    };
    assert_eq!(extended(), extended());
}

#[test]
fn an_extended_lowering_keeps_its_own_and_numbers_after_it() {
    let (claims, graph) = lowered();
    let mut b = GraphBuilder::extend(graph.clone(), room_after(&graph)).unwrap();
    let x = b.input(&[4], Kind::Advice, BYTE, 0, 255).unwrap();
    assert_eq!(x, TensorId(graph.tensors().len() as u32));
    let table = b.table(TableKind::AesSbox).unwrap();
    assert_eq!(
        table,
        TableId(graph.tables().len() as u32),
        "after the lowering's tables"
    );
    let c = b.constant(&[1, 2, 3, 4]).unwrap();
    assert_eq!(c as usize, graph.constant_count());
    let n = b
        .node(Op::ConstAdd { c }, &[x], &[(&[4], Kind::Virtual)])
        .unwrap();
    let g = b.finish();
    assert_eq!(g.node(NodeId(graph.nodes().len() as u32)), Some(&n));
    assert_eq!(g.length(), 4);
    assert_eq!(g.checkpoints(), graph.checkpoints());
    assert_eq!(
        g.route(tv_core::claims::Site::MlpResidual),
        graph.route(tv_core::claims::Site::MlpResidual)
    );
    assert!(graph.route(tv_core::claims::Site::MlpResidual).is_some());
    assert_eq!(&g.nodes()[..graph.nodes().len()], graph.nodes());
    assert_eq!(&g.tensors()[..graph.tensors().len()], graph.tensors());
    assert_eq!(g.inputs().last().unwrap().role, Role::Input(0));
    // The lowering's constants are one value each, and the flat list is them in order.
    assert_eq!(graph.constant_count(), graph.constants().len());
    for (i, &value) in graph.constants().iter().enumerate() {
        assert_eq!(graph.constant(i as u32), Some(&[value][..]));
    }
    assert!(matches!(
        graph.route(tv_core::claims::Site::AttnResidual),
        Some(Route::FusedKey | Route::TwoSteps)
    ));
    let _ = claims;
}

#[test]
fn perm_and_bool_fn_read_as_specified() {
    assert_eq!(perm(&[2, 0, 1]).as_slice(), &[2, 0, 1]);
    assert_eq!(perm(&[1, 0]).len(), 2);
    assert_eq!(Perm::new(&[0, 1, 2, 3, 4]), None);
    assert_eq!(Perm::try_from(vec![1, 0, 2]), Ok(perm(&[1, 0, 2])));
    assert_eq!(Perm::try_from(vec![0; 5]), Err(vec![0; 5]));
    let arities: Vec<usize> = [
        BoolFn::Xor,
        BoolFn::Xor3,
        BoolFn::And,
        BoolFn::Ch,
        BoolFn::Maj,
    ]
    .map(BoolFn::arity)
    .to_vec();
    assert_eq!(arities, [2, 3, 2, 3, 3]);
}

// ----- Refusals downstream -----------------------------------------------------------------------

/// The operations the Llama lowering never creates, each appended alone to a lowered graph, and
/// a lowering's operation (an `Add`) in their place.
fn appended(op: &str) -> Graph {
    let (_, graph) = lowered();
    let mut b = GraphBuilder::extend(graph.clone(), room_after(&graph)).unwrap();
    let a = b.input(&[2, 4], Kind::Public, BIT, 0, 1).unwrap();
    let a2 = b.input(&[2, 4], Kind::Public, BIT, 0, 1).unwrap();
    let bytes = b.input(&[4], Kind::Advice, BYTE, 0, 255).unwrap();
    let bytes2 = b.input(&[4], Kind::Advice, BYTE, 0, 255).unwrap();
    let v = Kind::Virtual;
    match op {
        "Add" => {
            b.node1(Op::Add, &[a, a2], &[2, 4], v).unwrap();
        }
        "Sub" => {
            b.node1(Op::Sub, &[a, a2], &[2, 4], v).unwrap();
        }
        "ConstAdd" => {
            let c = b.constant(&[1]).unwrap();
            b.node1(Op::ConstAdd { c }, &[a], &[2, 4], v).unwrap();
        }
        "Sum" => {
            b.node1(Op::Sum { axis: 0 }, &[a], &[4], v).unwrap();
        }
        "Slice" => {
            b.node1(
                Op::Slice {
                    axis: 0,
                    start: 1,
                    len: 1,
                },
                &[a],
                &[1, 4],
                v,
            )
            .unwrap();
        }
        "GatherPublic" => {
            let index = b.constant(&[1, 0]).unwrap();
            b.node1(Op::GatherPublic { index }, &[a], &[2, 4], v)
                .unwrap();
        }
        "Bitwise" => {
            let table = b.table(TableKind::Bitwise8 { op: BitOp::Xor }).unwrap();
            b.node1(
                Op::Bitwise { table },
                &[bytes, bytes2],
                &[4],
                Kind::Lookup { table },
            )
            .unwrap();
        }
        "BitDecompose" => {
            b.node1(
                Op::BitDecompose { low: 8, high: 0 },
                &[bytes],
                &[4, 8],
                Kind::Advice,
            )
            .unwrap();
        }
        "BitFn" => {
            b.node1(Op::BitFn { f: BoolFn::Xor }, &[a, a2], &[2, 4], v)
                .unwrap();
        }
        "AssertEqual" => {
            b.node(Op::AssertEqual, &[a, a2], &[]).unwrap();
        }
        "ConstMul per channel" => {
            let c = b.constant(&[1, 2, 3, 4]).unwrap();
            b.node1(Op::ConstMul { c }, &[a], &[2, 4], v).unwrap();
        }
        "TableMap into AesSbox" => {
            let table = b.table(TableKind::AesSbox).unwrap();
            b.node1(
                Op::TableMap { table },
                &[bytes],
                &[4],
                Kind::Lookup { table },
            )
            .unwrap();
        }
        "TableMap into Gf256Mul" => {
            let table = b.table(TableKind::Gf256Mul { c: 2 }).unwrap();
            b.node1(
                Op::TableMap { table },
                &[bytes],
                &[4],
                Kind::Lookup { table },
            )
            .unwrap();
        }
        other => panic!("{other}"),
    }
    b.finish()
}

const ANCHOR_OPERATIONS: [&str; 12] = [
    "Sub",
    "ConstAdd",
    "Sum",
    "Slice",
    "GatherPublic",
    "Bitwise",
    "BitDecompose",
    "BitFn",
    "AssertEqual",
    "ConstMul per channel",
    "TableMap into AesSbox",
    "TableMap into Gf256Mul",
];

/// Each check after the lowering, run on `graph`: its result, or its error, as text.
fn checks(claims: &ClaimList, graph: &Graph) -> Vec<(&'static str, Result<(), String>)> {
    let policy = claims.gadget_policy;
    let show = |e: &dyn std::fmt::Debug| format!("{e:?}");
    let base = lowered().1;
    let selected = selection::derive(&base, policy).unwrap();
    let config = Config::new(&claims.arch, policy, 10, 1);
    vec![
        (
            "ranges",
            ranges::analyze(graph).map(drop).map_err(|e| show(&e)),
        ),
        (
            "selection",
            selection::derive(graph, policy)
                .map(drop)
                .map_err(|e| show(&e)),
        ),
        (
            "schedule",
            schedule::derive(graph).map(drop).map_err(|e| show(&e)),
        ),
        (
            "composed schedule",
            schedule::derive_composed(graph, &[], &[])
                .map(drop)
                .map_err(|e| show(&e)),
        ),
        (
            "token axes",
            TokenAxes::derive(graph).map(drop).map_err(|e| show(&e)),
        ),
        (
            "advice",
            advice::Layout::derive(graph, &selected, config, 4)
                .map(drop)
                .map_err(|e| show(&e)),
        ),
        (
            "gadgets",
            gadgets::Plan::derive(graph, policy)
                .map(drop)
                .map_err(|e| show(&e)),
        ),
        (
            "leaf",
            ModelSchedule::derive(graph, policy)
                .map(drop)
                .map_err(|e| show(&e)),
        ),
        (
            "leaf sites",
            split::model_sites(graph, policy)
                .map(drop)
                .map_err(|e| show(&e)),
        ),
    ]
}

#[test]
fn every_check_after_the_lowering_refuses_each_anchor_operation_with_its_typed_error() {
    let (claims, base) = lowered();
    // The control: the lowering, and the lowering with an `Add` appended, pass every check.
    for graph in [base.clone(), appended("Add")] {
        for (check, result) in checks(&claims, &graph) {
            assert_eq!(result, Ok(()), "{check} on the control");
        }
    }
    let node = NodeId(base.nodes().len() as u32);
    for op in ANCHOR_OPERATIONS {
        let graph = appended(op);
        assert_eq!(graph.nodes().len(), base.nodes().len() + 1, "{op}");
        assert_eq!(
            ranges::analyze(&graph).err(),
            Some(RangeError::Unsupported { node }),
            "{op}"
        );
        assert_eq!(
            selection::derive(&graph, claims.gadget_policy).err(),
            Some(ClaimsError::Internal),
            "{op}"
        );
        assert_eq!(
            schedule::derive(&graph).err(),
            Some(ScheduleError::Unsupported { node }),
            "{op}"
        );
        assert_eq!(
            schedule::derive_composed(&graph, &[], &[]).err(),
            Some(ScheduleError::Unsupported { node }),
            "{op}"
        );
        assert_eq!(
            TokenAxes::derive(&graph).err(),
            Some(AxisError::Refused { node }),
            "{op}"
        );
        for (check, result) in checks(&claims, &graph) {
            let expected = match check {
                "ranges" => format!("{:?}", RangeError::Unsupported { node }),
                "schedule" | "composed schedule" => {
                    format!("{:?}", ScheduleError::Unsupported { node })
                }
                "token axes" => format!("{:?}", AxisError::Refused { node }),
                _ => format!("{:?}", ClaimsError::Internal),
            };
            assert_eq!(result, Err(expected), "{check} on {op}");
        }
    }
}

#[test]
fn the_schedule_refuses_an_anchor_graph_at_its_first_anchor_node_before_its_layers() {
    // Not `Template` for want of checkpoints: the first operation it cannot prove, in node order.
    assert_eq!(
        schedule::derive(&sha256_fragment()).err(),
        Some(ScheduleError::Unsupported { node: NodeId(0) })
    );
    let aes = aes_fragment();
    assert_eq!(
        schedule::derive(&aes).err(),
        Some(ScheduleError::Unsupported { node: NodeId(0) })
    );
    assert_eq!(
        ranges::analyze(&aes).err(),
        Some(RangeError::Unsupported { node: NodeId(0) })
    );
    assert_eq!(
        TokenAxes::derive(&aes).err(),
        Some(AxisError::Refused { node: NodeId(0) })
    );
    // A graph of the lowering's operations alone, without checkpoints, is the template's refusal.
    let mut b = GraphBuilder::new(roomy()).unwrap();
    let x = b.input(&[2, 4], Kind::Public, BIT, 0, 1).unwrap();
    b.node1(Op::Add, &[x, x], &[2, 4], Kind::Virtual).unwrap();
    assert_eq!(
        schedule::derive(&b.finish()).err(),
        Some(ScheduleError::Template { node: NodeId(0) })
    );
}
