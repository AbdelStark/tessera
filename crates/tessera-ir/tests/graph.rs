// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The IR's graph rules (RFC-0006 'Operation set', 03 §3): a block that uses every operation is
//! accepted, builds identically twice (INV-DM-5) and validates as built; and every rule, broken
//! alone in an otherwise valid node or graph, is refused with its exact error, leaving the
//! builder as it was.

use tessera_ir::{
    BoolFn, Broadcast, CommitmentClass, Graph, GraphBuilder, IntRange, IntType, IrError, Node,
    NodeId, NodeKind, Op, OpClass, PrfRef, TableId, TensorDecl, TensorId, WeightRef, WeightType,
};

const T: u32 = 4; // tokens
const D: u32 = 8; // model width
const F: u32 = 12; // MLP width
const VOCAB: u32 = 16;

fn decl(shape: &[u32], kind: NodeKind) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: IntType::I32,
        range: IntRange {
            lo: -1000,
            hi: 1000,
        },
        kind,
    }
}

fn virt(shape: &[u32]) -> TensorDecl {
    decl(shape, NodeKind::Virtual)
}

fn advice(shape: &[u32]) -> TensorDecl {
    decl(
        shape,
        NodeKind::Committed {
            commitment: CommitmentClass::Advice,
        },
    )
}

fn lookup(shape: &[u32], table: u32) -> TensorDecl {
    decl(
        shape,
        NodeKind::LookupOutput {
            table: TableId(table),
        },
    )
}

fn public(shape: &[u32]) -> TensorDecl {
    decl(shape, NodeKind::Public)
}

fn weight(ty: WeightType) -> WeightRef {
    WeightRef { ty, layer: 0 }
}

/// One node, expected to pass, returning its single output.
fn one(b: &mut GraphBuilder, op: Op, inputs: &[TensorId], output: TensorDecl) -> TensorId {
    b.node(op, inputs, vec![output]).expect("a valid node")[0]
}

/// A block of a Llama-like layer, then the head and the certificates: every operation of RFC-0006
/// at least once, in the shapes the lowering uses.
fn block() -> Graph {
    let mut b = GraphBuilder::new();
    let x = b.input(advice(&[T, D])).expect("x");
    let positions = b.input(public(&[T])).expect("positions");
    let tokens = b.input(advice(&[T])).expect("tokens");
    let classes = b
        .input(decl(
            &[D],
            NodeKind::Committed {
                commitment: CommitmentClass::Weight,
            },
        ))
        .expect("channel classes");
    let embed = b
        .weight(weight(WeightType::Embed), D, VOCAB)
        .expect("embed");
    let [wq, wk, wv, wo] = [WeightType::Q, WeightType::K, WeightType::V, WeightType::O]
        .map(|ty| b.weight(weight(ty), D, D).expect("attention weight"));
    let wg = b.weight(weight(WeightType::Gate), F, D).expect("gate");
    let wu = b.weight(weight(WeightType::Up), F, D).expect("up");
    let wd = b.weight(weight(WeightType::Down), D, F).expect("down");
    let head = b
        .weight(weight(WeightType::LmHead), VOCAB, D)
        .expect("head");
    let one_class = b.shift_classes(vec![4]).expect("one class");
    let two_classes = b.shift_classes(vec![3, 5]).expect("two classes");
    let scalar = b.constant(vec![3]).expect("a scalar");
    let per_channel = b.constant(vec![1; D as usize]).expect("one per channel");
    let rows = b.constant(vec![3, 0]).expect("row indices");

    let h = one(
        &mut b,
        Op::OneHot { vocab: VOCAB },
        &[tokens],
        advice(&[T, VOCAB]),
    );
    let e = one(
        &mut b,
        Op::MatMulWeight {
            weight: weight(WeightType::Embed),
        },
        &[h, embed],
        virt(&[T, D]),
    );
    let r = one(&mut b, Op::Add, &[x, e], virt(&[T, D]));
    let ss = one(&mut b, Op::SumSquares { axis: 1 }, &[r], virt(&[T]));
    let rho = one(
        &mut b,
        Op::RsqrtRow {
            kappa: 16,
            eps_int: 5,
            dim: D,
        },
        &[ss],
        advice(&[T]),
    );
    let n = one(
        &mut b,
        Op::Mul {
            broadcast: Broadcast::Trailing,
        },
        &[r, rho],
        virt(&[T, D]),
    );
    let requant = |shift| Op::Requant {
        shift,
        lo: -128,
        hi: 127,
    };
    let n8 = one(&mut b, requant(one_class), &[n], lookup(&[T, D], 0));
    let [q, k, v] = [
        (WeightType::Q, wq),
        (WeightType::K, wk),
        (WeightType::V, wv),
    ]
    .map(|(ty, w)| {
        one(
            &mut b,
            Op::MatMulWeight { weight: weight(ty) },
            &[n8, w],
            virt(&[T, D]),
        )
    });
    let rope = Op::Rotate { rope: TableId(9) };
    let qr = one(&mut b, rope.clone(), &[q, positions], virt(&[T, D]));
    let kr = one(&mut b, rope, &[k, positions], virt(&[T, D]));
    let [q8, k8, v8] =
        [qr, kr, v].map(|t| one(&mut b, requant(one_class), &[t], lookup(&[T, D], 0)));
    let scores = one(
        &mut b,
        Op::MatMulAct {
            transpose_rhs: true,
        },
        &[q8, k8],
        virt(&[T, T]),
    );
    let softmax = Op::SoftmaxRow {
        exp_table: TableId(1),
        sigma: 16,
    };
    let [exps, recip] = b
        .node(softmax, &[scores], vec![lookup(&[T, T], 1), advice(&[T])])
        .expect("softmax")[..]
    else {
        unreachable!("two outputs")
    };
    let pv = one(
        &mut b,
        Op::MatMulAct {
            transpose_rhs: false,
        },
        &[exps, v8],
        virt(&[T, D]),
    );
    let attn = one(
        &mut b,
        Op::Mul {
            broadcast: Broadcast::Trailing,
        },
        &[pv, recip],
        virt(&[T, D]),
    );
    let attn8 = one(
        &mut b,
        requant(two_classes),
        &[attn, classes],
        advice(&[T, D]),
    );
    let o = one(
        &mut b,
        Op::MatMulWeight {
            weight: weight(WeightType::O),
        },
        &[attn8, wo],
        virt(&[T, D]),
    );
    let res = one(&mut b, Op::Add, &[r, o], virt(&[T, D]));
    let res8 = one(&mut b, requant(one_class), &[res], lookup(&[T, D], 0));
    let g = one(
        &mut b,
        Op::MatMulWeight {
            weight: weight(WeightType::Gate),
        },
        &[res8, wg],
        virt(&[T, F]),
    );
    let u = one(
        &mut b,
        Op::MatMulWeight {
            weight: weight(WeightType::Up),
        },
        &[res8, wu],
        virt(&[T, F]),
    );
    let g8 = one(
        &mut b,
        Op::TableMap { table: TableId(3) },
        &[g],
        lookup(&[T, F], 3),
    );
    let u8_ = one(&mut b, requant(one_class), &[u], lookup(&[T, F], 0));
    let p = one(
        &mut b,
        Op::Mul {
            broadcast: Broadcast::Trailing,
        },
        &[g8, u8_],
        virt(&[T, F]),
    );
    let p8 = one(&mut b, requant(one_class), &[p], lookup(&[T, F], 0));
    let down = one(
        &mut b,
        Op::MatMulWeight {
            weight: weight(WeightType::Down),
        },
        &[p8, wd],
        virt(&[T, D]),
    );
    let scaled = one(&mut b, Op::ConstMul { c: scalar }, &[down], virt(&[T, D]));
    let shifted = one(
        &mut b,
        Op::ConstAdd { c: per_channel },
        &[scaled],
        virt(&[T, D]),
    );
    let out = one(&mut b, Op::Sub, &[res, shifted], virt(&[T, D]));
    one(&mut b, Op::Sum { axis: 0 }, &[out], virt(&[D]));
    one(&mut b, Op::Reshape, &[out], virt(&[D, T]));
    // A slice that ends exactly at the end of its axis.
    one(
        &mut b,
        Op::Slice {
            axis: 1,
            start: 5,
            len: 3,
        },
        &[out],
        virt(&[T, 3]),
    );
    one(
        &mut b,
        Op::GatherPublic { index: rows },
        &[out],
        virt(&[2, D]),
    );
    let swapped = one(
        &mut b,
        Op::Transpose { perm: vec![1, 0] },
        &[out],
        virt(&[D, T]),
    );
    one(
        &mut b,
        Op::Expand { axis: 1, len: 3 },
        &[swapped],
        virt(&[D, 3, T]),
    );
    // A linear output that is committed advice, proven equal to its relation (RFC-0037).
    one(&mut b, Op::ConstMul { c: scalar }, &[out], advice(&[T, D]));
    let out8 = one(&mut b, requant(one_class), &[out], lookup(&[T, D], 0));
    let logits = one(
        &mut b,
        Op::MatMulWeight {
            weight: weight(WeightType::LmHead),
        },
        &[out8, head],
        virt(&[T, VOCAB]),
    );
    assert_eq!(b.node(Op::Argmax, &[logits, h], vec![]), Ok(vec![]));
    assert_eq!(
        b.node(Op::InverseCdf { prf: PrfRef(0) }, &[logits, h], vec![]),
        Ok(vec![])
    );
    let byte = b.input(advice(&[VOCAB])).expect("bytes");
    one(
        &mut b,
        Op::Bitwise { table: TableId(4) },
        &[byte, byte],
        lookup(&[VOCAB], 4),
    );
    // RFC-0066: a byte's bits and two of carry, a Boolean function of them, and an equality.
    let bits = b
        .node(
            Op::BitDecompose { low: 8, high: 2 },
            &[byte],
            vec![advice(&[VOCAB, 8]), advice(&[VOCAB, 2])],
        )
        .expect("bits");
    let maj = one(
        &mut b,
        Op::BitFn { f: BoolFn::Maj },
        &[bits[0], bits[0], bits[0]],
        virt(&[VOCAB, 8]),
    );
    assert_eq!(b.node(Op::AssertEqual, &[maj, bits[0]], vec![]), Ok(vec![]));
    b.finish()
}

#[test]
fn a_block_with_every_operation_builds_identically_and_validates() {
    let graph = block();
    assert_eq!(
        graph,
        block(),
        "two builds from the same calls are equal (INV-DM-5)"
    );
    assert_eq!(graph.validate(), Ok(()));
    let names: std::collections::BTreeSet<&str> = graph.nodes.iter().map(|n| n.op.name()).collect();
    assert_eq!(names.len(), 26, "every operation of RFC-0006: {names:?}");
    for (position, tensor) in (0u32..).zip(&graph.tensors) {
        assert_eq!(tensor.id, TensorId(position));
    }
    // The constraint-system view: inputs have no producer, every other tensor has its node, and
    // classes follow the operation set: a linear or degree-2 node's output is virtual, or
    // committed advice when it is not a matmul (RFC-0037).
    let advice_kind = NodeKind::Committed {
        commitment: CommitmentClass::Advice,
    };
    let mut committed_linear = 0;
    for node in &graph.nodes {
        for &output in &node.outputs {
            assert_eq!(graph.producer(output), Some(node.id));
        }
        if node.op.class() != OpClass::Gadget {
            for &output in &node.outputs {
                let kind = graph.tensors[output.0 as usize].kind;
                if kind == advice_kind {
                    committed_linear += 1;
                } else {
                    assert_eq!(kind, NodeKind::Virtual);
                }
            }
        }
    }
    assert_eq!(
        committed_linear, 1,
        "the block's one committed linear output"
    );
    assert_eq!(graph.producer(TensorId(0)), None);
    assert_eq!(
        Op::MatMulAct {
            transpose_rhs: true
        }
        .class(),
        OpClass::Degree2
    );
    assert_eq!(Op::Rotate { rope: TableId(0) }.class(), OpClass::Linear);
    assert_eq!(
        Op::SoftmaxRow {
            exp_table: TableId(0),
            sigma: 16
        }
        .class(),
        OpClass::Gadget
    );
}

/// A builder with the declarations every scenario below needs. Tensors 0 to 7 are `a [T, D]`,
/// `b [T, D]` (advice), the positions `[T]` (public), the channel classes `[D]` (weights), the
/// weight Q `[D, D]`, a public `[D]`, an advice `[T, 3]` with an odd last axis, and square scores
/// `[T, T]` (advice); shift classes 0 and 1 have one and two classes; constants 0, 1 and 2 are a
/// scalar, the rows 3 and 0, and the rows 0 and `T`, one past the end of `[T]`.
fn base() -> (GraphBuilder, [TensorId; 8]) {
    let mut b = GraphBuilder::new();
    let a = b.input(advice(&[T, D])).expect("a");
    let c = b.input(advice(&[T, D])).expect("b");
    let positions = b.input(public(&[T])).expect("positions");
    let classes = b
        .input(decl(
            &[D],
            NodeKind::Committed {
                commitment: CommitmentClass::Weight,
            },
        ))
        .expect("classes");
    let wq = b.weight(weight(WeightType::Q), D, D).expect("Q");
    let wide = b.input(public(&[D])).expect("a public [D]");
    let odd = b.input(advice(&[T, 3])).expect("an odd last axis");
    let square = b.input(advice(&[T, T])).expect("square scores");
    b.shift_classes(vec![4]).expect("one class");
    b.shift_classes(vec![3, 5]).expect("two classes");
    b.constant(vec![3]).expect("scalar");
    b.constant(vec![3, 0]).expect("rows");
    b.constant(vec![0, i64::from(T)])
        .expect("a row past the end");
    (b, [a, c, positions, classes, wq, wide, odd, square])
}

fn op_error(op: &'static str, rule: &'static str) -> IrError {
    IrError::Op {
        node: NodeId(0),
        op,
        rule,
    }
}

#[test]
fn each_operation_rule_broken_alone_is_refused() {
    let (base_builder, [a, c, positions, classes, wq, wide, odd, square]) = base();
    let q = weight(WeightType::Q);
    let shift = |n| tessera_ir::ShiftClassRef(n);
    let konst = |n| tessera_ir::ConstRef(n);
    let requant = |s| Op::Requant {
        shift: shift(s),
        lo: -128,
        hi: 127,
    };
    type Case = (Op, Vec<TensorId>, Vec<TensorDecl>, IrError);
    let cases: Vec<Case> = vec![
        // Arity, and virtual outputs of linear and degree-2 operations.
        (
            Op::Add,
            vec![a],
            vec![virt(&[T, D])],
            op_error("Add", "takes two tensors and produces one"),
        ),
        (
            Op::Add,
            vec![a, c],
            vec![decl(
                &[T, D],
                NodeKind::Committed {
                    commitment: CommitmentClass::Weight,
                },
            )],
            op_error(
                "Add",
                "a linear or degree-2 operation produces a virtual tensor, or committed advice if \
                 it is not a matmul",
            ),
        ),
        (
            Op::Mul {
                broadcast: Broadcast::Trailing,
            },
            vec![a, c],
            vec![lookup(&[T, D], 0)],
            op_error(
                "Mul",
                "a linear or degree-2 operation produces a virtual tensor, or committed advice if \
                 it is not a matmul",
            ),
        ),
        (
            Op::MatMulWeight {
                weight: weight(WeightType::Q),
            },
            vec![a, wq],
            vec![advice(&[T, D])],
            op_error(
                "MatMulWeight",
                "a linear or degree-2 operation produces a virtual tensor, or committed advice if \
                 it is not a matmul",
            ),
        ),
        // MatMulWeight.
        (
            Op::MatMulWeight {
                weight: weight(WeightType::K),
            },
            vec![a, wq],
            vec![virt(&[T, D])],
            IrError::UnknownReference {
                node: NodeId(0),
                what: "weight",
            },
        ),
        (
            Op::MatMulWeight { weight: q },
            vec![a, c],
            vec![virt(&[T, D])],
            op_error(
                "MatMulWeight",
                "the second input is the weight tensor its WeightRef declares",
            ),
        ),
        (
            Op::MatMulWeight { weight: q },
            vec![positions, wq],
            vec![virt(&[D])],
            op_error(
                "MatMulWeight",
                "the weight's columns match the input's last axis",
            ),
        ),
        (
            Op::MatMulWeight { weight: q },
            vec![a, wq],
            vec![virt(&[T, D + 1])],
            op_error(
                "MatMulWeight",
                "the output is the input with its last axis the weight's rows",
            ),
        ),
        // Elementwise operands.
        (
            Op::Sub,
            vec![a, positions],
            vec![virt(&[T, D])],
            op_error("Sub", "the operands and the output have one shape"),
        ),
        (
            Op::Add,
            vec![a, c],
            vec![virt(&[D, T])],
            op_error("Add", "the operands and the output have one shape"),
        ),
        // Constants.
        (
            Op::ConstMul { c: konst(9) },
            vec![a],
            vec![virt(&[T, D])],
            IrError::UnknownReference {
                node: NodeId(0),
                what: "constant",
            },
        ),
        (
            Op::ConstAdd { c: konst(1) },
            vec![a],
            vec![virt(&[T, D])],
            op_error(
                "ConstAdd",
                "the constant is a scalar or one value per channel",
            ),
        ),
        (
            Op::ConstMul { c: konst(0) },
            vec![a],
            vec![virt(&[T])],
            op_error("ConstMul", "the output has the input's shape"),
        ),
        // Rotate.
        (
            Op::Rotate { rope: TableId(9) },
            vec![a, c],
            vec![virt(&[T, D])],
            op_error("Rotate", "the positions are public"),
        ),
        (
            Op::Rotate { rope: TableId(9) },
            vec![a, wide],
            vec![virt(&[T, D])],
            op_error("Rotate", "one position per row of the outermost axis"),
        ),
        (
            Op::Rotate { rope: TableId(9) },
            vec![odd, positions],
            vec![virt(&[T, 3])],
            op_error("Rotate", "the last axis holds pairs"),
        ),
        (
            Op::Rotate { rope: TableId(9) },
            vec![a, positions],
            vec![virt(&[T, D + 2])],
            op_error("Rotate", "the output has the input's shape"),
        ),
        // Sum, SumSquares, Reshape, Slice, GatherPublic.
        (
            Op::Sum { axis: 2 },
            vec![a],
            vec![virt(&[T])],
            op_error("Sum", "the axis is one of the input's"),
        ),
        (
            Op::SumSquares { axis: 1 },
            vec![a],
            vec![virt(&[D])],
            op_error("SumSquares", "the output is the input without the axis"),
        ),
        (
            Op::Reshape,
            vec![a],
            vec![virt(&[T, D + 1])],
            op_error("Reshape", "the output has as many elements as the input"),
        ),
        (
            Op::Slice {
                axis: 2,
                start: 0,
                len: 1,
            },
            vec![a],
            vec![virt(&[T, D])],
            op_error("Slice", "the axis is one of the input's"),
        ),
        (
            Op::Slice {
                axis: 1,
                start: 6,
                len: 3,
            },
            vec![a],
            vec![virt(&[T, 3])],
            op_error("Slice", "the slice is non-empty and inside the axis"),
        ),
        (
            Op::Slice {
                axis: 1,
                start: 0,
                len: 0,
            },
            vec![a],
            vec![virt(&[T, 1])],
            op_error("Slice", "the slice is non-empty and inside the axis"),
        ),
        (
            Op::Slice {
                axis: 0,
                start: 1,
                len: 2,
            },
            vec![a],
            vec![virt(&[T, 2])],
            op_error(
                "Slice",
                "the output is the input with the axis cut to the slice",
            ),
        ),
        (
            Op::GatherPublic { index: konst(9) },
            vec![positions],
            vec![virt(&[2])],
            IrError::UnknownReference {
                node: NodeId(0),
                what: "constant",
            },
        ),
        (
            Op::GatherPublic { index: konst(2) },
            vec![positions],
            vec![virt(&[2])],
            op_error("GatherPublic", "every index is a row of the outermost axis"),
        ),
        (
            Op::GatherPublic { index: konst(1) },
            vec![positions],
            vec![virt(&[3])],
            op_error("GatherPublic", "the output has one row per index"),
        ),
        // Transpose and Expand.
        (
            Op::Transpose { perm: vec![0] },
            vec![a],
            vec![virt(&[T, D])],
            op_error(
                "Transpose",
                "the permutation names each of the input's axes once",
            ),
        ),
        (
            Op::Transpose { perm: vec![1, 1] },
            vec![a],
            vec![virt(&[D, D])],
            op_error(
                "Transpose",
                "the permutation names each of the input's axes once",
            ),
        ),
        (
            Op::Transpose { perm: vec![0, 2] },
            vec![a],
            vec![virt(&[T, D])],
            op_error(
                "Transpose",
                "the permutation names each of the input's axes once",
            ),
        ),
        (
            Op::Transpose { perm: vec![1, 0] },
            vec![a],
            vec![virt(&[T, D])],
            op_error(
                "Transpose",
                "the output is the input with its axes permuted",
            ),
        ),
        (
            Op::Expand { axis: 3, len: 2 },
            vec![a],
            vec![virt(&[T, D, 2])],
            op_error("Expand", "the new axis is at most the input's rank"),
        ),
        (
            Op::Expand { axis: 2, len: 4 },
            vec![a],
            vec![virt(&[T, 4, D])],
            op_error(
                "Expand",
                "the output is the input with the new axis inserted",
            ),
        ),
        // MatMulAct.
        (
            Op::MatMulAct {
                transpose_rhs: false,
            },
            vec![a, positions],
            vec![virt(&[T, T])],
            op_error(
                "MatMulAct",
                "the operands are matrices with equal leading axes",
            ),
        ),
        (
            Op::MatMulAct {
                transpose_rhs: false,
            },
            vec![a, c],
            vec![virt(&[T, D])],
            op_error("MatMulAct", "the inner dimensions agree"),
        ),
        (
            Op::MatMulAct {
                transpose_rhs: true,
            },
            vec![a, c],
            vec![virt(&[T, D])],
            op_error("MatMulAct", "the output is the operands' product shape"),
        ),
        // Mul's broadcast.
        (
            Op::Mul {
                broadcast: Broadcast::Trailing,
            },
            vec![a, classes],
            vec![virt(&[T, D])],
            op_error(
                "Mul",
                "the second operand has the first's shape or a prefix of it",
            ),
        ),
        (
            Op::Mul {
                broadcast: Broadcast::Trailing,
            },
            vec![a, positions],
            vec![virt(&[T])],
            op_error("Mul", "the output has the first operand's shape"),
        ),
        // A leading broadcast takes one value per channel of the last axis, not per row.
        (
            Op::Mul {
                broadcast: Broadcast::Leading,
            },
            vec![a, positions],
            vec![virt(&[T, D])],
            op_error("Mul", "the second operand has the first's last axis"),
        ),
        // OneHot.
        (
            Op::OneHot { vocab: 0 },
            vec![positions],
            vec![advice(&[T, 1])],
            op_error("OneHot", "the vocabulary is not empty"),
        ),
        (
            Op::OneHot { vocab: VOCAB },
            vec![positions],
            vec![advice(&[VOCAB, T])],
            op_error("OneHot", "the output adds an axis of the vocabulary's size"),
        ),
        (
            Op::OneHot { vocab: VOCAB },
            vec![positions],
            vec![lookup(&[T, VOCAB], 0)],
            op_error("OneHot", "the one-hot vectors are committed advice"),
        ),
        // Requant.
        (
            requant(7),
            vec![a],
            vec![lookup(&[T, D], 0)],
            IrError::UnknownReference {
                node: NodeId(0),
                what: "set of shift classes",
            },
        ),
        (
            requant(1),
            vec![a],
            vec![lookup(&[T, D], 0)],
            op_error(
                "Requant",
                "takes the tensor, with the channel classes when there are several, and produces one",
            ),
        ),
        (
            requant(0),
            vec![a, classes],
            vec![lookup(&[T, D], 0)],
            op_error(
                "Requant",
                "takes the tensor, with the channel classes when there are several, and produces one",
            ),
        ),
        (
            Op::Requant {
                shift: shift(0),
                lo: 1,
                hi: 0,
            },
            vec![a],
            vec![lookup(&[T, D], 0)],
            op_error("Requant", "lo ≤ hi"),
        ),
        (
            requant(1),
            vec![a, positions],
            vec![lookup(&[T, D], 0)],
            op_error(
                "Requant",
                "the channel classes are committed weights, one per channel",
            ),
        ),
        (
            requant(0),
            vec![a],
            vec![lookup(&[D, T], 0)],
            op_error("Requant", "the output has the input's shape"),
        ),
        (
            requant(0),
            vec![a],
            vec![virt(&[T, D])],
            op_error(
                "Requant",
                "the output is committed, as a lookup output or as advice",
            ),
        ),
        // TableMap and Bitwise.
        (
            Op::TableMap { table: TableId(3) },
            vec![a],
            vec![lookup(&[T, D], 4)],
            op_error("TableMap", "the output is the table's lookup output"),
        ),
        (
            Op::Bitwise { table: TableId(4) },
            vec![a, positions],
            vec![lookup(&[T, D], 4)],
            op_error("Bitwise", "the operands and the output have one shape"),
        ),
        (
            Op::TableMap { table: TableId(3) },
            vec![a, c],
            vec![lookup(&[T, D], 3)],
            op_error(
                "TableMap",
                "the channel classes are committed weights, one per channel",
            ),
        ),
        // SoftmaxRow and RsqrtRow.
        (
            Op::SoftmaxRow {
                exp_table: TableId(1),
                sigma: 16,
            },
            vec![square],
            vec![lookup(&[T, T], 1)],
            op_error(
                "SoftmaxRow",
                "takes the scores and produces the exponentials and reciprocals",
            ),
        ),
        (
            Op::SoftmaxRow {
                exp_table: TableId(1),
                sigma: 16,
            },
            vec![a],
            vec![lookup(&[T, D], 1), advice(&[T])],
            op_error(
                "SoftmaxRow",
                "the scores' last two axes are the query and key positions of one sequence",
            ),
        ),
        (
            Op::SoftmaxRow {
                exp_table: TableId(1),
                sigma: 16,
            },
            vec![square],
            vec![lookup(&[T, T], 1), advice(&[D])],
            op_error(
                "SoftmaxRow",
                "one exponential per score and one reciprocal per row",
            ),
        ),
        (
            Op::SoftmaxRow {
                exp_table: TableId(1),
                sigma: 16,
            },
            vec![square],
            vec![lookup(&[T, T], 2), advice(&[T])],
            op_error(
                "SoftmaxRow",
                "the exponentials are Exp lookup outputs, the reciprocals advice",
            ),
        ),
        (
            Op::RsqrtRow {
                kappa: 16,
                eps_int: 0,
                dim: 0,
            },
            vec![positions],
            vec![advice(&[T])],
            op_error("RsqrtRow", "the rows have at least one element"),
        ),
        (
            Op::RsqrtRow {
                kappa: 32,
                eps_int: 0,
                dim: D,
            },
            vec![positions],
            vec![advice(&[T])],
            op_error("RsqrtRow", "κ ≤ 31"),
        ),
        (
            Op::RsqrtRow {
                kappa: 16,
                eps_int: 0,
                dim: D,
            },
            vec![positions],
            vec![advice(&[D])],
            op_error("RsqrtRow", "one root per sum"),
        ),
        (
            Op::RsqrtRow {
                kappa: 16,
                eps_int: 0,
                dim: D,
            },
            vec![positions],
            vec![lookup(&[T], 0)],
            op_error("RsqrtRow", "the roots are committed advice"),
        ),
        // The certificates.
        (
            Op::Argmax,
            vec![a, positions],
            vec![],
            op_error("Argmax", "one one-hot entry per logit"),
        ),
        (
            Op::InverseCdf { prf: PrfRef(0) },
            vec![a, c],
            vec![virt(&[T])],
            op_error(
                "InverseCdf",
                "takes the logits and the one-hot tokens, and produces nothing",
            ),
        ),
        // Tensors and inputs.
        (
            Op::Add,
            vec![a, TensorId(99)],
            vec![virt(&[T, D])],
            IrError::UnknownTensor {
                node: NodeId(0),
                input: 1,
            },
        ),
        (
            Op::Add,
            vec![a, c],
            vec![virt(&[T, 0])],
            IrError::Tensor {
                // The node's output, the first tensor after `base`'s inputs.
                tensor: TensorId(square.0 + 1),
                rule: "a shape has at least one axis, each of length at least 1",
            },
        ),
        // Each operation's arity, one input or output away from it.
        (
            Op::MatMulWeight { weight: q },
            vec![a],
            vec![virt(&[T, D])],
            op_error(
                "MatMulWeight",
                "takes the input and the weight, and produces one tensor",
            ),
        ),
        (
            Op::ConstMul { c: konst(0) },
            vec![a, c],
            vec![virt(&[T, D])],
            op_error("ConstMul", "takes one tensor and produces one"),
        ),
        (
            Op::Rotate { rope: TableId(9) },
            vec![a],
            vec![virt(&[T, D])],
            op_error(
                "Rotate",
                "takes the tensor and its rows' positions, and produces one",
            ),
        ),
        (
            Op::Sum { axis: 0 },
            vec![a],
            vec![],
            op_error("Sum", "takes one tensor and produces one"),
        ),
        (
            Op::Reshape,
            vec![a, c],
            vec![virt(&[T, D])],
            op_error("Reshape", "takes one tensor and produces one"),
        ),
        (
            Op::Slice {
                axis: 0,
                start: 0,
                len: 1,
            },
            vec![],
            vec![virt(&[1, D])],
            op_error("Slice", "takes one tensor and produces one"),
        ),
        (
            Op::GatherPublic { index: konst(1) },
            vec![a],
            vec![virt(&[2, D]), virt(&[2, D])],
            op_error("GatherPublic", "takes one tensor and produces one"),
        ),
        (
            Op::MatMulAct {
                transpose_rhs: true,
            },
            vec![a],
            vec![virt(&[T, T])],
            op_error("MatMulAct", "takes two tensors and produces one"),
        ),
        (
            Op::Mul {
                broadcast: Broadcast::Trailing,
            },
            vec![a],
            vec![virt(&[T, D])],
            op_error("Mul", "takes two tensors and produces one"),
        ),
        (
            Op::OneHot { vocab: VOCAB },
            vec![positions, positions],
            vec![advice(&[T, VOCAB])],
            op_error(
                "OneHot",
                "takes the tokens and produces their one-hot vectors",
            ),
        ),
        (
            Op::TableMap { table: TableId(3) },
            vec![a, classes, classes],
            vec![lookup(&[T, D], 3)],
            op_error(
                "TableMap",
                "takes the tensor, with the channel classes of a classed table, and produces one",
            ),
        ),
        (
            Op::Bitwise { table: TableId(4) },
            vec![a],
            vec![lookup(&[T, D], 4)],
            op_error("Bitwise", "takes its operands and produces one tensor"),
        ),
        (
            Op::RsqrtRow {
                kappa: 16,
                eps_int: 0,
                dim: D,
            },
            vec![positions],
            vec![advice(&[T]), advice(&[T])],
            op_error(
                "RsqrtRow",
                "takes the sums of squares and produces one root per row",
            ),
        ),
    ];
    let bit_cases: Vec<Case> = vec![
        // BitDecompose, BitFn and AssertEqual (RFC-0066).
        (
            Op::BitDecompose { low: 8, high: 2 },
            vec![a],
            vec![advice(&[T, D, 8])],
            op_error(
                "BitDecompose",
                "takes the integer and produces its low bits, then its high bits",
            ),
        ),
        (
            Op::BitDecompose { low: 0, high: 2 },
            vec![a],
            vec![advice(&[T, D, 1]), advice(&[T, D, 2])],
            op_error("BitDecompose", "1 ≤ low and low + high ≤ 62"),
        ),
        (
            Op::BitDecompose { low: 32, high: 31 },
            vec![a],
            vec![advice(&[T, D, 32]), advice(&[T, D, 31])],
            op_error("BitDecompose", "1 ≤ low and low + high ≤ 62"),
        ),
        (
            Op::BitDecompose { low: 8, high: 0 },
            vec![a],
            vec![advice(&[T, D])],
            op_error(
                "BitDecompose",
                "each output is the input with an innermost axis of its bits",
            ),
        ),
        (
            Op::BitDecompose { low: 8, high: 2 },
            vec![a],
            vec![advice(&[T, D, 8]), advice(&[T, D, 3])],
            op_error(
                "BitDecompose",
                "each output is the input with an innermost axis of its bits",
            ),
        ),
        (
            Op::BitDecompose { low: 8, high: 0 },
            vec![a],
            vec![virt(&[T, D, 8])],
            op_error("BitDecompose", "the bits are committed advice"),
        ),
        (
            Op::BitFn { f: BoolFn::Xor3 },
            vec![a, c],
            vec![virt(&[T, D])],
            op_error("BitFn", "takes its bits and produces one tensor"),
        ),
        (
            Op::BitFn { f: BoolFn::And },
            vec![a, odd],
            vec![virt(&[T, D])],
            op_error("BitFn", "the inputs and the output have one shape"),
        ),
        (
            Op::BitFn { f: BoolFn::Xor },
            vec![a, c],
            vec![virt(&[T, 3])],
            op_error("BitFn", "the inputs and the output have one shape"),
        ),
        (
            Op::BitFn { f: BoolFn::Ch },
            vec![a, c, a],
            vec![advice(&[T, D])],
            op_error("BitFn", "a Boolean function's value is virtual"),
        ),
        (
            Op::AssertEqual,
            vec![a, c],
            vec![virt(&[T, D])],
            op_error("AssertEqual", "takes two tensors and produces nothing"),
        ),
        (
            Op::AssertEqual,
            vec![a, odd],
            vec![],
            op_error("AssertEqual", "the two tensors have one shape"),
        ),
    ];
    for (op, inputs, outputs, expected) in cases.into_iter().chain(bit_cases) {
        let mut b = base_builder.clone();
        let before = b.clone().finish();
        let name = op.name();
        assert_eq!(b.node(op, &inputs, outputs), Err(expected), "{name}");
        assert_eq!(
            b.finish(),
            before,
            "{name}: a refused node leaves the builder as it was"
        );
    }
}

#[test]
fn malformed_declarations_and_inputs_are_refused() {
    let mut b = GraphBuilder::new();
    assert_eq!(
        b.input(virt(&[T])),
        Err(IrError::InputKind {
            tensor: TensorId(0)
        })
    );
    assert_eq!(
        b.input(lookup(&[T], 0)),
        Err(IrError::InputKind {
            tensor: TensorId(0)
        })
    );
    let tensor = |rule| {
        Err(IrError::Tensor {
            tensor: TensorId(0),
            rule,
        })
    };
    assert_eq!(
        b.input(public(&[])),
        tensor("a shape has at least one axis, each of length at least 1")
    );
    assert_eq!(
        b.input(public(&[1 << 24, 1 << 25])),
        tensor("a tensor holds at most 2^48 elements")
    );
    assert_eq!(
        b.input(public(&[u32::MAX, u32::MAX, u32::MAX])),
        tensor("a tensor holds at most 2^48 elements")
    );
    let mut wide = public(&[T]);
    wide.ty = IntType {
        bits: 65,
        signed: true,
    };
    assert_eq!(b.input(wide), tensor("an integer type has 1 to 64 bits"));
    let mut empty = public(&[T]);
    empty.range = IntRange { lo: 1, hi: 0 };
    assert_eq!(b.input(empty), tensor("a range has lo ≤ hi"));
    assert_eq!(
        b.input(public(&[1 << 24, 1 << 24])).map(|t| t.0),
        Ok(0),
        "2^48 elements fit"
    );

    let classes = "a set of shift classes has one to four, each with 1 ≤ r_c ≤ 63";
    for r_by_class in [vec![], vec![1; 5], vec![0], vec![64, 1]] {
        assert_eq!(
            b.shift_classes(r_by_class),
            Err(IrError::Declaration { rule: classes })
        );
    }
    assert_eq!(
        b.constant(vec![]),
        Err(IrError::Declaration {
            rule: "a constant holds at least one value"
        })
    );
    let q = weight(WeightType::Q);
    b.weight(q, D, D).expect("first declaration");
    assert_eq!(
        b.weight(q, D, D),
        Err(IrError::DuplicateWeight { weight: q })
    );
}

#[test]
fn a_graph_assembled_otherwise_is_validated_rule_by_rule() {
    let valid = block();
    assert_eq!(valid.validate(), Ok(()));
    let refused = |change: fn(&mut Graph), expected| {
        let mut graph = valid.clone();
        change(&mut graph);
        assert_eq!(graph.validate(), Err(expected));
    };
    refused(
        |g| g.tensors[3].id = TensorId(4),
        IrError::Id {
            rule: "tensor ids are their positions",
        },
    );
    refused(
        |g| g.nodes[2].id = NodeId(7),
        IrError::Id {
            rule: "node ids are their positions",
        },
    );
    // Node 1 reads the output of node 2: out of topological order.
    refused(
        |g| {
            let later = g.nodes[2].outputs[0];
            g.nodes[1].inputs[0] = later;
        },
        IrError::Order {
            node: NodeId(1),
            input: 0,
        },
    );
    // A node reading its own output: the shortest cycle.
    refused(
        |g| {
            let own = g.nodes[2].outputs[0];
            g.nodes[2].inputs[0] = own;
        },
        IrError::Order {
            node: NodeId(2),
            input: 0,
        },
    );
    let twice = valid.nodes[1].outputs[0];
    let mut graph = valid.clone();
    graph.nodes[2].outputs[0] = twice;
    assert_eq!(
        graph.validate(),
        Err(IrError::ProducedTwice { tensor: twice })
    );
    refused(
        |g| g.nodes[2].inputs[1] = TensorId(10_000),
        IrError::UnknownTensor {
            node: NodeId(2),
            input: 1,
        },
    );
    // A node's output kind changed after the fact.
    refused(
        |g| {
            let output = g.nodes[2].outputs[0];
            g.tensors[output.0 as usize].kind = NodeKind::Public;
        },
        op_error_at(
            2,
            "Add",
            "a linear or degree-2 operation produces a virtual tensor, or committed advice if it \
             is not a matmul",
        ),
    );
    // A graph input that no node produces but is virtual.
    refused(
        |g| g.tensors[0].kind = NodeKind::Virtual,
        IrError::InputKind {
            tensor: TensorId(0),
        },
    );
    refused(
        |g| {
            let first = g.weights[0];
            g.weights.push(first);
        },
        IrError::DuplicateWeight {
            weight: weight(WeightType::Embed),
        },
    );
    refused(
        |g| g.shift_classes[0] = vec![],
        IrError::Declaration {
            rule: "a set of shift classes has one to four, each with 1 ≤ r_c ≤ 63",
        },
    );
    refused(
        |g| g.constants[0] = vec![],
        IrError::Declaration {
            rule: "a constant holds at least one value",
        },
    );
    // Every rule the builder checks on a node, validate checks too: a node replaced by one the
    // builder would refuse.
    refused(
        |g| {
            g.nodes[2] = Node {
                op: Op::Sub,
                inputs: vec![g.nodes[2].inputs[0]],
                ..g.nodes[2].clone()
            };
        },
        op_error_at(2, "Sub", "takes two tensors and produces one"),
    );
}

fn op_error_at(node: u32, op: &'static str, rule: &'static str) -> IrError {
    IrError::Op {
        node: NodeId(node),
        op,
        rule,
    }
}

#[test]
fn each_boolean_function_is_its_truth_table_on_bits() {
    // RFC-0066 'BitFn': the polynomial agrees with the function on {0, 1}, by FIPS 180-4 §4.1.2
    // for Ch and Maj; and the classes the schedule places them by.
    type Truth = fn(bool, bool, bool) -> bool;
    let functions: [(BoolFn, usize, Truth); 5] = [
        (BoolFn::Xor, 2, |a, b, _| a != b),
        (BoolFn::Xor3, 3, |a, b, c| (a != b) != c),
        (BoolFn::And, 2, |a, b, _| a && b),
        (BoolFn::Ch, 3, |e, f, g| (e && f) != (!e && g)),
        (BoolFn::Maj, 3, |a, b, c| ((a && b) != (a && c)) != (b && c)),
    ];
    for (f, arity, truth) in functions {
        assert_eq!(f.arity(), arity, "{f:?}");
        for x in 0..1i64 << arity {
            let bits: Vec<i64> = (0..arity).map(|i| (x >> i) & 1).collect();
            let bit = |i: usize| bits.get(i).is_some_and(|&b| b == 1);
            assert_eq!(
                f.eval(&bits),
                i64::from(truth(bit(0), bit(1), bit(2))),
                "{f:?} at {bits:?}"
            );
        }
        assert_eq!(Op::BitFn { f }.class(), OpClass::Degree3);
    }
    assert_eq!(
        Op::BitDecompose { low: 8, high: 0 }.class(),
        OpClass::Gadget
    );
    assert_eq!(Op::AssertEqual.class(), OpClass::Gadget);
}
