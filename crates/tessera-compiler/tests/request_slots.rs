// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Conformance and adversarial tests for batched request-slot lowering, schedules, runtime plans,
//! and advice layouts (RFC-0071, RFC-0072, #551).

mod common;

use common::public_vectors::{models, read_json, vectors};
use serde_json::Value;
use tessera_codec::Ascii;
use tessera_codec::error::ErrorCode;
use tessera_compiler::schedule::{
    NativeKind, derive_composed_schedule, model_natives_batch, protocol2_stages,
};
use tessera_compiler::{
    CompileError, ExecutionPlan, ModelAdviceLayout, PublicModel, RuntimePlan, analyze,
    check_batch_advice_rows, claim_list_for_protocol, derive_schedule, lower, lower_batch,
    lower_batch_for_protocol, plan, positions_values, select,
};
use tessera_ir::{
    AdviceLayoutParams, BatchSlots, ClaimList, CommitmentClass, CommitmentLayout, ConstRef,
    GadgetPolicy, GraphBuilder, IntRange, IntType, NodeKind, Op, PROTOCOL_VERSION,
    PROTOCOL_VERSION_2, ParamSetId, TensorDecl, WeightLayoutParams,
};

fn vec_u32(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| u32::try_from(x.as_u64().expect("u64")).expect("u32"))
        .collect()
}

fn map_protocol(oracle_protocol: u64) -> u16 {
    match oracle_protocol {
        1 => PROTOCOL_VERSION,
        2 => PROTOCOL_VERSION_2,
        other => panic!("unexpected oracle protocol {other}"),
    }
}

fn default_layout() -> CommitmentLayout {
    CommitmentLayout {
        advice: AdviceLayoutParams {
            log2_k: 12,
            log2_inv_rate: 1,
            mask_coeffs: 200,
            row_block: 16,
            segments: 4,
        },
        weights: WeightLayoutParams {
            log2_k: 8,
            log2_inv_rate: 2,
            mask_coeffs: 16,
            blinding_rows: 64,
        },
    }
}

#[test]
fn request_slots_lowering_schedule_and_layout_match_oracle_vectors() {
    let all_models = models();
    let path = vectors().join("request-slots/slots.json");
    let root = read_json(&path);
    let cases = root["lowering"].as_array().expect("lowering array");
    assert_eq!(cases.len(), 14);

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let model_name = case["model"].as_str().expect("model");
        let protocol = map_protocol(case["protocol"].as_u64().expect("protocol"));
        let lengths = vec_u32(&case["lengths"]);
        let slot_bits = u32::try_from(case["slot_bits"].as_u64().expect("slot_bits")).expect("u32");
        let slot_len = u32::try_from(case["slot_len"].as_u64().expect("slot_len")).expect("u32");
        let token_len = u32::try_from(case["token_len"].as_u64().expect("token_len")).expect("u32");
        let padded_tokens =
            usize::try_from(case["padded_tokens"].as_u64().expect("padded_tokens")).expect("usize");

        let model = &all_models[model_name];
        let public = PublicModel::of(model);
        let lowering = lower_batch_for_protocol(protocol, public, &lengths)
            .unwrap_or_else(|e| panic!("{name}: {e}"));

        if lengths.len() == 1 {
            assert!(lowering.graph.slots.is_none(), "{name}: R=1 slots");
            let single = lower(public, lengths[0]).expect("single lower");
            assert_eq!(lowering, single, "{name}: R=1 identity");
        } else {
            let slots = lowering.graph.slots.as_ref().expect("R>=2 slots");
            assert_eq!(slots.slot_bits(), slot_bits, "{name}: slot_bits");
            assert_eq!(slots.slot_len, slot_len, "{name}: slot_len");
            assert_eq!(slots.token_len(), token_len, "{name}: token_len");
            assert_eq!(
                slots.padded_tokens(),
                padded_tokens,
                "{name}: padded_tokens"
            );
            let real_rows: Vec<bool> = (0..token_len as usize)
                .map(|tau| slots.is_real_row(tau))
                .collect();
            let expected_real: Vec<bool> = case["real_rows"]
                .as_array()
                .expect("real_rows")
                .iter()
                .map(|b| b.as_bool().expect("bool"))
                .collect();
            assert_eq!(real_rows, expected_real, "{name}: real_rows");
        }

        let expected_positions: Vec<i64> = case["positions"]
            .as_array()
            .expect("positions")
            .iter()
            .map(|p| p.as_i64().expect("i64"))
            .collect();
        assert_eq!(
            positions_values(&lowering),
            expected_positions,
            "{name}: positions_values"
        );
        let pos_spec = &lowering.graph.tensors[lowering.positions.0 as usize];
        let bounds = case["positions_bounds"].as_array().expect("bounds");
        assert_eq!(
            [pos_spec.range.lo, pos_spec.range.hi],
            [
                i128::from(bounds[0].as_i64().expect("lo")),
                i128::from(bounds[1].as_i64().expect("hi")),
            ],
            "{name}: positions range"
        );
        assert_eq!(
            lowering.graph.nodes.len(),
            usize::try_from(case["node_count"].as_u64().expect("nodes")).expect("usize"),
            "{name}: nodes"
        );
        assert_eq!(
            lowering.graph.tensors.len(),
            usize::try_from(case["tensor_count"].as_u64().expect("tensors")).expect("usize"),
            "{name}: tensors"
        );
        assert_eq!(
            lowering.tables.len(),
            usize::try_from(case["table_count"].as_u64().expect("tables")).expect("usize"),
            "{name}: tables"
        );

        let shapes = &case["shapes"];
        let shape_of = |t: tessera_ir::TensorId| lowering.graph.tensors[t.0 as usize].shape.clone();
        assert_eq!(
            shape_of(lowering.tokens),
            vec_u32(&shapes["tokens"]),
            "{name}: tokens"
        );
        assert_eq!(
            shape_of(lowering.positions),
            vec_u32(&shapes["positions"]),
            "{name}: positions"
        );
        assert_eq!(
            shape_of(lowering.one_hot),
            vec_u32(&shapes["one_hot"]),
            "{name}: one_hot"
        );
        assert_eq!(
            shape_of(lowering.checkpoints[0]),
            vec_u32(&shapes["embed"]),
            "{name}: embed"
        );
        assert_eq!(
            shape_of(lowering.logits),
            vec_u32(&shapes["logits"]),
            "{name}: logits"
        );

        let qk_node = lowering
            .graph
            .nodes
            .iter()
            .find(|n| {
                matches!(
                    n.op,
                    Op::MatMulAct {
                        transpose_rhs: true
                    }
                )
            })
            .expect("QK^T node");
        assert_eq!(
            shape_of(qk_node.inputs[0]),
            vec_u32(&shapes["q"]),
            "{name}: q"
        );
        assert_eq!(
            shape_of(qk_node.inputs[1]),
            vec_u32(&shapes["k"]),
            "{name}: k"
        );
        assert_eq!(
            shape_of(qk_node.outputs[0]),
            vec_u32(&shapes["scores"]),
            "{name}: scores"
        );

        let sm_node = lowering
            .graph
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::SoftmaxRow { .. }))
            .expect("SoftmaxRow node");
        assert_eq!(
            shape_of(sm_node.outputs[0]),
            vec_u32(&shapes["exp"]),
            "{name}: exp"
        );
        assert_eq!(
            shape_of(sm_node.outputs[1]),
            vec_u32(&shapes["reciprocal"]),
            "{name}: reciprocal"
        );

        let pv_node = lowering
            .graph
            .nodes
            .iter()
            .find(|n| {
                matches!(
                    n.op,
                    Op::MatMulAct {
                        transpose_rhs: false
                    }
                )
            })
            .expect("PV node");
        assert_eq!(
            shape_of(pv_node.inputs[1]),
            vec_u32(&shapes["v"]),
            "{name}: v"
        );
        assert_eq!(
            shape_of(pv_node.outputs[0]),
            vec_u32(&shapes["pv"]),
            "{name}: pv"
        );

        // Plain schedule + composed native schedules (k = 0 and k = all requests).
        derive_schedule(&lowering).unwrap_or_else(|e| panic!("{name}: plain schedule: {e}"));
        for active in [0usize, lengths.len()] {
            let natives = model_natives_batch(&lowering, active)
                .unwrap_or_else(|e| panic!("{name}: natives({active}): {e}"));
            let composed = derive_composed_schedule(&lowering, &natives.deferred, &natives.native)
                .unwrap_or_else(|e| panic!("{name}: composed({active}): {e}"));
            if protocol == PROTOCOL_VERSION_2 {
                protocol2_stages(&lowering.graph, &composed, 8)
                    .unwrap_or_else(|e| panic!("{name}: p2 stages({active}): {e}"));
            }
        }

        // Execution plan tile expansion and advice layout derivation.
        let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
        let policy = GadgetPolicy::default();
        let selection =
            select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("select");
        let claims = claim_list_for_protocol(
            protocol,
            public,
            &selection,
            &policy,
            default_layout(),
            ParamSetId(Ascii::new("conformance-p1").expect("ascii")),
        )
        .expect("claim list");
        let exec_plan: ExecutionPlan = plan(public, &claims).expect("plan");
        let expanded = exec_plan.expand(&lowering).expect("expand");
        assert!(!expanded.is_empty());

        let requests: Vec<(u32, u32)> = lengths
            .iter()
            .map(|&n| if n >= 2 { (1, n - 1) } else { (1, 0) })
            .collect();
        let runtime =
            RuntimePlan::derive_batch(&lowering, &selection, &claims, &requests).expect("runtime");
        ModelAdviceLayout::runtime(&lowering, &runtime, &claims)
            .unwrap_or_else(|e| panic!("{name}: advice layout: {e}"));
    }
}

#[test]
fn greedy_batch_runtime_and_natives_match_oracle_vectors() {
    let all_models = models();
    let model = &all_models["tiny-2-minimal.d"];
    let path = vectors().join("request-slots/slots.json");
    let root = read_json(&path);
    let cases = root["greedy"].as_array().expect("greedy array");
    assert_eq!(cases.len(), 9);

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let protocol = map_protocol(case["protocol"].as_u64().expect("protocol"));
        let vocab = u32::try_from(case["vocab"].as_u64().expect("vocab")).expect("u32");
        let has_maximum = case["has_maximum"].as_bool().expect("has_maximum");
        let requests: Vec<(u32, u32)> = case["requests"]
            .as_array()
            .expect("requests")
            .iter()
            .map(|r| {
                (
                    u32::try_from(r["prompt_len"].as_u64().expect("pr")).expect("u32"),
                    u32::try_from(r["output_len"].as_u64().expect("out")).expect("u32"),
                )
            })
            .collect();

        let mut arch = model.arch.clone();
        arch.vocab = vocab;
        let public = PublicModel {
            arch: &arch,
            profile: model.profile,
            quant: &model.quant.public,
            silu_in_exponents: &model.quant.silu_in_exponents,
        };
        let lengths: Vec<u32> = requests.iter().map(|&(pr, out)| pr + out).collect();
        let lowering = lower_batch_for_protocol(protocol, public, &lengths)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
        let policy = GadgetPolicy::default();
        let selection =
            select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("select");
        let claims = claim_list_for_protocol(
            protocol,
            public,
            &selection,
            &policy,
            default_layout(),
            ParamSetId(Ascii::new("conformance-p1").expect("ascii")),
        )
        .expect("claim list");

        let runtime = RuntimePlan::derive_batch(&lowering, &selection, &claims, &requests)
            .unwrap_or_else(|e| panic!("{name}: runtime: {e}"));
        assert_eq!(
            runtime.certificate().is_some(),
            has_maximum,
            "{name}: has_maximum"
        );

        let actual_keys: Vec<(u32, u32, u32, u64)> = runtime
            .key_columns()
            .iter()
            .map(|col| {
                (
                    col.request_index,
                    col.prompt,
                    col.positions - col.prompt,
                    1u64 << col.variables,
                )
            })
            .collect();
        let expected_keys: Vec<(u32, u32, u32, u64)> = case["key_columns"]
            .as_array()
            .expect("key_columns")
            .iter()
            .map(|col| {
                (
                    u32::try_from(col["request_index"].as_u64().expect("req")).expect("u32"),
                    u32::try_from(col["prompt_len"].as_u64().expect("pr")).expect("u32"),
                    u32::try_from(col["output_len"].as_u64().expect("out")).expect("u32"),
                    col["entries"].as_u64().expect("entries"),
                )
            })
            .collect();
        assert_eq!(actual_keys, expected_keys, "{name}: key_columns");

        let natives = model_natives_batch(&lowering, runtime.key_columns().len()).expect("natives");
        let tail_len = if has_maximum {
            1 + expected_keys.len() + 1
        } else {
            2
        };
        let tail = &natives.kinds[natives.kinds.len() - tail_len..];
        let mut key_iter = runtime.key_columns().iter();
        let formatted: Vec<String> = tail
            .iter()
            .map(|kind| match kind {
                NativeKind::Tokens => "tokens".to_owned(),
                NativeKind::Observe => "observe".to_owned(),
                NativeKind::GreedySelected => "greedy_selected".to_owned(),
                NativeKind::GreedyKey => {
                    let col = key_iter.next().expect("key column");
                    format!("greedy_key:{}", col.request_index)
                }
                other => panic!("unexpected tail native {other:?}"),
            })
            .collect();
        let expected_natives: Vec<String> = case["native_kinds"]
            .as_array()
            .expect("native_kinds")
            .iter()
            .map(|s| s.as_str().expect("str").to_owned())
            .collect();
        assert_eq!(formatted, expected_natives, "{name}: native_kinds");
    }
}

#[test]
fn slotted_schedule_refuses_const_add_and_assert_equal_downstream_of_tokens_only() {
    let slots = BatchSlots::new(vec![2, 1]).expect("slots");
    let n = slots.token_len(); // 4
    let u8_ty = IntType {
        signed: false,
        bits: 8,
    };
    let u1_ty = IntType {
        signed: false,
        bits: 1,
    };
    let i16_ty = IntType {
        signed: true,
        bits: 16,
    };

    // 1. ConstAdd downstream of slotted tokens -> ERR-INTERNAL.
    let mut b = GraphBuilder::with_slots(slots.clone()).expect("builder");
    let c = b.constant(vec![3]).expect("const");
    let tok = b
        .input(TensorDecl {
            shape: vec![n],
            ty: u8_ty,
            range: IntRange { lo: 0, hi: 3 },
            kind: NodeKind::Committed {
                commitment: CommitmentClass::Advice,
            },
        })
        .expect("tok");
    let _oh = b
        .node(
            Op::OneHot { vocab: 4 },
            &[tok],
            vec![TensorDecl {
                shape: vec![n, 4],
                ty: u1_ty,
                range: IntRange { lo: 0, hi: 1 },
                kind: NodeKind::Committed {
                    commitment: CommitmentClass::Advice,
                },
            }],
        )
        .expect("one_hot");
    let _shifted = b
        .node(
            Op::ConstAdd { c },
            &[tok],
            vec![TensorDecl {
                shape: vec![n],
                ty: i16_ty,
                range: IntRange { lo: 3, hi: 6 },
                kind: NodeKind::Virtual,
            }],
        )
        .expect("const_add");
    let graph = b.finish();
    let err = tessera_compiler::derive_graph_schedule(&graph, &[], &[], None).unwrap_err();
    assert_eq!(err.code(), ErrorCode::Internal);

    // 2. AssertEqual downstream of slotted tokens -> ERR-INTERNAL.
    let mut b = GraphBuilder::with_slots(slots.clone()).expect("builder");
    let tok = b
        .input(TensorDecl {
            shape: vec![n],
            ty: u8_ty,
            range: IntRange { lo: 0, hi: 3 },
            kind: NodeKind::Committed {
                commitment: CommitmentClass::Advice,
            },
        })
        .expect("tok");
    let _oh = b
        .node(
            Op::OneHot { vocab: 4 },
            &[tok],
            vec![TensorDecl {
                shape: vec![n, 4],
                ty: u1_ty,
                range: IntRange { lo: 0, hi: 1 },
                kind: NodeKind::Committed {
                    commitment: CommitmentClass::Advice,
                },
            }],
        )
        .expect("one_hot");
    b.node(Op::AssertEqual, &[tok, tok], Vec::new())
        .expect("assert_equal");
    let graph = b.finish();
    let err = tessera_compiler::derive_graph_schedule(&graph, &[], &[], None).unwrap_err();
    assert_eq!(err.code(), ErrorCode::Internal);

    // 3. GatherPublic selecting ONLY real rows (0, 1 in slot 0 and 2 in slot 1; row 3 is padded)
    // cuts slotted padding, so ConstAdd downstream of GatherPublic succeeds!
    let mut b = GraphBuilder::with_slots(slots).expect("builder");
    let real_idx: ConstRef = b.constant(vec![0, 1, 2]).expect("indices");
    let shift: ConstRef = b.constant(vec![5]).expect("shift");
    let tok = b
        .input(TensorDecl {
            shape: vec![n],
            ty: u8_ty,
            range: IntRange { lo: 0, hi: 3 },
            kind: NodeKind::Committed {
                commitment: CommitmentClass::Advice,
            },
        })
        .expect("tok");
    let _oh = b
        .node(
            Op::OneHot { vocab: 4 },
            &[tok],
            vec![TensorDecl {
                shape: vec![n, 4],
                ty: u1_ty,
                range: IntRange { lo: 0, hi: 1 },
                kind: NodeKind::Committed {
                    commitment: CommitmentClass::Advice,
                },
            }],
        )
        .expect("one_hot");
    let gathered = b
        .node(
            Op::GatherPublic { index: real_idx },
            &[tok],
            vec![TensorDecl {
                shape: vec![3],
                ty: u8_ty,
                range: IntRange { lo: 0, hi: 3 },
                kind: NodeKind::Virtual,
            }],
        )
        .expect("gather")[0];
    let _out = b
        .node(
            Op::ConstAdd { c: shift },
            &[gathered],
            vec![TensorDecl {
                shape: vec![3],
                ty: i16_ty,
                range: IntRange { lo: 5, hi: 8 },
                kind: NodeKind::Committed {
                    commitment: CommitmentClass::Advice,
                },
            }],
        )
        .expect("const_add")[0];
    let graph = b.finish();
    tessera_compiler::derive_graph_schedule(&graph, &[], &[], None)
        .expect("GatherPublic of real rows cuts slotted padding");
}

#[test]
fn lower_batch_and_advice_rows_refuse_invalid_or_oversized_batches_with_typed_codes() {
    let all_models = models();
    let model = &all_models["tiny-2-minimal.d"];
    let public = PublicModel::of(model);

    // Empty batch -> ERR-USAGE.
    let err = lower_batch(public, &[]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::Usage);

    // Zero-length sequence -> ERR-USAGE.
    let err = lower_batch(public, &[4, 0]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::Usage);

    // Exceeds max_context -> ERR-USAGE.
    let err = lower_batch(public, &[4, model.arch.max_context + 1]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::Usage);

    // Unknown protocol_version -> ERR-USAGE.
    let err = lower_batch_for_protocol(99, public, &[4, 4]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::Usage);

    // Oversized R * S > 2^31 -> ERR-USAGE.
    let mut big_arch = model.arch.clone();
    big_arch.max_context = 1 << 17;
    let big_public = PublicModel {
        arch: &big_arch,
        ..public
    };
    let huge = vec![1 << 17; (1 << 14) + 1];
    let err = lower_batch(big_public, &huge).unwrap_err();
    assert!(matches!(err, CompileError::Usage { .. }));
    assert_eq!(err.code(), ErrorCode::Usage);

    // Valid single-sequence claim list whose multi-request batch pushes advice rows past 2^32
    // -> ERR-QUANT-ARCH.
    let mut wide_arch = model.arch.clone();
    wide_arch.max_context = 1 << 17;
    wide_arch.d_ff = 1 << 20;
    let wide_public = PublicModel {
        arch: &wide_arch,
        ..public
    };
    let one = lower(wide_public, 16).expect("lower 16");
    let ranges = analyze(&one.graph, &one.tables).expect("ranges");
    let policy = GadgetPolicy::default();
    let selection = select(&one.graph, &ranges, &one.tables, &policy).expect("select");
    let raw_claims = ClaimList {
        protocol_version: PROTOCOL_VERSION,
        arch: wide_arch,
        profile: model.profile,
        quant: model.quant.public.clone(),
        silu_in_exponents: model.quant.silu_in_exponents.clone(),
        tables: selection.tables,
        gadget_policy: policy,
        layout: CommitmentLayout {
            advice: AdviceLayoutParams {
                log2_k: 8,
                log2_inv_rate: 1,
                mask_coeffs: 16,
                row_block: 16,
                segments: 4,
            },
            weights: WeightLayoutParams {
                log2_k: 12,
                log2_inv_rate: 2,
                mask_coeffs: 256,
                blinding_rows: 64,
            },
        },
        params: ParamSetId(Ascii::new("ni-100/1").expect("ascii")),
    };
    let many_reqs = vec![(1u32, (1 << 14) - 1); 16];
    let err = check_batch_advice_rows(&raw_claims, &many_reqs).unwrap_err();
    assert_eq!(err.code(), ErrorCode::QuantArch);
}
