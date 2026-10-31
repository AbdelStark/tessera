// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0071 witness tests: slot isolation, padded-row zeros, Resident vs Regenerator bit-exact
//! tile equivalence (including PROTOCOL_VERSION_2 when n_max < S), and multi-request greedy
//! certificate keys, multiplicities, and ExactMismatch refusals.

mod common;

use allocation_counter::measure;
use tessera_codec::{Ascii, Secret, SecretBuf};
use tessera_compiler::{
    ModelAdviceLayout, PublicModel, RuntimePlan, analyze, claim_list, lower, lower_batch,
    lower_batch_for_protocol, select,
};
use tessera_ir::{GadgetPolicy, PROTOCOL_VERSION, PROTOCOL_VERSION_2, ParamSetId};
use tessera_witness::{
    Checkpoints, Engine, Regenerator, Resident, TileWitness, WitnessError, WitnessSource,
};

fn pack_batch_tokens(slot_len: u32, requests: &[&[u32]]) -> Vec<u32> {
    let s = slot_len as usize;
    let mut out = vec![0u32; requests.len() * s];
    for (r, req) in requests.iter().enumerate() {
        out[r * s..r * s + req.len()].copy_from_slice(req);
    }
    out
}

fn greedy_argmax(logits: &[i64]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (idx, &val) in logits.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx as u32
}

fn generate_greedy_sequence(
    model: &tessera_ir::IntegerModel,
    prompt: &[u32],
    output_len: usize,
) -> Vec<u32> {
    let public = PublicModel::of(model);
    let mut seq = prompt.to_vec();
    for _ in 0..output_len {
        let lowering = lower(public, seq.len() as u32).expect("lower step");
        let engine = Engine::new(&lowering).expect("engine");
        let witness = engine
            .run(model, &Secret::new(seq.clone()))
            .expect("step run");
        let logits = witness.logits().expect("logits");
        let next = greedy_argmax(logits.positions.last().expect("last row"));
        seq.push(next);
    }
    seq
}

fn assert_same_tile(a: &TileWitness, b: &TileWitness, context: &str) {
    assert_eq!(a.shares().len(), b.shares().len(), "{context}: shares len");
    for (i, (sa, sb)) in a.shares().iter().zip(b.shares()).enumerate() {
        assert_eq!(&sa[..], &sb[..], "{context}: share {i}");
    }
    assert_eq!(a.gadget().len(), b.gadget().len(), "{context}: gadget len");
    for (i, ((ra, ga), (rb, gb))) in a.gadget().iter().zip(b.gadget()).enumerate() {
        assert_eq!(ra, rb, "{context}: gadget role {i}");
        assert_eq!(&ga[..], &gb[..], "{context}: gadget values {i}");
    }
    assert_eq!(
        a.greedy_maximum(),
        b.greedy_maximum(),
        "{context}: greedy maximum"
    );
    let la: Vec<_> = a.lookups().collect();
    let lb: Vec<_> = b.lookups().collect();
    assert_eq!(la.len(), lb.len(), "{context}: lookups len");
    for (i, (xa, xb)) in la.iter().zip(&lb).enumerate() {
        assert_eq!(xa.table(), xb.table(), "{context}: lookup table {i}");
        assert_eq!(xa.keys(), xb.keys(), "{context}: lookup keys {i}");
        assert_eq!(xa.values(), xb.values(), "{context}: lookup values {i}");
    }
    match (a.multiplicities(), b.multiplicities()) {
        (None, None) => {}
        (Some(ma), Some(mb)) => {
            assert_eq!(ma.rows(), mb.rows(), "{context}: multiplicity rows");
            let len = usize::try_from(ma.rows()).expect("rows fit usize");
            let mut ba = SecretBuf::filled(0, len);
            let mut bb = SecretBuf::filled(0, len);
            ma.read_rows(0, &mut ba).expect("read a");
            mb.read_rows(0, &mut bb).expect("read b");
            assert_eq!(&ba[..], &bb[..], "{context}: multiplicity values");
        }
        _ => panic!("{context}: multiplicity presence mismatch"),
    }
}

#[test]
fn batched_witness_isolates_slots_zeros_padded_rows_and_refuses_nonzero_padded_tokens() {
    let fixture = &common::conformance()[0];
    let public = PublicModel::of(&fixture.model);
    let req0 = &fixture.sequences[0].tokens[..3];
    let req1 = &fixture.sequences[1].tokens[..5];
    let req2 = &fixture.sequences[0].tokens[1..3];

    let standalone: Vec<_> = [req0, req1, req2]
        .into_iter()
        .map(|seq| {
            let l = lower(public, seq.len() as u32).expect("standalone lower");
            let e = Engine::new(&l).expect("engine");
            e.run(&fixture.model, &Secret::new(seq.to_vec()))
                .expect("standalone run")
                .logits()
                .expect("logits")
                .positions
        })
        .collect();

    let lowering = lower_batch(public, &[3, 5, 2]).expect("batch lower");
    let slots = lowering.graph.slots.as_ref().expect("R=3 slots");
    assert_eq!(slots.slot_len, 8);
    let engine = Engine::new(&lowering).expect("batch engine");
    let packed = pack_batch_tokens(slots.slot_len, &[req0, req1, req2]);
    let canonical = {
        let concatenated = Secret::new([req0, req1, req2].concat());
        engine
            .canonical_batch_tokens(&fixture.model, &concatenated)
            .expect("owned canonical tokens")
    };
    let copied = engine
        .run(&fixture.model, &canonical)
        .expect("retained token replay");
    assert!(
        copied
            .tensor(lowering.tokens)
            .unwrap()
            .iter()
            .zip(&packed)
            .all(|(&a, &b)| a == i64::from(b))
    );
    for bad in [vec![0; 9], vec![fixture.model.arch.vocab; 10]] {
        let bad = Secret::new(bad);
        let info = measure(|| {
            assert_eq!(
                engine.canonical_batch_tokens(&fixture.model, &bad).err(),
                Some(WitnessError::Usage {
                    rule: "concatenated batch tokens match real request lengths and vocabulary",
                })
            );
        });
        assert_eq!((info.count_total, info.bytes_max), (0, 0));
    }
    let batch_logits = engine
        .run(&fixture.model, &Secret::new(packed.clone()))
        .expect("batch run")
        .logits()
        .expect("batch logits");
    assert_eq!(batch_logits.positions.len(), 24);

    let zero_row = vec![0i64; fixture.model.arch.vocab as usize];
    for (r, want) in standalone.iter().enumerate() {
        for s_pos in 0..8usize {
            let tau = r * 8 + s_pos;
            if s_pos < want.len() {
                assert_eq!(
                    batch_logits.positions[tau], want[s_pos],
                    "slot {r} pos {s_pos} matches standalone"
                );
            } else {
                assert_eq!(
                    batch_logits.positions[tau], zero_row,
                    "padded row {tau} is identically zero"
                );
            }
        }
    }

    // Mutating slot 1's tokens leaves slots 0 and 2 bit-for-bit unchanged.
    let mut alt_req1 = req1.to_vec();
    alt_req1[0] = (alt_req1[0] + 1) % fixture.model.arch.vocab;
    alt_req1[4] = (alt_req1[4] + 2) % fixture.model.arch.vocab;
    let alt_packed = pack_batch_tokens(slots.slot_len, &[req0, &alt_req1, req2]);
    let alt_logits = engine
        .run(&fixture.model, &Secret::new(alt_packed))
        .expect("alt batch run")
        .logits()
        .expect("alt batch logits");
    for r in [0usize, 2usize] {
        for p in 0..standalone[r].len() {
            assert_eq!(
                alt_logits.positions[r * 8 + p],
                batch_logits.positions[r * 8 + p],
                "slot {r} pos {p} isolated from slot 1 mutation"
            );
        }
    }

    // Non-zero token on an inter-slot padded row is refused with ERR-USAGE.
    let mut bad_padded = packed;
    bad_padded[3] = 1;
    assert!(matches!(
        engine.run(&fixture.model, &Secret::new(bad_padded)),
        Err(WitnessError::Usage { .. })
    ));
}

#[test]
fn resident_and_regenerator_agree_across_batched_runtime_tiles_and_protocols() {
    let fixture = &common::conformance()[0];
    let public = PublicModel::of(&fixture.model);

    // Case 1: PROTOCOL_VERSION, R = 3 with mixed active/prompt-only requests [(2, 1), (2, 2), (2, 0)].
    let r0 = generate_greedy_sequence(&fixture.model, &[1, 2], 1);
    let r1 = generate_greedy_sequence(&fixture.model, &[3, 0], 2);
    let r2 = vec![2u32, 1];
    check_batch_runtime_equivalence(
        &fixture.model,
        PROTOCOL_VERSION,
        &[(2, 1), (2, 2), (2, 0)],
        &[&r0, &r1, &r2],
    );

    // Case 2: PROTOCOL_VERSION_2, R = 2 with n_max = 2 < S = 4 [(1, 0), (1, 1)].
    let p2_r0 = vec![1u32];
    let p2_r1 = generate_greedy_sequence(&fixture.model, &[2], 1);
    check_batch_runtime_equivalence(
        &fixture.model,
        PROTOCOL_VERSION_2,
        &[(1, 0), (1, 1)],
        &[&p2_r0, &p2_r1],
    );

    // Non-greedy output token in request 2 (slot 1) is refused with ExactMismatch by both sources.
    let lowering = lower_batch(public, &[3, 4, 2]).expect("lower");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = common::policy(2);
    let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("select");
    let params = ParamSetId(Ascii::new("test-small/1").expect("ascii"));
    let claims = claim_list(public, &selection, &policy, common::TEST_SMALL, params).expect("cl");
    let runtime =
        RuntimePlan::derive_batch(&lowering, &selection, &claims, &[(2, 1), (2, 2), (2, 0)])
            .expect("runtime");
    let advice = ModelAdviceLayout::runtime(&lowering, &runtime, &claims).expect("advice");
    let mut wrong_r1 = r1.clone();
    wrong_r1[3] = (wrong_r1[3] + 1) % fixture.model.arch.vocab;
    let slot_len = lowering.graph.slots.as_ref().expect("slots").slot_len;
    let bad_tokens = Secret::new(pack_batch_tokens(slot_len, &[&r0, &wrong_r1, &r2]));
    let engine = Engine::new(&lowering).expect("engine");
    // Public planning precedes any witness construction, even for a later refused continuation.
    let plans = engine
        .runtime_multiplicity_plans(&fixture.model.arch, &runtime)
        .expect("public plans");
    assert!(
        plans
            .windows(2)
            .all(|pair| pair[0].table() < pair[1].table())
    );
    assert!(
        plans
            .iter()
            .any(|plan| plan.table() == runtime.certificate().unwrap().table)
    );
    let first = engine
        .run(&fixture.model, &bad_tokens)
        .expect("model pass still runs");
    let checkpoints = Checkpoints::new(&engine, &fixture.model, &bad_tokens).expect("checkpoints");
    let resident =
        Resident::runtime(&engine, &runtime, &fixture.model, &first).expect("resident runtime");
    let regenerator =
        Regenerator::runtime(&engine, &runtime, &fixture.model, &bad_tokens, &checkpoints)
            .expect("regenerator runtime");
    let cert = runtime.certificate().expect("active certificate");
    for tile in advice.tiles() {
        let should_fail = match tile.tile {
            tessera_compiler::Tile::GreedyMaximum { .. } => true,
            tessera_compiler::Tile::Table { table } => table == cert.table.0,
            _ => false,
        };
        if should_fail {
            assert_eq!(
                resident.tile(tile).expect_err("resident exact mismatch"),
                WitnessError::ExactMismatch
            );
            assert_eq!(
                regenerator
                    .tile(tile)
                    .expect_err("regenerator exact mismatch"),
                WitnessError::ExactMismatch
            );
        }
    }
}

fn check_batch_runtime_equivalence(
    model: &tessera_ir::IntegerModel,
    protocol: u16,
    requests: &[(u32, u32)],
    sequences: &[&[u32]],
) {
    let public = PublicModel::of(model);
    let lengths: Vec<u32> = requests.iter().map(|&(p, o)| p + o).collect();
    let lowering = lower_batch_for_protocol(protocol, public, &lengths).expect("lower batch");
    let slot_len = lowering.graph.slots.as_ref().expect("slots").slot_len;
    let tokens = Secret::new(pack_batch_tokens(slot_len, sequences));
    let engine = Engine::new(&lowering).expect("engine");
    let first = engine.run(model, &tokens).expect("first pass");
    let checkpoints = Checkpoints::new(&engine, model, &tokens).expect("checkpoints");
    let cached = tessera_witness::CheckpointTensors::new(&engine, model, &tokens, &checkpoints)
        .expect("nonweight checkpoint cache");
    for q in [1u32, 2, 4, 512] {
        let policy = GadgetPolicy {
            query_block: q,
            ..GadgetPolicy::default()
        };
        let single = lower(public, 1).expect("single lower");
        let single_ranges = analyze(&single.graph, &single.tables).expect("single ranges");
        let single_sel =
            select(&single.graph, &single_ranges, &single.tables, &policy).expect("single select");
        let params = ParamSetId(Ascii::new("test-small/1").expect("ascii"));
        let claims =
            claim_list(public, &single_sel, &policy, common::TEST_SMALL, params).expect("claims");
        let batch_ranges = analyze(&lowering.graph, &lowering.tables).expect("batch ranges");
        let batch_sel =
            select(&lowering.graph, &batch_ranges, &lowering.tables, &policy).expect("batch sel");
        let runtime =
            RuntimePlan::derive_batch(&lowering, &batch_sel, &claims, requests).expect("runtime");
        let advice = ModelAdviceLayout::runtime(&lowering, &runtime, &claims).expect("advice");
        let plans = engine
            .runtime_multiplicity_plans(&model.arch, &runtime)
            .expect("public count plans");
        let planned: std::collections::BTreeSet<_> =
            plans.iter().map(|plan| plan.table()).collect();
        let required: std::collections::BTreeSet<_> = advice
            .tiles()
            .iter()
            .filter_map(|tile| {
                if let tessera_compiler::Tile::Table { table } = tile.tile {
                    Some(tessera_ir::TableId(table))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(plans.len(), planned.len());
        assert_eq!(planned, required);
        let mut work = None;
        let planning =
            measure(|| work = Some(engine.runtime_source_tile_work_memory(&runtime).unwrap()));
        assert_eq!(planning.count_total, 0);
        let work = work.unwrap();
        let resident = Resident::runtime(&engine, &runtime, model, &first).expect("resident");
        let borrowed = Resident::runtime_from_tensors(&engine, &runtime, model, &cached)
            .expect("borrowed INT8 source");
        let regenerator =
            Regenerator::runtime(&engine, &runtime, model, &tokens, &checkpoints).expect("regen");
        if q == 1 {
            for tensor in &lowering.graph.tensors {
                assert_eq!(
                    &regenerator.tensor(tensor.id).expect("tensor replay")[..],
                    &first.tensor(tensor.id).expect("resident tensor")[..],
                    "protocol={protocol}, tensor {:?}",
                    tensor.id
                );
            }
        }
        for (idx, tile) in advice.tiles().iter().enumerate() {
            let count_bytes = if let tessera_compiler::Tile::Table { table } = tile.tile {
                plans
                    .iter()
                    .find(|plan| plan.table() == tessera_ir::TableId(table))
                    .unwrap()
                    .payload_bytes()
            } else {
                0
            };
            let mut rt = None;
            let requests = measure(|| rt = Some(resident.tile(tile).expect("resident tile")));
            assert!(
                requests.bytes_total <= work + count_bytes,
                "protocol={protocol} q={q} tile={idx} {requests:?} work={work} counts={count_bytes}"
            );
            let rt = rt.unwrap();
            if let Some(column) = rt.multiplicities() {
                let plan = plans
                    .iter()
                    .find(|plan| plan.table() == column.table())
                    .unwrap();
                assert_eq!(plan.rows(), column.rows());
                assert_eq!(plan.payload_bytes(), rt.bytes() as u64);
            }
            let gt = regenerator.tile(tile).expect("regenerator tile");
            assert_same_tile(&rt, &gt, &format!("protocol={protocol} q={q} tile={idx}"));
            let mut bt = None;
            let requests = measure(|| bt = Some(borrowed.tile(tile).expect("borrowed INT8 tile")));
            assert!(
                requests.bytes_total <= work + count_bytes,
                "borrowed protocol={protocol} q={q} tile={idx} {requests:?} work={work} counts={count_bytes}"
            );
            let bt = bt.unwrap();
            assert_same_tile(
                &rt,
                &bt,
                &format!("borrowed protocol={protocol} q={q} tile={idx}"),
            );
        }
    }
}

#[test]
fn public_count_plans_include_each_runtime_request_without_a_witness() {
    let fixture = &common::conformance()[0];
    let public = PublicModel::of(&fixture.model);
    for protocol in [PROTOCOL_VERSION, PROTOCOL_VERSION_2] {
        let lengths = [3u32, 4, 2];
        let lowering = lower_batch_for_protocol(protocol, public, &lengths).expect("lower");
        let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
        let policy = common::policy(2);
        let selection =
            select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("select");
        let params = ParamSetId(Ascii::new("test-small/1").expect("ascii"));
        let claims =
            claim_list(public, &selection, &policy, common::TEST_SMALL, params).expect("claims");
        let engine = Engine::new(&lowering).expect("engine");
        let inactive = [(3, 0), (4, 0), (2, 0)];
        let base_runtime =
            RuntimePlan::derive_batch(&lowering, &selection, &claims, &inactive).expect("base");
        let base = engine
            .runtime_multiplicity_plans(&fixture.model.arch, &base_runtime)
            .expect("base plans");
        for requests in [inactive, [(2, 1), (2, 2), (2, 0)], [(1, 2), (2, 2), (1, 1)]] {
            let runtime = RuntimePlan::derive_batch(&lowering, &selection, &claims, &requests)
                .expect("runtime");
            let plans = engine
                .runtime_multiplicity_plans(&fixture.model.arch, &runtime)
                .expect("plans");
            // Count each active request's padded output/vocabulary rectangle independently of
            // GreedyKeyColumn::entries and the production update-bound derivation (RFC-0071).
            let power = |n: u32| {
                let mut p = 1u64;
                while p < u64::from(n) {
                    p *= 2;
                }
                p
            };
            let entries: u64 = requests
                .iter()
                .filter(|&&(_, output)| output != 0)
                .map(|&(_, output)| power(output) * power(fixture.model.arch.vocab))
                .sum();
            let certificate = runtime.certificate();
            // Use an independent consumer for table coverage: two calls to this planner
            // could otherwise omit the same table and agree (mutation control, RFC-0018).
            let advice = ModelAdviceLayout::runtime(&lowering, &runtime, &claims).expect("advice");
            let expected: std::collections::BTreeSet<_> = advice
                .tiles()
                .iter()
                .filter_map(|tile| {
                    if let tessera_compiler::Tile::Table { table } = tile.tile {
                        Some(tessera_ir::TableId(table))
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(
                plans.iter().map(|plan| plan.table()).collect::<Vec<_>>(),
                expected.into_iter().collect::<Vec<_>>()
            );
            for plan in plans {
                let baseline = base
                    .iter()
                    .find(|p| p.table() == plan.table())
                    .map_or(1, |p| p.max_updates());
                let added = if certificate.is_some_and(|c| c.table == plan.table()) {
                    entries
                } else {
                    0
                };
                assert_eq!(plan.max_updates(), baseline + added);
                if let Some(c) = certificate.filter(|c| c.table == plan.table()) {
                    assert_eq!(plan.rows(), 1u64 << c.range_bits);
                }
            }
        }
    }
}
