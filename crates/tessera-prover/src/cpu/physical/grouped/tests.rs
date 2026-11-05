// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::*;
use tessera_codec::{DecodeLimits, envelope, schema};

#[test]
fn canonical_group_work_planning_is_bounded_and_refuses_before_field_work() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/tamper/claims.tcl");
    let claims: tessera_ir::ClaimList = envelope::open(
        &std::fs::read(path).unwrap(),
        &schema::CLAIM_LIST,
        DecodeLimits::MESSAGE,
    )
    .unwrap();
    let mut cases = 0;
    for (lengths, output) in [(vec![3], false), (vec![5], true), (vec![3, 1, 2], true)] {
        let requests: Vec<_> = lengths
            .iter()
            .map(|&n| if output { (1, n - 1) } else { (n, 0) })
            .collect();
        let lowering = compiler::lower_batch(PublicModel::of_claims(&claims), &lengths).unwrap();
        let ranges = compiler::analyze(&lowering.graph, &lowering.tables).unwrap();
        let selected = compiler::select_model(
            claims.protocol_version,
            &lowering,
            &ranges,
            &claims.gadget_policy,
        )
        .unwrap();
        let runtime = RuntimePlan::derive_batch(&lowering, &selected, &claims, &requests).unwrap();
        let advice = ModelAdviceLayout::runtime(&lowering, &runtime, &claims).unwrap();
        let layers = LayerMap::new(&lowering).unwrap();
        let leaf = LeafLayout::runtime(&lowering, &layers, &runtime, &claims).unwrap();
        let native = ModelNativePlan::derive_runtime(&lowering, &runtime, &claims, &leaf).unwrap();
        let weights = pcs::WeightLayout::new(
            claims.layout.weights,
            pcs::WeightShapes {
                layers: claims.arch.n_layers,
                shapes: claims.arch.weight_shapes().unwrap(),
                channels: lowering
                    .classes
                    .iter()
                    .chain(&lowering.multipliers)
                    .map(|(w, _)| (w.ty.matrix(), w.layer))
                    .collect(),
            },
        )
        .unwrap();
        let statement = protocol::PublicStatement {
            claim_list_digest: [0; 32],
            weights_root: [0; 32],
            weights_epoch: 0,
            manifest_digest: [0; 32],
            batch: protocol::BatchDescriptor {
                requests: requests
                    .iter()
                    .enumerate()
                    .map(|(r, &(prompt_len, output_len))| protocol::RequestRef {
                        request_id: r as u64,
                        session_id: [0; 16],
                        prompt_len,
                        output_len,
                    })
                    .collect(),
            },
            predictor: protocol::PredictorSpec::Exact {
                decoding: protocol::Decoding::Greedy,
            },
        };
        for (token_bits, layer_bits) in [(0, 1), (1, 0), (1, 1)] {
            let grouping = GroupedLayout::new(
                &leaf,
                &layers,
                protocol::LeafGroups {
                    token_bits,
                    layer_bits,
                },
            )
            .unwrap();
            let input = Inputs {
                claims: &claims,
                statement: &statement,
                lowering: &lowering,
                layers: &layers,
                runtime: &runtime,
                advice: &advice,
                leaf: &leaf,
                native: &native,
                grouped: Some(&grouping),
                weights: &weights,
                session: None,
                caller_batch: None,
                planner_caller: 0,
            };
            let mut bill = None;
            let measured = allocation_counter::measure(|| {
                bill = Some(GroupedWorkMemory::new(&input, 64 << 30).unwrap());
            });
            let bill = bill.unwrap();
            assert_eq!(measured.bytes_current, 0);
            assert!(
                measured.bytes_max <= bill.planning,
                "{measured:?} > {bill:?}"
            );
            assert!(bill.peak > bill.messages.driver_bytes());
            assert_eq!(
                GroupedWorkMemory::new(&input, bill.planning - 1)
                    .unwrap_err()
                    .code(),
                ErrorCode::ResMemory
            );
            let mut refused = None;
            let tiny =
                allocation_counter::measure(|| refused = Some(GroupedWorkMemory::new(&input, 0)));
            assert_eq!(tiny.count_total, 0);
            assert_eq!(refused.unwrap().unwrap_err().code(), ErrorCode::ResMemory);
            // Neither a component nor an unlimited allowance bypasses either production guard.
            assert!(matches!(
                leaf_messages(&input, 0),
                Err(ProverError::Usage { .. })
            ));
            cases += 1;
        }
    }
    assert_eq!(cases, 9);
}
