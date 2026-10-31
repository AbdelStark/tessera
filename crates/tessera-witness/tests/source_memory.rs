// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained source metadata measured independently of the accounting (RFC-0018, #641).
mod common;

use allocation_counter::measure;
use tessera_codec::{Ascii, Secret};
use tessera_compiler::{PublicModel, RuntimePlan, analyze, claim_list, lower_batch, select};
use tessera_ir::ParamSetId;
use tessera_witness::{Checkpoints, Engine, Regenerator, Resident};

fn check<T>(make: impl FnOnce() -> T, bytes: impl Fn(&T) -> Option<usize>) {
    let mut value = None;
    let construction = measure(|| value = Some(make()));
    let value = value.unwrap();
    assert!(construction.bytes_current > 0);
    assert_eq!(
        usize::try_from(construction.bytes_current).unwrap(),
        bytes(&value).unwrap(),
        "every retained allocation, including spare capacity, is charged"
    );
    let querying = measure(|| assert!(bytes(&value).is_some()));
    assert_eq!(querying.count_total, 0);
}

#[test]
fn resident_and_replay_metadata_count_all_gadget_templates_and_group_capacity() {
    for fixture in common::conformance() {
        for n in [1, 3, 17] {
            let (lowering, selection) = common::lowered(&fixture.model, n);
            let engine = Engine::new(&lowering)
                .unwrap()
                .threads(std::num::NonZeroUsize::MIN);
            let tokens = Secret::new(vec![0; n as usize]);
            let witness = engine.run(&fixture.model, &tokens).unwrap();
            let checkpoints = Checkpoints::new(&engine, &fixture.model, &tokens).unwrap();
            for spare in [0, 17] {
                check(
                    || {
                        Resident::new(&engine, &selection, &fixture.model, &witness)
                            .unwrap()
                            .with_grouped_columns(Vec::with_capacity(spare))
                            .unwrap()
                    },
                    Resident::metadata_bytes,
                );
                check(
                    || {
                        Regenerator::new(&engine, &selection, &fixture.model, &tokens, &checkpoints)
                            .unwrap()
                            .with_grouped_columns(Vec::with_capacity(spare))
                            .unwrap()
                    },
                    Regenerator::metadata_bytes,
                );
            }
        }
    }
}

#[test]
fn runtime_metadata_counts_extra_reads_and_ragged_request_key_columns() {
    let fixture = &common::conformance()[0];
    let public = PublicModel::of(&fixture.model);
    for requests in [vec![(2, 1)], vec![(2, 1), (2, 2), (2, 0)]] {
        let lengths = requests
            .iter()
            .map(|(prompt, output)| prompt + output)
            .collect::<Vec<_>>();
        let lowering = lower_batch(public, &lengths).unwrap();
        let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
        let policy = common::policy(2);
        let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).unwrap();
        let params = ParamSetId(Ascii::new("test-small/1").unwrap());
        let claims = claim_list(public, &selection, &policy, common::TEST_SMALL, params).unwrap();
        let runtime = RuntimePlan::derive_batch(&lowering, &selection, &claims, &requests).unwrap();
        let engine = Engine::new(&lowering)
            .unwrap()
            .threads(std::num::NonZeroUsize::MIN);
        let concatenated = Secret::new(vec![0; lengths.iter().sum::<u32>() as usize]);
        let tokens = engine
            .canonical_batch_tokens(&fixture.model, &concatenated)
            .unwrap();
        let witness = engine.run(&fixture.model, &tokens).unwrap();
        let checkpoints = Checkpoints::new(&engine, &fixture.model, &tokens).unwrap();
        // Source construction validates public shapes. Exact greedy output is checked when
        // requesting its tile, outside this metadata-only allocation measurement.
        check(
            || Resident::runtime(&engine, &runtime, &fixture.model, &witness).unwrap(),
            Resident::metadata_bytes,
        );
        check(
            || {
                Regenerator::runtime(&engine, &runtime, &fixture.model, &tokens, &checkpoints)
                    .unwrap()
            },
            Regenerator::metadata_bytes,
        );
    }
}
