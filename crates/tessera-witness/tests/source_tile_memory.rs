// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Allocator requests of complete resident/cached source tiles (RFC-0018/0050).
mod common;

use allocation_counter::measure;
use tessera_codec::{Ascii, Secret};
use tessera_compiler::{ModelAdviceLayout, PublicModel, RuntimePlan, Tile, claim_list};
use tessera_ir::ParamSetId;
use tessera_witness::{
    CheckpointTensors, Checkpoints, Engine, Resident, WitnessError, WitnessSource,
};

#[test]
fn every_model_tile_and_table_recount_fits_public_work_and_count_backing() {
    for fixture in common::conformance() {
        for (n, q) in [(1u32, 1), (3, 2), (17, 4)] {
            let (lowering, selection) = common::lowered(&fixture.model, n);
            let public = PublicModel::of(&fixture.model);
            let policy = common::policy(q);
            let claims = claim_list(
                public,
                &selection,
                &policy,
                common::TEST_SMALL,
                ParamSetId(Ascii::new("test-small/1").unwrap()),
            )
            .unwrap();
            let runtime = RuntimePlan::derive(&lowering, &selection, &claims, n, 0).unwrap();
            let advice = ModelAdviceLayout::runtime(&lowering, &runtime, &claims).unwrap();
            let engine = Engine::new(&lowering)
                .unwrap()
                .threads(std::num::NonZeroUsize::MIN);
            let mut work = None;
            let planning =
                measure(|| work = Some(engine.runtime_source_tile_work_memory(&runtime).unwrap()));
            assert_eq!(planning.count_total, 0);
            let work = work.unwrap();
            assert_eq!(work, engine.source_tile_work_memory().unwrap());
            let tokens = Secret::new(vec![0; n as usize]);
            let first = engine.run(&fixture.model, &tokens).unwrap();
            let checkpoints = Checkpoints::new(&engine, &fixture.model, &tokens).unwrap();
            let cached =
                CheckpointTensors::new(&engine, &fixture.model, &tokens, &checkpoints).unwrap();
            let resident = Resident::runtime(&engine, &runtime, &fixture.model, &first).unwrap();
            let borrowed =
                Resident::runtime_from_tensors(&engine, &runtime, &fixture.model, &cached).unwrap();
            let counts = engine
                .runtime_multiplicity_plans(&fixture.model.arch, &runtime)
                .unwrap();
            let mut tables = 0;
            for tile in advice.tiles() {
                let count_bytes = if let Tile::Table { table } = tile.tile {
                    tables += 1;
                    counts
                        .iter()
                        .find(|plan| plan.table().0 == table)
                        .unwrap()
                        .payload_bytes()
                } else {
                    0
                };
                let mut expected = None;
                let requests = measure(|| expected = Some(resident.tile(tile).unwrap()));
                assert!(
                    requests.bytes_total <= work + count_bytes,
                    "{} n={n} {:?} {requests:?} work={work} counts={count_bytes}",
                    fixture.name,
                    tile.tile
                );
                let mut actual = None;
                let requests = measure(|| actual = Some(borrowed.tile(tile).unwrap()));
                assert!(
                    requests.bytes_total <= work + count_bytes,
                    "cached {} n={n} {:?} {requests:?} work={work} counts={count_bytes}",
                    fixture.name,
                    tile.tile
                );
                let expected = expected.unwrap();
                let actual = actual.unwrap();
                assert_eq!(expected.bytes(), actual.bytes());
                if let Tile::Table { .. } = tile.tile {
                    assert_eq!(actual.bytes() as u64, count_bytes);
                }
            }
            assert_eq!(tables, counts.len());
        }
    }
}

#[test]
fn tile_work_planning_requires_the_lowering_and_checks_tensor_geometry() {
    let fixture = &common::conformance()[0];
    let (mut lowering, _) = common::lowered(&fixture.model, 1);
    let engine = Engine::for_graph(&lowering.graph, &lowering.tables).unwrap();
    assert_eq!(
        engine.source_tile_work_memory(),
        Err(WitnessError::Usage {
            rule: "a model runs on the engine of its lowering",
        })
    );
    lowering.tokens = tessera_ir::TensorId(u32::MAX);
    let engine = Engine::new(&lowering).unwrap();
    let planning = measure(|| {
        assert_eq!(
            engine.source_tile_work_memory(),
            Err(WitnessError::Memory {
                tensor: lowering.tokens
            })
        )
    });
    assert_eq!(planning.count_total, 0);
}
