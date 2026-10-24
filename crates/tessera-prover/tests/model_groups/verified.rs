// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The prover's grouped leaf stage against the independent verifier (DEP-2): `tv-core` derives
//! the groups from its own lowering, replays the stage on its own transcript with its own forest
//! roots and tables, and must accept it, its merged claims the prover's own.

use super::*;
use tessera_prover::leaf::{MergeMode, ModelWitness, grouped_multiplicities, prove_grouped_leaf};
use tv_core::leaf::grouped::{Proof, Stage, TableTree};

/// The prover's levels, as the verifier decodes them.
fn levels(
    levels: &[tessera_sumcheck::ProvenTreeLevel<F>],
) -> Vec<tv_core::sumcheck::SumcheckProof> {
    levels
        .iter()
        .map(|l| tv_core::sumcheck::SumcheckProof {
            rounds: l.proof().rounds.iter().map(|r| es(r)).collect(),
            final_evals: es(&l.proof().final_evals),
        })
        .collect()
}

#[test]
fn the_independent_verifier_accepts_a_grouped_stage_at_the_provers_merged_claims() {
    let model = layered(bounded_model());
    let n = 5;
    let (p, layout) = stage(&model, n);
    let engine = Engine::new(&p.lowering).unwrap();
    let input: Vec<_> = (0..n).map(|i| (i + 1) % model.arch.vocab).collect();
    let witness = engine.run(&model, &Secret::new(input)).unwrap();
    let source = Resident::new(&engine, &p.selection, &model, &witness).unwrap();
    let advice = ModelAdviceLayout::new(&p.lowering, &p.selection, &p.claims).unwrap();
    let actual =
        ModelWitness::new(&layout, &p.lowering, &p.layers, &witness, &source, &advice).unwrap();
    let spec = |id| p.selection.tables.iter().find(|t| t.id == id).unwrap();
    let tables: Vec<_> = layout
        .instances()
        .iter()
        .map(|i| {
            TableColumns::<G>::new(&GeneratedTable::from_spec(spec(i.table)).unwrap()).unwrap()
        })
        .collect();
    let vt_tables: Vec<_> = layout
        .instances()
        .iter()
        .map(|i| {
            tv_tables::Table::from_spec(
                &tv_codec::decode(
                    &tessera_codec::encode(spec(i.table)),
                    &tv_codec::DecodeLimits::MESSAGE,
                    tv_tables::TableSpec::decode,
                )
                .unwrap(),
            )
            .unwrap()
        })
        .collect();
    let (graph, claims) = verifier(&p, n);
    let schedule = tv::ModelSchedule::derive(&graph, claims.gadget_policy).unwrap();
    let index = |id: tv_tables::TableId| layout.instances().iter().position(|i| i.table.0 == id.0);
    let trees: Vec<TableTree> = vt_tables
        .iter()
        .zip(layout.instances())
        .map(|(t, i)| TableTree {
            table: tv_tables::TableId(i.table.0),
            height: t.rows().next_power_of_two().trailing_zeros(),
        })
        .collect();
    let tensors = actual.tensors();
    for (g_t, g_l) in [(1, 0), (0, 1), (1, 1), (2, 1)] {
        let grouped = GroupedLayout::new(&layout, &p.layers, groups(g_t, g_l)).unwrap();
        let multiplicities = grouped_multiplicities(&layout, &grouped, &tables, &tensors).unwrap();
        let committed: Vec<&[G]> = multiplicities.iter().map(|m| &m[..]).collect();
        let mut tr = transcript(Mode::NonInteractive);
        let proven = prove_grouped_leaf(
            &mut tr,
            &layout,
            &grouped,
            &tables,
            &tensors,
            &committed,
            &actual.stage_forests(),
            // The backend's mode: the merge's table rounds, then one regeneration pass.
            MergeMode::TwoPass { j_star: None },
        )
        .unwrap();
        let trees_proof = tv_core::protocol::TreeProof {
            levels: levels(proven.tables().levels()),
            groups: proven
                .groups()
                .iter()
                .zip(proven.splits())
                .map(|(group, splits)| tv_core::protocol::GroupProof {
                    roots: es(group.roots()),
                    levels: levels(group.levels()),
                    values: es(group.values()),
                    splits: es(splits),
                })
                .collect(),
            table_roots: es(proven.tables().roots()),
            merge: tv_core::sumcheck::SumcheckProof {
                rounds: proven
                    .merge()
                    .proof()
                    .rounds
                    .iter()
                    .map(|r| es(r))
                    .collect(),
                final_evals: es(&proven.merge().proof().final_evals),
            },
            splits: Vec::new(),
        };
        let logup = tv_core::protocol::LogUpProof {
            roots: Vec::new(),
            values: Vec::new(),
        };
        let theirs = tv_core::leaf::groups::Grouped::derive(
            &graph,
            claims.gadget_policy,
            tv_core::protocol::LeafGroups {
                token_bits: g_t,
                layer_bits: g_l,
            },
        )
        .unwrap();
        let stage = Stage::new(&theirs, &trees).unwrap();
        let mut t2 = vt(Mode::NonInteractive);
        let merged = stage
            .verify(
                &mut t2,
                Proof {
                    logup: &logup,
                    trees: &trees_proof,
                },
                |tree, point| schedule.forest_root(tree, point),
                |id, beta, point| vt_tables[index(id)?].eval_mle(point, beta).ok(),
                |id| {
                    let table = &vt_tables[index(id)?];
                    let (k, v) = table.row(0)?;
                    let range = matches!(table.spec().kind, tv_tables::TableKind::Range { .. });
                    Some((k, (!range).then_some(v)))
                },
            )
            .unwrap_or_else(|e| panic!("{g_t} and {g_l} bits: {e}"));
        // The merged claims are the prover's: the tensors', the forests', the multiplicities'.
        let tensor_count = grouped.polynomials().len();
        assert_eq!(merged.columns.len(), tensor_count);
        for (theirs, ours) in merged.columns.iter().zip(proven.merged()) {
            assert_eq!(
                (&theirs.point, theirs.value),
                (&es(&ours.point), e(ours.value))
            );
        }
        assert_eq!(merged.forests.len(), proven.merged().len() - tensor_count);
        assert!(!merged.forests.is_empty(), "the stage has forests");
        for (theirs, ours) in merged.forests.iter().zip(&proven.merged()[tensor_count..]) {
            assert_eq!(
                (&theirs.claim.point, theirs.claim.value),
                (&es(&ours.point), e(ours.value))
            );
        }
        for ((_, theirs), ours) in merged
            .multiplicities
            .iter()
            .zip(proven.tables().multiplicities())
        {
            assert_eq!(
                (&theirs.point, theirs.value),
                (&es(&ours.point), e(ours.value))
            );
        }
        // A changed merge round is rejected.
        let mut tampered = trees_proof.clone();
        tampered.merge.rounds[0][0] += tv_field::Fp2::ONE;
        let mut t3 = vt(Mode::NonInteractive);
        let refused = stage.verify(
            &mut t3,
            Proof {
                logup: &logup,
                trees: &tampered,
            },
            |tree, point| schedule.forest_root(tree, point),
            |id, beta, point| vt_tables[index(id)?].eval_mle(point, beta).ok(),
            |id| {
                let table = &vt_tables[index(id)?];
                let (k, v) = table.row(0)?;
                let range = matches!(table.spec().kind, tv_tables::TableKind::Range { .. });
                Some((k, (!range).then_some(v)))
            },
        );
        assert!(
            refused.is_err(),
            "{g_t} and {g_l} bits: a changed merge round"
        );
        // Both transcripts end in one state: the next challenge agrees.
        let ours: F = tr.challenge("stage/0/next");
        let theirs = t2
            .challenge_fp2(tv_hash::transcript::Ascii::new("stage/0/next").unwrap())
            .unwrap();
        assert_eq!(e(ours), theirs, "{g_t} and {g_l} bits");
    }
}
