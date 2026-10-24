// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The prover's grouped leaf stage against the independent verifier (DEP-2): `tv-core` derives
//! the groups from its own lowering, replays the stage on its own transcript with its own forest
//! roots and tables, and must accept it, its merged claims the prover's own.

use super::*;
use tessera_prover::leaf::{
    MergeMode, ModelWitness, ProvenGroupedLeaf, grouped_multiplicities, prove_grouped_leaf,
};
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

/// The prover's grouped stage as the verifier decodes its `TreeProof`; no atom split.
fn tree_proof(proven: &ProvenGroupedLeaf) -> tv_core::protocol::TreeProof {
    tv_core::protocol::TreeProof {
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
    }
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
        let trees_proof = tree_proof(&proven);
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

        // RFC-0054 'Testing Strategy', 'The forests', at one token bit: group 1 holds the odd
        // queries. The forest's trees are each query row's keys, 2^vars(n) leaves; a real row's
        // tree already holds a zero, so its public root is 0, and a padded row's holds ones.
        if (g_t, g_l) == (1, 0) {
            let site = actual.forest_sites().next().unwrap();
            let honest = forest_truth(&p, &witness, site);
            let height = vars(n);
            let query = |x: usize| (x >> height) & ((1 << height) - 1);
            let tree = |x: usize| (x >> height) << height..((x >> height) + 1) << height;
            let forests = actual.stage_forests();
            let prove_over = |leaves: &[i64]| {
                let leaves: Vec<G> = leaves.iter().map(|&v| G::from_i64(v)).collect();
                let mut cheating = actual.stage_forests();
                cheating[0] = forests[0].with_leaves(&leaves);
                let mut tr = transcript(Mode::NonInteractive);
                prove_grouped_leaf(
                    &mut tr,
                    &layout,
                    &grouped,
                    &tables,
                    &tensors,
                    &committed,
                    &cheating,
                    MergeMode::TwoPass { j_star: None },
                )
                .unwrap()
            };
            let check = |proven: &ProvenGroupedLeaf| {
                let mut t = vt(Mode::NonInteractive);
                stage
                    .verify(
                        &mut t,
                        Proof {
                            logup: &logup,
                            trees: &tree_proof(proven),
                        },
                        |tree, point| schedule.forest_root(tree, point),
                        |id, beta, point| vt_tables[index(id)?].eval_mle(point, beta).ok(),
                        |id| {
                            let table = &vt_tables[index(id)?];
                            let (k, v) = table.row(0)?;
                            let range =
                                matches!(table.spec().kind, tv_tables::TableKind::Range { .. });
                            Some((k, (!range).then_some(v)))
                        },
                    )
                    .map_err(|e| Box::new(e.verdict(tv_core::verdict::ArtifactKind(0x0520))))
            };
            // The control: the forest rebuilt from the witness is the prover's own.
            let control = check(&prove_over(&honest));
            assert!(control.is_ok(), "{control:?}");

            // A second zero in a real row's tree, at a valid key: its root stays 0, so the leaf
            // stage cannot see it. It carries the cheating leaves to the merged forest claim,
            // the false claim stage 1 rejects as it rejects the ungrouped stage's
            // (tv-core driver_grouped: forest claims moved off their leaves fail stage 1).
            let at = (0..honest.len())
                .find(|&x| {
                    query(x) % 2 == 1
                        && (query(x) as u32) < n
                        && honest[x] != 0
                        && honest[tree(x)].contains(&0)
                })
                .expect("a nonzero leaf in an odd real row's tree");
            let mut second = honest.clone();
            second[at] = 0;
            let merged = check(&prove_over(&second)).expect("the leaf stage cannot see it");
            let claim = &merged.forests[0].claim;
            assert_eq!(claim.value, mle(&second, &claim.point));
            assert_ne!(claim.value, mle(&honest, &claim.point));

            // A zero in a padded row's tree, all ones: its root becomes 0, not the public 1.
            // The forest joins group 1's levels at L − height, its root claim from the public
            // roots. Each round sends p(0) and derives p(1) from the claim (RFC-0010 'Sumcheck
            // messages'), so the false root claim fails that level's final check, at round
            // L − height: the tallest tree's variables there.
            let at = (0..honest.len())
                .find(|&x| query(x) % 2 == 1 && (query(x) as u32) >= n)
                .expect("an odd padded row");
            assert!(
                honest[tree(at)].iter().all(|&v| v == 1),
                "a padded tree holds ones"
            );
            let mut padded = honest.clone();
            padded[at] = 0;
            let joins = proven.groups()[1].levels().len() as u32 - height;
            let refused = check(&prove_over(&padded)).map(drop).map_err(|v| *v);
            assert!(
                matches!(
                    refused,
                    Err(tv_core::Verdict::Reject(tv_core::RejectReason::TreeLevel {
                        group: 1,
                        level,
                        round,
                    })) if (level, round) == (joins, joins)
                ),
                "{refused:?}"
            );
        }
    }
}
