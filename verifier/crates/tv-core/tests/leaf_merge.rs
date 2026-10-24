// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A grouped leaf stage's checks through the merge (RFC-0054), against an honest prover written
//! from the RFCs (`merge_support`): acceptance on synthetic and model groupings with the merged
//! claims equal to the polynomials' extensions; the transcript replayed label by label from
//! RFC-0014's rows 4 and 5; and, for every check, a proof that fails exactly it, with its exact
//! reason.
mod claim_support;
mod group_support;
mod merge_support;

use group_support::{bits, model};
use merge_support::{
    Proved, Table, Tamper, Witness, cover, extension, forest_roots, from_fe, prove, recount,
    row_zero, table_eval, table_trees, to_fe,
};

use tv_core::advice::Auxiliary;
use tv_core::leaf::grouped::{Error, Merged, Message, Proof, Stage, TableTree};
use tv_core::leaf::groups::{Extent, Grouped, Source};
use tv_core::leaf::split::{Affine, Column, Layer, Lookup, Read, Site, TokenAxis};
use tv_core::leaf::{ForestShape, ModelSchedule};
use tv_core::protocol::{LeafGroups, TreeProof};
use tv_core::sumcheck::SumcheckProof;
use tv_core::verdict::{ArtifactKind, ErrorCode, RejectReason, Verdict};
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_tables::TableId;
use tv_testkit::SplitMix64;
use tv_testkit::sumcheck::{Fe, Transcript as Reference, tce};

const NI: ArtifactKind = ArtifactKind(0x0520);

// ----- A synthetic stage ------------------------------------------------------------------------

fn groups(token_bits: u8, layer_bits: u8) -> LeafGroups {
    LeafGroups {
        token_bits,
        layer_bits,
    }
}
fn affine(constant: u64, terms: &[(u64, Column, &[u32])]) -> Affine {
    Affine {
        constant: Fp::reduce(constant),
        terms: terms
            .iter()
            .map(|&(a, column, coordinates)| {
                (
                    Fp::reduce(a),
                    Read {
                        column,
                        coordinates: coordinates.to_vec(),
                    },
                )
            })
            .collect(),
    }
}
fn lookup(
    table: u32,
    variables: u32,
    token: Option<(u32, u32)>,
    key: Affine,
    value: Option<Affine>,
) -> Lookup {
    Lookup {
        table: TableId(table),
        variables,
        key,
        value,
        token: token.map(|(low, bits)| TokenAxis { low, bits }),
    }
}
fn site(nodes: &[u32], lookups: impl Fn() -> Vec<Lookup>) -> Site {
    Site {
        layers: nodes
            .iter()
            .map(|&node| Layer {
                node: tv_core::lowering::NodeId(node),
                lookups: lookups(),
            })
            .collect(),
    }
}

/// Four tokens (`t = 2`) and three real layers (`n_ℓ = 2`): a prologue site, node 5, with a
/// tokenless Range lookup beside a tokened one that shares its tensor `A`; a template site of
/// three layers, nodes 10, 20 and 30, with a `[4, 8]` lookup keyed `X + 16·K` (`K` a channel
/// class, read without the token axis) and valued `Y`, a `[4]` lookup keyed `a` and valued `b`,
/// and a Range lookup keyed `u + 2·g` over `[4, 8]`; and an epilogue site, node 40, over
/// `[2, 4, 8]` with its token axis in the middle.
fn sites() -> Vec<Site> {
    let full: &[u32] = &[0, 1, 2, 3, 4];
    vec![
        site(&[5], || {
            vec![
                lookup(
                    2,
                    3,
                    None,
                    affine(
                        0,
                        &[
                            (1, Column::Input(0), &[0, 1, 2]),
                            (2, Column::Input(1), &[0, 1, 2]),
                        ],
                    ),
                    None,
                ),
                lookup(
                    5,
                    5,
                    Some((3, 2)),
                    affine(
                        7,
                        &[
                            (1, Column::Output(0), full),
                            (4, Column::Input(0), &[0, 1, 2]),
                        ],
                    ),
                    Some(affine(0, &[(1, Column::Output(1), full)])),
                ),
            ]
        }),
        site(&[10, 20, 30], || {
            vec![
                lookup(
                    1,
                    5,
                    Some((3, 2)),
                    affine(
                        0,
                        &[
                            (1, Column::Input(0), full),
                            (16, Column::Input(1), &[0, 1, 2]),
                        ],
                    ),
                    Some(affine(0, &[(1, Column::Output(0), full)])),
                ),
                lookup(
                    1,
                    2,
                    Some((0, 2)),
                    affine(0, &[(1, Column::SoftmaxA, &[0, 1])]),
                    Some(affine(0, &[(1, Column::SoftmaxB, &[0, 1])])),
                ),
                lookup(
                    3,
                    5,
                    Some((3, 2)),
                    affine(
                        0,
                        &[(1, Column::RootInput, full), (2, Column::GreedyKey, full)],
                    ),
                    None,
                ),
            ]
        }),
        site(&[40], || {
            vec![lookup(
                0,
                6,
                Some((3, 2)),
                affine(
                    0,
                    &[(
                        1,
                        Column::Auxiliary(Auxiliary::RequantLow),
                        &[0, 1, 2, 3, 4, 5],
                    )],
                ),
                None,
            )]
        }),
    ]
}
/// A forest stacked over the three layers, two heads and four queries, of height 2; and a
/// forest outside the template, of one head and three queries, of height 1.
fn forests() -> Vec<ForestShape> {
    vec![
        ForestShape {
            tree: 100,
            height: 2,
            queries: 4,
            heads: 2,
            layers: 3,
        },
        ForestShape {
            tree: 101,
            height: 1,
            queries: 3,
            heads: 1,
            layers: 1,
        },
    ]
}
const EXTENT: Extent = Extent {
    token_bits: 2,
    layers: 3,
};
const GROUPINGS: [(u8, u8); 6] = [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)];

fn synthetic(g_t: u8, g_l: u8) -> Grouped {
    Grouped::new(&sites(), &forests(), EXTENT, groups(g_t, g_l)).unwrap()
}

fn session() -> SessionContext<'static> {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x54; 32],
        session_id: [0x45; 16],
    }
}
fn reference() -> Reference {
    Reference::new(&session().to_tce())
}

/// A stage's honest proof and everything the verifier's callbacks read.
struct Case {
    grouped: Grouped,
    witness: Witness,
    tables: Vec<Table>,
    proved: Proved,
    /// The prover's transcript after its last message.
    prover: Reference,
}

impl Case {
    fn new(grouped: Grouped, witness: Witness) -> Self {
        let tables = cover(&grouped, &witness);
        Self::with(grouped, witness, tables, |_| {})
    }
    fn with(
        grouped: Grouped,
        witness: Witness,
        tables: Vec<Table>,
        tamper: impl FnMut(Tamper<'_>),
    ) -> Self {
        let mut prover = reference();
        let proved = prove(&mut prover, &grouped, &witness, &tables, tamper);
        Self {
            grouped,
            witness,
            tables,
            proved,
            prover,
        }
    }
    fn stage(&self) -> Stage<'_> {
        Stage::new(&self.grouped, &table_trees(&self.tables)).unwrap()
    }
    /// The forests' public roots, from the witness's leaves.
    fn roots(&self) -> Vec<(usize, Vec<Fp2>)> {
        self.grouped
            .polynomials()
            .iter()
            .enumerate()
            .filter_map(|(i, p)| match p.source {
                Source::Forest { forest, tree } => Some((
                    tree,
                    forest_roots(&self.witness, i, forests()[forest].height),
                )),
                Source::Split { .. } => None,
            })
            .collect()
    }
    fn verify_with(&self, trees: &TreeProof, transcript: &mut Transcript) -> Result<Merged, Error> {
        let roots = self.roots();
        self.stage().verify(
            transcript,
            Proof {
                logup: &self.proved.logup,
                trees,
            },
            |tree, point| {
                let (_, table) = roots.iter().find(|(t, _)| *t == tree)?;
                (table.len() == 1 << point.len()).then(|| extension(table, point))
            },
            |id, beta, point| table_eval(&self.tables, id, beta, point),
            |id| row_zero(&self.tables, id),
        )
    }
    fn verify(&self, trees: &TreeProof) -> Result<Merged, Error> {
        self.verify_with(trees, &mut Transcript::new(&session()))
    }
    fn reason(&self, trees: &TreeProof) -> RejectReason {
        let error = self.verify(trees).expect_err("a rejection");
        match error.verdict(NI) {
            Verdict::Reject(reason) => {
                assert_eq!(error.rejection(NI), Some(reason.clone()));
                reason
            }
            other => panic!("{error} gives {other:?}, not a rejection"),
        }
    }
    fn honest(&self) -> TreeProof {
        self.proved.trees.clone()
    }
}

/// The merged claims are the polynomials' extensions at their points, the multiplicities' too,
/// and polynomials of one size and one `S` lie at one point (RFC-0054 'Variable order').
fn check_merged(case: &Case, merged: &Merged) {
    let polynomials = case.grouped.polynomials();
    assert_eq!(merged.point, case.proved.merge_point);
    let rounds = polynomials.iter().map(|p| p.variables).max().unwrap();
    assert_eq!(merged.point.len(), rounds as usize);
    assert_eq!(
        merged.columns.len() + merged.forests.len(),
        polynomials.len()
    );
    let mut points = Vec::new();
    for (i, polynomial) in polynomials.iter().enumerate() {
        let (point, value) = match polynomial.source {
            Source::Split { node, column } => {
                let claim = &merged.columns[i];
                assert_eq!((claim.node, claim.column), (node, column));
                (&claim.point, claim.value)
            }
            Source::Forest { tree, .. } => {
                let claim = &merged.forests[i - case.grouped.values()];
                assert_eq!(claim.tree, tree);
                (&claim.claim.point, claim.claim.value)
            }
        };
        assert_eq!(point.len(), polynomial.variables as usize);
        assert_eq!(
            extension(&case.witness.fp2(i), point),
            value,
            "polynomial {i}"
        );
        // Its own variables take the last `n_i` challenges: outside `S` first, then `S`.
        let own = &merged.point[merged.point.len() - point.len()..];
        let outside = (0..polynomial.variables).filter(|v| !polynomial.group_variables.contains(v));
        let order: Vec<u32> = outside
            .chain(polynomial.group_variables.iter().copied())
            .collect();
        for (j, &v) in order.iter().enumerate() {
            assert_eq!(point[v as usize], own[j]);
        }
        points.push((
            polynomial.variables,
            polynomial.group_variables.clone(),
            point.clone(),
        ));
    }
    for a in &points {
        for b in &points {
            if (a.0, &a.1) == (b.0, &b.1) {
                assert_eq!(a.2, b.2);
            }
        }
    }
    assert_eq!(merged.multiplicities.len(), case.tables.len());
    for ((id, claim), table) in merged.multiplicities.iter().zip(&case.tables) {
        assert_eq!(*id, table.id);
        let m: Vec<Fp2> = table
            .multiplicities
            .iter()
            .map(|&m| Fp2::from(Fp::reduce(m)))
            .collect();
        assert_eq!(extension(&m, &claim.point), claim.value);
    }
}

// ----- Acceptance -------------------------------------------------------------------------------

#[test]
fn honest_grouped_stages_are_accepted_and_merged_claims_are_the_polynomials_extensions() {
    let mut rng = SplitMix64::new(0x0054_4d52);
    for (g_t, g_l) in GROUPINGS {
        for _ in 0..2 {
            let grouped = synthetic(g_t, g_l);
            let witness = Witness::random(&grouped, &mut rng, 4);
            let case = Case::new(grouped, witness);
            let mut transcript = Transcript::new(&session());
            let merged = case
                .verify_with(&case.honest(), &mut transcript)
                .unwrap_or_else(|e| panic!("g = ({g_t}, {g_l}): {e}"));
            check_merged(&case, &merged);
            // The verifier read exactly what the prover wrote.
            assert_eq!(*transcript.state(), case.prover.state());
            assert_eq!(transcript.counter(), case.prover.counter());
        }
    }
}

#[test]
fn stage_metadata_it_cannot_lay_out_is_a_fault_of_the_verifier() {
    let grouped = synthetic(1, 1);
    let witness = Witness::random(&grouped, &mut SplitMix64::new(1), 4);
    let trees = table_trees(&cover(&grouped, &witness));
    assert!(Stage::new(&grouped, &trees).is_ok());
    let one = Grouped::new(&sites(), &forests(), EXTENT, LeafGroups::NONE).unwrap();
    assert_eq!(Stage::new(&one, &trees).err(), Some(Error::Internal));
    let mut reversed = trees.clone();
    reversed.swap(0, 1);
    let mut missing = trees.clone();
    missing.remove(2);
    let mut extra = trees.clone();
    extra.push(TableTree {
        table: TableId(9),
        height: 1,
    });
    let mut tall = trees.clone();
    tall[0].height = 64;
    for bad in [reversed, missing, extra, tall] {
        assert_eq!(Stage::new(&grouped, &bad).err(), Some(Error::Internal));
    }
}

/// A model's grouping, with the verifier's own Softmax roots (`ModelSchedule::forest_root`):
/// each real query row's tree holds a zero leaf, each padded tree ones alone.
#[test]
fn a_model_grouping_is_accepted_with_the_verifiers_own_forest_roots() {
    let mut rng = SplitMix64::new(0x0054_0001);
    for (name, layers, n, (g_t, g_l)) in [
        ("tiny-2-servable", Some(3), 5, (1, 1)),
        ("tiny-2-minimal", None, 4, (2, 0)),
    ] {
        let (graph, policy) = model(name, layers, n);
        let grouped = Grouped::derive(&graph, policy, groups(g_t, g_l)).unwrap();
        let schedule = ModelSchedule::derive(&graph, policy).unwrap();
        let shapes: Vec<ForestShape> = schedule.forests().collect();
        let mut witness = Witness::random(&grouped, &mut rng, 4);
        for (i, polynomial) in grouped.polynomials().iter().enumerate() {
            let Source::Forest { forest, .. } = polynomial.source else {
                continue;
            };
            let shape = shapes[forest];
            let (q, hd) = (bits(shape.queries), bits(shape.heads));
            let leaves = 1usize << shape.height;
            for (y, tree) in witness.tables[i].chunks_mut(leaves).enumerate() {
                let query = y % (1 << q);
                let head = (y >> q) % (1 << hd);
                let layer = y >> (q + hd);
                let real = query < shape.queries as usize
                    && head < shape.heads as usize
                    && layer < shape.layers as usize;
                for (x, leaf) in tree.iter_mut().enumerate() {
                    *leaf = if !real {
                        1
                    } else if x == 0 {
                        0
                    } else {
                        1 + rng.next_u64() % 1000
                    };
                }
            }
        }
        let tables = cover(&grouped, &witness);
        let mut prover = reference();
        let proved = prove(&mut prover, &grouped, &witness, &tables, |_| {});
        let stage = Stage::new(&grouped, &table_trees(&tables)).unwrap();
        let mut transcript = Transcript::new(&session());
        let merged = stage
            .verify(
                &mut transcript,
                Proof {
                    logup: &proved.logup,
                    trees: &proved.trees,
                },
                |tree, point| schedule.forest_root(tree, point),
                |id, beta, point| table_eval(&tables, id, beta, point),
                |id| row_zero(&tables, id),
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!merged.forests.is_empty());
        let case = Case {
            grouped,
            witness,
            tables,
            proved,
            prover,
        };
        check_merged(&case, &merged);
        assert_eq!(*transcript.state(), case.prover.state());
    }
}

// ----- The transcript ---------------------------------------------------------------------------

/// RFC-0014 'Transcript schedule', rows 4 and 5 with groups, replayed label by label on a fresh
/// transcript from the proof's messages and the public layout alone: the verifier ends in the
/// same state, so it absorbed and squeezed exactly these labels in this order.
#[test]
fn the_transcript_follows_rfc_0014_rows_4_and_5_with_groups() {
    let mut rng = SplitMix64::new(0x0054_0014);
    for (g_t, g_l) in [(1, 1), (2, 2)] {
        let grouped = synthetic(g_t, g_l);
        let witness = Witness::random(&grouped, &mut rng, 4);
        let case = Case::new(grouped, witness);
        let trees = case.honest();
        let fe = |values: &[Fp2]| -> Vec<Fe> { values.iter().copied().map(to_fe).collect() };
        let mut replay = reference();
        let mut labels: Vec<String> = Vec::new();
        let mut absorb = |t: &mut Reference, label: String, values: &[Fp2]| {
            t.absorb(&label, &tce(&fe(values)));
            labels.push(format!("+{label}"));
        };
        // A level's sumcheck (RFC-0010 'Sumcheck messages'), after its γ and λ.
        let sumcheck = |t: &mut Reference,
                        absorb: &mut dyn FnMut(&mut Reference, String, &[Fp2]),
                        prefix: &str,
                        proof: &SumcheckProof| {
            for (j, round) in proof.rounds.iter().enumerate() {
                absorb(t, format!("{prefix}/round/{j}"), round);
                t.challenge_fp2(&format!("{prefix}/r/{j}"));
            }
            absorb(t, format!("{prefix}/evals"), &proof.final_evals);
        };
        // Row 4.
        replay.challenge_fp2("logup/alpha");
        replay.challenge_fp2("logup/beta");
        absorb(&mut replay, "logup/roots".into(), &[]);
        absorb(&mut replay, "logup/values".into(), &[]);
        // Row 5, per group.
        for (h, sent) in (0..case.grouped.count()).zip(&trees.groups) {
            let group = case.grouped.group(h).unwrap();
            let height = group.levels();
            absorb(&mut replay, format!("group/{h}/logup/roots"), &sent.roots);
            for (level, proof) in (0..height).zip(&sent.levels) {
                let joining = group
                    .forests
                    .iter()
                    .filter(|s| height - s.height == level)
                    .map(|s| s.trees_log)
                    .max()
                    .unwrap_or(0);
                for c in 0..joining {
                    replay.challenge_fp2(&format!("group/{h}/tree/{level}/root/{c}"));
                }
                replay.challenge_fp2(&format!("group/{h}/tree/{level}/gamma"));
                if group
                    .instances
                    .iter()
                    .any(|i| i.height > 0 && height - i.height <= level)
                {
                    replay.challenge_fp2(&format!("group/{h}/tree/{level}/lambda"));
                }
                sumcheck(
                    &mut replay,
                    &mut absorb,
                    &format!("group/{h}/tree/{level}"),
                    proof,
                );
                replay.challenge_fp2(&format!("group/{h}/tree/{level}/line"));
            }
            absorb(&mut replay, format!("group/{h}/logup/values"), &sent.values);
            absorb(&mut replay, format!("group/{h}/split"), &sent.splits);
            replay.challenge_fp2(&format!("group/{h}/merge/gamma"));
        }
        // Row 5, then: the table trees, every level with fraction trees alone.
        absorb(&mut replay, "tables/logup/roots".into(), &trees.table_roots);
        for (level, proof) in trees.levels.iter().enumerate() {
            replay.challenge_fp2(&format!("tables/tree/{level}/gamma"));
            replay.challenge_fp2(&format!("tables/tree/{level}/lambda"));
            sumcheck(
                &mut replay,
                &mut absorb,
                &format!("tables/tree/{level}"),
                proof,
            );
            replay.challenge_fp2(&format!("tables/tree/{level}/line"));
        }
        // And the merge.
        replay.challenge_fp2("merge/beta");
        sumcheck(&mut replay, &mut absorb, "merge", &trees.merge);
        let mut transcript = Transcript::new(&session());
        let _ = case.verify_with(&trees, &mut transcript).unwrap();
        assert_eq!(*transcript.state(), replay.state(), "g = ({g_t}, {g_l})");
        assert_eq!(transcript.counter(), replay.counter());
        assert_eq!(replay.state(), case.prover.state());
        // The merge's messages: degree 2, `p(0), p(2)`, one round per variable of the largest
        // polynomial.
        let n_max = case
            .grouped
            .polynomials()
            .iter()
            .map(|p| p.variables)
            .max()
            .unwrap();
        assert_eq!(trees.merge.rounds.len(), n_max as usize);
        assert!(trees.merge.rounds.iter().all(|r| r.len() == 2));
        assert_eq!(labels[..2], ["+logup/roots", "+logup/values"]);
        assert_eq!(labels[2], "+group/0/logup/roots");
        assert_eq!(labels.last().unwrap(), "+merge/evals");
    }
}

// ----- One rejection per check ------------------------------------------------------------------

fn case(g_t: u8, g_l: u8, seed: u64) -> Case {
    let grouped = synthetic(g_t, g_l);
    let witness = Witness::random(&grouped, &mut SplitMix64::new(seed), 4);
    Case::new(grouped, witness)
}
fn tree_level(group: u32, level: u32, round: u32) -> RejectReason {
    RejectReason::TreeLevel {
        group,
        level,
        round,
    }
}
fn lookup_reason(table: TableId) -> RejectReason {
    RejectReason::Lookup { table }
}
fn malformed() -> RejectReason {
    RejectReason::Malformed {
        artifact: NI,
        code: ErrorCode::CodecNoncanonical,
    }
}

type Edit = Box<dyn Fn(&mut TreeProof)>;

/// Checks 1 to 4: every message whose length the grouping fixes, too short and too long, is
/// `Malformed` before any other check, even with a zero denominator in group 0.
#[test]
fn wrong_message_counts_are_malformed_before_any_other_check() {
    let case = case(1, 1, 11);
    let honest = case.honest();
    let groups = case.grouped.count();
    let mut edits: Vec<(Message, Edit)> = vec![
        (
            Message::Groups,
            Box::new(|t| {
                t.groups.pop();
            }),
        ),
        (
            Message::Groups,
            Box::new(|t| t.groups.push(t.groups[0].clone())),
        ),
        (
            Message::TableRoots,
            Box::new(|t| t.table_roots.push(Fp2::ONE)),
        ),
        (
            Message::TableRoots,
            Box::new(|t| {
                t.table_roots.pop();
            }),
        ),
        (
            Message::TableLevels,
            Box::new(|t| {
                t.levels.pop();
            }),
        ),
        (
            Message::TableLevels,
            Box::new(|t| t.levels.push(SumcheckProof::default())),
        ),
    ];
    for h in 0..groups {
        let g = h as usize;
        edits.push((
            Message::Roots { group: h },
            Box::new(move |t| t.groups[g].roots.push(Fp2::ONE)),
        ));
        edits.push((
            Message::Levels { group: h },
            Box::new(move |t| {
                t.groups[g].levels.pop();
            }),
        ));
        edits.push((
            Message::Values { group: h },
            Box::new(move |t| t.groups[g].values.push(Fp2::ONE)),
        ));
        edits.push((
            Message::Splits { group: h },
            Box::new(move |t| {
                t.groups[g].splits.pop();
            }),
        ));
    }
    for (message, edit) in &edits {
        let mut trees = honest.clone();
        // A zero denominator in group 0 too: the count is still reported first.
        trees.groups[0].roots[1] = Fp2::ZERO;
        edit(&mut trees);
        let error = case.verify(&trees).unwrap_err();
        assert!(
            matches!(error, Error::Count { message: m, .. } if m == *message),
            "{message:?}: {error}"
        );
        assert_eq!(error.rejection(NI), Some(malformed()));
    }
    // The ungrouped stage's own messages are empty in a grouped one.
    for roots in [true, false] {
        let mut logup = case.proved.logup.clone();
        if roots {
            logup.roots.push(Fp2::ONE);
        } else {
            logup.values.push(Fp2::ONE);
        }
        let error = case
            .stage()
            .verify(
                &mut Transcript::new(&session()),
                Proof {
                    logup: &logup,
                    trees: &honest,
                },
                |_, _| None,
                |_, _, _| None,
                |_| None,
            )
            .unwrap_err();
        let message = if roots {
            Message::LogUpRoots
        } else {
            Message::LogUpValues
        };
        assert_eq!(
            error,
            Error::Count {
                message,
                expected: 0,
                found: 1
            }
        );
    }
}

/// Check 5: a zero root denominator in any group, at its tree's table; the first group at fault
/// is reported.
#[test]
fn a_zero_denominator_in_a_group_is_rejected_at_its_table() {
    let case = case(1, 1, 12);
    for h in 0..case.grouped.count() {
        let group = case.grouped.group(h).unwrap();
        for (i, instance) in group.instances.iter().enumerate() {
            let mut trees = case.honest();
            trees.groups[h as usize].roots[2 * i] = Fp2::ZERO;
            trees.groups[h as usize].roots[2 * i + 1] = Fp2::ZERO;
            assert_eq!(case.reason(&trees), lookup_reason(instance.table));
            assert_eq!(
                case.verify(&trees).unwrap_err(),
                Error::Denominator {
                    group: h,
                    table: instance.table
                }
            );
        }
    }
    // Two groups at fault: the first is reported.
    let mut trees = case.honest();
    trees.groups[3].roots[1] = Fp2::ZERO;
    trees.groups[1].roots[1] = Fp2::ZERO;
    let first = case.grouped.group(1).unwrap().instances[0].table;
    assert_eq!(
        case.verify(&trees).unwrap_err(),
        Error::Denominator {
            group: 1,
            table: first
        }
    );
}

/// Check 6: a root the prover sends but its levels do not prove is rejected at the level its
/// tree joins, in its own group; and a group's root moved to another group, which keeps every
/// instance's identity, is rejected by that group's levels.
#[test]
fn a_root_its_group_does_not_prove_fails_that_groups_level() {
    let grouped = synthetic(1, 1);
    let witness = Witness::random(&grouped, &mut SplitMix64::new(13), 4);
    let tables = cover(&grouped, &witness);
    let target = 2u32;
    let group = grouped.group(target).unwrap();
    let height = group.levels();
    let (i, instance) = group
        .instances
        .iter()
        .enumerate()
        .find(|(_, i)| i.height > 0)
        .unwrap();
    let join = height - instance.height;
    let case = Case::with(grouped, witness.clone(), tables, |tamper| {
        if let Tamper::GroupRoots(h, roots) = tamper
            && h == target
        {
            roots[2 * i] = roots[2 * i] + Fe::ONE;
        }
    });
    let rounds = case.honest().groups[target as usize].levels[join as usize]
        .rounds
        .len() as u32;
    assert_eq!(
        case.reason(&case.honest()),
        tree_level(target, join, rounds)
    );
    // Groups 2 and 3, the token subsets of layer subset 1, hold the same instances; their roots
    // exchanged leave every identity's sum unchanged.
    let case = Case::new(synthetic(1, 1), witness);
    let mut trees = case.honest();
    let (two, three) = (trees.groups[2].roots.clone(), trees.groups[3].roots.clone());
    assert_eq!(two.len(), three.len());
    assert_ne!(two, three);
    trees.groups[2].roots = three;
    trees.groups[3].roots = two;
    let group = case.grouped.group(2).unwrap();
    let first = group
        .instances
        .iter()
        .filter(|i| i.height > 0)
        .map(|i| group.levels() - i.height)
        .min()
        .unwrap();
    let rounds = trees.groups[2].levels[first as usize].rounds.len() as u32;
    assert_eq!(case.reason(&trees), tree_level(2, first, rounds));
}

/// Check 6, and 12 for the table batch: one changed round value or evaluation of any level is
/// rejected at that level's final check, and a round of another length at that round; with
/// `group` the group's index, or `G` for the table batch.
#[test]
fn a_changed_level_message_is_rejected_at_its_group_and_level() {
    let case = case(2, 1, 14);
    let groups = case.grouped.count();
    let honest = case.honest();
    let mut seen = 0;
    for h in 0..=groups {
        let levels = if h < groups {
            honest.groups[h as usize].levels.clone()
        } else {
            honest.levels.clone()
        };
        for (level, proof) in levels.iter().enumerate() {
            let n = proof.rounds.len() as u32;
            let edit = |trees: &mut TreeProof, change: &dyn Fn(&mut SumcheckProof)| {
                let proofs = if h < groups {
                    &mut trees.groups[h as usize].levels
                } else {
                    &mut trees.levels
                };
                change(&mut proofs[level]);
            };
            for j in 0..proof.rounds.len() {
                let mut trees = honest.clone();
                edit(&mut trees, &|p| p.rounds[j][1] += Fp2::ONE);
                assert_eq!(case.reason(&trees), tree_level(h, level as u32, n));
                let mut trees = honest.clone();
                edit(&mut trees, &|p| p.rounds[j].push(Fp2::ONE));
                assert_eq!(case.reason(&trees), tree_level(h, level as u32, j as u32));
            }
            for k in 0..proof.final_evals.len() {
                let mut trees = honest.clone();
                edit(&mut trees, &|p| p.final_evals[k] += Fp2::ONE);
                assert_eq!(case.reason(&trees), tree_level(h, level as u32, n));
            }
            let mut trees = honest.clone();
            edit(&mut trees, &|p| p.rounds.push(vec![Fp2::ONE; 3]));
            assert_eq!(case.reason(&trees), tree_level(h, level as u32, n));
            let mut trees = honest.clone();
            edit(&mut trees, &|p| {
                p.final_evals.pop();
            });
            assert_eq!(case.reason(&trees), tree_level(h, level as u32, n));
            seen += 1;
        }
    }
    assert!(seen > groups as usize);
    assert!(!honest.levels.is_empty());
}

/// Check 7: a lookup tree whose leaves' numerators are not all 1, its levels proving them, is
/// rejected at its table.
#[test]
fn a_leaf_numerator_other_than_one_is_rejected_at_its_table() {
    let grouped = synthetic(1, 1);
    let witness = Witness::random(&grouped, &mut SplitMix64::new(15), 4);
    let tables = cover(&grouped, &witness);
    let target = 3u32;
    let table = grouped.group(target).unwrap().instances[1].table;
    let case = Case::with(grouped, witness, tables, |tamper| {
        if let Tamper::LookupLeaves(h, 1, leaves) = tamper
            && h == target
        {
            leaves[0].0 = Fe::base(2);
        }
    });
    let error = case.verify(&case.honest()).unwrap_err();
    assert_eq!(
        error,
        Error::LeafNumerator {
            group: target,
            table
        }
    );
    assert_eq!(error.rejection(NI), Some(lookup_reason(table)));
}

/// Check 8: a changed split value or value evaluation of any group is rejected by its group's
/// check, at its level count and round 0.
#[test]
fn a_changed_split_value_fails_its_groups_check() {
    let case = case(1, 2, 16);
    let honest = case.honest();
    let mut checked = 0;
    for h in 0..case.grouped.count() {
        let group = case.grouped.group(h).unwrap();
        let expected = tree_level(h, group.levels(), 0);
        for i in 0..group.split.len() {
            let mut trees = honest.clone();
            trees.groups[h as usize].splits[i] += Fp2::ONE;
            assert_eq!(case.reason(&trees), expected, "group {h}, value {i}");
            checked += 1;
        }
        for i in 0..honest.groups[h as usize].values.len() {
            let mut trees = honest.clone();
            trees.groups[h as usize].values[i] += Fp2::ONE;
            assert_eq!(case.reason(&trees), expected);
        }
    }
    assert!(checked > 0);
}

/// Checks 15 to 17: a split change that RFC-0045's identity cannot see, `u + 2·g` kept, passes its
/// group's check and is rejected by the merge's final check.
#[test]
fn a_split_change_its_groups_check_cannot_see_fails_the_merge() {
    let grouped = synthetic(1, 1);
    let witness = Witness::random(&grouped, &mut SplitMix64::new(17), 4);
    let tables = cover(&grouped, &witness);
    let target = 1u32;
    let split = grouped.group(target).unwrap().split;
    let u = split
        .iter()
        .position(|v| v.column == Column::RootInput)
        .unwrap();
    let g = split
        .iter()
        .position(|v| v.column == Column::GreedyKey && v.node == split[u].node)
        .unwrap();
    let case = Case::with(grouped, witness, tables, |tamper| {
        if let Tamper::Splits(h, splits) = tamper
            && h == target
        {
            splits[u] = splits[u] + Fe::base(2);
            splits[g] = splits[g] - Fe::ONE;
        }
    });
    let n_max = case.stage().rounds();
    assert_eq!(
        case.reason(&case.honest()),
        RejectReason::LeafMerge { round: n_max }
    );
}

/// Check 10 and 11: a table root of zero denominator, and a lookup that is no row, whose
/// group's levels, values and split are all consistent, so that only the identity sees it.
#[test]
fn a_zero_table_denominator_and_a_false_group_root_fail_the_table_checks() {
    let case = case(1, 1, 18);
    for (i, table) in case.tables.iter().enumerate() {
        let mut trees = case.honest();
        trees.table_roots[2 * i + 1] = Fp2::ZERO;
        assert_eq!(
            case.verify(&trees).unwrap_err(),
            Error::TableDenominator { table: table.id }
        );
        assert_eq!(case.reason(&trees), lookup_reason(table.id));
        // A table root changed in its numerator unbalances its identity first.
        let mut trees = case.honest();
        trees.table_roots[2 * i] += Fp2::ONE;
        assert_eq!(
            case.verify(&trees).unwrap_err(),
            Error::Unbalanced { table: table.id }
        );
    }
    // The epilogue's tensor changed at one entry, after the tables were fixed: group 1's lookup
    // of table 0 is no row, and its root is a false one.
    let grouped = synthetic(1, 1);
    let mut witness = Witness::random(&grouped, &mut SplitMix64::new(19), 4);
    let mut tables = cover(&grouped, &witness);
    let polynomial = grouped
        .polynomials()
        .iter()
        .position(|p| {
            matches!(
                p.source,
                Source::Split {
                    column: Column::Auxiliary(Auxiliary::RequantLow),
                    ..
                }
            )
        })
        .unwrap();
    // Entry 8 has token bit 0 set: token subset 1's.
    witness.tables[polynomial][8] += 1000;
    recount(&grouped, &witness, &mut tables);
    let case = Case::with(grouped, witness, tables, |_| {});
    assert_eq!(
        case.verify(&case.honest()).unwrap_err(),
        Error::Unbalanced { table: TableId(0) }
    );
    assert_eq!(case.reason(&case.honest()), lookup_reason(TableId(0)));
}

/// Check 13: a table batch proven over a table that is not the public one fails the table
/// tree's leaf check, at that table.
#[test]
fn a_table_other_than_the_public_one_fails_its_leaf_check() {
    let case = case(1, 1, 20);
    let target = case
        .tables
        .iter()
        .position(|t| t.rows.len() > 1 && t.rows[1] != t.rows[0])
        .unwrap();
    let mut public = case.tables.clone();
    public[target].rows[1].0 = public[target].rows[1].0 + Fe::base(1 << 40);
    let roots = case.roots();
    let error = case
        .stage()
        .verify(
            &mut Transcript::new(&session()),
            Proof {
                logup: &case.proved.logup,
                trees: &case.honest(),
            },
            |tree, point| {
                let (_, table) = roots.iter().find(|(t, _)| *t == tree)?;
                Some(extension(table, point))
            },
            |id, beta, point| table_eval(&public, id, beta, point),
            |id| row_zero(&public, id),
        )
        .unwrap_err();
    let table = case.tables[target].id;
    assert_eq!(error, Error::TableEvaluation { table });
    assert_eq!(error.rejection(NI), Some(lookup_reason(table)));
}

/// Checks 15 to 17: a changed merge round value or evaluation fails the final check, a round of
/// another length at its round, a missing round at the first missing, an extra round and a
/// wrong evaluation count at `n_max`.
#[test]
fn a_changed_merge_round_or_evaluation_fails_the_merge() {
    let case = case(2, 2, 21);
    let honest = case.honest();
    let n = case.stage().rounds();
    assert_eq!(honest.merge.rounds.len(), n as usize);
    let merge = |round| RejectReason::LeafMerge { round };
    for j in 0..honest.merge.rounds.len() {
        for k in 0..2 {
            let mut trees = honest.clone();
            trees.merge.rounds[j][k] += Fp2::ONE;
            assert_eq!(case.reason(&trees), merge(n), "round {j}, value {k}");
        }
        let mut trees = honest.clone();
        trees.merge.rounds[j].push(Fp2::ONE);
        assert_eq!(case.reason(&trees), merge(j as u32));
        let mut trees = honest.clone();
        trees.merge.rounds.truncate(j);
        assert_eq!(case.reason(&trees), merge(j as u32));
    }
    for i in 0..honest.merge.final_evals.len() {
        let mut trees = honest.clone();
        trees.merge.final_evals[i] += Fp2::ONE;
        assert_eq!(case.reason(&trees), merge(n), "evaluation {i}");
    }
    let mut trees = honest.clone();
    trees.merge.rounds.push(vec![Fp2::ONE; 2]);
    assert_eq!(case.reason(&trees), merge(n));
    let mut trees = honest.clone();
    trees.merge.final_evals.pop();
    assert_eq!(case.reason(&trees), merge(n));
    let mut trees = honest.clone();
    trees.merge.final_evals.push(Fp2::ONE);
    assert_eq!(case.reason(&trees), merge(n));
}

/// Check 17: two merged claims swapped between tensors of one class, `u` and `g` at one site and
/// layer, at one point with the same groups, are told apart by their `γ_h^k`; and so are one
/// tensor's claims at two layers of one size and `S`.
#[test]
fn a_merged_claim_swapped_between_two_tensors_of_one_class_fails_the_merge() {
    let case = case(1, 1, 22);
    let polynomials = case.grouped.polynomials();
    let position = |node: u32, column: Column| {
        polynomials
            .iter()
            .position(|p| {
                p.source
                    == Source::Split {
                        node: tv_core::lowering::NodeId(node),
                        column,
                    }
            })
            .unwrap()
    };
    let n = case.stage().rounds();
    for (a, b) in [
        (
            position(20, Column::RootInput),
            position(20, Column::GreedyKey),
        ),
        (
            position(10, Column::RootInput),
            position(30, Column::RootInput),
        ),
    ] {
        let (p, q) = (&polynomials[a], &polynomials[b]);
        assert_eq!(
            (p.variables, &p.group_variables),
            (q.variables, &q.group_variables)
        );
        let mut trees = case.honest();
        assert_ne!(trees.merge.final_evals[a], trees.merge.final_evals[b]);
        trees.merge.final_evals.swap(a, b);
        assert_eq!(case.reason(&trees), RejectReason::LeafMerge { round: n });
    }
}

/// The first failing check in RFC-0054's order is the one reported: a group's level before a
/// later group's root, a group's split before the table batch, the table batch before the merge.
#[test]
fn the_first_failing_check_in_order_is_reported() {
    let case = case(1, 1, 23);
    let honest = case.honest();
    let mut trees = honest.clone();
    trees.groups[0].levels[0].final_evals[0] += Fp2::ONE;
    trees.groups[2].roots[1] = Fp2::ZERO;
    trees.merge.final_evals[0] += Fp2::ONE;
    let rounds = honest.groups[0].levels[0].rounds.len() as u32;
    assert_eq!(case.reason(&trees), tree_level(0, 0, rounds));
    let mut trees = honest.clone();
    trees.groups[1].splits[0] += Fp2::ONE;
    trees.table_roots[1] = Fp2::ZERO;
    let group = case.grouped.group(1).unwrap();
    assert_eq!(case.reason(&trees), tree_level(1, group.levels(), 0));
    let mut trees = honest.clone();
    trees.levels[0].final_evals[0] += Fp2::ONE;
    trees.merge.rounds[0].push(Fp2::ONE);
    let rounds = honest.levels[0].rounds.len() as u32;
    assert_eq!(case.reason(&trees), tree_level(4, 0, rounds));
}

/// Public data the verifier lacks, a forest root or a table, is its own fault: `Inconclusive`,
/// never a rejection.
#[test]
fn missing_public_data_is_a_fault_of_the_verifier() {
    let case = case(1, 1, 24);
    let roots = case.roots();
    for missing in 0..3 {
        let error = case
            .stage()
            .verify(
                &mut Transcript::new(&session()),
                Proof {
                    logup: &case.proved.logup,
                    trees: &case.honest(),
                },
                |tree, point| {
                    let (_, table) = roots.iter().find(|(t, _)| *t == tree)?;
                    (missing != 0).then(|| extension(table, point))
                },
                |id, beta, point| {
                    (missing != 1).then(|| table_eval(&case.tables, id, beta, point))?
                },
                |id| (missing != 2).then(|| row_zero(&case.tables, id))?,
            )
            .unwrap_err();
        assert_eq!(error, Error::Internal);
        assert_eq!(error.rejection(NI), None);
        assert!(matches!(
            error.verdict(NI),
            Verdict::Inconclusive(e) if e.code == ErrorCode::Internal
        ));
    }
    // The honest proof verifies with every callback.
    let _ = case.verify(&case.honest()).unwrap();
    let _ = from_fe(Fe::ONE);
}
