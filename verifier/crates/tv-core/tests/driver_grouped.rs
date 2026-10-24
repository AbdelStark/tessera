// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The drivers verify a leaf stage in token groups end to end (RFC-0054), on the recorded CPU
//! proofs of `fixtures/grouped/` (`leaf_groups` (0, 1), (0, 2) and (1, 2)): each is accepted;
//! the grouping's bounds and counts reject before any transcript work; every grouped check
//! rejects a proof that fails exactly it with its exact reason; and the reduction stages accept
//! the merged claims at their merged points, and only there. The fixtures are producer output,
//! inputs here; every verdict below is RFC-0054's.
#![cfg(feature = "test-params")]

use std::path::Path;
use tv_codec::{DecodeLimits, WeightCommitmentPublic};
use tv_core::claims::ClaimList;
use tv_core::gadgets::{GadgetError, model, runtime};
use tv_core::leaf::grouped::{self, Stage, TableTree};
use tv_core::leaf::groups::Grouped;
use tv_core::leaf::split::{Bound, LeafPoint};
use tv_core::leaf::{ModelSchedule, Tree};
use tv_core::lowering::{self, Graph};
use tv_core::params::ParamSet;
use tv_core::protocol::{FixedBits, LeafGroups, NiProof, ProverSeedShare, PublicStatement};
use tv_core::verdict::ArtifactKind;
use tv_core::{
    ErrorCode, RejectReason as R, ResourceLimits, Verdict, VerifierConfig, validation,
    verify_non_interactive, verify_non_interactive_bytes, weights,
};
use tv_field::Fp2;
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_tables::TableId;

const NI: ArtifactKind = ArtifactKind(0x0520);
const ROUND1: ArtifactKind = ArtifactKind(0x0510);
const ROUND2: ArtifactKind = ArtifactKind(0x0511);

struct Fixture {
    claims: ClaimList,
    proof: NiProof,
    bytes: Vec<u8>,
    cfg: VerifierConfig,
}
impl Fixture {
    fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/grouped")
            .join(name);
        let read = |file: &str| std::fs::read(dir.join(file)).unwrap();
        let limits = DecodeLimits::MESSAGE;
        let weights = tv_codec::decode(
            &read("weights.tce"),
            &limits,
            WeightCommitmentPublic::decode,
        )
        .unwrap();
        let bytes = read("ni.tnp");
        Self {
            claims: ClaimList::open(&read("claims.tcl"), &limits).unwrap(),
            proof: NiProof::open(&bytes, &limits).unwrap(),
            bytes,
            cfg: VerifierConfig {
                params: ParamSet::TestSmall,
                limits: ResourceLimits::default(),
                accept_bound: FixedBits(0),
                weights,
            },
        }
    }
    /// The statement the fixture's README pins: the proof's own.
    fn expected(&self) -> PublicStatement {
        self.proof.round1.statement.clone()
    }
    fn verify(&self, proof: &NiProof) -> Verdict {
        verify_non_interactive(&self.claims, &self.expected(), proof, &self.cfg)
    }
    fn rejects(&self, proof: &NiProof, reason: &R) {
        match self.verify(proof) {
            Verdict::Reject(actual) => assert_eq!(&actual, reason),
            other => panic!("expected {reason:?}, got {other:?}"),
        }
    }
    fn graph(&self) -> Graph {
        let r = &self.expected().batch.requests[0];
        lowering::lower(
            &validation::check_public(&self.claims).unwrap(),
            r.prompt_len + r.output_len,
        )
        .unwrap()
    }
}

fn groups(token_bits: u8, layer_bits: u8) -> LeafGroups {
    LeafGroups {
        token_bits,
        layer_bits,
    }
}

/// The verifier's own layouts for the fixture's statement: the runtime plan's grouped stage and
/// its table trees, each from public data alone.
struct Layouts<'g> {
    graph: &'g Graph,
    validated: validation::Validated,
    claims: &'g ClaimList,
}
impl<'g> Layouts<'g> {
    fn new(f: &'g Fixture, graph: &'g Graph) -> Self {
        Self {
            graph,
            validated: validation::validate(&f.claims).unwrap(),
            claims: &f.claims,
        }
    }
    fn runtime(&self, prompt: u32) -> runtime::Plan<'_> {
        runtime::Plan::derive(
            self.graph,
            self.claims.gadget_policy,
            prompt,
            0,
            self.validated.tables(),
        )
        .unwrap()
    }
}
fn table_trees(schedule: &ModelSchedule) -> Vec<TableTree> {
    schedule
        .schedule()
        .trees()
        .iter()
        .filter_map(|t| match *t {
            Tree::Table { instance, height } => Some(TableTree {
                table: schedule.lookups()[instance].0,
                height,
            }),
            _ => None,
        })
        .collect()
}

#[test]
fn the_recorded_grouped_proofs_are_accepted_identically_from_bytes() {
    for (name, expected) in [
        ("g1", groups(0, 1)),
        ("g2", groups(0, 2)),
        ("g3", groups(1, 2)),
    ] {
        let f = Fixture::new(name);
        assert_eq!(f.proof.round1.leaf_groups, expected, "{name}");
        let report = match f.verify(&f.proof) {
            Verdict::Accept(report) => report,
            other => panic!("{name}: {other:?}"),
        };
        let from_bytes = verify_non_interactive_bytes(&f.claims, &f.expected(), &f.bytes, &f.cfg);
        assert_eq!(
            from_bytes.to_json().unwrap(),
            Verdict::Accept(report).to_json().unwrap(),
            "{name}"
        );
    }
}

/// RFC-0054 'Messages': the base's `n = 5` tokens give `t = 3`, its 3 layers `n_ℓ = 2`. A
/// grouping past either is `Malformed { NiProof, ERR-CODEC-LIMIT }`, and past 14 bits the
/// decoder's own; a grouping within them fixes the stage's counts.
#[test]
fn the_grouping_bounds_and_counts_reject_before_any_transcript_work() {
    let f = Fixture::new("g3");
    let limit = R::Malformed {
        artifact: NI,
        code: ErrorCode::CodecLimit,
    };
    let counts = R::Malformed {
        artifact: NI,
        code: ErrorCode::CodecNoncanonical,
    };
    for (bad, reason) in [
        (groups(4, 0), &limit),
        (groups(0, 3), &limit),
        (groups(4, 3), &limit),
        // Within the bounds: 4 groups where the proof holds 8, and the ungrouped stage's empty
        // groups where it holds 8.
        (groups(1, 1), &counts),
        (groups(3, 2), &counts),
        (LeafGroups::NONE, &counts),
    ] {
        let mut proof = f.proof.clone();
        proof.round1.leaf_groups = bad;
        f.rejects(&proof, reason);
    }
    // Past 14 bits the codec refuses the envelope itself, at the struct's first byte.
    let mut bytes = f.bytes.clone();
    let round1 = f.proof.round1.to_tce().unwrap();
    // Round1 is the payload's prefix; its grouping is its last two bytes.
    assert_eq!(&f.bytes[20..20 + round1.len()], round1.as_slice());
    let groups_at = 20 + round1.len() - 2;
    bytes[groups_at..groups_at + 2].copy_from_slice(&[8, 7]);
    let resealed = reseal(&bytes);
    match verify_non_interactive_bytes(&f.claims, &f.expected(), &resealed, &f.cfg) {
        Verdict::Reject(actual) => assert_eq!(actual, limit),
        other => panic!("{other:?}"),
    }
    // A count the grouping fixes, in a group, the table batch and the empty LogUpProof.
    let mut proof = f.proof.clone();
    proof.round2.trees.groups[5].splits.pop();
    f.rejects(&proof, &counts);
    let mut proof = f.proof.clone();
    proof.round2.trees.table_roots.push(Fp2::ONE);
    f.rejects(&proof, &counts);
    let mut proof = f.proof.clone();
    proof.round2.logup.values.push(Fp2::ONE);
    f.rejects(&proof, &counts);
    // The atom splits' count is the schedule's: none in this model.
    let mut proof = f.proof.clone();
    proof.round2.trees.splits.push(Fp2::ONE);
    f.rejects(&proof, &counts);
}

/// RFC-0003 'Envelope': the digest of everything before it, over an edited payload.
fn reseal(bytes: &[u8]) -> Vec<u8> {
    let body = &bytes[..bytes.len() - 32];
    let mut out = body.to_vec();
    out.extend_from_slice(&tv_hash::hash(tv_hash::Domain::Envelope, body));
    out
}

/// Every grouped check, failed alone, with its reason (RFC-0054 'Rejections').
#[test]
fn each_grouped_check_rejects_with_its_own_reason() {
    let f = Fixture::new("g3");
    let graph = f.graph();
    let layouts = Layouts::new(&f, &graph);
    let prompt = f.expected().batch.requests[0].prompt_len;
    let runtime = layouts.runtime(prompt);
    let grouped = Grouped::derive_runtime(&graph, &runtime, f.proof.round1.leaf_groups).unwrap();
    let tables = table_trees(&runtime.leaf_schedule().unwrap());
    let count = grouped.count();
    assert_eq!(count, 8);
    let trees = &f.proof.round2.trees;
    let first_table = |h: u32| -> TableId { grouped.group(h).unwrap().instances[0].table };
    let shift = |x: &mut Fp2| *x += Fp2::ONE;

    // A group's roots: a zero denominator, and a numerator that unbalances its table's sum.
    let mut proof = f.proof.clone();
    proof.round2.trees.groups[1].roots[1] = Fp2::ZERO;
    f.rejects(
        &proof,
        &R::Lookup {
            table: first_table(1),
        },
    );
    // A changed numerator: the group's levels run before the table batch's identity, and the
    // roots are absorbed before the group's first challenge (INV-FS-1). Level 0 has no round, so
    // its check is an identity on the tallest trees' honest roots whatever its challenges; level
    // 1 is the first to meet new challenges, and fails its final check.
    let mut proof = f.proof.clone();
    shift(&mut proof.round2.trees.groups[4].roots[0]);
    assert!(trees.groups[4].levels[0].rounds.is_empty());
    f.rejects(
        &proof,
        &R::TreeLevel {
            group: 4,
            level: 1,
            round: trees.groups[4].levels[1].rounds.len() as u32,
        },
    );
    // A group's level: a round value and an evaluation, caught by the level's final check.
    for (h, l) in [(0usize, 9usize), (2, 3), (7, 2)] {
        let rounds = trees.groups[h].levels[l].rounds.len() as u32;
        let reason = R::TreeLevel {
            group: h as u32,
            level: l as u32,
            round: rounds,
        };
        let mut proof = f.proof.clone();
        shift(&mut proof.round2.trees.groups[h].levels[l].rounds[0][0]);
        f.rejects(&proof, &reason);
        let mut proof = f.proof.clone();
        shift(&mut proof.round2.trees.groups[h].levels[l].final_evals[0]);
        f.rejects(&proof, &reason);
    }
    // A group's value and split values: the group's split check, at its own level count.
    for h in [0usize, 1, 5] {
        let reason = R::TreeLevel {
            group: h as u32,
            level: trees.groups[h].levels.len() as u32,
            round: 0,
        };
        for at in [0, trees.groups[h].splits.len() - 1] {
            let mut proof = f.proof.clone();
            shift(&mut proof.round2.trees.groups[h].splits[at]);
            f.rejects(&proof, &reason);
        }
        let mut proof = f.proof.clone();
        shift(&mut proof.round2.trees.groups[h].values[0]);
        f.rejects(&proof, &reason);
    }
    // The table batch: a zero denominator, an unbalanced identity, and its levels as group G.
    let mut proof = f.proof.clone();
    proof.round2.trees.table_roots[3] = Fp2::ZERO;
    f.rejects(
        &proof,
        &R::Lookup {
            table: tables[1].table,
        },
    );
    let mut proof = f.proof.clone();
    shift(&mut proof.round2.trees.table_roots[4]);
    f.rejects(
        &proof,
        &R::Lookup {
            table: tables[2].table,
        },
    );
    for l in [1usize, 17] {
        let mut proof = f.proof.clone();
        shift(&mut proof.round2.trees.levels[l].final_evals[0]);
        f.rejects(
            &proof,
            &R::TreeLevel {
                group: count,
                level: l as u32,
                round: trees.levels[l].rounds.len() as u32,
            },
        );
    }
    // The merge: a round value, a short round, an evaluation, a missing evaluation, and two
    // merged claims of one class swapped.
    let rounds = trees.merge.rounds.len() as u32;
    assert_eq!(
        rounds,
        grouped
            .polynomials()
            .iter()
            .map(|p| p.variables)
            .max()
            .unwrap()
    );
    let mut proof = f.proof.clone();
    shift(&mut proof.round2.trees.merge.rounds[0][0]);
    f.rejects(&proof, &R::LeafMerge { round: rounds });
    let mut proof = f.proof.clone();
    proof.round2.trees.merge.rounds[3].pop();
    f.rejects(&proof, &R::LeafMerge { round: 3 });
    let mut proof = f.proof.clone();
    shift(&mut proof.round2.trees.merge.final_evals[17]);
    f.rejects(&proof, &R::LeafMerge { round: rounds });
    let mut proof = f.proof.clone();
    proof.round2.trees.merge.final_evals.pop();
    f.rejects(&proof, &R::LeafMerge { round: rounds });
    let polynomials = grouped.polynomials();
    let (i, j) = (0..polynomials.len())
        .flat_map(|i| (i + 1..polynomials.len()).map(move |j| (i, j)))
        .find(|&(i, j)| {
            let (a, b) = (&polynomials[i], &polynomials[j]);
            a.variables == b.variables
                && a.group_variables == b.group_variables
                && a.readers == b.readers
                && trees.merge.final_evals[i] != trees.merge.final_evals[j]
        })
        .unwrap();
    let mut proof = f.proof.clone();
    proof.round2.trees.merge.final_evals.swap(i, j);
    f.rejects(&proof, &R::LeafMerge { round: rounds });
    // The stages that take the merged claims: stage 1's evaluation, and stage 2's round.
    let mut proof = f.proof.clone();
    let stage = &mut proof.round2.stages[0].sumcheck;
    let n1 = stage.rounds.len() as u32;
    shift(&mut stage.final_evals[0]);
    f.rejects(
        &proof,
        &R::Sumcheck {
            stage: 1,
            round: n1,
        },
    );
    let mut proof = f.proof.clone();
    let stage = &mut proof.round2.stages[1].sumcheck;
    let n2 = stage.rounds.len() as u32;
    shift(&mut stage.rounds[0][0]);
    f.rejects(
        &proof,
        &R::Sumcheck {
            stage: 2,
            round: n2,
        },
    );
}

/// The verifier's transcript through the leaf stage of a non-interactive proof at
/// `test-small/1` (RFC-0014 rows 1 to 5, RFC-0054 step 5), and the stage's outputs.
fn leaf_bound(f: &Fixture, graph: &Graph, runtime: &runtime::Plan<'_>) -> (Transcript, Bound) {
    let proof = &f.proof;
    let expected = f.expected();
    let mut t = Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: expected.digest(),
        session_id: proof.round1.session_id,
    });
    t.absorb(
        Ascii::new("round1").unwrap(),
        &proof.round1.to_tce().unwrap(),
    );
    let mut t = t
        .check_grind(Ascii::new("seed-pow").unwrap(), 0, proof.seed_nonce)
        .unwrap();
    let _ = t.squeeze(Ascii::new("seed").unwrap());
    for l in weights::derive(&f.claims, f.cfg.weights.epoch).unwrap() {
        if l.committed_rows() > 0 {
            let label = format!("pcs/weights/{}/xi", l.matrix().name());
            for _ in 0..l.proximity_vars() {
                t.challenge_fp2(Ascii::new(&label).unwrap()).unwrap();
            }
        }
    }
    let schedule = runtime.leaf_schedule().unwrap();
    let grouped = Grouped::derive_runtime(graph, runtime, proof.round1.leaf_groups).unwrap();
    let stage = Stage::new(&grouped, &table_trees(&schedule)).unwrap();
    let row_zero = |id| {
        let table = runtime.table(id)?;
        let (k, v) = table.row(0)?;
        Some((
            k,
            (!matches!(table.spec().kind, tv_tables::TableKind::Range { .. })).then_some(v),
        ))
    };
    let merged = stage
        .verify(
            &mut t,
            grouped::Proof {
                logup: &proof.round2.logup,
                trees: &proof.round2.trees,
            },
            |i, p| schedule.forest_root(i, p),
            |id, b, p| runtime.table(id)?.eval_mle(p, b).ok(),
            row_zero,
        )
        .unwrap();
    let native = model::Plan::derive_runtime(runtime).unwrap();
    let bound = merged
        .bind(
            &mut t,
            &proof.round2.trees.splits,
            native.schedule().leaf_splits(),
        )
        .unwrap();
    (t, bound)
}

/// RFC-0054 'The stage's outputs': the gadget checks and the reduction stages take the merged
/// claims at `r(n, S)`. With `g_T = 1`, `S` is not empty for any tensor with a token axis, so
/// placing the very same merged claims at the merge's last `n` challenges in order, `r(n, ∅)`,
/// consistently in the split and at the sites' points alike, makes a fixed gadget check fail;
/// and the forests' merged leaf claims so placed make stage 1 fail. The honest downstream claims
/// are consistent with the merged points only.
#[test]
fn the_stages_accept_the_merged_claims_only_at_their_merged_points() {
    let f = Fixture::new("g3");
    assert_eq!(f.proof.round1.leaf_groups, groups(1, 2));
    let graph = f.graph();
    let layouts = Layouts::new(&f, &graph);
    let runtime = layouts.runtime(f.expected().batch.requests[0].prompt_len);
    let native = model::Plan::derive_runtime(&runtime).unwrap();
    let rope = runtime
        .catalog()
        .iter()
        .find_map(|s| runtime.table(s.id)?.rope())
        .unwrap();
    let stages = &f.proof.round2.stages;

    let (mut t, bound) = leaf_bound(&f, &graph, &runtime);
    let LeafPoint::Merged {
        challenges,
        token_bits,
    } = bound.point.clone()
    else {
        panic!("a grouped stage leaves merged points");
    };
    assert_eq!(token_bits, 1);
    let prepared = native.bind_runtime(&bound).unwrap();
    let reduced = prepared.verify(&mut t, rope, stages).unwrap();
    assert!(!reduced.sources.is_empty());

    let (_, mut moved) = leaf_bound(&f, &graph, &runtime);
    moved.point = LeafPoint::Merged {
        challenges: challenges.clone(),
        token_bits: 0,
    };
    let mut changed = 0;
    for column in &mut moved.columns {
        let own = challenges[challenges.len() - column.point.len()..].to_vec();
        changed += usize::from(own != column.point);
        column.point = own;
    }
    assert!(changed > 0, "some tensor has group variables");
    // RFC-0037 'Native gadget targets': each layer's affine checks move public terms, such as
    // the root's and the softmax's row selectors at the site's point, into the native target.
    // The honest claims satisfy them at the merged points alone.
    match native.bind_runtime(&moved) {
        Err(error) => assert_eq!(
            error,
            GadgetError::Rejected {
                gadget: validation::GadgetId(1)
            }
        ),
        Ok(_) => panic!("the fixed gadget checks must reject claims moved off their points"),
    }

    // The forests' leaf claims are their merged claims at r(n_f, S_f), the claim RFC-0030's
    // relation 2 consumes in stage 1, and no fixed check reads them: moved to r(n_f, ∅), the
    // stage itself rejects.
    let (mut t, mut moved) = leaf_bound(&f, &graph, &runtime);
    assert!(!moved.forests.is_empty());
    for forest in &mut moved.forests {
        let own = challenges[challenges.len() - forest.claim.point.len()..].to_vec();
        assert_ne!(own, forest.claim.point, "a forest has group variables");
        forest.claim.point = own;
    }
    let prepared = native.bind_runtime(&moved).unwrap();
    let error = prepared.verify(&mut t, rope, stages).err().unwrap();
    assert_eq!(
        error.rejection().map(|r| (r.stage, r.round)),
        Some((1, stages[0].sumcheck.rounds.len() as u32)),
        "{error:?}"
    );
}

/// Live mode (RFC-0054 'Messages'): the grouping's bounds reject `Round1` itself, before any
/// seed is revealed; a grouping within them passes Round1, and its counts reject the
/// `Round2Response`. No grouped live session is recorded, so live acceptance is not tested.
#[cfg(unix)]
#[test]
fn a_live_session_judges_the_grouping_on_round1_and_its_counts_on_round2() {
    let f = Fixture::new("g3");
    let start = |round1_groups: LeafGroups| {
        let mut live = tv_core::LiveVerifier::new(f.claims.clone(), f.expected(), f.cfg.clone());
        let commit = live.start().unwrap();
        let mut round1 = f.proof.round1.clone();
        round1.session_id = commit.session_id;
        round1.leaf_groups = round1_groups;
        let share = ProverSeedShare {
            session_id: commit.session_id,
            share: [17; 32],
        };
        (live.on_round1(&round1, &share), live)
    };
    for bad in [groups(4, 0), groups(0, 3)] {
        let (outcome, _) = start(bad);
        match outcome {
            Err(Verdict::Reject(reason)) => assert_eq!(
                reason,
                R::Malformed {
                    artifact: ROUND1,
                    code: ErrorCode::CodecLimit
                }
            ),
            other => panic!("{other:?}"),
        }
    }
    let (outcome, mut live) = start(groups(1, 1));
    outcome.unwrap();
    match live.on_round2(&f.proof.round2) {
        Err(Verdict::Reject(reason)) => assert_eq!(
            reason,
            R::Malformed {
                artifact: ROUND2,
                code: ErrorCode::CodecNoncanonical
            }
        ),
        other => panic!("{other:?}"),
    }
    // The proof's own grouping passes Round1: its layouts derive from public data alone.
    let (outcome, _) = start(f.proof.round1.leaf_groups);
    outcome.unwrap();
}
