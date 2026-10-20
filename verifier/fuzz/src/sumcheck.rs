// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Cases and oracles for the `sumcheck` target.
//!
//! A case is an honest proof of a seeded instance, made by the test kit's reference prover,
//! which shares no code with `tv-core`: a batched sumcheck of products of multilinear tables,
//! front-padded, or a batch of product trees of different heights, level by level. A script the
//! fuzz input spells out then changes the proof. The oracle judges the verifier's result from
//! the specification alone, without `tv-core`'s code:
//!
//! - an unchanged proof verifies, at the prover's own challenges, and a batch of trees ends
//!   with true leaf claims;
//! - a changed proof never verifies with true claims. Its first structural defect, in the
//!   crate's order of checks, is the error, exactly; a proof whose shape is right and whose
//!   rounds changed fails the final check, which an honest proof's evaluations at another point
//!   pass only with probability `(n·d + m)/|Fp2|`; a stage whose evaluations alone changed is
//!   accepted exactly when the final check, computed here from the prover's `γ` and last claim,
//!   holds, and then a changed evaluation is a false claim on a table;
//! - a batch of trees whose first changed level passes, as one with `u_i` and `w_i` exchanged
//!   does, is rejected by a later level or leaves a false leaf claim, checked here against the
//!   leaves.

use tv_core::sumcheck::{Instance, MAX_DEGREE, Prefix, SumcheckError, SumcheckProof, verify_batch};
use tv_core::tree::{TreeClaim, verify_product_level};
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_testkit::SplitMix64;
use tv_testkit::sumcheck::{self as reference, Fe, Product, ProductTree};

use crate::proofs::Script;

/// The most instances of a stage case and trees of a tree case.
pub const MAX_INSTANCES: u8 = 4;
/// The most variables of a stage case's instance.
pub const MAX_VARS: u8 = 6;
/// The tallest tree of a tree case.
pub const MAX_HEIGHT: u8 = 5;
/// The changes a script knows.
pub const CHANGES: u8 = 16;

/// The session every case runs in.
pub fn context() -> SessionContext<'static> {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x59; 32],
        session_id: *b"fuzz: sumcheck  ",
    }
}

fn from_reference(value: Fe) -> Fp2 {
    Fp2::new(
        Fp::from_canonical(value.0).unwrap(),
        Fp::from_canonical(value.1).unwrap(),
    )
}

fn to_reference(value: Fp2) -> Fe {
    Fe(value.c0.to_canonical(), value.c1.to_canonical())
}

fn proof_of(proof: &reference::Proof) -> SumcheckProof {
    SumcheckProof {
        rounds: proof
            .rounds
            .iter()
            .map(|round| round.iter().copied().map(from_reference).collect())
            .collect(),
        final_evals: proof.evals.iter().copied().map(from_reference).collect(),
    }
}

fn product(_: usize, _: &[Fp2], evals: &[Fp2]) -> Option<Fp2> {
    Some(evals.iter().fold(Fp2::ONE, |p, e| p * *e))
}

/// A batched sumcheck of products, honest and changed.
pub struct StageCase {
    pub prefix: Prefix,
    pub degree: usize,
    pub instances: Vec<Instance>,
    pub honest: SumcheckProof,
    pub gamma: Fp2,
    pub challenges: Vec<Fp2>,
    pub last: Fp2,
    pub changed: SumcheckProof,
}

/// A batch of product trees, honest and changed.
pub struct TreeCase {
    pub trees: Vec<ProductTree>,
    pub honest: Vec<SumcheckProof>,
    /// The honest claims entering each level, in schedule order.
    pub entering: Vec<Vec<TreeClaim>>,
    pub changed: Vec<SumcheckProof>,
}

/// A case of the target.
pub enum Case {
    Stage(StageCase),
    Trees(TreeCase),
}

/// An element the script spells, both coefficients reduced.
fn element(script: &mut Script<'_>) -> Fp2 {
    Fp2::new(Fp::reduce(script.u64()), Fp::reduce(script.u64()))
}

/// Changes `proof` as one script operation says.
fn change(proof: &mut SumcheckProof, script: &mut Script<'_>) {
    let rounds = proof.rounds.len();
    let evals = proof.final_evals.len();
    match script.u8() % CHANGES {
        0 if rounds > 0 => {
            let j = script.below(rounds);
            if !proof.rounds[j].is_empty() {
                let k = script.below(proof.rounds[j].len());
                proof.rounds[j][k] += element(script);
            }
        }
        1 if rounds > 0 => {
            let j = script.below(rounds);
            if !proof.rounds[j].is_empty() {
                let k = script.below(proof.rounds[j].len());
                proof.rounds[j][k] = element(script);
            }
        }
        2 if rounds > 0 => {
            let j = script.below(rounds);
            let k = script.below(proof.rounds[j].len() + 1);
            proof.rounds[j].insert(k, element(script));
        }
        3 if rounds > 0 => {
            let j = script.below(rounds);
            if !proof.rounds[j].is_empty() {
                let k = script.below(proof.rounds[j].len());
                proof.rounds[j].remove(k);
            }
        }
        4 => {
            let length = usize::from(script.u8()) % (MAX_DEGREE + 3);
            let round = (0..length).map(|_| element(script)).collect();
            proof.rounds.insert(script.below(rounds + 1), round);
        }
        5 if rounds > 0 => {
            proof.rounds.remove(script.below(rounds));
        }
        6 if rounds > 0 => {
            let copy = proof.rounds[script.below(rounds)].clone();
            proof.rounds.insert(script.below(rounds + 1), copy);
        }
        7 if rounds > 0 => {
            let (a, b) = (script.below(rounds), script.below(rounds));
            proof.rounds.swap(a, b);
        }
        8 if rounds > 0 => {
            let j = script.below(rounds);
            let length = proof.rounds[j].len();
            if length > 0 {
                let (a, b) = (script.below(length), script.below(length));
                proof.rounds[j].swap(a, b);
            }
        }
        9 if evals > 0 => {
            let k = script.below(evals);
            proof.final_evals[k] += element(script);
        }
        10 if evals > 0 => {
            let k = script.below(evals);
            proof.final_evals[k] = element(script);
        }
        11 => {
            let k = script.below(evals + 1);
            proof.final_evals.insert(k, element(script));
        }
        12 if evals > 0 => {
            proof.final_evals.remove(script.below(evals));
        }
        13 if evals > 0 => {
            let (a, b) = (script.below(evals), script.below(evals));
            proof.final_evals.swap(a, b);
        }
        14 => {
            let keep = script.below(rounds + 1);
            proof.rounds.truncate(keep);
        }
        _ => {
            let keep = script.below(evals + 1);
            proof.final_evals.truncate(keep);
        }
    }
}

/// The case a fuzz input spells. Byte 0 picks a stage (even) or trees (odd).
///
/// - A stage: a byte for the degree `1 + b mod 7`, one for `1 + b mod 4` instances, and for
///   each instance one for its variables, `b mod 7`, and one for its tables, `1 + b mod d`;
///   then a byte for the stage's index and 8 bytes seeding the tables.
/// - Trees: a byte for `1 + b mod 4` trees, and for each one for its height, `1 + b mod 5`,
///   and one that, below 64, sets the leaf `b mod 2^h` to zero; then 8 bytes seeding the leaves.
///
/// Then changes, an operation byte and its arguments each, until the input ends; a tree case's
/// change starts with a byte picking the level it changes.
pub fn case(data: &[u8]) -> Case {
    let mut script = Script::new(data);
    if script.u8() & 1 == 0 {
        let degree = 1 + usize::from(script.u8() % 7);
        let shape: Vec<(usize, usize)> = (0..1 + script.u8() % MAX_INSTANCES)
            .map(|_| {
                let vars = usize::from(script.u8() % (MAX_VARS + 1));
                let tables = 1 + usize::from(script.u8()) % degree;
                (vars, tables)
            })
            .collect();
        let prefix = Prefix::Stage(u32::from(script.u8()));
        let mut rng = SplitMix64::new(script.u64());
        let products: Vec<Product> = shape
            .iter()
            .map(|&(vars, tables)| {
                Product::new(
                    (0..tables)
                        .map(|_| {
                            (0..1 << vars)
                                .map(|_| Fe::new(rng.next_u64(), rng.next_u64()))
                                .collect()
                        })
                        .collect(),
                )
            })
            .collect();
        let mut prover = reference::Transcript::new(&context().to_tce());
        let proof = reference::prove(&mut prover, &prefix.to_string(), degree, &products, None);
        let honest = proof_of(&proof);
        let mut changed = honest.clone();
        while !script.is_empty() {
            change(&mut changed, &mut script);
        }
        Case::Stage(StageCase {
            prefix,
            degree,
            instances: products
                .iter()
                .map(|p| Instance {
                    claim: from_reference(p.claim()),
                    vars: p.vars(),
                    evals: p.tables.len(),
                })
                .collect(),
            honest,
            gamma: from_reference(proof.gamma),
            challenges: proof
                .challenges
                .iter()
                .copied()
                .map(from_reference)
                .collect(),
            last: from_reference(*proof.claims.last().unwrap()),
            changed,
        })
    } else {
        let shape: Vec<(usize, u8)> = (0..1 + script.u8() % MAX_INSTANCES)
            .map(|_| (1 + usize::from(script.u8() % MAX_HEIGHT), script.u8()))
            .collect();
        let mut rng = SplitMix64::new(script.u64());
        let trees: Vec<ProductTree> = shape
            .iter()
            .map(|&(height, zero)| {
                let mut leaves: Vec<Fe> =
                    (0..1 << height).map(|_| Fe::base(rng.next_u64())).collect();
                if zero < 64 {
                    leaves[usize::from(zero) % (1 << height)] = Fe::ZERO;
                }
                ProductTree::new(leaves)
            })
            .collect();
        let mut prover = reference::Transcript::new(&context().to_tce());
        let levels = reference::prove_trees(&mut prover, &trees);
        let honest: Vec<SumcheckProof> =
            levels.iter().map(|level| proof_of(&level.proof)).collect();
        let entering = levels
            .iter()
            .map(|level| {
                level
                    .claims
                    .iter()
                    .map(|claim| TreeClaim {
                        point: claim.point.iter().copied().map(from_reference).collect(),
                        value: from_reference(claim.value),
                    })
                    .collect()
            })
            .collect();
        let mut changed = honest.clone();
        while !script.is_empty() {
            let level = script.below(changed.len());
            change(&mut changed[level], &mut script);
        }
        Case::Trees(TreeCase {
            trees,
            honest,
            entering,
            changed,
        })
    }
}

/// The first check of the crate's order that a proof's shape fails, or `None` when its shape is
/// right, from the crate documentation alone.
pub fn structural(
    degree: usize,
    instances: &[Instance],
    proof: &SumcheckProof,
) -> Option<SumcheckError> {
    if degree == 0 || degree > MAX_DEGREE {
        return Some(SumcheckError::Degree { degree });
    }
    let Some(n_max) = instances.iter().map(|i| i.vars).max() else {
        return Some(SumcheckError::NoInstances);
    };
    let Some(expected) = instances
        .iter()
        .try_fold(0usize, |t, i| t.checked_add(i.evals))
    else {
        return Some(SumcheckError::EvaluationTotal);
    };
    for j in 0..n_max {
        let round = j as u32;
        match proof.rounds.get(j) {
            None => return Some(SumcheckError::MissingRound { round }),
            Some(message) if message.len() != degree => {
                return Some(SumcheckError::RoundLength {
                    round,
                    expected: degree,
                    found: message.len(),
                });
            }
            Some(_) => {}
        }
    }
    if proof.rounds.len() > n_max {
        return Some(SumcheckError::ExtraRound {
            round: n_max as u32,
        });
    }
    if proof.final_evals.len() != expected {
        return Some(SumcheckError::EvaluationCount {
            round: n_max as u32,
            expected,
            found: proof.final_evals.len(),
        });
    }
    None
}

impl StageCase {
    /// What verifying the changed proof must give.
    pub fn expected(&self) -> Result<(), SumcheckError> {
        if self.changed == self.honest {
            return Ok(());
        }
        if let Some(error) = structural(self.degree, &self.instances, &self.changed) {
            return Err(error);
        }
        let round = self.challenges.len() as u32;
        if self.changed.rounds != self.honest.rounds {
            return Err(SumcheckError::FinalMismatch { round });
        }
        // Only evaluations changed: the final check at the honest point, computed here.
        let mut rest = self.changed.final_evals.iter();
        let mut weight = Fp2::ONE;
        let mut sum = Fp2::ZERO;
        for instance in &self.instances {
            let value = rest
                .by_ref()
                .take(instance.evals)
                .fold(Fp2::ONE, |p, e| p * *e);
            sum += weight * value;
            weight *= self.gamma;
        }
        if sum == self.last {
            Ok(())
        } else {
            Err(SumcheckError::FinalMismatch { round })
        }
    }

    /// Verifies the changed proof and checks the result against the oracle.
    pub fn check(&self) {
        let mut transcript = Transcript::new(&context());
        let outcome = verify_batch(
            &mut transcript,
            self.prefix,
            self.degree,
            &self.instances,
            &self.changed,
            product,
        );
        assert_eq!(
            outcome.as_ref().map(|_| ()).map_err(|e| *e),
            self.expected(),
            "degree {}, instances {:?}",
            self.degree,
            self.instances
                .iter()
                .map(|i| (i.vars, i.evals))
                .collect::<Vec<_>>()
        );
        if let Ok(batch) = outcome {
            assert_eq!(batch.point, self.challenges, "the prover's challenges");
            assert_eq!(batch.gamma, self.gamma);
            if self.changed != self.honest {
                // Accepted only with a changed evaluation: a false claim.
                assert_eq!(self.changed.rounds, self.honest.rounds);
                assert_ne!(self.changed.final_evals, self.honest.final_evals);
            }
        }
    }
}

/// What a batch of trees comes to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeOutcome {
    /// Every level passed and every leaf claim is true.
    Accepted,
    /// Level `level` was rejected.
    Rejected { level: u32, error: SumcheckError },
    /// Every level passed and tree `tree`'s leaf claim is false.
    FalseLeaf { tree: usize },
}

impl TreeCase {
    /// Runs every level of the changed proof as the caller does, trees joining at level
    /// `H − h` with their root claims, then checks the leaf claims against the leaves.
    pub fn run(&self) -> TreeOutcome {
        let tallest = self.trees.iter().map(ProductTree::height).max().unwrap();
        let mut current: Vec<Option<TreeClaim>> = vec![None; self.trees.len()];
        let mut transcript = Transcript::new(&context());
        for (level, proof) in self.changed.iter().enumerate() {
            for (t, tree) in self.trees.iter().enumerate() {
                if tallest - tree.height() == level {
                    current[t] = Some(TreeClaim {
                        point: Vec::new(),
                        value: from_reference(tree.root()),
                    });
                }
            }
            let present: Vec<usize> = (0..self.trees.len())
                .filter(|&t| current[t].is_some())
                .collect();
            let claims: Vec<TreeClaim> = present
                .iter()
                .map(|&t| current[t].clone().unwrap())
                .collect();
            match verify_product_level(&mut transcript, level as u32, &claims, proof) {
                Err(error) => {
                    return TreeOutcome::Rejected {
                        level: level as u32,
                        error,
                    };
                }
                Ok(outcome) => {
                    assert_eq!(outcome.claims.len(), present.len());
                    for (&t, claim) in present.iter().zip(outcome.claims) {
                        current[t] = Some(claim);
                    }
                }
            }
        }
        for (t, claim) in current.iter().enumerate() {
            let claim = claim.as_ref().unwrap();
            let point: Vec<Fe> = claim.point.iter().copied().map(to_reference).collect();
            if reference::mle(self.trees[t].leaves(), &point) != to_reference(claim.value) {
                return TreeOutcome::FalseLeaf { tree: t };
            }
        }
        TreeOutcome::Accepted
    }

    /// Runs the changed proof and checks the result against the oracle.
    pub fn check(&self) {
        let outcome = self.run();
        let Some(first) = (0..self.honest.len()).find(|&l| self.changed[l] != self.honest[l])
        else {
            assert_eq!(outcome, TreeOutcome::Accepted, "the honest proof");
            return;
        };
        assert_ne!(
            outcome,
            TreeOutcome::Accepted,
            "a changed proof, first at level {first}"
        );
        let instances: Vec<Instance> = self.entering[first]
            .iter()
            .map(|claim| Instance {
                claim: claim.value,
                vars: claim.point.len(),
                evals: 2,
            })
            .collect();
        let at_first = |error| TreeOutcome::Rejected {
            level: first as u32,
            error,
        };
        if let Some(error) = structural(3, &instances, &self.changed[first]) {
            assert_eq!(
                outcome,
                at_first(error),
                "a structural defect at level {first}"
            );
        } else if self.changed[first].rounds != self.honest[first].rounds {
            let round = first as u32;
            assert_eq!(
                outcome,
                at_first(SumcheckError::FinalMismatch { round }),
                "changed rounds at level {first}"
            );
        }
    }
}

/// Builds the case `data` spells and checks the verifier's result against the oracle.
/// Returns whether the gadget oracle handled the input.
pub fn check(data: &[u8]) -> bool {
    if let Some(gadget) = data.strip_prefix(b"GDG") {
        crate::gadgets::check(gadget);
        return true;
    }
    match case(data) {
        Case::Stage(case) => case.check(),
        Case::Trees(case) => case.check(),
    }
    false
}
