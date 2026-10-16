// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Test support shared by the integration tests:
//!
//! - the vectors of `spec/vectors/sumcheck/`, written by an independent Python prover, read into
//!   typed sessions, stages, tree batches and levels;
//! - the caller's side of the checks, as the stage schedule and the driver will hold it: the
//!   vectors' relation, the product of an instance's evaluations; the joining of trees of
//!   different heights, level by level; and a stand-in for the stage that consumes the leaf
//!   claims, which evaluates each tree's leaves at its leaf claim's point;
//! - conversions from the test kit's reference prover, whose proofs of seeded and of cheating
//!   instances the vectors cannot hold.
//!
//! The leaf check evaluates a multilinear extension by summing the table against `eq` from its
//! definition, an algorithm of its own: `tv-core` evaluates no table, and the reference prover
//! binds one variable at a time.

#![allow(dead_code)]

use tv_core::sumcheck::{Batch, Instance, Prefix, SumcheckError, SumcheckProof, verify_batch};
use tv_core::tree::{ProductLevel, TreeClaim, verify_product_level};
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_testkit::{Json, read_vectors, sumcheck as reference};

// ----- Reading the vectors ----------------------------------------------------------------------

/// The member `key` of a vector object, which must be a string.
pub fn text<'a>(object: &'a Json, key: &str) -> &'a str {
    object
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("`{key}` is not a string"))
}

/// The member `key` of a vector object, which must be an array.
pub fn list<'a>(object: &'a Json, key: &str) -> &'a [Json] {
    object
        .get(key)
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("`{key}` is not an array"))
}

/// The member `key` of a vector object, which must be a `u64`.
pub fn number(object: &Json, key: &str) -> u64 {
    object
        .get(key)
        .and_then(Json::as_u64)
        .unwrap_or_else(|| panic!("`{key}` is not a u64"))
}

/// An `Fp` element written `"0x"` and 16 hex digits, canonical.
pub fn fp(text: &Json) -> Fp {
    let word = text.as_hex_u64().expect("an Fp element");
    Fp::from_canonical(word).expect("a canonical Fp element")
}

/// An `Fp2` element written `[c0, c1]`.
pub fn fp2(pair: &Json) -> Fp2 {
    match pair.as_array().expect("an Fp2 element") {
        [c0, c1] => Fp2::new(fp(c0), fp(c1)),
        other => panic!("an Fp2 element has two coefficients, not {}", other.len()),
    }
}

/// A list of `Fp2` elements.
pub fn fp2s(values: &Json) -> Vec<Fp2> {
    values
        .as_array()
        .expect("a list of Fp2 elements")
        .iter()
        .map(fp2)
        .collect()
}

/// The 32 bytes of a hexadecimal digest.
pub fn digest(text: &str) -> [u8; 32] {
    tv_testkit::hex(text).try_into().expect("32 bytes")
}

/// A session context, with its text owned.
pub struct Context {
    protocol: String,
    mode: Mode,
    params: String,
    statement_digest: [u8; 32],
    session_id: [u8; 16],
}

impl Context {
    pub fn from_json(json: &Json) -> Self {
        Self {
            protocol: text(json, "protocol").to_owned(),
            mode: match text(json, "mode") {
                "Live" => Mode::Live,
                "NonInteractive" => Mode::NonInteractive,
                other => panic!("unknown mode {other}"),
            },
            params: text(json, "params").to_owned(),
            statement_digest: digest(text(json, "statement_digest")),
            session_id: tv_testkit::hex(text(json, "session_id"))
                .try_into()
                .expect("16 bytes"),
        }
    }

    pub fn session(&self) -> SessionContext<'_> {
        SessionContext {
            protocol: Ascii::new(&self.protocol).unwrap(),
            mode: self.mode,
            params: ParamSetId(Ascii::new(&self.params).unwrap()),
            statement_digest: self.statement_digest,
            session_id: self.session_id,
        }
    }

    /// A fresh transcript of this session.
    pub fn transcript(&self) -> Transcript {
        Transcript::new(&self.session())
    }

    /// A fresh transcript of this session for the reference prover.
    pub fn reference_transcript(&self) -> reference::Transcript {
        reference::Transcript::new(&self.session().to_tce())
    }
}

/// The prefix written `stage/{k}` or `tree/{ℓ}`.
pub fn prefix(text: &str) -> Prefix {
    let (kind, index) = text.split_once('/').expect("a prefix");
    let index = index.parse().expect("a decimal index");
    match kind {
        "stage" => Prefix::Stage(index),
        "tree" => Prefix::Tree(index),
        other => panic!("unknown prefix {other}"),
    }
}

/// One stage of a session: a batched sumcheck and every value the oracle recorded.
#[derive(Clone)]
pub struct Stage {
    pub prefix: Prefix,
    pub degree: usize,
    pub instances: Vec<Instance>,
    pub gamma: Fp2,
    pub proof: SumcheckProof,
    pub challenges: Vec<Fp2>,
    pub claims: Vec<Fp2>,
    pub points: Vec<Vec<Fp2>>,
    pub values: Vec<Fp2>,
    pub state: [u8; 32],
    pub counter: u64,
}

impl Stage {
    fn from_json(json: &Json) -> Self {
        let instances = list(json, "instances")
            .iter()
            .map(|instance| Instance {
                claim: fp2(instance.get("claim").unwrap()),
                vars: number(instance, "vars") as usize,
                evals: number(instance, "factors") as usize,
            })
            .collect();
        Self {
            prefix: prefix(text(json, "prefix")),
            degree: number(json, "degree") as usize,
            instances,
            gamma: fp2(json.get("gamma").unwrap()),
            proof: SumcheckProof {
                rounds: list(json, "rounds").iter().map(fp2s).collect(),
                final_evals: fp2s(json.get("final_evals").unwrap()),
            },
            challenges: fp2s(json.get("challenges").unwrap()),
            claims: fp2s(json.get("claims").unwrap()),
            points: list(json, "points").iter().map(fp2s).collect(),
            values: fp2s(json.get("values").unwrap()),
            state: digest(text(json, "state")),
            counter: number(json, "counter"),
        }
    }

    /// The number of rounds, `n_max`.
    pub fn rounds(&self) -> usize {
        self.challenges.len()
    }

    /// The round index a final check reports.
    pub fn final_round(&self) -> u32 {
        self.rounds() as u32
    }
}

/// A session: stages run one after another on one transcript.
pub struct Session {
    pub name: String,
    pub context: Context,
    pub stages: Vec<Stage>,
}

impl Session {
    /// A transcript that has run the session's stages before `stage`, each verified.
    pub fn transcript_before(&self, stage: usize) -> Transcript {
        let mut transcript = self.context.transcript();
        for earlier in &self.stages[..stage] {
            // Only the transcript an earlier stage leaves matters here.
            let _ = verify(&mut transcript, earlier, &earlier.instances, &earlier.proof)
                .unwrap_or_else(|e| panic!("{}: an earlier stage fails: {e}", self.name));
        }
        transcript
    }
}

/// The sessions of the vectors.
pub fn sessions() -> Vec<Session> {
    let file = read_vectors("sumcheck/sumcheck.json");
    list(&file, "sessions")
        .iter()
        .map(|session| Session {
            name: text(session, "name").to_owned(),
            context: Context::from_json(session.get("context").unwrap()),
            stages: list(session, "stages")
                .iter()
                .map(Stage::from_json)
                .collect(),
        })
        .collect()
}

/// A product tree of the vectors.
#[derive(Clone)]
pub struct Tree {
    pub height: usize,
    pub root: Fp,
    pub leaves: Vec<Fp>,
}

/// A claim of the vectors, with the index of its tree.
fn tree_claim(json: &Json) -> (usize, TreeClaim) {
    (
        number(json, "tree") as usize,
        TreeClaim {
            point: fp2s(json.get("point").unwrap()),
            value: fp2(json.get("value").unwrap()),
        },
    )
}

/// One level of a tree batch and every value the oracle recorded.
#[derive(Clone)]
pub struct Level {
    pub prefix: Prefix,
    pub entering: Vec<(usize, TreeClaim)>,
    pub gamma: Fp2,
    pub proof: SumcheckProof,
    pub challenges: Vec<Fp2>,
    pub claims: Vec<Fp2>,
    pub rho: Fp2,
    pub next: Vec<(usize, TreeClaim)>,
    pub state: [u8; 32],
    pub counter: u64,
}

impl Level {
    pub fn rounds(&self) -> usize {
        self.challenges.len()
    }
}

/// A batch of product trees and its levels.
pub struct TreeBatch {
    pub name: String,
    pub context: Context,
    pub trees: Vec<Tree>,
    pub levels: Vec<Level>,
}

impl TreeBatch {
    /// The level proofs.
    pub fn proofs(&self) -> Vec<SumcheckProof> {
        self.levels
            .iter()
            .map(|level| level.proof.clone())
            .collect()
    }
}

/// The tree batches of the vectors.
pub fn tree_batches() -> Vec<TreeBatch> {
    let file = read_vectors("sumcheck/sumcheck.json");
    list(&file, "tree_batches")
        .iter()
        .map(|batch| TreeBatch {
            name: text(batch, "name").to_owned(),
            context: Context::from_json(batch.get("context").unwrap()),
            trees: list(batch, "trees")
                .iter()
                .map(|tree| Tree {
                    height: number(tree, "height") as usize,
                    root: fp(tree.get("root").unwrap()),
                    leaves: list(tree, "leaves").iter().map(fp).collect(),
                })
                .collect(),
            levels: list(batch, "levels")
                .iter()
                .map(|level| Level {
                    prefix: prefix(text(level, "prefix")),
                    entering: list(level, "instances").iter().map(tree_claim).collect(),
                    gamma: fp2(level.get("gamma").unwrap()),
                    proof: SumcheckProof {
                        rounds: list(level, "rounds").iter().map(fp2s).collect(),
                        final_evals: fp2s(level.get("final_evals").unwrap()),
                    },
                    challenges: fp2s(level.get("challenges").unwrap()),
                    claims: fp2s(level.get("claims").unwrap()),
                    rho: fp2(level.get("rho").unwrap()),
                    next: list(level, "next").iter().map(tree_claim).collect(),
                    state: digest(text(level, "state")),
                    counter: number(level, "counter"),
                })
                .collect(),
        })
        .collect()
}

// ----- The caller's side ------------------------------------------------------------------------

/// The vectors' relation: an instance's relation is the product of its tables, so its value at
/// its point is the product of its evaluations.
pub fn product(_: usize, _: &[Fp2], evals: &[Fp2]) -> Option<Fp2> {
    Some(evals.iter().fold(Fp2::ONE, |p, e| p * *e))
}

/// Verifies `proof` as stage `stage` with `instances`, under the product relation.
pub fn verify(
    transcript: &mut Transcript,
    stage: &Stage,
    instances: &[Instance],
    proof: &SumcheckProof,
) -> Result<Batch, SumcheckError> {
    verify_batch(
        transcript,
        stage.prefix,
        stage.degree,
        instances,
        proof,
        product,
    )
}

/// `f̃(point) = Σ_x f[x]·eq(point, x)`, `x_1` the lowest bit of `x`: the multilinear extension
/// from its definition.
pub fn extension(table: &[Fp], point: &[Fp2]) -> Fp2 {
    assert_eq!(table.len(), 1 << point.len());
    let mut sum = Fp2::ZERO;
    for (x, &entry) in table.iter().enumerate() {
        let mut weight = Fp2::ONE;
        for (i, &r) in point.iter().enumerate() {
            weight *= if (x >> i) & 1 == 1 { r } else { Fp2::ONE - r };
        }
        sum += weight * entry;
    }
    sum
}

/// What a batch of trees comes to, its levels run by [`run_trees`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeOutcome {
    /// Every level passed and every leaf claim is true.
    Accepted,
    /// Level `level` was rejected with `error`.
    Rejected { level: u32, error: SumcheckError },
    /// Every level passed, but the leaf claim of tree `tree` is false, which the stage that
    /// consumes it rejects.
    FalseLeaf { tree: usize },
}

/// A level that passed in a run of [`run_trees`].
pub struct LevelRun {
    /// The claims that entered it, by tree, in schedule order.
    pub entering: Vec<(usize, TreeClaim)>,
    /// What it returned.
    pub outcome: ProductLevel,
    /// The transcript's state and squeeze count after it.
    pub state: [u8; 32],
    pub counter: u64,
}

/// A run of every level of a batch of trees.
pub struct TreeRun {
    /// Each level that passed.
    pub levels: Vec<LevelRun>,
    /// The leaf claims, by tree, when every level passed.
    pub leaves: Vec<(usize, TreeClaim)>,
    pub outcome: TreeOutcome,
}

/// Runs every level of a batch of `trees` with `proofs` as the caller does: at level `ℓ`, with
/// `H` the tallest height, the trees of height `h ≥ H − ℓ` are present in the order of `trees`,
/// a tree joining at level `H − h` with its root claim on no variable, and the others with the
/// claims the level before returned. `tamper` may change the claims entering a level. After
/// the last level, each leaf claim is checked against its tree's leaves.
pub fn run_trees(
    transcript: &mut Transcript,
    trees: &[Tree],
    proofs: &[SumcheckProof],
    mut tamper: impl FnMut(u32, &mut [TreeClaim]),
) -> TreeRun {
    let tallest = trees.iter().map(|tree| tree.height).max().unwrap();
    assert_eq!(proofs.len(), tallest, "one proof per level");
    let mut current: Vec<Option<TreeClaim>> = vec![None; trees.len()];
    let mut levels = Vec::new();
    for (level, proof) in proofs.iter().enumerate() {
        for (index, tree) in trees.iter().enumerate() {
            if tallest - tree.height == level {
                current[index] = Some(TreeClaim {
                    point: Vec::new(),
                    value: Fp2::from(tree.root),
                });
            }
        }
        let present: Vec<usize> = (0..trees.len()).filter(|&t| current[t].is_some()).collect();
        let mut claims: Vec<TreeClaim> = present
            .iter()
            .map(|&t| current[t].clone().unwrap())
            .collect();
        tamper(level as u32, &mut claims);
        match verify_product_level(transcript, level as u32, &claims, proof) {
            Err(error) => {
                return TreeRun {
                    levels,
                    leaves: Vec::new(),
                    outcome: TreeOutcome::Rejected {
                        level: level as u32,
                        error,
                    },
                };
            }
            Ok(outcome) => {
                assert_eq!(outcome.claims.len(), present.len());
                for (&t, claim) in present.iter().zip(&outcome.claims) {
                    current[t] = Some(claim.clone());
                }
                levels.push(LevelRun {
                    entering: present.iter().copied().zip(claims).collect(),
                    outcome,
                    state: *transcript.state(),
                    counter: transcript.counter(),
                });
            }
        }
    }
    let leaves: Vec<(usize, TreeClaim)> = current
        .into_iter()
        .enumerate()
        .map(|(t, claim)| (t, claim.unwrap()))
        .collect();
    let outcome = leaves
        .iter()
        .find(|(t, claim)| extension(&trees[*t].leaves, &claim.point) != claim.value)
        .map_or(TreeOutcome::Accepted, |(t, _)| TreeOutcome::FalseLeaf {
            tree: *t,
        });
    TreeRun {
        levels,
        leaves,
        outcome,
    }
}

// ----- The reference prover ---------------------------------------------------------------------

/// A reference-prover element in `Fp2`.
pub fn from_reference(value: reference::Fe) -> Fp2 {
    Fp2::new(
        Fp::from_canonical(value.0).unwrap(),
        Fp::from_canonical(value.1).unwrap(),
    )
}

/// A reference proof as the verifier receives it.
pub fn proof_of(proof: &reference::Proof) -> SumcheckProof {
    SumcheckProof {
        rounds: proof
            .rounds
            .iter()
            .map(|round| round.iter().copied().map(from_reference).collect())
            .collect(),
        final_evals: proof.evals.iter().copied().map(from_reference).collect(),
    }
}

/// The instances of a reference batch, as the schedule gives them to the verifier.
pub fn instances_of(products: &[reference::Product]) -> Vec<Instance> {
    products
        .iter()
        .map(|product| Instance {
            claim: from_reference(product.claim()),
            vars: product.vars(),
            evals: product.tables.len(),
        })
        .collect()
}
