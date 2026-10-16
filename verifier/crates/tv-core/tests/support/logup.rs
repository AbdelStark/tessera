// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Test support for LogUp-GKR's leaf stage:
//!
//! - the `logup_batches` of `spec/vectors/sumcheck/`, written by an independent Python prover,
//!   read into typed batches, levels and leaf checks;
//! - the caller's side of the checks, as the verifier's driver will hold it, written from
//!   RFC-0008 'Fraction tree' and RFC-0010 'Tree levels': `α`, `β` and the roots; the joining
//!   of lookup trees, table trees and product forests of different heights, level by level, with
//!   the forests' root challenges and root claims; the table evaluation `t̃(ρ_t)` from the table's
//!   rows; the leaf checks; and stand-ins for the stages and openings that take the claims left,
//!   which evaluate the key, value, multiplicity and forest leaf polynomials at their points;
//! - the fraction trees of an instance, built here from its rows, lookups and multiplicities, so
//!   that every claim can be checked against its tree's level;
//! - conversions from the test kit's reference prover of the leaf stage, whose proofs of seeded
//!   and of cheating batches the vectors cannot hold.
//!
//! Every multilinear extension here is a sum against `eq` from its definition: `tv-core`
//! evaluates no table, and the reference prover binds one variable at a time.

use std::rc::Rc;
use tv_core::logup::{LeafClaims, Leaves, LookupError, Roots, check_leaves, check_roots};
use tv_core::sumcheck::{Prefix, SumcheckError, SumcheckProof};
use tv_core::tree::{FractionClaim, LevelClaim, TreeClaim, TreeLevel, verify_level};
use tv_field::{Fp, Fp2};

use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_testkit::logup as reference;
use tv_testkit::sumcheck::Fe;
use tv_testkit::{Json, SplitMix64};

use super::{
    Context, Forest, digest, fp, fp2, fp2s, from_reference, list, number, prefix, root_claim, text,
};

// ----- The batches ------------------------------------------------------------------------------

/// A LogUp instance: its table, its lookups and its multiplicities (RFC-0008 'Relation').
#[derive(Clone, Debug)]
pub struct Instance {
    /// The table's `T ≥ 1` rows `(key, value)`; a table without values has every value 0.
    pub rows: Vec<(Fp, Fp)>,
    /// Whether the values message holds this instance's `v`.
    pub has_values: bool,
    /// The `N ≥ 1` lookups, in leaf order.
    pub lookups: Vec<Lookup>,
    /// `m_j` per table-tree leaf, `2^{h_t}` of them.
    pub multiplicities: Vec<Fp>,
}

/// A lookup: the index of its row, or, for a cheating prover's lookup that is no row, its pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    /// Row `j` of the table.
    Row(usize),
    /// `(key, value)`, which is no row.
    Pair(Fp, Fp),
}

/// `⌈log2 n⌉`, for `n ≥ 1`.
pub fn ceil_log2(n: usize) -> usize {
    assert!(n >= 1);
    n.next_power_of_two().trailing_zeros() as usize
}

impl Instance {
    /// `h_a = ⌈log2 N⌉`, from the number of lookups.
    pub fn lookup_height(&self) -> usize {
        ceil_log2(self.lookups.len())
    }

    /// `h_t = ⌈log2 T⌉`, from the number of rows.
    pub fn table_height(&self) -> usize {
        ceil_log2(self.rows.len())
    }

    /// An honest instance over `rows`: the lookups are the rows `lookups`, and each multiplicity
    /// counts them, row 0's the lookup tree's padding too.
    pub fn honest(rows: Vec<(Fp, Fp)>, has_values: bool, lookups: Vec<usize>) -> Self {
        let mut counts = vec![0u64; 1 << ceil_log2(rows.len())];
        for &j in &lookups {
            counts[j] += 1;
        }
        counts[0] += ((1usize << ceil_log2(lookups.len())) - lookups.len()) as u64;
        Self {
            rows,
            has_values,
            lookups: lookups.into_iter().map(Lookup::Row).collect(),
            multiplicities: counts.into_iter().map(Fp::reduce).collect(),
        }
    }

    /// Lookup `i`'s `(key, value)`.
    pub fn pair(&self, i: usize) -> (Fp, Fp) {
        match self.lookups[i] {
            Lookup::Row(j) => self.rows[j],
            Lookup::Pair(key, value) => (key, value),
        }
    }

    /// `key + β·value` of row `j`.
    pub fn row(&self, j: usize, beta: Fp2) -> Fp2 {
        let (key, value) = self.rows[j];
        Fp2::from(key) + beta * value
    }

    /// The lookup side's keys and values, `2^{h_a}` each, the padding slots holding row 0's.
    pub fn lookup_pairs(&self) -> Vec<(Fp, Fp)> {
        let mut pairs: Vec<(Fp, Fp)> = (0..self.lookups.len()).map(|i| self.pair(i)).collect();
        pairs.resize(1 << self.lookup_height(), self.rows[0]);
        pairs
    }

    /// The lookup tree's leaves `(1, α − a_i)`, padded with `(1, α − t_0)`.
    pub fn lookup_leaves(&self, alpha: Fp2, beta: Fp2) -> Vec<(Fp2, Fp2)> {
        self.lookup_pairs()
            .into_iter()
            .map(|(key, value)| (Fp2::ONE, alpha - (Fp2::from(key) + beta * value)))
            .collect()
    }

    /// The table's `2^{h_t}` rows, its padding rows copies of row 0 (RFC-0008 'Fraction tree').
    pub fn padded_rows(&self) -> Vec<(Fp, Fp)> {
        let mut rows = self.rows.clone();
        rows.resize(1 << self.table_height(), self.rows[0]);
        rows
    }

    /// The table tree's leaves `(−m_j, α − t_j)` over the padded rows, a padding row's
    /// `(−m_j, α − t_0)`.
    pub fn table_leaves(&self, alpha: Fp2, beta: Fp2) -> Vec<(Fp2, Fp2)> {
        self.padded_rows()
            .into_iter()
            .zip(&self.multiplicities)
            .map(|((key, value), &m)| (-Fp2::from(m), alpha - (Fp2::from(key) + beta * value)))
            .collect()
    }

    /// `t̃(point)`, the table polynomial `key + β·value` over the padded rows, at `point`.
    pub fn table_eval(&self, beta: Fp2, point: &[Fp2]) -> Fp2 {
        let table: Vec<Fp2> = self
            .padded_rows()
            .into_iter()
            .map(|(key, value)| Fp2::from(key) + beta * value)
            .collect();
        extension2(&table, point)
    }
}

/// `f̃(point) = Σ_x f[x]·eq(point, x)` for a table of `Fp2` values, `x_1` the lowest bit of `x`.
pub fn extension2(table: &[Fp2], point: &[Fp2]) -> Fp2 {
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

/// Every level of a fraction tree over `leaves`, the root first:
/// `(n, d) = (n_L·d_R + n_R·d_L, d_L·d_R)`, the left child the even entry.
pub fn fraction_levels(leaves: Vec<(Fp2, Fp2)>) -> Vec<Vec<(Fp2, Fp2)>> {
    let mut levels = vec![leaves];
    while levels[0].len() > 1 {
        let above = levels[0]
            .chunks_exact(2)
            .map(|pair| {
                let ((n_l, d_l), (n_r, d_r)) = (pair[0], pair[1]);
                (n_l * d_r + n_r * d_l, d_l * d_r)
            })
            .collect();
        levels.insert(0, above);
    }
    levels
}

/// Whose a tree of the schedule is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// An instance's lookup tree.
    Lookup,
    /// An instance's table tree.
    Table,
    /// A product forest.
    Forest,
}

/// A tree in the schedule order: its kind and its index into the instances or the forests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TreeRef {
    pub kind: Kind,
    pub index: usize,
}

/// One level of a LogUp batch and every value the oracle recorded.
#[derive(Clone, Debug)]
pub struct LogUpLevel {
    pub prefix: Prefix,
    pub root_challenges: Vec<Fp2>,
    /// The claims entering it, each with its tree's index in `trees`.
    pub entering: Vec<(usize, LevelClaim)>,
    pub gamma: Fp2,
    pub lambda: Option<Fp2>,
    pub proof: SumcheckProof,
    pub challenges: Vec<Fp2>,
    pub claims: Vec<Fp2>,
    pub rho: Fp2,
    pub next: Vec<(usize, LevelClaim)>,
    pub state: [u8; 32],
    pub counter: u64,
}

impl LogUpLevel {
    pub fn rounds(&self) -> usize {
        self.challenges.len()
    }
}

/// The lookup tree's side of a leaf check.
#[derive(Clone, Debug)]
pub struct LookupLeaf {
    pub point: Vec<Fp2>,
    pub numerator: Fp2,
    pub denominator: Fp2,
    /// `v`, or `None` without values.
    pub value: Option<Fp2>,
    /// `α − q − β·v`.
    pub key: Fp2,
}

/// The table tree's side of a leaf check.
#[derive(Clone, Debug)]
pub struct TableLeaf {
    pub point: Vec<Fp2>,
    pub numerator: Fp2,
    /// `−p`.
    pub multiplicity: Fp2,
    pub denominator: Fp2,
    /// `t̃(ρ_t)`.
    pub table_eval: Fp2,
}

/// One instance's leaf checks as the oracle recorded them.
#[derive(Clone, Debug)]
pub struct LeafCheck {
    pub lookup: LookupLeaf,
    pub table: TableLeaf,
}

/// What a batch must come to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expected {
    Accept,
    /// `reason` is `Lookup` or `Sumcheck`; `check` is `roots`, `leaf numerator`,
    /// `table evaluation` or `level ℓ`; `instance` is the LogUp instance at fault, or the
    /// level's instance for a sumcheck.
    Reject {
        reason: String,
        check: String,
        instance: usize,
    },
}

/// A LogUp batch of the vectors: one leaf stage on one transcript.
#[derive(Clone)]
pub struct LogUpBatch {
    pub name: String,
    pub context: Rc<Context>,
    pub alpha: Fp2,
    pub beta: Fp2,
    pub instances: Vec<Instance>,
    pub forests: Vec<Forest>,
    pub trees: Vec<TreeRef>,
    pub roots: Vec<Fp2>,
    pub levels: Vec<LogUpLevel>,
    pub values: Vec<Fp2>,
    pub leaf_checks: Vec<LeafCheck>,
    pub state: Option<[u8; 32]>,
    pub counter: Option<u64>,
    pub expected: Expected,
}

/// What the prover sends in a leaf stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogUpProof {
    /// The roots message.
    pub roots: Vec<Fp2>,
    /// Each level's sumcheck.
    pub levels: Vec<SumcheckProof>,
    /// The values message.
    pub values: Vec<Fp2>,
}

impl LogUpBatch {
    /// The proof the batch records.
    pub fn proof(&self) -> LogUpProof {
        LogUpProof {
            roots: self.roots.clone(),
            levels: self
                .levels
                .iter()
                .map(|level| level.proof.clone())
                .collect(),
            values: self.values.clone(),
        }
    }

    /// The tree heights, in schedule order, from the instances' sizes and the forests.
    pub fn heights(&self) -> Vec<usize> {
        heights(&self.instances, &self.forests, &self.trees)
    }

    /// Runs the batch's own proof.
    pub fn run(&self) -> LogUpRun {
        self.run_proof(&self.proof())
    }

    /// Runs `proof` in the batch's context, on its instances, forests and trees.
    pub fn run_proof(&self, proof: &LogUpProof) -> LogUpRun {
        let mut transcript = self.context.transcript();
        run_logup(
            &mut transcript,
            &self.instances,
            &self.forests,
            &self.trees,
            proof,
            |_, _| {},
        )
    }
}

/// The trees' heights in schedule order.
pub fn heights(instances: &[Instance], forests: &[Forest], trees: &[TreeRef]) -> Vec<usize> {
    trees
        .iter()
        .map(|tree| match tree.kind {
            Kind::Lookup => instances[tree.index].lookup_height(),
            Kind::Table => instances[tree.index].table_height(),
            Kind::Forest => forests[tree.index].height,
        })
        .collect()
}

fn boolean(json: &Json, key: &str) -> bool {
    match json.get(key) {
        Some(Json::Bool(value)) => *value,
        _ => panic!("`{key}` is not a boolean"),
    }
}

fn optional_fp2(json: &Json, key: &str) -> Option<Fp2> {
    json.get(key).filter(|value| !value.is_null()).map(fp2)
}

/// A level claim of the vectors: `{tree, point, value}` for a forest, `{tree, point, numerator,
/// denominator}` for a fraction tree.
fn level_claim(json: &Json) -> (usize, LevelClaim) {
    let point = fp2s(json.get("point").unwrap());
    let claim = match json.get("value") {
        Some(value) => LevelClaim::Product(TreeClaim {
            point,
            value: fp2(value),
        }),
        None => LevelClaim::Fraction(FractionClaim {
            point,
            numerator: fp2(json.get("numerator").unwrap()),
            denominator: fp2(json.get("denominator").unwrap()),
        }),
    };
    (number(json, "tree") as usize, claim)
}

fn logup_level(json: &Json) -> LogUpLevel {
    let claims = |key: &str| -> Vec<(usize, LevelClaim)> {
        list(json, key).iter().map(level_claim).collect()
    };
    LogUpLevel {
        prefix: prefix(text(json, "prefix")),
        root_challenges: fp2s(json.get("root_challenges").unwrap()),
        entering: claims("instances"),
        gamma: fp2(json.get("gamma").unwrap()),
        lambda: optional_fp2(json, "lambda"),
        proof: SumcheckProof {
            rounds: list(json, "rounds").iter().map(fp2s).collect(),
            final_evals: fp2s(json.get("final_evals").unwrap()),
        },
        challenges: fp2s(json.get("challenges").unwrap()),
        claims: fp2s(json.get("claims").unwrap()),
        rho: fp2(json.get("rho").unwrap()),
        next: claims("next"),
        state: digest(text(json, "state")),
        counter: number(json, "counter"),
    }
}

fn leaf_check(json: &Json) -> LeafCheck {
    let lookup = json.get("lookup").unwrap();
    let table = json.get("table").unwrap();
    LeafCheck {
        lookup: LookupLeaf {
            point: fp2s(lookup.get("point").unwrap()),
            numerator: fp2(lookup.get("numerator").unwrap()),
            denominator: fp2(lookup.get("denominator").unwrap()),
            value: optional_fp2(lookup, "value"),
            key: fp2(lookup.get("key").unwrap()),
        },
        table: TableLeaf {
            point: fp2s(table.get("point").unwrap()),
            numerator: fp2(table.get("numerator").unwrap()),
            multiplicity: fp2(table.get("multiplicity").unwrap()),
            denominator: fp2(table.get("denominator").unwrap()),
            table_eval: fp2(table.get("table_eval").unwrap()),
        },
    }
}

fn forest(json: &Json) -> Forest {
    let forest = Forest {
        rows_log: number(json, "rows_log") as usize,
        height: number(json, "height") as usize,
        roots: list(json, "roots").iter().map(fp).collect(),
        leaves: list(json, "leaves").iter().map(fp).collect(),
    };
    assert!(forest.height >= 1 && forest.rows_log + forest.height < 32);
    assert_eq!(forest.roots.len(), 1 << forest.rows_log);
    assert_eq!(forest.leaves.len(), 1 << (forest.rows_log + forest.height));
    forest
}

fn instance(json: &Json) -> Instance {
    let instance = Instance {
        rows: list(json, "rows")
            .iter()
            .map(|row| match row.as_array().expect("a row") {
                [key, value] => (fp(key), fp(value)),
                other => panic!("a row is a pair, not {} values", other.len()),
            })
            .collect(),
        has_values: boolean(json, "has_values"),
        lookups: list(json, "lookups")
            .iter()
            .map(|lookup| match lookup {
                Json::Array(pair) => match &pair[..] {
                    [key, value] => Lookup::Pair(fp(key), fp(value)),
                    other => panic!("a lookup's pair has two values, not {}", other.len()),
                },
                index => Lookup::Row(index.as_u64().expect("a row index") as usize),
            })
            .collect(),
        multiplicities: list(json, "multiplicities").iter().map(fp).collect(),
    };
    // Small shapes only, and the heights the RFC derives from the sizes.
    assert!(!instance.rows.is_empty() && !instance.lookups.is_empty());
    assert_eq!(number(json, "h_a") as usize, instance.lookup_height());
    assert_eq!(number(json, "h_t") as usize, instance.table_height());
    assert!(instance.table_height() < 24 && instance.lookup_height() < 24);
    assert_eq!(instance.multiplicities.len(), 1 << instance.table_height());
    instance
}

fn tree_ref(json: &Json) -> TreeRef {
    TreeRef {
        kind: match text(json, "kind") {
            "lookup" => Kind::Lookup,
            "table" => Kind::Table,
            "forest" => Kind::Forest,
            other => panic!("unknown tree kind {other}"),
        },
        index: number(json, "index") as usize,
    }
}

fn expected(json: &Json) -> Expected {
    match json {
        Json::String(text) if text == "accept" => Expected::Accept,
        Json::Object(_) => Expected::Reject {
            reason: text(json, "reason").to_owned(),
            check: text(json, "check").to_owned(),
            instance: number(json, "instance") as usize,
        },
        other => panic!("unknown expectation {other:?}"),
    }
}

/// The LogUp batches of the vectors, or `None` while the file has none.
pub fn logup_batches() -> Option<Vec<LogUpBatch>> {
    let file = tv_testkit::read_vectors("sumcheck/sumcheck.json");
    let batches = file.get("logup_batches")?.as_array()?;
    Some(
        batches
            .iter()
            .map(|batch| {
                let instances: Vec<Instance> =
                    list(batch, "instances").iter().map(instance).collect();
                LogUpBatch {
                    name: text(batch, "name").to_owned(),
                    context: Rc::new(Context::from_json(batch.get("context").unwrap())),
                    alpha: fp2(batch.get("alpha").unwrap()),
                    beta: fp2(batch.get("beta").unwrap()),
                    instances,
                    forests: list(batch, "forests").iter().map(forest).collect(),
                    trees: list(batch, "trees").iter().map(tree_ref).collect(),
                    roots: fp2s(batch.get("roots").unwrap()),
                    levels: batch
                        .get("levels")
                        .and_then(Json::as_array)
                        .map(|levels| levels.iter().map(logup_level).collect())
                        .unwrap_or_default(),
                    values: batch.get("values").map(fp2s).unwrap_or_default(),
                    leaf_checks: batch
                        .get("leaf_checks")
                        .and_then(Json::as_array)
                        .map(|checks| checks.iter().map(leaf_check).collect())
                        .unwrap_or_default(),
                    state: batch.get("state").and_then(Json::as_str).map(digest),
                    counter: batch.get("counter").and_then(Json::as_u64),
                    expected: expected(batch.get("expected").unwrap()),
                }
            })
            .collect(),
    )
}

// ----- The caller's side ------------------------------------------------------------------------

/// A claim left by the leaf checks, or a forest's leaf claim, that is false: the stage or the
/// opening that takes it rejects it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FalseClaim {
    /// Instance `i`'s key claim.
    Key(usize),
    /// Instance `i`'s value claim.
    Value(usize),
    /// Instance `i`'s multiplicity claim.
    Multiplicity(usize),
    /// Forest `f`'s leaf claim.
    ForestLeaf(usize),
}

/// What a leaf stage comes to, run by [`run_logup`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogUpOutcome {
    /// Every check passed and every claim left is true.
    Accepted,
    /// The roots were rejected, or `α` or `β` found no value.
    Roots(LookupError),
    /// Level `level`'s root challenges found no value, a fault of the verifier.
    Inconclusive { level: u32, why: String },
    /// Level `level` was rejected.
    Level { level: u32, error: SumcheckError },
    /// The leaf checks were rejected.
    Leaves(LookupError),
    /// Every check passed, but a claim left for later is false.
    False(FalseClaim),
}

/// A level that passed in a run of [`run_logup`].
#[derive(Clone, Debug)]
pub struct LogUpLevelRun {
    pub root_challenges: Vec<Fp2>,
    /// The claims that entered it, by tree, in schedule order.
    pub entering: Vec<(usize, LevelClaim)>,
    pub outcome: TreeLevel,
    pub state: [u8; 32],
    pub counter: u64,
}

/// A run of a leaf stage.
#[derive(Clone, Debug)]
pub struct LogUpRun {
    /// What the roots returned.
    pub roots: Option<Roots>,
    /// Each level that passed.
    pub levels: Vec<LogUpLevelRun>,
    /// The claims that entered the level that was rejected, by tree, in schedule order.
    pub rejected: Vec<(usize, LevelClaim)>,
    /// Each tree's claims after its last level, in schedule order, when every level passed.
    pub leaves: Vec<LevelClaim>,
    /// `t̃(ρ_t)` per instance, as the leaf checks asked for it.
    pub table_evals: Vec<Fp2>,
    /// What the leaf checks returned.
    pub claims: Option<LeafClaims>,
    /// The transcript after the values, when the leaf checks ran.
    pub state: Option<[u8; 32]>,
    pub counter: Option<u64>,
    pub outcome: LogUpOutcome,
}

/// Runs a leaf stage as a verifier's driver does, from RFC-0008 'Fraction tree' and RFC-0010
/// 'Tree levels':
///
/// 1. [`check_roots`] squeezes `α` and `β` and checks the roots message.
/// 2. With `H` the tallest tree's height, at level `ℓ` the forests of height `H − ℓ` join under
///    the root challenges `tree/{ℓ}/root/{c}`, `c < s_max`, each at the last `s` of them with its
///    root claim `Σ_y v_y·eq(r, y)`; the fraction trees of height `H − ℓ ≥ 1` join with their
///    root claims; every tree that has joined is present, in the order of `trees`, and
///    [`verify_level`] checks the level. `tamper` may change the claims entering a level, or
///    their number.
/// 3. A fraction tree of height 0 joins no level: its root claims are its leaf claims.
/// 4. [`check_leaves`] checks the values and each instance's leaves, `t̃(ρ_t)` computed here from
///    the rows.
/// 5. The stand-ins for the later stages: each instance's key, value and multiplicity claims, and
///    each forest's leaf claim, against the polynomials computed here.
pub fn run_logup(
    transcript: &mut Transcript,
    instances: &[Instance],
    forests: &[Forest],
    trees: &[TreeRef],
    proof: &LogUpProof,
    mut tamper: impl FnMut(u32, &mut Vec<LevelClaim>),
) -> LogUpRun {
    let mut run = LogUpRun {
        roots: None,
        levels: Vec::new(),
        rejected: Vec::new(),
        leaves: Vec::new(),
        table_evals: Vec::new(),
        claims: None,
        state: None,
        counter: None,
        outcome: LogUpOutcome::Accepted,
    };
    let roots = match check_roots(transcript, instances.len(), &proof.roots) {
        Ok(roots) => roots,
        Err(error) => {
            run.outcome = LogUpOutcome::Roots(error);
            return run;
        }
    };
    run.roots = Some(roots.clone());
    let challenges = roots.challenges;
    let heights = heights(instances, forests, trees);
    let tallest = heights.iter().copied().max().unwrap_or(0);
    assert_eq!(proof.levels.len(), tallest, "one proof per level");
    let root_of = |tree: &TreeRef| -> Option<FractionClaim> {
        let claims = roots.instances.get(tree.index)?;
        match tree.kind {
            Kind::Lookup => Some(claims.lookup.clone()),
            Kind::Table => Some(claims.table.clone()),
            Kind::Forest => None,
        }
    };
    let mut current: Vec<Option<LevelClaim>> = vec![None; trees.len()];
    for (level, level_proof) in proof.levels.iter().enumerate() {
        let index = level as u32;
        let joins = |t: usize| heights[t] > 0 && tallest - heights[t] == level;
        let s_max = (0..trees.len())
            .filter(|&t| trees[t].kind == Kind::Forest && joins(t))
            .map(|t| forests[trees[t].index].rows_log)
            .max()
            .unwrap_or(0);
        let mut root_challenges = Vec::with_capacity(s_max);
        for c in 0..s_max {
            let label = format!("tree/{level}/root/{c}");
            match transcript.challenge_fp2(Ascii::new(&label).unwrap()) {
                Ok(challenge) => root_challenges.push(challenge),
                Err(error) => {
                    run.outcome = LogUpOutcome::Inconclusive {
                        level: index,
                        why: error.to_string(),
                    };
                    return run;
                }
            }
        }
        for t in (0..trees.len()).filter(|&t| joins(t)) {
            current[t] = Some(match trees[t].kind {
                Kind::Forest => {
                    let forest = &forests[trees[t].index];
                    let point = root_challenges[s_max - forest.rows_log..].to_vec();
                    let value = root_claim(&forest.roots, &point).expect("2^s roots");
                    LevelClaim::Product(TreeClaim { point, value })
                }
                _ => LevelClaim::Fraction(root_of(&trees[t]).expect("an instance's root")),
            });
        }
        let present: Vec<usize> = (0..trees.len()).filter(|&t| current[t].is_some()).collect();
        let mut claims: Vec<LevelClaim> = present
            .iter()
            .map(|&t| current[t].clone().unwrap())
            .collect();
        tamper(index, &mut claims);
        match verify_level(transcript, index, &claims, level_proof) {
            Err(error) => {
                run.outcome = LogUpOutcome::Level {
                    level: index,
                    error,
                };
                run.rejected = present.iter().copied().zip(claims).collect();
                return run;
            }
            Ok(outcome) => {
                assert_eq!(
                    outcome.claims.len(),
                    present.len(),
                    "a level whose instances were added or dropped passed"
                );
                for (&t, claim) in present.iter().zip(&outcome.claims) {
                    current[t] = Some(claim.clone());
                }
                run.levels.push(LogUpLevelRun {
                    root_challenges,
                    entering: present.iter().copied().zip(claims).collect(),
                    outcome,
                    state: *transcript.state(),
                    counter: transcript.counter(),
                });
            }
        }
    }
    run.leaves = (0..trees.len())
        .map(|t| match &current[t] {
            Some(claim) => claim.clone(),
            None => {
                assert_eq!(heights[t], 0, "every tree of height at least 1 joined");
                LevelClaim::Fraction(root_of(&trees[t]).expect("a fraction tree"))
            }
        })
        .collect();
    let fraction_of = |kind: Kind, i: usize| -> FractionClaim {
        let t = trees
            .iter()
            .position(|tree| *tree == TreeRef { kind, index: i })
            .expect("every instance's trees are scheduled");
        match &run.leaves[t] {
            LevelClaim::Fraction(claim) => claim.clone(),
            LevelClaim::Product(_) => panic!("a fraction tree's claims"),
        }
    };
    let leaves: Vec<Leaves> = instances
        .iter()
        .enumerate()
        .map(|(i, instance)| Leaves {
            lookup: fraction_of(Kind::Lookup, i),
            table: fraction_of(Kind::Table, i),
            has_values: instance.has_values,
        })
        .collect();
    let mut table_evals = Vec::new();
    let checked = check_leaves(
        transcript,
        &challenges,
        &leaves,
        &proof.values,
        |i, point| {
            let evaluation = instances.get(i)?.table_eval(challenges.beta, point);
            table_evals.push(evaluation);
            Some(evaluation)
        },
    );
    run.table_evals = table_evals;
    run.state = Some(*transcript.state());
    run.counter = Some(transcript.counter());
    let claims = match checked {
        Ok(claims) => claims,
        Err(error) => {
            run.outcome = LogUpOutcome::Leaves(error);
            return run;
        }
    };
    run.outcome = false_claim(instances, forests, trees, &run.leaves, &claims)
        .map_or(LogUpOutcome::Accepted, LogUpOutcome::False);
    run.claims = Some(claims);
    run
}

/// The first claim left for later that is false, the instances' in order (key, value,
/// multiplicity), then the forests' leaf claims: the stand-ins for the stages and openings that
/// take them, from the instances' polynomials and the forests' stacked leaves.
fn false_claim(
    instances: &[Instance],
    forests: &[Forest],
    trees: &[TreeRef],
    leaves: &[LevelClaim],
    claims: &LeafClaims,
) -> Option<FalseClaim> {
    for (i, (instance, claims)) in instances.iter().zip(&claims.instances).enumerate() {
        let pairs = instance.lookup_pairs();
        let keys: Vec<Fp2> = pairs.iter().map(|&(key, _)| Fp2::from(key)).collect();
        let values: Vec<Fp2> = pairs.iter().map(|&(_, value)| Fp2::from(value)).collect();
        let multiplicities: Vec<Fp2> = instance
            .multiplicities
            .iter()
            .copied()
            .map(Fp2::from)
            .collect();
        if claims.key.point.len() != instance.lookup_height()
            || extension2(&keys, &claims.key.point) != claims.key.value
        {
            return Some(FalseClaim::Key(i));
        }
        match &claims.value {
            Some(claim) => {
                if !instance.has_values
                    || claim.point.len() != instance.lookup_height()
                    || extension2(&values, &claim.point) != claim.value
                {
                    return Some(FalseClaim::Value(i));
                }
            }
            None => {
                if instance.has_values {
                    return Some(FalseClaim::Value(i));
                }
            }
        }
        if claims.multiplicity.point.len() != instance.table_height()
            || extension2(&multiplicities, &claims.multiplicity.point) != claims.multiplicity.value
        {
            return Some(FalseClaim::Multiplicity(i));
        }
    }
    for (tree, leaf) in trees.iter().zip(leaves) {
        if tree.kind != Kind::Forest {
            continue;
        }
        let forest = &forests[tree.index];
        let holds = match leaf {
            LevelClaim::Product(claim) => {
                claim.point.len() == forest.rows_log + forest.height
                    && super::extension(&forest.leaves, &claim.point) == claim.value
            }
            LevelClaim::Fraction(_) => false,
        };
        if !holds {
            return Some(FalseClaim::ForestLeaf(tree.index));
        }
    }
    None
}

// ----- The reference prover ---------------------------------------------------------------------

fn to_fe(value: Fp) -> Fe {
    Fe::base(value.to_canonical())
}

fn to_fe2(value: Fp2) -> Fe {
    Fe(value.c0.to_canonical(), value.c1.to_canonical())
}

/// The reference prover's instance for `instance`: its rows, lookups and multiplicities as they
/// are, so that a cheating instance stays one.
pub fn reference_logup(instance: &Instance) -> reference::LogUp {
    let rows: Vec<(Fe, Fe)> = instance
        .rows
        .iter()
        .map(|&(key, value)| (to_fe(key), to_fe(value)))
        .collect();
    reference::LogUp {
        lookups: (0..instance.lookups.len())
            .map(|i| {
                let (key, value) = instance.pair(i);
                (to_fe(key), to_fe(value))
            })
            .collect(),
        rows,
        has_values: instance.has_values,
        multiplicities: instance.multiplicities.iter().copied().map(to_fe).collect(),
    }
}

/// The reference prover's forest for `forest`.
pub fn reference_forest(forest: &Forest) -> reference::Forest {
    reference::Forest::new(
        forest.rows_log,
        forest.leaves.iter().copied().map(to_fe).collect(),
    )
}

/// The reference prover's schedule for `trees`.
pub fn reference_trees(trees: &[TreeRef]) -> Vec<reference::StageTree> {
    trees
        .iter()
        .map(|tree| match tree.kind {
            Kind::Lookup => reference::StageTree::Lookup(tree.index),
            Kind::Table => reference::StageTree::Table(tree.index),
            Kind::Forest => reference::StageTree::Forest(tree.index),
        })
        .collect()
}

/// A reference claim as the verifier holds it: a forest's or a fraction tree's.
pub fn claim_of(claim: &reference::StageClaim) -> LevelClaim {
    let point: Vec<Fp2> = claim.point.iter().copied().map(from_reference).collect();
    match claim.values[..] {
        [value] => LevelClaim::Product(TreeClaim {
            point,
            value: from_reference(value),
        }),
        [numerator, denominator] => LevelClaim::Fraction(FractionClaim {
            point,
            numerator: from_reference(numerator),
            denominator: from_reference(denominator),
        }),
        _ => panic!("a claim holds one value or two"),
    }
}

/// A reference leaf stage's messages, as the verifier receives them.
pub fn proof_of_stage(stage: &reference::LeafStage) -> LogUpProof {
    LogUpProof {
        roots: stage.roots.iter().copied().map(from_reference).collect(),
        levels: stage
            .levels
            .iter()
            .map(|level| super::proof_of(&level.proof))
            .collect(),
        values: stage.values.iter().copied().map(from_reference).collect(),
    }
}

/// `x` as the reference prover's element.
pub fn fe2(value: Fp2) -> Fe {
    to_fe2(value)
}

// ----- Seeded batches and subjects ---------------------------------------------------------------

/// A number below `bound`.
pub fn below(rng: &mut SplitMix64, bound: usize) -> usize {
    (rng.next_u64() % bound as u64) as usize
}

/// An honest instance of `rows` random rows and `lookups` random lookups into them, with values
/// or not.
pub fn random_instance(
    rng: &mut SplitMix64,
    rows: usize,
    lookups: usize,
    has_values: bool,
) -> Instance {
    let table = (0..rows)
        .map(|_| {
            let value = if has_values {
                Fp::reduce(rng.next_u64())
            } else {
                Fp::ZERO
            };
            (Fp::reduce(rng.next_u64()), value)
        })
        .collect();
    let picks = (0..lookups).map(|_| below(rng, rows)).collect();
    Instance::honest(table, has_values, picks)
}

/// A forest of `2^s` random trees of height `h`, its roots the trees' products.
pub fn random_forest(rng: &mut SplitMix64, s: usize, h: usize) -> Forest {
    let leaves: Vec<Fp> = (0..1 << (s + h))
        .map(|_| Fp::reduce(rng.next_u64()))
        .collect();
    let roots = leaves
        .chunks_exact(1 << h)
        .map(|tree| tree.iter().fold(Fp::ONE, |p, leaf| p * *leaf))
        .collect();
    Forest {
        rows_log: s,
        height: h,
        roots,
        leaves,
    }
}

/// Every tree of `instances` instances and `forests` forests, in a random order.
pub fn random_schedule(rng: &mut SplitMix64, instances: usize, forests: usize) -> Vec<TreeRef> {
    let mut trees: Vec<TreeRef> = (0..instances)
        .flat_map(|index| [Kind::Lookup, Kind::Table].map(|kind| TreeRef { kind, index }))
        .chain((0..forests).map(|index| TreeRef {
            kind: Kind::Forest,
            index,
        }))
        .collect();
    for i in (1..trees.len()).rev() {
        trees.swap(i, below(rng, i + 1));
    }
    trees
}

/// An honest leaf stage to run and to change: a batch of the vectors, or one the reference
/// prover proved.
#[derive(Clone)]
pub struct Subject {
    pub name: String,
    pub context: Rc<Context>,
    pub instances: Vec<Instance>,
    pub forests: Vec<Forest>,
    pub trees: Vec<TreeRef>,
    pub proof: LogUpProof,
}

impl Subject {
    /// The honest batch `batch`.
    pub fn of_batch(batch: &LogUpBatch) -> Self {
        assert_eq!(batch.expected, Expected::Accept, "{}", batch.name);
        Self {
            name: batch.name.clone(),
            context: Rc::clone(&batch.context),
            instances: batch.instances.clone(),
            forests: batch.forests.clone(),
            trees: batch.trees.clone(),
            proof: batch.proof(),
        }
    }

    /// The reference prover's honest leaf stage of `instances` and `forests`, scheduled as
    /// `trees`, in `context`.
    pub fn proven(
        name: String,
        context: &SessionContext<'_>,
        instances: Vec<Instance>,
        forests: Vec<Forest>,
        trees: Vec<TreeRef>,
    ) -> Self {
        let logups: Vec<reference::LogUp> = instances.iter().map(reference_logup).collect();
        let reference_forests: Vec<reference::Forest> =
            forests.iter().map(reference_forest).collect();
        let mut prover = tv_testkit::sumcheck::Transcript::new(&context.to_tce());
        let stage = reference::prove_leaf_stage(
            &mut prover,
            &logups,
            &reference_forests,
            &reference_trees(&trees),
            |_| {},
        );
        Self {
            name,
            context: Rc::new(Context::of(context)),
            instances,
            forests,
            trees,
            proof: proof_of_stage(&stage),
        }
    }

    /// The trees' heights in schedule order.
    pub fn heights(&self) -> Vec<usize> {
        heights(&self.instances, &self.forests, &self.trees)
    }

    /// Runs `proof` on the subject's instances.
    pub fn run(&self, proof: &LogUpProof) -> LogUpRun {
        self.run_with(proof, |_, _| {})
    }

    /// Runs `proof`, `tamper` changing the claims entering a level, their number too.
    pub fn run_with(
        &self,
        proof: &LogUpProof,
        tamper: impl FnMut(u32, &mut Vec<LevelClaim>),
    ) -> LogUpRun {
        let mut transcript = self.context.transcript();
        run_logup(
            &mut transcript,
            &self.instances,
            &self.forests,
            &self.trees,
            proof,
            tamper,
        )
    }
}

/// Runs `subject`'s own proof with the claims entering level `level` replaced by `claims`.
pub fn run_replacing(subject: &Subject, level: u32, claims: Vec<LevelClaim>) -> LogUpRun {
    let mut replacement = Some(claims);
    subject.run_with(&subject.proof, |at, entering| {
        if at == level {
            if let Some(claims) = replacement.take() {
                *entering = claims;
            }
        }
    })
}

/// The session the seeded subjects run in.
pub fn seeded_context() -> SessionContext<'static> {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::Live,
        params: ParamSetId(Ascii::new("live-64/1").unwrap()),
        statement_digest: [0x62; 32],
        session_id: *b"logup subjects  ",
    }
}

/// `count` seeded honest subjects of the reference prover: zero to three instances of 1 to 9
/// lookups into 1 to 9 rows, with values or not, a third of those with padding rows giving one of
/// row 0's counts to a padding row, beside zero to two forests of up to 4 trees of heights 1 to 3,
/// a third of them SoftmaxRow forests, in a random schedule order.
pub fn seeded_subjects(seed: u64, count: usize) -> Vec<Subject> {
    let mut rng = SplitMix64::new(seed);
    (0..count)
        .map(|case| {
            let instances: Vec<Instance> = (0..below(&mut rng, 4))
                .map(|_| {
                    let (rows, lookups) = (1 + below(&mut rng, 9), 1 + below(&mut rng, 9));
                    let has_values = rng.next_u64() & 1 == 0;
                    let mut instance = random_instance(&mut rng, rows, lookups, has_values);
                    let padding = instance.multiplicities.len() - rows;
                    if padding > 0
                        && instance.multiplicities[0] != Fp::ZERO
                        && below(&mut rng, 3) == 0
                    {
                        // An honest multiplicity on a padding row: a count of row 0 on one of
                        // its copies.
                        instance.multiplicities[0] -= Fp::ONE;
                        instance.multiplicities[rows + below(&mut rng, padding)] += Fp::ONE;
                    }
                    instance
                })
                .collect();
            let forests: Vec<Forest> = (0..below(&mut rng, 3))
                .map(|_| {
                    let (s, h) = (below(&mut rng, 3), 1 + below(&mut rng, 3));
                    let mut forest = random_forest(&mut rng, s, h);
                    if below(&mut rng, 3) == 0 {
                        // A SoftmaxRow forest: every tree's first leaf is zero, so are its roots,
                        // and its u is zero where it joins.
                        for tree in forest.leaves.chunks_exact_mut(1 << h) {
                            tree[0] = Fp::ZERO;
                        }
                        forest.roots.iter_mut().for_each(|root| *root = Fp::ZERO);
                    }
                    forest
                })
                .collect();
            let trees = random_schedule(&mut rng, instances.len(), forests.len());
            Subject::proven(
                format!("seeded subject {case}"),
                &seeded_context(),
                instances,
                forests,
                trees,
            )
        })
        .collect()
}

/// The honest batches of the vectors as subjects, then `seeded` subjects of the reference
/// prover.
pub fn subjects(seeded: usize) -> Vec<Subject> {
    let batches = logup_batches().expect("`logup_batches` in spec/vectors/sumcheck/sumcheck.json");
    batches
        .iter()
        .filter(|batch| batch.expected == Expected::Accept)
        .map(Subject::of_batch)
        .chain(seeded_subjects(0x5eed_0062_00ff, seeded))
        .collect()
}
