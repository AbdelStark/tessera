// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared test support: seeded inputs, a session's context on both sides, and a verifier driver
//! written from RFC-0008 'Fraction tree' and RFC-0010 'Tree levels' around the verifier's
//! independent checks (DEP-2): `tv_core::logup::check_roots`, then `tv_core::tree::verify_level`
//! at each level, then `tv_core::logup::check_leaves`.
//!
//! The driver holds the schedule as a verifier does: from the instances' and forests' shapes it
//! knows where each tree joins. It squeezes the forests' root challenges itself and forms their
//! root claims from the trees' products; the fraction trees enter with the root claims
//! `check_roots` returns. The table polynomial `t̃ = kẽy_t + β·valũe_t` is evaluated here from the
//! table's rows in `tv-field`, its padding rows copies of row 0, and so are the extensions every
//! claim must equal: the lookup-side keys and values, padded with row 0's, and the multiplicities.
//! None of it uses the prover's code.

#![allow(dead_code)] // each test binary uses part of it

use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_logup::{LogUpInstance, ProvenLogUp, TreeId, Tuples, prove_logup};
use tessera_sumcheck::{Claim, Forest, LevelClaim};
use tv_core::logup::{self as tvl, LookupError};
use tv_core::sumcheck as tv;
use tv_core::tree as tvt;

pub type Fp = Goldilocks;
pub type Fp2 = GoldilocksExt2;
pub type TvFp = tv_field::Fp;
pub type TvFp2 = tv_field::Fp2;

/// SplitMix64 (Steele, Lea and Flood, 2014): reproducible inputs, nothing secret.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    pub fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    pub fn fp(&mut self) -> Fp {
        Fp::from_u64_reduced(self.next_u64())
    }

    pub fn fps(&mut self, n: usize) -> Vec<Fp> {
        (0..n).map(|_| self.fp()).collect()
    }

    pub fn bytes<const N: usize>(&mut self) -> [u8; N] {
        core::array::from_fn(|_| self.next_u64() as u8)
    }
}

// ----- Conversions to the verifier's field ---------------------------------------------------

pub fn tv_fp(x: Fp) -> TvFp {
    TvFp::from_canonical(x.to_canonical_u64()).expect("a canonical element")
}

pub fn tv_fp2(x: Fp2) -> TvFp2 {
    let [c0, c1] = x.coeffs();
    TvFp2::new(tv_fp(c0), tv_fp(c1))
}

pub fn tv_fp2s(xs: &[Fp2]) -> Vec<TvFp2> {
    xs.iter().copied().map(tv_fp2).collect()
}

pub fn tv_proof(proof: &tessera_sumcheck::SumcheckProof<Fp2>) -> tv::SumcheckProof {
    tv::SumcheckProof {
        rounds: proof.rounds.iter().map(|round| tv_fp2s(round)).collect(),
        final_evals: tv_fp2s(&proof.final_evals),
    }
}

// ----- Sessions --------------------------------------------------------------------------------

pub const PROTOCOL: &str = "tessera-protocol/1";

/// A session's context, built for each side from the same fields.
pub struct Session {
    pub mode: Mode,
    pub statement_digest: [u8; 32],
    pub session_id: [u8; 16],
}

impl Session {
    pub fn random(rng: &mut SplitMix64) -> Self {
        Self {
            mode: if rng.coin() {
                Mode::Live
            } else {
                Mode::NonInteractive
            },
            statement_digest: rng.bytes(),
            session_id: rng.bytes(),
        }
    }

    pub fn transcript(&self) -> Transcript {
        Transcript::new(&SessionContext {
            protocol: Ascii::new(PROTOCOL).expect("ASCII"),
            mode: self.mode,
            params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
            statement_digest: self.statement_digest,
            session_id: self.session_id,
        })
    }

    pub fn tv_transcript(&self) -> tv_hash::transcript::Transcript {
        use tv_hash::transcript as t;
        t::Transcript::new(&t::SessionContext {
            protocol: t::Ascii::new(PROTOCOL).expect("ASCII"),
            mode: match self.mode {
                Mode::Live => t::Mode::Live,
                Mode::NonInteractive => t::Mode::NonInteractive,
            },
            params: t::ParamSetId(t::Ascii::new("test-small/1").expect("ASCII")),
            statement_digest: self.statement_digest,
            session_id: self.session_id,
        })
    }
}

// ----- Extensions in the verifier's field ------------------------------------------------------

/// The `2^n` weights `eq(point, x)`, built by doubling: after `r_1, …, r_j` the table holds the
/// products of the first `j` factors, and `r_{j+1}` splits each entry `e` into `e·(1 − r_{j+1})`
/// and `e·r_{j+1}`.
pub fn eq_weights(point: &[TvFp2]) -> Vec<TvFp2> {
    let mut weights = vec![TvFp2::ONE];
    for &r in point {
        let high: Vec<TvFp2> = weights.iter().map(|&e| e * r).collect();
        for (e, h) in weights.iter_mut().zip(&high) {
            *e -= *h;
        }
        weights.extend(high);
    }
    weights
}

/// `f̃(point) = Σ_x f[x]·eq(point, x)` of a table over `Fp2`.
pub fn mle(table: &[TvFp2], point: &[TvFp2]) -> TvFp2 {
    assert_eq!(
        table.len(),
        1 << point.len(),
        "2^n entries at n coordinates"
    );
    table
        .iter()
        .zip(eq_weights(point))
        .fold(TvFp2::ZERO, |sum, (&f, e)| sum + f * e)
}

/// `table` lifted, then padded with `padding` to `2^height` entries.
pub fn padded(table: &[Fp], padding: Fp, height: usize) -> Vec<TvFp2> {
    let mut lifted: Vec<TvFp2> = table.iter().map(|&x| TvFp2::from(tv_fp(x))).collect();
    lifted.resize(1 << height, TvFp2::from(tv_fp(padding)));
    lifted
}

/// `⌈log2 n⌉`, 0 for `n ≤ 1`.
pub fn height(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}

// ----- Instances -------------------------------------------------------------------------------

/// A table of rows `(key, value)`, values absent for a table without values.
#[derive(Clone)]
pub struct TestTable {
    pub keys: Vec<Fp>,
    pub values: Option<Vec<Fp>>,
}

impl TestTable {
    /// `rows` rows of distinct random keys, and random values when `with_values`; without values,
    /// the keys `0, …, rows − 1`, as a `Range` table's.
    pub fn random(rng: &mut SplitMix64, rows: usize, with_values: bool) -> Self {
        match with_values {
            true => {
                let keys = rng.fps(rows);
                Self {
                    keys,
                    values: Some(rng.fps(rows)),
                }
            }
            false => Self {
                keys: (0..rows as u64).map(Fp::from_u64_reduced).collect(),
                values: None,
            },
        }
    }

    pub fn rows(&self) -> usize {
        self.keys.len()
    }

    pub fn tuples(&self) -> Tuples<'_, Fp> {
        Tuples::new(&self.keys, self.values.as_deref())
    }

    /// Row `j`'s value, 0 without values.
    pub fn value(&self, j: usize) -> Fp {
        self.values.as_ref().map_or(Fp::ZERO, |v| v[j])
    }

    /// `t̃(ρ) = Σ_j eq(ρ, j)·(key_j + β·value_j)` over the `2^{|ρ|}` rows, the padding rows copies
    /// of row 0 (RFC-0008 'Structured evaluation'), by enumeration.
    pub fn tv_eval(&self, rho: &[TvFp2], beta: TvFp2) -> TvFp2 {
        let keys = padded(&self.keys, self.keys[0], rho.len());
        let values = padded(
            &(0..self.rows()).map(|j| self.value(j)).collect::<Vec<_>>(),
            self.value(0),
            rho.len(),
        );
        let rows: Vec<TvFp2> = keys
            .iter()
            .zip(&values)
            .map(|(&k, &v)| k + beta * v)
            .collect();
        mle(&rows, rho)
    }
}

/// An instance: a table, the lookups' columns, and the committed multiplicities.
#[derive(Clone)]
pub struct TestInstance {
    pub table: TestTable,
    pub keys: Vec<Fp>,
    pub values: Option<Vec<Fp>>,
    pub multiplicities: Vec<Fp>,
}

impl TestInstance {
    /// Honest lookups of the rows `rows`, with the multiplicities counted here from the
    /// definition: the lookups of each row, row 0 also counting the lookup tree's padding.
    pub fn of_rows(table: TestTable, rows: &[usize]) -> Self {
        let keys = rows.iter().map(|&r| table.keys[r]).collect();
        let values = table
            .values
            .as_ref()
            .map(|v| rows.iter().map(|&r| v[r]).collect());
        let mut counts = vec![0u64; 1 << height(table.rows())];
        for &r in rows {
            counts[r] += 1;
        }
        counts[0] += ((1usize << height(rows.len())) - rows.len()) as u64;
        Self {
            table,
            keys,
            values,
            multiplicities: counts.into_iter().map(Fp::from_u64_reduced).collect(),
        }
    }

    /// `lookups` random rows of `table`.
    pub fn random(rng: &mut SplitMix64, table: TestTable, lookups: usize) -> Self {
        let rows: Vec<usize> = (0..lookups)
            .map(|_| rng.below(table.rows() as u64) as usize)
            .collect();
        Self::of_rows(table, &rows)
    }

    pub fn instance(&self) -> LogUpInstance<'_, Fp> {
        LogUpInstance::new(
            self.table.tuples(),
            Tuples::new(&self.keys, self.values.as_deref()),
            &self.multiplicities,
        )
    }

    pub fn lookup_height(&self) -> usize {
        height(self.keys.len())
    }

    pub fn table_height(&self) -> usize {
        height(self.table.rows())
    }
}

/// `2^s` product trees of height `h`, their leaves stacked: entry `y·2^h + x` is leaf `x` of
/// tree `y`.
#[derive(Clone)]
pub struct TestForest {
    pub trees_log: usize,
    pub height: usize,
    pub leaves: Vec<Fp>,
}

impl TestForest {
    pub fn random(rng: &mut SplitMix64, trees_log: usize, height: usize) -> Self {
        Self {
            trees_log,
            height,
            leaves: rng.fps(1 << (trees_log + height)),
        }
    }

    pub fn forest(&self) -> Forest<'_, Fp2> {
        Forest::new(&self.leaves, self.height)
    }

    /// The trees' products `v_y`: the forest's public root values.
    pub fn roots(&self) -> Vec<TvFp2> {
        self.leaves
            .chunks_exact(1 << self.height)
            .map(|tree| TvFp2::from(tree.iter().fold(TvFp::ONE, |p, &x| p * tv_fp(x))))
            .collect()
    }
}

// ----- The verifier driver ---------------------------------------------------------------------

/// The first check that failed.
#[derive(Debug, PartialEq, Eq)]
pub enum Rejected {
    /// `check_roots` or `check_leaves`.
    Lookup(LookupError),
    /// The tree level `level`.
    Level(usize, tv::SumcheckError),
}

/// What a verifier that accepted every check holds.
pub struct Verified {
    pub roots: tvl::Roots,
    pub levels: Vec<tvt::TreeLevel>,
    /// The forests' root challenges per level, as squeezed.
    pub root_challenges: Vec<Vec<TvFp2>>,
    /// The trees present per level, as schedule positions.
    pub present: Vec<Vec<usize>>,
    /// Each tree's leaf claims, in schedule order.
    pub leaf_claims: Vec<tvt::LevelClaim>,
    pub claims: tvl::LeafClaims,
    pub state: [u8; 32],
}

/// The messages a verifier reads, in its types.
pub struct Messages {
    pub roots: Vec<TvFp2>,
    pub levels: Vec<tv::SumcheckProof>,
    pub values: Vec<TvFp2>,
}

impl Messages {
    pub fn of(proven: &ProvenLogUp<Fp2>) -> Self {
        Self {
            roots: tv_fp2s(proven.roots()),
            levels: proven
                .levels()
                .iter()
                .map(|level| tv_proof(level.proof()))
                .collect(),
            values: tv_fp2s(proven.values()),
        }
    }
}

/// Checks `messages` as a verifier holding the schedule does, on a fresh `tv-hash` transcript of
/// `session`: the roots, every level, then the leaf checks against `t̃(ρ_t)` computed here from
/// each instance's table.
pub fn verify(
    session: &Session,
    instances: &[TestInstance],
    forests: &[TestForest],
    schedule: &[TreeId],
    messages: &Messages,
) -> Result<Verified, Rejected> {
    let mut transcript = session.tv_transcript();
    let roots = tvl::check_roots(&mut transcript, instances.len(), &messages.roots)
        .map_err(Rejected::Lookup)?;
    let heights: Vec<usize> = schedule
        .iter()
        .map(|&id| match id {
            TreeId::Lookup(k) => instances[k].lookup_height(),
            TreeId::Table(k) => instances[k].table_height(),
            TreeId::Forest(k) => forests[k].height,
        })
        .collect();
    let tallest = heights.iter().copied().max().unwrap_or(0);
    assert_eq!(messages.levels.len(), tallest, "one proof per level");
    let mut claims: Vec<Option<tvt::LevelClaim>> = schedule
        .iter()
        .map(|&id| match id {
            TreeId::Lookup(k) => Some(tvt::LevelClaim::Fraction(roots.instances[k].lookup.clone())),
            TreeId::Table(k) => Some(tvt::LevelClaim::Fraction(roots.instances[k].table.clone())),
            TreeId::Forest(_) => None,
        })
        .collect();
    let mut levels = Vec::new();
    let mut root_challenges = Vec::new();
    let mut present_by_level = Vec::new();
    for (level, proof) in messages.levels.iter().enumerate() {
        let joins = |p: usize| tallest - heights[p] == level;
        let s_max = (0..schedule.len())
            .filter(|&p| joins(p))
            .map(|p| match schedule[p] {
                TreeId::Forest(k) => forests[k].trees_log,
                _ => 0,
            })
            .max()
            .unwrap_or(0);
        let root: Vec<TvFp2> = (0..s_max)
            .map(|c| {
                let label = format!("tree/{level}/root/{c}");
                transcript
                    .challenge_fp2(tv_hash::transcript::Ascii::new(&label).expect("ASCII"))
                    .expect("a challenge")
            })
            .collect();
        for p in (0..schedule.len()).filter(|&p| joins(p)) {
            if let TreeId::Forest(k) = schedule[p] {
                let r = root[s_max - forests[k].trees_log..].to_vec();
                let weights = eq_weights(&r);
                let value = forests[k]
                    .roots()
                    .iter()
                    .zip(weights)
                    .fold(TvFp2::ZERO, |sum, (&v, e)| sum + v * e);
                claims[p] = Some(tvt::LevelClaim::Product(tvt::TreeClaim { point: r, value }));
            }
        }
        let present: Vec<usize> = (0..schedule.len())
            .filter(|&p| tallest - heights[p] <= level)
            .collect();
        let entering: Vec<tvt::LevelClaim> = present
            .iter()
            .map(|&p| claims[p].clone().expect("a tree that has joined"))
            .collect();
        let checked = tvt::verify_level(&mut transcript, level as u32, &entering, proof)
            .map_err(|error| Rejected::Level(level, error))?;
        for (&p, next) in present.iter().zip(&checked.claims) {
            claims[p] = Some(next.clone());
        }
        root_challenges.push(root);
        present_by_level.push(present);
        levels.push(checked);
    }
    let leaf_claims: Vec<tvt::LevelClaim> = claims
        .into_iter()
        .map(|claim| claim.expect("every tree has claims"))
        .collect();
    let fraction = |id: TreeId| {
        let p = schedule.iter().position(|&x| x == id).expect("scheduled");
        match &leaf_claims[p] {
            tvt::LevelClaim::Fraction(claim) => claim.clone(),
            tvt::LevelClaim::Product(_) => panic!("a fraction tree's claims"),
        }
    };
    let leaves: Vec<tvl::Leaves> = (0..instances.len())
        .map(|k| tvl::Leaves {
            lookup: fraction(TreeId::Lookup(k)),
            table: fraction(TreeId::Table(k)),
            has_values: instances[k].table.values.is_some(),
        })
        .collect();
    let beta = roots.challenges.beta;
    let checked = tvl::check_leaves(
        &mut transcript,
        &roots.challenges,
        &leaves,
        &messages.values,
        |i, rho| Some(instances.get(i)?.table.tv_eval(rho, beta)),
    )
    .map_err(Rejected::Lookup)?;
    Ok(Verified {
        roots,
        levels,
        root_challenges,
        present: present_by_level,
        leaf_claims,
        claims: checked,
        state: *transcript.state(),
    })
}

/// Proves the stage on a fresh transcript of `session`, and returns the result and the
/// transcript's final state.
pub fn prove(
    session: &Session,
    instances: &[TestInstance],
    forests: &[TestForest],
    schedule: &[TreeId],
) -> (ProvenLogUp<Fp2>, [u8; 32]) {
    let borrowed: Vec<LogUpInstance<'_, Fp>> =
        instances.iter().map(TestInstance::instance).collect();
    let forests: Vec<Forest<'_, Fp2>> = forests.iter().map(TestForest::forest).collect();
    let mut transcript = session.transcript();
    let proven =
        prove_logup(&mut transcript, &borrowed, &forests, schedule).expect("a well-formed stage");
    (proven, transcript.state())
}

fn tv_claim(claim: &Claim<Fp2>) -> tvl::Claim {
    tvl::Claim {
        point: tv_fp2s(&claim.point),
        value: tv_fp2(claim.value),
    }
}

fn tv_level_claim(claim: &LevelClaim<Fp2>) -> tvt::LevelClaim {
    match claim {
        LevelClaim::Product(c) => tvt::LevelClaim::Product(tvt::TreeClaim {
            point: tv_fp2s(&c.point),
            value: tv_fp2(c.value),
        }),
        LevelClaim::Fraction(c) => tvt::LevelClaim::Fraction(tvt::FractionClaim {
            point: tv_fp2s(&c.point),
            numerator: tv_fp2(c.numerator),
            denominator: tv_fp2(c.denominator),
        }),
    }
}

/// Proves the stage and checks it with the verifier driver: it accepts; its challenges, every
/// level's instances, `λ`, point, `ρ` and next claims are the prover's; its transcript ends in the
/// prover's state; the claims it leaves are the prover's; and each claim is the extension of its
/// polynomial at its point, from the instances' columns: the lookup-side keys and values padded
/// with row 0's, the multiplicities, and each forest's stacked leaves.
pub fn prove_and_verify(
    session: &Session,
    instances: &[TestInstance],
    forests: &[TestForest],
    schedule: &[TreeId],
) -> ProvenLogUp<Fp2> {
    let (proven, state) = prove(session, instances, forests, schedule);
    let verified = verify(
        session,
        instances,
        forests,
        schedule,
        &Messages::of(&proven),
    )
    .unwrap_or_else(|rejected| panic!("the verifier rejects an honest proof: {rejected:?}"));
    assert_eq!(
        verified.state, state,
        "the transcripts end in the same state"
    );
    assert_eq!(
        tv_fp2(*proven.alpha()),
        verified.roots.challenges.alpha,
        "α"
    );
    assert_eq!(tv_fp2(*proven.beta()), verified.roots.challenges.beta, "β");
    assert_eq!(proven.levels().len(), verified.levels.len(), "the levels");
    for (level, ((mine, theirs), (present, root))) in proven
        .levels()
        .iter()
        .zip(&verified.levels)
        .zip(verified.present.iter().zip(&verified.root_challenges))
        .enumerate()
    {
        assert_eq!(mine.trees(), present, "level {level}: the instances");
        assert_eq!(&tv_fp2s(mine.root_challenges()), root, "level {level}");
        assert_eq!(tv_fp2(*mine.gamma()), theirs.gamma, "level {level}: γ");
        assert_eq!(
            mine.lambda().map(|l| tv_fp2(*l)),
            theirs.lambda,
            "level {level}: λ"
        );
        let fractions = present
            .iter()
            .any(|&p| !matches!(schedule[p], TreeId::Forest(_)));
        assert_eq!(
            theirs.lambda.is_some(),
            fractions,
            "level {level}: λ where a fraction tree is"
        );
        assert_eq!(
            tv_fp2s(mine.point()),
            theirs.point,
            "level {level}: the point"
        );
        assert_eq!(tv_fp2(*mine.rho()), theirs.rho, "level {level}: ρ");
        let next: Vec<tvt::LevelClaim> = mine.next_claims().iter().map(tv_level_claim).collect();
        assert_eq!(next, theirs.claims, "level {level}: the next claims");
    }
    let leaf_claims: Vec<tvt::LevelClaim> =
        proven.leaf_claims().iter().map(tv_level_claim).collect();
    assert_eq!(leaf_claims, verified.leaf_claims, "the leaf claims");

    assert_eq!(
        proven.claims().len(),
        instances.len(),
        "claims per instance"
    );
    for (k, ((mine, theirs), instance)) in proven
        .claims()
        .iter()
        .zip(&verified.claims.instances)
        .zip(instances)
        .enumerate()
    {
        assert_eq!(
            tv_claim(&mine.key),
            theirs.key,
            "instance {k}: the key claim"
        );
        assert_eq!(
            mine.value.as_ref().map(tv_claim),
            theirs.value,
            "instance {k}: the value claim"
        );
        assert_eq!(
            tv_claim(&mine.multiplicity),
            theirs.multiplicity,
            "instance {k}: the multiplicity claim"
        );
        // Each claim is its polynomial's extension at its point.
        let h_a = instance.lookup_height();
        let keys = padded(&instance.keys, instance.table.keys[0], h_a);
        assert_eq!(theirs.key.point.len(), h_a);
        assert_eq!(
            theirs.key.value,
            mle(&keys, &theirs.key.point),
            "instance {k}: kẽy(ρ_a)"
        );
        match (&instance.values, &theirs.value) {
            (Some(values), Some(claim)) => {
                let values = padded(values, instance.table.value(0), h_a);
                assert_eq!(claim.point, theirs.key.point);
                assert_eq!(
                    claim.value,
                    mle(&values, &claim.point),
                    "instance {k}: valũe(ρ_a)"
                );
            }
            (None, None) => {}
            _ => panic!("instance {k}: a value claim exactly with values"),
        }
        let m: Vec<TvFp2> = instance
            .multiplicities
            .iter()
            .map(|&m| TvFp2::from(tv_fp(m)))
            .collect();
        assert_eq!(theirs.multiplicity.point.len(), instance.table_height());
        assert_eq!(
            theirs.multiplicity.value,
            mle(&m, &theirs.multiplicity.point),
            "instance {k}: m̃(ρ_t)"
        );
    }
    assert_eq!(
        proven.forest_claims().len(),
        forests.len(),
        "a leaf claim per forest"
    );
    for (k, (claim, forest)) in proven.forest_claims().iter().zip(forests).enumerate() {
        let p = schedule
            .iter()
            .position(|&id| id == TreeId::Forest(k))
            .expect("scheduled");
        let tvt::LevelClaim::Product(theirs) = &verified.leaf_claims[p] else {
            panic!("forest {k}: a product claim");
        };
        assert_eq!(
            tv_fp2s(&claim.point),
            theirs.point,
            "forest {k}: the leaf point"
        );
        assert_eq!(
            tv_fp2(claim.value),
            theirs.value,
            "forest {k}: the leaf value"
        );
        let leaves: Vec<TvFp2> = forest
            .leaves
            .iter()
            .map(|&x| TvFp2::from(tv_fp(x)))
            .collect();
        assert_eq!(
            theirs.value,
            mle(&leaves, &theirs.point),
            "forest {k}: the leaves"
        );
    }
    proven
}

/// A random `Fp2` element.
pub fn fp2(rng: &mut SplitMix64) -> Fp2 {
    let c0 = rng.fp();
    Fp2::new(c0, rng.fp())
}
