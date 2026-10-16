// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The engine over the calibration field (07 §4): the same code proves batches over
//! `F65537Ext2`, with tables over `F65537` and over the extension, and over `F65537` itself,
//! whose challenges then lie in the prime field, on a transcript the test supplies. `tv-core`
//! checks Goldilocks proofs only, so each proof is checked here as RFC-0010 'Sumcheck messages'
//! says a verifier does, round by round from the batched claim with `p(1) = claim − p(0)` and
//! `tessera_poly::RoundPoly`, which the prover does not use, and each final evaluation against
//! the table's multilinear extension at the instance's point.
//!
//! Batches of product forests are checked the same way, level by level as RFC-0010 'Tree levels'
//! says: root challenges, root claims from the trees' products, the level's sumcheck and its
//! final check `Σ_i γ^i·eq(r_i, r'_i)·u_i·w_i`, `ρ` and the next claims, and the leaf claims
//! against the leaves' multilinear extension.

mod common;

use common::SplitMix64;
use tessera_codec::{Ascii, Encode};
use tessera_field::{F65537, F65537Ext2, Field, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_poly::{PolyField, RoundPoly, evaluate, evaluate_base};
use tessera_sumcheck::{
    Forest, Instance, MAX_DEGREE, Prefix, SumcheckProof, SumcheckTranscript, Table, prove_batch,
    prove_forests,
};

/// A transcript over `F65537` and its extension: messages absorbed as TCE `Vec`, challenges
/// drawn exactly uniformly by rejecting 17-bit words at or above 65537.
struct SmallTranscript {
    inner: Transcript,
    /// The 17-bit words of the challenge's last squeeze not yet used.
    words: Vec<u64>,
}

impl SmallTranscript {
    fn new(seed: u8) -> Self {
        let context = SessionContext {
            protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
            mode: Mode::NonInteractive,
            params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
            statement_digest: [seed; 32],
            session_id: [seed; 16],
        };
        Self {
            inner: Transcript::new(&context),
            words: Vec::new(),
        }
    }

    /// The next element of the challenge being drawn under `label`.
    fn element(&mut self, label: &str) -> F65537 {
        loop {
            if self.words.is_empty() {
                let out = self.inner.squeeze(label);
                self.words = out
                    .chunks_exact(8)
                    .map(|w| u64::from_le_bytes(w.try_into().expect("8 bytes")) & 0x1_FFFF)
                    .collect();
            }
            if let Some(x) = F65537::from_canonical_u64(self.words.remove(0)) {
                return x;
            }
        }
    }

    fn absorb<T: Encode + Clone>(&mut self, label: &str, message: &[T]) {
        self.inner.absorb(label, &message.to_vec());
    }
}

impl SumcheckTranscript<F65537Ext2> for SmallTranscript {
    fn absorb_message(&mut self, label: &str, message: &[F65537Ext2]) {
        self.absorb(label, message);
    }

    fn challenge(&mut self, label: &str) -> F65537Ext2 {
        self.words.clear();
        let c0 = self.element(label);
        F65537Ext2::new(c0, self.element(label))
    }
}

impl SumcheckTranscript<F65537> for SmallTranscript {
    fn absorb_message(&mut self, label: &str, message: &[F65537]) {
        self.absorb(label, message);
    }

    fn challenge(&mut self, label: &str) -> F65537 {
        self.words.clear();
        self.element(label)
    }
}

/// A table over the prime field or over `F`.
enum Values<F: PolyField> {
    Base(Vec<F::Base>),
    Ext(Vec<F>),
}

impl<F: PolyField> Values<F> {
    fn table(&self) -> Table<'_, F> {
        match self {
            Self::Base(v) => Table::base(v),
            Self::Ext(v) => Table::ext(v),
        }
    }

    fn at(&self, x: usize) -> F {
        match self {
            Self::Base(v) => F::from(v[x]),
            Self::Ext(v) => v[x],
        }
    }

    fn mle(&self, point: &[F]) -> F {
        match self {
            Self::Base(v) => evaluate_base(v, point),
            Self::Ext(v) => evaluate(v, point),
        }
        .expect("a table of 2^n entries at n coordinates")
    }
}

/// Checks a proof as a verifier would, from the instances' sums over their hypercubes.
fn check<F: PolyField>(
    t: &mut SmallTranscript,
    prefix: &str,
    degree: usize,
    instances: &[(usize, Vec<Values<F>>)],
    proof: &SumcheckProof<F>,
) where
    SmallTranscript: SumcheckTranscript<F>,
{
    let n_max = instances
        .iter()
        .map(|(n, _)| *n)
        .max()
        .expect("an instance");
    let gamma = t.challenge(&format!("{prefix}/gamma"));
    let mut claim = F::ZERO;
    let mut weight = F::ONE;
    for (vars, tables) in instances {
        let sum = (0..1 << vars).fold(F::ZERO, |s, x| {
            s + tables.iter().fold(F::ONE, |p, table| p * table.at(x))
        });
        let scale = F::from(F::Base::from_u64_reduced(1 << (n_max - vars)));
        claim += weight * scale * sum;
        weight *= gamma;
    }
    assert_eq!(proof.rounds.len(), n_max);
    let mut point = Vec::new();
    for (j, round) in proof.rounds.iter().enumerate() {
        assert_eq!(round.len(), degree, "round {j} holds d values");
        t.absorb_message(&format!("{prefix}/round/{j}"), round);
        let r = t.challenge(&format!("{prefix}/r/{j}"));
        claim = RoundPoly::new(round).expect("a degree").evaluate(claim, r);
        point.push(r);
    }
    t.absorb_message(&format!("{prefix}/evals"), &proof.final_evals);
    let mut evals = proof.final_evals.iter();
    let mut batched = F::ZERO;
    let mut weight = F::ONE;
    for (vars, tables) in instances {
        let own = &point[n_max - vars..];
        let mut value = F::ONE;
        for table in tables {
            let eval = *evals.next().expect("an evaluation per table");
            assert_eq!(
                eval,
                table.mle(own),
                "the table's extension at the instance's point"
            );
            value *= eval;
        }
        batched += weight * value;
        weight *= gamma;
    }
    assert!(evals.next().is_none());
    assert_eq!(batched, claim, "the final check");
}

trait Draw: PolyField {
    fn draw(rng: &mut SplitMix64) -> Self;
}

impl Draw for F65537 {
    fn draw(rng: &mut SplitMix64) -> Self {
        Self::from_u64_reduced(rng.next_u64())
    }
}

impl Draw for F65537Ext2 {
    fn draw(rng: &mut SplitMix64) -> Self {
        let c0 = F65537::draw(rng);
        Self::new(c0, F65537::draw(rng))
    }
}

/// Random batches over `F`: 1 to 4 instances on 0 to 8 variables, degree 1 to 7, tables of
/// random kinds.
fn random_batches<F: Draw<Base = F65537>>(seed: u64)
where
    SmallTranscript: SumcheckTranscript<F>,
{
    let mut rng = SplitMix64::new(seed);
    for case in 0..100u8 {
        let degree = 1 + rng.below(MAX_DEGREE as u64) as usize;
        let count = 1 + rng.below(4) as usize;
        let instances: Vec<(usize, Vec<Values<F>>)> = (0..count)
            .map(|_| {
                let vars = rng.below(9) as usize;
                let k = 1 + rng.below(degree as u64) as usize;
                let tables = (0..k)
                    .map(|_| match rng.coin() {
                        true => {
                            Values::Base((0..1 << vars).map(|_| F65537::draw(&mut rng)).collect())
                        }
                        false => Values::Ext((0..1 << vars).map(|_| F::draw(&mut rng)).collect()),
                    })
                    .collect();
                (vars, tables)
            })
            .collect();
        let borrowed: Vec<Instance<'_, F>> = instances
            .iter()
            .map(|(_, tables)| Instance::new(tables.iter().map(Values::table).collect()))
            .collect();
        let mut prover = SmallTranscript::new(case);
        let proven = prove_batch(&mut prover, Prefix::Stage(5), degree, &borrowed)
            .expect("a well-formed batch");
        let mut verifier = SmallTranscript::new(case);
        check(&mut verifier, "stage/5", degree, &instances, proven.proof());
        assert_eq!(
            verifier.inner.state(),
            prover.inner.state(),
            "the transcripts end in the same state"
        );
    }
}

#[test]
fn batches_over_the_extension_of_f65537_verify() {
    random_batches::<F65537Ext2>(0x53_0401);
}

#[test]
fn batches_with_challenges_in_f65537_verify() {
    random_batches::<F65537>(0x53_0402);
}

/// `eq(r, y)` at the vertex `y`, from the definition: `Π_i (r_i·y_i + (1 − r_i)·(1 − y_i))`,
/// `y_i` bit `i − 1` of `y`.
fn eq_at_vertex<F: PolyField>(r: &[F], y: usize) -> F {
    r.iter().enumerate().fold(F::ONE, |product, (i, &ri)| {
        product * if (y >> i) & 1 == 1 { ri } else { F::ONE - ri }
    })
}

/// `eq(a, b) = Π_i (a_i·b_i + (1 − a_i)·(1 − b_i))`.
fn eq_points<F: PolyField>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(F::ONE, |product, (&a, &b)| {
        product * (a * b + (F::ONE - a) * (F::ONE - b))
    })
}

/// A batch of forests over `F`, `(s, h, zero rows)` each: heights 4, 6, 2, 3 and 1 join at
/// levels 2, 0, 4, 3 and 5, the forests of 2^2, 2^3 and 2^5 trees under root challenges, the
/// single trees under none. Proven, then checked level by level as a verifier would.
fn forest_batch<F: Draw<Base = F65537>>(seed: u8)
where
    SmallTranscript: SumcheckTranscript<F>,
{
    let mut rng = SplitMix64::new(0x58_0400 + u64::from(seed));
    let shapes = [
        (2usize, 4usize, true),
        (0, 6, false),
        (3, 2, false),
        (5, 3, true),
        (0, 1, true),
    ];
    let leaves: Vec<Vec<F65537>> = shapes
        .iter()
        .map(|&(s, h, zero_rows)| {
            let mut leaves: Vec<F65537> =
                (0..1 << (s + h)).map(|_| F65537::draw(&mut rng)).collect();
            if zero_rows {
                for y in 0..1 << s {
                    leaves[(y << h) + rng.below(1 << h) as usize] = F65537::ZERO;
                }
            }
            leaves
        })
        .collect();
    let forests: Vec<Forest<'_, F>> = leaves
        .iter()
        .zip(&shapes)
        .map(|(leaves, &(_, h, _))| Forest::new(leaves, h))
        .collect();
    let mut prover = SmallTranscript::new(seed);
    let proven = prove_forests(&mut prover, &forests).expect("well-formed forests");

    // The verifier, from RFC-0010 'Tree levels', with the trees' true products as public roots.
    let roots: Vec<Vec<F>> = leaves
        .iter()
        .zip(&shapes)
        .map(|(leaves, &(_, h, _))| {
            leaves
                .chunks_exact(1 << h)
                .map(|tree| F::from(tree.iter().fold(F65537::ONE, |p, &x| p * x)))
                .collect()
        })
        .collect();
    let tallest = shapes.iter().map(|&(_, h, _)| h).max().expect("a forest");
    assert_eq!(proven.levels().len(), tallest);
    let mut t = SmallTranscript::new(seed);
    let mut claims: Vec<Option<(Vec<F>, F)>> = vec![None; shapes.len()];
    for (level, record) in proven.levels().iter().enumerate() {
        let prefix = format!("tree/{level}");
        let joins = |i: usize| tallest - shapes[i].1;
        let s_max = (0..shapes.len())
            .filter(|&i| joins(i) == level)
            .map(|i| shapes[i].0)
            .max()
            .unwrap_or(0);
        let root: Vec<F> = (0..s_max)
            .map(|c| t.challenge(&format!("{prefix}/root/{c}")))
            .collect();
        assert!(
            record.root_challenges() == root,
            "level {level}: the root challenges"
        );
        for i in (0..shapes.len()).filter(|&i| joins(i) == level) {
            let r = root[s_max - shapes[i].0..].to_vec();
            let value = roots[i]
                .iter()
                .enumerate()
                .fold(F::ZERO, |sum, (y, &v)| sum + v * eq_at_vertex(&r, y));
            claims[i] = Some((r, value));
        }
        let present: Vec<usize> = (0..shapes.len()).filter(|&i| joins(i) <= level).collect();
        assert_eq!(record.forests(), present, "level {level}: the instances");
        let entering: Vec<(Vec<F>, F)> = present
            .iter()
            .map(|&i| claims[i].clone().expect("joined"))
            .collect();

        let gamma = t.challenge(&format!("{prefix}/gamma"));
        assert_eq!(*record.gamma(), gamma, "level {level}: γ");
        let n_max = entering
            .iter()
            .map(|(r, _)| r.len())
            .max()
            .expect("an instance");
        let mut claim = F::ZERO;
        let mut weight = F::ONE;
        for (r, value) in &entering {
            let scale = F::from(F65537::from_u64_reduced(1 << (n_max - r.len())));
            claim += weight * scale * *value;
            weight *= gamma;
        }
        let proof = record.proof();
        assert_eq!(
            proof.rounds.len(),
            n_max,
            "level {level}: one round per variable"
        );
        let mut point = Vec::new();
        for (j, round) in proof.rounds.iter().enumerate() {
            assert_eq!(round.len(), 3, "level {level}: round {j} holds 3 values");
            t.absorb_message(&format!("{prefix}/round/{j}"), round);
            let r = t.challenge(&format!("{prefix}/r/{j}"));
            claim = RoundPoly::new(round).expect("degree 3").evaluate(claim, r);
            point.push(r);
        }
        assert!(record.point() == point, "level {level}: the point");
        assert_eq!(proof.final_evals.len(), 2 * entering.len());
        t.absorb_message(&format!("{prefix}/evals"), &proof.final_evals);
        let rho = t.challenge(&format!("{prefix}/line"));
        assert_eq!(*record.rho(), rho, "level {level}: ρ");
        let mut batched = F::ZERO;
        let mut weight = F::ONE;
        for (k, ((r, _), evals)) in entering
            .iter()
            .zip(proof.final_evals.chunks_exact(2))
            .enumerate()
        {
            let own = &point[n_max - r.len()..];
            let (u, w) = (evals[0], evals[1]);
            batched += weight * eq_points(r, own) * u * w;
            weight *= gamma;
            let mut next = vec![rho];
            next.extend_from_slice(own);
            let value = u + rho * (w - u);
            let mine = &record.next_claims()[k];
            assert!(
                mine.point == next,
                "level {level}: instance {k}'s next point"
            );
            assert_eq!(
                mine.value, value,
                "level {level}: instance {k}'s next value"
            );
            claims[present[k]] = Some((next, value));
        }
        assert_eq!(batched, claim, "level {level}: the final check");
    }
    for (i, (claim, leaves)) in proven.leaf_claims().iter().zip(&leaves).enumerate() {
        assert_eq!(claim.point.len(), shapes[i].0 + shapes[i].1);
        assert_eq!(
            claim.value,
            evaluate_base(leaves, &claim.point).expect("2^n leaves at n coordinates"),
            "forest {i}: the leaves' extension at the leaf point"
        );
    }
    assert_eq!(
        t.inner.state(),
        prover.inner.state(),
        "the transcripts end in the same state"
    );
}

#[test]
fn forests_over_the_extension_of_f65537_verify() {
    for seed in 0..20 {
        forest_batch::<F65537Ext2>(seed);
    }
}

#[test]
fn forests_with_challenges_in_f65537_verify() {
    for seed in 0..20 {
        forest_batch::<F65537>(seed);
    }
}
