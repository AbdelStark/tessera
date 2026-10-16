// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A reference prover for the sumchecks of RFC-0010, for tests only: batched sumchecks of
//! products of multilinear tables with front-padding, and batches of product trees proven level
//! by level, honestly or as a prover that lies in one round.
//!
//! It is written from RFC-0010 ('Multilinear extensions', 'Tree levels', 'Batching within a
//! stage', 'Sumcheck messages') and RFC-0004 'Transcript', and shares no code with `tv-field`,
//! `tv-hash` or `tv-core`, which its proofs test: its field is `u128` arithmetic modulo `P`, its
//! transcript runs on this crate's reference BLAKE3, and it pads a smaller instance implicitly,
//! a padded round contributing a constant, where the Python oracle of `spec/vectors/sumcheck/`
//! pads the tables themselves. It gives the verifier's tests what conformance vectors cannot
//! hold: proofs of seeded instances of any shape, and of cheating provers.

use std::ops::{Add, Mul, Neg, Sub};

use crate::blake3::derive_key;

/// The Goldilocks modulus `2^64 − 2^32 + 1`.
pub const P: u64 = 0xFFFF_FFFF_0000_0001;

/// An element `c0 + c1·X` of `Fp2 = Fp[X]/(X^2 − 7)`, both coefficients below `P`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Fe(pub u64, pub u64);

fn reduce(x: u128) -> u64 {
    (x % u128::from(P)) as u64
}

impl Fe {
    /// 0.
    pub const ZERO: Fe = Fe(0, 0);
    /// 1.
    pub const ONE: Fe = Fe(1, 0);

    /// `c0 + c1·X`, each coefficient reduced modulo `P`.
    pub fn new(c0: u64, c1: u64) -> Self {
        Fe(c0 % P, c1 % P)
    }

    /// The base-field element `x mod P`.
    pub fn base(x: u64) -> Self {
        Fe(x % P, 0)
    }

    /// The canonical 16-byte encoding, `c0` then `c1`, little-endian.
    pub fn to_le_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.0.to_le_bytes());
        out[8..].copy_from_slice(&self.1.to_le_bytes());
        out
    }

    /// `self^e` by square-and-multiply.
    pub fn pow(self, mut e: u128) -> Self {
        let (mut acc, mut square) = (Fe::ONE, self);
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * square;
            }
            square = square * square;
            e >>= 1;
        }
        acc
    }

    /// The inverse, through the norm `c0^2 − 7·c1^2` of `Fp`, or `None` for zero.
    pub fn inverse(self) -> Option<Self> {
        let norm = (self * Fe(self.0, (P - self.1) % P)).0;
        if norm == 0 {
            return None;
        }
        let inverse_norm = Fe::base(norm).pow(u128::from(P - 2)).0;
        Some(Fe(self.0, (P - self.1) % P) * Fe::base(inverse_norm))
    }
}

impl Add for Fe {
    type Output = Fe;
    fn add(self, rhs: Fe) -> Fe {
        Fe(
            reduce(u128::from(self.0) + u128::from(rhs.0)),
            reduce(u128::from(self.1) + u128::from(rhs.1)),
        )
    }
}

impl Neg for Fe {
    type Output = Fe;
    fn neg(self) -> Fe {
        Fe((P - self.0) % P, (P - self.1) % P)
    }
}

impl Sub for Fe {
    type Output = Fe;
    fn sub(self, rhs: Fe) -> Fe {
        self + -rhs
    }
}

impl Mul for Fe {
    type Output = Fe;
    /// `(a0 + a1·X)(b0 + b1·X)` with `X^2 = 7`.
    fn mul(self, rhs: Fe) -> Fe {
        let (a0, a1, b0, b1) = (
            u128::from(self.0),
            u128::from(self.1),
            u128::from(rhs.0),
            u128::from(rhs.1),
        );
        let p = u128::from(P);
        Fe(
            reduce(a0 * b0 % p + 7 * (a1 * b1 % p)),
            reduce(a0 * b1 % p + a1 * b0 % p),
        )
    }
}

// ----- The transcript (RFC-0004 'Transcript') ----------------------------------------------------

/// TCE `Ascii`: the `u64` length, then the bytes.
fn ascii(text: &str) -> Vec<u8> {
    assert!(text.bytes().all(|b| (0x20..=0x7E).contains(&b)), "{text:?}");
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    out
}

/// `TCE(SessionContext)`: `Ascii(protocol) ‖ u16(mode) ‖ Ascii(params) ‖ statement_digest ‖
/// session_id`, the mode tag 0 for live and 1 for non-interactive.
pub fn context_tce(
    protocol: &str,
    mode: u16,
    params: &str,
    statement_digest: &[u8; 32],
    session_id: &[u8; 16],
) -> Vec<u8> {
    let mut out = ascii(protocol);
    out.extend_from_slice(&mode.to_le_bytes());
    out.extend(ascii(params));
    out.extend_from_slice(statement_digest);
    out.extend_from_slice(session_id);
    out
}

/// The prover's copy of the Fiat–Shamir transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transcript {
    state: [u8; 32],
    counter: u64,
}

impl Transcript {
    /// `state = H("transcript-init", context_tce)`.
    pub fn new(context_tce: &[u8]) -> Self {
        Self {
            state: derive_key("tessera/v1/transcript-init", context_tce),
            counter: 0,
        }
    }

    /// The state.
    pub fn state(&self) -> [u8; 32] {
        self.state
    }

    /// The squeezes so far.
    pub fn counter(&self) -> u64 {
        self.counter
    }

    /// `state = H("transcript-absorb", state ‖ Ascii(label) ‖ tce)`.
    pub fn absorb(&mut self, label: &str, tce: &[u8]) {
        let input = [&self.state[..], &ascii(label), tce].concat();
        self.state = derive_key("tessera/v1/transcript-absorb", &input);
    }

    /// `out = H("transcript-squeeze", state ‖ Ascii(label) ‖ u64(counter))`, then the counter
    /// moves on and `state = H("transcript-ratchet", state ‖ out)`.
    pub fn squeeze(&mut self, label: &str) -> [u8; 32] {
        let input = [&self.state[..], &ascii(label), &self.counter.to_le_bytes()].concat();
        let out = derive_key("tessera/v1/transcript-squeeze", &input);
        self.counter += 1;
        self.state = derive_key("tessera/v1/transcript-ratchet", &[self.state, out].concat());
        out
    }

    /// The first squeeze whose first two little-endian words are below `P`, as `(c0, c1)`.
    pub fn challenge_fp2(&mut self, label: &str) -> Fe {
        loop {
            let out = self.squeeze(label);
            let word = |i: usize| u64::from_le_bytes(out[8 * i..8 * i + 8].try_into().unwrap());
            if word(0) < P && word(1) < P {
                return Fe(word(0), word(1));
            }
        }
    }
}

/// TCE `Vec<Fp2>`: the `u64` length, then each element's 16 bytes.
pub fn tce(values: &[Fe]) -> Vec<u8> {
    let mut out = (values.len() as u64).to_le_bytes().to_vec();
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

// ----- Multilinear extensions (RFC-0010 'Multilinear extensions') --------------------------------

/// `eq(a, b) = Π_i (a_i·b_i + (1 − a_i)·(1 − b_i))`.
pub fn eq(a: &[Fe], b: &[Fe]) -> Fe {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(Fe::ONE, |acc, (&x, &y)| {
        acc * (x * y + (Fe::ONE - x) * (Fe::ONE - y))
    })
}

/// The table of `eq(point, x)` over `x` in `0..2^n`, `x_1` the lowest bit: built a coordinate at
/// a time, each doubling the table, `x_i = 0` in the low half.
pub fn eq_table(point: &[Fe]) -> Vec<Fe> {
    let mut table = vec![Fe::ONE];
    for &r in point {
        let low: Vec<Fe> = table.iter().map(|&t| t * (Fe::ONE - r)).collect();
        let high: Vec<Fe> = table.iter().map(|&t| t * r).collect();
        table = [low, high].concat();
    }
    table
}

/// `f[2k] + r·(f[2k + 1] − f[2k])`: the table with its lowest variable bound to `r`.
pub fn bind(table: &[Fe], r: Fe) -> Vec<Fe> {
    table
        .chunks_exact(2)
        .map(|pair| pair[0] + r * (pair[1] - pair[0]))
        .collect()
}

/// `f̃(point)`, by binding the variables one at a time from the lowest.
pub fn mle(table: &[Fe], point: &[Fe]) -> Fe {
    assert_eq!(
        table.len(),
        1 << point.len(),
        "a table of 2^n entries takes n coordinates"
    );
    point
        .iter()
        .fold(table.to_vec(), |table, &r| bind(&table, r))[0]
}

/// `p(r)` for the polynomial through `(k, values[k])`, `k = 0, 1, …`, by Lagrange's formula.
pub fn lagrange(values: &[Fe], r: Fe) -> Fe {
    let mut out = Fe::ZERO;
    for (i, &value) in values.iter().enumerate() {
        let (mut numerator, mut denominator) = (Fe::ONE, Fe::ONE);
        for m in 0..values.len() {
            if m != i {
                numerator = numerator * (r - Fe::base(m as u64));
                denominator = denominator * (Fe::base(i as u64) - Fe::base(m as u64));
            }
        }
        out = out + value * numerator * denominator.inverse().unwrap();
    }
    out
}

// ----- Batched sumchecks (RFC-0010 'Batching within a stage', 'Sumcheck messages') --------------

/// One instance: the product of its tables, all of `2^n` entries for its `n` variables, or a
/// LogUp fraction tree's level relation over four of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Product {
    /// The tables whose values at the instance's point the proof sends, in order.
    pub tables: Vec<Vec<Fe>>,
    /// A factor the verifier computes itself, such as `eq(r, x)`: multiplied in, never sent.
    pub public: Option<Vec<Fe>>,
    /// `None` for the product of the tables. `Some(λ)` for a fraction tree's level (RFC-0010
    /// 'Tree levels'): the tables are `N(0 ‖ x)`, `N(1 ‖ x)`, `D(0 ‖ x)` and `D(1 ‖ x)`, and
    /// the relation is `N_0·D_1 + N_1·D_0 + λ·D_0·D_1`, times the public factor.
    pub lambda: Option<Fe>,
}

impl Product {
    /// A product of `tables`, all of one size, with no public factor.
    pub fn new(tables: Vec<Vec<Fe>>) -> Self {
        Self {
            tables,
            public: None,
            lambda: None,
        }
    }

    /// Every factor, the sent ones first.
    fn factors(&self) -> impl Iterator<Item = &Vec<Fe>> {
        self.tables.iter().chain(&self.public)
    }

    /// Its number of variables.
    pub fn vars(&self) -> usize {
        self.factors()
            .next()
            .expect("a product of at least one table")
            .len()
            .trailing_zeros() as usize
    }

    /// The relation's degree in each variable.
    fn degree(&self) -> usize {
        let public = usize::from(self.public.is_some());
        match self.lambda {
            None => self.tables.len() + public,
            Some(_) => 2 + public,
        }
    }

    /// The relation's value from its factors' values, in the order of [`Product::factors`].
    fn combine(&self, values: &[Fe]) -> Fe {
        match self.lambda {
            None => values.iter().fold(Fe::ONE, |p, &v| p * v),
            Some(lambda) => {
                let [n0, n1, d0, d1] = values[..4] else {
                    panic!("a fraction tree's level has four tables")
                };
                let public = values[4..].iter().fold(Fe::ONE, |p, &v| p * v);
                (n0 * d1 + n1 * d0 + lambda * d0 * d1) * public
            }
        }
    }

    /// Its claim: the sum of the relation over the hypercube.
    pub fn claim(&self) -> Fe {
        (0..1usize << self.vars()).fold(Fe::ZERO, |sum, x| {
            let values: Vec<Fe> = self.factors().map(|table| table[x]).collect();
            sum + self.combine(&values)
        })
    }
}

/// How a cheating prover deviates in one round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lie {
    /// The round, from 0.
    pub round: usize,
    /// Added to the values the round's message sends, `p(0), p(2), …, p(d)`.
    pub offset: Fe,
    /// The value it is added to, by its position in the message, or every value for `None`.
    pub value: Option<usize>,
    /// Whether the final evaluations are forged so that the final check passes: the first sent
    /// evaluation of the first instance is solved for.
    pub forge: bool,
}

/// A prover's messages for one batched sumcheck, and what it knows of the verifier's view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    /// `γ`.
    pub gamma: Fe,
    /// Round `j`'s message: `p_j(0), p_j(2), …, p_j(d)`.
    pub rounds: Vec<Vec<Fe>>,
    /// The challenges, in round order.
    pub challenges: Vec<Fe>,
    /// The verifier's claims `claim_1, …, claim_{n_max + 1}`, from the messages sent, by
    /// Lagrange's formula.
    pub claims: Vec<Fe>,
    /// The final evaluations sent, instance by instance.
    pub evals: Vec<Fe>,
    /// The true values of the sent tables at each instance's point, instance by instance.
    pub true_evals: Vec<Fe>,
    /// Each instance's public factor at its point, or 1.
    pub publics: Vec<Fe>,
}

/// The prover of one batched sumcheck under `prefix`, of degree `degree`, ending with the
/// final evaluations absorbed under `<prefix>/evals`.
///
/// A smaller instance is padded implicitly: in the first `n_max − n_i` rounds it is constant in
/// the variable bound, so it adds `2^{n_max − n_i − j − 1}·c_i` to every value of round `j`, and
/// its tables are bound only in the rounds of its own variables.
pub fn prove(
    transcript: &mut Transcript,
    prefix: &str,
    degree: usize,
    instances: &[Product],
    lie: Option<Lie>,
) -> Proof {
    let gamma = transcript.challenge_fp2(&format!("{prefix}/gamma"));
    prove_after_gamma(transcript, prefix, degree, gamma, instances, lie)
}

/// [`prove`] from its batching challenge `γ`, already squeezed: a tree level squeezes `λ`
/// between the two, and its fraction trees' instances carry it.
pub fn prove_after_gamma(
    transcript: &mut Transcript,
    prefix: &str,
    degree: usize,
    gamma: Fe,
    instances: &[Product],
    lie: Option<Lie>,
) -> Proof {
    let n_max = instances
        .iter()
        .map(Product::vars)
        .max()
        .expect("an instance");
    for instance in instances {
        assert!(instance.factors().all(|t| t.len() == 1 << instance.vars()));
        assert!(
            instance.degree() <= degree,
            "degree at least the relation's"
        );
        assert!(
            instance.lambda.is_none() || lie.is_none_or(|lie| !lie.forge),
            "forging solves for a product's evaluation"
        );
    }
    let weights: Vec<Fe> = (0..instances.len()).map(|i| gamma.pow(i as u128)).collect();
    let mut tables: Vec<Vec<Vec<Fe>>> = instances
        .iter()
        .map(|i| i.factors().cloned().collect())
        .collect();
    let pads: Vec<usize> = instances.iter().map(|i| n_max - i.vars()).collect();
    let sums: Vec<Fe> = instances.iter().map(Product::claim).collect();
    let mut claim = weights
        .iter()
        .zip(&sums)
        .zip(&pads)
        .fold(Fe::ZERO, |acc, ((&w, &c), &m)| {
            acc + w * c * Fe::base(2).pow(m as u128)
        });
    let (mut rounds, mut challenges, mut claims) = (Vec::new(), Vec::new(), vec![claim]);
    for j in 0..n_max {
        let at = |t: Fe| -> Fe {
            let mut value = Fe::ZERO;
            for (((weight, factors), (&pad, &sum)), instance) in weights
                .iter()
                .zip(&tables)
                .zip(pads.iter().zip(&sums))
                .zip(instances)
            {
                let contribution = if j < pad {
                    sum * Fe::base(2).pow((pad - j - 1) as u128)
                } else {
                    (0..factors[0].len() / 2).fold(Fe::ZERO, |acc, x| {
                        let values: Vec<Fe> = factors
                            .iter()
                            .map(|f| f[2 * x] + t * (f[2 * x + 1] - f[2 * x]))
                            .collect();
                        acc + instance.combine(&values)
                    })
                };
                value = value + *weight * contribution;
            }
            value
        };
        let mut message: Vec<Fe> = [0]
            .into_iter()
            .chain(2..=degree as u64)
            .map(|k| at(Fe::base(k)))
            .collect();
        if let Some(lie) = lie.filter(|lie| lie.round == j) {
            for (position, value) in message.iter_mut().enumerate() {
                if lie.value.is_none_or(|only| only == position) {
                    *value = *value + lie.offset;
                }
            }
        }
        transcript.absorb(&format!("{prefix}/round/{j}"), &tce(&message));
        let r = transcript.challenge_fp2(&format!("{prefix}/r/{j}"));
        let nodes: Vec<Fe> = [message[0], claim - message[0]]
            .into_iter()
            .chain(message[1..].iter().copied())
            .collect();
        claim = lagrange(&nodes, r);
        for (factors, &pad) in tables.iter_mut().zip(&pads) {
            if j >= pad {
                for f in factors.iter_mut() {
                    *f = bind(f, r);
                }
            }
        }
        rounds.push(message);
        challenges.push(r);
        claims.push(claim);
    }
    let mut true_evals = Vec::new();
    let mut publics = Vec::new();
    for (instance, factors) in instances.iter().zip(&tables) {
        true_evals.extend(factors[..instance.tables.len()].iter().map(|f| f[0]));
        publics.push(if instance.public.is_some() {
            factors[instance.tables.len()][0]
        } else {
            Fe::ONE
        });
    }
    let mut evals = true_evals.clone();
    if lie.is_some_and(|lie| lie.forge) {
        // Solve Σ_i γ^i · public_i · Π_k e_ik = claim for e_00.
        let mut offset = 0;
        let mut rest = Fe::ZERO;
        let mut first = Fe::ONE;
        for (i, instance) in instances.iter().enumerate() {
            let own = &evals[offset..offset + instance.tables.len()];
            if i == 0 {
                first = weights[0] * publics[0] * own[1..].iter().fold(Fe::ONE, |p, &e| p * e);
            } else {
                rest = rest + weights[i] * publics[i] * own.iter().fold(Fe::ONE, |p, &e| p * e);
            }
            offset += instance.tables.len();
        }
        evals[0] = (claim - rest) * first.inverse().expect("a nonzero cofactor");
    }
    transcript.absorb(&format!("{prefix}/evals"), &tce(&evals));
    Proof {
        gamma,
        rounds,
        challenges,
        claims,
        evals,
        true_evals,
        publics,
    }
}

// ----- Product trees (RFC-0010 'Tree levels') ----------------------------------------------------

/// Every level of a product tree, the root first: `V_ℓ[x] = V_{ℓ+1}[2x]·V_{ℓ+1}[2x + 1]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductTree {
    /// `levels[0]` is the root, one entry; `levels[h]` the leaves.
    pub levels: Vec<Vec<Fe>>,
}

impl ProductTree {
    /// The tree over `leaves`, a power of two of them.
    pub fn new(leaves: Vec<Fe>) -> Self {
        assert!(leaves.len().is_power_of_two());
        let mut levels = vec![leaves];
        while levels[0].len() > 1 {
            let above = levels[0]
                .chunks_exact(2)
                .map(|pair| pair[0] * pair[1])
                .collect();
            levels.insert(0, above);
        }
        Self { levels }
    }

    /// Its height `h`: the leaves are level `h`.
    pub fn height(&self) -> usize {
        self.levels.len() - 1
    }

    /// The root, `V_0[0]`.
    pub fn root(&self) -> Fe {
        self.levels[0][0]
    }

    /// The leaves, level `h`.
    pub fn leaves(&self) -> &[Fe] {
        &self.levels[self.height()]
    }
}

/// A claim on a tree's level: `Ṽ(point) = value`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    /// The tree, by its index in the batch.
    pub tree: usize,
    /// The point, low variables first.
    pub point: Vec<Fe>,
    /// The value.
    pub value: Fe,
}

/// One level of a batch of product trees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Level {
    /// The claims entering the level, in schedule order.
    pub claims: Vec<Claim>,
    /// The level's sumcheck; its evaluations are `u_0, w_0, u_1, w_1, …`.
    pub proof: Proof,
    /// The line challenge `ρ`.
    pub rho: Fe,
    /// The claims on the next level, in the same order.
    pub next: Vec<Claim>,
}

/// The levels of a batch of product trees, from the roots: at level `ℓ` of a batch whose
/// tallest tree has height `H`, the trees of height `h ≥ H − ℓ` are present, in the order of
/// `trees`, a tree joining at level `H − h` with its root claim on no variable.
pub fn prove_trees(transcript: &mut Transcript, trees: &[ProductTree]) -> Vec<Level> {
    let tallest = trees.iter().map(ProductTree::height).max().expect("a tree");
    let mut current: Vec<Option<Claim>> = vec![None; trees.len()];
    let mut levels = Vec::new();
    for level in 0..tallest {
        for (index, tree) in trees.iter().enumerate() {
            if tallest - tree.height() == level {
                current[index] = Some(Claim {
                    tree: index,
                    point: Vec::new(),
                    value: tree.root(),
                });
            }
        }
        let claims: Vec<Claim> = current.iter().flatten().cloned().collect();
        let instances: Vec<Product> = claims
            .iter()
            .map(|claim| {
                let own = claim.point.len();
                let below = &trees[claim.tree].levels[own + 1];
                assert_eq!(below.len(), 2 << own);
                Product {
                    tables: vec![
                        below.iter().step_by(2).copied().collect(),
                        below.iter().skip(1).step_by(2).copied().collect(),
                    ],
                    public: Some(eq_table(&claim.point)),
                    lambda: None,
                }
            })
            .collect();
        let prefix = format!("tree/{level}");
        let proof = prove(transcript, &prefix, 3, &instances, None);
        let rho = transcript.challenge_fp2(&format!("{prefix}/line"));
        let next: Vec<Claim> = claims
            .iter()
            .zip(proof.evals.chunks_exact(2))
            .map(|(claim, pair)| {
                let own = &proof.challenges[proof.challenges.len() - claim.point.len()..];
                Claim {
                    tree: claim.tree,
                    point: [&[rho][..], own].concat(),
                    value: pair[0] + rho * (pair[1] - pair[0]),
                }
            })
            .collect();
        for claim in &next {
            current[claim.tree] = Some(claim.clone());
        }
        levels.push(Level {
            claims,
            proof,
            rho,
            next,
        });
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SplitMix64;

    fn random(rng: &mut SplitMix64) -> Fe {
        Fe::new(rng.next_u64(), rng.next_u64())
    }

    #[test]
    fn field_arithmetic_follows_its_laws() {
        let mut rng = SplitMix64::new(1);
        for _ in 0..200 {
            let (a, b, c) = (random(&mut rng), random(&mut rng), random(&mut rng));
            assert_eq!(a * (b + c), a * b + a * c);
            assert_eq!((a - b) + b, a);
            if a != Fe::ZERO {
                assert_eq!(a * a.inverse().unwrap(), Fe::ONE);
            }
        }
        // X^2 = 7.
        assert_eq!(Fe(0, 1) * Fe(0, 1), Fe::base(7));
        assert_eq!(Fe::ZERO.inverse(), None);
    }

    #[test]
    fn extensions_agree_with_their_definitions() {
        let mut rng = SplitMix64::new(2);
        for n in 0..6 {
            let point: Vec<Fe> = (0..n).map(|_| random(&mut rng)).collect();
            let table: Vec<Fe> = (0..1 << n).map(|_| random(&mut rng)).collect();
            let weights = eq_table(&point);
            let by_definition = (0..1usize << n).fold(Fe::ZERO, |acc, x| {
                let bits: Vec<Fe> = (0..n).map(|i| Fe::base(((x >> i) & 1) as u64)).collect();
                assert_eq!(weights[x], eq(&point, &bits));
                acc + table[x] * eq(&point, &bits)
            });
            assert_eq!(mle(&table, &point), by_definition);
        }
    }

    #[test]
    fn an_honest_proof_satisfies_its_own_claims() {
        let mut rng = SplitMix64::new(3);
        let table = |n: usize, rng: &mut SplitMix64| -> Vec<Fe> {
            (0..1 << n).map(|_| random(rng)).collect()
        };
        let instances = vec![
            Product::new(vec![table(4, &mut rng), table(4, &mut rng)]),
            Product::new(vec![table(1, &mut rng)]),
            Product::new(vec![table(0, &mut rng), table(0, &mut rng)]),
        ];
        let mut transcript = Transcript::new(b"context");
        let proof = prove(&mut transcript, "stage/0", 2, &instances, None);
        assert_eq!(proof.evals, proof.true_evals);
        let mut combination = Fe::ZERO;
        let mut offset = 0;
        for (i, instance) in instances.iter().enumerate() {
            let point = &proof.challenges[4 - instance.vars()..];
            let mut value = Fe::ONE;
            for table in &instance.tables {
                assert_eq!(mle(table, point), proof.evals[offset]);
                value = value * proof.evals[offset];
                offset += 1;
            }
            combination = combination + proof.gamma.pow(i as u128) * value;
        }
        assert_eq!(combination, *proof.claims.last().unwrap());
    }

    #[test]
    fn tree_levels_end_at_the_leaves() {
        let mut rng = SplitMix64::new(4);
        let trees: Vec<ProductTree> = [2usize, 4, 1]
            .iter()
            .map(|&h| ProductTree::new((0..1 << h).map(|_| Fe::base(rng.next_u64())).collect()))
            .collect();
        let mut transcript = Transcript::new(b"context");
        let levels = prove_trees(&mut transcript, &trees);
        assert_eq!(levels.len(), 4);
        assert_eq!(levels[0].claims.len(), 1);
        assert_eq!(levels[3].next.len(), 3);
        for claim in &levels[3].next {
            assert_eq!(mle(trees[claim.tree].leaves(), &claim.point), claim.value);
        }
    }
}
