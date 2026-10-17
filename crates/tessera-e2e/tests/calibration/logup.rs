// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lookups outside their table, and a verifier from RFC-0008 'Fraction tree' and RFC-0010 'Tree
//! levels' (RFC-0008 'Testing Strategy': "the empirical false-accept rate of a single random-`α`
//! trial stays at or below `(N + T)/|F|`").
//!
//! An instance holds `T` rows and `N` lookups, keys without values, and the multiplicities the
//! prover commits. With `N` keys outside the table, the identity
//! `R(α) = Σ_i 1/(α − a_i) − Σ_j m_j/(α − t_j) = 0` is a rational function whose numerator, over
//! the `N + T` distinct poles, has degree at most `N + T − 1`: it holds for at most that many
//! values of `α` (RFC-0008 'Relation'). The prover is `tessera_logup::prove_logup` itself: it
//! builds honest trees over the bad leaves, so every tree level is a true statement and passes,
//! and the verifier accepts exactly when the roots balance, that is when `α` is no pole and
//! `R(α) = 0`.
//!
//! [`Lookups::adversarial`] chooses the multiplicities so that `R` has `N + T − 1` roots in the
//! field, the most it can: `T` of them chosen at the top of the field, and `N − 1` more that the
//! lookups' residues force, kept when they lie in the field. Every instance's accepting set is
//! computed by brute force over the field, from the leaves.
//!
//! The levels' own sumchecks are not attacked: each is honest, so they add nothing to the rate,
//! which measures the identity alone.

use tessera_field::{Field, PrimeField};
use tessera_logup::{LogUpInstance, TreeId, Tuples, prove_logup};
use tessera_sumcheck::{SumcheckProof, SumcheckTranscript};

use super::{F, FIELD, SplitMix64, Verdict, eq, int, mle, round_value};

/// `⌈log2 n⌉`: the height of a tree of `n` leaves before padding.
fn height(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}

/// `values` padded to `2^h` entries with `fill`.
fn padded(values: &[F], h: usize, fill: F) -> Vec<F> {
    let mut out = values.to_vec();
    out.resize(1 << h, fill);
    out
}

/// One LogUp instance over a table without values: its rows, its lookups' keys and the
/// committed multiplicities, with the values of `α` at which a verifier accepts it.
pub struct Lookups {
    rows: Vec<F>,
    keys: Vec<F>,
    multiplicities: Vec<F>,
    /// The leaves' distinct keys, in order: the values of `α` at which a denominator is zero.
    poles: Vec<F>,
    /// The values of `α` at which the identity holds, in order.
    accepting: Vec<F>,
}

/// What the prover sends (RFC-0014 'Transcript schedule', rows 4 and 5): the roots, each level's
/// messages and the values.
struct Proof {
    roots: Vec<F>,
    levels: Vec<SumcheckProof<F>>,
    values: Vec<F>,
}

/// A fraction tree's two claims on one level, at one point (RFC-0010 'Tree levels').
struct Claims {
    point: Vec<F>,
    numerator: F,
    denominator: F,
}

impl Lookups {
    /// The instance of `keys` into `rows` with the multiplicities `m`, `2^{h_t}` of them, and its
    /// accepting set.
    fn new(rows: Vec<F>, keys: Vec<F>, m: Vec<F>) -> Self {
        assert_eq!(
            m.len(),
            1 << height(rows.len()),
            "a multiplicity per table leaf"
        );
        let mut instance = Self {
            rows,
            keys,
            multiplicities: m,
            poles: Vec::new(),
            accepting: Vec::new(),
        };
        let leaves = instance.leaves();
        instance.poles = leaves.iter().map(|&(_, key)| key).collect();
        instance.poles.sort();
        instance.poles.dedup();
        instance.accepting = accepting(&leaves);
        instance
    }

    /// `lookups` lookups of random rows of a table of `rows` keys `0, …, T − 1`, with the honest
    /// multiplicities, the lookup tree's padding counted on row 0 (RFC-0008 'Fraction tree').
    pub fn honest(lookups: usize, rows: usize, rng: &mut SplitMix64) -> Self {
        let table: Vec<F> = (0..rows).map(int).collect();
        let hits: Vec<usize> = (0..lookups)
            .map(|_| rng.below(rows as u64) as usize)
            .collect();
        let mut m = vec![F::ZERO; 1 << height(rows)];
        for &j in &hits {
            m[j] += F::ONE;
        }
        m[0] += int((1 << height(lookups)) - lookups);
        let keys = hits.iter().map(|&j| table[j]).collect();
        Self::new(table, keys, m)
    }

    /// `lookups` keys `T, …, T + N − 1` outside a table of `rows` keys `0, …, T − 1`, and
    /// multiplicities that make the identity hold at `N + T − 1` values of `α`: the first
    /// candidate, by increasing `c`, whose `N − 1` forced roots lie in the field, the chosen
    /// roots being `P − 1 − c − s`, `s < T`.
    ///
    /// Over the poles `S`, the rows and the keys, `R(α) = Σ_{x∈S} c_x/(α − x)` with residue
    /// `c_x = 1` at each key: its numerator `R·Π_{x∈S}(α − x)` is `Z·G`, `Z = Π_s (α − z_s)`
    /// over the chosen roots, and `G` of degree `N − 1` interpolates `G(a_i) = Q'(a_i)/Z(a_i)`,
    /// `Q'(x) = Π_{y∈S, y≠x}(x − y)`, which makes the residues at the keys 1. A row's residue is
    /// then `Z(t_j)·G(t_j)/Q'(t_j)`: the multiplicity `m_j` is its opposite, plus the lookup
    /// tree's padding at row 0.
    pub fn adversarial(lookups: usize, rows: usize) -> Self {
        (0..FIELD)
            .find_map(|c| candidate(lookups, rows, c))
            .expect("an instance whose forced roots lie in the field")
    }

    /// The rows.
    pub fn rows(&self) -> &[F] {
        &self.rows
    }

    /// The lookups' keys.
    pub fn keys(&self) -> &[F] {
        &self.keys
    }

    /// The leaves' distinct keys, in order.
    pub fn poles(&self) -> &[F] {
        &self.poles
    }

    /// The values of `α` at which the identity holds, no pole among them, in order.
    pub fn accepting(&self) -> &[F] {
        &self.accepting
    }

    /// Every leaf of both trees as `(numerator, key)`, each padded with row 0 (RFC-0008 'Fraction
    /// tree'): the lookup tree's `(1, a_i)`, then `(1, t_0)`; the table tree's `(−m_j, t_j)`,
    /// then `(−m_j, t_0)`.
    fn leaves(&self) -> Vec<(F, F)> {
        let lookups = padded(&self.keys, height(self.keys.len()), self.rows[0]);
        let table = padded(&self.rows, height(self.rows.len()), self.rows[0]);
        lookups
            .into_iter()
            .map(|key| (F::ONE, key))
            .chain(self.multiplicities.iter().map(|&m| -m).zip(table))
            .collect()
    }

    /// One trial: the instance proven by `prove_logup` on `prover`'s transcript, after the
    /// committed columns, and checked on `verifier`'s. The verdict carries `α`.
    pub fn trial(
        &self,
        prover: &mut impl SumcheckTranscript<F>,
        verifier: &mut impl SumcheckTranscript<F>,
    ) -> Verdict {
        // The lookups and the multiplicities are bound before α, as their commitments are
        // (RFC-0014 'Transcript schedule', row 2).
        let committed: Vec<F> = self
            .keys
            .iter()
            .chain(&self.multiplicities)
            .copied()
            .collect();
        prover.absorb_message("round1", &committed);
        let instance = LogUpInstance::new(
            Tuples::new(&self.rows, None),
            Tuples::new(&self.keys, None),
            &self.multiplicities,
        );
        let proven = prove_logup::<F, _>(prover, &[instance], &[], &TreeId::paired(1))
            .expect("a well-formed instance");
        let proof = Proof {
            roots: proven.roots().to_vec(),
            levels: proven
                .levels()
                .iter()
                .map(|level| level.proof().clone())
                .collect(),
            values: proven.values().to_vec(),
        };
        verifier.absorb_message("round1", &committed);
        self.verify(verifier, &proof)
    }

    /// Checks a proof of the instance as RFC-0008 'Fraction tree' and RFC-0010 'Tree levels' say
    /// a verifier does, the committed columns at hand in place of their openings:
    ///
    /// 1. `α`, `β`; the roots `n_a, d_a, n_t, d_t`, absorbed, with `d_a ≠ 0`, `d_t ≠ 0` and
    ///    `n_a·d_t + n_t·d_a = 0`;
    /// 2. each level `ℓ`: `γ`, `λ`, the instances present, the lookup tree then the table tree,
    ///    each joining at `H − h` with its root, combined as `Σ_i γ^i·2^{n_max − n_i}·(p_i +
    ///    λ·q_i)`; rounds of 3 values; four evaluations `N_0, N_1, D_0, D_1` per instance with
    ///    `Σ_i γ^i·eq(r_i, r'_i)·(N_0·D_1 + N_1·D_0 + λ·D_0·D_1)` the last claim; `ρ` and the
    ///    claims on the next level at `ρ ‖ r'_i`;
    /// 3. no values, absorbed; the lookup tree's leaf numerator 1, the table tree's leaf
    ///    denominator `α − t̃(ρ_t)`, and the claims the openings owe: `kẽy(ρ_a) = α − q_a` and
    ///    `m̃(ρ_t) = −p_t`.
    fn verify(&self, t: &mut impl SumcheckTranscript<F>, proof: &Proof) -> Verdict {
        let alpha = t.challenge("logup/alpha");
        let _beta = t.challenge("logup/beta");
        let reject = Verdict {
            accepted: false,
            challenge: alpha,
        };
        let &[n_a, d_a, n_t, d_t] = proof.roots.as_slice() else {
            return reject;
        };
        t.absorb_message("logup/roots", &proof.roots);
        if d_a == F::ZERO || d_t == F::ZERO || n_a * d_t + n_t * d_a != F::ZERO {
            return reject;
        }

        let heights = [height(self.keys.len()), height(self.rows.len())];
        let tallest = heights[0].max(heights[1]);
        if proof.levels.len() != tallest {
            return reject;
        }
        let root = |numerator, denominator| Claims {
            point: Vec::new(),
            numerator,
            denominator,
        };
        let mut claims = [root(n_a, d_a), root(n_t, d_t)];
        for (level, sumcheck) in proof.levels.iter().enumerate() {
            let present: Vec<usize> = (0..2).filter(|&i| tallest - heights[i] <= level).collect();
            let gamma = t.challenge(&format!("tree/{level}/gamma"));
            let lambda = t.challenge(&format!("tree/{level}/lambda"));
            let vars: Vec<usize> = present.iter().map(|&i| claims[i].point.len()).collect();
            let n_max = vars.iter().copied().max().unwrap_or(0);
            let (mut claim, mut weight) = (F::ZERO, F::ONE);
            for (&i, &n) in present.iter().zip(&vars) {
                let c = claims[i].numerator + lambda * claims[i].denominator;
                claim += weight * int(1 << (n_max - n)) * c;
                weight *= gamma;
            }
            if sumcheck.rounds.len() != n_max {
                return reject;
            }
            let mut point = Vec::with_capacity(n_max);
            for (j, round) in sumcheck.rounds.iter().enumerate() {
                if round.len() != 3 {
                    return reject;
                }
                t.absorb_message(&format!("tree/{level}/round/{j}"), round);
                let r = t.challenge(&format!("tree/{level}/r/{j}"));
                claim = round_value(claim, round, r);
                point.push(r);
            }
            if sumcheck.final_evals.len() != 4 * present.len() {
                return reject;
            }
            t.absorb_message(&format!("tree/{level}/evals"), &sumcheck.final_evals);
            let evals: Vec<&[F]> = sumcheck.final_evals.chunks_exact(4).collect();
            let (mut batched, mut weight) = (F::ZERO, F::ONE);
            for ((&i, &n), e) in present.iter().zip(&vars).zip(&evals) {
                let g = e[0] * e[3] + e[1] * e[2] + lambda * e[2] * e[3];
                batched += weight * eq(&claims[i].point, &point[n_max - n..]) * g;
                weight *= gamma;
            }
            if batched != claim {
                return reject;
            }
            let rho = t.challenge(&format!("tree/{level}/line"));
            for ((&i, &n), e) in present.iter().zip(&vars).zip(&evals) {
                let mut next = vec![rho];
                next.extend_from_slice(&point[n_max - n..]);
                claims[i] = Claims {
                    point: next,
                    numerator: e[0] + rho * (e[1] - e[0]),
                    denominator: e[2] + rho * (e[3] - e[2]),
                };
            }
        }

        // A table without values sends none, and `v = 0`.
        if !proof.values.is_empty() {
            return reject;
        }
        t.absorb_message("logup/values", &proof.values);
        let [lookup, table] = &claims;
        let row_0 = self.rows[0];
        let rows = padded(&self.rows, heights[1], row_0);
        let keys = padded(&self.keys, heights[0], row_0);
        let accepted = lookup.numerator == F::ONE
            && table.denominator == alpha - mle(&rows, &table.point)
            && mle(&keys, &lookup.point) == alpha - lookup.denominator
            && mle(&self.multiplicities, &table.point) == -table.numerator;
        Verdict {
            accepted,
            challenge: alpha,
        }
    }
}

/// The adversarial instance of [`Lookups::adversarial`] with the chosen roots `P − 1 − c − s`,
/// if its forced roots are `N − 1` distinct elements of the field, none a pole or a chosen root.
fn candidate(lookups: usize, rows: usize, c: u64) -> Option<Lookups> {
    let table: Vec<F> = (0..rows).map(int).collect();
    let keys: Vec<F> = (rows..rows + lookups).map(int).collect();
    let chosen: Vec<F> = (0..rows as u64)
        .map(|s| F::from_u64_reduced(FIELD - 1 - c - s))
        .collect();
    let poles: Vec<F> = table.iter().chain(&keys).copied().collect();
    if chosen.iter().any(|z| poles.contains(z)) {
        return None;
    }
    // Q'(x) = Π_{y∈S, y≠x} (x − y) at a pole x, and Z(x) = Π_s (x − z_s).
    let q_prime = |x: F| {
        poles
            .iter()
            .filter(|&&y| y != x)
            .fold(F::ONE, |p, &y| p * (x - y))
    };
    let z_at = |x: F| chosen.iter().fold(F::ONE, |p, &s| p * (x - s));
    // G through (a_i, Q'(a_i)/Z(a_i)), in Lagrange's form over the keys.
    let weights: Vec<F> = keys
        .iter()
        .map(|&a| {
            let nodes = keys
                .iter()
                .filter(|&&b| b != a)
                .fold(F::ONE, |p, &b| p * (a - b));
            q_prime(a)
                * (z_at(a) * nodes)
                    .inverse()
                    .expect("no chosen root is a pole")
        })
        .collect();
    let g = |x: F| {
        keys.iter().zip(&weights).fold(F::ZERO, |sum, (&a, &w)| {
            let basis = keys
                .iter()
                .filter(|&&b| b != a)
                .fold(F::ONE, |p, &b| p * (x - b));
            sum + w * basis
        })
    };
    let forced: Vec<F> = (0..FIELD)
        .map(F::from_u64_reduced)
        .filter(|&x| g(x) == F::ZERO)
        .collect();
    if forced.len() != lookups - 1
        || forced
            .iter()
            .any(|x| poles.contains(x) || chosen.contains(x))
    {
        return None;
    }
    let mut m: Vec<F> = table
        .iter()
        .map(|&x| -(z_at(x) * g(x) * q_prime(x).inverse().expect("distinct poles")))
        .collect();
    m[0] += int((1 << height(lookups)) - lookups);
    m.resize(1 << height(rows), F::ZERO);
    Some(Lookups::new(table, keys, m))
}

/// The values of `α`, in order, at which no leaf's denominator is zero and the leaves' fractions
/// sum to zero, which is what the roots' check accepts: by brute force over the field, each `α`
/// with one inversion for all the leaves (Montgomery's trick).
fn accepting(leaves: &[(F, F)]) -> Vec<F> {
    let mut denominators = vec![F::ZERO; leaves.len()];
    let mut prefix = vec![F::ZERO; leaves.len()];
    (0..FIELD)
        .map(F::from_u64_reduced)
        .filter(|&alpha| {
            let mut product = F::ONE;
            for ((&(_, key), d), before) in leaves.iter().zip(&mut denominators).zip(&mut prefix) {
                *d = alpha - key;
                if *d == F::ZERO {
                    return false;
                }
                *before = product;
                product *= *d;
            }
            let mut inverse = product.inverse().expect("no zero denominator");
            let mut sum = F::ZERO;
            for ((&(numerator, _), &d), &before) in
                leaves.iter().zip(&denominators).zip(&prefix).rev()
            {
                sum += numerator * inverse * before;
                inverse *= d;
            }
            sum == F::ZERO
        })
        .collect()
}
