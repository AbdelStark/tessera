// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A reference prover for LogUp-GKR's leaf stage, for tests only: each LogUp instance's lookup
//! tree and table tree, beside product forests, proven level by level from their roots, honestly
//! or as a prover that changes its leaves, its roots or its values before it sends them.
//!
//! It is written from RFC-0008 ('Relation', 'Fraction tree', 'Batching'), RFC-0010 ('Tree
//! levels') and RFC-0014 ('Transcript schedule', rows 4 and 5), on the field, the transcript and
//! the batched prover of [`sumcheck`](crate::sumcheck), and shares no code with `tv-core`, whose
//! checks its proofs test. It builds every tree whole, level by level from the leaves, and proves
//! a level with the tables of the level below split by the pair bit, where `tv-core` checks a
//! level from its claims alone. It gives the verifier's tests what conformance vectors cannot
//! hold: proofs of seeded batches of any shape, and of cheating provers.

use crate::sumcheck::{
    Fe, Form, Product, Proof, Transcript, eq_table, mle, prove_after_gamma, tce,
};

/// `⌈log2 n⌉` for `n ≥ 1`: the height of a tree over `n` leaves padded to a power of two.
pub fn ceil_log2(n: usize) -> usize {
    assert!(n >= 1, "a tree has at least one leaf");
    n.next_power_of_two().trailing_zeros() as usize
}

/// `key + β·value`: a pair compressed to one element (RFC-0008 'Relation').
pub fn compress((key, value): (Fe, Fe), beta: Fe) -> Fe {
    key + beta * value
}

/// A LogUp instance as its prover holds it (RFC-0008 'Relation').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogUp {
    /// The table's rows `(key, value)`, `T ≥ 1` of them; a table without values has every value
    /// 0.
    pub rows: Vec<(Fe, Fe)>,
    /// Whether the table has values, so that the values message holds `valũe(ρ_a)`.
    pub has_values: bool,
    /// The lookups `(key, value)`, `N ≥ 1` of them; an honest prover's are rows.
    pub lookups: Vec<(Fe, Fe)>,
    /// `m_j` per table-tree leaf, `2^{h_t}` of them: row 0's counts the lookup tree's padding,
    /// and an honest prover's padding rows hold 0.
    pub multiplicities: Vec<Fe>,
}

impl LogUp {
    /// An honest instance: the lookups are the rows `lookup_rows`, in order, and each row's
    /// multiplicity counts them, row 0's the lookup tree's padding too.
    pub fn honest(rows: Vec<(Fe, Fe)>, has_values: bool, lookup_rows: &[usize]) -> Self {
        assert!(!rows.is_empty() && !lookup_rows.is_empty());
        let lookups = lookup_rows.iter().map(|&j| rows[j]).collect();
        let mut counts = vec![0u64; 1 << ceil_log2(rows.len())];
        for &j in lookup_rows {
            counts[j] += 1;
        }
        counts[0] += ((1usize << ceil_log2(lookup_rows.len())) - lookup_rows.len()) as u64;
        Self {
            rows,
            has_values,
            lookups,
            multiplicities: counts.into_iter().map(Fe::base).collect(),
        }
    }

    /// `h_a = ⌈log2 N⌉`.
    pub fn lookup_height(&self) -> usize {
        ceil_log2(self.lookups.len())
    }

    /// `h_t = ⌈log2 T⌉`.
    pub fn table_height(&self) -> usize {
        ceil_log2(self.rows.len())
    }

    /// The lookup tree's leaves: `(1, α − a_i)`, then `(1, α − t_0)` in the padding.
    pub fn lookup_leaves(&self, alpha: Fe, beta: Fe) -> Vec<(Fe, Fe)> {
        let padding = (Fe::ONE, alpha - compress(self.rows[0], beta));
        let mut leaves: Vec<(Fe, Fe)> = self
            .lookups
            .iter()
            .map(|&lookup| (Fe::ONE, alpha - compress(lookup, beta)))
            .collect();
        leaves.resize(1 << self.lookup_height(), padding);
        leaves
    }

    /// The table's `2^{h_t}` rows, its padding rows copies of row 0 (RFC-0008 'Fraction tree'), so
    /// that a multiplicity there only counts more lookups of row 0.
    pub fn padded_rows(&self) -> Vec<(Fe, Fe)> {
        let mut rows = self.rows.clone();
        rows.resize(1 << self.table_height(), self.rows[0]);
        rows
    }

    /// The table tree's leaves: `(−m_j, α − t_j)` over the padded rows, a padding row's
    /// `(−m_j, α − t_0)`, where an honest `m_j` is 0.
    pub fn table_leaves(&self, alpha: Fe, beta: Fe) -> Vec<(Fe, Fe)> {
        assert_eq!(self.multiplicities.len(), 1 << self.table_height());
        self.padded_rows()
            .into_iter()
            .zip(&self.multiplicities)
            .map(|(row, &m)| (-m, alpha - compress(row, beta)))
            .collect()
    }

    /// The lookup side's key polynomial: `2^{h_a}` entries, the padding slots row 0's key.
    pub fn lookup_keys(&self) -> Vec<Fe> {
        self.padded_lookups().map(|(key, _)| key).collect()
    }

    /// The lookup side's value polynomial: `2^{h_a}` entries, the padding slots row 0's value.
    pub fn lookup_values(&self) -> Vec<Fe> {
        self.padded_lookups().map(|(_, value)| value).collect()
    }

    fn padded_lookups(&self) -> impl Iterator<Item = (Fe, Fe)> + '_ {
        let padding = (1usize << self.lookup_height()) - self.lookups.len();
        self.lookups
            .iter()
            .copied()
            .chain(std::iter::repeat_n(self.rows[0], padding))
    }

    /// The table polynomial `t = key + β·value` over the padded rows: `2^{h_t}` entries, the
    /// padding rows copies of row 0.
    pub fn table_poly(&self, beta: Fe) -> Vec<Fe> {
        self.padded_rows()
            .into_iter()
            .map(|row| compress(row, beta))
            .collect()
    }
}

/// Every level of a fraction tree, the root first: `(n, d) = (n_L·d_R + n_R·d_L, d_L·d_R)`, the
/// left child the even entry (RFC-0008 'Fraction tree').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FractionTree {
    /// `levels[0]` is the root, one entry; `levels[h]` the leaves.
    pub levels: Vec<Vec<(Fe, Fe)>>,
}

impl FractionTree {
    /// The tree over `leaves`, a power of two of them.
    pub fn new(leaves: Vec<(Fe, Fe)>) -> Self {
        assert!(leaves.len().is_power_of_two());
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
        Self { levels }
    }

    /// Its height `h`.
    pub fn height(&self) -> usize {
        self.levels.len() - 1
    }

    /// The root `(n, d)`.
    pub fn root(&self) -> (Fe, Fe) {
        self.levels[0][0]
    }
}

/// A product forest: `2^s` product trees of one height `h ≥ 1`, stacked, so that its level `ℓ`
/// has `2^{s+ℓ}` entries, entry `y·2^ℓ + x` being entry `x` of tree `y`'s level `ℓ` (RFC-0010
/// 'Tree levels').
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Forest {
    /// `s`.
    pub rows_log: usize,
    /// `levels[0]` holds the trees' roots, `2^s` of them; `levels[h]` the stacked leaves.
    pub levels: Vec<Vec<Fe>>,
}

impl Forest {
    /// The forest of `2^rows_log` trees over its stacked leaves, `2^{s+h}` of them with `h ≥ 1`.
    pub fn new(rows_log: usize, leaves: Vec<Fe>) -> Self {
        assert!(leaves.len().is_power_of_two() && leaves.len() > 1 << rows_log);
        let mut levels = vec![leaves];
        while levels[0].len() > 1 << rows_log {
            let above = levels[0]
                .chunks_exact(2)
                .map(|pair| pair[0] * pair[1])
                .collect();
            levels.insert(0, above);
        }
        Self { rows_log, levels }
    }

    /// Its height `h`.
    pub fn height(&self) -> usize {
        self.levels.len() - 1
    }

    /// The public root values, one per tree: level 0.
    pub fn roots(&self) -> &[Fe] {
        &self.levels[0]
    }

    /// The stacked leaves.
    pub fn leaves(&self) -> &[Fe] {
        &self.levels[self.height()]
    }
}

/// A tree of the leaf stage, as the schedule lists it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StageTree {
    /// Instance `i`'s lookup tree.
    Lookup(usize),
    /// Instance `i`'s table tree.
    Table(usize),
    /// Forest `f`.
    Forest(usize),
}

/// A tree's claims on one of its levels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageClaim {
    /// The tree, by its position in the schedule order.
    pub tree: usize,
    /// The point, low variables first.
    pub point: Vec<Fe>,
    /// `Ṽ(point)` for a forest; `Ñ(point)` and `D̃(point)` for a fraction tree.
    pub values: Vec<Fe>,
}

/// One level of the leaf stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageLevel {
    /// The challenges squeezed under `tree/{ℓ}/root/{c}` before `γ`.
    pub root_challenges: Vec<Fe>,
    /// The claims entering the level, in schedule order.
    pub entering: Vec<StageClaim>,
    /// `λ`, squeezed after `γ` when a fraction tree is present.
    pub lambda: Option<Fe>,
    /// The level's sumcheck; its evaluations are `u, w` for a forest and `N_0, N_1, D_0, D_1`
    /// for a fraction tree, instance by instance.
    pub proof: Proof,
    /// The line challenge `ρ`.
    pub rho: Fe,
    /// The claims on the next level, in the same order.
    pub next: Vec<StageClaim>,
}

/// A proof of the leaf stage and what its prover knows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafStage {
    /// `α`.
    pub alpha: Fe,
    /// `β`.
    pub beta: Fe,
    /// The roots message: `n_a, d_a, n_t, d_t` per instance.
    pub roots: Vec<Fe>,
    /// The levels, from 0.
    pub levels: Vec<StageLevel>,
    /// The values message: `valũe(ρ_a)` per instance whose table has values.
    pub values: Vec<Fe>,
    /// Each tree's claims after its last level, in schedule order; a fraction tree of height 0
    /// holds its root as sent.
    pub leaves: Vec<StageClaim>,
}

/// What a cheating prover may change before it uses or sends it.
pub enum Tamper<'a> {
    /// Instance `i`'s lookup-tree leaves, before its tree is built.
    LookupLeaves(usize, &'a mut Vec<(Fe, Fe)>),
    /// Instance `i`'s table-tree leaves, before its tree is built.
    TableLeaves(usize, &'a mut Vec<(Fe, Fe)>),
    /// The roots message, before it is absorbed; the levels still prove the trees' true sums.
    Roots(&'a mut Vec<Fe>),
    /// The values message, before it is absorbed.
    Values(&'a mut Vec<Fe>),
}

/// The leaf stage of `logups` and `forests`, its trees in the schedule order `trees`, which
/// lists each instance's lookup tree and table tree and each forest once.
///
/// RFC-0014 rows 4 and 5: `α` and `β`, then the roots absorbed under `logup/roots`; then level by
/// level, with `H` the tallest tree's height, the forests of height `H − ℓ` join under the root
/// challenges `tree/{ℓ}/root/{c}`, the last `s` of them each, and the fraction trees of height
/// `H − ℓ` with their roots as sent; `γ`, then `λ` when a fraction tree is present; the batched
/// sumcheck of degree 3; `ρ`; and after the last level the values absorbed under
/// `logup/values`. A fraction tree of height 0 joins no level. `tamper` sees each value it may
/// change; an honest prover changes nothing.
pub fn prove_leaf_stage(
    transcript: &mut Transcript,
    logups: &[LogUp],
    forests: &[Forest],
    trees: &[StageTree],
    mut tamper: impl FnMut(Tamper<'_>),
) -> LeafStage {
    let alpha = transcript.challenge_fp2("logup/alpha");
    let beta = transcript.challenge_fp2("logup/beta");
    let mut lookup_trees = Vec::new();
    let mut table_trees = Vec::new();
    for (i, logup) in logups.iter().enumerate() {
        let mut leaves = logup.lookup_leaves(alpha, beta);
        tamper(Tamper::LookupLeaves(i, &mut leaves));
        lookup_trees.push(FractionTree::new(leaves));
        let mut leaves = logup.table_leaves(alpha, beta);
        tamper(Tamper::TableLeaves(i, &mut leaves));
        table_trees.push(FractionTree::new(leaves));
    }
    let mut roots: Vec<Fe> = lookup_trees
        .iter()
        .zip(&table_trees)
        .flat_map(|(lookup, table)| {
            let ((n_a, d_a), (n_t, d_t)) = (lookup.root(), table.root());
            [n_a, d_a, n_t, d_t]
        })
        .collect();
    tamper(Tamper::Roots(&mut roots));
    transcript.absorb("logup/roots", &tce(&roots));

    let fraction = |tree: StageTree| -> Option<&FractionTree> {
        match tree {
            StageTree::Lookup(i) => Some(&lookup_trees[i]),
            StageTree::Table(i) => Some(&table_trees[i]),
            StageTree::Forest(_) => None,
        }
    };
    let sent_root = |tree: StageTree| -> Option<Vec<Fe>> {
        match tree {
            StageTree::Lookup(i) => Some(roots[4 * i..4 * i + 2].to_vec()),
            StageTree::Table(i) => Some(roots[4 * i + 2..4 * i + 4].to_vec()),
            StageTree::Forest(_) => None,
        }
    };
    let height = |tree: StageTree| match tree {
        StageTree::Forest(f) => forests[f].height(),
        other => fraction(other).map_or(0, FractionTree::height),
    };
    let tallest = trees.iter().map(|&tree| height(tree)).max().unwrap_or(0);
    let mut current: Vec<Option<StageClaim>> = vec![None; trees.len()];
    let mut levels = Vec::new();
    for level in 0..tallest {
        let joins = |tree: StageTree| height(tree) > 0 && tallest - height(tree) == level;
        let s_max = trees
            .iter()
            .filter_map(|&tree| match tree {
                StageTree::Forest(f) if joins(tree) => Some(forests[f].rows_log),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let root_challenges: Vec<Fe> = (0..s_max)
            .map(|c| transcript.challenge_fp2(&format!("tree/{level}/root/{c}")))
            .collect();
        for (t, &tree) in trees.iter().enumerate() {
            if !joins(tree) {
                continue;
            }
            current[t] = Some(match tree {
                StageTree::Forest(f) => {
                    let point = root_challenges[s_max - forests[f].rows_log..].to_vec();
                    let value = mle(forests[f].roots(), &point);
                    StageClaim {
                        tree: t,
                        point,
                        values: vec![value],
                    }
                }
                other => StageClaim {
                    tree: t,
                    point: Vec::new(),
                    values: sent_root(other).expect("a fraction tree"),
                },
            });
        }
        let entering: Vec<StageClaim> = current.iter().flatten().cloned().collect();
        let prefix = format!("tree/{level}");
        let gamma = transcript.challenge_fp2(&format!("{prefix}/gamma"));
        let lambda = entering
            .iter()
            .any(|claim| fraction(trees[claim.tree]).is_some())
            .then(|| transcript.challenge_fp2(&format!("{prefix}/lambda")));
        let evens = |table: &[Fe]| -> Vec<Fe> { table.iter().step_by(2).copied().collect() };
        let odds = |table: &[Fe]| -> Vec<Fe> { table.iter().skip(1).step_by(2).copied().collect() };
        let instances: Vec<Product> = entering
            .iter()
            .map(|claim| {
                let own = claim.point.len();
                let public = vec![eq_table(&claim.point)];
                match trees[claim.tree] {
                    StageTree::Forest(f) => {
                        let below = &forests[f].levels[own - forests[f].rows_log + 1];
                        assert_eq!(below.len(), 2 << own);
                        Product {
                            tables: vec![evens(below), odds(below)],
                            public,
                            lambda: None,
                            form: Form::Product,
                        }
                    }
                    other => {
                        let below = &fraction(other).expect("a fraction tree").levels[own + 1];
                        assert_eq!(below.len(), 2 << own);
                        let n: Vec<Fe> = below.iter().map(|&(n, _)| n).collect();
                        let d: Vec<Fe> = below.iter().map(|&(_, d)| d).collect();
                        Product {
                            tables: vec![evens(&n), odds(&n), evens(&d), odds(&d)],
                            public,
                            lambda,
                            form: Form::Product,
                        }
                    }
                }
            })
            .collect();
        let proof = prove_after_gamma(transcript, &prefix, 3, gamma, &instances, None);
        let rho = transcript.challenge_fp2(&format!("{prefix}/line"));
        let mut evals = proof.evals.iter().copied();
        let next: Vec<StageClaim> = entering
            .iter()
            .zip(&instances)
            .map(|(claim, instance)| {
                let own = &proof.challenges[proof.challenges.len() - claim.point.len()..];
                let mine: Vec<Fe> = evals.by_ref().take(instance.tables.len()).collect();
                StageClaim {
                    tree: claim.tree,
                    point: [&[rho][..], own].concat(),
                    values: mine
                        .chunks_exact(2)
                        .map(|pair| pair[0] + rho * (pair[1] - pair[0]))
                        .collect(),
                }
            })
            .collect();
        for claim in &next {
            current[claim.tree] = Some(claim.clone());
        }
        levels.push(StageLevel {
            root_challenges,
            entering,
            lambda,
            proof,
            rho,
            next,
        });
    }
    let leaves: Vec<StageClaim> = trees
        .iter()
        .enumerate()
        .map(|(t, &tree)| match &current[t] {
            Some(claim) => claim.clone(),
            None => StageClaim {
                tree: t,
                point: Vec::new(),
                values: sent_root(tree).expect("only a fraction tree of height 0 joins no level"),
            },
        })
        .collect();
    let mut values = Vec::new();
    for (i, logup) in logups.iter().enumerate() {
        if logup.has_values {
            let t = trees
                .iter()
                .position(|&tree| tree == StageTree::Lookup(i))
                .expect("every lookup tree is scheduled");
            values.push(mle(&logup.lookup_values(), &leaves[t].point));
        }
    }
    tamper(Tamper::Values(&mut values));
    transcript.absorb("logup/values", &tce(&values));
    LeafStage {
        alpha,
        beta,
        roots,
        levels,
        values,
        leaves,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SplitMix64;

    fn random(rng: &mut SplitMix64) -> Fe {
        Fe::new(rng.next_u64(), rng.next_u64())
    }

    /// A root's `n/d` is the sum of its leaves' fractions, and an honest instance's two roots
    /// cancel, `n_a·d_t + n_t·d_a = 0`, for any `α` and `β`.
    #[test]
    fn honest_roots_cancel() {
        let mut rng = SplitMix64::new(7);
        for (rows, lookups) in [(1, 1), (3, 5), (8, 8), (5, 9)] {
            let table: Vec<(Fe, Fe)> = (0..rows)
                .map(|_| (Fe::base(rng.next_u64()), Fe::base(rng.next_u64())))
                .collect();
            let picks: Vec<usize> = (0..lookups)
                .map(|_| (rng.next_u64() % rows as u64) as usize)
                .collect();
            let logup = LogUp::honest(table, true, &picks);
            let (alpha, beta) = (random(&mut rng), random(&mut rng));
            let lookup = FractionTree::new(logup.lookup_leaves(alpha, beta));
            let table = FractionTree::new(logup.table_leaves(alpha, beta));
            let sum = |leaves: &[(Fe, Fe)]| {
                leaves
                    .iter()
                    .fold(Fe::ZERO, |s, &(n, d)| s + n * d.inverse().unwrap())
            };
            let ((n_a, d_a), (n_t, d_t)) = (lookup.root(), table.root());
            assert_eq!(
                n_a * d_a.inverse().unwrap(),
                sum(&lookup.levels[lookup.height()])
            );
            assert_eq!(n_a * d_t + n_t * d_a, Fe::ZERO);
        }
    }

    /// Every claim of an honest leaf stage is its tree level's extension at its point.
    #[test]
    fn an_honest_leaf_stage_makes_true_claims() {
        let mut rng = SplitMix64::new(8);
        let logups = vec![
            LogUp::honest(
                (0..5).map(|j| (Fe::base(j), Fe::base(3 * j))).collect(),
                true,
                &[1, 4, 4],
            ),
            LogUp::honest(
                (0..2).map(|j| (Fe::base(j), Fe::ZERO)).collect(),
                false,
                &[1],
            ),
        ];
        let forests = vec![Forest::new(
            1,
            (0..8).map(|_| Fe::base(rng.next_u64())).collect(),
        )];
        let trees = [
            StageTree::Forest(0),
            StageTree::Lookup(0),
            StageTree::Table(0),
            StageTree::Lookup(1),
            StageTree::Table(1),
        ];
        let mut transcript = Transcript::new(b"context");
        let stage = prove_leaf_stage(&mut transcript, &logups, &forests, &trees, |_| {});
        assert_eq!(stage.levels.len(), 3);
        // Level 0 holds the table tree of instance 0 alone; level 1 adds the forest and the
        // lookup tree of instance 0; level 2 the table tree of instance 1.
        let present: Vec<Vec<usize>> = stage
            .levels
            .iter()
            .map(|level| level.entering.iter().map(|claim| claim.tree).collect())
            .collect();
        assert_eq!(present, [vec![2], vec![0, 1, 2], vec![0, 1, 2, 4]]);
        assert!(stage.levels.iter().all(|level| level.lambda.is_some()));
        for (t, leaf) in stage.leaves.iter().enumerate() {
            match trees[t] {
                StageTree::Forest(f) => {
                    assert_eq!(mle(forests[f].leaves(), &leaf.point), leaf.values[0])
                }
                StageTree::Lookup(i) | StageTree::Table(i) => {
                    let leaves = if matches!(trees[t], StageTree::Lookup(_)) {
                        logups[i].lookup_leaves(stage.alpha, stage.beta)
                    } else {
                        logups[i].table_leaves(stage.alpha, stage.beta)
                    };
                    let n: Vec<Fe> = leaves.iter().map(|&(n, _)| n).collect();
                    let d: Vec<Fe> = leaves.iter().map(|&(_, d)| d).collect();
                    assert_eq!(leaf.values, [mle(&n, &leaf.point), mle(&d, &leaf.point)]);
                }
            }
        }
        // The lookup tree of instance 1 has one leaf: it joins no level and keeps its root.
        assert!(stage.leaves[3].point.is_empty());
        assert_eq!(stage.values.len(), 1);
    }
}
