// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sums of weighted products over one shared table list: the terms of an instance, and the plan
//! the kernel follows to evaluate them (RFC-0010 'Tree levels', `FractionTreeLevel`).

use core::fmt;

use tessera_field::Field;

/// One term of a sum-of-products [`Instance`](crate::Instance): `weight · Π_{k ∈ factors} T̃_k`,
/// the factors named by their positions in the instance's table list.
///
/// A factor may repeat, as `T̃_k^2`; the term's degree in each variable is its number of factors.
/// [`prove_batch`](crate::prove_batch) checks that a term has between 1 and the batch's degree
/// factors, each naming a table of its instance.
///
/// The weight is protocol-class data, such as a challenge: the debug form shows the factors only.
#[derive(Clone, PartialEq, Eq)]
pub struct Term<F> {
    weight: F,
    factors: Vec<usize>,
}

impl<F> Term<F> {
    /// `weight` times the product of the tables at the positions `factors`.
    pub fn new(weight: F, factors: Vec<usize>) -> Self {
        Self { weight, factors }
    }

    /// The weight.
    pub fn weight(&self) -> &F {
        &self.weight
    }

    /// The positions of the factors in the instance's table list, in order.
    pub fn factors(&self) -> &[usize] {
        &self.factors
    }
}

/// Shows the factors, never the weight (05 §3).
impl<F> fmt::Debug for Term<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Term")
            .field("factors", &self.factors)
            .finish_non_exhaustive()
    }
}

/// Terms that share a weight: `weight · C · Σ_k R_k`, with `C` the factors common to every term
/// of the instance and `R_k` what each of these terms multiplies beyond them.
pub(crate) struct Group<F> {
    pub(crate) weight: F,
    /// Each term's remaining factors, as line slots; an empty list is the product 1.
    pub(crate) rests: Vec<Vec<usize>>,
}

/// How the kernel evaluates an instance's terms, from their structure alone (public data,
/// INV-ARCH-2): `Σ_t w_t·Π_{f ∈ t} f = C · Σ_g w_g·Σ_{t ∈ g} R_t`.
///
/// - **Lines.** Only the tables some term multiplies get line values, each once per pair of
///   entries, at slots in table order. A table that no term reads, such as a lookup tree's leaf
///   numerators of 1, is bound for its evaluation only.
/// - **Common factors.** The factors every term has, counted with their least multiplicity, are
///   multiplied once per pair, not once per term: `eq(r, ·)` in a tree level.
/// - **Groups.** Terms of equal weight share one sum, and each group's weight multiplies its sum
///   once per call, never per pair: a tree level's `N_0·D_1 + N_1·D_0` and its `λ·D_0·D_1` are two
///   sums, `λ` applied at the end.
///
/// A fraction tree's level, `eq·(N_0·D_1 + N_1·D_0 + λ·D_0·D_1)`, so costs 5 products per pair and
/// node, and with numerators of 1, `eq·(D_1 + D_0 + λ·D_0·D_1)`, 3.
pub(crate) struct Plan<F> {
    /// The table at each line slot.
    pub(crate) lines: Vec<usize>,
    /// The common factors, as line slots, with their multiplicity.
    pub(crate) common: Vec<usize>,
    pub(crate) groups: Vec<Group<F>>,
}

impl<F: Field> Plan<F> {
    /// The plan of `terms` over `tables` tables. Every factor names one of them and there is at
    /// least one term, which [`prove_batch`](crate::prove_batch) checks first.
    pub(crate) fn new(terms: &[Term<F>], tables: usize) -> Self {
        // counts[t][k]: how many times term t multiplies table k.
        let counts: Vec<Vec<usize>> = terms
            .iter()
            .map(|term| {
                let mut count = vec![0; tables];
                for &k in term.factors() {
                    if let Some(c) = count.get_mut(k) {
                        *c += 1;
                    }
                }
                count
            })
            .collect();
        let common: Vec<usize> = (0..tables)
            .map(|k| counts.iter().map(|count| count[k]).min().unwrap_or(0))
            .collect();
        let rests: Vec<Vec<usize>> = counts
            .iter()
            .map(|count| {
                (0..tables)
                    .flat_map(|k| core::iter::repeat_n(k, count[k] - common[k]))
                    .collect()
            })
            .collect();
        let read: Vec<bool> = (0..tables)
            .map(|k| common[k] > 0 || rests.iter().any(|rest| rest.contains(&k)))
            .collect();
        let lines: Vec<usize> = (0..tables).filter(|&k| read[k]).collect();
        // slot[k]: table k's line slot, for the tables some term reads.
        let mut slot = vec![0; tables];
        for (s, &k) in lines.iter().enumerate() {
            slot[k] = s;
        }
        let common_slots: Vec<usize> = (0..tables)
            .flat_map(|k| core::iter::repeat_n(slot[k], common[k]))
            .collect();
        let mut groups: Vec<Group<F>> = Vec::new();
        for (term, rest) in terms.iter().zip(rests) {
            let rest = rest.into_iter().map(|k| slot[k]).collect();
            match groups.iter_mut().find(|group| group.weight == term.weight) {
                Some(group) => group.rests.push(rest),
                None => groups.push(Group {
                    weight: term.weight,
                    rests: vec![rest],
                }),
            }
        }
        Self {
            lines,
            common: common_slots,
            groups,
        }
    }
}

#[cfg(test)]
mod tests {
    use tessera_field::{Field, Goldilocks, PrimeField};

    use super::{Plan, Term};

    type Fp = Goldilocks;

    /// The term loop against its definition, `Σ_t Σ_k w_k·Π_{f ∈ k} (a_f + z·Δ_f)`, through every
    /// shape of plan: common factors or none, groups of one or several terms, rests of 0 to 3
    /// factors, tables of both fields read by several terms or by none. At the nodes of a round
    /// message and at every node, over ranges of pairs that span several blocks or none, and the
    /// sums over the two halves of a range add up to the sum over the range.
    #[test]
    fn term_sums_match_their_definition_on_any_range() {
        use tessera_field::{GoldilocksExt2, QuadraticExtension};

        use crate::kernel::{Entries, NODES, TermScratch, term_sums};

        type Fp2 = GoldilocksExt2;
        let mut seed = 0x60_0101u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            seed >> 1
        };
        let entries = 1 << 10;
        let mut fp2 = || {
            let c0 = Fp::from_u64_reduced(next());
            Fp2::new(c0, Fp::from_u64_reduced(next()))
        };
        // Tables 0, 2 over Fp, 1, 3, 4 over Fp2.
        let base: Vec<Vec<Fp>> = (0..2)
            .map(|_| (0..entries).map(|_| fp2().coeffs()[0]).collect())
            .collect();
        let ext: Vec<Vec<Fp2>> = (0..3)
            .map(|_| (0..entries).map(|_| fp2()).collect())
            .collect();
        let tables = [
            Entries::<Fp2>::Base(&base[0]),
            Entries::Ext(&ext[0]),
            Entries::Base(&base[1]),
            Entries::Ext(&ext[1]),
            Entries::Ext(&ext[2]),
        ];
        let lifted: Vec<Vec<Fp2>> = vec![
            base[0].iter().map(|&b| Fp2::from(b)).collect(),
            ext[0].clone(),
            base[1].iter().map(|&b| Fp2::from(b)).collect(),
            ext[1].clone(),
            ext[2].clone(),
        ];
        let (w0, w1) = (fp2(), fp2());
        let plans: Vec<Vec<Term<Fp2>>> = vec![
            // A fraction level: eq common, weights 1, 1 and λ.
            vec![
                Term::new(Fp2::ONE, vec![0, 1, 4]),
                Term::new(Fp2::ONE, vec![0, 2, 3]),
                Term::new(w0, vec![0, 3, 4]),
            ],
            // Numerators of 1: tables 1 and 2 read by no term.
            vec![
                Term::new(Fp2::ONE, vec![0, 4]),
                Term::new(Fp2::ONE, vec![0, 3]),
                Term::new(w0, vec![0, 3, 4]),
            ],
            // Two common factors, a term equal to them, rests of 1 to 3 factors in one group.
            vec![
                Term::new(w1, vec![1, 3]),
                Term::new(w1, vec![3, 1, 0]),
                Term::new(w1, vec![1, 3, 0, 2, 4]),
                Term::new(w0, vec![1, 3, 2, 2]),
            ],
            // No common factor, a single term of one factor, repeated factors.
            vec![Term::new(w0, vec![2])],
            vec![Term::new(w0, vec![4, 4, 4]), Term::new(w1, vec![0, 1])],
            vec![
                Term::new(Fp2::ONE, vec![3]),
                Term::new(Fp2::ONE, vec![1, 2, 4]),
                Term::new(Fp2::ONE, vec![0]),
            ],
        ];
        let mut scratch = TermScratch::<Fp2>::new();
        for terms in &plans {
            let plan = Plan::new(terms, tables.len());
            scratch.fit(plan.lines.len(), plan.groups.len());
            let d = terms.iter().map(|t| t.factors().len()).max().unwrap_or(1);
            let naive = |pairs: core::ops::Range<usize>, nodes: &[u64]| -> Vec<Fp2> {
                nodes
                    .iter()
                    .map(|&z| {
                        let z = Fp2::from(Fp::from_u64_reduced(z));
                        pairs.clone().fold(Fp2::ZERO, |sum, t| {
                            sum + terms.iter().fold(Fp2::ZERO, |s, term| {
                                s + *term.weight()
                                    * term.factors().iter().fold(Fp2::ONE, |p, &k| {
                                        let (a, b) = (lifted[k][2 * t], lifted[k][2 * t + 1]);
                                        p * (a + z * (b - a))
                                    })
                            })
                        })
                    })
                    .collect()
            };
            let message_nodes: Vec<u64> = core::iter::once(0).chain(2..=d as u64).collect();
            let all_nodes: Vec<u64> = (0..=d as u64).collect();
            for pairs in [0..entries / 2, 3..200, 64..65, 7..7, 130..300] {
                let mid = (pairs.start + pairs.end) / 2;
                for (nodes, all) in [(&message_nodes, false), (&all_nodes, true)] {
                    let mut sums = |range: core::ops::Range<usize>| -> [Fp2; NODES] {
                        match (d, all) {
                            (1, false) => {
                                term_sums::<_, _, 1, true>(&tables[..], &plan, range, &mut scratch)
                            }
                            (1, true) => {
                                term_sums::<_, _, 2, false>(&tables[..], &plan, range, &mut scratch)
                            }
                            (2, false) => {
                                term_sums::<_, _, 2, true>(&tables[..], &plan, range, &mut scratch)
                            }
                            (2, true) => {
                                term_sums::<_, _, 3, false>(&tables[..], &plan, range, &mut scratch)
                            }
                            (3, false) => {
                                term_sums::<_, _, 3, true>(&tables[..], &plan, range, &mut scratch)
                            }
                            (3, true) => {
                                term_sums::<_, _, 4, false>(&tables[..], &plan, range, &mut scratch)
                            }
                            (5, false) => {
                                term_sums::<_, _, 5, true>(&tables[..], &plan, range, &mut scratch)
                            }
                            (5, true) => {
                                term_sums::<_, _, 6, false>(&tables[..], &plan, range, &mut scratch)
                            }
                            _ => unreachable!("the plans' degrees"),
                        }
                    };
                    let whole = sums(pairs.clone());
                    let (low, high) = (sums(pairs.start..mid), sums(mid..pairs.end));
                    let expected = naive(pairs.clone(), nodes);
                    let at = format!("{terms:?}, {pairs:?}, all nodes {all}");
                    assert!(whole[..nodes.len()] == expected[..], "{at}");
                    assert!(whole[nodes.len()..].iter().all(|&x| x == Fp2::ZERO), "{at}");
                    for ((w, l), h) in whole.iter().zip(low).zip(high) {
                        assert!(*w == l + h, "{at}: the halves add up");
                    }
                }
            }
        }
    }
}
