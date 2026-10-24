// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054's merge ('The merge', 'The prover in two passes').
//!
//! - **Completeness.** A reference verifier replays the merge's challenges on the verifier's own
//!   transcript, checks every round against its claim, and checks the final evaluations against
//!   the weights it computes from the claims alone. The final evaluations are the tensors' own
//!   evaluations at the merge's point.
//! - **Soundness.** A false claim fails the final check.
//! - **Two passes.** The table-backed rounds and the merge pass send the in-memory merge's
//!   messages, byte for byte, at every `j*` from 0 to the largest the polynomials allow, and
//!   regenerate each polynomial once.
//! - **Refusals.** Every shape rule refuses with its own rule.

mod common;

use common::{Fp, Fp2, Session, SplitMix64, tv_fp2};
use tessera_codec::Secret;
use tessera_field::{Field, PrimeField};
use tessera_poly::evaluate_base;
use tessera_sumcheck::{
    Merge, MergeClaim, MergePolynomial, MergeTables, SumcheckError, largest_table_rounds,
    prove_merge, prove_merge_two_pass,
};
use tv_hash::transcript::Ascii;

/// A polynomial's shape and table: synthetic test data, kept in the clear beside the `Secret` the
/// provers take.
struct Poly {
    shape: MergePolynomial,
    values: Vec<Fp>,
    table: Secret<Vec<Fp>>,
}

impl Poly {
    fn new(rng: &mut SplitMix64, vars: usize, group_vars: Vec<usize>) -> Self {
        let values = rng.fps(1 << vars);
        Self {
            shape: MergePolynomial { vars, group_vars },
            table: Secret::new(values.clone()),
            values,
        }
    }

    /// The positions outside `S`, ascending, then `S`.
    fn order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.shape.vars)
            .filter(|v| !self.shape.group_vars.contains(v))
            .collect();
        order.extend(&self.shape.group_vars);
        order
    }

    /// `P̃` at a point in binding order, evaluated over the natural table.
    fn at(&self, binding: &[Fp2]) -> Fp2 {
        let mut natural = vec![Fp2::ZERO; self.shape.vars];
        for (t, &v) in self.order().iter().enumerate() {
            natural[v] = binding[t];
        }
        evaluate_base::<Fp2>(&self.values, &natural).expect("a point per variable")
    }

    /// The claim `P̃(z ‖ s)` at a random `z` and the bits `s`.
    fn claim(&self, rng: &mut SplitMix64, index: usize, bits: u64) -> MergeClaim<Fp2> {
        let point = rng.fp2s(self.shape.free_vars());
        let mut binding = point.clone();
        binding.extend(
            (0..self.shape.group_vars.len()).map(|u| match (bits >> u) & 1 {
                1 => Fp2::ONE,
                _ => Fp2::ZERO,
            }),
        );
        MergeClaim {
            polynomial: index,
            point,
            group_bits: bits,
            value: self.at(&binding),
        }
    }

    /// The group block: the table with its group variables set to `bits`, over its free
    /// variables in increasing position.
    fn block(&self, bits: u64) -> Secret<Vec<Fp>> {
        let free: Vec<usize> = (0..self.shape.vars)
            .filter(|v| !self.shape.group_vars.contains(v))
            .collect();
        let fixed = self
            .shape
            .group_vars
            .iter()
            .enumerate()
            .fold(0usize, |x, (u, &v)| x | ((((bits >> u) & 1) as usize) << v));
        let table = &self.values;
        Secret::new(
            (0..1usize << free.len())
                .map(|a| {
                    let x = free
                        .iter()
                        .enumerate()
                        .fold(fixed, |x, (t, &v)| x | (((a >> t) & 1) << v));
                    table[x]
                })
                .collect(),
        )
    }
}

/// Five polynomials and four groups, every class kind: group variables at either end and in the
/// middle, a polynomial without them read by two groups, front padding, two claims of one group
/// at one point on two polynomials of one shape (one class), and two claims of one group on one
/// polynomial at two points.
fn fixture(rng: &mut SplitMix64) -> (Vec<Poly>, Merge<Fp2>) {
    let polys = vec![
        Poly::new(rng, 5, vec![1, 3]),
        Poly::new(rng, 3, vec![]),
        Poly::new(rng, 5, vec![0]),
        Poly::new(rng, 4, vec![3]),
        Poly::new(rng, 4, vec![3]),
    ];
    let mut claims: Vec<Vec<MergeClaim<Fp2>>> = (0..4u64)
        .map(|h| vec![polys[0].claim(rng, 0, h), polys[2].claim(rng, 2, h & 1)])
        .collect();
    claims[0].push(polys[1].claim(rng, 1, 0));
    claims[2].push(polys[1].claim(rng, 1, 0));
    claims[2].push(polys[1].claim(rng, 1, 0));
    // One class: the same point and bits on two polynomials of one shape.
    let shared = polys[3].claim(rng, 3, 1);
    let mut twin = shared.clone();
    twin.polynomial = 4;
    let mut binding = twin.point.clone();
    binding.push(Fp2::ONE);
    twin.value = polys[4].at(&binding);
    claims[1].push(shared);
    claims[1].push(twin);
    for h in [0u64, 2, 3] {
        claims[h as usize].push(polys[3].claim(rng, 3, h & 1));
        claims[h as usize].push(polys[4].claim(rng, 4, (h >> 1) & 1));
    }
    let gammas = rng.fp2s(4);
    let merge = Merge::new(
        polys.iter().map(|p| p.shape.clone()).collect(),
        claims,
        gammas,
        rng.fp2(),
    )
    .expect("a merge");
    (polys, merge)
}

/// `p(r)` of a degree-2 round from `p(0)`, `p(2)` and the claim `p(0) + p(1)`.
fn interpolate(claim: Fp2, sent: &[Fp2], r: Fp2) -> Fp2 {
    let (p0, p2) = (sent[0], sent[1]);
    let p1 = claim - p0;
    let half = Fp2::from(Fp::from_u64_reduced(2))
        .inverse()
        .expect("2 is invertible");
    let one = Fp2::ONE;
    let two = one.double();
    p0 * (r - one) * (r - two) * half - p1 * r * (r - two) + p2 * r * (r - one) * half
}

/// `β^h·γ_h^k` for every claim, written out here rather than taken from the merge.
fn weighted(merge: &Merge<Fp2>) -> Vec<(Fp2, MergeClaim<Fp2>)> {
    let mut out = Vec::new();
    let mut beta_h = Fp2::ONE;
    for (group, &gamma) in merge.claims().iter().zip(merge.gammas()) {
        let mut gamma_k = Fp2::ONE;
        for claim in group {
            out.push((beta_h * gamma_k, claim.clone()));
            gamma_k *= gamma;
        }
        beta_h *= merge.beta();
    }
    out
}

/// `eq(a, b)`, coordinate by coordinate.
fn eq(a: &[Fp2], b: &[Fp2]) -> Fp2 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(Fp2::ONE, |acc, (&x, &y)| {
        acc * (x * y + (Fp2::ONE - x) * (Fp2::ONE - y))
    })
}

/// The first claim, `Σ β^h·γ_h^k·2^{n_max − n_i}·v`.
fn claimed_sum(merge: &Merge<Fp2>) -> Fp2 {
    let n_max = merge.rounds();
    weighted(merge).iter().fold(Fp2::ZERO, |acc, (w, c)| {
        let pad =
            (0..n_max - merge.polynomials()[c.polynomial].vars).fold(Fp2::ONE, |x, _| x.double());
        acc + *w * pad * c.value
    })
}

/// `W_i` at a point in binding order: `Σ β^h·γ_h^k·eq(z, r_free)·eq(s, r_group)`.
fn weight_at(merge: &Merge<Fp2>, i: usize, point: &[Fp2]) -> Fp2 {
    let p = &merge.polynomials()[i];
    let m = p.free_vars();
    weighted(merge)
        .iter()
        .filter(|(_, c)| c.polynomial == i)
        .fold(Fp2::ZERO, |acc, (w, c)| {
            let bits: Vec<Fp2> = (0..p.group_vars.len())
                .map(|u| match (c.group_bits >> u) & 1 {
                    1 => Fp2::ONE,
                    _ => Fp2::ZERO,
                })
                .collect();
            acc + *w * eq(&c.point, &point[..m]) * eq(&bits, &point[m..])
        })
}

/// The reference verifier: the challenges replayed on the verifier's own transcript, every round
/// against its claim, and the final check against the weights it computes from the claims.
fn verify(
    session: &Session,
    merge: &Merge<Fp2>,
    proof: &tessera_sumcheck::ProvenBatch<Fp2>,
) -> bool {
    let mut transcript = session.tv_transcript();
    let mut claim = claimed_sum(merge);
    assert_eq!(claim, merge.claimed_sum(), "the merge's own sum");
    let rounds = &proof.proof().rounds;
    assert_eq!(rounds.len(), merge.rounds());
    for (j, sent) in rounds.iter().enumerate() {
        assert_eq!(sent.len(), 2, "degree 2: p(0) and p(2)");
        let label = format!("merge/round/{j}");
        transcript.absorb(
            Ascii::new(&label).expect("ASCII"),
            &tessera_codec::encode(sent),
        );
        let label = format!("merge/r/{j}");
        let r = transcript
            .challenge_fp2(Ascii::new(&label).expect("ASCII"))
            .expect("a challenge");
        assert_eq!(r, tv_fp2(proof.point()[j]), "round {j}'s challenge");
        claim = interpolate(claim, sent, proof.point()[j]);
    }
    let evals = &proof.proof().final_evals;
    assert_eq!(evals.len(), merge.polynomials().len());
    let mut total = Fp2::ZERO;
    for (i, &value) in evals.iter().enumerate() {
        let point = proof.instance_point(i).expect("a point");
        let weight = weight_at(merge, i, point);
        assert_eq!(
            merge.weight_at(i, point),
            Some(weight),
            "the merge's own weight"
        );
        total += weight * value;
    }
    total == claim
}

#[test]
fn the_merge_proves_its_claims_and_ends_at_the_tensors_evaluations() {
    for seed in 0..8 {
        let mut rng = SplitMix64::new(0x5400 + seed);
        let session = Session::random(&mut rng);
        let (polys, merge) = fixture(&mut rng);
        let tables: Vec<&Secret<Vec<Fp>>> = polys.iter().map(|p| &p.table).collect();
        let proof = prove_merge(&mut session.transcript(), &merge, &tables).expect("a proof");
        assert!(verify(&session, &merge, &proof), "seed {seed}");
        // The merged claims: each polynomial's evaluation at its own point, the last n_i
        // challenges in binding order.
        for (i, p) in polys.iter().enumerate() {
            let point = proof.instance_point(i).expect("a point");
            assert_eq!(point.len(), p.shape.vars);
            assert_eq!(
                proof.instance_evals(i),
                Some(&[p.at(point)][..]),
                "polynomial {i}"
            );
        }
    }
}

#[test]
fn a_false_claim_fails_the_final_check() {
    let mut rng = SplitMix64::new(0x5410);
    let session = Session::random(&mut rng);
    let (polys, merge) = fixture(&mut rng);
    let tables: Vec<&Secret<Vec<Fp>>> = polys.iter().map(|p| &p.table).collect();
    // Each claim in turn one more than its value: the verifier starts from a false sum.
    let shapes: Vec<MergePolynomial> = polys.iter().map(|p| p.shape.clone()).collect();
    let (_, honest) = fixture(&mut SplitMix64::new(0x5410));
    let groups = honest_claims(&honest);
    for h in 0..groups.len() {
        for k in 0..groups[h].len() {
            let mut claims = groups.clone();
            claims[h][k].value += Fp2::ONE;
            let lying = Merge::new(shapes.clone(), claims, gammas(&honest), beta(&honest))
                .expect("a merge");
            let proof = prove_merge(&mut session.transcript(), &lying, &tables).expect("a proof");
            assert!(!verify(&session, &lying, &proof), "claim {k} of group {h}");
        }
    }
    assert!(verify(
        &session,
        &merge,
        &prove_merge(&mut session.transcript(), &merge, &tables).expect("a proof")
    ));
}

#[test]
fn the_two_pass_merge_sends_the_in_memory_merges_messages() {
    for seed in 0..4 {
        let mut rng = SplitMix64::new(0x5420 + seed);
        let session = Session::random(&mut rng);
        let (polys, merge) = fixture(&mut rng);
        let tables: Vec<&Secret<Vec<Fp>>> = polys.iter().map(|p| &p.table).collect();
        let expected = prove_merge(&mut session.transcript(), &merge, &tables).expect("a proof");
        // The largest j* that fits: no polynomial's free variables end before it.
        let largest = largest_table_rounds(merge.polynomials());
        assert!(largest >= 2, "the fixture reaches into its own rounds");
        for j_star in 0..=largest {
            // Each group's tables from its own claims and γ_h alone, before β exists.
            let mut kept = MergeTables::new(merge.polynomials(), merge.claims().len(), j_star)
                .expect("tables");
            for h in 0..4 {
                let blocks: Vec<Vec<(u64, Secret<Vec<Fp>>)>> = polys
                    .iter()
                    .map(|p| {
                        (0..1u64 << p.shape.group_vars.len())
                            .map(|bits| (bits, p.block(bits)))
                            .collect()
                    })
                    .collect();
                kept.add_group(h, &merge.claims()[h], merge.gammas()[h], |i, bits| {
                    blocks[i].iter().find(|(b, _)| *b == bits).map(|(_, t)| t)
                })
                .expect("the group's tables");
            }
            let mut regenerated = vec![0; polys.len()];
            let proof = prove_merge_two_pass(&mut session.transcript(), &merge, &kept, |i| {
                regenerated[i] += 1;
                Secret::new(polys[i].values.clone())
            })
            .expect("a proof");
            assert_eq!(
                proof.proof(),
                expected.proof(),
                "seed {seed}, j* = {j_star}"
            );
            assert_eq!(
                proof.point(),
                expected.point(),
                "seed {seed}, j* = {j_star}"
            );
            assert_eq!(regenerated, vec![1; polys.len()], "one merge pass");
        }
    }
}

#[test]
fn every_shape_rule_refuses_with_its_own_rule() {
    let mut rng = SplitMix64::new(0x5430);
    let (polys, merge) = fixture(&mut rng);
    let shapes: Vec<MergePolynomial> = polys.iter().map(|p| p.shape.clone()).collect();
    let refused = |rule| Err::<(), _>(SumcheckError::Merge { rule });
    let claims = honest_claims(&merge);
    let new =
        |shapes: Vec<MergePolynomial>, claims: Vec<Vec<MergeClaim<Fp2>>>, gammas: Vec<Fp2>| {
            Merge::new(shapes, claims, gammas, Fp2::ONE).map(drop)
        };
    assert_eq!(
        new(vec![], vec![], vec![]),
        refused("a merge has a polynomial")
    );
    let mut wide = shapes.clone();
    wide[0].vars = 41;
    assert_eq!(
        new(wide, claims.clone(), gammas(&merge)),
        refused("a polynomial has at most MAX_MERGE_VARS variables")
    );
    for group_vars in [vec![3, 1], vec![1, 1], vec![5]] {
        let mut odd = shapes.clone();
        odd[0].group_vars = group_vars;
        assert_eq!(
            new(odd, claims.clone(), gammas(&merge)),
            refused("group variables are increasing positions among the variables")
        );
    }
    assert_eq!(
        new(shapes.clone(), claims.clone(), rng.fp2s(3)),
        refused("one gamma per group")
    );
    let edit = |f: &dyn Fn(&mut MergeClaim<Fp2>)| {
        let mut claims = claims.clone();
        f(&mut claims[1][0]);
        new(shapes.clone(), claims, gammas(&merge))
    };
    assert_eq!(
        edit(&|c| c.polynomial = 5),
        refused("a claim is on a polynomial of the merge")
    );
    assert_eq!(
        edit(&|c| {
            c.point.pop();
        }),
        refused("a claim's point has one coordinate per free variable")
    );
    assert_eq!(
        edit(&|c| c.group_bits = 4),
        refused("a claim's bits fit its polynomial's group variables")
    );
    // The provers' inputs.
    let session = Session::random(&mut rng);
    let mut tables: Vec<&Secret<Vec<Fp>>> = polys.iter().map(|p| &p.table).collect();
    let short = Secret::new(rng.fps(8));
    tables[0] = &short;
    assert_eq!(
        prove_merge(&mut session.transcript(), &merge, &tables).map(drop),
        refused("a polynomial's table has 2^n_i entries")
    );
    assert_eq!(
        prove_merge(&mut session.transcript(), &merge, &tables[1..]).map(drop),
        refused("one table per polynomial")
    );
    let largest = largest_table_rounds(merge.polynomials());
    let groups = merge.claims().len();
    assert_eq!(
        MergeTables::<Fp2>::new(merge.polynomials(), groups, largest + 1).map(drop),
        refused("table rounds end before any polynomial's free variables do")
    );
    assert_eq!(
        MergeTables::<Fp2>::new(&[], groups, 0).map(drop),
        refused("a merge has a polynomial")
    );
    let mut kept = MergeTables::new(merge.polynomials(), groups, 1).expect("tables");
    let (claims_0, gamma_0) = (&merge.claims()[0], merge.gammas()[0]);
    assert_eq!(
        kept.add_group(4, claims_0, gamma_0, |_, _| None),
        refused("the group is one of the merge's")
    );
    assert_eq!(
        kept.add_group(0, claims_0, gamma_0, |_, _| None),
        refused("every claim's block is given")
    );
    let wrong = Secret::new(rng.fps(2));
    assert_eq!(
        kept.add_group(0, claims_0, gamma_0, |_, _| Some(&wrong)),
        refused("a block has 2^m_i entries")
    );
    // A group's claims are checked as the merge checks them.
    let edited = |f: &dyn Fn(&mut MergeClaim<Fp2>)| {
        let mut claims = claims_0.clone();
        f(&mut claims[0]);
        claims
    };
    for (claims, rule) in [
        (
            edited(&|c| c.polynomial = 5),
            "a claim is on a polynomial of the merge",
        ),
        (
            edited(&|c| {
                c.point.pop();
            }),
            "a claim's point has one coordinate per free variable",
        ),
        (
            edited(&|c| c.group_bits = 4),
            "a claim's bits fit its polynomial's group variables",
        ),
    ] {
        assert_eq!(
            kept.add_group(0, &claims, gamma_0, |_, _| None),
            refused(rule)
        );
    }
    let two_pass = prove_merge_two_pass(&mut session.transcript(), &merge, &kept, |i| {
        Secret::new(polys[i].values.clone())
    });
    assert_eq!(
        two_pass.map(drop),
        refused("every group's tables are kept, for this merge")
    );
    // A group's tables, kept, cannot be kept again.
    let blocks: Vec<Vec<Secret<Vec<Fp>>>> = polys
        .iter()
        .map(|p| {
            (0..1u64 << p.shape.group_vars.len())
                .map(|b| p.block(b))
                .collect()
        })
        .collect();
    let block = |i: usize, bits: u64| blocks[i].get(bits as usize);
    assert_eq!(kept.add_group(0, claims_0, gamma_0, block), Ok(()));
    assert_eq!(
        kept.add_group(0, claims_0, gamma_0, block),
        refused("each group's tables are kept once")
    );
    // Tables kept for another γ_h, or with other claims, are another merge's.
    let mut other = MergeTables::new(merge.polynomials(), groups, 1).expect("tables");
    for h in 0..groups {
        let gamma = match h {
            2 => merge.gammas()[h] + Fp2::ONE,
            _ => merge.gammas()[h],
        };
        other
            .add_group(h, &merge.claims()[h], gamma, block)
            .expect("the group's tables");
    }
    assert_eq!(
        prove_merge_two_pass(&mut session.transcript(), &merge, &other, |i| {
            Secret::new(polys[i].values.clone())
        })
        .map(drop),
        refused("every group's tables are kept, for this merge")
    );
    let mut fewer = MergeTables::new(merge.polynomials(), groups, 1).expect("tables");
    for h in 0..groups {
        let claims = &merge.claims()[h];
        let claims = match h {
            1 => &claims[..claims.len() - 1],
            _ => &claims[..],
        };
        fewer
            .add_group(h, claims, merge.gammas()[h], block)
            .expect("the group's tables");
    }
    assert_eq!(
        prove_merge_two_pass(&mut session.transcript(), &merge, &fewer, |i| {
            Secret::new(polys[i].values.clone())
        })
        .map(drop),
        refused("every group's tables are kept, for this merge")
    );
}

fn honest_claims(merge: &Merge<Fp2>) -> Vec<Vec<MergeClaim<Fp2>>> {
    merge.claims().to_vec()
}

fn gammas(merge: &Merge<Fp2>) -> Vec<Fp2> {
    merge.gammas().to_vec()
}

fn beta(merge: &Merge<Fp2>) -> Fp2 {
    merge.beta()
}
