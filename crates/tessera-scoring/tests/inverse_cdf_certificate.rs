// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0017's test plan: the reference sampler and the `InverseCdf` certificate agree on 10^5
//! random distributions (RFC-0009 Lemma 10, RFC-0058). For each, the gadget's honest token is
//! `inverse_cdf`'s, and its witness passes every row equality and range request.

use std::collections::HashMap;
use tessera_field::{Field, Goldilocks as Fp, PrimeField};
use tessera_gadgets::{CdfConstraints, CdfTerm, CdfWitness, InverseCdf};
use tessera_scoring::inverse_cdf;
use tessera_tables::TableKind as Kind;
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

/// SplitMix64 (Steele, Lea, Flood 2014), for seeded random distributions.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Whether every equality is zero and every range request is a row of its table, on the gadget's
/// witness, with the sums `Z`, `B` and `S` from its evaluation.
fn passes(c: &CdfConstraints<Fp>, specs: &[TableSpec], w: &CdfWitness, sums: &[Fp], u: Fp) -> bool {
    let value = |t| match t {
        CdfTerm::Draw => u,
        CdfTerm::Total => sums[0],
        CdfTerm::Below => sums[1],
        CdfTerm::Selected => sums[2],
        CdfTerm::RowSelector => Fp::ONE,
        CdfTerm::Column(column) => Fp::from_i64(w.value(column).unwrap()),
    };
    c.equalities.iter().all(|e| e.evaluate(value) == Fp::ZERO)
        && c.lookups.iter().all(|l| {
            let Kind::Range { bits } = specs[l.table.0 as usize - 1].kind else {
                return false;
            };
            l.key.evaluate(value).to_canonical_u64() < 1 << bits
        })
}

fn specs() -> Vec<TableSpec> {
    (1..=32)
        .map(|bits| {
            GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                .unwrap()
                .spec()
                .clone()
        })
        .collect()
}

#[test]
fn the_certificate_agrees_with_the_reference_sampler_on_100000_distributions() {
    let specs = specs();
    let mut gadgets: HashMap<(usize, u8), (InverseCdf<Fp>, CdfConstraints<Fp>)> = HashMap::new();
    let mut rng = SplitMix64(0x0144_0017);
    for case in 0..100_000 {
        let vocab = 1 + (rng.next() % 48) as usize;
        let sigma = (rng.next() % 33) as u8;
        let zeros = rng.next() % 4;
        let mut weights: Vec<u64> = (0..vocab)
            .map(|_| {
                let w = rng.next() % ((1 << sigma) + 1);
                if rng.next() % 4 < zeros { 0 } else { w }
            })
            .collect();
        // A real row's vocabulary certificate gives its maximal logit 2^σ.
        let top = (rng.next() % vocab as u64) as usize;
        weights[top] = 1 << sigma;
        let u = rng.next() as u32;
        let expected = inverse_cdf(&weights, u).unwrap();
        let (gadget, constraints) = gadgets.entry((vocab, sigma)).or_insert_with(|| {
            let gadget = InverseCdf::new(vocab, sigma, &specs).unwrap();
            let constraints = gadget.constraints().unwrap();
            (gadget, constraints)
        });
        let mut e: Vec<Fp> = weights.iter().map(|&w| Fp::from_u64_reduced(w)).collect();
        e.resize(vocab.next_power_of_two(), Fp::ZERO);
        let draw = Fp::from_u64_reduced(u64::from(u));
        let h = gadget.sample(&e, draw, true).unwrap();
        let ones: Vec<usize> = (0..h.indicator().len())
            .filter(|&v| h.indicator()[v] == Fp::ONE)
            .collect();
        assert_eq!(ones, [expected], "case {case}");
        let witness = gadget.witness(&e, draw, &h, true).unwrap();
        let sums = gadget.evaluate(&e, &h).unwrap().sums;
        assert!(
            passes(constraints, &specs, &witness, &sums, draw),
            "case {case}"
        );
    }
}

/// The rows of `spec/vectors/scoring/inverse_cdf.txt`, written by an independent bisection,
/// whose weights the vocabulary table can produce: at most `2^18` weights in `[0, 2^32]`, with
/// mass. Each row's `σ` is the least that holds its weights.
#[test]
fn the_certificate_selects_the_shared_vectors_tokens() {
    let specs = specs();
    let mut checked = 0;
    for line in include_str!("../../../spec/vectors/scoring/inverse_cdf.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let row: Vec<&str> = line.split_whitespace().collect();
        let (u, token) = (row[0].parse::<u64>().unwrap(), row[1]);
        let weights: Vec<u64> = row[2..].iter().map(|w| w.parse().unwrap()).collect();
        let largest = weights.iter().copied().max().unwrap_or(0);
        if token == "none" || largest > 1 << 32 {
            continue;
        }
        let sigma = (64 - largest.leading_zeros()).saturating_sub(1) as u8;
        let sigma = if largest > 1 << sigma {
            sigma + 1
        } else {
            sigma
        };
        let gadget = InverseCdf::<Fp>::new(weights.len(), sigma, &specs).unwrap();
        let mut e: Vec<Fp> = weights.iter().map(|&w| Fp::from_u64_reduced(w)).collect();
        e.resize(weights.len().next_power_of_two(), Fp::ZERO);
        let h = gadget.sample(&e, Fp::from_u64_reduced(u), true).unwrap();
        let token: usize = token.parse().unwrap();
        assert_eq!(h.indicator()[token], Fp::ONE, "u {u}, weights {weights:?}");
        gadget
            .witness(&e, Fp::from_u64_reduced(u), &h, true)
            .unwrap();
        checked += 1;
    }
    assert_eq!(checked, 131, "the rows that fit the vocabulary table");
}
