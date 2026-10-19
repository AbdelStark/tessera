// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Small scripted-challenge arithmetic control, written using only tv-field arithmetic.
//! This checks the polynomial relations directly; zero scripted denominators do not mean Accept.
use crate::driver::{Case, Record, tvf, tvx};
use tv_field::Fp2 as E;
fn sum(xs: impl Iterator<Item = E>) -> E {
    xs.fold(E::ZERO, |a, b| a + b)
}
fn base(x: u64) -> E {
    E::new(tv_field::Fp::from_canonical(x).unwrap(), tv_field::Fp::ZERO)
}
fn eq(a: &[E], b: &[E]) -> E {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(E::ONE, |q, (&a, &b)| {
        q * (a * b + (E::ONE - a) * (E::ONE - b))
    })
}
fn mle(xs: &[E], point: &[E]) -> E {
    assert_eq!(xs.len(), 1 << point.len());
    sum(xs.iter().enumerate().map(|(i, &x)| {
        let w = point.iter().enumerate().fold(E::ONE, |w, (j, &r)| {
            w * if i >> j & 1 == 1 { r } else { E::ONE - r }
        });
        w * x
    }))
}
fn scalar(xs: &[crate::Fp], point: &[E]) -> E {
    mle(
        &xs.iter()
            .map(|&x| E::new(tvf(x), tv_field::Fp::ZERO))
            .collect::<Vec<_>>(),
        point,
    )
}
fn child_evals(level: &[(E, E)], point: &[E]) -> [E; 4] {
    core::array::from_fn(|j| {
        let values: Vec<_> = level
            .iter()
            .skip(j % 2)
            .step_by(2)
            .map(|x| if j < 2 { x.0 } else { x.1 })
            .collect();
        mle(&values, point)
    })
}
/// Returns the number of explicitly checked cubic message evaluations.
pub fn check(case: &Case, record: &Record) -> usize {
    let alpha = tvx(record.alpha);
    let beta = tvx(record.beta);
    let mut trees = Vec::new();
    for table in [false, true] {
        let keys = if table { &case.tk } else { &case.k };
        let values = if table { &case.tv } else { &case.v };
        let leaves: Vec<_> = keys
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                let v = values
                    .as_ref()
                    .map_or(E::ZERO, |v| E::new(tvf(v[i]), tv_field::Fp::ZERO));
                let n = if table {
                    -E::new(tvf(case.m[i]), tv_field::Fp::ZERO)
                } else {
                    E::ONE
                };
                (n, alpha - E::new(tvf(k), tv_field::Fp::ZERO) - beta * v)
            })
            .collect();
        assert!(leaves.len().is_power_of_two());
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            let above = levels
                .last()
                .unwrap()
                .chunks_exact(2)
                .map(|p| (p[0].0 * p[1].1 + p[1].0 * p[0].1, p[0].1 * p[1].1))
                .collect();
            levels.push(above);
        }
        levels.reverse();
        trees.push(levels);
    }
    let heights: Vec<_> = trees.iter().map(|t| t.len() - 1).collect();
    let height = *heights.iter().max().unwrap();
    assert_eq!(record.levels.len(), height);
    assert_eq!(record.roots.len(), 4);
    for (i, tree) in trees.iter().enumerate() {
        assert_eq!(tvx(record.roots[2 * i]), tree[0][0].0);
        assert_eq!(tvx(record.roots[2 * i + 1]), tree[0][0].1);
    }
    // A polynomial identity, without inversion or an acceptance assertion for zero roots.
    assert_eq!(
        trees[0][0][0].0 * trees[1][0][0].1 + trees[1][0][0].0 * trees[0][0][0].1,
        E::ZERO
    );
    let mut incoming = [Vec::<E>::new(), Vec::<E>::new()];
    let mut checked = 0;
    for (i, level) in record.levels.iter().enumerate() {
        let active: Vec<_> = (0..2).filter(|&t| i >= height - heights[t]).collect();
        assert_eq!(active, level.trees);
        assert_eq!(level.proof.rounds.len(), i);
        assert_eq!(level.point.len(), i);
        assert_eq!(level.proof.final_evals.len(), active.len() * 4);
        assert_eq!(level.next.len(), active.len());
        let lambda = tvx(level.lambda);
        let gamma = tvx(level.gamma);
        let point: Vec<_> = level.point.iter().copied().map(tvx).collect();
        let polynomial = |x: &[E]| {
            let mut weight = E::ONE;
            let mut out = E::ZERO;
            for &t in &active {
                let own = i - (height - heights[t]);
                let ownpoint = &x[x.len() - own..];
                let [n0, n1, d0, d1] = child_evals(&trees[t][own + 1], ownpoint);
                out += weight * eq(&incoming[t], ownpoint) * (n0 * d1 + n1 * d0 + lambda * d0 * d1);
                weight *= gamma;
            }
            out
        };
        for (j, message) in level.proof.rounds.iter().enumerate() {
            assert_eq!(message.len(), 3);
            for (z, actual) in [0, 2, 3].into_iter().zip(message) {
                let expected = sum((0..1usize << (i - j - 1)).map(|bits| {
                    let mut x = point[..j].to_vec();
                    x.push(base(z));
                    for k in 0..i - j - 1 {
                        x.push(base((bits >> k & 1) as u64));
                    }
                    polynomial(&x)
                }));
                assert_eq!(
                    tvx(*actual),
                    expected,
                    "scripted direct round i={i} j={j} z={z}"
                );
                checked += 1;
            }
        }
        for (ordinal, &t) in active.iter().enumerate() {
            let own = i - (height - heights[t]);
            let ownpoint = &point[point.len() - own..];
            let ev = child_evals(&trees[t][own + 1], ownpoint);
            assert_eq!(
                level.proof.final_evals[ordinal * 4..ordinal * 4 + 4]
                    .iter()
                    .copied()
                    .map(tvx)
                    .collect::<Vec<_>>(),
                ev
            );
            let mut nextpoint = vec![tvx(level.rho)];
            nextpoint.extend_from_slice(ownpoint);
            let next = &level.next[ordinal];
            assert_eq!(
                next.point.iter().copied().map(tvx).collect::<Vec<_>>(),
                nextpoint
            );
            let ns: Vec<_> = trees[t][own + 1].iter().map(|x| x.0).collect();
            let ds: Vec<_> = trees[t][own + 1].iter().map(|x| x.1).collect();
            assert_eq!(tvx(next.n), mle(&ns, &nextpoint));
            assert_eq!(tvx(next.d), mle(&ds, &nextpoint));
            incoming[t] = nextpoint;
        }
    }
    assert_eq!(record.leaves.len(), 2);
    assert_eq!(record.claims.len(), 1);
    assert!(record.forest_claims.is_empty());
    for t in 0..2 {
        let leaf = &record.leaves[t];
        assert_eq!(
            leaf.point.iter().copied().map(tvx).collect::<Vec<_>>(),
            incoming[t]
        );
        let ns: Vec<_> = trees[t].last().unwrap().iter().map(|x| x.0).collect();
        let ds: Vec<_> = trees[t].last().unwrap().iter().map(|x| x.1).collect();
        assert_eq!(tvx(leaf.n), mle(&ns, &incoming[t]));
        assert_eq!(tvx(leaf.d), mle(&ds, &incoming[t]));
    }
    let claims = &record.claims[0];
    assert_eq!(
        claims
            .key
            .point
            .iter()
            .copied()
            .map(tvx)
            .collect::<Vec<_>>(),
        incoming[0]
    );
    assert_eq!(tvx(claims.key.value), scalar(&case.k, &incoming[0]));
    assert_eq!(
        claims
            .multiplicity
            .point
            .iter()
            .copied()
            .map(tvx)
            .collect::<Vec<_>>(),
        incoming[1]
    );
    assert_eq!(
        tvx(claims.multiplicity.value),
        scalar(&case.m, &incoming[1])
    );
    if let Some(v) = &case.v {
        assert_eq!(record.values.len(), 1);
        let c = claims.value.as_ref().unwrap();
        assert_eq!(
            c.point.iter().copied().map(tvx).collect::<Vec<_>>(),
            incoming[0]
        );
        assert_eq!(tvx(c.value), scalar(v, &incoming[0]));
        assert_eq!(tvx(record.values[0]), scalar(v, &incoming[0]));
    } else {
        assert!(claims.value.is_none());
        assert!(record.values.is_empty());
    }
    checked
}
