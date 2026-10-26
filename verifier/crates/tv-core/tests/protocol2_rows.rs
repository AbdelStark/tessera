// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0013's per-row degree test, from the normative summands and prescribed tiny shapes.
//! Uses the test kit's u128 field arithmetic, not tv-field or the prover's relation evaluators.
use tv_core::schedule::degree::{Constant, Key, degree_row};
use tv_testkit::SplitMix64;
use tv_testkit::sumcheck::{Fe, eq, mle};

type Formula = Box<dyn Fn(&[Fe]) -> Fe>;

struct Tensor {
    values: Vec<Fe>,
    mask: Option<Vec<Fe>>,
}
impl Tensor {
    fn new(rng: &mut SplitMix64, vars: usize, public: bool) -> Self {
        let mut table = || {
            (0..1 << vars)
                .map(|_| Fe::new(rng.next_u64(), rng.next_u64()))
                .collect()
        };
        let values = table();
        let mask = if public { None } else { Some(table()) };
        Self { values, mask }
    }
    fn at(&self, p: &[Fe]) -> Fe {
        let value = mle(&self.values, p);
        match &self.mask {
            None => value,
            Some(mask) => {
                value
                    + mle(mask, p)
                        * p.iter()
                            .skip(1)
                            .fold(Fe::ONE, |z, &x| z * x * (Fe::ONE - x))
            }
        }
    }
}
fn random(rng: &mut SplitMix64, n: usize) -> Vec<Fe> {
    (0..n)
        .map(|_| Fe::new(rng.next_u64(), rng.next_u64()))
        .collect()
}
fn selector(n: usize, predicate: impl Fn(usize) -> bool) -> Vec<Fe> {
    (0..1 << n)
        .map(|i| if predicate(i) { Fe::ONE } else { Fe::ZERO })
        .collect()
}
fn two() -> Fe {
    Fe::base(2)
}

struct Case {
    relation: &'static str,
    public: Vec<bool>,
    constant: Option<Constant>,
    stacked: bool,
    vars: usize,
    formula: Formula,
}
impl Case {
    fn key(&self) -> Key<'_> {
        Key {
            relation: self.relation,
            public: &self.public,
            constant: self.constant,
            stacked: self.stacked,
        }
    }
}

/// Build the RFC's mathematical relation, including genuine padded selectors. The key selects
/// the table row, while this formula never reads that row's degree or factors.
fn build(
    relation: &'static str,
    public: Vec<bool>,
    constant: Option<Constant>,
    stacked: bool,
    rng: &mut SplitMix64,
) -> Case {
    let vars = if matches!(
        relation,
        "OneHotIndex"
            | "OneHotCount"
            | "Tokens"
            | "GreedyKey"
            | "GreedySelected"
            | "Observe"
            | "Recomposition"
    ) || constant == Some(Constant::Layer)
        || stacked
    {
        4
    } else {
        3
    };
    let rho = random(rng, vars);
    let p = |i: usize| public.get(i).copied().unwrap_or(false);
    let formula: Formula = match relation {
        "MatMulWeight" | "MatMulAct" => {
            let a = Tensor::new(rng, vars + 1, false);
            let b = Tensor::new(rng, vars + 1, false);
            let fixed = random(rng, 2);
            Box::new(move |x| {
                let mut u = x.to_vec();
                u.push(fixed[0]);
                let mut v = x.to_vec();
                v.push(fixed[1]);
                let weight = if stacked {
                    eq(&rho[vars - 1..], &x[vars - 1..])
                } else {
                    Fe::ONE
                };
                weight * a.at(&u) * b.at(&v)
            })
        }
        "Mul" | "Equality.Mul" => {
            let equality = relation == "Equality.Mul";
            let offset = usize::from(equality);
            let a = Tensor::new(rng, vars, p(offset));
            let b = Tensor::new(rng, vars, p(offset + 1));
            let y = Tensor::new(rng, vars, false);
            Box::new(move |x| {
                eq(&rho, x)
                    * if equality {
                        y.at(x) - a.at(x) * b.at(x)
                    } else {
                        a.at(x) * b.at(x)
                    }
            })
        }
        "Xor" | "And" | "Ch" | "Xor3" | "Maj" => {
            let a = Tensor::new(rng, vars, p(0));
            let b = Tensor::new(rng, vars, p(1));
            let c = Tensor::new(rng, vars, p(2));
            Box::new(move |x| {
                let (a, b, c) = (a.at(x), b.at(x), c.at(x));
                eq(&rho, x)
                    * match relation {
                        "And" => a * b,
                        "Xor" => a + b - two() * a * b,
                        "Ch" => a * b + c - a * c,
                        "Xor3" => {
                            a + b + c - two() * (a * b + a * c + b * c) + Fe::base(4) * a * b * c
                        }
                        "Maj" => a * b + a * c + b * c - two() * a * b * c,
                        _ => unreachable!(),
                    }
            })
        }
        "Equality.ConstMul" => {
            let y = Tensor::new(rng, vars, false);
            let a = Tensor::new(rng, vars, p(1));
            let class = constant.unwrap();
            let c = Tensor::new(
                rng,
                match class {
                    Constant::Scalar => 0,
                    Constant::Layer => 1,
                    Constant::Channel => vars,
                },
                true,
            );
            Box::new(move |x| {
                let at = match class {
                    Constant::Scalar => &[][..],
                    Constant::Layer => &x[vars - 1..],
                    Constant::Channel => x,
                };
                eq(&rho, x) * (y.at(x) - c.at(at) * a.at(x))
            })
        }
        "Equality.ConstAdd" | "Equality.Add" | "Equality.Sub" => {
            let y = Tensor::new(rng, vars, false);
            let a = Tensor::new(rng, vars, p(1));
            let b = Tensor::new(rng, vars, p(2));
            let shift = random(rng, 1)[0];
            let psi = selector(vars, |i| i >> 1 < 3);
            Box::new(move |x| {
                let rhs = match relation {
                    "Equality.ConstAdd" => a.at(x) + shift * mle(&psi, x),
                    "Equality.Add" => a.at(x) + b.at(x),
                    _ => a.at(x) - b.at(x),
                };
                eq(&rho, x) * (y.at(x) - rhs)
            })
        }
        "Rotate" => {
            let kernel = Tensor::new(rng, vars, true);
            let a = Tensor::new(rng, vars, false);
            Box::new(move |x| kernel.at(x) * a.at(x))
        }
        "Selection" => {
            let kernel = Tensor::new(rng, vars, true);
            let a = Tensor::new(rng, vars + 2, false);
            let fixed = random(rng, 2);
            Box::new(move |x| {
                let mut at = vec![fixed[0]];
                at.extend_from_slice(x);
                at.push(fixed[1]);
                kernel.at(x) * a.at(&at)
            })
        }
        "ScaleChannels" => {
            let a = Tensor::new(rng, vars, false);
            let c = Tensor::new(rng, vars, true);
            Box::new(move |x| eq(&rho, x) * c.at(x) * a.at(x))
        }
        "SumSquares" | "Booleanity" => {
            let a = Tensor::new(rng, vars, false);
            Box::new(move |x| {
                if relation == "SumSquares" {
                    eq(&rho[1..], &x[1..]) * a.at(x) * a.at(x)
                } else {
                    eq(&rho, x) * a.at(x) * (Fe::ONE - a.at(x))
                }
            })
        }
        "AssertEqual" => {
            let a = Tensor::new(rng, vars, p(0));
            let b = Tensor::new(rng, vars, p(1));
            let psi = selector(vars, |i| i >> 1 < 3);
            Box::new(move |x| eq(&rho, x) * mle(&psi, x) * (a.at(x) - b.at(x)))
        }
        "OneHotVocabulary" | "InputPadding" | "BitPadding" | "RootPad" | "ResidualPad"
        | "MaxPad" | "ReciprocalPad" => {
            let a = Tensor::new(rng, vars, false);
            let psi = if matches!(relation, "OneHotVocabulary" | "BitPadding") {
                selector(vars, |i| i < 5)
            } else {
                selector(vars, |i| i >> 1 < 3)
            };
            Box::new(move |x| eq(&rho, x) * (Fe::ONE - mle(&psi, x)) * a.at(x))
        }
        "OneHotCount" | "OneHotIndex" | "Tokens" => {
            let h = Tensor::new(rng, vars, false);
            let t = Tensor::new(rng, 2, relation == "OneHotIndex" && p(1));
            let real = selector(2, |i| i < 3);
            Box::new(move |x| {
                let v = x[0] + two() * x[1];
                let value = if relation == "OneHotCount" {
                    h.at(x) - mle(&real, &x[2..]) * Fe::base(4).inverse().unwrap()
                } else {
                    v * h.at(x) - t.at(&x[2..]) * Fe::base(4).inverse().unwrap()
                };
                eq(&rho[2..], &x[2..]) * value
            })
        }
        "Recomposition" => {
            let a = Tensor::new(rng, 2, p(0));
            let b = Tensor::new(rng, 3, false);
            let h = Tensor::new(rng, 4, false);
            let wb = Tensor::new(rng, 2, true);
            let wh = Tensor::new(rng, 2, true);
            Box::new(move |x| {
                eq(&rho[2..], &x[2..])
                    * (a.at(&x[2..]) * Fe::base(4).inverse().unwrap()
                        - wb.at(&x[..2]) * b.at(&x[1..])
                        - wh.at(&x[..2]) * h.at(x))
            })
        }
        "Product" => {
            let a = Tensor::new(rng, vars, false);
            let b = Tensor::new(rng, vars, false);
            let c = Tensor::new(rng, vars, false);
            Box::new(move |x| eq(&rho, x) * a.at(x) * b.at(x) * c.at(x))
        }
        "Key" | "Forest" => {
            let a = Tensor::new(rng, vars, false);
            let causal = selector(vars, |i| i & 1 <= i >> 1);
            Box::new(move |x| eq(&rho, x) * mle(&causal, x) * a.at(x))
        }
        "Reciprocal" => {
            let a = Tensor::new(rng, vars, false);
            let b = Tensor::new(rng, vars, false);
            Box::new(move |x| eq(&rho, x) * (a.at(x) * b.at(x) - Fe::ONE))
        }
        "GreedyKey" => {
            let a = Tensor::new(rng, vars, false);
            let time = Tensor::new(rng, 2, true);
            let vocab = Tensor::new(rng, 2, true);
            Box::new(move |x| time.at(&x[2..]) * vocab.at(&x[..2]) * a.at(x))
        }
        "GreedySelected" => {
            let logits = Tensor::new(rng, 4, false);
            let h = Tensor::new(rng, 4, false);
            let coefficients: Vec<_> = (0..4).map(|_| Tensor::new(rng, 2, true)).collect();
            Box::new(move |x| {
                let hplus = (0..4).fold(Fe::ZERO, |s, j| {
                    // Boolean time kills H's own vanishing mask, even at non-Boolean vocabulary.
                    let point = [
                        x[0],
                        x[1],
                        Fe::base(j as u64 & 1),
                        Fe::base((j >> 1) as u64),
                    ];
                    s + coefficients[j].at(&x[2..]) * h.at(&point)
                });
                eq(&rho, x) * logits.at(x) * hplus
            })
        }
        "Observe" => Box::new(|_| Fe::ZERO),
        "Fold" => {
            let mask = Tensor::new(rng, vars, true);
            Box::new(move |x| eq(&rho, x) * mask.at(x))
        }
        _ => panic!("missing independent formula: {relation}"),
    };
    Case {
        relation,
        public,
        constant,
        stacked,
        vars,
        formula,
    }
}

fn patterns(n: usize) -> impl Iterator<Item = Vec<bool>> {
    (0..1 << n).map(move |m| (0..n).map(|i| m >> i & 1 == 1).collect())
}
fn cases() -> Vec<Case> {
    let mut rng = SplitMix64::new(0x0565_1010);
    let mut cases = Vec::new();
    let mut add = |name, public, constant, stacked| {
        cases.push(build(name, public, constant, stacked, &mut rng))
    };
    for name in ["MatMulWeight", "MatMulAct"] {
        for stacked in [false, true] {
            add(name, vec![false; 2], None, stacked);
        }
    }
    // Head stacking has the same one-axis formula, independently of the layer interpretation.
    add("MatMulAct", vec![false; 2], None, true);
    for name in ["Mul", "Xor", "And"] {
        for p in patterns(2).filter(|p| !p.iter().all(|v| *v)) {
            add(name, p, None, false);
        }
    }
    for name in ["Xor3", "Maj", "Ch"] {
        for p in patterns(3).filter(|p| !p.iter().all(|v| *v)) {
            add(name, p, None, false);
        }
    }
    for constant in [Constant::Scalar, Constant::Channel, Constant::Layer] {
        for input in [false, true] {
            add(
                "Equality.ConstMul",
                vec![false, input],
                Some(constant),
                constant == Constant::Layer,
            );
        }
    }
    for input in [false, true] {
        add("Equality.ConstAdd", vec![false, input], None, false);
    }
    for name in ["Equality.Add", "Equality.Sub", "Equality.Mul"] {
        for inputs in patterns(2) {
            let mut p = vec![false];
            p.extend(inputs);
            add(name, p, None, false);
        }
    }
    for name in [
        "SumSquares",
        "Rotate",
        "Selection",
        "ScaleChannels",
        "Booleanity",
        "OneHotVocabulary",
        "InputPadding",
        "BitPadding",
        "OneHotCount",
    ] {
        add(name, vec![false], None, false);
    }
    for input in [false, true] {
        add("OneHotIndex", vec![false, input], None, false);
        add("Recomposition", vec![input, false, false], None, false);
    }
    for p in patterns(2) {
        add("AssertEqual", p, None, false);
    }
    for name in [
        "Product",
        "RootPad",
        "ResidualPad",
        "Key",
        "Forest",
        "Reciprocal",
        "MaxPad",
        "ReciprocalPad",
        "Tokens",
        "GreedyKey",
        "GreedySelected",
        "Observe",
        "Fold",
    ] {
        add(name, vec![], None, false);
    }
    cases
}

/// Enumerate each round's Boolean tail, evaluate at fourteen nodes, and recover the degree by
/// successive forward differences. Previous round coordinates are generic non-Boolean points.
fn measured(case: &Case) -> usize {
    let mut rng = SplitMix64::new(0x0565_1111);
    let prefix = random(&mut rng, case.vars);
    (0..case.vars)
        .map(|round| {
            let mut values: Vec<Fe> = (0..14)
                .map(|node| {
                    (0..1 << (case.vars - round - 1)).fold(Fe::ZERO, |s, tail| {
                        let mut x = prefix[..round].to_vec();
                        x.push(Fe::base(node));
                        x.extend((0..case.vars - round - 1).map(|i| Fe::base((tail >> i) & 1)));
                        s + (case.formula)(&x)
                    })
                })
                .collect();
            let mut degree = 0;
            for k in 0..14 {
                if values[0] != Fe::ZERO {
                    degree = k;
                }
                values = values.windows(2).map(|p| p[1] - p[0]).collect();
            }
            degree
        })
        .max()
        .unwrap()
}

#[test]
fn every_normative_row_bounds_its_polynomial_and_reaches_its_tight_degree() {
    let cases = cases();
    let mut names = std::collections::BTreeSet::new();
    for case in &cases {
        let row = degree_row(case.key()).unwrap();
        let actual = measured(case);
        assert!(
            actual <= row.degree,
            "{:?}: degree {actual}, bound {}",
            case.key(),
            row.degree
        );
        assert_eq!(actual, row.reach, "{:?}", case.key());
        assert_eq!(row.degree, row.base + 2 * row.masked);
        names.insert(case.relation);
    }
    assert_eq!(names.len(), 38);
    assert_eq!(cases.len(), 85);
}

#[test]
fn lowering_any_tight_row_by_one_fails_the_independent_measurement() {
    let mut checked = 0;
    for case in cases() {
        let row = degree_row(case.key()).unwrap();
        if row.degree == row.reach {
            assert!(measured(&case) > row.degree - 1, "{:?}", case.key());
            checked += 1;
        }
    }
    assert_eq!(checked, 77);
}
