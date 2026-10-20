// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bounded gadget cases for the sumcheck target's `GDG` input format.
//! Literal cube sums, integer prefixes and the independent frozen model fixtures
//! judge arithmetic, exact errors and retained debts. A final relation alone is
//! never called proof acceptance; source/opening/lookup obligations remain explicit.

mod models;
use crate::proofs::Script;
use tv_core::gadgets::{
    GadgetError,
    argmax::{Argmax, Role},
    comparison::{Comparison, Values as CompareValues},
    decomposition::{Decomposition, Values as Digits},
    root_products::Products,
};
use tv_core::validation::GadgetId;
use tv_field::{Fp, Fp2};

/// The nine gadget families in the bounded script format.
pub const FAMILIES: u8 = 9;
fn element(s: &mut Script<'_>) -> Fp2 {
    Fp2::new(Fp::reduce(s.u64()), Fp::reduce(s.u64()))
}
fn nonzero(s: &mut Script<'_>) -> Fp2 {
    let x = element(s);
    if x == Fp2::ZERO { Fp2::ONE } else { x }
}
fn field(n: u64) -> Fp2 {
    Fp2::from(Fp::reduce(n))
}
fn point(s: &mut Script<'_>, n: usize) -> Vec<Fp2> {
    (0..n).map(|_| element(s)).collect()
}
fn reshape<T: Clone>(v: &mut Vec<T>, shape: u8, fill: T) {
    match shape % 4 {
        1 => {
            v.pop();
        }
        2 => v.push(fill),
        3 => v.clear(),
        _ => (),
    }
}
fn eq_index(point: &[Fp2], index: usize) -> Fp2 {
    point.iter().enumerate().fold(Fp2::ONE, |a, (i, &x)| {
        a * if index >> i & 1 == 0 { Fp2::ONE - x } else { x }
    })
}
// Sum both equality tables rather than use tv-core's product formula.
fn equality(a: &[Fp2], b: &[Fp2]) -> Fp2 {
    assert_eq!(a.len(), b.len());
    (0..1usize << a.len()).fold(Fp2::ZERO, |sum, i| sum + eq_index(a, i) * eq_index(b, i))
}
fn less(point: &[Fp2], bound: usize) -> Fp2 {
    (0..bound).fold(Fp2::ZERO, |a, i| a + eq_index(point, i))
}
fn triangular(target: &[Fp2], point: &[Fp2]) -> Fp2 {
    (0..1usize << point.len()).fold(Fp2::ZERO, |a, v| {
        a + (0..v).fold(Fp2::ZERO, |a, i| {
            a + eq_index(target, i) * eq_index(point, v)
        })
    })
}
fn rejected(gadget: GadgetId) -> GadgetError {
    GadgetError::Rejected { gadget }
}
fn product(width: u8, j: usize, increment: bool, values: &[Fp2]) -> Fp2 {
    let count = usize::from(width).div_ceil(8);
    let mut sum = Fp2::ZERO;
    for a in 0..count {
        for b in 0..count {
            for k in 0..4 {
                if a + b + 2 * k == j {
                    let x = values[a]
                        + if increment && a == 0 {
                            Fp2::ONE
                        } else {
                            Fp2::ZERO
                        };
                    let y = values[b]
                        + if increment && b == 0 {
                            Fp2::ONE
                        } else {
                            Fp2::ZERO
                        };
                    sum += x * y * values[count + k];
                }
            }
        }
    }
    sum
}
fn decomposition(s: &mut Script<'_>) {
    let b = s.u8();
    let l = s.u8();
    let bits = if b == 255 { 0 } else { 1 + b % 62 };
    let limb = if l == 255 { 0 } else { 1 + l % 32 };
    let derived = Decomposition::derive(bits, limb);
    if bits == 0 || limb == 0 {
        assert!(matches!(derived, Err(GadgetError::Internal)));
        return;
    }
    let plan = derived.unwrap();
    let count = usize::from(bits).div_ceil(usize::from(limb));
    assert_eq!(plan.limbs(), count);
    let variables = usize::from(s.u8() % 5);
    let p = point(s, variables);
    let mut keys: Vec<_> = (0..count).map(|_| element(s)).collect();
    let mut low = keys[..count - 1].to_vec();
    let radix = field(1u64 << limb);
    let mut value = keys.iter().rev().fold(Fp2::ZERO, |a, &x| a * radix + x);
    if s.u8() & 1 != 0 {
        value += nonzero(s);
    }
    let shape = s.u8();
    reshape(&mut low, shape, Fp2::ZERO);
    reshape(&mut keys, s.u8(), Fp2::ZERO);
    let expected = low.len() == count - 1
        && keys.len() == count
        && low.iter().zip(&keys).all(|(a, b)| a == b)
        && keys.iter().rev().fold(Fp2::ZERO, |a, &x| a * radix + x) == value;
    match plan.check(
        GadgetId(73),
        Digits {
            point: &p,
            value,
            low: &low,
            keys: &keys,
        },
    ) {
        Err(e) => {
            assert!(!expected);
            assert_eq!(e, rejected(GadgetId(73)));
        }
        Ok(checked) => {
            assert!(expected);
            let claims: Vec<_> = checked.claims().collect();
            assert_eq!(claims.len(), count + 1);
            for (i, c) in claims.iter().enumerate() {
                assert_eq!(c.point, p);
                if i == 0 {
                    assert_eq!(c.limb, None);
                    assert_eq!(c.value, value);
                    assert_eq!(c.range_bits, None);
                } else {
                    assert_eq!(c.limb, Some(i - 1));
                    assert_eq!(c.value, keys[i - 1]);
                    assert_eq!(
                        c.range_bits,
                        Some((bits - usize::from(limb).saturating_mul(i - 1) as u8).min(limb))
                    );
                }
            }
        }
    }
}
fn comparison(s: &mut Script<'_>) {
    let left = [u64::from(s.u16()), u64::from(s.u16()), 0, 0];
    let right = [u64::from(s.u16()), u64::from(s.u16()), 0, 0];
    let x = left[0] + 256 * left[1];
    let y = right[0] + 256 * right[1];
    let plan = Comparison::derive(8, &[65535, 65535, 0, 0], &[65535, 65535, 0, 0]).unwrap();
    let mut arrays: Vec<Vec<Fp2>> = (0..6).map(|_| Vec::new()).collect();
    for j in 0..4 {
        let radix = 1u64 << (8 * (j + 1));
        let scale = 1u64 << (8 * j);
        let prefix = |input: &[u64]| {
            input
                .iter()
                .take(j + 1)
                .enumerate()
                .map(|(k, &a)| a * (1u64 << (8 * k)))
                .sum::<u64>()
        };
        for (a, v) in [
            x / scale % 256,
            prefix(&left) / radix,
            y / scale % 256,
            prefix(&right) / radix,
            y.wrapping_sub(x) / scale % 256,
            u64::from(x % radix > y % radix),
        ]
        .into_iter()
        .enumerate()
        {
            arrays[a].push(field(v));
        }
    }
    let mut l: Vec<_> = left.into_iter().map(field).collect();
    let r: Vec<_> = right.into_iter().map(field).collect();
    let mode = s.u8() % 4;
    if mode == 1 {
        let a = s.below(6);
        let j = s.below(4);
        arrays[a][j] += nonzero(s);
    } else if mode == 2 {
        let j = s.below(4);
        l[j] += nonzero(s);
    } else if mode == 3 {
        let a = s.below(6);
        arrays[a].pop();
    }
    let p = point(s, 2);
    let expected = x <= y && mode == 0;
    let values = CompareValues {
        point: &p,
        left: &l,
        right: &r,
        arrays: [
            &arrays[0], &arrays[1], &arrays[2], &arrays[3], &arrays[4], &arrays[5],
        ],
    };
    match plan.check(GadgetId(74), values) {
        Err(e) => {
            assert!(!expected);
            assert_eq!(e, rejected(GadgetId(74)));
        }
        Ok(checked) => {
            assert!(expected);
            let claims: Vec<_> = checked.claims().collect();
            assert_eq!(claims.len(), 32);
            use tv_core::advice::ComparisonRole as R;
            use tv_core::gadgets::comparison::Input;
            let inputs = [
                Input::Left,
                Input::Right,
                Input::Witness(R::LeftDigit),
                Input::Witness(R::LeftCarry),
                Input::Witness(R::RightDigit),
                Input::Witness(R::RightCarry),
                Input::Witness(R::Difference),
                Input::Witness(R::Borrow),
            ];
            for (i, c) in claims.iter().enumerate() {
                let group = i / 4;
                let j = i % 4;
                assert_eq!(c.point, p);
                assert_eq!(c.limb, j);
                assert_eq!(c.input, inputs[group]);
                assert_eq!(
                    c.value,
                    match group {
                        0 => l[j],
                        1 => r[j],
                        _ => arrays[group - 2][j],
                    }
                );
                let width = match group {
                    0 | 1 => None,
                    3 | 5 => Some([8, 9, 1, 1][j]),
                    7 => Some(1),
                    _ => Some(8),
                };
                assert_eq!(c.range_bits, width);
            }
        }
    }
}
fn products(s: &mut Script<'_>) {
    let width = 1 + s.u8() % 32;
    let plan = Products::derive(width).unwrap();
    let j = usize::from(s.u8()) % (plan.coefficients() + 1);
    let inc = s.u8() & 1 != 0;
    let mut values = point(s, plan.evaluations());
    reshape(&mut values, s.u8(), Fp2::ZERO);
    let expected = if values.len() != plan.evaluations() || j == plan.coefficients() {
        None
    } else {
        Some(product(width, j, inc, &values))
    };
    assert_eq!(plan.evaluate(j, inc, &values), expected);
}
fn argmax(s: &mut Script<'_>) {
    let vocab = 1 + u32::from(s.u8() % 8);
    let rows = 1 + u32::from(s.u8() % 4);
    let site = Argmax::derive(GadgetId(75), vocab, rows).unwrap();
    let vb = site.variables() - site.row_variables();
    let target = point(s, site.variables());
    let row = point(s, site.row_variables());
    let m = element(s);
    let key = site.lookup(&target, element(s), Fp2::ZERO).unwrap();
    let selected = site.selected(&row, m).unwrap();
    let suffix = site.suffix(&target, element(s)).unwrap();
    assert_eq!(key.lookup().bits(), 29);
    assert_eq!(selected.input_claim().role, Role::Maximum);
    assert_eq!(selected.input_claim().value, m);
    let kind = s.u8() % 3;
    let count = [3, 2, 1][usize::from(kind)];
    let mut p = point(s, site.variables());
    let mut values = point(s, count);
    reshape(&mut values, s.u8(), Fp2::ZERO);
    if s.u8() % 8 == 1 {
        p.push(Fp2::ONE);
    }
    let result = match kind {
        0 => key.evaluate(&p, &values),
        1 => selected.evaluate(&p, &values),
        _ => suffix.evaluate(&p, &values),
    };
    if p.len() != site.variables() {
        assert!(matches!(result, Err(GadgetError::Internal)));
        return;
    }
    if values.len() != count {
        assert!(matches!(result,Err(e)if e==rejected(GadgetId(75))));
        return;
    }
    let checked = result.unwrap();
    let (value, roles) = match kind {
        0 => (
            equality(&target, &p)
                * less(&p[..vb], vocab as usize)
                * (values[0] - values[1] - values[2]),
            vec![Role::Maximum, Role::Logits, Role::Suffix],
        ),
        1 => (
            equality(&row, &p[vb..]) * values[0] * values[1],
            vec![Role::Indicator, Role::Logits],
        ),
        _ => (
            equality(&target[vb..], &p[vb..]) * triangular(&target[..vb], &p[..vb]) * values[0],
            vec![Role::Indicator],
        ),
    };
    assert_eq!(checked.value(), value);
    let claims: Vec<_> = checked.claims().collect();
    assert_eq!(claims.iter().map(|c| c.role).collect::<Vec<_>>(), roles);
    for (i, c) in claims.iter().enumerate() {
        assert_eq!(c.value, values[i]);
        assert_eq!(
            c.point,
            if c.role == Role::Maximum {
                &p[vb..]
            } else {
                &p
            }
        );
    }
}
/// Feed a bounded gadget case to the verifier and judge it from independent data/math.
pub fn check(data: &[u8]) {
    let mut s = Script::new(data);
    match s.u8() % FAMILIES {
        0 => decomposition(&mut s),
        1 => comparison(&mut s),
        2 => products(&mut s),
        3 => models::requant(&mut s),
        4 => models::root(&mut s),
        5 => models::softmax(&mut s),
        6 => models::onehot(&mut s),
        7 => argmax(&mut s),
        _ => models::outputs(&mut s),
    }
}
