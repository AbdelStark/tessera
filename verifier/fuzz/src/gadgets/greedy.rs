// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Dense literal coefficient sums judge the bounded native boundary and every endpoint.
use super::{element, eq_index, equality, field, reshape};
use crate::proofs::Script;
use std::sync::OnceLock;
use tv_core::claims::QuantPoint;
use tv_core::gadgets::{
    GadgetError, Plan,
    greedy::{Boundary, Kind, Source},
};
use tv_core::lowering::{self, TensorId};
use tv_core::validation::{self, GadgetId};
use tv_field::{Fp, Fp2};
use tv_testkit::{Json, read_vectors};

struct Case {
    b: Boundary,
    n: usize,
    v: usize,
    p: usize,
    nv: usize,
    nt: usize,
    target: Vec<Fp2>,
    zero: Vec<Fp2>,
    point: Vec<Fp2>,
    raw: Vec<Fp2>,
    tokens: Vec<Fp2>,
    sources: [TensorId; 3],
}
fn member<'a>(j: &'a Json, k: &str) -> &'a Json {
    j.get(k).unwrap()
}
fn num(j: &Json) -> u64 {
    j.as_u64()
        .unwrap_or_else(|| j.as_str().unwrap().parse().unwrap())
}
fn fields(j: &Json) -> Vec<Fp2> {
    j.as_array()
        .unwrap()
        .iter()
        .map(|j| {
            let x = j.as_array().unwrap();
            Fp2::new(
                Fp::from_canonical(num(&x[0])).unwrap(),
                Fp::from_canonical(num(&x[1])).unwrap(),
            )
        })
        .collect()
}
fn cases() -> &'static [Case] {
    static C: OnceLock<Vec<Case>> = OnceLock::new();
    C.get_or_init(|| {
        let f = read_vectors("greedy-boundary/greedy-boundary.json");
        member(&f, "cases")
            .as_array()
            .unwrap()
            .iter()
            .map(|x| {
                let (n, v, p) = (
                    num(member(x, "n")) as usize,
                    num(member(x, "vocab")) as usize,
                    num(member(x, "prompt")) as usize,
                );
                let mut c = super::models::base();
                c.arch.d_model = 2;
                c.arch.n_heads = 1;
                c.arch.n_kv_heads = 1;
                c.arch.d_head = 2;
                c.arch.d_ff = 6;
                c.arch.vocab = v as u32;
                c.arch.max_context = n.max(8) as u32;
                c.quant.residual_bits = 8;
                c.quant.rsqrt_kappa = 0;
                c.quant.exp_sigma = 3;
                c.quant.residual_exponent = c
                    .quant
                    .act_exponents
                    .iter()
                    .find(|(p, _)| *p == QuantPoint::Embed)
                    .unwrap()
                    .1;
                for (_, s) in &mut c.quant.shift_class_sets {
                    *s = vec![12];
                }
                for (_, s) in &mut c.quant.score_scales {
                    *s = tv_codec::Rational::new(1, 1).unwrap();
                }
                let public = validation::check_public(&c).unwrap();
                let graph = lowering::lower(&public, n as u32).unwrap();
                let plan = Plan::derive(&graph, c.gadget_policy).unwrap();
                let b = plan.greedy_boundary(GadgetId(73), p as u32).unwrap();
                let first = graph.nodes().first().unwrap();
                let last = graph.nodes().last().unwrap();
                let sources = [
                    *last.outputs().first().unwrap(),
                    *first.outputs().first().unwrap(),
                    *first.inputs().first().unwrap(),
                ];
                let mut target = fields(member(x, "target_v"));
                target.extend(fields(member(x, "target_o")));
                Case {
                    b,
                    n,
                    v,
                    p,
                    nv: usize::BITS.saturating_sub(v.saturating_sub(1).leading_zeros()) as usize,
                    nt: usize::BITS.saturating_sub(n.saturating_sub(1).leading_zeros()) as usize,
                    target,
                    zero: fields(member(x, "zero")),
                    point: fields(member(x, "point")),
                    raw: fields(member(x, "raw")),
                    tokens: fields(member(x, "token_raw")),
                    sources,
                }
            })
            .collect()
    })
}

pub(super) fn check(s: &mut Script<'_>) {
    let fixtures = cases();
    let c = &fixtures[usize::from(s.u8()) % fixtures.len()];
    let kind = match s.u8() % 3 {
        0 => Kind::Key,
        1 => Kind::Selected,
        _ => Kind::Tokens,
    };
    let flags = s.u8();
    let shapes = s.u8();
    let mut target = if kind == Kind::Key {
        c.target.clone()
    } else {
        c.zero.clone()
    };
    let mut point = c.point.clone();
    let mut raw = if kind == Kind::Tokens {
        c.tokens.clone()
    } else {
        c.raw.clone()
    };
    if flags & 1 != 0 {
        for v in &mut point {
            *v = element(s);
        }
    }
    if flags & 2 != 0 {
        for v in &mut raw {
            *v = element(s);
        }
    }
    if flags & 4 != 0 {
        for v in &mut target {
            *v = element(s);
        }
    }
    reshape(&mut raw, shapes, Fp2::ZERO);
    reshape(&mut point, shapes >> 2, Fp2::ZERO);
    reshape(&mut target, shapes >> 4, Fp2::ZERO);
    let target_len = if kind == Kind::Key {
        c.target.len()
    } else {
        c.zero.len()
    };
    let incoming = if kind == Kind::Key {
        c.b.key(&target, Fp2::ZERO)
    } else {
        c.b.zero(kind, &target)
    };
    if target.len() != target_len {
        assert!(matches!(incoming, Err(GadgetError::Internal)));
        return;
    }
    let incoming = incoming.unwrap();
    let result = incoming.evaluate(&point, &raw);
    let expected_error = if point.len() != c.nv + c.nt {
        Some(GadgetError::Internal)
    } else if raw.len() != c.b.evaluations(kind) {
        Some(GadgetError::Rejected {
            gadget: GadgetId(73),
        })
    } else {
        None
    };
    if let Some(error) = expected_error {
        match result {
            Err(e) => assert_eq!(e, error),
            Ok(_) => panic!("malformed boundary accepted"),
        };
        return;
    }
    let got = result.unwrap();
    let (v, t) = point.split_at(c.nv);
    let domain = 1usize << c.nt;
    let delta = eq_index(v, 0);
    let expected = if kind == Kind::Tokens {
        let index =
            (0..1usize << c.nv).fold(Fp2::ZERO, |a, i| a + field(i as u64) * eq_index(v, i));
        equality(&target, t) * (index * raw[0] - delta * raw[1])
    } else {
        let mut shifted = Fp2::ZERO;
        for k in 0..=c.nt {
            let coefficient = (0..domain)
                .filter(|&i| {
                    if k == c.nt {
                        i == domain - 1
                    } else {
                        i & ((1 << (k + 1)) - 1) == (1 << k) - 1
                    }
                })
                .fold(Fp2::ZERO, |a, i| a + eq_index(t, i));
            shifted += coefficient * raw[2 + k];
        }
        if kind == Kind::Key {
            let (pv, po) = target.split_at(c.nv);
            let time = (0..c.n - c.p).fold(Fp2::ZERO, |a, u| {
                a + eq_index(po, u) * eq_index(t, c.p - 1 + u)
            });
            let tv = (0..c.v).fold(Fp2::ZERO, |a, i| a + eq_index(pv, i) * eq_index(v, i));
            let mv = (0..c.v).fold(Fp2::ZERO, |a, i| a + eq_index(pv, i));
            let sv = (0..c.v).fold(Fp2::ZERO, |a, w| {
                a + eq_index(v, w) * (0..w).fold(Fp2::ZERO, |a, i| a + eq_index(pv, i))
            });
            time * (mv * delta * raw[1] - tv * raw[0] - sv * shifted)
        } else {
            let active = (c.p - 1..c.n - 1)
                .fold(Fp2::ZERO, |a, i| a + eq_index(&target, i) * eq_index(t, i));
            let mask = (0..c.v).fold(Fp2::ZERO, |a, i| a + eq_index(v, i));
            active * mask * raw[0] * shifted - equality(&target, t) * delta * raw[1]
        }
    };
    assert_eq!(got.value, expected);
    assert_eq!(got.claims.len(), raw.len());
    for (i, claim) in got.claims.iter().enumerate() {
        let (source, expected_point) = if kind == Kind::Tokens {
            if i == 0 {
                (Source::OneHot(c.sources[1]), point.clone())
            } else {
                (Source::Tokens(c.sources[2]), t.to_vec())
            }
        } else if i == 0 {
            (Source::Logits(c.sources[0]), point.clone())
        } else if i == 1 {
            (Source::Maximum(GadgetId(73)), t.to_vec())
        } else {
            let k = i - 2;
            let time = t.iter().enumerate().map(|(j, &r)| {
                if k == c.nt || j < k {
                    Fp2::ZERO
                } else if j == k {
                    Fp2::ONE
                } else {
                    r
                }
            });
            (
                Source::OneHot(c.sources[1]),
                v.iter().copied().chain(time).collect(),
            )
        };
        assert_eq!(claim.source, source);
        assert_eq!(claim.point, expected_point);
        assert_eq!(claim.value, raw[i]);
    }
}
