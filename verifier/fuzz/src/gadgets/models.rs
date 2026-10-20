// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Frozen independent model fixtures plus bounded changes and literal public factors.
use super::*;
use std::sync::OnceLock;
use tv_codec::{DecodeLimits, Rational};
use tv_core::advice::{Auxiliary, Comparison as Side, ComparisonRole, Source};
use tv_core::claims::{ClaimList, GadgetPolicy, QuantPoint, Site};
use tv_core::gadgets::{Plan, argmax_outputs::Binding, onehot::Relation, softmax::Padding};
use tv_core::lowering::{self, Graph, NodeId, TensorId};
use tv_core::validation;
use tv_testkit::{Json, hex, read_vectors};
fn member<'a>(v: &'a Json, k: &str) -> &'a Json {
    v.get(k).unwrap()
}
fn list<'a>(v: &'a Json, k: &str) -> &'a [Json] {
    member(v, k).as_array().unwrap()
}
fn num(v: &Json) -> u64 {
    v.as_u64().unwrap()
}
fn fp(v: &Json) -> Fp2 {
    let a = v.as_array().unwrap();
    Fp2::new(
        Fp::from_canonical(num(&a[0])).unwrap(),
        Fp::from_canonical(num(&a[1])).unwrap(),
    )
}
fn fields(v: &Json) -> Vec<Fp2> {
    v.as_array().unwrap().iter().map(fp).collect()
}
fn base() -> ClaimList {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    let bytes = BYTES.get_or_init(|| {
        let f = read_vectors("claims/claim-lists.json");
        let c = list(&f, "cases")
            .iter()
            .find(|c| member(c, "model").as_str() == Some("tiny-2-servable.d"))
            .unwrap();
        hex(member(c, "hex").as_str().unwrap())
    });
    ClaimList::open(bytes, &DecodeLimits::MESSAGE).unwrap()
}
struct Model {
    graph: Graph,
    policy: GadgetPolicy,
}
fn model(case: Option<(&Json, u8)>) -> Model {
    let mut c = base();
    let a = &mut c.arch;
    a.d_model = 2;
    a.n_heads = 1;
    a.n_kv_heads = 1;
    a.d_head = 2;
    a.d_ff = 6;
    a.vocab = 5;
    a.max_context = 8;
    let q = &mut c.quant;
    q.residual_bits = 8;
    q.residual_exponent = q
        .act_exponents
        .iter()
        .find(|(p, _)| *p == QuantPoint::Embed)
        .unwrap()
        .1;
    q.rsqrt_kappa = 0;
    q.exp_sigma = 3;
    for (_, s) in &mut q.shift_class_sets {
        *s = vec![12];
    }
    for (_, s) in &mut q.score_scales {
        *s = Rational::new(1, 1).unwrap();
    }
    c.gadget_policy.max_log2_rows = 32;
    c.gadget_policy.limb_bits = 16;
    let mut n = 3;
    if let Some((v, family)) = case {
        n = num(member(v, "n")) as u32;
        match family {
            0 => {
                c.arch.vocab = 8;
                c.gadget_policy.max_log2_rows = num(member(v, "maximum")) as u8;
                c.gadget_policy.limb_bits = 8;
                let shifts: Vec<_> = list(v, "shifts").iter().map(|v| num(v) as i8).collect();
                for (p, s) in &mut q.shift_class_sets {
                    if matches!(p,QuantPoint::Layer{site,..}if matches!(site,Site::RopeQ|Site::RopeK|Site::V|Site::AttnResidual|Site::Silu|Site::Up|Site::MlpResidual))
                    {
                        *s = shifts.clone();
                    }
                }
            }
            1 => {
                c.arch.vocab = 8;
                c.arch.d_model = num(member(v, "dim")) as u32;
                c.arch.d_head = c.arch.d_model;
                q.rsqrt_kappa = num(member(v, "kappa")) as u8;
                q.rms_eps_int = num(member(v, "epsilon"));
            }
            2 => {
                let h = num(member(v, "heads")) as u32;
                c.arch.vocab = 8;
                c.arch.d_model = 2 * h;
                c.arch.n_heads = h;
                c.arch.n_kv_heads = h;
                q.exp_sigma = num(member(v, "sigma")) as u8;
                c.gadget_policy.limb_bits = num(member(v, "limb")) as u8;
                for (p, s) in &mut q.shift_class_sets {
                    if matches!(p,QuantPoint::Layer{site,..}if *site!=Site::Silu) {
                        *s = vec![20];
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    let public = validation::check_public(&c).unwrap();
    Model {
        graph: lowering::lower(&public, n).unwrap(),
        policy: c.gadget_policy,
    }
}
struct Snapshot {
    data: Json,
    models: Vec<Model>,
}
impl Snapshot {
    fn load(file: &str, family: u8) -> Self {
        let data = read_vectors(file);
        let models = list(&data, "cases")
            .iter()
            .map(|v| model(Some((v, family))))
            .collect();
        Self { data, models }
    }
}
fn requants() -> &'static Snapshot {
    static S: OnceLock<Snapshot> = OnceLock::new();
    S.get_or_init(|| Snapshot::load("gadget-finals/requant.json", 0))
}
fn roots() -> &'static Snapshot {
    static S: OnceLock<Snapshot> = OnceLock::new();
    S.get_or_init(|| Snapshot::load("gadget-finals/roots.json", 1))
}
fn softmaxes() -> &'static Snapshot {
    static S: OnceLock<Snapshot> = OnceLock::new();
    S.get_or_init(|| Snapshot::load("gadget-finals/softmax.json", 2))
}
fn small() -> &'static Model {
    static M: OnceLock<Model> = OnceLock::new();
    M.get_or_init(|| model(None))
}
fn source(v: &Json) -> Source {
    let a = v.as_array().unwrap();
    if a[0].as_str() == Some("tensor") {
        return Source::Tensor(TensorId(num(&a[1]) as u32));
    }
    let node = NodeId(num(&a[1]) as u32);
    let name = a[2].as_str().unwrap();
    let j = a.get(3).map_or(0, |v| num(v) as u32);
    let aux = match name {
        "requant.low" => Auxiliary::RequantLow,
        "requant.middle" => Auxiliary::RequantMiddle,
        "requant.saturation" => Auxiliary::RequantSaturation,
        "root.input" => Auxiliary::RootInput(j),
        "root.low" => Auxiliary::RootLow(j),
        _ => {
            let (side, name) = if let Some(n) = name.strip_prefix("root.lower.") {
                (Side::Lower, n)
            } else {
                (Side::Upper, name.strip_prefix("root.upper.").unwrap())
            };
            let role = match name {
                "left.digit" => ComparisonRole::LeftDigit,
                "left.carry" => ComparisonRole::LeftCarry,
                "right.digit" => ComparisonRole::RightDigit,
                "right.carry" => ComparisonRole::RightCarry,
                "difference" => ComparisonRole::Difference,
                "borrow" => ComparisonRole::Borrow,
                _ => unreachable!(),
            };
            Auxiliary::RootComparison(side, role, j)
        }
    };
    Source::Auxiliary(node, aux)
}
// Each nonzero single-field change violates an affine binding of the frozen valid case.
fn tweak(
    s: &mut Script<'_>,
    p: &mut Vec<Fp2>,
    v: &mut Vec<Fp2>,
    keys: &mut Vec<Fp2>,
    values: &mut Vec<Fp2>,
) -> u8 {
    let mode = s.u8() % 8;
    let delta = nonzero(s);
    match mode {
        1 => {
            let i = s.below(v.len());
            v[i] += delta;
        }
        2 => {
            let i = s.below(keys.len());
            keys[i] += delta;
        }
        3 => {
            let i = s.below(values.len());
            values[i] += delta;
        }
        4 => {
            v.pop();
        }
        5 => {
            keys.push(Fp2::ZERO);
            values.push(Fp2::ZERO);
        }
        6 => p.push(Fp2::ONE),
        7 => {
            v.clear();
            keys.clear();
            values.clear();
        }
        _ => (),
    };
    mode
}
fn judge(error: GadgetError, mode: u8, gid: GadgetId) {
    assert_ne!(mode, 0);
    assert_eq!(
        error,
        if mode == 6 {
            GadgetError::Internal
        } else {
            rejected(gid)
        }
    );
}
pub(super) fn requant(s: &mut Script<'_>) {
    let f = requants();
    let ci = s.below(f.models.len());
    let case = &list(&f.data, "cases")[ci];
    let m = &f.models[ci];
    let plan = Plan::derive(&m.graph, m.policy).unwrap();
    let cases = list(case, "sites");
    let i = s.below(cases.len());
    let expected = &cases[i];
    let site = plan.requantizations().nth(i).unwrap().unwrap();
    let gid = GadgetId(num(member(expected, "gadget")) as u32);
    assert_eq!(site.gadget(), gid);
    assert_eq!(site.node(), NodeId(num(member(expected, "node")) as u32));
    let mut p = fields(member(expected, "point"));
    let mut v = fields(member(expected, "evaluations"));
    let mut keys = fields(member(expected, "key_evals"));
    let mut values = fields(member(expected, "value_evals"));
    let mode = tweak(s, &mut p, &mut v, &mut keys, &mut values);
    let pairs: Vec<_> = keys.iter().copied().zip(values.iter().copied()).collect();
    match site.check(&p, &v, &pairs) {
        Err(e) => judge(e, mode, site.gadget()),
        Ok(checked) => {
            assert_eq!(mode, 0);
            let claims: Vec<_> = checked.claims().collect();
            assert_eq!(claims.len(), v.len());
            for (i, c) in claims.iter().enumerate() {
                assert_eq!(c.input.source, source(&list(expected, "sources")[i]));
                assert_eq!(c.value, v[i]);
                let source = source(&list(expected, "sources")[i]);
                let dims: Vec<u64> = match source {
                    Source::Tensor(t) => m
                        .graph
                        .tensor(t)
                        .unwrap()
                        .shape
                        .dims()
                        .iter()
                        .map(|&d| u64::from(d))
                        .collect(),
                    Source::Auxiliary(_, _) => list(expected, "shape").iter().map(num).collect(),
                    Source::Multiplicity(_) => unreachable!(),
                };
                let variables: usize = dims
                    .iter()
                    .map(|&n| (u64::BITS - (n - 1).leading_zeros()) as usize)
                    .sum();
                assert_eq!(c.input.point, &p[..variables]);
            }
            let lookups: Vec<_> = checked.lookups().collect();
            assert_eq!(lookups.len(), pairs.len());
            for (i, l) in lookups.iter().enumerate() {
                assert_eq!(
                    l.table.0,
                    u32::try_from(num(&list(expected, "tables")[i])).unwrap()
                );
                assert_eq!(l.point, p);
                assert_eq!((l.key, l.value), pairs[i]);
            }
        }
    }
}
pub(super) fn root(s: &mut Script<'_>) {
    let f = roots();
    let ci = s.below(f.models.len());
    let case = &list(&f.data, "cases")[ci];
    let m = &f.models[ci];
    let plan = Plan::derive(&m.graph, m.policy).unwrap();
    let cases = list(case, "roots");
    let i = s.below(cases.len());
    let expected = &cases[i];
    let site = plan.roots().nth(i).unwrap().unwrap();
    let gid = GadgetId(num(member(expected, "gadget")) as u32);
    assert_eq!(site.gadget(), gid);
    assert_eq!(site.node(), NodeId(num(member(expected, "node")) as u32));
    assert_eq!(
        site.range_bits(),
        list(expected, "range_bits")
            .iter()
            .map(|v| num(v) as u8)
            .collect::<Vec<_>>()
    );
    let mut p = fields(member(expected, "point"));
    let mut v = fields(member(expected, "values"));
    let mut keys = fields(member(expected, "keys"));
    let mut values = vec![Fp2::ZERO; keys.len()];
    let mode = tweak(s, &mut p, &mut v, &mut keys, &mut values);
    let checked = match site.check(&p, &v, &keys, &values) {
        Err(e) => {
            judge(e, mode, site.gadget());
            return;
        }
        Ok(c) => {
            assert_eq!(mode, 0);
            c
        }
    };
    let claims: Vec<_> = checked.claims().collect();
    assert_eq!(claims.len(), v.len());
    for (i, c) in claims.iter().enumerate() {
        assert_eq!(c.input.source, source(&list(expected, "sources")[i]));
        assert_eq!(c.input.point, p);
        assert_eq!(c.value, v[i]);
    }
    let lookups: Vec<_> = checked.lookups().collect();
    assert_eq!(lookups.len(), keys.len());
    for (i, l) in lookups.iter().enumerate() {
        assert_eq!(l.table.0, num(&list(expected, "tables")[i]) as u32);
        assert_eq!(l.point, p);
        assert_eq!(l.key, keys[i]);
        assert_eq!(l.value, Fp2::ZERO);
    }
    let kind = s.u8() % 3;
    if kind != 0 {
        let residual = kind == 2;
        let vars = if residual {
            site.residual_padding_instance().vars
        } else {
            p.len()
        };
        let mut zero = point(s, vars);
        let mut at = point(s, vars);
        let mut operands = point(s, 1);
        reshape(&mut operands, s.u8(), Fp2::ZERO);
        match s.u8() % 8 {
            1 => at.push(Fp2::ONE),
            2 => zero.push(Fp2::ONE),
            _ => (),
        }
        let result = if residual {
            site.residual_padding(&zero, &at, &operands)
        } else {
            site.padding(&zero, &at, &operands)
        };
        if zero.len() != vars || at.len() != vars {
            assert!(matches!(result, Err(GadgetError::Internal)));
            return;
        }
        if operands.len() != 1 {
            assert!(matches!(result, Err(e) if e == rejected(site.gadget())));
            return;
        }
        let final_value = result.unwrap();
        let features = vars - p.len();
        let psi = if residual {
            less(&at[..features], num(member(case, "dim")) as usize)
                * less(&at[features..], num(member(case, "n")) as usize)
        } else {
            less(&at, num(member(case, "n")) as usize)
        };
        assert_eq!(
            final_value.value(),
            equality(&zero, &at) * (Fp2::ONE - psi) * operands[0]
        );
        let c = final_value.claim();
        assert_eq!(c.input.point, at);
        assert_eq!(c.value, operands[0]);
        assert_eq!(
            c.input.source,
            if residual {
                Source::Tensor(TensorId(num(member(expected, "residual_tensor")) as u32))
            } else {
                source(&list(expected, "sources")[1])
            }
        );
        return;
    }
    let width = num(member(expected, "width")) as u8;
    assert_eq!(
        checked.products().count(),
        list(expected, "product_targets").len()
    );
    for (product, expected) in checked.products().zip(list(expected, "product_targets")) {
        assert_eq!(product.instance().claim, fp(expected));
    }
    let count = usize::from(width).div_ceil(8);
    let coefficients = (2 * usize::from(width) + 62) / 8;
    let i = s.below(2 * coefficients);
    let product_case = checked.products().nth(i).unwrap();
    let mut final_point = point(s, p.len());
    let mut inputs = point(s, count + 4);
    reshape(&mut inputs, s.u8(), Fp2::ZERO);
    if s.u8() % 8 == 1 {
        final_point.push(Fp2::ONE);
    }
    let result = product_case.evaluate(&final_point, &inputs);
    if final_point.len() != p.len() {
        assert!(matches!(result, Err(GadgetError::Internal)));
        return;
    }
    if inputs.len() != count + 4 {
        assert!(matches!(result,Err(e)if e==rejected(site.gadget())));
        return;
    }
    let final_value = result.unwrap();
    assert_eq!(
        final_value.value(),
        equality(&p, &final_point) * product(width, i % coefficients, i >= coefficients, &inputs)
    );
    let claims: Vec<_> = final_value.claims().collect();
    assert_eq!(claims.len(), count + 4);
    let rho = inputs[..count]
        .iter()
        .rev()
        .fold(Fp2::ZERO, |a, &x| a * field(256) + x);
    let u = inputs[count..]
        .iter()
        .rev()
        .fold(Fp2::ZERO, |a, &x| a * field(65536) + x);
    let sq = u - field(num(member(case, "epsilon")))
        * less(&final_point, num(member(case, "n")) as usize);
    let expected_values: Vec<_> = [sq, rho]
        .into_iter()
        .chain(inputs[count..count + 3].iter().copied())
        .chain(inputs[..count - 1].iter().copied())
        .collect();
    for (j, c) in claims.iter().enumerate() {
        assert_eq!(c.input.source, source(&list(expected, "sources")[j]));
        assert_eq!(c.input.point, final_point);
        assert_eq!(c.value, expected_values[j]);
    }
}
fn causal(p: &[Fp2], n: usize, heads: usize) -> Fp2 {
    let t = (usize::BITS - (n - 1).leading_zeros()) as usize;
    let padded = 1usize << t;
    let mut sum = Fp2::ZERO;
    for h in 0..heads {
        for q in 0..n {
            for i in 0..=q {
                sum += eq_index(p, (h * padded + q) * padded + i);
            }
        }
    }
    sum
}
fn rows(p: &[Fp2], n: usize, heads: usize) -> Fp2 {
    let t = (usize::BITS - (n - 1).leading_zeros()) as usize;
    let padded = 1usize << t;
    (0..heads).fold(Fp2::ZERO, |a, h| {
        a + (0..n).fold(Fp2::ZERO, |a, q| a + eq_index(p, h * padded + q))
    })
}
pub(super) fn softmax(s: &mut Script<'_>) {
    let f = softmaxes();
    let ci = s.below(f.models.len());
    let expected = &list(&f.data, "cases")[ci];
    let m = &f.models[ci];
    let plan = Plan::derive(&m.graph, m.policy).unwrap();
    let gid = GadgetId(num(member(expected, "gadget")) as u32);
    let site = plan
        .softmaxes()
        .map(Result::unwrap)
        .find(|x| x.gadget() == gid)
        .unwrap();
    let mut p = fields(member(expected, "row_target"));
    let mut v = fields(member(expected, "source_values"));
    let mut keys = fields(member(expected, "range_keys"));
    let mut values = vec![Fp2::ZERO; keys.len()];
    let mode = tweak(s, &mut p, &mut v, &mut keys, &mut values);
    let checked = match site.check(&p, &v, &keys, &values) {
        Err(e) => {
            judge(e, mode, gid);
            return;
        }
        Ok(c) => {
            assert_eq!(mode, 0);
            c
        }
    };
    let claims: Vec<_> = checked.claims().collect();
    assert_eq!(claims.len(), v.len());
    let half = field(2).inverse().unwrap();
    let n = num(member(expected, "n")) as usize;
    let heads = num(member(expected, "heads")) as usize;
    let tvars = (usize::BITS - (n - 1).leading_zeros()) as usize;
    let hvars = (usize::BITS - (heads - 1).leading_zeros()) as usize;
    assert_eq!(site.variables(), 2 * tvars + hvars);
    assert_eq!(site.row_variables(), tvars + hvars);
    assert_eq!(site.node(), NodeId(num(member(expected, "node")) as u32));
    assert_eq!(checked.instance().claim, fp(&list(expected, "claims")[2]));
    let ep: Vec<_> = vec![half; tvars]
        .into_iter()
        .chain(p.iter().copied())
        .collect();
    let lows = list(expected, "widths").len() - 1;
    for (i, c) in claims.iter().enumerate() {
        let source = match i {
            0 => Source::Tensor(TensorId(num(member(expected, "reciprocal")) as u32)),
            1 => Source::Tensor(TensorId(num(member(expected, "exponentials")) as u32)),
            _ if i < 2 + lows => {
                Source::Auxiliary(site.node(), Auxiliary::SoftmaxA((i - 2) as u32))
            }
            _ => Source::Auxiliary(site.node(), Auxiliary::SoftmaxB((i - 2 - lows) as u32)),
        };
        assert_eq!(c.input.source, source);
        assert_eq!(c.input.point, if i == 1 { &ep } else { &p });
        assert_eq!(c.value, v[i]);
    }
    let lookups: Vec<_> = checked.lookups().collect();
    assert_eq!(lookups.len(), keys.len());
    for (i, l) in lookups.iter().enumerate() {
        assert_eq!(l.point, p);
        assert_eq!(l.table.0, num(&list(expected, "tables")[i + 1]) as u32);
        assert_eq!(l.key, keys[i]);
        assert_eq!(l.value, Fp2::ZERO);
    }
    let n = num(member(expected, "n")) as usize;
    let heads = num(member(expected, "heads")) as usize;
    let kind = s.u8() % 4;
    let target = point(s, site.variables());
    let keyvalue = element(s);
    let evalvalue = element(s);
    let lookup = site.lookup(&target, keyvalue, evalvalue).unwrap();
    let tree = site.tree(&target, keyvalue).unwrap();
    let mut final_point = point(
        s,
        if kind < 2 {
            site.variables()
        } else {
            site.row_variables()
        },
    );
    let count = if kind == 3 { 1 } else { 2 };
    let mut operands = point(s, count);
    reshape(&mut operands, s.u8(), Fp2::ZERO);
    if s.u8() % 8 == 1 {
        final_point.push(Fp2::ONE);
    }
    let correct = if kind < 2 {
        site.variables()
    } else {
        site.row_variables()
    };
    if final_point.len() != correct || operands.len() != count {
        let expected_error = if final_point.len() != correct {
            GadgetError::Internal
        } else {
            rejected(gid)
        };
        match kind {
            0 | 1 => assert!(
                matches!((if kind==0{&lookup}else{&tree}).evaluate(&final_point,&operands),Err(e)if e==expected_error)
            ),
            2 => assert!(
                matches!(checked.evaluate(&final_point,&operands),Err(e)if e==expected_error)
            ),
            _ => {
                let zero = point(s, correct);
                assert!(
                    matches!(site.padding(Padding::Maximum,&zero,&final_point,&operands),Err(e)if e==expected_error)
                );
            }
        }
        return;
    }
    match kind {
        0 | 1 => {
            let key = if kind == 0 { &lookup } else { &tree };
            let mask = causal(&target, n, heads);
            let constant = if kind == 0 {
                field((1u64 << num(member(expected, "key_bits"))) - 1)
            } else {
                Fp2::ONE
            };
            assert_eq!(
                key.instance().claim,
                keyvalue - constant * (Fp2::ONE - mask)
            );
            if kind == 0 {
                let input = key.input_claim().unwrap();
                assert_eq!(
                    input.input.source,
                    Source::Tensor(TensorId(num(member(expected, "exponentials")) as u32))
                );
                assert_eq!(input.input.point, target);
                assert_eq!(input.value, evalvalue);
                let l = key.lookup().unwrap();
                assert_eq!(l.table.0, num(&list(expected, "tables")[0]) as u32);
                assert_eq!(l.point, target);
                assert_eq!((l.key, l.value), (keyvalue, evalvalue));
            } else {
                assert!(key.input_claim().is_none());
                assert!(key.lookup().is_none());
            }
            let result = key.evaluate(&final_point, &operands).unwrap();
            assert_eq!(
                result.value(),
                equality(&target, &final_point)
                    * causal(&final_point, n, heads)
                    * (operands[0] - operands[1])
            );
            let cs: Vec<_> = result.claims().collect();
            assert_eq!(cs.len(), 2);
            assert_eq!(
                cs[0].input.source,
                Source::Auxiliary(site.node(), Auxiliary::SoftmaxMaximum)
            );
            assert_eq!(cs[0].input.point, &final_point[tvars..]);
            assert_eq!(
                cs[1].input.source,
                Source::Tensor(TensorId(num(member(expected, "scores")) as u32))
            );
            assert_eq!(cs[1].input.point, final_point);
            for (i, c) in cs.iter().enumerate() {
                assert_eq!(c.value, operands[i]);
            }
        }
        2 => {
            let result = checked.evaluate(&final_point, &operands).unwrap();
            assert_eq!(
                result.value(),
                equality(&p, &final_point) * operands[0] * field(1u64 << tvars) * operands[1]
            );
            let cs: Vec<_> = result.claims().collect();
            assert_eq!(cs.len(), 2);
            assert_eq!(
                cs[0].input.source,
                Source::Tensor(TensorId(num(member(expected, "reciprocal")) as u32))
            );
            assert_eq!(cs[0].input.point, final_point);
            let ep: Vec<_> = vec![half; tvars]
                .into_iter()
                .chain(final_point.iter().copied())
                .collect();
            assert_eq!(cs[1].input.point, ep);
            assert_eq!(
                cs[1].input.source,
                Source::Tensor(TensorId(num(member(expected, "exponentials")) as u32))
            );
            for (i, c) in cs.iter().enumerate() {
                assert_eq!(c.value, operands[i]);
            }
        }
        _ => {
            let z = point(s, site.row_variables());
            let kind = if s.u8() & 1 == 0 {
                Padding::Maximum
            } else {
                Padding::Reciprocal
            };
            let result = site.padding(kind, &z, &final_point, &operands).unwrap();
            assert_eq!(
                result.value(),
                equality(&z, &final_point)
                    * (Fp2::ONE - rows(&final_point, n, heads))
                    * operands[0]
            );
            let c = result.claim();
            assert_eq!(c.input.point, final_point);
            assert_eq!(c.value, operands[0]);
            assert_eq!(
                c.input.source,
                if kind == Padding::Maximum {
                    Source::Auxiliary(site.node(), Auxiliary::SoftmaxMaximum)
                } else {
                    Source::Tensor(TensorId(num(member(expected, "reciprocal")) as u32))
                }
            );
            assert_eq!(
                site.forest_root(&final_point).unwrap(),
                Fp2::ONE - rows(&final_point, n, heads)
            );
        }
    }
}
pub(super) fn onehot(s: &mut Script<'_>) {
    let m = small();
    let plan = Plan::derive(&m.graph, m.policy).unwrap();
    let site = plan.one_hots().next().unwrap().unwrap();
    let row = point(s, 2);
    let token = element(s);
    let at = site.at(&row, token).unwrap();
    let token_claim = at.input_claim();
    assert_eq!(token_claim.input.source, Source::Tensor(TensorId(0)));
    assert_eq!(token_claim.input.point, row);
    assert_eq!(token_claim.value, token);
    let relation = site.relations().nth(usize::from(s.u8() % 4)).unwrap();
    assert_eq!(
        at.instance(relation).claim,
        match relation {
            Relation::Booleanity | Relation::Vocabulary => Fp2::ZERO,
            Relation::RowSum => less(&row, 3),
            Relation::Index => token,
        }
    );
    let zero = point(s, 5);
    let mut p = point(s, 5);
    let mut values = point(s, 1);
    reshape(&mut values, s.u8(), Fp2::ZERO);
    if s.u8() % 8 == 1 {
        p.push(Fp2::ONE);
    }
    let result = at.evaluate(relation, &zero, &p, &values);
    if p.len() != 5 {
        assert!(matches!(result, Err(GadgetError::Internal)));
        return;
    }
    if values.len() != 1 {
        assert!(matches!(result,Err(e)if e==rejected(GadgetId(0))));
        return;
    }
    let checked = result.unwrap();
    let h = values[0];
    let value = match relation {
        Relation::Booleanity => equality(&zero, &p) * h * (Fp2::ONE - h),
        Relation::Vocabulary => equality(&zero, &p) * (Fp2::ONE - less(&p[..3], 5)) * h,
        Relation::RowSum => equality(&row, &p[3..]) * h,
        Relation::Index => {
            let index = (0..8).fold(Fp2::ZERO, |a, v| a + field(v as u64) * eq_index(&p[..3], v));
            equality(&row, &p[3..]) * index * h
        }
    };
    assert_eq!(checked.value(), value);
    let cs: Vec<_> = checked.claims().collect();
    assert_eq!(cs.len(), 1);
    assert_eq!(cs[0].input.source, Source::Tensor(TensorId(2)));
    assert_eq!(cs[0].input.point, p);
    assert_eq!(cs[0].value, h);
}
pub(super) fn outputs(s: &mut Script<'_>) {
    let m = small();
    let plan = Plan::derive(&m.graph, m.policy).unwrap();
    let context = plan.argmax_outputs(GadgetId(76), 1).unwrap();
    let site = context.argmax();
    let role = match s.u8() % 4 {
        0 => Role::Indicator,
        1 => Role::Logits,
        2 => Role::Maximum,
        _ => Role::Suffix,
    };
    let target = point(
        s,
        if role == Role::Maximum {
            site.row_variables()
        } else {
            site.variables()
        },
    );
    let claim = element(s);
    let binding = context.bind(tv_core::gadgets::argmax::Claim {
        role,
        point: &target,
        value: claim,
    });
    if role == Role::Suffix {
        assert!(matches!(binding, Err(GadgetError::Internal)));
        return;
    }
    let view = match binding.unwrap() {
        Binding::Maximum(m) => {
            assert_eq!(role, Role::Maximum);
            assert_eq!(m.gadget, GadgetId(76));
            assert_eq!(m.point, target);
            assert_eq!(m.value, claim);
            return;
        }
        Binding::Graph(v) => v,
    };
    assert_eq!(view.instance().claim, claim);
    assert_eq!(view.instance().vars, 2);
    let mut p = point(s, 2);
    let mut values = point(s, 1);
    reshape(&mut values, s.u8(), Fp2::ZERO);
    if s.u8() % 8 == 1 {
        p.push(Fp2::ONE);
    }
    let result = view.evaluate(&p, &values);
    if p.len() != 2 {
        assert!(matches!(result, Err(GadgetError::Internal)));
        return;
    }
    if values.len() != 1 {
        assert!(matches!(result,Err(e)if e==rejected(GadgetId(76))));
        return;
    }
    let final_value = result.unwrap();
    let vb = 3;
    let offset = usize::from(role == Role::Indicator);
    let kernel = (0..2).fold(Fp2::ZERO, |a, u| {
        a + eq_index(&target[vb..], u) * eq_index(&p, offset + u)
    });
    assert_eq!(final_value.value(), kernel * values[0]);
    let c = final_value.claim();
    assert_eq!(
        c.input.point,
        target[..vb]
            .iter()
            .copied()
            .chain(p.iter().copied())
            .collect::<Vec<_>>()
    );
    assert_eq!(c.value, values[0]);
    let source = if role == Role::Indicator {
        TensorId(2)
    } else {
        *m.graph.nodes().last().unwrap().outputs().first().unwrap()
    };
    assert_eq!(c.input.source, Source::Tensor(source));
}
