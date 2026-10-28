// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Callback algebra only: direct frames deliberately omit fixed-model binding.
//! Literal operands/finals precede producer comparison; full proof tests must bind the frame.
use super::*;
use crate::claims::{ClaimList, QuantPoint};
use tv_testkit::{Json, hex, read_vectors};

fn member<'a>(j: &'a Json, k: &str) -> &'a Json {
    j.get(k).unwrap()
}
fn values(j: &Json) -> &[Json] {
    j.as_array().unwrap()
}
fn number(j: &Json) -> u64 {
    j.as_u64()
        .unwrap_or_else(|| j.as_str().unwrap().parse().unwrap())
}
fn field(j: &Json) -> Fp2 {
    let v = values(j);
    Fp2::new(
        Fp::from_canonical(number(&v[0])).unwrap(),
        Fp::from_canonical(number(&v[1])).unwrap(),
    )
}
fn fields(j: &Json) -> Vec<Fp2> {
    values(j).iter().map(field).collect()
}
fn claims(n: u32, vocab: u32) -> ClaimList {
    let file = read_vectors("claims/claim-lists.json");
    let c = values(member(&file, "cases"))
        .iter()
        .find(|c| member(c, "model").as_str() == Some("tiny-2-servable.d"))
        .unwrap();
    let mut c = ClaimList::open(
        &hex(member(c, "hex").as_str().unwrap()),
        &tv_codec::DecodeLimits::MESSAGE,
    )
    .unwrap();
    c.arch.d_model = 2;
    c.arch.n_heads = 1;
    c.arch.n_kv_heads = 1;
    c.arch.d_head = 2;
    c.arch.d_ff = 6;
    c.arch.vocab = vocab;
    c.arch.max_context = n.max(8);
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
    c
}
#[test]
fn runtime_callbacks_match_literal_finals_and_retain_every_original_obligation() {
    let fixtures = read_vectors("greedy-boundary/greedy-boundary.json");
    for case in values(member(&fixtures, "cases")) {
        let c = claims(
            number(member(case, "n")) as u32,
            number(member(case, "vocab")) as u32,
        );
        let public = crate::validation::check_public(&c).unwrap();
        let graph = crate::lowering::lower(&public, number(member(case, "n")) as u32).unwrap();
        let selected = crate::selection::derive(&graph, c.gadget_policy).unwrap();
        let tables = selected
            .tables
            .iter()
            .map(|t| tv_tables::Table::generate(t.id, t.kind.clone()).unwrap())
            .collect::<Vec<_>>();
        let runtime = runtime::Plan::derive(
            &graph,
            c.gadget_policy,
            number(member(case, "prompt")) as u32,
            0,
            &tables,
        )
        .unwrap();
        let certificate = runtime.certificate().unwrap();
        let plan = Plan::derive_runtime(&runtime).unwrap();
        let node = NodeId(graph.nodes().len() as u32 - 1);
        let mut target = fields(member(case, "target_v"));
        target.extend(fields(member(case, "target_o")));
        let bound = split::Bound {
            columns: vec![split::ColumnClaim {
                node,
                column: split::Column::GreedyKey,
                point: target,
                value: field(member(case, "key_claim")),
            }],
            multiplicities: vec![],
            forests: vec![],
            atoms: vec![],
            point: split::LeafPoint::Shared(vec![]),
        };
        let mut columns = vec![None; graph.nodes().len()];
        columns[node.0 as usize] = Some((0, 1));
        let prepared = Prepared {
            plan: &plan,
            bound: &bound,
            data: (0..graph.nodes().len()).map(|_| None).collect(),
            columns,
            leaf: vec![],
            original: vec![],
            external: None,
        };
        let nt = bits(graph.length()).unwrap();
        let mut zero = vec![Fp2::ONE; plan.schedule.stages()[0].zero_vars];
        let start = zero.len() - nt;
        zero[start..].copy_from_slice(&fields(member(case, "zero")));
        let layer = vec![Fp2::ONE; plan.schedule.layer_vars()];
        let ctx = NativeContext {
            stage: 1,
            layer: &layer,
            zero: &zero,
        };
        let point = fields(member(case, "point"));
        for (relation, raw, expected) in [
            (
                Relation::GreedyKey,
                fields(member(case, "raw")),
                field(member(case, "key_final")),
            ),
            (
                Relation::GreedySelected,
                fields(member(case, "raw")),
                field(member(case, "selected_final")),
            ),
            (
                Relation::Tokens,
                fields(member(case, "token_raw")),
                field(member(case, "token_final")),
            ),
        ] {
            let i = plan
                .slots
                .iter()
                .position(|s| {
                    std::mem::discriminant(&s.relation) == std::mem::discriminant(&relation)
                })
                .unwrap();
            let tag = plan.native[i].tag;
            let instance = prepared.instance(tag, ctx).unwrap();
            assert_eq!(instance.vars, point.len());
            assert_eq!(instance.evals, raw.len());
            assert_eq!(
                instance.claim,
                if matches!(relation, Relation::GreedyKey) {
                    field(member(case, "key_claim"))
                } else {
                    Fp2::ZERO
                }
            );
            let final_value = prepared.component(tag, ctx, &point, &raw).unwrap();
            assert_eq!(final_value.value, expected);
            let (final_value, remaining) = prepared.transform(tag, &point, final_value).unwrap();
            assert_eq!(final_value.value, expected);
            if matches!(relation, Relation::Tokens) {
                assert_eq!(final_value.graph, raw);
                assert!(remaining.is_empty());
            } else {
                assert_eq!(final_value.graph, vec![raw[0]]);
                assert_eq!(remaining.len(), raw.len() - 1);
                assert_eq!(
                    remaining[0].sources,
                    vec![Source::GreedyMaximum(certificate.gadget())]
                );
                assert_eq!(remaining[0].point, point[point.len() - nt..]);
                assert_eq!(remaining[0].value, raw[1]);
                for (got, want) in remaining[1..]
                    .iter()
                    .zip(values(member(case, "original_onehot")))
                {
                    assert_eq!(
                        got.sources,
                        vec![Source::Tensor(graph.nodes()[0].outputs()[0])]
                    );
                    assert_eq!(got.point, fields(member(want, "point")));
                    assert_eq!(got.value, field(member(want, "value")));
                }
            }
        }
    }
}
#[test]
fn empty_output_observation_still_routes_the_actual_lm_head_value() {
    let c = claims(1, 3);
    let public = crate::validation::check_public(&c).unwrap();
    let graph = crate::lowering::lower(&public, 1).unwrap();
    let selected = crate::selection::derive(&graph, c.gadget_policy).unwrap();
    let tables = selected
        .tables
        .iter()
        .map(|t| tv_tables::Table::generate(t.id, t.kind.clone()).unwrap())
        .collect::<Vec<_>>();
    let runtime = runtime::Plan::derive(&graph, c.gadget_policy, 1, 0, &tables).unwrap();
    let plan = Plan::derive_runtime(&runtime).unwrap();
    let bound = split::Bound {
        columns: vec![],
        multiplicities: vec![],
        forests: vec![],
        atoms: vec![],
        point: split::LeafPoint::Shared(vec![]),
    };
    let prepared = Prepared {
        plan: &plan,
        bound: &bound,
        data: vec![],
        columns: vec![],
        leaf: vec![],
        original: vec![],
        external: None,
    };
    let zero = vec![Fp2::ONE; plan.schedule.stages()[0].zero_vars];
    let layer = vec![Fp2::ONE; plan.schedule.layer_vars()];
    let ctx = NativeContext {
        stage: 1,
        layer: &layer,
        zero: &zero,
    };
    let tokens = plan
        .slots
        .iter()
        .position(|s| matches!(s.relation, Relation::Tokens))
        .unwrap();
    let point = [Fp2::from(Fp::reduce(7)); 2];
    let raw = [Fp2::from(Fp::reduce(2)), Fp2::from(Fp::reduce(5))];
    let checked = prepared
        .component(plan.native[tokens].tag, ctx, &point, &raw)
        .unwrap();
    assert_eq!(
        checked.value,
        Fp2::from(Fp::reduce(42)) - Fp2::from(Fp::reduce(180))
    );
    let (final_value, remaining) = prepared
        .transform(plan.native[tokens].tag, &point, checked)
        .unwrap();
    assert_eq!(final_value.graph, raw);
    assert!(remaining.is_empty());
    let i = plan
        .slots
        .iter()
        .position(|s| matches!(s.relation, Relation::ObserveLogits))
        .unwrap();
    let meta = &plan.native[i];
    let point = vec![Fp2::from(Fp::reduce(7)); meta.vars];
    let raw = [Fp2::from(Fp::reduce(123))];
    let checked = prepared.component(meta.tag, ctx, &point, &raw).unwrap();
    let (final_value, remaining) = prepared.transform(meta.tag, &point, checked).unwrap();
    assert_eq!(final_value.value, Fp2::ZERO);
    assert_eq!(final_value.graph, raw);
    assert!(remaining.is_empty());
    assert_eq!(
        meta.outputs[0].tensor,
        graph.nodes().last().unwrap().outputs()[0]
    );
}

#[test]
fn batched_runtime_registers_each_key_then_one_selected_union() {
    for requests in [vec![(2u32, 3u32), (1, 2)], vec![(1, 0), (2, 2), (1, 1)]] {
        let c = claims(8, 11);
        let public = crate::validation::check_public(&c).unwrap();
        let lengths: Vec<_> = requests.iter().map(|&(p, o)| p + o).collect();
        let graph = crate::lowering::lower_batch(&public, &lengths).unwrap();
        let selected = crate::selection::derive(&graph, c.gadget_policy).unwrap();
        let tables: Vec<_> = selected
            .tables
            .iter()
            .map(|t| tv_tables::Table::generate(t.id, t.kind.clone()).unwrap())
            .collect();
        let runtime =
            runtime::Plan::derive_batch(&graph, c.gadget_policy, &requests, &tables).unwrap();
        let plan = Plan::derive_runtime(&runtime).unwrap();
        let epilogue: Vec<_> = plan
            .slots
            .iter()
            .filter(|s| matches!(s.relation, Relation::GreedyKey | Relation::GreedySelected))
            .collect();
        let count = requests.iter().filter(|&&(_, o)| o > 0).count();
        assert_eq!(epilogue.len(), count + 1);
        for slot in &epilogue[..count] {
            assert!(matches!(slot.relation, Relation::GreedyKey));
        }
        assert!(matches!(epilogue[count].relation, Relation::GreedySelected));
        assert_eq!(plan.windows.len(), count);
        let ordinals: Vec<_> = plan.windows.iter().map(|w| w.ordinal).collect();
        assert_eq!(
            ordinals,
            requests
                .iter()
                .enumerate()
                .filter(|(_, (_, o))| *o > 0)
                .map(|(r, _)| r as u32)
                .collect::<Vec<_>>()
        );
    }
}
