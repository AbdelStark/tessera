// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Complete runtime geometry from the pre-implementation independent Python oracle.
mod claim_support;
use claim_support::{apply_variant, claims_of, items, member, number, text};
use tv_core::advice::{Config, Layout, PointError, Source, TileKey};
use tv_core::gadgets::runtime::Plan;
use tv_core::leaf::split::{self, Column};
use tv_core::validation::{self, ClaimsError, GadgetId};
use tv_core::{lowering, selection};
use tv_field::{Fp, Fp2};
use tv_tables::{Table, TableId, TableKind};
use tv_testkit::{Json, read_vectors};

fn point(n: usize) -> Vec<Fp2> {
    (0..n)
        .map(|i| Fp2::new(Fp::reduce(5 + 7 * i as u64), Fp::reduce(3 + 11 * i as u64)))
        .collect()
}
fn eq(index: u64, point: &[Fp2]) -> Fp2 {
    point.iter().enumerate().fold(Fp2::ONE, |v, (j, &r)| {
        v * if index >> j & 1 == 0 { Fp2::ONE - r } else { r }
    })
}
fn table_key(json: &Json) -> TileKey {
    let v = items(json);
    match v[0].as_str().unwrap() {
        "table" => TileKey::Table(TableId(number(&v[1]))),
        "greedy.maximum" => TileKey::GreedyMaximum(GadgetId(number(&v[1]))),
        other => panic!("unexpected tail key {other}"),
    }
}
#[test]
fn complete_catalog_leaf_counts_maxima_and_original_images_match_literal_geometry() {
    let vectors = read_vectors("greedy-runtime/runtime.json");
    let cases = items(member(&vectors, "cases"));
    assert_eq!(cases.len(), 141);
    let mut cache: Vec<(String, Vec<Table>)> = Vec::new();
    let mut nonempty = 0;
    for case in cases {
        let mut claims = claims_of(text(case, "file"));
        if !matches!(member(case, "variant"), Json::Null) {
            apply_variant(member(case, "variant"), &mut claims);
        }
        if !matches!(member(case, "vocab"), Json::Null) {
            claims.arch.vocab = number(member(case, "vocab"));
        }
        if let custom = member(case, "custom")
            && !matches!(custom, Json::Null)
        {
            claims.arch.d_model = number(member(custom, "d_model"));
            claims.arch.n_heads = number(member(custom, "n_heads"));
            claims.arch.n_kv_heads = number(member(custom, "n_kv_heads"));
            claims.arch.d_head = number(member(custom, "d_head"));
            claims.arch.d_ff = number(member(custom, "d_ff"));
            claims.quant.rsqrt_kappa = number(member(custom, "rsqrt_kappa"));
            claims.quant.exp_sigma = number(member(custom, "exp_sigma"));
            claims.quant.residual_bits = number(member(custom, "residual_bits"));
            claims.quant.residual_exponent = number(member(custom, "residual_exponent"));
        }
        let public = validation::check_public(&claims).unwrap();
        let n = number(member(case, "n"));
        let graph = lowering::lower(&public, n).unwrap();
        let selected = selection::derive(&graph, claims.gadget_policy).unwrap();
        let cache_key = format!(
            "{}/{:?}/{:?}/{:?}",
            text(case, "file"),
            member(case, "variant"),
            member(case, "vocab"),
            member(case, "custom")
        );
        let slot = cache
            .iter()
            .position(|(key, _)| key == &cache_key)
            .unwrap_or_else(|| {
                cache.push((
                    cache_key,
                    selected
                        .tables
                        .iter()
                        .map(|t| Table::generate(t.id, t.kind.clone()).unwrap())
                        .collect(),
                ));
                cache.len() - 1
            });
        let models = &cache[slot].1;
        let plan = Plan::derive(
            &graph,
            claims.gadget_policy,
            number(member(case, "prompt")),
            number(member(case, "ordinal")),
            models,
        )
        .unwrap();
        let expected_tables = items(member(case, "tables"));
        assert_eq!(plan.catalog().len(), expected_tables.len());
        assert_eq!(&plan.catalog()[..models.len()], selected.tables);
        for (found, expected) in plan.catalog().iter().zip(expected_tables) {
            let fields = items(expected);
            assert_eq!(found.id.0, number(&fields[0]));
            assert_eq!(found.log2_rows, number(&fields[2]));
            if let Some(range) = fields[1].get("Range") {
                assert_eq!(
                    found.kind,
                    TableKind::Range {
                        bits: number(member(range, "bits"))
                    }
                );
            }
            let actual = plan.table(found.id).unwrap();
            assert_eq!(actual.spec().id, found.id);
            assert_eq!(actual.spec().kind, found.kind);
            assert_eq!(actual.spec().log2_rows, found.log2_rows);
        }
        assert!(plan.table(TableId(u32::MAX)).is_none());
        let config = Config {
            log_k: number(member(case, "log2_k")),
            segments: number(member(case, "segments")),
            query_block: number(member(case, "query_block")),
            heads: claims.arch.n_heads,
            kv_heads: claims.arch.n_kv_heads,
            limb_bits: claims.gadget_policy.limb_bits,
        };
        let layout = plan.advice_layout(config).unwrap();
        let model = Layout::derive(&graph, &selected, config, n).unwrap();
        assert_eq!(layout.rows(), number(member(case, "rows")));
        assert_eq!(layout.nu(), number(member(case, "nu")));
        let model_tiles: usize = number(member(case, "model_tiles"));
        for (new, old) in layout.tiles().take(model_tiles).zip(model.tiles()) {
            assert_eq!(
                (new.key(), new.rows(), new.columns(), new.segment()),
                (old.key(), old.rows(), old.columns(), old.segment())
            );
            for col in 0..new.columns() {
                let a = new.location(col).unwrap();
                let b = old.location(col).unwrap();
                assert_eq!(
                    (a.source, a.domain, a.base, a.zero_at),
                    (b.source, b.domain, b.base, b.zero_at)
                );
            }
        }
        assert_eq!(
            layout.tiles().nth(model_tiles - 1).unwrap().rows().1,
            number(member(case, "model_end"))
        );
        let tail = items(member(case, "tail"));
        assert_eq!(layout.tiles().count(), model_tiles + tail.len());
        for (tile, expected) in layout.tiles().skip(model_tiles).zip(tail) {
            assert_eq!(tile.key(), table_key(member(expected, "key")));
            assert_eq!(tile.segment(), number(member(expected, "segment")));
            let rows = items(member(expected, "rows"));
            assert_eq!(tile.rows(), (number(&rows[0]), number(&rows[1])));
            let [value] = items(member(expected, "values")) else {
                panic!("one source")
            };
            let [identity, shape, permutation, base, rows, zero] = items(value) else {
                panic!("one location")
            };
            let source = match tile.key() {
                TileKey::Table(id) => Source::Multiplicity(id),
                TileKey::GreedyMaximum(id) => Source::GreedyMaximum(id),
                _ => panic!("runtime tail"),
            };
            assert_eq!(
                match source {
                    Source::GreedyMaximum(id) => id.0,
                    Source::Multiplicity(id) => id.0,
                    _ => panic!("tail source"),
                },
                number(&items(identity)[1])
            );
            let loc = tile.location(0).unwrap();
            assert_eq!(loc.source, source);
            assert_eq!(
                loc.domain.shape(),
                items(shape).iter().map(number).collect::<Vec<u64>>()
            );
            let perm = items(permutation).iter().map(number).collect::<Vec<u8>>();
            assert_eq!(loc.domain.permutation(), perm);
            assert_eq!(
                (loc.base, loc.domain.rows(), loc.zero_at),
                (number(base), number(rows), number(zero))
            );
            assert!(tile.location(1).is_none());
            let p = point(perm.len());
            let images = layout
                .claim(source, &p)
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(images.len(), 1);
            let image = &images[0];
            assert_eq!(image.weight, Fp2::ONE);
            assert_eq!(image.source, source);
            assert_eq!(image.tile, tile.key());
            for natural in [0u64, 1, 0x5555_5555, u64::MAX] {
                let local = perm
                    .iter()
                    .enumerate()
                    .fold(0, |a, (j, &old)| a | ((natural >> old & 1) << j));
                assert_eq!(
                    eq(loc.base << config.log_k | local, &image.point),
                    eq(natural, &p)
                );
            }
        }
        for (i, e) in items(member(case, "segment_rows")).iter().enumerate() {
            let r = items(e);
            assert_eq!(
                layout.segment(i as u32),
                Some((number(&r[0]), number(&r[1])))
            );
        }
        let leaf = plan.leaf_schedule().unwrap();
        let expected = items(member(case, "lookups"));
        assert_eq!(
            leaf.lookups(),
            expected
                .iter()
                .map(|e| {
                    let p = items(e);
                    (TableId(number(&p[0])), number(&p[1]))
                })
                .collect::<Vec<_>>()
        );
        let composed = tv_core::gadgets::model::Plan::derive_runtime(&plan).unwrap();
        let base = tv_core::gadgets::model::Plan::derive(plan.gadgets()).unwrap();
        let extras = if plan.certificate().is_some() { 3 } else { 2 };
        assert_eq!(composed.native().len(), base.native().len() + extras);
        assert_eq!(&composed.native()[..base.native().len()], base.native());
        let native = items(member(case, "native"));
        assert_eq!(native.len(), extras);
        for (offset, (actual, expected)) in composed.native()[base.native().len()..]
            .iter()
            .zip(native)
            .enumerate()
        {
            assert_eq!(actual.tag as usize, base.native().len() + offset + 1);
            assert_eq!(actual.node.0, number(member(expected, "owner")));
            assert_eq!(actual.vars, number(member(expected, "variables")));
            assert_eq!(actual.degree, number(member(expected, "degree")));
            assert_eq!(actual.evaluations, number(member(expected, "evaluations")));
            assert_eq!(
                actual.zero,
                matches!(member(expected, "zero"), Json::Bool(true))
            );
            assert_eq!(
                actual.outputs.len(),
                items(member(expected, "outputs")).len()
            );
            for (projection, expected) in actual
                .outputs
                .iter()
                .zip(items(member(expected, "outputs")))
            {
                let e = items(expected);
                assert_eq!(projection.tensor.0, number(&e[0]));
                assert_eq!(projection.from, number(&e[1]));
                assert_eq!(projection.vars, number(&e[2]));
            }
        }
        assert!(
            composed
                .schedule()
                .leaf()
                .iter()
                .all(|l| l.input.is_some() && !(l.site.0 == 0 && l.input == Some(0)))
        );
        let count = |model: &tv_core::gadgets::model::Plan<'_>| {
            model.schedule().stages()[0]
                .instances
                .iter()
                .filter(|i| matches!(i.relation, tv_core::schedule::Relation::OneHotIndex { .. }))
                .count()
        };
        assert_eq!(count(&base), 1);
        assert_eq!(count(&composed), 0);
        for kind in [0, 1, 2] {
            let present = composed.schedule().stages()[0].instances.iter().any(|i| {
                matches!(
                    (kind, &i.relation),
                    (0, tv_core::schedule::Relation::Booleanity)
                        | (1, tv_core::schedule::Relation::Vocabulary { .. })
                        | (2, tv_core::schedule::Relation::OneHotCount { .. })
                )
            });
            assert!(present);
        }
        let sites = plan.split_sites().unwrap();
        let split = split::Layout::new(&sites).unwrap();
        assert_eq!(split.instances().collect::<Vec<_>>(), leaf.lookups());
        match (plan.certificate(), member(case, "certificate")) {
            (None, Json::Null) => assert_eq!(layout.rows(), model.rows()),
            (Some(c), expected) => {
                nonempty += 1;
                assert_eq!(c.gadget().0, number(member(expected, "id")));
                assert_eq!(c.table().0, number(member(expected, "range")));
                assert_eq!(c.boundary().range_bits(), number(member(expected, "bits")));
                assert_eq!(c.entries(), number(member(expected, "entries")));
                assert_eq!(1u64 << c.variables(), c.entries());
                let site = sites.last().unwrap();
                assert_eq!(site.layers.len(), 1);
                assert_eq!(site.layers[0].node.0 as usize, graph.nodes().len() - 1);
                let key = &site.layers[0].lookups[0];
                assert_eq!(key.table, c.table());
                assert_eq!(key.variables, c.variables());
                assert_eq!(key.key.terms.len(), 1);
                assert_eq!(key.key.terms[0].0, Fp::ONE);
                assert_eq!(key.key.terms[0].1.column, Column::GreedyKey);
                assert_eq!(
                    key.key.terms[0].1.coordinates,
                    (0..c.variables()).collect::<Vec<_>>()
                );
                assert!(key.value.is_none());
                let wrong = Source::GreedyMaximum(GadgetId(c.gadget().0 + 1));
                assert!(matches!(layout.claim(wrong, &[]), Err(PointError::Source)));
                assert!(matches!(
                    layout.claim(Source::GreedyMaximum(c.gadget()), &[]),
                    Err(PointError::Arity)
                ));
            }
            _ => panic!("certificate presence"),
        }
    }
    assert_eq!(nonempty, 89);
}

#[test]
fn runtime_refuses_invalid_own_requests_ordinals_and_model_catalogs() {
    let c = claims_of("tiny-2-servable.d");
    let public = validation::check_public(&c).unwrap();
    let g = lowering::lower(&public, 4).unwrap();
    let s = selection::derive(&g, c.gadget_policy).unwrap();
    let models = s
        .tables
        .iter()
        .map(|t| Table::generate(t.id, t.kind.clone()).unwrap())
        .collect::<Vec<_>>();
    for prompt in [0, 5, u32::MAX] {
        assert!(matches!(
            Plan::derive(&g, c.gadget_policy, prompt, 0, &models),
            Err(ClaimsError::Internal)
        ));
    }
    assert!(matches!(
        Plan::derive(&g, c.gadget_policy, 1, u32::MAX, &models),
        Err(ClaimsError::Internal)
    ));
    assert!(matches!(
        Plan::derive(&g, c.gadget_policy, 1, 0, &models[..models.len() - 1]),
        Err(ClaimsError::Internal)
    ));
    let mut wrong = models;
    wrong.swap(0, 1);
    assert!(matches!(
        Plan::derive(&g, c.gadget_policy, 1, 0, &wrong),
        Err(ClaimsError::Internal)
    ));
}

#[test]
fn only_the_exact_native_token_descriptor_can_replace_the_weighted_onehot_check() {
    use tv_core::lowering::NodeId;
    use tv_core::schedule::{self, Native, NativeOutput, ScheduleError};
    let c = claims_of("tiny-2-servable.d");
    let public = validation::check_public(&c).unwrap();
    let g = lowering::lower(&public, 4).unwrap();
    let node = &g.nodes()[0];
    let h = node.outputs()[0];
    let token = node.inputs()[0];
    let nv = 9;
    let nt = 2;
    let vars = nv + nt;
    let own = Native {
        tag: 1,
        node: NodeId(0),
        vars,
        degree: 2,
        evaluations: 2,
        zero: true,
        outputs: vec![
            NativeOutput {
                tensor: h,
                from: 0,
                vars,
            },
            NativeOutput {
                tensor: token,
                from: nv,
                vars: nt,
            },
        ],
    };
    let deferred = [(NodeId(0), Some(0))];
    assert!(schedule::derive_composed(&g, &deferred, std::slice::from_ref(&own)).is_ok());
    for change in 0..10 {
        let mut bad = own.clone();
        match change {
            0 => bad.degree = 3,
            1 => bad.evaluations = 3,
            2 => bad.vars -= 1,
            3 => bad.zero = false,
            4 => bad.outputs[0].from = 1,
            5 => bad.outputs[0].tensor = token,
            6 => bad.outputs[1].from = 0,
            7 => bad.outputs[1].vars -= 1,
            8 => bad.outputs.swap(0, 1),
            9 => {
                bad.outputs.pop();
            }
            _ => unreachable!(),
        }
        assert!(
            matches!(
                schedule::derive_composed(&g, &deferred, &[bad]),
                Err(ScheduleError::Internal)
            ),
            "change {change}"
        );
    }
}
