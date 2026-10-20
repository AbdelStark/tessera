// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Stream the frozen public micro-models through the compiler, PCS and canonical verifier.
//! Synthetic affine rows test the opening component, not inference or certified parameters.
use serde_json::Value;
use std::{fs, num::NonZeroUsize, path::Path};
use tessera_codec::{Ascii, DecodeLimits, Rational, SecretBuf, envelope, schema};
use tessera_compiler::{
    AdviceSource, AuxColumn, ModelAdviceLayout, PublicModel, analyze, lower, select,
};
use tessera_field::{
    Field, Goldilocks as Fp, GoldilocksExt2 as Fp2, PrimeField, QuadraticExtension,
};
use tessera_ir::{
    ClaimList, CommitmentLayout, GadgetPolicy, IntegerModel, NodeId, ParamSetId, QuantPoint,
    TableId, TensorId, WeightLayoutParams,
};
use tessera_pcs::{
    AdviceClaim, AdviceLayoutParams, AdviceOpening, AdvicePoint, CommitmentParams, MaskSource,
    RowEncoder, StreamingCommitter,
};
use tv_core::advice as tv;

fn list(v: &Value) -> &[Value] {
    v.as_array().unwrap()
}
fn num(v: &Value) -> u64 {
    v.as_u64().unwrap()
}
fn fp(v: &Value) -> Fp {
    Fp::from_canonical_u64(
        u64::from_str_radix(v.as_str().unwrap().trim_start_matches("0x"), 16).unwrap(),
    )
    .unwrap()
}
fn fp2(v: &Value) -> Fp2 {
    Fp2::new(fp(&v[0]), fp(&v[1]))
}
fn point(v: &Value) -> Vec<Fp2> {
    list(v).iter().map(fp2).collect()
}
fn digest(v: &Value) -> [u8; 32] {
    let s = v.as_str().unwrap();
    std::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
}
fn tv_fp(x: Fp) -> tv_field::Fp {
    tv_field::Fp::from_canonical(x.to_canonical_u64()).unwrap()
}
fn tv_fp2(x: Fp2) -> tv_field::Fp2 {
    let [a, b] = x.coeffs();
    tv_field::Fp2::new(tv_fp(a), tv_fp(b))
}
fn sources(v: &Value) -> (AdviceSource, tv::Source) {
    let a = list(v);
    let id = num(&a[1]) as u32;
    match a[0].as_str().unwrap() {
        "tensor" => (
            AdviceSource::Tensor(TensorId(id)),
            tv::Source::Tensor(tv_core::lowering::TensorId(id)),
        ),
        "multiplicity" => (
            AdviceSource::Multiplicity(TableId(id)),
            tv::Source::Multiplicity(tv_tables::TableId(id)),
        ),
        "aux" => {
            let (root, other) = match a[2].as_str().unwrap() {
                "requant.low" => (AuxColumn::Low, tv::Auxiliary::RequantLow),
                "requant.middle" => (AuxColumn::Middle, tv::Auxiliary::RequantMiddle),
                "softmax.maximum" => (AuxColumn::Maximum, tv::Auxiliary::SoftmaxMaximum),
                _ => panic!("a role used by the frozen original claims"),
            };
            (
                AdviceSource::Auxiliary(NodeId(id), root),
                tv::Source::Auxiliary(tv_core::lowering::NodeId(id), other),
            )
        }
        _ => panic!("a frozen source"),
    }
}
struct Masks<I> {
    values: I,
    drawn: usize,
}
impl<I: Iterator<Item = Fp>> MaskSource for Masks<I> {
    fn draw(&mut self) -> Fp {
        self.drawn += 1;
        self.values.next().unwrap()
    }
}
fn masks(layout: &tessera_pcs::AdviceLayout) -> Masks<impl Iterator<Item = Fp> + '_> {
    Masks {
        drawn: 0,
        values: layout
            .segments()
            .iter()
            .flat_map(|s| s.nonzero.iter().cloned().flatten())
            .flat_map(|r| (0..4).map(move |i| Fp::from_u64_reduced((r + 1) * (i + 1) + 7))),
    }
}
fn row(r: u64, powers: &[Fp], out: &mut [Fp]) {
    let c = Fp::from_u64_reduced(17 * r + 5);
    let d = Fp::from_u64_reduced(r * r + 3);
    for (v, &x) in out.iter_mut().zip(powers) {
        *v = c + d * x;
    }
}

#[test]
fn original_model_claims_stream_to_the_frozen_wire_and_canonical_verifier() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors");
    let vectors: Value =
        serde_json::from_str(&fs::read_to_string(dir.join("advice-opening/opening.json")).unwrap())
            .unwrap();
    let model: IntegerModel = envelope::open(
        &fs::read(dir.join("models/tiny-2-servable.d")).unwrap(),
        &schema::INTEGER_MODEL,
        DecodeLimits::INTEGER_MODEL,
    )
    .unwrap();
    let mut arch = model.arch.clone();
    arch.d_model = 4;
    arch.n_heads = 2;
    arch.n_kv_heads = 1;
    arch.d_head = 2;
    arch.d_ff = 6;
    arch.vocab = 8;
    arch.max_context = 8;
    let mut quant = model.quant.public.clone();
    quant.residual_bits = 8;
    quant.residual_exponent = quant
        .act_exponents
        .iter()
        .find(|(p, _)| *p == QuantPoint::Embed)
        .unwrap()
        .1;
    quant.rsqrt_kappa = 0;
    quant.exp_sigma = 3;
    for (_, s) in &mut quant.shift_class_sets {
        *s = vec![12];
    }
    for (_, s) in &mut quant.score_scales {
        *s = Rational::new(1, 1).unwrap();
    }
    let public = PublicModel {
        arch: &arch,
        profile: model.profile,
        quant: &quant,
        silu_in_exponents: &model.quant.silu_in_exponents,
    };
    let policy = GadgetPolicy {
        max_log2_rows: 17,
        limb_bits: 8,
        query_block: 2,
    };
    let maximum = lower(public, 8).unwrap();
    let selection = select(
        &maximum.graph,
        &analyze(&maximum.graph, &maximum.tables).unwrap(),
        &maximum.tables,
        &policy,
    )
    .unwrap();
    for data in list(&vectors["openings"]) {
        let params = AdviceLayoutParams {
            log2_k: 6,
            log2_inv_rate: 1,
            mask_coeffs: 4,
            row_block: num(&data["row_block"]) as u32,
            segments: list(&data["segments"]).len() as u32,
        };
        let claims = ClaimList {
            protocol_version: tessera_ir::PROTOCOL_VERSION,
            arch: arch.clone(),
            profile: model.profile,
            quant: quant.clone(),
            silu_in_exponents: model.quant.silu_in_exponents.clone(),
            tables: selection.tables.clone(),
            gadget_policy: policy,
            layout: CommitmentLayout {
                advice: params,
                weights: WeightLayoutParams {
                    log2_k: 8,
                    log2_inv_rate: 2,
                    mask_coeffs: 64,
                    blinding_rows: 32,
                },
            },
            params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        };
        let lowering = lower(public, num(&data["n"]) as u32).unwrap();
        let mapped = ModelAdviceLayout::new(&lowering, &selection, &claims).unwrap();
        let layout = mapped.layout();
        assert_eq!(layout.allocated_rows(), num(&data["allocated_rows"]));
        let p = CommitmentParams {
            log2_k: 6,
            log2_inv_rate: 1,
            mask_coeffs: 4,
            row_block: NonZeroUsize::new(params.row_block as usize).unwrap(),
            commitment_id: digest(&data["commitment_id"]),
        };
        let encoder = RowEncoder::new(64, 1, 4).unwrap();
        let omega = Fp::two_adic_generator(6);
        let powers: Vec<_> = (0..64).map(|i| omega.pow(i)).collect();
        let mut buffer = SecretBuf::filled(Fp::ZERO, 64);
        let mut rng = masks(layout);
        let mut parts = Vec::new();
        for (s, part) in layout.segments().iter().enumerate() {
            let mut pass =
                StreamingCommitter::new(p, layout, s as u32, &encoder, &mut rng).unwrap();
            for r in part.nonzero.iter().cloned().flatten() {
                row(r, &powers, &mut buffer);
                pass.push_row(r, &buffer).unwrap();
            }
            parts.push(pass.finish().unwrap());
        }
        assert_eq!(rng.drawn, layout.rows() as usize * 4);
        let state = AdviceOpening::new(&parts).unwrap();
        assert_eq!(state.commitment().root, digest(&data["root"]));
        let mut originals = Vec::new();
        let mut tv_points = Vec::new();
        let mut tv_pieces = Vec::new();
        let mut values = Vec::new();
        for claim in list(&data["claims"]) {
            let natural = point(&claim["point"]);
            let mut pieces = Vec::new();
            let mut verifier = Vec::new();
            for p in list(&claim["pieces"]) {
                let (source, tv_source) = sources(&p["source"]);
                let coefficient = fp2(&p["coefficient"]);
                for image in mapped.claim(source, &natural).unwrap() {
                    let image = image.unwrap();
                    pieces.push(AdvicePoint {
                        point: image.point,
                        weight: coefficient * image.weight,
                    });
                }
                verifier.push(tv::opening::Piece {
                    source: tv_source,
                    coefficient: tv_fp2(coefficient),
                });
            }
            originals.push(AdviceClaim { pieces });
            tv_points.push(natural.into_iter().map(tv_fp2).collect::<Vec<_>>());
            tv_pieces.push(verifier);
            values.push(tv_fp2(fp2(&claim["value"])));
        }
        let gamma = fp2(&data["gamma"]);
        let xi = point(&data["xi_point"]);
        let prepared = state.prepare(&originals, gamma, &xi).unwrap();
        assert_eq!(prepared.group_count(), list(&data["groups"]).len());
        let mut rng = masks(layout);
        let mut pass = prepared.begin(&encoder, &mut rng).unwrap();
        for part in layout.segments() {
            for r in part.nonzero.iter().cloned().flatten() {
                row(r, &powers, &mut buffer);
                pass.push_row(r, &buffer).unwrap();
            }
        }
        let checked = pass.finish().unwrap();
        assert_eq!(rng.drawn, layout.rows() as usize * 4);
        for (actual, expected) in checked.combined().iter().zip(list(&data["combined"])) {
            assert_eq!(actual.u, point(&expected["u"]));
            assert_eq!(actual.mask, point(&expected["mask"]));
        }
        let queries: Vec<_> = list(&data["queries"]).iter().map(num).collect();
        let mut rng = masks(layout);
        let mut pass = checked.answer(&encoder, &mut rng, &queries).unwrap();
        for part in layout.segments() {
            for r in part.nonzero.iter().cloned().flatten() {
                row(r, &powers, &mut buffer);
                pass.push_row(r, &buffer).unwrap();
            }
        }
        let answer = pass.finish().unwrap();
        assert_eq!(rng.drawn, layout.rows() as usize * 4);
        for (actual, expected) in answer.columns.iter().zip(list(&data["columns"])) {
            assert_eq!(*actual, list(expected).iter().map(fp).collect::<Vec<_>>());
        }
        assert_eq!(
            answer.proof.siblings,
            list(&data["siblings"])
                .iter()
                .map(digest)
                .collect::<Vec<_>>()
        );
        let encoded = envelope::seal(&schema::CLAIM_LIST, &claims);
        let other =
            tv_core::claims::ClaimList::open(&encoded, &tv_codec::DecodeLimits::MESSAGE).unwrap();
        let public = tv_core::validation::check_public(&other).unwrap();
        let graph = tv_core::lowering::lower(&public, 8).unwrap();
        let selected = tv_core::selection::derive(&graph, other.gadget_policy).unwrap();
        let geometry = tv::Layout::derive(
            &graph,
            &selected,
            tv::Config::new(&other.arch, other.gadget_policy, 6, params.segments),
            num(&data["n"]) as u32,
        )
        .unwrap();
        let vp = tv_core::claims::AdviceLayoutParams {
            log2_k: 6,
            log2_inv_rate: 1,
            mask_coeffs: 4,
            row_block: params.row_block,
            segments: params.segments,
        };
        let context = tv::opening::Commitment::bind(
            &geometry,
            vp,
            &digest(&data["statement_digest"]),
            num(&data["unit"]) as u32,
            state.commitment().root,
            state.commitment().layout_digest,
        )
        .unwrap();
        let views: Vec<_> = tv_points
            .iter()
            .zip(&tv_pieces)
            .zip(values)
            .map(|((point, pieces), value)| tv::opening::Claim {
                point,
                pieces,
                value,
            })
            .collect();
        let xi: Vec<_> = xi.into_iter().map(tv_fp2).collect();
        let opening = context.prepare(&views, tv_fp2(gamma), &xi).unwrap();
        let combined: Vec<_> = checked
            .combined()
            .iter()
            .map(|r| tv_core::opening::CombinedRow {
                u: r.u.iter().copied().map(tv_fp2).collect(),
                mask: r.mask.iter().copied().map(tv_fp2).collect(),
            })
            .collect();
        let columns: Vec<_> = answer
            .columns
            .iter()
            .map(|r| r.iter().copied().map(tv_fp).collect())
            .collect();
        assert_eq!(
            opening.check_evaluations(&combined).unwrap().check_columns(
                &queries,
                &columns,
                &answer.proof.siblings
            ),
            Ok(())
        );
    }
}
