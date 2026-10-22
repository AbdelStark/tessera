// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Real current-model P1/P4/P5, checked by independent canonical PCS components (DEP-2).
//! Header, semantic certificates, verdicts and the protocol driver remain separate.
use super::*;
use std::num::{NonZeroU32, NonZeroUsize};
use tessera_compiler::AdviceSource;
use tessera_pcs as pcs;
use tessera_protocol as protocol;
use tv_core::{advice as va, weights as vw};

// Test-only replayable masks; this protocol/1 fixture makes no zero-knowledge claim.
struct Masks(u64);
impl pcs::MaskSource for Masks {
    fn draw(&mut self) -> G {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        G::from_u64_reduced(self.0)
    }
}
pub struct Committed<'a> {
    p: &'a Public,
    advice: &'a ModelAdviceLayout<'a>,
    source: &'a dyn WitnessSource,
    replay: &'a dyn WitnessSource,
    claims: &'a tv_core::claims::ClaimList,
    advice_state: pcs::AdviceOpening<'a>,
    encoder: pcs::RowEncoder,
    weight_layout: pcs::WeightLayout,
    model_weights: pcs::ModelWeights<'a>,
    weights: pcs::WeightCommitment,
    statement: protocol::PublicStatement,
    digest: [u8; 32],
}
fn ascii(s: &str) -> tv_hash::transcript::Ascii<'_> {
    tv_hash::transcript::Ascii::new(s).unwrap()
}
fn from_e(x: E) -> F {
    F::new(
        G::from_u64_reduced(x.c0.to_canonical()),
        G::from_u64_reduced(x.c1.to_canonical()),
    )
}
fn tv_combined(c: &pcs::CombinedRow) -> tv_core::opening::CombinedRow {
    tv_core::opening::CombinedRow {
        u: es(&c.u),
        mask: es(&c.mask),
    }
}
fn wire(c: &pcs::CombinedRow) -> protocol::CombinedRow {
    protocol::CombinedRow {
        u: c.u.clone(),
        mask: c.mask.clone(),
    }
}
fn source(s: va::Source) -> AdviceSource {
    match s {
        va::Source::Tensor(t) => AdviceSource::Tensor(tessera_ir::TensorId(t.0)),
        va::Source::Multiplicity(t) => AdviceSource::Multiplicity(tessera_ir::TableId(t.0)),
        va::Source::GreedyMaximum(g) => AdviceSource::GreedyMaximum(tessera_ir::GadgetId(g.0)),
        va::Source::Auxiliary(n, a) => {
            use va::{Auxiliary as A, Comparison as C, ComparisonRole as R};
            let role = match a {
                A::RequantLow => AuxColumn::Low,
                A::RequantMiddle => AuxColumn::Middle,
                A::RequantSaturation => AuxColumn::Saturation,
                A::SoftmaxMaximum => AuxColumn::Maximum,
                A::SoftmaxA(j) => AuxColumn::Slack {
                    second: false,
                    limb: j,
                },
                A::SoftmaxB(j) => AuxColumn::Slack {
                    second: true,
                    limb: j,
                },
                A::RootInput(j) => AuxColumn::InputLimb(j),
                A::RootLow(j) => AuxColumn::RootByte(j),
                A::RootComparison(c, r, limb) => AuxColumn::Comparison {
                    upper: c == C::Upper,
                    column: match r {
                        R::LeftDigit => CompareColumn::LeftByte,
                        R::LeftCarry => CompareColumn::LeftCarry,
                        R::RightDigit => CompareColumn::RightByte,
                        R::RightCarry => CompareColumn::RightCarry,
                        R::Difference => CompareColumn::Difference,
                        R::Borrow => CompareColumn::Borrow,
                    },
                    limb,
                },
            };
            AdviceSource::Auxiliary(NodeId(n.0), role)
        }
    }
}
fn rows<Failure: Into<tessera_prover::advice::Error>>(
    p: &Public,
    layout: &ModelAdviceLayout<'_>,
    source: &dyn WitnessSource,
    segment: u32,
    mut sink: impl FnMut(u64, &[G]) -> Result<(), Failure>,
) {
    tessera_prover::advice::visit_rows(&p.lowering, layout, source, segment, |r, v| {
        sink(r, v).map_err(Into::into)
    })
    .unwrap();
}
impl<'a> Committed<'a> {
    pub fn new(
        p: &'a Public,
        model: &'a IntegerModel,
        advice: &'a ModelAdviceLayout<'a>,
        source: &'a dyn WitnessSource,
        replay: &'a dyn WitnessSource,
        claims: &'a tv_core::claims::ClaimList,
    ) -> Self {
        let channels = p
            .lowering
            .classes
            .iter()
            .map(|(w, _)| {
                let c = model
                    .quant
                    .channels
                    .iter()
                    .find(|c| c.ty == w.ty && c.layer == w.layer)
                    .unwrap();
                pcs::ChannelParamsRef {
                    matrix: w.ty.matrix(),
                    layer: w.layer,
                    values: pcs::ChannelValues::Classes(&c.class),
                }
            })
            .chain(p.lowering.multipliers.iter().map(|(w, _)| {
                let c = model
                    .quant
                    .channels
                    .iter()
                    .find(|c| c.ty == w.ty && c.layer == w.layer)
                    .unwrap();
                pcs::ChannelParamsRef {
                    matrix: w.ty.matrix(),
                    layer: w.layer,
                    values: pcs::ChannelValues::Multipliers(&c.multiplier),
                }
            }))
            .collect::<Vec<_>>();
        let weight_layout = pcs::WeightLayout::new(
            p.claims.layout.weights,
            pcs::WeightShapes {
                layers: p.claims.arch.n_layers,
                shapes: p.claims.arch.weight_shapes().unwrap(),
                channels: channels.iter().map(|c| (c.matrix, c.layer)).collect(),
            },
        )
        .unwrap();
        let model_weights = pcs::ModelWeights {
            tensors: model
                .weights
                .iter()
                .map(|w| pcs::WeightTensorRef {
                    matrix: w.ty.matrix(),
                    layer: w.layer,
                    data: &w.data,
                })
                .collect(),
            channels,
        };
        let weights = pcs::WeightCommitment::commit(
            &weight_layout,
            NonZeroU32::new(p.claims.layout.advice.row_block).unwrap(),
            0,
            &model_weights,
        )
        .unwrap();
        let public = tv_codec::decode(
            &tessera_codec::encode(weights.public()),
            &tv_codec::DecodeLimits::MESSAGE,
            tv_codec::WeightCommitmentPublic::decode,
        )
        .unwrap();
        vw::check_public(claims, &public, &weights.weights_root(), 0).unwrap();
        let statement = protocol::PublicStatement {
            claim_list_digest: tessera_hash::hash(
                tessera_hash::Domain::ClaimList,
                &tessera_codec::encode(&p.claims),
            ),
            weights_root: weights.weights_root(),
            weights_epoch: 0,
            manifest_digest: [0; 32],
            batch: protocol::BatchDescriptor {
                requests: vec![protocol::RequestRef {
                    request_id: 0,
                    session_id: [130; 16],
                    prompt_len: p.lowering.graph.tensors[p.lowering.tokens.0 as usize].shape[0],
                    output_len: 0,
                }],
            },
            predictor: protocol::PredictorSpec::Exact {
                decoding: protocol::Decoding::Greedy,
            },
        };
        let digest = statement.digest();
        let layout = p.claims.layout.advice;
        let id = pcs::CommitmentDescriptor::Advice {
            statement_digest: digest,
            unit: 0,
            layout,
            rows: advice.layout().rows(),
        }
        .commitment_id();
        let params = pcs::CommitmentParams {
            log2_k: u32::from(layout.log2_k),
            log2_inv_rate: u32::from(layout.log2_inv_rate),
            mask_coeffs: layout.mask_coeffs as usize,
            row_block: NonZeroUsize::new(layout.row_block as usize).unwrap(),
            commitment_id: id,
        };
        let encoder =
            pcs::RowEncoder::new(1 << layout.log2_k, params.log2_inv_rate, params.mask_coeffs)
                .unwrap();
        let commit = |source| {
            let mut masks = Masks(831);
            let mut parts = Vec::new();
            for segment in 0..layout.segments {
                let mut pass = pcs::StreamingCommitter::new(
                    params,
                    advice.layout(),
                    segment,
                    &encoder,
                    &mut masks,
                )
                .unwrap();
                rows(p, advice, source, segment, |r, v| pass.push_row(r, v));
                parts.push(pass.finish().unwrap());
            }
            pcs::AdviceOpening::new(&parts).unwrap()
        };
        let advice_state = commit(source);
        assert_eq!(advice_state.commitment(), commit(replay).commitment());
        Self {
            p,
            advice,
            source,
            replay,
            claims,
            advice_state,
            encoder,
            weight_layout,
            model_weights,
            weights,
            statement,
            digest,
        }
    }
    pub fn start(&self, mode: Mode) -> (Transcript, tv_hash::transcript::Transcript, [Vec<F>; 10]) {
        let context = SessionContext {
            protocol: Ascii::new("tessera-protocol/1").unwrap(),
            mode,
            params: ParamSetId(Ascii::new("test-small/1").unwrap()),
            statement_digest: self.digest,
            session_id: [130; 16],
        };
        let mut t = Transcript::new(&context);
        let mut v = tv_hash::transcript::Transcript::new(&tv_hash::transcript::SessionContext {
            protocol: ascii("tessera-protocol/1"),
            mode: if mode == Mode::Live {
                tv_hash::transcript::Mode::Live
            } else {
                tv_hash::transcript::Mode::NonInteractive
            },
            params: tv_hash::transcript::ParamSetId(ascii("test-small/1")),
            statement_digest: self.digest,
            session_id: [130; 16],
        });
        let round1 = protocol::Round1 {
            protocol_version: tessera_ir::PROTOCOL_VERSION,
            session_id: [130; 16],
            statement: self.statement.clone(),
            advice: self.advice_state.commitment(),
        };
        t.absorb("round1", &round1);
        v.absorb(ascii("round1"), &tessera_codec::encode(&round1));
        if mode == Mode::Live {
            let seed = protocol::coin_toss(&[13; 32], &[17; 32]);
            t.absorb("seed", &seed);
            v.absorb(ascii("seed"), &seed);
        } else {
            let nonce = t.grind("seed-pow", ParamSet::TEST_SMALL.seed_pow as u8);
            v = v
                .check_grind(
                    ascii("seed-pow"),
                    ParamSet::TEST_SMALL.seed_pow as u8,
                    nonce,
                )
                .unwrap();
            assert_eq!(t.squeeze("seed"), v.squeeze(ascii("seed")));
        }
        let mut xi: [Vec<F>; 10] = core::array::from_fn(|_| Vec::new());
        let vl = vw::derive(self.claims, 0).unwrap();
        for (i, m) in pcs::WeightMatrix::ALL.into_iter().enumerate() {
            if self.weight_layout.matrix(m).rows() > 0 {
                let label = format!("pcs/weights/{m:?}/xi");
                for _ in 0..vl[i].proximity_vars() {
                    let x = t.challenge(&label);
                    assert_eq!(e(x), v.challenge_fp2(ascii(&label)).unwrap());
                    xi[i].push(x);
                }
            }
        }
        (t, v, xi)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn check(
        &self,
        graph: &tv_core::lowering::Graph,
        reduced: tv_core::gadgets::model::Reduced<'_>,
        t: &mut Transcript,
        mut v: tv_hash::transcript::Transcript,
        xi: &[Vec<F>; 10],
        mode: Mode,
    ) {
        let selected = tv_core::selection::derive(graph, self.claims.gadget_policy).unwrap();
        let a = va::Layout::derive(
            graph,
            &selected,
            va::Config::new(
                &self.claims.arch,
                self.claims.gadget_policy,
                self.claims.layout.advice.log2_k,
                self.claims.layout.advice.segments,
            ),
            graph.length(),
        )
        .unwrap();
        let wl = vw::derive(self.claims, 0).unwrap();
        let claims =
            tv_core::model_openings::Claims::derive(graph, self.claims, &a, &wl, reduced).unwrap();
        let original = claims.advice_claims().unwrap();
        let mut cpu = Vec::new();
        for claim in &original {
            let point = claim.point.iter().copied().map(from_e).collect::<Vec<_>>();
            let mut pieces = Vec::new();
            for piece in claim.pieces {
                for image in self.advice.claim(source(piece.source), &point).unwrap() {
                    let image = image.unwrap();
                    pieces.push(pcs::AdvicePoint {
                        point: image.point,
                        weight: from_e(piece.coefficient) * image.weight,
                    });
                }
            }
            cpu.push(pcs::AdviceClaim { pieces });
        }
        let gamma = t.challenge("pcs/advice/gamma");
        assert_eq!(
            e(gamma),
            v.challenge_fp2(ascii("pcs/advice/gamma")).unwrap()
        );
        let av = (0..self.advice.layout().row_vars())
            .map(|_| {
                let x = t.challenge("pcs/advice/xi");
                assert_eq!(e(x), v.challenge_fp2(ascii("pcs/advice/xi")).unwrap());
                x
            })
            .collect::<Vec<_>>();
        let mut theta = [F::ZERO; 10];
        for (i, m) in vw::Matrix::ALL.into_iter().enumerate() {
            if !claims.weight_claims(m, E::ONE).unwrap().is_empty() {
                let label = format!("pcs/weights/{m:?}/gamma");
                theta[i] = t.challenge(&label);
                assert_eq!(e(theta[i]), v.challenge_fp2(ascii(&label)).unwrap());
            }
        }
        let prepared = self.advice_state.prepare(&cpu, gamma, &av).unwrap();
        let combine = |source| {
            let mut masks = Masks(831);
            let mut pass = prepared.begin(&self.encoder, &mut masks).unwrap();
            for s in 0..self.p.claims.layout.advice.segments {
                rows(self.p, self.advice, source, s, |r, x| pass.push_row(r, x));
            }
            pass.finish().unwrap()
        };
        let checked = combine(self.source);
        let replayed = combine(self.replay);
        assert_eq!(checked.combined(), replayed.combined());
        let binding = va::opening::Commitment::bind(
            &a,
            self.claims.layout.advice,
            &self.digest,
            0,
            self.advice_state.commitment().root,
            self.advice_state.commitment().layout_digest,
        )
        .unwrap();
        let av = es(&av);
        let bound = binding.prepare(&original, e(gamma), &av).unwrap();
        assert_eq!(bound.group_count(), prepared.group_count());
        let ac = checked
            .combined()
            .iter()
            .map(tv_combined)
            .collect::<Vec<_>>();
        let attacks =
            graph.length() == 3 && self.claims.arch.n_layers == 2 && mode == Mode::NonInteractive;
        if attacks {
            use tv_core::opening::OpeningError as O;
            for index in 0..original.len() {
                let mut bad = original.clone();
                bad[index].value += E::ONE;
                let changed = binding.prepare(&bad, e(gamma), &av).unwrap();
                assert!(
                    matches!(
                        changed.check_evaluations(&ac),
                        Err(O::Evaluation { group: Some(_) })
                    ),
                    "original advice obligation {index}"
                );
            }
            let mut bad = ac.clone();
            for value in &mut bad[0].u {
                *value += E::ONE;
            }
            assert_eq!(
                bound.check_evaluations(&bad).err(),
                Some(O::Evaluation { group: Some(0) })
            );
            assert_eq!(
                bound.check_evaluations(&ac[..ac.len() - 1]).err(),
                Some(O::Evaluation { group: None })
            );
        }
        let advice_evaluated = bound.check_evaluations(&ac).unwrap();
        let advice_wire = checked.combined().iter().map(wire).collect::<Vec<_>>();
        t.absorb("pcs/advice/combined", &advice_wire);
        v.absorb(
            ascii("pcs/advice/combined"),
            &tessera_codec::encode(&advice_wire),
        );
        let mut wc: Vec<(
            usize,
            Vec<vw::Claim<'_>>,
            Vec<tv_core::opening::CombinedRow>,
        )> = Vec::new();
        for (i, m) in pcs::WeightMatrix::ALL.into_iter().enumerate() {
            let original = claims
                .weight_claims(vw::Matrix::ALL[i], e(theta[i]))
                .unwrap();
            if original.is_empty() {
                continue;
            }
            let mut reader = self.weight_layout.rows(m, &self.model_weights).unwrap();
            let rowpoints = original
                .iter()
                .map(|c| c.row_point.iter().copied().map(from_e).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let layer;
            let fold = if m == pcs::WeightMatrix::ChannelParams {
                layer = Vec::new();
                pcs::WeightFold::channel_params(
                    &self.weight_layout,
                    &mut reader,
                    &rowpoints.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &xi[i],
                )
                .unwrap()
            } else {
                let full = original[0]
                    .column_point
                    .iter()
                    .chain(original[0].row_point)
                    .copied()
                    .map(from_e)
                    .collect::<Vec<_>>();
                let pcs::MatrixKind::Weight { shape, a, h, .. } =
                    self.weight_layout.matrix(m).kind()
                else {
                    unreachable!()
                };
                let input = shape.inputs.next_power_of_two().trailing_zeros() as usize;
                let output = (a + h) as usize;
                layer = full[input + output..].to_vec();
                pcs::WeightFold::new(
                    &self.weight_layout,
                    m,
                    &mut reader,
                    &full[input..input + output],
                    &xi[i],
                )
                .unwrap()
            };
            let combined = fold.combined_rows(&layer, theta[i]).unwrap();
            let packet = protocol::WeightCombinedRows {
                matrix: m.index() as u8,
                evaluations: combined.evaluations.iter().map(wire).collect(),
                proximity: wire(&combined.proximity),
            };
            let label = format!("pcs/weights/{m:?}/combined");
            t.absorb(&label, &packet);
            v.absorb(ascii(&label), &tessera_codec::encode(&packet));
            let rows = combined
                .evaluations
                .iter()
                .chain([&combined.proximity])
                .map(tv_combined)
                .collect::<Vec<_>>();
            wc.push((i, original, rows));
        }
        let vxis: [Vec<E>; 10] = core::array::from_fn(|i| es(&xi[i]));
        let weights_evaluated = wc
            .iter()
            .map(|(i, original, rows)| {
                vw::check_evaluations(
                    vw::Opening {
                        layout: &wl[*i],
                        root: self.weights.public().roots[*i],
                        claims: original,
                        xi_point: &vxis[*i],
                    },
                    rows,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        t.absorb("scoring", &vec![0u64]);
        v.absorb(ascii("scoring"), &tessera_codec::encode(&vec![0u64]));
        if mode == Mode::NonInteractive {
            let nonce = t.grind("query-pow", ParamSet::TEST_SMALL.query_pow as u8);
            v = v
                .check_grind(
                    ascii("query-pow"),
                    ParamSet::TEST_SMALL.query_pow as u8,
                    nonce,
                )
                .unwrap();
        }
        let query = |t: &mut Transcript,
                     v: &mut tv_hash::transcript::Transcript,
                     label: &str,
                     bits: u32,
                     count: u32| {
            if mode == Mode::Live {
                // Supplied verifier-query fixture; the production Live RNG belongs to the driver.
                (0..count)
                    .map(|i| u64::from(i * 7 + 3) & ((1u64 << bits) - 1))
                    .collect::<Vec<_>>()
            } else {
                let q = t.challenge_indices(label, bits, count);
                assert_eq!(q, v.challenge_indices(ascii(label), bits, count).unwrap());
                q
            }
        };
        let aq = query(
            t,
            &mut v,
            "pcs/advice/queries",
            self.encoder.codeword_len().trailing_zeros(),
            ParamSet::TEST_SMALL.advice_queries,
        );
        let mut wq = Vec::new();
        for (i, _, _) in &wc {
            let m = pcs::WeightMatrix::ALL[*i];
            let label = format!("pcs/weights/{m:?}/queries");
            wq.push((
                *i,
                query(
                    t,
                    &mut v,
                    &label,
                    self.weight_layout.matrix(m).log2_k() + 2,
                    ParamSet::TEST_SMALL.weight_queries,
                ),
            ));
        }
        if mode == Mode::Live {
            let packet = protocol::ColumnQueries {
                advice: aq.iter().map(|&q| q as u32).collect(),
                weights: wq
                    .iter()
                    .map(|(i, q)| protocol::MatrixQueries {
                        matrix: *i as u8,
                        indices: q.iter().map(|&q| q as u32).collect(),
                    })
                    .collect(),
            };
            t.absorb("queries", &packet);
            v.absorb(ascii("queries"), &tessera_codec::encode(&packet));
        }
        let answer = |source| {
            let mut masks = Masks(831);
            let mut pass = checked.answer(&self.encoder, &mut masks, &aq).unwrap();
            for s in 0..self.p.claims.layout.advice.segments {
                rows(self.p, self.advice, source, s, |r, x| pass.push_row(r, x));
            }
            pass.finish().unwrap()
        };
        let answer = answer(self.source);
        let mut masks = Masks(831);
        let mut replay_pass = replayed.answer(&self.encoder, &mut masks, &aq).unwrap();
        for s in 0..self.p.claims.layout.advice.segments {
            rows(self.p, self.advice, self.replay, s, |r, x| {
                replay_pass.push_row(r, x)
            });
        }
        assert_eq!(answer, replay_pass.finish().unwrap());
        let columns = answer
            .columns
            .iter()
            .map(|c| c.iter().copied().map(super::v).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        if attacks {
            use tv_core::opening::OpeningError as O;
            let first = *aq.iter().min().unwrap() as u32;
            let mut bad = columns.clone();
            bad[0][0] += V::ONE;
            assert_eq!(
                bound.check_evaluations(&ac).unwrap().check_columns(
                    &aq,
                    &bad,
                    &answer.proof.siblings
                ),
                Err(O::MerklePath { column: first })
            );
            let mut siblings = answer.proof.siblings.clone();
            siblings.push([0; 32]);
            assert_eq!(
                bound
                    .check_evaluations(&ac)
                    .unwrap()
                    .check_columns(&aq, &columns, &siblings),
                Err(O::MerklePath { column: first })
            );
            let mut bad = ac.clone();
            for value in &mut bad.last_mut().unwrap().u {
                *value += E::ONE;
            }
            assert_eq!(
                bound.check_evaluations(&bad).unwrap().check_columns(
                    &aq,
                    &columns,
                    &answer.proof.siblings
                ),
                Err(O::Proximity { column: first })
            );
        }
        advice_evaluated
            .check_columns(&aq, &columns, &answer.proof.siblings)
            .unwrap();
        for ((i, queries), evaluated) in wq.iter().zip(weights_evaluated) {
            let m = pcs::WeightMatrix::ALL[*i];
            let answer = self.weights.matrix(m).answer(queries).unwrap();
            let columns = answer
                .columns
                .iter()
                .map(|c| c.iter().copied().map(super::v).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            if attacks {
                let (_, original, combined) = wc.iter().find(|(index, _, _)| index == i).unwrap();
                let opening = vw::Opening {
                    layout: &wl[*i],
                    root: self.weights.public().roots[*i],
                    claims: original,
                    xi_point: &vxis[*i],
                };
                for index in 0..original.len() {
                    let mut bad = original.clone();
                    bad[index].value += E::ONE;
                    assert_eq!(
                        vw::check_evaluations(
                            vw::Opening {
                                claims: &bad,
                                ..opening
                            },
                            combined
                        )
                        .err(),
                        Some(vw::Error::Evaluation {
                            matrix: vw::Matrix::ALL[*i]
                        })
                    );
                }
                let first = *queries.iter().min().unwrap() as u32;
                let mut bad = columns.clone();
                bad[0][0] += V::ONE;
                assert_eq!(
                    vw::check_evaluations(opening, combined)
                        .unwrap()
                        .check_columns(queries, &bad, &answer.proof.siblings),
                    Err(vw::Error::MerklePath {
                        matrix: vw::Matrix::ALL[*i],
                        column: first
                    })
                );
                let mut bad = combined.clone();
                for value in &mut bad.last_mut().unwrap().u {
                    *value += E::ONE;
                }
                assert_eq!(
                    vw::check_evaluations(opening, &bad).unwrap().check_columns(
                        queries,
                        &columns,
                        &answer.proof.siblings
                    ),
                    Err(vw::Error::Proximity {
                        matrix: vw::Matrix::ALL[*i],
                        column: first
                    })
                );
            }
            evaluated
                .check_columns(queries, &columns, &answer.proof.siblings)
                .unwrap();
        }
        assert_eq!(
            e(t.challenge("test/after-openings")),
            v.challenge_fp2(ascii("test/after-openings")).unwrap()
        );
    }
}

#[test]
fn canonical_rows_omit_public_padding_and_preserve_source_and_sink_errors() {
    use tessera_prover::advice::{Error, visit_rows};
    let model = bounded_model_named("tiny-2-affine");
    let mut p = public(
        &model,
        3,
        GadgetPolicy {
            max_log2_rows: 18,
            query_block: 2,
            ..GadgetPolicy::default()
        },
    );
    p.claims.layout.advice.segments = 5;
    let engine = Engine::new(&p.lowering).unwrap();
    let tokens = Secret::new(vec![1, 2, 3]);
    let witness = engine.run(&model, &tokens).unwrap();
    let source = Resident::new(&engine, &p.selection, &model, &witness).unwrap();
    let layout = ModelAdviceLayout::new(&p.lowering, &p.selection, &p.claims).unwrap();
    let first = layout
        .location(0, AdviceSource::Tensor(p.lowering.tokens))
        .unwrap()
        .first_row();
    let width = 1usize << layout.layout().params().log2_k;
    let table_counts = layout
        .tiles()
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            let [name @ AdviceSource::Multiplicity(_)] = layout.sources(i).unwrap() else {
                return None;
            };
            let tile = source.tile(t).unwrap();
            let column = tile.multiplicities().unwrap();
            let mut counts = tessera_codec::SecretBuf::new();
            counts.resize(column.rows() as usize, 0);
            column.read_rows(0, &mut counts).unwrap();
            Some((layout.location(i, *name).unwrap().clone(), counts))
        })
        .collect::<Vec<_>>();
    let mut saw_tokens = false;
    for (segment, expected) in layout.layout().segments().iter().enumerate() {
        let mut rows = Vec::new();
        visit_rows(
            &p.lowering,
            &layout,
            &source,
            segment as u32,
            |r, values| {
                rows.push(r);
                for (place, counts) in &table_counts {
                    if place.nonzero().iter().any(|run| run.contains(&r)) {
                        let start = (r - place.first_row()) as usize * width;
                        for (i, value) in values.iter().enumerate() {
                            assert_eq!(
                                value.to_canonical_u64(),
                                counts.get(start + i).copied().unwrap_or(0)
                            );
                        }
                    }
                }
                if r == first {
                    assert_eq!(
                        &values[..3],
                        &[G::ONE, G::from_u64_reduced(2), G::from_u64_reduced(3)]
                    );
                    assert!(values[3..].iter().all(|&x| x == G::ZERO));
                    saw_tokens = true;
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            rows,
            expected
                .nonzero
                .iter()
                .cloned()
                .flatten()
                .collect::<Vec<_>>()
        );
    }
    assert!(saw_tokens);
    assert!(
        layout
            .layout()
            .segments()
            .iter()
            .any(|s| s.nonzero.is_empty())
    );
    assert!(
        layout
            .tiles()
            .iter()
            .enumerate()
            .any(
                |(i, _)| layout.sources(i).unwrap().iter().any(|&s| matches!(
                    s,
                    AdviceSource::Multiplicity(_)
                ) && {
                    let place = layout.location(i, s).unwrap();
                    place.nonzero().iter().map(|r| r.end - r.start).sum::<u64>() < place.row_count()
                })
            )
    );
    struct Swapped<'a> {
        source: &'a dyn WitnessSource,
        from: &'a tessera_compiler::SequenceTile,
        to: &'a tessera_compiler::SequenceTile,
    }
    impl WitnessSource for Swapped<'_> {
        fn tile(
            &self,
            tile: &tessera_compiler::SequenceTile,
        ) -> Result<tessera_witness::TileWitness, tessera_witness::WitnessError> {
            self.source
                .tile(if tile == self.from { self.to } else { tile })
        }
    }
    let other = layout
        .tiles()
        .iter()
        .find(|t| {
            t.shares.len() == 1
                && t.shares[0]
                    .ranges
                    .iter()
                    .map(|r| r.end - r.start)
                    .product::<u32>()
                    != 3
        })
        .unwrap();
    let bad = Swapped {
        source: &source,
        from: &layout.tiles()[0],
        to: other,
    };
    assert_eq!(
        visit_rows(&p.lowering, &layout, &bad, 0, |_, _| Ok(())),
        Err(Error::Internal)
    );
    let tables = layout
        .tiles()
        .iter()
        .enumerate()
        .filter_map(|(i, t)| match layout.sources(i).unwrap() {
            [s @ AdviceSource::Multiplicity(_)] => {
                Some((t, layout.location(i, *s).unwrap().vars()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let (from, vars) = tables[0];
    let (to, _) = tables
        .iter()
        .copied()
        .find(|(_, bits)| *bits != vars)
        .unwrap();
    let bad = Swapped {
        source: &source,
        from,
        to,
    };
    for segment in 0..p.claims.layout.advice.segments {
        let got = visit_rows(&p.lowering, &layout, &bad, segment, |_, _| Ok(()));
        let index = layout.tiles().iter().position(|t| t == from).unwrap();
        assert_eq!(
            got,
            if layout.layout().tiles()[index].segment == segment {
                Err(Error::Internal)
            } else {
                Ok(())
            }
        );
    }
    struct Failing;
    impl WitnessSource for Failing {
        fn tile(
            &self,
            _: &tessera_compiler::SequenceTile,
        ) -> Result<tessera_witness::TileWitness, tessera_witness::WitnessError> {
            Err(tessera_witness::WitnessError::Usage {
                rule: "public source failure",
            })
        }
    }
    let mut calls = 0;
    let mut sink = |_, _: &[G]| {
        calls += 1;
        Ok(())
    };
    assert_eq!(
        visit_rows(&p.lowering, &layout, &Failing, 5, &mut sink),
        Err(Error::Internal)
    );
    assert_eq!(
        visit_rows(&p.lowering, &layout, &Failing, 0, sink),
        Err(Error::Witness(tessera_witness::WitnessError::Usage {
            rule: "public source failure"
        }))
    );
    assert_eq!(calls, 0);
    for error in [
        Error::Pcs(pcs::PcsError::NoQueries),
        Error::Opening(pcs::AdviceOpeningError::Randomness),
    ] {
        let mut calls = 0;
        assert_eq!(
            visit_rows(&p.lowering, &layout, &source, 0, |_, _| {
                calls += 1;
                Err(error)
            }),
            Err(error)
        );
        assert_eq!(calls, 1);
    }
}
