// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Real current-model P1/P4/P5, checked by independent canonical PCS components (DEP-2).
//! The accepting driver consumes these real messages without sharing producer code.
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
pub static NI_REPLAYS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static LIVE_ACCEPTANCES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub struct First {
    pub round1: tv_core::protocol::Round1,
    pub seed_nonce: u64,
    pub live: Option<tv_core::LiveVerifier>,
    pub replay: Option<Transcript>,
    pub live_seed: Option<[u8; 32]>,
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
    prompt: Option<u32>,
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
    pub fn statement(&self) -> protocol::PublicStatement {
        self.statement.clone()
    }
    pub fn public_weights(&self) -> pcs::WeightCommitmentPublic {
        *self.weights.public()
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        p: &'a Public,
        model: &'a IntegerModel,
        advice: &'a ModelAdviceLayout<'a>,
        source: &'a dyn WitnessSource,
        replay: &'a dyn WitnessSource,
        claims: &'a tv_core::claims::ClaimList,
        prompt: Option<u32>,
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
                    prompt_len: prompt
                        .unwrap_or(p.lowering.graph.tensors[p.lowering.tokens.0 as usize].shape[0]),
                    output_len: prompt.map_or(0, |prompt| {
                        p.lowering.graph.tensors[p.lowering.tokens.0 as usize].shape[0] - prompt
                    }),
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
            prompt,
        }
    }
    fn verifier_config(&self) -> Option<tv_core::VerifierConfig> {
        let params = tv_core::params::ParamSet::accept(&self.claims.params)?;
        let weights = tv_codec::decode(
            &tessera_codec::encode(self.weights.public()),
            &tv_codec::DecodeLimits::MESSAGE,
            tv_codec::WeightCommitmentPublic::decode,
        )
        .unwrap();
        Some(tv_core::VerifierConfig {
            params,
            limits: Default::default(),
            accept_bound: tv_core::protocol::FixedBits(0),
            weights,
        })
    }
    pub fn start(
        &self,
        mode: Mode,
    ) -> (
        Transcript,
        tv_hash::transcript::Transcript,
        [Vec<F>; 10],
        First,
    ) {
        let mut live = if mode == Mode::Live && self.prompt.is_some() {
            self.verifier_config().map(|cfg| {
                let context = protocol::Round1 {
                    protocol_version: tessera_ir::PROTOCOL_VERSION,
                    session_id: [0; 16],
                    statement: self.statement.clone(),
                    advice: self.advice_state.commitment(),
                    leaf_groups: protocol::LeafGroups::NONE,
                };
                let decoded = tv_codec::decode(
                    &tessera_codec::encode(&context),
                    &tv_codec::DecodeLimits::MESSAGE,
                    tv_core::protocol::Round1::decode,
                )
                .unwrap();
                tv_core::LiveVerifier::new(self.claims.clone(), decoded.statement, cfg)
            })
        } else {
            None
        };
        let commit = live.as_mut().map(|v| v.start().unwrap());
        let session_id = commit.as_ref().map_or([130; 16], |c| c.session_id);
        let context = SessionContext {
            protocol: Ascii::new("tessera-protocol/1").unwrap(),
            mode,
            params: ParamSetId(Ascii::new("test-small/1").unwrap()),
            statement_digest: self.digest,
            session_id,
        };
        let mut t = Transcript::new(&context);
        let mut replay = Transcript::new(&context);
        let mut v = tv_hash::transcript::Transcript::new(&tv_hash::transcript::SessionContext {
            protocol: ascii("tessera-protocol/1"),
            mode: if mode == Mode::Live {
                tv_hash::transcript::Mode::Live
            } else {
                tv_hash::transcript::Mode::NonInteractive
            },
            params: tv_hash::transcript::ParamSetId(ascii("test-small/1")),
            statement_digest: self.digest,
            session_id,
        });
        let round1 = protocol::Round1 {
            protocol_version: tessera_ir::PROTOCOL_VERSION,
            session_id,
            statement: self.statement.clone(),
            advice: self.advice_state.commitment(),
            leaf_groups: protocol::LeafGroups::NONE,
        };
        let wire_round1 = tv_codec::decode(
            &tessera_codec::encode(&round1),
            &tv_codec::DecodeLimits::MESSAGE,
            tv_core::protocol::Round1::decode,
        )
        .unwrap();
        t.absorb("round1", &round1);
        replay.absorb("round1", &round1);
        v.absorb(ascii("round1"), &tessera_codec::encode(&round1));
        let mut seed_nonce = 0;
        let mut live_seed = None;
        if mode == Mode::Live {
            let share = if let Some(driver) = live.as_mut() {
                let reveal = driver
                    .on_round1(
                        &wire_round1,
                        &tv_core::protocol::ProverSeedShare {
                            session_id,
                            share: [17; 32],
                        },
                    )
                    .unwrap();
                assert_eq!(
                    tessera_hash::hash(tessera_hash::Domain::CoinCommit, &reveal.share),
                    commit.as_ref().unwrap().commitment
                );
                reveal.share
            } else {
                [13; 32]
            };
            let seed = protocol::coin_toss(&share, &[17; 32]);
            live_seed = Some(seed);
            t.absorb("seed", &seed);
            replay.absorb("seed", &seed);
            v.absorb(ascii("seed"), &seed);
        } else {
            let nonce = t.grind("seed-pow", ParamSet::TEST_SMALL.seed_pow as u8);
            seed_nonce = nonce;
            assert!(replay.check_grind("seed-pow", ParamSet::TEST_SMALL.seed_pow as u8, nonce));
            v = v
                .check_grind(
                    ascii("seed-pow"),
                    ParamSet::TEST_SMALL.seed_pow as u8,
                    nonce,
                )
                .unwrap();
            let seed = t.squeeze("seed");
            assert_eq!(seed, replay.squeeze("seed"));
            assert_eq!(seed, v.squeeze(ascii("seed")));
        }
        let mut xi: [Vec<F>; 10] = core::array::from_fn(|_| Vec::new());
        let vl = vw::derive(self.claims, 0).unwrap();
        for (i, m) in pcs::WeightMatrix::ALL.into_iter().enumerate() {
            if self.weight_layout.matrix(m).rows() > 0 {
                let label = format!("pcs/weights/{m:?}/xi");
                for _ in 0..vl[i].proximity_vars() {
                    let x = t.challenge(&label);
                    assert!(
                        x == replay.challenge(&label),
                        "initial proximity transcript differs"
                    );
                    assert_eq!(e(x), v.challenge_fp2(ascii(&label)).unwrap());
                    xi[i].push(x);
                }
            }
        }

        (
            t,
            v,
            xi,
            First {
                round1: wire_round1,
                seed_nonce,
                live,
                replay: Some(replay),
                live_seed,
            },
        )
    }
    pub fn openings(
        &self,
        layout: &LeafLayout<ModelColumn>,
        leaf: &tessera_prover::ProvenLeaf,
        native: &tessera_prover::native::ModelNativePlan,
        stages: &tessera_prover::StageProver<'_, tessera_witness::Witness>,
        witness: &tessera_witness::Witness,
    ) -> tessera_prover::ModelOpenings {
        let unfinished =
            tessera_prover::StageProver::new(&self.p.lowering, native.schedule(), witness);
        assert!(matches!(
            tessera_prover::ModelOpenings::derive(
                &self.p.lowering,
                layout,
                leaf,
                native,
                &unfinished,
                self.advice,
                &self.weight_layout
            ),
            Err(tessera_prover::StageError::Internal {
                rule: "INV-SCH-3: original opening sources, points and order follow the runtime schedule"
            })
        ));
        tessera_prover::ModelOpenings::derive(
            &self.p.lowering,
            layout,
            leaf,
            native,
            stages,
            self.advice,
            &self.weight_layout,
        )
        .unwrap()
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
        first: First,
        round2: tv_core::protocol::Round2Response,
        cpu_openings: tessera_prover::ModelOpenings,
    ) {
        let First {
            round1,
            seed_nonce,
            mut live,
            live_seed,
            ..
        } = first;
        let mut ni = tv_core::protocol::NiProof {
            round1,
            seed_nonce,
            round2,
            query_nonce: 0,
            round3: tv_core::protocol::Round3Response {
                advice_columns: Vec::new(),
                advice_proof: tv_core::protocol::MultiProof {
                    siblings: Vec::new(),
                },
                weight_columns: Vec::new(),
            },
        };

        let selected = tv_core::selection::derive(graph, self.claims.gadget_policy).unwrap();
        let config = va::Config::new(
            &self.claims.arch,
            self.claims.gadget_policy,
            self.claims.layout.advice.log2_k,
            self.claims.layout.advice.segments,
        );
        let tables = selected
            .tables
            .iter()
            .map(|t| tv_tables::Table::generate(t.id, t.kind.clone()).unwrap())
            .collect::<Vec<_>>();
        let runtime = self.prompt.map(|prompt| {
            tv_core::gadgets::runtime::Plan::derive(
                graph,
                self.claims.gadget_policy,
                prompt,
                0,
                &tables,
            )
            .unwrap()
        });
        let a = if let Some(runtime) = &runtime {
            runtime.advice_layout(config)
        } else {
            va::Layout::derive(graph, &selected, config, graph.length())
        }
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
        assert_eq!(
            cpu.len(),
            cpu_openings.advice().len(),
            "original advice count"
        );
        for (i, (expected, actual)) in cpu.iter().zip(cpu_openings.advice()).enumerate() {
            if expected.pieces.len() == actual.pieces.len()
                && expected
                    .pieces
                    .iter()
                    .zip(&actual.pieces)
                    .all(|(a, b)| a.point == b.point && a.weight == b.weight)
            {
                continue;
            }
            let k = usize::from(self.p.claims.layout.advice.log2_k);
            let columns = |claim: &pcs::AdviceClaim| {
                let mut out = Vec::new();
                for p in &claim.pieces {
                    let c = p.point[..k].to_vec();
                    if !out.contains(&c) {
                        out.push(c);
                    }
                }
                out
            };
            let groups = columns(expected);
            assert_eq!(groups, columns(actual), "original claim {i} column groups");
            // Independent implementations may decompose the same linear map differently.
            // Compare EVERY Boolean row coefficient, not samples or the current witness values.
            for group in groups {
                let coefficients = |claim: &pcs::AdviceClaim| {
                    let n = 1usize << (claim.pieces[0].point.len() - k);
                    let mut row = vec![F::ZERO; n];
                    for p in claim.pieces.iter().filter(|p| p.point[..k] == group) {
                        let eq = tessera_poly::eq_table(&p.point[k..]).unwrap();
                        for (v, c) in row.iter_mut().zip(eq) {
                            *v += p.weight * c;
                        }
                    }
                    row
                };
                assert!(
                    coefficients(expected) == coefficients(actual),
                    "original claim {i} complete row coefficients"
                );
            }
        }
        for matrix in pcs::WeightMatrix::ALL {
            let expected = claims
                .weight_claims(vw::Matrix::ALL[matrix.index()], E::ONE)
                .unwrap();
            let points = expected
                .iter()
                .map(|c| {
                    c.column_point
                        .iter()
                        .chain(c.row_point)
                        .copied()
                        .map(from_e)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            assert_eq!(
                cpu_openings.weight_points(matrix),
                points,
                "matrix {matrix:?} original points"
            );
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
        let prepared = self
            .advice_state
            .prepare(cpu_openings.advice(), gamma, &av)
            .unwrap();
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
        if let Some(runtime) = &runtime {
            let g = tv_core::runtime::Geometry::derive(self.claims, runtime).unwrap();
            let rounds: Vec<_> = g
                .levels()
                .iter()
                .copied()
                .chain(g.stages().iter().map(|s| s.sumcheck))
                .map(|f| tv_core::soundness::RoundGroup {
                    count: f.rounds as u64,
                    degree: f.degree as u64,
                })
                .collect();
            let tree_levels: Vec<_> = g.levels().iter().map(|f| f.instances as u64).collect();
            let report =
                if let Some(params) = tv_core::params::ParamSet::accept(&self.claims.params) {
                    let budget = g
                        .check_budget(
                            params,
                            if mode == Mode::Live {
                                tv_hash::transcript::Mode::Live
                            } else {
                                tv_hash::transcript::Mode::NonInteractive
                            },
                            &bound,
                            &claims,
                            &wl,
                        )
                        .unwrap();
                    budget.report().clone()
                } else {
                    tv_core::soundness::SoundnessReport::evaluate(
                        tv_core::soundness::SoundnessParams::test_small(),
                        g.statement_sizes(&bound, &claims, &wl).unwrap(),
                    )
                    .unwrap()
                };
            assert_eq!(report.statement.rounds, rounds);
            assert_eq!(report.statement.tree_levels, tree_levels);
            let r = tessera_compiler::RuntimePlan::derive(
                &self.p.lowering,
                &self.p.selection,
                &self.p.claims,
                self.prompt.unwrap(),
                0,
            )
            .unwrap();
            let leaf =
                LeafLayout::runtime(&self.p.lowering, &self.p.layers, &r, &self.p.claims).unwrap();
            let plan = tessera_prover::native::ModelNativePlan::derive_runtime(
                &self.p.lowering,
                &r,
                &self.p.claims,
                &leaf,
            )
            .unwrap();
            let stages = &plan.schedule().stages[1..];
            assert_eq!(
                report.statement.stages,
                stages
                    .iter()
                    .map(|s| s.instances.len() as u64)
                    .collect::<Vec<_>>()
            );
            let zero_checks: Vec<_> = stages
                .iter()
                .flat_map(|s| &s.instances)
                .filter(|i| {
                    i.source == tessera_compiler::schedule::Source::ZeroCheck
                        || matches!(
                            i.relation,
                            tessera_compiler::schedule::Relation::Native { zero: true, .. }
                        )
                })
                .map(|i| u64::from(i.vars))
                .collect();
            assert_eq!(report.statement.zero_checks, zero_checks);
            let lookup_sizes: Vec<_> = leaf
                .instances()
                .iter()
                .map(|i| {
                    let table = r
                        .selection()
                        .tables
                        .iter()
                        .find(|t| t.id == i.table)
                        .unwrap();
                    tv_core::soundness::LookupSize {
                        lookups: i.entries.next_power_of_two(),
                        rows: 1u64 << table.log2_rows,
                    }
                })
                .collect();
            assert_eq!(report.statement.lookups, lookup_sizes);
            let schedule = tessera_prover::derive_runtime_schedule(
                &self.p.claims,
                graph.length(),
                self.prompt.unwrap(),
                0,
            )
            .unwrap();
            assert_eq!(
                report.statement.forests,
                schedule
                    .sizes
                    .trees
                    .iter()
                    .filter_map(|t| match *t {
                        protocol::LeafTree::Forest { trees_log, .. } => Some(u64::from(trees_log)),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                report
                    .statement
                    .matrices
                    .iter()
                    .map(|m| m.matrix.tag())
                    .collect::<Vec<_>>(),
                schedule
                    .sizes
                    .opened
                    .iter()
                    .map(|m| u16::from(m.matrix))
                    .collect::<Vec<_>>()
            );
            for size in &report.statement.matrices {
                let matrix = self
                    .weight_layout
                    .matrix(pcs::WeightMatrix::ALL[usize::from(size.matrix.tag())]);
                assert_eq!(
                    (size.width, size.rows, size.variables),
                    (
                        1 << matrix.log2_k(),
                        matrix.rows(),
                        u64::from(matrix.log2_k() + matrix.row_vars())
                    )
                );
            }
            assert_eq!(
                report.statement.claim_groups.iter().sum::<u64>(),
                original.len() as u64
            );
            assert!(
                report
                    .statement
                    .matrices
                    .iter()
                    .any(|m| m.matrix == vw::Matrix::LmHead)
            );
            assert_eq!(
                report.statement.advice_rows,
                1u64 << a
                    .nu()
                    .saturating_sub(u32::from(self.claims.layout.advice.log2_k))
            );
        }
        assert_eq!(bound.group_count(), prepared.group_count());
        let ac = checked
            .combined()
            .iter()
            .map(tv_combined)
            .collect::<Vec<_>>();
        let attacks = (graph.length() == 3 || self.prompt.is_some())
            && self.claims.arch.n_layers == 2
            && mode == Mode::NonInteractive;
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
        ni.round2.advice_combined = ac.clone();
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
            let full_points = cpu_openings.weight_points(m);
            let k = self.weight_layout.matrix(m).log2_k() as usize;
            let rowpoints = full_points
                .iter()
                .map(|p| p[k..].to_vec())
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
                let full = &full_points[0];
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
            ni.round2
                .weight_combined
                .push(tv_core::protocol::WeightCombinedRows {
                    matrix: i as u8,
                    evaluations: combined.evaluations.iter().map(tv_combined).collect(),
                    proximity: tv_combined(&combined.proximity),
                });
            wc.push((i, original, rows));
        }
        let vxis: [Vec<E>; 10] = core::array::from_fn(|i| es(&xi[i]));
        if attacks {
            for (i, original, rows) in &wc {
                for index in 0..original.len() {
                    let mut bad = original.clone();
                    bad[index].value += E::ONE;
                    assert_eq!(
                        vw::check_evaluations(
                            vw::Opening {
                                layout: &wl[*i],
                                root: self.weights.public().roots[*i],
                                claims: &bad,
                                xi_point: &vxis[*i],
                            },
                            rows
                        )
                        .err(),
                        Some(vw::Error::Evaluation {
                            matrix: vw::Matrix::ALL[*i]
                        })
                    );
                }
            }
        }
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
            ni.query_nonce = nonce;
            v = v
                .check_grind(
                    ascii("query-pow"),
                    ParamSet::TEST_SMALL.query_pow as u8,
                    nonce,
                )
                .unwrap();
        }
        let supplied_queries = live
            .as_mut()
            .map(|driver| driver.on_round2(&ni.round2).unwrap());
        let query = |t: &mut Transcript,
                     v: &mut tv_hash::transcript::Transcript,
                     label: &str,
                     bits: u32,
                     count: u32| {
            if mode == Mode::Live {
                if let Some(q) = &supplied_queries {
                    let values = if label == "pcs/advice/queries" {
                        &q.advice
                    } else {
                        &q.weights
                            .iter()
                            .find(|p| {
                                format!(
                                    "pcs/weights/{:?}/queries",
                                    pcs::WeightMatrix::ALL[usize::from(p.matrix)]
                                ) == label
                            })
                            .unwrap()
                            .indices
                    };
                    assert_eq!(values.len(), count as usize);
                    values.iter().copied().map(u64::from).collect()
                } else {
                    (0..count)
                        .map(|i| u64::from(i * 7 + 3) & ((1u64 << bits) - 1))
                        .collect()
                }
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
        ni.round3.advice_columns = columns.clone();
        ni.round3.advice_proof.siblings = answer.proof.siblings.clone();
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
            ni.round3
                .weight_columns
                .push(tv_core::protocol::MatrixColumns {
                    matrix: *i as u8,
                    columns: columns.clone(),
                    proof: tv_core::protocol::MultiProof {
                        siblings: answer.proof.siblings.clone(),
                    },
                });
            evaluated
                .check_columns(queries, &columns, &answer.proof.siblings)
                .unwrap();
        }
        if let Some(driver) = live.as_mut() {
            let result = driver.on_round3(&ni.round3);
            assert!(
                matches!(result, tv_core::Verdict::Accept(_)),
                "actual Live session: {result:?}"
            );
            let parsed: serde_json::Value =
                serde_json::from_str(&result.to_json().unwrap()).unwrap();
            assert_eq!(parsed["test_parameters"], true);
            assert_eq!(parsed["anchor_verified"], false);
            if graph.length() == 5
                && self.prompt == Some(4)
                && self.p.claims.arch.d_model == 2
                && let Some(folder) = std::env::var_os("TESSERA_CAPTURE_DRIVER_FIXTURE")
            {
                let folder = PathBuf::from(folder);
                fs::create_dir(&folder).unwrap();
                let decoded: protocol::NiProof =
                    tessera_codec::decode(&ni.to_tce().unwrap(), DecodeLimits::MESSAGE).unwrap();
                fs::write(
                    folder.join("live-messages.tnp"),
                    envelope::seal(&schema::NI_PROOF, &decoded),
                )
                .unwrap();
                fs::write(
                    folder.join("claims.tcl"),
                    envelope::seal(&schema::CLAIM_LIST, &self.p.claims),
                )
                .unwrap();
                fs::write(
                    folder.join("weights.tce"),
                    tessera_codec::encode(self.weights.public()),
                )
                .unwrap();
                fs::write(folder.join("live-seed.bin"), live_seed.unwrap()).unwrap();
                fs::write(
                    folder.join("queries.tce"),
                    supplied_queries.as_ref().unwrap().to_tce().unwrap(),
                )
                .unwrap();
            }
            LIVE_ACCEPTANCES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            assert!(matches!(
                driver.on_round3(&ni.round3),
                tv_core::Verdict::Inconclusive(tv_core::VerifierError {
                    code: tv_core::ErrorCode::ProtocolOrder
                })
            ));
        }
        if self.prompt.is_some()
            && mode == Mode::NonInteractive
            && let Some(cfg) = self.verifier_config()
        {
            let result =
                tv_core::verify_non_interactive(self.claims, &ni.round1.statement, &ni, &cfg);
            assert!(
                matches!(result, tv_core::Verdict::Accept(_)),
                "full NI driver: {result:?}"
            );
            let json = result.to_json().unwrap();
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed["verdict"]["status"], "accept");
            assert_eq!(parsed["test_parameters"], true);
            assert_eq!(parsed["anchor_verified"], false);
            assert_eq!(parsed["zero_knowledge"], false);
            assert_eq!(parsed["checks"].as_array().unwrap().len(), 13);
            let rejected = |bad: &tv_core::protocol::NiProof, expected: tv_core::RejectReason| {
                match tv_core::verify_non_interactive(self.claims, &ni.round1.statement, bad, &cfg)
                {
                    tv_core::Verdict::Reject(actual) => assert_eq!(actual, expected),
                    other => panic!("expected {expected:?}, got {other:?}"),
                }
            };
            use tv_core::verdict::{
                ArtifactKind, CommitmentClass as C, ErrorCode, RejectReason as R,
            };
            let mut bad = ni.clone();
            bad.round2.logup.roots[1] = E::ZERO;
            let table = runtime.as_ref().unwrap().leaf_schedule().unwrap().lookups()[0].0;
            rejected(&bad, R::Lookup { table });
            let mut bad = ni.clone();
            bad.round2.stages.clear();
            rejected(
                &bad,
                R::Malformed {
                    artifact: ArtifactKind(0x0520),
                    code: ErrorCode::CodecNoncanonical,
                },
            );
            let mut bad = ni.clone();
            bad.round2.stages[0].sumcheck.final_evals[0] += E::ONE;
            rejected(
                &bad,
                R::Sumcheck {
                    stage: 1,
                    round: ni.round2.stages[0].sumcheck.rounds.len() as u32,
                },
            );
            let mut bad = ni.clone();
            bad.round2.advice_combined[0].u[0] += E::ONE;
            rejected(
                &bad,
                R::Evaluation {
                    commitment: C::Advice,
                },
            );
            let mut bad = ni.clone();
            bad.round2.weight_combined[0].evaluations[0].u[0] += E::ONE;
            rejected(
                &bad,
                R::Evaluation {
                    commitment: C::Weight,
                },
            );
            let mut bad = ni.clone();
            bad.round3.advice_columns[0][0] += V::ONE;
            rejected(
                &bad,
                R::MerklePath {
                    commitment: C::Advice,
                    column: *aq.iter().min().unwrap() as u32,
                },
            );
            let mut bad = ni.clone();
            bad.round3.weight_columns[0].columns[0][0] += V::ONE;
            rejected(
                &bad,
                R::MerklePath {
                    commitment: C::Weight,
                    column: *wq[0].1.iter().min().unwrap() as u32,
                },
            );
            if graph.length() == 5 && self.prompt == Some(4) && self.p.claims.arch.d_model == 2 {
                for _ in 0..100 {
                    let repeated = tv_core::verify_non_interactive(
                        self.claims,
                        &ni.round1.statement,
                        &ni,
                        &cfg,
                    );
                    assert!(matches!(repeated, tv_core::Verdict::Accept(_)));
                    assert_eq!(repeated.to_json().unwrap(), json);
                    NI_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
            let encoded = ni.to_tce().unwrap();
            let root_proof: protocol::NiProof =
                tessera_codec::decode(&encoded, DecodeLimits::MESSAGE).unwrap();
            let envelope = envelope::seal(&schema::NI_PROOF, &root_proof);
            if graph.length() == 5
                && self.prompt == Some(4)
                && self.p.claims.arch.d_model == 2
                && let Some(folder) = std::env::var_os("TESSERA_CAPTURE_DRIVER_FIXTURE")
            {
                let file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(PathBuf::from(folder).join("ni.tnp"))
                    .unwrap();
                use std::io::Write;
                let mut file = file;
                file.write_all(&envelope).unwrap();
            }

            let result = tv_core::verify_non_interactive_bytes(
                self.claims,
                &ni.round1.statement,
                &envelope,
                &cfg,
            );
            assert!(
                matches!(result, tv_core::Verdict::Accept(_)),
                "canonical NI envelope: {result:?}"
            );
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
