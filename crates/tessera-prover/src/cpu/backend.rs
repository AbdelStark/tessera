// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::*;
use crate::leaf::{LeafLayout, ModelColumn, ModelWitness, prepare_leaf};
use crate::native::{ModelNativePlan, ModelNativeProver};
use compiler::{Lowering, ModelAdviceLayout, RuntimePlan, schedule::LayerMap};
use tessera_field::{Goldilocks as G, GoldilocksExt2 as F};
use tessera_gadgets::TableColumns;
use tessera_ir::{ClaimList, StageId};
use tessera_sumcheck::SumcheckTranscript;
use tessera_tables::GeneratedTable;
use tessera_witness::{Engine, Resident, Witness};

struct Unit {
    lowering: Lowering,
    layers: LayerMap,
    runtime: RuntimePlan,
    leaf: LeafLayout<ModelColumn>,
    native: ModelNativePlan,
}
struct Data {
    statement: protocol::PublicStatement,
    unit: Unit,
    witness: Witness,
    seed: random::Seed,
    p1: pcs::AdviceOpening<'static>,
    permit: Option<pcs::ColumnPermit>,
    transcript: Option<Transcript>,
}
pub(super) struct CpuBackend<'a> {
    prepared: Prepared<'a>,
    data: Option<Data>,
}

fn memory() -> ProverError {
    ProverError::Resource(ErrorCode::ResMemory)
}
fn power(n: u64) -> Result<u64, ProverError> {
    n.checked_next_power_of_two().ok_or_else(memory)
}
fn reserve(budget: MemoryBudget, bytes: Option<u64>) -> Result<(), ProverError> {
    if bytes.is_none_or(|n| n > budget.host) {
        return Err(memory());
    }
    Ok(())
}

// shortcut: conservative resident estimates; replace with tile/pass accounting in issue #132.
pub(super) fn check_weight_memory(c: &ClaimList, budget: MemoryBudget) -> Result<(), ProverError> {
    reserve(budget, Some(weight_memory(c)?))
}
fn weight_memory(c: &ClaimList) -> Result<u64, ProverError> {
    let shapes = c
        .arch
        .weight_shapes()
        .map_err(|_| usage("invalid weight shapes"))?;
    let mut entries = 0u64;
    let mut channels = 1;
    for (i, shape) in shapes.iter().enumerate() {
        channels = channels.max(power(u64::from(shape.out))?);
        let layers = if i == 0 || i == 8 {
            1
        } else {
            u64::from(c.arch.n_layers)
        };
        let cells = power(u64::from(shape.inputs))?
            .checked_mul(power(u64::from(shape.out))?)
            .and_then(|n| n.checked_mul(power(layers).ok()?))
            .ok_or_else(memory)?;
        entries = entries.checked_add(cells).ok_or_else(memory)?;
    }
    entries = entries
        .checked_add(
            16u64
                .checked_mul(power(u64::from(c.arch.n_layers))?)
                .and_then(|n| n.checked_mul(channels))
                .ok_or_else(memory)?,
        )
        .ok_or_else(memory)?;
    // Encoded weights, Merkle levels, folding scratch and allocation slack.
    entries.checked_mul(256).ok_or_else(memory)
}

pub(super) fn model_weights<'a>(
    model: &'a IntegerModel,
    lowering: &Lowering,
) -> Result<pcs::ModelWeights<'a>, ProverError> {
    let mut channels = Vec::new();
    for (multipliers, entries) in [(false, &lowering.classes), (true, &lowering.multipliers)] {
        for (w, _) in entries {
            let c = model
                .quant
                .channels
                .iter()
                .find(|c| c.ty == w.ty && c.layer == w.layer)
                .ok_or_else(|| internal("validated channel parameters exist"))?;
            channels.push(pcs::ChannelParamsRef {
                matrix: w.ty.matrix(),
                layer: w.layer,
                values: if multipliers {
                    pcs::ChannelValues::Multipliers(&c.multiplier)
                } else {
                    pcs::ChannelValues::Classes(&c.class)
                },
            });
        }
    }
    Ok(pcs::ModelWeights {
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
    })
}
fn encoder(c: &ClaimList) -> Result<pcs::RowEncoder, ProverError> {
    let a = c.layout.advice;
    Ok(pcs::RowEncoder::new(
        1usize.checked_shl(u32::from(a.log2_k)).ok_or_else(memory)?,
        u32::from(a.log2_inv_rate),
        a.mask_coeffs as usize,
    )?)
}
fn wire(c: pcs::CombinedRow) -> protocol::CombinedRow {
    protocol::CombinedRow {
        u: c.u,
        mask: c.mask,
    }
}

impl<'a> CpuBackend<'a> {
    pub(super) fn new(prepared: Prepared<'a>, batch: &ProofBatch) -> Result<Self, ProverError> {
        let mut backend = Self {
            prepared,
            data: None,
        };
        backend.commit_batch(batch)?;
        Ok(backend)
    }
    pub(super) fn session(&self) -> Result<[u8; 16], ProverError> {
        Ok(self
            .data
            .as_ref()
            .ok_or_else(|| internal("P1 has a fresh seed"))?
            .seed
            .session())
    }
    fn unit(&self, batch: &ProofBatch) -> Result<Unit, ProverError> {
        let c = &self.prepared.plan.claims;
        let s = &batch.statement;
        if s.claim_list_digest
            != tessera_hash::hash(tessera_hash::Domain::ClaimList, &tessera_codec::encode(c))
            || s.weights_root != self.prepared.weights.weights_root()
            || s.weights_epoch != self.prepared.weights.public().epoch
            || s.predictor
                != (protocol::PredictorSpec::Exact {
                    decoding: protocol::Decoding::Greedy,
                })
            || s.batch.requests.len() != 1
        {
            return Err(usage(
                "one exact-greedy request must bind this claim list and weight epoch",
            ));
        }
        let r = &s.batch.requests[0];
        let n = r
            .prompt_len
            .checked_add(r.output_len)
            .ok_or_else(|| usage("request length overflow"))?;
        if r.prompt_len == 0 || r.output_len == 0 || n > c.arch.max_context {
            return Err(usage(
                "request requires a nonempty prompt and continuation within max_context",
            ));
        }
        let lowering = compiler::lower(PublicModel::of(self.prepared.model), n)?;
        let ranges = compiler::analyze(&lowering.graph, &lowering.tables)?;
        let selected =
            compiler::select(&lowering.graph, &ranges, &lowering.tables, &c.gadget_policy)?;
        let runtime = RuntimePlan::derive(&lowering, &selected, c, r.prompt_len, 0)?;
        let layers = LayerMap::new(&lowering)?;
        let leaf = LeafLayout::runtime(&lowering, &layers, &runtime, c)?;
        let native = ModelNativePlan::derive_runtime(&lowering, &runtime, c, &leaf)?;
        // Bound the dense reference implementation before loading tables or secret witness data.
        let mut cells = 0u64;
        for t in &lowering.graph.tensors {
            let padded = t
                .shape
                .iter()
                .try_fold(1u64, |p, &d| p.checked_mul(power(u64::from(d)).ok()?));
            cells = cells
                .checked_add(padded.ok_or_else(memory)?)
                .ok_or_else(memory)?;
        }
        for i in leaf.instances() {
            let table = runtime
                .selection()
                .tables
                .iter()
                .find(|t| t.id == i.table)
                .ok_or_else(|| internal("leaf table is selected"))?;
            cells = cells
                .checked_add(power(i.entries)?)
                .and_then(|n| n.checked_add(1u64.checked_shl(u32::from(table.log2_rows))?))
                .ok_or_else(memory)?;
        }
        let advice = ModelAdviceLayout::runtime(&lowering, &runtime, c)?;
        cells = cells
            .checked_add(
                advice
                    .layout()
                    .rows()
                    .checked_mul(1u64 << c.layout.advice.log2_k)
                    .ok_or_else(memory)?,
            )
            .ok_or_else(memory)?;
        // Multiple live column/forest/relation representations and proof-message slack.
        reserve(
            self.prepared.cfg.memory_budget,
            cells
                .checked_mul(4096)
                .and_then(|n| n.checked_add(weight_memory(c).ok()?)),
        )?;
        Ok(Unit {
            lowering,
            layers,
            runtime,
            leaf,
            native,
        })
    }
    pub(super) fn round2(
        &mut self,
        tr: &mut Transcript,
    ) -> Result<(protocol::Round2Response, protocol::QueryShape), ProverError> {
        let p = &self.prepared;
        let d = self
            .data
            .as_mut()
            .ok_or_else(|| internal("P1 precedes proof computation"))?;
        let u = &d.unit;
        let c = &p.plan.claims;
        let advice = ModelAdviceLayout::runtime(&u.lowering, &u.runtime, c)?;
        let engine = Engine::new(&u.lowering)?.threads(p.threads);
        let source = Resident::runtime(&engine, &u.runtime, p.model, &d.witness)?;
        let actual = ModelWitness::new(
            &u.leaf,
            &u.lowering,
            &u.layers,
            &d.witness,
            &source,
            &advice,
        )?;
        let mut xi: [Vec<F>; 10] = core::array::from_fn(|_| Vec::new());
        for m in pcs::WeightMatrix::ALL {
            let rows = p.layout.matrix(m).rows();
            if rows > 0 {
                xi[m.index()] = (0..rows.next_power_of_two().trailing_zeros())
                    .map(|_| tr.challenge(&format!("pcs/weights/{m:?}/xi")))
                    .collect();
            }
        }
        let tables: Vec<TableColumns<G>> = u
            .leaf
            .instances()
            .iter()
            .map(|i| {
                let spec = u
                    .runtime
                    .selection()
                    .tables
                    .iter()
                    .find(|t| t.id == i.table)
                    .ok_or_else(|| internal("leaf table exists"))?;
                let generated = GeneratedTable::from_spec(spec)
                    .map_err(|_| internal("selected table regenerates"))?;
                TableColumns::new(&generated)
                    .map_err(|e| ProverError::Leaf(crate::LeafError::Gadget(e)))
            })
            .collect::<Result<_, _>>()?;
        let tensors = actual.tensors();
        let forests = actual.forests();
        let lookup = prepare_leaf(tr, &u.leaf, &tables, &tensors, &forests)?;
        let mut stages = crate::StageProver::new(&u.lowering, u.native.schedule(), &d.witness);
        let points: Vec<_> = u.native.schedule().stages[0]
            .claims
            .iter()
            .map(|c| lookup.point()[..c.point.len()].to_vec())
            .collect();
        let _graph_leaf = stages.leaf(&points)?;
        let leaf = lookup.finish(&stages.leaf_split_values()?)?;
        let mut native =
            ModelNativeProver::new(&u.native, &u.lowering, &actual, &d.witness, &leaf)?;
        let mut proofs = Vec::new();
        for k in 1..u.native.schedule().stages.len() {
            let result = stages.prove_stage_composed(tr, k as u16, &mut native)?;
            proofs.push(protocol::StageProof {
                stage: StageId(result.stage),
                sumcheck: result.sumcheck,
                splits: result.splits,
            });
        }
        let targets = crate::ModelOpenings::derive(
            &u.lowering,
            &u.leaf,
            &leaf,
            &u.native,
            &stages,
            &advice,
            &p.layout,
        )?;
        let gamma = tr.challenge("pcs/advice/gamma");
        let av: Vec<_> = (0..advice.layout().row_vars())
            .map(|_| tr.challenge("pcs/advice/xi"))
            .collect();
        let mut theta: [Option<F>; 10] = [None; 10];
        for m in pcs::WeightMatrix::ALL {
            if !targets.weight_points(m).is_empty() && p.layout.matrix(m).rows() > 0 {
                theta[m.index()] = Some(tr.challenge(&format!("pcs/weights/{m:?}/gamma")));
            }
        }
        let encoder = encoder(c)?;
        let prepared = d.p1.prepare(targets.advice(), gamma, &av)?;
        let mut masks = d.seed.masks();
        let mut pass = prepared.begin(&encoder, &mut masks)?;
        for segment in 0..c.layout.advice.segments {
            crate::advice::visit_rows(&u.lowering, &advice, &source, segment, |r, v| {
                pass.push_row(r, v).map_err(Into::into)
            })?;
        }
        let (combined, permit) = pass.finish()?.into_parts();
        let advice_combined: Vec<_> = combined.into_iter().map(wire).collect();
        tr.absorb("pcs/advice/combined", &advice_combined);
        let weights = model_weights(p.model, &u.lowering)?;
        let mut weight_combined = Vec::new();
        let mut matrices = Vec::new();
        for m in pcs::WeightMatrix::ALL {
            let Some(theta) = theta[m.index()] else {
                continue;
            };
            let full = targets.weight_points(m);
            let matrix = p.layout.matrix(m);
            let mut reader = p.layout.rows(m, &weights)?;
            let layer;
            let fold = if m == pcs::WeightMatrix::ChannelParams {
                layer = Vec::new();
                let points = full
                    .iter()
                    .map(|p| &p[matrix.log2_k() as usize..])
                    .collect::<Vec<_>>();
                pcs::WeightFold::channel_params(&p.layout, &mut reader, &points, &xi[m.index()])?
            } else {
                if full.iter().any(|point| point != &full[0]) {
                    return Err(internal("ordinary matrix aliases share one point"));
                }
                let pcs::MatrixKind::Weight { shape, a, h, .. } = matrix.kind() else {
                    return Err(internal("ordinary matrix kind"));
                };
                let input = shape.inputs.next_power_of_two().trailing_zeros() as usize;
                let output = (a + h) as usize;
                layer = full[0][input + output..].to_vec();
                pcs::WeightFold::new(
                    &p.layout,
                    m,
                    &mut reader,
                    &full[0][input..input + output],
                    &xi[m.index()],
                )?
            };
            let combined = fold.combined_rows(&layer, theta)?;
            let packet = protocol::WeightCombinedRows {
                matrix: m.index() as u8,
                evaluations: combined.evaluations.into_iter().map(wire).collect(),
                proximity: wire(combined.proximity),
            };
            tr.absorb(&format!("pcs/weights/{m:?}/combined"), &packet);
            weight_combined.push(packet);
            matrices.push(protocol::MatrixCodeword {
                matrix: m.index() as u8,
                len: 1u64 << (matrix.log2_k() + u32::from(c.layout.weights.log2_inv_rate)),
            });
        }
        let scoring = protocol::ScoringClaims {
            requests: vec![protocol::FixedBits(0)],
        };
        tr.absorb("scoring", &scoring);
        let round2 = protocol::Round2Response {
            logup: protocol::LogUpProof {
                roots: leaf.logup().roots().to_vec(),
                values: leaf.logup().values().to_vec(),
            },
            trees: protocol::TreeProof {
                levels: leaf
                    .logup()
                    .levels()
                    .iter()
                    .map(|p| p.proof().clone())
                    .collect(),
                splits: leaf.splits().to_vec(),
            },
            stages: proofs,
            advice_combined,
            weight_combined,
            scoring,
        };
        let shape = protocol::QueryShape {
            advice_count: p.cfg.params.advice_queries,
            advice_len: encoder.codeword_len() as u64,
            matrix_count: p.cfg.params.weight_queries,
            matrices,
        };
        d.permit = Some(permit);
        Ok((round2, shape))
    }
}

impl CpuBackend<'_> {
    fn commit_batch(&mut self, batch: &ProofBatch) -> Result<(), ProverError> {
        let unit = self.unit(batch)?;
        let c = &self.prepared.plan.claims;
        let engine = Engine::new(&unit.lowering)?.threads(self.prepared.threads);
        let witness = engine.run(self.prepared.model, &batch.tokens)?;
        let source = Resident::runtime(&engine, &unit.runtime, self.prepared.model, &witness)?;
        let advice = ModelAdviceLayout::runtime(&unit.lowering, &unit.runtime, c)?;
        let id = pcs::CommitmentDescriptor::Advice {
            statement_digest: batch.statement.digest(),
            unit: 0,
            layout: c.layout.advice,
            rows: advice.layout().rows(),
        }
        .commitment_id();
        let encoder = encoder(c)?;
        let seed = random::Seed::new(self.prepared.cfg.mask_rng)?;
        let mut masks = seed.masks();
        let a = c.layout.advice;
        let params = pcs::CommitmentParams {
            log2_k: u32::from(a.log2_k),
            log2_inv_rate: u32::from(a.log2_inv_rate),
            mask_coeffs: a.mask_coeffs as usize,
            row_block: NonZeroUsize::new(a.row_block as usize)
                .ok_or_else(|| usage("zero row block"))?,
            commitment_id: id,
        };
        let mut parts = Vec::new();
        for segment in 0..a.segments {
            let mut pass = pcs::StreamingCommitter::new(
                params,
                advice.layout(),
                segment,
                &encoder,
                &mut masks,
            )?;
            crate::advice::visit_rows(&unit.lowering, &advice, &source, segment, |r, v| {
                pass.push_row(r, v).map_err(Into::into)
            })?;
            parts.push(pass.finish()?);
        }
        let p1 = pcs::AdviceOpening::new(&parts)?.into_owned();
        self.data = Some(Data {
            statement: batch.statement.clone(),
            unit,
            witness,
            seed,
            p1,
            permit: None,
            transcript: None,
        });
        Ok(())
    }
}

impl ProverEngine for CpuBackend<'_> {
    type Error = ProverError;
    fn commit(
        &mut self,
        _session_id: &[u8; 16],
    ) -> Result<(protocol::PublicStatement, protocol::AdviceCommitment), ProverError> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| internal("P1 was prepared for this one-shot session"))?;
        Ok((data.statement.clone(), data.p1.commitment()))
    }
    fn prover_share(&mut self) -> Result<[u8; 32], ProverError> {
        Ok(self
            .data
            .as_ref()
            .ok_or_else(|| internal("P1 has a per-proof seed"))?
            .seed
            .share())
    }
    fn prove(
        &mut self,
        round1: &protocol::Round1,
        seed: &tessera_hash::Digest,
    ) -> Result<(protocol::Round2Response, protocol::QueryShape), ProverError> {
        let mut tr = transcript(Mode::Live, self.prepared.cfg.params, round1)?;
        tr.absorb("seed", seed);
        let answer = self.round2(&mut tr)?;
        self.data
            .as_mut()
            .ok_or_else(|| internal("P4 retains its P1"))?
            .transcript = Some(tr);
        Ok(answer)
    }
    fn answer(
        &mut self,
        queries: &protocol::ColumnQueries,
    ) -> Result<protocol::Round3Response, ProverError> {
        // Taking Data clears every per-proof allocation on both success and error.
        let mut d = self
            .data
            .take()
            .ok_or_else(|| internal("P1 precedes column answers"))?;
        // Live queries are absorbed even though P5 draws no further challenges.
        if self.prepared.cfg.mode == Mode::Live {
            d.transcript
                .as_mut()
                .ok_or_else(|| internal("Live P5 retains its Round2 transcript"))?
                .absorb("queries", queries);
        }
        let p = &self.prepared;
        let c = &p.plan.claims;
        let u = &d.unit;
        let engine = Engine::new(&u.lowering)?.threads(p.threads);
        let source = Resident::runtime(&engine, &u.runtime, p.model, &d.witness)?;
        let advice = ModelAdviceLayout::runtime(&u.lowering, &u.runtime, c)?;
        let encoder = encoder(c)?;
        let permit = d
            .permit
            .as_ref()
            .ok_or_else(|| internal("P4 precedes column answers"))?;
        let mut masks = d.seed.masks();
        let indices = queries
            .advice
            .iter()
            .map(|&i| u64::from(i))
            .collect::<Vec<_>>();
        let mut pass = d.p1.begin_columns(permit, &encoder, &mut masks, &indices)?;
        for segment in 0..c.layout.advice.segments {
            crate::advice::visit_rows(&u.lowering, &advice, &source, segment, |r, v| {
                pass.push_row(r, v).map_err(Into::into)
            })?;
        }
        let answer = pass.finish()?;
        let mut weight_columns = Vec::new();
        for q in &queries.weights {
            let m = pcs::WeightMatrix::ALL
                .get(q.matrix as usize)
                .ok_or_else(|| internal("checked query matrix"))?;
            let indices = q.indices.iter().map(|&i| u64::from(i)).collect::<Vec<_>>();
            let a = p.weights.matrix(*m).answer(&indices)?;
            weight_columns.push(protocol::MatrixColumns {
                matrix: q.matrix,
                columns: a.columns,
                proof: protocol::MultiProof {
                    siblings: a.proof.siblings,
                },
            });
        }
        Ok(protocol::Round3Response {
            advice_columns: answer.columns,
            advice_proof: protocol::MultiProof {
                siblings: answer.proof.siblings,
            },
            weight_columns,
        })
    }
    fn discard(&mut self) {
        self.data = None;
    }
}
