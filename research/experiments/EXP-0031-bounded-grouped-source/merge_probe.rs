// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Current merge kernels over statelessly regenerated original synthetic sources.

use serde_json::{Value, json};
use std::cell::Cell;
use tessera_codec::{Secret, SecretBuf};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_poly::evaluate_base;
use tessera_protocol::LeafGroups;
use tessera_prover::leaf::{GroupedLayout, GroupedMessageMemory, GroupedMessageShape};
use tessera_sumcheck::{
    Merge, MergeClaim, MergeMemory, MergePolynomial, MergeTables, ProvenBatch, SumcheckTranscript,
    largest_table_rounds, prove_merge, prove_merge_two_pass,
};

use crate::adapter::insert;
use crate::fixture::{self, Fixture, ForestGeometry};
use crate::memory::{add, growing};

fn point(h: usize, coordinate: usize) -> F {
    let word = fixture::splitmix(fixture::SEED ^ ((h as u64) << 32) ^ coordinate as u64);
    F::new(
        G::from_u64_reduced(word),
        G::from_u64_reduced(fixture::splitmix(word)),
    )
}

/// Fixed-state test transcript: no callback allocations enter kernel measurements.
struct Recording {
    hash: blake3::Hasher,
}
impl Recording {
    fn new() -> Self {
        let mut hash = blake3::Hasher::new();
        hash.update(b"EXP-0031 synthetic merge transcript");
        Self { hash }
    }
    fn digest(&self) -> String {
        self.hash.finalize().to_hex().to_string()
    }
}
impl SumcheckTranscript<F> for Recording {
    fn absorb_message(&mut self, label: &str, message: &[F]) {
        self.hash.update(&(label.len() as u64).to_le_bytes());
        self.hash.update(label.as_bytes());
        self.hash.update(&(message.len() as u64).to_le_bytes());
        for value in message {
            for coefficient in value.coeffs() {
                self.hash
                    .update(&coefficient.to_canonical_u64().to_le_bytes());
            }
        }
    }
    fn challenge(&mut self, label: &str) -> F {
        self.absorb_message(label, &[]);
        let bytes = self.hash.finalize();
        let a = u64::from_le_bytes(
            bytes.as_bytes()[..8]
                .try_into()
                .expect("eight public hash bytes"),
        );
        let b = u64::from_le_bytes(
            bytes.as_bytes()[8..16]
                .try_into()
                .expect("eight public hash bytes"),
        );
        F::new(G::from_u64_reduced(a), G::from_u64_reduced(b))
    }
}

struct Factory<'a> {
    fixture: &'a Fixture,
    forest: &'a ForestGeometry,
    sources: Vec<(usize, usize, u8)>,
    full_calls: Cell<usize>,
}
impl<'a> Factory<'a> {
    fn new(fixture: &'a Fixture, forest: &'a ForestGeometry) -> Self {
        let mut sources = Vec::new();
        for (s, site) in fixture.stage.sites().iter().enumerate() {
            for l in 0..site.layers.len() {
                for read in fixture.stage.reads(s, l).expect("source read registry") {
                    sources.push((s, l, read.term));
                }
            }
        }
        Self {
            fixture,
            forest,
            sources,
            full_calls: Cell::new(0),
        }
    }
    fn datum(&self, polynomial: usize, index: u64) -> G {
        if let Some(&(s, l, t)) = self.sources.get(polynomial) {
            G::from_i64(self.fixture.datum(s, l, t, index))
        } else {
            assert_eq!(
                polynomial,
                self.sources.len(),
                "registered forest polynomial"
            );
            self.forest.datum(self.fixture, index)
        }
    }
    fn values(&self, i: usize, shape: &MergePolynomial) -> SecretBuf<G> {
        self.full_calls.set(self.full_calls.get() + 1);
        let n = 1usize << shape.vars;
        let mut values = SecretBuf::try_with_capacity(n).expect("finite source capacity");
        for j in 0..n {
            values.push(self.datum(i, j as u64));
        }
        values
    }
    fn table(&self, i: usize, shape: &MergePolynomial) -> Secret<Vec<G>> {
        // The clearing source and the returned owned copy coexist here.
        Secret::new(self.values(i, shape).to_vec())
    }
    fn block(&self, i: usize, shape: &MergePolynomial, bits: u64) -> Secret<Vec<G>> {
        Secret::new(self.block_values(i, shape, bits).to_vec())
    }
    fn block_values(&self, i: usize, shape: &MergePolynomial, bits: u64) -> SecretBuf<G> {
        let mut values =
            SecretBuf::try_with_capacity(1 << shape.free_vars()).expect("finite group block");
        for j in 0..1u64 << shape.free_vars() {
            let x = insert(j, shape.vars as u32, |v| {
                shape
                    .group_vars
                    .iter()
                    .position(|&p| p == v as usize)
                    .map(|p| (bits >> p) & 1)
            });
            values.push(self.datum(i, x));
        }
        values
    }
}

fn same(a: &ProvenBatch<F>, b: &ProvenBatch<F>) {
    assert!(
        a.proof().rounds == b.proof().rounds,
        "complete merge round messages"
    );
    assert!(
        a.proof().final_evals == b.proof().final_evals,
        "complete merge evaluations"
    );
    assert!(a.point() == b.point(), "complete merge point");
}

pub fn run() -> Value {
    let mut cases = Vec::new();
    for id in 0..fixture::LENGTHS.len() {
        let f = Fixture::new(id);
        for height in [1, 2] {
            let forest = ForestGeometry::new(&f, height);
            let factory = Factory::new(&f, &forest);
            for (gt, gl) in fixture::GROUPS {
                if u32::from(gt) > fixture::bits(f.tokens) {
                    continue;
                } // Explicit negatives live in source stage.
                let grouped = GroupedLayout::new(
                    &f.stage,
                    &f.layers,
                    LeafGroups {
                        token_bits: gt,
                        layer_bits: gl,
                    },
                )
                .expect("registered legal grouping");
                let mut shapes = grouped.polynomials().to_vec();
                assert_eq!(shapes.len(), factory.sources.len(), "source identity order");
                shapes.push(MergePolynomial {
                    vars: forest.vars as usize,
                    group_vars: (forest.queries..forest.queries + u32::from(gt))
                        .chain(forest.layers..forest.layers + u32::from(gl))
                        .map(|x| x as usize)
                        .collect(),
                });
                let n_max = shapes
                    .iter()
                    .map(|p| p.vars)
                    .max()
                    .expect("polynomial shapes");
                let full_max = shapes
                    .iter()
                    .map(|p| 1u64 << p.vars)
                    .max()
                    .expect("source size");
                let group_max = shapes
                    .iter()
                    .map(|p| 1u64 << p.free_vars())
                    .max()
                    .expect("group source size");
                let mut message_vars = n_max.max(4); // Both synthetic public tables have 16 rows.
                let mut group_layout_bill = 0;
                for h in 0..grouped.count() {
                    group_layout_bill = group_layout_bill.max(
                        grouped
                            .group_memory(h)
                            .expect("public group layout bill")
                            .peak_bytes(),
                    );
                    let layout = grouped.group(h).expect("public message tree geometry");
                    for instance in layout.instances() {
                        message_vars = message_vars.max(instance.height as usize);
                    }
                }
                let driver = GroupedMessageMemory::new(GroupedMessageShape {
                    groups: grouped.count() as u64,
                    lookups: 2,
                    forests: 1,
                    tensors: factory.sources.len() as u64,
                    max_vars: message_vars as u32,
                    atom_splits: 0,
                })
                .expect("public driver metadata")
                .driver_bytes();
                let input_bill = add(
                    group_layout_bill,
                    add(
                        driver,
                        add(
                            growing::<G>(group_max)
                                .expect("restricted group source evaluation input"),
                            add(
                                growing::<F>(group_max).expect("restricted evaluation scratch"),
                                growing::<F>((2 * n_max) as u64).expect("points"),
                            )
                            .expect("evaluation temporaries"),
                        )
                        .expect("input scratch"),
                    )
                    .expect("input phase"),
                )
                .expect("group layout beside merge inputs");
                let full_before_group = factory.full_calls.get();
                let mut merge = None;
                let preparation = allocation_counter::measure(|| {
                    let mut claims = Vec::with_capacity(grouped.count());
                    let mut gammas = Vec::with_capacity(grouped.count());
                    for h in 0..grouped.count() {
                        let layout = grouped.group(h).expect("group layout");
                        let mut group = Vec::with_capacity(layout.reads().len() + 1);
                        for read in layout.reads() {
                            let p = &shapes[read.polynomial];
                            let original =
                                f.read(read.site, read.layer, factory.sources[read.polynomial].2);
                            let lookup = f.stage.sites()[read.site].layers[read.layer]
                                .lookups
                                .iter()
                                .find(|l| {
                                    l.lookup
                                        .key
                                        .terms
                                        .iter()
                                        .chain(l.lookup.value.iter().flat_map(|v| &v.terms))
                                        .any(|&(_, t)| t == original.term)
                                })
                                .expect("original read domain");
                            let fixed = lookup.token.map(|axis| {
                                lookup.domain[axis + 1..]
                                    .iter()
                                    .map(|&n| fixture::bits(n))
                                    .sum::<u32>()
                            });
                            let mut coordinate = 0;
                            let free: Vec<F> = original
                                .coords
                                .iter()
                                .filter(|&&c| {
                                    !fixed.is_some_and(|p| c >= p && c < p + u32::from(gt))
                                })
                                .map(|&c| {
                                    let c = c - fixed
                                        .map_or(0, |p| u32::from(gt).min(c.saturating_sub(p)));
                                    assert_eq!(
                                        Some(&c),
                                        read.coords.get(coordinate),
                                        "split coordinate order"
                                    );
                                    coordinate += 1;
                                    point(h, c as usize)
                                })
                                .collect();
                            assert_eq!(
                                coordinate,
                                read.coords.len(),
                                "all free split coordinates transported"
                            );
                            assert_eq!(free.len(), p.free_vars(), "free coordinate dimension");
                            let values = factory.block_values(read.polynomial, p, read.group_bits);
                            let value = evaluate_base::<F>(&values, &free)
                                .expect("source split evaluation");
                            group.push(MergeClaim {
                                polynomial: read.polynomial,
                                point: free,
                                group_bits: read.group_bits,
                                value,
                            });
                        }
                        let i = shapes.len() - 1;
                        let p = &shapes[i];
                        let free: Vec<_> = (0..p.free_vars()).map(|c| point(h, c)).collect();
                        let values = factory.block_values(i, p, h as u64);
                        let value =
                            evaluate_base::<F>(&values, &free).expect("forest split evaluation");
                        group.push(MergeClaim {
                            polynomial: i,
                            point: free,
                            group_bits: h as u64,
                            value,
                        });
                        claims.push(group);
                        gammas.push(point(h, n_max + 1));
                    }
                    merge = Some(
                        Merge::new(
                            shapes.clone(),
                            claims,
                            gammas,
                            point(grouped.count(), n_max + 2),
                        )
                        .expect("canonical synthetic merge"),
                    );
                });
                assert!(
                    preparation.bytes_max <= input_bill,
                    "merge input and split scratch bill"
                );
                assert_eq!(
                    factory.full_calls.get(),
                    full_before_group,
                    "no full original source during group pass"
                );
                let merge = merge.expect("retained merge inputs");
                // Materialized full values are an oracle only, outside candidate counters.
                let oracle_values: Vec<_> = shapes
                    .iter()
                    .enumerate()
                    .map(|(i, p)| factory.values(i, p))
                    .collect();
                for claims in merge.claims() {
                    for claim in claims {
                        let at = natural_point(
                            &shapes[claim.polynomial],
                            &claim.point,
                            claim.group_bits,
                        );
                        let expected = evaluate_base::<F>(&oracle_values[claim.polynomial], &at)
                            .expect("full polynomial reference evaluation");
                        assert!(
                            claim.value == expected,
                            "split/free/full coordinate transport"
                        );
                    }
                }
                let input_actual = preparation.bytes_current;
                assert!(input_actual >= 0, "retained input allocation");
                let registry: Vec<Vec<usize>> = merge
                    .claims()
                    .iter()
                    .map(|g| g.iter().map(|c| c.polynomial).collect())
                    .collect();
                let baseline_tables: Vec<_> = shapes
                    .iter()
                    .enumerate()
                    .map(|(i, p)| factory.table(i, p))
                    .collect();
                let references: Vec<_> = baseline_tables.iter().collect();
                let mut baseline_tr = Recording::new();
                let baseline = prove_merge(&mut baseline_tr, &merge, &references)
                    .expect("materialized reference merge");
                let mut cuts = Vec::new();
                for cut in 0..=largest_table_rounds(&shapes) {
                    let memory = MergeMemory::for_two_pass::<F>(&shapes, &registry, cut)
                        .expect("public merge geometry");
                    type Block = (usize, u64, Secret<Vec<G>>);
                    let mut block_bill = 0;
                    for group in merge.claims() {
                        let mut bytes =
                            growing::<Block>(group.len() as u64).expect("callback metadata");
                        for claim in group {
                            bytes = add(
                                bytes,
                                growing::<G>(1u64 << shapes[claim.polynomial].free_vars())
                                    .expect("caller block and owned-copy overlap"),
                            )
                            .expect("caller blocks");
                        }
                        block_bill = block_bill.max(bytes);
                    }
                    let construct_bill = add(memory.construction_peak_bytes(), block_bill)
                        .expect("construction overlap");
                    let mut kept = None;
                    let constructing = allocation_counter::measure(|| {
                        let mut tables = MergeTables::new(&shapes, grouped.count(), cut)
                            .expect("merge table constructor");
                        for (h, claims) in merge.claims().iter().enumerate() {
                            let blocks: Vec<Block> = claims
                                .iter()
                                .map(|c| {
                                    (
                                        c.polynomial,
                                        c.group_bits,
                                        factory.block(
                                            c.polynomial,
                                            &shapes[c.polynomial],
                                            c.group_bits,
                                        ),
                                    )
                                })
                                .collect();
                            tables
                                .add_group(h, claims, merge.gammas()[h], |i, b| {
                                    blocks
                                        .iter()
                                        .find(|(p, s, _)| (*p, *s) == (i, b))
                                        .map(|(_, _, v)| v)
                                })
                                .expect("bounded group table accumulation");
                        }
                        kept = Some(tables);
                    });
                    assert!(
                        constructing.bytes_max <= construct_bill,
                        "kept tables and callback blocks overlap"
                    );
                    assert!(
                        constructing.bytes_current <= memory.table_bytes() as i64,
                        "retained merge tables"
                    );
                    let kept = kept.expect("retained group tables");
                    let callback_bill =
                        growing::<G>(full_max).expect("original P2b regeneration/copy");
                    let proving_bill =
                        add(memory.proving_peak_bytes(), callback_bill).expect("P2b overlap");
                    let mut calls = vec![0usize; shapes.len()];
                    let mut tr = Recording::new();
                    let mut candidate = None;
                    let full_before_p2b = factory.full_calls.get();
                    let proving = allocation_counter::measure(|| {
                        candidate = Some(
                            prove_merge_two_pass(&mut tr, &merge, &kept, |i| {
                                calls[i] += 1;
                                factory.table(i, &shapes[i])
                            })
                            .expect("two-pass candidate merge"),
                        );
                    });
                    assert!(
                        proving.bytes_max <= proving_bill,
                        "P2b engine and original callback source overlap"
                    );
                    assert!(
                        proving.bytes_current <= memory.output_bytes() as i64,
                        "retained merge output"
                    );
                    assert!(
                        calls.iter().all(|&n| n == 1),
                        "one regeneration per original polynomial"
                    );
                    assert_eq!(
                        factory.full_calls.get() - full_before_p2b,
                        shapes.len(),
                        "only P2b regenerates each full original once"
                    );
                    same(&baseline, candidate.as_ref().expect("merge output"));
                    assert_eq!(
                        baseline_tr.digest(),
                        tr.digest(),
                        "every transcript label and message"
                    );
                    cuts.push(json!({"cut":cut,"construction_bill":construct_bill,"construction_observed":constructing.bytes_max,"kept_bill":memory.table_bytes(),"kept_observed":constructing.bytes_current,"proving_bill":proving_bill,"proving_observed":proving.bytes_max,"callback_bill":callback_bill,"block_bill":block_bill,"engine_construction":memory.construction_peak_bytes(),"engine_proving":memory.proving_peak_bytes(),"output_bill":memory.output_bytes(),"regeneration_calls":calls,"transcript_digest":tr.digest()}));
                }
                cases.push(json!({"fixture":id,"height":height,"gt":gt,"gl":gl,"shapes":shapes.iter().map(|p|json!({"vars":p.vars,"group_vars":p.group_vars})).collect::<Vec<_>>(),"claims":registry,"input_bill":input_bill,"input_observed_peak":preparation.bytes_max,"input_retained":input_actual,"driver_bill":driver,"message_vars":message_vars,"group_layout_bill":group_layout_bill,"full_source_calls_during_group":0,"cuts":cuts}));
            }
        }
    }
    json!({"stage":"merge-prototype-only","verdict":"not judged","registered_criteria_complete":false,"production_admission":false,"real_model_acceptance":false,"cases":cases,"scope":"finite current merge-kernel equivalence and requested allocations; borrowed factory/public descriptors and materialized oracle are additional, full P1-P5 composition remains unimplemented"})
}

fn natural_point(shape: &MergePolynomial, free: &[F], bits: u64) -> Vec<F> {
    let mut at = Vec::with_capacity(shape.vars);
    let mut next = 0;
    for v in 0..shape.vars {
        at.push(match shape.group_vars.iter().position(|&s| s == v) {
            Some(j) => {
                if bits >> j & 1 == 1 {
                    F::ONE
                } else {
                    F::ZERO
                }
            }
            None => {
                let x = free[next];
                next += 1;
                x
            }
        });
    }
    assert_eq!(
        next,
        free.len(),
        "full point consumes every free coordinate"
    );
    at
}
