// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A graph's lookups in the real leaf stage (RFC-0072 'Gadgets and leaf sites' 'AES's lookups',
//! 'Public tensors' 'In the leaf stage', 'Padded positions'; RFC-0045; #526's fourth slice):
//! `TableMap` and `Bitwise` sites of a graph without a lowering, laid out by
//! `LeafLayout::graph`, their tensors over their padded shapes lent by `GraphLeafWitness` from a
//! `PaddedWitness`, proven by `tessera-logup` and the split, then by the stage prover.
//!
//! **The verifier** is this module's, written from RFC-0045, RFC-0008 and RFC-0072. It derives
//! the sites, blocks, offsets, heights and reads from the graph and the tables alone; runs the
//! production shared-tree driver (`tv_core::leaf`) with the verifier's tables (`tv-tables`) on its
//! own transcript (`tv-hash`); checks every instance's key and value claims against the split by
//! RFC-0045's identity, evaluating each public read itself at its point, its table padded with
//! zeros; derives each leaf claim's value from the split or from public data; and continues with
//! the stage verifier of [`super`] on the same transcript. The openings' part, which no message
//! here carries, is the test's: the pool, the lookup outputs' claims and the multiplicities'
//! claims must be the committed tables' extensions, and the prover's opening targets
//! (`GraphOpenings`) must be exactly those claims.
//!
//! - **Rejections.** A lookup off its table, and a padded block whose S-box output is taken as 0,
//!   fail LogUp's roots, `Reject(Lookup { table })` (RFC-0008); a changed split value, and a
//!   public input changed on the verifier's side, fail the split's identity, `Reject(Sumcheck {
//!   stage: 0, round: 0 })` (RFC-0045 'The check'); multiplicities that miss a block's padded
//!   positions fail LogUp's roots.
//! - **Multiplicities.** Each instance's multiplicities sum to its column's `2^h` entries: every
//!   block's padded positions are counted, beside the column's row-0 tail.

use super::padding::PublicOf;
use super::*;
use tessera_compiler::AdviceSource;
use tessera_compiler::{Selection, analyze, select, settle};
use tessera_gadgets::TableColumns;
use tessera_ir::{GadgetPolicy, NodeId};
use tessera_logup::{LogUpInstance, ProvenLogUp, Tuples, prove_logup};
use tessera_prover::GraphOpenings;
use tessera_prover::leaf::{
    GraphLeafWitness, LeafColumns, LeafLayout, ModelColumn, SPLIT_LABEL, leaf_point, prepare_leaf,
    split_values,
};
use tv_core::logup::LookupError;
use tv_core::sumcheck as tv;

type TvFp = tv_field::Fp;
type TvFp2 = tv_field::Fp2;

// ----- The verifier's field and transcript ------------------------------------------------------

pub(super) fn tv_fp(x: Goldilocks) -> TvFp {
    TvFp::from_canonical(x.to_canonical_u64()).expect("a canonical element")
}

pub(super) fn tv_fp2(x: F) -> TvFp2 {
    let [c0, c1] = x.coeffs();
    TvFp2::new(tv_fp(c0), tv_fp(c1))
}

/// A verifier's element in the prover's field, for the stage verifier of [`super`].
pub(super) fn fp2(x: TvFp2) -> F {
    let bytes = x.to_le_bytes();
    let half = |at: usize| {
        let mut word = [0u8; 8];
        word.copy_from_slice(&bytes[at..at + 8]);
        Goldilocks::from_u64_reduced(u64::from_le_bytes(word))
    };
    F::new(half(0), half(8))
}

/// The verifier's own transcript (`tv-hash`) on the session of [`super::transcript`], lent to the
/// stage verifier of [`super`]: each message is absorbed as the TCE `Vec<Fp2>` of RFC-0003, its
/// length as a little-endian `u64`, then each element's 16 bytes.
pub(super) struct Tv(pub(super) tv_hash::transcript::Transcript);

impl Tv {
    pub(super) fn new() -> Self {
        use tv_hash::transcript as t;
        Self(t::Transcript::new(&t::SessionContext {
            protocol: t::Ascii::new("tessera-protocol/1").expect("ASCII"),
            mode: t::Mode::NonInteractive,
            params: t::ParamSetId(t::Ascii::new("test-small/1").expect("ASCII")),
            statement_digest: [0x13; 32],
            session_id: [0x37; 16],
        }))
    }

    /// Absorbs `message` under `label`.
    fn absorb_tv(&mut self, label: &str, message: &[TvFp2]) {
        let mut tce = Vec::with_capacity(8 + 16 * message.len());
        tce.extend_from_slice(&(message.len() as u64).to_le_bytes());
        for x in message {
            tce.extend_from_slice(&x.to_le_bytes());
        }
        let label = tv_hash::transcript::Ascii::new(label).expect("ASCII");
        self.0.absorb(label, &tce);
    }
}

impl SumcheckTranscript<F> for Tv {
    fn absorb_message(&mut self, label: &str, message: &[F]) {
        let message: Vec<TvFp2> = message.iter().copied().map(tv_fp2).collect();
        self.absorb_tv(label, &message);
    }

    fn challenge(&mut self, label: &str) -> F {
        let label = tv_hash::transcript::Ascii::new(label).expect("ASCII");
        fp2(self.0.challenge_fp2(label).expect("a challenge"))
    }
}

/// `f̃(point)` of `2^{|point|}` integers in the centered representation, by `eq`'s weights.
pub(super) fn mle(values: &[i64], point: &[TvFp2]) -> TvFp2 {
    assert_eq!(values.len(), 1 << point.len(), "2^n entries at n points");
    let mut weights = vec![TvFp2::ONE];
    for &r in point {
        let high: Vec<TvFp2> = weights.iter().map(|&e| e * r).collect();
        for (e, h) in weights.iter_mut().zip(&high) {
            *e -= *h;
        }
        weights.extend(high);
    }
    values
        .iter()
        .zip(weights)
        .fold(TvFp2::ZERO, |sum, (&v, e)| {
            sum + e * TvFp2::from(tv_fp(Goldilocks::from_i64(v)))
        })
}

/// `values`, row-major of `shape`, padded per dimension with zeros.
pub(super) fn zero_padded(values: &[i64], shape: &[u32]) -> Vec<i64> {
    let vars: u32 = shape.iter().map(|&d| axis_vars(d)).sum();
    let mut table = vec![0; 1 << vars];
    for (flat, &v) in values.iter().enumerate() {
        let (mut rest, mut index, mut shift) = (flat, 0usize, 0);
        for &len in shape.iter().rev() {
            index |= (rest % len as usize) << shift;
            rest /= len as usize;
            shift += axis_vars(len);
        }
        table[index] = v;
    }
    table
}

// ----- The leaf stage's sites, as the verifier derives them -------------------------------------

/// A tensor a site's lookup reads, at its coordinates of `P`; the split sends no value for a
/// public one.
#[derive(Clone, Debug, PartialEq, Eq)]
struct VRead {
    tensor: TensorId,
    coords: Vec<u32>,
    public: bool,
}

/// An affine form of reads: `constant + Σ a·read`.
#[derive(Clone, Debug)]
struct VForm {
    constant: TvFp,
    terms: Vec<(TvFp, usize)>,
}

/// One lookup site of the graph, from RFC-0008's keys and RFC-0045's reads.
#[derive(Clone, Debug)]
struct VSite {
    node: NodeId,
    table: TableId,
    vars: u32,
    reads: Vec<VRead>,
    key: VForm,
    value: VForm,
}

/// One instance's column: its blocks, larger first in site order, their offsets, its entries
/// `m` and height `h`.
#[derive(Clone, Debug)]
struct VInstance {
    table: TableId,
    blocks: Vec<(usize, u32, u64)>,
    entries: u64,
    height: u32,
}

/// The sites, every `TableMap` and `Bitwise` of `graph` in node order (RFC-0072 'Gadgets and leaf
/// sites'): the key is the input for a `TableMap`, or `pack(c, z) = (z + 2^{b−1}) + 2^b·c` into a
/// `Requant` table, and `a + 2^8·b` for a `Bitwise8`; the value is the output. A site reads each
/// tensor once, the key's before the value's, over its domain, its input's shape, or a class over
/// the channel axis alone (RFC-0045 'What is sent').
fn sites(graph: &Graph, specs: &[TableSpec]) -> Vec<VSite> {
    let public = public_tensors(graph);
    let one = TvFp::ONE;
    let mut sites = Vec::new();
    for node in &graph.nodes {
        let table = match node.op {
            Op::TableMap { table } | Op::Bitwise { table } => table,
            ref op => {
                // `BitDecompose` and `AssertEqual` take no site (RFC-0072 'Gadgets and leaf
                // sites'); this verifier knows no other gadget.
                assert!(
                    matches!(op, Op::BitDecompose { .. } | Op::AssertEqual)
                        || op.class() != tessera_ir::OpClass::Gadget,
                    "a graph of lookups"
                );
                continue;
            }
        };
        let x = node.inputs[0];
        let domain = graph.tensors[x.0 as usize].shape.clone();
        let vars: u32 = domain.iter().map(|&d| axis_vars(d)).sum();
        let mut reads: Vec<VRead> = Vec::new();
        let mut read = |tensor: TensorId, coords: Vec<u32>| {
            if let Some(i) = reads.iter().position(|r| r.tensor == tensor) {
                return i;
            }
            let public = public[tensor.0 as usize];
            reads.push(VRead {
                tensor,
                coords,
                public,
            });
            reads.len() - 1
        };
        let all: Vec<u32> = (0..vars).collect();
        let kind = &specs
            .iter()
            .find(|s| s.id == table)
            .expect("the table")
            .kind;
        let key = match (&node.op, kind) {
            (Op::Bitwise { .. }, _) => VForm {
                constant: TvFp::ZERO,
                terms: vec![
                    (one, read(x, all.clone())),
                    (TvFp::reduce(256), read(node.inputs[1], all.clone())),
                ],
            },
            (_, TableKind::Requant { b, r_by_class, .. }) => {
                let mut terms = vec![(one, read(x, all.clone()))];
                if r_by_class.len() > 1 {
                    let channel = axis_vars(*domain.last().expect("a channel axis"));
                    terms.push((
                        TvFp::reduce(1 << b),
                        read(node.inputs[1], (0..channel).collect()),
                    ));
                }
                VForm {
                    constant: TvFp::reduce(1 << (b - 1)),
                    terms,
                }
            }
            _ => VForm {
                constant: TvFp::ZERO,
                terms: vec![(one, read(x, all.clone()))],
            },
        };
        let value = VForm {
            constant: TvFp::ZERO,
            terms: vec![(one, read(node.outputs[0], all))],
        };
        sites.push(VSite {
            node: node.id,
            table,
            vars,
            reads,
            key,
            value,
        });
    }
    sites
}

/// The instances, one per table in table order, each its blocks larger first, ties in site
/// order, at prefix-sum offsets (RFC-0045 'The column').
fn instances(sites: &[VSite]) -> Vec<VInstance> {
    let mut tables: Vec<TableId> = sites.iter().map(|s| s.table).collect();
    tables.sort();
    tables.dedup();
    tables
        .into_iter()
        .map(|table| {
            let mut blocks: Vec<(usize, u32, u64)> = Vec::new();
            for vars in (0..64).rev() {
                for (s, site) in sites.iter().enumerate() {
                    if site.table == table && site.vars == vars {
                        blocks.push((s, vars, 0));
                    }
                }
            }
            let mut offset = 0u64;
            for block in &mut blocks {
                block.2 = offset;
                offset += 1 << block.1;
            }
            let height = (0..64).find(|&h| (1u64 << h) >= offset).expect("a height");
            VInstance {
                table,
                blocks,
                entries: offset,
                height,
            }
        })
        .collect()
}

/// The verifier's table of a specification: its TCE bytes, decoded and regenerated by the
/// verifier's own crates.
fn tv_table(spec: &TableSpec) -> tv_tables::Table {
    let bytes = tessera_codec::encode(spec);
    let decoded = tv_codec::decode(
        &bytes,
        &tv_codec::DecodeLimits::MESSAGE,
        tv_tables::TableSpec::decode,
    )
    .expect("a specification");
    tv_tables::Table::from_spec(&decoded).expect("a table")
}

// ----- Proofs -----------------------------------------------------------------------------------

/// What the leaf stage and the stages send, the leaf stage's in the verifier's types.
#[derive(Clone)]
pub(super) struct LeafProof {
    pub(super) roots: Vec<TvFp2>,
    pub(super) levels: Vec<tv::SumcheckProof>,
    pub(super) values: Vec<TvFp2>,
    pub(super) splits: Vec<TvFp2>,
    pub(super) stages: Vec<StageProof>,
}

impl LeafProof {
    fn of(logup: &ProvenLogUp<F>, splits: &[F], stages: Vec<StageProof>) -> Self {
        let tvs = |xs: &[F]| xs.iter().copied().map(tv_fp2).collect::<Vec<_>>();
        Self {
            roots: tvs(logup.roots()),
            levels: logup
                .levels()
                .iter()
                .map(|level| tv::SumcheckProof {
                    rounds: level.proof().rounds.iter().map(|r| tvs(r)).collect(),
                    final_evals: tvs(&level.proof().final_evals),
                })
                .collect(),
            values: tvs(logup.values()),
            splits: tvs(splits),
            stages,
        }
    }
}

/// A graph with its tables and gadgets, its schedule and leaf layout, ready to prove.
pub(super) struct Setup {
    pub(super) graph: Graph,
    pub(super) specs: Vec<TableSpec>,
    pub(super) schedule: StageSchedule,
    pub(super) layout: LeafLayout<ModelColumn>,
    pub(super) columns: Vec<TableColumns<Goldilocks>>,
}

impl Setup {
    /// `graph`'s ranges settled, its gadgets selected, its schedule and leaf layout derived.
    pub(super) fn new(mut graph: Graph, specs: Vec<TableSpec>) -> Self {
        let ranges = analyze(&graph, &specs).expect("the ranges");
        settle(&mut graph, &ranges).expect("bounded ranges");
        let selection: Selection =
            select(&graph, &ranges, &specs, &GadgetPolicy::default()).expect("the gadgets");
        let schedule = derive_graph_schedule(&graph, &specs, &[], None).expect("a schedule");
        let layout =
            LeafLayout::graph(&graph, &schedule.layers, &selection).expect("the leaf layout");
        let columns = layout
            .instances()
            .iter()
            .map(|i| {
                let spec = specs.iter().find(|s| s.id == i.table).expect("a table");
                TableColumns::new(&GeneratedTable::from_spec(spec).expect("a table"))
                    .expect("the columns")
            })
            .collect();
        Self {
            graph,
            specs,
            schedule,
            layout,
            columns,
        }
    }

    /// The witness of `inputs`, from the CPU engine.
    pub(super) fn witness(&self, inputs: Vec<(TensorId, Vec<i64>)>) -> Witness {
        Engine::for_graph(&self.graph, &self.specs)
            .expect("an engine")
            .evaluate(
                inputs
                    .into_iter()
                    .map(|(t, v)| (t, v.into_iter().collect()))
                    .collect(),
            )
            .expect("a witness")
    }

    /// The proof of the leaf stage over `leaf`'s tensors, then of the stages over `stages`, on
    /// one transcript, and the prover's pool.
    pub(super) fn prove<W: WitnessSource + ?Sized>(
        &self,
        leaf: &GraphLeafWitness,
        stages: &W,
    ) -> (LeafProof, Vec<F>) {
        self.prove_with(&self.schedule, leaf, stages)
    }

    /// [`prove`](Self::prove) under `schedule`, the graph's or one with checks removed.
    pub(super) fn prove_with<W: WitnessSource + ?Sized>(
        &self,
        schedule: &StageSchedule,
        leaf: &GraphLeafWitness,
        stages: &W,
    ) -> (LeafProof, Vec<F>) {
        let (proof, pool, _) = self.prove_targets(schedule, leaf, stages);
        (proof, pool)
    }

    /// [`prove_with`](Self::prove_with), and the producer's opening targets, which no message
    /// carries ([`GraphOpenings`]).
    pub(super) fn prove_targets<W: WitnessSource + ?Sized>(
        &self,
        schedule: &StageSchedule,
        leaf: &GraphLeafWitness,
        stages: &W,
    ) -> (LeafProof, Vec<F>, GraphOpenings) {
        let mut t = transcript();
        let tensors = leaf.tensors();
        let lookup =
            prepare_leaf(&mut t, &self.layout, &self.columns, &tensors, &[]).expect("the leaf");
        let points: Vec<Vec<F>> = schedule.stages[0]
            .claims
            .iter()
            .map(|c| lookup.point()[..c.point.len()].to_vec())
            .collect();
        let mut prover = StageProver::for_graph(&self.graph, &self.specs, schedule, stages);
        prover.leaf(&points).expect("the leaf claims");
        let atoms = prover.leaf_split_values().expect("the atoms' splits");
        let proven = lookup.finish(&atoms).expect("one split");
        let proofs = (1..schedule.stages.len())
            .map(|k| {
                let k = u16::try_from(k).expect("small");
                prover.prove_stage(&mut t, k).expect("a stage")
            })
            .collect();
        let pool = prover.pool().iter().map(|v| v.expect("a value")).collect();
        let targets = GraphOpenings::derive(&self.graph, schedule, &self.layout, &proven, &prover)
            .expect("the opening targets");
        (
            LeafProof::of(proven.logup(), proven.splits(), proofs),
            pool,
            targets,
        )
    }

    /// The honest proof of `witness`: its padded tables, its leaf tensors, its proof and pool.
    pub(super) fn honest<'w>(&self, witness: &'w Witness) -> Honest<'w> {
        let padded = PaddedWitness::new(&self.graph, &self.specs, witness).expect("padded");
        let leaf = GraphLeafWitness::new(&self.layout, &self.graph, &self.schedule.layers, &padded)
            .expect("the leaf tensors");
        let (proof, pool, targets) = self.prove_targets(&self.schedule, &leaf, &padded);
        Honest {
            padded,
            leaf,
            proof,
            pool,
            targets,
        }
    }
}

/// An honest run: the padded witness, the leaf tensors, the proof, the prover's pool and its
/// opening targets.
pub(super) struct Honest<'w> {
    pub(super) padded: PaddedWitness<'w, Witness>,
    pub(super) leaf: GraphLeafWitness,
    pub(super) proof: LeafProof,
    pub(super) pool: Vec<F>,
    pub(super) targets: GraphOpenings,
}

// ----- The verifier ----------------------------------------------------------------------------

/// Why the verifier rejects (04 §2): LogUp's roots or leaf checks, naming the instance's table
/// (`Reject(Lookup { table })`), a tree level, or a stage's check, the split's at stage 0 round 0
/// (`Reject(Sumcheck { stage, round })`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rejection {
    Lookup { table: TableId },
    Level(u32),
    Sumcheck { stage: u16, round: u32 },
}

impl From<Reject> for Rejection {
    fn from(r: Reject) -> Self {
        Self::Sumcheck {
            stage: r.stage,
            round: r.round,
        }
    }
}

/// How the verifier takes the leaf stage's claims on one tensor at one point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Claims {
    /// Each claim its own value and its own instance (RFC-0072 'Combined selections' 'No
    /// merging across claims').
    Own,
    /// A wrong verifier: claims on one tensor at one point take the first one's value, merged
    /// across the stage (RFC-0072 'Alternatives Considered' 5).
    Merged,
}

/// What the verifier ends with: the pool's values and its coordinates, then the claims the leaf
/// stage leaves for the openings: each lookup output's, at its point, and each instance's
/// multiplicities', in table order.
pub(super) struct Verdict {
    pub(super) pool: Vec<F>,
    pub(super) coords: HashMap<Coord, F>,
    pub(super) outputs: Vec<(TensorId, Vec<TvFp2>, TvFp2)>,
    pub(super) multiplicities: Vec<(TableId, Vec<TvFp2>, TvFp2)>,
}

/// The verifier's verdict on `proof` of `graph` with tables `specs` and schedule `schedule`,
/// holding the public tensors `statement` (RFC-0045, RFC-0072; module documentation).
pub(super) fn verify(
    graph: &Graph,
    specs: &[TableSpec],
    schedule: &StageSchedule,
    statement: &dyn WitnessSource,
    proof: &LeafProof,
    claims: Claims,
) -> Result<Verdict, Rejection> {
    let sites = sites(graph, specs);
    let instances = instances(&sites);
    let tables: Vec<tv_tables::Table> = instances
        .iter()
        .map(|i| tv_table(specs.iter().find(|s| s.id == i.table).expect("a table")))
        .collect();
    let trees: Vec<tv_core::leaf::Tree> = instances
        .iter()
        .zip(&tables)
        .enumerate()
        .flat_map(|(instance, (layout, table))| {
            [
                tv_core::leaf::Tree::Lookup {
                    instance,
                    height: layout.height,
                    has_values: true,
                },
                tv_core::leaf::Tree::Table {
                    instance,
                    height: u32::from(table.log2_rows()),
                },
            ]
        })
        .collect();
    let mut t = Tv::new();
    let checked = tv_core::leaf::Schedule::new(&trees)
        .expect("the verifier's trees")
        .verify(
            &mut t.0,
            tv_core::leaf::Proof {
                roots: &proof.roots,
                levels: &proof.levels,
                values: &proof.values,
            },
            |_, _| None,
            |instance, beta, point| tables.get(instance)?.eval_mle(point, beta).ok(),
        )
        .map_err(|error| match error {
            tv_core::leaf::Error::Lookup(error) if error.is_rejection() => Rejection::Lookup {
                table: instances[rejected(error)].table,
            },
            tv_core::leaf::Error::Level { level, .. } => Rejection::Level(level),
            _ => panic!("a verifier fault: {error}"),
        })?;
    // The split: one value per read that is not public, then the graph atoms' splits.
    let split = Rejection::Sumcheck { stage: 0, round: 0 };
    let sent: usize = sites
        .iter()
        .map(|s| s.reads.iter().filter(|r| !r.public).count())
        .sum();
    let atoms: usize = schedule.stages[0]
        .claims
        .iter()
        .filter(|c| c.split)
        .map(|c| c.parts.len())
        .sum();
    if proof.splits.len() != sent + atoms {
        return Err(split);
    }
    t.absorb_tv(SPLIT_LABEL, &proof.splits);
    let p = &checked.point;
    let mut next = 0;
    let values: Vec<Vec<TvFp2>> = sites
        .iter()
        .map(|site| {
            site.reads
                .iter()
                .map(|read| {
                    let at: Vec<TvFp2> = read.coords.iter().map(|&c| p[c as usize]).collect();
                    if read.public {
                        let shape = &graph.tensors[read.tensor.0 as usize].shape;
                        let values = statement.values(read.tensor).expect("a public tensor");
                        mle(&zero_padded(values, shape), &at)
                    } else {
                        next += 1;
                        proof.splits[next - 1]
                    }
                })
                .collect()
        })
        .collect();
    let form = |f: &VForm, s: usize| {
        f.terms
            .iter()
            .fold(TvFp2::from(f.constant), |sum, &(a, r)| {
                sum + TvFp2::from(a) * values[s][r]
            })
    };
    for ((instance, table), claim) in instances
        .iter()
        .zip(&tables)
        .zip(&checked.lookups.instances)
    {
        let h = instance.height as usize;
        let point = &p[..h];
        let mut leaf = point.first().copied().into_iter().collect::<Vec<_>>();
        leaf.extend(point.iter().skip(1).rev());
        assert_eq!(claim.key.point, leaf, "the lookup tree's leaf point");
        let (key0, value0) = table.row(0).expect("row 0");
        let tail = TvFp2::ONE - tv_core::selectors::lt(point, instance.entries).expect("LT");
        let (mut key, mut value) = (tail * TvFp2::from(key0), tail * TvFp2::from(value0));
        for &(s, vars, offset) in &instance.blocks {
            let weight =
                point[vars as usize..]
                    .iter()
                    .enumerate()
                    .fold(TvFp2::ONE, |w, (bit, &r)| {
                        w * match (offset >> (vars as usize + bit)) & 1 {
                            0 => TvFp2::ONE - r,
                            _ => r,
                        }
                    });
            key += weight * form(&sites[s].key, s);
            value += weight * form(&sites[s].value, s);
        }
        let value_claim = claim.value.as_ref().expect("a table with values");
        if key != claim.key.value || value != value_claim.value {
            return Err(split);
        }
    }
    // The leaf claims: each site input's value at its point, sent or public.
    let mut seen: Vec<(TensorId, Vec<u32>, TvFp2)> = Vec::new();
    let mut leaf_claims = Vec::with_capacity(schedule.stages[0].claims.len());
    for site in &schedule.sites {
        let tessera_compiler::schedule::Site::Gadget {
            node: NodeRef::Node(node),
            input,
        } = *site
        else {
            panic!("a graph's sites are outside the layers");
        };
        let tensor = graph.nodes[node.0 as usize].inputs[input as usize];
        let s = sites.iter().position(|v| v.node == node).expect("a site");
        let r = sites[s]
            .reads
            .iter()
            .position(|r| r.tensor == tensor)
            .expect("a read");
        let coords = sites[s].reads[r].coords.clone();
        let mut value = values[s][r];
        if claims == Claims::Merged {
            match seen.iter().find(|(t, c, _)| *t == tensor && *c == coords) {
                Some(&(_, _, first)) => value = first,
                None => seen.push((tensor, coords, value)),
            }
        }
        leaf_claims.push(vec![fp2(value)]);
    }
    let outputs = sites
        .iter()
        .enumerate()
        .flat_map(|(s, site)| {
            let output = graph.nodes[site.node.0 as usize].outputs[0];
            let r = site.reads.iter().position(|r| r.tensor == output)?;
            let at = site.reads[r]
                .coords
                .iter()
                .map(|&c| p[c as usize])
                .collect();
            Some((output, at, values[s][r]))
        })
        .collect();
    let multiplicities = instances
        .iter()
        .zip(&checked.lookups.instances)
        .map(|(i, c)| (i.table, c.multiplicity.point.clone(), c.multiplicity.value))
        .collect();
    let outcome = LeafOutcome {
        point: p.iter().copied().map(fp2).collect(),
        claims: leaf_claims,
        atoms: proof.splits[sent..].iter().copied().map(fp2).collect(),
    };
    let publics = Publics {
        values: statement,
        shift: Shift::Extension,
        kernel: Kernel::Closed,
    };
    let (pool, coords) = verify_from_leaf(
        graph,
        specs,
        schedule,
        publics,
        outcome,
        &proof.stages,
        &mut t,
    )?;
    Ok(Verdict {
        pool,
        coords,
        outputs,
        multiplicities,
    })
}

/// The instance a LogUp rejection names.
fn rejected(error: LookupError) -> usize {
    error
        .rejected_instance()
        .expect("a rejection names its instance")
}

/// `tensor`'s committed table over its padded shape, as the openings read it: `source`'s padded
/// values, or its values padded with zeros.
pub(super) fn committed(graph: &Graph, source: &dyn WitnessSource, tensor: TensorId) -> Vec<i64> {
    match source.padded(tensor) {
        Some(values) => values.to_vec(),
        None => zero_padded(
            source.values(tensor).expect("a tensor"),
            &graph.tensors[tensor.0 as usize].shape,
        ),
    }
}

/// The openings' part (RFC-0011, RFC-0012), which the test plays: the verifier's pool, its lookup
/// outputs' claims and its multiplicities' claims are the extensions of the tables `source`
/// commits and of the multiplicities `multiplicities` holds, in table order.
pub(super) fn opened(
    setup: &Setup,
    verdict: &Verdict,
    source: &dyn WitnessSource,
    multiplicities: &[Vec<Goldilocks>],
) {
    opened_under(setup, &setup.schedule, verdict, source, multiplicities);
}

/// [`opened`] under `schedule`, the graph's or one with checks removed.
pub(super) fn opened_under(
    setup: &Setup,
    schedule: &StageSchedule,
    verdict: &Verdict,
    source: &dyn WitnessSource,
    multiplicities: &[Vec<Goldilocks>],
) {
    let graph = &setup.graph;
    let pool = pool_of_tables(&verdict.coords, graph, schedule, &|t| {
        committed(graph, source, t)
            .into_iter()
            .map(Goldilocks::from_i64)
            .collect()
    });
    assert_eq!(
        verdict.pool, pool,
        "the pool is the committed tables' extensions"
    );
    for (tensor, point, value) in &verdict.outputs {
        assert_eq!(
            *value,
            mle(&committed(graph, source, *tensor), point),
            "lookup output {tensor:?}"
        );
    }
    for ((table, point, value), m) in verdict.multiplicities.iter().zip(multiplicities) {
        let m: Vec<i64> = m.iter().map(|x| x.to_canonical_u64() as i64).collect();
        assert_eq!(*value, mle(&m, point), "the multiplicities of {table:?}");
    }
}

/// RFC-0011, RFC-0045, RFC-0072: the producer's opening targets ([`GraphOpenings`]) are what
/// the verifier holds once the stages pass, in order. The pool's claims, at the verifier's points,
/// each the extension of its pieces over the committed tables `source` gives; then each lookup
/// output's split claim, a committed tensor the schedule leaves no claim on; then each
/// multiplicities' claim. And every leaf claim on a committed tensor, a lookup input, is ready in
/// the pool, so it reaches a target: no committed leaf claim goes unopened.
pub(super) fn targets_are_the_verifiers(
    setup: &Setup,
    targets: &GraphOpenings,
    verdict: &Verdict,
    source: &dyn WitnessSource,
) {
    let graph = &setup.graph;
    let schedule = &setup.schedule;
    let coord = |c: &Coord| match c {
        Coord::Half => F::from(Goldilocks::from_u64_reduced(2))
            .inverse()
            .expect("2 ≠ 0"),
        other => verdict.coords[other],
    };
    let value = |target: &tessera_prover::OpeningTarget| {
        let point: Vec<TvFp2> = target.point.iter().copied().map(tv_fp2).collect();
        target.pieces.iter().fold(TvFp2::ZERO, |sum, &(of, a)| {
            let AdviceSource::Tensor(t) = of else {
                panic!("a tensor's target");
            };
            sum + tv_fp2(a) * mle(&committed(graph, source, t), &point)
        })
    };
    let (mut advice, mut weights) = (targets.advice().iter(), targets.weights().iter());
    for (claim, &v) in schedule.pool.iter().zip(&verdict.pool) {
        let target = match claim.commitment {
            CommitmentClass::Advice => advice.next(),
            CommitmentClass::Weight => weights.next(),
        }
        .expect("a target per pool claim");
        let point: Vec<F> = claim.point.iter().map(coord).collect();
        assert_eq!(target.point, point, "a pool claim's point");
        assert_eq!(value(target), tv_fp2(v), "a pool claim's value");
    }
    assert!(weights.next().is_none(), "no other weight target");
    for (tensor, point, v) in &verdict.outputs {
        let target = advice.next().expect("a target per lookup output");
        assert_eq!(target.pieces, [(AdviceSource::Tensor(*tensor), F::ONE)]);
        assert_eq!(
            target.point,
            point.iter().copied().map(fp2).collect::<Vec<_>>()
        );
        assert_eq!(value(target), *v, "lookup output {tensor:?}");
    }
    for (table, point, _) in &verdict.multiplicities {
        let target = advice.next().expect("a target per instance");
        assert_eq!(
            target.pieces,
            [(AdviceSource::Multiplicity(*table), F::ONE)]
        );
        assert_eq!(
            target.point,
            point.iter().copied().map(fp2).collect::<Vec<_>>()
        );
    }
    assert!(advice.next().is_none(), "no other advice target");
    // Every leaf claim on a committed tensor is ready in the pool, its claim a target's.
    for claim in &schedule.stages[0].claims {
        let tessera_compiler::schedule::Kind::Tensor(t) = claim.kind else {
            panic!("a graph's kinds are tensors");
        };
        let committed = matches!(
            graph.tensors[t.0 as usize].kind,
            NodeKind::Committed { .. } | NodeKind::LookupOutput { .. }
        );
        if !committed {
            continue;
        }
        let [Part::Pool { claim: pool, .. }] = claim.parts[..] else {
            panic!("a committed leaf claim is ready in the pool");
        };
        assert!(
            schedule.pool[pool]
                .atoms
                .iter()
                .any(|a| a.kind == claim.kind),
            "its pool claim holds it"
        );
    }
}

/// The multiplicities the prover commits for `leaf`'s tensors, in table order.
pub(super) fn multiplicities(setup: &Setup, leaf: &GraphLeafWitness) -> Vec<Vec<Goldilocks>> {
    let columns =
        LeafColumns::new(&setup.layout, &setup.columns, &leaf.tensors()).expect("columns");
    columns
        .columns()
        .iter()
        .map(|c| c.multiplicities().to_vec())
        .collect()
}

// ----- A synthetic graph -----------------------------------------------------------------------

/// The tables: the S-box (0), `Gf256Mul { 2 }` (1), `Bitwise8 { Xor }` (2) and a requantization
/// table non-zero at padding, `Requant { b: 9, [1], 1, 255 }` (3), `rq(0) = 1`.
fn byte_tables() -> Vec<TableSpec> {
    [
        TableKind::AesSbox,
        TableKind::Gf256Mul { c: 2 },
        TableKind::Bitwise8 {
            op: tessera_tables::BitOp::Xor,
        },
        TableKind::Requant {
            b: 9,
            r_by_class: vec![1],
            lo: 1,
            hi: 255,
        },
    ]
    .into_iter()
    .zip(0..)
    .map(|(kind, id)| {
        GeneratedTable::new(TableId(id), kind)
            .expect("a table")
            .spec()
            .clone()
    })
    .collect()
}

/// A graph of byte lookups (RFC-0072 'AES's lookups'): committed bytes `x` of `[4, 3]`, range-
/// checked by a `BitDecompose { 8, 0 }`, whose padding check pins its padded column; public bytes
/// `p` of `[4, 3]`; `s = AesSbox(x)`, non-zero at padding; `m = Gf256Mul{2}(s)`; `c = p ⊕ m`,
/// whose first operand is public; `d = c ⊕ c`, one tensor twice; `q = Requant-table(s)` and
/// `r = Requant-table(x)`, a lookup non-zero at padding of a zero-padded input; and `e =
/// AesSbox(p)`, a lookup of public bytes alone. The graph and its inputs `x` and `p`.
fn byte_graph() -> (Graph, TensorId, TensorId) {
    let mut g = GraphBuilder::new();
    let byte = |kind| {
        let mut d = decl(&[4, 3], kind);
        d.range = IntRange { lo: 0, hi: 255 };
        d
    };
    let x = g.input(byte(ADVICE)).expect("x");
    let p = g.input(byte(NodeKind::Public)).expect("p");
    let mut bits = decl(&[4, 3, 8], ADVICE);
    bits.range = IntRange { lo: 0, hi: 1 };
    g.node(Op::BitDecompose { low: 8, high: 0 }, &[x], vec![bits])
        .expect("the range check");
    let lookup = |g: &mut GraphBuilder, table: u32, inputs: &[TensorId]| {
        let op = match inputs.len() {
            1 => Op::TableMap {
                table: TableId(table),
            },
            _ => Op::Bitwise {
                table: TableId(table),
            },
        };
        let output = decl(
            &[4, 3],
            NodeKind::LookupOutput {
                table: TableId(table),
            },
        );
        g.node(op, inputs, vec![output]).expect("a lookup")[0]
    };
    let s = lookup(&mut g, 0, &[x]);
    let m = lookup(&mut g, 1, &[s]);
    let c = lookup(&mut g, 2, &[p, m]);
    lookup(&mut g, 2, &[c, c]);
    lookup(&mut g, 3, &[s]);
    lookup(&mut g, 3, &[x]);
    lookup(&mut g, 0, &[p]);
    (g.finish(), x, p)
}

/// The byte graph's setup, its inputs' values and its witness.
fn byte_run(seed: u64) -> (Setup, Vec<(TensorId, Vec<i64>)>, Witness) {
    let (graph, x, p) = byte_graph();
    let setup = Setup::new(graph, byte_tables());
    let mut state = seed;
    let mut bytes = || -> Vec<i64> {
        (0..12)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                i64::try_from(state >> 56).expect("a byte")
            })
            .collect()
    };
    let inputs = vec![(x, bytes()), (p, bytes())];
    let witness = setup.witness(inputs.clone());
    (setup, inputs, witness)
}

/// The verifier's public data on the honest witness: its public tensors.
fn statement<'w>(setup: &Setup, witness: &'w Witness) -> PublicOf<'w> {
    PublicOf {
        witness,
        public: public_tensors(&setup.graph),
    }
}

/// RFC-0072 'Gadgets and leaf sites' 'AES's lookups', 'Public tensors' 'In the leaf stage',
/// 'Padded positions': a graph's `TableMap` and `Bitwise` lookups, with padded positions, a
/// public operand, one tensor read twice, lookups non-zero at padding and a lookup of public
/// bytes alone, prove through the real leaf stage and the stages, and the verifier accepts, its
/// openings the committed padded tables' extensions.
///
/// The layout sends no value for a public read (`p`, twice), and one for a tensor read twice
/// (`c ⊕ c`); its sites, blocks and heights are the verifier's. The S-box's padded column holds
/// `0x63`, and the requantization table's of the zero-padded `x` holds `rq(0) = 1`.
#[test]
fn a_graphs_lookups_prove_through_the_real_leaf_stage() {
    let (setup, _, witness) = byte_run(0x0526_4B01);
    // Seven sites: twelve reads, of which `p` twice public; `c ⊕ c` reads `c` once.
    let reads: Vec<usize> = (0..7)
        .map(|s| setup.layout.reads(s, 0).expect("a site").len())
        .collect();
    assert_eq!(reads, [2, 2, 3, 2, 2, 2, 2]);
    assert_eq!(setup.layout.split_len(), 15 - 2);
    let derived = instances(&sites(&setup.graph, &setup.specs));
    let ours: Vec<(TableId, u64, u32)> = setup
        .layout
        .instances()
        .iter()
        .map(|i| (i.table, i.entries, i.height))
        .collect();
    let theirs: Vec<(TableId, u64, u32)> = derived
        .iter()
        .map(|i| (i.table, i.entries, i.height))
        .collect();
    assert_eq!(ours, theirs, "the verifier's instances");
    let honest = setup.honest(&witness);
    let s = setup.graph.nodes[1].outputs[0];
    assert_eq!(honest.padded.padded(s).expect("the S-box")[3], 0x63);
    let r = setup.graph.nodes[6].outputs[0];
    assert_eq!(honest.padded.padded(r).expect("rq(x)")[3], 1);
    let public = statement(&setup, &witness);
    let verdict = verify(
        &setup.graph,
        &setup.specs,
        &setup.schedule,
        &public,
        &honest.proof,
        Claims::Own,
    )
    .expect("the honest proof verifies");
    assert_eq!(verdict.pool, honest.pool, "the prover's pool");
    opened(
        &setup,
        &verdict,
        &honest.padded,
        &multiplicities(&setup, &honest.leaf),
    );
    targets_are_the_verifiers(&setup, &honest.targets, &verdict, &honest.padded);
}

/// RFC-0045 'The check', RFC-0072 'In the leaf stage': each value the split sends is checked, a
/// changed one is `Reject(Sumcheck { stage: 0, round: 0 })`, and so is a public input changed on
/// the verifier's side, which it evaluates itself: one entry of `p`, real or not.
#[test]
fn a_changed_split_value_or_public_input_is_rejected_at_stage_0() {
    let (setup, inputs, witness) = byte_run(0x0526_4B02);
    let honest = setup.honest(&witness);
    let public = statement(&setup, &witness);
    let split = Err(Rejection::Sumcheck { stage: 0, round: 0 });
    let verdict = |proof: &LeafProof, statement: &dyn WitnessSource| {
        verify(
            &setup.graph,
            &setup.specs,
            &setup.schedule,
            statement,
            proof,
            Claims::Own,
        )
        .map(|_| ())
    };
    for index in 0..honest.proof.splits.len() {
        let mut bad = honest.proof.clone();
        bad.splits[index] += TvFp2::ONE;
        assert_eq!(verdict(&bad, &public), split, "split value {index}");
    }
    let mut short = honest.proof.clone();
    short.splits.pop();
    assert_eq!(verdict(&short, &public), split, "a value short");
    // The public bytes changed on the verifier's side, entry by entry.
    let (p, values) = &inputs[1];
    for entry in [0, 5, 11] {
        let mut changed = values.clone();
        changed[entry] ^= 1;
        let overridden = Override {
            inner: &public,
            tensor: *p,
            values: changed,
        };
        assert_eq!(verdict(&honest.proof, &overridden), split, "p[{entry}]");
    }
}

/// A public source with one tensor's values replaced.
pub(super) struct Override<'a> {
    pub(super) inner: &'a dyn WitnessSource,
    pub(super) tensor: TensorId,
    pub(super) values: Vec<i64>,
}

impl WitnessSource for Override<'_> {
    fn values(&self, tensor: TensorId) -> Option<&[i64]> {
        match tensor == self.tensor {
            true => Some(&self.values),
            false => self.inner.values(tensor),
        }
    }
}

/// A source whose tensors `tables` are replaced over their padded shapes: a prover that commits
/// them so, its other tensors `inner`'s.
pub(super) struct Committing<'a> {
    pub(super) inner: &'a dyn WitnessSource,
    pub(super) tables: Vec<(TensorId, Vec<i64>)>,
}

impl WitnessSource for Committing<'_> {
    fn values(&self, tensor: TensorId) -> Option<&[i64]> {
        self.inner.values(tensor)
    }

    fn padded(&self, tensor: TensorId) -> Option<&[i64]> {
        match self.tables.iter().find(|(t, _)| *t == tensor) {
            Some((_, table)) => Some(table),
            None => self.inner.padded(tensor),
        }
    }
}

/// RFC-0008 'Fraction tree', RFC-0072 'Padded positions': a lookup output off its table at a real
/// position, and the S-box's padded column taken as 0, as the zero-padded tables of #526's first
/// slices held, are each `Reject(Lookup { table })` of the S-box's table, at LogUp's roots. The
/// rest of the witness is honest, so the S-box's instance is the one that fails.
#[test]
fn a_lookup_off_its_table_is_rejected_at_logup() {
    let (setup, _, witness) = byte_run(0x0526_4B03);
    let honest = setup.honest(&witness);
    let public = statement(&setup, &witness);
    let s = setup.graph.nodes[1].outputs[0];
    let table = committed(&setup.graph, &honest.padded, s);
    let mut off = table.clone();
    off[5] ^= 1;
    let mut zero = table;
    for row in 0..4 {
        zero[4 * row + 3] = 0;
    }
    for (name, cheat) in [("a real entry", off), ("the padded column", zero)] {
        let source = Committing {
            inner: &honest.padded,
            tables: vec![(s, cheat)],
        };
        let leaf =
            GraphLeafWitness::new(&setup.layout, &setup.graph, &setup.schedule.layers, &source)
                .expect("the leaf tensors");
        let (proof, _) = setup.prove(&leaf, &source);
        assert_eq!(
            verify(
                &setup.graph,
                &setup.specs,
                &setup.schedule,
                &public,
                &proof,
                Claims::Own
            )
            .err(),
            Some(Rejection::Lookup { table: TableId(0) }),
            "{name}"
        );
    }
}

/// RFC-0045 'The column', RFC-0008 'Fraction tree': each instance's multiplicities count every
/// entry of its column, the blocks' padded positions among them, and the row-0 tail, so they sum
/// to `2^h`. Multiplicities that miss the blocks' padded positions, counting real positions and
/// the tail alone, fail LogUp's roots: `Reject(Lookup { table })` of the first instance with a
/// padded position, the S-box's.
#[test]
fn the_multiplicities_count_the_padded_positions() {
    let (setup, _, witness) = byte_run(0x0526_4B04);
    let honest = setup.honest(&witness);
    let tensors = honest.leaf.tensors();
    let columns = LeafColumns::new(&setup.layout, &setup.columns, &tensors).expect("columns");
    // Per instance, its real entries: each block's domain, `[4, 3]`, twelve of sixteen.
    let mut real = Vec::new();
    for (instance, column) in setup.layout.instances().iter().zip(columns.columns()) {
        let total: u64 = column
            .multiplicities()
            .iter()
            .map(|m| m.to_canonical_u64())
            .sum();
        assert_eq!(total, 1 << instance.height, "{:?}", instance.table);
        real.push(instance.blocks.len() as u64 * 12 + ((1 << instance.height) - instance.entries));
    }
    // Recount without the padded positions: each padded entry's row loses one. Its key, from
    // the graph: the input, `2^8` more into the requantization table, `a + 2^8·b` for a XOR.
    let graph = &setup.graph;
    let mut missing: Vec<Vec<Goldilocks>> = Vec::new();
    for (i, (instance, column)) in setup
        .layout
        .instances()
        .iter()
        .zip(columns.columns())
        .enumerate()
    {
        let mut m = column.multiplicities().to_vec();
        for block in &instance.blocks {
            let NodeRef::Node(id) = setup.layout.sites()[block.site].node else {
                panic!("a site outside the layers");
            };
            let node = &graph.nodes[id.0 as usize];
            let table = |k: usize| committed(graph, &honest.padded, node.inputs[k]);
            let (a, b) = (table(0), node.inputs.get(1).map(|_| table(1)));
            for x in 0..block.entries() as usize {
                if (x >> 2) < 4 && (x & 3) < 3 {
                    continue;
                }
                let key = match (&node.op, &b) {
                    (Op::Bitwise { .. }, Some(b)) => a[x] + 256 * b[x],
                    (Op::TableMap { table: TableId(3) }, _) => a[x] + 256,
                    _ => a[x],
                };
                m[usize::try_from(key).expect("a row")] -= Goldilocks::ONE;
            }
        }
        let total: u64 = m.iter().map(|x| x.to_canonical_u64()).sum();
        assert_eq!(
            total, real[i],
            "{:?}: real positions and the tail",
            instance.table
        );
        missing.push(m);
    }
    let instances: Vec<LogUpInstance<'_, Goldilocks>> = columns
        .columns()
        .iter()
        .zip(&missing)
        .zip(&setup.columns)
        .map(|((column, m), table)| {
            LogUpInstance::new(
                table.tuples(),
                Tuples::new(column.keys(), column.values()),
                m,
            )
        })
        .collect();
    let mut t = transcript();
    let logup = prove_logup(&mut t, &instances, &[], &setup.layout.trees()).expect("the trees");
    let splits = split_values(&setup.layout, &leaf_point(&logup), &tensors).expect("the split");
    let proof = LeafProof::of(&logup, &splits, Vec::new());
    let public = statement(&setup, &witness);
    assert_eq!(
        verify(
            &setup.graph,
            &setup.specs,
            &setup.schedule,
            &public,
            &proof,
            Claims::Own
        )
        .err(),
        Some(Rejection::Lookup { table: TableId(0) })
    );
}

/// RFC-0072 'Non-Goals' 'Token groups', RFC-0045 'The split': a graph without a lowering has no
/// token axis, so its layout runs ungrouped, and RFC-0054's merge, which sends a polynomial per
/// read, has no rule for a public read: `GroupedLayout::new` refuses the layout with groups, by
/// the token rule, and in one group, by its public reads. A graph with a gadget other than a
/// lookup at a leaf site, a requantization, takes a model's layout instead.
#[test]
fn a_graph_layout_runs_ungrouped_and_lays_out_lookups_alone() {
    use tessera_protocol::LeafGroups;
    use tessera_prover::leaf::{GroupedLayout, LeafError};
    let (setup, ..) = byte_run(0x0526_4B05);
    let rule = setup.layout.ungroupable().expect("ungrouped");
    let grouped = |token_bits| {
        GroupedLayout::new(
            &setup.layout,
            &setup.schedule.layers,
            LeafGroups {
                token_bits,
                layer_bits: 0,
            },
        )
        .err()
    };
    assert_eq!(grouped(1), Some(LeafError::Internal { rule }));
    assert_eq!(
        grouped(0),
        Some(LeafError::Internal {
            rule: "a grouped leaf stage reads no public tensor"
        })
    );
    // A requantization's site.
    let mut g = GraphBuilder::new();
    let mut input = decl(&[4, 3], ADVICE);
    input.range = IntRange { lo: 0, hi: 255 };
    let x = g.input(input).expect("x");
    let mut bits = decl(&[4, 3, 8], ADVICE);
    bits.range = IntRange { lo: 0, hi: 1 };
    g.node(Op::BitDecompose { low: 8, high: 0 }, &[x], vec![bits])
        .expect("the range check");
    requant(&mut g, x, &[4, 3], 100);
    let mut graph = g.finish();
    let specs = Vec::new();
    let ranges = analyze(&graph, &specs).expect("the ranges");
    settle(&mut graph, &ranges).expect("bounded ranges");
    let selection = select(&graph, &ranges, &specs, &GadgetPolicy::default()).expect("gadgets");
    let schedule = derive_graph_schedule(&graph, &selection.tables, &[], None).expect("a schedule");
    assert_eq!(
        LeafLayout::graph(&graph, &schedule.layers, &selection).err(),
        Some(LeafError::Internal {
            rule: "a graph's leaf sites are TableMap and Bitwise lookups; a model's layout lays \
                   out the other gadgets"
        })
    );
}

/// RFC-0045 'Blocks', RFC-0072 'Padded positions': a lookup into a table of two classes reads its
/// classes over the channel axis, and a padded channel's class is the class vector's padded
/// entry, 0. The S-box's padded column, 0x63, then looks up class 0's `rq(99) = 50` by its shift
/// 1, where class 1's shift 2 would give 25; the block's key packs that class, so LogUp accepts
/// the honest proof only with it.
#[test]
fn a_classed_lookup_takes_class_0_at_a_padded_channel() {
    let mut specs = byte_tables();
    specs.push(
        GeneratedTable::new(
            TableId(4),
            TableKind::Requant {
                b: 9,
                r_by_class: vec![1, 2],
                lo: 1,
                hi: 255,
            },
        )
        .expect("a table")
        .spec()
        .clone(),
    );
    let mut g = GraphBuilder::new();
    let mut byte = decl(&[4, 3], ADVICE);
    byte.range = IntRange { lo: 0, hi: 255 };
    let x = g.input(byte).expect("x");
    let mut bits = decl(&[4, 3, 8], ADVICE);
    bits.range = IntRange { lo: 0, hi: 1 };
    g.node(Op::BitDecompose { low: 8, high: 0 }, &[x], vec![bits])
        .expect("the range check");
    let mut classes = decl(
        &[3],
        NodeKind::Committed {
            commitment: CommitmentClass::Weight,
        },
    );
    classes.range = IntRange { lo: 0, hi: 1 };
    let c = g.input(classes).expect("the classes");
    let output = |table| decl(&[4, 3], NodeKind::LookupOutput { table });
    let s = g
        .node(
            Op::TableMap { table: TableId(0) },
            &[x],
            vec![output(TableId(0))],
        )
        .expect("the S-box")[0];
    let q = g
        .node(
            Op::TableMap { table: TableId(4) },
            &[s, c],
            vec![output(TableId(4))],
        )
        .expect("a classed lookup")[0];
    let setup = Setup::new(g.finish(), specs);
    assert_eq!(setup.layout.reads(1, 0).expect("the site").len(), 3);
    let witness = setup.witness(vec![
        (x, (0..12).map(|i| (53 * i + 7) % 256).collect()),
        (c, vec![1, 0, 1]),
    ]);
    let honest = setup.honest(&witness);
    let table = honest.padded.padded(q).expect("the classed lookup's table");
    assert!(
        (0..4).all(|row| table[4 * row + 3] == 50),
        "class 0's rq(99)"
    );
    let verdict = verify(
        &setup.graph,
        &setup.specs,
        &setup.schedule,
        &statement(&setup, &witness),
        &honest.proof,
        Claims::Own,
    )
    .expect("the honest proof verifies");
    opened(
        &setup,
        &verdict,
        &honest.padded,
        &multiplicities(&setup, &honest.leaf),
    );
    // The classes are committed with the weights: their claim is a weight target.
    assert_eq!(honest.targets.weights().len(), 1);
    targets_are_the_verifiers(&setup, &honest.targets, &verdict, &honest.padded);
}
