// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The leaf stage's layout and split (RFC-0045), end to end on a conformance model, checked by a
//! production independent shared-tree and RFC-0045 split verifiers.
//!
//! - **The model.** `tiny-2-servable` is lowered for its first sequence, its gadgets selected, and
//!   its forward pass computed by the CPU engine (`tessera-witness`). Its leaf sites here are the
//!   requantizations whose tables hold at most `2^20` rows in every layer: positions 13 and 14,
//!   `RequantSplit` with classes, 32, `RequantSplit` with one class, and 47, `Requant1` with a
//!   table of its own in each layer. `tessera-gadgets` implements the requantizations, and the
//!   model's larger tables, up to `2^25` rows, would hold gigabytes of trees. `tessera-gadgets`
//!   computes each site's committed tensors from the engine's keys and classes, and each output is
//!   the engine's.
//! - **Synthetic sites** at the model's positions and in its prologue pad a `Requant` table, whose
//!   row 0 has a value other than 0.
//! - **The verifier** runs the production shared-tree driver (`tv_core::leaf`, `tv-tables`;
//!   DEP-2). It then derives the blocks, their offsets, the heights
//!   and the tensors each site reads from the sites alone, reads `P` from its own last tree
//!   level, and checks every instance's key and value claims against the split values by
//!   RFC-0045's identity: `Reject(Sumcheck { stage: 0, round: 0 })` on a mismatch.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use tessera_codec::{Ascii, Secret, encode, envelope, schema};
use tessera_compiler::schedule::{LayerMap, NodeRef};
use tessera_compiler::{Gadget, Lowering, PublicModel, Selection, analyze, lower, select};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_gadgets::{Affine, Committed, Requant, TableColumns, Term};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_ir::{GadgetPolicy, IntegerModel, NodeId};
use tessera_logup::{LogUpInstance, ProvenLogUp, Tuples, prove_logup};
use tessera_prover::leaf::{
    LeafColumns, LeafError, LeafLayout, SPLIT_LABEL, Site, SiteGadget, SiteLookup, SiteTensors,
    gadget_nodes, leaf_index, leaf_point, prove_leaf, split_values,
};
use tessera_sumcheck::{Forest, SumcheckTranscript};
use tessera_tables::{GeneratedTable, TableId, TableSpec};
use tessera_witness::{Engine, Witness};
use tv_core::logup::{self as tvl, LookupError};
use tv_core::sumcheck as tv;

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;
type TvFp = tv_field::Fp;
type TvFp2 = tv_field::Fp2;

/// The largest table, in `log2` of its rows, among the sites' tables in every layer.
const MAX_TABLE_VARS: u8 = 20;

// ----- The model -------------------------------------------------------------------------------

/// A conformance model of `spec/vectors/models/` with its first sequence's tokens.
fn conformance(name: &str) -> (IntegerModel, Vec<u32>) {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models");
    let index: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("models.json")).expect("the index"))
            .expect("JSON");
    let entry = index["models"]
        .as_array()
        .expect("models")
        .iter()
        .find(|m| m["name"] == name)
        .expect("the model");
    let bytes = fs::read(dir.join(entry["file"].as_str().expect("a file"))).expect("the file");
    let model = envelope::open(
        &bytes,
        &schema::INTEGER_MODEL,
        tessera_codec::DecodeLimits::INTEGER_MODEL,
    )
    .expect("an IntegerModel");
    let tokens = entry["sequences"][0]["tokens"]
        .as_array()
        .expect("tokens")
        .iter()
        .map(|t| u32::try_from(t.as_u64().expect("a token")).expect("small"))
        .collect();
    (model, tokens)
}

/// `tiny-2-servable` lowered for its first sequence, its layers, its gadgets and its witness.
struct Model {
    lowering: Lowering,
    layers: LayerMap,
    selection: Selection,
    witness: Witness,
}

fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| {
        let (model, tokens) = conformance("tiny-2-servable");
        let n = u32::try_from(tokens.len()).expect("small");
        let lowering = lower(PublicModel::of(&model), n).expect("the model lowers");
        let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
        let selection = select(
            &lowering.graph,
            &ranges,
            &lowering.tables,
            &GadgetPolicy::default(),
        )
        .expect("every gadget fits");
        let layers = LayerMap::new(&lowering).expect("one template");
        let witness = Engine::new(&lowering)
            .expect("an engine")
            .run(&model, &Secret::new(tokens))
            .expect("a witness");
        Model {
            lowering,
            layers,
            selection,
            witness,
        }
    })
}

/// The nodes of `site`: its node in each real layer, or itself outside the layers.
fn nodes_of(layers: &LayerMap, site: NodeRef) -> Vec<NodeId> {
    match site {
        NodeRef::Position(p) => (0..layers.count).map(|l| layers.node(p, l)).collect(),
        NodeRef::Node(id) => vec![id],
    }
}

/// The tables a requantization gadget reads, or `None` for another gadget.
fn requant_tables(gadget: &Gadget) -> Option<Vec<TableId>> {
    match *gadget {
        Gadget::Requant1 { table } => Some(vec![table]),
        Gadget::RequantSplit { low, high, .. } => Some(vec![low, high]),
        Gadget::RequantWide {
            low,
            mid,
            top,
            sign,
            high,
            ..
        } => Some(low.into_iter().chain([mid, top, sign, high]).collect()),
        _ => None,
    }
}

/// `⌈log2 len⌉`.
fn bits(len: u32) -> u32 {
    u64::from(len).next_power_of_two().trailing_zeros()
}

/// `values`, row-major of `shape`, padded per dimension with zeros: the last axis the lowest
/// bits, each axis a block of `⌈log2 len⌉` bits (RFC-0011 rule 1).
fn pad(values: &[i64], shape: &[u32]) -> Vec<i64> {
    let vars: u32 = shape.iter().map(|&d| bits(d)).sum();
    let mut out = vec![0i64; 1 << vars];
    for (flat, &v) in values.iter().enumerate() {
        let (mut rest, mut index, mut shift) = (flat, 0usize, 0u32);
        for &len in shape.iter().rev() {
            index |= (rest % len as usize) << shift;
            rest /= len as usize;
            shift += bits(len);
        }
        out[index] = v;
    }
    out
}

/// The tensors one site reads in one layer: each term's values over its own axes, padded.
type Terms = Vec<(Term, Vec<i64>)>;

/// The tensors one site reads in one layer, as the verifier derives them: each term with its
/// point, as indices into `P`.
type Reads = Vec<(Term, Vec<u32>)>;

/// The sites, their layout, the tensors each reads in each layer, and the instances' tables.
struct Leaf {
    sites: Vec<Site>,
    layout: LeafLayout,
    /// `tensors[s][l]`: each term's values over its own axes, padded.
    tensors: Vec<Vec<Terms>>,
    tables: Vec<TableColumns<Fp>>,
    specs: BTreeMap<TableId, TableSpec>,
}

impl Leaf {
    fn witness(&self) -> Vec<Vec<SiteTensors<'_>>> {
        self.tensors
            .iter()
            .map(|layers| {
                layers
                    .iter()
                    .map(|terms| {
                        SiteTensors::new(terms.iter().map(|(t, v)| (*t, v.as_slice())).collect())
                    })
                    .collect()
            })
            .collect()
    }
}

/// The requantization sites of `model` whose tables hold at most `2^20` rows in every layer, in
/// node order, and each one's tensors from the engine's witness.
fn leaf() -> &'static Leaf {
    static LEAF: OnceLock<Leaf> = OnceLock::new();
    LEAF.get_or_init(|| {
        let m = model();
        let (lowering, layers, selection) = (&m.lowering, &m.layers, &m.selection);
        let spec = |id: TableId| {
            selection
                .tables
                .iter()
                .find(|s| s.id == id)
                .expect("a table of the selection")
        };
        let small = |site: NodeRef| {
            nodes_of(layers, site).iter().all(|&id| {
                selection
                    .of(id)
                    .and_then(|(_, gadget)| requant_tables(&gadget))
                    .is_some_and(|tables| {
                        tables.iter().all(|&t| spec(t).log2_rows <= MAX_TABLE_VARS)
                    })
            })
        };
        let chosen: Vec<NodeRef> = gadget_nodes(lowering, layers)
            .into_iter()
            .filter(|&site| small(site))
            .collect();
        assert_eq!(
            chosen,
            [
                NodeRef::Position(13),
                NodeRef::Position(14),
                NodeRef::Position(32),
                NodeRef::Position(47)
            ],
            "the test's sites"
        );
        let sites: Vec<Site> = chosen
            .iter()
            .map(|&site| Site::requant(lowering, layers, selection, site).expect("a site"))
            .collect();
        let mut tensors = Vec::with_capacity(sites.len());
        for site in &sites {
            let mut per_layer = Vec::new();
            for id in nodes_of(layers, site.node) {
                let node = &lowering.graph.nodes[id.0 as usize];
                let (_, gadget) = selection.of(id).expect("a gadget");
                let requant = Requant::new(&gadget, &selection.tables).expect("a requantization");
                let shape = lowering.graph.tensors[node.inputs[0].0 as usize]
                    .shape
                    .clone();
                let engine = |t: tessera_ir::TensorId| -> Vec<i64> {
                    m.witness.tensor(t).expect("the engine's tensor").to_vec()
                };
                let keys = pad(&engine(node.inputs[0]), &shape);
                // Classes over the output channels, the last axis; each element takes its
                // channel's, the low bits of its padded index.
                let channels = *shape.last().expect("an axis");
                let classes =
                    (requant.classes() > 1).then(|| pad(&engine(node.inputs[1]), &[channels]));
                let per_element: Option<Vec<u8>> = classes.as_ref().map(|c| {
                    (0..keys.len())
                        .map(|x| u8::try_from(c[x & (c.len() - 1)]).expect("a class"))
                        .collect()
                });
                let witness = requant
                    .witness(&keys, per_element.as_deref())
                    .expect("the engine's keys and classes");
                // The gadget's output is D's.
                assert_eq!(
                    witness.tensor(Committed::Y).expect("y"),
                    pad(&engine(node.outputs[0]), &shape),
                    "{:?}: the gadget's y is the engine's",
                    site.node
                );
                let mut terms = vec![(Term::Input, keys.clone())];
                if let Some(c) = classes {
                    terms.push((Term::Class, c));
                }
                for c in requant.constraints::<Fp>().committed {
                    terms.push((
                        Term::Committed(c),
                        witness.tensor(c).expect("a tensor").to_vec(),
                    ));
                }
                per_layer.push(terms);
            }
            tensors.push(per_layer);
        }
        let layout = LeafLayout::new(layers, sites.clone()).expect("a layout");
        let specs: BTreeMap<TableId, TableSpec> = layout
            .instances()
            .iter()
            .map(|i| (i.table, spec(i.table).clone()))
            .collect();
        let tables = specs
            .values()
            .map(|s| {
                TableColumns::new(&GeneratedTable::from_spec(s).expect("a table")).expect("rows")
            })
            .collect();
        Leaf {
            sites,
            layout,
            tensors,
            tables,
            specs,
        }
    })
}

// ----- Sessions and messages -------------------------------------------------------------------

const PROTOCOL: &str = "tessera-protocol/1";
const STATEMENT: [u8; 32] = [0x45; 32];
const SESSION: [u8; 16] = [0x67; 16];

fn transcript() -> Transcript {
    transcript_mode(Mode::NonInteractive)
}

fn transcript_mode(mode: Mode) -> Transcript {
    Transcript::new(&SessionContext {
        protocol: Ascii::new(PROTOCOL).expect("ASCII"),
        mode,
        params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
        statement_digest: STATEMENT,
        session_id: SESSION,
    })
}

fn tv_transcript_mode(mode: Mode) -> tv_hash::transcript::Transcript {
    use tv_hash::transcript as t;
    t::Transcript::new(&t::SessionContext {
        protocol: t::Ascii::new(PROTOCOL).expect("ASCII"),
        mode: match mode {
            Mode::Live => t::Mode::Live,
            Mode::NonInteractive => t::Mode::NonInteractive,
        },
        params: t::ParamSetId(t::Ascii::new("test-small/1").expect("ASCII")),
        statement_digest: STATEMENT,
        session_id: SESSION,
    })
}

fn tv_fp(x: Fp) -> TvFp {
    TvFp::from_canonical(x.to_canonical_u64()).expect("a canonical element")
}

fn tv_fp2(x: Fp2) -> TvFp2 {
    let [c0, c1] = x.coeffs();
    TvFp2::new(tv_fp(c0), tv_fp(c1))
}

fn tv_fp2s(xs: &[Fp2]) -> Vec<TvFp2> {
    xs.iter().copied().map(tv_fp2).collect()
}

/// What the leaf stage sends, in the verifier's types.
#[derive(Clone)]
struct Messages {
    roots: Vec<TvFp2>,
    levels: Vec<tv::SumcheckProof>,
    values: Vec<TvFp2>,
    splits: Vec<TvFp2>,
}

impl Messages {
    fn of(logup: &ProvenLogUp<Fp2>, splits: &[Fp2]) -> Self {
        Self {
            roots: tv_fp2s(logup.roots()),
            levels: logup
                .levels()
                .iter()
                .map(|level| tv::SumcheckProof {
                    rounds: level.proof().rounds.iter().map(|r| tv_fp2s(r)).collect(),
                    final_evals: tv_fp2s(&level.proof().final_evals),
                })
                .collect(),
            values: tv_fp2s(logup.values()),
            splits: tv_fp2s(splits),
        }
    }
}

/// The honest leaf stage: its messages, and the challenge `stage/1/layer/0` the prover's
/// transcript squeezes after it.
struct Honest {
    messages: Messages,
    point: Vec<Fp2>,
    next: Fp2,
}

fn honest() -> &'static Honest {
    static HONEST: OnceLock<Honest> = OnceLock::new();
    HONEST.get_or_init(|| {
        let leaf = leaf();
        let mut t = transcript();
        let proven = prove_leaf(&mut t, &leaf.layout, &leaf.tables, &leaf.witness())
            .expect("the leaf stage");
        let next = t.challenge("stage/1/layer/0");
        Honest {
            messages: Messages::of(proven.logup(), proven.splits()),
            point: proven.point().to_vec(),
            next,
        }
    })
}

/// The leaf stage proven over `instances` in place of the layout's columns, with the honest split
/// values at the leaf point those instances' levels end at.
fn prove_instances(leaf: &Leaf, instances: &[LogUpInstance<'_, Fp>]) -> Messages {
    let mut t = transcript();
    let logup = prove_logup(&mut t, instances, &[], &leaf.layout.trees()).expect("instances");
    let point = leaf_point(&logup);
    let splits = split_values(&leaf.layout, &point, &leaf.witness()).expect("the split");
    t.absorb_message(SPLIT_LABEL, &splits);
    Messages::of(&logup, &splits)
}

// ----- A verifier written from RFC-0045 --------------------------------------------------------

/// Why the verifier rejects, 04 §2: LogUp's roots or leaf checks, a tree level, or the split's
/// check, `Reject(Sumcheck { stage: 0, round: 0 })`.
#[derive(Debug, PartialEq, Eq)]
enum Reject {
    Lookup(LookupError),
    Level(usize, tv::SumcheckError),
    Sumcheck { stage: u16, round: u32 },
    Malformed,
}

const SPLIT: Reject = Reject::Sumcheck { stage: 0, round: 0 };

/// One block, as the verifier derives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VBlock {
    site: usize,
    layer: usize,
    lookup: usize,
    vars: u32,
    offset: u64,
}

/// One instance's column, as the verifier derives it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct VInstance {
    table: TableId,
    blocks: Vec<VBlock>,
    entries: u64,
    height: u32,
}

/// RFC-0045 'The column', from the sites alone: the blocks of each table, by site, layer and
/// lookup, then larger first with ties in that order, at prefix-sum offsets; the tables in table
/// order.
fn derive_instances(sites: &[Site]) -> Vec<VInstance> {
    let mut tables: Vec<(TableId, Vec<VBlock>)> = Vec::new();
    for (site, s) in sites.iter().enumerate() {
        for (layer, gadget) in s.layers.iter().enumerate() {
            for (lookup, l) in gadget.lookups.iter().enumerate() {
                let vars: u32 = l.domain.iter().map(|&d| bits(d)).sum();
                let block = VBlock {
                    site,
                    layer,
                    lookup,
                    vars,
                    offset: 0,
                };
                match tables.iter_mut().find(|(t, _)| *t == l.lookup.table) {
                    Some((_, blocks)) => blocks.push(block),
                    None => tables.push((l.lookup.table, vec![block])),
                }
            }
        }
    }
    tables.sort_by_key(|(table, _)| *table);
    tables
        .into_iter()
        .map(|(table, blocks)| {
            // Larger first, each size's blocks in their order: a bucket per size.
            let mut sorted: Vec<VBlock> = Vec::new();
            for size in (0..64).rev() {
                sorted.extend(blocks.iter().filter(|b| b.vars == size));
            }
            let mut offset = 0u64;
            for block in &mut sorted {
                block.offset = offset;
                offset += 1 << block.vars;
            }
            let height = (0..64).find(|&h| (1u64 << h) >= offset).expect("a height");
            VInstance {
                table,
                blocks: sorted,
                entries: offset,
                height,
            }
        })
        .collect()
}

/// RFC-0045 'What is sent', from a site's gadget alone: the tensors its lookups read, in the order
/// they first read them, the key's terms before the value's, each with its point `π_t(P[0..k])` as
/// indices into `P`.
fn derive_reads(gadget: &SiteGadget) -> Reads {
    let mut reads: Reads = Vec::new();
    for lookup in &gadget.lookups {
        let widths: Vec<u32> = lookup.domain.iter().map(|&d| bits(d)).collect();
        let forms = [Some(&lookup.lookup.key), lookup.lookup.value.as_ref()];
        for form in forms.into_iter().flatten() {
            for &(_, term) in &form.terms {
                if reads.iter().any(|(t, _)| *t == term) {
                    continue;
                }
                let axes: Vec<usize> = lookup
                    .axes
                    .iter()
                    .find(|(t, _)| *t == term)
                    .map_or_else(|| (0..widths.len()).collect(), |(_, a)| a.clone());
                // Axis `a` takes the coordinates after those of the axes behind it.
                let mut coords = Vec::new();
                for &a in axes.iter().rev() {
                    let start: u32 = widths[a + 1..].iter().sum();
                    coords.extend(start..start + widths[a]);
                }
                reads.push((term, coords));
            }
        }
    }
    reads
}

/// Test-data adapter only: production verifier metadata never imports prover descriptors.
fn split_layout(sites: &[Site]) -> tv_core::leaf::split::Layout {
    use tv_core::advice::Auxiliary;
    use tv_core::leaf::split as v;
    let column = |term| match term {
        Term::Input => v::Column::Input(0),
        Term::Class => v::Column::Input(1),
        Term::Committed(Committed::Y) => v::Column::Output(0),
        Term::Committed(role) => v::Column::Auxiliary(match role {
            Committed::ZLo => Auxiliary::RequantLow,
            Committed::H0 => Auxiliary::RequantMiddle,
            Committed::G => Auxiliary::RequantSaturation,
            Committed::Limb(j) => Auxiliary::SoftmaxA(u32::from(j)),
            Committed::Y => unreachable!(),
        }),
    };
    let sites: Vec<_> = sites
        .iter()
        .enumerate()
        .map(|(s, site)| v::Site {
            layers: site
                .layers
                .iter()
                .enumerate()
                .map(|(l, gadget)| {
                    let reads = derive_reads(gadget);
                    let form = |f: &Affine<Fp>| v::Affine {
                        constant: tv_fp(f.constant),
                        terms: f
                            .terms
                            .iter()
                            .map(|&(a, t)| {
                                (
                                    tv_fp(a),
                                    v::Read {
                                        column: column(t),
                                        coordinates: reads
                                            .iter()
                                            .find(|(term, _)| *term == t)
                                            .unwrap()
                                            .1
                                            .clone(),
                                    },
                                )
                            })
                            .collect(),
                    };
                    v::Layer {
                        node: tv_core::lowering::NodeId((s + l * sites.len()) as u32),
                        lookups: gadget
                            .lookups
                            .iter()
                            .map(|lookup| v::Lookup {
                                table: tv_tables::TableId(lookup.lookup.table.0),
                                variables: lookup.domain.iter().map(|&d| bits(d)).sum(),
                                key: form(&lookup.lookup.key),
                                value: lookup.lookup.value.as_ref().map(form),
                                // The ungrouped split reads no token axis.
                                token: None,
                            })
                            .collect(),
                    }
                })
                .collect(),
        })
        .collect();
    v::Layout::new(&sites).expect("independent split descriptors")
}

/// What the verifier ends with on an accepted leaf stage.
struct Accepted {
    /// `P`, from its own last tree level.
    point: Vec<TvFp2>,
    /// LogUp's leaf claims, instance by instance.
    leaves: tvl::LeafClaims,
    /// Its instances.
    instances: Vec<VInstance>,
    /// The challenge `stage/1/layer/0` after the split.
    next: TvFp2,
}

/// The verifier's table of a prover specification: its TCE bytes, decoded and regenerated by the
/// verifier's own crates.
fn tv_table(spec: &TableSpec) -> tv_tables::Table {
    let bytes = encode(spec);
    let decoded = tv_codec::decode(
        &bytes,
        &tv_codec::DecodeLimits::MESSAGE,
        tv_tables::TableSpec::decode,
    )
    .expect("a specification");
    tv_tables::Table::from_spec(&decoded).expect("a table")
}

/// Checks the leaf stage's `messages` for the public `sites` and table `specs`, with the
/// instances the verifier derives.
fn verify(
    sites: &[Site],
    specs: &BTreeMap<TableId, TableSpec>,
    messages: &Messages,
) -> Result<Accepted, Reject> {
    verify_combined(
        sites,
        specs,
        messages,
        derive_instances(sites),
        &[],
        0,
        Mode::NonInteractive,
    )
}

/// Component oracle: the tail remains an outstanding graph claim; its length is public.
/// Forest roots and final leaves are evaluated independently from their explicit test tables.
fn verify_combined(
    sites: &[Site],
    specs: &BTreeMap<TableId, TableSpec>,
    messages: &Messages,
    instances: Vec<VInstance>,
    forests: &[(usize, Vec<i64>)],
    atom_count: usize,
    mode: Mode,
) -> Result<Accepted, Reject> {
    let tables: Vec<tv_tables::Table> = instances
        .iter()
        .map(|i| tv_table(&specs[&i.table]))
        .collect();
    let mut t = tv_transcript_mode(mode);
    let mut trees: Vec<_> = instances
        .iter()
        .zip(&tables)
        .enumerate()
        .flat_map(|(instance, (layout, table))| {
            [
                tv_core::leaf::Tree::Lookup {
                    instance,
                    height: layout.height,
                    has_values: !matches!(table.spec().kind, tv_tables::TableKind::Range { .. }),
                },
                tv_core::leaf::Tree::Table {
                    instance,
                    height: u32::from(table.log2_rows()),
                },
            ]
        })
        .collect();
    let first_forest = trees.len();
    trees.extend(
        forests
            .iter()
            .map(|(height, leaves)| tv_core::leaf::Tree::Forest {
                height: *height as u32,
                trees_log: leaves.len().trailing_zeros() - *height as u32,
            }),
    );
    let schedule = tv_core::leaf::Schedule::new(&trees).expect("independent public layout");
    let checked = schedule
        .verify(
            &mut t,
            tv_core::leaf::Proof {
                roots: &messages.roots,
                levels: &messages.levels,
                values: &messages.values,
            },
            |tree, point| {
                let (height, leaves) = forests.get(tree.checked_sub(first_forest)?)?;
                let roots: Vec<_> = leaves
                    .chunks_exact(1 << height)
                    .map(|tree| tree.iter().product())
                    .collect();
                Some(mle(&roots, point))
            },
            |instance, beta, point| tables.get(instance)?.eval_mle(point, beta).ok(),
        )
        .map_err(|error| match error {
            tv_core::leaf::Error::Lookup(error) => Reject::Lookup(error),
            tv_core::leaf::Error::Level { level, error } => Reject::Level(level as usize, error),
            tv_core::leaf::Error::LevelCount { .. } => Reject::Malformed,
            _ => panic!("verifier metadata or resource fault: {error}"),
        })?;
    for forest in &checked.forests {
        let index = forest.tree - first_forest;
        assert_eq!(
            forest.claim.value,
            mle(&forests[index].1, &forest.claim.point),
            "forest {index} leaves are still owed by the model stages"
        );
    }
    let leaves = checked.lookups.clone();
    let forests = checked.forests.clone();
    let layout = split_layout(sites);
    let bound = layout
        .bind(&mut t, checked, &messages.splits, atom_count, |id| {
            let index = instances.iter().position(|i| i.table.0 == id.0)?;
            let table = &tables[index];
            let (key, value) = table.row(0)?;
            Some((
                key,
                (!matches!(table.spec().kind, tv_tables::TableKind::Range { .. })).then_some(value),
            ))
        })
        .map_err(|error| match error {
            tv_core::leaf::split::Error::Mismatch | tv_core::leaf::split::Error::Count { .. } => {
                SPLIT
            }
            _ => panic!("verifier metadata or resource fault: {error}"),
        })?;
    assert_eq!(bound.atoms, messages.splits[layout.values()..]);
    assert_eq!(bound.forests, forests);
    assert_eq!(
        bound.multiplicities,
        instances
            .iter()
            .zip(&leaves.instances)
            .map(|(i, claim)| (tv_tables::TableId(i.table.0), claim.multiplicity.clone()))
            .collect::<Vec<_>>()
    );
    let point = bound.point;
    let next = t
        .challenge_fp2(tv_hash::transcript::Ascii::new("stage/1/layer/0").expect("ASCII"))
        .expect("a challenge");
    Ok(Accepted {
        point,
        leaves,
        instances,
        next,
    })
}

// ----- Extensions in the verifier's field ------------------------------------------------------

/// The `2^n` weights `eq(point, x)`, built by doubling.
fn eq_weights(point: &[TvFp2]) -> Vec<TvFp2> {
    let mut weights = vec![TvFp2::ONE];
    for &r in point {
        let high: Vec<TvFp2> = weights.iter().map(|&e| e * r).collect();
        for (e, h) in weights.iter_mut().zip(&high) {
            *e -= *h;
        }
        weights.extend(high);
    }
    weights
}

/// `f̃(point)` of `2^{|point|}` integers in the centered representation.
fn mle(values: &[i64], point: &[TvFp2]) -> TvFp2 {
    assert_eq!(
        values.len(),
        1 << point.len(),
        "2^n entries at n coordinates"
    );
    values
        .iter()
        .zip(eq_weights(point))
        .fold(TvFp2::ZERO, |sum, (&v, e)| {
            sum + e * TvFp2::from(tv_fp(Fp::from_i64(v)))
        })
}

// ----- End to end ------------------------------------------------------------------------------

/// The leaf stage of a conformance model, through `tessera-logup` and the split, is accepted, and
/// each split value is its tensor's extension at its point: RFC-0037's per-layer claim.
#[test]
fn the_leaf_stage_of_a_conformance_model_is_accepted() {
    let leaf = leaf();
    let honest = honest();
    let accepted = verify(&leaf.sites, &leaf.specs, &honest.messages)
        .unwrap_or_else(|reject| panic!("rejected: {reject:?}"));
    assert_eq!(
        accepted.point,
        tv_fp2s(&honest.point),
        "the prover's P is the verifier's"
    );
    // The split is absorbed as one TCE Vec<Fp2> under stage/0/split: both transcripts agree after.
    assert_eq!(
        accepted.next,
        tv_fp2(honest.next),
        "the transcripts agree after the split"
    );
    // Each value is its tensor's extension at π_t(P[0..k]), in the order of the reads.
    let mut sent = honest.messages.splits.iter();
    for (s, site) in leaf.sites.iter().enumerate() {
        for (l, gadget) in site.layers.iter().enumerate() {
            for (term, coords) in derive_reads(gadget) {
                let values = &leaf.tensors[s][l]
                    .iter()
                    .find(|(t, _)| *t == term)
                    .expect("a tensor")
                    .1;
                let at: Vec<TvFp2> = coords.iter().map(|&c| accepted.point[c as usize]).collect();
                assert_eq!(
                    sent.next(),
                    Some(&mle(values, &at)),
                    "site {s}, layer {l}, {term:?}"
                );
            }
        }
    }
    assert!(
        sent.next().is_none(),
        "one value per tensor, site and layer"
    );
}

/// Every split value is checked: one changed by 1, a missing one or an extra one is
/// `Reject(Sumcheck { stage: 0, round: 0 })`, after LogUp accepts.
#[test]
fn a_tampered_split_value_is_rejected_at_stage_0() {
    let leaf = leaf();
    let honest = &honest().messages;
    assert_eq!(
        honest.splits.len(),
        26,
        "4, 4, 3 and 2 tensors in each of 2 layers"
    );
    for i in 0..honest.splits.len() {
        let mut bad = honest.clone();
        bad.splits[i] += TvFp2::ONE;
        assert_eq!(
            verify(&leaf.sites, &leaf.specs, &bad).err(),
            Some(SPLIT),
            "split value {i}"
        );
    }
    let mut longer = honest.clone();
    longer.splits.push(TvFp2::ONE);
    assert_eq!(verify(&leaf.sites, &leaf.specs, &longer).err(), Some(SPLIT));
    let mut shorter = honest.clone();
    shorter.splits.pop();
    assert_eq!(
        verify(&leaf.sites, &leaf.specs, &shorter).err(),
        Some(SPLIT)
    );
}

/// A wrong value in a block, `y + 1` at one element of site 47's layer-0 output, is no row of its
/// table: LogUp's root check rejects its instance, whatever the split.
#[test]
fn a_tampered_lookup_in_a_block_fails_the_root_check() {
    let leaf = leaf();
    let site = leaf
        .sites
        .iter()
        .position(|s| s.node == NodeRef::Position(47))
        .expect("site 47");
    let mut tensors = leaf.tensors.clone();
    let y = tensors[site][0]
        .iter_mut()
        .find(|(t, _)| *t == Term::Committed(Committed::Y))
        .expect("y");
    y.1[3] += 1;
    let witness: Vec<Vec<SiteTensors<'_>>> = tensors
        .iter()
        .map(|layers| {
            layers
                .iter()
                .map(|terms| {
                    SiteTensors::new(terms.iter().map(|(t, v)| (*t, v.as_slice())).collect())
                })
                .collect()
        })
        .collect();
    let proven = prove_leaf(&mut transcript(), &leaf.layout, &leaf.tables, &witness)
        .expect("the prover builds honest trees over any lookups");
    let table = leaf.sites[site].layers[0].lookups[0].lookup.table;
    let instance = derive_instances(&leaf.sites)
        .iter()
        .position(|i| i.table == table)
        .expect("its instance");
    assert_eq!(
        verify(
            &leaf.sites,
            &leaf.specs,
            &Messages::of(proven.logup(), proven.splits())
        )
        .err(),
        Some(Reject::Lookup(LookupError::Unbalanced { instance }))
    );
}

// ----- One point per size ----------------------------------------------------------------------

/// Two instances of different heights hold blocks of `2^k` entries of one site: both blocks lie at
/// `P[0..k]`. Each instance's lookup tree ends at `σ_h(P[0..h])`, so the natural layout would put
/// the two blocks at different points, and a prover that lays its columns out naturally is
/// rejected by the split's check after LogUp accepts.
#[test]
fn blocks_of_one_size_share_their_point_in_instances_of_every_height() {
    let leaf = leaf();
    let accepted = verify(&leaf.sites, &leaf.specs, &honest().messages).expect("accepted");
    let p = &accepted.point;
    // Each lookup tree's leaf point is P[0..h] with bits 1 to h − 1 reversed: (ρ, r_{n−h+1}, …).
    for (k, instance) in accepted.instances.iter().enumerate() {
        let leaf_point = &accepted.leaves.instances[k].key.point;
        let h = instance.height as usize;
        assert_eq!(leaf_point.len(), h, "instance {k}");
        for (j, &coordinate) in leaf_point.iter().enumerate() {
            let column = if j == 0 { 0 } else { h - j };
            assert_eq!(coordinate, p[column], "instance {k}, leaf bit {j}");
        }
    }
    // Site 13 reads Range { 19 } and its high table in each layer: blocks of 2^9 entries in
    // instances of heights 11 and 9, its layers' high tables differing.
    let site = leaf
        .sites
        .iter()
        .position(|s| s.node == NodeRef::Position(13))
        .expect("site 13");
    let holding = |layer: usize, lookup: usize| {
        accepted
            .instances
            .iter()
            .position(|i| {
                i.blocks
                    .iter()
                    .any(|b| (b.site, b.layer, b.lookup) == (site, layer, lookup))
            })
            .expect("an instance")
    };
    let (low, high) = (holding(0, 0), holding(0, 1));
    let (a, b) = (&accepted.instances[low], &accepted.instances[high]);
    assert_eq!((a.height, b.height), (11, 9), "two heights");
    let k = 9;
    assert!(a.blocks.iter().chain(&b.blocks).any(|bl| bl.vars == k));
    assert_ne!(
        accepted.instances[holding(0, 1)].table,
        accepted.instances[holding(1, 1)].table,
        "the site's layers read different high tables"
    );
    // Laid out naturally, the blocks would lie at the trees' own low coordinates: different.
    let natural = |i: usize| accepted.leaves.instances[i].key.point[..k as usize].to_vec();
    assert_ne!(
        natural(low),
        natural(high),
        "a natural layout splits the point"
    );
    // A prover that lays out its columns naturally passes LogUp and fails the split's check.
    let columns = LeafColumns::new(&leaf.layout, &leaf.tables, &leaf.witness()).expect("columns");
    let reordered: Vec<(Vec<Fp>, Option<Vec<Fp>>)> = columns
        .columns()
        .iter()
        .zip(leaf.layout.instances())
        .map(|(c, i)| {
            let h = i.height;
            let in_order = |leaves: &[Fp]| -> Vec<Fp> {
                (0..leaves.len() as u64)
                    .map(|q| leaves[leaf_index(h, q) as usize])
                    .collect()
            };
            (in_order(c.keys()), c.values().map(in_order))
        })
        .collect();
    let instances = instances_of(leaf, &columns, &reordered);
    assert_eq!(
        verify(&leaf.sites, &leaf.specs, &prove_instances(leaf, &instances)).err(),
        Some(SPLIT)
    );
}

/// The LogUp instances of `columns` with each instance's lookups replaced by `lookups`.
fn instances_of<'a>(
    leaf: &'a Leaf,
    columns: &'a LeafColumns,
    lookups: &'a [(Vec<Fp>, Option<Vec<Fp>>)],
) -> Vec<LogUpInstance<'a, Fp>> {
    columns
        .columns()
        .iter()
        .zip(lookups)
        .map(|(c, (keys, values))| {
            let table = leaf
                .tables
                .iter()
                .find(|t| t.table() == c.table())
                .expect("its table");
            LogUpInstance::new(
                table.tuples(),
                Tuples::new(keys, values.as_deref()),
                c.multiplicities(),
            )
        })
        .collect()
}

// ----- One claim per input ---------------------------------------------------------------------

/// Every tensor a site's lookups read has exactly one value per layer: the split reads each term of
/// the forms once, in each real layer.
#[test]
fn every_tensor_a_site_reads_has_one_value_per_layer() {
    let leaf = leaf();
    let mut total = 0;
    for (s, site) in leaf.sites.iter().enumerate() {
        assert_eq!(site.layers.len(), 2, "a template position's real layers");
        for (l, gadget) in site.layers.iter().enumerate() {
            let reads = leaf.layout.reads(s, l).expect("its reads");
            let mut read: Vec<Term> = Vec::new();
            for lookup in &gadget.lookups {
                let forms = [Some(&lookup.lookup.key), lookup.lookup.value.as_ref()];
                for &(_, term) in forms.into_iter().flatten().flat_map(|f| &f.terms) {
                    if !read.contains(&term) {
                        read.push(term);
                    }
                }
            }
            let sent: Vec<Term> = reads.iter().map(|r| r.term).collect();
            assert_eq!(
                sent, read,
                "site {s}, layer {l}: each tensor once, in first-read order"
            );
            let classes = reads.iter().find(|r| r.term == Term::Class);
            if let Some(class) = classes {
                // A channel's class is read over the channel axis: the lowest coordinates.
                let channels = bits(*gadget.lookups[0].domain.last().expect("an axis"));
                assert_eq!(class.coords, (0..channels).collect::<Vec<_>>());
            }
            total += reads.len();
        }
    }
    assert_eq!(leaf.layout.split_len(), total);
    assert_eq!(honest().messages.splits.len(), total);
}

/// A gadget whose lookups read one tensor over two sets of axes is refused, `ERR-INTERNAL`.
#[test]
fn a_gadget_reading_one_tensor_over_two_sets_of_axes_is_refused() {
    let leaf = leaf();
    let mut sites = leaf.sites.clone();
    // Site 13's RequantSplit reads z_lo in both its lookups: its low part over the channel axis
    // alone, its high key over every axis.
    let site = &mut sites[0];
    assert_eq!(site.node, NodeRef::Position(13));
    let low = &mut site.layers[1].lookups[0];
    assert_eq!(
        low.lookup.key.terms,
        [(Fp::ONE, Term::Committed(Committed::ZLo))]
    );
    low.axes = vec![(Term::Committed(Committed::ZLo), vec![1])];
    let refused = LeafLayout::new(&model().layers, sites).expect_err("two sets of axes");
    assert_eq!(
        refused,
        LeafError::Internal {
            rule: "a gadget's lookups read each tensor over one set of axes (RFC-0045 'One \
                   claim per input')"
        }
    );
    assert_eq!(refused.code().as_str(), "ERR-INTERNAL");
}

// ----- The layout ------------------------------------------------------------------------------

/// Asserts RFC-0045 'The column' of `layout`: blocks larger first, each at a multiple of its size,
/// one after another from 0, `m` their total and `h = ⌈log2 m⌉`.
fn assert_aligned(layout: &LeafLayout) {
    for instance in layout.instances() {
        let mut next = 0u64;
        for pair in instance.blocks.windows(2) {
            assert!(
                pair[0].vars >= pair[1].vars,
                "{:?}: larger first",
                instance.table
            );
        }
        for block in &instance.blocks {
            assert_eq!(
                block.offset, next,
                "{:?}: one after another",
                instance.table
            );
            assert_eq!(
                block.offset % block.entries(),
                0,
                "{:?}: aligned",
                instance.table
            );
            next += block.entries();
        }
        assert_eq!(instance.entries, next);
        let h = instance.height;
        assert!(
            next <= 1 << h && (h == 0 || next > 1 << (h - 1)),
            "h = ⌈log2 m⌉"
        );
    }
}

/// The conformance layout and seeded synthetic ones: offsets aligned to their blocks' sizes, and
/// the padding, row 0's tuple, exactly the tail of each column.
#[test]
fn offsets_are_aligned_and_the_padding_is_the_tail() {
    let leaf = leaf();
    assert_aligned(&leaf.layout);
    assert_eq!(
        leaf.layout.instances(),
        &derive_instances_as_layout(&leaf.sites)[..]
    );
    // Range { 19 } holds 2^9 + 2^9 + 2^8 + 2^8 entries: its last quarter pads.
    let columns = LeafColumns::new(&leaf.layout, &leaf.tables, &leaf.witness()).expect("columns");
    let mut padded = 0;
    for (instance, column) in leaf.layout.instances().iter().zip(columns.columns()) {
        let rows = leaf
            .tables
            .iter()
            .find(|t| t.table() == instance.table)
            .expect("rows")
            .tuples();
        let h = instance.height;
        for q in 0..1u64 << h {
            let leaf_at = leaf_index(h, q) as usize;
            let is_row_0 = column.keys()[leaf_at] == rows.keys[0]
                && column.values().map(|v| v[leaf_at]) == rows.values.map(|v| v[0]);
            if q >= instance.entries {
                assert!(is_row_0, "{:?}: entry {q} pads with row 0", instance.table);
                padded += 1;
            }
        }
    }
    assert_eq!(
        padded, 512,
        "Range {{ 19 }}'s 2048 − 1536 entries, the only padding"
    );
    // Synthetic sites at the model's positions, blocks of random sizes into four tables: one domain
    // per site, and in each layer one to three lookups that read its input.
    let mut rng = SplitMix64(0x0045);
    let layers = &model().layers;
    for _ in 0..64 {
        let mut sites = Vec::new();
        for position in 0..layers.width {
            if rng.below(3) != 0 {
                continue;
            }
            let domain: Vec<u32> = (0..1 + rng.below(3))
                .map(|_| 1 + rng.below(40) as u32)
                .collect();
            let gadgets = (0..layers.count)
                .map(|_| {
                    let domain = domain.clone();
                    let lookups = (0..1 + rng.below(3))
                        .map(|_| SiteLookup {
                            domain: domain.clone(),
                            lookup: tessera_gadgets::Lookup {
                                table: TableId(rng.below(4) as u32),
                                key: Affine {
                                    constant: Fp::ZERO,
                                    terms: vec![(Fp::ONE, Term::Input)],
                                },
                                value: None,
                            },
                            axes: Vec::new(),
                            token: None,
                        })
                        .collect();
                    SiteGadget { lookups }
                })
                .collect();
            sites.push(Site {
                node: NodeRef::Position(position),
                layers: gadgets,
            });
        }
        let layout = LeafLayout::new(layers, sites.clone()).expect("a layout");
        assert_aligned(&layout);
        assert_eq!(layout.instances(), &derive_instances_as_layout(&sites)[..]);
    }
}

/// The verifier's derivation, as the layout's type.
fn derive_instances_as_layout(sites: &[Site]) -> Vec<tessera_prover::leaf::InstanceLayout> {
    derive_instances(sites)
        .into_iter()
        .map(|i| tessera_prover::leaf::InstanceLayout {
            table: i.table,
            blocks: i
                .blocks
                .iter()
                .map(|b| tessera_prover::leaf::Block {
                    site: b.site,
                    layer: b.layer,
                    lookup: b.lookup,
                    vars: b.vars,
                    offset: b.offset,
                })
                .collect(),
            entries: i.entries,
            height: i.height,
        })
        .collect()
}

/// `σ_h` keeps bit 0, reverses bits 1 to `h − 1`, and is its own inverse.
#[test]
fn the_leaf_order_is_an_involution() {
    let by_bits = |h: u32, i: u64| -> u64 {
        let mut out = i & 1;
        for j in 1..h {
            out |= ((i >> j) & 1) << (h - j);
        }
        out
    };
    for h in 0..=12u32 {
        for i in 0..1u64 << h {
            let s = leaf_index(h, i);
            assert_eq!(s, by_bits(h, i), "σ_{h}({i})");
            assert_eq!(leaf_index(h, s), i, "σ_{h}(σ_{h}({i}))");
        }
    }
    let mut rng = SplitMix64(7);
    for h in 13..=48u32 {
        for _ in 0..256 {
            let i = rng.next() & ((1u64 << h) - 1);
            assert_eq!(leaf_index(h, i), by_bits(h, i), "σ_{h}({i})");
            assert_eq!(leaf_index(h, leaf_index(h, i)), i, "σ_{h}(σ_{h}({i}))");
        }
    }
}

/// Two blocks of one size in one instance, site 13's layers' low parts in `Range { 19 }`, swapped:
/// the verifier that expects them swapped rejects the honest proof, and a prover that swaps them
/// passes LogUp, whose lookups are the same multiset, and fails the split's check. The order is
/// load-bearing.
#[test]
fn swapping_two_blocks_of_one_size_changes_the_check() {
    let leaf = leaf();
    let instances = derive_instances(&leaf.sites);
    let (k, a, b) = instances
        .iter()
        .enumerate()
        .find_map(|(k, i)| {
            let a = i
                .blocks
                .iter()
                .position(|b| (b.site, b.layer, b.lookup) == (0, 0, 0))?;
            let b = i
                .blocks
                .iter()
                .position(|b| (b.site, b.layer, b.lookup) == (0, 1, 0))?;
            Some((k, a, b))
        })
        .expect("site 13's low parts share Range { 19 }");
    assert_eq!(
        instances[k].blocks[a].vars, instances[k].blocks[b].vars,
        "one size"
    );
    // The prover's order: the two blocks' keys swapped in the column, the rest as laid out.
    let columns = LeafColumns::new(&leaf.layout, &leaf.tables, &leaf.witness()).expect("columns");
    let layout = &leaf.layout.instances()[k];
    let lookups: Vec<(Vec<Fp>, Option<Vec<Fp>>)> = columns
        .columns()
        .iter()
        .enumerate()
        .map(|(j, c)| {
            let mut keys = c.keys().to_vec();
            if j == k {
                let (x, y) = (layout.blocks[a], layout.blocks[b]);
                for e in 0..x.entries() {
                    let (p, q) = (
                        leaf_index(layout.height, x.offset + e) as usize,
                        leaf_index(layout.height, y.offset + e) as usize,
                    );
                    keys.swap(p, q);
                }
            }
            (keys, c.values().map(<[Fp]>::to_vec))
        })
        .collect();
    let instances = instances_of(leaf, &columns, &lookups);
    assert_eq!(
        verify(&leaf.sites, &leaf.specs, &prove_instances(leaf, &instances)).err(),
        Some(SPLIT),
        "the prover's order"
    );
}

// ----- Synthetic sites: a padded table with values ---------------------------------------------

/// Small tables and the requantizations over them: `Requant1` into `Requant { 4, [2], −1, 1 }`,
/// whose row 0 is `(0, −1)`, and `RequantSplit` into `Range { 2 }` and `Requant { 4, [1], −1, 1 }`.
struct Small {
    tables: [GeneratedTable; 3],
    specs: Vec<TableSpec>,
    requant1: Requant,
    split: Requant,
}

fn small() -> Small {
    let requant = |id: u32, r: u8| {
        GeneratedTable::new(
            TableId(id),
            tessera_tables::TableKind::Requant {
                b: 4,
                r_by_class: vec![r],
                lo: -1,
                hi: 1,
            },
        )
        .expect("a table")
    };
    let low = GeneratedTable::new(TableId(1), tessera_tables::TableKind::Range { bits: 2 })
        .expect("a table");
    let tables = [requant(0, 2), low, requant(2, 1)];
    let specs: Vec<TableSpec> = tables
        .iter()
        .map(|t| tessera_tables::Table::spec(t).clone())
        .collect();
    let requant1 = Requant::new(&Gadget::Requant1 { table: TableId(0) }, &specs).expect("Requant1");
    let split = Requant::new(
        &Gadget::RequantSplit {
            split: 2,
            low: TableId(1),
            high: TableId(2),
        },
        &specs,
    )
    .expect("RequantSplit");
    Small {
        tables,
        specs,
        requant1,
        split,
    }
}

/// Synthetic requantization sites at the model's positions 1 and 2 and in its prologue, over the
/// small tables: `Requant1` from a prologue site of 5 elements and a template site of `3 × 5` in
/// each layer, and `RequantSplit` over `2 × 2`.
fn synthetic() -> Leaf {
    let Small {
        tables: small_tables,
        specs,
        requant1,
        split,
    } = small();
    let mut rng = SplitMix64(0x5e);
    let mut site = |node: NodeRef, gadget: &Requant, shape: &[u32], layers: usize| {
        let half = 1i64 << (gadget.key_bits() - 1);
        let mut per_layer = Vec::new();
        for _ in 0..layers {
            let count: usize = shape.iter().map(|&d| d as usize).product();
            let keys: Vec<i64> = (0..count)
                .map(|_| rng.below(2 * half as u64) as i64 - half)
                .collect();
            let keys = pad(&keys, shape);
            let witness = gadget.witness(&keys, None).expect("keys in range");
            let mut terms = vec![(Term::Input, keys.clone())];
            for c in gadget.constraints::<Fp>().committed {
                terms.push((
                    Term::Committed(c),
                    witness.tensor(c).expect("a tensor").to_vec(),
                ));
            }
            per_layer.push(terms);
        }
        let gadgets = vec![SiteGadget::requant(gadget, shape, None); layers];
        (
            Site {
                node,
                layers: gadgets,
            },
            per_layer,
        )
    };
    let built = [
        site(NodeRef::Node(NodeId(0)), &requant1, &[5], 1),
        site(NodeRef::Position(1), &requant1, &[3, 5], 2),
        site(NodeRef::Position(2), &split, &[2, 2], 2),
    ];
    let (sites, tensors): (Vec<Site>, Vec<_>) = built.into_iter().unzip();
    let layout = LeafLayout::new(&model().layers, sites.clone()).expect("a layout");
    let tables = small_tables
        .iter()
        .map(|t| TableColumns::new(t).expect("rows"))
        .collect();
    Leaf {
        sites,
        layout,
        tensors,
        tables,
        specs: specs.into_iter().map(|s| (s.id, s)).collect(),
    }
}

/// A prologue site and template sites into a `Requant` table of `32 + 32 + 8` entries: its column
/// pads 56 entries with row 0's tuple, value included, and the leaf stage is accepted. A prover
/// that pads with another row, row 1, and counts it, passes LogUp, every lookup being a row, and
/// fails the split's check, whose padding term is row 0's.
#[test]
fn the_padding_is_row_0s_tuple_which_the_check_pins() {
    let leaf = synthetic();
    let padded = &leaf.layout.instances()[0];
    assert_eq!(
        (padded.table, padded.entries, padded.height),
        (TableId(0), 72, 7)
    );
    assert_eq!(
        padded
            .blocks
            .iter()
            .map(|b| (b.site, b.layer, b.vars, b.offset))
            .collect::<Vec<_>>(),
        [(1, 0, 5, 0), (1, 1, 5, 32), (0, 0, 3, 64)],
        "larger first: the template site's layers, then the prologue's"
    );
    let mut t = transcript();
    let proven =
        prove_leaf(&mut t, &leaf.layout, &leaf.tables, &leaf.witness()).expect("the leaf stage");
    let messages = Messages::of(proven.logup(), proven.splits());
    let accepted = verify(&leaf.sites, &leaf.specs, &messages)
        .unwrap_or_else(|reject| panic!("rejected: {reject:?}"));
    assert_eq!(accepted.point.len(), 7, "H = 7, the padded column's height");
    assert_eq!(
        messages.splits.len(),
        2 + 2 * 2 + 2 * 3,
        "z and y once and twice, then z_lo, z and y twice"
    );
    // The padding with row 1's tuple, counted on row 1: every lookup is a row.
    let columns = LeafColumns::new(&leaf.layout, &leaf.tables, &leaf.witness()).expect("columns");
    let rows = leaf.tables[0].tuples();
    let (key_1, value_1) = (rows.keys[1], rows.values.expect("values")[1]);
    let mut keys = columns.columns()[0].keys().to_vec();
    let mut values = columns.columns()[0].values().expect("values").to_vec();
    assert_ne!(
        (
            keys[leaf_index(7, 72) as usize],
            values[leaf_index(7, 72) as usize]
        ),
        (key_1, value_1),
        "row 0 pads"
    );
    for q in 72..128 {
        let at = leaf_index(7, q) as usize;
        keys[at] = key_1;
        values[at] = value_1;
    }
    let counted: Vec<u32> = keys
        .iter()
        .map(|k| u32::try_from(k.to_canonical_u64()).expect("a row"))
        .collect();
    let multiplicities =
        tessera_logup::multiplicities(rows, &Secret::new(counted)).expect("rows of the table");
    let mut instances = columns.instances(&leaf.tables).expect("instances");
    instances[0] = LogUpInstance::new(rows, Tuples::new(&keys, Some(&values)), &multiplicities);
    assert_eq!(
        verify(
            &leaf.sites,
            &leaf.specs,
            &prove_instances(&leaf, &instances)
        )
        .err(),
        Some(SPLIT)
    );
}

/// A tensor read over an outer axis alone, a row's value broadcast over the channels as
/// `SoftmaxRow`'s maxima are over the keys, lies at its axis's coordinates `π_t(P[0..k])`, the high
/// ones of its block: the block's extension at `P[0..k]` is the form at that value, and the leaf
/// stage is accepted. Read at `P[0..k_t]`, its value would break the identity.
#[test]
fn a_tensor_read_over_an_outer_axis_lies_at_its_coordinates() {
    let table = GeneratedTable::new(TableId(0), tessera_tables::TableKind::Range { bits: 4 })
        .expect("a table");
    let spec = tessera_tables::Table::spec(&table).clone();
    // The key `x + 4·c_row` over `[4, 8]`, `c` read over the rows, axis 0.
    let lookup = SiteLookup {
        domain: vec![4, 8],
        token: None,
        lookup: tessera_gadgets::Lookup {
            table: TableId(0),
            key: Affine {
                constant: Fp::ZERO,
                terms: vec![
                    (Fp::ONE, Term::Input),
                    (Fp::from_u64_reduced(4), Term::Class),
                ],
            },
            value: None,
        },
        axes: vec![(Term::Class, vec![0])],
    };
    let gadget = SiteGadget {
        lookups: vec![lookup],
    };
    let mut rng = SplitMix64(0x0a);
    let layers = &model().layers;
    let mut tensors = Vec::new();
    for _ in 0..layers.count {
        let x: Vec<i64> = (0..32).map(|_| rng.below(4) as i64).collect();
        let c: Vec<i64> = (0..4).map(|_| rng.below(4) as i64).collect();
        tensors.push(vec![(Term::Input, x), (Term::Class, c)]);
    }
    let sites = vec![Site {
        node: NodeRef::Position(5),
        layers: vec![gadget; layers.count as usize],
    }];
    let layout = LeafLayout::new(layers, sites.clone()).expect("a layout");
    let class = &layout.reads(0, 0).expect("reads")[1];
    assert_eq!((class.term, &class.coords[..]), (Term::Class, &[3, 4][..]));
    let leaf = Leaf {
        sites,
        layout,
        tensors: vec![tensors],
        tables: vec![TableColumns::new(&table).expect("rows")],
        specs: [(TableId(0), spec)].into_iter().collect(),
    };
    let proven = prove_leaf(
        &mut transcript(),
        &leaf.layout,
        &leaf.tables,
        &leaf.witness(),
    )
    .expect("the leaf stage");
    let messages = Messages::of(proven.logup(), proven.splits());
    let accepted = verify(&leaf.sites, &leaf.specs, &messages)
        .unwrap_or_else(|reject| panic!("rejected: {reject:?}"));
    // The value sent is the rows' extension at P[3..5], not at P[0..2].
    let rows = &leaf.tensors[0][0][1].1;
    let p = &accepted.point;
    assert_eq!(messages.splits[1], mle(rows, &p[3..5]));
    assert_ne!(messages.splits[1], mle(rows, &p[0..2]));
}

/// Every real layer of a site reads the same tensors over the same axes: a site whose layers'
/// gadgets read different tensors, `Requant1` in one layer and `RequantSplit`'s `z_lo` besides in
/// the other, or one tensor over different axes, the classes over the rows in one layer, is
/// refused, `ERR-INTERNAL`.
#[test]
fn a_site_whose_layers_read_different_tensors_is_refused() {
    let layers = &model().layers;
    let rule = LeafError::Internal {
        rule: "a site's real layers read the same tensors over the same axes (RFC-0045 'One \
               claim per input')",
    };
    let small = small();
    let mixed = Site {
        node: NodeRef::Position(1),
        layers: vec![
            SiteGadget::requant(&small.requant1, &[3, 5], None),
            SiteGadget::requant(&small.split, &[3, 5], None),
        ],
    };
    let refused = LeafLayout::new(layers, vec![mixed]).expect_err("different tensors");
    assert_eq!(refused, rule);
    assert_eq!(refused.code().as_str(), "ERR-INTERNAL");
    // Site 13's classes, read over the channels in layer 0, over the rows in layer 1.
    let mut site = leaf().sites[0].clone();
    assert_eq!(site.node, NodeRef::Position(13));
    let high = &mut site.layers[1].lookups[1];
    assert_eq!(high.axes, [(Term::Class, vec![1])]);
    high.axes = vec![(Term::Class, vec![0])];
    assert_eq!(
        LeafLayout::new(layers, vec![site]).expect_err("different axes"),
        rule
    );
}

/// The layout refuses sites out of node order or repeated, a site without one gadget per real
/// layer, and a node inside the layers named as a node: `ERR-INTERNAL`.
#[test]
fn the_layout_refuses_sites_out_of_order_or_without_their_layers() {
    let leaf = leaf();
    let layers = &model().layers;
    let refused = |sites: Vec<Site>| LeafLayout::new(layers, sites).expect_err("refused");
    let order = LeafError::Internal {
        rule: "the sites come in node order, each once",
    };
    let swapped = vec![leaf.sites[1].clone(), leaf.sites[0].clone()];
    assert_eq!(refused(swapped), order);
    let repeated = vec![leaf.sites[0].clone(), leaf.sites[0].clone()];
    assert_eq!(refused(repeated), order);
    let mut short = leaf.sites[0].clone();
    short.layers.pop();
    assert_eq!(
        refused(vec![short]),
        LeafError::Internal {
            rule: "a site has one gadget per real layer, or one outside the layers"
        }
    );
    let mut inside = leaf.sites[3].clone();
    inside.node = NodeRef::Node(layers.node(47, 1));
    inside.layers.truncate(1);
    let error = refused(vec![inside]);
    assert_eq!(
        error,
        LeafError::Internal {
            rule: "a site is a template position or a node outside the layers"
        }
    );
    assert_eq!(error.code().as_str(), "ERR-INTERNAL");
}

/// SplitMix64 (Steele, Lea and Flood, 2014): reproducible inputs, nothing secret.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// RFC-0037/0045: lookup values and graph atom values share one transcript message.
/// This composes components with synthetic lookup sites and forest roots; it is not a
/// complete model proof. Every returned forest claim is checked against its explicit leaves.
#[test]
fn lookup_and_graph_splits_have_one_transcript_message() {
    struct Trace {
        inner: Transcript,
        split_messages: usize,
        first_layer: Option<Fp2>,
    }
    impl SumcheckTranscript<Fp2> for Trace {
        fn absorb_message(&mut self, label: &str, values: &[Fp2]) {
            if label == SPLIT_LABEL {
                self.split_messages += 1;
            }
            self.inner.absorb_message(label, values);
        }
        fn challenge(&mut self, label: &str) -> Fp2 {
            let value = self.inner.challenge(label);
            if label == "stage/1/layer/0" {
                self.first_layer = Some(value);
            }
            value
        }
    }
    let (model, tokens) = conformance("tiny-2-affine");
    let lowering = lower(PublicModel::of(&model), tokens.len() as u32).expect("lowering");
    let witness = Engine::new(&lowering)
        .expect("engine")
        .run(&model, &Secret::new(tokens))
        .expect("witness");
    let schedule = tessera_compiler::derive_schedule(&lowering).expect("graph schedule");
    let leaf = synthetic();
    let longest = schedule.stages[0]
        .claims
        .iter()
        .map(|c| c.point.len())
        .max()
        .unwrap();
    // Forests join before, beside and after the lookup trees; one is a single product tree.
    let forest_data: Vec<_> = [(longest.max(9), 1), (3, 2), (1, 0)]
        .into_iter()
        .map(|(height, vars)| {
            let mut leaves = vec![1i64; 1 << (height + vars)];
            for (tree, row) in leaves.chunks_exact_mut(1 << height).enumerate() {
                row[0] = tree as i64 + 2;
                let last = row.len() - 1;
                row[last] = 3;
            }
            (height, leaves)
        })
        .collect();
    let fields: Vec<Vec<_>> = forest_data
        .iter()
        .map(|(_, leaves)| leaves.iter().map(|&x| Fp::from_i64(x)).collect())
        .collect();
    let forests: Vec<_> = forest_data
        .iter()
        .zip(&fields)
        .map(|((h, _), leaves)| Forest::new(leaves, *h))
        .collect();
    for mode in [Mode::Live, Mode::NonInteractive] {
        let mut trace = Trace {
            inner: transcript_mode(mode),
            split_messages: 0,
            first_layer: None,
        };
        let lookup = tessera_prover::leaf::prepare_leaf(
            &mut trace,
            &leaf.layout,
            &leaf.tables,
            &leaf.witness(),
            &forests,
        )
        .expect("shared trees");
        assert_eq!(lookup.logup().forest_claims().len(), forests.len());
        let points: Vec<_> = schedule.stages[0]
            .claims
            .iter()
            .map(|c| lookup.point()[..c.point.len()].to_vec())
            .collect();
        let mut graph = tessera_prover::StageProver::new(&lowering, &schedule, &witness);
        assert!(
            graph.leaf_split_values().is_err(),
            "split before claims is refused"
        );
        let _claims = graph.leaf(&points).expect("graph claims");
        let atoms = graph.leaf_split_values().expect("atom splits");
        assert!(!atoms.is_empty(), "a real mixed virtual/committed key");
        let proven = lookup.finish(&atoms).expect("one combined split");
        assert_eq!(&proven.splits()[leaf.layout.split_len()..], &atoms);
        assert!(graph.leaf_split_values().is_err(), "no second graph split");
        assert_eq!(trace.split_messages, 1, "INV-FS-4: a single stage/0/split");
        let messages = Messages::of(proven.logup(), proven.splits());
        let verify = |m: &Messages| {
            verify_combined(
                &leaf.sites,
                &leaf.specs,
                m,
                derive_instances(&leaf.sites),
                &forest_data,
                atoms.len(),
                mode,
            )
        };
        let accepted = verify(&messages).expect("independent leaf checks");
        assert_eq!(accepted.point, tv_fp2s(proven.point()));
        for stage in 1..schedule.stages.len() {
            let _proof = graph
                .prove_stage(&mut trace, stage as u16)
                .expect("graph reduction");
        }
        assert_eq!(
            accepted.next,
            tv_fp2(trace.first_layer.expect("graph layer challenge")),
            "the full combined vector binds the next stage"
        );
        assert!(graph.pool().iter().all(Option::is_some));
        // The first two levels are forest-only, then fractions join the same batch.
        for level in [0, messages.levels.len() - 1] {
            let mut bad = messages.clone();
            bad.levels[level].final_evals[0] += TvFp2::ONE;
            assert_eq!(
                verify(&bad).err(),
                Some(Reject::Level(
                    level,
                    tv::SumcheckError::FinalMismatch {
                        round: bad.levels[level].rounds.len() as u32
                    }
                ))
            );
        }
        let mut bad = messages.clone();
        bad.splits[0] += TvFp2::ONE;
        assert_eq!(verify(&bad).err(), Some(SPLIT));
        let mut bad = messages.clone();
        bad.splits.pop();
        assert_eq!(verify(&bad).err(), Some(SPLIT));
        let mut bad = messages.clone();
        *bad.splits.last_mut().unwrap() += TvFp2::ONE;
        assert_ne!(
            verify(&bad).unwrap().next,
            accepted.next,
            "atom tail is transcript-bound; graph checks remain outstanding"
        );
    }
}

#[test]
fn a_malformed_forest_is_refused_before_the_transcript_moves() {
    let leaf = synthetic();
    let mut t = transcript();
    let before = transcript().challenge("test/unchanged");
    let fields = [Fp::ONE; 3];
    let error = tessera_prover::leaf::prepare_leaf(
        &mut t,
        &leaf.layout,
        &leaf.tables,
        &leaf.witness(),
        &[Forest::new(&fields, 1)],
    )
    .unwrap_err();
    assert_eq!(
        error,
        LeafError::LogUp(tessera_logup::LogUpError::Sumcheck(
            tessera_sumcheck::SumcheckError::LeafCount { forest: 0, len: 3 }
        ))
    );
    assert_eq!(t.challenge("test/unchanged"), before);
}

#[test]
fn forest_only_and_empty_leaf_stages_bind_their_split_message() {
    let layout = LeafLayout::<Term>::new(&model().layers, vec![]).unwrap();
    let fields: Vec<_> = [2, 3, 4, 5].into_iter().map(Fp::from_i64).collect();
    for with_forest in [false, true] {
        let forests = if with_forest {
            vec![Forest::new(&fields, 1)]
        } else {
            vec![]
        };
        let forest_data = if with_forest {
            vec![(1, vec![2, 3, 4, 5])]
        } else {
            vec![]
        };
        for mode in [Mode::Live, Mode::NonInteractive] {
            let mut t = transcript_mode(mode);
            let proven = tessera_prover::leaf::prepare_leaf(&mut t, &layout, &[], &[], &forests)
                .unwrap()
                .finish(&[])
                .unwrap();
            assert!(proven.splits().is_empty());
            assert_eq!(proven.logup().forest_claims().len(), forests.len());
            let messages = Messages::of(proven.logup(), proven.splits());
            let accepted = verify_combined(
                &[],
                &BTreeMap::new(),
                &messages,
                vec![],
                &forest_data,
                0,
                mode,
            )
            .unwrap();
            assert_eq!(accepted.point, tv_fp2s(proven.point()));
            assert_eq!(accepted.next, tv_fp2(t.challenge("stage/1/layer/0")));
        }
    }
}
