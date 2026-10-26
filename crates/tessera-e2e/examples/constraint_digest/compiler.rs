// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compiler-side diagnostic extraction. It reconstructs producer/role metadata from the
//! compiler's dataflow graph and expands the compiler's factored schedule representation.

use serde_json::{Value, json};
use std::collections::BTreeMap;
use tessera_compiler::schedule as s;
use tessera_compiler::{Lowering, PublicModel, StageSchedule, analyze, derive_schedule, lower};
use tessera_field::{Field, Goldilocks as F, PrimeField};
use tessera_ir::{
    Broadcast, ClaimList, CommitmentClass, NodeId, NodeKind, Op, QuantPoint, Site, TensorId,
    WeightType,
};
use tessera_tables::TableKind;

fn weight(w: WeightType) -> &'static str {
    match w {
        WeightType::Embed => "Embed",
        WeightType::Q => "Q",
        WeightType::K => "K",
        WeightType::V => "V",
        WeightType::O => "O",
        WeightType::Gate => "Gate",
        WeightType::Up => "Up",
        WeightType::Down => "Down",
        WeightType::LmHead => "LmHead",
    }
}
fn broadcast(b: Broadcast) -> &'static str {
    match b {
        Broadcast::Trailing => "trailing",
        Broadcast::Leading => "leading",
    }
}
fn kind(k: NodeKind) -> Value {
    match k {
        NodeKind::Virtual => json!(["virtual"]),
        NodeKind::Public => json!(["public"]),
        NodeKind::Committed {
            commitment: CommitmentClass::Advice,
        } => json!(["advice"]),
        NodeKind::Committed {
            commitment: CommitmentClass::Weight,
        } => json!(["weight"]),
        NodeKind::LookupOutput { table } => json!(["lookup", table.0]),
    }
}
fn op(o: &Op) -> Value {
    let (tag, params) = match o {
        Op::MatMulWeight { weight: w } => ("MatMulWeight", json!([weight(w.ty), w.layer])),
        Op::OneHot { vocab } => ("OneHot", json!([vocab])),
        Op::ConstMul { c } => ("ConstMul", json!([c.0])),
        Op::Add => ("Add", json!([])),
        Op::Reshape => ("Reshape", json!([])),
        Op::SumSquares { axis } => ("SumSquares", json!([axis])),
        Op::RsqrtRow {
            kappa,
            eps_int,
            dim,
        } => ("RsqrtRow", json!([kappa, eps_int, dim])),
        Op::Mul { broadcast: b } => ("Mul", json!([broadcast(*b)])),
        Op::Requant { shift, lo, hi } => (
            "Requant",
            json!([shift.0.to_string(), lo.to_string(), hi.to_string()]),
        ),
        Op::Rotate { rope } => ("Rotate", json!([rope.0])),
        Op::Transpose { perm } => ("Transpose", json!(perm)),
        Op::Expand { axis, len } => ("Expand", json!([axis, len])),
        Op::MatMulAct { transpose_rhs } => ("MatMulAct", json!([transpose_rhs])),
        Op::SoftmaxRow { exp_table, sigma } => ("SoftmaxRow", json!([exp_table.0, sigma])),
        Op::TableMap { table } => ("TableMap", json!([table.0])),
        other => panic!("unsupported Llama operation {}", other.name()),
    };
    let mut object = serde_json::Map::new();
    object.insert(tag.to_owned(), params);
    Value::Object(object)
}
fn table(t: &TableKind) -> Value {
    let (name, fields) = match t {
        TableKind::Rope {
            theta,
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max,
            bits,
            max_context,
            d_head,
        } => (
            "Rope",
            json!([
                theta,
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max,
                bits,
                max_context,
                d_head
            ]),
        ),
        TableKind::Exp {
            key_bits,
            sigma,
            scale,
        } => (
            "Exp",
            json!([
                key_bits,
                sigma,
                scale.num().to_string(),
                scale.den().to_string()
            ]),
        ),
        TableKind::Silu {
            b,
            r_by_class,
            in_exp,
            out_exp,
        } => ("Silu", json!([b, r_by_class, in_exp, out_exp])),
        TableKind::Range { bits } => ("Range", json!([bits])),
        TableKind::Requant {
            b,
            r_by_class,
            lo,
            hi,
        } => ("Requant", json!([b, r_by_class, lo, hi])),
        _ => panic!("unsupported table kind in Llama constraints"),
    };
    let mut object = serde_json::Map::new();
    object.insert(name.to_owned(), fields);
    Value::Object(object)
}
fn point(p: &[s::Coord]) -> Vec<String> {
    p.iter()
        .map(|c| match c {
            s::Coord::Leaf { site, index } => format!("leaf:{site}:{index}"),
            s::Coord::Layer(i) => format!("layer:{i}"),
            s::Coord::Zero(i) => format!("zero:1:{i}"),
            s::Coord::Round { stage, round } => format!("round:{stage}:{round}"),
            s::Coord::Half => "half".to_owned(),
        })
        .collect()
}
fn shape_vars(shape: &[u32]) -> usize {
    shape
        .iter()
        .map(|d| (32 - (d - 1).leading_zeros()) as usize)
        .sum()
}

struct Context<'a> {
    low: &'a Lowering,
    q: &'a StageSchedule,
    producers: Vec<Option<NodeId>>,
}
impl Context<'_> {
    fn node(&self, n: s::NodeRef, layer: u32) -> NodeId {
        match n {
            s::NodeRef::Node(n) => n,
            s::NodeRef::Position(p) => self.q.layers.node(p, layer),
        }
    }
    fn physical(&self, k: s::Kind) -> TensorId {
        self.q.layers.tensor(&self.low.graph, k, 0).unwrap()
    }
    fn tensor(&self, k: s::Kind) -> Value {
        match k {
            s::Kind::LayerInput => json!(["layer-input"]),
            s::Kind::Tensor(t) => json!(["single", t.0]),
            _ => json!(["stacked", self.physical(k).0]),
        }
    }
    fn layers(&self, l: &s::Layers) -> Value {
        match l {
            s::Layers::None => json!(["none"]),
            s::Layers::One(i) => json!(["one", i]),
            s::Layers::Each => json!([
                "point",
                point(
                    &(0..self.q.layers.vars)
                        .map(s::Coord::Layer)
                        .collect::<Vec<_>>()
                )
            ]),
            s::Layers::Eq(p) => json!(["point", point(p)]),
        }
    }
    fn coefficient(&self, terms: &[Vec<s::Factor>]) -> Value {
        let count = self.q.layers.count as usize;
        let mut sums: BTreeMap<Vec<(Vec<String>, u64)>, Vec<F>> = BTreeMap::new();
        for term in terms {
            let mut factors = Vec::new();
            let mut constants = vec![F::ONE; count];
            for factor in term {
                match factor {
                    s::Factor::Minus => {
                        for c in &mut constants {
                            *c = -*c;
                        }
                    }
                    s::Factor::Pow2(bits) => {
                        let mut factor = F::ONE;
                        for _ in 0..*bits {
                            factor += factor;
                        }
                        for c in &mut constants {
                            *c *= factor;
                        }
                    }
                    s::Factor::Constant(n) => {
                        for (i, c) in constants.iter_mut().enumerate() {
                            let node = self.node(*n, i as u32);
                            let Op::ConstMul { c: idx } = self.low.graph.nodes[node.0 as usize].op
                            else {
                                panic!("a constant factor names ConstMul")
                            };
                            let values = &self.low.graph.constants[idx.0 as usize];
                            assert_eq!(values.len(), 1);
                            *c *= F::from_i64(values[0]);
                        }
                    }
                    s::Factor::Lt { coords, len } => {
                        if u64::from(*len) != 1u64 << coords.len() {
                            factors.push((point(coords), u64::from(*len)));
                        }
                    }
                }
            }
            factors.sort();
            let sum = sums.entry(factors).or_insert_with(|| vec![F::ZERO; count]);
            for (s, c) in sum.iter_mut().zip(constants) {
                *s += c;
            }
        }
        json!(sums.into_iter().filter(|(_,v)|v.iter().any(|c|*c!=F::ZERO)).map(|(f,c)|json!({"constants":c.into_iter().map(|v|v.to_canonical_u64()).collect::<Vec<_>>(),"lt":f})).collect::<Vec<_>>())
    }
    fn claim(&self, c: &s::Claim) -> Value {
        let atoms:Vec<_>=c.atoms.iter().map(|a|{
            let class=match a.class {
                s::AtomClass::Virtual=>json!(["virtual",self.producers[self.physical(a.kind).0 as usize].unwrap().0]),
                s::AtomClass::Committed(CommitmentClass::Advice)=>json!(["advice"]),
                s::AtomClass::Committed(CommitmentClass::Weight)=>match self.q.weight_type(&self.low.graph,a.kind) {Some(w)=>json!(["weight",weight(w)]),None=>json!(["channel-params"])},
            };
            json!({"tensor":self.tensor(a.kind),"point":point(&a.point),"layers":self.layers(&a.layers),"coefficient":self.coefficient(&a.coefficient),"class":class})
        }).collect();
        let route = if c.split {
            json!([
                "split",
                c.parts
                    .iter()
                    .map(|p| match p {
                        s::Part::Instance { atom, instance } => json!(["virtual", atom, instance]),
                        s::Part::Pool { atoms, claim: _ } => json!(["group", atoms]),
                    })
                    .collect::<Vec<_>>()
            ])
        } else {
            assert_eq!(c.parts.len(), 1);
            match &c.parts[0] {
                s::Part::Instance { atom: _, instance } => json!(["instance", instance]),
                s::Part::Pool { atoms: _, claim: _ } => json!(["pool"]),
            }
        };
        json!({"tensor":self.tensor(c.kind),"vars":shape_vars(&self.low.graph.tensors[self.physical(c.kind).0 as usize].shape),"atoms":atoms,"route":route})
    }
    fn relation(&self, i: &s::Instance) -> (Value, usize) {
        let g = &self.low.graph;
        let node = &g.nodes[self.node(i.node, 0).0 as usize];
        let out = &g.tensors[node.outputs[0].0 as usize].shape;
        let lvars = if matches!(i.layers, s::Layers::Eq(_) | s::Layers::Each) {
            self.q.layers.vars as usize
        } else {
            0
        };
        match i.relation {
            s::Relation::Native {
                tag,
                degree,
                evaluations,
                zero,
            } => (json!({"Native":[tag,evaluations,zero]}), degree),
            s::Relation::MatMulWeight => {
                let w = &g.tensors[node.inputs[1].0 as usize].shape;
                (
                    json!({"MatMulWeight":[shape_vars(&w[1..]),shape_vars(&w[..1])]}),
                    2 + usize::from(lvars > 0),
                )
            }
            s::Relation::MatMulAct { transposed } => {
                let a = &g.tensors[node.inputs[0].0 as usize].shape;
                let lead = shape_vars(&out[..out.len() - 2]);
                (
                    json!({"MatMulAct":[json!(transposed),json!(shape_vars(&a[a.len()-1..])),json!(shape_vars(&out[out.len()-2..out.len()-1])),json!(shape_vars(&out[out.len()-1..])),json!(lead)]}),
                    2 + usize::from(lead + lvars > 0),
                )
            }
            s::Relation::Mul(b) => {
                let bvars = shape_vars(&g.tensors[node.inputs[1].0 as usize].shape);
                let start = match b {
                    Broadcast::Leading => 0,
                    Broadcast::Trailing => shape_vars(out) - bvars,
                };
                (
                    json!({"Elementwise":[json!(broadcast(b)),json!(start),json!(bvars)]}),
                    2 + usize::from(i.vars > 0),
                )
            }
            s::Relation::SumSquares => {
                let a = &g.tensors[node.inputs[0].0 as usize].shape;
                (
                    json!({"SumSquares":[shape_vars(&a[a.len()-1..])]}),
                    2 + usize::from(i.vars > 0),
                )
            }
            s::Relation::Rotate => {
                let a = &g.tensors[node.inputs[0].0 as usize].shape;
                (
                    json!({"Rotate":[json!(a[0]),json!(shape_vars(&a[1..a.len()-1]))]}),
                    2,
                )
            }
            s::Relation::Equality => {
                let Op::ConstMul { c } = node.op else {
                    panic!("Llama's equality binds the first checkpoint")
                };
                (
                    json!({"Equality":[F::from_i64(g.constants[c.0 as usize][0]).to_canonical_u64()]}),
                    1 + usize::from(shape_vars(out) + lvars > 0),
                )
            }
            s::Relation::Booleanity => (json!({"Booleanity":[]}), 3),
            s::Relation::OneHotVocabulary => {
                (json!({"Vocabulary":[out[1],shape_vars(&out[1..])]}), 3)
            }
            s::Relation::OneHotCount => (json!({"OneHotCount":[shape_vars(&out[1..]),out[0]]}), 2),
            s::Relation::OneHotIndex => (json!({"OneHotIndex":[shape_vars(&out[1..])]}), 2),
            s::Relation::ScaleChannels => panic!("no per-channel ConstMul in the Llama lowering"),
            s::Relation::InputPadding
            | s::Relation::BitPadding
            | s::Relation::Recomposition
            | s::Relation::AssertEqual => panic!("no bit operation in the Llama lowering"),
        }
    }
    fn schedule(&self) -> Value {
        let leaf_stage = &self.q.stages[0];
        let mut leaf = Vec::new();
        let mut leaf_splits = 0;
        for (site, c) in self.q.sites.iter().zip(&leaf_stage.claims) {
            let (node, input) = match site {
                s::Site::Gadget { node, input } => (self.node(*node, 0), Some(*input)),
                s::Site::Logits => (self.producers[self.low.logits.0 as usize].unwrap(), None),
            };
            let each = matches!(c.layers, s::Layers::Each);
            let layers = if each { self.q.layers.count } else { 1 };
            leaf.push(json!({"site":node.0,"input":input,"stacked":each,"claim":self.claim(c)}));
            if c.split {
                leaf_splits += c.parts.len() * layers as usize;
            }
        }
        let stages:Vec<_>=self.q.stages.iter().skip(1).map(|stage|{
            let mut degree=1;
            let instances:Vec<_>=stage.instances.iter().map(|i|{
                let (relation,d)=self.relation(i);degree=degree.max(d);
                let zero=matches!(i.source,s::Source::ZeroCheck) || matches!(i.relation,s::Relation::Native{zero:true,..});
                let mut z=if zero {point(&i.point)}else{Vec::new()};if zero && let s::Layers::Eq(r)=&i.layers {z.extend(point(r));}
                json!({"relation":relation,"node":self.node(i.node,0).0,"tensor":self.tensor(i.kind),"stacked":matches!(i.node,s::NodeRef::Position(_)),"vars":i.vars,"degree":d,"point":if zero {Vec::new()}else{point(&i.point)},"layers":self.layers(&i.layers),"zero":z,"evals":stage.claims[i.claims.clone()].iter().map(|c|self.claim(c)).collect::<Vec<_>>()})
            }).collect();
            json!({"index":stage.index,"degree":degree,"rounds":stage.rounds,"layer_vars":if stage.index==1 {self.q.layers.vars}else{0},"zero_vars":stage.zero_vars,"splits":stage.claims.iter().filter(|c|c.split).map(|c|c.parts.len()).sum::<usize>(),"instances":instances})
        }).collect();
        json!({"layers":self.q.layers.count,"layer_vars":self.q.layers.vars,"first_checkpoint":self.low.checkpoints[0].0,"last_output":self.low.checkpoints[1].0,"leaf_splits":leaf_splits,"leaf":leaf,"stages":stages})
    }
}

/// Compute the compiler's diagnostic tree without reading any model tensor values.
pub fn snapshot(c: &ClaimList, n: u32) -> Value {
    let low = lower(PublicModel::of_claims(c), n).expect("compiler lowering");
    let q = derive_schedule(&low).expect("compiler stage schedule");
    let ranges = analyze(&low.graph, &low.tables).expect("compiler range analysis");
    let selected = tessera_compiler::select(&low.graph, &ranges, &low.tables, &c.gadget_policy)
        .expect("compiler gadget choices");
    let g = &low.graph;
    let mut producers = vec![None; g.tensors.len()];
    for node in &g.nodes {
        for t in &node.outputs {
            assert!(producers[t.0 as usize].replace(node.id).is_none());
        }
    }
    let inputs:Vec<_>=g.tensors.iter().filter(|t|producers[t.id.0 as usize].is_none()).map(|t|{
        let role=if t.id==low.tokens {json!(["tokens"])}else if t.id==low.positions {json!(["positions"])}else {
            let (tag,w)=g.weights.iter().map(|(w,t)|("weight",w,t)).chain(low.classes.iter().map(|(w,t)|("classes",w,t))).chain(low.multipliers.iter().map(|(w,t)|("multipliers",w,t))).find(|(_,_,id)|**id==t.id).map(|(tag,w,_)|(tag,w)).expect("every public model input has a role");json!([tag,[weight(w.ty),w.layer]])
        };
        json!({"tensor":t.id.0,"role":role,"type":[t.ty.bits,u8::from(t.ty.signed)],"range":[t.range.lo.to_string(),t.range.hi.to_string()]})
    }).collect();
    let residual = |site| {
        let output = low
            .sites
            .iter()
            .find(|(p, _)| *p == QuantPoint::Layer { layer: 0, site })
            .unwrap()
            .1;
        let req = &g.nodes[producers[output.0 as usize].unwrap().0 as usize];
        let key = &g.nodes[producers[req.inputs[0].0 as usize].unwrap().0 as usize];
        match key.op {
            Op::Add => "fused",
            Op::ConstMul { c } => {
                assert_eq!(g.constants[c.0 as usize], [2]);
                "two-step"
            }
            _ => panic!("a residual uses the RFC-0038 route"),
        }
    };
    let graph = json!({"length":n,"tensors":g.tensors.iter().map(|t|json!({"shape":t.shape,"kind":kind(t.kind),"producer":producers[t.id.0 as usize].map(|p|p.0)})).collect::<Vec<_>>(),"inputs":inputs,"nodes":g.nodes.iter().map(|node|json!({"op":op(&node.op),"inputs":node.inputs.iter().map(|t|t.0).collect::<Vec<_>>(),"outputs":node.outputs.iter().map(|t|t.0).collect::<Vec<_>>()})).collect::<Vec<_>>(),"constants":g.constants.iter().map(|v|{assert_eq!(v.len(),1);v[0].to_string()}).collect::<Vec<_>>(),"shifts":g.shift_classes,"weights":g.weights.iter().map(|(w,t)|json!([[weight(w.ty),w.layer],t.0])).collect::<Vec<_>>(),"tables":low.tables.iter().map(|t|json!({"id":t.id.0,"kind":table(&t.kind),"log2_rows":t.log2_rows})).collect::<Vec<_>>(),"checkpoints":low.checkpoints.iter().map(|t|t.0).collect::<Vec<_>>(),"routes":[residual(Site::AttnResidual),residual(Site::MlpResidual)]});
    json!({"version":1,"graph":graph,"selection":selection_snapshot(&selected),"ranges":ranges.all().iter().map(|r|r.bounded().map(|v|[v.lo.to_string(),v.hi.to_string()])).collect::<Vec<_>>(),"policy":[u64::from(c.gadget_policy.max_log2_rows),u64::from(c.gadget_policy.query_block),u64::from(c.gadget_policy.limb_bits)],"schedule":Context{low:&low,q:&q,producers}.schedule()})
}

/// Recursive canonical framing, independently written from the CI format description.
pub fn frame(value: &Value) -> Vec<u8> {
    fn walk(value: &Value, path: &str, bytes: &mut Vec<u8>) {
        let (tag, data) = match value {
            Value::Null => ("null", String::new()),
            Value::String(s) => ("string", s.to_owned()),
            Value::Bool(b) => ("bool", if *b { "1" } else { "0" }.to_owned()),
            Value::Number(n) => {
                assert!(n.is_u64() || n.is_i64());
                ("integer", n.to_string())
            }
            Value::Array(a) => ("array", a.len().to_string()),
            Value::Object(m) => ("object", m.len().to_string()),
        };
        for part in [path, tag, &data] {
            bytes.extend_from_slice(&(part.len() as u64).to_le_bytes());
            bytes.extend_from_slice(part.as_bytes());
        }
        match value {
            Value::Array(a) => {
                for (i, child) in a.iter().enumerate() {
                    walk(child, &format!("{path}/{i}"), bytes);
                }
            }
            Value::Object(m) => {
                let mut keys: Vec<_> = m.keys().collect();
                keys.sort_unstable();
                for key in keys {
                    let mut escaped = String::new();
                    for c in key.chars() {
                        match c {
                            '/' => escaped.push_str("~1"),
                            '~' => escaped.push_str("~0"),
                            other => escaped.push(other),
                        }
                    }
                    walk(&m[key], &format!("{path}/{escaped}"), bytes);
                }
            }
            _ => {}
        }
    }
    let mut bytes = Vec::new();
    walk(value, "", &mut bytes);
    bytes
}
/// BLAKE3 over the compiler's canonical diagnostic bytes, using its native hash dependency.
pub fn digest(tree: &Value) -> [u8; 32] {
    blake3::derive_key("tessera/ci/constraint-system-digest/1", &frame(tree))
}

fn selection_snapshot(selection: &tessera_compiler::Selection) -> Value {
    use tessera_compiler::Gadget;
    let choices: Vec<_> = selection
        .gadgets
        .iter()
        .map(|(node, g)| {
            let pattern = match *g {
                Gadget::Requant1 { .. } => json!(["Requant1"]),
                Gadget::RequantSplit { split, .. } => json!(["RequantSplit", split]),
                Gadget::RequantWide { split, middle, .. } => json!(["RequantWide", split, middle]),
                Gadget::Lookup { .. } => json!(["Lookup"]),
                Gadget::SoftmaxRow { .. } => json!(["SoftmaxRow"]),
                Gadget::RsqrtRow => json!(["RsqrtRow"]),
                Gadget::OneHot => json!(["OneHot"]),
                Gadget::Argmax { .. } | Gadget::InverseCdf => {
                    panic!("scoring gadgets are outside the Llama lowering")
                }
                Gadget::BitDecompose { .. } | Gadget::AssertEqual => {
                    panic!("the anchors' bit gadgets are outside the Llama lowering")
                }
            };
            let tables: Vec<_> = selection
                .reads(*node)
                .expect("every gadget has its ordered requests")
                .iter()
                .map(|t| t.0)
                .collect();
            json!({"node": node.0, "pattern": pattern, "tables": tables})
        })
        .collect();
    json!({"choices":choices,"tables":selection.tables.iter().map(|t|json!({"id":t.id.0,"kind":table(&t.kind),"log2_rows":t.log2_rows})).collect::<Vec<_>>()})
}
