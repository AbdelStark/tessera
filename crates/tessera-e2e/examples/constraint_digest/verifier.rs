// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent diagnostic collector, written from the RFCs and tv-* public representation.
//! No compiler/prover types or helpers are imported here (DEP-2, SP-6).

use serde_json::{Value, json};
use std::collections::BTreeMap;
use tv_core::claims::{ClaimList, Site, WeightType};
use tv_core::lowering::{self, Broadcast, Graph, Kind, Op, Role, Route, WeightRef};
use tv_core::{schedule as s, selection, validation};
use tv_field::Fp;
use tv_tables::TableKind;

fn weight(t: WeightType) -> &'static str {
    match t {
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
fn wref(w: WeightRef) -> Value {
    json!([weight(w.ty), w.layer])
}
fn kind(k: Kind) -> Value {
    match k {
        Kind::Advice => json!(["advice"]),
        Kind::Weight => json!(["weight"]),
        Kind::Public => json!(["public"]),
        Kind::Virtual => json!(["virtual"]),
        Kind::Lookup { table } => json!(["lookup", table.0]),
    }
}
fn broadcast(b: Broadcast) -> &'static str {
    match b {
        Broadcast::Leading => "leading",
        Broadcast::Trailing => "trailing",
    }
}
fn table(t: &TableKind) -> Value {
    match t {
        TableKind::Rope {
            theta,
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max,
            bits,
            max_context,
            d_head,
        } => {
            json!({"Rope":[theta,factor,low_freq_factor,high_freq_factor,original_max,bits,max_context,d_head]})
        }
        TableKind::Exp {
            key_bits,
            sigma,
            scale,
        } => json!({"Exp":[key_bits,sigma,scale.num().to_string(),scale.den().to_string()]}),
        TableKind::Silu {
            b,
            r_by_class,
            in_exp,
            out_exp,
        } => json!({"Silu":[b,r_by_class,in_exp,out_exp]}),
        TableKind::Range { bits } => json!({"Range":[bits]}),
        TableKind::Requant {
            b,
            r_by_class,
            lo,
            hi,
        } => json!({"Requant":[b,r_by_class,lo,hi]}),
        _ => panic!("unsupported table kind in Llama constraints"),
    }
}
fn op(o: &Op) -> Value {
    match o {
        Op::OneHot { vocab } => json!({"OneHot":[vocab]}),
        Op::MatMulWeight { weight } => json!({"MatMulWeight":wref(*weight)}),
        Op::ConstMul { c } => json!({"ConstMul":[c]}),
        Op::Add => json!({"Add":[]}),
        Op::Reshape => json!({"Reshape":[]}),
        Op::SumSquares { axis } => json!({"SumSquares":[axis]}),
        Op::RsqrtRow {
            kappa,
            eps_int,
            dim,
        } => json!({"RsqrtRow":[u64::from(*kappa),*eps_int,u64::from(*dim)]}),
        Op::Mul { broadcast: b } => json!({"Mul":[broadcast(*b)]}),
        Op::Requant { shift, lo, hi } => {
            json!({"Requant":[shift.to_string(),lo.to_string(),hi.to_string()]})
        }
        Op::Rotate { rope } => json!({"Rotate":[rope.0]}),
        Op::Transpose { perm } => json!({"Transpose":perm}),
        Op::Expand { axis, len } => json!({"Expand":[u64::from(*axis),u64::from(*len)]}),
        Op::MatMulAct { transpose_rhs } => json!({"MatMulAct":[transpose_rhs]}),
        Op::SoftmaxRow { exp_table, sigma } => {
            json!({"SoftmaxRow":[exp_table.0,u32::from(*sigma)]})
        }
        Op::TableMap { table } => json!({"TableMap":[table.0]}),
    }
}

/// Every lowered graph component, in its normative creation order.
pub fn graph(g: &Graph) -> Value {
    let tensors:Vec<_>=g.tensors().iter().map(|t|json!({"shape":t.shape.dims(),"kind":kind(t.kind),"producer":t.producer.map(|p|p.0)})).collect();
    let inputs:Vec<_>=g.inputs().iter().map(|i|{
        let role=match i.role {Role::Tokens=>json!(["tokens"]),Role::Positions=>json!(["positions"]),Role::Weight(w)=>json!(["weight",wref(w)]),Role::Classes(w)=>json!(["classes",wref(w)]),Role::Multipliers(w)=>json!(["multipliers",wref(w)])};
        json!({"tensor":i.tensor.0,"role":role,"type":[i.ty.bits,u8::from(i.ty.signed)],"range":[i.lo.to_string(),i.hi.to_string()]})
    }).collect();
    let nodes:Vec<_>=g.nodes().iter().map(|n|json!({"op":op(n.op()),"inputs":n.inputs().iter().map(|t|t.0).collect::<Vec<_>>(),"outputs":n.outputs().iter().map(|t|t.0).collect::<Vec<_>>()})).collect();
    let tables: Vec<_> = g
        .tables()
        .iter()
        .map(|t| json!({"id":t.id.0,"kind":table(&t.kind),"log2_rows":t.log2_rows}))
        .collect();
    let route = |site| match g.route(site).unwrap() {
        Route::FusedKey => "fused",
        Route::TwoSteps => "two-step",
    };
    json!({"length":g.length(),"tensors":tensors,"inputs":inputs,"nodes":nodes,"constants":g.constants().iter().map(i64::to_string).collect::<Vec<_>>(),"shifts":g.shift_classes().iter().map(|v|v.as_slice()).collect::<Vec<_>>(),"weights":g.weights().iter().map(|(w,t)|json!([wref(*w),t.0])).collect::<Vec<_>>(),"tables":tables,"checkpoints":g.checkpoints().iter().map(|t|t.0).collect::<Vec<_>>(),"routes":[route(Site::AttnResidual),route(Site::MlpResidual)]})
}

fn point(p: &s::Point) -> Vec<String> {
    p.runs()
        .iter()
        .flat_map(|r| {
            let prefix = match r.source {
                s::Source::Leaf(i) => format!("leaf:{i}"),
                s::Source::Layer => "layer".to_owned(),
                s::Source::Rounds(i) => format!("round:{i}"),
                s::Source::Zero(i) => format!("zero:{i}"),
            };
            (r.start..r.start + r.len).map(move |i| format!("{prefix}:{i}"))
        })
        .collect()
}
fn layers(l: &s::Layers) -> Value {
    match l {
        s::Layers::None => json!(["none"]),
        s::Layers::One(i) => json!(["one", i]),
        s::Layers::Point(p) => json!(["point", point(p)]),
    }
}
fn tensor(t: s::TensorRef) -> Value {
    match t {
        s::TensorRef::Single(t) => json!(["single", t.0]),
        s::TensorRef::Stacked(t) => json!(["stacked", t.0]),
        s::TensorRef::LayerInput => json!(["layer-input"]),
    }
}
fn class(c: s::Class) -> Value {
    match c {
        s::Class::Virtual { node } => json!(["virtual", node.0]),
        s::Class::Committed(c) => match c {
            s::Commitment::Advice => json!(["advice"]),
            s::Commitment::Weight(w) => json!(["weight", weight(w)]),
            s::Commitment::ChannelParams => json!(["channel-params"]),
        },
    }
}
fn coefficient(c: &s::Coefficient, l: usize) -> Value {
    let mut polynomials: BTreeMap<Vec<(Vec<String>, u64)>, Vec<Fp>> = BTreeMap::new();
    for t in c.terms() {
        let mut factors: Vec<_> = t
            .lts
            .iter()
            .map(|(p, b)| (point(p), *b))
            .filter(|(p, b)| *b != (1u64 << p.len()))
            .collect();
        factors.sort();
        let sum = polynomials
            .entry(factors)
            .or_insert_with(|| vec![Fp::ZERO; l]);
        assert!(t.constant.len() == 1 || t.constant.len() == l);
        for (i, s) in sum.iter_mut().enumerate() {
            *s += t.constant[if t.constant.len() == 1 { 0 } else { i }];
        }
    }
    json!(polynomials.into_iter().filter(|(_,c)|c.iter().any(|x|*x!=Fp::ZERO)).map(|(f,c)|json!({"constants":c.into_iter().map(Fp::to_canonical).collect::<Vec<_>>(),"lt":f})).collect::<Vec<_>>())
}
fn claim(c: &s::Claim, l: usize) -> Value {
    let route = match &c.route {
        s::Route::Instance(i) => json!(["instance", i]),
        s::Route::Pool => json!(["pool"]),
        s::Route::Split(parts) => json!([
            "split",
            parts
                .iter()
                .map(|p| match p {
                    s::Part::Virtual { atom, instance } => json!(["virtual", atom, instance]),
                    s::Part::Group { atoms } => json!(["group", atoms]),
                })
                .collect::<Vec<_>>()
        ]),
    };
    json!({"tensor":tensor(c.tensor),"vars":c.vars,"atoms":c.atoms.iter().map(|a|json!({"tensor":tensor(a.tensor),"point":point(&a.point),"layers":layers(&a.layers),"coefficient":coefficient(&a.coefficient,l),"class":class(a.class)})).collect::<Vec<_>>(),"route":route})
}
fn relation(r: &s::Relation) -> Value {
    match r {
        s::Relation::MatMulWeight { in_vars, out_vars } => {
            json!({"MatMulWeight":[in_vars,out_vars]})
        }
        s::Relation::MatMulAct {
            transpose_rhs,
            k_vars,
            m_vars,
            n_vars,
            lead_vars,
        } => {
            json!({"MatMulAct":[json!(transpose_rhs),json!(k_vars),json!(m_vars),json!(n_vars),json!(lead_vars)]})
        }
        s::Relation::Elementwise {
            broadcast: b,
            b_from,
            b_vars,
        } => json!({"Elementwise":[json!(broadcast(*b)),json!(b_from),json!(b_vars)]}),
        s::Relation::SumSquares { col_vars } => json!({"SumSquares":[col_vars]}),
        s::Relation::Rotate {
            positions,
            middle_vars,
        } => json!({"Rotate":[json!(positions),json!(middle_vars)]}),
        s::Relation::Equality { constant } => json!({"Equality":[constant.to_canonical()]}),
    }
}
fn schedule(q: &s::Schedule) -> Value {
    let l = q.layers() as usize;
    json!({"layers":q.layers(),"layer_vars":q.layer_vars(),"first_checkpoint":q.first_checkpoint().0,"last_output":q.last_output().0,"leaf_splits":q.leaf_splits(),"leaf":q.leaf().iter().map(|c|json!({"site":c.site.0,"input":c.input,"stacked":c.stacked,"claim":claim(&c.claim,l)})).collect::<Vec<_>>(),"stages":q.stages().iter().map(|s|json!({"index":s.index,"degree":s.degree,"rounds":s.rounds,"layer_vars":s.layer_vars,"zero_vars":s.zero_vars,"splits":s.splits,"instances":s.instances.iter().map(|i|json!({"relation":relation(&i.relation),"node":i.node.0,"tensor":tensor(i.tensor),"stacked":i.stacked,"vars":i.vars,"degree":i.degree,"point":point(&i.point),"layers":layers(&i.layers),"zero":point(&i.zero),"evals":i.evals.iter().map(|c|claim(c,l)).collect::<Vec<_>>()})).collect::<Vec<_>>() })).collect::<Vec<_>>()})
}

/// Lower, analyze and derive independently from the claim list's public metadata.
pub fn snapshot(c: &ClaimList, n: u32) -> Value {
    let public = validation::check_public(c).expect("valid public metadata");
    let g = lowering::lower(&public, n).expect("independent lowering");
    let catalog =
        selection::derive(&g, c.gadget_policy).expect("independent ranges and gadget choices");
    let r = &catalog.ranges;
    let q = s::derive(&g).expect("independent stage derivation");
    json!({"version":1,"graph":graph(&g),"selection":selection_snapshot(&catalog),"ranges":r.iter().map(|v|v.bounds().map(|(a,b)|[a.to_string(),b.to_string()])).collect::<Vec<_>>(),"policy":[u64::from(c.gadget_policy.max_log2_rows),u64::from(c.gadget_policy.query_block),u64::from(c.gadget_policy.limb_bits)],"schedule":schedule(&q)})
}

/// Iterative canonical encoder and independent reference BLAKE3 for this CI-only tree.
pub fn frame(tree: &Value) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut pending = vec![(String::new(), tree)];
    while let Some((path, v)) = pending.pop() {
        let (tag, payload) = match v {
            Value::Null => ("null", String::new()),
            Value::Bool(b) => ("bool", u8::from(*b).to_string()),
            Value::Number(n) => {
                assert!(n.is_i64() || n.is_u64());
                ("integer", n.to_string())
            }
            Value::String(s) => ("string", s.clone()),
            Value::Array(a) => {
                for (i, v) in a.iter().enumerate().rev() {
                    pending.push((format!("{path}/{i}"), v));
                }
                ("array", a.len().to_string())
            }
            Value::Object(m) => {
                let mut entries: Vec<_> = m.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(b.0));
                for (key, v) in entries.into_iter().rev() {
                    let escaped = key.replace('~', "~0").replace('/', "~1");
                    pending.push((format!("{path}/{escaped}"), v));
                }
                ("object", m.len().to_string())
            }
        };
        for part in [&path, tag, &payload] {
            let size = part.len() as u64;
            for shift in 0..8 {
                bytes.push((size >> (shift * 8)) as u8);
            }
            bytes.extend_from_slice(part.as_bytes());
        }
    }
    bytes
}

/// Independently hash the canonical diagnostic records; no protocol hash domain is used.
pub fn digest(tree: &Value) -> [u8; 32] {
    tv_testkit::blake3::derive_key("tessera/ci/constraint-system-digest/1", &frame(tree))
}

fn selection_snapshot(c: &selection::Selection) -> Value {
    use selection::Pattern;
    let choices:Vec<_>=c.choices.iter().map(|g| {
        let pattern=match g.pattern {
            Pattern::Requant1=>json!(["Requant1"]),Pattern::RequantSplit{split}=>json!(["RequantSplit",split]),Pattern::RequantWide{split,middle}=>json!(["RequantWide",split,middle]),Pattern::Lookup=>json!(["Lookup"]),Pattern::SoftmaxRow=>json!(["SoftmaxRow"]),Pattern::RsqrtRow=>json!(["RsqrtRow"]),Pattern::OneHot=>json!(["OneHot"]),
        };
        json!({"node":g.node.0,"pattern":pattern,"tables":g.tables.iter().map(|t|t.0).collect::<Vec<_>>()})
    }).collect();
    json!({"choices":choices,"tables":c.tables.iter().map(|t|json!({"id":t.id.0,"kind":table(&t.kind),"log2_rows":t.log2_rows})).collect::<Vec<_>>()})
}
