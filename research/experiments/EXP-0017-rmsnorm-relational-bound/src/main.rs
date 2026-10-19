// SPDX-License-Identifier: Apache-2.0 OR MIT

//! E0 public compiler probe; no production range or verifier changes.
#![forbid(unsafe_code)]

mod fixture;

use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::Read;
use tessera_compiler::{Gadget, PublicModel, Range, Selection, analyze, lower, report, select};
use tessera_ir::{
    CommitmentClass, GadgetPolicy, GraphBuilder, IntRange, IntType, NodeKind, Op, Profile, TableId,
    TensorDecl,
};
use tessera_tables::{GeneratedTable, Table, TableKind, TableSpec};

fn signed_width(lo: i128, hi: i128) -> u32 {
    let bits = |v: i128| 128 - v.max(0).unsigned_abs().leading_zeros();
    1 + bits(hi).max(bits(!lo))
}

fn table_json(t: &TableSpec) -> Value {
    let kind = match &t.kind {
        TableKind::Range { bits } => json!({"tag":"Range","bits":bits}),
        TableKind::Requant {
            b,
            r_by_class,
            lo,
            hi,
        } => json!({"tag":"Requant","b":b,"shifts":r_by_class,"lo":lo,"hi":hi}),
        TableKind::Silu {
            b,
            r_by_class,
            in_exp,
            out_exp,
        } => json!({"tag":"Silu","b":b,"shifts":r_by_class,"in_exp":in_exp,"out_exp":out_exp}),
        TableKind::Exp {
            key_bits,
            sigma,
            scale,
        } => json!({"tag":"Exp","key_bits":key_bits,"sigma":sigma,"scale":format!("{scale:?}")}),
        TableKind::Rope {
            max_context,
            d_head,
            ..
        } => {
            json!({"tag":"Rope","max_context":max_context,"d_head":d_head,"definition":format!("{:?}",t.kind)})
        }
        _ => panic!("unexpected fixture table {:?}", t.kind),
    };
    json!({"id":t.id.0,"kind":kind,"log2_rows":t.log2_rows,"rows":1u64<<t.log2_rows,"digest":t.digest})
}

fn refs(g: Gadget) -> Vec<TableId> {
    match g {
        Gadget::Requant1 { table } | Gadget::Lookup { table } => vec![table],
        Gadget::RequantSplit { low, high, .. } => vec![low, high],
        Gadget::RequantWide {
            low,
            mid,
            top,
            sign,
            high,
            ..
        } => low.into_iter().chain([mid, top, sign, high]).collect(),
        Gadget::SoftmaxRow { exp } => vec![exp],
        _ => vec![],
    }
}

fn gadget_json(g: Gadget, tables: &[TableSpec]) -> Value {
    let (name, split, middle, commits, lookups) = match g {
        Gadget::Requant1 { .. } => ("Requant1", None, None, 1, 1),
        Gadget::RequantSplit { split, .. } => ("RequantSplit", Some(split), None, 2, 2),
        Gadget::RequantWide {
            split, middle, low, ..
        } => (
            "RequantWide",
            Some(split),
            Some(middle),
            3 + u32::from(low.is_some()),
            4 + u32::from(low.is_some()),
        ),
        Gadget::Lookup { .. } => ("Lookup", None, None, 1, 1),
        Gadget::SoftmaxRow { .. } => ("SoftmaxRow", None, None, 0, 0),
        Gadget::RsqrtRow => ("RsqrtRow", None, None, 20, 30),
        Gadget::OneHot => ("OneHot", None, None, 1, 0),
        _ => panic!("not a forward fixture"),
    };
    json!({"tag":name,"split":split,"middle":middle,"commits_per_element":commits,"lookups_per_element":lookups,
        "tables":refs(g).iter().map(|id|table_json(&tables[id.0 as usize])).collect::<Vec<_>>()})
}

fn mini(lo: i128, hi: i128, shifts: &[u8]) -> Selection {
    let mut g = GraphBuilder::new();
    let advice = NodeKind::Committed {
        commitment: CommitmentClass::Advice,
    };
    let x = g
        .input(TensorDecl {
            shape: vec![1],
            ty: IntType::I32,
            range: IntRange { lo, hi },
            kind: advice,
        })
        .expect("public interval");
    assert_eq!(shifts.len(), 1, "norms have one class");
    let shift = g.shift_classes(shifts.to_vec()).expect("shift");
    g.node(
        Op::Requant {
            shift,
            lo: -128,
            hi: 127,
        },
        &[x],
        vec![TensorDecl {
            shape: vec![1],
            ty: IntType::I32,
            range: IntRange { lo: 0, hi: 0 },
            kind: advice,
        }],
    )
    .expect("requant");
    let graph = g.finish();
    let ranges = analyze(&graph, &[]).expect("mini ranges");
    select(&graph, &ranges, &[], &GadgetPolicy::default()).expect("mini selection")
}

fn remap(g: Gadget, source: &[TableSpec], out: &mut Vec<TableSpec>) -> Gadget {
    let mut id = |old: TableId| {
        let kind = &source[old.0 as usize].kind;
        if let Some(t) = out.iter().find(|t| &t.kind == kind) {
            return t.id;
        }
        let next = TableId(out.len() as u32);
        out.push(
            GeneratedTable::new(next, kind.clone())
                .expect("generated table")
                .spec()
                .clone(),
        );
        next
    };
    match g {
        Gadget::Requant1 { table } => Gadget::Requant1 { table: id(table) },
        Gadget::RequantSplit { split, low, high } => Gadget::RequantSplit {
            split,
            low: id(low),
            high: id(high),
        },
        Gadget::RequantWide {
            split,
            middle,
            low,
            mid,
            top,
            sign,
            high,
        } => Gadget::RequantWide {
            split,
            middle,
            low: low.map(&mut id),
            mid: id(mid),
            top: id(top),
            sign: id(sign),
            high: id(high),
        },
        Gadget::Lookup { table } => Gadget::Lookup { table: id(table) },
        Gadget::SoftmaxRow { exp } => Gadget::SoftmaxRow { exp: id(exp) },
        other => other,
    }
}

fn extrema(n: u128, e: u64, lo: i128, hi: i128, step: u32) -> Value {
    let max_for = |top: u32| {
        let mut best = (0u128, 0u32, (n / u128::from(e)).isqrt());
        for x in (0..=top).step_by(step as usize) {
            let root = (n / (u128::from(x) * u128::from(x) + u128::from(e))).isqrt();
            let z = u128::from(x) * root;
            if z > best.0 {
                best = (z, x, root);
            }
        }
        json!({"magnitude":best.0 as u64,"x":best.1,"rho":best.2 as u64,"last_x":top,"step":step})
    };
    let neg = max_for((-lo) as u32);
    let pos = max_for(hi as u32);
    let l = -i128::from(neg["magnitude"].as_u64().expect("u64"));
    let h = i128::from(pos["magnitude"].as_u64().expect("u64"));
    json!({"negative":neg,"positive":pos,"lo":l as i64,"hi":h as i64,"width":signed_width(l,h)})
}

fn case(d: u32, layers: u32, mode: u8) -> Value {
    let (arch, quant, silu) = fixture::fixture(Profile::Servable, 2, d, layers, mode);
    let model = PublicModel {
        arch: &arch,
        profile: Profile::Servable,
        quant: &quant,
        silu_in_exponents: &silu,
    };
    let lowering = lower(model, 16).expect("fixture lowers");
    let graph = &lowering.graph;
    let ranges = analyze(graph, &lowering.tables).expect("ranges");
    let policy = GadgetPolicy::default();
    let selection = select(graph, &ranges, &lowering.tables, &policy).expect("selection");
    let baseline = report(model, &lowering, &selection, &policy)
        .expect("report")
        .to_json(&json!({}));
    let mut replacement = BTreeMap::new();
    let mut cache = BTreeMap::new();
    let mut norms = vec![];
    let mut original_nodes = vec![];
    let n = u128::from(d) << (2 * quant.rsqrt_kappa);
    for &(node, gadget) in &selection.gadgets {
        let nd = &graph.nodes[node.0 as usize];
        let output = nd.outputs[0];
        let elements = graph.tensors[output.0 as usize]
            .shape
            .iter()
            .map(|&v| u64::from(v))
            .product::<u64>();
        let point = lowering
            .sites
            .iter()
            .find(|(_, t)| *t == output)
            .map(|(p, _)| format!("{p:?}"));
        original_nodes.push(json!({"node":node.0,"point":point,"elements":elements,"gadget":gadget_json(gadget,&selection.tables)}));
        let Op::Requant { shift, .. } = nd.op else {
            continue;
        };
        let Some(mul) = graph.producer(nd.inputs[0]) else {
            continue;
        };
        let mul = &graph.nodes[mul.0 as usize];
        if !matches!(mul.op, Op::Mul { .. }) {
            continue;
        }
        let root = mul.inputs.iter().find_map(|&t| {
            let p = graph.producer(t)?;
            matches!(graph.nodes[p.0 as usize].op, Op::RsqrtRow { .. }).then_some(t)
        });
        let Some(root) = root else { continue };
        let residual = *mul.inputs.iter().find(|&&t| t != root).expect("residual");
        let Range::Bounded(res) = ranges.of(residual).expect("res range") else {
            panic!("bounded")
        };
        let Range::Bounded(rho) = ranges.of(root).expect("root range") else {
            panic!("bounded")
        };
        let Range::Bounded(key) = ranges.of(nd.inputs[0]).expect("key range") else {
            panic!("bounded")
        };
        let first = norms.is_empty();
        // Embedding weights are [-127,127] and shift1: even [-254,254].
        let step = if first { 2 } else { 1 };
        let ex = cache
            .entry((res.lo, res.hi, step))
            .or_insert_with(|| extrema(n, quant.rms_eps_int, res.lo, res.hi, step))
            .clone();
        let refined = mini(
            i128::from(ex["lo"].as_i64().expect("lo")),
            i128::from(ex["hi"].as_i64().expect("hi")),
            &graph.shift_classes[shift.0 as usize],
        );
        let baseline_mini = mini(key.lo, key.hi, &graph.shift_classes[shift.0 as usize]);
        assert_eq!(
            gadget_json(gadget, &selection.tables)["tag"],
            gadget_json(baseline_mini.gadgets[0].1, &baseline_mini.tables)["tag"]
        );
        let root_nd = &graph.nodes[graph.producer(root).expect("root producer").0 as usize];
        norms.push(json!({"node":node.0,"point":point,"first_embedding":first,"elements":elements,"residual_range":[res.lo as i64,res.hi as i64],"step":step,"sum_range":format!("{:?}",ranges.of(root_nd.inputs[0])),"root_range":[rho.lo as i64,rho.hi as i64],"key_range":[key.lo as i64,key.hi as i64],"baseline_width":signed_width(key.lo,key.hi),"shifts":graph.shift_classes[shift.0 as usize],"extrema":ex,"baseline":gadget_json(gadget,&selection.tables),"candidate":gadget_json(refined.gadgets[0].1,&refined.tables)}));
        replacement.insert(node, refined);
    }
    assert_eq!(norms.len(), 2 * layers as usize + 1);
    let mut candidate = Selection {
        gadgets: vec![],
        tables: lowering.tables.clone(),
    };
    for &(node, gadget) in &selection.gadgets {
        let (g, ts) = replacement
            .get(&node)
            .map_or((gadget, selection.tables.as_slice()), |s| {
                (s.gadgets[0].1, s.tables.as_slice())
            });
        let new = remap(g, ts, &mut candidate.tables);
        candidate.gadgets.push((node, new));
    }
    let candidate_report = report(model, &lowering, &candidate, &policy)
        .expect("candidate report")
        .to_json(&json!({}));
    json!({"d":d,"layers":layers,"mode":mode,"kappa":quant.rsqrt_kappa,"epsilon":quant.rms_eps_int,"N":n.to_string(),"positions":16,"initial_tables":lowering.tables.iter().map(table_json).collect::<Vec<_>>(),"baseline":baseline,"candidate":candidate_report,"baseline_tables":selection.tables.iter().map(table_json).collect::<Vec<_>>(),"candidate_tables":candidate.tables.iter().map(table_json).collect::<Vec<_>>(),"norms":norms,"all_gadgets":original_nodes})
}

fn invalid_roots() -> Vec<Value> {
    [(0u64,16u8,16384u32),(1,26,4096),(1,16,16384)].into_iter().map(|(eps,kappa,dim)|{
        let mut b=GraphBuilder::new();
        let kind=NodeKind::Committed{commitment:CommitmentClass::Advice};
        let x=b.input(TensorDecl{shape:vec![1],ty:IntType::I32,range:IntRange{lo:0,hi:0},kind}).expect("input");
        b.node(Op::RsqrtRow{kappa,eps_int:eps,dim},&[x],vec![TensorDecl{shape:vec![1],ty:IntType::I32,range:IntRange{lo:0,hi:0},kind}]).expect("root graph");
        let result=analyze(&b.finish(),&[]);
        json!({"epsilon":eps,"kappa":kappa,"d":dim,"accepted":result.is_ok(),"error":result.err().map(|v|format!("{v:?}"))})
    }).collect()
}

fn lookup_mode() {
    let mut text = String::new();
    std::io::stdin()
        .read_to_string(&mut text)
        .expect("requests");
    let groups: Vec<Value> = serde_json::from_str(&text).expect("JSON requests");
    let output=groups.iter().map(|group| {
        let lo=i128::from(group["lo"].as_i64().expect("lo"));
        let hi=i128::from(group["hi"].as_i64().expect("hi"));
        let shift=group["shift"].as_u64().expect("shift") as u8;
        let selection=mini(lo,hi,&[shift]);
        let tables=selection.tables.iter().map(|t|GeneratedTable::from_spec(t).expect("table")).collect::<Vec<_>>();
        let values=group["keys"].as_array().expect("keys").iter().map(|key| {
            let z=key.as_i64().expect("key");
            match selection.gadgets[0].1 {
                Gadget::Requant1{table}=>tables[table.0 as usize].lookup(z,0).expect("key in table"),
                Gadget::RequantSplit{split,low,high}=>{
                    let bottom=(z+(1i64<<(signed_width(lo,hi)-1))).rem_euclid(1i64<<split);
                    assert_eq!(tables[low.0 as usize].lookup(bottom,0),Some(0));
                    tables[high.0 as usize].lookup(z.div_euclid(1i64<<split),0).expect("high key")
                },
                _=>panic!("norm takes single or split"),
            }
        }).collect::<Vec<_>>();
        json!({"name":group["name"],"gadget":gadget_json(selection.gadgets[0].1,&selection.tables),"values":values})
    }).collect::<Vec<_>>();
    println!("{}", serde_json::to_string(&output).expect("JSON"));
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("lookup") {
        lookup_mode();
        return;
    }
    let cases = [(4096, 32), (16384, 126)]
        .into_iter()
        .flat_map(|(d, l)| [30, 16, 31].into_iter().map(move |m| case(d, l, m)))
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"schema":"EXP-0017/compiler/1","cases":cases,"root_controls":invalid_roots()})
        )
        .expect("JSON")
    );
}
