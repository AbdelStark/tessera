// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Actual requantization witnesses/forms against the pre-verifier integer oracle.
use serde_json::Value;
use tessera_compiler::Gadget;
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::{Committed, Requant, Term};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind};

fn array(value: &Value) -> &[Value] {
    value.as_array().unwrap()
}
fn id(value: &Value) -> TableId {
    TableId(value.as_u64().unwrap() as u32)
}
fn field(value: &Value) -> F {
    F::new(
        G::from_canonical_u64(value[0].as_u64().unwrap()).unwrap(),
        G::from_canonical_u64(value[1].as_u64().unwrap()).unwrap(),
    )
}
fn generated(row: &Value) -> GeneratedTable {
    let kind = row[1].as_object().unwrap();
    let (name, p) = kind.iter().next().unwrap();
    let shifts = || {
        array(&p["r_by_class"])
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect()
    };
    let kind = match name.as_str() {
        "Range" => TableKind::Range {
            bits: p["bits"].as_u64().unwrap() as u8,
        },
        "Requant" => TableKind::Requant {
            b: p["b"].as_u64().unwrap() as u8,
            r_by_class: shifts(),
            lo: p["lo"].as_i64().unwrap() as i32,
            hi: p["hi"].as_i64().unwrap() as i32,
        },
        "Silu" => TableKind::Silu {
            b: p["b"].as_u64().unwrap() as u8,
            r_by_class: shifts(),
            in_exp: p["in_exp"].as_i64().unwrap() as i8,
            out_exp: p["out_exp"].as_i64().unwrap() as i8,
        },
        _ => panic!("unexpected table {name}"),
    };
    GeneratedTable::new(id(&row[0]), kind).unwrap()
}
fn selected(site: &Value) -> Gadget {
    let pattern = array(&site["pattern"]);
    let tables: Vec<_> = array(&site["tables"]).iter().map(id).collect();
    match pattern[0].as_str().unwrap() {
        "Requant1" => Gadget::Requant1 { table: tables[0] },
        "RequantSplit" => Gadget::RequantSplit {
            split: pattern[1].as_u64().unwrap() as u8,
            low: tables[0],
            high: tables[1],
        },
        "RequantWide" => {
            let split = pattern[1].as_u64().unwrap() as u8;
            let offset = usize::from(split > 0);
            Gadget::RequantWide {
                split,
                middle: pattern[2].as_u64().unwrap() as u8,
                low: (split > 0).then_some(tables[0]),
                mid: tables[offset],
                top: tables[offset + 1],
                sign: tables[offset + 2],
                high: tables[offset + 3],
            }
        }
        _ => panic!("not a requantization"),
    }
}
fn committed(source: &Value, index: usize) -> Option<Committed> {
    if index == 1 {
        return Some(Committed::Y);
    }
    if source[0] != "auxiliary" {
        return None;
    }
    Some(match source[2].as_str().unwrap() {
        "requant.low" => Committed::ZLo,
        "requant.middle" => Committed::H0,
        "requant.saturation" => Committed::G,
        _ => panic!("unknown role"),
    })
}

#[test]
fn actual_cpu_columns_and_lookup_forms_match_all_frozen_model_sites() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../spec/vectors/gadget-finals/requant.json"
    ))
    .unwrap();
    let (mut requants, mut silus) = (0, 0);
    for case in array(&fixture["cases"]) {
        let tables: Vec<_> = array(&case["tables"]).iter().map(generated).collect();
        let specs: Vec<_> = tables.iter().map(|t| t.spec().clone()).collect();
        for site in array(&case["sites"]) {
            let columns: Vec<Vec<_>> = array(&site["columns"])
                .iter()
                .map(|c| array(c).iter().map(|v| v.as_i64().unwrap()).collect())
                .collect();
            let sources = array(&site["sources"]);
            let class = sources.get(2).is_some_and(|v| v[0] == "tensor");
            let classes: Option<Vec<u8>> =
                class.then(|| columns[2].iter().map(|&v| v as u8).collect());
            if site["pattern"][0] == "Lookup" {
                let table = tables
                    .iter()
                    .find(|t| t.spec().id == id(&site["tables"][0]))
                    .unwrap();
                for (row, &z) in columns[0].iter().enumerate() {
                    assert_eq!(
                        table.lookup(z, classes.as_ref().map_or(0, |v| v[row])),
                        Some(columns[1][row])
                    );
                }
                silus += 1;
                continue;
            }
            let gadget = Requant::new(&selected(site), &specs).unwrap();
            let witness = gadget.witness(&columns[0], classes.as_deref()).unwrap();
            for (i, source) in sources.iter().enumerate() {
                if let Some(role) = committed(source, i) {
                    assert_eq!(witness.tensor(role).unwrap(), columns[i]);
                }
            }
            let point: Vec<_> = array(&site["point"]).iter().map(field).collect();
            let evaluations: Vec<_> = columns
                .iter()
                .enumerate()
                .map(|(i, column)| {
                    let value = column.iter().enumerate().fold(F::ZERO, |out, (row, &v)| {
                        let weight = point.iter().enumerate().fold(F::ONE, |w, (bit, &x)| {
                            w * if row >> bit & 1 == 0 { F::ONE - x } else { x }
                        });
                        out + weight * F::from(G::from_i64(v))
                    });
                    assert_eq!(value, field(&site["evaluations"][i]));
                    value
                })
                .collect();
            let read = |term| match term {
                Term::Input => evaluations[0],
                Term::Class => {
                    if class {
                        evaluations[2]
                    } else {
                        F::ZERO
                    }
                }
                Term::Committed(role) => {
                    let slot = sources
                        .iter()
                        .enumerate()
                        .position(|(i, s)| committed(s, i) == Some(role))
                        .unwrap();
                    evaluations[slot]
                }
            };
            let constraints = gadget.constraints::<G>();
            assert_eq!(constraints.lookups.len(), array(&site["tables"]).len());
            for (i, lookup) in constraints.lookups.iter().enumerate() {
                assert_eq!(lookup.table, id(&site["tables"][i]));
                assert_eq!(lookup.key.evaluate(read), field(&site["key_evals"][i]));
                assert_eq!(
                    lookup.value.as_ref().map_or(F::ZERO, |v| v.evaluate(read)),
                    field(&site["value_evals"][i])
                );
            }
            requants += 1;
        }
    }
    assert_eq!((requants, silus), (109, 10));
}
