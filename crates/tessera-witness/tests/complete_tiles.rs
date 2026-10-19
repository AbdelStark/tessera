// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0050: each selected current-model gadget supplies every canonical column and lookup.

mod common;

use std::fs;

use serde_json::Value;
use tessera_codec::Secret;
use tessera_compiler::{AuxColumn, CompareColumn, Tile};
use tessera_witness::{Checkpoints, Engine, Regenerator, Resident, WitnessSource};

#[test]
fn every_component_tile_has_the_complete_independent_column_and_request_schema() {
    let path = common::models_dir().join("../advice/layout.json");
    let oracle: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("advice oracle")).expect("JSON");
    for fixture in common::conformance() {
        let expected = oracle["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["model"] == fixture.name && case["n"] == 5 && case["log2_k"] == 10)
            .expect("independently derived model schema");
        let tokens = Secret::new(fixture.sequences[0].tokens[..5].to_vec());
        let (lowering, selected) = common::lowered(&fixture.model, 5);
        let engine = Engine::new(&lowering).expect("engine");
        let first = engine.run(&fixture.model, &tokens).expect("first pass");
        let checkpoints = Checkpoints::new(&engine, &fixture.model, &tokens).expect("checkpoints");
        let resident = Resident::new(&engine, &selected, &fixture.model, &first).expect("resident");
        let regenerated =
            Regenerator::new(&engine, &selected, &fixture.model, &tokens, &checkpoints)
                .expect("regenerator");
        for q in [1, 4, 512] {
            for tile in common::plan_of(&fixture.model, q)
                .expand(&lowering)
                .expect("plan")
            {
                let node = match tile.tile {
                    Tile::Node { node } | Tile::Attention { node, .. } => node,
                    _ => continue,
                };
                let Some(choice) = expected["choices"]
                    .as_array()
                    .expect("choices")
                    .iter()
                    .find(|choice| choice["node"] == node)
                else {
                    continue;
                };
                for source in [&resident as &dyn WitnessSource, &regenerated] {
                    let values = source.tile(&tile).expect("complete component tile");
                    assert_eq!(
                        values.gadget().len(),
                        choice["roles"].as_array().expect("roles").len(),
                        "{} node={node} Q={q}: all canonical auxiliary roles",
                        fixture.name
                    );
                    let roles: Vec<_> = values
                        .gadget()
                        .iter()
                        .map(|(role, _)| role_name(*role))
                        .collect();
                    assert_eq!(
                        serde_json::to_value(roles).expect("roles JSON"),
                        choice["roles"],
                        "{} node={node} Q={q}: exact identities and order",
                        fixture.name
                    );
                    let tables: Vec<_> = values.lookups().map(|lookup| lookup.table().0).collect();
                    let want: Vec<_> = choice["tables"]
                        .as_array()
                        .expect("requests")
                        .iter()
                        .map(|table| table.as_u64().expect("table id") as u32)
                        .collect();
                    assert_eq!(
                        tables, want,
                        "{} node={node} Q={q}: all ordered requests",
                        fixture.name
                    );
                }
            }
        }
    }
}

fn role_name(role: AuxColumn) -> (String, u32) {
    let (name, index) = match role {
        AuxColumn::Low => ("requant.low".into(), 0),
        AuxColumn::Middle => ("requant.middle".into(), 0),
        AuxColumn::Saturation => ("requant.saturation".into(), 0),
        AuxColumn::Maximum => ("softmax.maximum".into(), 0),
        AuxColumn::Slack { second, limb } => (
            (if second { "softmax.b" } else { "softmax.a" }).into(),
            limb,
        ),
        AuxColumn::InputLimb(j) => ("root.input".into(), j),
        AuxColumn::RootByte(j) => ("root.low".into(), j),
        AuxColumn::Comparison {
            upper,
            column,
            limb,
        } => {
            let name = match column {
                CompareColumn::LeftByte => "left.digit",
                CompareColumn::LeftCarry => "left.carry",
                CompareColumn::RightByte => "right.digit",
                CompareColumn::RightCarry => "right.carry",
                CompareColumn::Difference => "difference",
                CompareColumn::Borrow => "borrow",
            };
            (
                format!("root.{}.{name}", if upper { "upper" } else { "lower" }),
                limb,
            )
        }
    };
    (name, index)
}

#[test]
fn a_selection_from_another_gadget_schema_is_refused_before_regeneration() {
    use tessera_compiler::Gadget;
    use tessera_witness::WitnessError;
    let fixture = common::conformance().remove(0);
    let (lowering, selection) = common::lowered(&fixture.model, 3);
    let engine = Engine::new(&lowering).expect("engine");
    let tokens = Secret::new(vec![1, 2, 3]);
    let store = Checkpoints::new(&engine, &fixture.model, &tokens).expect("checkpoints");
    let witness = engine.run(&fixture.model, &tokens).expect("first pass");
    let error = WitnessError::Internal {
        rule: "a selected gadget has its complete canonical column and lookup schema",
    };
    let mut absent_root = selection.clone();
    absent_root
        .gadgets
        .retain(|(_, g)| !matches!(g, Gadget::RsqrtRow));
    let mut other_limb_width = selection.clone();
    let byte = selection
        .tables
        .iter()
        .find(|t| matches!(t.kind, tessera_tables::TableKind::Range { bits: 8 }))
        .expect("byte table")
        .id;
    for (_, gadget) in &mut other_limb_width.gadgets {
        if let Gadget::SoftmaxRow { limb, .. } = gadget {
            *limb = Some(byte);
        }
    }
    for wrong in [&absent_root, &other_limb_width] {
        assert_eq!(
            Regenerator::new(&engine, wrong, &fixture.model, &tokens, &store)
                .expect_err("schema mismatch"),
            error
        );
        assert_eq!(
            Resident::new(&engine, wrong, &fixture.model, &witness).expect_err("schema mismatch"),
            error
        );
    }
}
