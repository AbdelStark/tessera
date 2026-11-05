// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::*;
use std::cell::Cell;
use tessera_tables::{Table as _, TableKind};
use tessera_witness::{TileWitness, WitnessError, WitnessSource};

struct Refuse<'a> {
    source: &'a dyn WitnessSource,
    calls: Cell<usize>,
    at: usize,
    tensor: tessera_ir::TensorId,
}
impl WitnessSource for Refuse<'_> {
    fn tile(&self, tile: &tessera_compiler::SequenceTile) -> Result<TileWitness, WitnessError> {
        let n = self.calls.get();
        self.calls.set(n + 1);
        if n == self.at {
            Err(WitnessError::Memory {
                tensor: self.tensor,
            })
        } else {
            self.source.tile(tile)
        }
    }
}

#[test]
fn partial_scoped_sources_release_every_owned_buffer_on_callback_refusal() {
    let model = bounded_model();
    let lowering = tessera_compiler::lower(PublicModel::of(&model), 3).unwrap();
    let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
    let policy = GadgetPolicy {
        max_log2_rows: 21,
        query_block: 2,
        ..Default::default()
    };
    let selection =
        select_model(tessera_ir::PROTOCOL_VERSION, &lowering, &ranges, &policy).unwrap();
    let params = ParamSet::TEST_SMALL;
    let claims = claim_list(
        PublicModel::of(&model),
        &selection,
        &policy,
        params.layout,
        ParamSetId(Ascii::new(params.id).unwrap()),
    )
    .unwrap();
    let layers = LayerMap::new(&lowering).unwrap();
    let layout = LeafLayout::model(&lowering, &layers, &selection, &claims).unwrap();
    let advice = ModelAdviceLayout::new(&lowering, &selection, &claims).unwrap();
    let grouped = GroupedLayout::new(
        &layout,
        &layers,
        LeafGroups {
            token_bits: 1,
            layer_bits: 1,
        },
    )
    .unwrap();
    let engine = Engine::new(&lowering).unwrap();
    let witness = engine.run(&model, &Secret::new(vec![1, 2, 3])).unwrap();
    let source = Resident::new(&engine, &selection, &model, &witness).unwrap();
    for at in [0, 2] {
        let refused = Refuse {
            source: &source,
            calls: Cell::new(0),
            at,
            tensor: lowering.tokens,
        };
        let factory = ModelSourceFactory {
            layout: &layout,
            grouped: &grouped,
            lowering: &lowering,
            layers: &layers,
            witness: &witness,
            source: &refused,
            advice: &advice,
        };
        let mut outcome = None;
        let measured = allocation_counter::measure(|| {
            outcome = Some(factory.group(0));
        });
        assert!(
            matches!(outcome.unwrap(),Err(LeafError::Witness(WitnessError::Memory {tensor})) if tensor==lowering.tokens)
        );
        assert_eq!(refused.calls.get(), at + 1);
        assert!(measured.count_total > 0 && measured.bytes_max > 0);
        assert_eq!(
            measured.bytes_current, 0,
            "no failed scope retains raw columns or forest allocations"
        );
    }
    let padding: Vec<_> = layout
        .instances()
        .iter()
        .map(|i| {
            let spec = selection.tables.iter().find(|t| t.id == i.table).unwrap();
            let generated = GeneratedTable::from_spec(spec).unwrap();
            let (key, value) = generated.row(0).unwrap();
            (
                i.table,
                [key],
                (!matches!(spec.kind, TableKind::Range { .. })).then_some([value]),
            )
        })
        .collect();
    let factory = ModelSourceFactory {
        layout: &layout,
        grouped: &grouped,
        lowering: &lowering,
        layers: &layers,
        witness: &witness,
        source: &source,
        advice: &advice,
    };
    for at in [0, 2] {
        let calls = Cell::new(0);
        let mut outcome = None;
        let measured = allocation_counter::measure(|| {
            outcome = Some(factory.group_inputs(0, |id| {
                let n = calls.get();
                calls.set(n + 1);
                if n == at {
                    return Err(LeafError::Memory);
                }
                let (_, key, value) = padding.iter().find(|p| p.0 == id).unwrap();
                Ok(Tuples::new(key, value.as_ref().map(|v| &v[..])))
            }));
        });
        assert!(matches!(outcome.unwrap(), Err(LeafError::Memory)));
        assert_eq!(calls.get(), at + 1);
        assert!(measured.count_total > 0 && measured.bytes_max > 0);
        assert_eq!(
            measured.bytes_current, 0,
            "no failed conversion retains i64/field/forest allocations"
        );
    }
}
