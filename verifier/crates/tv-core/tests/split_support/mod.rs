// SPDX-License-Identifier: Apache-2.0 OR MIT

use tv_core::advice::Source;
use tv_core::leaf::split::{Affine, Column, Layer};
use tv_core::lowering::{Graph, NodeId};
use tv_field::Fp2;

pub fn reads(layer: &Layer) -> Vec<Column> {
    let mut found = Vec::new();
    for lookup in &layer.lookups {
        for form in [Some(&lookup.key), lookup.value.as_ref()]
            .into_iter()
            .flatten()
        {
            for (_, read) in &form.terms {
                if !found.contains(&read.column) {
                    found.push(read.column);
                }
            }
        }
    }
    found
}

pub fn local(graph: &Graph, owner: NodeId, source: Source) -> Column {
    match source {
        Source::Auxiliary(node, role) => {
            assert_eq!(node, owner);
            Column::Auxiliary(role)
        }
        Source::Tensor(id) => {
            let node = graph.node(owner).unwrap();
            if let Some(i) = node.inputs().iter().position(|&t| t == id) {
                Column::Input(i.try_into().unwrap())
            } else {
                Column::Output(
                    node.outputs()
                        .iter()
                        .position(|&t| t == id)
                        .unwrap()
                        .try_into()
                        .unwrap(),
                )
            }
        }
        Source::Multiplicity(_) | Source::GreedyMaximum(_) => panic!("not a model gadget source"),
    }
}

pub fn check(
    layer: &Layer,
    values: &[(Column, Fp2)],
    keys: &[Fp2],
    outputs: &[Fp2],
    variables: impl Fn(Column) -> usize,
) {
    assert_eq!(layer.lookups.len(), keys.len());
    assert_eq!(keys.len(), outputs.len());
    let evaluate = |form: &Affine| {
        form.terms
            .iter()
            .fold(Fp2::from(form.constant), |sum, (coefficient, read)| {
                // All frozen gadget source points use natural prefix coordinates.
                assert_eq!(
                    read.coordinates,
                    (0..variables(read.column) as u32).collect::<Vec<_>>()
                );
                sum + Fp2::from(*coefficient)
                    * values.iter().find(|(c, _)| *c == read.column).unwrap().1
            })
    };
    for ((lookup, &key), &value) in layer.lookups.iter().zip(keys).zip(outputs) {
        assert_eq!(
            evaluate(&lookup.key),
            key,
            "node {:?}, table {:?}",
            layer.node,
            lookup.table
        );
        assert_eq!(
            lookup.value.as_ref().map(evaluate).unwrap_or(Fp2::ZERO),
            value
        );
    }
}
