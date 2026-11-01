// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Generated public rows paired with the exact canonical P1 table-count source.

use super::{LeafError, internal};
use tessera_codec::SecretBuf;
use tessera_field::{Field, Goldilocks as G, PrimeField};
use tessera_logup::{TableSource, Tuples};
use tessera_sumcheck::WindowError;
use tessera_tables::{GeneratedTable, Table, TableId, TableKind};
use tessera_witness::TileWitness;

pub(crate) struct GeneratedCounts {
    table: GeneratedTable,
    tile: TileWitness,
    key_0: [G; 1],
    value_0: Option<[G; 1]>,
    cache_depth: Option<usize>,
}

impl GeneratedCounts {
    pub(crate) fn new(table: GeneratedTable, tile: TileWitness) -> Result<Self, LeafError> {
        usize::try_from(table.row_count()).map_err(|_| LeafError::Memory)?;
        let counts = tile.multiplicities().ok_or(LeafError::Internal {
            rule: "a generated table source holds a canonical multiplicity tile",
        })?;
        if counts.table() != table.spec().id
            || Some(counts.rows()) != 1u64.checked_shl(u32::from(table.log2_rows()))
        {
            return internal(
                "the generated table and committed counts have identical table identity and padded domain",
            );
        }
        let (key, value) = table.row(0).ok_or(LeafError::Internal {
            rule: "a generated table has row zero",
        })?;
        let value_0 = (!matches!(table.spec().kind, TableKind::Range { .. })).then_some([value]);
        Ok(Self {
            table,
            tile,
            key_0: [key],
            value_0,
            cache_depth: None,
        })
    }
    pub(crate) fn with_cache_depth(mut self, depth: Option<usize>) -> Self {
        self.cache_depth = depth;
        self
    }
    pub(crate) fn id(&self) -> TableId {
        self.table.spec().id
    }
    pub(crate) fn check_keys(&self, keys: &[G]) -> Result<(), LeafError> {
        let mut fits = true;
        for key in keys {
            fits &= key.to_canonical_u64() < self.table.row_count();
        }
        if !fits {
            return internal("every lookup's key names a row of its table (INV-IR-2)");
        }
        Ok(())
    }
    pub(crate) fn padding(&self) -> Tuples<'_, G> {
        Tuples::new(&self.key_0, self.value_0.as_ref().map(|v| &v[..]))
    }
    pub(crate) fn bytes(&self) -> Result<u64, LeafError> {
        self.table
            .storage_bytes()
            .and_then(|n| n.checked_add(self.tile.bytes()))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(LeafError::Memory)
    }
}

impl TableSource<G> for GeneratedCounts {
    fn rows(&self) -> usize {
        self.table.row_count() as usize
    }
    fn has_values(&self) -> bool {
        self.value_0.is_some()
    }
    fn fraction_cache_depth(&self) -> Option<usize> {
        self.cache_depth
    }
    fn read_scratch_bytes(&self, rows: usize) -> Option<u64> {
        (rows as u64).checked_mul(size_of::<u64>() as u64)
    }
    fn read(
        &self,
        first: usize,
        keys: &mut [G],
        values: &mut [G],
        counts: &mut [G],
    ) -> Result<(), WindowError> {
        let column = self.tile.multiplicities().ok_or(WindowError::Source)?;
        if keys.len() != values.len()
            || keys.len() != counts.len()
            || first
                .checked_add(keys.len())
                .is_none_or(|end| end as u64 > column.rows())
        {
            return Err(WindowError::Shape(
                "a generated table chunk fits its padded count domain",
            ));
        }
        let mut raw =
            SecretBuf::try_with_capacity(keys.len()).map_err(|_| WindowError::Allocation)?;
        raw.resize(keys.len(), 0u64);
        column
            .read_rows(first as u64, &mut raw)
            .map_err(|_| WindowError::Source)?;
        for j in 0..keys.len() {
            let (key, value) = self
                .table
                .row((first + j) as u64)
                .ok_or(WindowError::Source)?;
            keys[j] = key;
            values[j] = if self.has_values() { value } else { G::ZERO };
            // RFC-0008: the canonical counter already enforces total < P, so no alias.
            counts[j] = G::from_u64_reduced(raw[j]);
        }
        Ok(())
    }
}

pub(crate) fn source(
    tables: &[GeneratedCounts],
    id: TableId,
) -> Result<&GeneratedCounts, LeafError> {
    tables
        .iter()
        .find(|t| t.id() == id)
        .ok_or(LeafError::Internal {
            rule: "every leaf instance has its generated table/count source",
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_codec::{DecodeLimits, Secret, envelope, schema};
    use tessera_compiler::{PublicModel, SequenceTile, Share, Tile};
    use tessera_ir::{CommitmentClass, GadgetPolicy, IntegerModel, NodeKind, Op};
    use tessera_witness::{Engine, Resident, WitnessSource};

    fn fixture(valued: bool) -> (GeneratedTable, TileWitness, SecretBuf<u64>) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../spec/vectors/models/tiny-2-servable.d");
        let model: IntegerModel = envelope::open(
            &std::fs::read(path).unwrap(),
            &schema::INTEGER_MODEL,
            DecodeLimits::INTEGER_MODEL,
        )
        .unwrap();
        let lowering = tessera_compiler::lower(PublicModel::of(&model), 3).unwrap();
        let ranges = tessera_compiler::analyze(&lowering.graph, &lowering.tables).unwrap();
        let selection = tessera_compiler::select(
            &lowering.graph,
            &ranges,
            &lowering.tables,
            &GadgetPolicy {
                max_log2_rows: 32,
                ..Default::default()
            },
        )
        .unwrap();
        let id = selection
            .gadgets
            .iter()
            .filter_map(|(node, _)| selection.reads(*node))
            .flatten()
            .filter(|&&id| {
                let spec = selection.tables.iter().find(|t| t.id == id).unwrap();
                if !valued {
                    return matches!(spec.kind, TableKind::Range { .. });
                }
                if !matches!(spec.kind, TableKind::Requant { .. }) {
                    return false;
                }
                let t = GeneratedTable::from_spec(spec).unwrap();
                t.row(0).unwrap().1 != G::ZERO || t.row(t.row_count() - 1).unwrap().1 != G::ZERO
            })
            .max_by_key(|&&id| {
                let spec = selection.tables.iter().find(|t| t.id == id).unwrap();
                (
                    spec.log2_rows,
                    matches!(
                        spec.kind,
                        TableKind::Range { .. } | TableKind::Requant { .. }
                    ),
                )
            })
            .copied()
            .unwrap();
        let table =
            GeneratedTable::from_spec(selection.tables.iter().find(|t| t.id == id).unwrap())
                .unwrap();
        let engine = Engine::new(&lowering).unwrap();
        let witness = engine.run(&model, &Secret::new(vec![1, 2, 3])).unwrap();
        let source = Resident::new(&engine, &selection, &model, &witness).unwrap();
        let node = selection
            .gadgets
            .iter()
            .find(|(node, _)| selection.reads(*node).is_some_and(|ids| ids.contains(&id)))
            .unwrap()
            .0;
        let node = &lowering.graph.nodes[node.0 as usize];
        assert!(!matches!(node.op, Op::SoftmaxRow { .. }));
        let shares = node
            .outputs
            .iter()
            .filter_map(|&t| {
                let spec = &lowering.graph.tensors[t.0 as usize];
                matches!(
                    spec.kind,
                    NodeKind::Committed {
                        commitment: CommitmentClass::Advice
                    } | NodeKind::LookupOutput { .. }
                )
                .then(|| Share {
                    tensor: t,
                    ranges: spec.shape.iter().map(|&n| 0..n).collect(),
                })
            })
            .collect();
        let node_witness = source
            .tile(&SequenceTile {
                tile: Tile::Node { node: node.id.0 },
                block: None,
                shares,
            })
            .unwrap();
        let key = node_witness
            .lookups()
            .find(|l| l.table() == id)
            .unwrap()
            .keys()[0];
        let known_row: SecretBuf<u64> = core::iter::once(u64::from(key)).collect();
        let tile = source
            .tile(&SequenceTile {
                tile: Tile::Table { table: id.0 },
                block: None,
                shares: Vec::new(),
            })
            .unwrap();
        (table, tile, known_row)
    }

    #[test]
    fn canonical_counts_and_generated_final_rows_match_under_the_chunk_bound() {
        for valued in [false, true] {
            let (table, tile, known_row) = fixture(valued);
            let column = tile.multiplicities().unwrap();
            if !valued {
                assert_eq!(column.rows(), 1u64 << 30);
            }
            let len = 13.min(column.rows() as usize);
            let first = column.rows() as usize - len;
            let mut expected = SecretBuf::filled(0u64, len);
            column.read_rows(first as u64, &mut expected).unwrap();
            let mut expected_hit = SecretBuf::filled(0u64, 1);
            column.read_rows(known_row[0], &mut expected_hit).unwrap();
            assert!(
                expected_hit[0] > 0,
                "the oracle row must exercise a nonzero multiplicity"
            );
            let source = GeneratedCounts::new(table.clone(), tile).unwrap();
            let mut k = SecretBuf::filled(G::ZERO, len);
            let mut v = SecretBuf::filled(G::ZERO, len);
            let mut m = SecretBuf::filled(G::ZERO, len);
            let measured =
                allocation_counter::measure(|| source.read(first, &mut k, &mut v, &mut m).unwrap());
            assert!(measured.bytes_max as u64 <= source.read_scratch_bytes(len).unwrap());
            assert_eq!(measured.bytes_current, 0);
            for at in 0..len {
                let (key, value) = table.row((first + at) as u64).unwrap();
                assert_eq!(k[at], key);
                assert_eq!(v[at], if source.has_values() { value } else { G::ZERO });
                assert_eq!(m[at], G::from_u64_reduced(expected[at]));
            }
            let mut one_k = SecretBuf::filled(G::ZERO, 1);
            let mut one_v = SecretBuf::filled(G::ZERO, 1);
            let mut one_m = SecretBuf::filled(G::ZERO, 1);
            source
                .read(known_row[0] as usize, &mut one_k, &mut one_v, &mut one_m)
                .unwrap();
            assert_eq!(one_m[0], G::from_u64_reduced(expected_hit[0]));
            let (key, value) = table.row(known_row[0]).unwrap();
            assert_eq!(one_k[0], key);
            assert_eq!(one_v[0], if source.has_values() { value } else { G::ZERO });
            assert_eq!(source.has_values(), valued);
            for row in [0, source.rows() - 1] {
                let mut raw = SecretBuf::filled(0u64, 1);
                source
                    .tile
                    .multiplicities()
                    .unwrap()
                    .read_rows(row as u64, &mut raw)
                    .unwrap();
                source
                    .read(row, &mut one_k, &mut one_v, &mut one_m)
                    .unwrap();
                let (key, value) = table.row(row as u64).unwrap();
                assert_eq!(one_k[0], key);
                assert_eq!(one_v[0], if valued { value } else { G::ZERO });
                assert_eq!(one_m[0], G::from_u64_reduced(raw[0]));
            }
            let before: SecretBuf<G> = k.iter().copied().collect();
            assert_eq!(
                source.read(column_rows(&source), &mut k, &mut v, &mut m),
                Err(WindowError::Shape(
                    "a generated table chunk fits its padded count domain"
                ))
            );
            assert!(k[..] == before[..]);
            let mut wrong = SecretBuf::filled(G::ZERO, len + 1);
            assert!(matches!(
                source.read(0, &mut k, &mut wrong, &mut m),
                Err(WindowError::Shape(_))
            ));
            assert!(k[..] == before[..]);
            let mut wrong_counts = SecretBuf::filled(G::ZERO, len + 1);
            assert!(matches!(
                source.read(0, &mut k, &mut v, &mut wrong_counts),
                Err(WindowError::Shape(_))
            ));
            assert!(k[..] == before[..]);
            assert!(matches!(
                source.read(usize::MAX, &mut k, &mut v, &mut m),
                Err(WindowError::Shape(_))
            ));
            assert!(k[..] == before[..]);
            assert!(source.check_keys(&[G::ZERO]).is_ok());
            for key in [G::from_u64_reduced(source.rows() as u64), -G::ONE] {
                assert_eq!(
                    source.check_keys(&[key]),
                    Err(LeafError::Internal {
                        rule: "every lookup's key names a row of its table (INV-IR-2)",
                    })
                );
            }
        }
    }
    fn column_rows(source: &GeneratedCounts) -> usize {
        source.tile.multiplicities().unwrap().rows() as usize
    }

    #[test]
    fn mismatched_table_identity_and_padded_count_domain_are_refused() {
        for id_mismatch in [true, false] {
            let (table, tile, _) = fixture(false);
            let counts = tile.multiplicities().unwrap();
            let id = if id_mismatch {
                TableId(table.spec().id.0 + 10_000)
            } else {
                table.spec().id
            };
            let bits = if counts.rows() == 2 { 2 } else { 1 };
            let wrong = if id_mismatch {
                GeneratedTable::new(id, table.spec().kind.clone()).unwrap()
            } else {
                GeneratedTable::new(id, TableKind::Range { bits }).unwrap()
            };
            let result = GeneratedCounts::new(wrong, tile);
            assert!(matches!(
                result,
                Err(LeafError::Internal {
                    rule: "the generated table and committed counts have identical table identity and padded domain",
                })
            ));
        }
    }
}
