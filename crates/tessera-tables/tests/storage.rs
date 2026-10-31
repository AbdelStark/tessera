// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public core geometry and one-allocation construction (RFC-0008, RFC-0018, #641).

use tessera_codec::Rational;
use tessera_tables::{GeneratedTable, TableError, TableId, TableKind};

#[test]
fn large_domains_are_planned_without_generating_values() {
    let kinds = [
        (TableKind::Range { bits: 32 }, Ok(0)),
        (
            TableKind::Silu {
                b: 24,
                r_by_class: vec![1, 2, 3, 3],
                in_exp: -10,
                out_exp: -4,
            },
            Ok(1 << 24),
        ),
        (
            TableKind::Silu {
                b: 24,
                r_by_class: vec![1, 1, 2, 3],
                in_exp: -10,
                out_exp: -4,
            },
            Err(TableError::Parameter {
                kind: "Silu",
                rule: "at most 2^24 values of h over the classes",
            }),
        ),
        (
            TableKind::Exp {
                key_bits: 32,
                sigma: 16,
                scale: Rational::new(1, 1 << 40).unwrap(),
            },
            Err(TableError::Parameter {
                kind: "Exp",
                rule: "a zero tail from key 2^24 or earlier",
            }),
        ),
    ];
    for (kind, expected) in kinds {
        let measured = allocation_counter::measure(|| {
            assert_eq!(GeneratedTable::core_storage_len(&kind), expected);
        });
        assert_eq!(measured.bytes_total, 0);
        assert_eq!(measured.bytes_max, 0);
    }
}

#[test]
fn mixed_silu_classes_reserve_the_exact_requested_payload_before_generation() {
    // RFC-0008: 2^11 + 2^9 + 2^4 values; this is not a power of two.
    let shifts = vec![1, 3, 8];
    let classes = shifts.capacity();
    let kind = TableKind::Silu {
        b: 12,
        r_by_class: shifts,
        in_exp: -10,
        out_exp: -4,
    };
    assert_eq!(GeneratedTable::core_storage_len(&kind), Ok(2576));
    let mut table = None;
    let measured = allocation_counter::measure(|| {
        table = Some(GeneratedTable::new(TableId(0), kind).unwrap());
    });
    let table = table.unwrap();
    assert_eq!(
        table.storage_bytes(),
        Some(size_of::<GeneratedTable>() + classes + 2576 * 8)
    );
    // The class vector was supplied by the caller; construction allocates one core only.
    // This measures requested allocator bytes, not process RSS or allocator bookkeeping.
    assert_eq!(measured.bytes_current, 2576 * 8);
    assert_eq!(measured.bytes_max, 2576 * 8);
}
