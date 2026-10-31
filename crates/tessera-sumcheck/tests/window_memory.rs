// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Invalid public geometry must fail before allocation or source access.
use allocation_counter::measure;
use tessera_field::GoldilocksExt2 as F;
use tessera_sumcheck::{WindowError, WindowMetadataMemory, WindowShape};

#[test]
fn invalid_window_metadata_is_refused_without_allocating() {
    let shape = WindowShape {
        vars: 3,
        tables: 2,
        delta: 2,
        windowable: true,
    };
    let measured = measure(|| {
        let mismatch = WindowError::Shape("window metadata matches canonical nonempty specs");
        for degree in [0, 8, usize::MAX] {
            assert_eq!(
                WindowMetadataMemory::new::<F>(&[shape], &[1], degree),
                Err(mismatch)
            );
        }
        assert_eq!(WindowMetadataMemory::new::<F>(&[], &[], 3), Err(mismatch));
        assert_eq!(
            WindowMetadataMemory::new::<F>(&[shape], &[], 3),
            Err(mismatch)
        );
        for bad in [
            WindowShape {
                vars: usize::BITS as usize,
                ..shape
            },
            WindowShape { tables: 0, ..shape },
            WindowShape { delta: 0, ..shape },
            WindowShape { delta: 4, ..shape },
        ] {
            assert_eq!(
                WindowMetadataMemory::new::<F>(&[bad], &[1], 3),
                Err(WindowError::Shape(
                    "window counts cover each table and term"
                ))
            );
        }
        assert_eq!(
            WindowMetadataMemory::new::<F>(&[shape], &[0], 3),
            Err(WindowError::Shape(
                "window counts cover each table and term"
            ))
        );
        assert!(WindowMetadataMemory::new::<F>(&[shape], &[1], 3).is_ok());
    });
    assert_eq!(measured.count_total, 0);
}

#[test]
fn unaddressable_window_vectors_are_refused_without_allocating() {
    let shape = WindowShape {
        vars: 1,
        tables: 1,
        delta: 1,
        windowable: false,
    };
    let measured = measure(|| {
        assert_eq!(
            WindowMetadataMemory::new::<F>(
                &[WindowShape {
                    tables: usize::MAX,
                    ..shape
                }],
                &[1],
                1
            ),
            Err(WindowError::Allocation)
        );
        assert_eq!(
            WindowMetadataMemory::new::<F>(&[shape], &[u64::MAX], 1),
            Err(WindowError::Allocation)
        );
    });
    assert_eq!(measured.count_total, 0);
}
