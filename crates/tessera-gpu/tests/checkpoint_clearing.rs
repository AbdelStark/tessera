// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The witness engine's checkpoints leave nothing behind when they are freed (05 §3 OBS-6, RFC-0018
//! 'Witness engine').
//!
//! The checkpoint store holds a proof unit's layer-boundary residuals, secret-class activations,
//! each in a buffer that clears its whole allocation when it drops. A recording allocator counts
//! the nonzero bytes of a block as it is freed, as `tessera-field`'s `zeroize` test does for
//! `SecretBuf` itself: every residual's block, freed with the store, holds zeros only. A plain copy
//! of the same residual, freed, holds its nonzero bytes as written, which shows that the residual
//! had some to clear.
//!
//! The test lives in this crate because a recording allocator is unsafe code, which RFC-0001 permits
//! only here and in `tessera-field`, and `tessera-field` lies below the witness engine (01 §3). It
//! needs no GPU.

#![deny(unsafe_code)]

use std::fs;
use std::path::Path;
use std::sync::atomic::Ordering;

use tessera_codec::{DecodeLimits, Secret, SecretBuf, envelope, schema};
use tessera_compiler::{PublicModel, SequenceTile, Tile, analyze, lower, select, tiles};
use tessera_field::{Field, Goldilocks as Fp, GoldilocksExt2 as Fp2, PrimeField};
use tessera_ir::{GadgetPolicy, IntegerModel};
use tessera_pcs::{
    AdviceClaim, AdviceLayout, AdviceOpening, AdvicePoint, Axis, CommitmentParams, DenseShape,
    LayoutParams, Padding, RowEncoder, StreamingCommitter, TensorShape, TileShape,
};
use tessera_witness::{Checkpoints, Engine, Regenerator, WitnessSource};

/// Forwards to `System` and, at the first free of the watched block, counts its nonzero bytes.
#[allow(unsafe_code)]
mod recorder {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::ptr;
    use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

    /// The block whose next free is inspected; null when none is.
    pub static WATCHED: AtomicPtr<u8> = AtomicPtr::new(ptr::null_mut());
    /// The nonzero bytes the watched block held when it was freed; `usize::MAX` until then.
    pub static NONZERO: AtomicUsize = AtomicUsize::new(usize::MAX);
    pub static WATCHED_SIZE: AtomicUsize = AtomicUsize::new(0);

    pub struct Recorder;

    // SAFETY: both methods forward to `System` with their own arguments, so its guarantees carry
    // over. `dealloc` first reads the block it frees: the block is live until the forwarded call,
    // `layout.size()` bytes long, and the test watches only blocks whose every byte is initialized:
    // a residual/copy, a fully initialized streamed codeword block, or the table tile's sole
    // fully initialized dense/sorted allocation. Opening blocks initialize their entire capacity;
    // completed P5 columns do too. The combination scratch has two u128 and two u64 fields,
    // 48 initialized bytes with no padding on these targets; its SecretBuf clears all capacity.
    // Sorted storage initializes its entire public capacity as two-u64 entries without padding.
    unsafe impl GlobalAlloc for Recorder {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: the caller's contract, forwarded unchanged.
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
            let watched = WATCHED.compare_exchange(
                block,
                ptr::null_mut(),
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
            let watched_size =
                WATCHED_SIZE.compare_exchange(layout.size(), 0, Ordering::SeqCst, Ordering::SeqCst);
            if watched.is_ok() || watched_size.is_ok() {
                // SAFETY: as stated on the impl: live, `layout.size()` bytes, initialized.
                let bytes = unsafe { std::slice::from_raw_parts(block, layout.size()) };
                NONZERO.store(bytes.iter().filter(|&&b| b != 0).count(), Ordering::SeqCst);
            }
            // SAFETY: the caller's contract, forwarded unchanged.
            unsafe { System.dealloc(block, layout) }
        }
    }
}

#[global_allocator]
static ALLOCATOR: recorder::Recorder = recorder::Recorder;

/// Runs `free` with `block` watched, and returns the nonzero bytes `block` held when freed.
fn nonzero_bytes_when_freed<T>(block: *const T, free: impl FnOnce()) -> usize {
    recorder::NONZERO.store(usize::MAX, Ordering::SeqCst);
    recorder::WATCHED.store(block.cast::<u8>().cast_mut(), Ordering::SeqCst);
    free();
    recorder::NONZERO.load(Ordering::SeqCst)
}

/// A table tile has one allocation; watch its public size without exposing private storage.
fn nonzero_bytes_when_size_freed(size: usize, free: impl FnOnce()) -> usize {
    assert!(size > 0);
    recorder::NONZERO.store(usize::MAX, Ordering::SeqCst);
    recorder::WATCHED_SIZE.store(size, Ordering::SeqCst);
    free();
    recorder::WATCHED_SIZE.store(0, Ordering::SeqCst);
    recorder::NONZERO.load(Ordering::SeqCst)
}

fn model() -> IntegerModel {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models/tiny-2-servable.d");
    let bytes = fs::read(path).expect("the conformance model");
    envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL)
        .expect("an IntegerModel")
}

#[test]
fn checkpoints_and_multiplicities_free_zeros_only() {
    streamed_codewords_are_cleared_on_abort();
    regenerated_opening_buffers_are_cleared();
    // One test, so that no other test of this binary frees memory while a block is watched.
    let model = model();
    let tokens = vec![7, 300, 11, 451, 0, 511];
    let n = u32::try_from(tokens.len()).expect("small");
    let lowering = lower(PublicModel::of(&model), n).expect("the model lowers");
    let engine = Engine::new(&lowering).expect("the tables regenerate");
    let secret = Secret::new(tokens);
    let checkpoints = lowering.checkpoints.len();
    for k in 0..checkpoints {
        let store = Checkpoints::new(&engine, &model, &secret).expect("the checkpoints");
        let residual = store.residual(k).expect("a checkpoint");
        let copy = residual.to_vec();
        let written: usize = copy
            .iter()
            .map(|v| v.to_le_bytes().iter().filter(|&&b| b != 0).count())
            .sum();
        assert!(written > 0, "c_{k} has nonzero bytes to clear");
        let freed = nonzero_bytes_when_freed(copy.as_ptr(), || drop(copy));
        assert_eq!(
            freed, written,
            "a plain copy of c_{k} frees its bytes as written"
        );
        let block = residual.as_ptr();
        let freed = nonzero_bytes_when_freed(block, || drop(store));
        assert_eq!(freed, 0, "c_{k}'s buffer frees zeros only");
    }
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let selection = select(
        &lowering.graph,
        &ranges,
        &lowering.tables,
        &GadgetPolicy::default(),
    )
    .expect("gadgets");
    let selected = tiles(PublicModel::of(&model), &lowering, &selection).expect("tiles");
    let store = Checkpoints::new(&engine, &model, &secret).expect("checkpoints");
    let source = Regenerator::new(&engine, &selection, &model, &secret, &store).expect("source");
    for bits in [1, 30] {
        let table = selection
            .tables
            .iter()
            .find(|table| table.log2_rows == bits)
            .expect("a canonical one-bit or wide table")
            .id;
        let table = Tile::Table { table: table.0 };
        assert!(selected.contains(&table));
        let tile = source
            .tile(&SequenceTile {
                tile: table,
                block: None,
                shares: Vec::new(),
            })
            .expect("multiplicities");
        assert!(tile.shares().is_empty() && tile.gadget().is_empty() && tile.lookups().len() == 0);
        let counts = tile.multiplicities().expect("a table tile");
        let mut probe = SecretBuf::filled(0, 2);
        counts.read_rows(0, &mut probe).expect("canonical prefix");
        let copy = probe.to_vec();
        let written: usize = copy
            .iter()
            .map(|v| v.to_le_bytes().iter().filter(|&&b| b != 0).count())
            .sum();
        assert!(written > 0);
        assert_eq!(
            nonzero_bytes_when_freed(copy.as_ptr(), || drop(copy)),
            written
        );
        let bytes = tile.bytes();
        if bits == 1 {
            assert_eq!(bytes, 16);
        } else {
            assert!((bytes as u64) < counts.rows() * 8);
        }
        assert_eq!(nonzero_bytes_when_size_freed(bytes, || drop(tile)), 0);
    }
}

fn streamed_codewords_are_cleared_on_abort() {
    let (k, n, block) = (8, 16, 5);
    let bytes = block * n * size_of::<Fp>();
    let layout = AdviceLayout::new(
        LayoutParams {
            log2_k: 3,
            segments: 1,
            layers: 1,
            group: 1,
            query_block: 1,
            max_context: 1,
        },
        1,
        &[TileShape {
            layer: Some(0),
            values: vec![TensorShape::Dense(DenseShape::new(
                vec![Axis::new(17 * k as u64)],
                Padding::Zero,
            ))],
        }],
    )
    .expect("layout");
    let params = CommitmentParams {
        log2_k: 3,
        log2_inv_rate: 1,
        mask_coeffs: 1,
        row_block: std::num::NonZeroUsize::new(block).expect("B"),
        commitment_id: [53; 32],
    };
    let encoder = RowEncoder::new(k, 1, 1).expect("encoder");
    let row = SecretBuf::filled(Fp::ONE, k);
    // A fully initialized plain allocation is a control: freeing it exposes nonzero bytes.
    let mut copy = vec![Fp::ZERO; block * n];
    encoder
        .encode_row(&row, &mut || Fp::ONE, &mut copy[..n])
        .expect("control codeword");
    let written: usize = copy
        .iter()
        .map(|v| {
            v.to_canonical_u64()
                .to_le_bytes()
                .iter()
                .filter(|&&b| b != 0)
                .count()
        })
        .sum();
    assert!(written > 0);
    assert_eq!(nonzero_bytes_when_size_freed(bytes, || drop(copy)), written);
    let mut source = || Fp::ONE;
    assert_eq!(
        nonzero_bytes_when_size_freed(bytes, || {
            let mut stream =
                StreamingCommitter::new(params, &layout, 0, &encoder, &mut source).expect("stream");
            stream.push_row(0, &row).expect("partial block");
            // Abort before any full-block or finish clearing: Drop must clear the entire allocation.
            drop(stream);
        }),
        0
    );
}

fn regenerated_opening_buffers_are_cleared() {
    let (k, n, b, rows, p) = (8usize, 16, 5, 17, 3);
    let layout = AdviceLayout::new(
        LayoutParams {
            log2_k: 3,
            segments: 1,
            layers: 1,
            group: 1,
            query_block: 1,
            max_context: 1,
        },
        1,
        &[TileShape {
            layer: Some(0),
            values: vec![TensorShape::Dense(DenseShape::new(
                vec![Axis::new((rows * k) as u64)],
                Padding::Zero,
            ))],
        }],
    )
    .unwrap();
    let params = CommitmentParams {
        log2_k: 3,
        log2_inv_rate: 1,
        mask_coeffs: p,
        row_block: std::num::NonZeroUsize::new(b).unwrap(),
        commitment_id: [59; 32],
    };
    let encoder = RowEncoder::new(k, 1, p).unwrap();
    assert_eq!(encoder.codeword_len(), n);
    let row = SecretBuf::filled(Fp::ONE, k);
    let mut rng = || Fp::ONE;
    let mut commit = StreamingCommitter::new(params, &layout, 0, &encoder, &mut rng).unwrap();
    for r in 0..rows as u64 {
        commit.push_row(r, &row).unwrap();
    }
    let parts = [commit.finish().unwrap()];
    let state = AdviceOpening::new(&parts).unwrap();
    let claims: Vec<_> = (0..2)
        .map(|i| {
            let mut point = vec![Fp2::from(Fp::from_u64_reduced(3)); layout.variables() as usize];
            point[0] = Fp2::from(Fp::from_u64_reduced(3 + i));
            AdviceClaim {
                pieces: vec![AdvicePoint {
                    point,
                    weight: Fp2::ONE,
                }],
            }
        })
        .collect();
    let groups = state
        .prepare(
            &claims,
            Fp2::from(Fp::from_u64_reduced(5)),
            &vec![Fp2::from(Fp::from_u64_reduced(7)); layout.row_vars() as usize],
        )
        .unwrap();
    for count in [b * k, b * p] {
        let copy = vec![Fp::ONE; count];
        assert_eq!(
            nonzero_bytes_when_size_freed(count * 8, || drop(copy)),
            count
        );
        assert_eq!(
            nonzero_bytes_when_size_freed(count * 8, || {
                let mut rng = || Fp::ONE;
                let mut pass = groups.begin(&encoder, &mut rng).unwrap();
                pass.push_row(0, &row).unwrap();
                drop(pass);
            }),
            0,
            "an aborted P4 block clears data and masks"
        );
    }
    // The actual shared combination kernel must clear its unreduced secret sums, including
    // the reference path. All 48 bytes of each Wide are initialized; the first free is watched.
    let scratch = 3 * 64 * 48;
    let copy = vec![1u64; scratch / 8];
    assert_eq!(
        nonzero_bytes_when_size_freed(scratch, || drop(copy)),
        scratch / 8
    );
    let mut rng = || Fp::ONE;
    let mut pass = groups.begin(&encoder, &mut rng).unwrap();
    assert_eq!(
        nonzero_bytes_when_size_freed(scratch, || {
            for r in 0..b as u64 {
                pass.push_row(r, &row).unwrap();
            }
        }),
        0,
        "combination scratch frees zeros, even while P4 remains live"
    );
    for r in b as u64..rows as u64 {
        pass.push_row(r, &row).unwrap();
    }
    let checked = pass.finish().unwrap();
    let copy = vec![Fp::ONE; rows];
    assert_eq!(nonzero_bytes_when_size_freed(rows * 8, || drop(copy)), rows);
    assert_eq!(
        nonzero_bytes_when_size_freed(rows * 8, || {
            let mut rng = || Fp::ONE;
            let mut pass = checked.answer(&encoder, &mut rng, &[1, 9]).unwrap();
            // Fill the complete reserved capacity before aborting, so a non-clearing mutant also
            // exposes initialized bytes only. No message has been returned yet.
            for r in 0..rows as u64 {
                pass.push_row(r, &row).unwrap();
            }
            drop(pass);
        }),
        0,
        "an unsent P5 answer clears its columns"
    );
}
