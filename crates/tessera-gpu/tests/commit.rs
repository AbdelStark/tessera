// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The encode-and-hash microkernel (#55) against the CPU references it must match bit for bit:
//! `tessera_pcs::RowEncoder` at rate 1/2 (RFC-0011 'Encoding') and `tessera_hash::ColumnChains`
//! (RFC-0004 'Column digests'). Runs only with the `gpu` feature, on a CUDA device.

#![cfg(feature = "gpu")]

use std::num::NonZeroUsize;

use tessera_field::{Goldilocks, PrimeField};
use tessera_gpu::{Device, GpuError};
use tessera_hash::ColumnChains;
use tessera_pcs::RowEncoder;

const P: u64 = 0xFFFF_FFFF_0000_0001;

/// SplitMix64 values reduced below `p` by one subtraction, for reproducible inputs.
fn inputs(n: usize, mut state: u64) -> Vec<u64> {
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            if z >= P { z - P } else { z }
        })
        .collect()
}

fn field(values: &[u64]) -> Vec<Goldilocks> {
    values
        .iter()
        .map(|&v| Goldilocks::from_canonical_u64(v).expect("canonical"))
        .collect()
}

/// The CPU codewords of `rows` with `masks` drawn row after row.
fn cpu_encode(log_k: u32, mask_coeffs: usize, rows: &[u64], masks: &[u64]) -> Vec<u64> {
    let encoder = RowEncoder::new(1 << log_k, 1, mask_coeffs).expect("encoder");
    let mut draws = field(masks).into_iter();
    let mut codewords = vec![Goldilocks::from_canonical_u64(0).expect("zero"); 2 * rows.len()];
    encoder
        .encode_rows(
            &field(rows),
            &mut || draws.next().expect("enough masks"),
            &mut codewords,
        )
        .expect("encode");
    codewords.iter().map(PrimeField::to_canonical_u64).collect()
}

fn gpu_encode(
    device: &Device,
    log_k: u32,
    mask_coeffs: usize,
    rows: &[u64],
    masks: &[u64],
) -> Vec<u64> {
    let encoder = device.encoder(log_k, mask_coeffs).expect("encoder");
    let mut rows = device.upload(rows).expect("upload");
    let masks = if masks.is_empty() {
        device.upload(&[0u64])
    } else {
        device.upload(masks)
    }
    .expect("upload");
    let mut codewords = device.zeroed::<u64>(2 * rows.len()).expect("alloc");
    device
        .encode_rows(&encoder, &mut rows, &masks, &mut codewords)
        .expect("encode");
    device.download(&codewords).expect("download")
}

#[test]
fn encoding_matches_the_cpu() {
    let device = Device::new(0).expect("a CUDA device with this crate's kernels");
    // Single-tile rows (log_k ≤ 12), one strided pass each way (13..=22), and two (24), with no
    // mask, one coefficient, a production-like count and P = k.
    for (log_k, count, mask_coeffs) in [
        (1, 3, 0),
        (1, 2, 2),
        (2, 3, 1),
        (5, 4, 32),
        (8, 3, 200),
        (12, 2, 200),
        (13, 2, 200),
        (14, 3, 1),
        (17, 2, 160),
        (22, 1, 200),
        (24, 1, 200),
    ] {
        let k = 1usize << log_k;
        let rows = inputs(count * k, u64::from(log_k));
        let masks = inputs(count * mask_coeffs, 1000 + u64::from(log_k));
        assert_eq!(
            gpu_encode(&device, log_k, mask_coeffs, &rows, &masks),
            cpu_encode(log_k, mask_coeffs, &rows, &masks),
            "log_k = {log_k}, rows = {count}, P = {mask_coeffs}"
        );
    }
}

#[test]
fn rate4_weight_encoding_matches_the_cpu() {
    let device = Device::new(0).expect("a CUDA device with this crate's kernels");
    // RFC-0040 weight rows: rate 1/4, no masks; one tile, one strided pass each way, and k = 2^20.
    for (log_k, count) in [(1, 3), (4, 2), (12, 2), (13, 2), (17, 2), (20, 2)] {
        let k = 1usize << log_k;
        let rows = inputs(count * k, 77 + u64::from(log_k));
        let encoder = RowEncoder::new(k, 2, 0).expect("encoder");
        let zero = Goldilocks::from_canonical_u64(0).expect("zero");
        let mut cpu = vec![zero; 4 * rows.len()];
        encoder
            .encode_rows(&field(&rows), &mut || zero, &mut cpu)
            .expect("encode");
        let [even, odd] = device.rate4_encoders(log_k).expect("encoders");
        let none = device.upload(&[0u64]).expect("upload");
        let mut halves = Vec::new();
        for e in [&even, &odd] {
            let mut scratch = device.upload(&rows).expect("upload");
            let mut half = device.zeroed::<u64>(2 * rows.len()).expect("alloc");
            device
                .encode_rows(e, &mut scratch, &none, &mut half)
                .expect("encode");
            halves.push(half);
        }
        let mut out = device.zeroed::<u64>(4 * rows.len()).expect("alloc");
        device
            .interleave(&halves[0], &halves[1], &mut out)
            .expect("interleave");
        assert_eq!(
            device.download(&out).expect("download"),
            cpu.iter()
                .map(PrimeField::to_canonical_u64)
                .collect::<Vec<_>>(),
            "log_k = {log_k}, rows = {count}"
        );
    }
}

#[test]
fn chains_match_the_cpu() {
    let device = Device::new(0).expect("a CUDA device with this crate's kernels");
    let id = [0xA5u8; 32];
    // Column counts off a block boundary; full blocks, a short last block, and the 124-row limit.
    for (columns, block, rows) in [
        (1usize, 32usize, 1usize),
        (1000, 32, 70),
        (4099, 32, 64),
        (257, 124, 130),
    ] {
        let symbols = inputs(rows * columns, columns as u64);
        let mut cpu = ColumnChains::new(
            &id,
            NonZeroUsize::new(columns).expect("columns"),
            NonZeroUsize::new(block).expect("block"),
        );
        let mut gpu = device.column_chains(&id, columns).expect("init");
        for start in (0..rows).step_by(block) {
            let count = block.min(rows - start);
            let part = &symbols[start * columns..(start + count) * columns];
            cpu.absorb_block(&field(part)).expect("cpu absorb");
            let buffer = device.upload(part).expect("upload");
            device
                .absorb_block(&mut gpu, &buffer, count)
                .expect("gpu absorb");
        }
        assert_eq!(
            device.download_chains(&gpu).expect("download"),
            cpu.finish(),
            "columns = {columns}, B = {block}, rows = {rows}"
        );
    }
}

#[test]
fn encode_then_hash_matches_the_cpu() {
    let device = Device::new(0).expect("a CUDA device with this crate's kernels");
    let (log_k, count, mask_coeffs) = (14u32, 33usize, 200usize);
    let k = 1usize << log_k;
    let rows = inputs(count * k, 7);
    let masks = inputs(count * mask_coeffs, 8);
    let codewords = gpu_encode(&device, log_k, mask_coeffs, &rows, &masks);
    let id = [3u8; 32];
    let n = NonZeroUsize::new(2 * k).expect("n");
    let mut cpu = ColumnChains::new(&id, n, NonZeroUsize::new(32).expect("B"));
    let expected = cpu_encode(log_k, mask_coeffs, &rows, &masks);
    let mut gpu = device.column_chains(&id, 2 * k).expect("init");
    for start in (0..count).step_by(32) {
        let c = 32.min(count - start);
        let part = &codewords[start * 2 * k..(start + c) * 2 * k];
        cpu.absorb_block(&field(&expected[start * 2 * k..(start + c) * 2 * k]))
            .expect("cpu absorb");
        device
            .absorb_block(&mut gpu, &device.upload(part).expect("upload"), c)
            .expect("gpu absorb");
    }
    assert_eq!(
        device.download_chains(&gpu).expect("download"),
        cpu.finish()
    );
}

#[test]
fn malformed_shapes_are_refused() {
    let device = Device::new(0).expect("a CUDA device with this crate's kernels");
    assert!(matches!(
        device.encoder(0, 0),
        Err(GpuError::Unsupported(_))
    ));
    assert!(matches!(
        device.encoder(27, 0),
        Err(GpuError::Unsupported(_))
    ));
    assert!(matches!(
        device.encoder(3, 9),
        Err(GpuError::Unsupported(_))
    ));
    let encoder = device.encoder(3, 2).expect("encoder");
    let masks = device.upload(&[1u64; 4]).expect("upload");
    // A partial row.
    let mut rows = device.upload(&[1u64; 12]).expect("upload");
    let mut out = device.zeroed::<u64>(24).expect("alloc");
    assert!(matches!(
        device.encode_rows(&encoder, &mut rows, &masks, &mut out),
        Err(GpuError::Unsupported(_))
    ));
    // Too few masks, then a codeword buffer of the wrong length.
    let mut rows = device.upload(&[1u64; 16]).expect("upload");
    let short = device.upload(&[1u64; 3]).expect("upload");
    assert!(matches!(
        device.encode_rows(&encoder, &mut rows, &short, &mut out),
        Err(GpuError::BufferLength {
            operand: "masks",
            expected: 4,
            found: 3
        })
    ));
    let mut wrong = device.zeroed::<u64>(31).expect("alloc");
    assert!(matches!(
        device.encode_rows(&encoder, &mut rows, &masks, &mut wrong),
        Err(GpuError::BufferLength {
            operand: "codewords",
            expected: 32,
            found: 31
        })
    ));
    let mut chains = device.column_chains(&[0; 32], 8).expect("init");
    let block = device.upload(&[1u64; 8 * 125]).expect("upload");
    assert!(matches!(
        device.absorb_block(&mut chains, &block, 0),
        Err(GpuError::Unsupported(_))
    ));
    assert!(matches!(
        device.absorb_block(&mut chains, &block, 125),
        Err(GpuError::Unsupported(_))
    ));
    let small = device.upload(&[1u64; 15]).expect("upload");
    assert!(matches!(
        device.absorb_block(&mut chains, &small, 2),
        Err(GpuError::BufferLength {
            operand: "codewords",
            ..
        })
    ));
}
