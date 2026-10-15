// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The template kernel against the host reference. Runs only with the `gpu` feature, on a
//! machine with a CUDA device: the GPU runner (#38).

#![cfg(feature = "gpu")]

use tessera_gpu::{Device, GpuError};

/// SplitMix64, for reproducible inputs without a dependency.
fn inputs(n: usize, mut state: u64) -> Vec<u64> {
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        })
        .collect()
}

#[test]
fn vector_add_matches_the_host() {
    let device = Device::new(0).expect("a CUDA device with this crate's kernels");
    // Sizes around a block boundary, and one larger than a single grid pass.
    for n in [0, 1, 1023, 1024, 1025, (1 << 20) + 7] {
        let (a, b) = (inputs(n, 1), inputs(n, 2));
        let sum = device
            .vector_add(
                &device.upload(&a).expect("upload"),
                &device.upload(&b).expect("upload"),
            )
            .expect("launch");
        let expected: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x.wrapping_add(*y)).collect();
        assert_eq!(
            device.download(&sum).expect("download"),
            expected,
            "n = {n}"
        );
    }
}

#[test]
fn mismatched_lengths_are_refused() {
    let device = Device::new(0).expect("a CUDA device with this crate's kernels");
    let (a, b) = (
        device.upload(&[1u64, 2]).expect("upload"),
        device.upload(&[1u64]).expect("upload"),
    );
    assert!(matches!(
        device.vector_add(&a, &b),
        Err(GpuError::LengthMismatch { left: 2, right: 1 })
    ));
}
