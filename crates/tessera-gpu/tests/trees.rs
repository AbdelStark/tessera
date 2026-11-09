// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The product-tree microkernel (#64) against `tessera_sumcheck::prove_forests`: the same
//! transcript, the same messages, the same leaf claim, byte for byte (RFC-0010 'Tree levels').
//! Runs only with the `gpu` feature, on a CUDA device.

#![cfg(feature = "gpu")]

use tessera_codec::Ascii;
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField};
use tessera_gpu::{Device, GpuError};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{Forest, SumcheckTranscript, prove_forests};

type Fp2 = GoldilocksExt2;
const P: u64 = 0xFFFF_FFFF_0000_0001;

fn transcript() -> Transcript {
    Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
        statement_digest: [0x65; 32],
        session_id: [0x13; 16],
    })
}

/// SplitMix64 values reduced below `p` by one subtraction.
fn leaves(n: usize, mut state: u64) -> Vec<u64> {
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

#[test]
fn forests_match_the_cpu_prover() {
    let device = Device::new(0).expect("a CUDA device with this crate's kernels");
    // (s, h): single trees, one level, tall and wide forests, and the benchmark's row height.
    for (s, h) in [
        (0, 1),
        (0, 2),
        (0, 5),
        (1, 1),
        (2, 1),
        (3, 4),
        (5, 6),
        (4, 10),
        (10, 4),
        (6, 14),
    ] {
        let values = leaves(1 << (s + h), (s * 64 + h) as u64);
        let fp: Vec<Goldilocks> = values
            .iter()
            .map(|&v| Goldilocks::from_canonical_u64(v).expect("canonical"))
            .collect();
        let mut cpu = transcript();
        let (cpu_proofs, cpu_claims) = prove_forests(&mut cpu, &[Forest::<Fp2>::new(&fp, h)])
            .expect("CPU prover")
            .into_parts();
        let mut gpu = transcript();
        let (gpu_proofs, gpu_claim) = device
            .prove_forest(&mut gpu, &device.upload(&values).expect("upload"), h)
            .expect("GPU prover");
        assert!(
            gpu_proofs == cpu_proofs,
            "messages differ at s = {s}, h = {h}"
        );
        assert!(
            gpu_claim == cpu_claims[0],
            "leaf claims differ at s = {s}, h = {h}"
        );
        // Both transcripts absorbed the same bytes in the same order.
        let next = |t: &mut Transcript| SumcheckTranscript::<Fp2>::challenge(t, "after");
        assert!(
            next(&mut gpu) == next(&mut cpu),
            "transcripts diverge at s = {s}, h = {h}"
        );
    }
}

#[test]
fn malformed_forests_are_refused() {
    let device = Device::new(0).expect("a CUDA device with this crate's kernels");
    let three = device.upload(&[1u64, 2, 3]).expect("upload");
    let four = device.upload(&[1u64, 2, 3, 4]).expect("upload");
    for (leaves, height) in [(&three, 1), (&four, 0), (&four, 3)] {
        assert!(matches!(
            device.prove_forest(&mut transcript(), leaves, height),
            Err(GpuError::Unsupported(_))
        ));
    }
}
