// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The fraction-tree microkernel (#63) against `tessera_sumcheck::prove_trees` on one LogUp
//! fraction tree: the same transcript, messages and leaf claim, byte for byte (RFC-0008
//! 'Fraction tree', RFC-0010 'Tree levels'). Runs only with the `gpu` feature, on a CUDA device.

#![cfg(feature = "gpu")]

use tessera_codec::Ascii;
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_gpu::{Device, FractionLeaves, GpuError};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{
    FractionTree, LeafNumerators, LevelClaim, SumcheckTranscript, Tree, prove_trees,
};

type Fp2 = GoldilocksExt2;

fn transcript() -> Transcript {
    Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
        statement_digest: [0x65; 32],
        session_id: [0x13; 16],
    })
}

fn fp2(a: u64, b: u64) -> Fp2 {
    Fp2::new(
        Goldilocks::from_canonical_u64(a).expect("canonical"),
        Goldilocks::from_canonical_u64(b).expect("canonical"),
    )
}

#[test]
fn fraction_trees_match_the_cpu_prover() {
    let device = Device::new(0).expect("a CUDA device with this crate's kernels");
    for height in [1usize, 2, 3, 5, 9, 14] {
        for unit in [true, false] {
            let leaves = FractionLeaves {
                key_seed: 11 * height as u64,
                value_seed: 1 << 40,
                multiplicity_seed: (!unit).then_some(1 << 50),
                alpha: fp2(0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321),
                beta: fp2(0x1111_2222_3333_4444, 0x5555_6666_7777_8888),
            };
            let count = 1u64 << height;
            let numerators: Vec<Goldilocks> =
                (0..count).filter_map(|j| leaves.numerator(j)).collect();
            let numerators = if unit {
                LeafNumerators::Unit
            } else {
                LeafNumerators::Base(&numerators)
            };
            let tree = FractionTree::new(height, numerators, |j| leaves.denominator(j as u64))
                .expect("CPU tree");
            let mut cpu = transcript();
            let (cpu_proofs, cpu_claims) = prove_trees(&mut cpu, vec![Tree::Fraction(tree)])
                .expect("CPU prover")
                .into_parts();
            let Some(LevelClaim::Fraction(cpu_claim)) = cpu_claims.first() else {
                panic!("one fraction claim");
            };
            let mut workspace = device.fraction_workspace(height, unit).expect("workspace");
            let mut gpu = transcript();
            let (gpu_proofs, gpu_claim) = device
                .prove_fraction_tree_in(&mut workspace, &mut gpu, &leaves)
                .expect("GPU prover");
            let at = format!("h = {height}, unit = {unit}");
            assert!(gpu_proofs == cpu_proofs, "messages differ at {at}");
            assert!(gpu_claim.point == cpu_claim.point, "points differ at {at}");
            assert!(
                gpu_claim.numerator == cpu_claim.numerator,
                "numerators differ at {at}"
            );
            assert!(
                gpu_claim.denominator == cpu_claim.denominator,
                "denominators differ at {at}"
            );
            let next = |t: &mut Transcript| SumcheckTranscript::<Fp2>::challenge(t, "after");
            assert!(
                next(&mut gpu) == next(&mut cpu),
                "transcripts diverge at {at}"
            );
        }
    }
}

#[test]
fn mismatched_workspaces_are_refused() {
    let device = Device::new(0).expect("a CUDA device with this crate's kernels");
    assert!(matches!(
        device.fraction_workspace(0, true),
        Err(GpuError::Unsupported(_))
    ));
    let mut workspace = device.fraction_workspace(3, true).expect("workspace");
    let table = FractionLeaves {
        key_seed: 0,
        value_seed: 0,
        multiplicity_seed: Some(0),
        alpha: fp2(1, 2),
        beta: fp2(3, 4),
    };
    assert!(matches!(
        device.prove_fraction_tree_in(&mut workspace, &mut transcript(), &table),
        Err(GpuError::Unsupported(_))
    ));
}
