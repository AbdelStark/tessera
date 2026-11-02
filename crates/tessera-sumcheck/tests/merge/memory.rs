// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054/0018: public sizing against actual constructor/prover allocations.

use super::super::*;
use tessera_sumcheck::{MAX_MERGE_VARS, MergeMemory, SumcheckTranscript, WindowError};

// No transcript allocations enter an engine allocation measurement. Real
// independent-transcript acceptance is exercised separately in the parent tests.
struct NoAllocationTranscript;
impl SumcheckTranscript<Fp2> for NoAllocationTranscript {
    fn absorb_message(&mut self, _: &str, _: &[Fp2]) {}
    fn challenge(&mut self, _: &str) -> Fp2 {
        Fp2::from(Fp::from_u64_reduced(17))
    }
}

fn check(polys: &[Poly], merge: &Merge<Fp2>) {
    let indices: Vec<Vec<_>> = merge
        .claims()
        .iter()
        .map(|group| group.iter().map(|c| c.polynomial).collect())
        .collect();
    // Source blocks are caller-owned and additional to the engine's bound.
    let blocks: Vec<Vec<_>> = polys
        .iter()
        .map(|p| {
            (0..1u64 << p.shape.group_vars.len())
                .map(|bits| p.block(bits))
                .collect()
        })
        .collect();
    for cut in 0..=largest_table_rounds(merge.polynomials()) {
        let mut memory = None;
        let sizing = allocation_counter::measure(|| {
            memory = Some(MergeMemory::for_two_pass::<Fp2>(
                merge.polynomials(),
                &indices,
                cut,
            ));
        });
        assert_eq!(sizing.count_total, 0, "planning allocates at cut {cut}");
        let memory = memory.unwrap().unwrap();
        let mut kept = None;
        let construction = allocation_counter::measure(|| {
            let mut tables = MergeTables::new(merge.polynomials(), indices.len(), cut).unwrap();
            for h in 0..indices.len() {
                tables
                    .add_group(h, &merge.claims()[h], merge.gammas()[h], |i, bits| {
                        blocks[i].get(bits as usize)
                    })
                    .unwrap();
            }
            kept = Some(tables);
        });
        assert!(
            construction.bytes_max <= memory.construction_peak_bytes(),
            "cut {cut}: constructor {construction:?} > {memory:?}"
        );
        assert!(
            construction.bytes_current <= memory.table_bytes() as i64,
            "cut {cut}: retained {construction:?} > {memory:?}"
        );
        let mut proof = None;
        let proving = allocation_counter::measure(|| {
            proof = Some(
                prove_merge_two_pass(
                    &mut NoAllocationTranscript,
                    merge,
                    kept.as_ref().unwrap(),
                    |i| Secret::new(polys[i].values.clone()),
                )
                .unwrap(),
            );
        });
        let source = polys.iter().map(|p| p.values.len()).max().unwrap() * size_of::<Fp>();
        assert!(
            proving.bytes_max <= memory.proving_peak_bytes() + source as u64,
            "cut {cut}: prover {proving:?} > {memory:?} + source {source}"
        );
        assert!(
            proving.bytes_current <= memory.output_bytes() as i64,
            "cut {cut}: output {proving:?} > {memory:?}"
        );
        assert!(proof.is_some());
    }
}

#[test]
fn public_merge_memory_covers_mixed_shared_and_distinct_classes() {
    let mut rng = SplitMix64::new(0x671);
    let (polys, merge) = fixture(&mut rng);
    check(&polys, &merge);
    // Same public claim indices, deliberately distinct public points. A sizing
    // function must not assume the point-class coincidences in the first fixture.
    let distinct = merge
        .claims()
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|c| polys[c.polynomial].claim(&mut rng, c.polynomial, c.group_bits))
                .collect()
        })
        .collect();
    let merge = Merge::new(
        merge.polynomials().to_vec(),
        distinct,
        merge.gammas().to_vec(),
        merge.beta(),
    )
    .unwrap();
    check(&polys, &merge);
}

#[test]
fn public_merge_memory_covers_zero_variables_and_empty_groups() {
    let mut rng = SplitMix64::new(0x6710);
    let polys = vec![Poly::new(&mut rng, 0, vec![])];
    for claims in [
        vec![],
        vec![vec![], vec![]],
        vec![vec![polys[0].claim(&mut rng, 0, 0)]],
    ] {
        let gammas = rng.fp2s(claims.len());
        let merge = Merge::new(vec![polys[0].shape.clone()], claims, gammas, rng.fp2()).unwrap();
        check(&polys, &merge);
    }
}

#[test]
fn public_merge_memory_covers_large_combined_and_compact_states() {
    let mut rng = SplitMix64::new(0x6711);
    let polys: Vec<_> = (0..12).map(|_| Poly::new(&mut rng, 12, vec![11])).collect();
    let claims: Vec<Vec<_>> = (0..2)
        .map(|bits| {
            polys
                .iter()
                .enumerate()
                .map(|(i, p)| p.claim(&mut rng, i, bits))
                .collect()
        })
        .collect();
    let merge = Merge::new(
        polys.iter().map(|p| p.shape.clone()).collect(),
        claims,
        rng.fp2s(2),
        rng.fp2(),
    )
    .unwrap();
    check(&polys, &merge);
}

#[test]
fn public_merge_memory_rejects_invalid_geometry_before_allocation() {
    let shape = |vars, group_vars| MergePolynomial { vars, group_vars };
    for polynomials in [
        vec![],
        vec![shape(MAX_MERGE_VARS + 1, vec![])],
        vec![shape(2, vec![1, 1])],
        vec![shape(2, vec![1, 0])],
        vec![shape(2, vec![2])],
        vec![shape(0, vec![0])],
    ] {
        let mut result = None;
        let measured = allocation_counter::measure(|| {
            result = Some(MergeMemory::for_two_pass::<Fp2>(&polynomials, &[], 0));
        });
        assert_eq!(measured.count_total, 0);
        assert_eq!(
            result.unwrap(),
            Err(WindowError::Shape("invalid merge polynomial geometry"))
        );
    }
    let polynomials = [shape(2, vec![1])];
    assert_eq!(
        MergeMemory::for_two_pass::<Fp2>(&polynomials, &[], 2),
        Err(WindowError::Shape("merge table cut exceeds free variables"))
    );
    assert_eq!(
        MergeMemory::for_two_pass::<Fp2>(&polynomials, &[vec![1]], 0),
        Err(WindowError::Shape("unknown merge claim polynomial"))
    );
}

#[test]
fn public_merge_memory_covers_front_padded_constants_and_fully_fixed_polynomials() {
    let mut rng = SplitMix64::new(0x6712);
    for group_vars in [vec![3], vec![0, 1, 2, 3]] {
        let polys = vec![
            Poly::new(&mut rng, 0, vec![]),
            Poly::new(&mut rng, 4, group_vars),
        ];
        let claims = vec![vec![
            polys[0].claim(&mut rng, 0, 0),
            polys[1].claim(&mut rng, 1, 1),
        ]];
        let merge = Merge::new(
            polys.iter().map(|p| p.shape.clone()).collect(),
            claims,
            rng.fp2s(1),
            rng.fp2(),
        )
        .unwrap();
        check(&polys, &merge);
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn public_merge_memory_refuses_aggregate_overflow_without_materializing_tables() {
    // 2^20 distinct permitted classes * 2^40 entries * 16 bytes = 2^64.
    // Only the 8 MiB public claim-index list exists; no witness/table is loaded.
    let polynomials = [MergePolynomial {
        vars: 40,
        group_vars: vec![],
    }];
    let claims = [vec![0; 1 << 20]];
    let mut result = None;
    let measured = allocation_counter::measure(|| {
        result = Some(MergeMemory::for_two_pass::<Fp2>(&polynomials, &claims, 0));
    });
    assert_eq!(measured.count_total, 0);
    assert_eq!(result.unwrap(), Err(WindowError::Allocation));
}

#[test]
fn public_merge_memory_covers_full_cut_without_group_variables() {
    let mut rng = SplitMix64::new(0x6713);
    let polys = vec![
        Poly::new(&mut rng, 0, vec![]),
        Poly::new(&mut rng, 2, vec![]),
        Poly::new(&mut rng, 5, vec![]),
    ];
    let claims = (0..3)
        .map(|h| {
            if h == 1 {
                Vec::new()
            } else {
                polys
                    .iter()
                    .enumerate()
                    .map(|(i, p)| p.claim(&mut rng, i, 0))
                    .collect()
            }
        })
        .collect();
    let merge = Merge::new(
        polys.iter().map(|p| p.shape.clone()).collect(),
        claims,
        rng.fp2s(3),
        rng.fp2(),
    )
    .unwrap();
    assert_eq!(largest_table_rounds(merge.polynomials()), merge.rounds());
    check(&polys, &merge);
    let session = Session::random(&mut rng);
    let refs: Vec<_> = polys.iter().map(|p| &p.table).collect();
    let honest = prove_merge(&mut session.transcript(), &merge, &refs).unwrap();
    assert!(verify(&session, &merge, &honest));
    for cut in 0..=merge.rounds() {
        let kept = kept_tables(&polys, &merge, cut);
        let actual = prove_merge_two_pass(&mut session.transcript(), &merge, &kept, |i| {
            Secret::new(polys[i].values.clone())
        })
        .unwrap();
        assert_eq!(actual.proof(), honest.proof(), "cut {cut}");
        assert_eq!(actual.point(), honest.point(), "cut {cut}");
        for (i, p) in polys.iter().enumerate() {
            assert_eq!(
                actual.proof().final_evals[i],
                p.at(actual.instance_point(i).unwrap())
            );
        }
        assert!(verify(&session, &merge, &actual));
    }
}
