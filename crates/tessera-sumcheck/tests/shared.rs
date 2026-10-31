// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0056: sharing is allocation identity, never equality of witness values. Duplicate
//! final evaluations and gamma powers stay in the original order, including front padding.

use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{
    BatchMemory, BatchShape, Computed, ComputedState, Instance, Prefix, Protocol, SumcheckError,
    Table, Term, prove_batch,
};

fn context() -> SessionContext {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::Live,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x46; 32],
        session_id: [0x61; 16],
    }
}

fn prove(instances: &[Instance<'_, F>]) -> tessera_sumcheck::SumcheckProof<F> {
    let shapes = instances
        .iter()
        .map(|i| BatchShape {
            vars: i
                .computed_vars()
                .unwrap_or_else(|| i.tables()[0].len().trailing_zeros() as usize),
            tables: i.tables().len(),
            terms: i.terms().map_or(0, <[Term<F>]>::len),
            evaluations: i.evaluations(),
            computed: i.computed_vars().is_some(),
        })
        .collect::<Vec<_>>();
    let mut memory = None;
    let planning =
        allocation_counter::measure(|| memory = Some(BatchMemory::new::<F>(3, &shapes).unwrap()));
    assert_eq!(planning.count_total, 0);
    let memory = memory.unwrap();
    let mut transcript = Transcript::new(&context());
    let mut result = None;
    let measured = allocation_counter::measure(|| {
        result = Some(
            prove_batch(
                &mut transcript,
                Protocol::One,
                Prefix::Stage(2),
                3,
                instances,
            )
            .unwrap(),
        );
    });
    assert!(
        measured.bytes_max <= memory.peak_bytes(),
        "batch {measured:?} {memory:?}"
    );
    assert!(
        u64::try_from(measured.bytes_current).unwrap() <= memory.output_bytes(),
        "output {measured:?} {memory:?}"
    );
    result.unwrap().into_proof()
}

#[test]
fn invalid_computed_geometry_is_rejected_before_start_or_transcript_movement() {
    // Adding an optional computed representation must retain the borrowed instance's
    // existing ability to cross worker threads.
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Instance<'static, F>>();
    struct Geometry(usize, usize);
    impl Computed<F> for Geometry {
        fn vars(&self) -> usize {
            self.0
        }
        fn degree(&self) -> usize {
            self.1
        }
        fn evaluations(&self) -> usize {
            0
        }
        fn start(&self) -> Box<dyn ComputedState<F> + '_> {
            panic!("invalid geometry must never start")
        }
    }
    let values = [G::from_u64_reduced(1), G::from_u64_reduced(2)];
    let valid = Instance::new(vec![Table::base(&values)]);
    for (geometry, error) in [
        (
            Geometry(63, 1),
            SumcheckError::ComputedVariables {
                instance: 0,
                vars: 63,
            },
        ),
        (Geometry(0, 0), SumcheckError::Degree { degree: 0, max: 3 }),
        (Geometry(0, 4), SumcheckError::Degree { degree: 4, max: 3 }),
    ] {
        let mut transcript = Transcript::new(&context());
        let result = prove_batch(
            &mut transcript,
            Protocol::One,
            Prefix::Stage(2),
            3,
            &[Instance::computed(&geometry)],
        );
        assert_eq!(result.unwrap_err(), error);
        let after = prove_batch(
            &mut transcript,
            Protocol::One,
            Prefix::Stage(2),
            3,
            std::slice::from_ref(&valid),
        )
        .unwrap()
        .into_proof();
        assert_eq!(after, prove(std::slice::from_ref(&valid)));
    }
}

#[test]
fn final_evaluations_of_unread_columns_are_computed_once() {
    use tessera_field::Field;
    let a = vec![G::ONE; 1 << 16];
    let c: Vec<_> = (0..1 << 16).map(G::from_u64_reduced).collect();
    let copies = vec![c.clone(); 16];
    let make = |c, i| {
        Instance::sum_of_products(
            vec![Table::base(&a), Table::base(c)],
            vec![Term::new(F::from(G::from_u64_reduced(i + 1)), vec![0])],
        )
    };
    let shared: Vec<_> = (0..16).map(|i| make(&c, i)).collect();
    let separate: Vec<_> = copies
        .iter()
        .enumerate()
        .map(|(i, c)| make(c, i as u64))
        .collect();
    let single = allocation_counter::measure(|| {
        prove(&shared[..1]);
    });
    let mut result = None;
    let repeated = allocation_counter::measure(|| {
        result = Some(
            prove_batch(
                &mut Transcript::new(&context()),
                Protocol::One,
                Prefix::Stage(2),
                3,
                &shared,
            )
            .unwrap(),
        );
    });
    let result = result.unwrap();
    // c[x] = x, so its multilinear extension is Σ_j 2^j r_j. Pin the final
    // opening against this identity even if both engine paths forgot an unread column.
    let expected = result
        .point()
        .iter()
        .enumerate()
        .fold(F::ZERO, |sum, (j, &r)| {
            sum + r * F::from(G::from_u64_reduced(1 << j))
        });
    for pair in result.proof().final_evals.chunks_exact(2) {
        assert_eq!(pair, [F::ONE, expected]);
    }
    assert_eq!(result.into_proof(), prove(&separate));
    assert!(repeated.bytes_total < single.bytes_total + 128 * 1024);
    assert!(repeated.bytes_max < single.bytes_max + 128 * 1024);
}

#[test]
fn thirty_identical_instances_hold_one_table_set_and_send_every_evaluation() {
    let n = 1 << 16;
    let source: Vec<_> = (0..n).map(|x| G::from_u64_reduced(x as u64)).collect();
    let separate = vec![source.clone(); 30];
    let make =
        |t| Instance::sum_of_products(vec![Table::base(t)], vec![Term::new(F::ONE, vec![0, 0])]);
    let shared: Vec<_> = (0..30).map(|_| make(&source)).collect();
    let separate: Vec<_> = separate.iter().map(|v| make(v)).collect();
    let single = allocation_counter::measure(|| {
        prove(&shared[..1]);
    });
    let mut result = None;
    let shared_peak = allocation_counter::measure(|| {
        result = Some(prove(&shared));
    });
    assert_eq!(result.unwrap(), prove(&separate));
    // Half of one Fp2 table is 512 KiB; metadata for 30 instances is small. A second
    // table allocation, even if freed before peak, violates the total allocation bound.
    assert!(shared_peak.bytes_max < single.bytes_max + 64 * 1024);
    assert!(shared_peak.bytes_total < single.bytes_total + 64 * 1024);
}

#[test]
fn distinct_products_share_operands_but_keep_their_terms_padding_and_public_flags() {
    let a: Vec<_> = (0..1 << 14)
        .map(|x| G::from_u64_reduced(x as u64))
        .collect();
    let b: Vec<_> = (0..1 << 14)
        .map(|x| G::from_u64_reduced(31 + x as u64))
        .collect();
    let copies_a = vec![a.clone(); 8];
    let copies_b = vec![b.clone(); 8];
    let build = |a, b, i| {
        Instance::sum_of_products(
            vec![Table::base(a), Table::base(b)],
            vec![Term::new(
                F::from(G::from_u64_reduced(1 + i)),
                if i % 2 == 0 {
                    vec![0, 0, 1]
                } else {
                    vec![0, 1, 1]
                },
            )],
        )
    };
    let mut shared: Vec<_> = (0..8).map(|i| build(&a, &b, i)).collect();
    let mut separate: Vec<_> = (0..8)
        .map(|i| build(&copies_a[i], &copies_b[i], i as u64))
        .collect();
    let tiny = [G::ONE, G::ZERO];
    // Same buffer twice, but only one instance sends its value.
    for instances in [&mut shared, &mut separate] {
        instances.push(Instance::new(vec![Table::base(&tiny)]));
        instances.push(Instance::new(vec![Table::base(&tiny).public()]));
    }
    let mut proof = None;
    let info = allocation_counter::measure(|| {
        proof = Some(prove(&shared));
    });
    let proof = proof.unwrap();
    assert_eq!(proof.final_evals.len(), 17);
    assert_eq!(proof, prove(&separate));
    // Two shared folded tables plus bounded kernel scratch, rather than sixteen tables.
    assert!(info.bytes_max < 1024 * 1024);
}

#[test]
fn batch_storage_covers_literal_terms_front_padding_and_unread_columns() {
    for vars in [0usize, 1, 3, 10, 11] {
        let base = (0..1usize << vars)
            .map(|i| G::from_u64_reduced(i as u64 + 1))
            .collect::<Vec<_>>();
        let ext = base.iter().copied().map(F::from).collect::<Vec<_>>();
        let short = [G::ONE, G::ONE];
        for terms in [1usize, 3, 32] {
            let instance = Instance::sum_of_products(
                vec![
                    Table::base(&base),
                    Table::ext(&ext),
                    Table::base(&base).public(),
                    Table::ext(&ext),
                ],
                (0..terms)
                    .map(|i| {
                        Term::new(
                            F::from(G::from_u64_reduced(i as u64 + 1)),
                            if i % 2 == 0 {
                                vec![0, 1]
                            } else {
                                vec![1, 2, 2]
                            },
                        )
                    })
                    .collect(),
            );
            let product = Instance::new(vec![Table::base(&short)]);
            let _ = prove(&[instance.clone(), product, instance]);
        }
    }
}

#[test]
fn batch_storage_covers_computed_wrappers_without_delegated_heap_payload() {
    // The zero polynomial has a zero-sized clearing state. No delegated heap is
    // excluded from the actual measured batch in this fixture.
    struct Zero(usize);
    struct ZeroState;
    impl Computed<F> for Zero {
        fn vars(&self) -> usize {
            self.0
        }
        fn degree(&self) -> usize {
            1
        }
        fn evaluations(&self) -> usize {
            1
        }
        fn start(&self) -> Box<dyn ComputedState<F> + '_> {
            Box::new(ZeroState)
        }
    }
    impl ComputedState<F> for ZeroState {
        fn round(&self, _: &[F], values: &mut [F]) {
            values.fill(F::ZERO);
        }
        fn bind(&mut self, _: F) {}
        fn evaluations(&self, out: &mut Vec<F>) {
            out.push(F::ZERO);
        }
    }
    let other = [G::ONE, G::ONE];
    for vars in [0usize, 1, 5, 12] {
        let zero = Zero(vars);
        let values = vec![G::ZERO; 1usize << vars];
        let computed = Instance::computed(&zero);
        let literal = Instance::new(vec![Table::base(&values)]);
        let extra = Instance::new(vec![Table::base(&other)]);
        assert_eq!(
            prove(&[computed.clone(), extra.clone(), computed]),
            prove(&[literal.clone(), extra, literal])
        );
    }
}

#[test]
fn batch_storage_covers_one_large_product_binding() {
    let values = (0..1u64 << 16).map(G::from_u64_reduced).collect::<Vec<_>>();
    let _ = prove(&[Instance::new(vec![Table::base(&values)])]);
}

#[test]
fn batch_storage_covers_sparse_large_term_plan_construction() {
    // Counts are tables*terms even though each term reads only two/three factors.
    // Singleton tables isolate constructor metadata from folded-table payload.
    let values = (1..=512u64).map(G::from_u64_reduced).collect::<Vec<_>>();
    let tables = values
        .iter()
        .map(|v| Table::base(std::slice::from_ref(v)))
        .collect();
    let terms = (0..8192u64)
        .map(|i| {
            Term::new(
                F::from(G::from_u64_reduced(i + 1)),
                if i % 2 == 0 {
                    vec![0, 1]
                } else {
                    vec![1, 2, 2]
                },
            )
        })
        .collect();
    let _ = prove(&[Instance::sum_of_products(tables, terms)]);
}
