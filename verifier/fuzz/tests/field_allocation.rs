// SPDX-License-Identifier: Apache-2.0 OR MIT

//! NTT scratch reservations are fallible and leave inputs unchanged on failure (SP-8).
use cap::Cap;
use std::alloc::System;
use std::fmt::Debug;
use tv_field::{
    Fp, Fp2, NttError, NttValue, coset_intt, coset_ntt, coset_ntt_with_cancel, intt,
    intt_with_cancel, ntt,
};

#[global_allocator]
static ALLOCATOR: Cap<System> = Cap::new(System, usize::MAX);

type Transform<T> = fn(&mut [T]) -> Result<(), NttError>;

fn check<T: NttValue + Debug + PartialEq>(values: &[T]) {
    let transforms: [Transform<T>; 4] = [
        ntt,
        intt,
        |v| coset_ntt(v, Fp::SEVEN),
        |v| coset_intt(v, Fp::SEVEN),
    ];
    // Invalid arguments take precedence over a failed scratch reservation.
    let mut invalid = values[..3].to_vec();
    let mut zero_shift = values.to_vec();
    let before = ALLOCATOR.total_allocated();
    ALLOCATOR.set_limit(ALLOCATOR.allocated()).unwrap();
    let invalid_results = [
        ntt(&mut invalid),
        intt(&mut invalid),
        coset_ntt(&mut invalid, Fp::ZERO),
        coset_intt(&mut invalid, Fp::ZERO),
    ];
    let zero_results = [
        coset_ntt(&mut zero_shift, Fp::ZERO),
        coset_intt(&mut zero_shift, Fp::ZERO),
    ];
    ALLOCATOR.set_limit(usize::MAX).unwrap();
    assert_eq!(ALLOCATOR.total_allocated() - before, 0);
    assert_eq!(
        invalid_results,
        [Err(NttError::NotPowerOfTwo { len: 3 }); 4]
    );
    assert_eq!(zero_results, [Err(NttError::ZeroShift); 2]);
    assert_eq!(invalid, values[..3]);
    assert_eq!(zero_shift, values);

    for inverse in [false, true] {
        let mut working = values.to_vec();
        let mut polled = false;
        let before = ALLOCATOR.total_allocated();
        ALLOCATOR.set_limit(ALLOCATOR.allocated()).unwrap();
        let mut cancel = || {
            polled = true;
            true
        };
        let result = if inverse {
            intt_with_cancel(&mut working, &mut cancel)
        } else {
            coset_ntt_with_cancel(&mut working, Fp::SEVEN, &mut cancel)
        };
        ALLOCATOR.set_limit(usize::MAX).unwrap();
        assert_eq!(ALLOCATOR.total_allocated() - before, 0);
        assert_eq!(result, Err(NttError::Cancelled));
        assert!(polled);
        assert_eq!(working, values);
    }

    let scratch = values.len() / 2 * std::mem::size_of::<Fp>();
    for (index, transform) in transforms.into_iter().enumerate() {
        for budget in [0, scratch - 1, scratch] {
            let mut working = values.to_vec();
            let before = ALLOCATOR.total_allocated();
            ALLOCATOR.set_limit(ALLOCATOR.allocated() + budget).unwrap();
            let result = transform(&mut working);
            ALLOCATOR.set_limit(usize::MAX).unwrap();
            let allocated = ALLOCATOR.total_allocated() - before;
            if budget < scratch {
                assert_eq!(result, Err(NttError::Memory), "transform {index}");
                assert_eq!(working, values, "transform {index} modified a failed input");
                assert_eq!(allocated, 0);
            } else {
                assert_eq!(result, Ok(()));
                assert_eq!(allocated, scratch);
            }
        }
        let mut singleton = [values[0]];
        ALLOCATOR.set_limit(ALLOCATOR.allocated()).unwrap();
        let result = transform(&mut singleton);
        ALLOCATOR.set_limit(usize::MAX).unwrap();
        assert_eq!(result, Ok(()));
        assert_eq!(singleton, [values[0]]);
    }
}

fn main() {
    let base: Vec<_> = (1..=8).map(Fp::reduce).collect();
    check(&base);
    let extension: Vec<_> = base.iter().map(|&x| Fp2::new(x, x + Fp::ONE)).collect();
    check(&extension);
    println!(
        "NTT scratch limits: base/extension fields, all four transforms, unchanged failures: ok"
    );
}
