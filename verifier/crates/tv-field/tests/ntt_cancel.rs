// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Explicit cancellation keeps validation precedence and honest NTT results (RFC-0019, SP-8).
use tv_field::{Fp, Fp2, NttError, coset_ntt, coset_ntt_with_cancel, intt, intt_with_cancel};

#[test]
fn uncancelled_transforms_equal_the_conformance_checked_interfaces() {
    for log_n in 0..=12 {
        let values: Vec<_> = (0..1 << log_n)
            .map(|i| Fp2::new(Fp::reduce(i), Fp::reduce(i + 7)))
            .collect();
        let mut expected = values.clone();
        let mut actual = values.clone();
        intt(&mut expected).unwrap();
        intt_with_cancel(&mut actual, || false).unwrap();
        assert_eq!(actual, expected);
        coset_ntt(&mut expected, Fp::SEVEN).unwrap();
        coset_ntt_with_cancel(&mut actual, Fp::SEVEN, || false).unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn cancellation_before_work_leaves_values_unchanged_and_invalid_arguments_do_not_poll() {
    for inverse in [false, true] {
        let mut values = vec![Fp::ONE; 8];
        let result = if inverse {
            intt_with_cancel(&mut values, || true)
        } else {
            coset_ntt_with_cancel(&mut values, Fp::SEVEN, || true)
        };
        assert_eq!(result, Err(NttError::Cancelled));
        assert_eq!(values, [Fp::ONE; 8]);
        let mut invalid = [Fp::ONE; 3];
        assert_eq!(
            intt_with_cancel(&mut invalid, || panic!("invalid length polled")),
            Err(NttError::NotPowerOfTwo { len: 3 })
        );
        assert_eq!(
            coset_ntt_with_cancel(&mut invalid, Fp::ZERO, || panic!("invalid length polled")),
            Err(NttError::NotPowerOfTwo { len: 3 })
        );
        assert_eq!(
            coset_ntt_with_cancel(&mut values, Fp::ZERO, || panic!("zero shift polled")),
            Err(NttError::ZeroShift)
        );
    }
}
