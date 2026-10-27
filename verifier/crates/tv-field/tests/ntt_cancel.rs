// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Cancellation keeps validation precedence and polls before completion (RFC-0019, SP-8).
use tv_field::{Fp, NttError, coset_ntt_with_cancel, intt_with_cancel};

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

#[test]
fn cancellation_is_polled_before_success_even_for_a_singleton() {
    for inverse in [false, true] {
        let mut values = [Fp::ONE];
        let mut polls = 0;
        let mut cancel = || {
            polls += 1;
            polls == 2
        };
        let result = if inverse {
            intt_with_cancel(&mut values, &mut cancel)
        } else {
            coset_ntt_with_cancel(&mut values, Fp::SEVEN, &mut cancel)
        };
        assert_eq!(result, Err(NttError::Cancelled));
        assert_eq!(polls, 2);
    }
}
