// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The complete target oracles must reach real acceptance and exact refusals.
#![cfg(feature = "test-params")]

#[test]
fn claim_controls_are_semantic_checks_and_both_raw_forms_reach_validation() {
    let bytes = tv_fuzz::driver::fixture_bytes(false, "claims.tcl");
    for mode in [0, 1] {
        let payload = if mode == 0 {
            &bytes[20..bytes.len() - 32]
        } else {
            &bytes
        };
        assert!(tv_fuzz::claims::check(&[&[mode], payload].concat()));
    }
    for mode in 2..=5 {
        assert!(!tv_fuzz::claims::check(&[mode]));
    }
    for byte in 0..=255 {
        assert!(!tv_fuzz::claims::check(&[0, byte]));
    }
}

#[test]
fn complete_ni_controls_accept_and_reject_at_the_declared_checks() {
    for (header, golden) in [
        (0, tv_fuzz::driver::golden()),
        (64, tv_fuzz::driver::tamper()),
    ] {
        assert!(tv_fuzz::driver::check(&[header | 2]));
        for mode in 3..=8 {
            assert!(!tv_fuzz::driver::check(&[header | mode]));
        }
        assert!(tv_fuzz::driver::check(&[&[header | 0x30], golden].concat()));
        assert!(tv_fuzz::driver::check(
            &[&[header | 1], &golden[20..golden.len() - 32]].concat()
        ));
    }
    for mode in [0x30, 1] {
        for byte in 0..=255 {
            assert!(!tv_fuzz::driver::check(&[mode, byte]));
        }
    }
}

#[test]
fn complete_ni_target_replays_every_declared_ni_tamper_case() {
    let indices = tv_fuzz::driver::tamper_cases();
    assert!(!indices.is_empty());
    for index in indices {
        let selector = u16::try_from(index).unwrap().to_le_bytes();
        assert!(!tv_fuzz::driver::check(&[9, selector[0], selector[1]]));
    }
}
