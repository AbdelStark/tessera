// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `sumcheck` target's cases and oracle on seeded inputs, so that `cargo test` runs the
//! target's code without the fuzzer: inputs of every length up to 96 bytes, which spell honest
//! cases when they end before any change, and changed ones otherwise.

use tv_fuzz::sumcheck::{Case, case, check};
use tv_testkit::SplitMix64;

#[test]
fn seeded_inputs_meet_the_oracle() {
    let mut rng = SplitMix64::new(0x5eed_f022_0059);
    let (mut stages, mut trees) = (0, 0);
    for _ in 0..1500 {
        let length = (rng.next_u64() % 97) as usize;
        let data: Vec<u8> = (0..length).map(|_| rng.next_u64() as u8).collect();
        check(&data);
        match case(&data) {
            Case::Stage(_) => stages += 1,
            Case::Trees(_) => trees += 1,
        }
    }
    assert!(
        stages > 500 && trees > 500,
        "{stages} stages, {trees} trees"
    );
}

#[test]
fn protocol_two_high_degrees_and_all_changes_meet_the_oracle() {
    use tv_core::sumcheck::Protocol;
    for degree in 8..=10u8 {
        let mut base = vec![
            2,
            degree - 1,
            2,
            3,
            degree - 1,
            1,
            degree - 1,
            0,
            degree - 1,
            0,
        ];
        base.extend(u64::from(degree).to_le_bytes());
        let Case::Stage(stage) = case(&base) else {
            panic!("a stage")
        };
        assert_eq!(stage.protocol, Protocol::Two);
        assert_eq!(stage.degree, usize::from(degree));
        stage.check();
        for change in 0..16 {
            let mut changed = base.clone();
            changed.extend([change, 1, 0, 2, 0, 3, 0]);
            changed.extend(5u128.to_le_bytes());
            check(&changed);
        }
    }
}
