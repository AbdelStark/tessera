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
