// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Replay the actual packet target on independent fixtures and seeded byte mutations.
use tv_fuzz::protocol::{TYPES, check};
use tv_testkit::{SplitMix64, hex, read_vectors};

#[test]
fn every_packet_seed_and_its_changes_reach_the_fuzz_entry_point() {
    let vectors = read_vectors("protocol/protocol.json");
    let mut rng = SplitMix64::new(0x0036_0128);
    for key in ["messages", "refused", "accepted"] {
        for case in vectors.get(key).unwrap().as_array().unwrap() {
            let selector = TYPES
                .iter()
                .position(|&s| Some(s) == case.get("type").unwrap().as_str())
                .unwrap() as u8;
            let payload = hex(case
                .get(if key == "messages" { "payload" } else { "hex" })
                .unwrap()
                .as_str()
                .unwrap());
            for config in 0..4 {
                let data = [&[selector, config][..], &payload].concat();
                let accepted = check(&data);
                if config == 0 && key != "refused" {
                    assert!(accepted, "valid {key} seed did not reach its decoder");
                }
                for _ in 0..20 {
                    let mut changed = data.clone();
                    let at = (rng.next_u64() as usize) % changed.len();
                    changed[at] ^= rng.next_u64() as u8 | 1;
                    check(&changed);
                }
            }
            if key == "messages" {
                let envelope = hex(case.get("envelope").unwrap().as_str().unwrap());
                for config in 4..8 {
                    let data = [&[selector, config][..], &envelope].concat();
                    let accepted = check(&data);
                    if config == 4 {
                        assert!(accepted, "valid envelope seed did not reach its decoder");
                    }
                    for end in 0..data.len() {
                        check(&data[..end]);
                    }
                }
            }
        }
    }
    for case in vectors.get("envelopes").unwrap().as_array().unwrap() {
        let selector = TYPES
            .iter()
            .position(|&s| Some(s) == case.get("schema").unwrap().as_str())
            .unwrap() as u8;
        let envelope = hex(case.get("hex").unwrap().as_str().unwrap());
        for config in 4..8 {
            assert!(!check(&[&[selector, config][..], &envelope].concat()));
        }
    }
    for _ in 0..2000 {
        let len = (rng.next_u64() % 1024) as usize;
        let data = (0..len).map(|_| rng.next_u64() as u8).collect::<Vec<_>>();
        check(&data);
    }
}
