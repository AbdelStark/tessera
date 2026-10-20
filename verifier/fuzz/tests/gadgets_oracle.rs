// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exercise the actual sumcheck target's gadget router without libFuzzer.
use tv_testkit::SplitMix64;

#[test]
fn every_gadget_family_checks_seeded_fields_lengths_and_retained_debts() {
    assert!(!tv_fuzz::sumcheck::check(&[]));
    let mut rng = SplitMix64::new(0x1240_f022_0009);
    for family in 0..tv_fuzz::gadgets::FAMILIES {
        for variation in 0..48 {
            let mut data = vec![0; 512];
            if variation > 0 {
                for b in &mut data {
                    *b = rng.next_u64() as u8;
                }
            }
            data[0] = family;
            // Keep affine fixtures valid in half the cases so their nonlinear
            // finals and source maps receive arbitrary extension-field operands.
            if variation % 2 == 0 {
                match family {
                    3 | 4 => data[5] = 0,
                    5 => data[3] = 0,
                    _ => (),
                }
            }
            let mut routed = b"GDG".to_vec();
            routed.extend_from_slice(&data);
            assert!(tv_fuzz::sumcheck::check(&routed));
        }
    }
}
#[test]
fn short_gadget_inputs_and_every_affine_perturbation_meet_the_oracle() {
    for family in 0..tv_fuzz::gadgets::FAMILIES {
        for len in 1..=64 {
            let mut data = vec![0; len];
            data[0] = family;
            tv_fuzz::gadgets::check(&data);
        }
    }
    for family in [3, 4, 5] {
        for variant in 0..32u8 {
            for change in 0..8u8 {
                let mut data = vec![0; 512];
                data[0] = family;
                data[1] = variant;
                let mode = if family == 5 { 3 } else { 5 };
                data[mode] = change;
                tv_fuzz::gadgets::check(&data);
            }
        }
    }
}

#[test]
fn every_generated_gadget_corpus_seed_runs_through_the_target() {
    let script = "import importlib.util,sys\ns=importlib.util.spec_from_file_location('gadget_seeds','seeds.py')\nm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nfor seed in m.gadget_seeds():\n sys.stdout.buffer.write(len(seed).to_bytes(4,'little')+seed)\n";
    let result = std::process::Command::new("python3")
        .args(["-I", "-B", "-c", script])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let mut rest = &result.stdout[..];
    let mut families = [0usize; tv_fuzz::gadgets::FAMILIES as usize];
    let mut count = 0;
    while !rest.is_empty() {
        let len = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
        assert!(len < 1 << 20);
        let seed = &rest[4..4 + len];
        assert!(seed.starts_with(b"GDG"));
        families[usize::from(seed[3])] += 1;
        assert!(tv_fuzz::sumcheck::check(seed));
        rest = &rest[4 + len..];
        count += 1;
    }
    assert!(count > 1000);
    assert!(families.iter().all(|&n| n > 0));
}
