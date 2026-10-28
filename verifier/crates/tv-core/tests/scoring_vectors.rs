// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Conformance vectors and adversarial domain/boundary tests for RFC-0017 and RFC-0061 scoring.

use std::fs;
use tv_codec::Rational;
use tv_core::protocol::{Decoding, FixedBits, PredictorSpec, RequestRef};
use tv_core::scoring::{
    SamplerKey, ScoringError, UhatError, aes128_encrypt_block, c_match, check_uhat_bound,
    derive_sampler_key, inverse_cdf, sampler_draw, stream_class, validate_decoding,
    validate_predictor, vocab_exp_scale,
};

fn hex16(s: &str) -> [u8; 16] {
    tv_testkit::hex(s).try_into().expect("16-byte hex")
}

fn hex32(s: &str) -> [u8; 32] {
    tv_testkit::hex(s).try_into().expect("32-byte hex")
}

fn rat(num: i64, den: u64) -> Rational {
    Rational::new(num, den).expect("canonical rational")
}

fn vector_lines(rel: &str) -> Vec<String> {
    let path = tv_testkit::vectors_dir().join(rel);
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    raw.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect()
}

#[test]
fn fips_197_appendix_b_and_c1_known_answers() {
    // FIPS 197 Appendix B cipher example.
    let key_b = SamplerKey::from_bytes(hex16("2b7e151628aed2a6abf7158809cf4f3c"));
    let pt_b = hex16("3243f6a8885a308d313198a2e0370734");
    let ct_b = hex16("3925841d02dc09fbdc118597196a0b32");
    assert_eq!(aes128_encrypt_block(&key_b, &pt_b), ct_b);

    // FIPS 197 Appendix C.1 AES-128 example vector.
    let key_c1 = SamplerKey::from_bytes(hex16("000102030405060708090a0b0c0d0e0f"));
    let pt_c1 = hex16("00112233445566778899aabbccddeeff");
    let ct_c1 = hex16("69c4e0d86a7b0430d8cdb78070b4c55a");
    assert_eq!(aes128_encrypt_block(&key_c1, &pt_c1), ct_c1);
}

#[test]
fn derive_vectors_match_all_46_rows() {
    let rows = vector_lines("scoring/derive.txt");
    assert_eq!(rows.len(), 46);
    let prompt = [1u32, 2, 3];
    for (idx, line) in rows.iter().enumerate() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(cols.len(), 7, "row {idx}");
        let base_key = hex16(cols[0]);
        let nonce = hex32(cols[1]);
        let bits: u8 = cols[2].parse().expect("bits");
        let from_ns: u64 = cols[3].parse().expect("from_ns");
        let expected_key = hex16(cols[4]);
        let expected_class: u64 = cols[5].parse().expect("class");
        let expected_u0: u32 = cols[6].parse().expect("u_0");

        let derived = derive_sampler_key(&base_key, &nonce, bits, from_ns)
            .unwrap_or_else(|e| panic!("row {idx} derive failed: {e}"));
        assert_eq!(derived.as_bytes(), &expected_key, "row {idx} key");

        let class = stream_class(&derived, bits, &prompt)
            .unwrap_or_else(|e| panic!("row {idx} stream_class failed: {e}"));
        assert_eq!(class, expected_class, "row {idx} class");

        let u0 = sampler_draw(&derived, class, 0);
        assert_eq!(u0, expected_u0, "row {idx} u_0");
    }
}

#[test]
fn prf_vectors_match_all_369_rows_for_class_ciphertext_and_draw() {
    let rows = vector_lines("scoring/prf.txt");
    assert_eq!(rows.len(), 369);
    for (idx, line) in rows.iter().enumerate() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(cols.len(), 7, "row {idx}");
        let key_bytes = hex16(cols[0]);
        let key = SamplerKey::from_bytes(key_bytes);
        let bits: u8 = cols[1].parse().expect("bits");
        let prompt: Vec<u32> = if cols[2] == "-" {
            Vec::new()
        } else {
            cols[2]
                .split(',')
                .map(|tok| tok.parse().expect("token"))
                .collect()
        };
        let t: u32 = cols[3].parse().expect("t");
        let expected_class: u64 = cols[4].parse().expect("class");
        let expected_ct = hex16(cols[5]);
        let expected_u: u32 = cols[6].parse().expect("u");

        let class = stream_class(&key, bits, &prompt)
            .unwrap_or_else(|e| panic!("row {idx} stream_class failed: {e}"));
        assert_eq!(class, expected_class, "row {idx} class");

        let mut block = [0u8; 16];
        block[..8].copy_from_slice(&class.to_le_bytes());
        block[8..12].copy_from_slice(&t.to_le_bytes());
        assert_eq!(
            aes128_encrypt_block(&key, &block),
            expected_ct,
            "row {idx} ciphertext"
        );
        assert_eq!(sampler_draw(&key, class, t), expected_u, "row {idx} u");
    }
}

#[test]
fn inverse_cdf_vectors_match_all_228_rows() {
    let rows = vector_lines("scoring/inverse_cdf.txt");
    assert_eq!(rows.len(), 228);
    for (idx, line) in rows.iter().enumerate() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        assert!(cols.len() >= 3, "row {idx}");
        let u: u32 = cols[0].parse().expect("u");
        let weights: Vec<u64> = cols[2..]
            .iter()
            .map(|w| w.parse().expect("weight"))
            .collect();
        let actual = inverse_cdf(&weights, u);
        if cols[1] == "none" {
            assert_eq!(actual, Err(ScoringError::NoMass), "row {idx}");
        } else {
            let expected: usize = cols[1].parse().expect("token");
            assert_eq!(actual, Ok(expected), "row {idx}");
        }
    }
}

#[test]
fn sampler_key_debug_redacts_key_material_and_domain_guards_reject() {
    let secret = SamplerKey::from_bytes([0xAB; 16]);
    let formatted = format!("{secret:?}");
    assert!(formatted.contains("REDACTED"), "{formatted}");
    assert!(!formatted.to_lowercase().contains("ab"), "{formatted}");

    for bad_bits in [65u8, 66, 128, 255] {
        assert_eq!(
            derive_sampler_key(&[0; 16], &[0; 32], bad_bits, 0).unwrap_err(),
            ScoringError::ClassBits { bits: bad_bits }
        );
        assert_eq!(
            stream_class(&secret, bad_bits, &[1, 2]),
            Err(ScoringError::ClassBits { bits: bad_bits })
        );
    }

    // Prompt tokens must satisfy token < 2^18 regardless of whether b == 0 or b > 0.
    assert_eq!(stream_class(&secret, 0, &[(1 << 18) - 1]), Ok(0));
    for bits in [0u8, 8, 64] {
        for bad_tok in [1u32 << 18, (1u32 << 18) + 7, u32::MAX] {
            assert_eq!(
                stream_class(&secret, bits, &[0, bad_tok, 1]),
                Err(ScoringError::PromptToken {
                    index: 1,
                    token: bad_tok
                })
            );
        }
    }
}

#[test]
fn inverse_cdf_boundary_and_zero_mass_cases() {
    assert_eq!(inverse_cdf(&[], 0), Err(ScoringError::NoMass));
    assert_eq!(inverse_cdf(&[0, 0, 0, 0], 42), Err(ScoringError::NoMass));

    // Exact threshold boundary: weights [1, 3], Z = 4.
    // floor((u * 4) / 2^32) == 0 for u <= 2^30 - 1, and == 1 at u == 2^30.
    assert_eq!(inverse_cdf(&[1, 3], (1u32 << 30) - 1), Ok(0));
    assert_eq!(inverse_cdf(&[1, 3], 1u32 << 30), Ok(1));

    // Maximum vocabulary size 2^18 with positive mass at the final token.
    let mut max_vocab = vec![0u64; 1 << 18];
    max_vocab[(1 << 18) - 1] = 1;
    assert_eq!(inverse_cdf(&max_vocab, u32::MAX), Ok((1 << 18) - 1));
}

#[test]
fn predictor_domain_scale_and_c_match_boundaries() {
    assert_eq!(validate_decoding(&Decoding::Greedy), Ok(()));
    assert_eq!(
        validate_decoding(&Decoding::Temperature { t: rat(1, 16) }),
        Ok(())
    );
    assert_eq!(
        validate_decoding(&Decoding::Temperature { t: rat(4, 1) }),
        Ok(())
    );
    for bad_temp in [rat(0, 1), rat(-1, 1), rat(-4, 1), rat(9, 2), rat(5, 1)] {
        let dec = Decoding::Temperature { t: bad_temp };
        assert_eq!(
            validate_decoding(&dec),
            Err(ScoringError::Temperature { t: bad_temp })
        );
        assert_eq!(
            validate_predictor(&PredictorSpec::Exact { decoding: dec }),
            Err(ScoringError::Temperature { t: bad_temp })
        );
    }

    for eps in [0u8, 1, 8, 16] {
        assert_eq!(
            validate_predictor(&PredictorSpec::Mixture {
                decoding: Decoding::Greedy,
                eps_log2: eps,
            }),
            Ok(())
        );
    }
    for bad_eps in [17u8, 18, 255] {
        assert_eq!(
            validate_predictor(&PredictorSpec::Mixture {
                decoding: Decoding::Greedy,
                eps_log2: bad_eps,
            }),
            Err(ScoringError::EpsLog2 { eps_log2: bad_eps })
        );
        assert_eq!(
            c_match(bad_eps),
            Err(ScoringError::EpsLog2 { eps_log2: bad_eps })
        );
    }

    let zero_t = rat(0, 1);
    assert_eq!(
        vocab_exp_scale(-16, zero_t),
        Err(ScoringError::Temperature { t: zero_t })
    );
    assert_eq!(vocab_exp_scale(-16, rat(1, 1)), Ok(rat(1, 65_536)));
    assert_eq!(vocab_exp_scale(-16, rat(4, 1)), Ok(rat(1, 262_144)));
    assert_eq!(vocab_exp_scale(-16, rat(3, 2)), Ok(rat(1, 98_304)));
    assert_eq!(vocab_exp_scale(2, rat(1, 2)), Ok(rat(8, 1)));
    assert_eq!(
        vocab_exp_scale(63, rat(1, 2)),
        Err(ScoringError::LogitsExponent {
            logits_exponent: 63
        })
    );
    assert_eq!(c_match(0), Ok(0));
    assert_eq!(c_match(8), Ok(2));
    assert_eq!(c_match(16), Ok(1));
}

#[test]
fn check_uhat_bound_enforces_exact_zero_and_mixture_per_request_and_batch_caps() {
    let reqs = [
        RequestRef {
            request_id: 101,
            session_id: [1; 16],
            prompt_len: 3,
            output_len: 2,
        },
        RequestRef {
            request_id: 202,
            session_id: [2; 16],
            prompt_len: 4,
            output_len: 1,
        },
    ];
    let exact = PredictorSpec::Exact {
        decoding: Decoding::Greedy,
    };
    let mixture = PredictorSpec::Mixture {
        decoding: Decoding::Greedy,
        eps_log2: 4,
    };

    // Malformed lengths.
    assert_eq!(
        check_uhat_bound(exact, &[], &[], FixedBits(0)),
        Err(UhatError::Malformed)
    );
    assert_eq!(
        check_uhat_bound(exact, &reqs, &[FixedBits(0)], FixedBits(0)),
        Err(UhatError::Malformed)
    );

    // Exact ignores positive accept_bound and requires every request's uhat == 0.
    assert_eq!(
        check_uhat_bound(
            exact,
            &reqs,
            &[FixedBits(0), FixedBits(0)],
            FixedBits(999_999)
        ),
        Ok(FixedBits(0))
    );
    assert_eq!(
        check_uhat_bound(
            exact,
            &reqs,
            &[FixedBits(0), FixedBits(5)],
            FixedBits(999_999)
        ),
        Err(UhatError::Exceeded {
            request_id: Some(202),
            uhat: FixedBits(5),
            bound: FixedBits(0),
        })
    );

    // Mixture accepts when every request and the batch sum are <= accept_bound.
    assert_eq!(
        check_uhat_bound(
            mixture,
            &reqs,
            &[FixedBits(40), FixedBits(60)],
            FixedBits(100)
        ),
        Ok(FixedBits(100))
    );

    // Mixture rejects an individual request exceeding accept_bound with Some(request_id).
    assert_eq!(
        check_uhat_bound(
            mixture,
            &reqs,
            &[FixedBits(101), FixedBits(0)],
            FixedBits(100)
        ),
        Err(UhatError::Exceeded {
            request_id: Some(101),
            uhat: FixedBits(101),
            bound: FixedBits(100),
        })
    );

    // Mixture rejects when each request is <= accept_bound but the batch sum exceeds it (request_id: None).
    assert_eq!(
        check_uhat_bound(
            mixture,
            &reqs,
            &[FixedBits(60), FixedBits(60)],
            FixedBits(100)
        ),
        Err(UhatError::Exceeded {
            request_id: None,
            uhat: FixedBits(120),
            bound: FixedBits(100),
        })
    );

    // Overflowing batch sum fails as Malformed rather than wrapping.
    assert_eq!(
        check_uhat_bound(
            mixture,
            &reqs,
            &[FixedBits(u64::MAX), FixedBits(1)],
            FixedBits(u64::MAX)
        ),
        Err(UhatError::Malformed)
    );
}
