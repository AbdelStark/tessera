// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The acceptance criteria of the encoding (RFC-0011 'Encoding', 'Testing Strategy'), checked
//! against definitions rather than the encoder's algorithm:
//!
//! - codewords lie in the code: the polynomial of degree below `k + P` through `k + P` random
//!   positions predicts every other symbol, and one corrupted symbol fails the test;
//! - the mask vanishes on `H`: `g`, recovered from the codeword by the inverse DFT as a sum, has
//!   the drawn mask as its coefficients `k, …, k + P − 1` and takes the row's values on `H`;
//! - a noiseless codeword decodes back to its row, whatever its mask.

mod common;

use common::{
    Fp, Replay, SplitMix64, codeword_points, horner, inverse_dft, passes_degree_test, root, seven,
};
use tessera_field::Field;
use tessera_pcs::{COSET_SHIFT, NttPlan, RowEncoder};

/// Mask counts worth testing for rows of `k` values: none, one, half, all but one, all.
fn mask_counts(k: usize) -> Vec<usize> {
    let mut counts = vec![0, 1, k / 2, k.saturating_sub(1), k];
    counts.sort_unstable();
    counts.dedup();
    counts
}

/// Encodes `row` with `P` fresh coefficients from `rng`, and returns the codeword and the mask.
fn encode(encoder: &RowEncoder, row: &[Fp], rng: &mut SplitMix64) -> (Vec<Fp>, Vec<Fp>) {
    let mask = rng.fps(encoder.mask_coeffs());
    let mut source = Replay::new(mask.clone());
    let mut codeword = vec![Fp::ZERO; encoder.codeword_len()];
    encoder
        .encode_row(row, &mut source, &mut codeword)
        .expect("matching lengths");
    assert_eq!(source.drawn, mask.len(), "P draws per row");
    (codeword, mask)
}

/// Decodes a noiseless codeword: `g` by the inverse coset NTT, then `f = g mod (X^k − 1)`, whose
/// values on `H` are the row. Checks on the way that `g` has degree below `k + P`.
fn decode(codeword: &[Fp], mask_coeffs: usize) -> Vec<Fp> {
    let (n, k) = (codeword.len(), codeword.len() / 2);
    let mut g = codeword.to_vec();
    NttPlan::new(n)
        .expect("a supported size")
        .coset_intt(&mut g, COSET_SHIFT)
        .expect("matching length");
    assert!(
        g[k + mask_coeffs..].iter().all(|&c| c == Fp::ZERO),
        "degree below k + P"
    );
    // X^k ≡ 1 modulo X^k − 1.
    let mut f: Vec<Fp> = (0..k).map(|i| g[i] + g[k + i]).collect();
    NttPlan::new(k)
        .expect("a supported size")
        .ntt(&mut f)
        .expect("matching length");
    f
}

#[test]
fn codewords_pass_the_degree_test_on_random_subsets_and_a_corrupted_one_fails() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0047_0201);
    for log_k in 0..=8 {
        let k = 1usize << log_k;
        let n = 2 * k;
        for p in mask_counts(k) {
            let degree = k + p;
            if degree == n {
                // P = k: degree below n, which every word of n symbols has.
                continue;
            }
            let encoder = RowEncoder::new(k, 1, p).expect("valid parameters");
            let row = rng.fps(k);
            let (codeword, _) = encode(&encoder, &row, &mut rng);
            for _ in 0..3 {
                let subset = rng.subset(n, degree);
                assert!(
                    passes_degree_test(&codeword, &subset, degree),
                    "k = {k}, P = {p}"
                );
            }
            // One symbol off: the interpolant through any k + P positions mispredicts some other
            // symbol, the corrupted one or, if it is among the nodes, every one outside them.
            let mut corrupted = codeword.clone();
            let j = rng.below(n);
            corrupted[j] += rng.nonzero();
            for _ in 0..3 {
                let subset = rng.subset(n, degree);
                assert!(
                    !passes_degree_test(&corrupted, &subset, degree),
                    "k = {k}, P = {p}, symbol {j} corrupted"
                );
            }
            // A word of degree exactly k + P fails too, as one more mask coefficient would make.
            let points = codeword_points(n);
            let mut beyond = vec![Fp::ZERO; degree + 1];
            beyond[degree] = Fp::ONE;
            let shifted: Vec<Fp> = codeword
                .iter()
                .zip(&points)
                .map(|(&c, &x)| c + horner(&beyond, x))
                .collect();
            let subset = rng.subset(n, degree);
            assert!(
                !passes_degree_test(&shifted, &subset, degree),
                "k = {k}, P = {p}, X^(k + P) added"
            );
        }
    }
}

#[test]
fn the_mask_vanishes_on_h_and_sits_in_the_top_coefficients() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0047_0202);
    for log_k in 0..=8 {
        let k = 1usize << log_k;
        for p in mask_counts(k) {
            let encoder = RowEncoder::new(k, 1, p).expect("valid parameters");
            let row = rng.fps(k);
            let (codeword, mask) = encode(&encoder, &row, &mut rng);
            // g's coefficients, by the inverse DFT on 7·L as a sum.
            let g = inverse_dft(&codeword, seven());
            // g = f − m + X^k·m: the mask is the coefficients k, …, k + P − 1, and nothing above.
            assert_eq!(&g[k..k + p], &mask[..], "k = {k}, P = {p}");
            assert!(
                g[k + p..].iter().all(|&c| c == Fp::ZERO),
                "k = {k}, P = {p}"
            );
            // On H, g takes the row's values: evaluated from its coefficients, point by point.
            let w = root(k);
            let mut x = Fp::ONE;
            for (i, &value) in row.iter().enumerate() {
                assert_eq!(horner(&g, x), value, "g(ω_k^{i}), k = {k}, P = {p}");
                x *= w;
            }
            // And g − f is (X^k − 1)·m: below k, the coefficients of f minus the mask.
            let f = inverse_dft(&row, Fp::ONE);
            for i in 0..k {
                let m_i = mask.get(i).copied().unwrap_or(Fp::ZERO);
                assert_eq!(g[i], f[i] - m_i, "coefficient {i}, k = {k}, P = {p}");
            }
        }
    }
}

#[test]
fn noiseless_codewords_decode_to_their_row_whatever_the_mask() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0047_0203);
    for log_k in [0, 1, 2, 3, 5, 8, 11, 14, 16] {
        let k = 1usize << log_k;
        for p in mask_counts(k) {
            let encoder = RowEncoder::new(k, 1, p).expect("valid parameters");
            let row = rng.fps(k);
            let (first, _) = encode(&encoder, &row, &mut rng);
            let (second, _) = encode(&encoder, &row, &mut rng);
            if p > 0 {
                assert_ne!(first, second, "fresh masks change the codeword, k = {k}");
            } else {
                assert_eq!(
                    first, second,
                    "without a mask the encoding is deterministic"
                );
            }
            assert!(decode(&first, p) == row, "k = {k}, P = {p}");
            assert!(decode(&second, p) == row, "k = {k}, P = {p}, second mask");
        }
    }
}

#[test]
fn a_zero_row_with_no_mask_is_the_zero_codeword_and_a_constant_row_a_constant_one() {
    for k in [1, 2, 64] {
        let encoder = RowEncoder::new(k, 1, 0).expect("valid parameters");
        let mut codeword = vec![Fp::ONE; 2 * k];
        let mut never = || -> Fp { unreachable!("P = 0 draws nothing") };
        encoder
            .encode_row(&vec![Fp::ZERO; k], &mut never, &mut codeword)
            .expect("matching lengths");
        assert!(codeword.iter().all(|&c| c == Fp::ZERO), "k = {k}");
        encoder
            .encode_row(&vec![seven(); k], &mut never, &mut codeword)
            .expect("matching lengths");
        assert!(codeword.iter().all(|&c| c == seven()), "k = {k}");
    }
}

#[test]
fn the_codeword_buffer_is_overwritten_whatever_it_held() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0047_0204);
    let encoder = RowEncoder::new(256, 1, 20).expect("valid parameters");
    let row = rng.fps(256);
    let mask = rng.fps(20);
    let mut clean = vec![Fp::ZERO; 512];
    let mut dirty = rng.fps(512);
    encoder
        .encode_row(&row, &mut Replay::new(mask.clone()), &mut clean)
        .expect("matching lengths");
    encoder
        .encode_row(&row, &mut Replay::new(mask), &mut dirty)
        .expect("matching lengths");
    assert_eq!(clean, dirty);
}

#[test]
fn a_batch_is_its_rows_encoded_in_turn_from_one_source() {
    let mut rng = SplitMix64::new(0x7E55_E7A0_0047_0205);
    for (k, p, count) in [
        (1, 1, 3),
        (8, 3, 5),
        (64, 64, 2),
        (1024, 200, 4),
        (16, 0, 0),
    ] {
        let encoder = RowEncoder::new(k, 1, p).expect("valid parameters");
        let rows = rng.fps(k * count);
        let masks = rng.fps(p * count);
        let mut batch = vec![Fp::ZERO; 2 * k * count];
        let mut source = Replay::new(masks.clone());
        encoder
            .encode_rows(&rows, &mut source, &mut batch)
            .expect("matching lengths");
        assert_eq!(source.drawn, p * count, "P draws per row");
        let mut one_by_one = Replay::new(masks);
        for (row, expected) in rows.chunks_exact(k).zip(batch.chunks_exact(2 * k)) {
            let mut codeword = vec![Fp::ZERO; 2 * k];
            encoder
                .encode_row(row, &mut one_by_one, &mut codeword)
                .expect("matching lengths");
            assert_eq!(codeword, expected, "k = {k}, P = {p}");
        }
    }
}

#[test]
fn mask_coefficients_are_drawn_in_order_m0_first() {
    // Swapping two drawn coefficients changes the codeword: the order is part of the encoding.
    let mut rng = SplitMix64::new(0x7E55_E7A0_0047_0206);
    let encoder = RowEncoder::new(32, 1, 4).expect("valid parameters");
    let row = rng.fps(32);
    let mask = rng.fps(4);
    let mut swapped = mask.clone();
    swapped.swap(0, 1);
    let mut a = vec![Fp::ZERO; 64];
    let mut b = vec![Fp::ZERO; 64];
    encoder
        .encode_row(&row, &mut Replay::new(mask.clone()), &mut a)
        .expect("matching lengths");
    encoder
        .encode_row(&row, &mut Replay::new(swapped), &mut b)
        .expect("matching lengths");
    assert_ne!(a, b);
    assert_eq!(&inverse_dft(&a, seven())[32..36], &mask[..]);
}

#[test]
fn a_closure_is_a_mask_source_and_so_is_a_trait_object() {
    let encoder = RowEncoder::new(4, 1, 2).expect("valid parameters");
    let row = [Fp::ONE; 4];
    let mut counter = 0u64;
    let mut closure = || {
        counter += 1;
        common::fp(counter)
    };
    let mut a = [Fp::ZERO; 8];
    encoder
        .encode_row(&row, &mut closure, &mut a)
        .expect("matching lengths");
    let mut replay = Replay::new(vec![common::fp(1), common::fp(2)]);
    let source: &mut dyn tessera_pcs::MaskSource = &mut replay;
    let mut b = [Fp::ZERO; 8];
    encoder
        .encode_row(&row, source, &mut b)
        .expect("matching lengths");
    assert_eq!(a, b);
    assert_eq!(counter, 2);
}

#[test]
fn accessors_and_debug_show_the_parameters_only() {
    let encoder = RowEncoder::new(1 << 10, 1, 200).expect("valid parameters");
    assert_eq!(encoder.message_len(), 1024);
    assert_eq!(encoder.codeword_len(), 2048);
    assert_eq!(encoder.mask_coeffs(), 200);
    assert_eq!(
        format!("{encoder:?}"),
        "RowEncoder { message_len: 1024, codeword_len: 2048, mask_coeffs: 200, .. }"
    );
    let plan = NttPlan::new(1 << 10).expect("a supported size");
    assert_eq!(format!("{plan:?}"), "NttPlan { size: 1024, .. }");
}
