// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Adversarial, algebraic, 512-bit integer oracle, and `tv-core`-verified tests for
//! [`OneSidedLog2`] and bounded [`Argmax`] (RFC-0017 'In-circuit scoring (mixture)', RFC-0008,
//! RFC-0009 Lemmas 0, 3, 4 & 9, RFC-0035, RFC-0050, RFC-0053).

mod common;

use common::{
    Fp, Fp2, Messages, Rejected as LogUpRejected, Session, Shape, SplitMix64, TvFp2, fp2, in_table,
    mle_fp, prove, spec_of, table_columns, tv_fp2, tv_fp2s, verify,
};
use tessera_codec::SecretBuf;
use tessera_field::{F65537, Field, PrimeField, QuadraticExtension};
use tessera_gadgets::{
    Argmax, CompareColumn, GadgetError, Log2BatchColumns, Log2Column, Log2Term, Lookups,
    OneSidedLog2, Side, c_match,
};
use tessera_sumcheck::{Elementwise, Prefix, Protocol, prove_batch};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};
use tv_core::sumcheck as tv;

/// Exact arbitrary-precision unsigned integer (`Vec<u64>` little-endian limbs) for float-free
/// `2^{B_t} · num_q^{256} >= den_q^{256}` checks across all `B_t ∈ 0..=20_223`.
fn big_mul_u128(limbs: &[u64], rhs: u128) -> Vec<u64> {
    if rhs == 0 {
        return vec![0];
    }
    let r = [rhs as u64, (rhs >> 64) as u64];
    let r_len = if r[1] == 0 { 1 } else { 2 };
    let mut out = vec![0u64; limbs.len() + r_len];
    for (i, &rw) in r[..r_len].iter().enumerate() {
        if rw == 0 {
            continue;
        }
        let mut carry = 0u128;
        for (j, &lw) in limbs.iter().enumerate() {
            let p = u128::from(lw) * u128::from(rw) + u128::from(out[i + j]) + carry;
            out[i + j] = p as u64;
            carry = p >> 64;
        }
        out[i + limbs.len()] += carry as u64;
    }
    while out.last() == Some(&0) && out.len() > 1 {
        out.pop();
    }
    out
}

fn big_pow_256(x: u128) -> Vec<u64> {
    let mut acc = vec![1u64];
    for _ in 0..256 {
        acc = big_mul_u128(&acc, x);
    }
    acc
}

fn big_shl(limbs: &[u64], bits: usize) -> Vec<u64> {
    if bits == 0 || limbs == [0] {
        return limbs.to_vec();
    }
    let ws = bits / 64;
    let bs = bits % 64;
    let mut out = vec![0u64; limbs.len() + ws + 1];
    for (i, &w) in limbs.iter().enumerate() {
        out[i + ws] |= w << bs;
        if bs > 0 {
            out[i + ws + 1] |= w >> (64 - bs);
        }
    }
    while out.last() == Some(&0) && out.len() > 1 {
        out.pop();
    }
    out
}

fn big_cmp(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
    let n = a.len().max(b.len());
    for i in (0..n).rev() {
        let av = a.get(i).copied().unwrap_or(0);
        let bv = b.get(i).copied().unwrap_or(0);
        match av.cmp(&bv) {
            core::cmp::Ordering::Equal => continue,
            ord => return ord,
        }
    }
    core::cmp::Ordering::Equal
}

fn log2_table_specs() -> Vec<TableSpec> {
    let mut specs = Vec::new();
    specs.push(
        GeneratedTable::new(TableId(0), TableKind::Pow2 { max_exp: 63 })
            .expect("Pow2(63)")
            .spec()
            .clone(),
    );
    specs.push(
        GeneratedTable::new(
            TableId(1),
            TableKind::Exp2Frac {
                frac_bits: 8,
                out_bits: 16,
            },
        )
        .expect("Exp2Frac(8, 16)")
        .spec()
        .clone(),
    );
    for bits in 1..=32u8 {
        specs.push(
            GeneratedTable::new(TableId(1 + u32::from(bits)), TableKind::Range { bits })
                .expect("Range")
                .spec()
                .clone(),
        );
    }
    specs
}

fn verify_elementwise_check(
    relation: &Elementwise<'_, Fp2>,
    point: &[Fp2],
    target: Fp2,
    operands: &[&[Fp]],
    degree: usize,
    rng: &mut SplitMix64,
    mut verifier_eval: impl FnMut(&[TvFp2], &[TvFp2]) -> TvFp2,
) -> Result<(), &'static str> {
    let vars = relation.shape().vars;
    let tables = relation.tables(point).map_err(|_| "tables")?;
    let session = Session::random(rng);
    let mut pt = session.transcript();
    let mut vt = session.tv_transcript();
    let proved = prove_batch(
        &mut pt,
        Protocol::One,
        Prefix::Stage(1),
        degree,
        &[tables.instance()],
    )
    .map_err(|_| "prove_batch")?;
    let proof = tv::SumcheckProof {
        rounds: proved.proof().rounds.iter().map(|r| tv_fp2s(r)).collect(),
        final_evals: tv_fp2s(&proved.proof().final_evals),
    };
    let checked = tv::verify_batch(
        &mut vt,
        tv::Prefix::Stage(1),
        degree,
        &[tv::Instance {
            claim: tv_fp2(target),
            vars,
            evals: operands.len(),
        }],
        &proof,
        |_, r, e| Some(verifier_eval(r, e)),
    )
    .map_err(|_| "sumcheck")?;
    for (&val, col) in proof.final_evals.iter().zip(operands) {
        let p = checked
            .point_of(col.len().trailing_zeros() as usize)
            .ok_or("point_of")?;
        if val != mle_fp(col, p) {
            return Err("opening");
        }
    }
    if pt.state() != *vt.state() {
        return Err("transcript");
    }
    Ok(())
}

fn check_lookups(
    rng: &mut SplitMix64,
    lookups: &Lookups<Fp>,
    specs: &[TableSpec],
) -> Result<(), &'static str> {
    let owned: Vec<GeneratedTable> = specs
        .iter()
        .map(|s| GeneratedTable::from_spec(s).expect("regen"))
        .collect();
    let refs: Vec<&GeneratedTable> = owned.iter().collect();
    let cols = table_columns(&refs);
    let mults = lookups.multiplicities(&cols).map_err(|_| "out_of_table")?;
    let instances = lookups.instances(&cols, &mults).map_err(|_| "instances")?;
    let shapes: Vec<Shape> = instances
        .iter()
        .enumerate()
        .map(|(idx, inst)| Shape::of(inst, spec_of(&refs, lookups, idx)))
        .collect();
    let session = Session::random(rng);
    let proven = prove(&session, &instances);
    verify(&session, &shapes, &Messages::of(&proven))
        .map(|_| ())
        .map_err(|_: LogUpRejected| "logup_verify")
}

#[test]
fn c_match_satisfies_exact_integer_ceil_condition_and_rejects_above_16() {
    assert_eq!(c_match(0).unwrap(), 0);
    assert!(matches!(c_match(17), Err(GadgetError::Shape { .. })));
    assert!(matches!(c_match(u8::MAX), Err(GadgetError::Shape { .. })));

    let mul_small = |limbs: &[u64], m: u64| -> Vec<u64> {
        let mut out = Vec::with_capacity(limbs.len() + 1);
        let mut carry = 0u128;
        for &w in limbs {
            let p = u128::from(w) * u128::from(m) + carry;
            out.push(p as u64);
            carry = p >> 64;
        }
        if carry > 0 {
            out.push(carry as u64);
        }
        out
    };
    let shl = |limbs: &[u64], bits: usize| -> Vec<u64> {
        let total_bits = limbs.len() * 64 + bits;
        let mut out = vec![0u64; total_bits.div_ceil(64) + 1];
        let ws = bits / 64;
        let bs = bits % 64;
        for (i, &w) in limbs.iter().enumerate() {
            out[i + ws] |= w << bs;
            if bs > 0 {
                out[i + ws + 1] |= w >> (64 - bs);
            }
        }
        while out.last() == Some(&0) && out.len() > 1 {
            out.pop();
        }
        out
    };
    let cmp_big = |a: &[u64], b: &[u64]| -> core::cmp::Ordering {
        let n = a.len().max(b.len());
        for i in (0..n).rev() {
            let av = a.get(i).copied().unwrap_or(0);
            let bv = b.get(i).copied().unwrap_or(0);
            match av.cmp(&bv) {
                core::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        core::cmp::Ordering::Equal
    };

    for k in 1..=16u8 {
        let c = c_match(k).unwrap() as usize;
        assert!(c >= 1);
        let base = (1u64 << k) - 1;
        let mut pow = vec![1u64];
        for _ in 0..256 {
            pow = mul_small(&pow, base);
        }
        let target = shl(&[1u64], 256 * usize::from(k));
        let upper = shl(&pow, c);
        let lower = shl(&pow, c - 1);
        assert!(
            cmp_big(&upper, &target).is_ge(),
            "c_match({k}) = {c} under-approximated -log2(1 - 2^-{k})"
        );
        assert!(
            cmp_big(&lower, &target).is_lt(),
            "c_match({k}) = {c} was not minimal ceiling"
        );
    }
}

#[test]
fn constructor_and_constraints_enforce_domain_modulus_sigma_and_tables() {
    let specs = log2_table_specs();
    assert!(matches!(
        OneSidedLog2::<Fp>::new(0, 16, 6, &specs),
        Err(GadgetError::Shape { .. })
    ));
    assert!(matches!(
        OneSidedLog2::<Fp>::new((1 << 18) + 1, 16, 6, &specs),
        Err(GadgetError::Shape { .. })
    ));
    assert!(matches!(
        OneSidedLog2::<Fp>::new(8, 33, 6, &specs),
        Err(GadgetError::Shape { .. })
    ));
    assert!(matches!(
        OneSidedLog2::<Fp>::new(8, 16, 17, &specs),
        Err(GadgetError::Shape { .. })
    ));
    assert!(matches!(
        OneSidedLog2::<F65537>::new(8, 16, 6, &specs),
        Err(GadgetError::Shape { .. })
    ));

    let g = OneSidedLog2::<Fp>::new(8, 16, 6, &specs).unwrap();
    assert!(matches!(
        g.constraints::<F65537>(),
        Err(GadgetError::Shape { .. })
    ));

    let no_pow2: Vec<_> = specs
        .iter()
        .filter(|s| !matches!(s.kind, TableKind::Pow2 { .. }))
        .cloned()
        .collect();
    assert!(matches!(
        OneSidedLog2::<Fp>::new(8, 16, 6, &no_pow2),
        Err(GadgetError::Shape { .. })
    ));
    let no_exp2: Vec<_> = specs
        .iter()
        .filter(|s| !matches!(s.kind, TableKind::Exp2Frac { .. }))
        .cloned()
        .collect();
    assert!(matches!(
        OneSidedLog2::<Fp>::new(8, 16, 6, &no_exp2),
        Err(GadgetError::Shape { .. })
    ));
    let no_r15: Vec<_> = specs
        .iter()
        .filter(|s| s.kind != TableKind::Range { bits: 15 })
        .cloned()
        .collect();
    assert!(matches!(
        OneSidedLog2::<Fp>::new(8, 16, 6, &no_r15),
        Err(GadgetError::Shape { .. })
    ));

    for (vocab, sigma, eps_log2) in [(1, 0, 0), (1, 0, 16), (3, 15, 6), (1 << 18, 32, 6)] {
        let kinds = OneSidedLog2::<Fp>::table_kinds(vocab, sigma).unwrap();
        let minimal_specs: Vec<TableSpec> = kinds
            .iter()
            .enumerate()
            .map(|(i, k)| {
                GeneratedTable::new(TableId(i as u32), k.clone())
                    .unwrap()
                    .spec()
                    .clone()
            })
            .collect();
        let gadget = OneSidedLog2::<Fp>::new(vocab, sigma, eps_log2, &minimal_specs).unwrap();
        let c = gadget.constraints::<Fp>().unwrap();
        assert_eq!(gadget.tables().len(), minimal_specs.len());
        assert!(!c.committed.is_empty());
    }
}

#[test]
fn argmax_with_span_and_batched_bounded_mode_isolate_every_check() {
    let mut rng = SplitMix64::new(0xA46A_5155);
    let r1 = GeneratedTable::new(TableId(1), TableKind::Range { bits: 1 })
        .unwrap()
        .spec()
        .clone();
    let r8 = GeneratedTable::new(TableId(8), TableKind::Range { bits: 8 })
        .unwrap()
        .spec()
        .clone();
    let pow2 = GeneratedTable::new(TableId(9), TableKind::Pow2 { max_exp: 8 })
        .unwrap()
        .spec()
        .clone();

    assert!(Argmax::<Fp>::with_span(4, 0, &r1).is_ok());
    assert!(Argmax::<Fp>::with_span(4, 1, &r1).is_ok());
    assert!(matches!(
        Argmax::<Fp>::with_span(4, 2, &r1),
        Err(GadgetError::Table { .. })
    ));
    assert!(Argmax::<Fp>::with_span(5, 255, &r8).is_ok());
    assert!(matches!(
        Argmax::<Fp>::with_span(5, 256, &r8),
        Err(GadgetError::Table { .. })
    ));
    assert!(matches!(
        Argmax::<Fp>::with_span(5, 10, &pow2),
        Err(GadgetError::Table { .. })
    ));

    let argmax = Argmax::<Fp>::with_span(5, 255, &r8).unwrap();
    let tie_row = [
        Fp::from_i64(-10),
        Fp::from_i64(42),
        Fp::from_i64(42),
        Fp::from_i64(7),
        Fp::from_i64(42),
        Fp::ZERO,
        Fp::ZERO,
        Fp::ZERO,
    ];
    let (h0, w0) = argmax.sample(&tie_row, true).unwrap();
    assert_eq!(h0.indicator()[1], Fp::ONE);
    assert_eq!(w0.maximum()[0], Fp::from_i64(42));

    let row1 = [
        Fp::from_i64(100),
        Fp::from_i64(99),
        Fp::from_i64(0),
        Fp::from_i64(-50),
        Fp::from_i64(100),
        Fp::ZERO,
        Fp::ZERO,
        Fp::ZERO,
    ];
    let (h1, w1) = argmax.sample(&row1, true).unwrap();
    assert_eq!(h1.indicator()[0], Fp::ONE);

    let padded_rows = 4usize;
    let real_rows = 2usize;
    let pv = 8usize;
    let mut logits = vec![Fp::ZERO; padded_rows * pv];
    logits[..pv].copy_from_slice(&tie_row);
    logits[pv..2 * pv].copy_from_slice(&row1);
    let mut indicators = vec![Fp::ZERO; padded_rows * pv];
    indicators[..pv].copy_from_slice(h0.indicator());
    indicators[pv..2 * pv].copy_from_slice(h1.indicator());
    let maxima = [w0.maximum()[0], w1.maximum()[0], Fp::ZERO, Fp::ZERO];

    let selector = argmax.selector(padded_rows, real_rows).unwrap();
    let eval = argmax
        .evaluate_batch(&logits, &indicators, &maxima, padded_rows)
        .unwrap();

    let sampled = argmax
        .batched_sampled_relations::<Fp2>(&selector, &indicators)
        .unwrap();
    let rels = argmax
        .batched_relations::<Fp2>(&selector, &logits, &indicators, &maxima, &eval)
        .unwrap();

    let r_row = [
        Fp2::new(Fp::from_u64_reduced(5), Fp::from_u64_reduced(2)),
        Fp2::new(Fp::from_u64_reduced(9), Fp::from_u64_reduced(4)),
    ];
    let r_all = [
        Fp2::new(Fp::from_u64_reduced(3), Fp::ONE),
        Fp2::new(Fp::from_u64_reduced(7), Fp::ONE),
        Fp2::new(Fp::from_u64_reduced(11), Fp::ONE),
        r_row[0],
        r_row[1],
    ];
    let tv_r_row = tv_fp2s(&r_row);
    let tv_r_all = tv_fp2s(&r_all);

    verify_elementwise_check(
        &sampled.booleanity,
        &r_all,
        Fp2::ZERO,
        &[&indicators],
        3,
        &mut rng,
        |p, e| tv::eq(&tv_r_all, p).unwrap() * e[0] * (TvFp2::ONE - e[0]),
    )
    .unwrap();
    let inv_valid: Vec<Fp> = (0..padded_rows * pv)
        .map(|idx| Fp::from_u64_reduced(u64::from(idx % pv >= 5)))
        .collect();
    verify_elementwise_check(
        &sampled.vocabulary,
        &r_all,
        Fp2::ZERO,
        &[&indicators],
        3,
        &mut rng,
        |p, e| tv::eq(&tv_r_all, p).unwrap() * mle_fp(&inv_valid, p) * e[0],
    )
    .unwrap();
    let cnt_claim = fp2(mle_fp(selector.row_selector(), &tv_r_row));
    verify_elementwise_check(
        &sampled.count,
        &r_row,
        cnt_claim,
        &[&indicators],
        2,
        &mut rng,
        |p, e| tv::eq(&tv_r_row, &p[3..]).unwrap() * e[0],
    )
    .unwrap();
    let max_claim = fp2(mle_fp(&maxima, &tv_r_row));
    verify_elementwise_check(
        &rels.selected,
        &r_row,
        max_claim,
        &[&indicators, &logits],
        3,
        &mut rng,
        |p, e| tv::eq(&tv_r_row, &p[3..]).unwrap() * e[0] * e[1],
    )
    .unwrap();
    let valid_mask: Vec<Fp> = (0..padded_rows * pv)
        .map(|idx| Fp::from_u64_reduced(u64::from(idx % pv < 5)))
        .collect();
    let key_claim = fp2(mle_fp(&eval.keys, &tv_r_all));
    verify_elementwise_check(
        &rels.key,
        &r_all,
        key_claim,
        &[&maxima, &logits, &eval.suffix],
        3,
        &mut rng,
        |p, e| tv::eq(&tv_r_all, p).unwrap() * mle_fp(&valid_mask, p) * (e[0] - e[1] - e[2]),
    )
    .unwrap();
    check_lookups(&mut rng, &eval.lookups, std::slice::from_ref(&r8)).unwrap();

    let weights = argmax.batched_suffix_weights(&r_all, padded_rows).unwrap();
    let dot = weights
        .iter()
        .zip(&indicators)
        .fold(Fp2::ZERO, |acc, (&w, &h)| acc + Fp2::from(h) * w);
    assert_eq!(dot, fp2(mle_fp(&eval.suffix, &tv_r_all)));

    // Adversarial: picking higher-index tie (v = 2 instead of v = 1) produces -1 (mod P) at v = 1!
    let mut tie_cheat = indicators.clone();
    tie_cheat[1] = Fp::ZERO;
    tie_cheat[2] = Fp::ONE;
    let bad_eval = argmax
        .evaluate_batch(&logits, &tie_cheat, &maxima, padded_rows)
        .unwrap();
    assert!(!in_table(
        argmax.table(),
        bad_eval.lookups.groups()[0].keys()[1],
        None
    ));
}

#[test]
fn exact_512bit_reference_verifies_soundness_and_tightness_across_boundaries() {
    let mut rng = SplitMix64::new(0x1550_0017);
    let specs = log2_table_specs();

    for &(vocab, sigma) in &[
        (1usize, 0u8),
        (3, 1),
        (5, 8),
        (17, 15),
        (19, 16),
        (33, 20),
        (65, 32),
    ] {
        let padded_vocab = vocab.next_power_of_two();
        for eps_log2 in [0u8, 1, 2, 6, 12, 16] {
            let gadget = OneSidedLog2::<Fp>::new(vocab, sigma, eps_log2, &specs).unwrap();
            let limit = 1u64 << sigma;
            for case in 0..12 {
                let mut exps = vec![Fp::ZERO; padded_vocab];
                let mut z_sum = 0u128;
                for (v, slot) in exps.iter_mut().enumerate().take(vocab) {
                    let val = match (case, v) {
                        (0, 0) => 1,
                        (0, _) => 0,
                        (1, _) => limit,
                        (2, 0) => limit,
                        (2, _) => 0,
                        _ => rng.below(limit.saturating_add(1)),
                    };
                    *slot = Fp::from_u64_reduced(val);
                    z_sum += u128::from(val);
                }
                if z_sum == 0 {
                    exps[0] = Fp::ONE;
                    z_sum = 1;
                }
                let pred_tok = (rng.below(vocab as u64)) as i64;
                let rec_tok = if case % 3 == 0 {
                    pred_tok
                } else {
                    (rng.below(vocab as u64)) as i64
                };
                let rec_h = gadget.domain().witness(rec_tok, true).unwrap();
                let pred_h = gadget.domain().witness(pred_tok, true).unwrap();
                let w = gadget.witness(&exps, &rec_h, &pred_h, true).unwrap();
                let b_units = w.bound_units() as usize;

                let y_exp = u128::from(exps[rec_tok as usize].to_canonical_u64());
                let z_plus_v = z_sum + (vocab as u128);
                let two_k = 1u128 << eps_log2;
                let den_q = two_k * z_plus_v;
                let num_q = if rec_tok == pred_tok {
                    (two_k - 1) * z_plus_v + (y_exp + 1)
                } else {
                    y_exp + 1
                };

                let num_256 = big_pow_256(num_q);
                let den_256 = big_pow_256(den_q);
                let lhs = big_shl(&num_256, b_units);
                assert!(
                    big_cmp(&lhs, &den_256).is_ge(),
                    "OneSidedLog2 unsound at vocab={vocab}, sigma={sigma}, k={eps_log2}: B_t={b_units}"
                );

                let m_bit = w.value(Log2Column::MatchBit).unwrap();
                let i_val = w.value(Log2Column::IntExp).unwrap();
                let f_val = w.value(Log2Column::FracExp).unwrap();
                if eps_log2 == 0 {
                    assert_eq!(m_bit, 0);
                }
                if m_bit == 0 && (i_val > 0 || f_val >= 2) && b_units >= 2 {
                    let y1_256 = big_pow_256(y_exp + 1);
                    let lhs_tight = big_shl(&y1_256, b_units - 2);
                    assert!(
                        big_cmp(&lhs_tight, &den_256).is_lt(),
                        "OneSidedLog2 loose by >= 2^-7 bits at vocab={vocab}, sigma={sigma}, k={eps_log2}: B_t={b_units}"
                    );
                }
            }
        }
    }
}

struct BatchFixture {
    gadget: OneSidedLog2<Fp>,
    specs: Vec<TableSpec>,
    padded_rows: usize,
    real_rows: usize,
    weights: Vec<Fp>,
    recorded: Vec<Fp>,
    predicted: Vec<Fp>,
    columns: Vec<(Log2Column, Vec<Fp>)>,
    uhat: u64,
}

impl BatchFixture {
    fn new(eps_log2: u8) -> Self {
        let specs = log2_table_specs();
        let vocab = 5usize;
        let pv = 8usize;
        let sigma = 16u8;
        let gadget = OneSidedLog2::<Fp>::new(vocab, sigma, eps_log2, &specs).unwrap();
        let padded_rows = 4usize;
        let real_rows = 3usize;

        let row_exps = [
            [100u64, 500, 65536, 1200, 300, 0, 0, 0],
            [0, 65536, 65536, 65536, 65536, 0, 0, 0],
            [17, 65536, 9000, 4000, 1234, 0, 0, 0],
            [0; 8],
        ];
        let rec_tokens = [2i64, 0, 0, 0];
        let pred_tokens = [2i64, 1, 1, 0];

        let mut weights = Vec::with_capacity(padded_rows * pv);
        let mut recorded = Vec::with_capacity(padded_rows * pv);
        let mut predicted = Vec::with_capacity(padded_rows * pv);
        let c_names = gadget.constraints::<Fp>().unwrap().committed;
        let mut col_vecs: Vec<(Log2Column, Vec<Fp>)> = c_names
            .iter()
            .map(|&c| (c, vec![Fp::ZERO; padded_rows]))
            .collect();
        let mut uhat = 0u64;

        for r in 0..padded_rows {
            let real = r < real_rows;
            let exps_fp: Vec<Fp> = row_exps[r]
                .iter()
                .map(|&v| Fp::from_u64_reduced(v))
                .collect();
            let rec_h = gadget.domain().witness(rec_tokens[r], real).unwrap();
            let pred_h = gadget.domain().witness(pred_tokens[r], real).unwrap();
            let w = gadget.witness(&exps_fp, &rec_h, &pred_h, real).unwrap();
            for ((_, dst), &val) in col_vecs.iter_mut().zip(w.values()) {
                dst[r] = Fp::from_i64(val);
            }
            if real {
                uhat += w.bound_units();
            }
            weights.extend(exps_fp);
            recorded.extend_from_slice(rec_h.indicator());
            predicted.extend_from_slice(pred_h.indicator());
        }

        Self {
            gadget,
            specs,
            padded_rows,
            real_rows,
            weights,
            recorded,
            predicted,
            columns: col_vecs,
            uhat,
        }
    }

    fn clone_fixture(&self) -> Self {
        Self {
            gadget: self.gadget.clone(),
            specs: self.specs.clone(),
            padded_rows: self.padded_rows,
            real_rows: self.real_rows,
            weights: self.weights.clone(),
            recorded: self.recorded.clone(),
            predicted: self.predicted.clone(),
            columns: self.columns.clone(),
            uhat: self.uhat,
        }
    }

    fn batch_columns(&self) -> Log2BatchColumns<Fp> {
        Log2BatchColumns::from_columns(
            self.columns
                .iter()
                .map(|(c, vals)| {
                    let mut b = SecretBuf::with_capacity(vals.len());
                    b.extend(vals.iter().copied());
                    (*c, b)
                })
                .collect(),
        )
    }

    fn recompute_uhat(&self) -> Fp {
        let cols = self.batch_columns();
        let selector = self
            .gadget
            .selector(self.padded_rows, self.real_rows)
            .unwrap();
        let eval = self
            .gadget
            .evaluate_batch(&self.weights, &self.recorded, &cols, &selector)
            .unwrap();
        eval.uhat_field()
    }

    fn set_col(&mut self, col: Log2Column, row: usize, val: Fp) {
        self.columns
            .iter_mut()
            .find(|(c, _)| *c == col)
            .expect("column")
            .1[row] = val;
    }

    fn get_col(&self, col: Log2Column, row: usize) -> Fp {
        self.columns
            .iter()
            .find(|(c, _)| *c == col)
            .expect("column")
            .1[row]
    }

    fn verify_all(&self, rng: &mut SplitMix64, claimed_uhat: Fp) -> Result<(), &'static str> {
        let cols = self.batch_columns();
        let selector = self
            .gadget
            .selector(self.padded_rows, self.real_rows)
            .unwrap();
        let eval = self
            .gadget
            .evaluate_batch(&self.weights, &self.recorded, &cols, &selector)
            .unwrap();
        let vocab_rels = self
            .gadget
            .batched_vocab_relations::<Fp2>(
                &selector,
                &self.weights,
                &self.recorded,
                &self.predicted,
                &cols,
            )
            .unwrap();
        let eq_rels = self
            .gadget
            .equality_relations::<Fp2>(&selector, &eval, &cols)
            .unwrap();
        let bound_rel = self.gadget.bound_relation::<Fp2>(&selector, &cols).unwrap();
        let constraints = self.gadget.constraints::<Fp>().unwrap();

        let r_row = [
            Fp2::new(Fp::from_u64_reduced(13), Fp::from_u64_reduced(3)),
            Fp2::new(Fp::from_u64_reduced(29), Fp::from_u64_reduced(5)),
        ];
        let r_all = [
            Fp2::new(Fp::from_u64_reduced(3), Fp::ONE),
            Fp2::new(Fp::from_u64_reduced(7), Fp::ONE),
            Fp2::new(Fp::from_u64_reduced(19), Fp::ONE),
            r_row[0],
            r_row[1],
        ];
        let tv_r_row = tv_fp2s(&r_row);
        let tv_r_all = tv_fp2s(&r_all);

        let pv = 8usize;
        let valid_mask: Vec<Fp> = (0..self.padded_rows * pv)
            .map(|idx| Fp::from_u64_reduced(u64::from(idx % pv < 5)))
            .collect();

        verify_elementwise_check(
            &vocab_rels.total,
            &r_row,
            fp2(mle_fp(&eval.totals, &tv_r_row)),
            &[&self.weights],
            3,
            rng,
            |p, e| tv::eq(&tv_r_row, &p[3..]).unwrap() * mle_fp(&valid_mask, p) * e[0],
        )?;
        verify_elementwise_check(
            &vocab_rels.selected,
            &r_row,
            fp2(mle_fp(&eval.selecteds, &tv_r_row)),
            &[&self.recorded, &self.weights],
            3,
            rng,
            |p, e| tv::eq(&tv_r_row, &p[3..]).unwrap() * e[0] * e[1],
        )?;
        let m_col = cols.column(Log2Column::MatchBit).unwrap();
        verify_elementwise_check(
            &vocab_rels.match_indicator,
            &r_all,
            Fp2::ZERO,
            &[m_col, &self.recorded, &self.predicted],
            3,
            rng,
            |p, e| tv::eq(&tv_r_all, p).unwrap() * e[0] * (e[1] - e[2]),
        )?;

        let mut eq_operands: Vec<&[Fp]> = vec![&eval.totals, &eval.selecteds];
        for &c in &constraints.committed {
            eq_operands.push(cols.column(c).unwrap());
        }
        for (eq_idx, rel) in eq_rels.iter().enumerate() {
            verify_elementwise_check(rel, &r_row, Fp2::ZERO, &eq_operands, 3, rng, |p, e| {
                let psi = fp2(mle_fp(selector.row_selector(), p));
                let poly = constraints.equalities[eq_idx].evaluate(|term| match term {
                    Log2Term::RowSelector => psi,
                    Log2Term::Total => fp2(e[0]),
                    Log2Term::Selected => fp2(e[1]),
                    Log2Term::Column(col) => {
                        let pos = constraints
                            .committed
                            .iter()
                            .position(|&c| c == col)
                            .unwrap();
                        fp2(e[2 + pos])
                    }
                });
                tv::eq(&tv_r_row, p).unwrap() * tv_fp2(poly)
            })?;
        }

        let i_col = cols.column(Log2Column::IntExp).unwrap();
        let f_col = cols.column(Log2Column::FracExp).unwrap();
        let c_val = Fp::from_u64_reduced(self.gadget.c_match_units());
        let k_val = Fp::from_u64_reduced(256 * u64::from(self.gadget.eps_log2()));
        let s256 = Fp::from_u64_reduced(256);
        verify_elementwise_check(
            &bound_rel,
            &[],
            Fp2::from(claimed_uhat),
            &[m_col, i_col, f_col],
            3,
            rng,
            |p, e| {
                let psi = mle_fp(selector.row_selector(), p);
                let c_tv = TvFp2::from(tv_fp2(Fp2::from(c_val)));
                let k_tv = TvFp2::from(tv_fp2(Fp2::from(k_val)));
                let s_tv = TvFp2::from(tv_fp2(Fp2::from(s256)));
                let m = e[0];
                psi * (m * c_tv + (TvFp2::ONE - m) * (k_tv + s_tv * e[1] + e[2]))
            },
        )?;

        let lookups = self
            .gadget
            .evaluate_lookups(&selector, &eval, &cols)
            .unwrap();
        check_lookups(rng, &lookups, &self.specs).map_err(|_| "logup")?;
        Ok(())
    }
}

#[test]
fn honest_batch_verifies_and_every_adversarial_mutation_is_rejected_by_tv_core() {
    let mut rng = SplitMix64::new(0xC0DE_0155);
    let base = BatchFixture::new(6);
    assert_eq!(
        base.verify_all(&mut rng, Fp::from_u64_reduced(base.uhat)),
        Ok(())
    );

    // 1. False match m_t = 1 on mismatched row 1 -> rejected by sumcheck (match_indicator).
    let mut m1 = base.clone_fixture();
    m1.set_col(Log2Column::MatchBit, 1, Fp::ONE);
    let m1_uhat = m1.recompute_uhat();
    assert_eq!(m1.verify_all(&mut rng, m1_uhat), Err("sumcheck"));

    // 2. eps_log2 = 0 (pure surprisal) enforces m_t = 0 even on matched row 0!
    let base_k0 = BatchFixture::new(0);
    assert_eq!(
        base_k0.verify_all(&mut rng, Fp::from_u64_reduced(base_k0.uhat)),
        Ok(())
    );
    let mut k0_match = base_k0.clone_fixture();
    k0_match.set_col(Log2Column::MatchBit, 0, Fp::ONE);
    let k0_uhat = k0_match.recompute_uhat();
    assert_eq!(k0_match.verify_all(&mut rng, k0_uhat), Err("sumcheck"));

    // 3. Non-Boolean m_t = 2 on matched row 0 (match_indicator still 0!) -> m_t(1 - m_t) = 0 fails.
    let mut non_bool_m = base.clone_fixture();
    non_bool_m.set_col(Log2Column::MatchBit, 0, Fp::from_u64_reduced(2));
    let nb_uhat = non_bool_m.recompute_uhat();
    assert_eq!(non_bool_m.verify_all(&mut rng, nb_uhat), Err("sumcheck"));

    // 4. Claiming I_t - 1 with f_t = 255 on mismatched row 1 -> BigIntCompare rejects (X > Y).
    let orig_i = base.get_col(Log2Column::IntExp, 1).to_canonical_u64();
    assert!(orig_i >= 1);
    let mut small_i = base.clone_fixture();
    let new_i = orig_i - 1;
    let new_p = 1u64 << new_i;
    let max_e = 130_816u64;
    small_i.set_col(Log2Column::IntExp, 1, Fp::from_u64_reduced(new_i));
    small_i.set_col(Log2Column::Pow2Val, 1, Fp::from_u64_reduced(new_p));
    small_i.set_col(
        Log2Column::Pow2Limb(0),
        1,
        Fp::from_u64_reduced(new_p & 0xffff),
    );
    small_i.set_col(
        Log2Column::Pow2Limb(1),
        1,
        Fp::from_u64_reduced((new_p >> 16) & 0xffff),
    );
    small_i.set_col(
        Log2Column::Pow2Limb(2),
        1,
        Fp::from_u64_reduced((new_p >> 32) & 0xffff),
    );
    small_i.set_col(Log2Column::FracExp, 1, Fp::from_u64_reduced(255));
    small_i.set_col(Log2Column::Exp2Val, 1, Fp::from_u64_reduced(max_e));
    small_i.set_col(Log2Column::ScaledSelected, 1, Fp::from_u64_reduced(max_e));
    small_i.set_col(
        Log2Column::ScaledLimb(0),
        1,
        Fp::from_u64_reduced(max_e & 0xffff),
    );
    small_i.set_col(
        Log2Column::ScaledLimb(1),
        1,
        Fp::from_u64_reduced((max_e >> 16) & 0xffff),
    );
    let si_uhat = small_i.recompute_uhat();
    assert_eq!(small_i.verify_all(&mut rng, si_uhat), Err("sumcheck"));

    // 5. Claiming f_t - 1 at minimal I_t on row 2 -> BigIntCompare rejects.
    let orig_f2 = base.get_col(Log2Column::FracExp, 2).to_canonical_u64();
    assert!(orig_f2 >= 1);
    let mut small_f = base.clone_fixture();
    let new_f = orig_f2 - 1;
    let exp2_tbl = GeneratedTable::from_spec(&base.specs[1]).unwrap();
    let new_e = exp2_tbl.lookup(new_f as i64, 0).unwrap() as u64;
    let new_q = (17 + 1) * new_e;
    small_f.set_col(Log2Column::FracExp, 2, Fp::from_u64_reduced(new_f));
    small_f.set_col(Log2Column::Exp2Val, 2, Fp::from_u64_reduced(new_e));
    small_f.set_col(Log2Column::ScaledSelected, 2, Fp::from_u64_reduced(new_q));
    small_f.set_col(
        Log2Column::ScaledLimb(0),
        2,
        Fp::from_u64_reduced(new_q & 0xffff),
    );
    small_f.set_col(
        Log2Column::ScaledLimb(1),
        2,
        Fp::from_u64_reduced((new_q >> 16) & 0xffff),
    );
    let sf_uhat = small_f.recompute_uhat();
    assert_eq!(small_f.verify_all(&mut rng, sf_uhat), Err("sumcheck"));

    // 6. Forged q_t (ScaledSelected + 1) -> q_t - (Y_t + psi)*E_t = 0 equality fails.
    let mut forged_q = base.clone_fixture();
    let q2 = forged_q.get_col(Log2Column::ScaledSelected, 2);
    forged_q.set_col(Log2Column::ScaledSelected, 2, q2 + Fp::ONE);
    let fq_uhat = forged_q.recompute_uhat();
    assert_eq!(forged_q.verify_all(&mut rng, fq_uhat), Err("sumcheck"));

    // 7. Forged Pow2Val (not matching 2^{I_t}) -> Pow2 LogUp fails.
    let mut forged_pow2 = base.clone_fixture();
    let p1 = forged_pow2
        .get_col(Log2Column::Pow2Val, 1)
        .to_canonical_u64();
    let fake_p = p1 * 2;
    forged_pow2.set_col(Log2Column::Pow2Val, 1, Fp::from_u64_reduced(fake_p));
    let fp_cols = forged_pow2.batch_columns();
    let fp_sel = forged_pow2
        .gadget
        .selector(forged_pow2.padded_rows, forged_pow2.real_rows)
        .unwrap();
    let fp_eval = forged_pow2
        .gadget
        .evaluate_batch(
            &forged_pow2.weights,
            &forged_pow2.recorded,
            &fp_cols,
            &fp_sel,
        )
        .unwrap();
    let fp_lookups = forged_pow2
        .gadget
        .evaluate_lookups(&fp_sel, &fp_eval, &fp_cols)
        .unwrap();
    assert!(check_lookups(&mut rng, &fp_lookups, &forged_pow2.specs).is_err());

    // 8. Forged Pow2Limb(0) without changing Pow2Val -> virtual P_3 Range(15) LogUp rejects!
    let mut forged_plimb = base.clone_fixture();
    let pl0 = forged_plimb.get_col(Log2Column::Pow2Limb(0), 0);
    forged_plimb.set_col(Log2Column::Pow2Limb(0), 0, pl0 + Fp::ONE);
    let pl_cols = forged_plimb.batch_columns();
    let pl_sel = forged_plimb
        .gadget
        .selector(forged_plimb.padded_rows, forged_plimb.real_rows)
        .unwrap();
    let pl_eval = forged_plimb
        .gadget
        .evaluate_batch(
            &forged_plimb.weights,
            &forged_plimb.recorded,
            &pl_cols,
            &pl_sel,
        )
        .unwrap();
    let pl_lookups = forged_plimb
        .gadget
        .evaluate_lookups(&pl_sel, &pl_eval, &pl_cols)
        .unwrap();
    assert!(check_lookups(&mut rng, &pl_lookups, &forged_plimb.specs).is_err());

    // 9. Nonzero Z_t on padded row 3 -> (psi_t - 1) * Z_t = 0 equality rejects!
    let mut bad_pad = base.clone_fixture();
    bad_pad.weights[3 * 8] = Fp::ONE;
    let bp_uhat = bad_pad.recompute_uhat();
    assert_eq!(bad_pad.verify_all(&mut rng, bp_uhat), Err("sumcheck"));

    // 10. Claiming a smaller public Û_r (uhat - 1) -> bound_relation sumcheck rejects!
    assert_eq!(
        base.verify_all(&mut rng, Fp::from_u64_reduced(base.uhat - 1)),
        Err("sumcheck")
    );

    // 11. Tampering BigIntCompare final borrow -> sumcheck rejects!
    let mut bad_borrow = base.clone_fixture();
    let m = base.gadget.comparison_limbs();
    bad_borrow.set_col(
        Log2Column::Compare(CompareColumn::Borrow(m - 1)),
        1,
        Fp::ONE,
    );
    let bb_uhat = bad_borrow.recompute_uhat();
    assert_eq!(bad_borrow.verify_all(&mut rng, bb_uhat), Err("sumcheck"));

    // 12. Tampering BigIntCompare LeftCarry -> sumcheck rejects!
    let mut bad_carry = base.clone_fixture();
    bad_carry.set_col(
        Log2Column::Compare(CompareColumn::Carry(Side::Left, 0)),
        1,
        Fp::ONE,
    );
    let bc_uhat = bad_carry.recompute_uhat();
    assert_eq!(bad_carry.verify_all(&mut rng, bc_uhat), Err("sumcheck"));
}

#[test]
fn gibbs_inequality_smoke_test_average_uhat_at_least_shannon_entropy() {
    let mut rng = SplitMix64::new(0x61BB_5017);
    let specs = log2_table_specs();
    let vocab = 4usize;
    let sigma = 12u8;
    let eps_log2 = 4u8;
    let gadget = OneSidedLog2::<Fp>::new(vocab, sigma, eps_log2, &specs).unwrap();

    let exps = [
        Fp::from_u64_reduced(2048),
        Fp::from_u64_reduced(1024),
        Fp::from_u64_reduced(512),
        Fp::from_u64_reduced(512),
    ];
    let pred_h = gadget.domain().witness(0, true).unwrap();

    let numerators = [63_549u64, 1_025, 513, 513];
    let denom = 65_600u64;
    assert_eq!(numerators.iter().sum::<u64>(), denom);

    let trials = 4_000u64;
    let mut total_units = 0u64;
    let mut counts = [0u64; 4];
    for _ in 0..trials {
        let draw = rng.below(denom);
        let mut acc = 0u64;
        let mut tok = 0usize;
        for (v, &num) in numerators.iter().enumerate() {
            acc += num;
            if draw < acc {
                tok = v;
                break;
            }
        }
        counts[tok] += 1;
        let rec_h = gadget.domain().witness(tok as i64, true).unwrap();
        let w = gadget.witness(&exps, &rec_h, &pred_h, true).unwrap();
        total_units += w.bound_units();
    }

    let avg_uhat_bits = (total_units as f64) / 256.0 / (trials as f64);
    let true_entropy: f64 = numerators
        .iter()
        .map(|&n| {
            let p = (n as f64) / (denom as f64);
            -p * p.log2()
        })
        .sum();
    let empirical_nll: f64 = counts
        .iter()
        .zip(&numerators)
        .map(|(&c, &n)| {
            let p = (n as f64) / (denom as f64);
            ((c as f64) / (trials as f64)) * (-p.log2())
        })
        .sum();

    assert!(
        avg_uhat_bits >= empirical_nll,
        "average Û ({avg_uhat_bits}) fell below empirical code length ({empirical_nll})"
    );
    assert!(
        avg_uhat_bits + 1e-3 >= true_entropy,
        "average Û ({avg_uhat_bits}) violated Gibbs' inequality against H(q) = {true_entropy}"
    );
}

#[test]
fn secret_witness_and_batch_debug_redact_all_values() {
    let specs = log2_table_specs();
    let gadget = OneSidedLog2::<Fp>::new(4, 16, 6, &specs).unwrap();
    let exps = [
        Fp::from_u64_reduced(12_345),
        Fp::from_u64_reduced(54_321),
        Fp::from_u64_reduced(7_777),
        Fp::from_u64_reduced(9_999),
    ];
    let rec_h = gadget.domain().witness(2, true).unwrap();
    let pred_h = gadget.domain().witness(1, true).unwrap();
    let w = gadget.witness(&exps, &rec_h, &pred_h, true).unwrap();
    let w_dbg = format!("{w:?}");
    let secret_bound = w.bound_units().to_string();
    assert!(!w_dbg.contains(&secret_bound));
    assert!(!w_dbg.contains("65536"));

    let mut cols_vec = Vec::new();
    for (&col, &val) in w.columns().iter().zip(w.values()) {
        let mut b = SecretBuf::with_capacity(1);
        b.push(Fp::from_i64(val));
        cols_vec.push((col, b));
    }
    let batch_cols = Log2BatchColumns::from_columns(cols_vec);
    let cols_dbg = format!("{batch_cols:?}");
    assert!(!cols_dbg.contains("65536"));

    let sel = gadget.selector(1, 1).unwrap();
    let eval = gadget
        .evaluate_batch(&exps, rec_h.indicator(), &batch_cols, &sel)
        .unwrap();
    let eval_dbg = format!("{eval:?}");
    assert!(!eval_dbg.contains(&secret_bound));
    assert!(!eval_dbg.contains("84342"));
}

#[test]
fn argmax_with_span_witness_rejects_out_of_span_logits() {
    let range4 = GeneratedTable::new(TableId(0), TableKind::Range { bits: 4 })
        .unwrap()
        .spec()
        .clone();
    let argmax = Argmax::<Fp>::with_span(4, 15, &range4).unwrap();

    // Span 15 fits Range(4) (`[0, 16)`); span 16 (`key[1] = 16`) and tie-offset overflow
    // (`logits = [0, 15, 0, 0]` => `best = 1`, `key[0]` = `15 - 0 - 1 = 14 < 16` fits, whereas
    // `logits = [0, 16, 0, 0]` => `key[2] = 16` overflows) must be rejected by both `sample` and
    // `witness` (SP-7).
    let ok_logits = [Fp::from_i64(15), Fp::ZERO, Fp::ZERO, Fp::ZERO];
    assert!(argmax.sample(&ok_logits, true).is_ok());
    assert!(argmax.witness(&ok_logits, true).is_ok());

    let overflow_logits = [Fp::from_i64(16), Fp::ZERO, Fp::ZERO, Fp::ZERO];
    assert!(matches!(
        argmax.sample(&overflow_logits, true),
        Err(GadgetError::Input { .. })
    ));
    assert!(matches!(
        argmax.witness(&overflow_logits, true),
        Err(GadgetError::Input { .. })
    ));
}
