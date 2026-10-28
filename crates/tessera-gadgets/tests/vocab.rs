// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Adversarial, boundary, and end-to-end tests for [`VocabMax`] and batched [`InverseCdf`]
//! (RFC-0009 Lemmas 5 & 10, RFC-0017 'Vocabulary softmax', RFC-0030, RFC-0035, RFC-0058).

mod common;

use common::{
    Fp, Fp2, Messages, Rejected as LogUpRejected, Session, Shape, SplitMix64, TvFp2, fp2, in_table,
    mle_fp, prove, spec_of, table_columns, tv_fp2, tv_fp2s, verify,
};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use tessera_codec::{Rational, SecretBuf, Zeroize};
use tessera_field::{F65537, Field, PrimeField, QuadraticExtension};
use tessera_gadgets::{
    CdfBatchColumns, CdfColumn, GadgetError, InverseCdf, VocabColumn, VocabMax, VocabTerm,
    VocabValue, VocabWitness,
};
use tessera_sumcheck::{Elementwise, Instance, Prefix, Protocol, Table as ScTable, prove_batch};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};
use tv_core::logup::LookupError;
use tv_core::sumcheck as tv;

fn exp_spec(id: u32, key_bits: u8, sigma: u8, scale: Rational) -> TableSpec {
    GeneratedTable::new(
        TableId(id),
        TableKind::Exp {
            key_bits,
            sigma,
            scale,
        },
    )
    .expect("valid Exp table")
    .spec()
    .clone()
}

fn range_specs() -> Vec<TableSpec> {
    (1..=32)
        .map(|bits| {
            GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                .expect("Range table")
                .spec()
                .clone()
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
enum VocabRejected {
    Lookup,
    Root(usize),
    Padding(usize),
}

fn vocab_rejected(
    gadget: &VocabMax<Fp>,
    logits: &[Fp],
    padded_rows: usize,
    real_rows: usize,
    witness: &VocabWitness<Fp>,
) -> Vec<VocabRejected> {
    let eval = gadget
        .evaluate_batch(logits, padded_rows, real_rows, witness)
        .expect("shape");
    let mut out = Vec::new();
    let group = &eval.lookups.groups()[0];
    let vals = group.values().expect("Exp has values");
    for (&key, &val) in group.keys().iter().zip(vals.iter()) {
        if !in_table(gadget.table(), key, Some(val)) && !out.contains(&VocabRejected::Lookup) {
            out.push(VocabRejected::Lookup);
        }
    }
    let pv = gadget.padded_vocab();
    for row in 0..padded_rows {
        let prod = eval.leaves[row * pv..(row + 1) * pv]
            .iter()
            .copied()
            .fold(Fp::ONE, |p, x| p * x);
        if prod != eval.roots[row] {
            out.push(VocabRejected::Root(row));
        }
        if eval.padding[row] != Fp::ZERO {
            out.push(VocabRejected::Padding(row));
        }
    }
    out
}

#[test]
fn constructor_enforces_domain_field_table_kind_sigma_and_zero_tail() {
    let good = exp_spec(10, 6, 16, Rational::new(1, 1).unwrap());
    assert!(matches!(
        VocabMax::<Fp>::new(0, &good),
        Err(GadgetError::Shape { .. })
    ));
    assert!(matches!(
        VocabMax::<Fp>::new((1 << 18) + 1, &good),
        Err(GadgetError::Shape { .. })
    ));
    assert!(VocabMax::<Fp>::new(1, &good).is_ok());
    assert!(VocabMax::<Fp>::new(1 << 18, &good).is_ok());

    // Small field P <= 2^30 is refused so c - ell cannot wrap modulo P.
    assert!(matches!(
        VocabMax::<F65537>::new(4, &good),
        Err(GadgetError::Shape { .. })
    ));

    // Over Mersenne31 (P = 2^31 - 1 > 2^30):
    // - key_bits = 12, sigma = 16 has 2^12 + 2^29 <= P and 2^16 <= (P - 1) / 2 -> accepted;
    // - key_bits = 31, sigma = 16 has 2^31 + 2^29 > P -> refused (prevents negative c - ell
    //   wrapping into Exp_T's zero tail);
    // - key_bits = 12, sigma = 30 has 2^30 > (P - 1) / 2 = 2^30 - 1 -> refused (Lemma 0).
    const M31_P: u64 = (1 << 31) - 1;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct M31(u64);
    impl Zeroize for M31 {
        fn zeroize(&mut self) {
            self.0.zeroize();
        }
    }
    impl Add for M31 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self((self.0 + rhs.0) % M31_P)
        }
    }
    impl Sub for M31 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self((self.0 + M31_P - rhs.0) % M31_P)
        }
    }
    impl Mul for M31 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self((self.0 * rhs.0) % M31_P)
        }
    }
    impl Neg for M31 {
        type Output = Self;
        fn neg(self) -> Self {
            Self((M31_P - self.0) % M31_P)
        }
    }
    impl AddAssign for M31 {
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }
    impl SubAssign for M31 {
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }
    impl MulAssign for M31 {
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }
    impl Field for M31 {
        const ZERO: Self = Self(0);
        const ONE: Self = Self(1);
        fn square(&self) -> Self {
            *self * *self
        }
        fn double(&self) -> Self {
            *self + *self
        }
        fn inverse(&self) -> Option<Self> {
            (self.0 != 0).then(|| self.pow(M31_P - 2))
        }
        fn pow(&self, mut exp: u64) -> Self {
            let (mut acc, mut sq) = (Self::ONE, *self);
            while exp > 0 {
                if exp & 1 == 1 {
                    acc *= sq;
                }
                sq = sq.square();
                exp >>= 1;
            }
            acc
        }
    }
    impl PrimeField for M31 {
        const MODULUS: u64 = M31_P;
        const TWO_ADICITY: u32 = 1;
        fn from_u64_reduced(x: u64) -> Self {
            Self(x % M31_P)
        }
        fn from_i64(x: i64) -> Self {
            Self(x.rem_euclid(M31_P as i64) as u64)
        }
        fn from_canonical_u64(x: u64) -> Option<Self> {
            (x < M31_P).then_some(Self(x))
        }
        fn to_canonical_u64(&self) -> u64 {
            self.0
        }
        fn to_centered_i128(&self) -> i128 {
            if self.0 > (M31_P - 1) / 2 {
                self.0 as i128 - M31_P as i128
            } else {
                self.0 as i128
            }
        }
        fn two_adic_generator(log_n: u32) -> Self {
            assert!(log_n <= 1);
            if log_n == 0 {
                Self::ONE
            } else {
                Self(M31_P - 1)
            }
        }
    }
    let m31_ok = exp_spec(1001, 12, 16, Rational::new(1, 1).unwrap());
    assert!(VocabMax::<M31>::new(4, &m31_ok).is_ok());
    let m31_wide_keys = exp_spec(1002, 31, 16, Rational::new(1, 1).unwrap());
    assert!(matches!(
        VocabMax::<M31>::new(4, &m31_wide_keys),
        Err(GadgetError::Shape { .. })
    ));
    let m31_wide_sigma = exp_spec(1003, 12, 30, Rational::new(1, 1).unwrap());
    assert!(matches!(
        VocabMax::<M31>::new(4, &m31_wide_sigma),
        Err(GadgetError::Shape { .. })
    ));

    // Non-Exp table is refused.
    let range = GeneratedTable::new(TableId(11), TableKind::Range { bits: 8 })
        .unwrap()
        .spec()
        .clone();
    assert!(matches!(
        VocabMax::<Fp>::new(4, &range),
        Err(GadgetError::Table {
            table: TableId(11),
            ..
        })
    ));

    // sigma > 32 is refused before table regeneration.
    let mut bad_sigma = good.clone();
    bad_sigma.kind = TableKind::Exp {
        key_bits: 6,
        sigma: 33,
        scale: Rational::new(1, 1).unwrap(),
    };
    assert!(matches!(
        VocabMax::<Fp>::new(4, &bad_sigma),
        Err(GadgetError::Shape { .. })
    ));

    // Exp table whose last key K_0 does not yet reach zero is refused (RFC-0030).
    let no_zero_tail = exp_spec(12, 4, 16, Rational::new(1, 4).unwrap());
    let regen = GeneratedTable::from_spec(&no_zero_tail).unwrap();
    assert_ne!(regen.row(regen.row_count() - 1).unwrap().1, Fp::ZERO);
    assert!(matches!(
        VocabMax::<Fp>::new(4, &no_zero_tail),
        Err(GadgetError::Table {
            table: TableId(12),
            ..
        })
    ));
}

#[test]
fn honest_witness_and_constraints_agree_across_shapes_and_refuse_out_of_domain_logits() {
    let spec = exp_spec(20, 8, 24, Rational::new(1, 2).unwrap());
    for vocab in [1, 3, 4, 7, 13] {
        let gadget = VocabMax::<Fp>::new(vocab, &spec).unwrap();
        let constraints = gadget.constraints::<Fp>();
        assert_eq!(
            constraints.committed,
            vec![VocabColumn::Exponentials, VocabColumn::Maximum]
        );
        let pv = gadget.padded_vocab();

        // Real row with negative, tied, and positive logits near +-2^28.
        let base = (1i64 << 28) - 20;
        let mut raw = vec![0i64; pv];
        for (v, slot) in raw.iter_mut().take(vocab).enumerate() {
            *slot = base - ((v % 3) as i64) * 5;
        }
        let logits: Vec<Fp> = raw.iter().map(|&x| Fp::from_i64(x)).collect();
        let witness = gadget.witness(&logits, true).unwrap();
        assert_eq!(witness.maximum(), &[Fp::from_i64(base)]);
        assert_eq!(witness.exponentials()[0], Fp::from_u64_reduced(1 << 24));
        for v in vocab..pv {
            assert_eq!(witness.exponentials()[v], Fp::ZERO);
        }
        assert!(vocab_rejected(&gadget, &logits, 1, 1, &witness).is_empty());

        let eval = gadget.evaluate(&logits, true, &witness).unwrap();
        for (v, &logit_v) in logits.iter().enumerate().take(pv) {
            let active = Fp::from_u64_reduced(u64::from(v < vocab));
            let val = |term: VocabTerm| match term {
                VocabTerm::Logit => logit_v,
                VocabTerm::Column(VocabColumn::Maximum) => witness.maximum()[0],
                VocabTerm::Column(VocabColumn::Exponentials) => witness.exponentials()[v],
                VocabTerm::Virtual(VocabValue::Key) => eval.keys[v],
                VocabTerm::Virtual(VocabValue::Leaf) => eval.leaves[v],
                VocabTerm::Virtual(VocabValue::PaddingMaximum) => eval.padding[0],
                VocabTerm::Active => active,
                VocabTerm::RowSelector => Fp::ONE,
            };
            assert_eq!(constraints.relations[0].evaluate(val), eval.keys[v]);
            assert_eq!(constraints.relations[1].evaluate(val), eval.leaves[v]);
            assert_eq!(constraints.relations[2].evaluate(val), Fp::ZERO);
        }

        // Padded row: all zeros -> c = 0, e = 0, root = 1.
        let zeros = vec![Fp::ZERO; pv];
        let pad_w = gadget.witness(&zeros, false).unwrap();
        assert_eq!(pad_w.maximum(), &[Fp::ZERO]);
        assert!(pad_w.exponentials().iter().all(|&x| x == Fp::ZERO));
        assert!(vocab_rejected(&gadget, &zeros, 1, 0, &pad_w).is_empty());

        // Out-of-domain inputs refused.
        let mut too_large = logits.clone();
        too_large[0] = Fp::from_i64(1 << 28);
        assert!(matches!(
            gadget.witness(&too_large, true),
            Err(GadgetError::Input { .. })
        ));
        if vocab < pv {
            let mut dirty_vocab_pad = logits.clone();
            dirty_vocab_pad[vocab] = Fp::ONE;
            assert!(matches!(
                gadget.witness(&dirty_vocab_pad, true),
                Err(GadgetError::Input { .. })
            ));
        }
        let mut dirty_row_pad = zeros;
        dirty_row_pad[0] = Fp::ONE;
        assert!(matches!(
            gadget.witness(&dirty_row_pad, false),
            Err(GadgetError::Input { .. })
        ));
        if vocab >= 2 {
            let mut wide_spread = logits;
            wide_spread[1] = Fp::from_i64(base - 500);
            assert!(matches!(
                gadget.witness(&wide_spread, true),
                Err(GadgetError::Input { .. })
            ));
        }
    }
}

#[test]
fn each_obligation_alone_rejects_its_targeted_forgery() {
    let spec = exp_spec(30, 6, 16, Rational::new(1, 1).unwrap());
    let gadget = VocabMax::<Fp>::new(5, &spec).unwrap();
    let logits: Vec<Fp> = [7i64, 10, 9, 4, 8, 0, 0, 0]
        .into_iter()
        .map(Fp::from_i64)
        .collect();
    let honest = gadget.witness(&logits, true).unwrap();
    assert!(vocab_rejected(&gadget, &logits, 1, 1, &honest).is_empty());

    // 1. Over-claimed maximum c' = 11: keys [4, 1, 2, 7, 3] are all valid Exp_T rows, so only Root
    //    rejects.
    let exp_hi: SecretBuf<Fp> = [7i64, 10, 9, 4, 8, 0, 0, 0]
        .into_iter()
        .enumerate()
        .map(|(v, ell)| {
            if v < 5 {
                Fp::from_i64(gadget.table().lookup(11 - ell, 0).unwrap())
            } else {
                Fp::ZERO
            }
        })
        .collect();
    let over_max = VocabWitness::from_columns([Fp::from_i64(11)].into_iter().collect(), exp_hi);
    assert_eq!(
        vocab_rejected(&gadget, &logits, 1, 1, &over_max),
        vec![VocabRejected::Root(0)]
    );

    // 2. Under-claimed maximum c' = 9 (= logit[2]): leaf product vanishes at v = 2, so Root and
    //    Padding pass and only Lookup rejects (key -1 = P - 1 at v = 1).
    let exp_lo: SecretBuf<Fp> = [7i64, 10, 9, 4, 8, 0, 0, 0]
        .into_iter()
        .enumerate()
        .map(|(v, ell)| {
            if v < 5 && 9 >= ell {
                Fp::from_i64(gadget.table().lookup(9 - ell, 0).unwrap())
            } else {
                Fp::ZERO
            }
        })
        .collect();
    let under_max = VocabWitness::from_columns([Fp::from_i64(9)].into_iter().collect(), exp_lo);
    assert_eq!(
        vocab_rejected(&gadget, &logits, 1, 1, &under_max),
        vec![VocabRejected::Lookup]
    );

    // 3. Forged active exponential: Root and Padding pass, Lookup alone rejects.
    let mut bad_active: SecretBuf<Fp> = honest.exponentials().iter().copied().collect();
    bad_active[0] += Fp::ONE;
    let forged_active =
        VocabWitness::from_columns(honest.maximum().iter().copied().collect(), bad_active);
    assert_eq!(
        vocab_rejected(&gadget, &logits, 1, 1, &forged_active),
        vec![VocabRejected::Lookup]
    );

    // 4. Forged vocabulary-padded exponential (v = 5 >= vocab): key is K_0 where Exp_T[K_0] = 0,
    //    so Lookup alone rejects.
    let mut bad_vpad: SecretBuf<Fp> = honest.exponentials().iter().copied().collect();
    bad_vpad[5] = Fp::ONE;
    let forged_vpad =
        VocabWitness::from_columns(honest.maximum().iter().copied().collect(), bad_vpad);
    assert_eq!(
        vocab_rejected(&gadget, &logits, 1, 1, &forged_vpad),
        vec![VocabRejected::Lookup]
    );

    // 5. Forged padded-row maximum c_0 = 42 with all exponentials 0: C_{0,v} = 0 makes every key
    //    K_0 (e = 0 passes Lookup) and every leaf 1 (product 1 passes Root), so Padding(0) alone
    //    rejects — both in evaluate_batch AND in constraints().relations[2]!
    let zeros = vec![Fp::ZERO; 8];
    let forged_pad = VocabWitness::from_columns(
        [Fp::from_i64(42)].into_iter().collect(),
        SecretBuf::filled(Fp::ZERO, 8),
    );
    assert_eq!(
        vocab_rejected(&gadget, &zeros, 1, 0, &forged_pad),
        vec![VocabRejected::Padding(0)]
    );
    let pad_eval = gadget.evaluate(&zeros, false, &forged_pad).unwrap();
    let pad_constraints = gadget.constraints::<Fp>();
    let pad_val = pad_constraints.relations[2].evaluate(|t| match t {
        VocabTerm::Column(VocabColumn::Maximum) => forged_pad.maximum()[0],
        VocabTerm::RowSelector => Fp::ZERO,
        _ => Fp::ZERO,
    });
    assert_eq!(pad_val, Fp::from_i64(42));
    assert_eq!(pad_val, pad_eval.padding[0]);
}

fn check_elementwise(
    relation: &Elementwise<'_, Fp2>,
    point: &[Fp2],
    target: Fp2,
    operands: &[&[Fp]],
    degree: usize,
    rng: &mut SplitMix64,
    mut verifier_eval: impl FnMut(&[TvFp2], &[TvFp2]) -> TvFp2,
) {
    let vars = relation.shape().vars;
    let tables = relation.tables(point).expect("tables");
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
    .expect("prove_batch");
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
    .expect("verify_batch");
    for (&val, col) in proof.final_evals.iter().zip(operands) {
        let p = checked
            .point_of(col.len().trailing_zeros() as usize)
            .expect("sub-point");
        assert_eq!(val, mle_fp(col, p));
    }
    assert_eq!(pt.state(), *vt.state());
}

#[test]
fn batched_vocab_max_and_inverse_cdf_prove_sumcheck_and_logup_and_reject_padded_cheat() {
    let mut rng = SplitMix64::new(0x1520_0058);
    let exp = exp_spec(100, 6, 16, Rational::new(1, 1).unwrap());
    let ranges = range_specs();
    let vocab = 5usize;
    let pv = 8usize;
    let padded_rows = 4usize;
    let real_rows = 3usize;

    let vocab_max = VocabMax::<Fp>::new(vocab, &exp).unwrap();
    let raw_rows: [[i64; 8]; 4] = [
        [12, 15, 11, 14, 9, 0, 0, 0],
        [-5, -2, -8, -3, -2, 0, 0, 0],
        [0, 3, 1, 2, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ];
    let draws: Vec<Fp> = [0x1234_5678u64, 0x9abc_def0, 0xffff_fffe, 0]
        .into_iter()
        .map(Fp::from_u64_reduced)
        .collect();

    let mut flat_logits = Vec::with_capacity(padded_rows * pv);
    let mut maxima = SecretBuf::with_capacity(padded_rows);
    let mut exps = SecretBuf::with_capacity(padded_rows * pv);
    for (row_idx, raw) in raw_rows.iter().enumerate() {
        let row_fp: Vec<Fp> = raw.iter().map(|&x| Fp::from_i64(x)).collect();
        let w = vocab_max.witness(&row_fp, row_idx < real_rows).unwrap();
        flat_logits.extend_from_slice(&row_fp);
        maxima.push(w.maximum()[0]);
        exps.extend(w.exponentials().iter().copied());
    }
    let vw = VocabWitness::from_columns(maxima, exps);
    let veval = vocab_max
        .evaluate_batch(&flat_logits, padded_rows, real_rows, &vw)
        .unwrap();
    let vsel = vocab_max.selector(padded_rows, real_rows).unwrap();
    let vrel = vocab_max
        .relations::<Fp2>(&vsel, &flat_logits, &vw)
        .unwrap();

    // Prove VocabMax's Exp_T lookups through LogUp and verify with tv-core.
    let exp_tables = [vocab_max.table()];
    let exp_cols = table_columns(&exp_tables);
    let mults = veval.lookups.multiplicities(&exp_cols).unwrap();
    let instances = veval.lookups.instances(&exp_cols, &mults).unwrap();
    let shapes = vec![Shape::of(
        &instances[0],
        spec_of(&exp_tables, &veval.lookups, 0),
    )];
    let session = Session::random(&mut rng);
    let proven = prove(&session, &instances);
    let leaf_claims = verify(&session, &shapes, &Messages::of(&proven)).unwrap();
    assert_eq!(
        leaf_claims.instances[0].key.value,
        mle_fp(&veval.keys, &leaf_claims.instances[0].key.point)
    );

    // Prove VocabMax's key, leaf, and padding Elementwise relations against tv-core sumcheck.
    let pt5: Vec<Fp2> = (0..5)
        .map(|i| {
            Fp2::new(
                Fp::from_u64_reduced(3 + i),
                Fp::from_u64_reduced(17 + 5 * i),
            )
        })
        .collect();
    let tv_pt5 = tv_fp2s(&pt5);
    let key_claim = fp2(mle_fp(&veval.keys, &tv_pt5));
    let k0_fp = Fp::from_u64_reduced(vocab_max.masked_key());
    let active_fp: Vec<Fp> = (0..padded_rows * pv)
        .map(|idx| Fp::from_u64_reduced(u64::from(idx / pv < real_rows && idx % pv < vocab)))
        .collect();
    let inactive_k0: Vec<Fp> = active_fp
        .iter()
        .map(|&a| if a == Fp::ONE { Fp::ZERO } else { k0_fp })
        .collect();
    check_elementwise(
        &vrel.key,
        &pt5,
        key_claim,
        &[vw.maximum(), &flat_logits],
        3,
        &mut rng,
        |p, e| {
            tv::eq(&tv_pt5, p).unwrap()
                * (mle_fp(&active_fp, p) * (e[0] - e[1]) + mle_fp(&inactive_k0, p))
        },
    );
    // Pin C_{t,v} = psi_t * [v < vocab] on inactive entries: placing non-zero garbage on
    // v >= vocab and row >= real_rows in the logit operand leaves key_claim and leaf_claim
    // completely unchanged because C_{t,v} annihilates inactive (c_t - ell_{t,v}) terms!
    let mut dirty_inactive_logits = flat_logits.clone();
    for (idx, slot) in dirty_inactive_logits.iter_mut().enumerate() {
        if idx / pv >= real_rows || idx % pv >= vocab {
            *slot = Fp::from_i64(777 + idx as i64);
        }
    }
    let dirty_vrel = vocab_max
        .relations::<Fp2>(&vsel, &dirty_inactive_logits, &vw)
        .unwrap();
    let leaf_claim = fp2(mle_fp(&veval.leaves, &tv_pt5));
    let inactive_one: Vec<Fp> = active_fp
        .iter()
        .map(|&a| if a == Fp::ONE { Fp::ZERO } else { Fp::ONE })
        .collect();
    check_elementwise(
        &dirty_vrel.key,
        &pt5,
        key_claim,
        &[vw.maximum(), &dirty_inactive_logits],
        3,
        &mut rng,
        |p, e| {
            tv::eq(&tv_pt5, p).unwrap()
                * (mle_fp(&active_fp, p) * (e[0] - e[1]) + mle_fp(&inactive_k0, p))
        },
    );
    check_elementwise(
        &dirty_vrel.leaf,
        &pt5,
        leaf_claim,
        &[vw.maximum(), &dirty_inactive_logits],
        3,
        &mut rng,
        |p, e| {
            tv::eq(&tv_pt5, p).unwrap()
                * (mle_fp(&active_fp, p) * (e[0] - e[1]) + mle_fp(&inactive_one, p))
        },
    );
    let pt2 = &pt5[..2];
    let tv_pt2 = tv_fp2s(pt2);
    let inv_psi: Vec<Fp> = (0..padded_rows)
        .map(|r| Fp::from_u64_reduced(u64::from(r >= real_rows)))
        .collect();
    check_elementwise(
        &vrel.padding,
        pt2,
        Fp2::ZERO,
        &[vw.maximum()],
        3,
        &mut rng,
        |p, e| tv::eq(&tv_pt2, p).unwrap() * mle_fp(&inv_psi, p) * e[0],
    );
    // Forging maximum[3] != 0 on padded row 3 fails vrel.padding against tv_core::sumcheck claim 0.
    let mut bad_max: SecretBuf<Fp> = vw.maximum().iter().copied().collect();
    bad_max[3] = Fp::from_i64(42);
    let bad_pad_vw =
        VocabWitness::from_columns(bad_max, vw.exponentials().iter().copied().collect());
    let bad_vrel = vocab_max
        .relations::<Fp2>(&vsel, &flat_logits, &bad_pad_vw)
        .unwrap();
    let bad_ptbls = bad_vrel.padding.tables(pt2).unwrap();
    let p_sess = Session::random(&mut rng);
    let mut p_pt = p_sess.transcript();
    let mut p_vt = p_sess.tv_transcript();
    let p_bad = prove_batch(
        &mut p_pt,
        Protocol::One,
        Prefix::Stage(6),
        3,
        &[bad_ptbls.instance()],
    )
    .unwrap();
    let proof_bad = tv::SumcheckProof {
        rounds: p_bad.proof().rounds.iter().map(|r| tv_fp2s(r)).collect(),
        final_evals: tv_fp2s(&p_bad.proof().final_evals),
    };
    assert!(
        tv::verify_batch(
            &mut p_vt,
            tv::Prefix::Stage(6),
            3,
            &[tv::Instance {
                claim: TvFp2::ZERO,
                vars: 2,
                evals: 1,
            }],
            &proof_bad,
            |_, r, e| Some(tv::eq(&tv_pt2, r).unwrap() * mle_fp(&inv_psi, r) * e[0]),
        )
        .is_err()
    );

    // Test batched InverseCdf in BOTH exact (bounded = false) and bounded (bounded = true) modes.
    for bounded in [false, true] {
        let cdf = InverseCdf::<Fp>::new(vocab, vocab_max.sigma(), &ranges).unwrap();
        let committed_names = cdf.constraints::<Fp>().unwrap().committed;
        let mut flat_h = Vec::with_capacity(padded_rows * pv);
        let mut cols: Vec<(CdfColumn, SecretBuf<Fp>)> = committed_names
            .iter()
            .map(|&c| (c, SecretBuf::with_capacity(padded_rows)))
            .collect();
        for (row_idx, &draw) in draws.iter().enumerate().take(padded_rows) {
            let real = row_idx < real_rows;
            let row_e = &vw.exponentials()[row_idx * pv..(row_idx + 1) * pv];
            let h = if real {
                cdf.sample(row_e, draw, true).unwrap()
            } else {
                cdf.domain().witness(0, false).unwrap()
            };
            let cw = cdf.witness(row_e, draw, &h, real).unwrap();
            flat_h.extend_from_slice(h.indicator());
            for ((_, buf), &val) in cols.iter_mut().zip(cw.values()) {
                buf.push(Fp::from_i64(val));
            }
        }
        let batch_cols = CdfBatchColumns::from_columns(cols);
        let csel = cdf.selector(padded_rows, real_rows).unwrap();
        let ceval = cdf
            .evaluate_batch(vw.exponentials(), &flat_h, padded_rows)
            .unwrap();
        let clookups = cdf
            .evaluate_lookups(&csel, &draws, &ceval, &batch_cols)
            .unwrap();

        // Prove all Range lookups through LogUp and verify with tv-core.
        let rtables = cdf.tables();
        let rcols = table_columns(&rtables);
        let rmults = clookups.multiplicities(&rcols).unwrap();
        let rinst = clookups.instances(&rcols, &rmults).unwrap();
        let rshapes: Vec<Shape> = rinst
            .iter()
            .enumerate()
            .map(|(k, inst)| Shape::of(inst, spec_of(&rtables, &clookups, k)))
            .collect();
        let rsession = Session::random(&mut rng);
        let rproven = prove(&rsession, &rinst);
        assert!(verify(&rsession, &rshapes, &Messages::of(&rproven)).is_ok());

        // Batched contractions ALL have degree 3 when padded_rows > 1 (RFC-0058).
        let crel = cdf
            .batched_relations::<Fp2>(&csel, vw.exponentials(), &flat_h, &ceval)
            .unwrap();
        assert_eq!(crel.total.degree(), 3);
        assert_eq!(crel.below.degree(), 3);
        assert_eq!(crel.selected.degree(), 3);

        // Suffix dot product reduces L~(pt5) directly to flat_h, verified by tv_core::sumcheck.
        let l_claim = fp2(mle_fp(&ceval.suffix, &tv_pt5));
        let weights = cdf.batched_suffix_weights(&pt5, padded_rows).unwrap();
        let s_sess = Session::random(&mut rng);
        let mut s_pt = s_sess.transcript();
        let mut s_vt = s_sess.tv_transcript();
        let proved_s = prove_batch(
            &mut s_pt,
            Protocol::One,
            Prefix::Stage(2),
            2,
            &[Instance::new(vec![
                ScTable::ext(&weights).public(),
                ScTable::base(&flat_h),
            ])],
        )
        .unwrap();
        let proof_s = tv::SumcheckProof {
            rounds: proved_s.proof().rounds.iter().map(|r| tv_fp2s(r)).collect(),
            final_evals: tv_fp2s(&proved_s.proof().final_evals),
        };
        let tv_weights = tv_fp2s(&weights);
        let checked_s = tv::verify_batch(
            &mut s_vt,
            tv::Prefix::Stage(2),
            2,
            &[tv::Instance {
                claim: tv_fp2(l_claim),
                vars: 5,
                evals: 1,
            }],
            &proof_s,
            |_, r, e| {
                let w_val = tv_weights
                    .iter()
                    .zip(common::eq_weights(r))
                    .fold(TvFp2::ZERO, |acc, (&w, eq)| acc + w * eq);
                Some(w_val * e[0])
            },
        )
        .unwrap();
        assert_eq!(proof_s.final_evals, [mle_fp(&flat_h, &checked_s.point)]);

        // Equality relations all sumcheck to zero on honest inputs AND agree at the random
        // sumcheck challenge point with CdfConstraints.equalities evaluated over TvFp2!
        let c_base = cdf.constraints::<Fp>().unwrap();
        let eq_rels = cdf
            .equality_relations::<Fp2>(&csel, &draws, &ceval, &batch_cols)
            .unwrap();
        assert_eq!(eq_rels.len(), c_base.equalities.len());
        for (eq_idx, rel) in eq_rels.iter().enumerate() {
            let tbls = rel.tables(pt2).unwrap();
            let inst = tbls.instance();
            let e_sess = Session::random(&mut rng);
            let mut e_pt = e_sess.transcript();
            let mut e_vt = e_sess.tv_transcript();
            let p_eq = prove_batch(&mut e_pt, Protocol::One, Prefix::Stage(3), 3, &[inst]).unwrap();
            let eval_count = p_eq.proof().final_evals.len();
            let proof_eq = tv::SumcheckProof {
                rounds: p_eq.proof().rounds.iter().map(|r| tv_fp2s(r)).collect(),
                final_evals: tv_fp2s(&p_eq.proof().final_evals),
            };
            let _ = tv::verify_batch(
                &mut e_vt,
                tv::Prefix::Stage(3),
                3,
                &[tv::Instance {
                    claim: TvFp2::ZERO,
                    vars: 2,
                    evals: eval_count,
                }],
                &proof_eq,
                |_, r, _| {
                    let pt_f: Vec<Fp2> = r.iter().copied().map(fp2).collect();
                    let eval_col = |col: &[Fp]| -> Fp2 { fp2(mle_fp(col, r)) };
                    let poly = c_base.equalities[eq_idx].evaluate(|term| match term {
                        tessera_gadgets::CdfTerm::Draw => eval_col(&draws),
                        tessera_gadgets::CdfTerm::Total => eval_col(&ceval.totals),
                        tessera_gadgets::CdfTerm::Below => eval_col(&ceval.belows),
                        tessera_gadgets::CdfTerm::Selected => eval_col(&ceval.selecteds),
                        tessera_gadgets::CdfTerm::RowSelector => eval_col(csel.row_selector()),
                        tessera_gadgets::CdfTerm::Column(col) => {
                            eval_col(batch_cols.column(col).unwrap())
                        }
                    });
                    let _ = pt_f;
                    Some(tv::eq(&tv_pt2, r).unwrap() * tv_fp2(poly))
                },
            )
            .unwrap_or_else(|e| panic!("equality {eq_idx} failed verify_batch: {e:?}"));
        }

        if bounded {
            let srel = cdf
                .batched_sampled_relations::<Fp2>(&csel, &flat_h)
                .unwrap();
            check_elementwise(
                &srel.booleanity,
                &pt5,
                Fp2::ZERO,
                &[&flat_h],
                3,
                &mut rng,
                |p, e| tv::eq(&tv_pt5, p).unwrap() * e[0] * (TvFp2::ONE - e[0]),
            );
            let inv_valid: Vec<Fp> = (0..padded_rows * pv)
                .map(|idx| Fp::from_u64_reduced(u64::from(idx % pv >= vocab)))
                .collect();
            check_elementwise(
                &srel.vocabulary,
                &pt5,
                Fp2::ZERO,
                &[&flat_h],
                3,
                &mut rng,
                |p, e| tv::eq(&tv_pt5, p).unwrap() * mle_fp(&inv_valid, p) * e[0],
            );
            let count_claim = fp2(mle_fp(csel.row_selector(), &tv_pt2));
            check_elementwise(
                &srel.count,
                pt2,
                count_claim,
                &[&flat_h],
                2,
                &mut rng,
                |p, e| tv::eq(&tv_pt2, &p[3..]).unwrap() * e[0],
            );
            // Adversarial bounded indicator: setting h = 0 on real row 0 passes booleanity and
            // vocabulary, and is rejected ONLY by srel.count against claim psi~(pt2)!
            let mut zero_row0_h = flat_h.clone();
            zero_row0_h[..pv].fill(Fp::ZERO);
            let bad_srel = cdf
                .batched_sampled_relations::<Fp2>(&csel, &zero_row0_h)
                .unwrap();
            let bad_ctbls = bad_srel.count.tables(pt2).unwrap();
            let c_sess = Session::random(&mut rng);
            let mut c_pt = c_sess.transcript();
            let mut c_vt = c_sess.tv_transcript();
            let c_p = prove_batch(
                &mut c_pt,
                Protocol::One,
                Prefix::Stage(7),
                2,
                &[bad_ctbls.instance()],
            )
            .unwrap();
            let c_proof = tv::SumcheckProof {
                rounds: c_p.proof().rounds.iter().map(|r| tv_fp2s(r)).collect(),
                final_evals: tv_fp2s(&c_p.proof().final_evals),
            };
            assert!(
                tv::verify_batch(
                    &mut c_vt,
                    tv::Prefix::Stage(7),
                    2,
                    &[tv::Instance {
                        claim: tv_fp2(count_claim),
                        vars: 5,
                        evals: 1,
                    }],
                    &c_proof,
                    |_, r, e| Some(tv::eq(&tv_pt2, &r[3..]).unwrap() * e[0]),
                )
                .is_err()
            );
        }

        // Adversarial padded row: setting Z_3 != 0 on row 3 (psi_3 = 0) while keeping all
        // committed comparison columns 0 passes all Range lookups (u_3 = 0 -> u*Z = 0) and fails
        // ONLY the padding equality (psi_3 - 1) * Z_3 = 0 (rejected by tv_core::sumcheck)!
        let mut forged_eval = cdf
            .evaluate_batch(vw.exponentials(), &flat_h, padded_rows)
            .unwrap();
        forged_eval.totals[3] = Fp::ONE;
        let forged_eqs = cdf
            .equality_relations::<Fp2>(&csel, &draws, &forged_eval, &batch_cols)
            .unwrap();
        let last_eq = forged_eqs.last().expect("padding equality");
        let neg_inv_psi: Vec<Fp> = (0..padded_rows)
            .map(|r| if r < real_rows { Fp::ZERO } else { -Fp::ONE })
            .collect();
        let tbls = last_eq.tables(pt2).unwrap();
        let inst = tbls.instance();
        let f_sess = Session::random(&mut rng);
        let mut f_pt = f_sess.transcript();
        let mut f_vt = f_sess.tv_transcript();
        let p = prove_batch(&mut f_pt, Protocol::One, Prefix::Stage(5), 3, &[inst]).unwrap();
        let proof_f = tv::SumcheckProof {
            rounds: p.proof().rounds.iter().map(|r| tv_fp2s(r)).collect(),
            final_evals: tv_fp2s(&p.proof().final_evals),
        };
        let forged_res = tv::verify_batch(
            &mut f_vt,
            tv::Prefix::Stage(5),
            3,
            &[tv::Instance {
                claim: TvFp2::ZERO,
                vars: 2,
                evals: 1,
            }],
            &proof_f,
            |_, r, e| Some(tv::eq(&tv_pt2, r).unwrap() * mle_fp(&neg_inv_psi, r) * e[0]),
        );
        assert!(
            forged_res.is_err(),
            "padded row Z_3 != 0 must be rejected by sumcheck against claim 0"
        );
    }

    // Also verify LogUp rejects a tampered exponential with LookupError::Unbalanced.
    let mut bad_exps: SecretBuf<Fp> = vw.exponentials().iter().copied().collect();
    bad_exps[0] += Fp::ONE;
    let bad_vw = VocabWitness::from_columns(vw.maximum().iter().copied().collect(), bad_exps);
    let bad_veval = vocab_max
        .evaluate_batch(&flat_logits, padded_rows, real_rows, &bad_vw)
        .unwrap();
    let bad_mults = bad_veval.lookups.multiplicities(&exp_cols).unwrap();
    let bad_inst = bad_veval.lookups.instances(&exp_cols, &bad_mults).unwrap();
    let bad_sess = Session::random(&mut rng);
    let bad_proven = prove(&bad_sess, &bad_inst);
    assert!(matches!(
        verify(&bad_sess, &shapes, &Messages::of(&bad_proven)),
        Err(LogUpRejected::Lookup(LookupError::Unbalanced {
            instance: 0
        }))
    ));
}
