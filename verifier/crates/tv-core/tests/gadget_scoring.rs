// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Conformance vectors, independent integer-power oracle checks, omission controls,
//! ground-truth MLE binding, and adversarial mutation tests for `tv_core::gadgets::scoring`
//! (`VocabMax`, `OneSidedLog2`, and `intern_scoring_tables`; RFC-0017, RFC-0030, RFC-0035,
//! RFC-0050, RFC-0053, RFC-0058).

mod claim_support;
mod support;

use claim_support::{integer, items, list, member};
use support::{Context, digest, extension};
use tv_codec::Rational;
use tv_core::advice::ComparisonRole;
use tv_core::gadgets::GadgetError;
use tv_core::gadgets::inverse_cdf::StageVerifyError;
use tv_core::gadgets::scoring::{
    ExpTableEntry, InternConfig, InternError, Log2LookupClaim, Log2Params, Log2Role, Log2RowValues,
    Log2Verified, OneSidedLog2, RequestTables, ScoringMode, ScoringRequest, VocabMax,
    VocabMaxLookupClaim, VocabMaxRole, VocabMaxSourceClaim, VocabMaxVerified,
    intern_scoring_tables, select_vocab_exp_table,
};
use tv_core::protocol::{Decoding, FixedBits, PredictorSpec};
use tv_core::scoring::{ScoringError, c_match};
use tv_core::sumcheck::{self, Batch, Prefix, SumcheckError, SumcheckProof};
use tv_core::validation::GadgetId;
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Transcript};
use tv_tables::{TableId, TableKind};
use tv_testkit::{Json, read_vectors};

// --- JSON & algebraic helpers ---

fn u64_val(v: &Json) -> u64 {
    v.as_u64()
        .unwrap_or_else(|| panic!("{v:?} is not an unsigned u64"))
}

fn u64_vec(v: &Json) -> Vec<u64> {
    items(v).iter().map(u64_val).collect()
}

fn u8_vec(v: &Json) -> Vec<u8> {
    items(v)
        .iter()
        .map(|x| u8::try_from(u64_val(x)).expect("u8"))
        .collect()
}

fn fp2_num(v: &Json) -> Fp2 {
    let a = items(v);
    assert_eq!(a.len(), 2);
    Fp2::new(
        Fp::from_canonical(u64_val(&a[0])).expect("canonical c0"),
        Fp::from_canonical(u64_val(&a[1])).expect("canonical c1"),
    )
}

fn fp2_vec(v: &Json) -> Vec<Fp2> {
    items(v).iter().map(fp2_num).collect()
}

fn rational_num(v: &Json) -> Rational {
    let a = items(v);
    assert_eq!(a.len(), 2);
    let num = i64::try_from(integer(&a[0])).expect("i64 num");
    Rational::new(num, u64_val(&a[1])).expect("canonical rational")
}

fn signed_col(v: &Json) -> Vec<Fp> {
    items(v)
        .iter()
        .map(|x| {
            let m = i128::from(Fp::MODULUS);
            let rem = u64::try_from(integer(x).rem_euclid(m)).expect("canonical rem");
            Fp::from_canonical(rem).expect("canonical Fp")
        })
        .collect()
}

fn parse_proof(stage_json: &Json) -> SumcheckProof {
    SumcheckProof {
        rounds: list(stage_json, "rounds").iter().map(fp2_vec).collect(),
        final_evals: fp2_vec(member(stage_json, "final_evals")),
    }
}

fn parse_table_kind(v: &Json) -> TableKind {
    let Json::Object(map) = v else {
        panic!("expected TableKind object, got {v:?}");
    };
    assert_eq!(map.len(), 1);
    let (tag, body) = &map[0];
    match tag.as_str() {
        "Range" => TableKind::Range {
            bits: u8::try_from(u64_val(member(body, "bits"))).expect("bits"),
        },
        "Pow2" => TableKind::Pow2 {
            max_exp: u8::try_from(u64_val(member(body, "max_exp"))).expect("max_exp"),
        },
        "Exp2Frac" => TableKind::Exp2Frac {
            frac_bits: u8::try_from(u64_val(member(body, "frac_bits"))).expect("frac_bits"),
            out_bits: u8::try_from(u64_val(member(body, "out_bits"))).expect("out_bits"),
        },
        "Exp" => TableKind::Exp {
            key_bits: u8::try_from(u64_val(member(body, "key_bits"))).expect("key_bits"),
            sigma: u8::try_from(u64_val(member(body, "sigma"))).expect("sigma"),
            scale: rational_num(member(body, "scale")),
        },
        other => panic!("unexpected TableKind variant {other}"),
    }
}

fn absorb_split(t: &mut Transcript, elems: &[Fp2]) {
    let mut bytes = (elems.len() as u64).to_le_bytes().to_vec();
    for e in elems {
        bytes.extend_from_slice(&e.to_le_bytes());
    }
    t.absorb(Ascii::new("stage/0/split").expect("label"), &bytes);
}

fn delta0(point: &[Fp2]) -> Fp2 {
    point.iter().fold(Fp2::ONE, |acc, &x| acc * (Fp2::ONE - x))
}

fn cmp_role_index(role: ComparisonRole) -> usize {
    match role {
        ComparisonRole::LeftDigit => 0,
        ComparisonRole::LeftCarry => 1,
        ComparisonRole::RightDigit => 2,
        ComparisonRole::RightCarry => 3,
        ComparisonRole::Difference => 4,
        ComparisonRole::Borrow => 5,
    }
}

// --- Independent exact multi-limb integer arithmetic for c_match ---

#[derive(Clone, Debug, PartialEq, Eq)]
struct BigUInt(Vec<u64>); // little-endian base 2^64, trimmed

impl BigUInt {
    fn one() -> Self {
        Self(vec![1])
    }

    fn shl_bits(bits: usize) -> Self {
        let limb_idx = bits / 64;
        let bit_off = bits % 64;
        let mut limbs = vec![0u64; limb_idx + 1];
        limbs[limb_idx] = 1u64 << bit_off;
        Self(limbs)
    }

    fn mul_small(&self, factor: u64) -> Self {
        if factor == 0 {
            return Self(Vec::new());
        }
        let mut out = Vec::with_capacity(self.0.len() + 1);
        let mut carry = 0u128;
        for &limb in &self.0 {
            let prod = u128::from(limb) * u128::from(factor) + carry;
            out.push(prod as u64);
            carry = prod >> 64;
        }
        if carry > 0 {
            out.push(carry as u64);
        }
        Self(out)
    }

    fn mul_pow2(&self, bits: usize) -> Self {
        if self.0.is_empty() || bits == 0 {
            return self.clone();
        }
        let word_shift = bits / 64;
        let bit_shift = bits % 64;
        let mut out = vec![0u64; word_shift];
        if bit_shift == 0 {
            out.extend_from_slice(&self.0);
        } else {
            let mut carry = 0u64;
            for &limb in &self.0 {
                out.push((limb << bit_shift) | carry);
                carry = limb >> (64 - bit_shift);
            }
            if carry > 0 {
                out.push(carry);
            }
        }
        Self(out)
    }

    fn cmp_val(&self, other: &Self) -> core::cmp::Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.0.iter().rev().cmp(other.0.iter().rev()))
    }
}

#[test]
fn c_match_exact_independent_oracle_and_vectors() {
    let vectors = read_vectors("gadget-finals/one-sided-log2.json");
    let expected = u64_vec(member(&vectors, "c_match"));
    assert_eq!(expected.len(), 17);

    assert_eq!(c_match(0), Ok(0));
    assert_eq!(expected[0], 0);

    for k in 1u8..=16 {
        let got = c_match(k).expect("in-domain k");
        assert_eq!(
            got,
            expected[usize::from(k)],
            "c_match({k}) vector mismatch"
        );
        assert!(got >= 1);

        // Exact integer inequality: (2^k - 1)^256 * 2^{c - 1} < 2^{256 * k} <= (2^k - 1)^256 * 2^c.
        let base_factor = (1u64 << k) - 1;
        let mut base_256 = BigUInt::one();
        for _ in 0..256 {
            base_256 = base_256.mul_small(base_factor);
        }
        let lhs = BigUInt::shl_bits(256 * usize::from(k));
        let lower = base_256.mul_pow2(usize::try_from(got - 1).expect("usize"));
        let upper = base_256.mul_pow2(usize::try_from(got).expect("usize"));
        assert_eq!(
            lower.cmp_val(&lhs),
            core::cmp::Ordering::Less,
            "c_match({k}) = {got} is not minimal"
        );
        assert!(
            lhs.cmp_val(&upper) != core::cmp::Ordering::Greater,
            "c_match({k}) = {got} under-bounds -log2(1 - 2^-{k}) * 256"
        );
    }

    assert_eq!(c_match(17), Err(ScoringError::EpsLog2 { eps_log2: 17 }));
    assert_eq!(
        c_match(u8::MAX),
        Err(ScoringError::EpsLog2 { eps_log2: u8::MAX })
    );
}

// --- VocabMax tests ---

fn vmax_initial_transcript(case: &Json) -> Transcript {
    let mut t = Context::from_json(member(case, "context")).transcript();
    absorb_split(
        &mut t,
        &[
            fp2_num(member(case, "lookup_key")),
            fp2_num(member(case, "lookup_value")),
            fp2_num(member(case, "tree_leaf")),
        ],
    );
    t
}

fn check_vmax_obligations(case: &Json, verified: &VocabMaxVerified) {
    let maxima = signed_col(member(case, "maxima"));
    let logits = signed_col(member(case, "logits"));
    let keys = signed_col(member(case, "keys"));
    let exps = signed_col(member(case, "exps"));

    let lookup: &VocabMaxLookupClaim = verified.lookup_claim();
    assert_eq!(lookup.point, fp2_vec(member(case, "lookup_target")));
    assert_eq!(lookup.key, extension(&keys, &lookup.point));
    assert_eq!(lookup.value, extension(&exps, &lookup.point));

    let [src_max, src_logits]: &[VocabMaxSourceClaim; 2] = verified.source_claims();
    assert_eq!(src_max.role, VocabMaxRole::Maximum);
    assert_eq!(src_max.value, extension(&maxima, &src_max.point));
    assert_eq!(src_logits.role, VocabMaxRole::Logits);
    assert_eq!(src_logits.value, extension(&logits, &src_logits.point));
}

#[test]
fn vocab_max_vectors_omission_and_adversarial_mutations() {
    let vectors = read_vectors("gadget-finals/one-sided-log2.json");
    let cases = list(&vectors, "vocab_max_cases");
    assert_eq!(cases.len(), 6);

    for case in cases {
        let name = member(case, "name").as_str().expect("name");
        let gadget = GadgetId(u32::try_from(u64_val(member(case, "gadget"))).expect("gadget"));
        let rows = u32::try_from(u64_val(member(case, "rows"))).expect("rows");
        let vocab = u32::try_from(u64_val(member(case, "vocab"))).expect("vocab");
        let logits_exp = i8::try_from(integer(member(case, "logits_exponent"))).expect("i8");
        let temp = rational_num(member(case, "temperature"));
        let sigma = u8::try_from(u64_val(member(case, "sigma"))).expect("sigma");
        let lo = i32::try_from(integer(member(case, "logit_lo"))).expect("lo");
        let hi = i32::try_from(integer(member(case, "logit_hi"))).expect("hi");
        let max_log2 = u8::try_from(u64_val(member(case, "max_log2_rows"))).expect("max_log2");

        let site = VocabMax::derive(
            gadget, vocab, rows, logits_exp, temp, sigma, lo, hi, max_log2,
        )
        .unwrap_or_else(|e| panic!("{name}: derive failed: {e}"));

        assert_eq!(site.gadget(), gadget, "{name}");
        assert_eq!(site.scale(), rational_num(member(case, "scale")), "{name}");
        assert_eq!(
            site.min_key_bits(),
            u8::try_from(u64_val(member(case, "min_key_bits"))).expect("min_kb"),
            "{name}"
        );
        assert_eq!(
            site.key_bits(),
            u8::try_from(u64_val(member(case, "key_bits"))).expect("kb"),
            "{name}"
        );
        assert_eq!(site.k0(), u64_val(member(case, "k0")), "{name}");

        // Explicit key_bits matches derive; narrower key_bits or above max_log2_rows fails.
        let explicit = VocabMax::derive_with_key_bits(
            gadget,
            vocab,
            rows,
            logits_exp,
            temp,
            sigma,
            lo,
            hi,
            max_log2,
            site.key_bits(),
        )
        .expect("explicit key_bits");
        assert_eq!(explicit, site, "{name}");
        if site.key_bits() > 1 {
            assert_eq!(
                VocabMax::derive_with_key_bits(
                    gadget,
                    vocab,
                    rows,
                    logits_exp,
                    temp,
                    sigma,
                    lo,
                    hi,
                    max_log2,
                    site.key_bits() - 1,
                ),
                Err(GadgetError::Internal),
                "{name}: key_bits - 1 must fail min_key_bits or zero-tail check"
            );
        }

        let lookup_target = fp2_vec(member(case, "lookup_target"));
        let tree_target = fp2_vec(member(case, "tree_target"));
        let forest_root_target = fp2_vec(member(case, "forest_root_target"));
        let lookup_key = fp2_num(member(case, "lookup_key"));
        let lookup_value = fp2_num(member(case, "lookup_value"));
        let tree_leaf = fp2_num(member(case, "tree_leaf"));
        let forest_root_claim = fp2_num(member(case, "forest_root_claim"));

        // Perturbing forest_root_claim must reject at check_claims.
        assert_eq!(
            site.check_claims(
                &lookup_target,
                lookup_key,
                lookup_value,
                &tree_target,
                tree_leaf,
                &forest_root_target,
                forest_root_claim + Fp2::ONE,
            )
            .unwrap_err(),
            GadgetError::Rejected { gadget },
            "{name}: bad forest root must reject"
        );
        // Wrong target point length is a caller bug (Internal).
        assert_eq!(
            site.check_claims(
                &[Fp2::ZERO; 33],
                lookup_key,
                lookup_value,
                &tree_target,
                tree_leaf,
                &forest_root_target,
                forest_root_claim,
            )
            .unwrap_err(),
            GadgetError::Internal
        );

        let checked = site
            .check_claims(
                &lookup_target,
                lookup_key,
                lookup_value,
                &tree_target,
                tree_leaf,
                &forest_root_target,
                forest_root_claim,
            )
            .expect("check_claims");
        assert_eq!(
            checked.claims().as_slice(),
            fp2_vec(member(case, "claims")).as_slice(),
            "{name}"
        );

        let proof_json = member(case, "proof");
        let proof = parse_proof(proof_json);

        let failed_json = member(case, "failed");
        if matches!(failed_json, Json::Null) {
            let mut t = vmax_initial_transcript(case);
            let s1 = checked.squeeze_stage1(&mut t).expect("squeeze_stage1");
            let (batch, verified): (Batch, VocabMaxVerified) = s1
                .verify(&mut t, &proof)
                .unwrap_or_else(|e| panic!("{name}: honest VocabMax failed: {e:?}"));
            assert_eq!(batch.point, fp2_vec(member(proof_json, "point")), "{name}");
            assert_eq!(batch.gamma, fp2_num(member(proof_json, "gamma")), "{name}");
            assert_eq!(
                *t.state(),
                digest(member(proof_json, "state").as_str().expect("state")),
                "{name}"
            );
            assert_eq!(
                t.counter(),
                u64_val(member(proof_json, "counter")),
                "{name}"
            );
            check_vmax_obligations(case, &verified);

            // Every single-slot perturbation of final_evals must be rejected.
            for slot in 0..proof.final_evals.len() {
                let mut t_bad = vmax_initial_transcript(case);
                let c_bad = site
                    .check_claims(
                        &lookup_target,
                        lookup_key,
                        lookup_value,
                        &tree_target,
                        tree_leaf,
                        &forest_root_target,
                        forest_root_claim,
                    )
                    .expect("check_claims");
                let s1_bad = c_bad.squeeze_stage1(&mut t_bad).expect("squeeze_stage1");
                let mut mutated = proof.clone();
                mutated.final_evals[slot] += Fp2::ONE;
                assert!(
                    s1_bad.verify(&mut t_bad, &mutated).is_err(),
                    "{name}: slot {slot} perturbation accepted"
                );
            }

            // Direct cross-instance consistency checks on finish([c, ell, c, ell, c]).
            for bad_idx in [2usize, 3, 4] {
                let mut t_f = vmax_initial_transcript(case);
                let s1_f = site
                    .check_claims(
                        &lookup_target,
                        lookup_key,
                        lookup_value,
                        &tree_target,
                        tree_leaf,
                        &forest_root_target,
                        forest_root_claim,
                    )
                    .expect("check_claims")
                    .squeeze_stage1(&mut t_f)
                    .expect("squeeze");
                let mut bad_evals = proof.final_evals.clone();
                bad_evals[bad_idx] += Fp2::ONE;
                assert_eq!(
                    s1_f.finish(batch.point.clone(), &bad_evals),
                    Err(GadgetError::Rejected { gadget }),
                    "{name}: inconsistent slot {bad_idx} must reject in finish"
                );
            }
        } else {
            let omit_idx = usize::try_from(u64_val(failed_json)).expect("omit_idx");
            let expected_err = StageVerifyError::Sumcheck(SumcheckError::FinalMismatch {
                round: u32::try_from(site.variables()).expect("u32"),
            });

            // 1. Real verifier rejects raw proof.
            let mut t_raw = vmax_initial_transcript(case);
            let s1_raw = checked.squeeze_stage1(&mut t_raw).expect("squeeze_stage1");
            assert_eq!(
                s1_raw.verify(&mut t_raw, &proof),
                Err(expected_err.clone()),
                "{name}: raw failed case must reject"
            );

            // 2. Real verifier also rejects omission proof.
            let om_json = member(case, "omission");
            let om_proof = parse_proof(om_json);
            let mut t_om_real = vmax_initial_transcript(case);
            let s1_om_real = site
                .check_claims(
                    &lookup_target,
                    lookup_key,
                    lookup_value,
                    &tree_target,
                    tree_leaf,
                    &forest_root_target,
                    forest_root_claim,
                )
                .expect("check_claims")
                .squeeze_stage1(&mut t_om_real)
                .expect("squeeze_stage1");
            assert_eq!(
                s1_om_real.verify(&mut t_om_real, &om_proof),
                Err(expected_err),
                "{name}: omission proof must reject under real verifier"
            );

            // 3. Adapted verifier omitting ONLY `omit_idx` accepts omission proof and matches state.
            let mut t_ctrl = vmax_initial_transcript(case);
            let s1_ctrl = site
                .check_claims(
                    &lookup_target,
                    lookup_key,
                    lookup_value,
                    &tree_target,
                    tree_leaf,
                    &forest_root_target,
                    forest_root_claim,
                )
                .expect("check_claims")
                .squeeze_stage1(&mut t_ctrl)
                .expect("squeeze_stage1");
            let batch = sumcheck::verify_batch(
                &mut t_ctrl,
                Prefix::Stage(1),
                s1_ctrl.degree(),
                s1_ctrl.instances(),
                &om_proof,
                |i, pt, vals| {
                    if i == omit_idx {
                        Some(s1_ctrl.instances()[i].claim * delta0(pt))
                    } else {
                        s1_ctrl.evaluate(i, pt, vals).ok()
                    }
                },
            )
            .unwrap_or_else(|e| panic!("{name}: omission control failed: {e:?}"));
            let verified = s1_ctrl
                .finish(batch.point, &om_proof.final_evals)
                .expect("finish");
            assert_eq!(
                *t_ctrl.state(),
                digest(member(om_json, "state").as_str().expect("state")),
                "{name}"
            );
            assert_eq!(
                t_ctrl.counter(),
                u64_val(member(om_json, "counter")),
                "{name}"
            );
            check_vmax_obligations(case, &verified);
        }
    }
}

// --- Runtime table interning tests ---

#[test]
fn table_interning_vectors_and_domain_guards() {
    let vectors = read_vectors("gadget-finals/one-sided-log2.json");
    let cases = list(&vectors, "table_interning_cases");
    assert_eq!(cases.len(), 3);

    for case in cases {
        let name = member(case, "name").as_str().expect("name");
        let base_tables: Vec<TableKind> = list(case, "base_tables")
            .iter()
            .map(parse_table_kind)
            .collect();
        let vocab = u32::try_from(u64_val(member(case, "vocab"))).expect("vocab");
        let sigma = u8::try_from(u64_val(member(case, "sigma"))).expect("sigma");
        let logits_exponent =
            i8::try_from(integer(member(case, "logits_exponent"))).expect("logits_exponent");
        let logit_lo = i32::try_from(integer(member(case, "logit_lo"))).expect("logit_lo");
        let logit_hi = i32::try_from(integer(member(case, "logit_hi"))).expect("logit_hi");
        let max_log2_rows =
            u8::try_from(u64_val(member(case, "max_log2_rows"))).expect("max_log2_rows");
        let eps_log2 = u8::try_from(u64_val(member(case, "eps_log2"))).expect("eps_log2");
        let mode = match member(case, "mode").as_str().expect("mode") {
            "Exact" => ScoringMode::Exact,
            "Mixture" => ScoringMode::Mixture { eps_log2 },
            other => panic!("unexpected mode {other}"),
        };
        let requests: Vec<ScoringRequest> = list(case, "requests")
            .iter()
            .map(|r| {
                let outputs = u32::try_from(u64_val(member(r, "outputs"))).expect("outputs");
                let decoding = match member(r, "decoding").as_str().expect("decoding") {
                    "Greedy" => Decoding::Greedy,
                    "Temperature" => Decoding::Temperature {
                        t: rational_num(member(r, "temperature")),
                    },
                    other => panic!("unexpected decoding {other}"),
                };
                ScoringRequest { outputs, decoding }
            })
            .collect();

        let got = intern_scoring_tables(InternConfig {
            base_tables: &base_tables,
            vocab,
            sigma,
            logits_exponent,
            logit_lo,
            logit_hi,
            max_log2_rows,
            mode,
            requests: &requests,
        })
        .unwrap_or_else(|e| panic!("{name}: intern_scoring_tables failed: {e}"));

        let expected_catalog: Vec<TableKind> =
            list(case, "catalog").iter().map(parse_table_kind).collect();
        assert_eq!(got.catalog, expected_catalog, "{name}: catalog mismatch");

        let expected_exp: Vec<ExpTableEntry> = list(case, "exp_tables")
            .iter()
            .map(|e| ExpTableEntry {
                temperature: rational_num(member(e, "temperature")),
                table_id: TableId(u32::try_from(u64_val(member(e, "table_id"))).expect("table_id")),
                kind: parse_table_kind(member(e, "kind")),
            })
            .collect();
        assert_eq!(got.exp_tables, expected_exp, "{name}: exp_tables mismatch");

        let expected_req: Vec<RequestTables> = list(case, "request_tables")
            .iter()
            .map(|rt| RequestTables {
                decoding_ids: u64_vec(member(rt, "decoding_ids"))
                    .into_iter()
                    .map(|id| TableId(u32::try_from(id).expect("id")))
                    .collect(),
                log2_ids: u64_vec(member(rt, "log2_ids"))
                    .into_iter()
                    .map(|id| TableId(u32::try_from(id).expect("id")))
                    .collect(),
            })
            .collect();
        assert_eq!(
            got.request_tables, expected_req,
            "{name}: request_tables mismatch"
        );
    }

    // ScoringMode::from_predictor mapping.
    assert_eq!(
        ScoringMode::from_predictor(PredictorSpec::Exact {
            decoding: Decoding::Greedy,
        }),
        ScoringMode::Exact
    );
    assert_eq!(
        ScoringMode::from_predictor(PredictorSpec::Mixture {
            decoding: Decoding::Greedy,
            eps_log2: 6,
        }),
        ScoringMode::Mixture { eps_log2: 6 }
    );

    // > 4 distinct effective temperatures rejected.
    let five_temps: Vec<ScoringRequest> = [(1, 2), (3, 4), (1, 1), (3, 2), (2, 1)]
        .into_iter()
        .map(|(n, d)| ScoringRequest {
            outputs: 1,
            decoding: Decoding::Temperature {
                t: Rational::new(n, d).expect("rational"),
            },
        })
        .collect();
    assert_eq!(
        intern_scoring_tables(InternConfig {
            base_tables: &[],
            vocab: 64,
            sigma: 12,
            logits_exponent: 0,
            logit_lo: -63,
            logit_hi: 63,
            max_log2_rows: 32,
            mode: ScoringMode::Exact,
            requests: &five_temps,
        }),
        Err(InternError::TooManyTemperatures { count: 5 })
    );

    // Out-of-domain eps_log2 and temperature rejected.
    assert_eq!(
        intern_scoring_tables(InternConfig {
            base_tables: &[],
            vocab: 64,
            sigma: 12,
            logits_exponent: 0,
            logit_lo: -63,
            logit_hi: 63,
            max_log2_rows: 32,
            mode: ScoringMode::Mixture { eps_log2: 17 },
            requests: &[],
        }),
        Err(InternError::Scoring(ScoringError::EpsLog2 { eps_log2: 17 }))
    );
    let bad_temp = Rational::new(9, 2).expect("9/2");
    assert_eq!(
        intern_scoring_tables(InternConfig {
            base_tables: &[],
            vocab: 64,
            sigma: 12,
            logits_exponent: 0,
            logit_lo: -63,
            logit_hi: 63,
            max_log2_rows: 32,
            mode: ScoringMode::Exact,
            requests: &[ScoringRequest {
                outputs: 1,
                decoding: Decoding::Temperature { t: bad_temp },
            }],
        }),
        Err(InternError::Scoring(ScoringError::Temperature {
            t: bad_temp
        }))
    );

    // Strict 28-bit signed logit endpoint bound (RFC-0009 Lemma 5, RFC-0053):
    // [-(1 << 28) + 1, (1 << 28) - 1] is the widest valid interval (min_key_bits = 29).
    let one_t = Rational::new(1, 1).expect("1/1");
    let max_valid_lo = -(1i32 << 28) + 1;
    let max_valid_hi = (1i32 << 28) - 1;
    let widest_greedy = intern_scoring_tables(InternConfig {
        base_tables: &[],
        vocab: 64,
        sigma: 8,
        logits_exponent: 0,
        logit_lo: max_valid_lo,
        logit_hi: max_valid_hi,
        max_log2_rows: 32,
        mode: ScoringMode::Exact,
        requests: &[ScoringRequest {
            outputs: 1,
            decoding: Decoding::Greedy,
        }],
    })
    .expect("widest valid 28-bit signed logit span must succeed in Greedy mode");
    assert_eq!(widest_greedy.catalog, vec![TableKind::Range { bits: 29 }]);

    let widest_temp = intern_scoring_tables(InternConfig {
        base_tables: &[],
        vocab: 64,
        sigma: 8,
        logits_exponent: 0,
        logit_lo: max_valid_lo,
        logit_hi: max_valid_hi,
        max_log2_rows: 32,
        mode: ScoringMode::Exact,
        requests: &[ScoringRequest {
            outputs: 1,
            decoding: Decoding::Temperature { t: one_t },
        }],
    })
    .expect("widest valid 28-bit signed logit span must succeed in Temperature mode");
    assert_eq!(
        widest_temp.exp_tables[0].kind,
        TableKind::Exp {
            key_bits: 29,
            sigma: 8,
            scale: one_t,
        }
    );

    let widest_vm = VocabMax::derive(
        GadgetId(0),
        64,
        4,
        0,
        one_t,
        8,
        max_valid_lo,
        max_valid_hi,
        32,
    )
    .expect("VocabMax::derive must accept [-(1 << 28) + 1, (1 << 28) - 1]");
    assert_eq!(widest_vm.min_key_bits(), 29);
    assert_eq!(widest_vm.key_bits(), 29);
    assert!(
        VocabMax::derive_with_key_bits(
            GadgetId(0),
            64,
            4,
            0,
            one_t,
            8,
            max_valid_lo,
            max_valid_hi,
            32,
            29,
        )
        .is_ok()
    );

    for (bad_lo, bad_hi) in [
        (-(1i32 << 28), 0),
        (-(1i32 << 28), max_valid_hi),
        (0, 1i32 << 28),
        (max_valid_lo, 1i32 << 28),
        (1i32 << 28, (1i32 << 28) + 7),
        (-(1i32 << 28) - 7, -(1i32 << 28)),
    ] {
        for mode in [ScoringMode::Exact, ScoringMode::Mixture { eps_log2: 6 }] {
            for decoding in [Decoding::Greedy, Decoding::Temperature { t: one_t }] {
                assert_eq!(
                    intern_scoring_tables(InternConfig {
                        base_tables: &[],
                        vocab: 64,
                        sigma: 8,
                        logits_exponent: 0,
                        logit_lo: bad_lo,
                        logit_hi: bad_hi,
                        max_log2_rows: 32,
                        mode,
                        requests: &[ScoringRequest {
                            outputs: 1,
                            decoding,
                        }],
                    }),
                    Err(InternError::Gadget(GadgetError::Internal)),
                    "logit bounds [{bad_lo}, {bad_hi}] in {mode:?}/{decoding:?} must be rejected"
                );
            }
        }
        assert_eq!(
            select_vocab_exp_table(0, one_t, 8, bad_lo, bad_hi, 32),
            Err(GadgetError::Internal),
            "select_vocab_exp_table must reject [{bad_lo}, {bad_hi}]"
        );
        assert_eq!(
            VocabMax::derive(GadgetId(0), 64, 4, 0, one_t, 8, bad_lo, bad_hi, 32),
            Err(GadgetError::Internal),
            "VocabMax::derive must reject [{bad_lo}, {bad_hi}]"
        );
        assert_eq!(
            VocabMax::derive_with_key_bits(GadgetId(0), 64, 4, 0, one_t, 8, bad_lo, bad_hi, 32, 29),
            Err(GadgetError::Internal),
            "VocabMax::derive_with_key_bits must reject [{bad_lo}, {bad_hi}]"
        );
    }

    // Narrow max_log2_rows unable to hit Exp_T zero tail rejected.
    assert_eq!(
        intern_scoring_tables(InternConfig {
            base_tables: &[],
            vocab: 64,
            sigma: 16,
            logits_exponent: -8,
            logit_lo: -3,
            logit_hi: 3,
            max_log2_rows: 4,
            mode: ScoringMode::Exact,
            requests: &[ScoringRequest {
                outputs: 1,
                decoding: Decoding::Temperature { t: one_t },
            }],
        }),
        Err(InternError::NoQualifyingExpWidth { temperature: one_t })
    );

    // GadgetPolicy.max_log2_rows enforced on every runtime table, even if pre-populated in base_tables (RFC-0053):
    // (a) Exact + Greedy with min_key_bits = 22 > max_log2_rows = 20.
    for base in [&[][..], &[TableKind::Range { bits: 22 }][..]] {
        assert_eq!(
            intern_scoring_tables(InternConfig {
                base_tables: base,
                vocab: 64,
                sigma: 8,
                logits_exponent: 0,
                logit_lo: -(1 << 20),
                logit_hi: 1 << 20,
                max_log2_rows: 20,
                mode: ScoringMode::Exact,
                requests: &[ScoringRequest {
                    outputs: 1,
                    decoding: Decoding::Greedy,
                }],
            }),
            Err(InternError::TableTooWide {
                log2_rows: 22,
                max_log2_rows: 20,
            })
        );
    }
    assert!(
        intern_scoring_tables(InternConfig {
            base_tables: &[],
            vocab: 64,
            sigma: 8,
            logits_exponent: 0,
            logit_lo: -(1 << 20),
            logit_hi: 1 << 20,
            max_log2_rows: 22,
            mode: ScoringMode::Exact,
            requests: &[ScoringRequest {
                outputs: 1,
                decoding: Decoding::Greedy,
            }],
        })
        .is_ok()
    );

    // (b) Mixture where Exp_1 fits within <= 15 bits (kb = 6 at sigma = 8, [-31, 31]),
    // yet OneSidedLog2 / InverseCdf Range { bits: 16 } exceeds max_log2_rows = 15,
    // both with empty base_tables and with Range { bits: 16 } pre-populated in base_tables.
    for base in [&[][..], &[TableKind::Range { bits: 16 }][..]] {
        for decoding in [Decoding::Greedy, Decoding::Temperature { t: one_t }] {
            assert_eq!(
                intern_scoring_tables(InternConfig {
                    base_tables: base,
                    vocab: 64,
                    sigma: 8,
                    logits_exponent: 0,
                    logit_lo: -31,
                    logit_hi: 31,
                    max_log2_rows: 15,
                    mode: ScoringMode::Mixture { eps_log2: 6 },
                    requests: &[ScoringRequest {
                        outputs: 1,
                        decoding,
                    }],
                }),
                Err(InternError::TableTooWide {
                    log2_rows: 16,
                    max_log2_rows: 15,
                }),
                "Mixture {decoding:?} with base={base:?} must enforce max_log2_rows = 15"
            );
        }
    }
    // Also check Pow2(63) (log2_rows = 6 > 5) and Exp2Frac(8, 16) (log2_rows = 8 > 7) in Mixture mode.
    for (max_rows, expected_rows, pre) in [
        (5u8, 6u8, TableKind::Pow2 { max_exp: 63 }),
        (
            7u8,
            8u8,
            TableKind::Exp2Frac {
                frac_bits: 8,
                out_bits: 16,
            },
        ),
    ] {
        for base in [&[][..], std::slice::from_ref(&pre)] {
            assert_eq!(
                intern_scoring_tables(InternConfig {
                    base_tables: base,
                    vocab: 16,
                    sigma: 4,
                    logits_exponent: 0,
                    logit_lo: -3,
                    logit_hi: 3,
                    max_log2_rows: max_rows,
                    mode: ScoringMode::Mixture { eps_log2: 6 },
                    requests: &[ScoringRequest {
                        outputs: 1,
                        decoding: Decoding::Greedy,
                    }],
                }),
                Err(InternError::TableTooWide {
                    log2_rows: expected_rows,
                    max_log2_rows: max_rows,
                })
            );
        }
    }

    // (c) Exact + Temperature with max_log2_rows = 15 (Range { bits: 16 } > 15) and with
    // max_log2_rows = 16 at vocab = 1 << 18, sigma = 32 (carry Range widths reach 17..=18).
    for base in [&[][..], &[TableKind::Range { bits: 16 }][..]] {
        assert_eq!(
            intern_scoring_tables(InternConfig {
                base_tables: base,
                vocab: 64,
                sigma: 8,
                logits_exponent: 0,
                logit_lo: -31,
                logit_hi: 31,
                max_log2_rows: 15,
                mode: ScoringMode::Exact,
                requests: &[ScoringRequest {
                    outputs: 1,
                    decoding: Decoding::Temperature { t: one_t },
                }],
            }),
            Err(InternError::TableTooWide {
                log2_rows: 16,
                max_log2_rows: 15,
            })
        );
    }
    for base in [&[][..], &[TableKind::Range { bits: 17 }][..]] {
        assert_eq!(
            intern_scoring_tables(InternConfig {
                base_tables: base,
                vocab: 1 << 18,
                sigma: 32,
                logits_exponent: 0,
                logit_lo: -31,
                logit_hi: 31,
                max_log2_rows: 16,
                mode: ScoringMode::Exact,
                requests: &[ScoringRequest {
                    outputs: 1,
                    decoding: Decoding::Temperature { t: one_t },
                }],
            }),
            Err(InternError::TableTooWide {
                log2_rows: 17,
                max_log2_rows: 16,
            })
        );
    }
    assert!(
        intern_scoring_tables(InternConfig {
            base_tables: &[],
            vocab: 1 << 18,
            sigma: 32,
            logits_exponent: 0,
            logit_lo: -31,
            logit_hi: 31,
            max_log2_rows: 17,
            mode: ScoringMode::Exact,
            requests: &[ScoringRequest {
                outputs: 1,
                decoding: Decoding::Temperature { t: one_t },
            }],
        })
        .is_ok()
    );
    for base in [&[][..], &[TableKind::Range { bits: 18 }][..]] {
        assert_eq!(
            intern_scoring_tables(InternConfig {
                base_tables: base,
                vocab: 1 << 18,
                sigma: 32,
                logits_exponent: 0,
                logit_lo: -31,
                logit_hi: 31,
                max_log2_rows: 17,
                mode: ScoringMode::Mixture { eps_log2: 6 },
                requests: &[ScoringRequest {
                    outputs: 1,
                    decoding: Decoding::Temperature { t: one_t },
                }],
            }),
            Err(InternError::TableTooWide {
                log2_rows: 18,
                max_log2_rows: 17,
            })
        );
    }

    // Zero-output requests vs effective-temperature ordering and vocab_exp_scale validation (RFC-0017 item 6):
    // (a) [outputs=0 @ T_A, outputs=3 @ T_B, outputs=3 @ T_A] interns Exp_{T_B} before Exp_{T_A}.
    let t_a = Rational::new(1, 2).expect("1/2");
    let t_b = Rational::new(3, 4).expect("3/4");
    let ordered = intern_scoring_tables(InternConfig {
        base_tables: &[],
        vocab: 64,
        sigma: 12,
        logits_exponent: 0,
        logit_lo: -63,
        logit_hi: 63,
        max_log2_rows: 32,
        mode: ScoringMode::Exact,
        requests: &[
            ScoringRequest {
                outputs: 0,
                decoding: Decoding::Temperature { t: t_a },
            },
            ScoringRequest {
                outputs: 3,
                decoding: Decoding::Temperature { t: t_b },
            },
            ScoringRequest {
                outputs: 3,
                decoding: Decoding::Temperature { t: t_a },
            },
        ],
    })
    .expect("zero-output T_A must not precede output-bearing T_B");
    assert_eq!(ordered.exp_tables.len(), 2);
    assert_eq!(ordered.exp_tables[0].temperature, t_b);
    assert_eq!(ordered.exp_tables[0].table_id, TableId(0));
    assert_eq!(ordered.exp_tables[1].temperature, t_a);
    assert_eq!(ordered.exp_tables[1].table_id, TableId(1));
    assert!(ordered.request_tables[0].decoding_ids.is_empty());

    // (b) Four zero-output requests at distinct temperatures + one output-bearing request at a fifth temperature succeeds.
    let fifth_t = Rational::new(2, 1).expect("2/1");
    let four_zero_plus_one = vec![
        ScoringRequest {
            outputs: 0,
            decoding: Decoding::Temperature {
                t: Rational::new(1, 2).expect("1/2"),
            },
        },
        ScoringRequest {
            outputs: 0,
            decoding: Decoding::Temperature {
                t: Rational::new(3, 4).expect("3/4"),
            },
        },
        ScoringRequest {
            outputs: 0,
            decoding: Decoding::Temperature { t: one_t },
        },
        ScoringRequest {
            outputs: 0,
            decoding: Decoding::Temperature {
                t: Rational::new(3, 2).expect("3/2"),
            },
        },
        ScoringRequest {
            outputs: 2,
            decoding: Decoding::Temperature { t: fifth_t },
        },
    ];
    let fifth_ok = intern_scoring_tables(InternConfig {
        base_tables: &[],
        vocab: 64,
        sigma: 12,
        logits_exponent: 0,
        logit_lo: -63,
        logit_hi: 63,
        max_log2_rows: 32,
        mode: ScoringMode::Exact,
        requests: &four_zero_plus_one,
    })
    .expect("zero-output temperatures do not consume the 4-temperature batch budget");
    assert_eq!(fifth_ok.exp_tables.len(), 1);
    assert_eq!(fifth_ok.exp_tables[0].temperature, fifth_t);

    // (c) Zero-output request whose 2^{logits_exponent} / T overflows canonical Rational is rejected.
    assert_eq!(
        intern_scoring_tables(InternConfig {
            base_tables: &[],
            vocab: 64,
            sigma: 12,
            logits_exponent: -128,
            logit_lo: -63,
            logit_hi: 63,
            max_log2_rows: 32,
            mode: ScoringMode::Exact,
            requests: &[ScoringRequest {
                outputs: 0,
                decoding: Decoding::Temperature { t: one_t },
            }],
        }),
        Err(InternError::Scoring(ScoringError::LogitsExponent {
            logits_exponent: -128,
        }))
    );
    assert_eq!(
        intern_scoring_tables(InternConfig {
            base_tables: &[],
            vocab: 64,
            sigma: 12,
            logits_exponent: -128,
            logit_lo: -63,
            logit_hi: 63,
            max_log2_rows: 32,
            mode: ScoringMode::Mixture { eps_log2: 6 },
            requests: &[ScoringRequest {
                outputs: 0,
                decoding: Decoding::Greedy,
            }],
        }),
        Err(InternError::Scoring(ScoringError::LogitsExponent {
            logits_exponent: -128,
        }))
    );
    let extreme_den_t = Rational::new(1, u64::MAX).expect("1 / u64::MAX");
    assert_eq!(
        intern_scoring_tables(InternConfig {
            base_tables: &[],
            vocab: 64,
            sigma: 12,
            logits_exponent: 0,
            logit_lo: -63,
            logit_hi: 63,
            max_log2_rows: 32,
            mode: ScoringMode::Exact,
            requests: &[ScoringRequest {
                outputs: 0,
                decoding: Decoding::Temperature { t: extreme_den_t },
            }],
        }),
        Err(InternError::Scoring(ScoringError::LogitsExponent {
            logits_exponent: 0,
        }))
    );
}

// --- OneSidedLog2 schema & component tests ---

fn log2_initial_transcript(case: &Json) -> Transcript {
    let mut t = Context::from_json(member(case, "context")).transcript();
    let lvals = fp2_vec(member(case, "lookup_value_evals"));
    let mut elems = vec![
        fp2_num(member(case, "z_eval")),
        fp2_num(member(case, "q_eval")),
        lvals[0],
        lvals[1],
    ];
    elems.extend(fp2_vec(member(case, "lookup_key_evals")));
    absorb_split(&mut t, &elems);
    t
}

fn log2_obligations_hold(case: &Json, params: &Log2Params, verified: &Log2Verified) -> bool {
    let rows = list(case, "row_records");
    let specs = params.lookup_specs();
    if verified.lookup_claims().len() != specs.len() {
        return false;
    }
    for (j, claim) in verified.lookup_claims().iter().enumerate() {
        let Log2LookupClaim {
            kind,
            point,
            key,
            value,
        } = claim;
        if kind != &specs[j] {
            return false;
        }
        let key_col: Vec<Fp> = rows
            .iter()
            .map(|r| Fp::reduce(u64_val(&list(r, "lookup_keys")[j])))
            .collect();
        let val_col: Vec<Fp> = rows
            .iter()
            .map(|r| Fp::reduce(u64_val(&list(r, "lookup_values")[j])))
            .collect();
        if *key != extension(&key_col, point) || *value != extension(&val_col, point) {
            return false;
        }
    }

    let z_low_len = params.k_z() - 1;
    let q_low_len = params.k_q() - 1;
    let cmp_start = 6 + z_low_len + q_low_len + 3;
    let m = params.m();

    for claim in verified.source_claims() {
        let expected = match claim.role {
            Log2Role::MatchBit => {
                let col: Vec<Fp> = rows
                    .iter()
                    .map(|r| Fp::reduce(u64_val(&list(r, "committed")[0])))
                    .collect();
                extension(&col, &claim.point)
            }
            Log2Role::ExponentKey => {
                let col: Vec<Fp> = rows
                    .iter()
                    .map(|r| Fp::reduce(u64_val(&list(r, "committed")[1])))
                    .collect();
                extension(&col, &claim.point)
            }
            Log2Role::Pow2Value => {
                let col: Vec<Fp> = rows
                    .iter()
                    .map(|r| Fp::reduce(u64_val(&list(r, "committed")[2])))
                    .collect();
                extension(&col, &claim.point)
            }
            Log2Role::FractionKey => {
                let col: Vec<Fp> = rows
                    .iter()
                    .map(|r| Fp::reduce(u64_val(&list(r, "committed")[3])))
                    .collect();
                extension(&col, &claim.point)
            }
            Log2Role::Exp2FracValue => {
                let col: Vec<Fp> = rows
                    .iter()
                    .map(|r| Fp::reduce(u64_val(&list(r, "committed")[4])))
                    .collect();
                extension(&col, &claim.point)
            }
            Log2Role::ScaledNumerator => {
                let col: Vec<Fp> = rows
                    .iter()
                    .map(|r| Fp::reduce(u64_val(&list(r, "committed")[5])))
                    .collect();
                extension(&col, &claim.point)
            }
            Log2Role::PartitionLow(h) => {
                let col: Vec<Fp> = rows
                    .iter()
                    .map(|r| Fp::reduce(u64_val(&list(r, "committed")[6 + h])))
                    .collect();
                extension(&col, &claim.point)
            }
            Log2Role::NumeratorLow(a) => {
                let col: Vec<Fp> = rows
                    .iter()
                    .map(|r| Fp::reduce(u64_val(&list(r, "committed")[6 + z_low_len + a])))
                    .collect();
                extension(&col, &claim.point)
            }
            Log2Role::Pow2Low(b) => {
                let col: Vec<Fp> = rows
                    .iter()
                    .map(|r| {
                        Fp::reduce(u64_val(
                            &list(r, "committed")[6 + z_low_len + q_low_len + b],
                        ))
                    })
                    .collect();
                extension(&col, &claim.point)
            }
            Log2Role::Compare(role, limb) => {
                let idx = cmp_start + cmp_role_index(role) * m + limb;
                let col: Vec<Fp> = rows
                    .iter()
                    .map(|r| Fp::reduce(u64_val(&list(r, "committed")[idx])))
                    .collect();
                extension(&col, &claim.point)
            }
            Log2Role::Exponentials => extension(&signed_col(member(case, "exps")), &claim.point),
            Log2Role::RecordedIndicator => {
                extension(&signed_col(member(case, "h_y")), &claim.point)
            }
            Log2Role::DecodedIndicator => {
                extension(&signed_col(member(case, "h_yhat")), &claim.point)
            }
        };
        if claim.value != expected {
            return false;
        }
    }
    true
}

fn check_log2_obligations(case: &Json, params: &Log2Params, verified: &Log2Verified) {
    assert!(
        log2_obligations_hold(case, params, verified),
        "OneSidedLog2 retained lookup/source obligations did not match ground-truth MLEs"
    );
}

#[test]
fn one_sided_log2_schema_vectors() {
    let vectors = read_vectors("gadget-finals/one-sided-log2.json");
    let schemas = list(&vectors, "schema_cases");
    assert_eq!(schemas.len(), 7);

    for sc in schemas {
        let name = member(sc, "name").as_str().expect("name");
        let vocab = u32::try_from(u64_val(member(sc, "vocab"))).expect("vocab");
        let sigma = u8::try_from(u64_val(member(sc, "sigma"))).expect("sigma");
        let eps_log2 = u8::try_from(u64_val(member(sc, "eps_log2"))).expect("eps_log2");

        let p = Log2Params::derive(vocab, sigma, eps_log2)
            .unwrap_or_else(|e| panic!("{name}: Log2Params::derive failed: {e}"));
        assert_eq!(p.vocab(), vocab, "{name}");
        assert_eq!(p.sigma(), sigma, "{name}");
        assert_eq!(p.eps_log2(), eps_log2, "{name}");
        assert_eq!(p.c_match(), u64_val(member(sc, "c_match")), "{name}");
        assert_eq!(
            p.w_z(),
            u8::try_from(u64_val(member(sc, "w_z"))).expect("w_z"),
            "{name}"
        );
        assert_eq!(
            p.k_z(),
            usize::try_from(u64_val(member(sc, "k_z"))).expect("k_z"),
            "{name}"
        );
        assert_eq!(
            p.top_z_bits(),
            u8::try_from(u64_val(member(sc, "top_z_bits"))).expect("top_z_bits"),
            "{name}"
        );
        assert_eq!(
            p.w_q(),
            u8::try_from(u64_val(member(sc, "w_q"))).expect("w_q"),
            "{name}"
        );
        assert_eq!(
            p.k_q(),
            usize::try_from(u64_val(member(sc, "k_q"))).expect("k_q"),
            "{name}"
        );
        assert_eq!(
            p.top_q_bits(),
            u8::try_from(u64_val(member(sc, "top_q_bits"))).expect("top_q_bits"),
            "{name}"
        );
        assert_eq!(
            p.m(),
            usize::try_from(u64_val(member(sc, "m"))).expect("m"),
            "{name}"
        );
        assert_eq!(
            p.committed_row_count(),
            usize::try_from(u64_val(member(sc, "committed_count"))).expect("committed_count"),
            "{name}"
        );
        assert_eq!(p.row_equality_count(), 3 * p.m() + 6, "{name}");

        let expected_specs: Vec<TableKind> = list(sc, "lookup_specs")
            .iter()
            .map(parse_table_kind)
            .collect();
        assert_eq!(p.lookup_specs(), expected_specs.as_slice(), "{name}");

        let bounds_json = list(sc, "bounds");
        let (b_left, b_right) = p.bounds();
        assert_eq!(b_left, u64_vec(&bounds_json[0]).as_slice(), "{name}");
        assert_eq!(b_right, u64_vec(&bounds_json[1]).as_slice(), "{name}");

        let cw_json = list(sc, "carry_widths");
        let (cw_left, cw_right) = p.carry_widths();
        assert_eq!(cw_left, u8_vec(&cw_json[0]).as_slice(), "{name}");
        assert_eq!(cw_right, u8_vec(&cw_json[1]).as_slice(), "{name}");

        // Schema row check (3 active rows + 1 padded row on 2 row variables => rows = 3).
        let site = OneSidedLog2::derive(GadgetId(99), vocab, 3, sigma, eps_log2)
            .expect("OneSidedLog2::derive");
        let point = fp2_vec(member(sc, "point"));
        let z_eval = fp2_num(member(sc, "z_eval"));
        let q_eval = fp2_num(member(sc, "q_eval"));
        let lookup_keys = fp2_vec(member(sc, "lookup_key_evals"));
        let lookup_values = fp2_vec(member(sc, "lookup_value_evals"));
        let rows = list(sc, "rows");
        let committed_evals: Vec<Fp2> = (0..p.committed_row_count())
            .map(|c_idx| {
                let col: Vec<Fp> = rows
                    .iter()
                    .map(|r| Fp::reduce(u64_val(&list(r, "committed")[c_idx])))
                    .collect();
                extension(&col, &point)
            })
            .collect();

        let _checked_none = site
            .check_row(Log2RowValues {
                row_target: &point,
                z_eval,
                q_eval,
                committed_evals: None,
                lookup_keys: &lookup_keys,
                lookup_values: &lookup_values,
            })
            .unwrap_or_else(|e| panic!("{name}: check_row(None) failed: {e}"));

        let _checked_some = site
            .check_row(Log2RowValues {
                row_target: &point,
                z_eval,
                q_eval,
                committed_evals: Some(&committed_evals),
                lookup_keys: &lookup_keys,
                lookup_values: &lookup_values,
            })
            .unwrap_or_else(|e| panic!("{name}: check_row(Some) failed: {e}"));

        // Tampering any committed scalar apart from m_c (index 0) must mismatch lookup/q bindings.
        for bad_idx in 1..committed_evals.len() {
            let mut tampered = committed_evals.clone();
            tampered[bad_idx] += Fp2::ONE;
            assert_eq!(
                site.check_row(Log2RowValues {
                    row_target: &point,
                    z_eval,
                    q_eval,
                    committed_evals: Some(&tampered),
                    lookup_keys: &lookup_keys,
                    lookup_values: &lookup_values,
                })
                .unwrap_err(),
                GadgetError::Rejected {
                    gadget: GadgetId(99)
                },
                "{name}: tampered committed_evals[{bad_idx}] accepted"
            );
        }
    }
}

#[test]
fn one_sided_log2_component_vectors_omission_and_adversarial_mutations() {
    let vectors = read_vectors("gadget-finals/one-sided-log2.json");
    let cases = list(&vectors, "cases");
    assert_eq!(cases.len(), 15);

    for case in cases {
        let name = member(case, "name").as_str().expect("name");
        let gadget = GadgetId(u32::try_from(u64_val(member(case, "gadget"))).expect("gadget"));
        let rows = u32::try_from(u64_val(member(case, "rows"))).expect("rows");
        let vocab = u32::try_from(u64_val(member(case, "vocab"))).expect("vocab");
        let sigma = u8::try_from(u64_val(member(case, "sigma"))).expect("sigma");
        let eps_log2 = u8::try_from(u64_val(member(case, "eps_log2"))).expect("eps_log2");
        let claimed_uhat = FixedBits(u64_val(member(case, "claimed_uhat")));

        let site = OneSidedLog2::derive(gadget, vocab, rows, sigma, eps_log2)
            .unwrap_or_else(|e| panic!("{name}: derive failed: {e}"));
        assert_eq!(site.gadget(), gadget, "{name}");
        assert_eq!(site.rows(), rows, "{name}");
        assert_eq!(
            site.params().c_match(),
            u64_val(member(case, "c_match")),
            "{name}"
        );

        let row_target = fp2_vec(member(case, "row_target"));
        let z_eval = fp2_num(member(case, "z_eval"));
        let q_eval = fp2_num(member(case, "q_eval"));
        let lookup_keys = fp2_vec(member(case, "lookup_key_evals"));
        let lookup_values = fp2_vec(member(case, "lookup_value_evals"));

        let row_vals = Log2RowValues {
            row_target: &row_target,
            z_eval,
            q_eval,
            committed_evals: None,
            lookup_keys: &lookup_keys,
            lookup_values: &lookup_values,
        };

        let local_reject = member(case, "local_reject");
        if !matches!(local_reject, Json::Null) {
            assert_eq!(
                site.check_row(row_vals).unwrap_err(),
                GadgetError::Rejected { gadget },
                "{name}: expected local_reject={local_reject:?}"
            );
            continue;
        }

        let checked = site
            .check_row(row_vals)
            .unwrap_or_else(|e| panic!("{name}: check_row failed: {e}"));
        assert_eq!(
            checked.x_targets(),
            fp2_vec(member(case, "x_targets")).as_slice(),
            "{name}: x_targets mismatch"
        );
        assert_eq!(
            checked.y_targets(),
            fp2_vec(member(case, "y_targets")).as_slice(),
            "{name}: y_targets mismatch"
        );

        let proof_bundle = member(case, "proof");
        let s1_json = member(proof_bundle, "stage1");
        let s2_json = member(proof_bundle, "stage2");
        let s1_proof = parse_proof(s1_json);
        let s2_proof = parse_proof(s2_json);

        let failed_stage = member(case, "failed_stage");
        if matches!(failed_stage, Json::Null) {
            let mut t = log2_initial_transcript(case);
            let s1 = checked
                .squeeze_stage1(&mut t, claimed_uhat)
                .expect("squeeze_stage1");
            let (b1, done1) = s1
                .verify(&mut t, &s1_proof)
                .unwrap_or_else(|e| panic!("{name}: stage1 failed: {e:?}"));
            assert_eq!(b1.point, fp2_vec(member(s1_json, "point")), "{name}");
            assert_eq!(b1.gamma, fp2_num(member(s1_json, "gamma")), "{name}");
            assert_eq!(
                *t.state(),
                digest(member(s1_json, "state").as_str().expect("s1 state")),
                "{name}"
            );
            assert_eq!(t.counter(), u64_val(member(s1_json, "counter")), "{name}");

            let s2 = done1.squeeze_stage2(&mut t).expect("squeeze_stage2");
            let (b2, verified) = s2
                .verify(&mut t, &s2_proof)
                .unwrap_or_else(|e| panic!("{name}: stage2 failed: {e:?}"));
            assert_eq!(b2.point, fp2_vec(member(s2_json, "point")), "{name}");
            assert_eq!(b2.gamma, fp2_num(member(s2_json, "gamma")), "{name}");
            assert_eq!(
                *t.state(),
                digest(member(s2_json, "state").as_str().expect("s2 state")),
                "{name}"
            );
            assert_eq!(t.counter(), u64_val(member(s2_json, "counter")), "{name}");

            check_log2_obligations(case, site.params(), &verified);

            // Perturb every single slot of Stage-1 final_evals (+1): Stage-1 verify MUST
            // reject immediately unless `slot` is `I(r_1)` or `f(r_1)` (the last 2 slots)
            // AND `1 - m(r_1) == 0` annihilates their cofactor in `B_t`, in which case the
            // retained PCS source claims MUST reject against the committed columns.
            let s1_len = s1_proof.final_evals.len();
            let m_r1_is_one = s1_proof.final_evals[0] == Fp2::ONE;
            for slot in 0..s1_len {
                let mut t_bad = log2_initial_transcript(case);
                let s1_bad = site
                    .check_row(row_vals)
                    .expect("check_row")
                    .squeeze_stage1(&mut t_bad, claimed_uhat)
                    .expect("squeeze_stage1");
                let mut mutated = s1_proof.clone();
                mutated.final_evals[slot] += Fp2::ONE;
                match s1_bad.verify(&mut t_bad, &mutated) {
                    Err(_) => {}
                    Ok((_, d1_bad)) => {
                        assert!(
                            slot >= s1_len - 2 && m_r1_is_one,
                            "{name}: stage1 slot {slot} had non-zero cofactor yet passed verify"
                        );
                        let s2_next = d1_bad.squeeze_stage2(&mut t_bad).expect("squeeze_stage2");
                        if let Ok((_, v_bad)) = s2_next.verify(&mut t_bad, &s2_proof) {
                            assert!(
                                !log2_obligations_hold(case, site.params(), &v_bad),
                                "{name}: stage1 slot {slot} perturbation escaped source claims"
                            );
                        }
                    }
                }
            }

            // Perturb every single slot of Stage-2 final_evals (+1): Stage-2 verify MUST
            // reject immediately unless `slot == 4` (`mt`) with `hy == hyhat` or `slot == 6`
            // (`hyhat`) with `mt == 0`, in which case the retained PCS source claims MUST reject.
            let hy_eq_hyhat = s2_proof.final_evals[2] == s2_proof.final_evals[6];
            let mt_is_zero = s2_proof.final_evals[4] == Fp2::ZERO;
            for slot in 0..s2_proof.final_evals.len() {
                let mut t_bad = log2_initial_transcript(case);
                let s1_ok = site
                    .check_row(row_vals)
                    .expect("check_row")
                    .squeeze_stage1(&mut t_bad, claimed_uhat)
                    .expect("squeeze_stage1");
                let (_, d1_ok) = s1_ok.verify(&mut t_bad, &s1_proof).expect("stage1");
                let s2_bad = d1_ok.squeeze_stage2(&mut t_bad).expect("squeeze_stage2");
                let mut mutated = s2_proof.clone();
                mutated.final_evals[slot] += Fp2::ONE;
                match s2_bad.verify(&mut t_bad, &mutated) {
                    Err(_) => {}
                    Ok((_, v_bad)) => {
                        let cofactor_zero = (slot == 4 && hy_eq_hyhat) || (slot == 6 && mt_is_zero);
                        assert!(
                            cofactor_zero,
                            "{name}: stage2 slot {slot} had non-zero cofactor yet passed verify"
                        );
                        assert!(
                            !log2_obligations_hold(case, site.params(), &v_bad),
                            "{name}: stage2 slot {slot} perturbation escaped source claims"
                        );
                    }
                }
            }
        } else {
            let fs = items(failed_stage);
            let fail_stage_no = usize::try_from(u64_val(&fs[0])).expect("stage_no");
            let omit_idx = usize::try_from(u64_val(&fs[1])).expect("omit_idx");

            let run_real = |bundle: &Json| -> Result<Log2Verified, (usize, StageVerifyError)> {
                let mut t = log2_initial_transcript(case);
                let s1 = site
                    .check_row(row_vals)
                    .expect("check_row")
                    .squeeze_stage1(&mut t, claimed_uhat)
                    .expect("squeeze_stage1");
                let (_, done1) = s1
                    .verify(&mut t, &parse_proof(member(bundle, "stage1")))
                    .map_err(|e| (1, e))?;
                let s2 = done1.squeeze_stage2(&mut t).expect("squeeze_stage2");
                let (_, v) = s2
                    .verify(&mut t, &parse_proof(member(bundle, "stage2")))
                    .map_err(|e| (2, e))?;
                Ok(v)
            };

            let expected_round = u32::try_from(if fail_stage_no == 1 {
                site.row_variables()
            } else {
                site.variables()
            })
            .expect("u32");
            let expected_err = (
                fail_stage_no,
                StageVerifyError::Sumcheck(SumcheckError::FinalMismatch {
                    round: expected_round,
                }),
            );

            // 1. Both raw proof and omission proof fail at exact stage under real verifier.
            assert_eq!(
                run_real(proof_bundle),
                Err(expected_err.clone()),
                "{name}: raw"
            );
            let om_bundle = member(case, "omission");
            assert_eq!(run_real(om_bundle), Err(expected_err), "{name}: omission");

            // 2. Adapted omission control omitting ONLY `(fail_stage_no, omit_idx)` passes all checks.
            let om_s1_json = member(om_bundle, "stage1");
            let om_s2_json = member(om_bundle, "stage2");
            let om_s1 = parse_proof(om_s1_json);
            let om_s2 = parse_proof(om_s2_json);

            let mut t_ctrl = log2_initial_transcript(case);
            let s1_ctrl = site
                .check_row(row_vals)
                .expect("check_row")
                .squeeze_stage1(&mut t_ctrl, claimed_uhat)
                .expect("squeeze_stage1");
            let b1 = sumcheck::verify_batch(
                &mut t_ctrl,
                Prefix::Stage(1),
                s1_ctrl.degree(),
                s1_ctrl.instances(),
                &om_s1,
                |i, pt, vals| {
                    if fail_stage_no == 1 && i == omit_idx {
                        Some(s1_ctrl.instances()[i].claim * delta0(pt))
                    } else {
                        s1_ctrl.evaluate(i, pt, vals).ok()
                    }
                },
            )
            .unwrap_or_else(|e| panic!("{name}: stage1 omission control failed: {e:?}"));
            let done1 = s1_ctrl
                .finish(b1.point, &om_s1.final_evals)
                .expect("stage1 finish");
            assert_eq!(
                *t_ctrl.state(),
                digest(member(om_s1_json, "state").as_str().expect("s1 state")),
                "{name}"
            );
            assert_eq!(
                t_ctrl.counter(),
                u64_val(member(om_s1_json, "counter")),
                "{name}"
            );

            let s2_ctrl = done1.squeeze_stage2(&mut t_ctrl).expect("squeeze_stage2");
            let b2 = sumcheck::verify_batch(
                &mut t_ctrl,
                Prefix::Stage(2),
                s2_ctrl.degree(),
                s2_ctrl.instances(),
                &om_s2,
                |i, pt, vals| {
                    if fail_stage_no == 2 && i == omit_idx {
                        Some(s2_ctrl.instances()[i].claim * delta0(pt))
                    } else {
                        s2_ctrl.evaluate(i, pt, vals).ok()
                    }
                },
            )
            .unwrap_or_else(|e| panic!("{name}: stage2 omission control failed: {e:?}"));
            let verified = s2_ctrl
                .finish(b2.point, &om_s2.final_evals)
                .expect("stage2 finish");
            assert_eq!(
                *t_ctrl.state(),
                digest(member(om_s2_json, "state").as_str().expect("s2 state")),
                "{name}"
            );
            assert_eq!(
                t_ctrl.counter(),
                u64_val(member(om_s2_json, "counter")),
                "{name}"
            );

            check_log2_obligations(case, site.params(), &verified);
        }
    }
}

#[test]
fn one_sided_log2_noncanonical_uhat_and_consistency_guards() {
    let vectors = read_vectors("gadget-finals/one-sided-log2.json");
    // Use `wide-sigma32` (k_z = 3, k_q = 4, y_active = 5) so z_low and repeated qp blocks are non-trivial.
    let case = list(&vectors, "cases")
        .iter()
        .find(|c| member(c, "name").as_str() == Some("wide-sigma32"))
        .expect("wide-sigma32 case");

    let gadget = GadgetId(777);
    let rows = u32::try_from(u64_val(member(case, "rows"))).expect("rows");
    let vocab = u32::try_from(u64_val(member(case, "vocab"))).expect("vocab");
    let sigma = u8::try_from(u64_val(member(case, "sigma"))).expect("sigma");
    let eps_log2 = u8::try_from(u64_val(member(case, "eps_log2"))).expect("eps_log2");
    let actual_uhat = u64_val(member(case, "claimed_uhat"));

    let site = OneSidedLog2::derive(gadget, vocab, rows, sigma, eps_log2).expect("derive");
    let row_target = fp2_vec(member(case, "row_target"));
    let z_eval = fp2_num(member(case, "z_eval"));
    let q_eval = fp2_num(member(case, "q_eval"));
    let lookup_keys = fp2_vec(member(case, "lookup_key_evals"));
    let lookup_values = fp2_vec(member(case, "lookup_value_evals"));

    let row_vals = Log2RowValues {
        row_target: &row_target,
        z_eval,
        q_eval,
        committed_evals: None,
        lookup_keys: &lookup_keys,
        lookup_values: &lookup_values,
    };

    // 1. Non-canonical claimed_uhat >= P (including P + actual_uhat which wraps mod P!) MUST reject.
    for noncanonical in [Fp::MODULUS, Fp::MODULUS + actual_uhat, u64::MAX] {
        let mut t = log2_initial_transcript(case);
        let checked = site.check_row(row_vals).expect("check_row");
        assert_eq!(
            checked
                .squeeze_stage1(&mut t, FixedBits(noncanonical))
                .unwrap_err(),
            StageVerifyError::Gadget(GadgetError::Rejected { gadget }),
            "noncanonical uhat {noncanonical} must reject without mod-P wrap"
        );
    }

    // 2. Non-zero Range lookup value at check_row must reject.
    let mut bad_lvals = lookup_values.clone();
    bad_lvals[2] = Fp2::ONE;
    assert_eq!(
        site.check_row(Log2RowValues {
            lookup_values: &bad_lvals,
            ..row_vals
        })
        .unwrap_err(),
        GadgetError::Rejected { gadget }
    );

    // 3. Wrong lookup slice length -> Rejected; wrong row_target length -> Internal.
    assert_eq!(
        site.check_row(Log2RowValues {
            lookup_keys: &lookup_keys[..lookup_keys.len() - 1],
            ..row_vals
        })
        .unwrap_err(),
        GadgetError::Rejected { gadget }
    );
    assert_eq!(
        site.check_row(Log2RowValues {
            row_target: &[Fp2::ZERO; 5],
            ..row_vals
        })
        .unwrap_err(),
        GadgetError::Internal
    );

    // 4. Direct cross-instance equality checks in Log2Stage1::finish and Log2Stage2::finish.
    let proof_bundle = member(case, "proof");
    let s1_proof = parse_proof(member(proof_bundle, "stage1"));
    let s2_proof = parse_proof(member(proof_bundle, "stage2"));
    let r1 = fp2_vec(member(member(proof_bundle, "stage1"), "point"));
    let r2 = fp2_vec(member(member(proof_bundle, "stage2"), "point"));

    let p = site.params();
    let z_low_len = p.k_z() - 1;
    let qp_stride = p.k_q() + 4;
    let x_evals_end = 4 + z_low_len + p.k_z();
    let y_evals_end = x_evals_end + p.y_active() * qp_stride;

    // Indices covering every distinct cross-instance tie in Log2Stage1::finish:
    // - 4: z_low_single[0] != z_low_top[0]
    // - x_evals_end: q_qp != q_r1 (when mutated together across all qp blocks, tested below, or single block mismatch)
    // - x_evals_end + qp_stride: second qp block disagrees with first
    // - y_evals_end: z_pad != z_r1
    // - y_evals_end + 1: m_uhat != m_r1
    for bad_slot in [
        4usize,
        x_evals_end + qp_stride,
        y_evals_end,
        y_evals_end + 1,
    ] {
        let mut t = log2_initial_transcript(case);
        let s1 = site
            .check_row(row_vals)
            .expect("check_row")
            .squeeze_stage1(&mut t, FixedBits(actual_uhat))
            .expect("squeeze_stage1");
        let mut bad_evals = s1_proof.final_evals.clone();
        bad_evals[bad_slot] += Fp2::ONE;
        assert_eq!(
            s1.finish(r1.clone(), &bad_evals).unwrap_err(),
            GadgetError::Rejected { gadget }
        );
    }
    // Mutating q across ALL qp blocks while leaving q_r1 (slot 1) untouched must still fail `q_qp != q_r1`!
    {
        let mut t = log2_initial_transcript(case);
        let s1 = site
            .check_row(row_vals)
            .expect("check_row")
            .squeeze_stage1(&mut t, FixedBits(actual_uhat))
            .expect("squeeze_stage1");
        let mut bad_evals = s1_proof.final_evals.clone();
        for blk in 0..p.y_active() {
            bad_evals[x_evals_end + blk * qp_stride] += Fp2::ONE;
        }
        assert_eq!(
            s1.finish(r1.clone(), &bad_evals).unwrap_err(),
            GadgetError::Rejected { gadget }
        );
    }

    // Stage-2 finish cross-instance ties: [ev0, ev1, hy0, ev2, mt, hy1, hyhat].
    for bad_slot in [1usize, 3, 5] {
        let mut t = log2_initial_transcript(case);
        let s1 = site
            .check_row(row_vals)
            .expect("check_row")
            .squeeze_stage1(&mut t, FixedBits(actual_uhat))
            .expect("squeeze_stage1");
        let (_, done1) = s1.verify(&mut t, &s1_proof).expect("stage1");
        let s2 = done1.squeeze_stage2(&mut t).expect("squeeze_stage2");
        let mut bad_evals = s2_proof.final_evals.clone();
        bad_evals[bad_slot] += Fp2::ONE;
        assert_eq!(
            s2.finish(r2.clone(), &bad_evals).unwrap_err(),
            GadgetError::Rejected { gadget }
        );
    }

    // 5. Out-of-domain OneSidedLog2::derive parameters return GadgetError::Internal.
    assert_eq!(
        OneSidedLog2::derive(gadget, 0, 4, 8, 6).unwrap_err(),
        GadgetError::Internal
    );
    assert_eq!(
        OneSidedLog2::derive(gadget, (1 << 18) + 1, 4, 8, 6).unwrap_err(),
        GadgetError::Internal
    );
    assert_eq!(
        OneSidedLog2::derive(gadget, 8, 0, 8, 6).unwrap_err(),
        GadgetError::Internal
    );
    assert_eq!(
        OneSidedLog2::derive(gadget, 8, (1 << 17) + 1, 8, 6).unwrap_err(),
        GadgetError::Internal
    );
    assert_eq!(
        OneSidedLog2::derive(gadget, 8, 4, 33, 6).unwrap_err(),
        GadgetError::Internal
    );
    assert_eq!(
        OneSidedLog2::derive(gadget, 8, 4, 8, 17).unwrap_err(),
        GadgetError::Internal
    );
}
