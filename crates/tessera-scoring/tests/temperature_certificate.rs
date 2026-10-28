// SPDX-License-Identifier: Apache-2.0 OR MIT

//! End-to-end conformance and adversarial tests for `VocabMax` + `InverseCdf` (`10^5` random logit
//! distributions), the in-circuit Sampler PRF (`derive.txt` + `prf.txt`), `cdf_columns_and_requests`
//! parity, and `ExactTemperature` (#152, RFC-0017, RFC-0058, RFC-0061).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use tessera_anchor::sampler_commitment;
use tessera_codec::error::{ErrorCode, RequestId, TokenPosition};
use tessera_codec::{Ascii, DecodeLimits, HeapKey, Rational, Secret, SecretBuf, envelope, schema};
use tessera_compiler::anchor::{
    AesTables, PinnedSamplerPrf, SamplerPinPublic, StandaloneSamplerPrf, build_pinned_sampler_prf,
    build_standalone_sampler_prf,
};
use tessera_compiler::schedule::derive_graph_schedule;
use tessera_compiler::{
    CdfAdviceColumn, CompareColumn as CompilerCompareColumn, Lowering, PublicModel, analyze,
    cdf_columns_and_requests, lower, settle, vocab_exp_scale,
};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gadgets::{
    CdfColumn, CdfConstraints, CdfTerm, CompareColumn as GadgetCompareColumn, Edge, InverseCdf,
    OneHotWitness, Side, VocabMax,
};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_ir::{Graph, GraphBuilder, IntRange, IntegerModel};
use tessera_sumcheck::{Instance, Prefix, Protocol, Table as ScTable, prove_batch};
use tessera_tables::{BitOp, GeneratedTable, Table, TableId, TableKind, TableSpec};
use tessera_witness::{Engine, WitnessError};

use tessera_scoring::{
    ClassBits, ExactTemperature, SamplerBaseKey, SamplerKey, ScoringError, TemperatureInputs,
    inverse_cdf,
};
use tv_core::sumcheck as tv;
use tv_field::{Fp as TvFp, Fp2 as TvFp2};
use tv_hash::transcript as vt;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn range_specs() -> Vec<TableSpec> {
    (1..=32)
        .map(|bits| {
            GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                .expect("Range")
                .spec()
                .clone()
        })
        .collect()
}

fn aes_tables_and_specs() -> (AesTables, Vec<TableSpec>) {
    let kinds = [
        (0u32, TableKind::AesSbox),
        (1, TableKind::Gf256Mul { c: 2 }),
        (2, TableKind::Gf256Mul { c: 3 }),
        (3, TableKind::Bitwise8 { op: BitOp::Xor }),
    ];
    let specs = kinds
        .into_iter()
        .map(|(id, kind)| {
            GeneratedTable::new(TableId(id), kind)
                .unwrap()
                .spec()
                .clone()
        })
        .collect();
    (
        AesTables {
            sbox: TableId(0),
            mul2: TableId(1),
            mul3: TableId(2),
            xor: TableId(3),
        },
        specs,
    )
}

fn hex_16(hex: &str) -> [u8; 16] {
    assert_eq!(hex.len(), 32);
    core::array::from_fn(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
}

fn hex_32(hex: &str) -> [u8; 32] {
    assert_eq!(hex.len(), 64);
    core::array::from_fn(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
}

fn parse_prompt(column: &str) -> Vec<u32> {
    match column {
        "-" => Vec::new(),
        list => list.split(',').map(|x| x.parse().unwrap()).collect(),
    }
}

fn map_col(col: CdfColumn) -> CdfAdviceColumn {
    let map_cmp = |c: GadgetCompareColumn| match c {
        GadgetCompareColumn::Digit(Side::Left, limb) => {
            (CompilerCompareColumn::LeftByte, limb as u32)
        }
        GadgetCompareColumn::Carry(Side::Left, limb) => {
            (CompilerCompareColumn::LeftCarry, limb as u32)
        }
        GadgetCompareColumn::Digit(Side::Right, limb) => {
            (CompilerCompareColumn::RightByte, limb as u32)
        }
        GadgetCompareColumn::Carry(Side::Right, limb) => {
            (CompilerCompareColumn::RightCarry, limb as u32)
        }
        GadgetCompareColumn::Difference(limb) => (CompilerCompareColumn::Difference, limb as u32),
        GadgetCompareColumn::Borrow(limb) => (CompilerCompareColumn::Borrow, limb as u32),
        GadgetCompareColumn::Input(_, _) => panic!("InverseCdf never commits CompareColumn::Input"),
    };
    match col {
        CdfColumn::DrawLimb => CdfAdviceColumn::DrawLimb,
        CdfColumn::TotalLimb(j) => CdfAdviceColumn::TotalLimb(j),
        CdfColumn::BelowLimb(j) => CdfAdviceColumn::BelowLimb(j),
        CdfColumn::Compare(edge, cmp) => {
            let (column, limb) = map_cmp(cmp);
            CdfAdviceColumn::Comparison {
                upper: edge == Edge::Upper,
                column,
                limb,
            }
        }
    }
}

#[test]
fn compiler_cdf_schema_matches_gadget_constraints_column_and_request_for_request() {
    let ranges = range_specs();
    for &(vocab, sigma) in &[
        (1usize, 0u8),
        (3, 8),
        (16, 11),
        (16, 12),
        (65, 16),
        (1024, 24),
        (1 << 18, 32),
    ] {
        for bounded in [false, true] {
            let (cols, widths) = cdf_columns_and_requests(vocab as u32, sigma, bounded).unwrap();
            let gadget = InverseCdf::<G>::new(vocab, sigma, &ranges).unwrap();
            let c: CdfConstraints<G> = gadget.constraints().unwrap();
            let mut gadget_cols = Vec::new();
            if bounded {
                gadget_cols.push(CdfAdviceColumn::SampledIndicator);
            }
            gadget_cols.extend(c.committed.iter().copied().map(map_col));
            let gadget_widths: Vec<u8> = c
                .lookups
                .iter()
                .map(|l| match ranges[l.table.0 as usize - 1].kind {
                    TableKind::Range { bits } => bits,
                    _ => unreachable!(),
                })
                .collect();
            assert_eq!(
                cols, gadget_cols,
                "vocab {vocab}, sigma {sigma}, bounded {bounded}"
            );
            assert_eq!(
                widths, gadget_widths,
                "vocab {vocab}, sigma {sigma}, bounded {bounded}"
            );
        }
    }
}

#[test]
fn vocab_max_plus_inverse_cdf_agrees_with_reference_sampler_on_100000_logit_rows() {
    let ranges = range_specs();
    let configs = [
        (8u8, -1i8, Rational::new(1, 2).unwrap(), 10u8),
        (16, 0, Rational::new(1, 1).unwrap(), 10),
        (24, -2, Rational::new(3, 2).unwrap(), 12),
        (32, -1, Rational::new(4, 1).unwrap(), 14),
    ];
    let exp_specs: Vec<TableSpec> = configs
        .iter()
        .enumerate()
        .map(|(i, &(sigma, exp_e, temp, key_bits))| {
            let scale = vocab_exp_scale(exp_e, temp).unwrap();
            GeneratedTable::new(
                TableId(100 + i as u32),
                TableKind::Exp {
                    key_bits,
                    sigma,
                    scale,
                },
            )
            .unwrap()
            .spec()
            .clone()
        })
        .collect();

    type CachedPair = (VocabMax<G>, InverseCdf<G>, CdfConstraints<G>);
    let mut cache: HashMap<(usize, usize), CachedPair> = HashMap::new();
    let mut rng = SplitMix64(0x0152_0017);

    for case in 0..100_000 {
        let cfg_idx = (rng.next() % configs.len() as u64) as usize;
        let vocab = 1 + (rng.next() % 32) as usize;
        let pv = vocab.next_power_of_two();
        let (vocab_max, cdf, constraints) = cache.entry((cfg_idx, vocab)).or_insert_with(|| {
            let vm = VocabMax::<G>::new(vocab, &exp_specs[cfg_idx]).unwrap();
            let ic = InverseCdf::<G>::new(vocab, vm.sigma(), &ranges).unwrap();
            let cc = ic.constraints().unwrap();
            (vm, ic, cc)
        });

        let center = (rng.next() % 401) as i64 - 200;
        let mut logits = vec![G::ZERO; pv];
        for slot in logits.iter_mut().take(vocab) {
            let delta = (rng.next() % 128) as i64 - 64;
            *slot = G::from_i64(center + delta);
        }
        let u = rng.next() as u32;
        let draw = G::from_u64_reduced(u64::from(u));

        let vw = vocab_max.witness(&logits, true).unwrap();
        let weights: Vec<u64> = vw.exponentials()[..vocab]
            .iter()
            .map(|x| x.to_canonical_u64())
            .collect();
        let expected_token = inverse_cdf(&weights, u).unwrap();

        let h = cdf.sample(vw.exponentials(), draw, true).unwrap();
        assert_eq!(h.indicator()[expected_token], G::ONE, "case {case}");
        let cw = cdf.witness(vw.exponentials(), draw, &h, true).unwrap();
        let sums = cdf.evaluate(vw.exponentials(), &h).unwrap().sums;
        let val = |t: CdfTerm| match t {
            CdfTerm::Draw => draw,
            CdfTerm::Total => sums[0],
            CdfTerm::Below => sums[1],
            CdfTerm::Selected => sums[2],
            CdfTerm::RowSelector => G::ONE,
            CdfTerm::Column(col) => G::from_i64(cw.value(col).unwrap()),
        };
        assert!(
            constraints
                .equalities
                .iter()
                .all(|e| e.evaluate(val) == G::ZERO),
            "case {case}"
        );
    }
}

#[test]
fn in_circuit_pinned_sampler_prf_matches_all_46_derive_vectors_and_rejects_wrong_commitment() {
    let (aes_tables, specs) = aes_tables_and_specs();
    let prompt = Secret::new(vec![1u32, 2, 3]);
    let mut cache: HashMap<u8, (Graph, PinnedSamplerPrf)> = HashMap::new();
    let salt_bytes = [0xa5u8; 16];
    let salt = Secret::new(HeapKey::from_box(Box::new(salt_bytes)));

    let mut count = 0usize;
    for line in include_str!("../../../spec/vectors/scoring/derive.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let cols: Vec<&str> = line.split_whitespace().collect();
        let [
            base_hex,
            nonce_hex,
            bits_str,
            from_ns_str,
            key_hex,
            class_str,
            u0_str,
        ] = cols[..]
        else {
            panic!("unexpected derive row: {cols:?}");
        };
        let class_bits: u8 = bits_str.parse().unwrap();
        let from_ns: u64 = from_ns_str.parse().unwrap();
        let expected_key = hex_16(key_hex);
        let expected_class: u64 = class_str.parse().unwrap();
        let expected_u0: u32 = u0_str.parse().unwrap();

        let (graph, circuit) = cache.entry(class_bits).or_insert_with(|| {
            let mut gb = GraphBuilder::new();
            let c = build_pinned_sampler_prf(&mut gb, &aes_tables, class_bits, 3, 1).unwrap();
            let mut graph = gb.finish();
            let ranges = analyze(&graph, &specs).expect("pinned PRF range analysis");
            assert_eq!(
                ranges.of(c.draws).unwrap().bounded(),
                Some(IntRange {
                    lo: 0,
                    hi: (1 << 32) - 1,
                })
            );
            settle(&mut graph, &ranges).unwrap();
            let _schedule = derive_graph_schedule(&graph, &specs, &[], Some(c.draws))
                .expect("pinned PRF schedule");
            (graph, c)
        });

        let base = SamplerBaseKey::new(Secret::new(HeapKey::from_box(Box::new(hex_16(base_hex)))));
        let pin = SamplerPinPublic {
            sampler_commitment: sampler_commitment(base.secret(), &salt),
            nonce: hex_32(nonce_hex),
            class_bits,
            from_ns,
        };
        let engine = Engine::for_graph(graph, &specs).unwrap();
        let inputs = base.pinned_prf_inputs(&salt, &pin, &prompt, &[0], circuit);
        let w = engine.evaluate_secret(inputs).unwrap();

        let got_key: Vec<u8> = w
            .tensor(circuit.derive.derived_key)
            .unwrap()
            .iter()
            .map(|&b| u8::try_from(b).unwrap())
            .collect();
        assert_eq!(got_key.as_slice(), &expected_key, "derive row {count} key");

        if let Some((_, _, ref class_in, _)) = circuit.class_stage {
            let cb = w.tensor(class_in.class_bytes).unwrap();
            let mut le = [0u8; 8];
            for (i, &b) in cb.iter().enumerate() {
                le[i] = u8::try_from(b).unwrap();
            }
            assert_eq!(
                u64::from_le_bytes(le),
                expected_class,
                "derive row {count} class"
            );
        } else {
            assert_eq!(expected_class, 0);
        }

        let got_u0 = u32::try_from(w.tensor(circuit.draws).unwrap()[0]).unwrap();
        assert_eq!(got_u0, expected_u0, "derive row {count} u_0");

        if count == 0 {
            // Wrong sampler_commitment or wrong salt must fail the SHA-256 AssertEqual binding
            // with WitnessError::Unequal (ERR-WITNESS-ANCHOR).
            let mut bad_pin = pin;
            bad_pin.sampler_commitment[0] ^= 1;
            let err = engine
                .evaluate_secret(base.pinned_prf_inputs(&salt, &bad_pin, &prompt, &[0], circuit))
                .unwrap_err();
            assert!(matches!(err, WitnessError::Unequal { .. }));
            assert_eq!(err.code(), ErrorCode::WitnessAnchor);

            let bad_salt = Secret::new(HeapKey::from_box(Box::new([0x5au8; 16])));
            let err_salt = engine
                .evaluate_secret(base.pinned_prf_inputs(&bad_salt, &pin, &prompt, &[0], circuit))
                .unwrap_err();
            assert!(matches!(err_salt, WitnessError::Unequal { .. }));
        }
        count += 1;
    }
    assert_eq!(count, 46);
}

#[test]
fn in_circuit_standalone_sampler_prf_matches_all_369_prf_vectors() {
    let (aes_tables, specs) = aes_tables_and_specs();
    let mut cache: HashMap<(u8, u32), (Graph, StandaloneSamplerPrf)> = HashMap::new();

    let mut count = 0usize;
    for line in include_str!("../../../spec/vectors/scoring/prf.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let cols: Vec<&str> = line.split_whitespace().collect();
        let [
            key_hex,
            bits_str,
            prompt_str,
            t_str,
            class_str,
            _cipher_hex,
            u_str,
        ] = cols[..]
        else {
            panic!("unexpected prf row: {cols:?}");
        };
        let class_bits: u8 = bits_str.parse().unwrap();
        let tokens = parse_prompt(prompt_str);
        let prompt_len = tokens.len() as u32;
        let t: u32 = t_str.parse().unwrap();
        let expected_class: u64 = class_str.parse().unwrap();
        let expected_u: u32 = u_str.parse().unwrap();

        let (graph, circuit) = cache.entry((class_bits, prompt_len)).or_insert_with(|| {
            let mut gb = GraphBuilder::new();
            let c = build_standalone_sampler_prf(&mut gb, &aes_tables, class_bits, prompt_len, 1)
                .unwrap();
            let mut graph = gb.finish();
            let ranges = analyze(&graph, &specs).expect("standalone PRF range analysis");
            settle(&mut graph, &ranges).unwrap();
            let _schedule = derive_graph_schedule(&graph, &specs, &[], Some(c.draws))
                .expect("standalone PRF schedule");
            (graph, c)
        });

        let sk = SamplerKey::new(
            Secret::new(HeapKey::from_box(Box::new(hex_16(key_hex)))),
            ClassBits::new(class_bits).unwrap(),
        );
        let prompt = Secret::new(tokens);
        let engine = Engine::for_graph(graph, &specs).unwrap();
        let w = engine
            .evaluate_secret(sk.standalone_prf_inputs(&prompt, &[t], circuit))
            .unwrap();

        if let Some((_, _, _, _, ref class_in, _)) = circuit.class_stage {
            let cb = w.tensor(class_in.class_bytes).unwrap();
            let mut le = [0u8; 8];
            for (i, &b) in cb.iter().enumerate() {
                le[i] = u8::try_from(b).unwrap();
            }
            assert_eq!(
                u64::from_le_bytes(le),
                expected_class,
                "prf row {count} class"
            );
        } else {
            assert_eq!(expected_class, 0);
        }

        let got_u = u32::try_from(w.tensor(circuit.draws).unwrap()[0]).unwrap();
        assert_eq!(got_u, expected_u, "prf row {count} u");
        count += 1;
    }
    assert_eq!(count, 369);
}

fn tiny_lowering(len: u32) -> (IntegerModel, Lowering) {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/models/tiny-2-servable.d");
    let bytes = fs::read(path).unwrap();
    let model: IntegerModel =
        envelope::open(&bytes, &schema::INTEGER_MODEL, DecodeLimits::INTEGER_MODEL).unwrap();
    let mut lowering = lower(PublicModel::of(&model), len).unwrap();
    let ranges = analyze(&lowering.graph, &lowering.tables).unwrap();
    settle(&mut lowering.graph, &ranges).unwrap();
    (model, lowering)
}

#[test]
fn exact_temperature_validates_tables_batches_degree_three_relations_and_redacts_secrets() {
    let (model, lowering) = tiny_lowering(5);
    let vocab = model.arch.vocab as usize;
    let pv = vocab.next_power_of_two();
    let logits_exp = model.quant.public.logits_exponent;
    let sigma = model.quant.public.exp_sigma;
    let ranges = range_specs();

    let logit_range = lowering.graph.tensors[lowering.logits.0 as usize].range;
    let span = (logit_range.hi - logit_range.lo) as u32;
    let min_bits = (u32::BITS - span.leading_zeros()) as u8;

    let t1 = Rational::new(1, 1).unwrap();
    let t2 = Rational::new(2, 1).unwrap();
    let scale1 = vocab_exp_scale(logits_exp, t1).unwrap();
    let scale2 = vocab_exp_scale(logits_exp, t2).unwrap();
    let exp1 = GeneratedTable::new(
        TableId(500),
        TableKind::Exp {
            key_bits: min_bits.max(20),
            sigma,
            scale: scale1,
        },
    )
    .unwrap()
    .spec()
    .clone();
    let exp2 = GeneratedTable::new(
        TableId(501),
        TableKind::Exp {
            key_bits: (min_bits + 1).max(21),
            sigma,
            scale: scale2,
        },
    )
    .unwrap()
    .spec()
    .clone();

    // Constructor refusals: empty, > 4 temperatures, duplicate temperature, wrong scale, narrow
    // spread coverage (SP-7), and missing zero tail.
    assert!(matches!(
        ExactTemperature::new(&lowering, logits_exp, &[], &ranges),
        Err(ScoringError::Usage { .. })
    ));
    let five: Vec<(Rational, TableSpec)> = [1i64, 2, 3, 4, 5]
        .into_iter()
        .map(|d| (Rational::new(4, d as u64).unwrap(), exp1.clone()))
        .collect();
    assert!(matches!(
        ExactTemperature::new(&lowering, logits_exp, &five, &ranges),
        Err(ScoringError::Usage { .. })
    ));
    assert!(matches!(
        ExactTemperature::new(
            &lowering,
            logits_exp,
            &[(t1, exp1.clone()), (t1, exp1.clone())],
            &ranges
        ),
        Err(ScoringError::Usage { .. })
    ));
    assert!(matches!(
        ExactTemperature::new(&lowering, logits_exp, &[(t1, exp2.clone())], &ranges),
        Err(ScoringError::Usage { .. })
    ));
    // At key_bits = 8 under the model's logits_exponent, Exp_T[255] != 0 so VocabMax rejects the
    // table before spread checking (GadgetError::Table -> ScoringError::Gadget).
    let no_zero_tail = GeneratedTable::new(
        TableId(502),
        TableKind::Exp {
            key_bits: 8,
            sigma,
            scale: scale1,
        },
    )
    .unwrap()
    .spec()
    .clone();
    assert!(matches!(
        ExactTemperature::new(&lowering, logits_exp, &[(t1, no_zero_tail)], &ranges),
        Err(ScoringError::Gadget(_))
    ));
    // With logits_exponent = 2 and T = 1/4 (scale = 16), key_bits = 6 already reaches Exp_T[63] = 0,
    // so VocabMax accepts the table and ExactTemperature rejects because 64 <= public logit spread (SP-7)!
    let steep_t = Rational::new(1, 4).unwrap();
    let steep_narrow = GeneratedTable::new(
        TableId(503),
        TableKind::Exp {
            key_bits: 6,
            sigma,
            scale: vocab_exp_scale(2, steep_t).unwrap(),
        },
    )
    .unwrap()
    .spec()
    .clone();
    assert!(matches!(
        ExactTemperature::new(&lowering, 2, &[(steep_t, steep_narrow)], &ranges),
        Err(ScoringError::Usage { .. })
    ));

    let scorer =
        ExactTemperature::new(&lowering, logits_exp, &[(t1, exp1), (t2, exp2)], &ranges).unwrap();
    assert!(!format!("{scorer:?}").contains("500"));

    // Prompt-only request (output_len == 0) has 0 padded rows and claims uhat == 0.
    let empty_req = scorer.request(RequestId(7), t1, 5, 0).unwrap();
    assert_eq!(empty_req.padded_rows(), 0);
    let empty_cert = empty_req.witness(core::iter::empty()).unwrap();
    assert_eq!(empty_cert.uhat(), 0);

    // 3 real outputs -> padded_rows == 4.
    let req = scorer.request(RequestId(99), t1, 2, 3).unwrap();
    assert_eq!(req.padded_rows(), 4);
    assert!(req.row(2).unwrap().is_some());
    assert!(req.row(3).unwrap().is_none());

    let make_row = |seed: i64, draw_u32: u32, force_token: Option<usize>| -> TemperatureInputs {
        let mut logits = SecretBuf::with_capacity(pv);
        for v in 0..pv {
            if v < vocab {
                logits.push(G::from_i64(seed + (v as i64 * 3) % 11));
            } else {
                logits.push(G::ZERO);
            }
        }
        let draw = G::from_u64_reduced(u64::from(draw_u32));
        let next_token = match force_token {
            None => {
                let vw = req.vocab_max().witness(&logits, true).unwrap();
                req.cdf().sample(vw.exponentials(), draw, true).unwrap()
            }
            Some(tok) => {
                let mut ind = SecretBuf::filled(G::ZERO, pv);
                ind[tok] = G::ONE;
                OneHotWitness::from_column(ind)
            }
        };
        TemperatureInputs {
            logits,
            next_token,
            draw,
        }
    };

    let honest_inputs = vec![
        make_row(100, 0x0102_0304, None),
        make_row(-50, 0x8000_0000, None),
        make_row(0, 0xffff_fffe, None),
    ];
    assert!(!format!("{:?}", honest_inputs[0]).contains("100"));

    // Mismatches at both index 1 and index 2 report the FIRST mismatch (position 2 + 1 = 3)
    // after scanning all rows, with no secret value in Debug/Display/context!
    let mut bad_inputs = vec![
        make_row(100, 0x0102_0304, None),
        make_row(-50, 0x8000_0000, None),
        make_row(0, 0xffff_fffe, None),
    ];
    for idx in [1usize, 2] {
        let chosen = (0..vocab)
            .find(|&v| bad_inputs[idx].next_token.indicator()[v] == G::ONE)
            .unwrap();
        bad_inputs[idx] = make_row(
            if idx == 1 { -50 } else { 0 },
            if idx == 1 { 0x8000_0000 } else { 0xffff_fffe },
            Some((chosen + 1) % vocab),
        );
    }
    let mismatch = req.witness(bad_inputs).unwrap_err();
    assert_eq!(mismatch.code(), ErrorCode::WitnessExactMismatch);
    assert_eq!(mismatch.context().request, Some(RequestId(99)));
    assert_eq!(mismatch.context().position, Some(TokenPosition(3)));

    // Honest witness + prepare_batch proves all batched degree-3 contractions and row equalities!
    let cert = req
        .witness(vec![
            make_row(100, 0x0102_0304, None),
            make_row(-50, 0x8000_0000, None),
            make_row(0, 0xffff_fffe, None),
        ])
        .unwrap();
    assert_eq!(cert.uhat(), 0);
    assert!(!format!("{cert:?}").contains("100"));

    let batch = cert.prepare_batch(&honest_inputs).unwrap();
    assert_eq!(batch.padded_rows(), 4);
    assert_eq!(batch.real_rows(), 3);

    let contractions = batch.cdf_contractions::<F>().unwrap();
    assert_eq!(contractions.total.degree(), 3);
    assert_eq!(contractions.below.degree(), 3);
    assert_eq!(contractions.selected.degree(), 3);

    let tvf = |x: F| -> TvFp2 {
        let [c0, c1] = x.coeffs();
        TvFp2::new(
            TvFp::from_canonical(c0.to_canonical_u64()).unwrap(),
            TvFp::from_canonical(c1.to_canonical_u64()).unwrap(),
        )
    };
    let lift = |x: TvFp2| -> F {
        F::new(
            G::from_canonical_u64(x.c0.to_canonical()).unwrap(),
            G::from_canonical_u64(x.c1.to_canonical()).unwrap(),
        )
    };
    let mle = |xs: &[G], p: &[TvFp2]| -> TvFp2 {
        assert_eq!(xs.len(), 1usize << p.len());
        xs.iter().enumerate().fold(TvFp2::ZERO, |s, (i, &x)| {
            let eq = p.iter().enumerate().fold(TvFp2::ONE, |a, (j, &r)| {
                a * if (i >> j) & 1 == 0 { TvFp2::ONE - r } else { r }
            });
            s + eq * TvFp2::from(TvFp::from_canonical(x.to_canonical_u64()).unwrap())
        })
    };
    let fresh_pair = |marker: u8| {
        let mut sid = [0x55; 16];
        sid[15] = marker;
        (
            Transcript::new(&SessionContext {
                protocol: Ascii::new("tessera-protocol/1").unwrap(),
                mode: Mode::NonInteractive,
                params: ParamSetId(Ascii::new("test-small/1").unwrap()),
                statement_digest: [0x44; 32],
                session_id: sid,
            }),
            vt::Transcript::new(&vt::SessionContext {
                protocol: vt::Ascii::new("tessera-protocol/1").unwrap(),
                mode: vt::Mode::NonInteractive,
                params: vt::ParamSetId(vt::Ascii::new("test-small/1").unwrap()),
                statement_digest: [0x44; 32],
                session_id: sid,
            }),
        )
    };

    let pt2 = [
        F::new(G::from_u64_reduced(5), G::from_u64_reduced(9)),
        F::new(G::from_u64_reduced(7), G::from_u64_reduced(13)),
    ];
    let pt2_tv: Vec<TvFp2> = pt2.into_iter().map(tvf).collect();
    let vocab_vars = pv.trailing_zeros() as usize;
    let valid_col: Vec<G> = (0..4 * pv)
        .map(|idx| G::from_u64_reduced(u64::from(idx % pv < vocab)))
        .collect();
    for (m_idx, (rel, claim_col, evals_count)) in [
        (&contractions.total, &batch.cdf_evaluated().totals, 1usize),
        (&contractions.below, &batch.cdf_evaluated().belows, 2usize),
        (
            &contractions.selected,
            &batch.cdf_evaluated().selecteds,
            2usize,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let claim = mle(claim_col, &pt2_tv);
        let tbls = rel.tables(&pt2).unwrap();
        let (mut pt, mut vtr) = fresh_pair(m_idx as u8);
        let proved = prove_batch(
            &mut pt,
            Protocol::One,
            Prefix::Stage(1),
            3,
            &[tbls.instance()],
        )
        .unwrap();
        let proof = tv::SumcheckProof {
            rounds: proved
                .proof()
                .rounds
                .iter()
                .map(|r| r.iter().copied().map(tvf).collect())
                .collect(),
            final_evals: proved
                .proof()
                .final_evals
                .iter()
                .copied()
                .map(tvf)
                .collect(),
        };
        let _ = tv::verify_batch(
            &mut vtr,
            tv::Prefix::Stage(1),
            3,
            &[tv::Instance {
                claim,
                vars: vocab_vars + 2,
                evals: evals_count,
            }],
            &proof,
            |_, r, e| {
                let row_eq = tv::eq(&pt2_tv, &r[vocab_vars..]).unwrap();
                Some(if m_idx == 0 {
                    row_eq * mle(&valid_col, r) * e[0]
                } else {
                    row_eq * e[0] * e[1]
                })
            },
        )
        .unwrap();
        assert_eq!(pt.state(), *vtr.state());
    }

    let c_base = batch.cdf().constraints::<G>().unwrap();
    let psi_col: Vec<G> = (0..4)
        .map(|row| G::from_u64_reduced(u64::from(row < 3)))
        .collect();
    for (eq_idx, eq_rel) in batch
        .cdf_equality_relations::<F>()
        .unwrap()
        .iter()
        .enumerate()
    {
        let tbls = eq_rel.tables(&pt2).unwrap();
        let (mut pt, mut vtr) = fresh_pair(20 + eq_idx as u8);
        let proved = prove_batch(
            &mut pt,
            Protocol::One,
            Prefix::Stage(2),
            3,
            &[tbls.instance()],
        )
        .unwrap();
        let proof = tv::SumcheckProof {
            rounds: proved
                .proof()
                .rounds
                .iter()
                .map(|r| r.iter().copied().map(tvf).collect())
                .collect(),
            final_evals: proved
                .proof()
                .final_evals
                .iter()
                .copied()
                .map(tvf)
                .collect(),
        };
        let eval_count = proof.final_evals.len();
        let _ = tv::verify_batch(
            &mut vtr,
            tv::Prefix::Stage(2),
            3,
            &[tv::Instance {
                claim: TvFp2::ZERO,
                vars: 2,
                evals: eval_count,
            }],
            &proof,
            |_, r, _| {
                let eval_col = |col: &[G]| -> F { lift(mle(col, r)) };
                let poly = c_base.equalities[eq_idx].evaluate(|term| match term {
                    CdfTerm::Draw => eval_col(batch.draws()),
                    CdfTerm::Total => eval_col(&batch.cdf_evaluated().totals),
                    CdfTerm::Below => eval_col(&batch.cdf_evaluated().belows),
                    CdfTerm::Selected => eval_col(&batch.cdf_evaluated().selecteds),
                    CdfTerm::RowSelector => eval_col(&psi_col),
                    CdfTerm::Column(col) => eval_col(batch.cdf_columns().column(col).unwrap()),
                });
                Some(tv::eq(&pt2_tv, r).unwrap() * tvf(poly))
            },
        )
        .unwrap();
        assert_eq!(pt.state(), *vtr.state());
    }

    // Suffix weights match batched suffix MLE and verify via tv_core::sumcheck.
    let pt_full: Vec<F> = (0..vocab_vars + 2)
        .map(|i| {
            F::new(
                G::from_u64_reduced(2 + i as u64),
                G::from_u64_reduced(19 + i as u64),
            )
        })
        .collect();
    let pt_full_tv: Vec<TvFp2> = pt_full.iter().copied().map(tvf).collect();
    let l_claim = mle(&batch.cdf_evaluated().suffix, &pt_full_tv);
    let weights = batch.suffix_weights(&pt_full).unwrap();
    let tv_weights: Vec<TvFp2> = weights.iter().copied().map(tvf).collect();
    let suffix_inst = Instance::new(vec![
        ScTable::ext(&weights).public(),
        ScTable::base(batch.indicators()),
    ]);
    let (mut s_pt, mut s_vt) = fresh_pair(99);
    let proved_s = prove_batch(
        &mut s_pt,
        Protocol::One,
        Prefix::Stage(3),
        2,
        &[suffix_inst],
    )
    .unwrap();
    let proof_s = tv::SumcheckProof {
        rounds: proved_s
            .proof()
            .rounds
            .iter()
            .map(|r| r.iter().copied().map(tvf).collect())
            .collect(),
        final_evals: proved_s
            .proof()
            .final_evals
            .iter()
            .copied()
            .map(tvf)
            .collect(),
    };
    let checked_s = tv::verify_batch(
        &mut s_vt,
        tv::Prefix::Stage(3),
        2,
        &[tv::Instance {
            claim: l_claim,
            vars: vocab_vars + 2,
            evals: 1,
        }],
        &proof_s,
        |_, r, e| {
            let w_val = tv_weights
                .iter()
                .enumerate()
                .fold(TvFp2::ZERO, |acc, (i, &w)| {
                    let eq = r.iter().enumerate().fold(TvFp2::ONE, |a, (j, &rj)| {
                        a * if (i >> j) & 1 == 0 {
                            TvFp2::ONE - rj
                        } else {
                            rj
                        }
                    });
                    acc + w * eq
                });
            Some(w_val * e[0])
        },
    )
    .unwrap();
    assert_eq!(
        proof_s.final_evals,
        [mle(batch.indicators(), &checked_s.point)]
    );
    assert_eq!(s_pt.state(), *s_vt.state());
}
