// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Protocol syntax judged by the independent Python schema oracle (RFC-0036, DEP-2).
use tv_codec::{DecodeError, DecodeLimits};
use tv_core::protocol as p;
use tv_field::{Fp, Fp2};
use tv_hash::{Domain, hash};
use tv_testkit::{Json as J, SplitMix64, hex, read_vectors};

fn member<'a>(j: &'a J, name: &str) -> &'a J {
    j.get(name).unwrap()
}
fn text(j: &J) -> &str {
    j.as_str().unwrap()
}
fn list(j: &J) -> &[J] {
    j.as_array().unwrap()
}
fn uint(j: &J) -> u64 {
    j.as_u64().unwrap_or_else(|| text(j).parse().unwrap())
}
fn n(v: impl ToString) -> J {
    J::String(v.to_string())
}
fn bytes(v: &[u8]) -> J {
    J::String(v.iter().map(|b| format!("{b:02x}")).collect())
}
fn a<T>(v: &[T], f: impl Fn(&T) -> J) -> J {
    J::Array(v.iter().map(f).collect())
}
fn o<const N: usize>(v: [(&str, J); N]) -> J {
    J::Object(v.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn fp(v: &Fp) -> J {
    n(v.to_canonical())
}
fn ext(v: &Fp2) -> J {
    J::Array(vec![fp(&v.c0), fp(&v.c1)])
}
fn decoding(v: &p::Decoding) -> J {
    match v {
        p::Decoding::Greedy => o([("Greedy", J::Null)]),
        p::Decoding::Temperature { t } => o([(
            "Temperature",
            o([("t", o([("num", n(t.num())), ("den", n(t.den()))]))]),
        )]),
    }
}
fn predictor(v: &p::PredictorSpec) -> J {
    match v {
        p::PredictorSpec::Exact { decoding: d } => o([("Exact", o([("decoding", decoding(d))]))]),
        p::PredictorSpec::Mixture {
            decoding: d,
            eps_log2,
        } => o([(
            "Mixture",
            o([("decoding", decoding(d)), ("eps_log2", n(eps_log2))]),
        )]),
    }
}
fn statement(v: &p::PublicStatement) -> J {
    o([
        ("claim_list_digest", bytes(&v.claim_list_digest)),
        ("weights_root", bytes(&v.weights_root)),
        ("weights_epoch", n(v.weights_epoch)),
        ("manifest_digest", bytes(&v.manifest_digest)),
        (
            "batch",
            o([(
                "requests",
                a(&v.batch.requests, |r| {
                    o([
                        ("request_id", n(r.request_id)),
                        ("session_id", bytes(&r.session_id)),
                        ("prompt_len", n(r.prompt_len)),
                        ("output_len", n(r.output_len)),
                    ])
                }),
            )]),
        ),
        ("predictor", predictor(&v.predictor)),
    ])
}
fn round1(v: &p::Round1) -> J {
    o([
        ("protocol_version", n(v.protocol_version)),
        ("session_id", bytes(&v.session_id)),
        ("statement", statement(&v.statement)),
        (
            "advice",
            o([
                ("root", bytes(&v.advice.root)),
                ("layout_digest", bytes(&v.advice.layout_digest)),
            ]),
        ),
        (
            "leaf_groups",
            o([
                ("token_bits", n(v.leaf_groups.token_bits)),
                ("layer_bits", n(v.leaf_groups.layer_bits)),
            ]),
        ),
    ])
}
fn sumcheck(v: &tv_core::sumcheck::SumcheckProof) -> J {
    o([
        ("rounds", a(&v.rounds, |r| a(r, ext))),
        ("final_evals", a(&v.final_evals, ext)),
    ])
}
fn combined(v: &tv_core::opening::CombinedRow) -> J {
    o([("u", a(&v.u, ext)), ("mask", a(&v.mask, ext))])
}
fn round2(v: &p::Round2Response) -> J {
    o([
        (
            "logup",
            o([
                ("roots", a(&v.logup.roots, ext)),
                ("values", a(&v.logup.values, ext)),
            ]),
        ),
        (
            "trees",
            o([
                ("levels", a(&v.trees.levels, sumcheck)),
                (
                    "groups",
                    a(&v.trees.groups, |g| {
                        o([
                            ("roots", a(&g.roots, ext)),
                            ("levels", a(&g.levels, sumcheck)),
                            ("values", a(&g.values, ext)),
                            ("splits", a(&g.splits, ext)),
                        ])
                    }),
                ),
                ("table_roots", a(&v.trees.table_roots, ext)),
                ("merge", sumcheck(&v.trees.merge)),
                ("splits", a(&v.trees.splits, ext)),
            ]),
        ),
        (
            "stages",
            a(&v.stages, |s| {
                o([
                    ("stage", n(s.stage)),
                    ("sumcheck", sumcheck(&s.sumcheck)),
                    ("splits", a(&s.splits, ext)),
                ])
            }),
        ),
        ("advice_combined", a(&v.advice_combined, combined)),
        (
            "weight_combined",
            a(&v.weight_combined, |r| {
                o([
                    ("matrix", n(r.matrix)),
                    ("evaluations", a(&r.evaluations, combined)),
                    ("proximity", combined(&r.proximity)),
                ])
            }),
        ),
        (
            "scoring",
            o([("requests", a(&v.scoring.requests, |b| n(b.0)))]),
        ),
    ])
}
fn queries(v: &p::ColumnQueries) -> J {
    o([
        ("advice", a(&v.advice, |v| n(*v))),
        (
            "weights",
            a(&v.weights, |r| {
                o([
                    ("matrix", n(r.matrix)),
                    ("indices", a(&r.indices, |v| n(*v))),
                ])
            }),
        ),
    ])
}
fn proof(v: &p::MultiProof) -> J {
    o([("siblings", a(&v.siblings, |x| bytes(x)))])
}
fn round3(v: &p::Round3Response) -> J {
    o([
        ("advice_columns", a(&v.advice_columns, |r| a(r, fp))),
        ("advice_proof", proof(&v.advice_proof)),
        (
            "weight_columns",
            a(&v.weight_columns, |r| {
                o([
                    ("matrix", n(r.matrix)),
                    ("columns", a(&r.columns, |r| a(r, fp))),
                    ("proof", proof(&r.proof)),
                ])
            }),
        ),
    ])
}
fn ni(v: &p::NiProof) -> J {
    o([
        ("round1", round1(&v.round1)),
        ("seed_nonce", n(v.seed_nonce)),
        ("round2", round2(&v.round2)),
        ("query_nonce", n(v.query_nonce)),
        ("round3", round3(&v.round3)),
    ])
}
#[derive(Debug)]
struct Decoded {
    value: J,
    encoded: Vec<u8>,
    shown: String,
}
fn decode(
    kind: &str,
    input: &[u8],
    limits: &DecodeLimits,
    envelope: bool,
) -> Result<Decoded, DecodeError> {
    macro_rules! read {
        ($ty:ident, $view:expr) => {{
            let value = if envelope {
                p::$ty::open(input, limits)
            } else {
                tv_codec::decode(input, limits, p::$ty::decode)
            }?;
            Ok(Decoded {
                value: ($view)(&value),
                encoded: value.to_tce().unwrap(),
                shown: format!("{value:?}"),
            })
        }};
    }
    match kind {
        "SeedCommit" => read!(SeedCommit, |v: &p::SeedCommit| o([
            ("session_id", bytes(&v.session_id)),
            ("commitment", bytes(&v.commitment))
        ])),
        "ProverSeedShare" => read!(ProverSeedShare, |v: &p::ProverSeedShare| o([
            ("session_id", bytes(&v.session_id)),
            ("share", bytes(&v.share))
        ])),
        "SeedReveal" => read!(SeedReveal, |v: &p::SeedReveal| o([
            ("session_id", bytes(&v.session_id)),
            ("share", bytes(&v.share))
        ])),
        "Round1" => read!(Round1, round1),
        "Round2Response" => read!(Round2Response, round2),
        "ColumnQueries" => read!(ColumnQueries, queries),
        "Round3Response" => read!(Round3Response, round3),
        "NiProof" => read!(NiProof, ni),
        _ => panic!("unregistered protocol vector type {kind}"),
    }
}
fn limits(j: &J) -> DecodeLimits {
    DecodeLimits {
        max_total_bytes: uint(member(j, "max_total_bytes")),
        max_vec_len: uint(member(j, "max_vec_len")),
        max_depth: uint(member(j, "max_depth")).try_into().unwrap(),
    }
}
fn case_limits(case: &J, default: DecodeLimits) -> DecodeLimits {
    case.get("limits").map(limits).unwrap_or(default)
}

#[test]
fn every_protocol_value_and_envelope_matches_the_independent_schema() {
    let vectors = read_vectors("protocol/protocol.json");
    let limits = limits(member(&vectors, "limits"));
    for case in list(member(&vectors, "messages")) {
        let kind = text(member(case, "type"));
        let payload = hex(text(member(case, "payload")));
        let decoded = decode(kind, &payload, &limits, false).unwrap();
        assert_eq!(&decoded.value, member(case, "value"), "{kind}");
        assert_eq!(decoded.encoded, payload, "{kind}");
        let opened = decode(kind, &hex(text(member(case, "envelope"))), &limits, true).unwrap();
        assert_eq!(opened.value, decoded.value);
        assert_eq!(opened.encoded, payload);
        if kind == "Round1" {
            let value = tv_codec::decode(&payload, &limits, p::Round1::decode).unwrap();
            assert_eq!(
                value.statement.digest().as_slice(),
                hex(text(member(case, "statement_digest")))
            );
            assert_eq!(
                value.statement.digest(),
                hash(Domain::Statement, &value.statement.to_tce().unwrap())
            );
        }
    }
}

#[test]
fn every_proper_prefix_reports_the_independent_first_missing_offset() {
    let vectors = read_vectors("protocol/protocol.json");
    let limits = limits(member(&vectors, "limits"));
    for case in list(member(&vectors, "messages")) {
        let kind = text(member(case, "type"));
        let payload = hex(text(member(case, "payload")));
        let runs = list(member(case, "truncated"));
        let mut run = 0;
        for len in 0..payload.len() {
            while run + 1 < runs.len() && uint(&list(&runs[run + 1])[0]) <= len as u64 {
                run += 1;
            }
            let error = decode(kind, &payload[..len], &limits, false).unwrap_err();
            assert_eq!(
                error.code().id(),
                "ERR-CODEC-TRUNCATED",
                "{kind} prefix {len}"
            );
            assert_eq!(
                error.offset(),
                uint(&list(&runs[run])[1]),
                "{kind} prefix {len}"
            );
        }
    }
}

#[test]
fn malformed_protocol_values_and_boundary_valid_inputs_keep_exact_codes_and_offsets() {
    let vectors = read_vectors("protocol/protocol.json");
    let default = limits(member(&vectors, "limits"));
    for section in ["refused", "envelopes"] {
        for case in list(member(&vectors, section)) {
            let kind = text(member(
                case,
                if section == "envelopes" {
                    "schema"
                } else {
                    "type"
                },
            ));
            let input = hex(text(member(case, "hex")));
            let error = decode(
                kind,
                &input,
                &case_limits(case, default),
                section == "envelopes",
            )
            .unwrap_err();
            assert_eq!(
                error.code().id(),
                text(member(case, "code")),
                "{section} {kind}: {}",
                text(member(case, "why"))
            );
            assert_eq!(
                error.offset(),
                uint(member(case, "offset")),
                "{section} {kind}: {}",
                text(member(case, "why"))
            );
        }
    }
    for case in list(member(&vectors, "accepted")) {
        let input = hex(text(member(case, "hex")));
        let d = decode(
            text(member(case, "type")),
            &input,
            &case_limits(case, default),
            false,
        )
        .unwrap();
        assert_eq!(d.encoded, input);
    }
}

#[test]
fn statement_policy_and_query_ranges_are_not_codec_predicates_and_debug_hides_messages() {
    let vectors = read_vectors("protocol/protocol.json");
    let limits = limits(member(&vectors, "limits"));
    for case in list(member(&vectors, "messages")) {
        let kind = text(member(case, "type"));
        let payload = hex(text(member(case, "payload")));
        let d = decode(kind, &payload, &limits, false).unwrap();
        if matches!(kind, "ProverSeedShare" | "SeedReveal" | "NiProof") {
            assert_eq!(d.shown, format!("{kind} {{ .. }}"));
        }
        if kind == "Round2Response" {
            let mut m = tv_codec::decode(&payload, &limits, p::Round2Response::decode).unwrap();
            let sentinel = Fp2::new(Fp::reduce(1_234_567_890_123), Fp::reduce(7_654_321_098_765));
            m.logup.roots = vec![sentinel];
            m.trees.splits = vec![sentinel];
            assert!(!format!("{m:?}").contains("1234567890123"));
            assert!(!format!("{:?}", m.logup).contains("1234567890123"));
            assert!(!format!("{:?}", m.trees).contains("1234567890123"));
        }
        if kind == "Round1" {
            let mut m = tv_codec::decode(&payload, &limits, p::Round1::decode).unwrap();
            m.protocol_version = u16::MAX;
            m.statement.predictor = p::PredictorSpec::Mixture {
                decoding: p::Decoding::Temperature {
                    t: tv_codec::Rational::new(-1, 2).unwrap(),
                },
                eps_log2: u8::MAX,
            };
            let bytes = m.to_tce().unwrap();
            assert_eq!(
                tv_codec::decode(&bytes, &limits, p::Round1::decode).unwrap(),
                m
            );
        }
    }
}

#[test]
fn hostile_bytes_never_panic_and_every_accepted_value_reencodes_from_its_fields() {
    let vectors = read_vectors("protocol/protocol.json");
    let limits = DecodeLimits {
        max_total_bytes: 4096,
        max_vec_len: 256,
        max_depth: 32,
    };
    let kinds = [
        "SeedCommit",
        "ProverSeedShare",
        "SeedReveal",
        "Round1",
        "Round2Response",
        "ColumnQueries",
        "Round3Response",
        "NiProof",
    ];
    let mut rng = SplitMix64::new(0x128_0036_0043);
    for case in list(member(&vectors, "messages")) {
        let kind = text(member(case, "type"));
        let original = hex(text(member(case, "payload")));
        for _ in 0..200 {
            let mut bytes = original.clone();
            let index = rng.next_u64() as usize % bytes.len();
            bytes[index] ^= (rng.next_u64() as u8) | 1;
            let d = std::panic::catch_unwind(|| decode(kind, &bytes, &limits, false))
                .expect("decoder panicked");
            if let Ok(d) = d {
                assert_eq!(d.encoded, bytes);
            }
        }
    }
    for kind in kinds {
        for _ in 0..1000 {
            let len = (rng.next_u64() % 513) as usize;
            let bytes = (0..len).map(|_| rng.next_u64() as u8).collect::<Vec<_>>();
            let d = std::panic::catch_unwind(|| decode(kind, &bytes, &limits, false))
                .expect("decoder panicked");
            if let Ok(d) = d {
                assert_eq!(d.encoded, bytes);
            }
        }
    }
}

#[test]
fn coin_toss_vectors_use_the_public_domains_and_verifier_share_first() {
    let vectors = read_vectors("protocol/protocol.json");
    for case in list(member(&vectors, "coin_toss")) {
        let sv = hex(text(member(case, "verifier_share")));
        let sp = hex(text(member(case, "prover_share")));
        assert_eq!(
            hash(Domain::CoinCommit, &sv).as_slice(),
            hex(text(member(case, "commitment")))
        );
        let mut input = sv;
        input.extend(sp);
        assert_eq!(
            hash(Domain::CoinToss, &input).as_slice(),
            hex(text(member(case, "seed")))
        );
    }
}
