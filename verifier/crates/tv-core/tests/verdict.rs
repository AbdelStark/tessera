// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independently specified stable error identities; no producer dependency (04 section 3).
use tv_core::ErrorCode;
use tv_testkit::read_vectors;
#[test]
fn every_operational_code_matches_the_normative_golden_table() {
    let doc = read_vectors("errors/codes.json");
    let expected = doc.get("codes").unwrap().as_array().unwrap();
    assert_eq!(expected.len(), ErrorCode::ALL.len());
    for (code, row) in ErrorCode::ALL.into_iter().zip(expected) {
        assert_eq!(Some(code.id()), row.get("code").unwrap().as_str());
        assert_eq!(Some(code.class()), row.get("class").unwrap().as_str());
        assert_eq!(
            Some(code.exit_code() as u64),
            row.get("exit").unwrap().as_u64()
        );
    }
}

#[test]
fn every_rejection_serializes_public_payloads_and_escapes_parameter_names() {
    use tv_core::claims::ParamSetId;
    use tv_core::protocol::FixedBits;
    use tv_core::validation::{GadgetId, StatementField};
    use tv_core::verdict::{
        AnchorFailure, ArtifactKind, CommitmentClass as C, PowPhase, RejectReason as R, Verdict,
    };
    let cases = [
        R::Malformed {
            artifact: ArtifactKind(0x0520),
            code: ErrorCode::CodecTruncated,
        },
        R::StatementMismatch {
            field: StatementField::Batch,
        },
        R::Sumcheck { stage: 2, round: 7 },
        R::TreeLevel {
            group: 4,
            level: 3,
            round: 9,
        },
        R::LeafMerge { round: 12 },
        R::LogUpIdentity { round: 11 },
        R::Lookup {
            table: tv_tables::TableId(5),
        },
        R::Gadget {
            gadget: GadgetId(6),
        },
        R::MerklePath {
            commitment: C::Advice,
            column: 13,
        },
        R::Proximity {
            commitment: C::Weight,
            column: 17,
        },
        R::Evaluation {
            commitment: C::Weight,
        },
        R::ProofOfWork {
            phase: PowPhase::Queries,
        },
        R::Anchor {
            request_id: None,
            kind: AnchorFailure::ManifestSignature,
        },
        R::Anchor {
            request_id: Some(u64::MAX),
            kind: AnchorFailure::Frames,
        },
        R::Anchor {
            request_id: Some(3),
            kind: AnchorFailure::SamplerKey,
        },
        R::Anchor {
            request_id: Some(0),
            kind: AnchorFailure::Session,
        },
        R::UBoundExceeded {
            request_id: Some(29),
            uhat: FixedBits(1024),
            bound: FixedBits(768),
        },
        R::Budget { epoch: 31 },
        R::Params {
            name: ParamSetId::new("bad\\\"/1").unwrap(),
        },
    ];
    for reason in cases {
        let code = reason.code();
        let verdict = Verdict::Reject(reason);
        let text = verdict.to_json().unwrap();
        let parsed = tv_testkit::Json::parse(&text).unwrap();
        let v = parsed.get("verdict").unwrap();
        assert_eq!(v.get("status").unwrap().as_str(), Some("reject"));
        assert_eq!(
            v.get("reason").unwrap().get("code").unwrap().as_str(),
            Some(code)
        );
        assert_eq!(verdict.exit_code(), 10);
        for _ in 0..100 {
            assert_eq!(verdict.to_json().unwrap(), text);
        }
    }
    // RFC-0054 'Messages': a tree level names its group, and the merge its round.
    for (reason, fields) in [
        (
            R::TreeLevel {
                group: 4,
                level: 3,
                round: 9,
            },
            &[("group", 4), ("level", 3), ("round", 9)][..],
        ),
        (R::LeafMerge { round: 12 }, &[("round", 12)][..]),
    ] {
        let code = reason.code();
        let text = Verdict::Reject(reason).to_json().unwrap();
        let parsed = tv_testkit::Json::parse(&text).unwrap();
        let reason = parsed.get("verdict").unwrap().get("reason").unwrap();
        for &(name, value) in fields {
            assert_eq!(
                reason.get(name).unwrap().as_u64(),
                Some(value),
                "{code}: {name}"
            );
        }
    }
    assert_eq!(R::LeafMerge { round: 0 }.code(), "REJ-LEAF-MERGE");
    let error = Verdict::Inconclusive(tv_core::VerifierError {
        code: ErrorCode::Memory,
    });
    let text = error.to_json().unwrap();
    let parsed = tv_testkit::Json::parse(&text).unwrap();
    assert_eq!(
        parsed
            .get("verdict")
            .unwrap()
            .get("error")
            .unwrap()
            .get("code")
            .unwrap()
            .as_str(),
        Some("ERR-RES-MEMORY")
    );
    assert_eq!(error.exit_code(), 20);
}

/// RFC-0043 'Anchor', with RFC-0061 and RFC-0062: the kinds are `ManifestSignature` (tag 0),
/// `Frames` (2), `SamplerKey` (3) and `Session` (4); tag 1, `Inclusion`, is retired and stays
/// reserved. The JSON names the kind and the request, `null` for the manifest's signature, as
/// `verdict-v1.schema.json` lists them.
#[test]
fn anchor_kinds_keep_their_tags_and_names() {
    use tv_core::verdict::{AnchorFailure as A, RejectReason as R, Verdict};
    let kinds = [
        (A::ManifestSignature, 0, "ManifestSignature"),
        (A::Frames, 2, "Frames"),
        (A::SamplerKey, 3, "SamplerKey"),
        (A::Session, 4, "Session"),
    ];
    let schema = tv_testkit::Json::parse(
        &std::fs::read_to_string(
            tv_testkit::vectors_dir().join("../schemas/verdict-v1.schema.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let listed = find_kind_enum(&schema).expect("the schema lists the anchor kinds");
    assert_eq!(
        listed,
        kinds
            .iter()
            .map(|(_, _, name)| name.to_string())
            .collect::<Vec<_>>()
    );
    for (kind, tag, name) in kinds {
        assert_eq!(kind.tag(), tag);
        for request_id in [None, Some(7)] {
            let text = Verdict::Reject(R::Anchor { request_id, kind })
                .to_json()
                .unwrap();
            let parsed = tv_testkit::Json::parse(&text).unwrap();
            let reason = parsed.get("verdict").unwrap().get("reason").unwrap();
            assert_eq!(reason.get("code").unwrap().as_str(), Some("REJ-ANCHOR"));
            assert_eq!(reason.get("kind").unwrap().as_str(), Some(name));
            assert_eq!(reason.get("request_id").unwrap().as_u64(), request_id);
            assert_eq!(
                reason.get("request_id").unwrap().is_null(),
                request_id.is_none()
            );
        }
    }
}

/// The `enum` of the schema's `kind` property, wherever it sits.
fn find_kind_enum(value: &tv_testkit::Json) -> Option<Vec<String>> {
    use tv_testkit::Json;
    match value {
        Json::Object(members) => {
            for (key, member) in members {
                if key == "kind"
                    && let Some(names) = member.get("enum").and_then(Json::as_array)
                {
                    return Some(
                        names
                            .iter()
                            .map(|n| n.as_str().unwrap().to_string())
                            .collect(),
                    );
                }
                if let Some(found) = find_kind_enum(member) {
                    return Some(found);
                }
            }
            None
        }
        Json::Array(items) => items.iter().find_map(find_kind_enum),
        _ => None,
    }
}
