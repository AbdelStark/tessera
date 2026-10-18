// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The crate's error crosses into `TesseraError` (04 §4) with the code 04 §3 gives each kind,
//! the schema of the message refused in its public context, and nothing else: no source, no
//! other context. The expected codes are written here from 04 §3, not taken from the crate.

use tessera_codec::error::{ErrorCode, PublicContext, SchemaId, TesseraError};
use tessera_protocol::{MessageKind, ProtocolError, QueryFault};

/// One error of every kind, with its code from 04 §3 and the schema its context names.
fn cases() -> Vec<(ProtocolError, ErrorCode, Option<u16>)> {
    let mut cases = Vec::new();
    for message in MessageKind::ALL {
        let schema = Some(message.schema().id.0);
        cases.push((
            ProtocolError::OutOfOrder { message },
            ErrorCode::ProtoOrder,
            schema,
        ));
        cases.push((
            ProtocolError::OtherSession { message },
            ErrorCode::ProtoOrder,
            schema,
        ));
    }
    let queries = Some(0x0512);
    cases.extend([
        (
            ProtocolError::Version { found: 1 },
            ErrorCode::ProtoVersion,
            Some(0x0510),
        ),
        (
            ProtocolError::CoinToss,
            ErrorCode::ProtoCointoss,
            Some(0x0503),
        ),
        (ProtocolError::Reuse, ErrorCode::ProtoReuse, Some(0x0503)),
        (
            ProtocolError::Queries(QueryFault::Matrices),
            ErrorCode::ProtoQueries,
            queries,
        ),
        (
            ProtocolError::Queries(QueryFault::Count {
                matrix: Some(8),
                expected: 97,
                found: 96,
            }),
            ErrorCode::ProtoQueries,
            queries,
        ),
        (
            ProtocolError::Queries(QueryFault::Range {
                matrix: None,
                position: 4,
                len: 1 << 23,
            }),
            ErrorCode::ProtoQueries,
            queries,
        ),
        (ProtocolError::Randomness, ErrorCode::ResIo, None),
        (
            ProtocolError::Internal { rule: "a rule" },
            ErrorCode::Internal,
            None,
        ),
        // A parameter set or sizes the caller chose: configuration (04 §3).
        (ProtocolError::UnknownParams, ErrorCode::Usage, None),
        (ProtocolError::TestParams, ErrorCode::Usage, None),
        (
            ProtocolError::Sizes { rule: "a rule" },
            ErrorCode::Usage,
            None,
        ),
    ]);
    cases
}

/// Fails to compile when a kind is added without a case above.
fn covered(error: ProtocolError) {
    match error {
        ProtocolError::OutOfOrder { .. }
        | ProtocolError::OtherSession { .. }
        | ProtocolError::Version { .. }
        | ProtocolError::CoinToss
        | ProtocolError::Reuse
        | ProtocolError::Queries(_)
        | ProtocolError::Randomness
        | ProtocolError::Internal { .. }
        | ProtocolError::UnknownParams
        | ProtocolError::TestParams
        | ProtocolError::Sizes { .. } => {}
    }
}

#[test]
fn every_error_crosses_with_its_code_and_its_schema() {
    for (error, code, schema) in cases() {
        covered(error);
        assert_eq!(error.code(), code, "{error:?}");
        let crossed = TesseraError::from(error);
        assert_eq!(crossed.code, code, "{error:?}");
        assert_eq!(
            crossed.context,
            PublicContext {
                schema: schema.map(SchemaId),
                ..PublicContext::default()
            },
            "{error:?}"
        );
        assert!(crossed.source.is_none(), "{error:?}");
        // 04 §3's exit codes: a second seed for one Round1 is an internal violation, 70, and a
        // configuration a usage error, 64.
        let exit = match code {
            ErrorCode::ProtoReuse | ErrorCode::Internal => 70,
            ErrorCode::Usage => 64,
            _ => 20,
        };
        assert_eq!(crossed.exit_code(), exit, "{error:?}");
    }
}

/// A refused query is named by its position and its codeword, never its index (05 §3).
#[test]
fn a_query_past_its_codeword_is_named_by_its_position() {
    let error = ProtocolError::Queries(QueryFault::Range {
        matrix: Some(1),
        position: 3,
        len: 4096,
    });
    assert_eq!(
        error.to_string(),
        "column queries refused: query 3 for matrix Q is past its codeword of 4096 symbols"
    );
}
