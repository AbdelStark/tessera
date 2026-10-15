// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera-codec` digests its envelopes itself, since it sits below this crate in the DAG
//! (01 §3). Its digest must be `H(envelope, header ‖ payload)` of RFC-0004, whose context
//! string `vectors.rs` checks against the independent oracle.

use tessera_codec::envelope::{DIGEST_CONTEXT, DIGEST_LEN, seal};
use tessera_codec::schema;
use tessera_hash::{Domain, hash};

#[test]
fn the_envelope_digest_is_the_envelope_domain() {
    assert_eq!(DIGEST_CONTEXT, Domain::Envelope.context());
    let values: [Vec<u64>; 3] = [vec![], vec![7], (0..20_000).collect()];
    for value in values {
        let sealed = seal(&schema::CLAIM_LIST, &value);
        let (body, digest) = sealed.split_at(sealed.len() - DIGEST_LEN);
        assert_eq!(
            digest,
            hash(Domain::Envelope, body),
            "{} elements",
            value.len()
        );
    }
}
