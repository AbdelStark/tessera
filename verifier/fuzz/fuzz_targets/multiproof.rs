// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `verify_multiproof` on canonical multiproofs of trees of 1 to 1024 leaves, changed by a script
//! the input spells out (`tv_fuzz::proofs::multiproof_case`): siblings removed, inserted,
//! duplicated, swapped or flipped; leaves with other indices or digests, repeated, removed or
//! swapped; another tree size or root; internal nodes opened as leaves.
//!
//! The result must be exactly the oracle's (`tv_fuzz::proofs::multiproof_outcome`), written from
//! RFC-0004 without `tv-hash`'s verification code: accepted only for the canonical multiproof of
//! the opened nodes, and otherwise rejected by the first check that fails, in the order `tv-hash`
//! documents. A panic, an accepted change or a rejection by the wrong check fails the target.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tv_fuzz::proofs::{multiproof_case, multiproof_outcome};
use tv_hash::merkle::verify_multiproof;

fuzz_target!(|data: &[u8]| {
    let case = multiproof_case(data);
    let outcome = verify_multiproof(&case.root, case.n, &case.leaves, &case.siblings);
    assert_eq!(
        outcome,
        multiproof_outcome(&case),
        "n = {}, leaves {:?}, {} siblings",
        case.n,
        case.leaves.iter().map(|leaf| leaf.0).collect::<Vec<_>>(),
        case.siblings.len()
    );
});
