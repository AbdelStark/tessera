// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `verify_inclusion` on RFC 6962 inclusion proofs in logs of 1 to 260 entries, changed by a
//! script the input spells out (`tv_fuzz::proofs::inclusion_case`): proof digests removed,
//! inserted, duplicated, swapped or flipped; another leaf, root, index or tree size, `u64`
//! extremes included.
//!
//! The result must be exactly the oracle's (`tv_fuzz::proofs::inclusion_outcome`), written from
//! RFC 6962's recursive definition of `PATH` rather than RFC 9162's iterative check: a proof of the
//! right length is accepted when it leads from the leaf to the root, and every other proof is
//! rejected by the step of RFC 9162 §2.1.3.2 that catches it first.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tv_fuzz::proofs::{inclusion_case, inclusion_outcome};
use tv_hash::rfc6962::verify_inclusion;

fuzz_target!(|data: &[u8]| {
    let case = inclusion_case(data);
    let outcome = verify_inclusion(case.index, case.size, &case.leaf, &case.proof, &case.root);
    assert_eq!(
        outcome,
        inclusion_outcome(&case),
        "leaf {} of {}, {} digests",
        case.index,
        case.size,
        case.proof.len()
    );
});
