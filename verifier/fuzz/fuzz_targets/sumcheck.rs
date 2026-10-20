// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GDG-prefixed inputs additionally exercise all final gadget families and canonical
//! output-row views against independent arithmetic/fixture oracles (`tv_fuzz::gadgets`).
//! Other inputs exercise `tv-core`'s sumcheck checks on honest proofs of seeded instances, made by the test kit's
//! reference prover, then changed by a script the input spells out (`tv_fuzz::sumcheck::case`):
//! batched sumchecks of products of up to four front-padded instances, of degree 1 to 7, and
//! batches of up to four product trees of heights 1 to 5, some with a zero leaf, whose changes
//! touch any level. Values are changed, replaced, inserted, removed or exchanged; rounds are
//! inserted, removed, duplicated, exchanged or cut short; evaluations likewise.
//!
//! The oracle (`tv_fuzz::sumcheck`), written from the specification without `tv-core`'s code:
//! an unchanged proof verifies at the prover's challenges and with true leaf claims; a changed
//! one never verifies with true claims, its first structural defect is the exact error, a
//! changed round fails the final check, and a stage whose evaluations alone changed passes
//! exactly when the final check, computed by the oracle, holds. A panic, an accepted change
//! with true claims, or another error fails the target.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    tv_fuzz::sumcheck::check(data);
});
