// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The audit sample's, the verifier ledger's and the pending toss's readers, payload and
//! envelope, on arbitrary bytes, against `tv_fuzz::audit`'s oracle: no panic, rejections that
//! are the input's fault and point inside it, and accepted inputs that are their value's one
//! encoding and keep its rules.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    tv_fuzz::audit::check(data);
});
