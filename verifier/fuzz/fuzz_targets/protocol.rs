// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Every protocol/1 packet's payload and envelope decoder (RFC-0036, SP-8).
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    tv_fuzz::protocol::check(data);
});
