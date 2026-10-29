// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public claim metadata under a fixed additional allocation cap (SP-8).
#![no_main]

use cap::Cap;
use std::alloc::System;

#[global_allocator]
static ALLOCATOR: Cap<System> = Cap::new(System, usize::MAX);

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    ALLOCATOR
        .set_limit(ALLOCATOR.allocated() + (64 << 20))
        .unwrap();
    let _ = tv_fuzz::claims::check(data);
    ALLOCATOR.set_limit(usize::MAX).unwrap();
});
