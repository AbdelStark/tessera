// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Layouts `#[derive(Encode, Decode)]` refuses at compile time, each because its encoding would
//! mislead or depend on the build: an explicit discriminant suggests a tag other than the
//! declaration index, a union has no single layout, and a variant under `#[cfg]` would move every
//! later tag in builds without it.

#[test]
fn layouts_tce_cannot_express_do_not_compile() {
    trybuild::TestCases::new().compile_fail("tests/ui/*.rs");
}
