// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Code that must not compile, each case pinned with its error in `tests/ui/`.
//!
//! - `#[derive(Encode, Decode)]` refuses layouts whose encoding would mislead or depend on the
//!   build. An explicit discriminant suggests a tag other than the declaration index; a union
//!   has no single layout; a variant under `#[cfg]` would move every later tag in builds
//!   without it.
//! - A `Secret` has no `Debug`, `Display`, `Clone` or `Encode`, so a secret-class value cannot be
//!   printed, copied or encoded by accident (05 §3), and neither has a `SecretBuf` of values
//!   derived from one (OBS-6).

#[test]
fn misuses_do_not_compile() {
    trybuild::TestCases::new().compile_fail("tests/ui/*.rs");
}
