// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `tessera` command-line interface (docs/spec/02-public-api.md).

#![forbid(unsafe_code)]

/// Demonstration only.
pub fn demo(v: &[u8]) -> bool {
    v.is_empty()
}

#[cfg(test)]
mod tests {
    #[test]
    fn demonstration_failure() {
        assert!(!super::demo(&[]), "deliberately failing test");
    }
}
