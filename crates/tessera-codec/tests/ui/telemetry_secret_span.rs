// SPDX-License-Identifier: Apache-2.0 OR MIT

// Span fields are public data too (05 §5).
fn main() {
    let tokens = tessera_codec::SecretBuf::<u32>::new();
    let _span = tessera_codec::span!(DEBUG, "prove", batch = tokens);
}
