// SPDX-License-Identifier: Apache-2.0 OR MIT

// A message is a literal template; formatting arguments would bypass the field types.
fn main() {
    let secret_bytes = [1u8, 2, 3];
    tessera_codec::event!(INFO, "codec.leak"; "{:?}", secret_bytes);
}
