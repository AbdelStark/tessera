// SPDX-License-Identifier: Apache-2.0 OR MIT

// A string built at run time is not a field; a caller wraps what it clears in `Public`.
fn main() {
    let text = String::from("anything");
    tessera_codec::event!(INFO, "codec.leak", text = text; "{text}");
}
