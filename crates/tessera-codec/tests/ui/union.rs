// SPDX-License-Identifier: Apache-2.0 OR MIT

use tessera_codec::{Decode, Encode};

#[derive(Encode, Decode)]
union Overlap {
    a: u32,
    b: f32,
}

fn main() {}
