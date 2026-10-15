// SPDX-License-Identifier: Apache-2.0 OR MIT

use tessera_codec::{Decode, Encode};

#[derive(Encode, Decode)]
enum Tagged {
    A = 1,
    B = 0,
}

fn main() {}
