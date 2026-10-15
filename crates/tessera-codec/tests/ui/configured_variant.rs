// SPDX-License-Identifier: Apache-2.0 OR MIT

use tessera_codec::{Decode, Encode};

#[derive(Encode, Decode)]
enum Configured {
    A,
    #[cfg(all())]
    B,
    C,
}

fn main() {}
