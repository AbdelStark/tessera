// SPDX-License-Identifier: Apache-2.0 OR MIT

use tessera_codec::Secret;

fn main() {
    let seed = Secret::new([0u8; 16]);
    let _ = format!("{:?}", seed);
}
