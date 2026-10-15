// SPDX-License-Identifier: Apache-2.0 OR MIT

use tessera_codec::Secret;

fn main() {
    let weights = Secret::new(vec![1i8, -2, 3]);
    let _ = tessera_codec::encode(&weights);
}
