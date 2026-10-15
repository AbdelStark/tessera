// SPDX-License-Identifier: Apache-2.0 OR MIT

use tessera_codec::Secret;

fn main() {
    let key = Secret::new([0u8; 16]);
    let _copy = key.clone();
}
