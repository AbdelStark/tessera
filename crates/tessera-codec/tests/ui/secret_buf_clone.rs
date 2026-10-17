// SPDX-License-Identifier: Apache-2.0 OR MIT

use tessera_codec::SecretBuf;

fn copy<T: Clone>(value: &T) -> T {
    value.clone()
}

fn main() {
    let rows = SecretBuf::filled(7u64, 4);
    let _copy = copy(&rows);
}
