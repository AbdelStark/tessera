// SPDX-License-Identifier: Apache-2.0 OR MIT

use tessera_codec::SecretBuf;

fn main() {
    let rows = SecretBuf::filled(7u64, 4);
    let _ = format!("{:?}", rows);
}
