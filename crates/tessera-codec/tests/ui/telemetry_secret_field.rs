// SPDX-License-Identifier: Apache-2.0 OR MIT

// A secret-class value is not a telemetry field (05 §3 OBS-1).
fn main() {
    let key = tessera_codec::Secret::new(7u64);
    tessera_codec::event!(INFO, "codec.leak", key = key; "{key}");
}
