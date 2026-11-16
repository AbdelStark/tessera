// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Times one P1 advice block on the device, step by step (#737): upload of 32 rows of 2^22
//! values, upload of their masks, encoding and column hashing.

#![allow(clippy::print_stdout)]

use std::time::Instant;
use tessera_field::{Goldilocks, PrimeField};
use tessera_gpu::Device;

fn main() -> Result<(), tessera_gpu::GpuError> {
    let d = Device::new(0)?;
    let (log_k, p, rows) = (22u32, 200usize, 32usize);
    let k = 1 << log_k;
    let enc = d.encoder(log_k, p)?;
    let host: Vec<Goldilocks> = (0..rows * k)
        .map(|i| Goldilocks::from_u64_reduced(i as u64))
        .collect();
    let masks: Vec<Goldilocks> = (0..rows * p)
        .map(|i| Goldilocks::from_u64_reduced(i as u64))
        .collect();
    let mut r = d.zeroed::<u64>(rows * k)?;
    let mut m = d.zeroed::<u64>(rows * p)?;
    let mut c = d.zeroed::<u64>(rows * 2 * k)?;
    let mut chains = d.column_chains(&[7; 32], 2 * k)?;
    for round in 0..3 {
        let t = Instant::now();
        d.upload_field_into(&host, &mut r)?;
        d.synchronize()?;
        let up = t.elapsed();
        let t = Instant::now();
        d.upload_field_into(&masks, &mut m)?;
        d.encode_rows(&enc, &mut r, &m, &mut c)?;
        d.synchronize()?;
        let encode = t.elapsed();
        let t = Instant::now();
        d.absorb_block(&mut chains, &c, rows)?;
        d.synchronize()?;
        let absorb = t.elapsed();
        println!(
            "round {round}: upload {up:?} ({:.1} GB/s), encode {encode:?}, absorb {absorb:?}",
            (rows * k * 8) as f64 / up.as_secs_f64() / 1e9
        );
    }
    Ok(())
}
