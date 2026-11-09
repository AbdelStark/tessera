// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU unit-cost microbenchmarks of RFC-0020 'Measurement protocol', with its statistics (3
//! warm-up runs, 10 measured runs, median, extremes, coefficient of variation) and a check of
//! every benchmark's output against the CPU reference before it is timed.
//!
//! ```sh
//! cargo run --release -p tessera-bench --features gpu --example gpu_unit_costs -- cc
//! ```
//!
//! `cc`: rate-1/2 coset Reed–Solomon encoding with a vanishing mask (`P = 200`, `ni-100`'s
//! `τ_A`) and BLAKE3 chained column hashing (`B = 32`) of rows of width `k = 2^22`, 4,096 rows
//! (2^34 committed elements) per run, inputs generated on the device by a counter-based
//! generator. A committed element is one row value; its codeword has two symbols.

#![allow(clippy::print_stdout)]

use std::num::NonZeroUsize;
use std::process::Command;
use std::time::Instant;

use tessera_bench::{Protocol, Statistics, measure};
use tessera_field::{Goldilocks, PrimeField};
use tessera_gpu::{Device, DeviceBuffer, GpuError};
use tessera_hash::ColumnChains;
use tessera_pcs::RowEncoder;

const LOG_K: u32 = 22;
const MASK_COEFFS: usize = 200;
const BLOCK: usize = 32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rows: usize = args
        .iter()
        .position(|a| a == "--rows")
        .and_then(|i| args.get(i + 1))
        .map_or(Ok(4096), |v| v.parse())?;
    environment();
    let device = Device::new(0)?;
    if args.first().map(String::as_str) != Some("cc") {
        return Err("usage: gpu_unit_costs cc [--rows N]".into());
    }
    cc(&device, rows)
}

/// The run manifest's environment lines: GPU, driver, clocks, CUDA.
fn environment() {
    let query = "name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,persistence_mode,ecc.mode.current,mig.mode.current,power.limit";
    if let Ok(out) = Command::new("nvidia-smi")
        .args([&format!("--query-gpu={query}"), "--format=csv"])
        .output()
    {
        print!("{}", String::from_utf8_lossy(&out.stdout));
    }
    if let Ok(out) = Command::new("nvcc").arg("--version").output() {
        let text = String::from_utf8_lossy(&out.stdout);
        if let Some(line) = text.lines().find(|l| l.contains("release")) {
            println!("nvcc: {line}");
        }
    }
}

struct Buffers {
    rows: DeviceBuffer<u64>,
    masks: DeviceBuffer<u64>,
    codewords: DeviceBuffer<u64>,
}

fn cc(device: &Device, rows: usize) -> Result<(), Box<dyn std::error::Error>> {
    if rows == 0 || !rows.is_multiple_of(BLOCK) {
        return Err("--rows must be a positive multiple of 32".into());
    }
    let encoder = device.encoder(LOG_K, MASK_COEFFS)?;
    let (k, n) = (encoder.message_len(), encoder.codeword_len());
    let mut buffers = Buffers {
        rows: device.zeroed(BLOCK * k)?,
        masks: device.zeroed(BLOCK * MASK_COEFFS)?,
        codewords: device.zeroed(BLOCK * n)?,
    };
    let id = [0x5Au8; 32];

    // The check: one block generated on the device, encoded and hashed there, against the CPU.
    let started = Instant::now();
    generate(device, &mut buffers, 0, k)?;
    let host_rows = device.download(&buffers.rows)?;
    let host_masks = device.download(&buffers.masks)?;
    let mut chains = device.column_chains(&id, n)?;
    device.encode_rows(
        &encoder,
        &mut buffers.rows,
        &buffers.masks,
        &mut buffers.codewords,
    )?;
    device.absorb_block(&mut chains, &buffers.codewords, BLOCK)?;
    let gpu = device.download_chains(&chains)?;
    let cpu = cpu_block(&host_rows, &host_masks, &id, n)?;
    if gpu != cpu {
        return Err("GPU column chains differ from the CPU reference".into());
    }
    println!(
        "check: {BLOCK} rows of 2^{LOG_K}, {n} column chains equal the CPU's ({:.1} s)",
        started.elapsed().as_secs_f64()
    );

    let elements = (rows * k) as f64;
    let full = timed(elements, || {
        let mut chains = device.column_chains(&id, n)?;
        for block in 0..rows / BLOCK {
            generate(device, &mut buffers, block, k)?;
            device.encode_rows(
                &encoder,
                &mut buffers.rows,
                &buffers.masks,
                &mut buffers.codewords,
            )?;
            device.absorb_block(&mut chains, &buffers.codewords, BLOCK)?;
        }
        device.synchronize()
    })?;
    let generation = timed(elements, || {
        for block in 0..rows / BLOCK {
            generate(device, &mut buffers, block, k)?;
        }
        device.synchronize()
    })?;
    let encode = timed(elements, || {
        for block in 0..rows / BLOCK {
            generate(device, &mut buffers, block, k)?;
            device.encode_rows(
                &encoder,
                &mut buffers.rows,
                &buffers.masks,
                &mut buffers.codewords,
            )?;
        }
        device.synchronize()
    })?;
    report("cc/encode+hash+generate", rows * k, &full);
    report("cc/generate", rows * k, &generation);
    report("cc/encode+generate", rows * k, &encode);
    println!(
        "c_c (median, generation subtracted): {:.2} ps per committed element; encoding {:.2} ps, hashing {:.2} ps",
        1e3 * (full.median - generation.median),
        1e3 * (encode.median - generation.median),
        1e3 * (full.median - encode.median),
    );
    Ok(())
}

/// Block `block`'s rows and masks from the counter-based generator.
fn generate(device: &Device, b: &mut Buffers, block: usize, k: usize) -> Result<(), GpuError> {
    device.fill_counter(&mut b.rows, (block * BLOCK * k) as u64)?;
    device.fill_counter(
        &mut b.masks,
        (1u64 << 62) + (block * BLOCK * MASK_COEFFS) as u64,
    )
}

fn cpu_block(
    rows: &[u64],
    masks: &[u64],
    id: &[u8; 32],
    n: usize,
) -> Result<Vec<[u8; 32]>, Box<dyn std::error::Error>> {
    let field = |v: &[u64]| -> Option<Vec<Goldilocks>> {
        v.iter()
            .map(|&x| Goldilocks::from_canonical_u64(x))
            .collect()
    };
    let rows = field(rows).ok_or("non-canonical row value")?;
    let mut draws = field(masks).ok_or("non-canonical mask")?.into_iter();
    let encoder = RowEncoder::new(1 << LOG_K, 1, MASK_COEFFS)?;
    let zero = Goldilocks::from_canonical_u64(0).ok_or("zero")?;
    let mut codewords = vec![zero; BLOCK * n];
    encoder.encode_rows(&rows, &mut || draws.next().unwrap_or(zero), &mut codewords)?;
    let mut chains = ColumnChains::new(
        id,
        NonZeroUsize::new(n).ok_or("n")?,
        NonZeroUsize::new(BLOCK).ok_or("B")?,
    );
    chains.absorb_block(&codewords)?;
    Ok(chains.finish())
}

/// [`measure`] under RFC-0020's protocol, failing on the first device error of any run.
fn timed(
    units: f64,
    mut run: impl FnMut() -> Result<(), GpuError>,
) -> Result<Statistics, Box<dyn std::error::Error>> {
    let mut failure = None;
    let statistics = measure(Protocol::RFC_0020, units, || {
        if let Err(e) = run() {
            failure.get_or_insert(e);
        }
    })?;
    match failure {
        Some(e) => Err(e.into()),
        None => Ok(statistics),
    }
}

fn report(name: &str, units: usize, s: &Statistics) {
    println!(
        "{name}: units {units}, median {:.2} ps, min {:.2} ps, max {:.2} ps, cv {:.2}%{}",
        1e3 * s.median,
        1e3 * s.min,
        1e3 * s.max,
        100.0 * s.cv,
        if s.noisy() {
            " (noisy: above 3%, repeat)"
        } else {
            ""
        }
    );
}
