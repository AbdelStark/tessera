// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU unit-cost microbenchmarks of RFC-0020 'Measurement protocol', with its statistics (3
//! warm-up runs, 10 measured runs, median, extremes, coefficient of variation) and a check of
//! every benchmark's output against the CPU reference before it is timed.
//!
//! ```sh
//! cargo run --release -p tessera-bench --features gpu --example gpu_unit_costs -- cc
//! ```
//!
//! `pi`: the product-tree levels and level sumchecks of a forest of 2^16 trees of 2^14 leaves
//! (2^30 leaves), RFC-0010 'Tree levels', leaves generated on the device; the check runs the
//! CPU prover `tessera_sumcheck::prove_forests` on the same leaves and transcript once.
//!
//! `l`: the LogUp fraction trees of 2^30 lookups into a 2^24-row table, RFC-0008 'Fraction
//! tree': the lookup tree (numerators 1) and the table tree (numerators `−m_j`), leaves
//! generated inside the kernels, each tree's levels proven as a batch of its own; the check runs
//! `tessera_sumcheck::prove_trees` on each tree once.
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
use tessera_codec::Ascii;
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_gpu::{Device, DeviceBuffer, FractionLeaves, GpuError};
use tessera_hash::{ColumnChains, Mode, ParamSetId, SessionContext, Transcript};
use tessera_pcs::RowEncoder;
use tessera_sumcheck::{
    Forest, FractionTree, LeafNumerators, LevelClaim, Tree, prove_forests, prove_trees,
};

const LOG_K: u32 = 22;
const MASK_COEFFS: usize = 200;
const BLOCK: usize = 32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rows = option(&args, "--rows", 4096)?;
    environment();
    let device = Device::new(0)?;
    match args.first().map(String::as_str) {
        Some("cc") => cc(&device, rows),
        Some("pi") => pi(&device, option(&args, "--trees-log", 16)?),
        Some("l") => lookups(
            &device,
            option(&args, "--lookups-log", 30)?,
            option(&args, "--rows-log", 24)?,
        ),
        _ => Err("usage: gpu_unit_costs cc [--rows N] | pi [--trees-log S] | l [--lookups-log A] [--rows-log T]".into()),
    }
}

/// The value after `name`, or `default`.
fn option(args: &[String], name: &str, default: usize) -> Result<usize, std::num::ParseIntError> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map_or(Ok(default), |v| v.parse())
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

const TREE_HEIGHT: usize = 14;

/// `2^trees_log` trees of `2^14` leaves: RFC-0020 fixes `2^16` (2^30 leaves).
fn pi(device: &Device, trees_log: usize) -> Result<(), Box<dyn std::error::Error>> {
    let count = 1usize << (trees_log + TREE_HEIGHT);
    let mut leaves = device.zeroed::<u64>(count)?;
    device.fill_counter(&mut leaves, 1 << 61)?;

    let started = Instant::now();
    let host = device.download(&leaves)?;
    let fp: Option<Vec<Goldilocks>> = host
        .iter()
        .map(|&v| Goldilocks::from_canonical_u64(v))
        .collect();
    let fp = fp.ok_or("non-canonical leaf")?;
    drop(host);
    let mut cpu = transcript();
    let (cpu_proofs, cpu_claims) =
        prove_forests(&mut cpu, &[Forest::<GoldilocksExt2>::new(&fp, TREE_HEIGHT)])?.into_parts();
    drop(fp);
    // One workspace for every run: a prover pools its device memory (RFC-0018).
    let mut workspace = device.forest_workspace(count, TREE_HEIGHT)?;
    let mut gpu = transcript();
    let (gpu_proofs, gpu_claim) = device.prove_forest_in(&mut workspace, &mut gpu, &leaves)?;
    if gpu_proofs != cpu_proofs || cpu_claims.first() != Some(&gpu_claim) {
        return Err("GPU forest proof differs from the CPU prover's".into());
    }
    println!(
        "check: 2^{trees_log} trees of 2^{TREE_HEIGHT} leaves, every level's messages and the leaf claim equal the CPU prover's ({:.1} s)",
        started.elapsed().as_secs_f64()
    );

    let statistics = timed(count as f64, || {
        device.prove_forest_in(&mut workspace, &mut transcript(), &leaves)?;
        device.synchronize()
    })?;
    report("pi/forest", count, &statistics);
    println!("c_pi (median): {:.2} ps per leaf", 1e3 * statistics.median);
    Ok(())
}

/// `2^lookups_log` lookups into `2^rows_log` rows: RFC-0020 fixes 2^30 and 2^24.
fn lookups(
    device: &Device,
    lookups_log: usize,
    rows_log: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let fp2 = |a: u64, b: u64| -> Result<GoldilocksExt2, Box<dyn std::error::Error>> {
        let c = |x| Goldilocks::from_canonical_u64(x).ok_or("non-canonical");
        Ok(GoldilocksExt2::new(c(a)?, c(b)?))
    };
    let (alpha, beta) = (
        fp2(0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321)?,
        fp2(0x1111_2222_3333_4444, 0x5555_6666_7777_8888)?,
    );
    let lookup = FractionLeaves {
        key_seed: 1 << 56,
        value_seed: 1 << 57,
        multiplicity_seed: None,
        alpha,
        beta,
    };
    let table = FractionLeaves {
        key_seed: 1 << 58,
        value_seed: 1 << 59,
        multiplicity_seed: Some(1 << 60),
        alpha,
        beta,
    };
    let mut lookup_space = device.fraction_workspace(lookups_log, true)?;
    let mut table_space = device.fraction_workspace(rows_log, false)?;

    for (name, leaves, height, space) in [
        ("lookup", &lookup, lookups_log, &mut lookup_space),
        ("table", &table, rows_log, &mut table_space),
    ] {
        let started = Instant::now();
        let count = 1u64 << height;
        let numerators: Vec<Goldilocks> = (0..count).filter_map(|j| leaves.numerator(j)).collect();
        let numerators = match leaves.multiplicity_seed {
            None => LeafNumerators::Unit,
            Some(_) => LeafNumerators::Base(&numerators),
        };
        let tree = FractionTree::new(height, numerators, |j| leaves.denominator(j as u64))?;
        let mut cpu = transcript();
        let (cpu_proofs, cpu_claims) =
            prove_trees(&mut cpu, vec![Tree::Fraction(tree)])?.into_parts();
        let mut gpu = transcript();
        let (gpu_proofs, gpu_claim) = device.prove_fraction_tree_in(space, &mut gpu, leaves)?;
        let same = matches!(cpu_claims.first(), Some(LevelClaim::Fraction(c))
            if c.point == gpu_claim.point && c.numerator == gpu_claim.numerator && c.denominator == gpu_claim.denominator);
        if gpu_proofs != cpu_proofs || !same {
            return Err(format!("GPU {name} tree proof differs from the CPU prover's").into());
        }
        println!(
            "check: {name} tree of 2^{height} leaves, every level's messages and the leaf claim equal the CPU prover's ({:.1} s)",
            started.elapsed().as_secs_f64()
        );
    }

    let lookups = (1usize << lookups_log) as f64;
    let lookup_tree = timed(lookups, || {
        device.prove_fraction_tree_in(&mut lookup_space, &mut transcript(), &lookup)?;
        device.synchronize()
    })?;
    let table_tree = timed(lookups, || {
        device.prove_fraction_tree_in(&mut table_space, &mut transcript(), &table)?;
        device.synchronize()
    })?;
    report("l/lookup-tree", 1 << lookups_log, &lookup_tree);
    report("l/table-tree", 1 << lookups_log, &table_tree);
    println!(
        "c_l (medians, per lookup): {:.2} ps; lookup tree {:.2} ps, table tree {:.2} ps",
        1e3 * (lookup_tree.median + table_tree.median),
        1e3 * lookup_tree.median,
        1e3 * table_tree.median
    );
    Ok(())
}

fn transcript() -> Transcript {
    Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap_or_else(|_| unreachable!()),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("bench/1").unwrap_or_else(|_| unreachable!())),
        statement_digest: [0x65; 32],
        session_id: [0x13; 16],
    })
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
