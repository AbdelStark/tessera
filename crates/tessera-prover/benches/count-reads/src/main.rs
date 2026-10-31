// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Count-read batching on a real integer-model source, independent of proof generation.
#![forbid(unsafe_code)]

use criterion::{BenchmarkId, Criterion, Throughput};
use std::{
    error::Error,
    fs::File,
    hint::black_box,
    io::Read,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tessera_codec::{Ascii, Decode, DecodeLimits, Secret, SecretBuf, envelope, schema};
use tessera_compiler::{ExecutionPlan, PublicModel, RuntimePlan, SequenceTile, Tile};
use tessera_ir::{GadgetPolicy, IntegerModel, PROTOCOL_VERSION, ParamSetId};
use tessera_witness::{
    CheckpointTensors, Checkpoints, Engine, Resident, TileWitness, WitnessSource,
};

fn read<T: Decode>(path: &Path, id: &schema::Schema) -> Result<T, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let size = usize::try_from(file.metadata()?.len())?;
    if size > 3usize << 30 {
        return Err("benchmark input exceeds 3 GiB".into());
    }
    let mut bytes = SecretBuf::try_with_capacity(size)?;
    bytes.resize(size, 0u8);
    file.read_exact(&mut bytes)?;
    Ok(envelope::open(&bytes, id, DecodeLimits::INTEGER_MODEL)?)
}

fn prepare(root: &Path, prefix: u64) -> Result<(TileWitness, u64), Box<dyn Error>> {
    let started = Instant::now();
    let path = std::env::var_os("COUNT_BENCH_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("spec/vectors/models/tiny-2-servable.d"));
    let model: IntegerModel = read(&path, &schema::INTEGER_MODEL)?;
    tessera_compiler::validate(&model)?;
    let n: u32 = std::env::var("COUNT_BENCH_POSITIONS")
        .unwrap_or_else(|_| "2".into())
        .parse()?;
    let saved: Option<ExecutionPlan> = std::env::var_os("COUNT_BENCH_PLAN")
        .map(|p| read(Path::new(&p), &schema::EXECUTION_PLAN))
        .transpose()?;
    let policy = saved
        .as_ref()
        .map(|p| p.claims.gadget_policy)
        .unwrap_or(GadgetPolicy {
            max_log2_rows: 32,
            ..GadgetPolicy::default()
        });
    let lowering = tessera_compiler::lower(PublicModel::of(&model), n)?;
    let ranges = tessera_compiler::analyze(&lowering.graph, &lowering.tables)?;
    let selected = tessera_compiler::select_model(PROTOCOL_VERSION, &lowering, &ranges, &policy)?;
    let claims = match saved {
        Some(p) => {
            if p.claims.protocol_version != PROTOCOL_VERSION || p.claims.arch != model.arch {
                return Err("benchmark metadata does not match this protocol/architecture".into());
            }
            p.claims
        }
        None => tessera_compiler::claim_list(
            PublicModel::of(&model),
            &selected,
            &policy,
            tessera_protocol::ParamSet::NI_100.layout,
            ParamSetId(Ascii::new("bench-counts/1")?),
        )?,
    };
    // Public zero input is a workload, never a recorded-greedy or fidelity claim.
    let runtime = RuntimePlan::derive(&lowering, &selected, &claims, n, 0)?;
    let engine = Engine::new(&lowering)?;
    let plans = engine.runtime_multiplicity_plans(&model.arch, &runtime)?;
    let chosen = plans
        .into_iter()
        .filter(|p| !p.is_dense() && p.rows() >= prefix)
        .max_by_key(|p| p.max_updates())
        .ok_or("no weighted table covers the requested prefix")?;
    let tokens = Secret::new(vec![0u32; n as usize]);
    let checkpoints = Checkpoints::new(&engine, &model, &tokens)?;
    let cache = CheckpointTensors::new(&engine, &model, &tokens, &checkpoints)?;
    let source = Resident::runtime_from_tensors(&engine, &runtime, &model, &cache)?;
    let tile = source.tile(&SequenceTile {
        tile: Tile::Table {
            table: chosen.table().0,
        },
        block: None,
        shares: Vec::new(),
    })?;
    if tile.bytes() as u64 != chosen.payload_bytes() {
        return Err("count backing differs from public plan".into());
    }
    eprintln!(
        "scope=count-reader-only positions={n} table={} logical_rows={} max_updates={} backing_bytes={} setup_seconds={}",
        chosen.table().0,
        chosen.rows(),
        chosen.max_updates(),
        chosen.payload_bytes(),
        started.elapsed().as_secs_f64()
    );
    // The clearing count tile owns its storage. Model/cache/core teardown is outside timing.
    Ok((tile, chosen.max_updates()))
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../..");
    let prefix: usize = std::env::var("COUNT_BENCH_PREFIX_ROWS")
        .unwrap_or_else(|_| ((1usize << 18) + 33).to_string())
        .parse()?;
    if prefix == 0 {
        return Err("benchmark prefix must be positive".into());
    }
    let (tile, updates) = prepare(&root, prefix as u64)?;
    let column = tile.multiplicities().ok_or("table tile")?;
    let mut reference = SecretBuf::try_with_capacity(prefix)?;
    reference.resize(prefix, 0u64);
    for first in [0u64, column.rows() - prefix as u64] {
        column.read_rows(first, &mut reference)?;
        for chunk in [256usize, 4096, 65536] {
            let mut output = SecretBuf::try_with_capacity(chunk)?;
            for offset in (0..prefix).step_by(chunk) {
                let count = chunk.min(prefix - offset);
                output.resize(count, 0);
                column.read_rows(first + offset as u64, &mut output)?;
                assert!(
                    output[..] == reference[offset..offset + count],
                    "public chunk sizes must preserve every count"
                );
            }
        }
    }
    drop(reference);
    let mut criterion = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .configure_from_args();
    let mut group = criterion.benchmark_group("weighted-count-prefix");
    group.throughput(Throughput::Elements(prefix as u64));
    for chunk in [256usize, 4096, 65536] {
        let mut output = SecretBuf::try_with_capacity(chunk)?;
        let visits = (prefix as u64)
            .div_ceil(chunk as u64)
            .checked_mul(updates)
            .ok_or("visit count overflow")?;
        eprintln!("chunk_rows={chunk} prefix_rows={prefix} fixed_update_visits={visits}");
        group.bench_with_input(BenchmarkId::from_parameter(chunk), &chunk, |b, &chunk| {
            b.iter(|| {
                for offset in (0..prefix).step_by(chunk) {
                    output.resize(chunk.min(prefix - offset), 0);
                    column
                        .read_rows(offset as u64, &mut output)
                        .expect("public chunk fits");
                    black_box(&output[..]);
                }
            })
        });
    }
    group.finish();
    criterion.final_summary();
    Ok(())
}
