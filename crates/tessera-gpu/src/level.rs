// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Offloaded window-level rounds (#734): [`GpuLevels`] implements
//! `tessera_sumcheck::LevelOffload`, so the host prover's tree levels run their rounds on the
//! device once their tables are folded. The kernels (`lvl_*` in `kernels/trees.cu`) compute the
//! same field elements as the host's window prover; a proof does not depend on the offload.

use std::sync::{Arc, Mutex, PoisonError};

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use tessera_field::GoldilocksExt2 as Fp2;
use tessera_sumcheck::{DeviceLevel, LevelOffload, LevelShape, WindowError};

use crate::device::{Device, DeviceBuffer, GpuError};
use crate::trees::{SUM_BLOCKS, SUM_THREADS, element, grid, words};

/// The kernels of `kernels/trees.cu` for offloaded levels.
pub(crate) struct Kernels {
    frac_sums: CudaFunction,
    prod_sums: CudaFunction,
    bind: CudaFunction,
}

impl Kernels {
    pub(crate) fn load(module: &Arc<CudaModule>) -> Result<Self, GpuError> {
        Ok(Self {
            frac_sums: module.load_function("lvl_frac_sums")?,
            prod_sums: module.load_function("lvl_prod_sums")?,
            bind: module.load_function("lvl_bind")?,
        })
    }
}

/// The coefficient words of `Fp2` elements, without a copy.
fn as_words(values: &[Fp2]) -> &[u64] {
    #[allow(unsafe_code)]
    // SAFETY: `QuadraticExt<Goldilocks>` is `#[repr(C)]` over two `Goldilocks`, each
    // `#[repr(transparent)]` over its canonical `u64`, so `values` is `2·len` initialized,
    // aligned `u64`s for the borrow's lifetime.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u64>(), 2 * values.len())
    }
}

fn window(e: GpuError) -> WindowError {
    tessera_codec::event!(WARN, "gpu.level_error", error = tessera_codec::Public(format!("{e}")); "GPU level: {error}");
    WindowError::Allocation
}

/// A device that takes level instances up to `budget` bytes in all, and none of fewer than
/// `min_entries` entries per table (the host is faster there).
pub struct GpuLevels {
    device: Arc<Device>,
    budget: u64,
    min_entries: usize,
    used: Arc<Mutex<u64>>,
}

impl GpuLevels {
    /// An offload on `device` holding at most `budget` bytes of level tables at once.
    pub fn new(device: Arc<Device>, budget: u64, min_entries: usize) -> Self {
        Self {
            device,
            budget,
            min_entries: min_entries.max(2),
            used: Arc::new(Mutex::new(0)),
        }
    }
}

impl LevelOffload for GpuLevels {
    fn load(
        &self,
        shape: LevelShape,
        tables: &[&[Fp2]],
    ) -> Result<Option<Box<dyn DeviceLevel>>, WindowError> {
        let len = tables.first().map_or(0, |t| t.len());
        let columns = match shape {
            LevelShape::Fraction { .. } => 4,
            LevelShape::Product => 2,
        };
        if tables.len() != columns
            || len < self.min_entries
            || !len.is_power_of_two()
            || tables.iter().any(|t| t.len() != len)
        {
            return Ok(None);
        }
        // Each table and its half-size partner, 16 bytes per entry.
        let bytes = (columns * (len + len / 2) * 16) as u64;
        {
            let mut used = self.used.lock().unwrap_or_else(PoisonError::into_inner);
            if used.saturating_add(bytes) > self.budget {
                return Ok(None);
            }
            *used += bytes;
        }
        let d = &self.device;
        let load = || -> Result<GpuLevel, GpuError> {
            let mut full = Vec::with_capacity(columns);
            let mut half = Vec::with_capacity(columns);
            for t in tables {
                full.push(d.upload(as_words(t))?);
                half.push(d.zeroed::<u64>(len)?);
            }
            Ok(GpuLevel {
                device: d.clone(),
                shape,
                buffers: [full, half],
                current: 0,
                len,
                scratch: Mutex::new(Scratch {
                    partials: d.zeroed(6 * SUM_BLOCKS as usize)?,
                    sums: d.zeroed(6 * SUM_THREADS as usize)?,
                    eq_lo: d.zeroed(2)?,
                    eq_hi: d.zeroed(2)?,
                }),
                bytes,
                used: self.used.clone(),
            })
        };
        match load() {
            Ok(level) => {
                tessera_codec::event!(DEBUG, "gpu.level_loaded", entries = len as u64; "level instance of {entries} entries on the GPU");
                Ok(Some(Box::new(level)))
            }
            Err(e) => {
                *self.used.lock().unwrap_or_else(PoisonError::into_inner) -= bytes;
                // The host proves any level the device cannot hold (out of device memory, say),
                // with the same messages: the load failing costs time, never the proof.
                tessera_codec::event!(WARN, "gpu.level_error", error = tessera_codec::Public(format!("{e}")); "GPU level: {error}; the host proves this level");
                Ok(None)
            }
        }
    }
}

struct Scratch {
    partials: DeviceBuffer<u64>,
    sums: DeviceBuffer<u64>,
    eq_lo: DeviceBuffer<u64>,
    eq_hi: DeviceBuffer<u64>,
}

/// One instance's tables: `buffers[current]` holds the live tables (`len` entries each), the
/// other set receives the next binding. 2 words per entry.
struct GpuLevel {
    device: Arc<Device>,
    shape: LevelShape,
    buffers: [Vec<DeviceBuffer<u64>>; 2],
    current: usize,
    len: usize,
    scratch: Mutex<Scratch>,
    bytes: u64,
    used: Arc<Mutex<u64>>,
}

impl GpuLevel {
    fn sums_on_device(
        &self,
        low: &[Fp2],
        high: &[Fp2],
        low_bits: usize,
    ) -> Result<[Fp2; 3], GpuError> {
        let d = &self.device;
        let pairs = (self.len / 2) as u64;
        if pairs == 0 || low.len() != 1 << low_bits || (high.len() << low_bits) < pairs as usize {
            return Err(GpuError::Unsupported("level weights cover the pairs"));
        }
        let mut s = self.scratch.lock().unwrap_or_else(PoisonError::into_inner);
        let Scratch {
            partials,
            sums,
            eq_lo,
            eq_hi,
        } = &mut *s;
        if eq_lo.len() < 2 * low.len() {
            *eq_lo = d.zeroed(2 * low.len())?;
        }
        if eq_hi.len() < 2 * high.len() {
            *eq_hi = d.zeroed(2 * high.len())?;
        }
        d.upload_into(as_words(low), eq_lo)?;
        d.upload_into(as_words(high), eq_hi)?;
        let blocks = grid(pairs, SUM_BLOCKS);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (SUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let lo_bits = low_bits as u32;
        let t = &self.buffers[self.current];
        match self.shape {
            LevelShape::Fraction { lambda } => {
                let [l0, l1] = words(lambda);
                let mut launch = d.stream.launch_builder(&d.level.frac_sums);
                launch
                    .arg(&t[0].0)
                    .arg(&t[1].0)
                    .arg(&t[2].0)
                    .arg(&t[3].0)
                    .arg(&pairs)
                    .arg(&l0)
                    .arg(&l1)
                    .arg(&eq_lo.0)
                    .arg(&lo_bits)
                    .arg(&eq_hi.0)
                    .arg(&mut partials.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `lvl_frac_sums`; it reads fp2 entries < 2·pairs = len of the
                // four live tables (each holding at least len entries), weights eq_lo[< 2^lo_bits]
                // and eq_hi[< pairs >> lo_bits] (uploaded above, lengths checked), and writes 3
                // entries per block, blocks ≤ SUM_BLOCKS, of `partials`.
                unsafe { launch.launch(config) }?;
            }
            LevelShape::Product => {
                let mut launch = d.stream.launch_builder(&d.level.prod_sums);
                launch
                    .arg(&t[0].0)
                    .arg(&t[1].0)
                    .arg(&pairs)
                    .arg(&eq_lo.0)
                    .arg(&lo_bits)
                    .arg(&eq_hi.0)
                    .arg(&mut partials.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `lvl_prod_sums`; as for the fraction kernel over two tables.
                unsafe { launch.launch(config) }?;
            }
        }
        d.reduce_sums(blocks, partials, sums)
    }

    fn bind_on_device(&mut self, r: Fp2) -> Result<(), GpuError> {
        let d = self.device.clone();
        let half = (self.len / 2) as u64;
        if half == 0 {
            return Err(GpuError::Unsupported("a bound table has no variable left"));
        }
        let [c0, c1] = words(r);
        let config = LaunchConfig {
            grid_dim: (grid(half, SUM_BLOCKS), 1, 1),
            block_dim: (SUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (from, to) = if self.current == 0 {
            let (a, b) = self.buffers.split_at_mut(1);
            (&a[0], &mut b[0])
        } else {
            let (a, b) = self.buffers.split_at_mut(1);
            (&b[0], &mut a[0])
        };
        for (src, dst) in from.iter().zip(to.iter_mut()) {
            let mut launch = d.stream.launch_builder(&d.level.bind);
            launch
                .arg(&src.0)
                .arg(&mut dst.0)
                .arg(&half)
                .arg(&c0)
                .arg(&c1);
            #[allow(unsafe_code)]
            // SAFETY: matches `lvl_bind`; it reads fp2 entries < 2·half = len of `src` and writes
            // entries < half of `dst`, a distinct buffer of at least len/2 entries.
            unsafe { launch.launch(config) }?;
        }
        self.current ^= 1;
        self.len /= 2;
        Ok(())
    }

    fn firsts_on_device(&self) -> Result<Vec<Fp2>, GpuError> {
        self.buffers[self.current]
            .iter()
            .map(|t| {
                let w = self.device.stream.clone_dtoh(&t.0.slice(0..2))?;
                element(w[0], w[1])
            })
            .collect()
    }
}

impl DeviceLevel for GpuLevel {
    fn sums(&self, low: &[Fp2], high: &[Fp2], low_bits: usize) -> Result<[Fp2; 3], WindowError> {
        self.sums_on_device(low, high, low_bits).map_err(window)
    }

    fn bind(&mut self, r: Fp2) -> Result<(), WindowError> {
        self.bind_on_device(r).map_err(window)
    }

    fn firsts(&self) -> Result<Vec<Fp2>, WindowError> {
        self.firsts_on_device().map_err(window)
    }
}

impl Drop for GpuLevel {
    fn drop(&mut self) {
        // Level tables are secret-class (05 §3 OBS-6).
        for set in &mut self.buffers {
            for b in set.iter_mut() {
                let _ = self.device.clear(b);
            }
        }
        let _ = self.device.synchronize();
        *self.used.lock().unwrap_or_else(PoisonError::into_inner) -= self.bytes;
    }
}
