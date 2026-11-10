// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU-backed stages (RFC-0018, #731). A [`Gpu`] is opened from `Backend::Gpu`; each stage it
//! serves computes exactly what the CPU computes, so a GPU proof equals the CPU proof byte for
//! byte. Today: P1's advice encoding and column hashing (#732), the rounds of P2's tree levels
//! (#734) through `tessera_sumcheck`'s level offload, and P5's re-encoding for column answers.

use tessera_hash::Digest;
use tessera_pcs::{BlockSink, ColumnSink, WeightEncoding};

use crate::ProverError;

#[cfg(feature = "gpu")]
pub(crate) use enabled::Gpu;

#[cfg(not(feature = "gpu"))]
/// The GPU context of a build without the `gpu` feature: it never opens.
pub(crate) struct Gpu;

#[cfg(not(feature = "gpu"))]
impl Gpu {
    pub(crate) fn open(_devices: &[u32]) -> Result<Self, ProverError> {
        Err(ProverError::Usage {
            rule: "the GPU backend needs a prover built with the `gpu` feature",
        })
    }

    pub(crate) fn advice_sink(
        &self,
        _: u32,
        _: usize,
        _: usize,
        _: &Digest,
    ) -> Result<Box<dyn BlockSink + '_>, ProverError> {
        Err(ProverError::Usage {
            rule: "the GPU backend needs a prover built with the `gpu` feature",
        })
    }

    pub(crate) fn column_sink(
        &self,
        _: u32,
        _: usize,
        _: usize,
        _: &[u64],
    ) -> Result<Box<dyn ColumnSink + '_>, ProverError> {
        Err(ProverError::Usage {
            rule: "the GPU backend needs a prover built with the `gpu` feature",
        })
    }

    pub(crate) fn weight_encoding(&self) -> Box<dyn WeightEncoding + '_> {
        Box::new(HostWeights)
    }
}

#[cfg(not(feature = "gpu"))]
/// Declines every matrix: the host encodes it.
struct HostWeights;

#[cfg(not(feature = "gpu"))]
impl WeightEncoding for HostWeights {
    fn matrix(
        &mut self,
        _: u32,
        _: u32,
        _: usize,
        _: &Digest,
        _: usize,
        _: &[u64],
    ) -> Result<Option<Box<dyn tessera_pcs::MatrixSink + '_>>, tessera_pcs::PcsError> {
        Ok(None)
    }
}

#[cfg(feature = "gpu")]
mod enabled {
    use super::*;
    use std::sync::Arc;
    use tessera_field::Goldilocks;
    use tessera_gpu::{ColumnChainsGpu, Device, DeviceBuffer, Encoder, GpuError, GpuLevels};

    /// Level tables the offload may hold at once, unless `TESSERA_GPU_LEVEL_GIB` says otherwise:
    /// room on an 80 GB device beside P1's buffers.
    const LEVEL_GIB: u64 = 60;
    /// Smaller level tables stay on the host, where a round costs less than a transfer.
    const LEVEL_MIN_ENTRIES: usize = 1 << 14;
    use tessera_pcs::PcsError;

    fn device(e: GpuError) -> PcsError {
        // The driver's message is public: it names no secret value (05 §3).
        tessera_codec::event!(WARN, "prover.gpu_error", error = tessera_codec::Public(format!("{e}")); "GPU: {error}");
        PcsError::Device {
            reason: "a CUDA call failed",
        }
    }

    /// Encoders by `(log_k, mask_coeffs)`.
    type Encoders = std::sync::Mutex<Vec<((u32, usize), Arc<Encoder>)>>;

    /// One CUDA device and the encoders built on it, reused across a proof's commitments.
    pub(crate) struct Gpu {
        device: Arc<Device>,
        encoders: Encoders,
    }

    impl Gpu {
        pub(crate) fn open(devices: &[u32]) -> Result<Self, ProverError> {
            let &[ordinal] = devices else {
                return Err(ProverError::Usage {
                    rule: "the GPU backend runs on exactly one device",
                });
            };
            let device = Device::new(ordinal as usize).map_err(|e| {
                tessera_codec::event!(WARN, "prover.gpu_open_failed", ordinal = ordinal, error = tessera_codec::Public(format!("{e}")); "GPU {ordinal}: {error}");
                ProverError::Usage {
                    rule: "the GPU backend's device opens",
                }
            })?;
            let device = Arc::new(device);
            let gib = std::env::var("TESSERA_GPU_LEVEL_GIB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(LEVEL_GIB);
            let min = std::env::var("TESSERA_GPU_LEVEL_MIN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(LEVEL_MIN_ENTRIES);
            tessera_sumcheck::set_level_offload(Some(Arc::new(GpuLevels::new(
                device.clone(),
                gib << 30,
                min,
            ))));
            Ok(Self {
                device,
                encoders: std::sync::Mutex::new(Vec::new()),
            })
        }

        fn encoder(&self, log_k: u32, mask_coeffs: usize) -> Result<Arc<Encoder>, PcsError> {
            let mut held = self
                .encoders
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some((_, e)) = held.iter().find(|(key, _)| *key == (log_k, mask_coeffs)) {
                return Ok(e.clone());
            }
            let e = Arc::new(self.device.encoder(log_k, mask_coeffs).map_err(device)?);
            held.push(((log_k, mask_coeffs), e.clone()));
            Ok(e)
        }

        /// A sink for one segment of a rate-1/2 advice commitment of rows of `2^log_k` values with
        /// `mask_coeffs` coefficients, in blocks of `block` rows, from the chains of `id`.
        pub(crate) fn advice_sink(
            &self,
            log_k: u32,
            mask_coeffs: usize,
            block: usize,
            id: &Digest,
        ) -> Result<Box<dyn BlockSink + '_>, ProverError> {
            let encoder = self.encoder(log_k, mask_coeffs).map_err(ProverError::Pcs)?;
            let (k, n) = (encoder.message_len(), encoder.codeword_len());
            let chains = self
                .device
                .column_chains(id, n)
                .map_err(|e| ProverError::Pcs(device(e)))?;
            let alloc = |len: usize| {
                self.device
                    .zeroed::<u64>(len)
                    .map_err(|e| ProverError::Pcs(device(e)))
            };
            Ok(Box::new(AdviceSink {
                device: &self.device,
                blocks: 0,
                busy: 0.0,
                last: std::time::Instant::now(),
                k,
                mask_coeffs,
                block,
                rows: alloc(block * k)?,
                masks: alloc((block * mask_coeffs).max(1))?,
                codewords: alloc(block * n)?,
                chains: Some(chains),
                encoder,
            }))
        }
    }

    impl Gpu {
        /// A sink re-encoding rate-1/2 advice rows in blocks of `block` and returning the
        /// symbols at `columns` (P5).
        pub(crate) fn column_sink(
            &self,
            log_k: u32,
            mask_coeffs: usize,
            block: usize,
            columns: &[u64],
        ) -> Result<Box<dyn ColumnSink + '_>, ProverError> {
            let encoder = self.encoder(log_k, mask_coeffs).map_err(ProverError::Pcs)?;
            let (k, n) = (encoder.message_len(), encoder.codeword_len());
            let pcs = |e| ProverError::Pcs(device(e));
            let d = &self.device;
            Ok(Box::new(ColumnsSink {
                device: d,
                k,
                n,
                mask_coeffs,
                block,
                m: columns.len(),
                columns: d.upload(columns).map_err(pcs)?,
                rows: d.zeroed(block * k).map_err(pcs)?,
                masks: d.zeroed((block * mask_coeffs).max(1)).map_err(pcs)?,
                codewords: d.zeroed(block * n).map_err(pcs)?,
                out: d.zeroed((block * columns.len()).max(1)).map_err(pcs)?,
                encoder,
            }))
        }
    }

    struct ColumnsSink<'d> {
        device: &'d Device,
        encoder: Arc<Encoder>,
        k: usize,
        n: usize,
        mask_coeffs: usize,
        block: usize,
        m: usize,
        columns: DeviceBuffer<u64>,
        rows: DeviceBuffer<u64>,
        masks: DeviceBuffer<u64>,
        codewords: DeviceBuffer<u64>,
        out: DeviceBuffer<u64>,
    }

    impl ColumnsSink<'_> {
        fn run(&mut self, rows: &[Goldilocks], masks: &[Goldilocks]) -> Result<Vec<u64>, GpuError> {
            let d = self.device;
            let count = rows.len() / self.k;
            let mut short = None;
            let (r, m, c) = if count == self.block {
                (&mut self.rows, &mut self.masks, &mut self.codewords)
            } else {
                let s = short.insert((
                    d.zeroed::<u64>(count * self.k)?,
                    d.zeroed::<u64>((count * self.mask_coeffs).max(1))?,
                    d.zeroed::<u64>(count * self.n)?,
                ));
                (&mut s.0, &mut s.1, &mut s.2)
            };
            d.upload_field_into(rows, r)?;
            d.upload_field_into(masks, m)?;
            d.encode_rows(&self.encoder, r, m, c)?;
            d.gather_columns(c, self.n, count, &self.columns, &mut self.out)?;
            let words = d.download(&self.out)?;
            if let Some((mut r, mut m, mut c)) = short {
                for b in [&mut r, &mut m, &mut c] {
                    d.clear(b)?;
                }
            }
            Ok(words[..count * self.m].to_vec())
        }
    }

    impl ColumnSink for ColumnsSink<'_> {
        fn select(
            &mut self,
            rows: &[Goldilocks],
            masks: &[Goldilocks],
        ) -> Result<Vec<Goldilocks>, PcsError> {
            let count = rows.len() / self.k;
            if count == 0 || count > self.block || rows.len() != count * self.k {
                return Err(PcsError::Device {
                    reason: "a block of whole rows",
                });
            }
            let words = self.run(rows, masks).map_err(device)?;
            words
                .into_iter()
                .map(|w| {
                    <Goldilocks as tessera_field::PrimeField>::from_canonical_u64(w).ok_or(
                        PcsError::Device {
                            reason: "a device symbol is canonical",
                        },
                    )
                })
                .collect()
        }
    }

    impl Drop for ColumnsSink<'_> {
        fn drop(&mut self) {
            for b in [
                &mut self.rows,
                &mut self.masks,
                &mut self.codewords,
                &mut self.out,
            ] {
                let _ = self.device.clear(b);
            }
            let _ = self.device.synchronize();
        }
    }

    impl Drop for Gpu {
        fn drop(&mut self) {
            tessera_sumcheck::set_level_offload(None);
        }
    }

    impl Gpu {
        /// The weight matrices' rate-1/4 encoding and column hashing on the device (RFC-0040):
        /// the weight recommit before a proof and P5's regenerated weight answers.
        pub(crate) fn weight_encoding(&self) -> Box<dyn WeightEncoding + '_> {
            Box::new(GpuWeights {
                device: &self.device,
            })
        }
    }

    struct GpuWeights<'d> {
        device: &'d Device,
    }

    impl WeightEncoding for GpuWeights<'_> {
        fn matrix(
            &mut self,
            log_k: u32,
            log2_inv_rate: u32,
            block: usize,
            commitment_id: &Digest,
            columns: usize,
            queries: &[u64],
        ) -> Result<Option<Box<dyn tessera_pcs::MatrixSink + '_>>, PcsError> {
            let k = 1usize << log_k;
            // The device encodes rate 1/4 (two rate-1/2 cosets); a block absorbs at most 124 rows.
            if log2_inv_rate != 2 || columns != 4 * k || block == 0 || block > 124 {
                return Ok(None);
            }
            let d = self.device;
            let [even, odd] = d.rate4_encoders(log_k).map_err(device)?;
            let alloc = |len: usize| d.zeroed::<u64>(len.max(1)).map_err(device);
            Ok(Some(Box::new(WeightSink {
                device: d,
                k,
                block,
                q: queries.len(),
                chains: Some(d.column_chains(commitment_id, columns).map_err(device)?),
                queries: d
                    .upload(if queries.is_empty() { &[0] } else { queries })
                    .map_err(device)?,
                even,
                odd,
                rows: alloc(block * k)?,
                halves: [alloc(block * 2 * k)?, alloc(block * 2 * k)?],
                codewords: alloc(block * 4 * k)?,
                out: alloc(block * queries.len())?,
            })))
        }
    }

    struct WeightSink<'d> {
        device: &'d Device,
        k: usize,
        block: usize,
        q: usize,
        chains: Option<ColumnChainsGpu>,
        queries: DeviceBuffer<u64>,
        even: Encoder,
        odd: Encoder,
        rows: DeviceBuffer<u64>,
        halves: [DeviceBuffer<u64>; 2],
        codewords: DeviceBuffer<u64>,
        out: DeviceBuffer<u64>,
    }

    impl WeightSink<'_> {
        fn run(&mut self, rows: &[Goldilocks], count: usize) -> Result<Vec<u64>, GpuError> {
            let d = self.device;
            let k = self.k;
            // A short final block gets buffers of its exact size: the kernels take whole buffers.
            let mut short = None;
            let (r, h, c) = if count == self.block {
                (&mut self.rows, &mut self.halves, &mut self.codewords)
            } else {
                let s = short.insert((
                    d.zeroed::<u64>(count * k)?,
                    [
                        d.zeroed::<u64>(count * 2 * k)?,
                        d.zeroed::<u64>(count * 2 * k)?,
                    ],
                    d.zeroed::<u64>(count * 4 * k)?,
                ));
                (&mut s.0, &mut s.1, &mut s.2)
            };
            let none = d.upload(&[0u64])?;
            // encode_rows uses its rows as scratch: each coset encodes a fresh upload.
            for (encoder, half) in [&self.even, &self.odd].into_iter().zip(h.iter_mut()) {
                d.upload_field_into(rows, r)?;
                d.encode_rows(encoder, r, &none, half)?;
            }
            d.interleave(&h[0], &h[1], c)?;
            let chains = self
                .chains
                .as_mut()
                .ok_or(GpuError::Unsupported("chains finished"))?;
            d.absorb_block(chains, c, count)?;
            let words = if self.q == 0 {
                Vec::new()
            } else {
                d.gather_columns(c, 4 * k, count, &self.queries, &mut self.out)?;
                d.download(&self.out)?[..count * self.q].to_vec()
            };
            if let Some((mut r, [mut h0, mut h1], mut c)) = short {
                for b in [&mut r, &mut h0, &mut h1, &mut c] {
                    d.clear(b)?;
                }
            }
            Ok(words)
        }
    }

    impl tessera_pcs::MatrixSink for WeightSink<'_> {
        fn absorb(&mut self, rows: &[Goldilocks]) -> Result<Vec<Goldilocks>, PcsError> {
            let count = rows.len() / self.k;
            if count == 0 || count > self.block || rows.len() != count * self.k {
                return Err(PcsError::Device {
                    reason: "a block of whole weight rows",
                });
            }
            let words = self.run(rows, count).map_err(device)?;
            words
                .into_iter()
                .map(|w| {
                    <Goldilocks as tessera_field::PrimeField>::from_canonical_u64(w).ok_or(
                        PcsError::Device {
                            reason: "a device symbol is canonical",
                        },
                    )
                })
                .collect()
        }

        fn finish(&mut self) -> Result<Vec<Digest>, PcsError> {
            let chains = self.chains.take().ok_or(PcsError::Device {
                reason: "chains finished twice",
            })?;
            self.device.download_chains(&chains).map_err(device)
        }
    }

    impl Drop for WeightSink<'_> {
        fn drop(&mut self) {
            // Weight rows and codewords are secret-class (05 §3 OBS-6).
            let [h0, h1] = &mut self.halves;
            for b in [&mut self.rows, h0, h1, &mut self.codewords, &mut self.out] {
                let _ = self.device.clear(b);
            }
            let _ = self.device.synchronize();
        }
    }

    struct AdviceSink<'d> {
        device: &'d Device,
        /// Blocks absorbed, the time inside the sink, and when the last block left it.
        blocks: u64,
        busy: f64,
        last: std::time::Instant,
        encoder: Arc<Encoder>,
        chains: Option<ColumnChainsGpu>,
        k: usize,
        mask_coeffs: usize,
        block: usize,
        rows: DeviceBuffer<u64>,
        masks: DeviceBuffer<u64>,
        codewords: DeviceBuffer<u64>,
    }

    impl AdviceSink<'_> {
        fn run(
            &mut self,
            rows: &[Goldilocks],
            masks: &[Goldilocks],
            count: usize,
        ) -> Result<(), GpuError> {
            let d = self.device;
            // A short final block gets buffers of its exact size: the kernels take whole buffers.
            let mut short = None;
            let (r, m, c) = if count == self.block {
                (&mut self.rows, &mut self.masks, &mut self.codewords)
            } else {
                let s = short.insert((
                    d.zeroed::<u64>(count * self.k)?,
                    d.zeroed::<u64>((count * self.mask_coeffs).max(1))?,
                    d.zeroed::<u64>(count * self.encoder.codeword_len())?,
                ));
                (&mut s.0, &mut s.1, &mut s.2)
            };
            d.upload_field_into(rows, r)?;
            d.upload_field_into(masks, m)?;
            d.encode_rows(&self.encoder, r, m, c)?;
            let chains = self
                .chains
                .as_mut()
                .ok_or(GpuError::Unsupported("chains finished"))?;
            d.absorb_block(chains, c, count)?;
            if let Some((mut r, mut m, mut c)) = short {
                for b in [&mut r, &mut m, &mut c] {
                    d.clear(b)?;
                }
            }
            Ok(())
        }
    }

    impl BlockSink for AdviceSink<'_> {
        fn absorb(&mut self, rows: &[Goldilocks], masks: &[Goldilocks]) -> Result<(), PcsError> {
            let count = rows.len() / self.k;
            if count == 0
                || count > self.block
                || rows.len() != count * self.k
                || masks.len() != count * self.mask_coeffs
            {
                return Err(PcsError::Device {
                    reason: "a block of whole rows and their masks",
                });
            }
            let started = std::time::Instant::now();
            let waited = started.duration_since(self.last).as_secs_f64();
            self.run(rows, masks, count).map_err(device)?;
            self.busy += started.elapsed().as_secs_f64();
            self.blocks += 1;
            self.last = std::time::Instant::now();
            if self.blocks == 1 || self.blocks.is_multiple_of(16) {
                let (blocks, busy_ms, gap_ms) =
                    (self.blocks, (self.busy * 1e3) as u64, (waited * 1e3) as u64);
                tessera_codec::event!(DEBUG, "prover.gpu_advice_blocks", blocks = blocks, busy_ms = busy_ms, gap_ms = gap_ms; "{blocks} advice blocks: {busy_ms} ms on the device path, last gap {gap_ms} ms");
            }
            Ok(())
        }

        fn finish(&mut self) -> Result<Vec<Digest>, PcsError> {
            let chains = self.chains.take().ok_or(PcsError::Device {
                reason: "chains finished twice",
            })?;
            let digests = self.device.download_chains(&chains).map_err(device)?;
            Ok(digests)
        }
    }

    impl Drop for AdviceSink<'_> {
        fn drop(&mut self) {
            // Rows, masks and codewords are secret-class (05 §3 OBS-6).
            for b in [&mut self.rows, &mut self.masks, &mut self.codewords] {
                let _ = self.device.clear(b);
            }
            let _ = self.device.synchronize();
        }
    }
}
