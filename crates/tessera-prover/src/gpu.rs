// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU-backed stages (RFC-0018, #731). A [`Gpu`] is opened from `Backend::Gpu`; each stage it
//! serves computes exactly what the CPU computes, so a GPU proof equals the CPU proof byte for
//! byte. Today: P1's advice encoding and column hashing (#732).

use tessera_hash::Digest;
use tessera_pcs::BlockSink;

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
}

#[cfg(feature = "gpu")]
mod enabled {
    use super::*;
    use tessera_field::Goldilocks;
    use tessera_gpu::{ColumnChainsGpu, Device, DeviceBuffer, Encoder, GpuError};
    use tessera_pcs::PcsError;

    fn device(e: GpuError) -> PcsError {
        // The driver's message is public: it names no secret value (05 §3).
        tessera_codec::event!(WARN, "prover.gpu_error", error = tessera_codec::Public(format!("{e}")); "GPU: {error}");
        PcsError::Device {
            reason: "a CUDA call failed",
        }
    }

    /// One CUDA device and the encoders built on it, reused across a proof's commitments.
    pub(crate) struct Gpu {
        device: Device,
        encoders: std::sync::Mutex<Vec<((u32, usize), std::sync::Arc<Encoder>)>>,
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
            Ok(Self {
                device,
                encoders: std::sync::Mutex::new(Vec::new()),
            })
        }

        fn encoder(
            &self,
            log_k: u32,
            mask_coeffs: usize,
        ) -> Result<std::sync::Arc<Encoder>, PcsError> {
            let mut held = self
                .encoders
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some((_, e)) = held.iter().find(|(key, _)| *key == (log_k, mask_coeffs)) {
                return Ok(e.clone());
            }
            let e = std::sync::Arc::new(self.device.encoder(log_k, mask_coeffs).map_err(device)?);
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

    struct AdviceSink<'d> {
        device: &'d Device,
        encoder: std::sync::Arc<Encoder>,
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
            self.run(rows, masks, count).map_err(device)
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
