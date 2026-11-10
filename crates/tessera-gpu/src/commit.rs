// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Encoding and column hashing on the device (#55): the microkernel behind the committed-element
//! unit cost `c_c` (RFC-0020 'Measurement protocol').
//!
//! [`Device::encode_rows`] computes the codewords of RFC-0011 'Encoding' at rate 1/2, the same
//! values as `tessera_pcs::RowEncoder` with `log2_inv_rate = 1`, and [`Device::absorb_block`]
//! advances the chained column digests of RFC-0004 'Column digests', the same bytes as
//! `tessera_hash::ColumnChains`. `kernels/commit.cu` describes the passes.

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use tessera_hash::Domain;

use crate::device::{Device, DeviceBuffer, GpuError};

/// The largest `log2 k` the encoder supports: twiddle indices and launch grids stay in range,
/// and `k = 2^22` is the advice row width (RFC-0011).
pub const ENCODE_MAX_LOG_K: u32 = 26;

/// Goldilocks modulus `p = 2^64 − 2^32 + 1` (RFC-0002).
const P: u64 = 0xFFFF_FFFF_0000_0001;
/// The coset shift and multiplicative generator 7 (RFC-0002, RFC-0011).
const SHIFT: u64 = 7;
/// Elements of the largest shared-memory tile: 64 KiB.
const TILE_LOG: u32 = 13;
/// `log2 G`: strided passes read runs of four consecutive elements (32 bytes, one sector).
const RUN_LOG: u32 = 2;
/// Elements of a strided pass's tile: smaller tiles keep more blocks resident per SM.
const PASS_TILE_LOG: u32 = 12;
/// `log2` of the tile of row values the fused pass handles; its codeword tile is twice as long.
const LOW_LOG: u32 = TILE_LOG - 1;
/// Shared-memory bytes of a tile of `elements`: one `u64` of padding per 16 (`pad` in
/// kernels/commit.cu).
const fn shared_bytes(elements: u32) -> u32 {
    8 * (elements + elements / 16)
}
/// Rows of one column-step input: at most `(1024 − 32) / 8` symbols fit one BLAKE3 chunk.
const MAX_STEP_ROWS: usize = 124;

fn mul(a: u64, b: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(P)) as u64
}

fn pow(mut base: u64, mut exp: u64) -> u64 {
    let mut acc = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul(acc, base);
        }
        base = mul(base, base);
        exp >>= 1;
    }
    acc
}

/// `[1, w, w^2, …]`, `len` entries.
fn powers(w: u64, len: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(len);
    let mut x = 1;
    for _ in 0..len {
        out.push(x);
        x = mul(x, w);
    }
    out
}

/// The kernels of `kernels/commit.cu`.
pub(crate) struct Kernels {
    fill_counter: CudaFunction,
    ntt_pass: CudaFunction,
    encode_low: CudaFunction,
    column_init: CudaFunction,
    column_step: CudaFunction,
    gather_columns: CudaFunction,
    interleave2: CudaFunction,
}

impl Kernels {
    pub(crate) fn load(module: &Arc<CudaModule>) -> Result<Self, GpuError> {
        let kernels = Self {
            fill_counter: module.load_function("fill_counter")?,
            ntt_pass: module.load_function("ntt_pass")?,
            encode_low: module.load_function("encode_low")?,
            column_init: module.load_function("column_init")?,
            column_step: module.load_function("column_step")?,
            gather_columns: module.load_function("gather_columns")?,
            interleave2: module.load_function("interleave2")?,
        };
        for f in [&kernels.ntt_pass, &kernels.encode_low] {
            f.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes(1 << TILE_LOG) as i32,
            )?;
        }
        Ok(kernels)
    }
}

/// The device tables of the rate-1/2 encoder of rows of `k = 2^log_k` values with `P` mask
/// coefficients (RFC-0011 'Encoding').
pub struct Encoder {
    log_k: u32,
    mask_coeffs: usize,
    /// `ω_n^{−i}` for `i < n/2`, `n = 2k`.
    inverse: DeviceBuffer<u64>,
    /// `ω_n^i` for `i < n/2`.
    forward: DeviceBuffer<u64>,
    /// `7^{rev_k(t)} / k` for `t < k`.
    scale: DeviceBuffer<u64>,
    /// `7^i` for `i < P` (one entry when `P = 0`, unread).
    mask_pow: DeviceBuffer<u64>,
    /// `7^k`.
    shift_pow_k: u64,
}

impl Encoder {
    /// Values per row, `k`.
    pub fn message_len(&self) -> usize {
        1 << self.log_k
    }

    /// Symbols per codeword, `n = 2k`.
    pub fn codeword_len(&self) -> usize {
        2 << self.log_k
    }

    /// Mask coefficients per row, `P`.
    pub fn mask_coeffs(&self) -> usize {
        self.mask_coeffs
    }
}

/// One `ntt_pass` launch: `c` stages over index bits `[lo_bit, lo_bit + c)`, runs of `2^log_g`.
struct Pass {
    lo_bit: u32,
    c: u32,
    log_g: u32,
}

/// The strided passes over bits `[from, to)`, each at most a tile of `2^PASS_TILE_LOG` elements, in
/// increasing bit order.
fn passes(from: u32, to: u32) -> Vec<Pass> {
    let mut out = Vec::new();
    let mut lo_bit = from;
    while lo_bit < to {
        let log_g = RUN_LOG.min(lo_bit);
        let c = (to - lo_bit).min(PASS_TILE_LOG - log_g);
        out.push(Pass { lo_bit, c, log_g });
        lo_bit += c;
    }
    out
}

fn check_len<T>(
    operand: &'static str,
    buffer: &DeviceBuffer<T>,
    expected: usize,
) -> Result<(), GpuError> {
    if buffer.len() == expected {
        Ok(())
    } else {
        Err(GpuError::BufferLength {
            operand,
            expected,
            found: buffer.len(),
        })
    }
}

impl Device {
    /// The encoder of rows of `k = 2^log_k` values into codewords of `2k` symbols with
    /// `mask_coeffs = P` mask coefficients, `1 ≤ log_k ≤` [`ENCODE_MAX_LOG_K`], `P ≤ k`.
    pub fn encoder(&self, log_k: u32, mask_coeffs: usize) -> Result<Encoder, GpuError> {
        self.coset_encoder(log_k, mask_coeffs, SHIFT)
    }

    /// The two encoders whose interleaved codewords are the rate-1/4 codeword of RFC-0011
    /// 'Encoding' without masks (a weight row of `tessera-protocol/1`, RFC-0040): symbol `2m` of
    /// `g(7·ω_{4k}^j)` is `g(7·ω_{2k}^m)` and symbol `2m + 1` is `g(7·ω_{4k}·ω_{2k}^m)`, the
    /// rate-1/2 encodings of the same row on the cosets `7·L` and `7·ω_{4k}·L`. Pass both
    /// codewords to [`Device::interleave`].
    pub fn rate4_encoders(&self, log_k: u32) -> Result<[Encoder; 2], GpuError> {
        if !(1..ENCODE_MAX_LOG_K).contains(&log_k) {
            return Err(GpuError::Unsupported("log_k outside 1..26"));
        }
        let omega_4k = pow(SHIFT, (P - 1) >> (log_k + 2));
        Ok([
            self.coset_encoder(log_k, 0, SHIFT)?,
            self.coset_encoder(log_k, 0, mul(SHIFT, omega_4k))?,
        ])
    }

    /// The rate-1/2 encoder evaluating on the coset `shift·L` (masks need `shift = 7`).
    fn coset_encoder(
        &self,
        log_k: u32,
        mask_coeffs: usize,
        shift: u64,
    ) -> Result<Encoder, GpuError> {
        if mask_coeffs > 0 && shift != SHIFT {
            return Err(GpuError::Unsupported("masks need the coset shift 7"));
        }
        if !(1..=ENCODE_MAX_LOG_K).contains(&log_k) {
            return Err(GpuError::Unsupported("log_k outside 1..=26"));
        }
        let k = 1usize << log_k;
        if mask_coeffs > k {
            return Err(GpuError::Unsupported(
                "more mask coefficients than row values",
            ));
        }
        let n = 2 * k;
        let omega = pow(SHIFT, (P - 1) / n as u64);
        let omega_inv = pow(omega, P - 2);
        let k_inv = pow(k as u64, P - 2);
        let shift_powers = powers(shift, k);
        let scale: Vec<u64> = (0..k)
            .map(|t| {
                mul(
                    shift_powers[t.reverse_bits() >> (usize::BITS - log_k)],
                    k_inv,
                )
            })
            .collect();
        let mask_pow = if mask_coeffs == 0 {
            vec![0]
        } else {
            shift_powers[..mask_coeffs].to_vec()
        };
        Ok(Encoder {
            log_k,
            mask_coeffs,
            inverse: self.upload(&powers(omega_inv, k))?,
            forward: self.upload(&powers(omega, k))?,
            scale: self.upload(&scale)?,
            mask_pow: self.upload(&mask_pow)?,
            shift_pow_k: pow(shift, k as u64),
        })
    }

    /// `out[2i] = even[i]`, `out[2i + 1] = odd[i]`: the rate-1/4 codewords of
    /// [`Device::rate4_encoders`] from their two rate-1/2 halves, row after row.
    pub fn interleave(
        &self,
        even: &DeviceBuffer<u64>,
        odd: &DeviceBuffer<u64>,
        out: &mut DeviceBuffer<u64>,
    ) -> Result<(), GpuError> {
        let half = even.len();
        check_len("odd", odd, half)?;
        check_len("out", out, 2 * half)?;
        if half == 0 {
            return Err(GpuError::Unsupported("interleave at least one symbol"));
        }
        let total = half as u64;
        let config = LaunchConfig::for_num_elems(u32::try_from(half).unwrap_or(u32::MAX));
        let mut launch = self.stream.launch_builder(&self.commit.interleave2);
        launch.arg(&even.0).arg(&odd.0).arg(&total).arg(&mut out.0);
        #[allow(unsafe_code)]
        // SAFETY: matches `interleave2` in kernels/commit.cu; it reads even[i] and odd[i] for
        // i < total = even.len() = odd.len() (checked) and writes out[2i], out[2i + 1] below
        // 2·total = out.len() (checked).
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Encodes `rows` (`r` rows of `k` canonical values, one after another) with `masks` (`P`
    /// canonical coefficients per row, `m_0` first; one ignored entry when `P = 0`) into
    /// `codewords` (`r` codewords of `2k` symbols): position `j` of codeword `i` is `g(7·ω_n^j)`
    /// for row `i`'s `g` (RFC-0011 'Encoding'). `rows` is scratch: it holds intermediate values
    /// afterwards. `1 ≤ r ≤ 65,535`.
    pub fn encode_rows(
        &self,
        encoder: &Encoder,
        rows: &mut DeviceBuffer<u64>,
        masks: &DeviceBuffer<u64>,
        codewords: &mut DeviceBuffer<u64>,
    ) -> Result<(), GpuError> {
        let (k, n) = (encoder.message_len(), encoder.codeword_len());
        let count = rows.len() / k;
        if count == 0 || count > 65_535 || rows.len() != count * k {
            return Err(GpuError::Unsupported(
                "rows must hold 1 to 65,535 whole rows",
            ));
        }
        check_len("masks", masks, (count * encoder.mask_coeffs).max(1))?;
        check_len("codewords", codewords, count * n)?;
        let log_k = encoder.log_k;
        let cx = log_k.min(LOW_LOG);
        let log_n = log_k + 1;
        // Pass 1: the size-k DIF stages above the fused tile, highest bits first.
        for pass in passes(cx, log_k).iter().rev() {
            self.ntt_pass(
                &mut rows.0,
                k,
                log_k,
                pass,
                &encoder.inverse,
                log_n,
                false,
                count,
            )?;
        }
        // Pass 2: the fused tile.
        let tile = 1u32 << cx;
        let config = LaunchConfig {
            grid_dim: (1 << (log_k - cx), count as u32, 1),
            // At least a quarter of the first DIT round's groups, as encode_low requires.
            block_dim: ((tile >> 4).max(1), 1, 1),
            shared_mem_bytes: shared_bytes(2 * tile),
        };
        let shift_pow_k = encoder.shift_pow_k;
        let mask_coeffs = encoder.mask_coeffs as u32;
        let mut launch = self.stream.launch_builder(&self.commit.encode_low);
        launch
            .arg(&rows.0)
            .arg(&mut codewords.0)
            .arg(&log_k)
            .arg(&cx)
            .arg(&encoder.inverse.0)
            .arg(&encoder.forward.0)
            .arg(&encoder.scale.0)
            .arg(&masks.0)
            .arg(&encoder.mask_pow.0)
            .arg(&shift_pow_k)
            .arg(&mask_coeffs);
        #[allow(unsafe_code)]
        // SAFETY: the arguments match `encode_low` in kernels/commit.cu in number, order and
        // type. Block (t, r) reads row values [r·k + t·2^cx, +2^cx) of `rows` (count·k checked),
        // writes codeword symbols [r·n + t·2^(cx+1), +2^(cx+1)) (count·n checked), reads
        // `scale` at indices < k (k entries), the twiddle tables at indices < n/2 (k entries
        // each), and `masks`/`mask_pow` at r·P + i and i for i < P (count·P and P entries).
        // Its shared memory is pad(2^(cx+1)) u64 = shared_mem_bytes, within the attribute set at
        // load, and it launches max(1, 2^(cx−4)) ≥ 2^(cx+1−R1)/4 threads (R1 = min(3, cx + 1)).
        unsafe { launch.launch(config) }?;
        // Pass 3: the size-n DIT stages above the fused tile, lowest bits first.
        for pass in passes(cx + 1, log_n) {
            self.ntt_pass(
                &mut codewords.0,
                n,
                log_n,
                &pass,
                &encoder.forward,
                log_n,
                true,
                count,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn ntt_pass(
        &self,
        data: &mut cudarc::driver::CudaSlice<u64>,
        row_stride: usize,
        log_n: u32,
        pass: &Pass,
        table: &DeviceBuffer<u64>,
        log_table: u32,
        dit: bool,
        rows: usize,
    ) -> Result<(), GpuError> {
        let size = 1u32 << (pass.c + pass.log_g);
        let config = LaunchConfig {
            grid_dim: (1 << (log_n - pass.c - pass.log_g), rows as u32, 1),
            block_dim: ((size / 32).clamp(1, 128), 1, 1),
            shared_mem_bytes: shared_bytes(size),
        };
        let row_stride = row_stride as u64;
        let dit = u32::from(dit);
        let mut launch = self.stream.launch_builder(&self.commit.ntt_pass);
        launch
            .arg(data)
            .arg(&row_stride)
            .arg(&log_n)
            .arg(&pass.lo_bit)
            .arg(&pass.c)
            .arg(&pass.log_g)
            .arg(&table.0)
            .arg(&log_table)
            .arg(&dit);
        #[allow(unsafe_code)]
        // SAFETY: the arguments match `ntt_pass` in kernels/commit.cu in number, order and type.
        // The 2^(log_n − c − log_g) tiles of a row partition its 2^log_n indices, so block
        // (t, r) touches only indices < 2^log_n of row r, at offset r·row_stride with
        // rows·row_stride ≤ data.len() (checked by the caller). Twiddle indices are below
        // 2^(log_table − 1), the table's length, since lo_bit + c ≤ log_n ≤ log_table. Shared
        // memory is pad(2^(c + log_g)) u64 with c + log_g ≤ PASS_TILE_LOG: shared_mem_bytes.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Fills `out` with `counter_value(base + i)`: SplitMix64 of the counter, reduced once
    /// below `p`, the counter-based generator of RFC-0020's protocol for benchmark inputs.
    pub fn fill_counter(&self, out: &mut DeviceBuffer<u64>, base: u64) -> Result<(), GpuError> {
        let len = out.len() as u64;
        if len == 0 {
            return Ok(());
        }
        let config = LaunchConfig::for_num_elems(u32::try_from(len).unwrap_or(u32::MAX));
        let mut launch = self.stream.launch_builder(&self.commit.fill_counter);
        launch.arg(&mut out.0).arg(&len).arg(&base);
        #[allow(unsafe_code)]
        // SAFETY: matches `fill_counter(uint64_t *out, uint64_t len, uint64_t base)`; the kernel
        // writes indices below len = out.len() only.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// The column chains of a commitment with `columns` columns: `d_j[0] = H("column-init",
    /// commitment_id ‖ u64(j))`, eight little-endian words per column (RFC-0004).
    pub fn column_chains(
        &self,
        commitment_id: &[u8; 32],
        columns: usize,
    ) -> Result<ColumnChainsGpu, GpuError> {
        if columns == 0 {
            return Err(GpuError::Unsupported(
                "a commitment has at least one column",
            ));
        }
        let words = |bytes: &[u8; 32]| -> Vec<u32> {
            bytes
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect()
        };
        let mut params = words(Domain::ColumnInit.context_key());
        params.extend(words(Domain::ColumnStep.context_key()));
        params.extend(words(commitment_id));
        let params = self.upload(&params)?;
        let mut chains = self.zeroed::<u32>(8 * columns)?;
        let n = columns as u64;
        let config = LaunchConfig::for_num_elems(
            u32::try_from(columns)
                .map_err(|_| GpuError::Unsupported("more than 2^32 − 1 columns"))?,
        );
        let mut launch = self.stream.launch_builder(&self.commit.column_init);
        launch.arg(&mut chains.0).arg(&n).arg(&params.0);
        #[allow(unsafe_code)]
        // SAFETY: matches `column_init(uint32_t *chains, uint64_t n, const uint32_t *params)`;
        // thread j < n writes chains[8j..8j+8) (8n words) and reads params[0..24) (24 words).
        unsafe { launch.launch(config) }?;
        Ok(ColumnChainsGpu {
            chains,
            params,
            columns,
        })
    }

    /// Absorbs one block of `count` rows (`1 ≤ count ≤ 124`, the first `count` codewords of
    /// `codewords`, each `columns` symbols long) into `chains`: `d_j ← H("column-step", d_j ‖
    /// symbols of the block at column j)`, symbols as 8-byte little-endian canonical values.
    pub fn absorb_block(
        &self,
        chains: &mut ColumnChainsGpu,
        codewords: &DeviceBuffer<u64>,
        count: usize,
    ) -> Result<(), GpuError> {
        if count == 0 || count > MAX_STEP_ROWS {
            return Err(GpuError::Unsupported("a block absorbs 1 to 124 rows"));
        }
        let n = chains.columns;
        if codewords.len() < count * n {
            return Err(GpuError::BufferLength {
                operand: "codewords",
                expected: count * n,
                found: codewords.len(),
            });
        }
        let config = LaunchConfig {
            grid_dim: (n.div_ceil(128) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let (n64, count32) = (n as u64, count as u32);
        let mut launch = self.stream.launch_builder(&self.commit.column_step);
        launch
            .arg(&mut chains.chains.0)
            .arg(&codewords.0)
            .arg(&n64)
            .arg(&count32)
            .arg(&chains.params.0);
        #[allow(unsafe_code)]
        // SAFETY: matches `column_step` in kernels/commit.cu; thread j < n reads and writes
        // chains[8j..8j+8) (8n words), reads codewords[r·n + j] for r < count (count·n ≤
        // codewords.len(), checked) and params[8..16) (24 words).
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// `out[r·m + j] = codewords[r·n + columns[j]]` for the first `rows` codewords of `n`
    /// symbols and the `m = columns.len()` columns: P5's queried symbols (#735).
    pub fn gather_columns(
        &self,
        codewords: &DeviceBuffer<u64>,
        n: usize,
        rows: usize,
        columns: &DeviceBuffer<u64>,
        out: &mut DeviceBuffer<u64>,
    ) -> Result<(), GpuError> {
        let m = columns.len();
        if rows * n > codewords.len() || out.len() < rows * m || m == 0 || rows == 0 {
            return Err(GpuError::Unsupported("gathered columns fit their buffers"));
        }
        let (n64, rows32, m32) = (
            n as u64,
            u32::try_from(rows).map_err(|_| GpuError::Unsupported("rows fit u32"))?,
            u32::try_from(m).map_err(|_| GpuError::Unsupported("columns fit u32"))?,
        );
        let config = LaunchConfig::for_num_elems(u32::try_from(rows * m).unwrap_or(u32::MAX));
        let mut launch = self.stream.launch_builder(&self.commit.gather_columns);
        launch
            .arg(&codewords.0)
            .arg(&n64)
            .arg(&rows32)
            .arg(&columns.0)
            .arg(&m32)
            .arg(&mut out.0);
        #[allow(unsafe_code)]
        // SAFETY: matches `gather_columns` in kernels/commit.cu; it reads columns[j] for j < m =
        // columns.len(), and codewords[r·n + c] only for r < rows and c < n (the kernel checks
        // c), so below rows·n ≤ codewords.len(); it writes out[i] for i < rows·m ≤ out.len().
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// The chain values, 32 bytes per column, in column order.
    pub fn download_chains(&self, chains: &ColumnChainsGpu) -> Result<Vec<[u8; 32]>, GpuError> {
        let words = self.download(&chains.chains)?;
        Ok(words
            .chunks_exact(8)
            .map(|w| {
                let mut d = [0u8; 32];
                for (out, word) in d.chunks_exact_mut(4).zip(w) {
                    out.copy_from_slice(&word.to_le_bytes());
                }
                d
            })
            .collect())
    }
}

/// The chain state of every column of one commitment on the device: 32 bytes per column.
pub struct ColumnChainsGpu {
    chains: DeviceBuffer<u32>,
    params: DeviceBuffer<u32>,
    columns: usize,
}
