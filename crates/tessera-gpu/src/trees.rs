// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The product-tree microkernel (#64): a forest's levels and its level sumchecks on the device,
//! the work behind the leaf unit cost `c_π` (RFC-0020 'Measurement protocol').
//!
//! [`Device::prove_forest`] runs the levels of RFC-0010 'Tree levels' for one forest on a
//! transcript: the same challenges, messages and leaf claim as `tessera_sumcheck::prove_forests`
//! on that forest alone. The device builds each round's split `eq` tables, binds the tables and
//! computes the round's three sums in one pass; the host runs the transcript. `kernels/trees.cu` describes the kernels.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_sumcheck::{Claim, SumcheckProof, SumcheckTranscript};

use crate::device::{Device, DeviceBuffer, GpuError};

type Fp2 = GoldilocksExt2;

/// Threads per block of the sum kernels; `kernels/trees.cu`'s `block_sums` assumes it.
pub(crate) const SUM_THREADS: u32 = 256;
/// The most blocks a sum launch uses: each grid-strides over its pairs.
pub(crate) const SUM_BLOCKS: u64 = 2048;

/// The kernels of `kernels/trees.cu`.
pub(crate) struct Kernels {
    tree_level: CudaFunction,
    level_sums_base: CudaFunction,
    bind_sums_base: CudaFunction,
    bind_sums_ext: CudaFunction,
    eq_table: CudaFunction,
    pub(crate) reduce_sums: CudaFunction,
    bind_base: CudaFunction,
    bind_ext: CudaFunction,
}

impl Kernels {
    pub(crate) fn load(module: &Arc<CudaModule>) -> Result<Self, GpuError> {
        Ok(Self {
            tree_level: module.load_function("tree_level")?,
            level_sums_base: module.load_function("level_sums_base")?,
            bind_sums_base: module.load_function("bind_sums_base")?,
            bind_sums_ext: module.load_function("bind_sums_ext")?,
            eq_table: module.load_function("eq_table")?,
            reduce_sums: module.load_function("reduce_sums")?,
            bind_base: module.load_function("bind_base")?,
            bind_ext: module.load_function("bind_ext")?,
        })
    }
}

pub(crate) fn words(x: Fp2) -> [u64; 2] {
    let [a, b] = x.coeffs();
    [a.to_canonical_u64(), b.to_canonical_u64()]
}

pub(crate) fn element(a: u64, b: u64) -> Result<Fp2, GpuError> {
    match (
        Goldilocks::from_canonical_u64(a),
        Goldilocks::from_canonical_u64(b),
    ) {
        (Some(a), Some(b)) => Ok(Fp2::new(a, b)),
        _ => Err(GpuError::Unsupported("a device value is not canonical")),
    }
}

pub(crate) fn grid(items: u64, cap: u64) -> u32 {
    items.div_ceil(u64::from(SUM_THREADS)).clamp(1, cap) as u32
}

/// Device memory a forest proof works in, for one shape: see [`Device::forest_workspace`].
pub struct ForestWorkspace {
    leaves: usize,
    height: usize,
    /// Level `t` at index `t − 1`, for `1 ≤ t < h`.
    levels: Vec<DeviceBuffer<u64>>,
    /// The bound tables `(E, O)` after the last round, and the pair the next round binds into.
    current: (DeviceBuffer<u64>, DeviceBuffer<u64>),
    spare: (DeviceBuffer<u64>, DeviceBuffer<u64>),
    partials: DeviceBuffer<u64>,
    sums: DeviceBuffer<u64>,
    eq_lo: DeviceBuffer<u64>,
    eq_hi: DeviceBuffer<u64>,
}

/// The split `eq` weights of a round: `w(z) = eq_lo[z mod 2^lo_bits]·eq_hi[z >> lo_bits]`.
#[derive(Clone, Copy)]
struct Weights<'a> {
    eq_lo: &'a DeviceBuffer<u64>,
    lo_bits: u32,
    eq_hi: &'a DeviceBuffer<u64>,
}

/// What a round's sum kernel reads, and binds first when it is fused with the round before.
enum Round<'a> {
    /// Round 0, from the base-field level.
    First(&'a DeviceBuffer<u64>),
    /// Round 1: binds the base-field level at round 0's challenge into the outputs.
    BindBase(
        &'a DeviceBuffer<u64>,
        Fp2,
        &'a mut DeviceBuffer<u64>,
        &'a mut DeviceBuffer<u64>,
    ),
    /// A later round: binds `(E, O)` at the last challenge into the outputs.
    BindExt(
        &'a DeviceBuffer<u64>,
        &'a DeviceBuffer<u64>,
        Fp2,
        &'a mut DeviceBuffer<u64>,
        &'a mut DeviceBuffer<u64>,
    ),
}

/// The level tables of the current round, as the rounds bind them.
enum Tables<'a> {
    /// The level itself, base-field, before its first round.
    Base(&'a DeviceBuffer<u64>),
    /// `E` and `O` in `Fp2`, two words per entry.
    Ext(&'a DeviceBuffer<u64>, &'a DeviceBuffer<u64>),
}

impl Device {
    /// The workspace of forest proofs over `leaves` leaves (`2^{s+h}`) in trees of height `h`:
    /// the levels `1, …, h − 1`, the bound tables and the round buffers, allocated once and
    /// reused by [`Device::prove_forest_in`], as RFC-0018's memory scheduler pools memory.
    pub fn forest_workspace(
        &self,
        leaves: usize,
        height: usize,
    ) -> Result<ForestWorkspace, GpuError> {
        if !leaves.is_power_of_two()
            || height == 0
            || height >= usize::BITS as usize
            || 1usize << height > leaves
        {
            return Err(GpuError::Unsupported(
                "a forest has 2^(s+h) leaves and height 1 ≤ h ≤ log2 of their count",
            ));
        }
        let vars = leaves.trailing_zeros() as usize; // s + h
        let width = (leaves / 2).max(2);
        // A round's weights cover at most s + h − 2 variables, split in two halves.
        let eq_len = 2usize << vars.saturating_sub(2).div_ceil(2);
        Ok(ForestWorkspace {
            leaves,
            height,
            levels: (1..height)
                .map(|t| self.zeroed::<u64>(leaves >> (height - t)))
                .collect::<Result<_, _>>()?,
            current: (self.zeroed(width)?, self.zeroed(width)?),
            spare: (self.zeroed(width)?, self.zeroed(width)?),
            partials: self.zeroed(6 * SUM_BLOCKS as usize)?,
            sums: self.zeroed(6 * SUM_THREADS as usize)?,
            eq_lo: self.zeroed(eq_len)?,
            eq_hi: self.zeroed(eq_len)?,
        })
    }

    /// [`Device::prove_forest_in`] in a workspace of its own.
    pub fn prove_forest<T: SumcheckTranscript<Fp2> + ?Sized>(
        &self,
        transcript: &mut T,
        leaves: &DeviceBuffer<u64>,
        height: usize,
    ) -> Result<(Vec<SumcheckProof<Fp2>>, Claim<Fp2>), GpuError> {
        let mut workspace = self.forest_workspace(leaves.len(), height)?;
        self.prove_forest_in(&mut workspace, transcript, leaves)
    }

    /// Proves the levels of the forest of `leaves` (`2^{s+h}` canonical values of `2^s` trees of
    /// the workspace's height `h`) on `transcript`, as `tessera_sumcheck::prove_forests` does for
    /// this forest alone (RFC-0010 'Tree levels', labels `tree/{ℓ}/…`). Returns each level's
    /// messages and the leaf claim.
    pub fn prove_forest_in<T: SumcheckTranscript<Fp2> + ?Sized>(
        &self,
        workspace: &mut ForestWorkspace,
        transcript: &mut T,
        leaves: &DeviceBuffer<u64>,
    ) -> Result<(Vec<SumcheckProof<Fp2>>, Claim<Fp2>), GpuError> {
        if leaves.len() != workspace.leaves {
            return Err(GpuError::BufferLength {
                operand: "leaves",
                expected: workspace.leaves,
                found: leaves.len(),
            });
        }
        let ForestWorkspace {
            height,
            levels,
            current,
            spare,
            partials,
            sums,
            eq_lo,
            eq_hi,
            ..
        } = workspace;
        let height = *height;
        // Level t from level t + 1 (the leaves for t = h − 1), from the leaves up: products of
        // adjacent pairs.
        for t in (1..height).rev() {
            let (below, above) = levels.split_at_mut(t);
            let child = above.first().unwrap_or(leaves);
            let parent = &mut below[t - 1];
            let n = parent.len() as u64;
            let config = LaunchConfig::for_num_elems(u32::try_from(n).unwrap_or(u32::MAX));
            let mut launch = self.stream.launch_builder(&self.trees.tree_level);
            launch.arg(&child.0).arg(&mut parent.0).arg(&n);
            #[allow(unsafe_code)]
            // SAFETY: matches `tree_level(const uint64_t *child, uint64_t *parent, uint64_t
            // parents)`; level t + 1 has twice level t's entries, so the kernel reads child[i]
            // for i < 2·parents = child.len() and writes parent[i] for i < parent.len().
            unsafe { launch.launch(config) }?;
        }
        let vars = leaves.len().trailing_zeros() as usize; // s + h
        let mut point: Vec<Fp2> = (0..vars - height)
            .map(|c| transcript.challenge(&format!("tree/0/root/{c}")))
            .collect();
        let mut proofs = Vec::with_capacity(height);
        let mut value = Fp2::ZERO;
        for level in 0..height {
            let prefix = format!("tree/{level}");
            let _gamma = transcript.challenge(&format!("{prefix}/gamma"));
            let table: &DeviceBuffer<u64> = levels.get(level).unwrap_or(leaves);
            let n = point.len();
            let mut rounds = Vec::with_capacity(n);
            let mut challenges: Vec<Fp2> = Vec::with_capacity(n);
            let mut scale = Fp2::ONE;
            let (u, w) = if n == 0 {
                let v = self.stream.clone_dtoh(&table.0.slice(0..2))?;
                (element(v[0], 0)?, element(v[1], 0)?)
            } else {
                let points =
                    self.upload(&point.iter().copied().flat_map(words).collect::<Vec<_>>())?;
                for (j, &r) in point.iter().enumerate() {
                    let pairs = 1u64 << (n - 1 - j);
                    let rest = (n - 1 - j) as u32;
                    let lo_bits = rest.div_ceil(2);
                    self.eq_table(&points, j as u32 + 1, lo_bits, eq_lo)?;
                    self.eq_table(&points, j as u32 + 1 + lo_bits, rest - lo_bits, eq_hi)?;
                    let weights = Weights {
                        eq_lo,
                        lo_bits,
                        eq_hi,
                    };
                    let s = match (j, challenges.last()) {
                        (0, _) | (_, None) => {
                            self.round_sums(Round::First(table), pairs, &weights, partials, sums)?
                        }
                        (1, Some(&c)) => self.round_sums(
                            Round::BindBase(table, c, &mut current.0, &mut current.1),
                            pairs,
                            &weights,
                            partials,
                            sums,
                        )?,
                        (_, Some(&c)) => {
                            let s = self.round_sums(
                                Round::BindExt(
                                    &current.0,
                                    &current.1,
                                    c,
                                    &mut spare.0,
                                    &mut spare.1,
                                ),
                                pairs,
                                &weights,
                                partials,
                                sums,
                            )?;
                            std::mem::swap(current, spare);
                            s
                        }
                    };
                    // p(c) = scale · eq_j(c) · S(c); S is quadratic, so S(3) = S(0) − 3S(1) + 3S(2).
                    let eq_at = |c: Fp2| (Fp2::ONE - r) * (Fp2::ONE - c) + r * c;
                    let two = Fp2::ONE + Fp2::ONE;
                    let three = two + Fp2::ONE;
                    let s3 = s[0] - three * s[1] + three * s[2];
                    let message = vec![
                        scale * eq_at(Fp2::ZERO) * s[0],
                        scale * eq_at(two) * s[2],
                        scale * eq_at(three) * s3,
                    ];
                    transcript.absorb_message(&format!("{prefix}/round/{j}"), &message);
                    let c = transcript.challenge(&format!("{prefix}/r/{j}"));
                    scale *= eq_at(c);
                    rounds.push(message);
                    challenges.push(c);
                }
                // The last round's binding, to one entry per table.
                let c = challenges.last().copied().unwrap_or(Fp2::ZERO);
                if n == 1 {
                    self.bind(&Tables::Base(table), 1, c, &mut current.0, &mut current.1)?;
                } else {
                    let source = Tables::Ext(&current.0, &current.1);
                    self.bind(&source, 1, c, &mut spare.0, &mut spare.1)?;
                    std::mem::swap(current, spare);
                }
                let e = self.stream.clone_dtoh(&current.0.0.slice(0..2))?;
                let o = self.stream.clone_dtoh(&current.1.0.slice(0..2))?;
                (element(e[0], e[1])?, element(o[0], o[1])?)
            };
            let final_evals = vec![u, w];
            transcript.absorb_message(&format!("{prefix}/evals"), &final_evals);
            let rho = transcript.challenge(&format!("{prefix}/line"));
            value = u + rho * (w - u);
            point = std::iter::once(rho).chain(challenges).collect();
            proofs.push(SumcheckProof {
                rounds,
                final_evals,
            });
        }
        Ok((proofs, Claim { point, value }))
    }

    /// `out[a] = eq(points[first..first + m], a)` for `a < 2^m`.
    pub(crate) fn eq_table(
        &self,
        points: &DeviceBuffer<u64>,
        first: u32,
        m: u32,
        out: &mut DeviceBuffer<u64>,
    ) -> Result<(), GpuError> {
        if points.len() < 2 * (first + m) as usize || out.len() < 2 << m {
            return Err(GpuError::Unsupported("eq table outside its buffers"));
        }
        let config = LaunchConfig::for_num_elems(1 << m);
        let mut launch = self.stream.launch_builder(&self.trees.eq_table);
        launch.arg(&points.0).arg(&first).arg(&m).arg(&mut out.0);
        #[allow(unsafe_code)]
        // SAFETY: matches `eq_table(const fp2 *points, uint32_t first, uint32_t m, fp2 *out)`;
        // it reads fp2 entries first..first + m of `points` and writes entries < 2^m of `out`,
        // both checked above.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// One round's three sums, after the binding of the round before when it is fused in.
    fn round_sums(
        &self,
        round: Round<'_>,
        pairs: u64,
        weights: &Weights<'_>,
        partials: &mut DeviceBuffer<u64>,
        sums: &mut DeviceBuffer<u64>,
    ) -> Result<[Fp2; 3], GpuError> {
        let blocks = grid(pairs, SUM_BLOCKS);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (SUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let Weights {
            eq_lo,
            lo_bits,
            eq_hi,
        } = *weights;
        match round {
            Round::First(v) => {
                let mut launch = self.stream.launch_builder(&self.trees.level_sums_base);
                launch
                    .arg(&v.0)
                    .arg(&pairs)
                    .arg(&eq_lo.0)
                    .arg(&lo_bits)
                    .arg(&eq_hi.0)
                    .arg(&mut partials.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `level_sums_base`; it reads v[4y..4y+4) for y < pairs, and
                // v.len() = 4·pairs (the level of a round-0 table); eq_lo and eq_hi hold the
                // 2^lo_bits and pairs/2^lo_bits entries its weights index; it writes 3 entries per
                // block, blocks ≤ SUM_BLOCKS, within `partials`.
                unsafe { launch.launch(config) }?;
            }
            Round::BindBase(v, c, e_out, o_out) => {
                let [c0, c1] = words(c);
                let mut launch = self.stream.launch_builder(&self.trees.bind_sums_base);
                launch
                    .arg(&v.0)
                    .arg(&pairs)
                    .arg(&c0)
                    .arg(&c1)
                    .arg(&mut e_out.0)
                    .arg(&mut o_out.0)
                    .arg(&eq_lo.0)
                    .arg(&lo_bits)
                    .arg(&eq_hi.0)
                    .arg(&mut partials.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `bind_sums_base`; it reads v[8z..8z+8) for z < pairs, v.len()
                // = 8·pairs (round 1 of the level), writes fp2 entries < 2·pairs of each output
                // (buffers of leaves/2 ≥ 4·pairs words), and indexes weights and partials as in
                // the first round.
                unsafe { launch.launch(config) }?;
            }
            Round::BindExt(e, o, c, e_out, o_out) => {
                let [c0, c1] = words(c);
                let mut launch = self.stream.launch_builder(&self.trees.bind_sums_ext);
                launch
                    .arg(&e.0)
                    .arg(&o.0)
                    .arg(&pairs)
                    .arg(&c0)
                    .arg(&c1)
                    .arg(&mut e_out.0)
                    .arg(&mut o_out.0)
                    .arg(&eq_lo.0)
                    .arg(&lo_bits)
                    .arg(&eq_hi.0)
                    .arg(&mut partials.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `bind_sums_ext`; it reads fp2 entries < 4·pairs of e and o
                // (bound by the round before to 4·pairs entries) and writes entries < 2·pairs of
                // the outputs, a distinct buffer pair; all four hold leaves/2 ≥ 8·pairs words.
                unsafe { launch.launch(config) }?;
            }
        }
        self.reduce_sums(blocks, partials, sums)
    }

    /// The three sums of a round: `partials` holds `blocks` blocks' partial sums, which one more
    /// launch adds up into `sums`, read back.
    pub(crate) fn reduce_sums(
        &self,
        blocks: u32,
        partials: &DeviceBuffer<u64>,
        sums: &mut DeviceBuffer<u64>,
    ) -> Result<[Fp2; 3], GpuError> {
        if partials.len() < 6 * blocks as usize || sums.len() < 6 {
            return Err(GpuError::Unsupported("partial sums outside their buffers"));
        }
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (SUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.trees.reduce_sums);
        launch.arg(&partials.0).arg(&blocks).arg(&mut sums.0);
        #[allow(unsafe_code)]
        // SAFETY: matches `reduce_sums(const fp2 *partials, uint32_t count, fp2 *out)`; it reads
        // 3·blocks fp2 entries of `partials` and writes 3 entries of `sums`, both checked above.
        unsafe { launch.launch(config) }?;
        let s = self.stream.clone_dtoh(&sums.0.slice(0..6))?;
        Ok([
            element(s[0], s[1])?,
            element(s[2], s[3])?,
            element(s[4], s[5])?,
        ])
    }

    fn bind(
        &self,
        source: &Tables<'_>,
        pairs: u64,
        c: Fp2,
        e_out: &mut DeviceBuffer<u64>,
        o_out: &mut DeviceBuffer<u64>,
    ) -> Result<(), GpuError> {
        let [c0, c1] = words(c);
        let config = LaunchConfig::for_num_elems(u32::try_from(pairs).unwrap_or(u32::MAX));
        match source {
            Tables::Base(v) => {
                let mut launch = self.stream.launch_builder(&self.trees.bind_base);
                launch
                    .arg(&v.0)
                    .arg(&pairs)
                    .arg(&c0)
                    .arg(&c1)
                    .arg(&mut e_out.0)
                    .arg(&mut o_out.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `bind_base`; reads v[4y..4y+4) for y < pairs (v.len() =
                // 4·pairs) and writes fp2 entry y < pairs of each output, whose buffers hold
                // 2·(leaves/2) ≥ 2·pairs words.
                unsafe { launch.launch(config) }?;
            }
            Tables::Ext(e, o) => {
                let mut launch = self.stream.launch_builder(&self.trees.bind_ext);
                launch
                    .arg(&e.0)
                    .arg(&o.0)
                    .arg(&pairs)
                    .arg(&c0)
                    .arg(&c1)
                    .arg(&mut e_out.0)
                    .arg(&mut o_out.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `bind_ext`; reads fp2 entries < 2·pairs of e and o and writes
                // entries < pairs of the outputs, a distinct buffer pair; all four hold at least
                // 2·pairs fp2 entries.
                unsafe { launch.launch(config) }?;
            }
        }
        Ok(())
    }
}
