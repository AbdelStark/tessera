// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The fraction-tree microkernel (#63): a LogUp tree's levels and level sumchecks on the device,
//! the work behind the lookup unit cost `c_ℓ` (RFC-0020 'Measurement protocol').
//!
//! [`Device::prove_fraction_tree_in`] runs the levels of RFC-0010 'Tree levels' for one fraction
//! tree (RFC-0008 'Fraction tree'): the same challenges, messages and leaf claim as
//! `tessera_sumcheck::prove_trees` on that tree alone. The leaves are generated inside the
//! kernels that read them and are never stored: leaf `j` is `(n_j, α − (k_j + β·v_j))`, with `k_j`,
//! `v_j` and the multiplicity behind `n_j = −m_j` drawn from the counter generator, or `n_j = 1`
//! for a lookup tree. [`FractionLeaves`] computes the same leaves on the host for a CPU check.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_sumcheck::{FractionClaim, SumcheckProof, SumcheckTranscript};

use crate::device::{Device, DeviceBuffer, GpuError};
use crate::trees::{SUM_BLOCKS, SUM_THREADS, element, grid, words};

type Fp2 = GoldilocksExt2;

const P: u64 = 0xFFFF_FFFF_0000_0001;

/// The kernels of `kernels/trees.cu` for fraction trees.
pub(crate) struct Kernels {
    level_leaves: CudaFunction,
    level: CudaFunction,
    sums_stored: CudaFunction,
    sums_leaves: CudaFunction,
    bind_sums_stored: CudaFunction,
    bind_sums_leaves: CudaFunction,
    bind_sums_bound: CudaFunction,
    last_stored: CudaFunction,
    last_leaves: CudaFunction,
    last_bound: CudaFunction,
    first_stored: CudaFunction,
    first_leaves: CudaFunction,
}

impl Kernels {
    pub(crate) fn load(module: &Arc<CudaModule>) -> Result<Self, GpuError> {
        Ok(Self {
            level_leaves: module.load_function("frac_level_leaves")?,
            level: module.load_function("frac_level")?,
            sums_stored: module.load_function("frac_sums_stored")?,
            sums_leaves: module.load_function("frac_sums_leaves")?,
            bind_sums_stored: module.load_function("frac_bind_sums_stored")?,
            bind_sums_leaves: module.load_function("frac_bind_sums_leaves")?,
            bind_sums_bound: module.load_function("frac_bind_sums_bound")?,
            last_stored: module.load_function("frac_last_stored")?,
            last_leaves: module.load_function("frac_last_leaves")?,
            last_bound: module.load_function("frac_last_bound")?,
            first_stored: module.load_function("frac_first_stored")?,
            first_leaves: module.load_function("frac_first_leaves")?,
        })
    }
}

/// `counter_value` of `kernels/goldilocks.cuh`: SplitMix64 of the counter, reduced once below `p`.
pub fn counter_value(counter: u64) -> u64 {
    let mut z = counter
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    if z >= P { z - P } else { z }
}

/// The generated leaves of a fraction tree: leaf `j` is `(n_j, α − (k_j + β·v_j))` with `k_j =
/// counter_value(key_seed + j)`, `v_j = counter_value(value_seed + j)`, and `n_j = 1` (a lookup
/// tree, `multiplicity_seed = None`) or `n_j = −m_j`, `m_j = counter_value(multiplicity_seed +
/// j) mod 2^16` (a table tree).
#[derive(Clone, Copy, Debug)]
pub struct FractionLeaves {
    /// The keys' counter offset.
    pub key_seed: u64,
    /// The values' counter offset.
    pub value_seed: u64,
    /// The multiplicities' counter offset, or `None` for numerators of 1.
    pub multiplicity_seed: Option<u64>,
    /// The LogUp challenge `α`.
    pub alpha: Fp2,
    /// The tuple-compression challenge `β`.
    pub beta: Fp2,
}

impl FractionLeaves {
    /// Leaf `j`'s denominator, `α − (k_j + β·v_j)`.
    pub fn denominator(&self, j: u64) -> Fp2 {
        let base = |x: u64| Goldilocks::from_canonical_u64(x).unwrap_or(Goldilocks::ZERO);
        let k = Fp2::from(base(counter_value(self.key_seed.wrapping_add(j))));
        let v = base(counter_value(self.value_seed.wrapping_add(j)));
        self.alpha - k - self.beta.mul_base(v)
    }

    /// Leaf `j`'s numerator in the prime field, or `None` when it is 1.
    pub fn numerator(&self, j: u64) -> Option<Goldilocks> {
        self.multiplicity_seed.map(|seed| {
            let m = counter_value(seed.wrapping_add(j)) & 0xFFFF;
            -Goldilocks::from_canonical_u64(m).unwrap_or(Goldilocks::ZERO)
        })
    }

    /// The kernels' leaf arguments, in their order.
    fn args(&self) -> ([u64; 7], u32) {
        let [a0, a1] = words(self.alpha);
        let [b0, b1] = words(self.beta);
        let unit = u32::from(self.multiplicity_seed.is_none());
        (
            [
                self.key_seed,
                self.value_seed,
                self.multiplicity_seed.unwrap_or(0),
                a0,
                a1,
                b0,
                b1,
            ],
            unit,
        )
    }
}

/// Device memory a fraction-tree proof works in, for one height and leaf kind: see
/// [`Device::fraction_workspace`].
pub struct FractionWorkspace {
    height: usize,
    unit: bool,
    /// Level `t`'s numerators and denominators at index `t − 1`, `1 ≤ t < h`, two words per entry.
    numerators: Vec<DeviceBuffer<u64>>,
    denominators: Vec<DeviceBuffer<u64>>,
    /// The bound tables `[Ne, No, De, Do]` after the last round, and those the next round binds
    /// into.
    current: [DeviceBuffer<u64>; 4],
    spare: [DeviceBuffer<u64>; 4],
    partials: DeviceBuffer<u64>,
    sums: DeviceBuffer<u64>,
    eq_lo: DeviceBuffer<u64>,
    eq_hi: DeviceBuffer<u64>,
    /// A level's four final evaluations.
    finals: DeviceBuffer<u64>,
}

/// Where a level's tables come from.
#[derive(Clone, Copy)]
enum Source<'a> {
    /// A stored level: its numerators and denominators.
    Stored(&'a DeviceBuffer<u64>, &'a DeviceBuffer<u64>),
    /// The generated leaves.
    Leaves,
}

impl Device {
    /// The workspace of proofs of fraction trees of height `h ≥ 1` (`2^h` leaves) with
    /// numerators of 1 (`unit`, a lookup tree) or not (a table tree), allocated once and reused
    /// by [`Device::prove_fraction_tree_in`].
    pub fn fraction_workspace(
        &self,
        height: usize,
        unit: bool,
    ) -> Result<FractionWorkspace, GpuError> {
        if !(1..=40).contains(&height) {
            return Err(GpuError::Unsupported("a fraction tree's height is 1 to 40"));
        }
        // Bound tables of a level hold 2^(n − 1) entries after its first binding, n ≤ h − 1 its
        // variables; the stored levels' numerators at most half that.
        let d_len = 2 * (1usize << height.saturating_sub(2)).max(2);
        let n_len = if unit {
            2 * (1usize << height.saturating_sub(3)).max(2)
        } else {
            d_len
        };
        let eq_len = 2usize << height.saturating_sub(2).div_ceil(2);
        let tables = || -> Result<[DeviceBuffer<u64>; 4], GpuError> {
            Ok([
                self.zeroed(n_len)?,
                self.zeroed(n_len)?,
                self.zeroed(d_len)?,
                self.zeroed(d_len)?,
            ])
        };
        Ok(FractionWorkspace {
            height,
            unit,
            numerators: (1..height)
                .map(|t| self.zeroed(2 << t))
                .collect::<Result<_, _>>()?,
            denominators: (1..height)
                .map(|t| self.zeroed(2 << t))
                .collect::<Result<_, _>>()?,
            current: tables()?,
            spare: tables()?,
            partials: self.zeroed(6 * SUM_BLOCKS as usize)?,
            sums: self.zeroed(6 * SUM_THREADS as usize)?,
            eq_lo: self.zeroed(eq_len)?,
            eq_hi: self.zeroed(eq_len)?,
            finals: self.zeroed(8)?,
        })
    }

    /// Proves the levels of the fraction tree of the workspace's height whose leaves `leaves`
    /// generates, on `transcript`, as `tessera_sumcheck::prove_trees` does for this tree alone
    /// (RFC-0010 'Tree levels', labels `tree/{ℓ}/…`). Returns each level's messages and the leaf
    /// claim. The caller sends the root (`logup/roots`, RFC-0008) before, as for the CPU prover.
    pub fn prove_fraction_tree_in<T: SumcheckTranscript<Fp2> + ?Sized>(
        &self,
        workspace: &mut FractionWorkspace,
        transcript: &mut T,
        leaves: &FractionLeaves,
    ) -> Result<(Vec<SumcheckProof<Fp2>>, FractionClaim<Fp2>), GpuError> {
        if workspace.unit != leaves.multiplicity_seed.is_none() {
            return Err(GpuError::Unsupported(
                "the workspace was made for the other kind of numerators",
            ));
        }
        let FractionWorkspace {
            height,
            numerators,
            denominators,
            current,
            spare,
            partials,
            sums,
            eq_lo,
            eq_hi,
            finals,
            ..
        } = workspace;
        let height = *height;
        let (leaf, unit) = leaves.args();
        self.build_fraction_levels(numerators, denominators, height, leaf, unit)?;

        let mut point: Vec<Fp2> = Vec::new();
        let mut proofs = Vec::with_capacity(height);
        let mut claim = FractionClaim {
            point: Vec::new(),
            numerator: Fp2::ZERO,
            denominator: Fp2::ZERO,
        };
        for level in 0..height {
            let prefix = format!("tree/{level}");
            let _gamma = transcript.challenge(&format!("{prefix}/gamma"));
            let lambda = transcript.challenge(&format!("{prefix}/lambda"));
            let source = match (numerators.get(level), denominators.get(level)) {
                (Some(n), Some(d)) => Source::Stored(n, d),
                _ => Source::Leaves,
            };
            // The level's numerators are 1 only at a lookup tree's leaves.
            let level_unit = matches!(source, Source::Leaves) && unit == 1;
            let n = point.len();
            let mut rounds = Vec::with_capacity(n);
            let mut challenges: Vec<Fp2> = Vec::with_capacity(n);
            let mut scale = Fp2::ONE;
            if n == 0 {
                self.fraction_first(source, leaf, unit, finals)?;
            } else {
                let points =
                    self.upload(&point.iter().copied().flat_map(words).collect::<Vec<_>>())?;
                for (j, &r) in point.iter().enumerate() {
                    let pairs = 1u64 << (n - 1 - j);
                    let rest = (n - 1 - j) as u32;
                    let lo_bits = rest.div_ceil(2);
                    self.eq_table(&points, j as u32 + 1, lo_bits, eq_lo)?;
                    self.eq_table(&points, j as u32 + 1 + lo_bits, rest - lo_bits, eq_hi)?;
                    let previous = challenges.last().copied();
                    let blocks = self.fraction_round(
                        source,
                        (leaf, unit),
                        level_unit,
                        j,
                        previous,
                        pairs,
                        lambda,
                        (eq_lo, lo_bits, eq_hi),
                        current,
                        spare,
                        partials,
                    )?;
                    let s = self.reduce_sums(blocks, partials, sums)?;
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
                let c = challenges.last().copied().unwrap_or(Fp2::ZERO);
                self.fraction_last(source, (leaf, unit), level_unit, n, c, current, finals)?;
            }
            let f = self.stream.clone_dtoh(&finals.0)?;
            let final_evals = vec![
                element(f[0], f[1])?,
                element(f[2], f[3])?,
                element(f[4], f[5])?,
                element(f[6], f[7])?,
            ];
            transcript.absorb_message(&format!("{prefix}/evals"), &final_evals);
            let rho = transcript.challenge(&format!("{prefix}/line"));
            let on_line = |zero: Fp2, one: Fp2| zero + rho * (one - zero);
            point = std::iter::once(rho).chain(challenges).collect();
            claim = FractionClaim {
                point: point.clone(),
                numerator: on_line(final_evals[0], final_evals[1]),
                denominator: on_line(final_evals[2], final_evals[3]),
            };
            proofs.push(SumcheckProof {
                rounds,
                final_evals,
            });
        }
        Ok((proofs, claim))
    }

    /// Levels `h − 1` down to 1: from the generated leaves, then from the level below.
    fn build_fraction_levels(
        &self,
        numerators: &mut [DeviceBuffer<u64>],
        denominators: &mut [DeviceBuffer<u64>],
        height: usize,
        leaf: [u64; 7],
        unit: u32,
    ) -> Result<(), GpuError> {
        for t in (1..height).rev() {
            let parents = 1u64 << t;
            let config = LaunchConfig::for_num_elems(u32::try_from(parents).unwrap_or(u32::MAX));
            let (n_below, n_above) = numerators.split_at_mut(t);
            let (d_below, d_above) = denominators.split_at_mut(t);
            let (n_out, d_out) = (&mut n_below[t - 1], &mut d_below[t - 1]);
            match (n_above.first(), d_above.first()) {
                (Some(n), Some(d)) => {
                    let mut launch = self.stream.launch_builder(&self.fraction.level);
                    launch
                        .arg(&n.0)
                        .arg(&d.0)
                        .arg(&mut n_out.0)
                        .arg(&mut d_out.0)
                        .arg(&parents);
                    #[allow(unsafe_code)]
                    // SAFETY: matches `frac_level`; level t + 1's tables hold 2·2^t fp2 entries,
                    // read at indices < 2·parents, and level t's 2^t, written at x < parents.
                    unsafe { launch.launch(config) }?;
                }
                _ => {
                    let mut launch = self.stream.launch_builder(&self.fraction.level_leaves);
                    for word in &leaf {
                        launch.arg(word);
                    }
                    launch
                        .arg(&unit)
                        .arg(&mut n_out.0)
                        .arg(&mut d_out.0)
                        .arg(&parents);
                    #[allow(unsafe_code)]
                    // SAFETY: matches `frac_level_leaves(LEAF_ARGS, fp2 *n_out, fp2 *d_out,
                    // uint64_t parents)`; it generates leaves and writes x < parents = 2^(h−1)
                    // entries of level h − 1's tables, which hold that many.
                    unsafe { launch.launch(config) }?;
                }
            }
        }
        Ok(())
    }

    /// A level on no variable: its tables' four entries into `finals`.
    fn fraction_first(
        &self,
        source: Source<'_>,
        leaf: [u64; 7],
        unit: u32,
        finals: &mut DeviceBuffer<u64>,
    ) -> Result<(), GpuError> {
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        match source {
            Source::Stored(n, d) => {
                let mut launch = self.stream.launch_builder(&self.fraction.first_stored);
                launch.arg(&n.0).arg(&d.0).arg(&mut finals.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `frac_first_stored`; it reads entries 0 and 1 of level 1's
                // tables (two entries each) and writes four fp2 entries of `finals` (8 words).
                unsafe { launch.launch(config) }?;
            }
            Source::Leaves => {
                let mut launch = self.stream.launch_builder(&self.fraction.first_leaves);
                for word in &leaf {
                    launch.arg(word);
                }
                launch.arg(&unit).arg(&mut finals.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `frac_first_leaves`; it generates two leaves and writes four
                // fp2 entries of `finals` (8 words).
                unsafe { launch.launch(config) }?;
            }
        }
        Ok(())
    }

    /// Launches round `j`'s sums (fused with the binding at the previous challenge from round 1
    /// on) and returns the number of blocks whose partial sums it wrote.
    #[allow(clippy::too_many_arguments)]
    fn fraction_round(
        &self,
        source: Source<'_>,
        (leaf, unit): ([u64; 7], u32),
        level_unit: bool,
        j: usize,
        previous: Option<Fp2>,
        pairs: u64,
        lambda: Fp2,
        (eq_lo, lo_bits, eq_hi): (&DeviceBuffer<u64>, u32, &DeviceBuffer<u64>),
        current: &mut [DeviceBuffer<u64>; 4],
        spare: &mut [DeviceBuffer<u64>; 4],
        partials: &mut DeviceBuffer<u64>,
    ) -> Result<u32, GpuError> {
        let blocks = grid(pairs, SUM_BLOCKS);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (SUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let [l0, l1] = words(lambda);
        let [c0, c1] = words(previous.unwrap_or(Fp2::ZERO));
        let bound_unit = u32::from(level_unit);
        match (j, source) {
            (0, Source::Stored(n, d)) => {
                let mut launch = self.stream.launch_builder(&self.fraction.sums_stored);
                launch.arg(&n.0).arg(&d.0);
                launch
                    .arg(&pairs)
                    .arg(&l0)
                    .arg(&l1)
                    .arg(&eq_lo.0)
                    .arg(&lo_bits)
                    .arg(&eq_hi.0)
                    .arg(&mut partials.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `frac_sums_stored(n, d, SUM_ARGS)`; it reads entries < 4·pairs
                // of the stored level's tables (4·pairs = 2^(n+1) entries), indexes the weights
                // below 2^lo_bits and pairs/2^lo_bits, and writes 3 fp2 per block, blocks ≤
                // SUM_BLOCKS, into `partials`.
                unsafe { launch.launch(config) }?;
            }
            (0, Source::Leaves) => {
                let mut launch = self.stream.launch_builder(&self.fraction.sums_leaves);
                for word in &leaf {
                    launch.arg(word);
                }
                launch.arg(&unit);
                launch
                    .arg(&pairs)
                    .arg(&l0)
                    .arg(&l1)
                    .arg(&eq_lo.0)
                    .arg(&lo_bits)
                    .arg(&eq_hi.0)
                    .arg(&mut partials.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `frac_sums_leaves(LEAF_ARGS, SUM_ARGS)`; the leaves are
                // generated, and the weights and partials are indexed as for a stored level.
                unsafe { launch.launch(config) }?;
            }
            (1, Source::Stored(n, d)) => {
                let [bne, bno, bde, bdd] = current;
                let mut launch = self.stream.launch_builder(&self.fraction.bind_sums_stored);
                launch.arg(&n.0).arg(&d.0);
                launch
                    .arg(&c0)
                    .arg(&c1)
                    .arg(&mut bne.0)
                    .arg(&mut bno.0)
                    .arg(&mut bde.0)
                    .arg(&mut bdd.0);
                launch
                    .arg(&pairs)
                    .arg(&l0)
                    .arg(&l1)
                    .arg(&eq_lo.0)
                    .arg(&lo_bits)
                    .arg(&eq_hi.0)
                    .arg(&mut partials.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `frac_bind_sums_stored(n, d, BIND_ARGS, SUM_ARGS)`; it reads
                // entries < 8·pairs of the stored level (its 2^(n+1) entries) and writes entries
                // < 2·pairs of the four bound tables, which hold at least 2^(n−1) = 2·pairs.
                unsafe { launch.launch(config) }?;
            }
            (1, Source::Leaves) => {
                let [bne, bno, bde, bdd] = current;
                let mut launch = self.stream.launch_builder(&self.fraction.bind_sums_leaves);
                for word in &leaf {
                    launch.arg(word);
                }
                launch.arg(&unit);
                launch
                    .arg(&c0)
                    .arg(&c1)
                    .arg(&mut bne.0)
                    .arg(&mut bno.0)
                    .arg(&mut bde.0)
                    .arg(&mut bdd.0);
                launch
                    .arg(&pairs)
                    .arg(&l0)
                    .arg(&l1)
                    .arg(&eq_lo.0)
                    .arg(&lo_bits)
                    .arg(&eq_hi.0)
                    .arg(&mut partials.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `frac_bind_sums_leaves(LEAF_ARGS, BIND_ARGS, SUM_ARGS)`; the
                // leaves are generated, and it writes entries < 2·pairs = 2^(h−2) of the bound
                // tables (the numerators' only when not unit), which hold that many.
                unsafe { launch.launch(config) }?;
            }
            _ => {
                {
                    let [ne, no, de, dd] = &*current;
                    let [bne, bno, bde, bdd] = &mut *spare;
                    let mut launch = self.stream.launch_builder(&self.fraction.bind_sums_bound);
                    launch
                        .arg(&ne.0)
                        .arg(&no.0)
                        .arg(&de.0)
                        .arg(&dd.0)
                        .arg(&bound_unit);
                    launch
                        .arg(&c0)
                        .arg(&c1)
                        .arg(&mut bne.0)
                        .arg(&mut bno.0)
                        .arg(&mut bde.0)
                        .arg(&mut bdd.0);
                    launch
                        .arg(&pairs)
                        .arg(&l0)
                        .arg(&l1)
                        .arg(&eq_lo.0)
                        .arg(&lo_bits)
                        .arg(&eq_hi.0)
                        .arg(&mut partials.0);
                    #[allow(unsafe_code)]
                    // SAFETY: matches `frac_bind_sums_bound`; it reads entries < 4·pairs of the
                    // current bound tables (bound by the round before to that many) and writes
                    // entries < 2·pairs of the spare ones, a distinct set of buffers of the same
                    // sizes.
                    unsafe { launch.launch(config) }?;
                }
                std::mem::swap(current, spare);
            }
        }
        Ok(blocks)
    }

    /// The last binding of a level on `n ≥ 1` variables at `c`, into `finals`.
    #[allow(clippy::too_many_arguments)]
    fn fraction_last(
        &self,
        source: Source<'_>,
        (leaf, unit): ([u64; 7], u32),
        level_unit: bool,
        n: usize,
        c: Fp2,
        current: &[DeviceBuffer<u64>; 4],
        finals: &mut DeviceBuffer<u64>,
    ) -> Result<(), GpuError> {
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let [c0, c1] = words(c);
        match (n, source) {
            (1, Source::Stored(nt, dt)) => {
                let mut launch = self.stream.launch_builder(&self.fraction.last_stored);
                launch
                    .arg(&nt.0)
                    .arg(&dt.0)
                    .arg(&c0)
                    .arg(&c1)
                    .arg(&mut finals.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `frac_last_stored`; it reads entries < 4 of a level of 4
                // entries and writes four fp2 entries of `finals`.
                unsafe { launch.launch(config) }?;
            }
            (1, Source::Leaves) => {
                let mut launch = self.stream.launch_builder(&self.fraction.last_leaves);
                for word in &leaf {
                    launch.arg(word);
                }
                launch.arg(&unit).arg(&c0).arg(&c1).arg(&mut finals.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `frac_last_leaves`; it generates four leaves and writes four
                // fp2 entries of `finals`.
                unsafe { launch.launch(config) }?;
            }
            _ => {
                let [ne, no, de, dd] = current;
                let bound_unit = u32::from(level_unit);
                let mut launch = self.stream.launch_builder(&self.fraction.last_bound);
                launch
                    .arg(&ne.0)
                    .arg(&no.0)
                    .arg(&de.0)
                    .arg(&dd.0)
                    .arg(&bound_unit)
                    .arg(&c0)
                    .arg(&c1)
                    .arg(&mut finals.0);
                #[allow(unsafe_code)]
                // SAFETY: matches `frac_last_bound`; it reads entries 0 and 1 of the bound
                // tables, bound to two entries by the round before, and writes four fp2 entries
                // of `finals`.
                unsafe { launch.launch(config) }?;
            }
        }
        Ok(())
    }
}
