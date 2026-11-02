// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public resident-tree storage for RFC-0010/0018/0054.

use core::ops::Range;

use tessera_codec::SecretBuf;
use tessera_poly::PolyField;

use super::{FractionLevel, LevelClaim, ProvenTreeLevel, State};
use crate::WindowError;
use crate::forest::Halves;
use crate::level::{LevelInstance, Split};

/// One canonical resident tree's public shape, in the actual batch order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentTreeShape {
    /// A single fraction tree; height zero is a root with no allocated level.
    Fraction {
        /// Number of binary levels below the root.
        height: usize,
        /// Explicit prime-subfield leaf numerators, rather than implicit ones.
        base_numerators: bool,
    },
    /// A product forest with `2^trees_log` trees sharing one leaf table.
    Forest {
        /// At least one binary level below each root.
        height: usize,
        /// Variables selecting the forest's trees above the leaf variables.
        trees_log: usize,
    },
}

impl ResidentTreeShape {
    fn height(self) -> usize {
        match self {
            Self::Fraction { height, .. } | Self::Forest { height, .. } => height,
        }
    }
    fn trees_log(self) -> usize {
        match self {
            Self::Fraction { .. } => 0,
            Self::Forest { trees_log, .. } => trees_log,
        }
    }
    fn fraction(self) -> bool {
        matches!(self, Self::Fraction { .. })
    }
}

/// Requested heap bounds for canonical fraction construction and `prove_trees_in`.
///
/// Borrowed forest leaves and numerator/denominator sources, the consumed input
/// vector (including spare capacity), transcript/encoding callbacks, stack,
/// allocator and process overhead are additional. Fraction trees must come from
/// `FractionTree::new` at these shapes. No field value or challenge is inspected,
/// and this component alone neither authenticates shapes nor admits a CPU route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentTreeMemory {
    fractions: u64,
    peak: u64,
    output: u64,
    levels: u64,
}

impl ResidentTreeMemory {
    /// Retained canonical fraction levels, also bounding their sequential
    /// construction peak. Forest levels are constructed inside the proving peak.
    pub fn fraction_bytes(self) -> u64 {
        self.fractions
    }

    /// Complete owned proving peak, including all fraction trees, constructed
    /// forest levels, kernel scratch and returned output. Exclusions above apply.
    pub fn peak_bytes(self) -> u64 {
        self.peak
    }

    /// Returned levels and leaf claims, including all nested points and messages.
    pub fn output_bytes(self) -> u64 {
        self.output
    }

    /// Returned level messages and carried claims, excluding final leaf claims.
    /// A wrapper can release those leaves while retaining the complete levels.
    pub fn level_bytes(self) -> u64 {
        self.levels
    }

    /// Price every scheduled tree, including repeated or equal shapes, without
    /// allocating or consulting a witness. An empty schedule is supported.
    ///
    /// # Errors
    /// `WindowError::Shape` for a height-zero forest or an unrepresentable variable
    /// count; `WindowError::Allocation` for checked storage overflow/addressability.
    pub fn new<F: PolyField>(shapes: &[ResidentTreeShape]) -> Result<Self, WindowError> {
        let (mut fractions, mut forests, mut state_points) = (0, 0, 0);
        let mut tallest = 0;
        for &shape in shapes {
            let height = shape.height();
            let vars = height
                .checked_add(shape.trees_log())
                .filter(|&n| n < usize::BITS as usize)
                .ok_or(WindowError::Shape("resident tree variables fit an index"))?;
            if !shape.fraction() && height == 0 {
                return Err(WindowError::Shape("a resident forest has positive height"));
            }
            tallest = tallest.max(height);
            // Old/current state points and cloned carried claims overlap while
            // advancing a level. Charge their maximum size throughout the batch.
            state_points = add(state_points, mul(4, array::<F>(vars as u64)?)?)?;
            match shape {
                ResidentTreeShape::Fraction {
                    base_numerators, ..
                } if height > 0 => {
                    let mut bytes = array::<FractionLevel<F>>((height + 1) as u64)?;
                    bytes = add(bytes, array::<F>(power(height)?)?)?;
                    if base_numerators {
                        bytes = add(bytes, array::<F::Base>(power(height)?)?)?;
                    }
                    for level in 1..height {
                        bytes = add(bytes, array::<F>(mul(2, power(level)?)?)?)?;
                    }
                    fractions = add(fractions, bytes)?;
                }
                ResidentTreeShape::Forest { trees_log, .. } => {
                    forests = add(forests, array::<Halves<F::Base>>((height + 1) as u64)?)?;
                    for level in 1..=height {
                        forests = add(forests, array::<F::Base>(power(trees_log + level)?)?)?;
                    }
                }
                ResidentTreeShape::Fraction { .. } => {}
            }
        }
        let count = shapes.len() as u64;
        let mut levels = array::<ProvenTreeLevel<F>>(tallest as u64)?;
        let mut leaves =
            // In-place iterator collection may retain the larger State allocation
            // when extracting leaf claims. Do not infer capacity from claim count.
            grow::<LevelClaim<F>>(count)?.max(grow::<State<F>>(count)?);
        for &shape in shapes {
            leaves = add(
                leaves,
                array::<F>((shape.height() + shape.trees_log()) as u64)?,
            )?;
        }
        let mut scratch = 0;
        for level in 0..tallest {
            let (mut active, mut rounds, mut evaluations, mut root_vars) = (0, 0, 0, 0);
            let (mut folded, mut points, mut equality) = (0, 0, 0);
            for &shape in shapes {
                let join = tallest - shape.height();
                if join > level {
                    continue;
                }
                let vars = shape.trees_log() + level - join;
                active += 1;
                rounds = rounds.max(vars as u64);
                evaluations = add(evaluations, if shape.fraction() { 4 } else { 2 })?;
                if join == level {
                    root_vars = root_vars.max(shape.trees_log() as u64);
                }
                points = add(points, array::<F>((vars + 1) as u64)?)?;
                if vars > 0 {
                    let tables = if shape.fraction() { 4 } else { 2 };
                    // Unit numerators can save two buffers at the final level;
                    // charge four fraction operands even in that case.
                    folded = add(folded, array::<F>(mul(tables, power(vars - 1)?)?)?)?;
                }
                // Two split equality tables plus the fraction kernel's scaled
                // low table. This also covers product_claim before prove_level.
                equality = equality.max(mul(3, array::<F>(power(vars.div_ceil(2))?)?)?);
            }
            let proof = add(
                add(array::<Vec<F>>(rounds)?, array::<F>(mul(3, rounds)?)?)?,
                grow::<F>(evaluations)?,
            )?;
            levels = add(
                levels,
                add(
                    proof,
                    add(
                        add(array::<F>(add(rounds, root_vars)?)?, grow::<usize>(active)?)?,
                        add(array::<LevelClaim<F>>(active)?, points)?,
                    )?,
                )?,
            )?;
            let kernel = add(
                add(proof, array::<F>(rounds)?)?,
                add(
                    array::<Range<usize>>(active)?,
                    add(
                        array::<usize>(active)?,
                        add(
                            array::<Split<'_, F>>(active)?,
                            add(array::<SecretBuf<F>>(mul(4, active)?)?, folded)?,
                        )?,
                    )?,
                )?,
            )?;
            scratch = scratch.max(add(
                kernel,
                add(array::<LevelInstance<'_, F>>(active)?, equality)?,
            )?);
        }
        // Labels::new starts at 32; dynamic group/level/round labels can grow.
        let label_len = "group/".len() + 10 + "/tree/".len() + 10 + "/round/".len() + 20;
        let labels = mul(2, grow::<u8>(label_len as u64)?)?;
        let output = add(levels, leaves)?;
        let peak = add(
            add(fractions, forests)?,
            add(
                output,
                add(
                    state_points,
                    add(
                        grow::<State<F>>(count)?,
                        add(array::<usize>(count)?, add(scratch, labels)?)?,
                    )?,
                )?,
            )?,
        )?;
        Ok(Self {
            fractions,
            peak,
            output,
            levels,
        })
    }
}

fn add(a: u64, b: u64) -> Result<u64, WindowError> {
    a.checked_add(b).ok_or(WindowError::Allocation)
}
fn mul(a: u64, b: u64) -> Result<u64, WindowError> {
    a.checked_mul(b).ok_or(WindowError::Allocation)
}
fn power(n: usize) -> Result<u64, WindowError> {
    u32::try_from(n)
        .ok()
        .and_then(|n| 1u64.checked_shl(n))
        .ok_or(WindowError::Allocation)
}
fn array<T>(n: u64) -> Result<u64, WindowError> {
    let bytes = mul(n, size_of::<T>() as u64)?;
    if usize::try_from(n).is_err() || bytes > isize::MAX as u64 {
        return Err(WindowError::Allocation);
    }
    Ok(bytes)
}
fn grow<T>(n: u64) -> Result<u64, WindowError> {
    if n == 0 {
        return Ok(0);
    }
    let minimum = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    array::<T>(mul(4, n.max(minimum))?)
}
