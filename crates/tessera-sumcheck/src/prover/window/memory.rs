// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public allocation model for the actual grid/fold lifetimes, excluding the source's tensors.

use super::{MAX_WINDOW_VARS, WindowError};

/// Geometry needed before any witness or claim point is loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowShape {
    /// Own variables.
    pub vars: usize,
    /// Source operands, including public and unused operands.
    pub tables: usize,
    /// Maximum factors in a product, excluding the factored EQ.
    pub delta: usize,
    /// Whether this instance may be windowed. Folded matmul operands stay resident.
    pub windowable: bool,
}

/// Engine storage, excluding the source's resident/regenerated witness and caller-owned inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowMemory {
    /// Global rounds answered from grids.
    pub cut: usize,
    /// Number of P3b regenerations.
    pub fold_passes: usize,
    /// Conservative peak live bytes, including grid construction scratch, split public EQ
    /// weights, clearing folded tables, round messages and per-instance metadata.
    pub working_bytes: u64,
}

fn allocation() -> WindowError {
    WindowError::Allocation
}
fn add(a: u64, b: u64) -> Result<u64, WindowError> {
    a.checked_add(b).ok_or_else(allocation)
}
fn mul(a: u64, b: u64) -> Result<u64, WindowError> {
    a.checked_mul(b).ok_or_else(allocation)
}
fn power(bits: usize) -> Result<u64, WindowError> {
    1u64.checked_shl(bits as u32).ok_or_else(allocation)
}
fn split_weights(bits: usize) -> Result<u64, WindowError> {
    add(power(bits / 2)?, power(bits - bits / 2)?)
}

impl WindowMemory {
    /// Charge a particular cut. Tables stay at their allocated capacity while binding; a
    /// contracted grid is released before the next grid or final folded tables are allocated.
    /// All arithmetic is checked; impossible geometry never wraps into a smaller charge.
    pub fn for_cut(
        shapes: &[WindowShape],
        cut: usize,
        element_bytes: usize,
    ) -> Result<Self, WindowError> {
        let rounds = shapes.iter().map(|s| s.vars).max().unwrap_or(0);
        if shapes.is_empty()
            || element_bytes == 0
            || rounds >= usize::BITS as usize
            || cut > 0 && cut >= rounds
            || shapes
                .iter()
                .any(|s| s.tables == 0 || s.delta == 0 || s.delta > 7)
        {
            return Err(WindowError::Shape("invalid window memory geometry"));
        }
        let active = |shape: &WindowShape| shape.windowable && cut > rounds - shape.vars;
        let first_padding = shapes
            .iter()
            .filter(|s| active(s))
            .map(|s| rounds - s.vars)
            .min();
        let first_boundary = first_padding.map_or(0, |p| cut.min(p + MAX_WINDOW_VARS));
        let mut retained = vec![0u64; shapes.len()];
        let mut live = 0;
        let mut peak = 0;
        let mut metadata = 16_384u64;
        for shape in shapes {
            metadata = add(metadata, 256)?;
            metadata = add(
                metadata,
                add(mul(shape.vars as u64, 256)?, mul(shape.tables as u64, 128)?)?,
            )?;
        }
        let mut boundary = 0;
        loop {
            let next = if boundary == 0 {
                first_boundary
            } else {
                (boundary + MAX_WINDOW_VARS).min(cut)
            };
            for (index, shape) in shapes.iter().enumerate() {
                if boundary != 0 && !active(shape) {
                    continue;
                }
                let padding = rounds - shape.vars;
                let bound = boundary.saturating_sub(padding);
                let end = next.saturating_sub(padding);
                let window = end - bound;
                let remaining = shape.vars - bound;
                let resident = !active(shape) || boundary == cut;
                let (target, scratch) = if resident {
                    (
                        mul(shape.tables as u64, power(remaining)?)?,
                        add(split_weights(bound)?, split_weights(remaining)?)?,
                    )
                } else {
                    let grid = (shape.delta as u64 + 1)
                        .checked_pow(window as u32)
                        .ok_or_else(allocation)?;
                    (
                        grid,
                        add(
                            mul(shape.tables as u64, grid)?,
                            add(split_weights(bound)?, split_weights(remaining - window)?)?,
                        )?,
                    )
                };
                let scratch = add(scratch, power(window.max(bound.min(MAX_WINDOW_VARS)))?)?;
                live -= retained[index];
                peak = peak.max(add(add(live, target)?, scratch)?);
                retained[index] = target;
                live = add(live, target)?;
            }
            peak = peak.max(live);
            if boundary == cut || first_boundary == 0 {
                break;
            }
            boundary = next;
        }
        Ok(Self {
            cut,
            fold_passes: first_padding.map_or(0, |p| (cut - p).div_ceil(MAX_WINDOW_VARS)),
            working_bytes: add(mul(peak, element_bytes as u64)?, metadata)?,
        })
    }

    /// Fewest global window rounds whose actual grids, construction scratch and folded tables
    /// fit `budget`. A resident batch has cut zero and no fold pass.
    pub fn select(
        shapes: &[WindowShape],
        budget: u64,
        element_bytes: usize,
    ) -> Result<Self, WindowError> {
        let rounds = shapes.iter().map(|s| s.vars).max().unwrap_or(0);
        for cut in 0..rounds.max(1) {
            match Self::for_cut(shapes, cut, element_bytes) {
                Ok(plan) if plan.working_bytes <= budget => return Ok(plan),
                Ok(_) | Err(WindowError::Allocation) => {}
                Err(error) => return Err(error),
            }
        }
        Err(WindowError::Allocation)
    }
}
