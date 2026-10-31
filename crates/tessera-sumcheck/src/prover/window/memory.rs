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
        Self::with_window_vars(shapes, cut, element_bytes, MAX_WINDOW_VARS)
    }

    /// Charge the same complete working sets with a smaller grid width, including
    /// RFC-0054 one-round table streaming. The width is between one and eight.
    pub fn with_window_vars(
        shapes: &[WindowShape],
        cut: usize,
        element_bytes: usize,
        window_vars: usize,
    ) -> Result<Self, WindowError> {
        if !(1..=MAX_WINDOW_VARS).contains(&window_vars) {
            return Err(WindowError::Shape(
                "a grid covers one through eight own variables",
            ));
        }
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
        let first_boundary = first_padding.map_or(0, |p| cut.min(p + window_vars));
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
                (boundary + window_vars).min(cut)
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
            fold_passes: first_padding.map_or(0, |p| (cut - p).div_ceil(window_vars)),
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

/// Additional window-engine specifications, bookkeeping and message storage.
/// Add this peak to WindowMemory::working_bytes and the source/callback storage.
/// The existing field/grid reservation is preserved, including its old margins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowMetadataMemory {
    peak: u64,
    output: u64,
}
impl WindowMetadataMemory {
    /// Additional constructor/proof storage beside the field/grid reservation.
    pub fn peak_bytes(self) -> u64 {
        self.peak
    }
    /// Returned proof, point and evaluation-index heap, included in the peak.
    pub fn output_bytes(self) -> u64 {
        self.output
    }
    /// Derive from canonical WindowInstance geometry without allocating or
    /// reading any field value. `terms` bounds each instance before equal weights
    /// or repeated factors could reduce work. Table/term/factor/EQ vectors must
    /// use canonical clone/push/collect capacities; arbitrary over-reservation,
    /// sources and transcript callbacks/encodings remain separately chargeable.
    ///
    /// # Errors
    /// Shape mismatch or overflowing/unaddressable allocations, as WindowMemory.
    pub fn new<F: tessera_poly::PolyField>(
        shapes: &[WindowShape],
        terms: &[u64],
        degree: usize,
    ) -> Result<Self, WindowError> {
        use super::{State, WindowInstance, WindowTable};
        use crate::Term;
        use core::ops::Range;
        use tessera_codec::SecretBuf;
        if shapes.is_empty() || shapes.len() != terms.len() || degree == 0 || degree > 7 {
            return Err(WindowError::Shape(
                "window metadata matches canonical nonempty specs",
            ));
        }
        let n = shapes.len() as u64;
        // The consumed source descriptor array may overlap the new State array.
        // The old field planner's retained-count Vec is priced as well.
        let mut owned = add(
            growing::<WindowInstance<F>>(n)?,
            add(allocation_of::<State<F>>(n)?, allocation_of::<u64>(n)?)?,
        )?;
        let (mut max_vars, mut max_tables, mut max_radix, mut evaluations) = (0, 0, 0, 0);
        for (shape, &count) in shapes.iter().zip(terms) {
            if shape.vars >= usize::BITS as usize
                || shape.tables == 0
                || shape.delta == 0
                || shape.delta > degree
                || count == 0
            {
                return Err(WindowError::Shape(
                    "window counts cover each table and term",
                ));
            }
            let (v, t) = (shape.vars as u64, shape.tables as u64);
            max_vars = max_vars.max(v);
            max_tables = max_tables.max(t);
            max_radix = max_radix.max(shape.delta as u64 + 1);
            evaluations = add(evaluations, t)?; // public columns may only reduce actual sends
            owned = add(
                owned,
                add(
                    growing::<WindowTable>(t)?,
                    add(
                        growing::<Term<F>>(count)?,
                        add(
                            separate_vectors::<usize>(mul(count, degree as u64)?, count)?,
                            growing::<Option<F>>(v)?,
                        )?,
                    )?,
                )?,
            )?;
            // Folded-table containers persist; the private true-sum buffer is one element.
            owned = add(
                owned,
                add(growing::<SecretBuf<F>>(t)?, allocation_of::<F>(1)?)?,
            )?;
        }
        let output = add(
            add(
                allocation_of::<Vec<F>>(max_vars)?,
                allocation_of::<F>(mul(max_vars, degree as u64)?)?,
            )?,
            add(
                add(allocation_of::<F>(max_vars)?, growing::<F>(evaluations)?)?,
                add(growing::<usize>(n)?, allocation_of::<Range<usize>>(n)?)?,
            )?,
        )?;
        // One prepare's column containers and Weights::new's temporary Option point.
        let prepare = add(
            growing::<SecretBuf<F>>(max_tables)?,
            allocation_of::<Option<F>>(max_vars)?,
        )?;
        // message(): returned values, grid/resident line, Lagrange basis, and suffix EQ.
        // Charge the message EQ independently of constructor order across instances.
        let message = add(
            allocation_of::<F>(degree as u64)?,
            add(
                allocation_of::<F>(max_radix)?,
                add(
                    growing::<F>(max_radix)?,
                    add(
                        allocation_of::<F>(max_tables)?,
                        allocation_of::<F>(split_weights(max_vars as usize)?)?,
                    )?,
                )?,
            )?,
        )?;
        let prefix = "group/".len() + 10 + "/tree/".len() + 10;
        let labels = growing::<u8>((prefix + "/challenge/".len() + 20) as u64)?;
        Ok(Self {
            peak: add(owned, add(output, add(labels, prepare.max(message))?)?)?,
            output,
        })
    }
}
fn allocation_of<T>(n: u64) -> Result<u64, WindowError> {
    let bytes = mul(n, size_of::<T>() as u64)?;
    if usize::try_from(n).is_err() || bytes > isize::MAX as u64 {
        return Err(WindowError::Allocation);
    }
    Ok(bytes)
}
fn growing<T>(n: u64) -> Result<u64, WindowError> {
    if n == 0 {
        return Ok(0);
    }
    let min = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    allocation_of::<T>(mul(4, n.max(min))?)
}
fn separate_vectors<T>(n: u64, vectors: u64) -> Result<u64, WindowError> {
    allocation_of::<T>(mul(4, add(n, mul(4, vectors)?)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metadata_allocations_obey_address_and_product_limits() {
        let measured = allocation_counter::measure(|| {
            assert_eq!(
                allocation_of::<u64>(isize::MAX as u64 / 8 + 1),
                Err(WindowError::Allocation)
            );
            assert_eq!(mul(u64::MAX, 2), Err(WindowError::Allocation));
            assert_eq!(growing::<u64>(u64::MAX), Err(WindowError::Allocation));
            assert_eq!(
                separate_vectors::<usize>(1, u64::MAX),
                Err(WindowError::Allocation)
            );
        });
        assert_eq!(measured.count_total, 0);
    }
}
