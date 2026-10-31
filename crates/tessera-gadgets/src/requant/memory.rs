// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Requant metadata construction: arithmetic tables and bounded affine forms (RFC-0018/0024).

use super::{Requant, Wide};
use crate::{Affine, Committed, Lookup, TemplateMemory, Term, Virtual};
use tessera_field::Goldilocks as F;

impl Requant {
    /// Allocation-free bound on template construction and complete Goldilocks
    /// constraint metadata for every valid Requant1/Split/Wide gadget and valid
    /// matching table specs. Arithmetic tables have no dense core. This does not
    /// validate supplied metadata or cover malformed-kind allocation, `.tables()`,
    /// scalar witnesses, borrowed specs, inline `Self` or runtime/allocator overhead.
    /// Returns `None` if a bound cannot fit the target address space.
    pub fn memory() -> Option<TemplateMemory> {
        // Wide owns a box, a singleton sign-class vector and <=4 output classes.
        // Its comparison with the expected sign kind creates one further byte.
        // One/Split own <=4 class bytes and no box.
        let template = (size_of::<Wide>() as u64).checked_add(6)?;
        // Wide bounds all forms: 4 committed, 5 virtuals, 5 lookups. Their affine
        // keys/values own <=24 coefficient/term pairs. Construction also requests
        // w_off(2), h1(2+4), doubled(3), high_key(2+4), and class(1): 18 pairs.
        // Rust 1.95's pair-sized Vec minimum capacity is 4, including 2->4 growth.
        let constraints = 4u64
            .checked_mul(size_of::<Committed>() as u64)?
            .checked_add(5u64.checked_mul(size_of::<(Virtual, Affine<F>)>() as u64)?)?
            .checked_add(5u64.checked_mul(size_of::<Lookup<F>>() as u64)?)?
            .checked_add(42u64.checked_mul(size_of::<(F, Term)>() as u64)?)?;
        if template > isize::MAX as u64 || constraints > isize::MAX as u64 {
            return None;
        }
        Some(TemplateMemory {
            template,
            constraints,
        })
    }
}
