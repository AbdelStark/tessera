// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The claim list's emission (RFC-0007 'Schema', RFC-0026): from the public part of `D`, the
//! tables its lowering and gadget selection define, the gadget policy, and the parameter set's
//! identifier and layout. It reads nothing hidden, so INV-CL-1 holds by construction: weights and
//! channel classes never reach it.
//!
//! The parameter set is the caller's: it is `tessera-protocol`'s, downstream of the compiler
//! (01 §3).

use tessera_ir::{ClaimList, CommitmentLayout, GadgetPolicy, PROTOCOL_VERSION, ParamSetId};

use crate::{CompileError, PublicModel, Selection, lower};

/// The claim list of `model`, whose gadgets `selection` chose under `policy`, for the parameter
/// set `params` with its `layout`.
///
/// Any lowering of `model` gives the same tables, whatever its length, so `selection` may come
/// from any of them.
///
/// # Errors
/// The public lowering's errors, or `Model` when the complete advice layout at
/// `max_context` exceeds `2^32` rows or a row cannot fit one key (RFC-0046/0047).
pub fn claim_list(
    model: PublicModel<'_>,
    selection: &Selection,
    policy: &GadgetPolicy,
    layout: CommitmentLayout,
    params: ParamSetId,
) -> Result<ClaimList, CompileError> {
    let claims = ClaimList {
        protocol_version: PROTOCOL_VERSION,
        arch: model.arch.clone(),
        profile: model.profile,
        quant: model.quant.clone(),
        silu_in_exponents: model.silu_in_exponents.to_vec(),
        tables: selection.tables.clone(),
        gadget_policy: *policy,
        layout,
        params,
    };
    let maximum = lower(model, model.arch.max_context)?;
    crate::advice::allocated_rows(&maximum, selection, &claims)?;
    Ok(claims)
}
