// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The claim list's emission (RFC-0007 'Schema', RFC-0026): from the public part of `D`, the
//! tables its lowering and gadget selection define, the gadget policy, and the parameter set's
//! identifier and layout. It reads nothing hidden, so INV-CL-1 holds by construction: weights and
//! channel classes never reach it.
//!
//! The parameter set is the caller's: it is `tessera-protocol`'s, downstream of the compiler
//! (01 §3).

use tessera_ir::{
    ClaimList, CommitmentLayout, GadgetPolicy, PROTOCOL_VERSION, PROTOCOL_VERSION_2, ParamSetId,
};
use tessera_pcs::{PcsError, WeightLayout, WeightShapes};

use crate::{CompileError, PublicModel, Selection, lower};

/// The claim list of `model`, whose gadgets `selection` chose under `policy`, for the parameter
/// set `params` with its `layout` in `tessera-protocol/1` ([`PROTOCOL_VERSION`]).
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
    claim_list_for_protocol(PROTOCOL_VERSION, model, selection, policy, layout, params)
}

/// The claim list of `model` at `protocol_version` ([`PROTOCOL_VERSION`] or
/// [`PROTOCOL_VERSION_2`]), whose gadgets `selection` chose under `policy`, for the parameter set
/// `params` with its `layout` (RFC-0007 'Schema', RFC-0012 'Layout', #564).
///
/// In `tessera-protocol/2` ([`PROTOCOL_VERSION_2`]), every committed weight matrix (`m_T > 0`) must
/// have message width `k_T ≥ k_W` so that its masked rate `(k_T + P_W) / n_T` stays within the
/// parameter set's `(k_W + P_W) / n_W`.
///
/// # Errors
///
/// - [`CompileError::Usage`] if `protocol_version` is neither [`PROTOCOL_VERSION`] nor
///   [`PROTOCOL_VERSION_2`], or if `layout.weights` is invalid (`log2_inv_rate == 0`, nominal
///   codeword `n_W > 2^32`, or in `tessera-protocol/2` `mask_coeffs == 0` or `k_W + P_W ≥ n_W`);
/// - the public lowering's errors;
/// - [`CompileError::Model`] (`ERR-QUANT-ARCH`) when the complete advice layout at `max_context`
///   exceeds `2^32` rows, a row cannot fit one key (RFC-0046/0047), a weight matrix's codeword
///   `n_T = k_T · 2^{log2_inv_rate}` exceeds `2^32` (RFC-0040), or in `tessera-protocol/2` a
///   committed weight matrix has `k_T < k_W` (RFC-0012 'Layout', #564).
pub fn claim_list_for_protocol(
    protocol_version: u16,
    model: PublicModel<'_>,
    selection: &Selection,
    policy: &GadgetPolicy,
    layout: CommitmentLayout,
    params: ParamSetId,
) -> Result<ClaimList, CompileError> {
    if protocol_version != PROTOCOL_VERSION && protocol_version != PROTOCOL_VERSION_2 {
        return Err(CompileError::Usage {
            rule: "protocol_version is PROTOCOL_VERSION or PROTOCOL_VERSION_2",
        });
    }
    if layout.weights.log2_inv_rate == 0
        || u32::from(layout.weights.log2_k).saturating_add(u32::from(layout.weights.log2_inv_rate))
            > 32
    {
        return Err(CompileError::Usage {
            rule: "layout.weights has inverse rate ≥ 2 and codewords of at most 2^32 symbols",
        });
    }
    let claims = ClaimList {
        protocol_version,
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
    let shapes = WeightShapes {
        layers: model.arch.n_layers,
        shapes: model.arch.weight_shapes()?,
        channels: maximum
            .classes
            .iter()
            .chain(&maximum.multipliers)
            .map(|&(w, _)| (w.ty.matrix(), w.layer))
            .collect(),
    };
    let weight_layout = WeightLayout::new(layout.weights, shapes).map_err(|err| match err {
        PcsError::TooLong { .. } => CompileError::Model {
            rule: "every weight matrix's codeword fits in at most 2^32 symbols (RFC-0040)",
        },
        _ => CompileError::Model {
            rule: "the model's weight shapes form a valid WeightLayout (RFC-0040)",
        },
    })?;
    if protocol_version == PROTOCOL_VERSION_2 {
        weight_layout.check_masked().map_err(|err| match err {
            PcsError::NarrowWeight { .. } => CompileError::Model {
                rule: "every committed weight matrix has message width k_T ≥ k_W in tessera-protocol/2 (RFC-0012, #564)",
            },
            _ => CompileError::Usage {
                rule: "layout.weights in tessera-protocol/2 has P_W > 0, codewords ≤ 2^32 and k_W + P_W < n_W (RFC-0012, #564)",
            },
        })?;
    }
    Ok(claims)
}
