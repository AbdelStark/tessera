// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Original-source greedy boundary polynomials (RFC-0009/0017, proposed RFC-0053).
//!
//! These reductions expose every original graph/PCS obligation. They neither allocate a
//! runtime certificate layout nor accept a proof. In particular, a carry-source evaluation
//! is individually opened, not replaced by an unchecked value for a shifted OneHot.

use super::{GadgetError, Plan};
use crate::lowering::TensorId;
use crate::selectors::lt;
use crate::sumcheck::{Instance, eq};
use crate::validation::GadgetId;
use tv_field::{Fp, Fp2};

/// Which exact original source a native final value must bind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Actual LM-head output; reduce it through the graph, never treat it as advice.
    Logits(TensorId),
    /// Actual committed model OneHot.
    OneHot(TensorId),
    /// Actual committed model token input.
    Tokens(TensorId),
    /// Certificate maximum on the model time domain; not a model node/auxiliary.
    Maximum(GadgetId),
}

/// One original-source evaluation, still owed at precisely this point.
#[derive(Clone, PartialEq, Eq)]
pub struct Claim {
    /// Identity independently derived from the model/certificate.
    pub source: Source,
    /// Natural LSB-first source coordinates.
    pub point: Vec<Fp2>,
    /// Native final-message value, fixed before PCS Gamma.
    pub value: Fp2,
}

/// Polynomial value and ALL original-source obligations, including repeated ones.
#[must_use = "check the batch equality and every original graph/PCS source claim"]
pub struct Final {
    /// Value used in the single mixed sumcheck's final equality.
    pub value: Fp2,
    /// No member may be dropped, including compensating carry evaluations.
    pub claims: Vec<Claim>,
}

/// Native relation kind; every descriptor is derived by the verifier, never decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Range key as a linear functional of original model sources.
    Key,
    /// Selected-logit equality and every inactive/padded maximum zero.
    Selected,
    /// Weighted OneHot equality against the actual token input.
    Tokens,
}

/// Complete public source identities and dimensions for one nonempty output request.
#[derive(Clone, Copy, Debug)]
pub struct Boundary {
    gadget: GadgetId,
    n: u32,
    vocab: u32,
    prompt: u32,
    nv: usize,
    nt: usize,
    no: usize,
    range_bits: u8,
    logits: TensorId,
    onehot: TensorId,
    tokens: TensorId,
}

/// Fixed incoming lookup claim, or zero-check point, before stage batching.
pub struct Check<'a> {
    boundary: Boundary,
    kind: Kind,
    target: &'a [Fp2],
    claim: Fp2,
}

fn copy(point: &[Fp2]) -> Result<Vec<Fp2>, GadgetError> {
    let mut v = Vec::new();
    v.try_reserve_exact(point.len())
        .map_err(|_| GadgetError::Memory)?;
    v.extend_from_slice(point);
    Ok(v)
}

fn range_bits(lo: i128, hi: i128) -> Option<u8> {
    if lo > hi || lo <= -(1i128 << 28) || hi >= 1i128 << 28 {
        return None;
    }
    let span = u128::try_from(hi.checked_sub(lo)?).ok()?;
    u8::try_from(u128::BITS.saturating_sub(span.leading_zeros()).max(1)).ok()
}

impl Plan<'_> {
    /// Derive actual sources and a physical Range width from re-derived public intervals.
    /// No supplied/observed logit bound enters this factory (SP-7).
    ///
    /// # Errors
    /// `Internal` for an invalid own request/graph or logits outside RFC-0009's strict box.
    pub fn greedy_boundary(&self, gadget: GadgetId, prompt: u32) -> Result<Boundary, GadgetError> {
        let outputs = self.argmax_outputs(gadget, prompt)?;
        let first = self.graph.nodes().first().ok_or(GadgetError::Internal)?;
        let last = self.graph.nodes().last().ok_or(GadgetError::Internal)?;
        let onehot = *first.outputs().first().ok_or(GadgetError::Internal)?;
        let tokens = *first.inputs().first().ok_or(GadgetError::Internal)?;
        let logits = *last.outputs().first().ok_or(GadgetError::Internal)?;
        let index = usize::try_from(logits.0).map_err(|_| GadgetError::Internal)?;
        let (lo, hi) = self
            .selected
            .ranges
            .get(index)
            .and_then(|r| r.bounds())
            .ok_or(GadgetError::Internal)?;
        let n = self.graph.length();
        Ok(Boundary {
            gadget,
            n,
            prompt,
            logits,
            onehot,
            tokens,
            vocab: outputs.argmax().vocab(),
            nv: outputs
                .argmax()
                .variables()
                .checked_sub(outputs.argmax().row_variables())
                .ok_or(GadgetError::Internal)?,
            nt: usize::try_from(u32::BITS.saturating_sub(n.saturating_sub(1).leading_zeros()))
                .map_err(|_| GadgetError::Internal)?,
            no: outputs.argmax().row_variables(),
            range_bits: range_bits(lo, hi).ok_or(GadgetError::Internal)?,
        })
    }
}

impl Boundary {
    /// Model vocabulary-plus-time variables, not a duplicated tensor-product domain.
    pub const fn variables(self) -> usize {
        self.nv.saturating_add(self.nt)
    }
    /// Public-span-derived Range width; the logical Argmax predicate remains 29 bits.
    pub const fn range_bits(self) -> u8 {
        self.range_bits
    }
    /// Degree of the complete native polynomial, including public factors.
    pub const fn degree(self, kind: Kind) -> usize {
        match kind {
            Kind::Selected => 3,
            Kind::Key | Kind::Tokens => 2,
        }
    }
    /// Raw final values; key/selected carry L, M, then every original carry/wrap H.
    pub const fn evaluations(self, kind: Kind) -> usize {
        match kind {
            Kind::Tokens => 2,
            Kind::Key | Kind::Selected => self.nt.saturating_add(3),
        }
    }
    /// Fix an earlier Range lookup's key evaluation and its natural output/vocabulary point.
    /// The caller still owes membership in the verifier-derived Range table.
    ///
    /// # Errors
    /// `Internal` for verifier-owned point geometry inconsistent with the request.
    pub fn key(self, point: &[Fp2], claim: Fp2) -> Result<Check<'_>, GadgetError> {
        if point.len() != self.nv.saturating_add(self.no) {
            return Err(GadgetError::Internal);
        }
        Ok(Check {
            boundary: self,
            kind: Kind::Key,
            target: point,
            claim,
        })
    }
    /// Fix the stage's earlier zero point (its time coordinates) for selected/token binding.
    ///
    /// # Errors
    /// `Internal` for Key or a verifier-owned point with the wrong time dimension.
    pub fn zero(self, kind: Kind, point: &[Fp2]) -> Result<Check<'_>, GadgetError> {
        if kind == Kind::Key || point.len() != self.nt {
            return Err(GadgetError::Internal);
        }
        Ok(Check {
            boundary: self,
            kind,
            target: point,
            claim: Fp2::ZERO,
        })
    }
}

// LSB recurrence for a truncated product of equality weights; constant memory.
fn prefix_eq(a: &[Fp2], b: &[Fp2], bound: u32) -> Option<Fp2> {
    if a.len() != b.len() {
        return None;
    }
    let domain = 1u64.checked_shl(u32::try_from(a.len()).ok()?)?;
    if u64::from(bound) > domain {
        return None;
    }
    let (mut below, mut total) = (Fp2::ZERO, Fp2::ONE);
    for (j, (&p, &r)) in a.iter().zip(b).enumerate() {
        let zero = (Fp2::ONE - p) * (Fp2::ONE - r);
        let one = p * r;
        below = if u64::from(bound)
            .checked_shr(u32::try_from(j).ok()?)
            .unwrap_or(0)
            & 1
            == 1
        {
            total * zero + below * one
        } else {
            below * zero
        };
        total *= zero + one;
    }
    Some(if u64::from(bound) == domain {
        total
    } else {
        below
    })
}
fn comparison(a: usize, b: usize, old: usize) -> usize {
    if a < b {
        0
    } else if a > b {
        2
    } else {
        old
    }
}
// Multilinear weight of sum(w<V) eq(point,w) sum(i<w) eq(target,i).
fn bounded_suffix(target: &[Fp2], point: &[Fp2], bound: u32) -> Option<Fp2> {
    if target.len() != point.len() {
        return None;
    }
    let domain = 1u64.checked_shl(u32::try_from(point.len()).ok()?)?;
    if u64::from(bound) > domain {
        return None;
    }
    let mut state = [[Fp2::ZERO; 3]; 3];
    *state.get_mut(1)?.get_mut(1)? = Fp2::ONE;
    for (j, (&p, &r)) in target.iter().zip(point).enumerate() {
        let limit = usize::from(
            u64::from(bound)
                .checked_shr(u32::try_from(j).ok()?)
                .unwrap_or(0)
                & 1
                != 0,
        );
        let mut next = [[Fp2::ZERO; 3]; 3];
        for (order, bounds) in state.iter().enumerate() {
            for (old_bound, &weight) in bounds.iter().enumerate() {
                for i in 0..2 {
                    for w in 0..2 {
                        let a = if i == 0 { Fp2::ONE - p } else { p };
                        let b = if w == 0 { Fp2::ONE - r } else { r };
                        *next
                            .get_mut(comparison(i, w, order))?
                            .get_mut(comparison(w, limit, old_bound))? += weight * a * b;
                    }
                }
            }
        }
        state = next;
    }
    if u64::from(bound) == domain {
        Some(state.first()?.iter().fold(Fp2::ZERO, |a, &b| a + b))
    } else {
        state.first()?.first().copied()
    }
}

impl Check<'_> {
    /// Fixed sumcheck instance; local success still owes every returned source binding.
    pub const fn instance(&self) -> Instance {
        Instance {
            claim: self.claim,
            vars: self.boundary.variables(),
            evals: self.boundary.evaluations(self.kind),
        }
    }
    /// Evaluate the native polynomial and retain each ORIGINAL source point/value.
    /// Key/selected raw order is L, M, H carry branches, wrap; tokens is H, Tokens.
    ///
    /// # Errors
    /// `Rejected { gadget }` for a wrong proof-value count; `Internal` for own point geometry;
    /// `Memory` for failed original-source reservations. This is not proof acceptance.
    pub fn evaluate(&self, point: &[Fp2], values: &[Fp2]) -> Result<Final, GadgetError> {
        let b = self.boundary;
        if point.len() != b.variables() {
            return Err(GadgetError::Internal);
        }
        if values.len() != b.evaluations(self.kind) {
            return Err(GadgetError::Rejected { gadget: b.gadget });
        }
        let v = point.get(..b.nv).ok_or(GadgetError::Internal)?;
        let time = point.get(b.nv..).ok_or(GadgetError::Internal)?;
        let delta = v.iter().fold(Fp2::ONE, |a, &r| a * (Fp2::ONE - r));
        let left = *values.first().ok_or(GadgetError::Internal)?;
        let right = *values.get(1).ok_or(GadgetError::Internal)?;
        let mut claims = Vec::new();
        claims
            .try_reserve_exact(values.len())
            .map_err(|_| GadgetError::Memory)?;
        if self.kind == Kind::Tokens {
            let mut index = Fp2::ZERO;
            for (j, &r) in v.iter().enumerate() {
                index += r * Fp2::from(Fp::reduce(
                    1u64.checked_shl(u32::try_from(j).map_err(|_| GadgetError::Internal)?)
                        .ok_or(GadgetError::Internal)?,
                ));
            }
            claims.push(Claim {
                source: Source::OneHot(b.onehot),
                point: copy(point)?,
                value: left,
            });
            claims.push(Claim {
                source: Source::Tokens(b.tokens),
                point: copy(time)?,
                value: right,
            });
            return Ok(Final {
                value: eq(self.target, time).ok_or(GadgetError::Internal)?
                    * (index * left - delta * right),
                claims,
            });
        }
        claims.push(Claim {
            source: Source::Logits(b.logits),
            point: copy(point)?,
            value: left,
        });
        claims.push(Claim {
            source: Source::Maximum(b.gadget),
            point: copy(time)?,
            value: right,
        });
        let (mut shifted, mut carry) = (Fp2::ZERO, Fp2::ONE);
        for (k, &r) in time.iter().enumerate() {
            let value = *values
                .get(k.saturating_add(2))
                .ok_or(GadgetError::Internal)?;
            let mut source = copy(point)?;
            for j in 0..k {
                *source
                    .get_mut(b.nv.saturating_add(j))
                    .ok_or(GadgetError::Internal)? = Fp2::ZERO;
            }
            *source
                .get_mut(b.nv.saturating_add(k))
                .ok_or(GadgetError::Internal)? = Fp2::ONE;
            shifted += carry * (Fp2::ONE - r) * value;
            carry *= r;
            claims.push(Claim {
                source: Source::OneHot(b.onehot),
                point: source,
                value,
            });
        }
        let wrap = *values.last().ok_or(GadgetError::Internal)?;
        let mut source = copy(point)?;
        for r in source.get_mut(b.nv..).ok_or(GadgetError::Internal)? {
            *r = Fp2::ZERO;
        }
        shifted += carry * wrap;
        claims.push(Claim {
            source: Source::OneHot(b.onehot),
            point: source,
            value: wrap,
        });
        let value = match self.kind {
            Kind::Key => {
                let target_v = self.target.get(..b.nv).ok_or(GadgetError::Internal)?;
                let target_o = self.target.get(b.nv..).ok_or(GadgetError::Internal)?;
                let row = super::argmax_outputs::kernel(
                    target_o,
                    time,
                    b.prompt.checked_sub(1).ok_or(GadgetError::Internal)?,
                    b.n.checked_sub(b.prompt).ok_or(GadgetError::Internal)?,
                )
                .ok_or(GadgetError::Internal)?;
                row * (lt(target_v, u64::from(b.vocab)).ok_or(GadgetError::Internal)?
                    * delta
                    * right
                    - prefix_eq(target_v, v, b.vocab).ok_or(GadgetError::Internal)? * left
                    - bounded_suffix(target_v, v, b.vocab).ok_or(GadgetError::Internal)? * shifted)
            }
            Kind::Selected => {
                let active = prefix_eq(
                    self.target,
                    time,
                    b.n.checked_sub(1).ok_or(GadgetError::Internal)?,
                )
                .ok_or(GadgetError::Internal)?
                    - prefix_eq(
                        self.target,
                        time,
                        b.prompt.checked_sub(1).ok_or(GadgetError::Internal)?,
                    )
                    .ok_or(GadgetError::Internal)?;
                active * lt(v, u64::from(b.vocab)).ok_or(GadgetError::Internal)? * shifted * left
                    - eq(self.target, time).ok_or(GadgetError::Internal)? * delta * right
            }
            Kind::Tokens => return Err(GadgetError::Internal),
        };
        Ok(Final { value, claims })
    }
}

#[cfg(test)]
mod tests {
    use super::range_bits;
    #[test]
    fn only_public_strict_logit_boxes_choose_complete_physical_ranges() {
        for (lo, hi, bits) in [
            (0, 0, 1),
            (-1, 0, 1),
            (-128, 127, 8),
            (-2048, 2047, 12),
            (-(1 << 28) + 1, (1 << 28) - 1, 29),
        ] {
            assert_eq!(range_bits(lo, hi), Some(bits));
        }
        for (lo, hi) in [
            (1, 0),
            (-(1 << 28), 0),
            (0, 1 << 28),
            (i128::MIN, i128::MAX),
        ] {
            assert_eq!(range_bits(lo, hi), None);
        }
    }
}
