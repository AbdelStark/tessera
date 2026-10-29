// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0050 root brackets bound to model-derived sources and lookup requests.
//! Product target claims are derived from the already fixed comparison arrays.
//! Their reductions join the existing stage batch, retaining all source debts.

use super::comparison::{Comparison, Values as ComparisonValues};
use super::root_products::Products;
use super::{Claim, GadgetError, Input, Plan};
use crate::advice::{Auxiliary, Source};
use crate::lowering::{Kind, NodeId, Op, TensorId};
use crate::selection::{Choice, Pattern};
use crate::sumcheck::{Instance, eq};
use crate::validation::{ClaimsError, GadgetId};
use tv_field::{Fp, Fp2};
use tv_tables::{TableId, TableKind};

/// A complete per-node root binding, derived only from the verifier's public graph.
#[derive(Debug)]
pub struct Root<'p> {
    choice: &'p Choice,
    pub(super) slots: Option<&'p crate::lowering::Slots>,
    gadget: GadgetId,
    products: Products,
    comparisons: [Comparison; 2],
    constants: [[u64; 15]; 2],
    sources: Vec<Source>,
    widths: Vec<u8>,
    low: usize,
    variables: usize,
    pub(super) n: u32,
    pub(super) epsilon: Fp2,
    root: TensorId,
    squares: TensorId,
    residual: TensorId,
    pub(super) features: usize,
    residual_vars: usize,
    pub(super) dim: u32,
}

/// A root leaf check with all source, lookup and nonlinear obligations retained.
#[must_use = "bind all source/lookup claims, reduce every product claim, and check root and residual padding"]
pub struct Checked<'a> {
    root: &'a Root<'a>,
    point: &'a [Fp2],
    sources: &'a [Fp2],
    keys: &'a [Fp2],
    products: [[Fp2; 15]; 2],
}

/// One canonical raw product coefficient to reduce in the stage batch.
#[derive(Clone, Copy)]
pub struct Product<'a> {
    root: &'a Root<'a>,
    target: &'a [Fp2],
    claim: Fp2,
    coefficient: usize,
    increment: bool,
}

/// A final product relation and the source claims its operands leave.
#[must_use = "check the batched relation, reduce the sum-of-squares claim and open every source"]
pub struct Final<'r, 'a> {
    root: &'r Root<'r>,
    point: &'a [Fp2],
    value: Fp2,
    rho: Fp2,
    squares: Fp2,
    root_low: &'a [Fp2],
    input_low: &'a [Fp2],
}

/// The root padding relation with its canonical output-opening obligation.
#[must_use = "check the padding relation in the stage batch and open its root claim"]
pub struct Padding<'a> {
    value: Fp2,
    claim: Claim<'a>,
}

impl Plan<'_> {
    /// Root sites in graph order. Each owns bounded, fallibly reserved metadata.
    /// An item is `Internal` for inconsistent verifier metadata, or `Memory`.
    pub fn roots(&self) -> impl Iterator<Item = Result<Root<'_>, GadgetError>> {
        self.selected
            .choices
            .iter()
            .enumerate()
            .filter(|(_, c)| c.pattern == Pattern::RsqrtRow)
            .map(|(id, c)| Root::derive(self, id, c))
    }
}

fn field(value: u64) -> Fp2 {
    Fp2::from(Fp::reduce(value))
}
fn metadata(error: ClaimsError) -> GadgetError {
    match error {
        ClaimsError::Memory => GadgetError::Memory,
        _ => GadgetError::Internal,
    }
}
fn recompose(values: &[Fp2], bits: u32) -> Result<Fp2, GadgetError> {
    let radix = field(1u64.checked_shl(bits).ok_or(GadgetError::Internal)?);
    Ok(values
        .iter()
        .rev()
        .fold(Fp2::ZERO, |out, &v| out * radix + v))
}
fn arrays(values: &[Fp2], limbs: usize) -> Result<[&[Fp2]; 6], GadgetError> {
    let mut out = [&[][..]; 6];
    for (i, slot) in out.iter_mut().enumerate() {
        let start = i.checked_mul(limbs).ok_or(GadgetError::Internal)?;
        *slot = values
            .get(start..start.checked_add(limbs).ok_or(GadgetError::Internal)?)
            .ok_or(GadgetError::Internal)?;
    }
    Ok(out)
}

impl<'p> Root<'p> {
    fn derive(plan: &'p Plan<'_>, id: usize, choice: &'p Choice) -> Result<Self, GadgetError> {
        let internal = GadgetError::Internal;
        let node = plan.graph.node(choice.node).ok_or(internal)?;
        let Op::RsqrtRow {
            kappa,
            eps_int,
            dim,
        } = *node.op()
        else {
            return Err(internal);
        };
        let &squares = node.inputs().first().ok_or(internal)?;
        let &root = node.outputs().first().ok_or(internal)?;
        let n = plan.graph.length();
        let input = plan.graph.tensor(squares).ok_or(internal)?;
        let sum = plan
            .graph
            .node(input.producer.ok_or(internal)?)
            .ok_or(internal)?;
        let &residual_source = sum.inputs().first().ok_or(internal)?;
        let residual = plan.graph.tensor(residual_source).ok_or(internal)?;
        // RFC-0050: residual_padding proves the premise S=0 on padded time rows.
        // Only then does psi*(S+e)=S+e*psi hold as a Boolean-domain/MLE identity.
        if !matches!(sum.op(), Op::SumSquares { axis: 1 })
            || !matches!(residual.kind, Kind::Advice | Kind::Lookup { .. })
            || residual.shape.dims() != [n, dim]
            || input.shape.dims() != [n]
            || plan.graph.tensor(root).ok_or(internal)?.shape.dims() != [n]
        {
            return Err(internal);
        }
        let (_, high) = plan
            .selected
            .ranges
            .get(usize::try_from(root.0).map_err(|_| internal)?)
            .and_then(crate::ranges::Interval::bounds)
            .ok_or(internal)?;
        let width = u8::try_from(
            u128::BITS
                .saturating_sub(u128::try_from(high).map_err(|_| internal)?.leading_zeros())
                .max(1),
        )
        .map_err(|_| internal)?;
        let products = Products::derive(width)?;
        let m = products.coefficients();
        let low = products.root_limbs().checked_sub(1).ok_or(internal)?;
        let constant = u128::from(dim)
            .checked_shl(u32::from(kappa).checked_mul(2).ok_or(internal)?)
            .ok_or(internal)?;
        let mut constants = [[0u64; 15]; 2];
        for (increment, digits) in constants.iter_mut().enumerate() {
            let value = constant
                .checked_add(u128::try_from(increment).map_err(|_| internal)?)
                .ok_or(internal)?;
            let shift = u32::try_from(m.checked_mul(8).ok_or(internal)?).map_err(|_| internal)?;
            if value.checked_shr(shift).ok_or(internal)? != 0 {
                return Err(internal);
            }
            for (j, digit) in digits.iter_mut().take(m).enumerate() {
                *digit = u64::try_from(
                    value
                        .checked_shr(
                            u32::try_from(j.checked_mul(8).ok_or(internal)?)
                                .map_err(|_| internal)?,
                        )
                        .ok_or(internal)?
                        & 255,
                )
                .map_err(|_| internal)?;
            }
        }
        let gadget = GadgetId(u32::try_from(id).map_err(|_| internal)?);
        let lower = crate::range_checks::root_coefficient_bounds(width, false, m, gadget)
            .map_err(metadata)?;
        let upper = crate::range_checks::root_coefficient_bounds(width, true, m, gadget)
            .map_err(metadata)?;
        let [cl, cu] = &constants;
        let comparisons = [
            Comparison::derive(
                8,
                lower.get(..m).ok_or(internal)?,
                cl.get(..m).ok_or(internal)?,
            )?,
            Comparison::derive(
                8,
                cu.get(..m).ok_or(internal)?,
                upper.get(..m).ok_or(internal)?,
            )?,
        ];
        let auxiliary = m
            .checked_mul(12)
            .and_then(|x| x.checked_add(low))
            .and_then(|x| x.checked_add(3))
            .ok_or(internal)?;
        let mut sources = Vec::new();
        sources
            .try_reserve_exact(auxiliary.checked_add(2).ok_or(internal)?)
            .map_err(|_| GadgetError::Memory)?;
        sources.extend([Source::Tensor(squares), Source::Tensor(root)]);
        for j in 0..auxiliary {
            let role = crate::advice::root_auxiliary(
                u32::try_from(low).map_err(|_| internal)?,
                u32::try_from(m).map_err(|_| internal)?,
                u32::try_from(j).map_err(|_| internal)?,
            )
            .ok_or(internal)?;
            sources.push(Source::Auxiliary(choice.node, role));
        }
        if choice.tables.len() != auxiliary.checked_add(3).ok_or(internal)? {
            return Err(internal);
        }
        let mut widths = Vec::new();
        widths
            .try_reserve_exact(choice.tables.len())
            .map_err(|_| GadgetError::Memory)?;
        for id in &choice.tables {
            let table = plan
                .selected
                .tables
                .get(usize::try_from(id.0).map_err(|_| internal)?)
                .ok_or(internal)?;
            let TableKind::Range { bits } = table.kind else {
                return Err(internal);
            };
            widths.push(bits);
        }
        let variables =
            usize::try_from(u32::BITS.saturating_sub(n.saturating_sub(1).leading_zeros()))
                .map_err(|_| internal)?;
        let features =
            usize::try_from(u32::BITS.saturating_sub(dim.saturating_sub(1).leading_zeros()))
                .map_err(|_| internal)?;
        Ok(Self {
            choice,
            slots: plan.graph.slots(),
            gadget,
            products,
            comparisons,
            constants,
            sources,
            widths,
            low,
            variables,
            n,
            epsilon: field(eps_int),
            root,
            squares,
            residual: residual_source,
            features,
            dim,
            residual_vars: variables.checked_add(features).ok_or(internal)?,
        })
    }

    pub(super) fn row_psi(&self, point: &[Fp2]) -> Result<Fp2, GadgetError> {
        match self.slots {
            Some(s) if s.lengths().len() >= 2 => crate::selectors::slot_selector(
                point,
                usize::try_from(s.slot_bits()).map_err(|_| GadgetError::Internal)?,
                s.lengths(),
            )
            .ok_or(GadgetError::Internal),
            _ => crate::selectors::lt(point, u64::from(self.n)).ok_or(GadgetError::Internal),
        }
    }

    /// The canonical rejection payload.
    pub const fn gadget(&self) -> GadgetId {
        self.gadget
    }
    /// The owner of all auxiliary roles.
    pub fn node(&self) -> NodeId {
        self.choice.node
    }
    /// The fixed source order: graph input, graph root, then RFC-0050 auxiliaries.
    pub fn sources(&self) -> &[Source] {
        &self.sources
    }
    /// Every range table id in request order, including repeated families.
    pub fn tables(&self) -> &[TableId] {
        &self.choice.tables
    }
    /// The prescribed range width of every request, including both virtual tops.
    pub fn range_bits(&self) -> &[u8] {
        &self.widths
    }

    /// Complete public byte-coefficient geometry, including narrowed virtual digits.
    pub fn product_geometry(&self) -> Products {
        self.products
    }

    /// The original residual tensor whose feature/time padding is a required premise.
    pub fn residual_source(&self) -> Source {
        Source::Tensor(self.residual)
    }

    /// Check affine lookup bindings and both complete comparisons at the leaf point.
    /// All values must already be fixed by the leaf-stage transcript. Membership
    /// is the earlier LogUp obligation, not an integer test on these evaluations.
    ///
    /// # Errors
    /// `Rejected { gadget }` for counts, nonzero Range values, wrong affine forms
    /// or comparison equations/terminals; `Internal` for the verifier's wrong point.
    pub fn check<'a>(
        &'a self,
        point: &'a [Fp2],
        sources: &'a [Fp2],
        keys: &'a [Fp2],
        values: &[Fp2],
    ) -> Result<Checked<'a>, GadgetError> {
        let internal = GadgetError::Internal;
        let reject = GadgetError::Rejected {
            gadget: self.gadget,
        };
        if point.len() != self.variables {
            return Err(internal);
        }
        if sources.len() != self.sources.len()
            || keys.len() != self.tables().len()
            || values.len() != keys.len()
            || values.iter().any(|&v| v != Fp2::ZERO)
        {
            return Err(reject);
        }
        for (i, &value) in sources.iter().enumerate().skip(1) {
            let at = if i == 1 {
                0
            } else if i < 5 {
                i.checked_sub(1).ok_or(internal)?
            } else if i < self.low.checked_add(5).ok_or(internal)? {
                i
            } else {
                i.checked_add(1).ok_or(internal)?
            };
            if keys.get(at).copied() != Some(value) {
                return Err(reject);
            }
        }
        let psi = self.row_psi(point)?;
        let u = sources.first().copied().ok_or(internal)? + self.epsilon * psi;
        if recompose(keys.get(1..5).ok_or(internal)?, 16)? != u
            || recompose(
                keys.get(5..self.low.checked_add(6).ok_or(internal)?)
                    .ok_or(internal)?,
                8,
            )? != sources.get(1).copied().ok_or(internal)?
        {
            return Err(reject);
        }
        let start = self.low.checked_add(5).ok_or(internal)?;
        let m = self.products.coefficients();
        let mut product_claims = [[Fp2::ZERO; 15]; 2];
        for (side, out) in product_claims.iter_mut().enumerate() {
            let offset = side
                .checked_mul(6)
                .and_then(|x| x.checked_mul(m))
                .and_then(|x| x.checked_add(start))
                .ok_or(internal)?;
            let a = arrays(sources.get(offset..).ok_or(internal)?, m)?;
            let (digits, carries) = if side == 0 {
                (a.first(), a.get(1))
            } else {
                (a.get(2), a.get(3))
            };
            let (mut previous, radix) = (Fp2::ZERO, field(256));
            for ((slot, &digit), &carry) in out
                .iter_mut()
                .take(m)
                .zip(digits.ok_or(internal)?.iter())
                .zip(carries.ok_or(internal)?.iter())
            {
                *slot = digit + radix * carry - previous;
                previous = carry;
            }
            let constant = self.constants.get(side).ok_or(internal)?;
            let mut c = [Fp2::ZERO; 15];
            for (slot, &byte) in c.iter_mut().zip(constant) {
                *slot = field(byte) * psi;
            }
            let (left, right) = if side == 0 {
                (out.get(..m).ok_or(internal)?, c.get(..m).ok_or(internal)?)
            } else {
                (c.get(..m).ok_or(internal)?, out.get(..m).ok_or(internal)?)
            };
            // Checked below retains these witnesses and raw product targets;
            // the other raw side is the verifier's public constant polynomial.
            let checked = self.comparisons.get(side).ok_or(internal)?.check(
                self.gadget,
                ComparisonValues {
                    point,
                    left,
                    right,
                    arrays: a,
                },
            )?;
            for claim in checked.claims() {
                if let super::comparison::Input::Witness(role) = claim.input {
                    use crate::advice::ComparisonRole as R;
                    let array = match role {
                        R::LeftDigit => 0usize,
                        R::LeftCarry => 1,
                        R::RightDigit => 2,
                        R::RightCarry => 3,
                        R::Difference => 4,
                        R::Borrow => 5,
                    };
                    let at = side
                        .checked_mul(6)
                        .and_then(|x| x.checked_add(array))
                        .and_then(|x| x.checked_mul(m))
                        .and_then(|x| x.checked_add(claim.limb))
                        .and_then(|x| x.checked_add(self.low))
                        .and_then(|x| x.checked_add(6))
                        .ok_or(internal)?;
                    if self.widths.get(at).copied() != claim.range_bits {
                        return Err(internal);
                    }
                }
            }
        }
        Ok(Checked {
            root: self,
            point,
            sources,
            keys,
            products: product_claims,
        })
    }

    /// Public metadata for the mandatory padded-root zero check.
    pub const fn padding_instance(&self) -> Instance {
        Instance {
            claim: Fp2::ZERO,
            vars: self.variables,
            evals: 1,
        }
    }
    /// Evaluate `eq(z,r)*(1-psi(r))*rho(r)` and retain its canonical root claim.
    ///
    /// # Errors
    /// `Internal` for wrong own points; `Rejected { gadget }` for a wrong count.
    pub fn padding<'a>(
        &self,
        zero: &[Fp2],
        point: &'a [Fp2],
        values: &[Fp2],
    ) -> Result<Padding<'a>, GadgetError> {
        if zero.len() != self.variables || point.len() != self.variables {
            return Err(GadgetError::Internal);
        }
        if values.len() != 1 {
            return Err(GadgetError::Rejected {
                gadget: self.gadget,
            });
        }
        let rho = values.first().copied().ok_or(GadgetError::Internal)?;
        let psi = self.row_psi(point)?;
        Ok(Padding {
            value: eq(zero, point).ok_or(GadgetError::Internal)? * (Fp2::ONE - psi) * rho,
            claim: Claim {
                input: Input {
                    source: Source::Tensor(self.root),
                    point,
                },
                value: rho,
            },
        })
    }

    /// Mandatory zero check on the upstream residual's padded time/feature entries.
    /// This discharges the padding premise of `u=S+epsilon*psi` and of SumSquares.
    pub const fn residual_padding_instance(&self) -> Instance {
        Instance {
            claim: Fp2::ZERO,
            vars: self.residual_vars,
            evals: 1,
        }
    }

    /// Evaluate the residual's zero-padding relation and retain its source claim.
    /// Its natural point lists feature coordinates first, then token coordinates.
    ///
    /// # Errors
    /// `Internal` for wrong own points; `Rejected { gadget }` for a wrong count.
    pub fn residual_padding<'a>(
        &self,
        zero: &[Fp2],
        point: &'a [Fp2],
        values: &[Fp2],
    ) -> Result<Padding<'a>, GadgetError> {
        let internal = GadgetError::Internal;
        if zero.len() != self.residual_vars || point.len() != self.residual_vars {
            return Err(internal);
        }
        if values.len() != self.residual_padding_instance().evals {
            return Err(GadgetError::Rejected {
                gadget: self.gadget,
            });
        }
        let value = values.first().copied().ok_or(internal)?;
        let (feature, row) = point.split_at_checked(self.features).ok_or(internal)?;
        let psi = crate::selectors::lt(feature, u64::from(self.dim)).ok_or(internal)?
            * self.row_psi(row)?;
        Ok(Padding {
            value: eq(zero, point).ok_or(internal)? * (Fp2::ONE - psi) * value,
            claim: Claim {
                input: Input {
                    source: Source::Tensor(self.residual),
                    point,
                },
                value,
            },
        })
    }
}

impl Checked<'_> {
    /// All source claims, including the virtual graph sum of squares.
    pub fn claims(&self) -> impl Iterator<Item = Claim<'_>> {
        self.root
            .sources
            .iter()
            .zip(self.sources)
            .map(|(&source, &value)| Claim {
                input: Input {
                    source,
                    point: self.point,
                },
                value,
            })
    }
    /// Every prescribed Range claim, in request order, still owed to LogUp.
    pub fn lookups(&self) -> impl Iterator<Item = super::requant::Lookup<'_>> {
        self.root
            .tables()
            .iter()
            .zip(self.keys)
            .map(|(&table, &key)| super::requant::Lookup {
                table,
                point: self.point,
                key,
                value: Fp2::ZERO,
            })
    }
    /// Both complete product vectors, lower first, then upper. No high limb is omitted.
    pub fn products(&self) -> impl Iterator<Item = Product<'_>> {
        self.products
            .iter()
            .enumerate()
            .flat_map(move |(side, raw)| {
                raw.iter()
                    .take(self.root.products.coefficients())
                    .enumerate()
                    .map(move |(coefficient, &claim)| Product {
                        root: self.root,
                        target: self.point,
                        claim,
                        coefficient,
                        increment: side == 1,
                    })
            })
    }
}

impl<'r> Product<'r> {
    /// Incoming claim and dimensions for this coefficient's existing stage instance.
    pub fn instance(self) -> Instance {
        Instance {
            claim: self.claim,
            vars: self.root.variables,
            evals: self.root.products.evaluations(),
        }
    }
    /// Its byte coefficient, least significant first.
    pub const fn coefficient(self) -> usize {
        self.coefficient
    }
    /// Whether this is the upper `(rho+1)^2*u` product.
    pub const fn increment(self) -> bool {
        self.increment
    }
    /// Evaluate the final polynomial and bind the operands to root/input decompositions.
    ///
    /// # Errors
    /// `Rejected { gadget }` for a wrong count; `Internal` for wrong own geometry.
    pub fn evaluate<'a>(
        self,
        point: &'a [Fp2],
        values: &'a [Fp2],
    ) -> Result<Final<'r, 'a>, GadgetError> {
        let internal = GadgetError::Internal;
        if point.len() != self.root.variables {
            return Err(internal);
        }
        if values.len() != self.root.products.evaluations() {
            return Err(GadgetError::Rejected {
                gadget: self.root.gadget,
            });
        }
        let (root_digits, input_digits) = values
            .split_at_checked(self.root.products.root_limbs())
            .ok_or(internal)?;
        let psi = self.root.row_psi(point)?;
        Ok(Final {
            root: self.root,
            point,
            value: eq(self.target, point).ok_or(internal)?
                * self
                    .root
                    .products
                    .evaluate(self.coefficient, self.increment, values)
                    .ok_or(internal)?,
            rho: recompose(root_digits, 8)?,
            squares: recompose(input_digits, 16)? - self.root.epsilon * psi,
            root_low: root_digits.get(..self.root.low).ok_or(internal)?,
            input_low: input_digits.get(..3).ok_or(internal)?,
        })
    }
}

impl<'a> Final<'_, 'a> {
    /// The value the existing stage's batched sumcheck must check.
    pub const fn value(&self) -> Fp2 {
        self.value
    }
    /// Graph sum-of-squares/root claims, then all committed low input/root limbs.
    /// The virtual top limbs are bound by recomposition, never allocated as advice.
    pub fn claims(&self) -> impl Iterator<Item = Claim<'a>> + '_ {
        let claim = move |source, value| Claim {
            input: Input {
                source,
                point: self.point,
            },
            value,
        };
        [
            claim(Source::Tensor(self.root.squares), self.squares),
            claim(Source::Tensor(self.root.root), self.rho),
        ]
        .into_iter()
        .chain((0u32..3).zip(self.input_low).map(move |(j, &v)| {
            claim(
                Source::Auxiliary(self.root.choice.node, Auxiliary::RootInput(j)),
                v,
            )
        }))
        .chain((0u32..).zip(self.root_low).map(move |(j, &v)| {
            claim(
                Source::Auxiliary(self.root.choice.node, Auxiliary::RootLow(j)),
                v,
            )
        }))
    }
}
impl<'a> Padding<'a> {
    /// The padding polynomial's final value.
    pub const fn value(&self) -> Fp2 {
        self.value
    }
    /// The canonical root source that remains to open.
    pub const fn claim(&self) -> Claim<'a> {
        self.claim
    }
}
