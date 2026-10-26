// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The elementwise relations of RFC-0010 'Relations': `Elementwise` (`Mul`, scalar-by-row),
//! `SumSquares` and `ZeroCheck`, each a sum of weighted products of its inputs against `eq` at a
//! point, proven as one sum-of-products instance of the stage. RoPE is [`Rotate`](crate::Rotate),
//! a kernel of its own (RFC-0037).
//!
//! A relation on `n` variables whose target has the high `n − m` of them states, for a claim on
//! the target at `r`,
//!
//! ```text
//! T̃(r) = Σ_{x ∈ {0,1}^n} eq(r, x_high) · Σ_t w_t · Π_{f ∈ t} I_f(x)
//! ```
//!
//! with `x_high` the high `n − m` variables of `x`. Elementwise relations and zero-checks have
//! `m = 0`; `SumSquares` sums its row's `2^m` entries. Every input `I_f` is a table of the
//! relation's `2^n` entries or read as one ([`Input`]): an operand, whose evaluation the proof
//! sends and which leaves a claim; an operand read at each entry's RoPE partner or broadcast over
//! some variables, which leave a claim at a point the instance's point fixes; or a public table,
//! which the verifier evaluates. `eq` is a public factor too.

use core::fmt;

use tessera_codec::{Secret, SecretBuf};
use tessera_field::{Field, PrimeField};
use tessera_poly::{MAX_DEGREE, PolyField};

use crate::contract::{eq_high, eq_weights};
use crate::{Claim, Instance, SumcheckError, SumcheckTranscript, Table, Term};

/// The shape of an [`Elementwise`] relation: the variables of its tables and of its instance,
/// and how many of the lowest its target does not have.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ElementwiseShape {
    /// `n`: the variables of the relation's tables of `2^n` entries, and of its instance.
    pub vars: usize,
    /// `m`: the lowest variables, which the relation sums out and its target does not have:
    /// `SumSquares`' axis, the columns of each row; 0 for an elementwise product or a
    /// zero-check. The target, and `eq`, take the other `n − m`.
    pub summed_vars: usize,
}

impl ElementwiseShape {
    /// The shape of an elementwise relation on `vars` variables, its target's: a product, a
    /// rotation or a zero-check.
    pub fn elementwise(vars: usize) -> Self {
        Self {
            vars,
            summed_vars: 0,
        }
    }

    /// The variables of the target and of `eq`, `n − m`.
    pub fn target_vars(&self) -> usize {
        self.vars.saturating_sub(self.summed_vars)
    }
}

/// One factor of an [`Elementwise`] relation's terms, read at each vertex `x` of the relation's
/// `n` variables. The operands are over the prime subfield, flattened row-major (RFC-0010
/// 'Multilinear extensions'), and secret-class unless public (05 §3).
pub enum Input<'a, F: PolyField> {
    /// `A(x)`: an operand of `2^n` entries. The proof sends its evaluation at the instance's
    /// point `r'`, a claim on `Ã` there.
    Operand(&'a [F::Base]),
    /// `X(x ⊕ 2^b)`: an operand of `2^n` entries read at each entry's partner, the entry whose
    /// index differs in bit `b` (variable `x_{b+1}`), as RoPE pairs `x_i` with `x_{i+m}`. Since
    /// `X̃_⊕(r') = X̃(r'_1, …, 1 − r'_{b+1}, …, r'_n)`, its evaluation is a claim on `X̃` at `r'`
    /// with coordinate `b` replaced by `1 − r'_{b+1}`.
    Partner {
        /// `X`, of `2^n` entries.
        operand: &'a [F::Base],
        /// `b`, from 0 for the lowest variable: the bit of the index that the pairs differ in.
        bit: usize,
    },
    /// `R(x_{k+1}, …, x_{k+j})`: an operand of `2^j` entries on the relation's variables from
    /// `k` on, constant in the others. A row's scalar broadcast over its row has `k = n − j`
    /// (`Mul`'s `Trailing`, such as `R_q` times the rows of `PV`), and a channel's over the rows
    /// `k = 0` (`Leading`). Its evaluation is a claim on `R̃` at the coordinates `k` to
    /// `k + j − 1` of `r'`.
    Broadcast {
        /// `R`, of `2^j` entries, `k + j ≤ n`.
        operand: &'a [F::Base],
        /// `k`: the low variables it is constant in.
        low_vars: usize,
    },
    /// `K(x)`: a public table of `2^n` entries, such as RoPE's constants or a selector. The
    /// verifier computes its value at the instance's point (RFC-0010 'Sumcheck messages'), so
    /// the proof sends nothing for it and it leaves no claim.
    Public(&'a [F::Base]),
}

impl<F: PolyField> Clone for Input<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: PolyField> Copy for Input<'_, F> {}

impl<'a, F: PolyField> Input<'a, F> {
    /// [`Input::Operand`] on a secret-class operand, in its [`Secret`]: the relation reads it
    /// in its sumcheck, and what it derives clears itself when it drops (05 §3 OBS-1, OBS-6).
    #[allow(clippy::disallowed_methods)]
    pub fn secret(operand: &'a Secret<Vec<F::Base>>) -> Self {
        Self::Operand(operand.expose_secret())
    }

    /// [`Input::Partner`] on a secret-class operand, as [`Input::secret`].
    #[allow(clippy::disallowed_methods)]
    pub fn secret_partner(operand: &'a Secret<Vec<F::Base>>, bit: usize) -> Self {
        Self::Partner {
            operand: operand.expose_secret(),
            bit,
        }
    }

    /// [`Input::Broadcast`] on a secret-class operand, as [`Input::secret`].
    #[allow(clippy::disallowed_methods)]
    pub fn secret_broadcast(operand: &'a Secret<Vec<F::Base>>, low_vars: usize) -> Self {
        Self::Broadcast {
            operand: operand.expose_secret(),
            low_vars,
        }
    }

    /// Whether the input is a public table, whose evaluation the proof does not send.
    pub fn is_public(&self) -> bool {
        matches!(self, Self::Public(_))
    }

    fn values(&self) -> &'a [F::Base] {
        match *self {
            Self::Operand(values)
            | Self::Partner {
                operand: values, ..
            }
            | Self::Broadcast {
                operand: values, ..
            }
            | Self::Public(values) => values,
        }
    }

    /// Checks that the input fits a relation on `vars` variables, `2^vars` countable.
    fn check(&self, input: usize, vars: usize) -> Result<(), SumcheckError> {
        let len = self.values().len();
        let fits = match *self {
            Self::Operand(_) | Self::Partner { .. } | Self::Public(_) => len == 1 << vars,
            Self::Broadcast { low_vars, .. } => {
                len.is_power_of_two()
                    && low_vars.saturating_add(len.trailing_zeros() as usize) <= vars
            }
        };
        if !fits {
            return Err(SumcheckError::InputLength { input, len, vars });
        }
        match *self {
            Self::Partner { bit, .. } if bit >= vars => {
                Err(SumcheckError::PartnerBit { input, bit, vars })
            }
            _ => Ok(()),
        }
    }
}

/// Shows the kind, the length and the variables, never an entry (05 §3).
impl<F: PolyField> fmt::Debug for Input<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let len = self.values().len();
        match *self {
            Self::Operand(_) => f.debug_struct("Operand").field("len", &len).finish(),
            Self::Partner { bit, .. } => f
                .debug_struct("Partner")
                .field("len", &len)
                .field("bit", &bit)
                .finish(),
            Self::Broadcast { low_vars, .. } => f
                .debug_struct("Broadcast")
                .field("len", &len)
                .field("low_vars", &low_vars)
                .finish(),
            Self::Public(_) => f.debug_struct("Public").field("len", &len).finish(),
        }
    }
}

/// An `Elementwise`, `SumSquares` or `ZeroCheck` relation of RFC-0010 'Relations': for a claim
/// on its target at `r`, of `n − m` coordinates,
///
/// ```text
/// T̃(r) = Σ_{x ∈ {0,1}^n} eq(r, x_high) · Σ_t w_t · Π_{f ∈ t} I_f(x),
/// ```
///
/// `x_high` the high `n − m` variables, over [`Input`]s `I_f` and [`Term`]s whose factors name
/// them by position. Its degree in each variable is the most factors a term has, plus one for
/// `eq` when the target has variables.
///
/// - **`Mul`** ([`mul`](Self::mul)): `Z̃(r) = Σ_x eq(r, x)·Ã(x)·B̃(x)`, either factor an operand,
///   a broadcast operand or a public table: elementwise products, scalar-by-row products and
///   products by constants. Degree 3.
/// - **`SumSquares`** ([`sum_squares`](Self::sum_squares)): `Ũ(r) = Σ_x eq(r, x_high)·X̃(x)²`, a
///   row's squares summed over its `2^m` columns. Degree 3.
/// - **`ZeroCheck`** ([`booleanity`](Self::booleanity), [`padding`](Self::padding)):
///   `0 = Σ_x eq(z, x)·h̃(x)·(1 − h̃(x))`, `OneHot`'s booleanity, and
///   `0 = Σ_x eq(z, x)·(1 − ψ̃(x))·w̃(x)` with `ψ` the public row selector (RFC-0035). `z` is the
///   stage's zero-check point ([`zero_check_point`]), and the claim 0. Degree 3.
///
/// Others, such as `SoftmaxRow`'s masked leaves or SHA-256's `Maj`, come from
/// [`new`](Self::new). Three steps prove one, the first two for the stage driver around
/// [`prove_batch`](crate::prove_batch), as for the matmul relations:
///
/// 1. [`tables`](Self::tables) builds the instance's tables from the input claim's point, never
///    its value: `eq(r, ·)`, and the partner and broadcast operands laid out at the relation's
///    `2^n` entries; [`ElementwiseTables::instance`] lends them to [`prove_batch`] as one instance
///    among the stage's.
/// 2. [`prove_batch`] runs the stage's sumcheck, of a degree at least [`degree`](Self::degree).
/// 3. [`ElementwiseTables::claims`] maps the instance's point and evaluations to one [`Claim`] per
///    input the proof evaluates, in input order.
///
/// The operands are secret-class values (05 §3): the debug form shows the shape and the inputs'
/// kinds and lengths only.
///
/// [`prove_batch`]: crate::prove_batch
#[derive(Clone)]
pub struct Elementwise<'a, F: PolyField> {
    shape: ElementwiseShape,
    inputs: Vec<Input<'a, F>>,
    terms: Vec<Term<F>>,
    degree: usize,
}

impl<'a, F: PolyField> Elementwise<'a, F> {
    /// The relation `Σ_x eq(r, x_high)·Σ_t w_t·Π_{f ∈ t} I_f(x)` of the shape `shape`, over
    /// `inputs`, with `terms` whose factors name inputs by position. A factor may repeat, as
    /// `X̃²`, and a term may have no factor when the target has variables: `Σ_x eq(r, x)·w = w`.
    ///
    /// # Errors
    ///
    /// In this order: [`SumcheckError::RelationShape`] when `m > n`, or `2^n` is not a count of
    /// a `usize`; input by input, [`SumcheckError::InputLength`] when it does not fit `n`
    /// variables, then [`SumcheckError::PartnerBit`] when it flips a bit outside them; term by
    /// term, [`SumcheckError::TermInput`] when a factor names no input, then
    /// [`SumcheckError::Degree`] when the term's degree, its factors plus `eq`'s, is outside
    /// `1..=MAX_DEGREE`, the largest any protocol allows; and [`SumcheckError::Degree`] with
    /// degree 0 when there is no term. The batch it joins refuses a degree above its own
    /// protocol's bound ([`prove_batch`](crate::prove_batch)).
    pub fn new(
        shape: ElementwiseShape,
        inputs: Vec<Input<'a, F>>,
        terms: Vec<Term<F>>,
    ) -> Result<Self, SumcheckError> {
        let ElementwiseShape { vars, summed_vars } = shape;
        if summed_vars > vars || vars >= usize::BITS as usize {
            return Err(SumcheckError::RelationShape { vars, summed_vars });
        }
        for (input, value) in inputs.iter().enumerate() {
            value.check(input, vars)?;
        }
        let with_eq = usize::from(shape.target_vars() > 0);
        let mut degree = 0;
        for (term, value) in terms.iter().enumerate() {
            if let Some(factor) = value.factors().iter().position(|&f| f >= inputs.len()) {
                return Err(SumcheckError::TermInput {
                    term,
                    factor,
                    inputs: inputs.len(),
                });
            }
            let term_degree = value.factors().len() + with_eq;
            if !(1..=MAX_DEGREE).contains(&term_degree) {
                return Err(SumcheckError::Degree {
                    degree: term_degree,
                    max: MAX_DEGREE,
                });
            }
            degree = degree.max(term_degree);
        }
        if terms.is_empty() {
            return Err(SumcheckError::Degree {
                degree: 0,
                max: MAX_DEGREE,
            });
        }
        Ok(Self {
            shape,
            inputs,
            terms,
            degree,
        })
    }

    /// `Mul`: `Z̃(r) = Σ_x eq(r, x)·Ã(x)·B̃(x)` on `vars` variables, `a` and `b` any inputs: two
    /// operands, an operand and a broadcast one (scalar-by-row), or an operand and a public
    /// table of constants.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn mul(a: Input<'a, F>, b: Input<'a, F>, vars: usize) -> Result<Self, SumcheckError> {
        Self::new(
            ElementwiseShape::elementwise(vars),
            vec![a, b],
            vec![Term::new(F::ONE, vec![0, 1])],
        )
    }

    /// `SumSquares`: `Ũ(r) = Σ_x eq(r, x_high)·X̃(x)²`, each row of `x`'s `2^m` lowest-indexed
    /// columns summed into one entry of `U`, with `m = shape.summed_vars`.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn sum_squares(x: Input<'a, F>, shape: ElementwiseShape) -> Result<Self, SumcheckError> {
        Self::new(shape, vec![x], vec![Term::new(F::ONE, vec![0, 0])])
    }

    /// `ZeroCheck`, `OneHot`'s booleanity: `0 = Σ_x eq(z, x)·h̃(x)·(1 − h̃(x))`, written
    /// `h − h²`, on `vars` variables. The claim is 0, at the zero-check point `z`.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn booleanity(h: Input<'a, F>, vars: usize) -> Result<Self, SumcheckError> {
        Self::new(
            ElementwiseShape::elementwise(vars),
            vec![h],
            vec![Term::new(F::ONE, vec![0]), Term::new(-F::ONE, vec![0, 0])],
        )
    }

    /// `ZeroCheck`, a padding zero-check of RFC-0035: `0 = Σ_x eq(z, x)·(1 − ψ̃(x))·w̃(x)`,
    /// written `w − ψ·w`, on `vars` variables, `psi` the public row selector at every entry. Its
    /// inputs are `[w, psi]`. The claim is 0, at the zero-check point `z`.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn padding(
        w: Input<'a, F>,
        psi: &'a [F::Base],
        vars: usize,
    ) -> Result<Self, SumcheckError> {
        Self::new(
            ElementwiseShape::elementwise(vars),
            vec![w, Input::Public(psi)],
            vec![Term::new(F::ONE, vec![0]), Term::new(-F::ONE, vec![1, 0])],
        )
    }

    /// The shape.
    pub fn shape(&self) -> ElementwiseShape {
        self.shape
    }

    /// The inputs, in order.
    pub fn inputs(&self) -> &[Input<'a, F>] {
        &self.inputs
    }

    /// The terms, their factors naming inputs.
    pub fn terms(&self) -> &[Term<F>] {
        &self.terms
    }

    /// The degree of its sumcheck: the most factors a term has, plus one for `eq` when the
    /// target has variables. The stage's degree must be at least this.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The variables of its instance, `n`.
    pub fn vars(&self) -> usize {
        self.shape.vars
    }

    /// The number of evaluations its instance sends: one per input that is not public.
    pub fn sent(&self) -> usize {
        self.inputs
            .iter()
            .filter(|input| !input.is_public())
            .count()
    }

    /// The instance's tables for a claim on the target at `point`, its `n − m` coordinates, the
    /// lowest first: `eq(point, ·)` on the high variables, broadcast over the `m` summed ones, a
    /// public factor, when the target has variables; then each input as a table of `2^n`
    /// entries, a partner or broadcast operand laid out in a buffer that clears itself when it
    /// drops (05 §3 OBS-6), the others borrowed as they are. For a zero-check, `point` is the
    /// last `n` coordinates of the stage's zero-check point.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::PointLength`] when the point does not have `n − m` coordinates, before
    /// anything is computed.
    pub fn tables(&self, point: &[F]) -> Result<ElementwiseTables<'a, F>, SumcheckError> {
        let target_vars = self.shape.target_vars();
        if point.len() != target_vars {
            return Err(SumcheckError::PointLength {
                len: point.len(),
                expected: target_vars,
            });
        }
        let eq = match (target_vars, self.shape.summed_vars) {
            (0, _) => Vec::new(),
            (_, 0) => eq_weights(point),
            (_, summed) => eq_high(point, summed),
        };
        let vars = self.shape.vars;
        let derived = self
            .inputs
            .iter()
            .map(|input| match *input {
                Input::Partner { operand, bit } => Some(partner(operand, bit)),
                Input::Broadcast { operand, low_vars } => Some(broadcast(operand, low_vars, vars)),
                Input::Operand(_) | Input::Public(_) => None,
            })
            .collect();
        Ok(ElementwiseTables {
            relation: self.clone(),
            eq,
            derived,
        })
    }
}

/// Shows the shape, the inputs' kinds and lengths and the terms' factors, never an entry or a
/// weight (05 §3).
impl<F: PolyField> fmt::Debug for Elementwise<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Elementwise")
            .field("shape", &self.shape)
            .field("inputs", &self.inputs)
            .field("terms", &self.terms)
            .finish()
    }
}

/// `X(x ⊕ 2^bit)` at every entry: the halves of each block of `2^{bit+1}` entries swapped.
fn partner<B: PrimeField>(operand: &[B], bit: usize) -> SecretBuf<B> {
    let half = 1usize << bit;
    let mut out = SecretBuf::with_capacity(operand.len());
    for block in operand.chunks_exact(2 * half) {
        let (low, high) = block.split_at(half);
        out.extend(high.iter().copied());
        out.extend(low.iter().copied());
    }
    out
}

/// `R` laid out at `2^vars` entries: entry `x` is `R[(x >> low_vars) mod 2^j]`, each entry
/// repeated `2^low_vars` times, and the whole repeated over the variables above.
fn broadcast<B: PrimeField>(operand: &[B], low_vars: usize, vars: usize) -> SecretBuf<B> {
    let len = 1usize << vars;
    let width = 1usize << low_vars;
    let mut out = SecretBuf::with_capacity(len);
    for _ in 0..len / (operand.len() * width) {
        for &value in operand {
            out.extend(core::iter::repeat_n(value, width));
        }
    }
    out
}

/// The tables of an [`Elementwise`] relation's instance for one claim on its target, from
/// [`Elementwise::tables`]: `eq(r, ·)` when the target has variables, and its inputs at `2^n`
/// entries each.
///
/// The partner and broadcast operands are secret-class, as the operands are (05 §3): the debug
/// form shows the relation's shape and inputs' kinds only.
pub struct ElementwiseTables<'a, F: PolyField> {
    relation: Elementwise<'a, F>,
    /// `eq(r, ·)` broadcast over the summed variables, empty when the target has none.
    eq: Vec<F>,
    /// Each partner and broadcast input laid out at `2^n` entries, by input.
    derived: Vec<Option<SecretBuf<F::Base>>>,
}

impl<F: PolyField> ElementwiseTables<'_, F> {
    /// The instance for [`prove_batch`](crate::prove_batch): `eq(r, ·)` as a public factor when
    /// the target has variables, then the inputs in order, public tables as public factors. Its
    /// evaluations are those of the inputs that are not public, in order.
    ///
    /// A single term of weight 1 that multiplies every input once is the product of the tables,
    /// on the engine's product loops; any other relation is a sum of products over them.
    pub fn instance(&self) -> Instance<'_, F> {
        let relation = &self.relation;
        let with_eq = !self.eq.is_empty();
        let mut tables = Vec::with_capacity(relation.inputs.len() + usize::from(with_eq));
        if with_eq {
            tables.push(Table::ext(&self.eq).public());
        }
        for (input, derived) in relation.inputs.iter().zip(&self.derived) {
            tables.push(match (input, derived) {
                (Input::Public(values), _) => Table::base(values).public(),
                (_, Some(values)) => Table::base(values),
                (_, None) => Table::base(input.values()),
            });
        }
        if is_product(&relation.terms, relation.inputs.len()) {
            return Instance::new(tables);
        }
        let shift = usize::from(with_eq);
        let terms = relation
            .terms
            .iter()
            .map(|term| {
                let mut factors = Vec::with_capacity(term.factors().len() + shift);
                if with_eq {
                    factors.push(0);
                }
                factors.extend(term.factors().iter().map(|&f| f + shift));
                Term::new(*term.weight(), factors)
            })
            .collect();
        Instance::sum_of_products(tables, terms)
    }

    /// The claims after the sumcheck, from the instance's point `r'`, its last `n` challenges
    /// ([`ProvenBatch::instance_point`]), and its evaluations
    /// ([`ProvenBatch::instance_evals`]): one per input that is not public, in input order, at
    /// `r'` for an operand, at `r'` with coordinate `b` replaced by `1 − r'_{b+1}` for a partner,
    /// and at the coordinates `k` to `k + j − 1` of `r'` for a broadcast.
    ///
    /// # Errors
    ///
    /// In this order: [`SumcheckError::PointLength`] when the point does not have `n`
    /// coordinates, and [`SumcheckError::EvaluationCount`] when the evaluations are not one per
    /// input that is not public.
    ///
    /// [`ProvenBatch::instance_point`]: crate::ProvenBatch::instance_point
    /// [`ProvenBatch::instance_evals`]: crate::ProvenBatch::instance_evals
    pub fn claims(&self, point: &[F], evals: &[F]) -> Result<Vec<Claim<F>>, SumcheckError> {
        let relation = &self.relation;
        if point.len() != relation.shape.vars {
            return Err(SumcheckError::PointLength {
                len: point.len(),
                expected: relation.shape.vars,
            });
        }
        let sent = relation.sent();
        if evals.len() != sent {
            return Err(SumcheckError::EvaluationCount {
                len: evals.len(),
                expected: sent,
            });
        }
        let points = relation.inputs.iter().filter_map(|input| match *input {
            Input::Operand(_) => Some(point.to_vec()),
            Input::Partner { bit, .. } => {
                let mut flipped = point.to_vec();
                flipped[bit] = F::ONE - flipped[bit];
                Some(flipped)
            }
            Input::Broadcast { operand, low_vars } => {
                let own = operand.len().trailing_zeros() as usize;
                Some(point[low_vars..low_vars + own].to_vec())
            }
            Input::Public(_) => None,
        });
        Ok(points
            .zip(evals)
            .map(|(point, &value)| Claim { point, value })
            .collect())
    }
}

/// Shows the relation's shape and inputs' kinds, never an entry (05 §3).
impl<F: PolyField> fmt::Debug for ElementwiseTables<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ElementwiseTables")
            .field("relation", &self.relation)
            .finish_non_exhaustive()
    }
}

/// Whether `terms` are one term of weight 1 multiplying each of `inputs` inputs once: the
/// product of the instance's tables, `eq` among them.
fn is_product<F: Field>(terms: &[Term<F>], inputs: usize) -> bool {
    let [term] = terms else {
        return false;
    };
    let mut seen = vec![false; inputs];
    *term.weight() == F::ONE
        && term.factors().len() == inputs
        && term
            .factors()
            .iter()
            .all(|&f| !core::mem::replace(&mut seen[f], true))
}

/// The zero-check point of stage `stage`: `vars` challenges squeezed under
/// `stage/{k}/zero/{c}`, `c` from 0, one per variable of the stage's largest `ZeroCheck`
/// (RFC-0010 'Batching within a stage', RFC-0014 'Transcript schedule' row 6). A `ZeroCheck` of
/// `n_i` variables takes the last `n_i` coordinates.
///
/// The stage driver squeezes it just before the stage's `γ`, which [`prove_batch`] squeezes,
/// and after `ρ` in `tessera-protocol/2`.
///
/// [`prove_batch`]: crate::prove_batch
pub fn zero_check_point<F, T>(transcript: &mut T, stage: u32, vars: usize) -> Vec<F>
where
    T: SumcheckTranscript<F> + ?Sized,
{
    (0..vars)
        .map(|c| transcript.challenge(&format!("stage/{stage}/zero/{c}")))
        .collect()
}
