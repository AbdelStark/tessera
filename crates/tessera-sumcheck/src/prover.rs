// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The batched sumcheck prover (RFC-0010 'Batching within a stage', 'Sumcheck messages').

pub mod window;

use core::fmt;
use core::fmt::Write as _;
use core::ops::Range;

use tessera_codec::{Secret, SecretBuf, Zeroize};
use tessera_field::PrimeField;
use tessera_poly::{PolyField, evaluate, evaluate_base};

use crate::kernel::{self, Entries, Lines, NODES, Scratch};
use crate::terms::{Plan, Term};
use crate::{MAX_DEGREE, Protocol, SumcheckError, SumcheckProof, SumcheckTranscript};

/// Which sumcheck of the protocol a proof belongs to: the prefix of its labels (RFC-0014
/// 'Transcript schedule', rows 5 and 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prefix {
    /// Stage `k`, counted from 0: labels `stage/{k}/…`.
    Stage(u32),
    /// Level `ℓ` of the leaf stage's trees, counted from 0 at the roots: labels `tree/{ℓ}/…`.
    Tree(u32),
    /// Level `ℓ` of token group `h`'s trees in a grouped leaf stage (RFC-0054): labels
    /// `group/{h}/tree/{ℓ}/…`.
    GroupTree(u32, u32),
    /// Level `ℓ` of a grouped leaf stage's table batch (RFC-0054 'The table batch'): labels
    /// `tables/tree/{ℓ}/…`.
    TableTree(u32),
    /// The merge of a grouped leaf stage's claims (RFC-0054 'The merge'): labels `merge/…`.
    Merge,
}

/// Where a leaf stage's tree levels run (RFC-0014 'Transcript schedule', row 5; RFC-0054): the
/// stage's one batch, a token group's batch, or a grouped stage's table batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TreeScope {
    /// The leaf stage in one batch: `tree/{ℓ}/…`.
    Stage,
    /// Token group `h`: `group/{h}/tree/{ℓ}/…`.
    Group(u32),
    /// The table batch: `tables/tree/{ℓ}/…`.
    Tables,
}

impl TreeScope {
    /// The prefix of level `ℓ`'s labels.
    pub fn level(self, level: u32) -> Prefix {
        match self {
            Self::Stage => Prefix::Tree(level),
            Self::Group(h) => Prefix::GroupTree(h, level),
            Self::Tables => Prefix::TableTree(level),
        }
    }
}

/// `stage/{k}` or `tree/{ℓ}`.
impl fmt::Display for Prefix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stage(k) => write!(f, "stage/{k}"),
            Self::Tree(level) => write!(f, "tree/{level}"),
            Self::GroupTree(h, level) => write!(f, "group/{h}/tree/{level}"),
            Self::TableTree(level) => write!(f, "tables/tree/{level}"),
            Self::Merge => write!(f, "merge"),
        }
    }
}

/// The entries of a table, in the prime subfield or in the field.
#[derive(Clone, Copy)]
enum Values<'a, F: PolyField> {
    Base(&'a [F::Base]),
    Ext(&'a [F]),
}

/// One factor of an instance's product: a borrowed table of `2^n` entries, the multilinear
/// polynomial `T̃(x_1, …, x_n)` equal to `T[Σ_i x_i·2^{i−1}]` on `{0, 1}^n` (RFC-0010
/// 'Multilinear extensions': `x_1` is the least significant bit of the index).
///
/// - [`Table::base`] holds entries in the prime subfield `F::Base`: witness data, such as
///   activations and committed advice. Until its first binding the prover multiplies such entries
///   in the prime subfield, which costs a fraction of an extension multiplication.
/// - [`Table::ext`] holds entries in `F`: a table such as `eq(r, ·)` at a challenge point, or a
///   table left by an earlier stage.
///
/// Both kinds mix freely within an instance. A table's evaluation at the instance's point is a
/// final evaluation of the proof, unless [`Table::public`] marks it as a public factor that the
/// verifier evaluates itself.
///
/// Table entries are secret-class values (05 §3): the debug form shows the table's field, its
/// length and whether it is public, never an entry. A table of activations or weights comes in
/// its [`Secret`], through [`Table::secret_base`] or [`Table::secret_ext`], so that only the
/// engine reads it (OBS-1); [`Table::base`] and [`Table::ext`] take public tables, such as
/// `eq(r, ·)` or a selector, and tables an engine derived itself.
pub struct Table<'a, F: PolyField> {
    values: Values<'a, F>,
    public: bool,
}

impl<F: PolyField> Clone for Table<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: PolyField> Copy for Table<'_, F> {}

impl<'a, F: PolyField> Table<'a, F> {
    /// A table over the prime subfield, whose evaluation is sent.
    pub fn base(values: &'a [F::Base]) -> Self {
        Self {
            values: Values::Base(values),
            public: false,
        }
    }

    /// A table over `F`, whose evaluation is sent.
    pub fn ext(values: &'a [F]) -> Self {
        Self {
            values: Values::Ext(values),
            public: false,
        }
    }

    /// A secret-class table over the prime subfield, such as activations or weights, whose
    /// evaluation is sent. The prover reads it in its rounds and keeps no copy: what it derives,
    /// the bound tables and the products of their entries, is cleared when the proof ends
    /// (05 §3 OBS-1, OBS-6).
    ///
    /// ```
    /// use tessera_codec::{Ascii, Secret};
    /// use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField};
    /// use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
    /// use tessera_sumcheck::{Instance, Prefix, Protocol, Table, prove_batch};
    ///
    /// let context = SessionContext {
    ///     protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
    ///     mode: Mode::NonInteractive,
    ///     params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
    ///     statement_digest: [0; 32],
    ///     session_id: [0; 16],
    /// };
    /// let fp = Goldilocks::from_u64_reduced;
    /// let activations = Secret::new((1..=8).map(fp).collect::<Vec<_>>());
    /// let selector: Vec<Goldilocks> = [1, 1, 1, 0, 0, 0, 0, 0].into_iter().map(fp).collect();
    /// let instance = Instance::new(vec![
    ///     Table::secret_base(&activations),
    ///     Table::base(&selector).public(),
    /// ]);
    /// let mut transcript = Transcript::new(&context);
    /// let proven = prove_batch::<GoldilocksExt2, _>(
    ///     &mut transcript,
    ///     Protocol::One,
    ///     Prefix::Stage(0),
    ///     2,
    ///     &[instance],
    /// )?;
    /// // The selector is a public factor: one evaluation, the activations'.
    /// assert_eq!(proven.proof().final_evals.len(), 1);
    /// # Ok::<(), tessera_sumcheck::SumcheckError>(())
    /// ```
    #[allow(clippy::disallowed_methods)]
    pub fn secret_base(values: &'a Secret<Vec<F::Base>>) -> Self {
        Self::base(values.expose_secret())
    }

    /// A secret-class table over `F`, whose evaluation is sent: as [`Table::secret_base`].
    #[allow(clippy::disallowed_methods)]
    pub fn secret_ext(values: &'a Secret<Vec<F>>) -> Self {
        Self::ext(values.expose_secret())
    }

    /// The same table as a public factor: a polynomial whose value at the instance's point the
    /// verifier computes itself, so its evaluation is left out of the final evaluations (RFC-0010
    /// 'Sumcheck messages'). It is either public, such as `eq(r, ·)` or a selector, or derived
    /// from evaluations the proof sends, such as the second factor of `X̃·X̃` or `1 − h̃` beside
    /// `h̃`.
    #[must_use]
    pub fn public(self) -> Self {
        Self {
            public: true,
            ..self
        }
    }

    /// Whether the table is a public factor.
    pub fn is_public(&self) -> bool {
        self.public
    }

    /// Whether the entries are in the prime subfield.
    pub fn is_base(&self) -> bool {
        matches!(self.values, Values::Base(_))
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        match self.values {
            Values::Base(values) => values.len(),
            Values::Ext(values) => values.len(),
        }
    }

    /// Whether the table has no entry, which no instance accepts.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The first entry, in `F`: the whole table of an instance on no variable.
    fn single(&self) -> F {
        match self.values {
            Values::Base(values) => values.first().map_or(F::ZERO, |&b| F::from(b)),
            Values::Ext(values) => values.first().copied().unwrap_or(F::ZERO),
        }
    }

    /// The entries, as the term loop reads them.
    fn entries(&self) -> Entries<'a, F> {
        match self.values {
            Values::Base(values) => Entries::Base(values),
            Values::Ext(values) => Entries::Ext(values),
        }
    }
}

/// Shows the field, length and kind, never an entry (05 §3).
impl<F: PolyField> fmt::Debug for Table<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Table")
            .field("field", &if self.is_base() { "base" } else { "extension" })
            .field("len", &self.len())
            .field("public", &self.public)
            .finish()
    }
}

/// One instance of a batched sumcheck, over tables all of `2^{n_i}` entries, whose relation's sum
/// over `{0, 1}^{n_i}` is the instance's claim. Its final evaluations are its tables' values at
/// its point, in the order of its tables, public factors left out. Its relation is either:
///
/// - **a product** ([`Instance::new`]): `Π_k T̃_k`, of at most the batch's degree tables;
/// - **a sum of weighted products** over the shared table list ([`Instance::sum_of_products`]):
///   `Σ_t w_t·Π_{k ∈ t} T̃_k`, each [`Term`] of at most the batch's degree factors. A fraction
///   tree's level is `eq·(N_0·D_1 + N_1·D_0 + λ·D_0·D_1)`, over the tables
///   `[eq (public), N_0, N_1, D_0, D_1]` with the terms `(1, {eq, N_0, D_1})`,
///   `(1, {eq, N_1, D_0})` and `(λ, {eq, D_0, D_1})`; its final evaluations are `N_0, N_1, D_0,
///   D_1`. Each table is bound once per round, however many terms read it.
#[derive(Clone)]
pub struct Instance<'a, F: PolyField> {
    tables: Vec<Table<'a, F>>,
    /// `None` for a product of every table.
    terms: Option<Vec<Term<F>>>,
}

impl<'a, F: PolyField> Instance<'a, F> {
    /// The product of `tables`, in order. [`prove_batch`] checks the shapes.
    pub fn new(tables: Vec<Table<'a, F>>) -> Self {
        Self {
            tables,
            terms: None,
        }
    }

    /// The sum of `terms`, each a weight times a product of some of `tables`, named by their
    /// positions. [`prove_batch`] checks the shapes: at least one term, each of 1 to the batch's
    /// degree factors that name tables of the list.
    pub fn sum_of_products(tables: Vec<Table<'a, F>>, terms: Vec<Term<F>>) -> Self {
        Self {
            tables,
            terms: Some(terms),
        }
    }

    /// The tables, in order.
    pub fn tables(&self) -> &[Table<'a, F>] {
        &self.tables
    }

    /// The terms of a sum-of-products instance, or `None` for a product.
    pub fn terms(&self) -> Option<&[Term<F>]> {
        self.terms.as_deref()
    }
}

/// Shows the tables' shapes and the terms' factors, never an entry or a weight (05 §3).
impl<F: PolyField> fmt::Debug for Instance<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("Instance");
        debug.field("tables", &self.tables);
        if let Some(terms) = &self.terms {
            debug.field("terms", terms);
        }
        debug.finish()
    }
}

/// A proven batch: the proof, and what the caller needs to form its next claims.
///
/// Instance `i` on `n_i` variables ends at its point, the last `n_i` challenges
/// ([`instance_point`](Self::instance_point)), where its tables take the values
/// [`instance_evals`](Self::instance_evals): each a claim on that table for the next stage or the
/// openings.
#[must_use]
pub struct ProvenBatch<F> {
    proof: SumcheckProof<F>,
    gamma: F,
    point: Vec<F>,
    vars: Vec<usize>,
    evals: Vec<Range<usize>>,
}

impl<F> ProvenBatch<F> {
    /// A batch proven elsewhere in the crate, as the tree levels are ([`crate::level`]): its
    /// messages, `γ`, its point, each instance's variables and the range of its final
    /// evaluations.
    pub(crate) fn from_parts(
        proof: SumcheckProof<F>,
        gamma: F,
        point: Vec<F>,
        vars: Vec<usize>,
        evals: Vec<Range<usize>>,
    ) -> Self {
        Self {
            proof,
            gamma,
            point,
            vars,
            evals,
        }
    }

    /// The messages.
    pub fn proof(&self) -> &SumcheckProof<F> {
        &self.proof
    }

    /// The messages, by value.
    pub fn into_proof(self) -> SumcheckProof<F> {
        self.proof
    }

    /// The batching challenge `γ`.
    pub fn gamma(&self) -> &F {
        &self.gamma
    }

    /// The challenges in round order, `r_1, …, r_{n_max}`: the batch's point, its lowest
    /// variable first.
    pub fn point(&self) -> &[F] {
        &self.point
    }

    /// The number of instances.
    pub fn instances(&self) -> usize {
        self.vars.len()
    }

    /// Instance `i`'s point, the last `n_i` challenges (front-padding), or `None` past the last
    /// instance.
    pub fn instance_point(&self, i: usize) -> Option<&[F]> {
        let vars = *self.vars.get(i)?;
        self.point.get(self.point.len() - vars..)
    }

    /// Instance `i`'s final evaluations, its tables' values at its point in the order of its
    /// tables, public factors left out; `None` past the last instance.
    pub fn instance_evals(&self, i: usize) -> Option<&[F]> {
        self.proof.final_evals.get(self.evals.get(i)?.clone())
    }
}

/// Shows the shapes: the evaluations are protocol-class data (05 §3).
impl<F> fmt::Debug for ProvenBatch<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProvenBatch")
            .field("proof", &self.proof)
            .field("vars", &self.vars)
            .finish_non_exhaustive()
    }
}

// `Degree` lists every degree up to `MAX_DEGREE`, `tessera-protocol/2`'s 10 (RFC-0013 'Degree').
const _: () = assert!(MAX_DEGREE == 10);

/// A degree the round polynomials support, `1..=MAX_DEGREE`, so that every degree selects
/// monomorphized loops.
#[derive(Clone, Copy)]
enum Degree {
    One = 1,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
}

impl Degree {
    fn new(degree: usize) -> Option<Self> {
        Some(match degree {
            1 => Self::One,
            2 => Self::Two,
            3 => Self::Three,
            4 => Self::Four,
            5 => Self::Five,
            6 => Self::Six,
            7 => Self::Seven,
            8 => Self::Eight,
            9 => Self::Nine,
            10 => Self::Ten,
            _ => return None,
        })
    }

    fn get(self) -> usize {
        self as usize
    }

    /// The sums of the product's values at the nodes of a round message, `0, 2, …, d`, in
    /// entries `0..d`.
    fn message_sums<F, B, X>(
        self,
        base: &[B],
        ext: &[X],
        pairs: Range<usize>,
        scratch: &mut Scratch<F>,
    ) -> [F; NODES]
    where
        F: PolyField,
        B: AsRef<[F::Base]>,
        X: AsRef<[F]>,
    {
        use kernel::product_sums as sums;
        match self {
            Self::One => sums::<F, B, X, 1, true>(base, ext, pairs, scratch),
            Self::Two => sums::<F, B, X, 2, true>(base, ext, pairs, scratch),
            Self::Three => sums::<F, B, X, 3, true>(base, ext, pairs, scratch),
            Self::Four => sums::<F, B, X, 4, true>(base, ext, pairs, scratch),
            Self::Five => sums::<F, B, X, 5, true>(base, ext, pairs, scratch),
            Self::Six => sums::<F, B, X, 6, true>(base, ext, pairs, scratch),
            Self::Seven => sums::<F, B, X, 7, true>(base, ext, pairs, scratch),
            Self::Eight => sums::<F, B, X, 8, true>(base, ext, pairs, scratch),
            Self::Nine => sums::<F, B, X, 9, true>(base, ext, pairs, scratch),
            Self::Ten => sums::<F, B, X, 10, true>(base, ext, pairs, scratch),
        }
    }

    /// The sums of the terms' values at the nodes of a round message, `0, 2, …, d`, in entries
    /// `0..d`.
    fn term_message_sums<F, L>(
        self,
        tables: &L,
        plan: &Plan<F>,
        pairs: Range<usize>,
        scratch: &mut Scratch<F>,
    ) -> [F; NODES]
    where
        F: PolyField,
        L: Lines<F> + ?Sized,
    {
        use kernel::term_sums as sums;
        let s = &mut scratch.terms;
        match self {
            Self::One => sums::<F, L, 1, true>(tables, plan, pairs, s),
            Self::Two => sums::<F, L, 2, true>(tables, plan, pairs, s),
            Self::Three => sums::<F, L, 3, true>(tables, plan, pairs, s),
            Self::Four => sums::<F, L, 4, true>(tables, plan, pairs, s),
            Self::Five => sums::<F, L, 5, true>(tables, plan, pairs, s),
            Self::Six => sums::<F, L, 6, true>(tables, plan, pairs, s),
            Self::Seven => sums::<F, L, 7, true>(tables, plan, pairs, s),
            Self::Eight => sums::<F, L, 8, true>(tables, plan, pairs, s),
            Self::Nine => sums::<F, L, 9, true>(tables, plan, pairs, s),
            Self::Ten => sums::<F, L, 10, true>(tables, plan, pairs, s),
        }
    }

    /// The sums of the terms' values at every node `0, 1, …, d`, in entries `0..=d`.
    fn term_node_sums<F, L>(
        self,
        tables: &L,
        plan: &Plan<F>,
        pairs: Range<usize>,
        scratch: &mut Scratch<F>,
    ) -> [F; NODES]
    where
        F: PolyField,
        L: Lines<F> + ?Sized,
    {
        use kernel::term_sums as sums;
        let s = &mut scratch.terms;
        match self {
            Self::One => sums::<F, L, 2, false>(tables, plan, pairs, s),
            Self::Two => sums::<F, L, 3, false>(tables, plan, pairs, s),
            Self::Three => sums::<F, L, 4, false>(tables, plan, pairs, s),
            Self::Four => sums::<F, L, 5, false>(tables, plan, pairs, s),
            Self::Five => sums::<F, L, 6, false>(tables, plan, pairs, s),
            Self::Six => sums::<F, L, 7, false>(tables, plan, pairs, s),
            Self::Seven => sums::<F, L, 8, false>(tables, plan, pairs, s),
            Self::Eight => sums::<F, L, 9, false>(tables, plan, pairs, s),
            Self::Nine => sums::<F, L, 10, false>(tables, plan, pairs, s),
            Self::Ten => sums::<F, L, 11, false>(tables, plan, pairs, s),
        }
    }

    /// The sums of the product's values at every node `0, 1, …, d`, in entries `0..=d`.
    fn node_sums<F, B, X>(
        self,
        base: &[B],
        ext: &[X],
        pairs: Range<usize>,
        scratch: &mut Scratch<F>,
    ) -> [F; NODES]
    where
        F: PolyField,
        B: AsRef<[F::Base]>,
        X: AsRef<[F]>,
    {
        use kernel::product_sums as sums;
        match self {
            Self::One => sums::<F, B, X, 2, false>(base, ext, pairs, scratch),
            Self::Two => sums::<F, B, X, 3, false>(base, ext, pairs, scratch),
            Self::Three => sums::<F, B, X, 4, false>(base, ext, pairs, scratch),
            Self::Four => sums::<F, B, X, 5, false>(base, ext, pairs, scratch),
            Self::Five => sums::<F, B, X, 6, false>(base, ext, pairs, scratch),
            Self::Six => sums::<F, B, X, 7, false>(base, ext, pairs, scratch),
            Self::Seven => sums::<F, B, X, 8, false>(base, ext, pairs, scratch),
            Self::Eight => sums::<F, B, X, 9, false>(base, ext, pairs, scratch),
            Self::Nine => sums::<F, B, X, 10, false>(base, ext, pairs, scratch),
            Self::Ten => sums::<F, B, X, 11, false>(base, ext, pairs, scratch),
        }
    }
}

/// Labels built from RFC-0014's patterns into one reused buffer, so that the rounds allocate
/// nothing for them.
pub(crate) struct Labels {
    prefix: Prefix,
    text: String,
}

impl Labels {
    pub(crate) fn new(prefix: Prefix) -> Self {
        Self {
            prefix,
            text: String::with_capacity(32),
        }
    }

    fn build(&mut self, suffix: fmt::Arguments<'_>) -> &str {
        self.text.clear();
        // Writing to a `String` cannot fail.
        let _ = write!(self.text, "{}/{suffix}", self.prefix);
        &self.text
    }

    /// `<prefix>/gamma`.
    fn gamma(&mut self) -> &str {
        self.build(format_args!("gamma"))
    }

    /// `<prefix>/round/{j}`, round `j` counted from 0.
    pub(crate) fn round(&mut self, j: usize) -> &str {
        self.build(format_args!("round/{j}"))
    }

    /// `<prefix>/r/{j}`, round `j` counted from 0.
    pub(crate) fn challenge(&mut self, j: usize) -> &str {
        self.build(format_args!("r/{j}"))
    }

    /// `<prefix>/evals`.
    pub(crate) fn evals(&mut self) -> &str {
        self.build(format_args!("evals"))
    }
}

/// The number of variables of a valid instance, or the first shape check it fails, in the order
/// of [`SumcheckError`].
fn instance_vars<F: PolyField>(
    instance: &Instance<'_, F>,
    index: usize,
    degree: usize,
) -> Result<usize, SumcheckError> {
    let tables = instance.tables();
    let Some(first) = tables.first() else {
        return Err(SumcheckError::NoTables { instance: index });
    };
    match instance.terms() {
        None if tables.len() > degree => {
            return Err(SumcheckError::TooManyTables {
                instance: index,
                tables: tables.len(),
                degree,
            });
        }
        None => {}
        Some([]) => return Err(SumcheckError::NoTerms { instance: index }),
        Some(terms) => {
            for (term, t) in terms.iter().enumerate() {
                let factors = t.factors();
                if factors.is_empty() || factors.len() > degree {
                    return Err(SumcheckError::TermFactors {
                        instance: index,
                        term,
                        factors: factors.len(),
                        degree,
                    });
                }
                if let Some(&factor) = factors.iter().find(|&&k| k >= tables.len()) {
                    return Err(SumcheckError::TermFactor {
                        instance: index,
                        term,
                        factor,
                        tables: tables.len(),
                    });
                }
            }
        }
    }
    let expected = first.len();
    for (table, t) in tables.iter().enumerate() {
        let len = t.len();
        if !len.is_power_of_two() {
            return Err(SumcheckError::TableLength {
                instance: index,
                table,
                len,
            });
        }
        if len != expected {
            return Err(SumcheckError::TableSizes {
                instance: index,
                table,
                len,
                expected,
            });
        }
    }
    Ok(expected.trailing_zeros() as usize)
}

/// One instance while the rounds run.
struct State<'a, F: PolyField> {
    tables: &'a [Table<'a, F>],
    /// The tables over the prime subfield, unbound, in order.
    base: Vec<&'a [F::Base]>,
    /// The tables over `F`, unbound, in order.
    ext: Vec<&'a [F]>,
    /// For a sum-of-products instance, its plan and its tables unbound in order; `None` for a
    /// product, which `base` and `ext` serve.
    terms: Option<(Plan<F>, Vec<Entries<'a, F>>)>,
    /// Whether the rounds read each table: every table of a product, the tables some term
    /// multiplies in a sum of products. A table no round reads is never bound: its evaluation is
    /// computed once, at the instance's point.
    read: Vec<bool>,
    vars: usize,
    /// `n_max − n_i`: the padding rounds, which the batch runs first.
    pad: usize,
    /// `γ^i`.
    weight: F,
    /// `γ^i·S_i`, `S_i` the product's sum over the instance's own hypercube; used by the padding
    /// rounds only.
    weighted_sum: F,
    /// The first own round's sums at the nodes `0, 1, …, d`, computed with `S_i` from the
    /// unbound tables of a padded instance; unused otherwise.
    first: [F; NODES],
    /// The tables, in order, as the rounds bind them: empty until the first own round binds
    /// them; for an instance on no variable, its tables' single entries.
    bound: Vec<SecretBuf<F>>,
}

/// The bound tables clear themselves; `first`, the instance's own sums, which no message carries
/// alone, is cleared here (05 §3 OBS-6).
impl<F: PolyField> Drop for State<'_, F> {
    fn drop(&mut self) {
        self.first.zeroize();
    }
}

impl<'a, F: PolyField> State<'a, F> {
    /// Prepares instance `i`, of `vars` variables in a batch of `n_max`, with weight `γ^i`:
    /// splits its tables by field, reserves its bound tables, and for a padded instance computes
    /// `S_i` from its tables, never from a claim.
    fn new(
        instance: &'a Instance<'a, F>,
        vars: usize,
        n_max: usize,
        weight: F,
        degree: Degree,
        scratch: &mut Scratch<F>,
    ) -> Self {
        let tables = instance.tables();
        let mut base = Vec::new();
        let mut ext = Vec::new();
        let terms = match instance.terms() {
            Some(terms) => {
                let plan = Plan::new(terms, tables.len());
                scratch.terms.fit(plan.lines.len(), plan.groups.len());
                Some((plan, tables.iter().map(Table::entries).collect::<Vec<_>>()))
            }
            None => {
                for table in tables {
                    match table.values {
                        Values::Base(values) => base.push(values),
                        Values::Ext(values) => ext.push(values),
                    }
                }
                None
            }
        };
        let read: Vec<bool> = match &terms {
            Some((plan, _)) => (0..tables.len()).map(|k| plan.lines.contains(&k)).collect(),
            None => vec![true; tables.len()],
        };
        let bound = if vars == 0 {
            tables
                .iter()
                .map(|table| SecretBuf::filled(table.single(), 1))
                .collect()
        } else {
            let half = 1 << (vars - 1);
            read.iter()
                .map(|&read| match read {
                    true => SecretBuf::with_capacity(half),
                    false => SecretBuf::new(),
                })
                .collect()
        };
        let pad = n_max - vars;
        let mut first = [F::ZERO; NODES];
        let mut weighted_sum = F::ZERO;
        if pad > 0 {
            // S_i: the product of the single entries, or p(0) + p(1) of the first own round,
            // which reads the tables unbound whatever round it is: compute it now, at every
            // node, in one pass.
            let sum = match (&terms, instance.terms()) {
                (_, Some(terms)) if vars == 0 => terms.iter().fold(F::ZERO, |sum, term| {
                    let product = term.factors().iter().fold(F::ONE, |product, &k| {
                        product * tables.get(k).map_or(F::ZERO, Table::single)
                    });
                    sum + *term.weight() * product
                }),
                _ if vars == 0 => tables
                    .iter()
                    .fold(F::ONE, |product, table| product * table.single()),
                (Some((plan, entries)), _) => {
                    let pairs = 0..1 << (vars - 1);
                    first = degree.term_node_sums(entries.as_slice(), plan, pairs, scratch);
                    first[0] + first[1]
                }
                (None, _) => {
                    first = degree.node_sums(&base, &ext, 0..1 << (vars - 1), scratch);
                    first[0] + first[1]
                }
            };
            weighted_sum = weight * sum;
        }
        Self {
            tables,
            base,
            ext,
            terms,
            read,
            vars,
            pad,
            weight,
            weighted_sum,
            first,
            bound,
        }
    }

    /// Adds `γ^i` times the instance's round polynomial at `0, 2, …, d` into `message`, for round
    /// `j` counted from 0.
    fn add_round(&self, j: usize, degree: Degree, message: &mut [F], scratch: &mut Scratch<F>) {
        if j < self.pad {
            // The padded product is constant in x_1, …, x_pad, so with x_{j+1} = z and the
            // later padding variables summed, it is 2^{pad − j − 1}·S_i at every z.
            let value = self
                .weighted_sum
                .mul_by_base(padding_scale(self.pad - 1 - j));
            for entry in message.iter_mut() {
                *entry += value;
            }
            return;
        }
        let sums = if j > self.pad {
            // Own round j − pad: the tables have 2^{vars − (j − pad)} entries left.
            let pairs = 0..1 << (self.vars - (j - self.pad) - 1);
            match &self.terms {
                Some((plan, _)) => {
                    degree.term_message_sums(self.bound.as_slice(), plan, pairs, scratch)
                }
                None => degree.message_sums(&[] as &[&[F::Base]], &self.bound, pairs, scratch),
            }
        } else if self.pad > 0 {
            // The first own round, computed at every node by `new`: drop p(1).
            let mut sums = [F::ZERO; NODES];
            sums[0] = self.first[0];
            sums[1..NODES - 1].copy_from_slice(&self.first[2..]);
            sums
        } else {
            let pairs = 0..1 << (self.vars - 1);
            match &self.terms {
                Some((plan, entries)) => {
                    degree.term_message_sums(entries.as_slice(), plan, pairs, scratch)
                }
                None => degree.message_sums(&self.base, &self.ext, pairs, scratch),
            }
        };
        for (entry, sum) in message.iter_mut().zip(sums) {
            *entry += self.weight * sum;
        }
    }

    /// Binds the instance's lowest unbound variable to `r` after round `j`, if the round was one
    /// of its own: the first own round binds the borrowed tables that the rounds read into the
    /// reserved ones, later rounds bind in place.
    fn bind(&mut self, j: usize, r: F) {
        let tables = self.tables.iter().zip(&mut self.bound).zip(&self.read);
        if j == self.pad {
            for ((table, bound), _) in tables.filter(|(_, read)| **read) {
                match table.values {
                    Values::Base(values) => kernel::bind_base(values, r, bound),
                    Values::Ext(values) => kernel::bind_ext(values, r, bound),
                }
            }
        } else if j > self.pad {
            for ((_, bound), _) in tables.filter(|(_, read)| **read) {
                kernel::bind_in_place(bound, r);
            }
        }
    }

    /// Appends the tables' values at the instance's point `point`, public factors left out: the
    /// bound tables' single entries, and the others evaluated there.
    fn final_evals(&self, point: &[F], out: &mut Vec<F>) {
        for ((table, bound), &read) in self.tables.iter().zip(&self.bound).zip(&self.read) {
            if table.public {
                continue;
            }
            let value = match (read || self.vars == 0, table.values) {
                (true, _) => bound.first().copied(),
                // A table of 2^n entries at n coordinates: the evaluation cannot fail.
                (false, Values::Base(values)) => evaluate_base(values, point).ok(),
                (false, Values::Ext(values)) => evaluate(values, point).ok(),
            };
            out.push(value.unwrap_or(F::ZERO));
        }
    }
}

/// Proves a batched sumcheck of degree `degree` over `instances`, under the labels of `prefix`,
/// on `transcript`, in `protocol` (RFC-0010 'Batching within a stage', 'Sumcheck messages';
/// RFC-0014 'Transcript schedule').
///
/// The degree is at most the bound of `protocol` at `prefix`,
/// [`protocol.max_degree(prefix)`](Protocol::max_degree): 10 at a reduction stage of
/// `tessera-protocol/2`, 7 everywhere else (RFC-0013 'Masked extensions' 'Degree'). The protocol
/// changes nothing else: a batch of a degree both protocols allow has the same messages in both.
///
/// Instance `i`, counted from 0 in the order given, on `n_i` variables, enters with weight
/// `γ^i`. The batched polynomial is `Σ_i γ^i·P_i`, where `P_i` is instance `i`'s product
/// front-padded to `n_max = max n_i` variables: constant in the lowest `n_max − n_i`, which the
/// first rounds bind, so its own variables take the last `n_i` challenges. Its sum is
/// `Σ_i γ^i·2^{n_max − n_i}·c_i`, `c_i` the instance's sum over its own hypercube; the prover
/// never needs the claims, and takes none.
///
/// On the transcript, in this order:
///
/// 1. `γ` is squeezed under `<prefix>/gamma`. The claims must already be bound by the transcript.
/// 2. Round `j`, from 0 to `n_max − 1`, absorbs the round polynomial's values `p(0), p(2), …,
///    p(d)` as TCE `Vec<F>` under `<prefix>/round/{j}`, then squeezes `r_j` under
///    `<prefix>/r/{j}`. An instance's padding round contributes `2^{n_max − n_i − j − 1}·S_i` at
///    every node, `S_i` computed from its tables; its own rounds read its tables, the first one
///    unbound, in the prime subfield for tables over it.
/// 3. The final evaluations, instance by instance and table by table, public factors left out,
///    are absorbed as one TCE `Vec<F>` under `<prefix>/evals`.
///
/// Nothing is absorbed or squeezed after that: a caller with more to derive, such as a tree
/// level's `ρ`, continues on the transcript.
///
/// # Errors
///
/// A [`SumcheckError`] when the degree is outside `1..=protocol.max_degree(prefix)`
/// ([`SumcheckError::Degree`]), there is no instance, an instance has no table, a product has more tables than the degree, a sum of products has no
/// term or a term with no factor, more factors than the degree or a factor that names no table,
/// or a table's length is not a power of two or differs from its instance's first table's. The
/// inputs are checked before the transcript moves.
pub fn prove_batch<F, T>(
    transcript: &mut T,
    protocol: Protocol,
    prefix: Prefix,
    degree: usize,
    instances: &[Instance<'_, F>],
) -> Result<ProvenBatch<F>, SumcheckError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    let (nodes, vars) = check_batch(protocol.max_degree(prefix), degree, instances)?;
    let mut labels = Labels::new(prefix);
    let gamma = transcript.challenge(labels.gamma());
    Ok(prove_checked(
        transcript, labels, nodes, vars, instances, gamma,
    ))
}

/// [`prove_batch`] with `γ` already squeezed under `<prefix>/gamma`, for a caller that squeezes
/// more challenges between `γ` and the first round, such as a tree level's `λ`, and builds its
/// instances from them (RFC-0014 'Transcript schedule', row 5). The tree levels run the
/// split-`eq` kernel of [`crate::level`] instead; this is the reference its tests compare it to.
#[cfg(test)]
pub(crate) fn prove_batch_after_gamma<F, T>(
    transcript: &mut T,
    prefix: Prefix,
    degree: usize,
    instances: &[Instance<'_, F>],
    gamma: F,
) -> Result<ProvenBatch<F>, SumcheckError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    let (nodes, vars) = check_batch(Protocol::One.max_degree(prefix), degree, instances)?;
    Ok(prove_checked(
        transcript,
        Labels::new(prefix),
        nodes,
        vars,
        instances,
        gamma,
    ))
}

/// [`prove_batch`] with every instance's weight 1 and no `γ`, for a batch whose weights its
/// tables already carry, as RFC-0054's merge carries `β^h·γ_h^k` in its `W_i`: rounds and final
/// evaluations only, under `<prefix>/round/{j}`, `<prefix>/r/{j}` and `<prefix>/evals`. The
/// result's `γ` is 1. Its degree is at most 7, the merge's bound in either protocol.
pub(crate) fn prove_batch_unweighted<F, T>(
    transcript: &mut T,
    prefix: Prefix,
    degree: usize,
    instances: &[Instance<'_, F>],
) -> Result<ProvenBatch<F>, SumcheckError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    let (nodes, vars) = check_batch(Protocol::One.max_degree(prefix), degree, instances)?;
    Ok(prove_checked(
        transcript,
        Labels::new(prefix),
        nodes,
        vars,
        instances,
        F::ONE,
    ))
}

/// The degree's loops and each instance's number of variables, or the first check the batch
/// fails, in the order of [`SumcheckError`]: first the degree, between 1 and `max`, the bound of
/// the batch's protocol and kind ([`Protocol::max_degree`]).
fn check_batch<F: PolyField>(
    max: usize,
    degree: usize,
    instances: &[Instance<'_, F>],
) -> Result<(Degree, Vec<usize>), SumcheckError> {
    let nodes = (degree <= max)
        .then(|| Degree::new(degree))
        .flatten()
        .ok_or(SumcheckError::Degree { degree, max })?;
    if instances.is_empty() {
        return Err(SumcheckError::NoInstances);
    }
    let vars = instances
        .iter()
        .enumerate()
        .map(|(i, instance)| instance_vars(instance, i, degree))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((nodes, vars))
}

/// The rounds and the final evaluations of a checked batch, from its `γ` on.
fn prove_checked<F, T>(
    transcript: &mut T,
    mut labels: Labels,
    nodes: Degree,
    vars: Vec<usize>,
    instances: &[Instance<'_, F>],
    gamma: F,
) -> ProvenBatch<F>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    let n_max = vars.iter().copied().max().unwrap_or(0);
    let mut scratch = Scratch::new();
    let mut weight = F::ONE;
    let mut states = Vec::with_capacity(instances.len());
    for (instance, &n) in instances.iter().zip(&vars) {
        states.push(State::new(instance, n, n_max, weight, nodes, &mut scratch));
        weight *= gamma;
    }

    let mut rounds = Vec::with_capacity(n_max);
    let mut point = Vec::with_capacity(n_max);
    for j in 0..n_max {
        let mut message = vec![F::ZERO; nodes.get()];
        for state in &states {
            state.add_round(j, nodes, &mut message, &mut scratch);
        }
        transcript.absorb_message(labels.round(j), &message);
        let r = transcript.challenge(labels.challenge(j));
        for state in &mut states {
            state.bind(j, r);
        }
        rounds.push(message);
        point.push(r);
    }

    let mut final_evals = Vec::new();
    let mut evals = Vec::with_capacity(states.len());
    for state in &states {
        let start = final_evals.len();
        state.final_evals(&point[n_max - state.vars..], &mut final_evals);
        evals.push(start..final_evals.len());
    }
    transcript.absorb_message(labels.evals(), &final_evals);
    ProvenBatch {
        proof: SumcheckProof {
            rounds,
            final_evals,
        },
        gamma,
        point,
        vars,
        evals,
    }
}

/// `2^exponent` in the prime subfield, `exponent ≤ 62`: the scale of a padding round. A unit test
/// replaces it by 1, to show that the verifier rejects a prover that omits it; the replacement
/// exists only in the crate's own test build.
pub(crate) fn padding_scale<B: PrimeField>(exponent: usize) -> B {
    #[cfg(test)]
    if tests::OMIT_SCALING.with(core::cell::Cell::get) {
        return B::ONE;
    }
    B::from_u64_reduced(1 << exponent)
}

#[cfg(test)]
mod tests {
    //! A prover that omits the scale `2^{n_max − n_i − j − 1}` of its padding rounds produces
    //! proofs that the independent verifier, `tv_core::sumcheck::verify_batch`, rejects
    //! (RFC-0010 'Testing Strategy'). The test-only path is [`OMIT_SCALING`], read by
    //! [`padding_scale`]: it changes nothing else.

    use core::cell::Cell;

    use tessera_codec::Ascii;
    use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
    use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
    use tv_core::sumcheck as tv;

    use super::{Instance, Prefix, Protocol, Table, prove_batch};
    use crate::SumcheckProof;

    thread_local! {
        /// Whether this thread's prover omits the padding scale.
        pub(super) static OMIT_SCALING: Cell<bool> = const { Cell::new(false) };
    }

    type Fp = Goldilocks;
    type Fp2 = GoldilocksExt2;

    fn tv_fp(x: Fp) -> tv_field::Fp {
        tv_field::Fp::from_canonical(x.to_canonical_u64()).expect("canonical")
    }

    fn tv_fp2(x: Fp2) -> tv_field::Fp2 {
        let [c0, c1] = x.coeffs();
        tv_field::Fp2::new(tv_fp(c0), tv_fp(c1))
    }

    /// The batch of 3, 7 and 12 variables, products of 3, 2 and 3 tables over `Fp`, proven by a
    /// prover that omits the padding scale or not, and checked by the verifier.
    fn prove_and_verify(omit: bool) -> Result<tv::Batch, tv::SumcheckError> {
        let mut seed = 0x53_0201u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            Fp::from_u64_reduced(seed >> 1)
        };
        let shapes = [(3usize, 3usize), (7, 2), (12, 3)];
        let tables: Vec<Vec<Vec<Fp>>> = shapes
            .iter()
            .map(|&(vars, k)| {
                (0..k)
                    .map(|_| (0..1 << vars).map(|_| next()).collect())
                    .collect()
            })
            .collect();
        let instances: Vec<Instance<'_, Fp2>> = tables
            .iter()
            .map(|factors| Instance::new(factors.iter().map(|t| Table::base(t)).collect()))
            .collect();
        let context = SessionContext {
            protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
            mode: Mode::Live,
            params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
            statement_digest: [3; 32],
            session_id: [9; 16],
        };
        OMIT_SCALING.with(|omits| omits.set(omit));
        let proven = prove_batch(
            &mut Transcript::new(&context),
            Protocol::One,
            Prefix::Stage(2),
            3,
            &instances,
        );
        OMIT_SCALING.with(|omits| omits.set(false));
        let SumcheckProof {
            rounds,
            final_evals,
        } = proven.expect("a well-formed batch").into_proof();

        let claims: Vec<tv::Instance> = tables
            .iter()
            .zip(shapes)
            .map(|(factors, (vars, k))| tv::Instance {
                claim: (0..1 << vars).fold(tv_field::Fp2::ZERO, |sum, x| {
                    sum + factors.iter().fold(tv_field::Fp2::ONE, |p, t| {
                        p * tv_field::Fp2::from(tv_fp(t[x]))
                    })
                }),
                vars,
                evals: k,
            })
            .collect();
        let proof = tv::SumcheckProof {
            rounds: rounds
                .iter()
                .map(|r| r.iter().copied().map(tv_fp2).collect())
                .collect(),
            final_evals: final_evals.into_iter().map(tv_fp2).collect(),
        };
        use tv_hash::transcript as t;
        let context = t::SessionContext {
            protocol: t::Ascii::new("tessera-protocol/1").expect("ASCII"),
            mode: t::Mode::Live,
            params: t::ParamSetId(t::Ascii::new("test-small/1").expect("ASCII")),
            statement_digest: [3; 32],
            session_id: [9; 16],
        };
        let product = |_: usize, _: &[tv_field::Fp2], evals: &[tv_field::Fp2]| {
            Some(evals.iter().fold(tv_field::Fp2::ONE, |p, e| p * *e))
        };
        tv::verify_batch(
            &mut t::Transcript::new(&context),
            tv::Prefix::Stage(2),
            3,
            &claims,
            &proof,
            product,
        )
    }

    /// `Σ_t Π_k (a_k + z·Δ_k)` at `nodes`, from the definition, each entry lifted to `Fp2`.
    fn naive(tables: &[Vec<Fp2>], pairs: core::ops::Range<usize>, nodes: &[u64]) -> Vec<Fp2> {
        nodes
            .iter()
            .map(|&z| {
                let z = Fp2::from(Fp::from_u64_reduced(z));
                pairs.clone().fold(Fp2::ZERO, |sum, t| {
                    sum + tables.iter().fold(Fp2::ONE, |product, table| {
                        let (a, b) = (table[2 * t], table[2 * t + 1]);
                        product * (a + z * (b - a))
                    })
                })
            })
            .collect()
    }

    /// Every loop of the kernel, through the degree dispatch, against the definition: products of
    /// 1 to `d` tables over `Fp`, over `Fp2` and of both, at the nodes of a round message and at
    /// every node, over ranges of pairs that span several blocks or none; and the sums over the
    /// two halves of a range add up to the sum over the range, as a parallel backend splitting
    /// it would need.
    #[test]
    fn product_sums_match_their_definition_on_any_range() {
        use super::{Degree, Scratch};
        let mut seed = 0x53_0202u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            seed >> 1
        };
        let mut scratch = Scratch::<Fp2>::new();
        let entries = 1 << 10;
        for d in 1..=crate::MAX_DEGREE {
            let degree = Degree::new(d).expect("a degree");
            for k in 1..=d {
                for kinds in 0..3 {
                    let base_at = |t: usize| match kinds {
                        0 => true,
                        1 => false,
                        _ => t.is_multiple_of(2),
                    };
                    let mut lifted = Vec::new();
                    let mut base: Vec<Vec<Fp>> = Vec::new();
                    let mut ext: Vec<Vec<Fp2>> = Vec::new();
                    for t in 0..k {
                        if base_at(t) {
                            let table: Vec<Fp> =
                                (0..entries).map(|_| Fp::from_u64_reduced(next())).collect();
                            lifted.push(table.iter().map(|&b| Fp2::from(b)).collect());
                            base.push(table);
                        } else {
                            let table: Vec<Fp2> = (0..entries)
                                .map(|_| {
                                    let c0 = Fp::from_u64_reduced(next());
                                    Fp2::new(c0, Fp::from_u64_reduced(next()))
                                })
                                .collect();
                            lifted.push(table.clone());
                            ext.push(table);
                        }
                    }
                    let message_nodes: Vec<u64> = core::iter::once(0).chain(2..=d as u64).collect();
                    let all_nodes: Vec<u64> = (0..=d as u64).collect();
                    for pairs in [0..entries / 2, 3..400, 200..201, 7..7, 130..300] {
                        let mid = (pairs.start + pairs.end) / 2;
                        for (nodes, all) in [(&message_nodes, false), (&all_nodes, true)] {
                            let mut sums = |range: core::ops::Range<usize>| {
                                if all {
                                    degree.node_sums(&base, &ext, range, &mut scratch)
                                } else {
                                    degree.message_sums(&base, &ext, range, &mut scratch)
                                }
                            };
                            let whole = sums(pairs.clone());
                            let (low, high) = (sums(pairs.start..mid), sums(mid..pairs.end));
                            let expected = naive(&lifted, pairs.clone(), nodes);
                            let at = format!("d = {d}, k = {k}, kinds {kinds}, {pairs:?}, {all}");
                            assert!(whole[..nodes.len()] == expected[..], "{at}");
                            assert!(whole[nodes.len()..].iter().all(|&x| x == Fp2::ZERO), "{at}");
                            for ((w, l), h) in whole.iter().zip(low).zip(high) {
                                assert!(*w == l + h, "{at}: the halves add up");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_prover_that_omits_the_padding_scale_is_rejected() {
        assert!(
            prove_and_verify(false).is_ok(),
            "the honest prover is accepted"
        );
        assert_eq!(
            prove_and_verify(true),
            Err(tv::SumcheckError::FinalMismatch { round: 12 })
        );
    }
}
