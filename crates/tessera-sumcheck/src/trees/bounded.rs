// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054 table streaming alongside already-resident lookup trees/forests.

use core::{fmt, mem::size_of};

use tessera_codec::SecretBuf;
use tessera_poly::PolyField;

use super::{
    FractionClaim, Kind, LevelClaim, Numerators, ProvenTreeLevel, ProvenTrees, State, Tree,
    build_levels, label, shape,
};
use crate::level::{LevelNumerators, LevelTables};
use crate::{
    FractionTreeRows, PreparedWindow, Protocol, SumcheckTranscript, Table, Term, TreeScope,
    WindowError, WindowInstance, WindowMemory, WindowPass, WindowShape, WindowSource, WindowTable,
};

/// One tree of a fixed canonical schedule, either retained or regenerated from bounded rows.
pub enum BoundedTree<'a, F: PolyField> {
    /// Existing lookup/fraction tree or forest, whose owned allocations are charged in full.
    Resident(Tree<'a, F>),
    /// A table fraction tree's immutable leaves; no complete level is retained.
    Generated(FractionTreeRows<'a, F>),
}

struct Generated<'a, F: PolyField> {
    rows: FractionTreeRows<'a, F>,
    root: (F, F),
    join: usize,
    point: Vec<F>,
    claim: Option<LevelClaim<F>>,
}

enum BatchState<'a, F: PolyField> {
    Resident(State<F>),
    Generated(Generated<'a, F>),
}

impl<F: PolyField> BatchState<'_, F> {
    fn join(&self) -> usize {
        match self {
            Self::Resident(s) => s.join,
            Self::Generated(s) => s.join,
        }
    }
    fn trees_log(&self) -> usize {
        match self {
            Self::Resident(s) => s.trees_log(),
            Self::Generated(_) => 0,
        }
    }
    fn fraction(&self) -> bool {
        match self {
            Self::Resident(s) => matches!(s.kind, Kind::Fraction(_)),
            Self::Generated(_) => true,
        }
    }
    fn point(&self) -> &[F] {
        match self {
            Self::Resident(s) => &s.point,
            Self::Generated(s) => &s.point,
        }
    }
    fn set_point(&mut self, point: Vec<F>) {
        match self {
            Self::Resident(s) => s.point = point,
            Self::Generated(s) => s.point = point,
        }
    }
    fn claim(&self, level: usize, lambda: F) -> F {
        match self {
            Self::Resident(s) => s.level_claim(level - s.join, lambda),
            Self::Generated(s) => match &s.claim {
                Some(LevelClaim::Fraction(c)) => c.numerator + lambda * c.denominator,
                _ => s.root.0 + lambda * s.root.1,
            },
        }
    }
    fn finish_level(&mut self, level: usize, claim: LevelClaim<F>, point: Vec<F>) {
        match self {
            Self::Resident(s) => {
                s.drop_level(level - s.join);
                s.point = point;
                s.claim = Some(claim);
            }
            Self::Generated(s) => {
                s.point = point;
                s.claim = Some(claim);
            }
        }
    }
    fn final_claim(self) -> Option<LevelClaim<F>> {
        match self {
            Self::Resident(s) => s.claim,
            Self::Generated(s) => s.claim,
        }
    }
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

fn fraction_bytes<F: PolyField>(tree: &super::FractionTree<F>) -> Result<u64, WindowError> {
    let mut bytes = mul(
        tree.levels.capacity() as u64,
        size_of::<super::FractionLevel<F>>() as u64,
    )?;
    for level in &tree.levels {
        bytes = add(
            bytes,
            mul(
                (level.denominators.even.capacity() + level.denominators.odd.capacity()) as u64,
                size_of::<F>() as u64,
            )?,
        )?;
        let n = match &level.numerators {
            Numerators::Unit => 0,
            Numerators::Base(n) => mul(
                (n.even.capacity() + n.odd.capacity()) as u64,
                size_of::<F::Base>() as u64,
            )?,
            Numerators::Ext(n) => mul(
                (n.even.capacity() + n.odd.capacity()) as u64,
                size_of::<F>() as u64,
            )?,
        };
        bytes = add(bytes, n)?;
    }
    Ok(bytes)
}

enum Operand<'s, 'r, F: PolyField> {
    Resident(Table<'s, F>),
    Constant {
        len: usize,
        value: F,
    },
    Generated {
        rows: &'s FractionTreeRows<'r, F>,
        depth: usize,
        part: usize,
    },
}

struct LevelSource<'s, 'r, F: PolyField> {
    operands: Vec<Operand<'s, 'r, F>>,
}

fn buffer<F: PolyField>(len: usize) -> Result<SecretBuf<F>, WindowError> {
    let mut v = SecretBuf::try_with_capacity(len).map_err(|_| allocation())?;
    v.resize(len, F::ZERO);
    Ok(v)
}

impl<F: PolyField> WindowSource<F> for LevelSource<'_, '_, F> {
    fn regenerate(&mut self, _: WindowPass) -> Result<(), WindowError> {
        Ok(())
    }
    fn resident(&self, table: usize) -> Option<Table<'_, F>> {
        match self.operands.get(table)? {
            Operand::Resident(table) => Some(*table),
            _ => None,
        }
    }
    fn read(&self, table: usize, start: usize, out: &mut [F]) -> Result<(), WindowError> {
        match self.operands.get(table).ok_or(WindowError::Source)? {
            Operand::Resident(_) => Err(WindowError::Source),
            Operand::Constant { len, value } => {
                if start.checked_add(out.len()).is_none_or(|end| end > *len) {
                    return Err(WindowError::Source);
                }
                out.fill(*value);
                Ok(())
            }
            Operand::Generated { rows, depth, part } => {
                if out.is_empty() {
                    return Ok(());
                }
                let count = rows.chunk_rows().min(out.len());
                let pair_count = count.checked_mul(2).ok_or_else(allocation)?;
                let mut n = buffer(pair_count)?;
                let mut d = buffer(pair_count)?;
                for (block, output) in out.chunks_mut(count).enumerate() {
                    let first = start
                        .checked_add(block * count)
                        .and_then(|first| first.checked_mul(2))
                        .ok_or(WindowError::Source)?;
                    let len = 2 * output.len();
                    rows.read_level(*depth, first, &mut n[..len], &mut d[..len])?;
                    let values = if *part < 2 { &n } else { &d };
                    for (at, value) in output.iter_mut().enumerate() {
                        *value = values[2 * at + (*part & 1)];
                    }
                }
                Ok(())
            }
        }
    }
}

impl<'s, 'r, F: PolyField> LevelSource<'s, 'r, F> {
    fn push(&mut self, operand: Operand<'s, 'r, F>) -> WindowTable {
        let id = self.operands.len();
        self.operands.push(operand);
        WindowTable { id, public: false }
    }
    fn instance(
        &mut self,
        state: &'s BatchState<'r, F>,
        level: usize,
        lambda: F,
    ) -> WindowInstance<F> {
        let vars = state.point().len();
        let mut tables = Vec::new();
        match state {
            BatchState::Generated(s) => {
                for part in 0..4 {
                    tables.push(self.push(Operand::Generated {
                        rows: &s.rows,
                        depth: level - s.join + 1,
                        part,
                    }));
                }
            }
            BatchState::Resident(s) => {
                let instance = s.level_instance(
                    level - s.join,
                    s.level_claim(level - s.join, lambda),
                    lambda,
                );
                match instance.tables {
                    LevelTables::Product { even, odd } => {
                        tables.push(self.push(Operand::Resident(Table::base(even))));
                        tables.push(self.push(Operand::Resident(Table::base(odd))));
                    }
                    LevelTables::Fraction {
                        numerators,
                        d_even,
                        d_odd,
                        ..
                    } => {
                        match numerators {
                            LevelNumerators::Unit => {
                                for _ in 0..2 {
                                    tables.push(self.push(Operand::Constant {
                                        len: 1usize << vars,
                                        value: F::ONE,
                                    }));
                                }
                            }
                            LevelNumerators::Base { even, odd } => {
                                tables.push(self.push(Operand::Resident(Table::base(even))));
                                tables.push(self.push(Operand::Resident(Table::base(odd))));
                            }
                            LevelNumerators::Ext { even, odd } => {
                                tables.push(self.push(Operand::Resident(Table::ext(even))));
                                tables.push(self.push(Operand::Resident(Table::ext(odd))));
                            }
                        }
                        tables.push(self.push(Operand::Resident(Table::ext(d_even))));
                        tables.push(self.push(Operand::Resident(Table::ext(d_odd))));
                    }
                }
            }
        }
        let terms = if state.fraction() {
            vec![
                Term::new(F::ONE, vec![0, 3]),
                Term::new(F::ONE, vec![1, 2]),
                Term::new(lambda, vec![2, 3]),
            ]
        } else {
            vec![Term::new(F::ONE, vec![0, 1])]
        };
        WindowInstance {
            vars,
            tables,
            terms,
            eq: state.point().iter().copied().map(Some).collect(),
            windowed: matches!(state, BatchState::Generated(_)),
        }
    }
}

/// A canonical batch whose source roots are prepared before the caller absorbs them.
///
/// The allowance covers owned resident trees, output/metadata, regeneration scratch and
/// every level's simultaneous grid/fold state. `source_bytes` charges the caller-owned source
/// backing storage that coexists with the batch; it must come from the owner's checked plan.
/// No model forward or witness regeneration occurs here.
pub struct PreparedTreeBatch<'a, F: PolyField> {
    scope: TreeScope,
    states: Vec<BatchState<'a, F>>,
    height: usize,
    allowance: u64,
    working_bytes: u64,
    cuts: Vec<usize>,
}

impl<F: PolyField> fmt::Debug for PreparedTreeBatch<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedTreeBatch")
            .field("scope", &self.scope)
            .field("trees", &self.states.len())
            .field("height", &self.height)
            .field("allowance", &self.allowance)
            .finish_non_exhaustive()
    }
}

impl<'a, F: PolyField> PreparedTreeBatch<'a, F> {
    /// Validate geometry/complete fixed allowance, then compute generated roots in bounded
    /// chunks. A constructor error sends nothing; source and allocation errors are fatal.
    /// The original borrowed forest leaf storage is included in `source_bytes` by its owner.
    pub fn new(
        scope: TreeScope,
        trees: Vec<BoundedTree<'a, F>>,
        allowance: u64,
        source_bytes: u64,
    ) -> Result<Self, WindowError> {
        let mut height = 0;
        let mut max_vars = 0;
        let mut owned = 0;
        let mut scratch = 0;
        for (at, tree) in trees.iter().enumerate() {
            let (h, s) = match tree {
                BoundedTree::Resident(Tree::Forest(forest)) => {
                    let s = shape(forest, at)
                        .map_err(|_| WindowError::Shape("resident forest geometry is valid"))?;
                    owned = add(
                        owned,
                        add(
                            mul(
                                mul(2, forest.leaves.len() as u64)?,
                                size_of::<F::Base>() as u64,
                            )?,
                            mul((forest.height + 1) as u64, 128)?,
                        )?,
                    )?;
                    (forest.height, s)
                }
                BoundedTree::Resident(Tree::Fraction(tree)) => {
                    owned = add(owned, fraction_bytes(tree)?)?;
                    (tree.height, 0)
                }
                BoundedTree::Generated(rows) => {
                    scratch = scratch.max(add(
                        rows.scratch_bytes(),
                        mul(rows.chunk_rows() as u64, mul(4, size_of::<F>() as u64)?)?,
                    )?);
                    (rows.height(), 0)
                }
            };
            height = height.max(h);
            max_vars = max_vars.max(h + s);
        }
        let metadata = add(
            65_536,
            mul(
                trees.len() as u64,
                mul(
                    (max_vars + 1) as u64,
                    mul(
                        (max_vars + 8) as u64,
                        add(256, mul(8, size_of::<F>() as u64)?)?,
                    )?,
                )?,
            )?,
        )?;
        let input_bytes = mul(
            trees.capacity() as u64,
            size_of::<BoundedTree<'_, F>>() as u64,
        )?;
        let fixed = add(source_bytes, add(owned, add(metadata, input_bytes)?)?)?;
        if add(fixed, scratch)? > allowance {
            return Err(WindowError::Allocation);
        }
        // No preparation allocation or source read precedes the complete fixed-budget check.
        let mut geometry = Vec::new();
        geometry
            .try_reserve_exact(trees.len())
            .map_err(|_| allocation())?;
        for tree in &trees {
            geometry.push(match tree {
                BoundedTree::Resident(Tree::Forest(forest)) => (
                    forest.height,
                    shape(forest, 0)
                        .map_err(|_| WindowError::Shape("resident forest geometry is valid"))?,
                ),
                BoundedTree::Resident(Tree::Fraction(tree)) => (tree.height, 0),
                BoundedTree::Generated(rows) => (rows.height(), 0),
            });
        }
        let mut cuts = Vec::new();
        cuts.try_reserve_exact(height).map_err(|_| allocation())?;
        let mut working_bytes = add(fixed, scratch)?;
        // Shape-only planning precedes every generated root read and every proof message.
        for level in 0..height {
            let shapes: Vec<_> = geometry
                .iter()
                .zip(&trees)
                .filter_map(|(&(h, s), tree)| {
                    let join = height - h;
                    (join <= level).then(|| WindowShape {
                        vars: s + level - join,
                        tables: if matches!(tree, BoundedTree::Resident(Tree::Forest(_))) {
                            2
                        } else {
                            4
                        },
                        delta: 2,
                        windowable: matches!(tree, BoundedTree::Generated(_)),
                    })
                })
                .collect();
            let rounds = shapes.iter().map(|s| s.vars).max().unwrap_or(0);
            let mut choice = None;
            for cut in 0..rounds.max(1) {
                match WindowMemory::with_window_vars(&shapes, cut, size_of::<F>(), 1) {
                    Ok(memory) => {
                        let bytes = add(add(fixed, scratch)?, memory.working_bytes)?;
                        if bytes <= allowance {
                            choice = Some((cut, bytes));
                            break;
                        }
                    }
                    Err(WindowError::Allocation) => {}
                    Err(error) => return Err(error),
                }
            }
            let (cut, bytes) = choice.ok_or_else(allocation)?;
            cuts.push(cut);
            working_bytes = working_bytes.max(bytes);
        }
        let mut states = Vec::with_capacity(trees.len());
        for (tree, (h, s)) in trees.into_iter().zip(geometry) {
            let join = height - h;
            states.push(match tree {
                BoundedTree::Resident(Tree::Forest(forest)) => BatchState::Resident(State {
                    kind: Kind::Forest {
                        trees_log: s,
                        levels: build_levels(forest.leaves, forest.height),
                    },
                    join,
                    point: Vec::new(),
                    claim: None,
                }),
                BoundedTree::Resident(Tree::Fraction(tree)) => {
                    let claim = (h == 0).then(|| {
                        LevelClaim::Fraction(FractionClaim {
                            point: Vec::new(),
                            numerator: tree.root.0,
                            denominator: tree.root.1,
                        })
                    });
                    BatchState::Resident(State {
                        kind: Kind::Fraction(tree),
                        join,
                        point: Vec::new(),
                        claim,
                    })
                }
                BoundedTree::Generated(rows) => {
                    let root = rows.root()?;
                    let claim = (h == 0).then(|| {
                        LevelClaim::Fraction(FractionClaim {
                            point: Vec::new(),
                            numerator: root.0,
                            denominator: root.1,
                        })
                    });
                    BatchState::Generated(Generated {
                        rows,
                        root,
                        join,
                        point: Vec::new(),
                        claim,
                    })
                }
            });
        }
        Ok(Self {
            scope,
            states,
            height,
            allowance,
            working_bytes,
            cuts,
        })
    }

    /// Complete conservative batch bound, including the supplied source backing storage.
    pub fn working_bytes(&self) -> u64 {
        self.working_bytes
    }

    /// Shape-only number of streamed global rounds at each level, before roots are read.
    pub fn level_cuts(&self) -> &[usize] {
        &self.cuts
    }

    /// A fraction root by original schedule index, for the original roots message.
    /// Forests have no LogUp root pair.
    pub fn fraction_root(&self, index: usize) -> Option<(F, F)> {
        match self.states.get(index)? {
            BatchState::Generated(s) => Some(s.root),
            BatchState::Resident(State {
                kind: Kind::Fraction(tree),
                ..
            }) => Some(tree.root),
            _ => None,
        }
    }

    /// Prove every level, streaming generated tables one round per halving until their folds
    /// fit. Supplied roots must already have been absorbed by the caller in canonical order.
    /// Errors discard the session; no root, level or final claim is omitted on success.
    pub fn prove<T: SumcheckTranscript<F> + ?Sized>(
        mut self,
        transcript: &mut T,
    ) -> Result<ProvenTrees<F>, WindowError> {
        let mut levels = Vec::with_capacity(self.height);
        for level in 0..self.height {
            let prefix = self.scope.level(level as u32);
            let s_max = self
                .states
                .iter()
                .filter(|s| s.join() == level)
                .map(BatchState::trees_log)
                .max()
                .unwrap_or(0);
            let root_challenges: Vec<_> = (0..s_max)
                .map(|at| transcript.challenge(&label(prefix, format_args!("root/{at}"))))
                .collect();
            for state in self.states.iter_mut().filter(|s| s.join() == level) {
                state.set_point(root_challenges[s_max - state.trees_log()..].to_vec());
            }
            let active: Vec<_> = (0..self.states.len())
                .filter(|&i| self.states[i].join() <= level)
                .collect();
            let gamma = transcript.challenge(&label(prefix, format_args!("gamma")));
            let lambda = active
                .iter()
                .any(|&i| self.states[i].fraction())
                .then(|| transcript.challenge(&label(prefix, format_args!("lambda"))));
            let lambda_value = lambda.unwrap_or(F::ZERO);
            let proven = {
                let mut source = LevelSource {
                    operands: Vec::new(),
                };
                let specs: Vec<_> = active
                    .iter()
                    .map(|&i| source.instance(&self.states[i], level, lambda_value))
                    .collect();
                let cut = self.cuts[level];
                let prepared =
                    PreparedWindow::with_window_vars(&mut source, Protocol::One, 3, specs, cut, 1)?;
                for (k, &i) in active.iter().enumerate() {
                    if prepared.sum(k) != Some(self.states[i].claim(level, lambda_value)) {
                        return Err(WindowError::Source);
                    }
                }
                prepared.prove_with_gamma(transcript, prefix, &mut source, gamma)?
            };
            let rho = transcript.challenge(&label(prefix, format_args!("line")));
            let mut next_claims = Vec::with_capacity(active.len());
            for (k, &i) in active.iter().enumerate() {
                let own = proven.instance_point(k).ok_or(WindowError::Source)?;
                let evals = proven.instance_evals(k).ok_or(WindowError::Source)?;
                let line = |at: usize| -> Result<F, WindowError> {
                    let zero = *evals.get(at).ok_or(WindowError::Source)?;
                    let one = *evals.get(at + 1).ok_or(WindowError::Source)?;
                    Ok(zero + rho * (one - zero))
                };
                let mut point = Vec::with_capacity(own.len() + 1);
                point.push(rho);
                point.extend_from_slice(own);
                let claim = if self.states[i].fraction() {
                    LevelClaim::Fraction(FractionClaim {
                        point: point.clone(),
                        numerator: line(0)?,
                        denominator: line(2)?,
                    })
                } else {
                    LevelClaim::Product(crate::Claim {
                        point: point.clone(),
                        value: line(0)?,
                    })
                };
                self.states[i].finish_level(level, claim.clone(), point);
                next_claims.push(claim);
            }
            let point = proven.point().to_vec();
            levels.push(ProvenTreeLevel {
                root_challenges,
                gamma,
                lambda,
                proof: proven.into_proof(),
                point,
                rho,
                trees: active,
                next_claims,
            });
        }
        let leaf_claims = self
            .states
            .into_iter()
            .map(BatchState::final_claim)
            .collect::<Option<Vec<_>>>()
            .ok_or(WindowError::Source)?;
        Ok(ProvenTrees {
            levels,
            leaf_claims,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Forest, FractionTree, FractionTreeSource, LeafNumerators, prove_trees_in};
    use core::{cell::Cell, num::NonZeroUsize};
    use tessera_codec::Ascii;
    use tessera_field::{Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
    use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};

    struct Source {
        height: usize,
        n: Vec<G>,
        d: Vec<F>,
        largest: Cell<usize>,
    }
    impl FractionTreeSource<F> for Source {
        fn height(&self) -> usize {
            self.height
        }
        fn read(&self, first: usize, n: &mut [F], d: &mut [F]) -> Result<(), WindowError> {
            self.largest.set(self.largest.get().max(n.len()));
            let end = first.checked_add(n.len()).ok_or(WindowError::Source)?;
            let input = self.n.get(first..end).ok_or(WindowError::Source)?;
            for (out, &value) in n.iter_mut().zip(input) {
                *out = F::from(value);
            }
            d.copy_from_slice(self.d.get(first..end).ok_or(WindowError::Source)?);
            Ok(())
        }
    }
    fn source(height: usize) -> Source {
        let count = 1usize << height;
        Source {
            height,
            n: (0..count)
                .map(|j| G::from_u64_reduced((5 * j + 3) as u64))
                .collect(),
            d: (0..count)
                .map(|j| {
                    F::new(
                        G::from_u64_reduced((j + 7) as u64),
                        G::from_u64_reduced((3 * j + 11) as u64),
                    )
                })
                .collect(),
            largest: Cell::new(0),
        }
    }
    fn transcript() -> Transcript {
        Transcript::new(&SessionContext {
            protocol: Ascii::new("tessera-protocol/1").unwrap(),
            mode: Mode::Live,
            params: ParamSetId(Ascii::new("test-small/1").unwrap()),
            statement_digest: [27; 32],
            session_id: [39; 16],
        })
    }
    fn inputs<'a>(sources: &'a [Source], forest: &'a [G], mixed: bool) -> Vec<BoundedTree<'a, F>> {
        let mut trees: Vec<_> = sources
            .iter()
            .map(|s| {
                BoundedTree::Generated(
                    FractionTreeRows::new(s, NonZeroUsize::new(13).unwrap(), u64::MAX).unwrap(),
                )
            })
            .collect();
        if mixed {
            trees.push(BoundedTree::Resident(Tree::Fraction(
                FractionTree::new(3, LeafNumerators::Unit, |j| {
                    F::from(G::from_u64_reduced((j + 19) as u64))
                })
                .unwrap(),
            )));
            trees.push(BoundedTree::Resident(Tree::Forest(Forest::new(forest, 3))));
        }
        trees
    }

    #[test]
    fn level_adapter_partner_scratch_is_bounded_by_chunk_not_destination() {
        let input = source(16);
        let chunk = 13;
        let rows =
            FractionTreeRows::new(&input, NonZeroUsize::new(chunk).unwrap(), u64::MAX).unwrap();
        let bound = rows.scratch_bytes() + (4 * chunk * size_of::<F>()) as u64 + 1024;
        let level = LevelSource {
            operands: vec![Operand::Generated {
                rows: &rows,
                depth: 16,
                part: 0,
            }],
        };
        let mut output = buffer::<F>(1 << 15).unwrap();
        let measured = allocation_counter::measure(|| level.read(0, 0, &mut output).unwrap());
        assert!(
            measured.bytes_max as u64 <= bound,
            "partner scratch must depend on chunk, not destination: {measured:?}, bound={bound}"
        );
        for (at, &value) in output.iter().enumerate() {
            assert_eq!(value, F::from(input.n[2 * at]));
        }
        assert!(input.largest.get() <= chunk);
    }

    #[test]
    fn complete_streamed_batches_match_literal_messages_claims_and_challenges() {
        for mixed in [false, true] {
            let scope = if mixed {
                TreeScope::Stage
            } else {
                TreeScope::Tables
            };
            let sources: Vec<_> = [0, 1, 4, 10].into_iter().map(source).collect();
            let forest: Vec<_> = (0..64)
                .map(|j| G::from_u64_reduced((7 * j + 9) as u64))
                .collect();
            let mut originals: Vec<_> = sources
                .iter()
                .map(|s| {
                    Tree::Fraction(
                        FractionTree::new(s.height, LeafNumerators::Base(&s.n), |j| s.d[j])
                            .unwrap(),
                    )
                })
                .collect();
            if mixed {
                originals.push(Tree::Fraction(
                    FractionTree::new(3, LeafNumerators::Unit, |j| {
                        F::from(G::from_u64_reduced((j + 19) as u64))
                    })
                    .unwrap(),
                ));
                originals.push(Tree::Forest(Forest::new(&forest, 3)));
            }
            let mut expected_transcript = transcript();
            let expected = prove_trees_in(&mut expected_transcript, scope, originals).unwrap();
            let wide = PreparedTreeBatch::new(scope, inputs(&sources, &forest, mixed), u64::MAX, 0)
                .unwrap()
                .working_bytes();
            for budget in [u64::MAX, wide - 1] {
                let prepared =
                    PreparedTreeBatch::new(scope, inputs(&sources, &forest, mixed), budget, 0)
                        .unwrap();
                assert!(prepared.working_bytes() <= budget);
                if budget != u64::MAX {
                    assert!(prepared.level_cuts().iter().any(|&cut| cut > 0));
                }
                let mut actual_transcript = transcript();
                let actual = prepared.prove(&mut actual_transcript).unwrap();
                assert_eq!(actual.levels().len(), expected.levels().len());
                for (a, b) in actual.levels().iter().zip(expected.levels()) {
                    assert_eq!(a.proof(), b.proof());
                    assert_eq!(a.point(), b.point());
                    assert_eq!(a.next_claims(), b.next_claims());
                }
                assert_eq!(actual.leaf_claims(), expected.leaf_claims());
                let mut expected_next = transcript();
                // Re-run the unchanged engine so both budget cases compare the same state.
                let mut trees: Vec<_> = sources
                    .iter()
                    .map(|s| {
                        Tree::Fraction(
                            FractionTree::new(s.height, LeafNumerators::Base(&s.n), |j| s.d[j])
                                .unwrap(),
                        )
                    })
                    .collect();
                if mixed {
                    trees.push(Tree::Fraction(
                        FractionTree::new(3, LeafNumerators::Unit, |j| {
                            F::from(G::from_u64_reduced((j + 19) as u64))
                        })
                        .unwrap(),
                    ));
                    trees.push(Tree::Forest(Forest::new(&forest, 3)));
                }
                let _ = prove_trees_in(&mut expected_next, scope, trees).unwrap();
                assert_eq!(
                    actual_transcript.challenge_fp2("after"),
                    expected_next.challenge_fp2("after")
                );
                assert!(sources.iter().all(|s| s.largest.get() <= 13));
            }
        }
    }
}
