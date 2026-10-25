// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Complete protocol/1 runtime message geometry and the mandatory RFC-0053 budget.
//!
//! A proof's budget is evaluated at its own grouping (RFC-0054 'A proof's budget'): an ungrouped
//! proof's statement has `groups.count = 1` and is the ungrouped statement; a grouped proof's
//! counts `G = 2^{g_T + g_L}`, the leaf stage's level rounds by degree (the levels `levels`
//! derives for the ungrouped stage, which bound every group's and the table batch's), the merge's
//! `n_max`, the most variables of a merged polynomial, and `P`, the merged polynomials' count, one
//! per ungrouped split value and one per forest leaf table (RFC-0054 'The merge'). Every size
//! comes from the verifier's own layouts ([`Grouped`], [`split::Layout`]), never from the proof.

use crate::advice::{self, opening::Opening};
use crate::claims::{ClaimList, CommitmentLayout, ParamSetId};
use crate::gadgets::{self, model};
use crate::leaf::groups::Grouped;
use crate::leaf::{Tree, split};
use crate::model_openings::Claims;
use crate::params::ParamSet;
use crate::schedule::{RelationKind, Stage};
use crate::soundness::{
    GroupSizes, LookupSize, MatrixSize, RoundGroup, SoundnessParams, SoundnessReport,
    StatementSizes,
};
use crate::weights::{self, Matrix};
use tv_hash::transcript::Mode;

/// A failure in runtime metadata or its accepted parameter set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Inconsistent own geometry or calculator inputs (`ERR-INTERNAL`).
    Internal,
    /// A fallible metadata/report reservation failed (`ERR-RES-MEMORY`).
    Memory,
    /// Name, mode or layout differ from the accepted set (`REJ-PARAMS`).
    Params,
    /// The first failing soundness row, zero-indexed (`REJ-PARAMS`).
    Targets {
        /// Index in the RFC-0014 report's fixed row order.
        row: usize,
    },
}

fn reserve<T>(n: usize) -> Result<Vec<T>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| Error::Memory)?;
    Ok(out)
}
fn number(n: usize) -> Result<u64, Error> {
    u64::try_from(n).map_err(|_| Error::Internal)
}
fn power(n: u32) -> Result<u64, Error> {
    1u64.checked_shl(n).ok_or(Error::Internal)
}
fn lookup_size(entries: u64, rows: u64) -> Result<LookupSize, Error> {
    if entries == 0 || rows == 0 {
        return Err(Error::Internal);
    }
    Ok(LookupSize {
        lookups: entries.checked_next_power_of_two().ok_or(Error::Internal)?,
        rows: rows.checked_next_power_of_two().ok_or(Error::Internal)?,
    })
}
fn claims_error(e: crate::validation::ClaimsError) -> Error {
    if matches!(e, crate::validation::ClaimsError::Memory) {
        Error::Memory
    } else {
        Error::Internal
    }
}
fn gadget_error(e: gadgets::GadgetError) -> Error {
    if matches!(e, gadgets::GadgetError::Memory) {
        Error::Memory
    } else {
        Error::Internal
    }
}

/// One sumcheck's exact packet shape and batch count, from verifier-owned metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fields {
    /// Number of round vectors.
    pub rounds: usize,
    /// Each round sends `degree` fields: p(1) is inferred (RFC-0010 'Sumcheck messages').
    pub degree: usize,
    /// Raw final fields, including every native source evaluation.
    pub evaluations: usize,
    /// Instances participating in this single batch.
    pub instances: usize,
}
impl Fields {
    /// Total Fp2 cells, excluding vector-length prefixes; `None` for overflowing own metadata.
    pub fn field_count(self) -> Option<usize> {
        self.rounds
            .checked_mul(self.degree)?
            .checked_add(self.evaluations)
    }
}
/// Exact stage fields, including its original split values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageFields {
    /// Stage number, starting at one.
    pub index: u16,
    /// This stage's one mixed graph/native sumcheck.
    pub sumcheck: Fields,
    /// Fields following its final evaluations.
    pub splits: usize,
}
fn levels(trees: &[Tree]) -> Result<Vec<Fields>, Error> {
    let height = trees.iter().map(|t| t.height()).max().unwrap_or(0);
    let mut out = reserve(usize::try_from(height).map_err(|_| Error::Internal)?)?;
    for depth in 0..height {
        let mut f = Fields {
            rounds: 0,
            degree: crate::tree::LEVEL_DEGREE,
            evaluations: 0,
            instances: 0,
        };
        for &tree in trees {
            let join = height.checked_sub(tree.height()).ok_or(Error::Internal)?;
            if depth < join {
                continue;
            }
            let (extra, evaluations) = match tree {
                Tree::Forest { trees_log, .. } => (trees_log, 2),
                _ => (0, 4),
            };
            let variables = depth
                .checked_sub(join)
                .and_then(|d| d.checked_add(extra))
                .ok_or(Error::Internal)?;
            f.rounds = f
                .rounds
                .max(usize::try_from(variables).map_err(|_| Error::Internal)?);
            f.evaluations = f
                .evaluations
                .checked_add(evaluations)
                .ok_or(Error::Internal)?;
            f.instances = f.instances.checked_add(1).ok_or(Error::Internal)?;
        }
        out.push(f);
    }
    Ok(out)
}
fn stage(s: &Stage) -> Result<StageFields, Error> {
    Ok(StageFields {
        index: s.index,
        sumcheck: Fields {
            rounds: s.rounds,
            degree: s.degree,
            instances: s.instances.len(),
            evaluations: s.instances.iter().try_fold(0usize, |n, i| {
                n.checked_add(i.input_evaluations()).ok_or(Error::Internal)
            })?,
        },
        splits: s.splits,
    })
}
/// RFC-0054 'A proof's budget': the grouped sizes of a proof of `grouped`'s `G`, over the ungrouped
/// stage's `levels`, whose split holds `split_values` values and whose schedule `forests` forests.
/// With `G = 1` they are [`GroupSizes::NONE`].
///
/// `P` is the merge's polynomials, one per ungrouped split value and one per forest leaf table
/// (RFC-0054 'The merge'): the grouped layout must count as many as RFC-0045's split layout and the
/// leaf schedule, else `Internal`. `n_max` is the most variables among them.
fn grouping(
    levels: &[Fields],
    grouped: &Grouped,
    split_values: usize,
    forests: usize,
) -> Result<GroupSizes, Error> {
    let count = grouped.count();
    if count == 0 {
        return Err(Error::Internal);
    }
    if count == 1 {
        return Ok(GroupSizes::NONE);
    }
    let polynomials = grouped.polynomials();
    let expected = split_values.checked_add(forests).ok_or(Error::Internal)?;
    if grouped.values() != split_values || polynomials.len() != expected {
        return Err(Error::Internal);
    }
    let mut level_rounds = reserve(levels.len())?;
    for f in levels {
        level_rounds.push(RoundGroup {
            count: number(f.rounds)?,
            degree: number(f.degree)?,
        });
    }
    Ok(GroupSizes {
        count: u64::from(count),
        level_rounds,
        merge_rounds: u64::from(polynomials.iter().map(|p| p.variables).max().unwrap_or(0)),
        merge_claims: number(polynomials.len())?,
    })
}
fn zero_checks_of(s: &Stage) -> impl Iterator<Item = usize> + '_ {
    s.instances
        .iter()
        .filter(|i| i.relation.kind() == RelationKind::ZeroCheck)
        .map(|i| i.vars)
}

/// Complete static runtime geometry; opening groups are filled only from the canonical plan.
#[derive(Debug)]
pub struct Geometry {
    levels: Vec<Fields>,
    stages: Vec<StageFields>,
    roots: usize,
    values: usize,
    leaf_splits: usize,
    layout: CommitmentLayout,
    name: ParamSetId,
    statement: StatementSizes,
}
impl Geometry {
    /// Derive all runtime leaf/native/GKR counts from the public model and request.
    /// Claim lists retain their prerelease version; this describes protocol/1 packets.
    ///
    /// `grouped` is the verifier's own layout of the proof's leaf groups over `runtime`
    /// ([`Grouped::derive_runtime`] at `Round1.leaf_groups`), `None` for an ungrouped proof: the
    /// statement's `groups` are the proof's own grouping (module documentation).
    ///
    /// # Errors
    /// `Internal` for inconsistent own metadata; `Memory` for failed reservations.
    pub fn derive(
        claims: &ClaimList,
        runtime: &gadgets::runtime::Plan<'_>,
        grouped: Option<&Grouped>,
    ) -> Result<Self, Error> {
        let leaf = runtime.leaf_schedule().map_err(claims_error)?;
        let model = model::Plan::derive_runtime(runtime).map_err(gadget_error)?;
        let splits =
            split::Layout::new(&runtime.split_sites().map_err(claims_error)?).map_err(|e| {
                if e == split::Error::Memory {
                    Error::Memory
                } else {
                    Error::Internal
                }
            })?;
        let advice = runtime
            .advice_layout(advice::Config::new(
                &claims.arch,
                claims.gadget_policy,
                claims.layout.advice.log2_k,
                claims.layout.advice.segments,
            ))
            .map_err(claims_error)?;
        let levels = levels(leaf.schedule().trees())?;
        let mut stages = reserve(model.schedule().stages().len())?;
        let mut rounds = reserve(
            levels
                .len()
                .checked_add(model.schedule().stages().len())
                .ok_or(Error::Internal)?,
        )?;
        let mut tree_levels = reserve(levels.len())?;
        for f in &levels {
            rounds.push(RoundGroup {
                count: number(f.rounds)?,
                degree: number(f.degree)?,
            });
            tree_levels.push(number(f.instances)?);
        }
        let mut stage_counts = reserve(model.schedule().stages().len())?;
        let checks = model.schedule().stages().iter().try_fold(0usize, |n, s| {
            n.checked_add(s.instances.len()).ok_or(Error::Internal)
        })?;
        let mut zero_checks = reserve(checks)?;
        for s in model.schedule().stages() {
            let f = stage(s)?;
            rounds.push(RoundGroup {
                count: number(f.sumcheck.rounds)?,
                degree: number(f.sumcheck.degree)?,
            });
            stage_counts.push(number(f.sumcheck.instances)?);
            for variables in zero_checks_of(s) {
                zero_checks.push(number(variables)?);
            }
            stages.push(f);
        }
        let mut forests = reserve(leaf.schedule().trees().len())?;
        for tree in leaf.schedule().trees() {
            if let Tree::Forest { trees_log, .. } = tree {
                forests.push(u64::from(*trees_log));
            }
        }
        let groups = match grouped {
            Some(grouped) => grouping(&levels, grouped, splits.values(), forests.len())?,
            None => GroupSizes::NONE,
        };
        let mut lookups = reserve(leaf.lookups().len())?;
        for &(id, entries) in leaf.lookups() {
            let rows = runtime.table(id).ok_or(Error::Internal)?.rows();
            lookups.push(lookup_size(entries, rows)?);
        }
        let name = claims.params.try_clone().ok_or(Error::Memory)?;
        Ok(Self {
            roots: leaf
                .schedule()
                .instances()
                .checked_mul(4)
                .ok_or(Error::Internal)?,
            values: leaf
                .schedule()
                .trees()
                .iter()
                .filter(|t| {
                    matches!(
                        t,
                        Tree::Lookup {
                            has_values: true,
                            ..
                        }
                    )
                })
                .count(),
            leaf_splits: splits
                .values()
                .checked_add(model.schedule().leaf_splits())
                .ok_or(Error::Internal)?,
            levels,
            stages,
            layout: claims.layout,
            name,
            statement: StatementSizes {
                protocol: 1,
                rounds,
                stages: stage_counts,
                tree_levels,
                zero_checks,
                forests,
                lookups,
                claim_groups: Vec::new(),
                matrices: Vec::new(),
                groups,
                advice_rows: power(
                    advice
                        .nu()
                        .checked_sub(u32::from(claims.layout.advice.log2_k))
                        .ok_or(Error::Internal)?,
                )?,
            },
        })
    }
    /// Four roots per complete runtime lookup instance.
    pub fn roots(&self) -> usize {
        self.roots
    }
    /// One final value field for every table with values.
    pub fn values(&self) -> usize {
        self.values
    }
    /// Canonical lookup/source fields plus the graph atom splits.
    pub fn leaf_splits(&self) -> usize {
        self.leaf_splits
    }
    /// Common levels, root first, including all joining trees.
    pub fn levels(&self) -> &[Fields] {
        &self.levels
    }
    /// Actual mixed graph/native stages, in packet order.
    pub fn stages(&self) -> &[StageFields] {
        &self.stages
    }

    /// Complete the statement using ALL original opening obligations, never proof row counts.
    ///
    /// # Errors
    /// `Internal` for inconsistent matrix metadata; `Memory` for reservations.
    pub fn statement_sizes(
        mut self,
        opening: &Opening<'_>,
        claims: &Claims<'_>,
        layouts: &[weights::Layout; 10],
    ) -> Result<StatementSizes, Error> {
        let counts = opening.claim_counts();
        self.statement.claim_groups = reserve(counts.len())?;
        self.statement.claim_groups.extend(counts);
        self.statement.matrices = reserve(Matrix::ALL.len())?;
        for matrix in claims.matrices() {
            let layout = layouts
                .get(usize::from(matrix.tag()))
                .ok_or(Error::Internal)?;
            if layout.matrix() != matrix {
                return Err(Error::Internal);
            }
            self.statement.matrices.push(MatrixSize {
                matrix,
                width: power(layout.log2_k())?,
                rows: layout.committed_rows(),
                variables: u64::from(
                    layout
                        .log2_k()
                        .checked_add(layout.row_vars())
                        .ok_or(Error::Internal)?,
                ),
                // `e_T` (RFC-0065 'The count'), which protocol/1's budget does not read.
                claims: number(claims.claim_count(matrix))?,
            });
        }
        Ok(self.statement)
    }

    /// Enforce the unchanged accepted parameter set against the complete runtime statement.
    /// A passing budget still owes every proof, opening, column and session check.
    ///
    /// # Errors
    /// `Params` for name/mode/layout mismatch; `Targets` for the first failed target;
    /// `Internal` for inconsistent own sizes; `Memory` for a failed reservation.
    pub fn check_budget(
        self,
        params: ParamSet,
        mode: Mode,
        opening: &Opening<'_>,
        claims: &Claims<'_>,
        layouts: &[weights::Layout; 10],
    ) -> Result<Budget, Error> {
        let mode_matches = match params {
            ParamSet::Live64 => mode == Mode::Live,
            ParamSet::Ni100 => mode == Mode::NonInteractive,
            #[cfg(feature = "test-params")]
            ParamSet::TestSmall => true,
        };
        if !mode_matches || self.name.as_str() != params.name() || self.layout != params.layout() {
            return Err(Error::Params);
        }
        let statement = self.statement_sizes(opening, claims, layouts)?;
        evaluate_budget(params, statement)
    }
}
fn evaluate_budget(params: ParamSet, statement: StatementSizes) -> Result<Budget, Error> {
    let params = SoundnessParams::try_from_param_set(params).map_err(|_| Error::Memory)?;
    let report = SoundnessReport::evaluate(params, statement).map_err(|e| {
        if e == crate::soundness::SoundnessError::Memory {
            Error::Memory
        } else {
            Error::Internal
        }
    })?;
    if let Some(row) = report.rows.iter().position(|r| r.pass == Some(false)) {
        return Err(Error::Targets { row });
    }
    Ok(Budget { report })
}

/// An evaluated runtime budget. This is one prerequisite of acceptance, not a verdict.
#[derive(Debug)]
#[must_use = "retain this budget report and discharge all proof/session obligations before Accept"]
pub struct Budget {
    report: SoundnessReport,
}
impl Budget {
    /// Complete deterministic RFC-0014 report with actual runtime counts.
    pub fn report(&self) -> &SoundnessReport {
        &self.report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tv_testkit::{Json, read_vectors};
    fn n(v: &Json, key: &str) -> u32 {
        v.get(key).unwrap().as_u64().unwrap() as u32
    }
    #[test]
    fn tree_fields_match_literal_parent_population_oracle() {
        let vectors = read_vectors("runtime-budget/geometry.json");
        for case in vectors.get("cases").unwrap().as_array().unwrap() {
            let trees = case
                .get("trees")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    if t.get("kind").unwrap().as_str() == Some("forest") {
                        Tree::Forest {
                            height: n(t, "height"),
                            trees_log: n(t, "trees_log"),
                        }
                    } else {
                        Tree::Lookup {
                            instance: i,
                            height: n(t, "height"),
                            has_values: true,
                        }
                    }
                })
                .collect::<Vec<_>>();
            let fields = levels(&trees).unwrap();
            let expected = case.get("levels").unwrap().as_array().unwrap();
            assert_eq!(fields.len(), expected.len());
            for (f, e) in fields.iter().zip(expected) {
                assert_eq!(
                    (f.rounds, f.degree, f.evaluations, f.instances),
                    (
                        n(e, "rounds") as usize,
                        n(e, "degree") as usize,
                        n(e, "evaluations") as usize,
                        n(e, "instances") as usize
                    )
                );
            }
        }
    }

    #[test]
    fn native_raw_evaluations_and_all_zero_checks_survive_graph_projection() {
        use crate::lowering::{NodeId, TensorId};
        use crate::schedule::{Claim, Instance, Layers, Point, Relation, Route, TensorRef};
        let dummy = Claim {
            tensor: TensorRef::Single(TensorId(0)),
            vars: 0,
            atoms: Vec::new(),
            route: Route::Pool,
        };
        let instances = [
            (
                Relation::Native {
                    tag: 1,
                    evaluations: 8,
                    zero: true,
                },
                5,
                4,
            ),
            (
                Relation::Native {
                    tag: 2,
                    evaluations: 6,
                    zero: false,
                },
                7,
                2,
            ),
            (Relation::Booleanity, 3, 3),
            (
                Relation::Native {
                    tag: 3,
                    evaluations: 1,
                    zero: true,
                },
                0,
                1,
            ),
        ]
        .into_iter()
        .map(|(relation, vars, degree)| Instance {
            relation,
            vars,
            degree,
            node: NodeId(0),
            tensor: TensorRef::Single(TensorId(0)),
            stacked: false,
            point: Point::default(),
            layers: Layers::None,
            zero: Point::default(),
            evals: vec![dummy.clone()],
        })
        .collect();
        let input = Stage {
            index: 1,
            rounds: 7,
            degree: 4,
            layer_vars: 0,
            zero_vars: 5,
            splits: 9,
            instances,
        };
        assert_eq!(
            stage(&input).unwrap(),
            StageFields {
                index: 1,
                sumcheck: Fields {
                    rounds: 7,
                    degree: 4,
                    evaluations: 16,
                    instances: 4
                },
                splits: 9,
            }
        );
        assert_eq!(zero_checks_of(&input).collect::<Vec<_>>(), [5, 3, 0]);
        assert_eq!(stage(&input).unwrap().sumcheck.field_count(), Some(44));
    }

    #[test]
    fn lookup_padding_and_geometry_overflow_are_checked() {
        let size = lookup_size(5, 3).unwrap();
        assert_eq!((size.lookups, size.rows), (8, 4));
        let size = lookup_size(8, 4).unwrap();
        assert_eq!((size.lookups, size.rows), (8, 4));
        for (entries, rows) in [(0, 1), (1, 0), (u64::MAX, 1), (1, u64::MAX)] {
            assert!(matches!(lookup_size(entries, rows), Err(Error::Internal)));
        }
        assert_eq!(
            Fields {
                rounds: usize::MAX,
                degree: 2,
                evaluations: 1,
                instances: 1,
            }
            .field_count(),
            None
        );
        assert_eq!(
            Fields {
                rounds: usize::MAX,
                degree: 1,
                evaluations: 1,
                instances: 1,
            }
            .field_count(),
            None
        );
    }

    #[test]
    fn unchanged_presets_reject_actual_field_and_seed_target_failures() {
        for params in [ParamSet::Live64, ParamSet::Ni100] {
            assert_eq!(
                evaluate_budget(params, StatementSizes::default_405b())
                    .unwrap()
                    .report()
                    .meets_targets,
                Some(true)
            );
            let mut sizes = StatementSizes::default_405b();
            sizes.zero_checks.push(u64::MAX);
            assert!(matches!(
                evaluate_budget(params, sizes),
                Err(Error::Targets { row: 0 })
            ));
            let mut sizes = StatementSizes::default_405b();
            sizes.advice_rows = 1 << 32;
            assert!(matches!(
                evaluate_budget(params, sizes),
                Err(Error::Targets { row: 6 })
            ));
        }
        let mut sizes = StatementSizes::default_405b();
        sizes.lookups.push(LookupSize {
            lookups: 1 << 49,
            rows: 1 << 32,
        });
        assert!(matches!(
            evaluate_budget(ParamSet::Ni100, sizes),
            Err(Error::Targets { row: 3 })
        ));
    }

    /// RFC-0054 'A proof's budget': the bound at the proof's own `G`. Sizes whose grouped union
    /// misses 100 bits at `G = 2^14` refuse with `Targets` at the union row, alone; the same
    /// sizes at `G = 2^13` pass, and without groups they pass at the ungrouped bound.
    #[test]
    fn a_grouped_bound_that_misses_a_target_refuses_while_the_ungrouped_passes() {
        for params in [ParamSet::Live64, ParamSet::Ni100] {
            let mut sizes = StatementSizes::default_405b();
            sizes.groups.level_rounds = vec![RoundGroup {
                count: 6_000,
                degree: 3,
            }];
            assert!(matches!(
                evaluate_budget(params, sizes.clone()),
                Err(Error::Targets { row: 1 })
            ));
            let mut fewer = sizes.clone();
            fewer.groups.count = 1 << 13;
            let report = evaluate_budget(params, fewer).unwrap();
            assert_eq!(report.report().meets_targets, Some(true));
            let mut ungrouped = sizes;
            ungrouped.groups = GroupSizes::NONE;
            let report = evaluate_budget(params, ungrouped).unwrap();
            assert_eq!(report.report().meets_targets, Some(true));
            assert_eq!(report.report().statement.groups, GroupSizes::NONE);
        }
    }

    /// RFC-0054 'A proof's budget', 'The merge': a grouped proof's sizes from the verifier's own
    /// layouts of the recorded `g3` proof, `(g_T, g_L) = (1, 2)`: `G = 8`, the ungrouped stage's
    /// levels by degree, `P` the split's values and the forests, `n_max` their most variables.
    /// A grouped layout that disagrees with the split layout or the schedule is the verifier's
    /// fault, and one group is no grouping.
    #[cfg(feature = "test-params")]
    #[test]
    fn grouped_sizes_come_from_the_layouts_and_refuse_a_disagreeing_count() {
        use crate::protocol::{LeafGroups, NiProof};
        let limits = tv_codec::DecodeLimits::MESSAGE;
        let claims = ClaimList::open(
            include_bytes!("../tests/fixtures/grouped/g3/claims.tcl"),
            &limits,
        )
        .unwrap();
        let proof = NiProof::open(
            include_bytes!("../tests/fixtures/grouped/g3/ni.tnp"),
            &limits,
        )
        .unwrap();
        let validated = crate::validation::validate(&claims).unwrap();
        let request = &proof.round1.statement.batch.requests[0];
        let graph = crate::lowering::lower(
            &crate::validation::check_public(&claims).unwrap(),
            request.prompt_len + request.output_len,
        )
        .unwrap();
        let runtime = gadgets::runtime::Plan::derive(
            &graph,
            claims.gadget_policy,
            request.prompt_len,
            0,
            validated.tables(),
        )
        .unwrap();
        let leaf = runtime.leaf_schedule().unwrap();
        let fields = levels(leaf.schedule().trees()).unwrap();
        let values = split::Layout::new(&runtime.split_sites().unwrap())
            .unwrap()
            .values();
        let forests = leaf
            .schedule()
            .trees()
            .iter()
            .filter(|t| matches!(t, Tree::Forest { .. }))
            .count();
        assert_eq!(
            proof.round1.leaf_groups,
            LeafGroups {
                token_bits: 1,
                layer_bits: 2
            }
        );
        let grouped = Grouped::derive_runtime(&graph, &runtime, proof.round1.leaf_groups).unwrap();
        let sizes = grouping(&fields, &grouped, values, forests).unwrap();
        assert_eq!(sizes.count, 8);
        assert!(forests >= 1 && values >= 1);
        assert_eq!(sizes.merge_claims, (values + forests) as u64);
        assert_eq!(
            sizes.merge_rounds,
            u64::from(
                grouped
                    .polynomials()
                    .iter()
                    .map(|p| p.variables)
                    .max()
                    .unwrap()
            )
        );
        assert_eq!(
            sizes.level_rounds,
            fields
                .iter()
                .map(|f| RoundGroup {
                    count: f.rounds as u64,
                    degree: 3,
                })
                .collect::<Vec<_>>()
        );
        assert_eq!(
            grouping(&fields, &grouped, values + 1, forests),
            Err(Error::Internal)
        );
        assert_eq!(
            grouping(&fields, &grouped, values, forests + 1),
            Err(Error::Internal)
        );
        let one = Grouped::derive_runtime(&graph, &runtime, LeafGroups::NONE).unwrap();
        assert_eq!(
            grouping(&fields, &one, values, forests),
            Ok(GroupSizes::NONE)
        );
    }
}
