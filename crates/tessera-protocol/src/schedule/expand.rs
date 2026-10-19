// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The expansion of the label patterns for one proof unit (RFC-0014 'Transcript schedule'): the
//! full ordered list of what the transcript absorbs and squeezes, and under which label.

use tessera_hash::Mode;

use super::label::{Label, Protocol, WEIGHTS_XI, WeightsXi};
use super::sizes::{LeafTree, MAX_VARS, ScheduleSizes};
use crate::{ParamSet, ProofOfWork, ProtocolError, WeightMatrix};

/// What a step does on the transcript (RFC-0004 'Transcript').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// Absorbs one message's TCE bytes under the label (RFC-0036 'What the transcript absorbs').
    Absorb,
    /// Squeezes 32 bytes under the label: the non-interactive seed.
    Digest,
    /// Draws `count` challenges in `Fp2` under the label, one after the other: one challenge, or
    /// every coordinate of a proximity point, in draw order. A point of no coordinate draws
    /// nothing.
    Fp2 {
        /// How many.
        count: u32,
    },
    /// Draws `count` column indices of `log_n` bits under the label, with replacement
    /// (`challenge_indices`).
    Indices {
        /// The codeword's length is `2^log_n`.
        log_n: u32,
        /// `τ_A` or `τ_W`.
        count: u32,
    },
    /// A proof of work: the nonce whose `H("pow", state ‖ Ascii(label) ‖ nonce)` has `bits`
    /// leading zero bits, which is then absorbed under the label (RFC-0014 'Grinding').
    Grind {
        /// Which proof of work.
        phase: ProofOfWork,
        /// `g_seed` or `g_query`.
        bits: u32,
    },
}

/// One step of a schedule: a label and what the transcript does under it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Step {
    /// The label.
    pub label: Label,
    /// The operation.
    pub op: Op,
}

/// The transcript schedule of one proof unit: every label it absorbs or squeezes under, in
/// order, for one mode and parameter set (RFC-0014 'Transcript schedule').
///
/// [`expand`](Self::expand) derives it from the [`ScheduleSizes`], which public data fixes, so
/// that the prover and the verifier each derive the same list. Each label appears once
/// (INV-FS-4), and every prover message is absorbed before the challenges that depend on it are
/// squeezed (INV-FS-1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TranscriptSchedule {
    protocol: Protocol,
    mode: Mode,
    params: ParamSet,
    steps: Vec<Step>,
}

impl TranscriptSchedule {
    /// The schedule of a proof unit of `sizes` in `mode` at `params`, in `protocol`.
    ///
    /// In `tessera-protocol/1`, in order (RFC-0014 'Transcript schedule' and 'Grinding', RFC-0036
    /// 'What the transcript absorbs', RFC-0010 'Tree levels' and 'Batching within a stage',
    /// RFC-0037 'Messages and transcript', RFC-0045 'The split', RFC-0049 'The schedule'):
    ///
    /// | Row | Steps |
    /// |---|---|
    /// | 2 | absorb `round1` |
    /// | 3 | live: absorb `seed`; non-interactive: grind `seed-pow` for `g_seed` bits, squeeze `seed`. Then each matrix with committed rows draws its proximity point under `pcs/weights/{T}/xi` ([`WEIGHTS_XI`]) |
    /// | 4 | `logup/alpha`, `logup/beta`, absorb `logup/roots` |
    /// | 5 | per tree level `ℓ`: `tree/{ℓ}/root/{c}` for the root point of the forests that join, `tree/{ℓ}/gamma`, `tree/{ℓ}/lambda` at a level with fraction trees, rounds `tree/{ℓ}/round/{j}` and `tree/{ℓ}/r/{j}`, absorb `tree/{ℓ}/evals`, `tree/{ℓ}/line`; then absorb `logup/values` and `stage/0/split` |
    /// | 6 | per stage `k`: in stage 1 `stage/1/layer/{c}`, the zero-check point `stage/{k}/zero/{c}`, `stage/{k}/gamma`, the rounds, absorb `stage/{k}/evals` and `stage/{k}/split` |
    /// | 7 | `pcs/advice/gamma`, `pcs/advice/xi`, then `pcs/weights/{T}/gamma` per opened matrix |
    /// | 8 | absorb `pcs/advice/combined`, then `pcs/weights/{T}/combined` per opened matrix |
    /// | 9 | absorb `scoring` |
    /// | 10 | live: absorb `queries`; non-interactive: grind `query-pow` for `g_query` bits, squeeze `pcs/advice/queries`, then `pcs/weights/{T}/queries` per opened matrix |
    ///
    /// A tree of height `h` joins the levels at `H − h`, `H` the tallest tree's height, and is an
    /// instance on `s + t` variables at its own level `t`; a level has as many rounds as its
    /// largest instance has variables. Split values are absorbed even when a stage splits
    /// nothing, as `logup/roots` and `logup/values` are even when empty.
    ///
    /// # Errors
    ///
    /// [`ProtocolError::Internal`], in this order: `tessera-protocol/2`, whose zero-knowledge
    /// labels are not defined yet (#148); a mode the parameter set does not allow; a proof of work
    /// of more than 64 bits; then the first rule of the sizes that fails, as
    /// [`ScheduleSizes`] documents them.
    pub fn expand(
        sizes: &ScheduleSizes,
        params: &ParamSet,
        mode: Mode,
        protocol: Protocol,
    ) -> Result<Self, ProtocolError> {
        Self::expand_with(sizes, params, mode, protocol, WEIGHTS_XI)
    }

    /// [`expand`](Self::expand) with the weights' proximity points placed at `weights_xi`.
    pub(crate) fn expand_with(
        sizes: &ScheduleSizes,
        params: &ParamSet,
        mode: Mode,
        protocol: Protocol,
        weights_xi: WeightsXi,
    ) -> Result<Self, ProtocolError> {
        if protocol != Protocol::One {
            return Err(ProtocolError::Internal {
                rule: "tessera-protocol/2's schedule adds its zero-knowledge labels (#148)",
            });
        }
        if !params.mode.allows(mode) {
            return Err(ProtocolError::Internal {
                rule: "the parameter set allows the proof's mode",
            });
        }
        if mode == Mode::NonInteractive
            && (params.seed_pow > MAX_VARS || params.query_pow > MAX_VARS)
        {
            return Err(ProtocolError::Internal {
                rule: "a proof of work asks at most 64 bits, which a u64 nonce can meet",
            });
        }
        sizes.validate(params)?;
        let mut steps = Steps::default();

        // Row 2.
        steps.absorb(Label::Round1);
        // Row 3.
        match mode {
            Mode::Live => steps.absorb(Label::Seed),
            Mode::NonInteractive => {
                steps.push(
                    Label::SeedPow,
                    Op::Grind {
                        phase: ProofOfWork::Seed,
                        bits: params.seed_pow,
                    },
                );
                steps.push(Label::Seed, Op::Digest);
            }
        }
        if weights_xi == WeightsXi::AfterSeed {
            for point in &sizes.proximity {
                steps.push(
                    Label::WeightsXi(point.matrix),
                    Op::Fp2 { count: point.vars },
                );
            }
        }
        // Row 4. `tessera-protocol/2` draws the masked LogUp identity sumcheck here, in place of
        // the roots (RFC-0032).
        steps.challenge(Label::LogUpAlpha);
        steps.challenge(Label::LogUpBeta);
        steps.absorb(Label::LogUpRoots);
        // Row 5.
        leaf_levels(&mut steps, &sizes.trees);
        steps.absorb(Label::LogUpValues);
        steps.absorb(Label::StageSplit { stage: 0 });
        // Row 6. `tessera-protocol/2` absorbs `stage/{k}/zk-sum` and squeezes `stage/{k}/zk-rho`
        // first in every stage (RFC-0013).
        for (k, stage) in (1..=u16::MAX).zip(&sizes.stages) {
            if k == 1 {
                for coord in 0..sizes.layer_vars {
                    steps.challenge(Label::StageLayer { coord });
                }
            }
            for coord in 0..stage.zero_vars {
                steps.challenge(Label::StageZero { stage: k, coord });
            }
            steps.challenge(Label::StageGamma { stage: k });
            for round in 0..stage.rounds {
                steps.absorb(Label::StageRound { stage: k, round });
                steps.challenge(Label::StageChallenge { stage: k, round });
            }
            steps.absorb(Label::StageEvals { stage: k });
            steps.absorb(Label::StageSplit { stage: k });
        }
        // Row 7. `tessera-protocol/2` absorbs the blinding claims first (RFC-0013, RFC-0012).
        let opened: Vec<WeightMatrix> = sizes
            .opened
            .iter()
            .filter_map(|codeword| WeightMatrix::from_id(codeword.matrix))
            .collect();
        steps.challenge(Label::AdviceGamma);
        steps.push(
            Label::AdviceXi,
            Op::Fp2 {
                count: sizes.advice_vars,
            },
        );
        for &matrix in &opened {
            steps.challenge(Label::WeightsGamma(matrix));
            if weights_xi == WeightsXi::Openings {
                let vars = sizes
                    .proximity
                    .iter()
                    .find(|p| p.matrix == matrix)
                    .map_or(0, |p| p.vars);
                steps.push(Label::WeightsXi(matrix), Op::Fp2 { count: vars });
            }
        }
        // Row 8.
        steps.absorb(Label::AdviceCombined);
        for &matrix in &opened {
            steps.absorb(Label::WeightsCombined(matrix));
        }
        // Row 9.
        steps.absorb(Label::Scoring);
        // Row 10.
        match mode {
            Mode::Live => steps.absorb(Label::Queries),
            Mode::NonInteractive => {
                steps.push(
                    Label::QueryPow,
                    Op::Grind {
                        phase: ProofOfWork::Query,
                        bits: params.query_pow,
                    },
                );
                let shape = sizes.query_shape(params);
                steps.push(
                    Label::AdviceQueries,
                    Op::Indices {
                        log_n: shape.advice_len.trailing_zeros(),
                        count: shape.advice_count,
                    },
                );
                for (codeword, &matrix) in sizes.opened.iter().zip(&opened) {
                    steps.push(
                        Label::WeightsQueries(matrix),
                        Op::Indices {
                            log_n: codeword.len.trailing_zeros(),
                            count: shape.matrix_count,
                        },
                    );
                }
            }
        }
        Ok(Self {
            protocol,
            mode,
            params: *params,
            steps: steps.0,
        })
    }

    /// The steps, in order.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// The labels, in order.
    pub fn labels(&self) -> impl Iterator<Item = Label> + '_ {
        self.steps.iter().map(|step| step.label)
    }

    /// The position of `label`'s step, if the schedule has it.
    pub fn position(&self, label: Label) -> Option<usize> {
        self.steps.iter().position(|step| step.label == label)
    }

    /// The protocol.
    pub fn protocol(&self) -> Protocol {
        self.protocol
    }

    /// The mode.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The parameter set.
    pub fn params(&self) -> &ParamSet {
        &self.params
    }
}

/// The steps of a schedule, as they are appended.
#[derive(Default)]
struct Steps(Vec<Step>);

impl Steps {
    fn push(&mut self, label: Label, op: Op) {
        self.0.push(Step { label, op });
    }

    fn absorb(&mut self, label: Label) {
        self.push(label, Op::Absorb);
    }

    /// One challenge in `Fp2`.
    fn challenge(&mut self, label: Label) {
        self.push(label, Op::Fp2 { count: 1 });
    }
}

/// The leaf stage's levels (RFC-0010 'Tree levels', RFC-0014 row 5): from the roots, level `ℓ`
/// takes the trees that have joined, a tree of height `h` joining at `H − h`.
fn leaf_levels(steps: &mut Steps, trees: &[LeafTree]) {
    let tallest = trees.iter().map(|t| t.height()).max().unwrap_or(0);
    for level in 0..tallest {
        // The trees that join here, and those already in: a tree of height 0 joins at `H`, past
        // the last level.
        let joins = |tree: &&LeafTree| tallest - tree.height() == level;
        let active = trees.iter().filter(|tree| tallest - tree.height() <= level);
        // The forests of more than one tree that join here share a root point, the last `s` of
        // `s_max` challenges.
        let roots = trees.iter().filter(joins).map(|t| t.trees_log()).max();
        for coord in 0..roots.unwrap_or(0) {
            steps.challenge(Label::TreeRoot { level, coord });
        }
        // `tessera-protocol/2` absorbs `tree/{ℓ}/zk-sum` and squeezes `tree/{ℓ}/zk-rho` here
        // (RFC-0032).
        steps.challenge(Label::TreeGamma { level });
        if active
            .clone()
            .any(|tree| matches!(tree, LeafTree::Fraction { .. }))
        {
            steps.challenge(Label::TreeLambda { level });
        }
        // At its own level `t = ℓ − (H − h)`, a tree is an instance on `s + t` variables.
        let rounds = active
            .map(|tree| tree.trees_log() + level - (tallest - tree.height()))
            .max()
            .unwrap_or(0);
        for round in 0..rounds {
            steps.absorb(Label::TreeRound { level, round });
            steps.challenge(Label::TreeChallenge { level, round });
        }
        steps.absorb(Label::TreeEvals { level });
        steps.challenge(Label::TreeLine { level });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MatrixCodeword;
    use crate::schedule::sizes::MatrixPoint;

    /// Two matrices with committed rows, of which the proof opens one.
    fn sizes() -> ScheduleSizes {
        ScheduleSizes {
            trees: Vec::new(),
            layer_vars: 0,
            stages: Vec::new(),
            advice_vars: 1,
            proximity: vec![
                MatrixPoint {
                    matrix: WeightMatrix::K,
                    vars: 2,
                },
                MatrixPoint {
                    matrix: WeightMatrix::LmHead,
                    vars: 3,
                },
            ],
            opened: vec![MatrixCodeword {
                matrix: WeightMatrix::LmHead.id(),
                len: 1 << 10,
            }],
        }
    }

    fn labels(placement: WeightsXi) -> Vec<String> {
        let schedule = TranscriptSchedule::expand_with(
            &sizes(),
            &ParamSet::TEST_SMALL,
            Mode::Live,
            Protocol::One,
            placement,
        )
        .expect("a schedule");
        schedule.labels().map(|l| l.to_string()).collect()
    }

    /// The one placement point moves the weights' proximity points and nothing else: after the
    /// seed for every matrix with committed rows (RFC-0049), or after each opened matrix's
    /// `θ_T` as RFC-0014 first wrote it.
    #[test]
    fn the_placement_moves_the_weights_points_alone() {
        let after_seed = labels(WeightsXi::AfterSeed);
        let openings = labels(WeightsXi::Openings);
        assert_eq!(
            after_seed,
            [
                "round1",
                "seed",
                "pcs/weights/K/xi",
                "pcs/weights/LmHead/xi",
                "logup/alpha",
                "logup/beta",
                "logup/roots",
                "logup/values",
                "stage/0/split",
                "pcs/advice/gamma",
                "pcs/advice/xi",
                "pcs/weights/LmHead/gamma",
                "pcs/advice/combined",
                "pcs/weights/LmHead/combined",
                "scoring",
                "queries",
            ]
        );
        assert_eq!(
            openings,
            [
                "round1",
                "seed",
                "logup/alpha",
                "logup/beta",
                "logup/roots",
                "logup/values",
                "stage/0/split",
                "pcs/advice/gamma",
                "pcs/advice/xi",
                "pcs/weights/LmHead/gamma",
                "pcs/weights/LmHead/xi",
                "pcs/advice/combined",
                "pcs/weights/LmHead/combined",
                "scoring",
                "queries",
            ]
        );
        // The crate's schedules follow the registry's one placement.
        let expanded =
            TranscriptSchedule::expand(&sizes(), &ParamSet::TEST_SMALL, Mode::Live, Protocol::One)
                .expect("a schedule");
        let placed = labels(WEIGHTS_XI);
        assert_eq!(
            expanded.labels().map(|l| l.to_string()).collect::<Vec<_>>(),
            placed
        );
    }
}
