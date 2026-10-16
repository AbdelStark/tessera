// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The model's projections at several contexts, and the `tessera.cost-model/1` JSON report that
//! `tessera bench model` writes (RFC-0020 'Executable cost model', 03 §5).

use serde_json::{Map, Number, Value};

use crate::error::{ModelError, Problem};
use crate::model::{
    Counts, MemoryModel, MuBreakdown, UnitCosts, detection_probability, mu, proof_time_s,
    sampling_rate, streaming_passes,
};

/// The report's schema identifier (03 §5).
pub const SCHEMA: &str = "tessera.cost-model/1";

/// The context lengths of the source's projections (source §5): 1K to 128K tokens, 1K being
/// 1,024.
pub const SOURCE_CONTEXTS: [u32; 8] = [
    1 << 10,
    1 << 11,
    1 << 12,
    1 << 13,
    1 << 14,
    1 << 15,
    1 << 16,
    1 << 17,
];

/// How the projections are used: the node a request is proven on, the batch it is proven in,
/// and the sampling policy whose rate and detection probability follow (source §8).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scenario {
    /// `G`: the GPUs that prove a request.
    pub gpus: u32,
    /// `T`: the tokens of a proof batch, which share its weight pass and its leaf levels.
    pub batch_tokens: u32,
    /// `η`: the ratio of ideal prefill time to serving time per token.
    pub eta: f64,
    /// The amortized overhead the sampling rate is set for.
    pub target_overhead: f64,
    /// The violating requests whose detection probability is reported.
    pub violations: u32,
}

impl Scenario {
    /// The source's: one 8×H100 node, batches of 16K tokens, `η = 0.3`, 10% amortized
    /// overhead, and 100 violating requests (source §8).
    pub const SOURCE: Self = Self {
        gpus: 8,
        batch_tokens: 16_384,
        eta: 0.3,
        target_overhead: 0.1,
        violations: 100,
    };
}

/// The model's outputs at one context.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projection {
    /// `S`: tokens of context.
    pub context: u32,
    /// The proof multiplier, term by term.
    pub mu: MuBreakdown,
    /// The streamed passes over the attention leaves of a batch, `c(S)`.
    pub streaming_passes: u32,
    /// Seconds to prove one request of `S` tokens on the scenario's GPUs.
    pub proof_time_s: f64,
    /// The sampling rate for the scenario's target overhead, in `[0, 1]`.
    pub sampling_rate: f64,
    /// The probability that a sample at that rate selects one of the scenario's violating
    /// requests.
    pub detection: f64,
}

/// Projections of one model at several contexts, with the inputs they rest on.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelReport {
    /// Whether the counts and unit costs are the source's.
    pub source_defaults: bool,
    /// The counts.
    pub counts: Counts,
    /// The unit costs.
    pub unit_costs: UnitCosts,
    /// The memory model of the streamed passes.
    pub memory: MemoryModel,
    /// How the projections are used.
    pub scenario: Scenario,
    /// One projection per context, in the order given.
    pub projections: Vec<Projection>,
}

impl ModelReport {
    /// The source's model (source §5 and §8): its counts, unit costs and scenario, with the
    /// memory of one 8×H100 node at 405B, at each of `contexts`.
    ///
    /// # Errors
    ///
    /// As [`ModelReport::new`].
    pub fn source(contexts: &[u32]) -> Result<Self, ModelError> {
        Self::new(
            true,
            Counts::SOURCE_405B,
            UnitCosts::SOURCE,
            MemoryModel::NODE_405B,
            Scenario::SOURCE,
            contexts,
        )
    }

    /// The projections of a model at each of `contexts`: [`mu`], [`streaming_passes`],
    /// [`proof_time_s`] from the total, then [`sampling_rate`] and [`detection_probability`].
    ///
    /// # Errors
    ///
    /// [`ModelError::Input`] for no context or an input one of those refuses, and
    /// [`ModelError::Output`] for an output that is not finite.
    pub fn new(
        source_defaults: bool,
        counts: Counts,
        unit_costs: UnitCosts,
        memory: MemoryModel,
        scenario: Scenario,
        contexts: &[u32],
    ) -> Result<Self, ModelError> {
        if contexts.is_empty() {
            return Err(ModelError::Input {
                name: "contexts",
                problem: Problem::Empty,
            });
        }
        let projections = contexts
            .iter()
            .map(|&context| {
                let breakdown = mu(
                    &counts,
                    &unit_costs,
                    &memory,
                    context,
                    scenario.batch_tokens,
                )?;
                let rate = sampling_rate(scenario.target_overhead, breakdown.total, scenario.eta)?;
                Ok(Projection {
                    context,
                    mu: breakdown,
                    streaming_passes: streaming_passes(
                        &counts,
                        &memory,
                        context,
                        scenario.batch_tokens,
                    )?,
                    proof_time_s: proof_time_s(
                        breakdown.total,
                        &counts,
                        &unit_costs,
                        context,
                        scenario.gpus,
                    )?,
                    sampling_rate: rate,
                    detection: detection_probability(rate, scenario.violations)?,
                })
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        Ok(Self {
            source_defaults,
            counts,
            unit_costs,
            memory,
            scenario,
            projections,
        })
    }

    /// The report as a `tessera.cost-model/1` JSON object, two-space indented, its keys sorted:
    /// `inputs`, one entry of `projections` per context in order, and `schema`. Every number is
    /// finite.
    ///
    /// # Errors
    ///
    /// [`ModelError::Output`] for a number that is not finite, which [`ModelReport::new`] never
    /// produces but a report built by hand can hold.
    pub fn to_json(&self) -> Result<String, ModelError> {
        let c = &self.counts;
        let u = &self.unit_costs;
        let m = &self.memory;
        let s = &self.scenario;
        let counts = object([
            (
                "commits_per_token",
                number("commits_per_token", c.commits_per_token)?,
            ),
            (
                "lookups_per_token",
                number("lookups_per_token", c.lookups_per_token)?,
            ),
            (
                "scores_per_token_per_ctx",
                number("scores_per_token_per_ctx", c.scores_per_token_per_ctx)?,
            ),
            (
                "native_ops_lin",
                number("native_ops_lin", c.native_ops_lin)?,
            ),
            (
                "native_ops_att_per_ctx",
                number("native_ops_att_per_ctx", c.native_ops_att_per_ctx)?,
            ),
            (
                "regen_full_equivalents",
                number("regen_full_equivalents", c.regen_full_equivalents)?,
            ),
        ]);
        let unit_costs = object([
            ("commit_ps", number("commit_ps", u.commit_ps)?),
            ("lookup_ps", number("lookup_ps", u.lookup_ps)?),
            (
                "product_leaf_ps",
                number("product_leaf_ps", u.product_leaf_ps)?,
            ),
            ("glue_fraction", number("glue_fraction", u.glue_fraction)?),
            (
                "weight_pass_gpu_s",
                number("weight_pass_gpu_s", u.weight_pass_gpu_s)?,
            ),
            (
                "native_ops_per_s",
                number("native_ops_per_s", u.native_ops_per_s)?,
            ),
            (
                "streaming_pass_ps",
                number("streaming_pass_ps", u.streaming_pass_ps)?,
            ),
            (
                "column_answer_ps",
                number("column_answer_ps", u.column_answer_ps)?,
            ),
        ]);
        let memory = object([
            ("free_bytes", number("free_bytes", m.free_bytes)?),
            (
                "bytes_per_leaf",
                number("bytes_per_leaf", m.bytes_per_leaf)?,
            ),
        ]);
        let scenario = object([
            ("gpus", Value::from(s.gpus)),
            ("batch_tokens", Value::from(s.batch_tokens)),
            ("eta", number("eta", s.eta)?),
            (
                "target_overhead",
                number("target_overhead", s.target_overhead)?,
            ),
            ("violations", Value::from(s.violations)),
        ]);
        let projections = self
            .projections
            .iter()
            .map(|p| {
                let mu = object([
                    ("linear", number("linear", p.mu.linear)?),
                    ("attention", number("attention", p.mu.attention)?),
                    ("weights", number("weights", p.mu.weights)?),
                    ("regen", number("regen", p.mu.regen)?),
                    ("streaming", number("streaming", p.mu.streaming)?),
                    (
                        "column_answers",
                        number("column_answers", p.mu.column_answers)?,
                    ),
                    ("total", number("total", p.mu.total)?),
                ]);
                Ok(object([
                    ("context", Value::from(p.context)),
                    ("mu", mu),
                    ("streaming_passes", Value::from(p.streaming_passes)),
                    ("proof_time_s", number("proof_time_s", p.proof_time_s)?),
                    ("sampling_rate", number("sampling_rate", p.sampling_rate)?),
                    ("detection", number("detection", p.detection)?),
                ]))
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        let report = object([
            ("schema", Value::from(SCHEMA)),
            (
                "inputs",
                object([
                    ("source_defaults", Value::from(self.source_defaults)),
                    ("counts", counts),
                    ("unit_costs", unit_costs),
                    ("memory", memory),
                    ("scenario", scenario),
                ]),
            ),
            ("projections", Value::Array(projections)),
        ]);
        // Every value is a string, a boolean, an integer or a finite number: serializing cannot
        // fail, and an empty string would only hide a bug.
        serde_json::to_string_pretty(&report).map_err(|_| ModelError::Output { name: "report" })
    }
}

/// A finite number, or the error that names it.
fn number(name: &'static str, value: f64) -> Result<Value, ModelError> {
    Number::from_f64(value)
        .map(Value::Number)
        .ok_or(ModelError::Output { name })
}

/// An object with `fields` in order.
fn object<const N: usize>(fields: [(&str, Value); N]) -> Value {
    let mut map = Map::new();
    for (key, value) in fields {
        map.insert(key.to_owned(), value);
    }
    Value::Object(map)
}
