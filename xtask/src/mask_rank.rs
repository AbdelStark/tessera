// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cargo xtask check-mask-rank`: R10 on the compiled schedule (`docs/zk/simulator.md` §9 R10,
//! #488). Every mask row's revealed functionals must be linearly independent, every secret value
//! of a matrix's pair `β_T` must lie outside the span of the pair's other functionals (RFC-0065
//! 'Zero knowledge'), and every revealed tensor must have the two variables RFC-0013's mask
//! needs. `tessera_compiler::schedule::masks` runs the check, under RFC-0013 as #488 amended it
//! (`ColumnRule::BelowRow`) and with committed tensors' mask columns as #554 fixed them
//! (`CommittedColumns::RULE`), on the composed schedule the prover proves; its documentation
//! lists what the compiled schedule does not hold yet.
//!
//! The cases:
//! - every claim list of `spec/vectors/claims/claim-lists.json`, at the lengths
//!   `spec/vectors/schedule/` covers (1, 4, 5 and `max_context`, `schedule_vectors.rs`) and 3,
//!   the shortest layout of `tessera-protocol/2`, for each request: none, a prefill and a
//!   generation (`Request`);
//! - Llama-3.1-405B's claim list, servable with two shift classes at every site after a weight,
//!   for `ni-100/1`'s layout (advice `log2 k = 22`, weights 20; the fixture of
//!   `crates/tessera-bench/tests/commitment_report.rs`, copied in [`llama_405b`]), at
//!   `n = 2^j` and `2^j + 1` for `j = 0..17` within `max_context`, for each request. The
//!   lengths cover every `⌈log2 n⌉`, on which the schedule's points depend: RFC-0013's amendment
//!   comes from `n ∈ (16384, 32768]`.
//!
//! Each length `n` is checked at the token-axis layout `tessera-protocol/2` gives it,
//! `masks::protocol2_length(n) = max(n, 3)` (RFC-0013 'Masked extensions', RFC-0047, RFC-0071
//! 'The slots'), and reported with both. A committed tensor's mask columns are its data claim's
//! column sub-point (RFC-0013 'Committed tensors', #554): a stacked kind's layer variables and an
//! attention tile's index bits are row bits, so two values that differ only there share one
//! functional, which the check refuses. Cases of one claim list, checked length and request share
//! one check, and the checks run on up to [`WORKERS`] threads.

use std::fs;
use std::io;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use serde_json::Value;
use tessera_codec::{Ascii, DecodeLimits, Rational, envelope, schema};
use tessera_compiler::schedule::masks::{
    ColumnRule, CommittedColumns, MaskReport, check_schedule, protocol2_length,
};
use tessera_compiler::schedule::{Request, StageSchedule, derive_composed_schedule, model_natives};
use tessera_compiler::{Lowering, PublicModel, analyze, claim_list, lower, select};
use tessera_ir::{
    AdviceLayoutParams, ArchFamily, ArchSpec, ClaimList, CommitmentLayout, GadgetPolicy,
    ParamSetId, Profile, PublicQuantSpec, QuantPoint, RopeSpec, Site, TensorId, WeightLayoutParams,
};

use crate::claim_vectors;

/// The requests each conformance claim list is checked for.
const REQUESTS: [Request; 3] = [Request::None, Request::Prefill, Request::Generate];

/// The most checks run at once: a 405B check holds about 0.6 GB.
const WORKERS: usize = 8;

/// Committed tensors' mask columns, as RFC-0013 'Committed tensors' fixes them: the data claim's
/// column sub-point (#554).
const COMMITTED: CommittedColumns = CommittedColumns::RULE;

/// One schedule to check: a claim list, a length and a request, and where the claim list is.
#[derive(Clone)]
pub(crate) struct Case {
    /// Where the claim list comes from: a file and line.
    pub(crate) source: String,
    /// The model's name.
    pub(crate) name: String,
    pub(crate) claims: ClaimList,
    /// The sequence's positions.
    pub(crate) n: u32,
    /// The length its schedule is derived at: `protocol2_length(n)`.
    pub(crate) length: u32,
    pub(crate) request: Request,
}

fn hex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| {
            text.get(i..i + 2)
                .and_then(|b| u8::from_str_radix(b, 16).ok())
        })
        .collect()
}

/// The conformance claim lists of the tree at `repo`, each at the lengths 1, 3, 4, 5 and its
/// `max_context`, for every request.
pub(crate) fn vector_cases(repo: &Path) -> Result<Vec<Case>, String> {
    let path = claim_vectors::PATH;
    let text = fs::read_to_string(repo.join(path)).map_err(|e| format!("{path}: {e}"))?;
    let file: Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    let entries = file["cases"]
        .as_array()
        .ok_or_else(|| format!("{path}: no `cases` array (RFC-0007 'Testing Strategy')"))?;
    let mut cases = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        let name = entry["name"]
            .as_str()
            .ok_or_else(|| format!("{path}: case {index} has no name"))?;
        let line = text
            .find(&format!("\"name\": \"{name}\""))
            .map_or(0, |at| text[..at].matches('\n').count() + 1);
        let source = format!("{path}:{line}");
        let bytes = entry["hex"]
            .as_str()
            .and_then(hex)
            .ok_or_else(|| format!("{source}: case `{name}` has no hex claim list"))?;
        let claims: ClaimList = envelope::open(&bytes, &schema::CLAIM_LIST, DecodeLimits::MESSAGE)
            .map_err(|e| format!("{source}: case `{name}` is no claim list (RFC-0007): {e}"))?;
        for n in [1, 3, 4, 5, claims.arch.max_context] {
            for request in REQUESTS {
                cases.push(Case {
                    source: source.clone(),
                    name: name.to_owned(),
                    claims: claims.clone(),
                    n,
                    length: protocol2_length(n),
                    request,
                });
            }
        }
    }
    Ok(cases)
}

/// `2^j` and `2^j + 1` for `j = 0..`, within `max`, each once.
pub(crate) fn lengths(max: u32) -> Vec<u32> {
    let mut lengths: Vec<u32> = (0..u32::BITS)
        .filter_map(|j| 1u32.checked_shl(j))
        .flat_map(|n| [n, n.saturating_add(1)])
        .filter(|&n| n <= max)
        .collect();
    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

const ROPE: RopeSpec = RopeSpec {
    theta: 500_000,
    factor: 8,
    low_freq_factor: 1,
    high_freq_factor: 4,
    original_max: 8192,
    bits: 14,
};

/// Where [`llama_405b`] is, for the gate's messages.
const LLAMA_405B: &str = concat!(file!(), ":", line!());

/// Llama-3.1-405B's claim list: servable, a residual scale of `2^{−7}`, two shift classes at
/// every site after a weight, `max_context` 131,072, and `ni-100/1`'s layout. It copies
/// `llama_405b(Profile::Servable, 2)` and the claim list of
/// `the_default_statements_merge_sizes_are_the_405b_lowerings` in
/// `crates/tessera-bench/tests/commitment_report.rs`.
pub(crate) fn llama_405b() -> Result<ClaimList, String> {
    const LAYERS: u32 = 126;
    let fail = |e: &dyn std::fmt::Display| format!("{LLAMA_405B}: the 405B claim list: {e}");
    let d = 16_384;
    let arch = ArchSpec {
        family: ArchFamily::LlamaV3,
        n_layers: LAYERS,
        d_model: d,
        n_heads: 128,
        n_kv_heads: 8,
        d_head: 128,
        d_ff: 53_248,
        vocab: 128_256,
        max_context: 131_072,
        rope: ROPE,
        rms_eps: Rational::new(1, 100_000).ok_or_else(|| fail(&"ε is a rational"))?,
    };
    // RMSNorm's e = ⌈d · 10^{−5} · 2^{14}⌉, and RFC-0023's κ.
    let eps = (u64::from(d) * (1 << 14)).div_ceil(100_000);
    let kappa = (0..=31u8)
        .rev()
        .find(|&k| (u128::from(d) << (2 * k)) / u128::from(eps) < 1 << 64)
        .ok_or_else(|| fail(&"κ = 0 fits"))?;
    let norm = i8::try_from(kappa).map_err(|e| fail(&e))? - 3;
    let sites: &[(Site, i8)] = &[
        (Site::AttnNorm, norm),
        (Site::RopeQ, 22),
        (Site::RopeK, 22),
        (Site::V, 8),
        (Site::AttnOut, 32),
        (Site::AttnResidual, 4),
        (Site::MlpNorm, norm),
        (Site::Silu, 20),
        (Site::Up, 8),
        (Site::Product, 5),
        (Site::MlpResidual, 4),
    ];
    let mut act_exponents = vec![(QuantPoint::Embed, -6)];
    let mut shift_class_sets = Vec::new();
    for layer in 0..LAYERS {
        for &(site, r) in sites {
            let point = QuantPoint::Layer { layer, site };
            let residual = matches!(site, Site::AttnResidual | Site::MlpResidual);
            if !residual {
                act_exponents.push((point, -3));
            }
            let after_a_weight = residual
                || matches!(
                    site,
                    Site::RopeQ | Site::RopeK | Site::V | Site::Silu | Site::Up
                );
            let count = if after_a_weight { 2 } else { 1 };
            shift_class_sets.push((point, (r..r + count).collect()));
        }
    }
    act_exponents.push((QuantPoint::FinalNorm, -3));
    shift_class_sets.push((QuantPoint::FinalNorm, vec![norm]));
    let quant = PublicQuantSpec {
        act_exponents,
        residual_exponent: -7,
        residual_bits: 20,
        rsqrt_kappa: kappa,
        rms_eps_int: eps,
        exp_sigma: 16,
        score_scales: (0..LAYERS)
            .map(|l| Rational::new(1, 1 << 14).map(|s| (l, s)))
            .collect::<Option<_>>()
            .ok_or_else(|| fail(&"a score scale is a rational"))?,
        shift_class_sets,
        affine_sites: Vec::new(),
        rope: ROPE,
        logits_bits: 28,
        logits_exponent: -11,
    };
    let silu: Vec<(u32, i8)> = (0..LAYERS).map(|l| (l, -6)).collect();
    let model = PublicModel {
        arch: &arch,
        profile: Profile::Servable,
        quant: &quant,
        silu_in_exponents: &silu,
    };
    let lowering = lower(model, 16).map_err(|e| fail(&e))?;
    let ranges = analyze(&lowering.graph, &lowering.tables).map_err(|e| fail(&e))?;
    let policy = GadgetPolicy {
        max_log2_rows: 32,
        ..GadgetPolicy::default()
    };
    let selection =
        select(&lowering.graph, &ranges, &lowering.tables, &policy).map_err(|e| fail(&e))?;
    let layout = CommitmentLayout {
        advice: AdviceLayoutParams {
            log2_k: 22,
            log2_inv_rate: 1,
            mask_coeffs: 200,
            row_block: 32,
            segments: 8,
        },
        weights: WeightLayoutParams {
            log2_k: 20,
            log2_inv_rate: 2,
            mask_coeffs: 65_536,
            blinding_rows: 4096,
        },
    };
    let params = ParamSetId(Ascii::new("ni-100/1").map_err(|_| fail(&"an ASCII identifier"))?);
    claim_list(model, &selection, &policy, layout, params).map_err(|e| fail(&e))
}

/// The 405B claim list's cases: every request at every length of [`lengths`].
pub(crate) fn llama_405b_cases(claims: &ClaimList) -> Vec<Case> {
    let mut cases = Vec::new();
    for n in lengths(claims.arch.max_context) {
        for request in REQUESTS {
            cases.push(Case {
                source: LLAMA_405B.to_owned(),
                name: "llama-3.1-405b".to_owned(),
                claims: claims.clone(),
                n,
                length: protocol2_length(n),
                request,
            });
        }
    }
    cases
}

/// `t` as the lowering builds it: its shape and what produces it.
fn describe(lowering: &Lowering, t: TensorId) -> String {
    let graph = &lowering.graph;
    let shape = graph
        .tensors
        .get(t.0 as usize)
        .map(|spec| format!("{:?}", spec.shape))
        .unwrap_or_default();
    let role = if t == lowering.tokens {
        "the tokens".to_owned()
    } else if t == lowering.logits {
        "the logits".to_owned()
    } else {
        match graph.nodes.iter().find(|n| n.outputs.contains(&t)) {
            Some(node) => {
                let op = format!("{:?}", node.op);
                let name: String = op
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                format!("{name} output, node {}", node.id.0)
            }
            None => "a graph input".to_owned(),
        }
    };
    format!("t{} {shape}: {role}", t.0)
}

/// The tensors a report's findings name, described.
fn named(report: &MaskReport, finding: &str) -> String {
    let mut names = Vec::new();
    for row in &report.system.rows {
        if let Some(t) = row.tensor
            && finding.contains(&format!("(t{})", t.0))
            && !names.contains(&t)
        {
            names.push(t);
        }
    }
    let described: Vec<String> = names
        .into_iter()
        .map(|t| describe(&report.lowering, t))
        .collect();
    described.join("; ")
}

/// One case's line and its violations, each naming the claim list's file and line, the case, the
/// finding and the rule it breaks.
pub(crate) struct Outcome {
    pub(crate) summary: String,
    pub(crate) violations: Vec<String>,
}

/// What one check finds: its system's counts (mask rows, functionals, folds, pair functionals),
/// and each finding with the tensors it names.
#[derive(Clone)]
struct Checked {
    counts: (usize, usize, usize, usize),
    findings: Vec<String>,
}

/// `report`'s counts and findings.
fn checked(report: &MaskReport) -> Checked {
    Checked {
        counts: report.counts(),
        findings: report
            .findings
            .iter()
            .map(|finding| {
                let text = finding.to_string();
                format!("{text} [{}]", named(report, &text))
            })
            .collect(),
    }
}

/// Where a case is: the claim list's file and line, the model, `n`, the checked length and the
/// request.
fn at(case: &Case) -> String {
    format!(
        "{} `{}` at n = {}, checked at {} ({:?})",
        case.source, case.name, case.n, case.length, case.request
    )
}

/// R10 on `case`'s composed schedule once `plant` has edited it, under `rule` and the reading
/// `committed`: the gate plants nothing, and its self-tests plant violations.
fn check_planted(
    case: &Case,
    rule: ColumnRule,
    committed: CommittedColumns,
    plant: impl FnOnce(&mut StageSchedule),
) -> Result<Checked, String> {
    let fail = |e: &dyn std::fmt::Display| format!("{}: the schedule's functionals: {e}", at(case));
    let claims = &case.claims;
    let lowering = lower(PublicModel::of_claims(claims), case.length).map_err(|e| fail(&e))?;
    let ranges = analyze(&lowering.graph, &lowering.tables).map_err(|e| fail(&e))?;
    let selection = select(
        &lowering.graph,
        &ranges,
        &lowering.tables,
        &claims.gadget_policy,
    )
    .map_err(|e| fail(&e))?;
    let natives = model_natives(&lowering, case.request).map_err(|e| fail(&e))?;
    let mut schedule = derive_composed_schedule(&lowering, &natives.deferred, &natives.native)
        .map_err(|e| fail(&e))?;
    plant(&mut schedule);
    let report = check_schedule(
        claims,
        lowering,
        &selection,
        &schedule,
        case.length,
        rule,
        &[committed],
    )
    .map_err(|e| fail(&e))?
    .pop()
    .ok_or_else(|| fail(&"no report"))?;
    Ok(checked(&report))
}

/// R10 on `case` under `rule` and the gate's columns, [`COMMITTED`], once `plant` has edited its
/// schedule: the gate plants nothing.
fn check_one(
    case: &Case,
    rule: ColumnRule,
    plant: impl FnOnce(&mut StageSchedule),
) -> Result<Checked, String> {
    check_planted(case, rule, COMMITTED, plant)
}

fn outcome(case: &Case, checked: &Checked) -> Outcome {
    let at = at(case);
    let (rows, functionals, folds, pairs) = checked.counts;
    Outcome {
        summary: format!(
            "{at}: {rows} mask rows, {functionals} functionals ({folds} folds), {pairs} pair \
             functionals, {} findings",
            checked.findings.len()
        ),
        violations: checked
            .findings
            .iter()
            .map(|finding| format!("{at}: {finding}"))
            .collect(),
    }
}

/// R10 on `case` under `rule`, at its checked length.
#[cfg(test)]
pub(crate) fn check_case(case: &Case, rule: ColumnRule) -> Result<Outcome, String> {
    Ok(outcome(case, &check_one(case, rule, |_| {})?))
}

/// R10 on every case under `rule`, the outcomes in the cases' order. Cases of one claim list,
/// checked length and request share one check, which up to [`WORKERS`] threads run.
pub(crate) fn check_all(cases: &[Case], rule: ColumnRule) -> Result<Vec<Outcome>, String> {
    let mut unique: Vec<usize> = Vec::new();
    let mut shared = Vec::with_capacity(cases.len());
    for (i, case) in cases.iter().enumerate() {
        let same = |&u: &usize| {
            let other = &cases[u];
            (&other.source, &other.name, other.length, other.request)
                == (&case.source, &case.name, case.length, case.request)
        };
        match unique.iter().position(same) {
            Some(j) => shared.push(j),
            None => {
                shared.push(unique.len());
                unique.push(i);
            }
        }
    }
    let workers = thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .clamp(1, WORKERS)
        .min(unique.len().max(1));
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Result<Checked, String>>>> =
        Mutex::new(unique.iter().map(|_| None).collect());
    thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&i) = unique.get(j) else { break };
                    let checked = check_one(&cases[i], rule, |_| {});
                    if let Ok(mut results) = results.lock() {
                        results[j] = Some(checked);
                    }
                }
            });
        }
    });
    let results = results
        .into_inner()
        .map_err(|_| "check-mask-rank: a check panicked".to_owned())?;
    cases
        .iter()
        .zip(shared)
        .map(|(case, j)| match &results[j] {
            Some(Ok(checked)) => Ok(outcome(case, checked)),
            Some(Err(error)) => Err(error.clone()),
            None => Err(format!("{}: not checked", at(case))),
        })
        .collect()
}

/// Runs the gate over the conformance claim lists of `repo` and the 405B claim list.
pub(crate) fn run(repo: &Path) -> io::Result<bool> {
    let mut cases = vector_cases(repo).map_err(io::Error::other)?;
    let claims = llama_405b().map_err(io::Error::other)?;
    cases.extend(llama_405b_cases(&claims));
    let outcomes = check_all(&cases, ColumnRule::BelowRow).map_err(io::Error::other)?;
    let mut failed = 0usize;
    let mut findings = 0usize;
    for outcome in &outcomes {
        let status = if outcome.violations.is_empty() {
            "ok"
        } else {
            failed += 1;
            "FAIL"
        };
        findings += outcome.violations.len();
        println!("{status:<6}{}", outcome.summary);
        for violation in &outcome.violations {
            println!("      {violation}");
        }
    }
    println!(
        "check-mask-rank: {} cases, {failed} failed, {findings} findings (R10, #488)",
        cases.len()
    );
    Ok(failed == 0)
}

#[cfg(test)]
mod tests {
    use super::{
        COMMITTED, Case, ColumnRule, CommittedColumns, Outcome, Request, StageSchedule, check_all,
        check_case, check_one, check_planted, lengths, llama_405b, llama_405b_cases, outcome,
        vector_cases,
    };
    use crate::claim_vectors;
    use crate::repo;
    use crate::testing::Fixture;
    use serde_json::Value;
    use std::fs;
    use tessera_compiler::schedule::masks::{MaskRow, Origin, Src, check_at};
    use tessera_compiler::schedule::{Coord, Kind, Layers};
    use tessera_ir::CommitmentClass;

    /// A fixture tree whose claim-list file holds the repository's case `name` alone.
    fn one_case(name: &str) -> Fixture {
        let text = fs::read_to_string(repo::root().join(claim_vectors::PATH)).expect("the file");
        let mut file: Value = serde_json::from_str(&text).expect("JSON");
        let case = file["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|c| c["name"].as_str() == Some(name))
            .cloned()
            .expect("the case");
        file["cases"] = Value::Array(vec![case]);
        let fixture = Fixture::new("mask-rank");
        fixture.write(
            claim_vectors::PATH,
            &serde_json::to_string_pretty(&file).expect("JSON"),
        );
        fixture
    }

    /// Every conformance claim list, at the lengths of `spec/vectors/schedule/`, for every
    /// request.
    #[test]
    fn the_cases_are_the_schedule_vectors_lengths_for_every_request() {
        let root = repo::root();
        let cases = vector_cases(&root).expect("the claim lists");
        let schedules: Value = serde_json::from_str(
            &fs::read_to_string(root.join("spec/vectors/schedule/schedule.json")).expect("file"),
        )
        .expect("JSON");
        let mut expected: Vec<(String, u32)> = schedules["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .map(|c| {
                let n = c["n"].as_u64().expect("n");
                (c["model"].as_str().expect("a model").to_owned(), n as u32)
            })
            .collect();
        expected.sort();
        let mut found: Vec<(String, u32)> = cases
            .iter()
            .filter(|c| c.request == Request::Generate)
            .map(|c| (c.name.clone(), c.n))
            .collect();
        // And 3, the shortest layout of `tessera-protocol/2`, for each model.
        let models: Vec<String> = expected.iter().map(|(m, _)| m.clone()).collect();
        for model in models {
            if !expected.contains(&(model.clone(), 3)) {
                expected.push((model, 3));
            }
        }
        expected.sort();
        found.sort();
        assert_eq!(found, expected);
        assert_eq!(cases.len(), 3 * expected.len());
    }

    /// `2^0 + 1 = 2^1`, and `2^17 + 1` is past `max_context`: 34 lengths.
    #[test]
    fn the_405b_lengths_cover_every_ceil_log2() {
        let n = lengths(131_072);
        assert_eq!(n.len(), 34);
        assert_eq!(&n[..6], [1, 2, 3, 4, 5, 8]);
        assert_eq!(n.last(), Some(&131_072));
        for j in 1..=17u32 {
            assert!(n.contains(&(1 << j)) && (j == 17 || n.contains(&((1 << j) + 1))));
        }
    }

    /// Lengths up to three are checked at three, the layout `tessera-protocol/2` gives them, a
    /// token axis padded to four rows laid out as a padded axis (RFC-0013 'Masked extensions',
    /// RFC-0047); longer ones at their own. The 405B claim list is checked for every request.
    #[test]
    fn lengths_one_to_three_are_checked_at_three() {
        let cases = vector_cases(&repo::root()).expect("the claim lists");
        for case in &cases {
            assert_eq!(case.length, case.n.max(3), "{}", case.name);
        }
        let claims = llama_405b().expect("the 405B claim list");
        let big = llama_405b_cases(&claims);
        assert_eq!(big.len(), 3 * 34);
        let generate: Vec<(u32, u32)> = big
            .iter()
            .filter(|c| c.request == Request::Generate)
            .map(|c| (c.n, c.length))
            .take(6)
            .collect();
        assert_eq!(generate, [(1, 3), (2, 3), (3, 3), (4, 4), (5, 5), (8, 8)]);
        for request in [Request::None, Request::Prefill, Request::Generate] {
            assert_eq!(big.iter().filter(|c| c.request == request).count(), 34);
        }
    }

    /// The clean case: a conformance claim list passes at its protocol/2 lengths, one position
    /// included, and each case reports its own `n` beside the length it shares.
    #[test]
    fn a_clean_claim_list_passes() {
        let fixture = one_case("tiny-2-affine");
        let cases = vector_cases(fixture.path()).expect("the fixture's claim list");
        assert_eq!(cases.len(), 15);
        let short: Vec<Case> = cases.into_iter().filter(|c| c.n <= 4).collect();
        let outcomes = check_all(&short, ColumnRule::BelowRow).expect("checked");
        assert_eq!(outcomes.len(), 9);
        for (case, outcome) in short.iter().zip(&outcomes) {
            assert_eq!(
                outcome.violations,
                Vec::<String>::new(),
                "{}",
                outcome.summary
            );
            let shown = format!(
                "at n = {}, checked at {} ({:?})",
                case.n,
                case.n.max(3),
                case.request
            );
            assert!(outcome.summary.contains(&shown), "{}", outcome.summary);
            // The system's counts, and its findings: none.
            assert!(
                outcome.summary.contains(" mask rows, ")
                    && outcome.summary.ends_with(", 0 findings"),
                "{}",
                outcome.summary
            );
        }
    }

    /// A planted violation of the unmaskable kind: a case still under two variables at its
    /// checked length, the token axis left at one row as `tessera-protocol/1` lays it out. The
    /// tokens have no variable, and the message names the file and line, the case, both lengths,
    /// the tensor and the rule.
    #[test]
    fn a_tensor_without_two_variables_fails_naming_it() {
        let fixture = one_case("tiny-2-affine");
        let cases = vector_cases(fixture.path()).expect("the fixture's claim list");
        let case = cases
            .iter()
            .find(|c| c.n == 1 && c.request == Request::Generate)
            .expect("n = 1");
        let unpadded = Case {
            length: 1,
            ..case.clone()
        };
        let outcome = check_case(&unpadded, ColumnRule::BelowRow).expect("checked");
        let tokens = outcome
            .violations
            .iter()
            .find(|v| v.contains("t0 [1]: the tokens"))
            .unwrap_or_else(|| panic!("the tokens: {:?}", outcome.violations));
        for part in [
            "spec/vectors/claims/claim-lists.json:",
            "`tiny-2-affine` at n = 1, checked at 1 (Generate)",
            "revealed with 0 variable(s)",
            "RFC-0013",
        ] {
            assert!(tokens.contains(part), "{part}: {tokens}");
        }
    }

    /// A planted violation of the rank kind: RFC-0013 before #488 on the 405B claim list, whose
    /// schedule at 32,768 positions folds the `SumSquares` outputs t6 and t53 at the point of
    /// their own values.
    #[test]
    fn the_old_column_rule_fails_on_the_405b_claim_list() {
        let claims = llama_405b().expect("the 405B claim list");
        assert_eq!(
            (claims.layout.advice.log2_k, claims.layout.weights.log2_k),
            (22, 20)
        );
        let cases = llama_405b_cases(&claims);
        let case = cases
            .iter()
            .find(|c| c.n == 32_768 && c.request == Request::Generate)
            .expect("n = 32,768");
        let old = check_case(case, ColumnRule::Row).expect("checked");
        assert_eq!(old.violations.len(), 2, "{:?}", old.violations);
        for (violation, t) in old.violations.iter().zip(["t6", "t53"]) {
            for part in [
                "xtask/src/mask_rank.rs:",
                "`llama-3.1-405b` at n = 32768, checked at 32768 (Generate)",
                "linearly dependent (R10",
                "fold at [R1.22",
                &format!("({t})"),
                &format!("{t} [32768]: SumSquares output"),
            ] {
                assert!(violation.contains(part), "{part}: {violation}");
            }
        }
        let amended = check_case(case, ColumnRule::BelowRow).expect("checked");
        assert_eq!(amended.violations, Vec::<String>::new());
    }

    /// The gate's outcome for `case` once `plant` has planted a violation in its schedule: the
    /// gate's own check, at RFC-0013 as #488 amended it.
    fn planted(case: &Case, plant: impl FnOnce(&mut StageSchedule)) -> Outcome {
        let checked = check_one(case, ColumnRule::BelowRow, plant).expect("checked");
        outcome(case, &checked)
    }

    /// The same under #554's rejected reading, a stacked kind's layer variables and a tile's
    /// index bits among a committed tensor's mask columns: the control.
    fn rejected_reading(case: &Case, plant: impl FnOnce(&mut StageSchedule)) -> Outcome {
        let checked = check_planted(
            case,
            ColumnRule::BelowRow,
            CommittedColumns::TensorOrder,
            plant,
        )
        .expect("checked");
        outcome(case, &checked)
    }

    /// A coordinate no challenge of the schedule sets: leaf point coordinates far past every
    /// site's.
    fn fresh(j: u32) -> Coord {
        Coord::Leaf {
            site: 0,
            index: 1000 + j,
        }
    }

    /// Adds a twin of the advice pool claim whose one atom is on `kind`, at fresh layer
    /// coordinates: a value at the claim's column sub-point and another layer point.
    fn twin_at_another_layer(kind: Kind) -> impl Fn(&mut StageSchedule) {
        move |schedule| {
            let index = schedule
                .pool
                .iter()
                .position(|p| {
                    p.commitment == CommitmentClass::Advice
                        && p.atoms.len() == 1
                        && p.atoms[0].kind == kind
                })
                .expect("a pool claim on the kind");
            let mut twin = schedule.pool[index].clone();
            let Layers::Eq(rho) = &twin.atoms[0].layers else {
                panic!("a stacked claim");
            };
            twin.atoms[0].layers = Layers::Eq((0..rho.len() as u32).map(fresh).collect());
            schedule.pool.push(twin);
        }
    }

    /// RFC-0013's rule refuses what it forbids (#554): a stacked committed kind's mask takes its
    /// data claim's column sub-point, not its layer variables, so a value at a claim's column
    /// sub-point and another layer point shares the claim's functional. On tiny-2-servable at 4
    /// positions the layer input has 8 variables of its own and one layer bit, below
    /// `log2 k = 10`. The gate's rule refuses the planted twin, naming the file and line, the
    /// case, the row and the rule; the rejected reading, which takes the layer bit as a column,
    /// accepts it.
    #[test]
    fn a_stacked_value_at_another_layer_point_fails_naming_it() {
        let fixture = one_case("tiny-2-servable");
        let cases = vector_cases(fixture.path()).expect("the fixture's claim list");
        let case = cases
            .iter()
            .find(|c| c.n == 4 && c.request == Request::Generate)
            .expect("n = 4");
        let plant = twin_at_another_layer(Kind::LayerInput);
        // The compiled schedule passes.
        assert_eq!(planted(case, |_| {}).violations, Vec::<String>::new());
        let refused = planted(case, &plant);
        let [violation] = refused.violations.as_slice() else {
            panic!("one violation: {:?}", refused.violations);
        };
        for part in [
            "spec/vectors/claims/claim-lists.json:",
            "`tiny-2-servable` at n = 4, checked at 4 (Generate)",
            "stacked advice kind LayerInput",
            "2 revealed functionals are linearly dependent (R10",
            "RFC-0013",
        ] {
            assert!(violation.contains(part), "{part}: {violation}");
        }
        assert!(
            refused.summary.ends_with(", 1 findings"),
            "{}",
            refused.summary
        );
        let control = rejected_reading(case, &plant);
        assert_eq!(control.violations, Vec::<String>::new());
    }

    /// The same at 405B, where the readings differ on real tensors: the `RsqrtRow` outputs
    /// `[16384]` at layer position 1 have 14 variables and 7 layer bits, below `log2 k = 22`.
    #[test]
    fn a_405b_stacked_value_at_another_layer_point_fails() {
        let claims = llama_405b().expect("the 405B claim list");
        let cases = llama_405b_cases(&claims);
        let case = cases
            .iter()
            .find(|c| c.n == 16_384 && c.request == Request::Generate)
            .expect("n = 16,384");
        let roots = Kind::Layer {
            position: 1,
            output: 0,
        };
        let plant = twin_at_another_layer(roots);
        let refused = planted(case, &plant);
        let [violation] = refused.violations.as_slice() else {
            panic!("one violation: {:?}", refused.violations);
        };
        for part in [
            "`llama-3.1-405b` at n = 16384, checked at 16384 (Generate)",
            "stacked advice kind Layer { position: 1, output: 0 } (t7)",
            "linearly dependent (R10",
            "t7 [16384]: RsqrtRow output",
        ] {
            assert!(violation.contains(part), "{part}: {violation}");
        }
        let control = rejected_reading(case, &plant);
        assert_eq!(control.violations, Vec::<String>::new());
    }

    /// An attention tile smaller than a row (RFC-0034, #554): tiny-2-servable's `SoftmaxRow`
    /// reciprocals at 4 positions are two tiles, one per KV head, each of 3 variables. The bit
    /// that selects the tile is a row bit, not a mask variable: a twin of a compiled read that
    /// differs in that bit alone is refused by the gate's rule and accepted by the rejected
    /// reading.
    #[test]
    fn a_value_in_another_tile_fails_naming_it() {
        let fixture = one_case("tiny-2-servable");
        let cases = vector_cases(fixture.path()).expect("the fixture's claim list");
        let case = cases
            .iter()
            .find(|c| c.n == 4 && c.request == Request::Generate)
            .expect("n = 4");
        let reciprocals = Kind::Layer {
            position: 27,
            output: 1,
        };
        // The compiled read on the reciprocals, and the coordinate of its point that the tile's
        // column sub-point leaves out: the tile bit.
        let report = check_at(
            &case.claims,
            case.length,
            case.request,
            ColumnRule::BelowRow,
            COMMITTED,
        )
        .expect("checked");
        assert_eq!(report.findings, Vec::new(), "the compiled schedule passes");
        let value = report
            .system
            .rows
            .iter()
            .find(|r| r.row == MaskRow::Stacked(reciprocals))
            .and_then(|r| r.functionals.first())
            .expect("a read on the reciprocals");
        let &Origin::Pool { claim, .. } = value.origins.first().expect("an origin") else {
            panic!("a pool claim: {value:?}");
        };
        let natural = value.key.len() - 1;
        let columns = value.point.iter().filter(|s| **s != Src::Zero).count();
        assert_eq!((natural, columns), (4, 3), "{value:?}");
        let tile = (0..natural)
            .find(|&j| !value.point.contains(&value.key[j]))
            .expect("a coordinate outside the column sub-point");
        let plant = |schedule: &mut StageSchedule| {
            let mut twin = schedule.pool[claim].clone();
            assert_eq!(twin.atoms[0].kind, reciprocals);
            twin.point[tile] = fresh(0);
            twin.atoms[0].point[tile] = fresh(0);
            schedule.pool.push(twin);
        };
        let refused = planted(case, plant);
        let [violation] = refused.violations.as_slice() else {
            panic!("one violation: {:?}", refused.violations);
        };
        for part in [
            "`tiny-2-servable` at n = 4, checked at 4 (Generate)",
            "stacked advice kind Layer { position: 27, output: 1 }",
            "linearly dependent (R10",
            "SoftmaxRow output",
        ] {
            assert!(violation.contains(part), "{part}: {violation}");
        }
        let control = rejected_reading(case, plant);
        assert_eq!(control.violations, Vec::<String>::new());
    }

    /// A claim-list file the gate cannot read is an error naming the file and the case.
    #[test]
    fn a_malformed_claim_list_is_an_error_naming_it() {
        let fixture = Fixture::new("mask-rank-bad");
        let missing = vector_cases(fixture.path()).err().expect("no file");
        assert!(missing.contains(claim_vectors::PATH), "{missing}");
        fixture.write(
            claim_vectors::PATH,
            "{\"cases\": [{\"name\": \"x\", \"hex\": \"zz\"}]}",
        );
        let bad_hex = vector_cases(fixture.path()).err().expect("bad hex");
        assert!(bad_hex.contains(":1: case `x` has no hex"), "{bad_hex}");
        fixture.write(
            claim_vectors::PATH,
            "{\"cases\": [\n{\"name\": \"x\", \"hex\": \"00ff\"}]}",
        );
        let not_a_claim_list = vector_cases(fixture.path()).err().expect("no claim list");
        assert!(
            not_a_claim_list.contains(":2: case `x` is no claim list (RFC-0007)"),
            "{not_a_claim_list}"
        );
    }
}
