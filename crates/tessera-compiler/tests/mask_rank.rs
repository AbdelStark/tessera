// SPDX-License-Identifier: Apache-2.0 OR MIT

//! R10's rank check on the compiled schedule (`docs/zk/simulator.md` §9 R10, #488): the
//! adversarial controls first, each a schedule that breaks exactly one rule, then the regression
//! control of RFC-0013's amendment on the 405B schedule, and the positive controls.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tessera_codec::{Ascii, DecodeLimits, Rational, envelope, schema};
use tessera_compiler::schedule::masks::{
    ColumnRule, Finding, MaskRow, MaskSystem, PairFunctional, PairRole, PairSystem, Role, SEED,
    Src, Var, check, check_at, check_claim_list, extract, protocol2_length, weight_layout,
};
use tessera_compiler::schedule::{
    Coord, Kind, Layers, Request, Source, StageSchedule, derive_composed_schedule, model_natives,
};
use tessera_compiler::{
    Lowering, ModelAdviceLayout, PublicModel, Selection, analyze, claim_list, lower, select,
};
use tessera_ir::{
    AdviceLayoutParams, ArchFamily, ArchSpec, ClaimList, CommitmentClass, CommitmentLayout,
    GadgetPolicy, ParamSetId, Profile, PublicQuantSpec, QuantPoint, RopeSpec, Site, TensorId,
    WeightLayoutParams,
};
use tessera_pcs::WeightMatrix;

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors")
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
        .collect()
}

/// The golden claim list of conformance model `name` (`spec/vectors/claims/claim-lists.json`).
fn golden(name: &str) -> ClaimList {
    let path = vectors().join("claims/claim-lists.json");
    let text = fs::read_to_string(&path).expect("the claim lists");
    let file: Value = serde_json::from_str(&text).expect("JSON");
    let case = file["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"].as_str() == Some(name))
        .expect("the model's claim list");
    let bytes = hex(case["hex"].as_str().expect("hex"));
    envelope::open(&bytes, &schema::CLAIM_LIST, DecodeLimits::MESSAGE).expect("a claim list")
}

/// A claim list's lowering at `n` positions, its selection and its composed schedule for
/// `request`: what the prover proves.
struct Compiled {
    claims: ClaimList,
    lowering: Lowering,
    selection: Selection,
    schedule: StageSchedule,
}

fn compiled(claims: ClaimList, n: u32, request: Request) -> Compiled {
    let lowering = lower(PublicModel::of_claims(&claims), n).expect("a lowering");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let selection = select(
        &lowering.graph,
        &ranges,
        &lowering.tables,
        &claims.gadget_policy,
    )
    .expect("a selection");
    let natives = model_natives(&lowering, request).expect("natives");
    let schedule = derive_composed_schedule(&lowering, &natives.deferred, &natives.native)
        .expect("a schedule");
    Compiled {
        claims,
        lowering,
        selection,
        schedule,
    }
}

impl Compiled {
    /// R10's functionals of `schedule`, a schedule of this lowering, under `rule`.
    fn system(&self, schedule: &StageSchedule, rule: ColumnRule) -> MaskSystem {
        let advice =
            ModelAdviceLayout::new(&self.lowering, &self.selection, &self.claims).expect("layout");
        let weights = weight_layout(&self.claims, &self.lowering).expect("weights");
        extract(&self.lowering, schedule, &advice, &weights, rule).expect("the functionals")
    }

    /// The pool claim whose single atom is on `tensor`'s kind, layer 0's tensor being `tensor`.
    fn pool_claim_on(&self, tensor: TensorId) -> usize {
        self.schedule
            .pool
            .iter()
            .position(|c| {
                c.atoms.len() == 1
                    && self
                        .schedule
                        .layers
                        .tensor(&self.lowering.graph, c.atoms[0].kind, 0)
                        == Some(tensor)
            })
            .expect("a pool claim on the tensor")
    }
}

/// A coordinate no challenge of the schedule sets: leaf point coordinates far past every site's.
fn fresh(j: u32) -> Coord {
    Coord::Leaf {
        site: 0,
        index: 1000 + j,
    }
}

fn dependent_rows(findings: &[Finding]) -> Vec<MaskRow> {
    findings
        .iter()
        .filter_map(|f| match f {
            Finding::Dependent { row, .. } => Some(*row),
            _ => None,
        })
        .collect()
}

/// The negative control of the issue: two values on one mask row at one column point, whose
/// row parts differ, here only in their layer weights. `LayerInput` at 64 positions has
/// `n_X = 12 ≥ log2 k = 10`, so its layer coordinate is in the row part. The values differ, so
/// the dedupe must not merge them, and their one functional reveals their difference.
#[test]
fn two_values_at_one_column_point_are_refused() {
    let c = compiled(golden("tiny-2-servable"), 64, Request::Generate);
    let mut schedule = c.schedule.clone();
    let index = schedule
        .pool
        .iter()
        .position(|p| {
            p.commitment == CommitmentClass::Advice
                && p.atoms.len() == 1
                && p.atoms[0].kind == Kind::LayerInput
        })
        .expect("a claim on the layer input");
    let mut twin = schedule.pool[index].clone();
    let Layers::Eq(rho) = &twin.atoms[0].layers else {
        panic!("a stacked claim");
    };
    twin.atoms[0].layers = Layers::Eq((0..rho.len() as u32).map(fresh).collect());
    schedule.pool.push(twin);

    let honest = check(&c.system(&c.schedule, ColumnRule::BelowRow), SEED);
    assert_eq!(honest, Vec::new(), "the compiled schedule passes");
    let findings = check(&c.system(&schedule, ColumnRule::BelowRow), SEED);
    let [
        Finding::Dependent {
            row, functionals, ..
        },
    ] = findings.as_slice()
    else {
        panic!("one dependency: {findings:?}");
    };
    assert_eq!(*row, MaskRow::Stacked(Kind::LayerInput));
    assert_eq!(functionals.len(), 2);
    assert_eq!(functionals[0].point, functionals[1].point);
    assert_ne!(functionals[0].key, functionals[1].key);
}

/// A row part that differs in the tensor's own variables: the one-hot tokens `[n, vocab]` at
/// 4 positions have 11 variables, the highest in the row part.
#[test]
fn a_value_that_differs_in_its_row_part_alone_is_refused() {
    let c = compiled(golden("tiny-2-servable"), 4, Request::Generate);
    let mut schedule = c.schedule.clone();
    let index = c.pool_claim_on(c.lowering.one_hot);
    let mut twin = schedule.pool[index].clone();
    let last = twin.point.len() - 1;
    twin.point[last] = fresh(0);
    twin.atoms[0].point[last] = fresh(0);
    schedule.pool.push(twin);
    let findings = check(&c.system(&schedule, ColumnRule::BelowRow), SEED);
    assert_eq!(
        dependent_rows(&findings),
        vec![MaskRow::Tensor(c.lowering.one_hot)],
        "{findings:?}"
    );
}

/// The negative control on a virtual row: a second proof of a virtual kind at the same point
/// but other layer weights. tiny-2-servable's stacked output at position 42 has `n_V = 10 + 1`,
/// so under the amended rule its mask takes 9 columns and the layer coordinate is not among
/// them: the two values share a functional, and differ.
#[test]
fn two_virtual_values_at_one_column_point_are_refused() {
    let c = compiled(golden("tiny-2-servable"), 4, Request::Generate);
    let kind = Kind::Layer {
        position: 42,
        output: 0,
    };
    let mut schedule = c.schedule.clone();
    let stage = &mut schedule.stages[1];
    let index = stage
        .instances
        .iter()
        .position(|i| i.kind == kind && matches!(i.source, Source::Part { .. }))
        .expect("an instance proving the kind");
    let mut twin = stage.instances[index].clone();
    let Layers::Eq(rho) = &twin.layers else {
        panic!("a stacked instance");
    };
    twin.layers = Layers::Eq((0..rho.len() as u32).map(fresh).collect());
    stage.instances.push(twin);
    let findings = check(&c.system(&schedule, ColumnRule::BelowRow), SEED);
    let [
        Finding::Dependent {
            row, functionals, ..
        },
    ] = findings.as_slice()
    else {
        panic!("one dependency: {findings:?}");
    };
    assert_eq!(*row, MaskRow::Virtual(kind));
    let roles: Vec<Role> = functionals.iter().map(|f| f.role).collect();
    assert_eq!(roles, [Role::Value, Role::Value]);
}

/// Every site's leaf claim lies at a prefix of one leaf point `P` (RFC-0045 'The leaf point',
/// the prover's `lookup.point()[..n]`): tiny-2-affine's V and Down multipliers (t28, t72) are
/// read at two sites, and their claims share `P[0..5]`.
#[test]
fn leaf_claims_of_every_site_share_the_leaf_point() {
    let (c, _) = affine();
    let site = |t| match c.schedule.pool[c.pool_claim_on(TensorId(t))].point[0] {
        Coord::Leaf { site, .. } => site,
        other => panic!("a leaf claim: {other:?}"),
    };
    assert_ne!(site(28), site(72));
    let system = c.system(&c.schedule, ColumnRule::BelowRow);
    let blinding = |t| {
        pair(&system, WeightMatrix::ChannelParams)
            .functionals
            .iter()
            .find(|f| f.tensor == Some(TensorId(t)))
            .map(|f| f.point.clone())
            .expect("the claim's blinding claim")
    };
    let shared: Vec<Src> = (0..5).map(|i| Src::Var(Var::Leaf(i))).collect();
    assert_eq!(blinding(28)[..5], shared[..]);
    assert_eq!(blinding(72)[..5], shared[..]);
}

/// The `ChannelParams` claims of tiny-2-affine's affine sites: the type tensor of Q's
/// multipliers (t22), its claim at `x = [R1.2..R1.7] ‖ ρ = [R1.10]`.
fn affine() -> (Compiled, usize) {
    let c = compiled(golden("tiny-2-affine"), 4, Request::Generate);
    let q = c.pool_claim_on(TensorId(22));
    (c, q)
}

fn pair(system: &MaskSystem, matrix: WeightMatrix) -> &PairSystem {
    system
        .pairs
        .iter()
        .find(|p| p.matrix == matrix)
        .expect("the matrix's pair")
}

/// RFC-0065 'Zero knowledge': a value of a type tensor at another point `y` whose column point is
/// another claim's carries `β_T`'s functional of that claim, so it lies in the span; with another
/// row point it carries a secret term, and is refused.
#[test]
fn a_secret_value_in_the_span_of_beta_is_refused() {
    let (c, q) = affine();
    let mut schedule = c.schedule.clone();
    let mut y = schedule.pool[q].clone();
    y.atoms[0].layers = Layers::Eq(vec![fresh(0)]);
    schedule.pool.push(y);
    let system = c.system(&schedule, ColumnRule::BelowRow);
    let values: Vec<&PairFunctional> = pair(&system, WeightMatrix::ChannelParams)
        .functionals
        .iter()
        .filter(|f| f.role != PairRole::Blinding)
        .collect();
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].role, PairRole::Value { secret: true });
    let findings = check(&system, SEED);
    let spanned: Vec<_> = findings
        .iter()
        .filter_map(|f| match f {
            Finding::Spanned {
                matrix,
                value,
                span,
            } => Some((*matrix, value.tensor, span.len())),
            _ => None,
        })
        .collect();
    // Its span is Q's blinding claim, or O's at the same column point (t50).
    assert_eq!(
        spanned,
        vec![(WeightMatrix::ChannelParams, Some(TensorId(22)), 1)],
        "{findings:?}"
    );
}

/// The symbolic `S ≡ 0` decision (RFC-0065 'Zero knowledge'): `y` keeps `x_i`'s row point and
/// `Z(y) ≡ Z(x_i)`, `Z` over every coordinate but the lowest. Each case changes one of the two.
#[test]
fn the_secret_term_is_decided_by_the_row_point_and_z() {
    let (c, q) = affine();
    let x = c.schedule.pool[q].clone();
    let Layers::Eq(rho) = &x.atoms[0].layers else {
        panic!("a stacked claim");
    };
    let rho0 = rho[0];
    // The lowest coordinate, which Z spares: no secret term (the pair-bit points of the leaf).
    let mut lowest = x.clone();
    lowest.point[0] = fresh(0);
    lowest.atoms[0].point[0] = fresh(0);
    // A coordinate Z multiplies: a secret term.
    let mut second = x.clone();
    second.point[1] = fresh(1);
    second.atoms[0].point[1] = fresh(1);
    // Z's coordinates kept as a multiset, the row point changed: a channel coordinate swapped
    // with the layer coordinate, which tiny-2-affine's layout puts in the row part.
    let mut swapped = x.clone();
    let channel = swapped.atoms[0].point[1];
    swapped.point[1] = rho0;
    swapped.atoms[0].point[1] = rho0;
    swapped.atoms[0].layers = Layers::Eq(vec![channel]);
    for (y, secret) in [(lowest, false), (second, true), (swapped, true)] {
        let mut schedule = c.schedule.clone();
        schedule.pool.push(y);
        let system = c.system(&schedule, ColumnRule::BelowRow);
        let roles: Vec<PairRole> = pair(&system, WeightMatrix::ChannelParams)
            .functionals
            .iter()
            .map(|f| f.role)
            .filter(|r| *r != PairRole::Blinding)
            .collect();
        assert_eq!(roles, vec![PairRole::Value { secret }]);
        // Each column point is new: none of them lies in the span.
        assert_eq!(check(&system, SEED), Vec::new());
    }
}

fn eq_point(coords: &[Src]) -> Vec<Src> {
    coords.to_vec()
}

fn var(i: u32) -> Src {
    Src::Var(Var::Zero(i))
}

/// The pair's pass on a structured span, as the leaf level's pair-bit points give one: `q` at
/// `(a, b)` and values at `(0, b)` and `(1, b)`, with `eq(a, b) = (1 − a)·eq(0, b) + a·eq(1, b)`.
/// Values without a secret term join the span and are never refused; one with a secret term in
/// it is.
#[test]
fn a_value_without_a_secret_term_joins_the_span_unrefused() {
    let functional = |role, point: Vec<Src>| PairFunctional {
        role,
        key: point.clone(),
        point,
        tensor: Some(TensorId(1)),
        origins: Vec::new(),
    };
    let system = |secret| MaskSystem {
        rows: Vec::new(),
        pairs: vec![PairSystem {
            matrix: WeightMatrix::ChannelParams,
            width: 2,
            functionals: vec![
                functional(PairRole::Blinding, eq_point(&[var(0), var(1)])),
                functional(
                    PairRole::Value { secret: false },
                    eq_point(&[Src::Zero, var(1)]),
                ),
                functional(PairRole::Value { secret }, eq_point(&[Src::One, var(1)])),
            ],
        }],
    };
    assert_eq!(check(&system(false), SEED), Vec::new());
    let findings = check(&system(true), SEED);
    let [Finding::Spanned { span, .. }] = findings.as_slice() else {
        panic!("one refusal: {findings:?}");
    };
    assert_eq!(span.len(), 2, "q and the value at (0, b)");
}

/// The positive control: two `ChannelParams` claims at one column point, each with its own mask
/// row, are accepted. tiny-2-affine's Q and O multipliers (t22, t50) are read at stage 1's point
/// `[R1.2..R1.7] ‖ [R1.10]`, and `ChannelParams`' layout puts the layer and type bits in the row
/// part, so `q_Q = q_O`, a public relation.
#[test]
fn channel_params_claims_at_one_column_point_are_accepted() {
    let (c, q) = affine();
    let o = c.pool_claim_on(TensorId(50));
    let system = c.system(&c.schedule, ColumnRule::BelowRow);
    let blinding = |t| {
        pair(&system, WeightMatrix::ChannelParams)
            .functionals
            .iter()
            .find(|f| f.role == PairRole::Blinding && f.tensor == Some(TensorId(t)))
            .map(|f| f.point.clone())
            .expect("the claim's blinding claim")
    };
    assert_eq!(blinding(22), blinding(50), "q_Q = q_O");
    for index in [q, o] {
        let row = MaskRow::Channel(c.schedule.pool[index].atoms[0].kind);
        let own = system
            .rows
            .iter()
            .find(|r| r.row == row)
            .expect("the claim's own row");
        assert_eq!(own.functionals.len(), 1);
    }
    assert_eq!(check(&system, SEED), Vec::new());
}

/// Identical values are one value: duplicate pool atoms, at one point with one layer weight,
/// are accepted. The compiled schedule has many already (t7, `RsqrtRow`'s output, is claimed
/// 32 times at one point); every pool claim is repeated here besides.
#[test]
fn identical_duplicate_pool_atoms_are_accepted() {
    let c = compiled(golden("tiny-2-servable"), 4, Request::Generate);
    let mut schedule = c.schedule.clone();
    let copies = schedule.pool.clone();
    schedule.pool.extend(copies.iter().cloned());
    schedule.pool.extend(copies);
    let system = c.system(&schedule, ColumnRule::BelowRow);
    let t7 = system
        .rows
        .iter()
        .find(|r| r.tensor == Some(TensorId(7)))
        .expect("RsqrtRow's output");
    assert_eq!(t7.functionals.len(), 3 * 32);
    assert_eq!(check(&system, SEED), Vec::new());
}

/// RFC-0047's layout order, not the natural one: a committed `[n, d_ff]` tensor, `d_ff = 192`
/// padded, puts its padded axis highest, so its column sub-point holds the position bits
/// first. tiny-2-servable's `[4, 192]` stacked output at position 44 fills its row of 2^10.
#[test]
fn committed_tensors_take_the_layout_order() {
    let c = compiled(golden("tiny-2-servable"), 4, Request::Generate);
    let system = c.system(&c.schedule, ColumnRule::BelowRow);
    let kind = Kind::Layer {
        position: 44,
        output: 0,
    };
    let tensor = c
        .schedule
        .layers
        .tensor(&c.lowering.graph, kind, 0)
        .expect("the tensor");
    assert_eq!(c.lowering.graph.tensors[tensor.0 as usize].shape, [4, 192]);
    let row = system
        .rows
        .iter()
        .find(|r| r.row == MaskRow::Stacked(kind))
        .expect("its row");
    for f in &row.functionals {
        // The key is the natural point: 8 channel bits, 2 position bits, the layer bit.
        let mut expected = f.key[8..10].to_vec();
        expected.extend(&f.key[..8]);
        assert_eq!(f.point, expected);
    }
}

/// A reduction stage with a fold has `max(n_max, log2 k)` rounds in `tessera-protocol/2`, its
/// instances front-padded (RFC-0013 'The stage's rounds'): at one position tiny-2-servable's
/// stage 1 has 9 compiled rounds, so its fold takes a fresh round 0 and the compiled rounds
/// move up by one.
#[test]
fn a_stage_below_log2_k_rounds_gains_fresh_leading_rounds() {
    let c = compiled(golden("tiny-2-servable"), 1, Request::Generate);
    assert_eq!(c.schedule.stages[1].rounds, 9);
    let system = c.system(&c.schedule, ColumnRule::BelowRow);
    let round = |round| Src::Var(Var::Round { stage: 1, round });
    let folds: Vec<&Vec<Src>> = system
        .rows
        .iter()
        .flat_map(|r| &r.functionals)
        .filter(|f| f.role == Role::Fold && f.point.first() == Some(&round(0)))
        .map(|f| &f.point)
        .collect();
    assert!(!folds.is_empty());
    assert!(
        folds
            .iter()
            .all(|p| **p == (0..10).map(round).collect::<Vec<_>>())
    );
    // Every stage-1 coordinate a value names is a compiled round shifted by one.
    let lowest = system
        .rows
        .iter()
        .flat_map(|r| &r.functionals)
        .filter(|f| f.role == Role::Value)
        .flat_map(|f| &f.key)
        .filter_map(|s| match s {
            Src::Var(Var::Round { stage: 1, round }) => Some(*round),
            _ => None,
        })
        .min();
    assert_eq!(lowest, Some(1), "no value reads the fresh round");
}

/// Tensors of fewer than two variables cannot carry RFC-0013's mask, and are their own finding.
/// The control is a schedule at one position, a token axis `tessera-protocol/2` never lays out
/// ([`protocol2_length`]): there the tokens `[1]` have no variable.
#[test]
fn tensors_of_fewer_than_two_variables_are_unmaskable() {
    let c = compiled(golden("tiny-2-servable"), 1, Request::Generate);
    let findings = check(&c.system(&c.schedule, ColumnRule::BelowRow), SEED);
    assert!(
        findings
            .iter()
            .all(|f| matches!(f, Finding::Unmaskable { vars, .. } if *vars < 2))
    );
    assert!(findings.iter().any(|f| matches!(
        f,
        Finding::Unmaskable { row: MaskRow::Tensor(t), vars: 0, .. } if *t == c.lowering.tokens
    )));
    let four = compiled(golden("tiny-2-servable"), 4, Request::Generate);
    assert_eq!(
        check(&four.system(&four.schedule, ColumnRule::BelowRow), SEED),
        Vec::new()
    );
}

/// `tessera-protocol/2` pads the token axis to `2^max(⌈log2 n⌉, 2)` rows (RFC-0013 'Masked
/// extensions', RFC-0071 'The slots'): lengths 1 to 4 are checked at 4, longer ones at their own,
/// and a sequence of one position is then accepted with the revealed points of four.
#[test]
fn the_token_axis_pads_to_four_rows() {
    let lengths: Vec<u32> = [1, 2, 3, 4, 5, 8, 9, 131_072]
        .map(protocol2_length)
        .to_vec();
    assert_eq!(lengths, [4, 4, 4, 4, 5, 8, 9, 131_072]);
    let claims = golden("tiny-2-servable");
    for n in [1, 2, 3] {
        let report = check_claim_list(&claims, n, Request::Generate, ColumnRule::BelowRow)
            .expect("the schedule");
        assert_eq!((report.n, report.length), (n, 4));
        assert_eq!(report.findings, Vec::new());
        let four = check_at(&claims, 4, Request::Generate, ColumnRule::BelowRow).expect("4");
        assert_eq!(
            report.system, four.system,
            "n = {n} reveals the points of n = 4"
        );
    }
    let one = check_at(&claims, 1, Request::Generate, ColumnRule::BelowRow).expect("1");
    assert!(
        !one.findings.is_empty(),
        "unpadded, one position is unmaskable"
    );
}

const ROPE: RopeSpec = RopeSpec {
    theta: 500_000,
    factor: 8,
    low_freq_factor: 1,
    high_freq_factor: 4,
    original_max: 8192,
    bits: 14,
};

/// Llama-3.1-405B's claim list, servable with two shift classes at every site after a weight,
/// for `ni-100/1`'s layout: `crates/tessera-bench/tests/commitment_report.rs`'s `llama_405b`
/// and `the_default_statements_merge_sizes_are_the_405b_lowerings`, copied.
fn llama_405b() -> ClaimList {
    const LAYERS: u32 = 126;
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
        rms_eps: Rational::new(1, 100_000).expect("a rational"),
    };
    let eps = (u64::from(d) * (1 << 14)).div_ceil(100_000);
    let kappa = (0..=31u8)
        .rev()
        .find(|&k| (u128::from(d) << (2 * k)) / u128::from(eps) < 1 << 64)
        .expect("κ = 0 fits");
    let norm = i8::try_from(kappa).expect("small") - 3;
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
            .map(|l| (l, Rational::new(1, 1 << 14).expect("a scale")))
            .collect(),
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
    let lowering = lower(model, 16).expect("405B's shape lowers");
    let ranges = analyze(&lowering.graph, &lowering.tables).expect("ranges");
    let policy = GadgetPolicy {
        max_log2_rows: 32,
        ..GadgetPolicy::default()
    };
    let selection = select(&lowering.graph, &ranges, &lowering.tables, &policy).expect("fits");
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
    let params = ParamSetId(Ascii::new("ni-100/1").expect("ASCII"));
    claim_list(model, &selection, &policy, layout, params).expect("a claim list")
}

/// The regression control of RFC-0013's #488 amendment. At 405B and 32,768 positions the
/// `SumSquares` outputs at layer positions 0 and 38 (t6, t53) have `n_V = 15 + 7 = 22 = log2 k`.
/// Stage 1 reveals their `V'` at its last 22 challenges, the point of their own folds. Under the
/// old rule, `c = log2 k`, the two functionals are one, which the check refuses; under the
/// amended rule, `c = log2 k − 1`, the schedule is accepted.
#[test]
fn the_405b_schedule_folds_below_a_full_row() {
    let claims = llama_405b();
    let old = check_claim_list(&claims, 32_768, Request::Generate, ColumnRule::Row)
        .expect("the schedule");
    let refused: Vec<(MaskRow, Option<TensorId>)> = old
        .findings
        .iter()
        .map(|f| match f {
            Finding::Dependent {
                row,
                tensor,
                functionals,
            } => {
                let roles: Vec<Role> = functionals.iter().map(|f| f.role).collect();
                assert_eq!(roles, [Role::Fold, Role::Value], "{f}");
                assert_eq!(functionals[0].point, functionals[1].point, "{f}");
                let last22: Vec<Src> = (22..44)
                    .map(|round| Src::Var(Var::Round { stage: 1, round }))
                    .collect();
                assert_eq!(functionals[0].point, last22, "{f}");
                (*row, *tensor)
            }
            other => panic!("only fold dependencies: {other}"),
        })
        .collect();
    let sum_squares = |position| {
        MaskRow::Virtual(Kind::Layer {
            position,
            output: 0,
        })
    };
    assert_eq!(
        refused,
        [
            (sum_squares(0), Some(TensorId(6))),
            (sum_squares(38), Some(TensorId(53)))
        ]
    );
    let amended = check_claim_list(&claims, 32_768, Request::Generate, ColumnRule::BelowRow)
        .expect("the schedule");
    assert_eq!(amended.findings, Vec::new());
}
