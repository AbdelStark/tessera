// SPDX-License-Identifier: Apache-2.0 OR MIT

//! R10's rank check on the compiled schedule (`docs/zk/simulator.md` §9 R10, #488): the
//! adversarial controls first, each a schedule that breaks exactly one rule, then the regression
//! control of RFC-0013's amendment on the 405B schedule, and the positive controls.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tessera_codec::{Ascii, DecodeLimits, Rational, envelope, schema};
use tessera_compiler::schedule::masks::{
    ColumnRule, CommittedColumns, Finding, Functional, MaskReport, MaskRow, MaskSystem, Origin,
    PairFunctional, PairRole, PairSystem, Role, RowSystem, SEED, ScaledWeight, Src, Var, check,
    check_at, check_claim_list, extract, protocol2_length, weight_layout,
};
use tessera_compiler::schedule::{
    Coord, Factor, Kind, Layers, NodeRef, Request, Source, StageSchedule, derive_composed_schedule,
    model_natives,
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
    /// R10's functionals of `schedule`, a schedule of this lowering, under `rule`, with committed
    /// tensors' mask columns RFC-0013's, their data claims' column sub-points (#554).
    fn system(&self, schedule: &StageSchedule, rule: ColumnRule) -> MaskSystem {
        self.system_with(schedule, rule, CommittedColumns::RULE)
    }

    /// R10's functionals of `schedule` under `rule` and the reading `committed`.
    fn system_with(
        &self,
        schedule: &StageSchedule,
        rule: ColumnRule,
        committed: CommittedColumns,
    ) -> MaskSystem {
        let advice =
            ModelAdviceLayout::new(&self.lowering, &self.selection, &self.claims).expect("layout");
        let weights = weight_layout(&self.claims, &self.lowering).expect("weights");
        extract(&self.lowering, schedule, &advice, &weights, rule, committed)
            .expect("the functionals")
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

/// tiny-4-servable's residual key `acc + 2^r·X_in` reads the layer input with a per-layer
/// constant (8, 8, 16 and 2 at position 34): its pool atom on `LayerInput` weighs the layers by
/// `eq(ρ, ℓ)·c_ℓ` (RFC-0037 'Claims and atoms'), a value other than `X'(P ‖ ρ)`.
fn scaled_layer_input() -> (Compiled, usize) {
    let c = compiled(golden("tiny-4-servable"), 4, Request::Generate);
    let layers = &c.schedule.layers;
    let index = c
        .schedule
        .pool
        .iter()
        .position(|p| {
            p.atoms.iter().any(|a| {
                a.kind == Kind::LayerInput
                    && a.coefficient.iter().flatten().any(|f| {
                        matches!(f, Factor::Constant(NodeRef::Position(p)) if !layers.uniform(*p))
                    })
            })
        })
        .expect("the scaled read of the layer input");
    (c, index)
}

/// The review's E14: an unscaled read of the layer input at the scaled read's point and layer
/// weights. The two values differ by `Σ_ℓ eq(ρ, ℓ)·(c_ℓ − 1)·X̃_ℓ(P)` and read `R_LayerInput`
/// through one functional, so their difference is revealed; the merge must not take them for
/// one value.
#[test]
fn an_unscaled_twin_of_a_per_layer_scaled_read_is_refused() {
    let (c, index) = scaled_layer_input();
    let mut twin = c.schedule.pool[index].clone();
    let atom = twin
        .atoms
        .iter_mut()
        .find(|a| a.kind == Kind::LayerInput)
        .expect("the atom");
    atom.coefficient = vec![Vec::new()];
    let mut schedule = c.schedule.clone();
    schedule.pool.push(twin.clone());
    let findings = check(&c.system(&schedule, ColumnRule::BelowRow), SEED);
    assert_eq!(
        dependent_rows(&findings),
        vec![MaskRow::Stacked(Kind::LayerInput)],
        "{findings:?}"
    );
    // The control: the same twin at a fresh layer point. The layer input fills its row of 2^10
    // here, so the layer point is in the row part and the twin shares the scaled read's
    // functional: it is refused whether or not the merge keys on the coefficient, so the twin
    // at the scaled read's own point was accepted by the merge alone.
    let atom = twin
        .atoms
        .iter_mut()
        .find(|a| a.kind == Kind::LayerInput)
        .expect("the atom");
    let Layers::Eq(rho) = &atom.layers else {
        panic!("a stacked read");
    };
    atom.layers = Layers::Eq((0..rho.len() as u32).map(fresh).collect());
    let mut schedule = c.schedule.clone();
    schedule.pool.push(twin);
    let control = check(&c.system(&schedule, ColumnRule::BelowRow), SEED);
    assert_eq!(
        dependent_rows(&control),
        vec![MaskRow::Stacked(Kind::LayerInput)],
        "{control:?}"
    );
    // The scaled read alone, as compiled, is accepted.
    assert_eq!(
        check(&c.system(&c.schedule, ColumnRule::BelowRow), SEED),
        Vec::new()
    );
}

/// RFC-0013 'Committed tensors' (#556): a stacked advice piece's mask coefficient in
/// `tessera-protocol/2` is `a_t` when `c_t(0) = … = c_t(L − 1) = a_t` (`0` when `a_t = 0`), and
/// `1` when `c_t(ℓ)` differs between real layers (`P'_t(r ‖ ρ) = Σ_ℓ eq(ρ, ℓ)·c_t(ℓ)·X̃_{κ,ℓ}(r)
/// + Z(r ‖ ρ)·R̃_{X_κ}(r_col)`). Classification and R10 merging therefore depend on the evaluated
/// per-layer polynomial `(c_t(0), …, c_t(L − 1))`, not on raw AST provenance:
/// 1. canceling paths (`c_ℓ − c_ℓ = 0`) contribute `0` to both data and mask rows;
/// 2. non-uniform paths summing to a uniform scalar (`[8, 8, 16, 2] + [2, 2, -6, 8] = [10; 4]`)
///    are uniform (`10·X'_in`), merging with an unscaled read at `(r ‖ ρ)` and refusing a
///    non-uniform twin there;
/// 3. another template position with the exact same per-layer constants `[8, 8, 16, 2]` (even
///    written through `Pow2(2)·LT([½, ½], 3)` in place of `ConstMul(3)`) merges at `(r ‖ ρ)`,
///    while `−c_ℓ` or `2·c_ℓ` at `(r ‖ ρ)` is refused because `(D + Z·R̃)` and `(−D + Z·R̃)` or
///    `(2D + Z·R̃)` unmask `D`.
#[test]
fn per_layer_scaled_advice_pieces_classify_and_merge_by_layer_polynomial() {
    let (base, index) = scaled_layer_input();
    let p1 = base.schedule.pool[index]
        .atoms
        .iter()
        .flat_map(|a| a.coefficient.iter().flatten())
        .find_map(|f| match f {
            Factor::Constant(NodeRef::Position(p)) if !base.schedule.layers.uniform(*p) => Some(*p),
            _ => None,
        })
        .expect("position 34");
    let p2 = (0..base.schedule.layers.width)
        .find(|&p| {
            p != p1
                && matches!(
                    base.lowering.graph.nodes[base.schedule.layers.node(p, 0).0 as usize].op,
                    tessera_ir::Op::ConstMul { .. }
                )
        })
        .expect("a second ConstMul position in the layer template");
    let p1_values: Vec<i64> = (0..base.schedule.layers.count)
        .map(|l| {
            let id = base.schedule.layers.node(p1, l);
            let tessera_ir::Op::ConstMul { c } = base.lowering.graph.nodes[id.0 as usize].op else {
                panic!("ConstMul");
            };
            base.lowering.graph.constants[c.0 as usize][0]
        })
        .collect();
    assert_eq!(p1_values, vec![8, 8, 16, 2]);

    let advice =
        ModelAdviceLayout::new(&base.lowering, &base.selection, &base.claims).expect("layout");
    let weights = weight_layout(&base.claims, &base.lowering).expect("weights");
    let system_of = |lowering: &Lowering, schedule: &StageSchedule| {
        extract(
            lowering,
            schedule,
            &advice,
            &weights,
            ColumnRule::BelowRow,
            CommittedColumns::RULE,
        )
        .expect("the functionals")
    };
    let with_p2 = |values: [i64; 4]| {
        let mut lowering = base.lowering.clone();
        for (l, v) in values.into_iter().enumerate() {
            let id = base.schedule.layers.node(p2, l as u32);
            let c_ref = tessera_ir::ConstRef(lowering.graph.constants.len() as u32);
            lowering.graph.constants.push(vec![v]);
            lowering.graph.nodes[id.0 as usize].op = tessera_ir::Op::ConstMul { c: c_ref };
        }
        lowering
    };

    // 1. Canceling paths `c_ℓ − c_ℓ = 0` at a fresh layer point reveal no functional of R_X.
    {
        let mut schedule = base.schedule.clone();
        let mut zero_twin = schedule.pool[index].clone();
        let atom = zero_twin
            .atoms
            .iter_mut()
            .find(|a| a.kind == Kind::LayerInput)
            .expect("the atom");
        let Layers::Eq(rho) = &atom.layers else {
            panic!("a stacked read");
        };
        atom.layers = Layers::Eq((0..rho.len() as u32).map(fresh).collect());
        atom.coefficient = vec![
            vec![Factor::Constant(NodeRef::Position(p1))],
            vec![Factor::Minus, Factor::Constant(NodeRef::Position(p1))],
        ];
        schedule.pool.push(zero_twin);
        assert_eq!(
            check(&system_of(&base.lowering, &schedule), SEED),
            Vec::new(),
            "a zero-coefficient piece contributes 0 to both data and mask rows"
        );
    }

    // 2. Non-uniform paths summing to `[10, 10, 10, 10]` are uniform (`scale = None`).
    {
        let lowering = with_p2([2, 2, -6, 8]);
        let mut schedule = base.schedule.clone();
        schedule.pool[index]
            .atoms
            .iter_mut()
            .find(|a| a.kind == Kind::LayerInput)
            .expect("the atom")
            .coefficient = vec![
            vec![Factor::Constant(NodeRef::Position(p1))],
            vec![Factor::Constant(NodeRef::Position(p2))],
        ];
        let mut unscaled_twin = schedule.pool[index].clone();
        unscaled_twin
            .atoms
            .iter_mut()
            .find(|a| a.kind == Kind::LayerInput)
            .expect("the atom")
            .coefficient = vec![Vec::new()];
        schedule.pool.push(unscaled_twin);
        assert_eq!(
            check(&system_of(&lowering, &schedule), SEED),
            Vec::new(),
            "paths summing to [10, 10, 10, 10] are uniform and merge with an unscaled read"
        );

        // And a genuinely non-uniform `[8, 8, 16, 2]` twin at the same point is refused.
        let mut schedule_nonuniform = base.schedule.clone();
        schedule_nonuniform.pool[index]
            .atoms
            .iter_mut()
            .find(|a| a.kind == Kind::LayerInput)
            .expect("the atom")
            .coefficient = vec![
            vec![Factor::Constant(NodeRef::Position(p1))],
            vec![Factor::Constant(NodeRef::Position(p2))],
        ];
        let mut scaled_twin = base.schedule.pool[index].clone();
        scaled_twin
            .atoms
            .iter_mut()
            .find(|a| a.kind == Kind::LayerInput)
            .expect("the atom")
            .coefficient = vec![vec![Factor::Constant(NodeRef::Position(p1))]];
        schedule_nonuniform.pool.push(scaled_twin);
        assert_eq!(
            dependent_rows(&check(&system_of(&lowering, &schedule_nonuniform), SEED)),
            vec![MaskRow::Stacked(Kind::LayerInput)]
        );
    }

    // 3. Another position with identical `[8, 8, 16, 2]` merges (including `4 · LT([½, ½], 3)` = 3
    //    vs `ConstMul(3)`), whereas `−c_ℓ` and `2·c_ℓ` are refused.
    {
        let mut lowering = with_p2([8, 8, 16, 2]);
        let three_ref = tessera_ir::ConstRef(lowering.graph.constants.len() as u32);
        lowering.graph.constants.push(vec![3]);
        let three_node = tessera_ir::NodeId(lowering.graph.nodes.len() as u32);
        lowering.graph.nodes.push(tessera_ir::Node {
            id: three_node,
            op: tessera_ir::Op::ConstMul { c: three_ref },
            inputs: vec![lowering.tokens],
            outputs: vec![lowering.tokens],
        });

        let mut schedule = base.schedule.clone();
        schedule.pool[index]
            .atoms
            .iter_mut()
            .find(|a| a.kind == Kind::LayerInput)
            .expect("the atom")
            .coefficient = vec![vec![
            Factor::Constant(NodeRef::Node(three_node)),
            Factor::Constant(NodeRef::Position(p1)),
        ]];
        let mut same_twin = base.schedule.pool[index].clone();
        same_twin
            .atoms
            .iter_mut()
            .find(|a| a.kind == Kind::LayerInput)
            .expect("the atom")
            .coefficient = vec![vec![
            Factor::Pow2(2),
            Factor::Lt {
                coords: vec![Coord::Half, Coord::Half],
                len: 3,
            },
            Factor::Constant(NodeRef::Position(p2)),
        ]];
        schedule.pool.push(same_twin);
        assert_eq!(
            check(&system_of(&lowering, &schedule), SEED),
            Vec::new(),
            "two per-layer-scaled reads with identical c_ℓ ∈ Fp^L merge"
        );

        for bad in [
            vec![vec![Factor::Minus, Factor::Constant(NodeRef::Position(p1))]],
            vec![vec![
                Factor::Pow2(1),
                Factor::Constant(NodeRef::Position(p1)),
            ]],
        ] {
            let mut bad_schedule = base.schedule.clone();
            let mut bad_twin = base.schedule.pool[index].clone();
            bad_twin
                .atoms
                .iter_mut()
                .find(|a| a.kind == Kind::LayerInput)
                .expect("the atom")
                .coefficient = bad.clone();
            bad_schedule.pool.push(bad_twin);
            assert_eq!(
                dependent_rows(&check(&system_of(&lowering, &bad_schedule), SEED)),
                vec![MaskRow::Stacked(Kind::LayerInput)],
                "scalar multiple {bad:?} with unit mask coefficient unmasks D and must be refused"
            );
        }
    }
}

/// The second review's N1: a twin of a stacked weight claim whose coefficient carries
/// tiny-4-servable's per-layer constants (position 34: 8, 8, 16, 2) weighs the matrix's layers by
/// `eq(ρ, ℓ)·c_ℓ`, another value than `W'(r ‖ ρ)` read through the same functional of `R_W`, and
/// one the pair would drop as the first claim's. No RFC masks it (#556), so it is refused by name.
/// A twin scaled by a public scalar uniform over the layers is the first claim's value up to that
/// scalar, and is accepted.
#[test]
fn a_per_layer_scaled_weight_claim_is_refused() {
    let (c, scaled) = scaled_layer_input();
    let layers = &c.schedule.layers;
    let position = c.schedule.pool[scaled]
        .atoms
        .iter()
        .flat_map(|a| a.coefficient.iter().flatten())
        .find_map(|f| match f {
            Factor::Constant(NodeRef::Position(p)) if !layers.uniform(*p) => Some(*p),
            _ => None,
        })
        .expect("the non-uniform position");
    let index = c
        .schedule
        .pool
        .iter()
        .position(|p| {
            p.commitment == CommitmentClass::Weight
                && p.atoms.len() == 1
                && matches!(&p.atoms[0].layers, Layers::Eq(rho) if !rho.is_empty())
        })
        .expect("a stacked weight claim");
    let claim = &c.schedule.pool[index];
    let matrix = c
        .schedule
        .weight_type(&c.lowering.graph, claim.atoms[0].kind)
        .map_or(WeightMatrix::ChannelParams, |w| w.matrix());
    let tensor = c
        .schedule
        .layers
        .tensor(&c.lowering.graph, claim.atoms[0].kind, 0);
    let twin = |coefficient| {
        let mut twin = claim.clone();
        twin.atoms[0].coefficient = coefficient;
        let mut schedule = c.schedule.clone();
        schedule.pool.push(twin);
        check(&c.system(&schedule, ColumnRule::BelowRow), SEED)
    };
    let findings = twin(vec![vec![Factor::Constant(NodeRef::Position(position))]]);
    assert_eq!(
        findings,
        vec![Finding::ScaledWeight(ScaledWeight {
            matrix,
            tensor,
            position,
            origin: Origin::Pool {
                stage: claim.stage,
                claim: c.schedule.pool.len(),
                atom: 0,
            },
        })]
    );
    let text = findings[0].to_string();
    assert!(
        text.contains("#556") && text.contains(&format!("{matrix:?}")),
        "{text}"
    );
    // A public scalar uniform over the layers: the first claim's value, negated.
    assert_eq!(twin(vec![vec![Factor::Minus]]), Vec::new());
    assert_eq!(
        check(&c.system(&c.schedule, ColumnRule::BelowRow), SEED),
        Vec::new()
    );
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
        scaled_weights: Vec::new(),
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

/// `tessera-protocol/2` pads the token axis to `2^max(⌈log2 n⌉, 2)` rows and lays an axis padded
/// to four rows out as a padded axis (RFC-0013 'Masked extensions', RFC-0047, RFC-0071 'The
/// slots'): lengths 1 to 3 take the layout of 3, longer ones their own. The layout of 3 is not
/// that of 4. RFC-0047 puts a padded axis highest, so tiny-2-servable's `[n, 192]` stacked output
/// at position 44 keeps its natural order at 3, both axes padded, and puts the position bits
/// lowest at 4, where only 192 is.
#[test]
fn short_sequences_take_the_layout_of_three_positions() {
    let lengths: Vec<u32> = [1, 2, 3, 4, 5, 8, 9, 131_072]
        .map(protocol2_length)
        .to_vec();
    assert_eq!(lengths, [3, 3, 3, 4, 5, 8, 9, 131_072]);
    let claims = golden("tiny-2-servable");
    let kind = Kind::Layer {
        position: 44,
        output: 0,
    };
    // The row's points, each with its key: the natural point, 8 channel bits, the position
    // bits, the layer bit.
    let points = |report: &MaskReport| -> Vec<(Vec<Src>, Vec<Src>)> {
        report
            .system
            .rows
            .iter()
            .find(|r| r.row == MaskRow::Stacked(kind))
            .expect("the [n, 192] row")
            .functionals
            .iter()
            .map(|f| (f.point.clone(), f.key.clone()))
            .collect()
    };
    let check = |n| {
        check_claim_list(
            &claims,
            n,
            Request::Generate,
            ColumnRule::BelowRow,
            CommittedColumns::RULE,
        )
        .expect("the schedule")
    };
    for n in [1, 2, 3] {
        let report = check(n);
        assert_eq!((report.n, report.length), (n, 3));
        assert_eq!(report.findings, Vec::new());
        for (point, key) in points(&report) {
            assert_eq!(point, key[..10], "n = {n}: the natural order");
        }
    }
    let four = check(4);
    assert_eq!((four.n, four.length), (4, 4));
    assert_eq!(four.findings, Vec::new());
    for (point, key) in points(&four) {
        let mut expected = key[8..10].to_vec();
        expected.extend(&key[..8]);
        assert_eq!(point, expected, "n = 4: the position bits lowest");
    }
    let one = check_at(
        &claims,
        1,
        Request::Generate,
        ColumnRule::BelowRow,
        CommittedColumns::RULE,
    )
    .expect("1");
    assert!(
        !one.findings.is_empty(),
        "unpadded, one position is unmaskable"
    );
}

/// #554's two readings of a committed tensor's mask columns differ where the tensor does not
/// fill a row: tiny-2-servable's layer input at 4 positions has 8 variables of its own and a
/// layer bit below `log2 k = 10`. RFC-0013's rule takes the data claim's column sub-point, which
/// the layer bit is not part of, so a second read at its point and other layer weights shares the
/// first's functional and is refused. The rejected reading, the layer bit a column, accepts it:
/// the control.
#[test]
fn the_two_readings_of_a_committed_mask_column_differ_below_a_row() {
    let c = compiled(golden("tiny-2-servable"), 4, Request::Generate);
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
    let tensor_order = check(
        &c.system_with(
            &schedule,
            ColumnRule::BelowRow,
            CommittedColumns::TensorOrder,
        ),
        SEED,
    );
    assert_eq!(tensor_order, Vec::new());
    let data_column = check(
        &c.system_with(&schedule, ColumnRule::BelowRow, CommittedColumns::RULE),
        SEED,
    );
    assert_eq!(
        dependent_rows(&data_column),
        vec![MaskRow::Stacked(Kind::LayerInput)],
        "{data_column:?}"
    );
    // The compiled schedule passes under both.
    for committed in [CommittedColumns::TensorOrder, CommittedColumns::RULE] {
        assert_eq!(
            check(
                &c.system_with(&c.schedule, ColumnRule::BelowRow, committed),
                SEED
            ),
            Vec::new(),
            "{committed:?}"
        );
    }
}

/// RFC-0013's rule for an attention tile smaller than a row (#554, RFC-0034): tiny-2-servable's
/// `SoftmaxRow` reciprocals at 4 positions are two tiles, one per KV head, each a tensor of
/// 3 variables below `log2 k = 10`. The bit that selects the tile is a row bit, not a mask
/// variable: a second read that differs from a compiled one in that bit alone shares its
/// functional and is refused. The rejected reading, the tile bit a column, accepts it.
#[test]
fn a_value_in_another_tile_at_one_column_point_is_refused() {
    let c = compiled(golden("tiny-2-servable"), 4, Request::Generate);
    let kind = Kind::Layer {
        position: 27,
        output: 1,
    };
    let tensor = c
        .schedule
        .layers
        .tensor(&c.lowering.graph, kind, 0)
        .expect("the reciprocals");
    assert_eq!(c.lowering.graph.tensors[tensor.0 as usize].shape, [4, 4]);
    let index = c.pool_claim_on(tensor);
    // The compiled value's functional, and the coordinate of its point the tile's column
    // sub-point leaves out: the tile bit.
    let system = c.system(&c.schedule, ColumnRule::BelowRow);
    let value = system
        .rows
        .iter()
        .find(|r| r.row == MaskRow::Stacked(kind))
        .and_then(|r| {
            r.functionals.iter().find(|f| {
                f.origins
                    .iter()
                    .any(|o| matches!(o, Origin::Pool { claim, .. } if *claim == index))
            })
        })
        .expect("the claim's value");
    let natural = c.schedule.pool[index].atoms[0].point.len();
    let columns = value.point.iter().filter(|s| **s != Src::Zero).count();
    assert_eq!((natural, columns), (4, 3), "{value:?}");
    let tile = (0..natural)
        .find(|&j| !value.point.contains(&value.key[j]))
        .expect("a coordinate outside the column sub-point");
    let mut schedule = c.schedule.clone();
    let mut twin = schedule.pool[index].clone();
    twin.point[tile] = fresh(0);
    twin.atoms[0].point[tile] = fresh(0);
    schedule.pool.push(twin);
    let rule = check(&c.system(&schedule, ColumnRule::BelowRow), SEED);
    assert_eq!(
        dependent_rows(&rule),
        vec![MaskRow::Stacked(kind)],
        "{rule:?}"
    );
    let rejected = check(
        &c.system_with(
            &schedule,
            ColumnRule::BelowRow,
            CommittedColumns::TensorOrder,
        ),
        SEED,
    );
    assert_eq!(rejected, Vec::new());
}

/// A functional of another length than its row's width is refused, on a row and on a pair,
/// never sketched over the coordinates it shares with the test vectors.
#[test]
fn a_functional_off_its_row_width_is_refused() {
    let value = |point: Vec<Src>| Functional {
        role: Role::Value,
        key: point.clone(),
        point,
        scale: None,
        origins: Vec::new(),
    };
    let row = MaskSystem {
        rows: vec![RowSystem {
            row: MaskRow::Tensor(TensorId(1)),
            tensor: Some(TensorId(1)),
            vars: 3,
            width: 3,
            functionals: vec![
                value(vec![var(0), var(1), var(2)]),
                value(vec![var(0), var(1)]),
            ],
        }],
        pairs: Vec::new(),
        scaled_weights: Vec::new(),
    };
    assert_eq!(
        check(&row, SEED),
        vec![Finding::Width {
            row: Some(MaskRow::Tensor(TensorId(1))),
            matrix: None,
            width: 3,
            found: 2,
        }]
    );
    let pair = MaskSystem {
        rows: Vec::new(),
        pairs: vec![PairSystem {
            matrix: WeightMatrix::ChannelParams,
            width: 2,
            functionals: vec![PairFunctional {
                role: PairRole::Blinding,
                point: vec![var(0), var(1), var(2)],
                key: Vec::new(),
                tensor: None,
                origins: Vec::new(),
            }],
        }],
        scaled_weights: Vec::new(),
    };
    assert_eq!(
        check(&pair, SEED),
        vec![Finding::Width {
            row: None,
            matrix: Some(WeightMatrix::ChannelParams),
            width: 2,
            found: 3,
        }]
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
    let old = check_claim_list(
        &claims,
        32_768,
        Request::Generate,
        ColumnRule::Row,
        CommittedColumns::RULE,
    )
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
    let amended = check_claim_list(
        &claims,
        32_768,
        Request::Generate,
        ColumnRule::BelowRow,
        CommittedColumns::RULE,
    )
    .expect("the schedule");
    assert_eq!(amended.findings, Vec::new());
}
