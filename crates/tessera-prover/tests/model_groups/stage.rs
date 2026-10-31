// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054 'The leaf stage with groups', on a model's leaf stage, checked from the messages:
//!
//! - **The schedule.** Every label in RFC-0054's order: the empty `logup/roots` and
//!   `logup/values`, each group's roots, levels, values, split and `γ_h`, the table batch, and the
//!   merge.
//! - **Each group's split.** Each group instance's key and value claims follow from the group's
//!   split values by RFC-0045's identity over the group's blocks, at `P_h`.
//! - **The identity.** Each instance's group roots and table root sum to RFC-0008's identity, so
//!   the multiplicities count every group's lookups and padding.
//! - **The merge.** Its rounds hold, its final check holds with weights computed here, and each
//!   merged claim is its tensor's or forest's extension at its point; a site's layers share it.
//! - **Determinism**, and the refusals before the transcript moves.

use super::*;
use tessera_field::GoldilocksExt2;
use tessera_prover::leaf::{GroupedClaims, LeafClaims};
use tessera_prover::leaf::{
    MergeMode, ModelWitness, ProvenGroupedLeaf, balanced_j_star, grouped_multiplicities,
    prove_grouped_leaf,
};
use tessera_sumcheck::largest_table_rounds;
use tessera_witness::Witness;

/// A transcript that records every label, message and challenge, in order.
struct Recording {
    inner: Transcript,
    events: Vec<(String, Vec<F>)>,
}

impl SumcheckTranscript<GoldilocksExt2> for Recording {
    fn absorb_message(&mut self, label: &str, message: &[F]) {
        self.events.push((label.to_owned(), message.to_vec()));
        self.inner.absorb_message(label, message);
    }

    fn challenge(&mut self, label: &str) -> F {
        let c = self.inner.challenge(label);
        self.events.push((label.to_owned(), vec![c]));
        c
    }
}

impl Recording {
    fn new() -> Self {
        Self {
            inner: transcript(Mode::NonInteractive),
            events: Vec::new(),
        }
    }

    fn one(&self, label: &str) -> F {
        let mut found = self.events.iter().filter(|(l, _)| l == label);
        let (Some((_, v)), None) = (found.next(), found.next()) else {
            panic!("{label} once");
        };
        v[0]
    }

    fn message(&self, label: &str) -> &[F] {
        &self.events.iter().find(|(l, _)| l == label).expect(label).1
    }
}

/// The model's leaf stage at `n` tokens, with its witness, run through `body`.
fn with_stage(
    model: &IntegerModel,
    n: u32,
    body: impl FnOnce(&Public, &LeafLayout<ModelColumn>, &Witness, &ModelWitness, &[TableColumns<G>]),
) {
    let (p, layout) = stage(model, n);
    let engine = Engine::new(&p.lowering).unwrap();
    let input: Vec<_> = (0..n).map(|i| (i + 1) % model.arch.vocab).collect();
    let witness = engine.run(model, &Secret::new(input)).unwrap();
    let source = Resident::new(&engine, &p.selection, model, &witness).unwrap();
    let advice = ModelAdviceLayout::new(&p.lowering, &p.selection, &p.claims).unwrap();
    let actual =
        ModelWitness::new(&layout, &p.lowering, &p.layers, &witness, &source, &advice).unwrap();
    let tables: Vec<_> = layout
        .instances()
        .iter()
        .map(|i| {
            let spec = p.selection.tables.iter().find(|t| t.id == i.table).unwrap();
            TableColumns::<G>::new(&GeneratedTable::from_spec(spec).unwrap()).unwrap()
        })
        .collect();
    body(&p, &layout, &witness, &actual, &tables);
}

fn prove(
    layout: &LeafLayout<ModelColumn>,
    grouped: &GroupedLayout,
    actual: &ModelWitness,
    tables: &[TableColumns<G>],
) -> (ProvenGroupedLeaf, Recording) {
    prove_in(layout, grouped, actual, tables, MergeMode::InMemory)
}

fn prove_in(
    layout: &LeafLayout<ModelColumn>,
    grouped: &GroupedLayout,
    actual: &ModelWitness,
    tables: &[TableColumns<G>],
    mode: MergeMode,
) -> (ProvenGroupedLeaf, Recording) {
    let tensors = actual.tensors();
    let multiplicities = grouped_multiplicities(layout, grouped, tables, &tensors).unwrap();
    let refs: Vec<&[G]> = multiplicities.iter().map(|m| &m[..]).collect();
    let mut recording = Recording::new();
    let proven = prove_grouped_leaf(
        &mut recording,
        layout,
        grouped,
        tables,
        &tensors,
        &refs,
        &actual.stage_forests(),
        mode,
    )
    .unwrap();
    (proven, recording)
}

/// Where a label falls in RFC-0054's order: before the groups, in group `h` at a step, in the
/// table batch, in the merge.
fn phase(label: &str) -> (u32, u64, u32) {
    let parts: Vec<&str> = label.split('/').collect();
    match parts.as_slice() {
        ["logup", "alpha"] => (0, 0, 0),
        ["logup", "beta"] => (0, 0, 1),
        ["logup", "roots"] => (0, 0, 2),
        ["logup", "values"] => (0, 0, 3),
        ["group", h, rest @ ..] => {
            let step = match rest {
                ["logup", "roots"] => 0,
                ["tree", ..] => 1,
                ["logup", "values"] => 2,
                ["split"] => 3,
                ["merge", "gamma"] => 4,
                _ => panic!("{label}"),
            };
            (1, h.parse().unwrap(), step)
        }
        ["tables", "logup", "roots"] => (2, 0, 0),
        ["tables", "tree", ..] => (2, 0, 1),
        ["merge", "beta"] => (3, 0, 0),
        ["merge", "round" | "r", _] => (3, 0, 1),
        ["merge", "evals"] => (3, 0, 2),
        _ => panic!("{label}"),
    }
}

fn lt(point: &[E], m: u64) -> E {
    (0..m).fold(E::ZERO, |sum, x| sum + eq(point, x))
}

#[test]
fn a_grouped_stage_runs_rfc_0054s_schedule_and_its_claims_hold() {
    let model = layered(bounded_model());
    with_stage(&model, 5, |p, layout, witness, actual, tables| {
        let sites: Vec<NodeRef> = actual.forest_sites().collect();
        assert!(!sites.is_empty(), "the stage has product forests");
        for (g_t, g_l) in [(1, 0), (0, 1), (2, 1)] {
            let grouped = GroupedLayout::new(layout, &p.layers, groups(g_t, g_l)).unwrap();
            let (proven, recording) = prove(layout, &grouped, actual, tables);
            let count = grouped.count();

            // The schedule.
            let phases: Vec<_> = recording.events.iter().map(|(l, _)| phase(l)).collect();
            assert!(phases.windows(2).all(|w| w[0] <= w[1]), "RFC-0054's order");
            for h in 0..count {
                for step in [0, 2, 3, 4] {
                    assert_eq!(
                        phases.iter().filter(|&&x| x == (1, h as u64, step)).count(),
                        1
                    );
                }
            }
            assert!(recording.message("logup/roots").is_empty());
            assert!(recording.message("logup/values").is_empty());

            for h in 0..count {
                let group = grouped.group(h).unwrap();
                let proven_group = &proven.groups()[h];
                let point = es(&proven.points()[h]);
                let splits = es(&proven.splits()[h]);
                assert_eq!(es(recording.message(&format!("group/{h}/split"))), splits);
                let value_of = |site: usize, layer: usize, read: usize| {
                    let at = group
                        .reads()
                        .iter()
                        .position(|r| (r.site, r.layer, r.read) == (site, layer, read))
                        .expect("the group sends the tensor");
                    splits[at]
                };
                // Each instance's claims, by RFC-0045's identity over the group's blocks.
                for (instance, claims) in group.instances().iter().zip(proven_group.claims()) {
                    let height = instance.height as usize;
                    let rows = tables
                        .iter()
                        .find(|t| t.table() == instance.table)
                        .unwrap()
                        .tuples();
                    let at = &point[..height];
                    let mut forms: Vec<(E, Option<E>)> = Vec::new();
                    for block in &instance.blocks {
                        let gadget = &layout.sites()[block.site].layers[block.layer];
                        let lookup = &gadget.lookups[block.lookup].lookup;
                        let reads = layout.reads(block.site, block.layer).unwrap();
                        let term = |t| {
                            let r = reads.iter().position(|read| read.term == t).unwrap();
                            value_of(block.site, block.layer, r)
                        };
                        let k = block.vars as usize;
                        let weight = eq(&at[k..], block.offset >> k);
                        forms.push((
                            weight * form(&lookup.key, term),
                            lookup.value.as_ref().map(|v| weight * form(v, term)),
                        ));
                    }
                    let tail = E::ONE - lt(at, instance.entries);
                    let key = forms
                        .iter()
                        .fold(tail * E::from(v(rows.keys[0])), |s, f| s + f.0);
                    assert_eq!(e(claims.key.value), key, "group {h}'s key claim");
                    if let Some(claim) = &claims.value {
                        let value = forms
                            .iter()
                            .fold(tail * E::from(v(rows.values.unwrap()[0])), |s, f| {
                                s + f.1.unwrap()
                            });
                        assert_eq!(e(claim.value), value, "group {h}'s value claim");
                    }
                }
            }

            // RFC-0008's identity over every group's roots and the table root.
            let table_roots = es(proven.tables().roots());
            for (k, instance) in layout.instances().iter().enumerate() {
                let (mut n_sum, mut d_sum) = (E::ZERO, E::ONE);
                for h in 0..count {
                    let group = grouped.group(h).unwrap();
                    if let Some(j) = group
                        .instances()
                        .iter()
                        .position(|i| i.table == instance.table)
                    {
                        let roots = es(proven.groups()[h].roots());
                        let (n, d) = (roots[2 * j], roots[2 * j + 1]);
                        assert_ne!(d, E::ZERO);
                        (n_sum, d_sum) = (n_sum * d + n * d_sum, d_sum * d);
                    }
                }
                let (n, d) = (table_roots[2 * k], table_roots[2 * k + 1]);
                assert_eq!(n_sum * d + n * d_sum, E::ZERO, "{:?}", instance.table);
            }

            // The merge, from the recorded messages and challenges.
            let polynomials = grouped.polynomials();
            let tensors = polynomials.len();
            let beta = e(recording.one("merge/beta"));
            let gammas: Vec<E> = (0..count)
                .map(|h| e(recording.one(&format!("group/{h}/merge/gamma"))))
                .collect();
            let g_t = u32::from(g_t);
            let forest_shape = |f: usize| {
                let vars = proven.merged()[tensors + f].point.len();
                let leaves = forest_truth(p, witness, sites[f]);
                assert_eq!(leaves.len(), 1 << vars);
                let q = vars_of(&leaves, p, sites[f]);
                (vars, q, leaves)
            };
            // Each claim: polynomial, free point, bits, value, weight.
            let mut claims: Vec<(usize, Vec<E>, u64, E, E)> = Vec::new();
            let mut outer = E::ONE;
            for (h, &gamma) in gammas.iter().enumerate() {
                let group = grouped.group(h).unwrap();
                let splits = es(&proven.splits()[h]);
                let point = es(&proven.points()[h]);
                let mut inner = E::ONE;
                for (read, &value) in group.reads().iter().zip(&splits) {
                    let free = read.coords.iter().map(|&c| point[c as usize]).collect();
                    claims.push((read.polynomial, free, read.group_bits, value, outer * inner));
                    inner *= gamma;
                }
                for (f, claim) in proven.groups()[h].forest_claims().iter().enumerate() {
                    claims.push((
                        tensors + f,
                        es(&claim.point),
                        h as u64,
                        e(claim.value),
                        outer * inner,
                    ));
                    inner *= gamma;
                }
                outer *= beta;
            }
            let vars_of_polynomial = |i: usize| match i < tensors {
                true => (polynomials[i].vars, polynomials[i].group_vars.clone()),
                false => {
                    let (vars, (q, l), _) = forest_shape(i - tensors);
                    let s: Vec<usize> = (q..q + g_t as usize)
                        .chain(l..l + usize::from(g_l))
                        .collect();
                    (vars, s)
                }
            };
            let shapes: Vec<(usize, Vec<usize>)> =
                (0..proven.merged().len()).map(vars_of_polynomial).collect();
            let n_max = shapes.iter().map(|s| s.0).max().unwrap();
            let two = E::ONE + E::ONE;
            let mut claim = claims.iter().fold(E::ZERO, |sum, (i, _, _, value, w)| {
                let pad = (0..n_max - shapes[*i].0).fold(E::ONE, |a, _| a * two);
                sum + *w * pad * *value
            });
            let mut challenges = Vec::new();
            for j in 0..n_max {
                let round = es(recording.message(&format!("merge/round/{j}")));
                let [p0, p2] = round[..] else {
                    panic!("degree 2")
                };
                let p1 = claim - p0;
                let r = e(recording.one(&format!("merge/r/{j}")));
                // The degree-2 polynomial through 0, 1, 2, at r.
                let half = (E::ONE + E::ONE).inverse().unwrap();
                claim = p0 * (r - E::ONE) * (r - two) * half - p1 * r * (r - two)
                    + p2 * r * (r - E::ONE) * half;
                challenges.push(r);
            }
            let evals = es(recording.message("merge/evals"));
            assert_eq!(evals.len(), shapes.len());
            let mut total = E::ZERO;
            for (i, (vars, s)) in shapes.iter().enumerate() {
                // The last n_i challenges, free variables first, then S.
                let bound = &challenges[n_max - vars..];
                let order: Vec<usize> = (0..*vars)
                    .filter(|x| !s.contains(x))
                    .chain(s.iter().copied())
                    .collect();
                let mut r = vec![E::ZERO; *vars];
                for (&var, &c) in order.iter().zip(bound) {
                    r[var] = c;
                }
                let free: Vec<E> = (0..*vars)
                    .filter(|x| !s.contains(x))
                    .map(|x| r[x])
                    .collect();
                let on_s: Vec<E> = s.iter().map(|&x| r[x]).collect();
                let weight =
                    claims
                        .iter()
                        .filter(|c| c.0 == i)
                        .fold(E::ZERO, |sum, (_, z, bits, _, w)| {
                            let bits_eq = on_s.iter().enumerate().fold(E::ONE, |a, (u, &x)| {
                                a * if bits >> u & 1 == 1 { x } else { E::ONE - x }
                            });
                            let z_eq = z.iter().zip(&free).fold(E::ONE, |a, (&zz, &x)| {
                                a * (zz * x + (E::ONE - zz) * (E::ONE - x))
                            });
                            sum + *w * z_eq * bits_eq
                        });
                total += weight * evals[i];
                // Each merged claim is its polynomial's extension at its point.
                let merged = &proven.merged()[i];
                assert_eq!(es(&merged.point), r);
                assert_eq!(e(merged.value), evals[i]);
                let truth = match i < tensors {
                    true => tensor_values(layout, actual, i),
                    false => forest_shape(i - tensors).2,
                };
                assert_eq!(mle(&truth, &r), evals[i], "polynomial {i}");
            }
            assert_eq!(total, claim, "the merge's final check");

            // A site's layers lie at one point, as RFC-0037's layer point needs.
            let mut polynomial = 0;
            for (s, site) in layout.sites().iter().enumerate() {
                let reads = layout.reads(s, 0).unwrap().len();
                for l in 1..site.layers.len() {
                    for r in 0..reads {
                        let first = &proven.merged()[polynomial + r].point;
                        let this = &proven.merged()[polynomial + l * reads + r].point;
                        assert_eq!(first, this, "site {s}, layer {l}");
                    }
                }
                polynomial += reads * site.layers.len();
            }

            // Determinism.
            let (again, _) = prove(layout, &grouped, actual, tables);
            assert_eq!(again.splits(), proven.splits());
            assert_eq!(again.merge().proof(), proven.merge().proof());
        }
    });
}

/// Polynomial `i`'s tensor: the ungrouped split's `i`-th value's tensor, from the witness.
fn tensor_values(layout: &LeafLayout<ModelColumn>, actual: &ModelWitness, i: usize) -> Vec<i64> {
    let mut polynomial = 0;
    for (s, site) in layout.sites().iter().enumerate() {
        for l in 0..site.layers.len() {
            for read in layout.reads(s, l).unwrap() {
                if polynomial == i {
                    return actual.values(s, l, read.term).unwrap().to_vec();
                }
                polynomial += 1;
            }
        }
    }
    panic!("polynomial {i}");
}

/// A forest's query and layer variables, from its scores' shape: keys and queries `⌈log2 n⌉`
/// each, the heads above them.
fn vars_of(_leaves: &[i64], p: &Public, site: NodeRef) -> (usize, usize) {
    let node = &p.lowering.graph.nodes[owner(p, site, 0).0 as usize];
    let shape = &p.lowering.graph.tensors[node.inputs[0].0 as usize].shape;
    let (heads, n) = (shape[0], shape[1]);
    let q = vars(n) as usize;
    (q, 2 * q + vars(heads) as usize)
}

#[test]
fn a_grouped_stage_is_refused_before_the_transcript_moves() {
    with_stage(&bounded_model(), 3, |p, layout, _, actual, tables| {
        let tensors = actual.tensors();
        let forests = actual.stage_forests();
        let refused = |grouped: &GroupedLayout, m: &[&[G]], rule: &'static str| {
            let mut recording = Recording::new();
            let result = prove_grouped_leaf(
                &mut recording,
                layout,
                grouped,
                tables,
                &tensors,
                m,
                &forests,
                MergeMode::InMemory,
            );
            assert_eq!(result.err(), Some(LeafError::Internal { rule }));
            assert!(recording.events.is_empty());
        };
        let none = GroupedLayout::new(layout, &p.layers, LeafGroups::NONE).unwrap();
        let multiplicities = grouped_multiplicities(layout, &none, tables, &tensors).unwrap();
        let refs: Vec<&[G]> = multiplicities.iter().map(|m| &m[..]).collect();
        refused(
            &none,
            &refs,
            "an ungrouped stage keeps RFC-0010's messages (prepare_leaf)",
        );
        let one = GroupedLayout::new(layout, &p.layers, groups(1, 0)).unwrap();
        refused(
            &one,
            &refs[1..],
            "one multiplicity column per instance of the stage",
        );
        // A j* past the merge's table rounds, before any message.
        let multiplicities = grouped_multiplicities(layout, &one, tables, &tensors).unwrap();
        let refs: Vec<&[G]> = multiplicities.iter().map(|m| &m[..]).collect();
        let mut recording = Recording::new();
        let result = prove_grouped_leaf(
            &mut recording,
            layout,
            &one,
            tables,
            &tensors,
            &refs,
            &forests,
            MergeMode::TwoPass { j_star: Some(64) },
        );
        assert_eq!(
            result.err(),
            Some(LeafError::Internal {
                rule: "j* is at most the merge's table rounds"
            })
        );
        assert!(recording.events.is_empty());
    });
}

/// RFC-0054 'Testing Strategy' ('Two passes', 'Boundaries'): the table-backed merge rounds and
/// the merge pass send the in-memory merge's messages and leave its merged claims, on every class
/// of a model's stage, at every `j*` from 0 to the largest that fits and at the balanced `j*` the
/// backend takes, under four groupings.
#[test]
fn the_two_pass_merge_sends_the_in_memory_merges_messages() {
    let model = layered(bounded_model());
    with_stage(&model, 5, |p, layout, _, actual, tables| {
        for (g_t, g_l) in [(1, 0), (0, 1), (1, 1), (2, 1)] {
            let grouped = GroupedLayout::new(layout, &p.layers, groups(g_t, g_l)).unwrap();
            let (expected, recorded) =
                prove_in(layout, &grouped, actual, tables, MergeMode::InMemory);
            let same = |mode: MergeMode| {
                let (proven, recording) = prove_in(layout, &grouped, actual, tables, mode);
                assert_eq!(recording.events, recorded.events, "{g_t}, {g_l}: {mode:?}");
                assert_eq!(proven.merged(), expected.merged(), "{g_t}, {g_l}: {mode:?}");
            };
            same(MergeMode::TwoPass { j_star: None });
            // Every j* the tables answer; the first refused ends them.
            let tensors = actual.tensors();
            let multiplicities =
                grouped_multiplicities(layout, &grouped, tables, &tensors).unwrap();
            let refs: Vec<&[G]> = multiplicities.iter().map(|m| &m[..]).collect();
            let mut largest = None;
            for j_star in 0..64 {
                let mode = MergeMode::TwoPass {
                    j_star: Some(j_star),
                };
                let fits = prove_grouped_leaf(
                    &mut Recording::new(),
                    layout,
                    &grouped,
                    tables,
                    &tensors,
                    &refs,
                    &actual.stage_forests(),
                    mode,
                )
                .is_ok();
                if !fits {
                    break;
                }
                same(mode);
                largest = Some(j_star);
            }
            assert!(largest >= Some(2), "{g_t}, {g_l}: tables answer rounds");
        }
    });
}

/// The backend's `j*` balances the tables' `2^{j*+1}` entries per class and group against the
/// folded state's `N_m/2^{j*}`, within the rounds the tables can answer (RFC-0054 'Memory').
#[test]
fn the_balanced_j_star_stays_within_the_table_rounds() {
    use tessera_sumcheck::MergePolynomial;
    let shape = |vars, group_vars: Vec<usize>| MergePolynomial { vars, group_vars };
    // 2^21 entries over 1·2 classes: 2^{2j*} = 2^20, j* = 10; the tables answer 19 rounds, each
    // polynomial's free variables. Over 2^10 groups, 2^{2j*} = 2^10.
    let wide = [shape(20, vec![19]), shape(20, vec![19])];
    assert_eq!(largest_table_rounds(&wide), 19);
    assert_eq!(balanced_j_star(&wide, 1), 10);
    assert_eq!(balanced_j_star(&wide, 1 << 10), 5);
    // An odd exponent rounds up: 2^21 entries over one class, 2^{2j*} ≥ 2^21, j* = 11.
    assert_eq!(balanced_j_star(&[shape(21, vec![20])], 1), 11);
    // A polynomial of group variables alone, unpadded: the tables answer no round.
    let low = [shape(3, vec![0, 1, 2]), shape(2, vec![])];
    assert_eq!(largest_table_rounds(&low), 0);
    assert_eq!(balanced_j_star(&low, 4), 0);
    // More classes than entries.
    assert_eq!(balanced_j_star(&[shape(2, vec![])], 1 << 10), 0);
}

/// RFC-0054 'The stage's outputs': where the stages and the openings read a grouped stage's
/// claims. Every split value's read point is its merged claim's, and it is also the merged point of
/// its tensor's shape and token axis, which the stages derive without the group layouts. The
/// forests' leaf claims are their merged claims, and the native relations build on them.
#[test]
fn the_stages_read_a_grouped_stages_claims_at_its_merged_points() {
    let model = layered(bounded_model());
    with_stage(&model, 5, |p, layout, witness, actual, tables| {
        let native = tessera_prover::native::ModelNativePlan::derive(
            &p.lowering,
            &p.selection,
            &p.claims,
            layout,
        )
        .unwrap();
        for (g_t, g_l) in [(1, 0), (2, 1)] {
            let grouped = GroupedLayout::new(layout, &p.layers, groups(g_t, g_l)).unwrap();
            let (proven, _) = prove(layout, &grouped, actual, tables);
            let claims = GroupedClaims::new(&proven, &grouped);
            assert_eq!(claims.token_bits(), u32::from(g_t));
            let mut polynomial = 0;
            let mut whole = 0;
            for (s, site) in layout.sites().iter().enumerate() {
                for l in 0..site.layers.len() {
                    for (r, read) in layout.reads(s, l).unwrap().iter().enumerate() {
                        let merged = &proven.merged()[polynomial];
                        assert_eq!(
                            claims.read_point(layout, s, l, r).as_ref(),
                            Some(&merged.point),
                            "site {s}, layer {l}, read {r}"
                        );
                        // Read over every axis of the first lookup that reads it: the domain's
                        // shape and token axis give the same point.
                        let lookup = site.layers[l]
                            .lookups
                            .iter()
                            .find(|k| {
                                std::iter::once(&k.lookup.key)
                                    .chain(&k.lookup.value)
                                    .any(|f| f.terms.iter().any(|&(_, t)| t == read.term))
                            })
                            .unwrap();
                        if read.axes.len() == lookup.domain.len() {
                            assert_eq!(
                                claims.tensor_point(&lookup.domain, lookup.token).as_ref(),
                                Some(&merged.point),
                                "site {s}, layer {l}, read {r}"
                            );
                            whole += 1;
                        }
                        polynomial += 1;
                    }
                }
            }
            assert!(whole > 0);
            assert_eq!(claims.read_point(layout, layout.sites().len(), 0, 0), None);
            let tensors = grouped.polynomials().len();
            for k in 0..actual.stage_forests().len() {
                assert_eq!(claims.forest_claim(k), proven.merged().get(tensors + k));
            }
            assert_eq!(claims.instances(), layout.instances().len());
            // The native relations take their targets at these points.
            let mut memory = None;
            let planning = allocation_counter::measure(|| {
                memory = Some(native.memory(&p.lowering, false).unwrap())
            });
            assert_eq!(planning.count_total, 0);
            let memory = memory.unwrap();
            let mut built = None;
            let measured = allocation_counter::measure(|| {
                built = Some(
                    tessera_prover::native::ModelNativeProver::new(
                        &native,
                        &p.lowering,
                        actual,
                        witness,
                        &claims,
                    )
                    .unwrap(),
                )
            });
            assert!(
                measured.bytes_max <= memory.construction_peak_bytes(),
                "grouped {measured:?} {memory:?}"
            );
        }
    });
}

/// RFC-0054 'When groups apply': the witness pass counts each table's multiplicities over every
/// group's column, its padding at row 0, so the advice commits exactly what the table batch needs:
/// the prover's own grouped count.
#[test]
fn the_witness_pass_counts_every_groups_padding() {
    use tessera_compiler::{SequenceTile, Tile};
    let model = layered(bounded_model());
    let n = 5;
    let (p, layout) = stage(&model, n);
    let engine = Engine::new(&p.lowering).unwrap();
    let input: Vec<_> = (0..n).map(|i| (i + 1) % model.arch.vocab).collect();
    let witness = engine.run(&model, &Secret::new(input)).unwrap();
    let source = Resident::new(&engine, &p.selection, &model, &witness).unwrap();
    let advice = ModelAdviceLayout::new(&p.lowering, &p.selection, &p.claims).unwrap();
    let actual =
        ModelWitness::new(&layout, &p.lowering, &p.layers, &witness, &source, &advice).unwrap();
    let tables: Vec<_> = layout
        .instances()
        .iter()
        .map(|i| {
            let spec = p.selection.tables.iter().find(|t| t.id == i.table).unwrap();
            TableColumns::<G>::new(&GeneratedTable::from_spec(spec).unwrap()).unwrap()
        })
        .collect();
    let read = |source: &Resident<'_>, table: TableId| {
        let tile = source
            .tile(&SequenceTile {
                tile: Tile::Table { table: table.0 },
                block: None,
                shares: Vec::new(),
            })
            .unwrap();
        let column = tile.multiplicities().unwrap();
        let mut rows = tessera_codec::SecretBuf::filled(0u64, column.rows() as usize);
        column.read_rows(0, &mut rows).unwrap();
        rows.to_vec()
    };
    // Each table once.
    let table = layout.instances()[0].table;
    assert!(matches!(
        Resident::new(&engine, &p.selection, &model, &witness)
            .unwrap()
            .with_grouped_columns(vec![(table, 8), (table, 16)]),
        Err(tessera_witness::WitnessError::Usage { .. })
    ));
    for (g_t, g_l) in [(0, 0), (1, 0), (1, 1), (2, 2)] {
        let grouped = GroupedLayout::new(&layout, &p.layers, groups(g_t, g_l)).unwrap();
        let columns = grouped.columns().unwrap();
        let ours = grouped_multiplicities(&layout, &grouped, &tables, &actual.tensors()).unwrap();
        let fresh = Resident::new(&engine, &p.selection, &model, &witness)
            .unwrap()
            .with_grouped_columns(columns.clone())
            .unwrap();
        assert_eq!(columns.len(), layout.instances().len());
        for ((instance, counted), &(table, entries)) in
            layout.instances().iter().zip(&ours).zip(&columns)
        {
            assert_eq!(instance.table, table);
            let witnessed = read(&fresh, table);
            let ours: Vec<u64> = counted.iter().map(|m| m.to_canonical_u64()).collect();
            assert_eq!(
                &witnessed[..ours.len()],
                &ours[..],
                "{table:?}, {g_t} and {g_l} bits"
            );
            // Every lookup of every group's column, its padding included.
            assert_eq!(witnessed.iter().sum::<u64>(), entries);
            if (g_t, g_l) == (0, 0) {
                // One group: the stage's own column.
                assert_eq!(entries, instance.entries.next_power_of_two());
                assert_eq!(witnessed, read(&source, table));
            }
        }
    }
}
