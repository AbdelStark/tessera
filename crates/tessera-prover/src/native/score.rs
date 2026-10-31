// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0056 score rounds. No table has both a query and a key axis. The ordinary
//! degree-three sumcheck and original maximum/score opening claims are unchanged.

use std::sync::Arc;

pub(super) mod inputs;

use tessera_codec::SecretBuf;
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField};
use tessera_poly::{PolyField, eq_table};
use tessera_sumcheck::{Computed, ComputedState};

#[cfg(test)]
thread_local! {
    static FAULT: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

fn fault(_which: u8) -> bool {
    #[cfg(test)]
    return FAULT.with(|f| f.get() == _which);
    #[cfg(not(test))]
    false
}

/// Padded INT8 operands, shared by the key and forest relations at one site.
pub(super) struct Inputs {
    pub layers: usize,
    pub heads: usize,
    pub key_heads: usize,
    pub positions: usize,
    pub dim: usize,
    pub layer_vars: usize,
    pub queries: SecretBuf<i8>,
    pub keys: SecretBuf<i8>,
    /// Padded `[layers, heads, queries]`, just as the original auxiliary column.
    pub maxima: SecretBuf<G>,
}

impl Inputs {
    fn np(&self) -> usize {
        self.positions.next_power_of_two()
    }
    fn hp(&self) -> usize {
        self.heads.next_power_of_two()
    }
    fn key_vars(&self) -> usize {
        self.np().trailing_zeros() as usize
    }
    fn row_vars(&self) -> usize {
        self.layer_vars + self.hp().trailing_zeros() as usize + self.key_vars()
    }
    fn query(&self, l: usize, h: usize, q: usize, d: usize) -> G {
        G::from_i64(i64::from(
            self.queries[((l * self.hp() + h) * self.np() + q) * self.dim + d],
        ))
    }
    fn key(&self, l: usize, h: usize, i: usize, d: usize) -> G {
        G::from_i64(i64::from(
            self.keys[((l * self.key_heads + h) * self.np() + i) * self.dim + d],
        ))
    }
    fn row(&self, l: usize, h: usize, q: usize) -> usize {
        (l * self.hp() + h) * self.np() + q
    }
}

pub(super) struct Score {
    pub inputs: Arc<Inputs>,
    pub point: Vec<F>,
}

/// The public causal selector at a point, including real head/layer boundaries.
pub(super) fn mask(point: &[F], heads: usize, n: usize, layers: usize) -> F {
    let np = n.next_power_of_two();
    let hp = heads.next_power_of_two();
    let k = np.trailing_zeros() as usize;
    let mut value = F::ZERO;
    for l in 0..layers {
        for h in 0..heads {
            for q in 0..n {
                value += vertex(&point[k..], (l * hp + h) * np + q) * lt(&point[..k], q + 1);
            }
        }
    }
    value
}

impl Computed<F> for Score {
    fn vars(&self) -> usize {
        self.inputs.key_vars() + self.inputs.row_vars()
    }
    fn degree(&self) -> usize {
        3
    }
    fn evaluations(&self) -> usize {
        2
    }
    fn start(&self) -> Box<dyn ComputedState<F> + '_> {
        let mut state = State {
            score: self,
            bound: Vec::new(),
            rows: None,
        };
        if self.inputs.key_vars() == 0 {
            state.make_rows();
        }
        Box::new(state)
    }
}

struct State<'a> {
    score: &'a Score,
    /// Public transcript challenges only.
    bound: Vec<F>,
    /// Maximum, score, equality and mask, after all key variables have been bound.
    rows: Option<[SecretBuf<F>; 4]>,
}

fn eq(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).fold(F::ONE, |v, (&a, &b)| {
        v * ((F::ONE - a) * (F::ONE - b) + a * b)
    })
}

fn vertex(point: &[F], index: usize) -> F {
    point.iter().enumerate().fold(F::ONE, |v, (j, &r)| {
        v * if (index >> j) & 1 == 1 { r } else { F::ONE - r }
    })
}

/// Closed form of the multilinear extension of `[i < end]`. Branches read public
/// row indices and lengths; no witness value chooses a branch or an address.
fn lt(point: &[F], end: usize) -> F {
    if end >= 1usize << point.len() {
        return F::ONE;
    }
    let (mut less, mut equal) = (F::ZERO, F::ONE);
    for (j, &r) in point.iter().enumerate().rev() {
        if (end >> j) & 1 == 1 {
            less += equal * (F::ONE - r);
            equal *= r;
        } else {
            equal *= F::ONE - r;
        }
    }
    less
}

impl State<'_> {
    fn key_round(&self, x: F) -> F {
        let a = &self.score.inputs;
        let j = self.bound.len();
        let mut low = self.bound.clone();
        low.push(x);
        // Weights are public challenges. Keys are folded anew one block at a time;
        // no folded key tensor or array of score-sized prefix sums is retained.
        let weights = eq_table(&low).expect("a bounded public key point");
        let width = 1usize << low.len();
        let blocks = a.np() / width;
        let z = &self.score.point;
        let high = &z[j + 1..a.key_vars()];
        let row_z = &z[a.key_vars()..];
        let mut total = F::ZERO;
        for l in 0..a.layers {
            for g in 0..a.key_heads {
                let key_head = if fault(3) { (g + 1) % a.key_heads } else { g };
                let mut prefix = SecretBuf::filled(F::ZERO, a.dim);
                let mut folded = SecretBuf::filled(F::ZERO, a.dim);
                let mut mass = F::ZERO;
                for block in 0..blocks {
                    folded.fill(F::ZERO);
                    for (lo, &w) in weights.iter().enumerate() {
                        let i = block * width + lo;
                        for (d, value) in folded.iter_mut().enumerate() {
                            *value += w.mul_by_base(a.key(l, key_head, i, d));
                        }
                    }
                    let e = vertex(high, block);
                    for h in g * a.heads / a.key_heads..(g + 1) * a.heads / a.key_heads {
                        for q in block * width..((block + 1) * width).min(a.positions) {
                            let boundary = e * lt(&low, q % width + usize::from(!fault(2)));
                            let extra = if fault(1) { e } else { F::ZERO };
                            let mut term =
                                F::from(a.maxima[a.row(l, h, q)]) * (mass + extra + boundary);
                            for d in 0..a.dim {
                                term -= (prefix[d] + (boundary + extra) * folded[d])
                                    .mul_by_base(a.query(l, h, q, d));
                            }
                            total += vertex(row_z, a.row(l, h, q)) * term;
                        }
                    }
                    mass += e;
                    for (p, &k) in prefix.iter_mut().zip(folded.iter()) {
                        *p += e * k;
                    }
                }
            }
        }
        total * eq(&z[..j + 1], &low)
    }

    fn make_rows(&mut self) {
        let a = &self.score.inputs;
        let key_vars = a.key_vars();
        let z = &self.score.point;
        let key_eq = eq(&z[..key_vars], &self.bound);
        let len = a.maxima.len();
        let mut rows = std::array::from_fn::<_, 4, _>(|_| SecretBuf::filled(F::ZERO, len));
        for (r, &c) in a.maxima.iter().enumerate() {
            rows[0][r] = F::from(c);
            rows[2][r] = key_eq * vertex(&z[key_vars..], r);
        }
        let weights = eq_table(&self.bound).expect("a bounded public key point");
        for l in 0..1usize << a.layer_vars {
            for g in 0..a.key_heads {
                let key_head = if fault(3) { (g + 1) % a.key_heads } else { g };
                let mut folded = SecretBuf::filled(F::ZERO, a.dim);
                for (i, &weight) in weights.iter().enumerate() {
                    for (d, k) in folded.iter_mut().enumerate() {
                        *k += weight.mul_by_base(a.key(l, key_head, i, d));
                    }
                }
                for h in g * a.heads / a.key_heads..if g + 1 == a.key_heads {
                    a.hp()
                } else {
                    (g + 1) * a.heads / a.key_heads
                } {
                    for q in 0..a.np() {
                        let row = a.row(l, h, q);
                        if l < a.layers && h < a.heads && q < a.positions {
                            rows[3][row] = lt(&self.bound, q + 1);
                        }
                        for (d, &k) in folded.iter().enumerate() {
                            rows[1][row] += k.mul_by_base(a.query(l, h, q, d));
                        }
                    }
                }
            }
        }
        self.rows = Some(rows);
    }
}

impl ComputedState<F> for State<'_> {
    fn round(&self, nodes: &[F], values: &mut [F]) {
        for (&x, out) in nodes.iter().zip(values) {
            *out = if let Some(rows) = &self.rows {
                if rows[0].len() == 1 {
                    (rows[0][0] - rows[1][0]) * rows[2][0] * rows[3][0]
                } else {
                    let mut total = F::ZERO;
                    for i in 0..rows[0].len() / 2 {
                        let at =
                            |k: usize| rows[k][2 * i] + x * (rows[k][2 * i + 1] - rows[k][2 * i]);
                        total += (at(0) - at(1)) * at(2) * at(3);
                    }
                    total
                }
            } else {
                self.key_round(x)
            };
        }
    }
    fn bind(&mut self, challenge: F) {
        self.bound.push(challenge);
        if let Some(rows) = &mut self.rows {
            for table in rows {
                let half = table.len() / 2;
                for i in 0..half {
                    table[i] = table[2 * i] + challenge * (table[2 * i + 1] - table[2 * i]);
                }
                table.truncate(half);
            }
        } else if self.bound.len() == self.score.inputs.key_vars() {
            self.make_rows();
        }
    }
    fn evaluations(&self, out: &mut Vec<F>) {
        let rows = self.rows.as_ref().expect("all key variables bound");
        out.extend([rows[0][0], rows[1][0]]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_codec::Ascii;
    use tessera_field::QuadraticExtension;
    use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
    use tessera_sumcheck::{Instance, Prefix, Protocol, Table, Term, prove_batch};

    #[test]
    fn score_input_packing_has_exact_retention_and_checked_constructor_peak() {
        use super::super::{Family, Kind, score_inputs};
        use tessera_codec::{DecodeLimits, Secret, envelope, schema};
        use tessera_compiler::{PublicModel, lower};
        use tessera_ir::{IntegerModel, Op};
        use tessera_witness::Engine;
        for name in [
            "tiny-2-minimal",
            "tiny-2-servable",
            "tiny-4-servable",
            "tiny-2-affine",
        ] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../spec/vectors/models")
                .join(format!("{name}.d"));
            let model: IntegerModel = envelope::open(
                &std::fs::read(path).unwrap(),
                &schema::INTEGER_MODEL,
                DecodeLimits::INTEGER_MODEL,
            )
            .unwrap();
            for n in [1, 3, 17] {
                let lowering = lower(PublicModel::of(&model), n).unwrap();
                let engine = Engine::new(&lowering)
                    .unwrap()
                    .threads(std::num::NonZeroUsize::MIN);
                let witness = engine
                    .run(&model, &Secret::new(vec![0; n as usize]))
                    .unwrap();
                let nodes = lowering
                    .graph
                    .nodes
                    .iter()
                    .filter(|node| matches!(node.op, Op::SoftmaxRow { .. }))
                    .map(|node| node.id)
                    .collect::<Vec<_>>();
                for repeat_source in [false, true] {
                    let mut nodes = nodes.clone();
                    // A repeated real source exercises a non-power-of-two packing domain;
                    // this is a buffer test, with no proof/claim or extra model layer.
                    if repeat_source {
                        nodes.push(nodes[0]);
                    }
                    let family = Family {
                        windows: Vec::new(),
                        site: 0,
                        lambda: nodes.len().next_power_of_two().trailing_zeros(),
                        nodes,
                        shape: Vec::new(),
                        kind: Kind::Softmax {
                            sigma: 0,
                            forest: None,
                        },
                    };
                    let mut corrupt = lower(PublicModel::of(&model), n).unwrap();
                    let (q, k) = inputs::ids(&corrupt, family.nodes[0]).unwrap();
                    for id in [q, k] {
                        let original = corrupt.graph.tensors[id.0 as usize].range;
                        for outside in [
                            tessera_ir::IntRange { lo: -129, hi: 127 },
                            tessera_ir::IntRange { lo: -128, hi: 128 },
                        ] {
                            corrupt.graph.tensors[id.0 as usize].range = outside;
                            let invalid = allocation_counter::measure(|| {
                                assert_eq!(
                                    inputs::Shape::derive(&family, &corrupt).err(),
                                    Some(super::super::bad())
                                );
                            });
                            assert_eq!(invalid.count_total, 0);
                        }
                        corrupt.graph.tensors[id.0 as usize].range = original;
                    }
                    // Change one layer while preserving its own GQA ratio. The family
                    // still needs one shared geometry, rather than accepting both shapes.
                    let original_heads = corrupt.graph.tensors[q.0 as usize].shape[0];
                    let key_heads = corrupt.graph.tensors[k.0 as usize].shape[0];
                    corrupt.graph.tensors[q.0 as usize].shape[0] += key_heads;
                    let inconsistent = allocation_counter::measure(|| {
                        assert_eq!(
                            inputs::Shape::derive(&family, &corrupt).err(),
                            Some(super::super::bad())
                        );
                    });
                    assert_eq!(inconsistent.count_total, 0);
                    corrupt.graph.tensors[q.0 as usize].shape[0] = original_heads;
                    let softmax = &corrupt.graph.nodes[family.nodes[0].0 as usize];
                    let product = corrupt.graph.producer(softmax.inputs[0]).unwrap();
                    corrupt.graph.nodes[product.0 as usize].id = tessera_ir::NodeId(u32::MAX);
                    let invalid = allocation_counter::measure(|| {
                        let result =
                            std::panic::catch_unwind(|| inputs::Shape::derive(&family, &corrupt));
                        assert!(result.is_ok(), "public metadata must yield a typed error");
                        if let Ok(value) = result {
                            assert_eq!(value.err(), Some(super::super::bad()));
                        }
                    });
                    assert_eq!(invalid.count_total, 0);
                    let shape = inputs::Shape::derive(&family, &lowering).unwrap();
                    // Pin source identity independently of the helper used by the value oracle.
                    assert_eq!(
                        (shape.heads, shape.key_heads, shape.dim),
                        (
                            model.arch.n_heads as usize,
                            model.arch.n_kv_heads as usize,
                            model.arch.d_head as usize
                        )
                    );
                    let plan = shape.memory().unwrap();
                    let maxima = SecretBuf::filled(G::ONE, shape.maxima);
                    let refusal = allocation_counter::measure(|| {
                        assert_eq!(
                            score_inputs(&family, &lowering, &witness, &maxima[..maxima.len() - 1])
                                .err(),
                            Some(super::super::bad())
                        );
                    });
                    assert_eq!(refusal.count_total, 0);
                    let mut packed = None;
                    let construction = allocation_counter::measure(|| {
                        packed = Some(score_inputs(&family, &lowering, &witness, &maxima).unwrap());
                    });
                    assert_eq!(
                        u64::try_from(construction.bytes_current).unwrap(),
                        plan.retained_bytes()
                    );
                    assert_eq!(construction.bytes_max, plan.peak_bytes());
                    let packed = packed.unwrap();
                    // Independent coordinate oracle reads each real tensor directly. Every
                    // padded coordinate stays zero; no field evaluation or proof is invented.
                    for (l, &node) in family.nodes.iter().enumerate() {
                        let (q, k) = inputs::ids(&lowering, node).unwrap();
                        for (id, columns, heads, padded_heads) in [
                            (
                                q,
                                &packed.queries,
                                shape.heads,
                                shape.heads.next_power_of_two(),
                            ),
                            (k, &packed.keys, shape.key_heads, shape.key_heads),
                        ] {
                            let values = witness.tensor(id).unwrap();
                            for head in 0..padded_heads {
                                for pos in 0..(n as usize).next_power_of_two() {
                                    for d in 0..shape.dim {
                                        let at = ((l * padded_heads + head)
                                            * (n as usize).next_power_of_two()
                                            + pos)
                                            * shape.dim
                                            + d;
                                        let expected = if head < heads && pos < n as usize {
                                            values[(head * n as usize + pos) * shape.dim + d] as i8
                                        } else {
                                            0
                                        };
                                        assert_eq!(columns[at], expected);
                                    }
                                }
                            }
                        }
                    }
                    assert_eq!(&packed.maxima[..], &maxima[..]);
                    for columns in [&packed.queries, &packed.keys] {
                        let block = columns.len() >> family.lambda;
                        assert!(
                            columns[family.nodes.len() * block..]
                                .iter()
                                .all(|&v| v == 0)
                        );
                    }
                }
            }
        }
    }

    fn context() -> SessionContext {
        SessionContext {
            protocol: Ascii::new("tessera-protocol/1").unwrap(),
            mode: Mode::Live,
            params: ParamSetId(Ascii::new("test-small/1").unwrap()),
            statement_digest: [0x46; 32],
            session_id: [0x61; 16],
        }
    }

    fn tv_g(x: G) -> tv_field::Fp {
        tv_field::Fp::from_canonical(x.to_canonical_u64()).unwrap()
    }
    fn tv_f(x: F) -> tv_field::Fp2 {
        let [a, b] = x.coeffs();
        tv_field::Fp2::new(tv_g(a), tv_g(b))
    }

    fn verify(
        proof: &tessera_sumcheck::SumcheckProof<F>,
        claim: F,
        point: &[F],
        mask: &[G],
        other_vars: usize,
    ) -> Result<tv_core::sumcheck::Batch, tv_core::sumcheck::SumcheckError> {
        use tv_core::sumcheck as tv;
        use tv_field::Fp2 as E;
        use tv_hash::transcript as t;
        let context = t::SessionContext {
            protocol: t::Ascii::new("tessera-protocol/1").unwrap(),
            mode: t::Mode::Live,
            params: t::ParamSetId(t::Ascii::new("test-small/1").unwrap()),
            statement_digest: [0x46; 32],
            session_id: [0x61; 16],
        };
        let proof = tv::SumcheckProof {
            rounds: proof
                .rounds
                .iter()
                .map(|r| r.iter().copied().map(tv_f).collect())
                .collect(),
            final_evals: proof.final_evals.iter().copied().map(tv_f).collect(),
        };
        let instances = [
            tv::Instance {
                claim: tv_f(claim),
                vars: point.len(),
                evals: 2,
            },
            tv::Instance {
                claim: E::from(tv_g(G::from_u64_reduced(17 * (1 << other_vars)))),
                vars: other_vars,
                evals: 1,
            },
        ];
        tv::verify_batch(
            &mut t::Transcript::new(&context),
            tv::Prefix::Stage(1),
            3,
            &instances,
            &proof,
            |i, r, evals| {
                if i == 1 {
                    return Some(evals[0]);
                }
                // Independent literal public-factor evaluation over tv-field. Do not call
                // the optimized selector, LT, prefix, head map or row evaluation here.
                let mut mask: Vec<_> = mask.iter().copied().map(|v| E::from(tv_g(v))).collect();
                for &x in r {
                    for k in 0..mask.len() / 2 {
                        mask[k] = mask[2 * k] + x * (mask[2 * k + 1] - mask[2 * k]);
                    }
                    mask.truncate(mask.len() / 2);
                }
                let equality = point.iter().zip(r).fold(E::ONE, |v, (&a, &b)| {
                    let a = tv_f(a);
                    v * ((E::ONE - a) * (E::ONE - b) + a * b)
                });
                Some((evals[0] - evals[1]) * equality * mask[0])
            },
        )
    }

    #[allow(clippy::print_stdout)] // Explicit benchmark prints public dimensions and counters only.
    fn check(
        layers: usize,
        heads: usize,
        key_heads: usize,
        positions: usize,
        dim: usize,
        seed: u64,
    ) {
        let adversarial = seed == u64::MAX;
        let benchmark = seed == 0x461beef;
        let mut seed = seed;
        let mut draw = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed
        };
        let np = positions.next_power_of_two();
        let hp = heads.next_power_of_two();
        let lp = layers.next_power_of_two();
        let mut inputs = Inputs {
            layers,
            heads,
            key_heads,
            positions,
            dim,
            layer_vars: lp.trailing_zeros() as usize,
            queries: SecretBuf::new(),
            keys: SecretBuf::new(),
            maxima: SecretBuf::new(),
        };
        inputs
            .queries
            .extend((0..lp * hp * np * dim).map(|_| draw() as i8));
        inputs
            .keys
            .extend((0..lp * key_heads * np * dim).map(|_| draw() as i8));
        // Nonzero maxima even on padded rows: the final maximum evaluation is unmasked.
        inputs
            .maxima
            .extend((0..lp * hp * np).map(|_| G::from_u64_reduced(draw())));
        let vars = inputs.key_vars() + inputs.row_vars();
        let point = (0..vars)
            .map(|_| F::new(G::from_u64_reduced(draw()), G::from_u64_reduced(draw())))
            .collect();
        let score = Score {
            inputs: Arc::new(inputs),
            point,
        };
        let a = &score.inputs;
        let len = lp * hp * np * np;
        let mut maxima = vec![G::ZERO; len];
        let mut scores = vec![G::ZERO; len];
        let mut mask = vec![G::ZERO; len];
        for r in 0..lp * hp * np {
            maxima[r * np..(r + 1) * np].fill(a.maxima[r]);
        }
        for l in 0..lp {
            for h in 0..hp {
                for q in 0..np {
                    for i in 0..np {
                        let index = a.row(l, h, q) * np + i;
                        scores[index] = (0..dim).fold(G::ZERO, |s, d| {
                            s + a.query(l, h, q, d)
                                * a.key(l, (h * key_heads / heads).min(key_heads - 1), i, d)
                        });
                        if l < layers && h < heads && q < positions && i <= q {
                            mask[index] = G::ONE;
                        }
                    }
                }
            }
        }
        let equality = eq_table(&score.point).unwrap();
        let literal = Instance::sum_of_products(
            vec![
                Table::base(&maxima),
                Table::base(&scores),
                Table::ext(&equality).public(),
                Table::base(&mask).public(),
            ],
            vec![
                Term::new(F::ONE, vec![0, 2, 3]),
                Term::new(-F::ONE, vec![1, 2, 3]),
            ],
        );
        if benchmark {
            use std::hint::black_box;
            use std::time::Instant;
            for (name, instance, resident) in [
                (
                    "structured",
                    Instance::computed(&score),
                    a.queries.len() + a.keys.len() + 8 * a.maxima.len(),
                ),
                ("literal", literal.clone(), 40 * len),
            ] {
                let prove = || {
                    black_box(
                        prove_batch(
                            &mut Transcript::new(&context()),
                            Protocol::One,
                            Prefix::Stage(1),
                            7,
                            std::slice::from_ref(&instance),
                        )
                        .unwrap(),
                    )
                };
                for _ in 0..3 {
                    let _ = prove();
                }
                let mut timings = Vec::new();
                let mut peak = 0;
                for _ in 0..11 {
                    let start = Instant::now();
                    let info = allocation_counter::measure(|| {
                        let _ = prove();
                    });
                    timings.push(start.elapsed().as_nanos());
                    peak = peak.max(info.bytes_max);
                }
                timings.sort_unstable();
                println!(
                    "{name}: L={layers} H={heads}/{key_heads} n={positions} d={dim}; median_ns={}; input_bytes={resident}; sumcheck_peak_bytes={peak}",
                    timings[5]
                );
            }
        }
        // A larger unrelated instance exercises the structured instance's front padding.
        let other = vec![G::from_u64_reduced(17); 1usize << (vars + 2)];
        let extra = Instance::new(vec![Table::base(&other)]);
        if adversarial {
            let claim = (0..len).fold(F::ZERO, |s, i| {
                s + equality[i] * F::from(mask[i] * (maxima[i] - scores[i]))
            });
            for fault in 0..=3 {
                FAULT.with(|f| f.set(fault));
                let proof = prove_batch(
                    &mut Transcript::new(&context()),
                    Protocol::One,
                    Prefix::Stage(1),
                    3,
                    &[Instance::computed(&score), extra.clone()],
                )
                .unwrap()
                .into_proof();
                FAULT.with(|f| f.set(0));
                let verdict = verify(&proof, claim, &score.point, &mask, vars + 2);
                if fault == 0 {
                    assert!(verdict.is_ok());
                } else {
                    assert_eq!(
                        verdict.unwrap_err(),
                        tv_core::sumcheck::SumcheckError::FinalMismatch {
                            round: (vars + 2) as u32
                        },
                        "fault {fault}"
                    );
                }
            }
        }
        for degree in [3, 7] {
            let fast = prove_batch(
                &mut Transcript::new(&context()),
                Protocol::One,
                Prefix::Stage(1),
                degree,
                &[Instance::computed(&score), extra.clone()],
            )
            .unwrap()
            .into_proof();
            let slow = prove_batch(
                &mut Transcript::new(&context()),
                Protocol::One,
                Prefix::Stage(1),
                degree,
                &[literal.clone(), extra.clone()],
            )
            .unwrap()
            .into_proof();
            assert_eq!(
                fast.rounds, slow.rounds,
                "L={layers} H={heads}/{key_heads} n={positions} d={dim}"
            );
            assert_eq!(fast.final_evals, slow.final_evals);
        }
    }

    #[test]
    fn structured_messages_equal_literal_on_padded_grouped_domains() {
        for n in [1, 2, 3, 5, 8, 9, 16] {
            for (h, k) in [(1, 1), (2, 1), (3, 1), (4, 2)] {
                for l in [1, 3] {
                    for d in [1, 3] {
                        for seed in [1, 0x461, 0x4580026] {
                            check(l, h, k, n, d, seed);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn independent_sumcheck_rejects_wrong_prefix_mask_boundary_and_key_head() {
        check(3, 4, 2, 5, 3, u64::MAX);
    }

    /// Reproducible CPU comparison at the conformance models' largest context (64).
    /// Like the repository's other sumcheck benches, this reports timings without a CI
    /// performance threshold; inputs are public deterministic INT8 fixtures.
    #[test]
    #[ignore = "CPU timing and allocation benchmark; run explicitly in release"]
    fn benchmark_structured_against_literal_at_context_64() {
        for (l, h, k, d) in [(2, 4, 2, 16), (4, 8, 2, 32)] {
            check(l, h, k, 64, d, 0x461beef);
        }
    }
}
