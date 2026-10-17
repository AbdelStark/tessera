// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared test support: graph-building shorthands and an exact interpreter of the IR, written from
//! RFC-0006 'Operation set' and RFC-0005 'Integer semantics' with `i128` values, so that no value
//! a test graph computes wraps. It is the reference the range tests check bounds against and the
//! lowering tests run the lowered graphs on; it shares no code with the compiler.

#![allow(dead_code)] // each test binary uses part of it

use tessera_codec::Rational;
use tessera_compiler::{Range, Ranges, analyze};
use tessera_ir::{
    ArchFamily, ArchSpec, Broadcast, CommitmentClass, Graph, GraphBuilder, IntRange, IntType, Node,
    NodeKind, Op, Profile, PublicQuantSpec, QuantPoint, RopeSpec, Site, TableId, TensorDecl,
    TensorId,
};
use tessera_tables::{GeneratedTable, RopeTable, Table, TableKind, TableSpec};

// ----- Building graphs -------------------------------------------------------------------------

pub const ADVICE: NodeKind = NodeKind::Committed {
    commitment: CommitmentClass::Advice,
};

pub const WEIGHT: NodeKind = NodeKind::Committed {
    commitment: CommitmentClass::Weight,
};

pub fn decl(shape: &[u32], kind: NodeKind, lo: i128, hi: i128) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: IntType::I32,
        range: IntRange { lo, hi },
        kind,
    }
}

/// An output: its declared range is the analysis's to compute, so any placeholder will do.
pub fn out(shape: &[u32], kind: NodeKind) -> TensorDecl {
    decl(shape, kind, 0, 0)
}

pub fn virt(shape: &[u32]) -> TensorDecl {
    out(shape, NodeKind::Virtual)
}

pub fn lookup(shape: &[u32], table: u32) -> TensorDecl {
    out(
        shape,
        NodeKind::LookupOutput {
            table: TableId(table),
        },
    )
}

pub fn one(
    builder: &mut GraphBuilder,
    op: Op,
    inputs: &[TensorId],
    output: TensorDecl,
) -> TensorId {
    builder
        .node(op, inputs, vec![output])
        .expect("a valid node")[0]
}

// ----- Tables ----------------------------------------------------------------------------------

/// What the interpreter needs of a table: the value of a key in a class, and RoPE's pairs.
pub trait Lookup {
    fn id(&self) -> TableId;
    fn value(&self, key: i128, class: i128) -> i128;
    fn rope(&self, _position: u32, _frequency: u32) -> (i128, i128) {
        unreachable!("not a RoPE table")
    }
}

impl Lookup for GeneratedTable {
    fn id(&self) -> TableId {
        self.spec().id
    }
    fn value(&self, key: i128, class: i128) -> i128 {
        let key = i64::try_from(key).expect("a key in i64");
        let class = u8::try_from(class).expect("a class");
        i128::from(self.lookup(key, class).expect("a key inside the table"))
    }
}

impl Lookup for RopeTable {
    fn id(&self) -> TableId {
        self.spec().id
    }
    fn value(&self, _key: i128, _class: i128) -> i128 {
        unreachable!("RoPE is no lookup table")
    }
    fn rope(&self, position: u32, frequency: u32) -> (i128, i128) {
        let (c, s) = self.constants().at(position, frequency).expect("a pair");
        (i128::from(c), i128::from(s))
    }
}

/// The tables a claim list states, regenerated from their specifications.
pub fn regenerate(specs: &[TableSpec]) -> Vec<Box<dyn Lookup>> {
    specs
        .iter()
        .map(|spec| -> Box<dyn Lookup> {
            if matches!(spec.kind, TableKind::Rope { .. }) {
                Box::new(RopeTable::from_spec(spec).expect("a RoPE table"))
            } else {
                Box::new(GeneratedTable::from_spec(spec).expect("a table"))
            }
        })
        .collect()
}

// ----- Integer helpers --------------------------------------------------------------------------

pub fn elements(shape: &[u32]) -> usize {
    shape.iter().map(|&d| d as usize).product()
}

/// `rq(x, r, lo, hi) = clamp(floor((x + 2^{r−1}) / 2^r), lo, hi)` of RFC-0005.
pub fn rq(x: i128, r: u8, lo: i64, hi: i64) -> i128 {
    ((x + (1 << (r - 1))) >> r).clamp(i128::from(lo), i128::from(hi))
}

/// `⌊√n⌋` for `n ≥ 0`.
pub fn isqrt(n: i128) -> i128 {
    assert!(n >= 0, "a square root of a non-negative number");
    let mut r = (n as f64).sqrt() as i128;
    while r * r > n {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= n {
        r += 1;
    }
    r
}

// ----- An exact interpreter of the IR ---------------------------------------------------------

/// The honest values of every tensor of `graph`, flat and row-major, from the inputs' values.
///
/// # Panics
///
/// When an input has no value or the wrong number of them, or a key misses its table.
pub fn evaluate(
    graph: &Graph,
    tables: &[Box<dyn Lookup>],
    inputs: Vec<(TensorId, Vec<i128>)>,
) -> Vec<Vec<i128>> {
    let mut values = vec![None; graph.tensors.len()];
    for (t, v) in inputs {
        assert_eq!(v.len(), elements(&graph.tensors[t.0 as usize].shape));
        values[t.0 as usize] = Some(v);
    }
    let mut run = Run {
        graph,
        tables,
        values,
    };
    for node in &graph.nodes {
        let outputs = run.node(node);
        for (&t, v) in node.outputs.iter().zip(outputs) {
            run.values[t.0 as usize] = Some(v);
        }
    }
    run.values
        .into_iter()
        .map(|v| v.expect("every tensor computed"))
        .collect()
}

struct Run<'a> {
    graph: &'a Graph,
    tables: &'a [Box<dyn Lookup>],
    values: Vec<Option<Vec<i128>>>,
}

impl Run<'_> {
    fn get(&self, t: TensorId) -> &[i128] {
        self.values[t.0 as usize].as_deref().expect("computed")
    }

    fn shape(&self, t: TensorId) -> &[u32] {
        &self.graph.tensors[t.0 as usize].shape
    }

    fn table(&self, id: TableId) -> &dyn Lookup {
        self.tables
            .iter()
            .find(|t| t.id() == id)
            .expect("a table")
            .as_ref()
    }

    fn node(&self, node: &Node) -> Vec<Vec<i128>> {
        let x = |k: usize| self.get(node.inputs[k]);
        let shape = |k: usize| self.shape(node.inputs[k]);
        let last = |k: usize| *shape(k).last().expect("an axis") as usize;
        // The channel class of element `i`: the class vector's entry for its last-axis position.
        let class = |i: usize| {
            if node.inputs.len() > 1 {
                x(1)[i % last(0)]
            } else {
                0
            }
        };
        match &node.op {
            Op::MatMulWeight { .. } => {
                let w = self.shape(node.inputs[1]);
                let (rows, cols) = (w[0] as usize, w[1] as usize);
                let (input, weight) = (x(0), x(1));
                let mut y = Vec::new();
                for row in input.chunks(cols) {
                    for o in 0..rows {
                        y.push((0..cols).map(|i| weight[o * cols + i] * row[i]).sum());
                    }
                }
                vec![y]
            }
            Op::Add => vec![x(0).iter().zip(x(1)).map(|(a, b)| a + b).collect()],
            Op::Sub => vec![x(0).iter().zip(x(1)).map(|(a, b)| a - b).collect()],
            Op::ConstMul { c } | Op::ConstAdd { c } => {
                let values = &self.graph.constants[c.0 as usize];
                let n = last(0);
                let y = x(0).iter().enumerate().map(|(i, &v)| {
                    let c = i128::from(if values.len() == 1 {
                        values[0]
                    } else {
                        values[i % n]
                    });
                    if matches!(node.op, Op::ConstMul { .. }) {
                        c * v
                    } else {
                        c + v
                    }
                });
                vec![y.collect()]
            }
            Op::Rotate { rope } => {
                // Pairs (x_i, x_{i+m}) of each last-axis vector of 2m, by its row's position.
                let table = self.table(*rope);
                let (rows, width) = (shape(0)[0] as usize, last(0));
                let per_row = elements(shape(0)) / rows;
                let half = width / 2;
                let positions = x(1);
                let mut y = x(0).to_vec();
                for (k, vector) in y.chunks_mut(width).enumerate() {
                    let row = k * width / per_row;
                    let p = u32::try_from(positions[row]).expect("a position");
                    for i in 0..half {
                        let (c, s) = table.rope(p, u32::try_from(i).expect("small"));
                        let (a, b) = (vector[i], vector[i + half]);
                        vector[i] = a * c - b * s;
                        vector[i + half] = a * s + b * c;
                    }
                }
                vec![y]
            }
            Op::Sum { axis } | Op::SumSquares { axis } => {
                let s = shape(0);
                let axis = usize::from(*axis);
                let inner: usize = s[axis + 1..].iter().map(|&d| d as usize).product();
                let len = s[axis] as usize;
                let outer = elements(s) / (inner * len);
                let square = matches!(node.op, Op::SumSquares { .. });
                let mut y = Vec::new();
                for o in 0..outer {
                    for i in 0..inner {
                        y.push(
                            (0..len)
                                .map(|k| x(0)[(o * len + k) * inner + i])
                                .map(|v| if square { v * v } else { v })
                                .sum(),
                        );
                    }
                }
                vec![y]
            }
            Op::Reshape => vec![x(0).to_vec()],
            Op::Slice { axis, start, len } => {
                let s = shape(0);
                let axis = usize::from(*axis);
                let inner: usize = s[axis + 1..].iter().map(|&d| d as usize).product();
                let full = s[axis] as usize;
                let outer = elements(s) / (inner * full);
                let mut y = Vec::new();
                for o in 0..outer {
                    for k in *start as usize..(*start + *len) as usize {
                        y.extend_from_slice(&x(0)[(o * full + k) * inner..][..inner]);
                    }
                }
                vec![y]
            }
            Op::GatherPublic { index } => {
                let row = elements(shape(0)) / shape(0)[0] as usize;
                let rows = &self.graph.constants[index.0 as usize];
                vec![
                    rows.iter()
                        .flat_map(|&r| x(0)[r as usize * row..][..row].to_vec())
                        .collect(),
                ]
            }
            Op::Transpose { perm } => {
                // Output element by element: decompose its index, then read the input there.
                let s: Vec<usize> = shape(0).iter().map(|&d| d as usize).collect();
                let out: Vec<usize> = perm.iter().map(|&a| s[usize::from(a)]).collect();
                let y = (0..elements(shape(0)))
                    .map(|flat| {
                        let mut digits = vec![0usize; out.len()];
                        let mut rest = flat;
                        for k in (0..out.len()).rev() {
                            digits[k] = rest % out[k];
                            rest /= out[k];
                        }
                        let mut at = vec![0usize; s.len()];
                        for (k, &a) in perm.iter().enumerate() {
                            at[usize::from(a)] = digits[k];
                        }
                        let index = at.iter().zip(&s).fold(0, |i, (&d, &n)| i * n + d);
                        x(0)[index]
                    })
                    .collect();
                vec![y]
            }
            Op::Expand { axis, len } => {
                // Output element by element: drop the new axis's digit from its index.
                let s: Vec<usize> = shape(0).iter().map(|&d| d as usize).collect();
                let axis = usize::from(*axis);
                let inner: usize = s[axis..].iter().product();
                let copies = *len as usize;
                let y = (0..elements(shape(0)) * copies)
                    .map(|flat| {
                        let (outer, within) = (flat / (copies * inner), flat % inner);
                        x(0)[outer * inner + within]
                    })
                    .collect();
                vec![y]
            }
            Op::MatMulAct { transpose_rhs } => {
                let (a, b) = (shape(0), shape(1));
                let r = a.len();
                let (m, k) = (a[r - 2] as usize, a[r - 1] as usize);
                let n = if *transpose_rhs { b[r - 2] } else { b[r - 1] } as usize;
                let batches = elements(a) / (m * k);
                let mut y = Vec::new();
                for batch in 0..batches {
                    let (lhs, rhs) = (&x(0)[batch * m * k..], &x(1)[batch * k * n..]);
                    for i in 0..m {
                        for j in 0..n {
                            y.push(
                                (0..k)
                                    .map(|t| {
                                        let right = if *transpose_rhs {
                                            rhs[j * k + t]
                                        } else {
                                            rhs[t * n + j]
                                        };
                                        lhs[i * k + t] * right
                                    })
                                    .sum(),
                            );
                        }
                    }
                }
                vec![y]
            }
            Op::Mul { broadcast } => {
                let (a, b) = (x(0), x(1));
                vec![match broadcast {
                    Broadcast::Trailing => {
                        let trailing = elements(shape(0)) / elements(shape(1));
                        a.iter()
                            .enumerate()
                            .map(|(i, &v)| v * b[i / trailing])
                            .collect()
                    }
                    Broadcast::Leading => a
                        .iter()
                        .zip(b.iter().cycle())
                        .map(|(&v, &m)| v * m)
                        .collect(),
                }]
            }
            Op::OneHot { vocab } => {
                let v = *vocab as usize;
                vec![
                    x(0).iter()
                        .flat_map(|&t| (0..v).map(move |j| i128::from(j as i128 == t)))
                        .collect(),
                ]
            }
            Op::Requant { shift, lo, hi } => {
                let classes = &self.graph.shift_classes[shift.0 as usize];
                let y = x(0).iter().enumerate().map(|(i, &v)| {
                    let c = usize::try_from(class(i)).expect("a class");
                    rq(v, classes[c], *lo, *hi)
                });
                vec![y.collect()]
            }
            Op::TableMap { table } => {
                let t = self.table(*table);
                vec![
                    x(0).iter()
                        .enumerate()
                        .map(|(i, &k)| t.value(k, class(i)))
                        .collect(),
                ]
            }
            Op::SoftmaxRow { exp_table, sigma } => {
                // Causal: row q of each [n, n] block sees keys i ≤ q; the others' exponentials
                // are 0 (RFC-0009 'SoftmaxRow').
                let t = self.table(*exp_table);
                let n = last(0);
                let (mut e, mut r) = (Vec::new(), Vec::new());
                for (k, row) in x(0).chunks(n).enumerate() {
                    let q = k % n;
                    let max = *row[..=q].iter().max().expect("a score");
                    let exps: Vec<i128> = (0..n)
                        .map(|i| if i <= q { t.value(max - row[i], 0) } else { 0 })
                        .collect();
                    let sum: i128 = exps.iter().sum();
                    r.push((1i128 << (2 * u32::from(*sigma))) / sum);
                    e.extend(exps);
                }
                vec![e, r]
            }
            Op::RsqrtRow {
                kappa,
                eps_int,
                dim,
            } => {
                // ρ = max{ρ : ρ^2 (u + e) ≤ N} = ⌊√⌊N / (u + e)⌋⌋, N = d · 2^{2κ}.
                let big = i128::from(*dim) << (2 * u32::from(*kappa));
                vec![
                    x(0).iter()
                        .map(|&u| isqrt(big / (u + i128::from(*eps_int))))
                        .collect(),
                ]
            }
            Op::Bitwise { table } => {
                let t = self.table(*table);
                vec![
                    x(0).iter()
                        .zip(x(1))
                        .map(|(&a, &b)| t.value(a + 256 * b, 0))
                        .collect(),
                ]
            }
            Op::Argmax | Op::InverseCdf { .. } => Vec::new(),
        }
    }
}

/// Every value of every tensor inside its range in `ranges`, where bounded.
///
/// # Panics
///
/// At the first value outside its range.
pub fn check_ranges(ranges: &[Range], values: &[Vec<i128>]) {
    for (k, (range, values)) in ranges.iter().zip(values).enumerate() {
        if let Range::Bounded(r) = range {
            for &v in values {
                assert!(
                    r.lo <= v && v <= r.hi,
                    "tensor {k}: {v} outside [{}, {}]",
                    r.lo,
                    r.hi
                );
            }
        }
    }
}

/// Analyzes `graph`, runs it on the inputs' values, and checks every tensor against its range.
pub fn check_run(
    graph: &Graph,
    tables: &[Box<dyn Lookup>],
    specs: &[TableSpec],
    inputs: Vec<(TensorId, Vec<i128>)>,
) -> (Ranges, Vec<Vec<i128>>) {
    let ranges = analyze(graph, specs).expect("the graph analyzes");
    let values = evaluate(graph, tables, inputs);
    check_ranges(ranges.all(), &values);
    (ranges, values)
}

/// A seeded generator for test inputs: SplitMix64.
pub struct Seeded(pub u64);

impl Seeded {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `[lo, hi]`.
    pub fn range(&mut self, lo: i128, hi: i128) -> i128 {
        let span = u128::try_from(hi - lo + 1).expect("lo ≤ hi");
        lo + i128::try_from(u128::from(self.next()) % span).expect("small")
    }
}

// ----- Llama-3.1 shapes -------------------------------------------------------------------------

pub const ROPE: RopeSpec = RopeSpec {
    theta: 500_000,
    factor: 8,
    low_freq_factor: 1,
    high_freq_factor: 4,
    original_max: 8192,
    bits: 14,
};

pub fn sites(profile: Profile) -> Vec<Site> {
    let all = [
        Site::AttnNorm,
        Site::RopeQ,
        Site::RopeK,
        Site::V,
        Site::AttnOut,
        Site::AttnResidual,
        Site::MlpNorm,
        Site::Silu,
        Site::Up,
        Site::Product,
        Site::MlpResidual,
    ];
    let servable_only = [Site::AttnOut, Site::Up, Site::Product];
    all.into_iter()
        .filter(|s| profile == Profile::Servable || !servable_only.contains(s))
        .collect()
}

/// The public part of a model of Llama-3.1's shape and `layers` layers, quantized as RFC-0005
/// describes, every site of one class. The residual's scale is `2^{−7}`, so that RMSNorm's
/// `e = ⌈d · ε / σ_res^2⌉` is below `d / 4` (#285's case), and `κ` is RFC-0023's: the largest,
/// at most 31, with `⌊d · 2^{2κ} / e⌋ < 2^64`.
pub fn llama(
    profile: Profile,
    layers: u32,
    d: u32,
    heads: u32,
    kv_heads: u32,
    d_ff: u32,
) -> (ArchSpec, PublicQuantSpec, Vec<(u32, i8)>) {
    let arch = ArchSpec {
        family: ArchFamily::LlamaV3,
        n_layers: layers,
        d_model: d,
        n_heads: heads,
        n_kv_heads: kv_heads,
        d_head: 128,
        d_ff,
        vocab: 128_256,
        max_context: 131_072,
        rope: ROPE,
        rms_eps: Rational::new(1, 100_000).expect("a rational"),
    };
    let mut act_exponents = vec![(QuantPoint::Embed, -6)];
    // ⌈d · 10^{−5} · 2^{14}⌉: 672 at 8B, 2685 at 405B; then κ = 30 at both.
    let eps = (u64::from(d) * (1 << 14)).div_ceil(100_000);
    let kappa = (0..=31u8)
        .rev()
        .find(|&k| (u128::from(d) << (2 * k)) / u128::from(eps) < 1 << 64)
        .expect("κ = 0 fits");
    let mut shift_class_sets = Vec::new();
    for layer in 0..layers {
        for site in sites(profile) {
            let point = QuantPoint::Layer { layer, site };
            if !matches!(site, Site::AttnResidual | Site::MlpResidual) {
                act_exponents.push((point, -3));
            }
            // Shifts of the size scales give: a norm's κ + e_y; RoPE's 14-bit constants and the
            // attention output's 2^{2σ} in theirs; a Silu shift wide enough that its 2^{b − r}
            // values of h stay below 2^24. `minimal`'s residual sites read unrequantized inputs.
            let minimal = profile == Profile::Minimal;
            let r = match site {
                Site::AttnNorm | Site::MlpNorm => i8::try_from(kappa).expect("small") - 3,
                Site::RopeQ | Site::RopeK => 22,
                Site::V | Site::Up => 8,
                Site::AttnOut => 32,
                Site::AttnResidual if minimal => 37,
                Site::MlpResidual if minimal => 17,
                Site::AttnResidual | Site::MlpResidual => 4,
                Site::Silu => 20,
                Site::Product => 5,
            };
            shift_class_sets.push((point, vec![r]));
        }
    }
    act_exponents.push((QuantPoint::FinalNorm, -3));
    let final_shift = i8::try_from(kappa).expect("small") - 3;
    shift_class_sets.push((QuantPoint::FinalNorm, vec![final_shift]));
    let quant = PublicQuantSpec {
        act_exponents,
        residual_exponent: -7,
        residual_bits: 20,
        rsqrt_kappa: kappa,
        rms_eps_int: eps,
        exp_sigma: 16,
        score_scales: (0..layers)
            .map(|l| (l, Rational::new(1, 1 << 14).expect("a scale")))
            .collect(),
        shift_class_sets,
        affine_sites: Vec::new(),
        rope: ROPE,
        logits_bits: 28,
        logits_exponent: -11,
    };
    let silu = (0..layers).map(|l| (l, -6)).collect();
    (arch, quant, silu)
}
