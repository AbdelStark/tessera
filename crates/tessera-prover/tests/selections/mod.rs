// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The selection relation, combined selections and `BitFn` (RFC-0072 'The selection relation',
//! 'Combined selections', 'Testing Strategy'; RFC-0066 '`BitFn`'), proven by the stage prover and
//! checked by the verifier of this test, which evaluates every public factor itself from public
//! data: a selection's kernel by RFC-0072's closed form `S̃(r_a, r'_k) = Σ_{j < len} eq(r_a, j)·
//! eq(r'_k, idx_j)`, a combined selection's `K̃ = Σ_t a_t·S̃_t`, `eq`, and `BitFn`'s polynomial
//! from RFC-0066's table.
//!
//! - **The selection relation.** A rotation (a `GatherPublic` of 32 rows), the value levels' even
//!   and odd rows, a `Slice` on axes 0, 1 and a middle one, `len`s that are not powers of two and
//!   the message's assembly prove and verify, each schedule's stage count the report's; a changed
//!   input entry that the selection reads is rejected at its stage's final check. The closed
//!   form is the kernel's extension enumerated over the padded cube.
//! - **Combined selections.** A level `Add(even, ConstMul(odd))` proves as one instance, and the
//!   five levels of a value as one per stage; a verifier that moves one member's rows rejects
//!   at the stage's final check. Two claims on one tensor at one point stay two instances.
//! - **`BitFn`.** Each function on random bits proves and verifies, its stage of degree 3, or 4
//!   for `Xor3` and `Maj`, and a changed input bit is rejected.

use super::*;
use tessera_compiler::schedule::{Instance as Planned, Kind};
use tessera_ir::BoolFn;

// ----- The verifier's closed forms (RFC-0072 'The selection relation', RFC-0066 '`BitFn`') -----

/// RFC-0066's multilinear polynomial of `f` at its inputs' values `x`.
pub(super) fn bit_function(f: BoolFn, x: &[F]) -> F {
    let c = |v: u64| F::from(Goldilocks::from_u64_reduced(v));
    let (a, b) = (x[0], x[1]);
    match f {
        BoolFn::Xor => a + b - c(2) * a * b,
        BoolFn::Xor3 => {
            let d = x[2];
            a + b + d - c(2) * (a * b + a * d + b * d) + c(4) * a * b * d
        }
        BoolFn::And => a * b,
        // `ef + g − eg`.
        BoolFn::Ch => a * b + x[2] - a * x[2],
        BoolFn::Maj => {
            let d = x[2];
            a * b + a * d + b * d - c(2) * a * b * d
        }
    }
}

/// The axis a selection's node selects along and its rows `idx_j`, `j < len`: a `Slice`'s
/// `start + j` along its axis, a `GatherPublic`'s constant along the outermost.
fn rows_of(graph: &Graph, node: &tessera_ir::Node) -> (usize, Vec<u64>) {
    match node.op {
        Op::Slice { axis, start, len } => (
            usize::from(axis),
            (u64::from(start)..u64::from(start) + u64::from(len)).collect(),
        ),
        Op::GatherPublic { index } => (
            0,
            graph.constants[index.0 as usize]
                .iter()
                .map(|&k| u64::try_from(k).expect("a row"))
                .collect(),
        ),
        _ => panic!("a selection"),
    }
}

/// `S̃(r_a, r'_k) = Σ_{j < len} eq(r_a, j)·eq(r'_k, idx_j)`, in `O(len·(⌈log2 len⌉ +
/// ⌈log2 n_a⌉))`.
fn kernel_at(index: &[u64], r_a: &[F], r_k: &[F]) -> F {
    (0u64..)
        .zip(index)
        .fold(F::ZERO, |sum, (j, &k)| sum + eq_at(r_a, j) * eq_at(r_k, k))
}

impl Verifier<'_> {
    /// A selection of stage `k` at its point `target = r_lo ‖ r_a ‖ r_hi`, from its final point
    /// `r = r'_k` and its evaluation `e`: `K̃(r_a, r'_k)·x̃`, the kernel its own `S̃` for a single
    /// selection, `Σ_t a_t·S̃_t` over its members for a combined one, each member's rows its
    /// node's (RFC-0072 'Combined selections').
    pub(super) fn selection_relation(
        &self,
        k: u16,
        instance: &Planned,
        node: &tessera_ir::Node,
        target: &[F],
        r: &[F],
        e: &[F],
    ) -> F {
        let (axis, _) = rows_of(self.graph, node);
        let y = &self.graph.tensors[node.outputs[0].0 as usize].shape;
        let at: usize = y[axis + 1..].iter().map(|&d| axis_vars(d) as usize).sum();
        let r_a = &target[at..at + axis_vars(y[axis]) as usize];
        let n_a = u64::from(self.graph.tensors[node.inputs[0].0 as usize].shape[axis]);
        let atom = self
            .schedule
            .atom_of(k, instance.source)
            .expect("a selection proves a part");
        let members: Vec<(F, &tessera_ir::Node)> = if atom.combined.is_empty() {
            vec![(F::ONE, node)]
        } else {
            atom.combined
                .iter()
                .map(|member| {
                    let Kind::Tensor(t) = member.kind else {
                        panic!("a selection outside the layers")
                    };
                    let producer = self
                        .graph
                        .nodes
                        .iter()
                        .find(|n| n.outputs.contains(&t))
                        .expect("a selection's output");
                    (self.coefficient(&member.coefficient, 0), producer)
                })
                .collect()
        };
        let last = members.len() - 1;
        let mut kernel = F::ZERO;
        for (t, (a, member)) in members.into_iter().enumerate() {
            let (_, mut index) = rows_of(self.graph, member);
            if self.publics.kernel == Kernel::LastMemberMoved && t == last {
                index = index.iter().map(|&i| (i + 1) % n_a).collect();
            }
            kernel += a * kernel_at(&index, r_a, r);
        }
        kernel * e[0]
    }
}

// ----- Graphs and runs --------------------------------------------------------------------------

/// The run of `graph` on the given values of every input, as [`Run::of`] on random ones: its
/// schedule's stage count is the report's, and the honest proof is proven, not yet verified.
pub(super) fn run_with(graph: &Graph, inputs: Vec<(TensorId, Vec<i64>)>, seed: u64) -> Run {
    let schedule = derive_graph_schedule(graph, &[], &[], None).expect("a schedule");
    assert_eq!(
        schedule.stage_count(),
        tessera_compiler::stage_count(graph) as usize,
        "the report's count"
    );
    let flags = public_tensors(graph);
    let public = inputs
        .iter()
        .filter(|(t, _)| flags[t.0 as usize])
        .cloned()
        .collect();
    let witness = Engine::for_graph(graph, &[])
        .expect("an engine")
        .evaluate(
            inputs
                .into_iter()
                .map(|(t, v)| (t, v.into_iter().collect()))
                .collect(),
        )
        .expect("a witness");
    let points = leaf_points(&schedule, seed);
    let (proof, pool) = prove_graph(graph, &[], &schedule, &witness, &points);
    let pool = pool.into_iter().map(|v| v.expect("a value")).collect();
    Run {
        schedule,
        witness,
        public,
        points,
        proof,
        pool,
    }
}

/// The verifier's verdict on the honest proof of `run` when it evaluates kernels by `kernel`.
fn verdict(run: &Run, graph: &Graph, kernel: Kernel) -> Result<Vec<F>, Reject> {
    let values = public_values(graph, &run.public);
    let publics = Publics {
        values: &values,
        shift: Shift::Extension,
        kernel,
    };
    verify_public(graph, &[], &run.schedule, &run.proof, &run.points, publics)
}

/// The proof of a prover whose tensor `tensor` has entry `entry` of its real values changed by
/// one, every other tensor the honest witness's: the table the selection reads, when `tensor`
/// is its input.
fn changed(run: &Run, graph: &Graph, tensor: TensorId, entry: usize) -> Proof {
    run.tampered(graph, tensor, entry)
}

/// The instances of stage `k`, by relation.
fn relations(run: &Run, k: usize) -> Vec<Relation> {
    run.schedule.stages[k]
        .instances
        .iter()
        .map(|i| i.relation)
        .collect()
}

/// A gather of `x`'s rows `index`, of `inner` positions each.
fn gather(g: &mut GraphBuilder, x: TensorId, index: Vec<i64>, inner: &[u32]) -> TensorId {
    let mut shape = vec![u32::try_from(index.len()).expect("small")];
    shape.extend_from_slice(inner);
    let index = g.constant(index).expect("rows");
    one(g, Op::GatherPublic { index }, &[x], wide(&shape))
}

/// `x[.., start..start + len, ..]` along `axis`.
fn slice(
    g: &mut GraphBuilder,
    x: TensorId,
    shape: &[u32],
    axis: u8,
    start: u32,
    len: u32,
) -> TensorId {
    let mut out = shape.to_vec();
    out[usize::from(axis)] = len;
    one(
        g,
        Op::Slice { axis, start, len },
        &[x],
        decl(&out, NodeKind::Virtual),
    )
}

/// A committed input of `shape` in `[lo, hi]`.
fn committed(g: &mut GraphBuilder, shape: &[u32], lo: i128, hi: i128) -> TensorId {
    let mut input = decl(shape, ADVICE);
    input.range = IntRange { lo, hi };
    g.input(input).expect("an input")
}

/// The flat index of `at` in a tensor of `shape`, row-major over its real positions.
fn flat(shape: &[u32], at: &[u32]) -> usize {
    shape
        .iter()
        .zip(at)
        .fold(0, |index, (&len, &i)| index * len as usize + i as usize)
}

// ----- The selection relation -------------------------------------------------------------------

/// A selection of a committed `x` whose output a requantization reads.
struct Case {
    name: &'static str,
    /// `x`'s shape.
    shape: &'static [u32],
    /// The selection of `x`, built into a graph.
    select: fn(&mut GraphBuilder, TensorId) -> (TensorId, Vec<u32>),
    /// A position of `x` the selection reads.
    read: &'static [u32],
}

/// RFC-0072 'Testing Strategy' 'Relations': the selection relation for a rotation, the value
/// levels' even and odd rows, a `Slice` on axes 0 and 1 and a middle one, and `len`s that are
/// not powers of two.
fn cases() -> Vec<Case> {
    fn rotation(g: &mut GraphBuilder, x: TensorId) -> (TensorId, Vec<u32>) {
        let y = gather(g, x, (0..32).map(|i| (i + 7) % 32).collect(), &[3]);
        (y, vec![32, 3])
    }
    fn even(g: &mut GraphBuilder, x: TensorId) -> (TensorId, Vec<u32>) {
        (
            gather(g, x, (0..32).step_by(2).collect(), &[3]),
            vec![16, 3],
        )
    }
    fn odd(g: &mut GraphBuilder, x: TensorId) -> (TensorId, Vec<u32>) {
        (
            gather(g, x, (1..32).step_by(2).collect(), &[3]),
            vec![16, 3],
        )
    }
    fn rows(g: &mut GraphBuilder, x: TensorId) -> (TensorId, Vec<u32>) {
        (slice(g, x, &[6, 5], 0, 1, 4), vec![4, 5])
    }
    fn columns(g: &mut GraphBuilder, x: TensorId) -> (TensorId, Vec<u32>) {
        (slice(g, x, &[6, 5], 1, 2, 3), vec![6, 3])
    }
    fn middle(g: &mut GraphBuilder, x: TensorId) -> (TensorId, Vec<u32>) {
        (slice(g, x, &[3, 6, 5], 1, 1, 5), vec![3, 5, 5])
    }
    fn uneven(g: &mut GraphBuilder, x: TensorId) -> (TensorId, Vec<u32>) {
        (gather(g, x, vec![4, 0, 4, 2, 1], &[3]), vec![5, 3])
    }
    fn short(g: &mut GraphBuilder, x: TensorId) -> (TensorId, Vec<u32>) {
        (slice(g, x, &[6, 5], 0, 1, 3), vec![3, 5])
    }
    fn word(g: &mut GraphBuilder, x: TensorId) -> (TensorId, Vec<u32>) {
        (slice(g, x, &[96, 3], 0, 32, 32), vec![32, 3])
    }
    vec![
        Case {
            name: "a rotation",
            shape: &[32, 3],
            select: rotation,
            read: &[7, 1],
        },
        Case {
            name: "a level's even rows",
            shape: &[32, 3],
            select: even,
            read: &[30, 2],
        },
        Case {
            name: "a level's odd rows",
            shape: &[32, 3],
            select: odd,
            read: &[31, 0],
        },
        Case {
            name: "a Slice on axis 0, len 4 of 6",
            shape: &[6, 5],
            select: rows,
            read: &[4, 3],
        },
        Case {
            name: "a Slice on axis 1, len 3 of 5",
            shape: &[6, 5],
            select: columns,
            read: &[5, 2],
        },
        Case {
            name: "a Slice on a middle axis, len 5 of 6",
            shape: &[3, 6, 5],
            select: middle,
            read: &[2, 5, 4],
        },
        Case {
            name: "a Slice on axis 0, len 3 of 6, rows past it",
            shape: &[6, 5],
            select: short,
            read: &[3, 4],
        },
        Case {
            name: "a gather of 5 rows of 5, one twice",
            shape: &[5, 3],
            select: uneven,
            read: &[4, 0],
        },
        Case {
            name: "a word of a message",
            shape: &[96, 3],
            select: word,
            read: &[63, 1],
        },
    ]
}

/// Each selection proves and verifies as one instance of `⌈log2 n_a⌉` variables and degree 2,
/// with one claim, on `x`; a changed entry of `x` that it reads is rejected at stage 1's final
/// check.
#[test]
fn each_selection_proves_and_verifies() {
    for case in cases() {
        let name = case.name;
        let mut g = GraphBuilder::new();
        let x = committed(&mut g, case.shape, -3, 3);
        let (y, shape) = (case.select)(&mut g, x);
        requant(&mut g, y, &shape, 100);
        let graph = g.finish();
        let run = Run::of(&graph, &[x], 0x0526_3100);
        assert_eq!(relations(&run, 1), [Relation::Selection], "{name}");
        let node = graph
            .nodes
            .iter()
            .find(|n| n.outputs == [y])
            .expect("the selection");
        let (axis, _) = rows_of(&graph, node);
        let instance = &run.schedule.stages[1].instances[0];
        assert_eq!(
            instance.vars,
            axis_vars(case.shape[axis]),
            "{name}: the selected axis's variables"
        );
        assert_eq!(instance.claims.len(), 1, "{name}: one claim");
        assert!(
            run.proof.stages[0]
                .sumcheck
                .rounds
                .iter()
                .all(|round| round.len() == 2),
            "{name}: degree 2"
        );
        run.accepted(&graph);
        let bad = changed(&run, &graph, x, flat(case.shape, case.read));
        assert_eq!(
            run.verify_with(&graph, &bad, &run.public, Shift::Extension),
            Err(run.end(1)),
            "{name}: a changed input entry"
        );
    }
}

/// The verifier's `S̃(r_a, r'_k)` of each selection above is the extension of its kernel
/// `S(j, k) = [j < len]·[idx_j = k]` enumerated over the padded cube of `y`'s and `x`'s axes, at
/// random points.
#[test]
fn the_kernels_closed_form_is_its_extension() {
    for case in cases() {
        let mut g = GraphBuilder::new();
        let x = committed(&mut g, case.shape, -3, 3);
        let (y, shape) = (case.select)(&mut g, x);
        let graph = g.finish();
        let node = graph
            .nodes
            .iter()
            .find(|n| n.outputs == [y])
            .expect("the selection");
        let (axis, index) = rows_of(&graph, node);
        let (rows, vars) = (
            axis_vars(shape[axis]) as usize,
            axis_vars(case.shape[axis]) as usize,
        );
        for r in random_points(rows + vars, 3, 0x0526_3200) {
            let (r_k, r_a) = r.split_at(vars);
            let mut table = vec![Goldilocks::ZERO; 1 << (rows + vars)];
            for (j, &k) in index.iter().enumerate() {
                table[(j << vars) | k as usize] = Goldilocks::ONE;
            }
            let enumerated = evaluate_base::<F>(&table, &r).expect("a point");
            assert_eq!(kernel_at(&index, r_a, r_k), enumerated, "{}", case.name);
        }
    }
}

/// RFC-0066 'The message': committed bytes decomposed, their bits transposed and reshaped to
/// `[8K, M]`, gathered into the 512 rows of a block's words and zeroed at public bytes by
/// `place`, then XORed with the public bits. It proves and verifies through the `Xor`, the
/// placing `Mul` and the gather's selection, beside stage 1's bit zero-checks; a changed entry of
/// the gathered bits, as the selection reads them, is rejected at its stage, the third.
#[test]
fn the_message_assembly_proves_and_verifies() {
    // Bytes 21 to 23 of three messages committed.
    let committed_offsets = [21usize, 22, 23];
    let (k, m) = (3u32, 3u32);
    let mut g = GraphBuilder::new();
    let message = g
        .input(TensorDecl {
            shape: vec![512, m],
            ty: IntType::I32,
            range: IntRange { lo: 0, hi: 1 },
            kind: PUBLIC,
        })
        .expect("message");
    let bytes = committed(&mut g, &[k, m], 0, 255);
    let place = g
        .input(TensorDecl {
            shape: vec![512],
            ty: IntType::I32,
            range: IntRange { lo: 0, hi: 1 },
            kind: PUBLIC,
        })
        .expect("place");
    let mut bits_decl = decl(&[k, m, 8], ADVICE);
    bits_decl.range = IntRange { lo: 0, hi: 1 };
    let b = g
        .node(
            Op::BitDecompose { low: 8, high: 0 },
            &[bytes],
            vec![bits_decl],
        )
        .expect("bits")[0];
    let virt = |shape: &[u32]| decl(shape, NodeKind::Virtual);
    let turned = one(
        &mut g,
        Op::Transpose {
            perm: vec![0, 2, 1],
        },
        &[b],
        virt(&[k, 8, m]),
    );
    let flat_bits = one(&mut g, Op::Reshape, &[turned], virt(&[8 * k, m]));
    // Bit `i` of word `w` is bit `i mod 8` of byte `4w + 3 − ⌊i/8⌋`.
    let byte_of = |row: usize| (4 * (row / 32) + 3 - (row % 32) / 8, row % 8);
    let slot = |byte: usize| committed_offsets.iter().position(|&o| o == byte);
    let index: Vec<i64> = (0..512)
        .map(|row| {
            let (byte, bit) = byte_of(row);
            slot(byte).map_or(0, |s| i64::try_from(8 * s + bit).expect("small"))
        })
        .collect();
    let gathered = gather(&mut g, flat_bits, index, &[m]);
    let placed = one(
        &mut g,
        product(tessera_ir::Broadcast::Trailing),
        &[gathered, place],
        virt(&[512, m]),
    );
    let mut xor = decl(&[512, m], NodeKind::Virtual);
    xor.range = IntRange { lo: 0, hi: 1 };
    let bits = one(
        &mut g,
        Op::BitFn { f: BoolFn::Xor },
        &[message, placed],
        xor,
    );
    requant(&mut g, bits, &[512, m], 4);
    let graph = g.finish();
    let place_values: Vec<i64> = (0..512)
        .map(|row| i64::from(slot(byte_of(row).0).is_some()))
        .collect();
    let mut state = 0x0526_3300u64;
    let mut next = |below: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        i64::try_from((state >> 33) % below).expect("small")
    };
    // The public bits are 0 at the committed bytes' rows.
    let message_values: Vec<i64> = (0..512 * m as usize)
        .map(|i| (1 - place_values[i / m as usize]) * next(2))
        .collect();
    let byte_values: Vec<i64> = (0..k * m).map(|_| next(256)).collect();
    let run = run_with(
        &graph,
        vec![
            (message, message_values),
            (bytes, byte_values),
            (place, place_values),
        ],
        0x0526_3300,
    );
    assert_eq!(run.schedule.stage_count(), 4, "Xor, Mul, the gather");
    assert!(relations(&run, 1).contains(&Relation::BitFn(BoolFn::Xor)));
    assert_eq!(
        relations(&run, 2),
        [Relation::Mul(tessera_ir::Broadcast::Trailing)]
    );
    assert_eq!(relations(&run, 3), [Relation::Selection]);
    assert_eq!(run.schedule.stages[3].instances[0].vars, 5, "8K = 24 rows");
    run.accepted(&graph);
    // Bit 3 of committed byte 1 of message 2, as the gather reads it.
    let bad = changed(&run, &graph, flat_bits, flat(&[8 * k, m], &[11, 2]));
    assert_eq!(
        run.verify_with(&graph, &bad, &run.public, Shift::Extension),
        Err(run.end(3)),
        "a changed gathered bit"
    );
}

// ----- Combined selections ----------------------------------------------------------------------

/// A virtual tensor of `shape` whose declared range holds a value's levels, below `2^40`.
fn wide(shape: &[u32]) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: IntType {
            bits: 64,
            signed: true,
        },
        range: IntRange {
            lo: -(1 << 40),
            hi: 1 << 40,
        },
        kind: NodeKind::Virtual,
    }
}

/// `Add(even, ConstMul(odd))` of `rows`, of `count` rows of `m`, the odd rows scaled by `weight`:
/// one level of a value (RFC-0066 'The integer value of rows').
fn level(g: &mut GraphBuilder, rows: TensorId, count: u32, m: u32, weight: i64) -> TensorId {
    let half = count / 2;
    let even = gather(g, rows, (0..i64::from(count)).step_by(2).collect(), &[m]);
    let odd = gather(g, rows, (1..i64::from(count)).step_by(2).collect(), &[m]);
    let c = g.constant(vec![weight]).expect("a weight");
    let scaled = one(g, Op::ConstMul { c }, &[odd], wide(&[half, m]));
    one(g, Op::Add, &[even, scaled], wide(&[half, m]))
}

/// RFC-0072 'Combined selections': a level `Add(even, ConstMul(odd))` is one instance, whose
/// atom combines the two gathers, `even` of coefficient 1 and `odd` of the constant, its own
/// coefficient 1 and its claim ready. It proves and verifies; a verifier that moves the odd
/// rows' kernel one row rejects it at stage 1's final check, and so does a changed row.
#[test]
fn a_level_proves_as_one_combined_selection() {
    let mut g = GraphBuilder::new();
    let rows = committed(&mut g, &[32, 3], 0, 3);
    let sum = level(&mut g, rows, 32, 3, 2);
    requant(&mut g, sum, &[16, 3], 1 << 20);
    let graph = g.finish();
    let run = Run::of(&graph, &[rows], 0x0526_3400);
    assert_eq!(relations(&run, 1), [Relation::Selection], "one instance");
    let leaf = &run.schedule.stages[0].claims[0];
    assert_eq!(
        (leaf.atoms.len(), leaf.split),
        (1, false),
        "one atom, ready"
    );
    let atom = &leaf.atoms[0];
    assert!(atom.is_unit(), "a combination's coefficient is 1");
    assert_eq!(atom.combined.len(), 2, "even and odd");
    assert_eq!(atom.combined[0].kind, atom.kind, "the first member's kind");
    assert_eq!(atom.combined[0].coefficient, [Vec::new()]);
    assert!(matches!(
        atom.combined[1].coefficient.as_slice(),
        [product] if matches!(product.as_slice(), [Factor::Constant(_)])
    ));
    assert_eq!(run.proof.stages[0].sumcheck.final_evals.len(), 1);
    run.accepted(&graph);
    assert_eq!(
        verdict(&run, &graph, Kernel::LastMemberMoved),
        Err(run.end(1)),
        "one member's kernel changed"
    );
    for (row, name) in [(6, "an even row"), (13, "an odd row")] {
        let bad = changed(&run, &graph, rows, flat(&[32, 3], &[row, 1]));
        assert_eq!(
            run.verify_with(&graph, &bad, &run.public, Shift::Extension),
            Err(run.end(1)),
            "{name}"
        );
    }
}

/// RFC-0037 'Splits', RFC-0072 'Its value': a combination beside a committed atom is split, one
/// value for the whole combination, weighed by 1, in its first member's place after the
/// committed group's; the next stage proves that value as its instance's claim. A combination's
/// value one off fails the leaf stage's split check.
#[test]
fn a_split_combination_sends_its_whole_value() {
    let mut g = GraphBuilder::new();
    let rows = committed(&mut g, &[32, 3], 0, 3);
    let w = committed(&mut g, &[16, 3], -3, 3);
    let sum = level(&mut g, rows, 32, 3, 4);
    let key = one(&mut g, Op::Add, &[w, sum], wide(&[16, 3]));
    requant(&mut g, key, &[16, 3], 1 << 20);
    let graph = g.finish();
    let run = Run::of(&graph, &[rows, w], 0x0526_3450);
    let leaf = &run.schedule.stages[0].claims[0];
    assert!(leaf.split);
    assert_eq!(leaf.atoms.len(), 2, "w, then the combination");
    assert_eq!(leaf.atoms[1].combined.len(), 2);
    assert!(matches!(
        leaf.parts.as_slice(),
        [Part::Pool { .. }, Part::Instance { atom: 1, .. }]
    ));
    assert_eq!(run.proof.leaf_splits.len(), 2);
    assert_eq!(relations(&run, 1), [Relation::Selection]);
    run.accepted(&graph);
    let mut bad = run.proof.clone();
    bad.leaf_splits[1] += F::ONE;
    assert_eq!(
        run.verify_with(&graph, &bad, &run.public, Shift::Extension),
        Err(Reject { stage: 0, round: 0 }),
        "the combination's value one off"
    );
    assert_eq!(
        verdict(&run, &graph, Kernel::LastMemberMoved),
        Err(run.end(1)),
        "one member's kernel changed"
    );
}

/// The five levels of a value (RFC-0066 'The integer value of rows') with its constant: one
/// combined selection per stage, five stages below the leaf, each claim on the level below
/// ready. A verifier that moves a member's rows rejects at stage 1, the last level's.
#[test]
fn a_value_proves_through_one_combined_selection_per_level() {
    let m = 3u32;
    let mut g = GraphBuilder::new();
    let rows = committed(&mut g, &[32, m], 0, 3);
    let (mut current, mut count, mut weight) = (rows, 32u32, 2i64);
    while count > 1 {
        current = level(&mut g, current, count, m, weight);
        count /= 2;
        weight = weight.saturating_mul(weight);
    }
    let value = one(&mut g, Op::Reshape, &[current], wide(&[m]));
    let k = g.constant(vec![0x428a_2f98]).expect("a constant");
    let shifted = one(&mut g, Op::ConstAdd { c: k }, &[value], wide(&[m]));
    requant(&mut g, shifted, &[m], 1 << 40);
    let graph = g.finish();
    let run = Run::of(&graph, &[rows], 0x0526_3500);
    assert_eq!(run.schedule.stage_count(), 6);
    for k in 1..=5 {
        assert_eq!(relations(&run, k), [Relation::Selection], "stage {k}");
        let instance = &run.schedule.stages[k].instances[0];
        let atom = run
            .schedule
            .atom_of(u16::try_from(k).expect("small"), instance.source)
            .expect("a part");
        assert_eq!(atom.combined.len(), 2, "stage {k}: even and odd");
        assert_eq!(instance.vars, u32::try_from(k).expect("small"), "stage {k}");
    }
    run.accepted(&graph);
    assert_eq!(
        verdict(&run, &graph, Kernel::LastMemberMoved),
        Err(run.end(1)),
        "one member's kernel changed"
    );
}

/// RFC-0072 'No merging across claims': `g·g`, `g` a gather, leaves two claims on `g` at one
/// point, which stay two claims and two instances of stage 2, each proving its own; the proof
/// verifies.
#[test]
fn two_claims_on_one_tensor_at_one_point_stay_two_instances() {
    let mut g = GraphBuilder::new();
    let x = committed(&mut g, &[8, 3], -3, 3);
    let picked = gather(&mut g, x, vec![5, 1, 2, 7, 0], &[3]);
    let square = one(
        &mut g,
        product(tessera_ir::Broadcast::Trailing),
        &[picked, picked],
        decl(&[5, 3], NodeKind::Virtual),
    );
    requant(&mut g, square, &[5, 3], 100);
    let graph = g.finish();
    let run = Run::of(&graph, &[x], 0x0526_3600);
    let stage = &run.schedule.stages[1];
    let (a, b) = (&stage.claims[0], &stage.claims[1]);
    assert_eq!(
        (a.kind, &a.point),
        (b.kind, &b.point),
        "one tensor, one point"
    );
    assert_eq!(
        relations(&run, 2),
        [Relation::Selection, Relation::Selection],
        "two instances"
    );
    let instances = &run.schedule.stages[2].instances;
    assert_eq!(instances[0].point, instances[1].point);
    assert_ne!(
        instances[0].source, instances[1].source,
        "each its own claim"
    );
    run.accepted(&graph);
}

// ----- BitFn ------------------------------------------------------------------------------------

/// RFC-0072 'Testing Strategy' 'Relations', RFC-0066 '`BitFn`': each function on random committed
/// bits of `[5, 3]` proves and verifies as RFC-0010's `Elementwise`, its stage of degree 3, or 4
/// for `Xor3` and `Maj`, one evaluation per input; a changed input bit that changes the function
/// is rejected at stage 1's final check. A public input sends nothing.
#[test]
fn each_bit_function_proves_and_verifies() {
    let shape = [5u32, 3];
    for f in [
        BoolFn::Xor,
        BoolFn::Xor3,
        BoolFn::And,
        BoolFn::Ch,
        BoolFn::Maj,
    ] {
        for public_first in [false, true] {
            let name = format!("{f:?}, public first input {public_first}");
            let mut g = GraphBuilder::new();
            let inputs: Vec<TensorId> = (0..f.arity())
                .map(|i| {
                    let kind = if public_first && i == 0 {
                        PUBLIC
                    } else {
                        ADVICE
                    };
                    let mut input = decl(&shape, kind);
                    input.range = IntRange { lo: 0, hi: 1 };
                    g.input(input).expect("bits")
                })
                .collect();
            let mut out = decl(&shape, NodeKind::Virtual);
            out.range = IntRange { lo: 0, hi: 1 };
            let z = one(&mut g, Op::BitFn { f }, &inputs, out);
            requant(&mut g, z, &shape, 4);
            let graph = g.finish();
            let run = Run::of(&graph, &inputs, 0x0526_3700);
            assert_eq!(relations(&run, 1), [Relation::BitFn(f)], "{name}");
            let degree = if matches!(f, BoolFn::Xor3 | BoolFn::Maj) {
                4
            } else {
                3
            };
            assert!(
                run.proof.stages[0]
                    .sumcheck
                    .rounds
                    .iter()
                    .all(|round| round.len() == degree),
                "{name}: degree {degree}"
            );
            let sent = f.arity() - usize::from(public_first);
            assert_eq!(
                run.proof.stages[0].sumcheck.final_evals.len(),
                sent,
                "{name}"
            );
            run.accepted(&graph);
            // The last input changed at an entry where the function changes with it.
            let values: Vec<Vec<i64>> = inputs
                .iter()
                .map(|&t| run.witness.tensor(t).expect("bits").to_vec())
                .collect();
            let last = f.arity() - 1;
            let entry = (0..values[0].len())
                .find(|&e| {
                    let bits: Vec<i64> = values.iter().map(|v| v[e]).collect();
                    let mut moved = bits.clone();
                    moved[last] += 1;
                    f.eval(&bits) != f.eval(&moved)
                })
                .expect("an entry where the last input matters");
            let bad = changed(&run, &graph, inputs[last], entry);
            assert_eq!(
                run.verify_with(&graph, &bad, &run.public, Shift::Extension),
                Err(run.end(1)),
                "{name}: a changed input bit"
            );
        }
    }
}

/// The verifier's polynomial of each function is its Boolean function on bits, and RFC-0066's
/// table off them: at random field points, `Xor3` is `Xor` twice, `Ch` is `ef + ¬e·g` and `Maj`
/// is `ab + c·(a ⊕ b)`, identities of multilinear polynomials.
#[test]
fn the_bit_functions_are_rfc_0066s_polynomials() {
    for bits in 0..8u64 {
        let x: Vec<i64> = (0..3)
            .map(|i| i64::try_from((bits >> i) & 1).expect("a bit"))
            .collect();
        let fx: Vec<F> = x
            .iter()
            .map(|&b| F::from(Goldilocks::from_i64(b)))
            .collect();
        for f in [
            BoolFn::Xor,
            BoolFn::Xor3,
            BoolFn::And,
            BoolFn::Ch,
            BoolFn::Maj,
        ] {
            assert_eq!(
                bit_function(f, &fx),
                F::from(Goldilocks::from_i64(f.eval(&x))),
                "{f:?} at {x:?}"
            );
        }
    }
    let xor = |a: F, b: F| bit_function(BoolFn::Xor, &[a, b]);
    let and = |a: F, b: F| bit_function(BoolFn::And, &[a, b]);
    for r in random_points(3, 4, 0x0526_3800) {
        let (a, b, c) = (r[0], r[1], r[2]);
        assert_eq!(bit_function(BoolFn::Xor3, &r), xor(xor(a, b), c));
        assert_eq!(bit_function(BoolFn::Ch, &r), and(a, b) + and(F::ONE - a, c));
        assert_eq!(bit_function(BoolFn::Maj, &r), and(a, b) + and(c, xor(a, b)));
    }
}
