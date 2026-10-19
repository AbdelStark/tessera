// SPDX-License-Identifier: Apache-2.0 OR MIT
use super::*;

pub(super) fn tail(rho: &[Fp2], p: usize) -> Fp2 {
    let n = rho.len();
    if p == 0 {
        return Fp2::ONE;
    }
    if p >= 1 << n {
        return Fp2::ZERO;
    }
    let mut lt = Fp2::ZERO;
    let mut suffix = Fp2::ONE;
    for i in (0..n).rev() {
        let one_minus = Fp2::ONE - rho[i];
        if (p >> i) & 1 == 1 {
            lt += one_minus * suffix;
            suffix *= rho[i];
        } else {
            suffix *= one_minus;
        }
    }
    Fp2::ONE - lt
}

/// The halves of one level of a forest, packed over its real trees: tree `y`'s even entries are
/// `even[e_off[y]..e_off[y + 1]]` and its odd entries `odd[o_off[y]..o_off[y + 1]]`; the entries
/// beyond are the constant `KAPPA`. Declared-dense forests fill every slot, so the layout is the
/// production's.
struct Level {
    even: SecretBuf<Fp>,
    odd: SecretBuf<Fp>,
    e_off: Vec<usize>,
    o_off: Vec<usize>,
}

impl Level {
    fn e_len(&self, y: usize) -> usize {
        self.e_off[y + 1] - self.e_off[y]
    }

    fn o_len(&self, y: usize) -> usize {
        self.o_off[y + 1] - self.o_off[y]
    }
}

/// Levels `1 … h` of a forest from its padded slice, at entries `1..=h` (entry 0 stays empty):
/// tree `y` has `declared[y]` real leaves, and the real count of a level is `⌈R/2⌉` of the level
/// below. The leaf level is de-interleaved in one pass; above it,
/// `V_t[z] = even_{t+1}[z]·odd_{t+1}[z]` for `z < R_t`, the one node with a padded child
/// multiplying by `KAPPA`, split into even and odd entries by the parity of `z`
/// (`forest.rs` `build_levels`, `leaf_level`, `level_above`).
fn build_levels(padded: &[Fp], height: usize, declared: &[usize]) -> Vec<Level> {
    let trees = declared.len();
    let mut levels: Vec<Level> = Vec::with_capacity(height + 1);
    // Entry 0: the roots, never built.
    levels.push(Level {
        even: SecretBuf::new(),
        odd: SecretBuf::new(),
        e_off: vec![0; trees + 1],
        o_off: vec![0; trees + 1],
    });
    for _ in 1..=height {
        levels.push(Level {
            even: SecretBuf::new(),
            odd: SecretBuf::new(),
            e_off: Vec::with_capacity(trees + 1),
            o_off: Vec::with_capacity(trees + 1),
        });
    }
    // The leaf level.
    {
        let even_total: usize = declared.iter().map(|r| r.div_ceil(2)).sum();
        let odd_total: usize = declared.iter().map(|r| r / 2).sum();
        let leaf = &mut levels[height];
        leaf.even.reserve(even_total);
        leaf.odd.reserve(odd_total);
        leaf.e_off.push(0);
        leaf.o_off.push(0);
        for (y, &r) in declared.iter().enumerate() {
            let slice = &padded[y << height..(y << height) + r];
            for pair in slice.chunks_exact(2) {
                leaf.even.push(pair[0]);
                leaf.odd.push(pair[1]);
            }
            if r % 2 == 1 {
                leaf.even.push(slice[r - 1]);
            }
            leaf.e_off.push(leaf.even.len());
            leaf.o_off.push(leaf.odd.len());
        }
    }
    // The levels above, from the one below each.
    for t in (1..height).rev() {
        let (upper, lower) = levels.split_at_mut(t + 1);
        let below = &lower[0];
        let level = &mut upper[t];
        let lengths = (0..trees)
            .map(|y| (below.e_len(y) + below.o_len(y)).div_ceil(2))
            .collect::<Vec<_>>();
        level
            .even
            .reserve(lengths.iter().map(|r| r.div_ceil(2)).sum());
        level.odd.reserve(lengths.iter().map(|r| r / 2).sum());
        level.e_off.push(0);
        level.o_off.push(0);
        for y in 0..trees {
            let e = &below.even[below.e_off[y]..below.e_off[y + 1]];
            let o = &below.odd[below.o_off[y]..below.o_off[y + 1]];
            let full = e.len().min(o.len());
            for (ep, op) in e[..full].chunks_exact(2).zip(o[..full].chunks_exact(2)) {
                level.even.push(ep[0] * op[0]);
                level.odd.push(ep[1] * op[1]);
            }
            let mut z = full - full % 2;
            if z < full {
                level.even.push(e[z] * o[z]);
                z += 1;
            }
            if e.len() > o.len() {
                let v = e[full] * KAPPA;
                if z % 2 == 0 {
                    level.even.push(v);
                } else {
                    level.odd.push(v);
                }
            }
            level.e_off.push(level.even.len());
            level.o_off.push(level.odd.len());
        }
    }
    levels
}

/// The tables an instance reads in its first own round, over the prime subfield.
enum Source<'a> {
    /// A level's packed halves.
    Level(&'a Level),
    /// The tree-level tables of a forest at its root level (`m = 0`), one entry per real tree.
    Owned {
        even: SecretBuf<Fp>,
        odd: SecretBuf<Fp>,
    },
}

/// Behavior-changing faults are built only in isolated mutation copies.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    Tail,
    Odd,
    Order,
    Packed,
    Fallback,
}
const FAULT: Fault = Fault::None;

#[derive(Clone, Copy)]
struct Group {
    tree: usize,
    e_start: usize,
    e_len: usize,
    o_start: usize,
    o_len: usize,
    tree_weight: Fp2,
}

fn vertex(point: &[Fp2], y: usize) -> Fp2 {
    point.iter().enumerate().fold(Fp2::ONE, |p, (i, &r)| {
        p * if (y >> i) & 1 == 1 { r } else { Fp2::ONE - r }
    })
}

/// Public equal-length runs share tail coefficients and fold bounds. No secret controls traversal.
fn runs(groups: &[Group]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < groups.len() {
        let mut end = start + 1;
        while end < groups.len()
            && (groups[end].e_len, groups[end].o_len) == (groups[start].e_len, groups[start].o_len)
        {
            end += 1;
        }
        runs.push((start, end));
        start = end;
    }
    runs
}

struct Inst<'a> {
    m: usize,
    s: usize,
    real_trees: usize,
    point: Vec<Fp2>,
    base: Source<'a>,
    groups: Vec<Group>,
    be: SecretBuf<Fp2>,
    bo: SecretBuf<Fp2>,
    bound: bool,
    phase2: bool,
    own: usize,
    pad: usize,
    weight: Fp2,
    claim: Fp2,
    scale: Fp2,
    sums: [Fp2; 3],
    fallbacks: usize,
}

impl Drop for Inst<'_> {
    fn drop(&mut self) {
        self.claim.zeroize();
        self.sums.zeroize();
    }
}

impl<'a> Inst<'a> {
    fn new(
        level: &'a Level,
        m: usize,
        s: usize,
        real_trees: usize,
        point: Vec<Fp2>,
        entering: Option<Fp2>,
    ) -> Self {
        assert_eq!(point.len(), m + s);
        let mut groups = Vec::with_capacity(real_trees.max(1));
        // Cached per-tree scalars are public boundary metadata, not a bound per-leaf eq table.
        let base = if m > 0 {
            for y in 0..real_trees {
                groups.push(Group {
                    tree: y,
                    e_start: level.e_off[y],
                    e_len: level.e_len(y),
                    o_start: level.o_off[y],
                    o_len: level.o_len(y),
                    tree_weight: vertex(&point[m..], y),
                });
            }
            groups.sort_by_key(|g| (g.e_len, g.o_len));
            Source::Level(level)
        } else {
            let even = (0..real_trees)
                .map(|y| {
                    if level.e_len(y) > 0 {
                        level.even[level.e_off[y]]
                    } else {
                        KAPPA
                    }
                })
                .collect::<SecretBuf<_>>();
            let odd = (0..real_trees)
                .map(|y| {
                    if level.o_len(y) > 0 {
                        level.odd[level.o_off[y]]
                    } else {
                        KAPPA
                    }
                })
                .collect::<SecretBuf<_>>();
            groups.push(Group {
                tree: 0,
                e_start: 0,
                e_len: real_trees,
                o_start: 0,
                o_len: real_trees,
                tree_weight: Fp2::ONE,
            });
            Source::Owned { even, odd }
        };
        let mut this = Self {
            m,
            s,
            real_trees,
            point,
            base,
            groups,
            be: SecretBuf::new(),
            bo: SecretBuf::new(),
            bound: false,
            phase2: m == 0,
            own: 0,
            pad: 0,
            weight: Fp2::ONE,
            claim: Fp2::ZERO,
            scale: Fp2::ONE,
            sums: [Fp2::ZERO; 3],
            fallbacks: 0,
        };
        this.claim = entering.unwrap_or_else(|| {
            assert_eq!(m, 0);
            let mut sum = Fp2::ZERO;
            for y in 0..real_trees {
                sum += vertex(&this.point, y).mul_by_base(this.base_even()[y] * this.base_odd()[y]);
            }
            sum + tail(&this.point, real_trees)
        });
        this
    }
    fn vars(&self) -> usize {
        self.m + self.s
    }
    fn base_even(&self) -> &[Fp] {
        match &self.base {
            Source::Level(l) => &l.even,
            Source::Owned { even, .. } => even,
        }
    }
    fn base_odd(&self) -> &[Fp] {
        match &self.base {
            Source::Level(l) => &l.odd,
            Source::Owned { odd, .. } => odd,
        }
    }
    fn base_at(&self, g: &Group, x: usize, odd: bool) -> Fp {
        let (start, len, buf) = if odd {
            (g.o_start, g.o_len, self.base_odd())
        } else {
            (g.e_start, g.e_len, self.base_even())
        };
        if x < len { buf[start + x] } else { KAPPA }
    }
    fn ext_at(&self, g: &Group, x: usize, odd: bool) -> Fp2 {
        let (start, len, buf) = if odd {
            (g.o_start, g.o_len, &self.bo)
        } else {
            (g.e_start, g.e_len, &self.be)
        };
        if x < len { buf[start + x] } else { Fp2::ONE }
    }

    /// Two split tables on the ORIGINAL padded remaining coordinates, never packed offsets.
    fn node_sums<const ONE: bool>(&self, lo: &[Fp2], hi: &[Fp2]) -> [Fp2; 3] {
        let mut sums = [Fp2::ZERO; 3];
        let local_bits = if self.phase2 {
            self.vars() - self.own - 1
        } else {
            self.m - self.own - 1
        };
        let local_pairs = 1usize << local_bits;
        let remainder = &self.point[self.own + 1..];
        for (start, end) in runs(&self.groups) {
            let first = self.groups[start];
            let full = (first.e_len / 2).min(first.o_len / 2);
            let padded = first.e_len.div_ceil(2).max(first.o_len.div_ceil(2));
            for g in &self.groups[start..end] {
                let base = if self.phase2 || FAULT == Fault::Packed {
                    0
                } else {
                    g.tree * local_pairs
                };
                // Hoist the high-half weight outside each contiguous low-half block, as production.
                let mut x = 0;
                while x < full {
                    let global = base + x;
                    let low = global & (lo.len() - 1);
                    let stop = (full - x).min(lo.len() - low);
                    let mut inner = [Fp2::ZERO; 3];
                    if self.bound {
                        let ev = &self.be[g.e_start + 2 * x..g.e_start + 2 * (x + stop)];
                        let od = &self.bo[g.o_start + 2 * x..g.o_start + 2 * (x + stop)];
                        for ((e, o), &w) in ev
                            .chunks_exact(2)
                            .zip(od.chunks_exact(2))
                            .zip(&lo[low..low + stop])
                        {
                            let e2 = e[1] + (e[1] - e[0]);
                            let o2 = o[1] + (o[1] - o[0]);
                            inner[0] += w * e[0] * o[0];
                            if ONE {
                                inner[1] += w * e[1] * o[1];
                            }
                            inner[2] += w * e2 * o2;
                        }
                    } else {
                        let ev = &self.base_even()[g.e_start + 2 * x..g.e_start + 2 * (x + stop)];
                        let od = &self.base_odd()[g.o_start + 2 * x..g.o_start + 2 * (x + stop)];
                        for ((e, o), &w) in ev
                            .chunks_exact(2)
                            .zip(od.chunks_exact(2))
                            .zip(&lo[low..low + stop])
                        {
                            let e2 = e[1] + (e[1] - e[0]);
                            let o2 = o[1] + (o[1] - o[0]);
                            inner[0] += w.mul_by_base(e[0] * o[0]);
                            if ONE {
                                inner[1] += w.mul_by_base(e[1] * o[1]);
                            }
                            inner[2] += w.mul_by_base(e2 * o2);
                        }
                    }
                    let w = hi[global / lo.len()];
                    for z in [0, 1, 2] {
                        sums[z] += w * inner[z];
                    }
                    x += stop;
                }
                // At most two mixed pairs; neither may be treated as a constant subtree.
                for x in full..padded {
                    let global = base + x;
                    let w = lo[global & (lo.len() - 1)] * hi[global / lo.len()];
                    if self.bound {
                        let (a, b, c, d) = (
                            self.ext_at(g, 2 * x, false),
                            self.ext_at(g, 2 * x + 1, false),
                            self.ext_at(g, 2 * x, true),
                            self.ext_at(g, 2 * x + 1, true),
                        );
                        sums[0] += w * a * c;
                        if ONE {
                            sums[1] += w * b * d;
                        }
                        sums[2] += w * (b + b - a) * (d + d - c);
                    } else {
                        let (a, b, c, d) = (
                            self.base_at(g, 2 * x, false),
                            self.base_at(g, 2 * x + 1, false),
                            self.base_at(g, 2 * x, true),
                            self.base_at(g, 2 * x + 1, true),
                        );
                        sums[0] += w.mul_by_base(a * c);
                        if ONE {
                            sums[1] += w.mul_by_base(b * d);
                        }
                        sums[2] += w.mul_by_base((b + b - a) * (d + d - c));
                    }
                }
            }
            if padded < local_pairs && FAULT != Fault::Tail {
                let tree_scale = if self.phase2 {
                    Fp2::ONE
                } else {
                    self.groups[start..end]
                        .iter()
                        .fold(Fp2::ZERO, |sum, g| sum + g.tree_weight)
                };
                let constant = tree_scale * tail(&remainder[..local_bits], padded);
                sums[0] += constant;
                if ONE {
                    sums[1] += constant;
                }
                sums[2] += constant;
            }
        }
        if !self.phase2 && self.real_trees < 1usize << self.s && FAULT != Fault::Tail {
            let constant = tail(&self.point[self.m..], self.real_trees);
            sums[0] += constant;
            if ONE {
                sums[1] += constant;
            }
            sums[2] += constant;
        }
        sums
    }
    fn prepare(&mut self, max: usize, weight: Fp2) {
        self.pad = max - self.vars();
        self.weight = weight;
    }
    fn add_round(&mut self, j: usize, message: &mut [Fp2; 3]) {
        if j < self.pad {
            let value = self.weight
                * self
                    .claim
                    .mul_by_base(Fp::from_u64_reduced(1u64 << (self.pad - 1 - j)));
            for v in message {
                *v += value;
            }
            return;
        }
        let k = j - self.pad;
        let r = self.point[k];
        let rest = &self.point[k + 1..];
        let (l, h) = rest.split_at(rest.len().div_ceil(2));
        let lo = eq_table(l).expect("checked split eq shape");
        let hi = eq_table(h).expect("checked split eq shape");
        let az = self.scale * (Fp2::ONE - r);
        let ao = self.scale * r;
        let inverse = ao.inverse();
        self.fallbacks += usize::from(inverse.is_none());
        let [v0, v1, v2] = if inverse.is_some() || FAULT == Fault::Fallback {
            self.node_sums::<false>(&lo, &hi)
        } else {
            self.node_sums::<true>(&lo, &hi)
        };
        let v1 = match inverse {
            Some(inv) => (self.claim - az * v0) * inv,
            None => v1,
        };
        let v3 = v0 + (v2 - v1).double() + (v2 - v1);
        message[0] += self.weight * az * v0;
        message[1] += self.weight * (ao.double() - az) * v2;
        message[2] += self.weight * (ao.double() + ao - az.double()) * v3;
        self.sums = [v0, v1, v2];
    }
    fn bind(&mut self, j: usize, r: Fp2) {
        if j < self.pad {
            return;
        }
        let k = j - self.pad;
        let az = self.scale * (Fp2::ONE - self.point[k]);
        let ao = self.scale * self.point[k];
        let [v0, v1, v2] = self.sums;
        let half = Fp2::from(Fp::from_u64_reduced(2))
            .inverse()
            .expect("two invertible");
        self.scale = az + r * (ao - az);
        self.claim =
            self.scale * (v0 + r * (v1 - v0) + r * (r - Fp2::ONE) * half * (v2 - v1.double() + v0));
        let groups = self.groups.clone();
        // Public run bounds and four independent odd-tail folds; no machine-SIMD claim.
        for odd in [false, true] {
            let mut out = if self.bound {
                SecretBuf::new()
            } else {
                SecretBuf::with_capacity(
                    groups
                        .iter()
                        .map(|g| {
                            if odd {
                                g.o_len.div_ceil(2)
                            } else {
                                g.e_len.div_ceil(2)
                            }
                        })
                        .sum(),
                )
            };
            for (start, end) in runs(&groups) {
                for (index, g) in groups.iter().enumerate().take(end).skip(start) {
                    let (offset, len) = if odd {
                        (g.o_start, g.o_len)
                    } else {
                        (g.e_start, g.e_len)
                    };
                    let next_offset = if self.bound { offset } else { out.len() };
                    if self.bound {
                        let buf = if odd { &mut self.bo } else { &mut self.be };
                        for x in 0..len / 2 {
                            let a = buf[offset + 2 * x];
                            let b = buf[offset + 2 * x + 1];
                            buf[offset + x] = a + r * (b - a);
                        }
                    } else {
                        let base = if odd {
                            self.base_odd()
                        } else {
                            self.base_even()
                        };
                        for pair in base[offset..offset + len - len % 2].chunks_exact(2) {
                            out.push(r.mul_by_base(pair[1] - pair[0]).add_base(pair[0]));
                        }
                        if len % 2 == 1 {
                            out.push(Fp2::ZERO);
                        }
                    }
                    if odd {
                        self.groups[index].o_start = next_offset;
                        self.groups[index].o_len = len.div_ceil(2);
                    } else {
                        self.groups[index].e_start = next_offset;
                        self.groups[index].e_len = len.div_ceil(2);
                    }
                }
                let len = if odd {
                    groups[start].o_len
                } else {
                    groups[start].e_len
                };
                if len % 2 == 1 {
                    for first in (start..end).step_by(4) {
                        let lanes = (end - first).min(4);
                        let mut values = [Fp2::ZERO; 4];
                        for lane in 0..lanes {
                            let g = &groups[first + lane];
                            values[lane] = if self.bound {
                                self.ext_at(g, len - 1, odd)
                            } else {
                                Fp2::from(self.base_at(g, len - 1, odd))
                            };
                        }
                        let padding = if FAULT == Fault::Odd {
                            Fp2::ZERO
                        } else {
                            Fp2::ONE
                        };
                        let folded = values.map(|a| a + r * (padding - a));
                        for (lane, value) in folded.iter().enumerate().take(lanes) {
                            let g = self.groups[first + lane];
                            let offset = if odd { g.o_start } else { g.e_start };
                            if self.bound {
                                let buf = if odd { &mut self.bo } else { &mut self.be };
                                buf[offset + len / 2] = *value;
                            } else {
                                out[offset + len / 2] = *value;
                            }
                        }
                    }
                }
            }
            if !self.bound {
                if odd {
                    self.bo = out;
                } else {
                    self.be = out;
                }
            }
        }
        self.bound = true;
        self.own += 1;
        if !self.phase2 && self.own == self.m {
            if FAULT != Fault::Order {
                self.groups.sort_unstable_by_key(|g| g.tree);
            }
            let be = self
                .groups
                .iter()
                .map(|g| self.ext_at(g, 0, false))
                .collect();
            let bo = self
                .groups
                .iter()
                .map(|g| self.ext_at(g, 0, true))
                .collect();
            self.be = be;
            self.bo = bo;
            self.groups = vec![Group {
                tree: 0,
                e_start: 0,
                e_len: self.real_trees,
                o_start: 0,
                o_len: self.real_trees,
                tree_weight: Fp2::ONE,
            }];
            self.phase2 = true;
        }
    }
    fn final_evals(&self) -> (Fp2, Fp2) {
        assert!(self.phase2);
        let g = &self.groups[0];
        if self.bound {
            (self.ext_at(g, 0, false), self.ext_at(g, 0, true))
        } else {
            (
                Fp2::from(self.base_at(g, 0, false)),
                Fp2::from(self.base_at(g, 0, true)),
            )
        }
    }
}

fn prove_level<T: SumcheckTranscript<Fp2>>(
    transcript: &mut T,
    tag: u32,
    instances: &mut [Inst<'_>],
) -> (SumcheckProof<Fp2>, Vec<Fp2>) {
    let prefix = Prefix::Tree(tag);
    let gamma = transcript.challenge(&format!("{prefix}/gamma"));
    let max = instances.iter().map(Inst::vars).max().unwrap_or(0);
    let mut weight = Fp2::ONE;
    for inst in instances.iter_mut() {
        inst.prepare(max, weight);
        weight *= gamma;
    }
    let mut rounds = Vec::with_capacity(max);
    let mut point = Vec::with_capacity(max);
    for j in 0..max {
        let mut message = [Fp2::ZERO; 3];
        for inst in instances.iter_mut() {
            inst.add_round(j, &mut message);
        }
        transcript.absorb_message(&format!("{prefix}/round/{j}"), &message);
        let r = transcript.challenge(&format!("{prefix}/r/{j}"));
        for inst in instances.iter_mut() {
            inst.bind(j, r);
        }
        rounds.push(message.to_vec());
        point.push(r);
    }
    let mut final_evals = Vec::with_capacity(2 * instances.len());
    for inst in instances.iter() {
        let (u, w) = inst.final_evals();
        final_evals.extend([u, w]);
    }
    transcript.absorb_message(&format!("{prefix}/evals"), &final_evals);
    (
        SumcheckProof {
            rounds,
            final_evals,
        },
        point,
    )
}

/// Research-only product forest driver; all allocation/build/drop work belongs to the proof.
pub(super) fn prove<T: SumcheckTranscript<Fp2>>(
    config: &Config,
    transcript: &mut T,
) -> (Vec<SumcheckProof<Fp2>>, Vec<Claim<Fp2>>, usize) {
    validate(config);
    struct State {
        trees_log: usize,
        join: usize,
        real_trees: usize,
        levels: Vec<Level>,
        point: Vec<Fp2>,
        claim: Option<Fp2>,
    }
    let tallest = config
        .forests
        .iter()
        .map(|f| f.shape.height)
        .max()
        .expect("a forest");
    let mut states: Vec<State> = config
        .forests
        .iter()
        .map(|f| State {
            trees_log: f.shape.trees_log,
            join: tallest - f.shape.height,
            real_trees: f.shape.real_trees,
            levels: build_levels(&f.padded, f.shape.height, &f.shape.real_leaves),
            point: Vec::new(),
            claim: None,
        })
        .collect();
    let mut proofs = Vec::with_capacity(tallest);
    let mut claims = Vec::new();
    let mut fallbacks = 0usize;
    for level in 0..tallest {
        let prefix = Prefix::Tree(level as u32);
        let smax = states
            .iter()
            .filter(|s| s.join == level)
            .map(|s| s.trees_log)
            .max()
            .unwrap_or(0);
        let root: Vec<Fp2> = (0..smax)
            .map(|c| transcript.challenge(&format!("{prefix}/root/{c}")))
            .collect();
        for state in states.iter_mut().filter(|s| s.join == level) {
            state.point = root[smax - state.trees_log..].to_vec();
        }
        let active: Vec<usize> = (0..states.len())
            .filter(|&i| states[i].join <= level)
            .collect();
        let mut instances: Vec<Inst<'_>> = active
            .iter()
            .map(|&i| {
                let s = &states[i];
                let t = level - s.join + 1;
                Inst::new(
                    &s.levels[t],
                    t - 1,
                    s.trees_log,
                    s.real_trees,
                    s.point.clone(),
                    s.claim,
                )
            })
            .collect();
        let (proof, point) = prove_level(transcript, level as u32, &mut instances);
        let vars: Vec<usize> = instances.iter().map(Inst::vars).collect();
        fallbacks += instances.iter().map(|i| i.fallbacks).sum::<usize>();
        drop(instances);
        let rho = transcript.challenge(&format!("{prefix}/line"));
        let mut next = Vec::with_capacity(active.len());
        for (k, &i) in active.iter().enumerate() {
            let own = &point[point.len() - vars[k]..];
            let mut cp = vec![rho];
            cp.extend_from_slice(own);
            let (u, w) = (proof.final_evals[2 * k], proof.final_evals[2 * k + 1]);
            let value = u + rho * (w - u);
            states[i].point.clone_from(&cp);
            states[i].claim = Some(value);
            let used = level - states[i].join + 1;
            states[i].levels[used] = Level {
                even: SecretBuf::new(),
                odd: SecretBuf::new(),
                e_off: Vec::new(),
                o_off: Vec::new(),
            };
            next.push(Claim { point: cp, value });
        }
        claims = next;
        proofs.push(proof);
    }
    (proofs, claims, fallbacks)
}

fn validate(config: &Config) {
    assert!(!config.forests.is_empty());
    let mut total = 0usize;
    for f in &config.forests {
        let s = &f.shape;
        assert!(s.height > 0);
        assert!(s.height.checked_add(s.trees_log).expect("bits") <= 24);
        assert!(s.real_trees <= 1usize << s.trees_log);
        assert_eq!(s.real_leaves.len(), s.real_trees);
        assert!(s.real_leaves.iter().all(|&r| r <= 1usize << s.height));
        assert_eq!(f.padded.len(), s.padded_len());
        total = total.checked_add(f.padded.len()).expect("batch count");
        assert!(
            f.padded
                .len()
                .checked_mul(16)
                .is_some_and(|b| b <= isize::MAX as usize)
        );
    }
    assert!(total < 1usize << 25);
}
