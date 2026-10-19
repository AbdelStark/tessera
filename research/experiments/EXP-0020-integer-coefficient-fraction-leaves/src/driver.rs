// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Public synthetic inputs and public independent verifier closure. No model/private data.
use crate::*;
use serde_json::{Value, json};
use std::{fs, path::Path};
use tessera_codec::{Ascii, Rational};
use tessera_hash::{Domain, Mode, ParamSetId, SessionContext, Transcript, hash};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind};
use tv_core::{logup as tl, tree as tv};
use tv_hash::transcript as tvt;
pub fn results() -> std::path::PathBuf {
    std::env::var_os("EXP20_RESULTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| RESULTS.into())
}
pub const RESULTS: &str =
    "research/experiments/EXP-0020-integer-coefficient-fraction-leaves/results";
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}
pub fn context() -> SessionContext {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x20; 32],
        session_id: [0x05; 16],
    }
}
fn tv_context() -> tvt::SessionContext<'static> {
    tvt::SessionContext {
        protocol: tvt::Ascii::new("tessera-protocol/1").unwrap(),
        mode: tvt::Mode::NonInteractive,
        params: tvt::ParamSetId(tvt::Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x20; 32],
        session_id: [0x05; 16],
    }
}
pub fn tvf(x: Fp) -> tv_field::Fp {
    tv_field::Fp::from_canonical(x.to_canonical_u64()).unwrap()
}
pub fn tvx(x: Fp2) -> tv_field::Fp2 {
    let [a, b] = x.coeffs();
    tv_field::Fp2::new(tvf(a), tvf(b))
}
pub fn xs(xs: &[Fp2]) -> Vec<tv_field::Fp2> {
    xs.iter().copied().map(tvx).collect()
}
fn hx(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}
fn nums(xs: &[Fp2]) -> Vec<[u64; 2]> {
    xs.iter().map(|&x| x.words()).collect()
}
fn proof_json(p: &SumcheckProof<Fp2>) -> Value {
    json!({"rounds":p.rounds.iter().map(|r|nums(r)).collect::<Vec<_>>(),"final_evals":nums(&p.final_evals)})
}
pub struct Case {
    pub profile: char,
    pub table_id: u32,
    pub name: String,
    pub tk: Vec<Fp>,
    pub tv: Option<Vec<Fp>>,
    pub k: Vec<Fp>,
    pub v: Option<Vec<Fp>>,
    pub m: Vec<Fp>,
    pub descriptor: Vec<u64>,
    pub domain: coeff::Domain,
}
impl Case {
    pub fn new(profile: char, lh: usize, tbits: usize, rng: &mut Rng) -> Self {
        let kind = match profile {
            'E' => TableKind::Exp {
                key_bits: tbits as u8,
                sigma: 16,
                scale: Rational::new(1, 16384).unwrap(),
            },
            'R' => TableKind::Range { bits: tbits as u8 },
            _ => TableKind::Requant {
                b: tbits as u8,
                r_by_class: vec![6],
                lo: -128,
                hi: 127,
            },
        };
        let table_id = match profile {
            'E' => 1,
            'R' => 3,
            _ => 4,
        };
        let table = GeneratedTable::new(TableId(table_id), kind).unwrap();
        let n = 1usize << lh;
        let t = 1usize << tbits;
        let mut tk = Vec::with_capacity(t);
        let mut tv = if profile == 'R' {
            None
        } else {
            Some(Vec::with_capacity(t))
        };
        let mut prev = 65536i128;
        for i in 0..t {
            let (k, v) = table.row(i as u64).unwrap();
            assert_eq!(k.to_canonical_u64(), i as u64);
            tk.push(k);
            if let Some(vs) = &mut tv {
                let z = v.to_centered_i128();
                if profile == 'E' {
                    assert!((0..=65536).contains(&z));
                    assert!(z <= prev);
                    prev = z;
                    if i == 0 {
                        assert_eq!(z, 65536);
                    }
                } else {
                    let a = i as i128 - (1i128 << (tbits - 1));
                    let expected = ((a + 32) >> 6).clamp(-128, 127);
                    assert_eq!(z, expected);
                }
                vs.push(v);
            }
        }
        let mut ids: Vec<usize> = (0..n).map(|_| rng.next() as usize % t).collect();
        let corners = [
            0,
            1,
            t - 1,
            t.saturating_sub(2),
            t / 2 - 1,
            t / 2,
            t / 2 + 1,
            0,
        ];
        for (i, &x) in corners.iter().enumerate().take(n) {
            ids[i] = x.min(t - 1);
        }
        let k = ids.iter().map(|&i| tk[i]).collect();
        let v = tv.as_ref().map(|vs| ids.iter().map(|&i| vs[i]).collect());
        let mut counts = vec![0u64; t];
        for &i in &ids {
            counts[i] += 1;
        }
        let m = counts.into_iter().map(Fp::from_u64_reduced).collect();
        let mut case = Self {
            profile,
            table_id,
            name: format!("{profile}-l{lh}-t{tbits}"),
            tk,
            tv,
            k,
            v,
            m,
            descriptor: vec![],
            domain: coeff::Domain {
                k: (t - 1) as u128,
                v: match profile {
                    'E' => 65536,
                    'R' => 0,
                    _ => 128,
                },
                m: n as u128,
            },
        };
        case.refresh_descriptor();
        case
    }
    pub fn refresh_descriptor(&mut self) {
        let mut data = Vec::new();
        for col in [&self.tk[..], &self.k[..], &self.m[..]] {
            for x in col {
                data.extend(x.to_canonical_u64().to_le_bytes());
            }
        }
        for col in [&self.tv, &self.v].into_iter().flatten() {
            for x in col {
                data.extend(x.to_canonical_u64().to_le_bytes());
            }
        }
        let digest = hash(Domain::ColumnMerge, &data);
        self.descriptor = vec![
            20,
            self.profile as u64,
            self.table_id as u64,
            self.k.len() as u64,
            self.tk.len() as u64,
            u64::from(self.v.is_some()),
        ];
        for a in digest.chunks_exact(8) {
            self.descriptor
                .push(u64::from_le_bytes(a.try_into().unwrap()));
        }
    }
    pub fn transcript(&self) -> Transcript {
        let mut t = Transcript::new(&context());
        t.absorb("exp20/case", &self.descriptor);
        t
    }
    pub fn independent_transcript(&self) -> tvt::Transcript {
        let mut t = tvt::Transcript::new(&tv_context());
        let mut bytes = Vec::new();
        bytes.extend((self.descriptor.len() as u64).to_le_bytes());
        for x in &self.descriptor {
            bytes.extend(x.to_le_bytes());
        }
        t.absorb(tvt::Ascii::new("exp20/case").unwrap(), &bytes);
        t
    }
    pub fn local(&self) -> LogUpInstance<'_, Fp> {
        LogUpInstance::new(
            Tuples::new(&self.tk, self.tv.as_deref()),
            Tuples::new(&self.k, self.v.as_deref()),
            &self.m,
        )
    }
    pub fn production(&self) -> tessera_logup::LogUpInstance<'_, Fp> {
        tessera_logup::LogUpInstance::new(
            tessera_logup::Tuples::new(&self.tk, self.tv.as_deref()),
            tessera_logup::Tuples::new(&self.k, self.v.as_deref()),
            &self.m,
        )
    }
}
#[derive(Clone)]
pub struct Fraction {
    pub point: Vec<Fp2>,
    pub n: Fp2,
    pub d: Fp2,
}
trait FractionView {
    fn rec(&self) -> Fraction;
}
impl FractionView for LevelClaim<Fp2> {
    fn rec(&self) -> Fraction {
        match self {
            Self::Fraction(c) => Fraction {
                point: c.point.clone(),
                n: c.numerator,
                d: c.denominator,
            },
            _ => panic!("not a fraction"),
        }
    }
}
impl FractionView for tessera_sumcheck::LevelClaim<Fp2> {
    fn rec(&self) -> Fraction {
        match self {
            Self::Fraction(c) => Fraction {
                point: c.point.clone(),
                n: c.numerator,
                d: c.denominator,
            },
            _ => panic!("not a fraction"),
        }
    }
}
#[derive(Clone)]
pub struct Level {
    pub proof: SumcheckProof<Fp2>,
    pub trees: Vec<usize>,
    pub next: Vec<Fraction>,
    pub gamma: Fp2,
    pub lambda: Fp2,
    pub rho: Fp2,
    pub point: Vec<Fp2>,
}
#[derive(Clone)]
pub struct ReturnedClaims {
    pub key: Claim<Fp2>,
    pub value: Option<Claim<Fp2>>,
    pub multiplicity: Claim<Fp2>,
}
pub struct Record {
    pub alpha: Fp2,
    pub beta: Fp2,
    pub claims: Vec<ReturnedClaims>,
    pub forest_claims: Vec<Claim<Fp2>>,
    pub roots: Vec<Fp2>,
    pub levels: Vec<Level>,
    pub values: Vec<Fp2>,
    pub leaves: Vec<Fraction>,
    pub state: [u8; 32],
}
macro_rules! record {
    ($p:expr,$state:expr) => {{
        let p = $p;
        Record {
            alpha: *p.alpha(),
            beta: *p.beta(),
            claims: p
                .claims()
                .iter()
                .map(|c| ReturnedClaims {
                    key: c.key.clone(),
                    value: c.value.clone(),
                    multiplicity: c.multiplicity.clone(),
                })
                .collect(),
            forest_claims: p.forest_claims().to_vec(),
            roots: p.roots().to_vec(),
            levels: p
                .levels()
                .iter()
                .map(|l| Level {
                    proof: l.proof().clone(),
                    trees: l.trees().to_vec(),
                    next: l.next_claims().iter().map(FractionView::rec).collect(),
                    gamma: *l.gamma(),
                    lambda: *l.lambda().unwrap(),
                    rho: *l.rho(),
                    point: l.point().to_vec(),
                })
                .collect(),
            values: p.values().to_vec(),
            leaves: p.leaf_claims().iter().map(FractionView::rec).collect(),
            state: $state,
        }
    }};
}
pub fn prove(case: &Case, a: u8) -> Record {
    ARM.with(|x| x.set(a));
    coeff::DOMAIN.with(|d| d.set(Some(case.domain)));
    let mut t = case.transcript();
    if a == 0 {
        let p = tessera_logup::prove_logup(
            &mut t,
            &[case.production()],
            &[],
            &tessera_logup::TreeId::paired(1),
        )
        .unwrap();
        record!(p, t.state())
    } else {
        let p =
            logup::prove_logup(&mut t, &[case.local()], &[], &logup::TreeId::paired(1)).unwrap();
        record!(p, t.state())
    }
}
pub fn production_record(p: tessera_logup::ProvenLogUp<Fp2>, state: [u8; 32]) -> Record {
    record!(p, state)
}
pub fn level_json(l: &Level) -> Value {
    json!({"proof":proof_json(&l.proof),"trees":l.trees,"gamma":l.gamma.words(),"lambda":l.lambda.words(),"rho":l.rho.words(),"point":nums(&l.point),"next":l.next.iter().map(|c|json!({"point":nums(&c.point),"n":c.n.words(),"d":c.d.words()})).collect::<Vec<_>>()})
}
fn leaf(c: &Fraction) -> tv::FractionClaim {
    tv::FractionClaim {
        point: xs(&c.point),
        numerator: tvx(c.n),
        denominator: tvx(c.d),
    }
}
fn vp(p: &SumcheckProof<Fp2>) -> tv_core::sumcheck::SumcheckProof {
    tv_core::sumcheck::SumcheckProof {
        rounds: p.rounds.iter().map(|r| xs(r)).collect(),
        final_evals: xs(&p.final_evals),
    }
}
fn mle(column: &[Fp], point: &[tv_field::Fp2], padding: Fp) -> tv_field::Fp2 {
    let one = tv_field::Fp2::ONE;
    let mut weights = vec![one];
    for &r in point {
        let n = weights.len();
        for i in 0..n {
            let x = weights[i];
            weights[i] = x * (one - r);
            weights.push(x * r);
        }
    }
    weights
        .into_iter()
        .enumerate()
        .fold(tv_field::Fp2::ZERO, |s, (i, w)| {
            s + w * tv_field::Fp2::new(
                tvf(column.get(i).copied().unwrap_or(padding)),
                tv_field::Fp::ZERO,
            )
        })
}
fn tv_descriptor(t: &mut tvt::Transcript, label: &str, descriptor: &[u64]) {
    let mut bytes = Vec::new();
    bytes.extend((descriptor.len() as u64).to_le_bytes());
    for x in descriptor {
        bytes.extend(x.to_le_bytes());
    }
    t.absorb(tvt::Ascii::new(label).unwrap(), &bytes);
}
pub fn verify(case: &Case, r: &Record) {
    verify_many(&[case], r)
}
fn verify_many(cases: &[&Case], r: &Record) {
    let mut t = cases[0].independent_transcript();
    for case in &cases[1..] {
        tv_descriptor(&mut t, "exp20/extra", &case.descriptor);
    }
    let roots = tl::check_roots(&mut t, cases.len(), &xs(&r.roots)).unwrap();
    assert_eq!(tvx(r.alpha), roots.challenges.alpha);
    assert_eq!(tvx(r.beta), roots.challenges.beta);
    assert_eq!(r.claims.len(), cases.len());
    assert!(r.forest_claims.is_empty());
    let mut claims: Vec<_> = roots
        .instances
        .iter()
        .flat_map(|r| {
            [
                tv::LevelClaim::Fraction(r.lookup.clone()),
                tv::LevelClaim::Fraction(r.table.clone()),
            ]
        })
        .collect();
    let hs: Vec<_> = cases
        .iter()
        .flat_map(|c| {
            [
                c.k.len().next_power_of_two().trailing_zeros() as usize,
                c.tk.len().next_power_of_two().trailing_zeros() as usize,
            ]
        })
        .collect();
    let h = *hs.iter().max().unwrap();
    assert_eq!(r.levels.len(), h);
    assert_eq!(r.leaves.len(), 2 * cases.len());
    assert_eq!(
        r.values.len(),
        cases.iter().filter(|c| c.tv.is_some()).count()
    );
    for (i, l) in r.levels.iter().enumerate() {
        assert_eq!(l.next.len(), l.trees.len());
        let active: Vec<_> = (0..hs.len()).filter(|&j| i >= h - hs[j]).collect();
        assert_eq!(active, l.trees);
        let c: Vec<_> = active.iter().map(|&j| claims[j].clone()).collect();
        let passed = tv::verify_level(&mut t, i as u32, &c, &vp(&l.proof)).unwrap();
        assert_eq!(passed.gamma, tvx(l.gamma));
        assert_eq!(passed.lambda, Some(tvx(l.lambda)));
        assert_eq!(passed.rho, tvx(l.rho));
        assert_eq!(passed.point, xs(&l.point));
        assert_eq!(passed.claims.len(), active.len());
        for (j, claim) in active.into_iter().zip(passed.claims) {
            claims[j] = claim;
        }
        for (&id, n) in l.trees.iter().zip(&l.next) {
            assert_eq!(claims[id], tv::LevelClaim::Fraction(leaf(n)));
        }
    }
    let leaves: Vec<_> = cases
        .iter()
        .enumerate()
        .map(|(i, c)| tl::Leaves {
            lookup: leaf(&r.leaves[2 * i]),
            table: leaf(&r.leaves[2 * i + 1]),
            has_values: c.tv.is_some(),
        })
        .collect();
    for (i, x) in r.leaves.iter().enumerate() {
        assert_eq!(claims[i], tv::LevelClaim::Fraction(leaf(x)));
    }
    let beta = roots.challenges.beta;
    let closed = tl::check_leaves(
        &mut t,
        &roots.challenges,
        &leaves,
        &xs(&r.values),
        |i, p| {
            let case = cases[i];
            let k = mle(&case.tk, p, case.tk[0]);
            let v = case
                .tv
                .as_ref()
                .map_or(tv_field::Fp2::ZERO, |v| mle(v, p, v[0]));
            Some(k + beta * v)
        },
    )
    .unwrap();
    assert_eq!(closed.instances.len(), cases.len());
    for (i, c) in closed.instances.iter().enumerate() {
        let case = cases[i];
        let actual = &r.claims[i];
        assert_eq!(xs(&actual.key.point), c.key.point);
        assert_eq!(tvx(actual.key.value), c.key.value);
        assert_eq!(xs(&actual.multiplicity.point), c.multiplicity.point);
        assert_eq!(tvx(actual.multiplicity.value), c.multiplicity.value);
        assert_eq!(
            actual.value.as_ref().map(|v| (xs(&v.point), tvx(v.value))),
            c.value.as_ref().map(|v| (v.point.clone(), v.value))
        );
        assert_eq!(c.key.value, mle(&case.k, &c.key.point, case.tk[0]));
        if let Some(v) = &c.value {
            let column = case.v.as_ref().unwrap();
            assert_eq!(v.value, mle(column, &v.point, case.tv.as_ref().unwrap()[0]));
        }
        assert_eq!(
            c.multiplicity.value,
            mle(&case.m, &c.multiplicity.point, Fp::ZERO)
        );
    }
    assert_eq!(*t.state(), r.state);
}
impl Record {
    pub fn json(&self) -> Value {
        json!({"alpha":self.alpha.words(),"beta":self.beta.words(),"claims":self.claims.iter().map(|c|json!({"key":{"point":nums(&c.key.point),"value":c.key.value.words()},"value":c.value.as_ref().map(|v|json!({"point":nums(&v.point),"value":v.value.words()})),"multiplicity":{"point":nums(&c.multiplicity.point),"value":c.multiplicity.value.words()}})).collect::<Vec<_>>(),"forest_claims":self.forest_claims.iter().map(|c|json!({"point":nums(&c.point),"value":c.value.words()})).collect::<Vec<_>>(),"roots":nums(&self.roots),"levels":self.levels.iter().map(|l|json!({"proof":proof_json(&l.proof),"trees":l.trees,"gamma":l.gamma.words(),"lambda":l.lambda.words(),"rho":l.rho.words(),"point":nums(&l.point),"next":l.next.iter().map(|c|json!({"point":nums(&c.point),"n":c.n.words(),"d":c.d.words()})).collect::<Vec<_>>()})).collect::<Vec<_>>(),"values":nums(&self.values),"leaves":self.leaves.iter().map(|c|json!({"point":nums(&c.point),"n":c.n.words(),"d":c.d.words()})).collect::<Vec<_>>(),"state":hx(&self.state)})
    }
    pub fn digest(&self) -> String {
        hx(&hash(
            Domain::ColumnMerge,
            &serde_json::to_vec(&self.json()).unwrap(),
        ))
    }
}
pub fn check() -> Value {
    let mut rng = Rng(0x7E55_E7A0_0005_0021);
    let mut cases = Vec::new();
    let mut diagnostics = Vec::new();
    for profile in ['E', 'R', 'Q'] {
        for h in 0..=10 {
            let mut c = Case::new(profile, h, h.clamp(1, 5), &mut rng); // h0 is legal but corners generator must respect row count.
            c.domain.m = 1 << h;
            let p = prove(&c, 0);
            verify(&c, &p);
            let expected = p.digest();
            let mut ds = Vec::new();
            for a in [1, 2] {
                for v in if a == 2 { vec![1, 2, 3] } else { vec![2] } {
                    WIDTH.with(|x| x.set(v));
                    let r = prove(&c, a);
                    println!("checking {} a{a} v{v}", c.name);
                    verify(&c, &r);
                    assert_eq!(r.digest(), expected, "{} arm{a} v{v}", c.name);
                    ds.push(json!({"arm":a,"v":v,"digest":r.digest(),"independent":true}));
                }
            }
            if h >= 1 {
                WIDTH.with(|x| x.set(2));
                let d = crate::measure::Diagnostic::new(&c);
                d.check(&c);
                diagnostics.push(json!({"case":c.name,"events":d.prefix.len(),"boundary":"prefix-replay+bottom","exact":true}));
            }
            cases.push(json!({"case":c.name,"production":p.json(),"digests":ds}));
        }
    }
    WIDTH.with(|x| x.set(2));
    json!({"status":"pass","cases":cases,"diagnostics":diagnostics})
}
pub fn padding_check() {
    let mut rng = Rng(1);
    let mut c = Case::new('R', 2, 2, &mut rng);
    c.k = vec![
        Fp::from_u64_reduced(1),
        Fp::from_u64_reduced(2),
        Fp::from_u64_reduced(3),
    ];
    c.m = vec![Fp::ONE; 4];
    c.domain.m = 4;
    c.refresh_descriptor();
    let p = prove(&c, 0);
    verify(&c, &p);
    let candidate = prove(&c, 2);
    fs::write(
        results().join("padding-debug.json"),
        serde_json::to_vec_pretty(&json!({"p":p.json(),"c":candidate.json()})).unwrap(),
    )
    .unwrap();
    println!(
        "ragged lookup3: first lookup=row1, table padding=row0; P={}, C={}",
        p.digest(),
        candidate.digest()
    );
    assert_eq!(
        candidate.digest(),
        p.digest(),
        "ragged padding exact output mismatch"
    );
    verify(&c, &candidate);
    println!("ragged padding independent acceptance pass");
}
pub fn arithmetic() -> Value {
    let mut rng = Rng(0x7E55_E7A0_0005_0021);
    let mut rows = Vec::new();
    let mut groups = vec![
        (0i128, 0u64, 1usize),
        (1, Fp::MODULUS - 1, 1024),
        (-1, Fp::MODULUS - 1, 1025),
        (18i128 * (1i128 << 112), Fp::MODULUS - 1, 2048),
        (-18i128 * (1i128 << 112), Fp::MODULUS - 1, 2048),
    ];
    for i in 0..96 {
        let c = (i128::from(rng.next()) << 36) | i128::from(rng.next() & ((1 << 36) - 1));
        let c = if i % 2 == 0 { c } else { -c };
        groups.push((
            c,
            rng.next() % Fp::MODULUS,
            [1, 31, 1023, 1024, 1025, 2049][i % 6],
        ));
    }
    for (c, w, count) in groups {
        let mut result = Fp::ZERO;
        let mut buckets = Vec::new();
        for start in (0..count).step_by(coeff::RESET) {
            let mut acc = arith::Acc::default();
            for _ in start..count.min(start + coeff::RESET) {
                acc.weighted(c, w);
            }
            let reduced = acc.signed();
            result += reduced;
            buckets.push(json!({"terms":count.min(start+coeff::RESET)-start,"words":acc.0,"reduced":reduced.to_canonical_u64()}));
        }
        let mag = c.unsigned_abs();
        let cf = Fp::from_u64_reduced(mag as u64)
            + Fp::from_u64_reduced((mag >> 64) as u64) * Fp::from_u64_reduced((1 << 32) - 1);
        let cf = if c < 0 { -cf } else { cf };
        let expected = cf * Fp::from_u64_reduced(w) * Fp::from_u64_reduced(count as u64);
        assert_eq!(
            result, expected,
            "signed weighted stress c={c} count={count}"
        );
        rows.push(json!({"coefficient":c.to_string(),"weight":w,"count":count,"buckets":buckets,"result":result.to_canonical_u64()}));
    }
    let d = coeff::Domain {
        k: 1 << 52,
        v: 0,
        m: 1 << 60,
    };
    assert!(coeff::bound(d, 1, 1024).is_some());
    assert!(coeff::bound(d, 1, 2048).is_none());
    assert!(
        coeff::bound(
            coeff::Domain {
                k: (1 << 32) - 1,
                v: 1 << 56,
                m: (1 << 61) - 1
            },
            3,
            1024
        )
        .is_none()
    );
    let c = coeff::channels(1, 2, 2, -1, 2, 3);
    assert_eq!(c, [3, 1, 2, 3, -2, 5, 7, 4]);
    let (a, b) = (3i128, 5i128);
    let prod = a * a - a * c[0] - a * b * c[1] + c[2] + b * c[3] + b * b * c[4];
    assert_eq!(prod, -48);
    assert_eq!(2 * a - c[0] - b * c[1] + 7 * prod, -338);
    assert_eq!(-a * c[5] + c[6] + b * c[7] + 7 * prod, -324);
    let direct_words = [
        [0, 0, 0],
        [u64::MAX, u64::MAX, (1u64 << 63) - 1],
        [0, 0, 1u64 << 63],
        [u64::MAX, u64::MAX, u64::MAX],
        [0, u64::MAX, u64::MAX],
        [u64::MAX, 0, 1u64 << 63],
    ];
    let direct: Vec<_> = direct_words
        .into_iter()
        .map(|words| json!({"words":words,"result":arith::Acc(words).signed().to_canonical_u64()}))
        .collect();
    let mut gates = Vec::new();
    for kind in ["coordinate", "coefficient", "bucket"] {
        let limit = match kind {
            "coordinate" => i64::MAX as u128,
            "coefficient" => (1u128 << 127) - 1,
            _ => 166153499511800110349651215382478848u128,
        };
        for x in [limit - 1, limit, limit + 1] {
            let admitted = match kind {
                "coordinate" => coeff::coordinate_ok(x),
                "coefficient" => coeff::coefficient_ok(x),
                _ => coeff::bucket_ok(x, 1024),
            };
            assert_eq!(admitted, x <= limit);
            gates.push(json!({"gate":kind,"value":x.to_string(),"admitted":admitted}));
        }
    }
    json!({"status":"pass","direct_signed_reductions":direct,"gate_boundaries":gates,"weighted":rows,"hand_channels":c.iter().map(ToString::to_string).collect::<Vec<_>>(),"hand_product":prod,"cadence":coeff::RESET})
}
pub fn extras() -> Value {
    let mut rng = Rng(0x7E55_E7A0_0005_0021);
    let mut out = Vec::new();
    for control in 0..5 {
        let mut c = Case::new(if control == 0 { 'Q' } else { 'R' }, 3, 2, &mut rng);
        if control == 0 {
            c.tk.truncate(3);
            c.tv.as_mut().unwrap().truncate(3);
            let ids = [2, 1, 2, 0, 1];
            c.k = ids.iter().map(|&i| c.tk[i]).collect();
            c.v = Some(ids.iter().map(|&i| c.tv.as_ref().unwrap()[i]).collect());
            c.m = vec![
                Fp::from_u64_reduced(4),
                Fp::from_u64_reduced(2),
                Fp::from_u64_reduced(2),
                Fp::ZERO,
            ];
        }
        if control == 1 {
            c.k.fill(c.tk[0]);
            c.m.fill(Fp::ZERO);
            c.m[0] = Fp::from_u64_reduced(8);
        }
        if control == 2 {
            c.tk = [0, (1u64 << 32) - 1, 1u64 << 31, 7]
                .map(Fp::from_u64_reduced)
                .to_vec();
            let ids = [3, 2, 1, 0, 0, 1, 2, 3];
            c.k = ids.iter().map(|&i| c.tk[i]).collect();
            c.tv = Some(vec![
                Fp::from_i64(-128),
                Fp::from_i64(127),
                Fp::ZERO,
                Fp::from_i64(-1),
            ]);
            c.v = Some(ids.iter().map(|&i| c.tv.as_ref().unwrap()[i]).collect());
            c.m = vec![Fp::from_u64_reduced(2); 4];
            c.domain.k = (1 << 32) - 1;
            c.domain.v = 128;
        }
        if control == 3 {
            c.domain = coeff::Domain {
                k: 1 << 52,
                v: 0,
                m: 1 << 60,
            };
        }
        if control == 4 {
            c.tk = [Fp::MODULUS - 1, Fp::MODULUS - 2, 0, 1 << 32]
                .map(Fp::from_u64_reduced)
                .to_vec();
            let ids = [0, 1, 2, 3, 3, 2, 1, 0];
            c.k = ids.iter().map(|&i| c.tk[i]).collect();
            c.m = vec![Fp::from_u64_reduced(2); 4];
            c.domain = coeff::Domain {
                k: u128::from(Fp::MODULUS - 1),
                v: 1 << 56,
                m: (1 << 61) - 1,
            };
        }
        c.name = format!("control{control}");
        c.refresh_descriptor();
        let p = prove(&c, 0);
        verify(&c, &p);
        let digest = p.digest();
        for a in [1, 2] {
            let r = prove(&c, a);
            verify(&c, &r);
            assert_eq!(r.digest(), digest);
        }
        out.push(json!({"control":control,"digest":digest,"integer_eligible":coeff::bound(c.domain,2,1024).is_some(),"table_len":c.tk.len(),"lookup_len":c.k.len(),"padded_table_m_last":c.m.last().unwrap().to_canonical_u64()}));
    }
    json!({"status":"pass","controls":out})
}
pub fn grid_vectors() -> Value {
    let mut rng = Rng(0x7E55_E7A0_0005_0021);
    let mut out = Vec::new();
    for case in 0..12 {
        let values = case % 3 != 0;
        let table = case % 2 == 0;
        let mut k = tessera_codec::SecretBuf::new();
        let mut v = tessera_codec::SecretBuf::new();
        let mut m = tessera_codec::SecretBuf::new();
        for i in 0..32 {
            k.push(if i == 0 {
                (1 << 32) - 1
            } else {
                (rng.next() & ((1 << 32) - 1)) as i64
            });
            if values {
                v.push(if i % 2 == 0 { 65536 } else { -128 });
            }
            if table {
                m.push((rng.next() & ((1 << 22) - 1)) as i64);
            }
        }
        let mut field = || {
            Fp2::new(
                Fp::from_u64_reduced(rng.next()),
                Fp::from_u64_reduced(rng.next()),
            )
        };
        let alpha = field();
        let beta = field();
        let lambda = field();
        let point: Vec<_> = (0..4).map(|_| field()).collect();
        let raw = coeff::Raw {
            k,
            v,
            m,
            alpha,
            beta,
            has_values: values,
        };
        for depth in 1..=3 {
            let w = coeff::Window::new(&raw, &point, lambda, depth);
            out.push(json!({"case":case,"depth":depth,"k":raw.k.to_vec(),"v":raw.v.to_vec(),"m":raw.m.to_vec(),"alpha":alpha.words(),"beta":beta.words(),"lambda":lambda.words(),"point":nums(&point),"grid":nums(&w.grid)}));
        }
    }
    json!({"status":"pass","vectors":out})
}
pub fn unit_tag_check() {
    let mut rng = Rng(5);
    let c = Case::new('E', 2, 2, &mut rng);
    let p = prove(&c, 0);
    verify(&c, &p);
    let cproof = prove(&c, 2);
    verify(&c, &cproof);
    assert_eq!(cproof.digest(), p.digest());
    println!("unit numerator classification independent closure pass");
}
pub fn multiple_check() -> Value {
    let mut rng = Rng(0x7E55_E7A0_0005_0021);
    let a = Case::new('Q', 5, 3, &mut rng);
    let b = Case::new('E', 2, 4, &mut rng);
    let cases = [&a, &b];
    let mut expected = None;
    for arm in [0, 1, 2] {
        ARM.with(|x| x.set(arm));
        WIDTH.with(|x| x.set(2));
        coeff::DOMAIN.with(|x| {
            x.set(Some(coeff::Domain {
                k: 31,
                v: 65536,
                m: 32,
            }))
        });
        let mut t = a.transcript();
        t.absorb("exp20/extra", &b.descriptor);
        let r = if arm == 0 {
            let p = tessera_logup::prove_logup(
                &mut t,
                &[a.production(), b.production()],
                &[],
                &tessera_logup::TreeId::paired(2),
            )
            .unwrap();
            record!(p, t.state())
        } else {
            let p = logup::prove_logup(
                &mut t,
                &[a.local(), b.local()],
                &[],
                &logup::TreeId::paired(2),
            )
            .unwrap();
            record!(p, t.state())
        };
        verify_many(&cases, &r);
        if let Some(d) = &expected {
            assert_eq!(r.digest(), *d);
        } else {
            expected = Some(r.digest());
        }
    }
    json!({"instances":2,"independent":true,"digest":expected.unwrap(),"lookup_heights":[5,2],"table_heights":[3,4]})
}

fn tx(x: tv_field::Fp2) -> Fp2 {
    Fp2::new(
        Fp::from_canonical_u64(x.c0.to_canonical()).unwrap(),
        Fp::from_canonical_u64(x.c1.to_canonical()).unwrap(),
    )
}
fn mle_ext(col: &[Fp2], point: &[tv_field::Fp2]) -> tv_field::Fp2 {
    let one = tv_field::Fp2::ONE;
    let mut weights = vec![one];
    for &r in point {
        let n = weights.len();
        for i in 0..n {
            let w = weights[i];
            weights[i] = w * (one - r);
            weights.push(w * r);
        }
    }
    assert_eq!(weights.len(), col.len());
    weights
        .into_iter()
        .zip(col)
        .fold(tv_field::Fp2::ZERO, |s, (w, &v)| s + w * tvx(v))
}
pub fn field_fallback() -> Value {
    let mut rng = Rng(0x7E55_E7A0_0005_0021);
    let mut draw = || {
        Fp2::new(
            Fp::from_u64_reduced(rng.next()),
            Fp::from_u64_reduced(rng.next()),
        )
    };
    let arrays: [Vec<Fp2>; 4] = core::array::from_fn(|_| (0..16).map(|_| draw()).collect());
    let point: Vec<_> = (0..4).map(|_| draw()).collect();
    let [ne, no, de, dd] = &arrays;
    let numerator: Vec<_> = (0..16).map(|i| ne[i] * dd[i] + no[i] * de[i]).collect();
    let denominator: Vec<_> = (0..16).map(|i| de[i] * dd[i]).collect();
    let n = tx(mle_ext(&numerator, &xs(&point)));
    let d = tx(mle_ext(&denominator, &xs(&point)));
    let mut expected = None;
    for arm in [0, 1, 2] {
        ARM.with(|x| x.set(arm));
        let mut t = Transcript::new(&context());
        let gamma = t.challenge("tree/0/gamma");
        let lambda = t.challenge("tree/0/lambda");
        let instance = level::LevelInstance {
            point: &point,
            claim: n + lambda * d,
            tables: level::LevelTables::Fraction {
                numerators: level::LevelNumerators::Ext { even: ne, odd: no },
                d_even: de,
                d_odd: dd,
                lambda,
                raw: None,
            },
        };
        let p = level::prove_level(&mut t, Prefix::Tree(0), &[instance], gamma);
        let rho = t.challenge("tree/0/line");
        let mut vt = tvt::Transcript::new(&tv_context());
        let input = [tv::LevelClaim::Fraction(tv::FractionClaim {
            point: xs(&point),
            numerator: tvx(n),
            denominator: tvx(d),
        })];
        let passed = tv::verify_level(&mut vt, 0, &input, &vp(p.proof())).unwrap();
        assert_eq!(passed.rho, tvx(rho));
        assert_eq!(*vt.state(), t.state());
        let digest = proof_json(p.proof());
        if let Some(old) = &expected {
            assert_eq!(&digest, old);
        } else {
            expected = Some(digest);
        }
        let mut target = vec![tvx(rho)];
        target.extend(xs(p.point()));
        let interleaved_n: Vec<_> = (0..16).flat_map(|i| [ne[i], no[i]]).collect();
        let interleaved_d: Vec<_> = (0..16).flat_map(|i| [de[i], dd[i]]).collect();
        let last = match &passed.claims[0] {
            tv::LevelClaim::Fraction(x) => x,
            _ => panic!("fraction"),
        };
        assert_eq!(last.numerator, mle_ext(&interleaved_n, &target));
        assert_eq!(last.denominator, mle_ext(&interleaved_d, &target));
    }
    json!({"scope":"full-field extension numerator/denominator algebra fallback, not a masked proof","independent":true,"proof":expected.unwrap()})
}
struct Script {
    mode: usize,
    index: usize,
    events: Vec<Value>,
}
impl Script {
    fn state(&self) -> [u8; 32] {
        hash(
            Domain::ColumnMerge,
            &serde_json::to_vec(&self.events).unwrap(),
        )
    }
}
impl SumcheckTranscript<Fp2> for Script {
    fn absorb_message(&mut self, l: &str, m: &[Fp2]) {
        self.events.push(json!({"label":l,"message":nums(m)}));
    }
    fn challenge(&mut self, l: &str) -> Fp2 {
        let x = match self.mode {
            0 => Fp2::ZERO,
            1 => Fp2::ONE,
            2 => {
                if self.index.is_multiple_of(2) {
                    Fp2::ZERO
                } else {
                    Fp2::ONE
                }
            }
            _ => [
                Fp2::ZERO,
                Fp2::ONE,
                Fp2::new(Fp::ZERO, Fp::ONE),
                Fp2::new(Fp::from_u64_reduced(Fp::MODULUS - 1), Fp::ZERO),
            ][self.index % 4],
        };
        self.index += 1;
        self.events.push(json!({"label":l,"challenge":x.words()}));
        x
    }
}
pub fn degenerate() -> Value {
    let mut rng = Rng(29);
    let c = Case::new('Q', 5, 3, &mut rng);
    let mut out = Vec::new();
    for mode in 0..4 {
        let mut expected = None;
        let mut records = Vec::new();
        for arm in [0, 1, 2] {
            ARM.with(|x| x.set(arm));
            WIDTH.with(|x| x.set(2));
            coeff::DOMAIN.with(|x| x.set(Some(c.domain)));
            let mut script = Script {
                mode,
                index: 0,
                events: vec![],
            };
            let r = if arm == 0 {
                let p = tessera_logup::prove_logup(
                    &mut script,
                    &[c.production()],
                    &[],
                    &tessera_logup::TreeId::paired(1),
                )
                .unwrap();
                record!(p, script.state())
            } else {
                let p =
                    logup::prove_logup(&mut script, &[c.local()], &[], &logup::TreeId::paired(1))
                        .unwrap();
                record!(p, script.state())
            };
            let checked = crate::script_oracle::check(&c, &r);
            records.push(json!({"arm":arm,"direct_round_evaluations":checked,"independent_arithmetic":true,"record":r.json(),"events":script.events}));
            if let Some(d) = &expected {
                assert_eq!(r.digest(), *d);
            } else {
                expected = Some(r.digest());
            }
        }
        out.push(json!({"mode":mode,"digest":expected.unwrap(),"scope":"independent tv-field direct tree/cubic-message/leaf arithmetic; scripted zero denominators do not imply public verification acceptance or ZK","arms":records}));
    }
    json!({"cases":out})
}
pub fn rejection_controls() -> Value {
    let mut rng = Rng(99);
    let c = Case::new('E', 3, 3, &mut rng);
    let r = prove(&c, 0);
    verify(&c, &r);
    let mut out = Vec::new();
    for kind in 0..3 {
        let mut roots = xs(&r.roots);
        let expected = match kind {
            0 => {
                roots[1] = tv_field::Fp2::ZERO;
                tl::LookupError::LookupDenominator { instance: 0 }
            }
            1 => {
                roots[3] = tv_field::Fp2::ZERO;
                tl::LookupError::TableDenominator { instance: 0 }
            }
            _ => {
                roots[2] = -roots[2];
                tl::LookupError::Unbalanced { instance: 0 }
            }
        };
        let error = tl::check_roots(&mut c.independent_transcript(), 1, &roots).unwrap_err();
        assert_eq!(error, expected);
        out.push(format!("{error:?}"));
    }
    for changed in [false, true] {
        let mut t = c.independent_transcript();
        let roots = tl::check_roots(&mut t, 1, &xs(&r.roots)).unwrap();
        let mut claims = vec![
            tv::LevelClaim::Fraction(roots.instances[0].lookup.clone()),
            tv::LevelClaim::Fraction(roots.instances[0].table.clone()),
        ];
        for (i, l) in r.levels.iter().enumerate() {
            let mut proof = vp(&l.proof);
            if i == 1 {
                if changed {
                    proof.final_evals[0] += tv_field::Fp2::ONE;
                } else {
                    proof.rounds[0][0] += tv_field::Fp2::ONE;
                }
                let error = tv::verify_level(&mut t, i as u32, &claims, &proof).unwrap_err();
                assert_eq!(
                    error,
                    tv_core::sumcheck::SumcheckError::FinalMismatch {
                        round: proof.rounds.len() as u32
                    }
                );
                out.push(format!("{error:?}"));
                break;
            }
            claims = tv::verify_level(&mut t, i as u32, &claims, &proof)
                .unwrap()
                .claims;
        }
    }
    for kind in 0..2 {
        let mut t = c.independent_transcript();
        let roots = tl::check_roots(&mut t, 1, &xs(&r.roots)).unwrap();
        let mut claims = vec![
            tv::LevelClaim::Fraction(roots.instances[0].lookup.clone()),
            tv::LevelClaim::Fraction(roots.instances[0].table.clone()),
        ];
        for (i, l) in r.levels.iter().enumerate() {
            claims = tv::verify_level(&mut t, i as u32, &claims, &vp(&l.proof))
                .unwrap()
                .claims;
        }
        let mut leaves = [tl::Leaves {
            lookup: leaf(&r.leaves[0]),
            table: leaf(&r.leaves[1]),
            has_values: true,
        }];
        let expected = if kind == 0 {
            leaves[0].lookup.numerator = tv_field::Fp2::ZERO;
            tl::LookupError::LeafNumerator { instance: 0 }
        } else {
            tl::LookupError::TableEvaluation { instance: 0 }
        };
        let beta = roots.challenges.beta;
        let error = tl::check_leaves(
            &mut t,
            &roots.challenges,
            &leaves,
            &xs(&r.values),
            |_, p| {
                let phi = mle(&c.tk, p, c.tk[0])
                    + beta * mle(c.tv.as_ref().unwrap(), p, c.tv.as_ref().unwrap()[0]);
                Some(
                    phi + if kind == 1 {
                        tv_field::Fp2::ONE
                    } else {
                        tv_field::Fp2::ZERO
                    },
                )
            },
        )
        .unwrap_err();
        assert_eq!(error, expected);
        out.push(format!("{error:?}"));
    }
    json!({"rejections":out,"independent_public_verifier":true})
}
pub fn run() {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("prepare") => {
            crate::measure::preflight();
        }
        Some("all") => {
            crate::measure::all(std::path::Path::new(args.get(2).expect("raw output path")));
        }
        Some("padding-check") => padding_check(),
        Some("unit-tag-check") => unit_tag_check(),
        Some("check") => {
            let a = arithmetic();
            fs::write(
                results().join("arithmetic.json"),
                serde_json::to_vec_pretty(&a).unwrap(),
            )
            .unwrap();
            padding_check();
            let field = field_fallback();
            fs::write(
                results().join("field-fallback.json"),
                serde_json::to_vec_pretty(&field).unwrap(),
            )
            .unwrap();
            let scripts = degenerate();
            fs::write(
                results().join("degenerate.json"),
                serde_json::to_vec_pretty(&scripts).unwrap(),
            )
            .unwrap();
            let rejects = rejection_controls();
            fs::write(
                results().join("rejections.json"),
                serde_json::to_vec_pretty(&rejects).unwrap(),
            )
            .unwrap();
            let mult = multiple_check();
            fs::write(
                results().join("multiple.json"),
                serde_json::to_vec_pretty(&mult).unwrap(),
            )
            .unwrap();
            let x = extras();
            fs::write(
                results().join("extras.json"),
                serde_json::to_vec_pretty(&x).unwrap(),
            )
            .unwrap();
            let g = grid_vectors();
            fs::write(
                results().join("grids.json"),
                serde_json::to_vec_pretty(&g).unwrap(),
            )
            .unwrap();
            let r = check();
            fs::write(
                results().join("check.json"),
                serde_json::to_vec_pretty(&r).unwrap(),
            )
            .unwrap();
            println!(
                "correctness pass, {} cases",
                r["cases"].as_array().unwrap().len()
            );
        }
        _ => panic!("usage: integer-fraction check"),
    }
}
