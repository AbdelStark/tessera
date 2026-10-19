// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Paid-prefix diagnostic. Amendment f8c2d71 was verified before this implementation.
use crate::driver::{Case, Fraction, Level, Record};
use crate::*;
use serde_json::{Value, json};
use tessera_codec::SecretBuf;
use tessera_hash::Transcript;
#[derive(Clone)]
pub enum Event {
    Message(String, Vec<Fp2>),
    Challenge(String, Fp2),
}
pub struct Capture {
    pub inner: Transcript,
    pub events: Vec<Event>,
    pub states: Vec<[u8; 32]>,
}
impl SumcheckTranscript<Fp2> for Capture {
    fn absorb_message(&mut self, label: &str, m: &[Fp2]) {
        self.inner.absorb_message(label, m);
        self.events.push(Event::Message(label.into(), m.to_vec()));
        self.states.push(self.inner.state());
    }
    fn challenge(&mut self, label: &str) -> Fp2 {
        let x = self.inner.challenge(label);
        self.events.push(Event::Challenge(label.into(), x));
        self.states.push(self.inner.state());
        x
    }
}
pub struct Diagnostic {
    pub prefix: Vec<Event>,
    pub incoming: Vec<Fraction>,
    pub level: u32,
    pub alpha: Fp2,
    pub beta: Fp2,
    pub expected: Value,
    pub expected_state: [u8; 32],
}
impl Diagnostic {
    pub fn new(case: &Case) -> Self {
        let mut capture = Capture {
            inner: case.transcript(),
            events: vec![],
            states: vec![],
        };
        let p = tessera_logup::prove_logup(
            &mut capture,
            &[case.production()],
            &[],
            &tessera_logup::TreeId::paired(1),
        )
        .unwrap();
        let record = driver::production_record(p, capture.inner.state());
        driver::verify(case, &record);
        let level = record.levels.len() - 1;
        let label = format!("tree/{level}/gamma");
        let end = capture
            .events
            .iter()
            .position(|e| matches!(e,Event::Challenge(l,_) if *l==label))
            .unwrap();
        let last = format!("tree/{level}/line");
        let last = capture
            .events
            .iter()
            .position(|e| matches!(e,Event::Challenge(l,_) if *l==last))
            .unwrap();
        let mut incoming = vec![
            Fraction {
                point: vec![],
                n: record.roots[0],
                d: record.roots[1],
            },
            Fraction {
                point: vec![],
                n: record.roots[2],
                d: record.roots[3],
            },
        ];
        for l in &record.levels[..level] {
            for (&id, c) in l.trees.iter().zip(&l.next) {
                incoming[id] = c.clone();
            }
        }
        Self {
            prefix: capture.events[..end].to_vec(),
            incoming,
            level: level as u32,
            alpha: record.alpha,
            beta: record.beta,
            expected: driver::level_json(&record.levels[level]),
            expected_state: capture.states[last],
        }
    }
    fn replay(&self, case: &Case) -> Transcript {
        let mut t = case.transcript();
        for event in &self.prefix {
            match event {
                Event::Message(label, m) => t.absorb_message(label, m),
                Event::Challenge(label, x) => assert_eq!(t.challenge(label), *x),
            }
        }
        t
    }
    pub fn run(&self, case: &Case, arm: u8) -> (Level, [u8; 32]) {
        ARM.with(|x| x.set(arm));
        coeff::DOMAIN.with(|x| x.set(Some(case.domain)));
        let mut t = self.replay(case);
        let gamma = t.challenge(&format!("tree/{}/gamma", self.level));
        let lambda = t.challenge(&format!("tree/{}/lambda", self.level));
        struct Prep {
            n0: SecretBuf<Fp>,
            n1: SecretBuf<Fp>,
            d0: SecretBuf<Fp2>,
            d1: SecretBuf<Fp2>,
            raw: Option<coeff::Raw<Fp2>>,
        }
        let depth = width().min(self.incoming.iter().map(|c| c.point.len()).max().unwrap());
        let integer = arm == 2 && coeff::bound(case.domain, depth, 1024).is_some();
        let mut preps = Vec::with_capacity(2);
        for table in [false, true] {
            let keys = if table { &case.tk } else { &case.k };
            let values = if table { &case.tv } else { &case.v };
            let count = keys.len().next_power_of_two();
            let mut n0 = SecretBuf::new();
            let mut n1 = SecretBuf::new();
            let mut d0 = SecretBuf::new();
            let mut d1 = SecretBuf::new();
            let raw = if integer {
                let neg: SecretBuf<Fp> = if table {
                    case.m.iter().map(|&m| -m).collect()
                } else {
                    SecretBuf::new()
                };
                Some(coeff::Raw::new(
                    keys,
                    values.as_deref(),
                    if table { Some(&neg) } else { None },
                    count,
                    self.alpha,
                    self.beta,
                    case.tk[0],
                    case.tv.as_ref().map(|v| v[0]),
                ))
            } else {
                d0 = SecretBuf::with_capacity(count / 2);
                d1 = SecretBuf::with_capacity(count / 2);
                if table {
                    n0 = SecretBuf::with_capacity(count / 2);
                    n1 = SecretBuf::with_capacity(count / 2);
                }
                for i in 0..count {
                    let key = keys.get(i).copied().unwrap_or(case.tk[0]);
                    let value = values.as_ref().map_or(Fp::ZERO, |v| {
                        v.get(i)
                            .copied()
                            .unwrap_or_else(|| case.tv.as_ref().unwrap()[0])
                    });
                    let d = self.alpha - Fp2::from(key) - self.beta.mul_by_base(value);
                    if i % 2 == 0 {
                        d0.push(d);
                        if table {
                            n0.push(-case.m[i]);
                        }
                    } else {
                        d1.push(d);
                        if table {
                            n1.push(-case.m[i]);
                        }
                    }
                }
                None
            };
            preps.push(Prep {
                n0,
                n1,
                d0,
                d1,
                raw,
            });
        }
        let instances: Vec<_> = preps
            .iter()
            .enumerate()
            .map(|(i, p)| level::LevelInstance {
                point: &self.incoming[i].point,
                claim: self.incoming[i].n + lambda * self.incoming[i].d,
                tables: level::LevelTables::Fraction {
                    numerators: if i == 0 {
                        level::LevelNumerators::Unit
                    } else {
                        level::LevelNumerators::Base {
                            even: &p.n0,
                            odd: &p.n1,
                        }
                    },
                    d_even: &p.d0,
                    d_odd: &p.d1,
                    lambda,
                    raw: p.raw.as_ref(),
                },
            })
            .collect();
        let p = level::prove_level(&mut t, Prefix::Tree(self.level), &instances, gamma);
        let rho = t.challenge(&format!("tree/{}/line", self.level));
        let mut next = Vec::new();
        for i in 0..2 {
            let ev = p.instance_evals(i).unwrap();
            let mut point = vec![rho];
            point.extend(p.instance_point(i).unwrap());
            next.push(Fraction {
                point,
                n: ev[0] + rho * (ev[1] - ev[0]),
                d: ev[2] + rho * (ev[3] - ev[2]),
            });
        }
        let point = p.point().to_vec();
        let level = Level {
            proof: p.into_proof(),
            trees: vec![0, 1],
            next,
            gamma,
            lambda,
            rho,
            point,
        };
        (level, t.state())
    }
    pub fn check(&self, case: &Case) {
        for a in [0, 1, 2] {
            let (l, s) = self.run(case, a);
            assert_eq!(driver::level_json(&l), self.expected);
            assert_eq!(s, self.expected_state);
        }
    }
}

pub fn peak_bound(case: &Case, arm: u8) -> u64 {
    let (l, t) = (
        case.k.len().next_power_of_two() as u64,
        case.tk.len().next_power_of_two() as u64,
    );
    let has = case.tv.is_some();
    let first = if arm == 2 && !has {
        40 * l + 56 * t
    } else {
        48 * l + 64 * t
    };
    // Conservative structural ceiling: all persistent levels plus every possible bound table,
    // old/new grid, scalar scratch, proof Vec capacity and full-capacity clearing storage.
    // The exact live-level/bind capacities are separately reported by count.py.
    let cap = 4 * first + 64 * 1024 * 1024;
    assert!(cap <= 4 * 1024 * 1024 * 1024, "registered scratch ceiling");
    cap
}
pub fn preflight() -> Value {
    let mut rng = driver::Rng(0x7E55_E7A0_0005_0020);
    let mut records = Vec::new();
    let mut counts = Vec::new();
    for (profile, tbits) in [('E', 22), ('R', 21), ('Q', 20)] {
        for h in [20, 22] {
            let case = Case::new(profile, h, tbits, &mut rng);
            println!("preflight {}", case.name);
            WIDTH.with(|x| x.set(2));
            let expected = driver::prove(&case, 0);
            driver::verify(&case, &expected);
            let digest = expected.digest();
            let mut arms = Vec::new();
            for arm in [0, 1, 2] {
                let cap = peak_bound(&case, arm);
                let r = driver::prove(&case, arm);
                driver::verify(&case, &r);
                assert_eq!(r.digest(), digest);
                let c = counts::run(&case, arm);
                assert_eq!(c["statement_state"], json!(expected.state));
                counts.push(c);
                arms.push(json!({"arm":arm,"digest":r.digest(),"independent":true,"scratch_upper_bytes":cap}));
            }
            let diagnostic = Diagnostic::new(&case);
            diagnostic.check(&case);
            records.push(json!({"case":case.name,"L":case.k.len(),"T":case.tk.len(),"profile":profile.to_string(),"table_id":case.table_id,"descriptor":case.descriptor,"m_max":case.m.iter().map(|m|m.to_canonical_u64()).max(),"m_sum":case.m.iter().map(|m|m.to_canonical_u64()).sum::<u64>(),"arms":arms,"prefix_events":diagnostic.prefix.len(),"bottom_exact":true,"production":expected.json()}));
            std::fs::write(
                driver::results().join("count-trace.json"),
                serde_json::to_vec_pretty(&json!({"primary_v":2,"records":counts})).unwrap(),
            )
            .unwrap();
            std::fs::write(
                driver::results().join("preflight.json"),
                serde_json::to_vec_pretty(&json!({"cases":records})).unwrap(),
            )
            .unwrap();
        }
    }
    json!({"cases":records})
}
fn statistics(x: &[f64]) -> (f64, f64) {
    let mut sorted = x.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let cv =
        (x.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt() / mean;
    ((sorted[4] + sorted[5]) / 2., cv)
}
fn summary(samples: &[Vec<Value>], bottom: bool) -> Value {
    let names = ["P", "P2", "D", "C"];
    let mut per: [Vec<f64>; 4] = core::array::from_fn(|_| Vec::new());
    let mut elapsed = [0.; 4];
    let mut reps = [0.; 4];
    let mut clean = true;
    for block in samples {
        for s in block {
            let i = names
                .iter()
                .position(|&n| n == s["arm"].as_str().unwrap())
                .unwrap();
            let ns = s["elapsed_ns"].as_u64().unwrap();
            let r = s["repetitions"].as_u64().unwrap();
            clean &= ns >= 20_000_000;
            per[i].push(ns as f64 / r as f64);
            elapsed[i] += ns as f64;
            reps[i] += r as f64;
        }
    }
    for x in &per {
        clean &= statistics(x).1 <= 0.03;
    }
    let ratios = |a: usize, b: usize| {
        per[a]
            .iter()
            .zip(&per[b])
            .map(|(a, b)| a / b)
            .collect::<Vec<_>>()
    };
    let cp = ratios(3, 0);
    let cd = ratios(3, 2);
    let sham = ratios(1, 0);
    let agg: Vec<_> = (0..4).map(|i| elapsed[i] / reps[i]).collect();
    let sm = statistics(&sham);
    clean &= statistics(&cp).1 <= 0.03
        && (bottom || statistics(&cd).1 <= 0.03)
        && sm.1 <= 0.03
        && (0.98..=1.02).contains(&sm.0)
        && (0.98..=1.02).contains(&(agg[1] / agg[0]));
    json!({"clean":clean,"paired_saving_P":1.-statistics(&cp).0,"paired_saving_D":1.-statistics(&cd).0,"aggregate_saving_P":1.-agg[3]/agg[0],"aggregate_saving_D":1.-agg[3]/agg[2],"arm_cv":per.iter().map(|x|statistics(x).1).collect::<Vec<_>>(),"ratio_cv":[statistics(&cp).1,statistics(&cd).1,sm.1],"sham_paired":sm.0,"sham_aggregate":agg[1]/agg[0]})
}
fn once(case: &Case, d: &Diagnostic, bottom: bool, arm: usize) {
    let code = [0, 0, 1, 2][arm];
    if bottom {
        drop(std::hint::black_box(d.run(case, code)));
    } else {
        drop(std::hint::black_box(driver::prove(case, code)));
    }
}
fn cohort(
    case: &Case,
    d: &Diagnostic,
    bottom: bool,
    attempt: usize,
    repetitions: &mut [u64; 4],
    origin: std::time::Instant,
) -> Value {
    let names = ["P", "P2", "D", "C"];
    let mut warmups = Vec::new();
    let mut samples = Vec::new();
    for block in 0..13 {
        let mut row = Vec::new();
        for j in 0..4 {
            let arm = (block + j) % 4;
            let begin = std::time::Instant::now();
            for _ in 0..repetitions[arm] {
                once(case, d, bottom, arm);
            }
            let end = std::time::Instant::now();
            let start = begin.duration_since(origin).as_nanos() as u64;
            let stop = end.duration_since(origin).as_nanos() as u64;
            row.push(json!({"arm":names[arm],"start_ns":start,"end_ns":stop,"elapsed_ns":stop-start,"repetitions":repetitions[arm]}));
        }
        if block < 3 {
            warmups.push(row);
        } else {
            samples.push(row);
        }
    }
    let s = summary(&samples, bottom);
    json!({"case":case.name,"boundary":if bottom{"prefix-replay+bottom"}else{"whole"},"attempt":attempt,"warmups":warmups,"samples":samples,"summary":s,"prefix_events":d.prefix.len()})
}
pub fn all(path: &std::path::Path) {
    let origin = std::time::Instant::now();
    let mut rng = driver::Rng(0x7E55_E7A0_0005_0020);
    let mut cohorts = Vec::new();
    let mut exact = Vec::new();
    for (profile, tbits) in [('E', 22), ('R', 21), ('Q', 20)] {
        for h in [20, 22] {
            let case = Case::new(profile, h, tbits, &mut rng);
            WIDTH.with(|x| x.set(2));
            for a in [0, 1, 2] {
                peak_bound(&case, a);
            }
            let p = driver::prove(&case, 0);
            driver::verify(&case, &p);
            let digest = p.digest();
            let mut before = vec![json!({"arm":"P","digest":digest,"independent":true})];
            for (name, a) in [("P2", 0), ("D", 1), ("C", 2)] {
                let r = driver::prove(&case, a);
                driver::verify(&case, &r);
                let value = r.digest();
                assert_eq!(value, digest);
                before.push(json!({"arm":name,"digest":value,"independent":true}));
            }
            let d = Diagnostic::new(&case);
            d.check(&case);
            for bottom in [false, true] {
                println!(
                    "timing {} {}",
                    case.name,
                    if bottom {
                        "prefix-replay+bottom"
                    } else {
                        "whole"
                    }
                );
                let mut reps = [1u64; 4];
                for (a, r) in reps.iter_mut().enumerate() {
                    let begin = std::time::Instant::now();
                    once(&case, &d, bottom, a);
                    let ns = begin.elapsed().as_nanos() as u64;
                    *r = 30_000_000u64.div_ceil(ns.max(1)).max(1);
                }
                for attempt in 0..2 {
                    let row = cohort(&case, &d, bottom, attempt, &mut reps, origin);
                    let clean = row["summary"]["clean"].as_bool().unwrap();
                    if !clean {
                        let mut minimum = [u64::MAX; 4];
                        for sample in row["samples"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .flat_map(|r| r.as_array().unwrap())
                        {
                            let a = ["P", "P2", "D", "C"]
                                .iter()
                                .position(|&a| a == sample["arm"].as_str().unwrap())
                                .unwrap();
                            minimum[a] = minimum[a].min(sample["elapsed_ns"].as_u64().unwrap());
                        }
                        for a in 0..4 {
                            if minimum[a] < 20_000_000 {
                                reps[a] = reps[a]
                                    .checked_mul(30_000_000u64.div_ceil(minimum[a].max(1)))
                                    .expect("public repetition bound");
                            }
                        }
                    }
                    println!("{} attempt{attempt}: {}", case.name, row["summary"]);
                    cohorts.push(row);
                    std::fs::write(
                        path,
                        serde_json::to_vec_pretty(
                            &json!({"verdict":"pending","cohorts":cohorts,"exact":exact}),
                        )
                        .unwrap(),
                    )
                    .unwrap();
                    if clean {
                        break;
                    }
                }
            }
            let mut after = Vec::new();
            for (name, a) in [("P", 0), ("P2", 0), ("D", 1), ("C", 2)] {
                let r = driver::prove(&case, a);
                driver::verify(&case, &r);
                let value = r.digest();
                assert_eq!(value, digest);
                after.push(json!({"arm":name,"digest":value,"independent":true}));
            }
            d.check(&case);
            exact.push(json!({"case":case.name,"descriptor":case.descriptor,"digest":digest,"before_after_all_equal":true,"independent":true,"before":before,"after":after,"prefix_events":d.prefix.len()}));
        }
    }
    let mut selected = std::collections::BTreeMap::new();
    for c in &cohorts {
        selected.insert(
            (c["case"].as_str().unwrap(), c["boundary"].as_str().unwrap()),
            c,
        );
    }
    let mut g = 1f64;
    let mut h = 1f64;
    let mut clean = true;
    for ((_, boundary), row) in selected {
        let s = &row["summary"];
        clean &= s["clean"].as_bool().unwrap();
        if boundary == "whole" {
            for key in [
                "paired_saving_P",
                "paired_saving_D",
                "aggregate_saving_P",
                "aggregate_saving_D",
            ] {
                g = g.min(s[key].as_f64().unwrap());
            }
        } else {
            for key in ["paired_saving_P", "aggregate_saving_P"] {
                h = h.min(s[key].as_f64().unwrap());
            }
        }
    }
    let verdict = if clean && g >= 0.05 && h >= 0.30 {
        "validated"
    } else if clean && (g < 0.02 || h < 0.15) {
        "refuted"
    } else {
        "inconclusive"
    };
    let raw = json!({"primary_v":2,"verdict":verdict,"G2":g,"H2":h,"all_noise_clean":clean,"cohorts":cohorts,"exact":exact});
    std::fs::write(path, serde_json::to_vec_pretty(&raw).unwrap()).unwrap();
    println!("terminal {verdict}: G2={g} H2={h} clean={clean}");
}
