// SPDX-License-Identifier: Apache-2.0 OR MIT
use super::*;
use std::time::{SystemTime, UNIX_EPOCH};
#[derive(Serialize, Deserialize, Clone)]
struct Sample {
    arm: String,
    elapsed_seconds: f64,
    repetitions: usize,
    seconds_per_proof: f64,
    start_unix_seconds: f64,
    end_unix_seconds: f64,
}
#[derive(Serialize, Deserialize, Clone)]
struct Block {
    index: usize,
    order: String,
    samples: Vec<Sample>,
}
fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_secs_f64()
}
fn sample(arm: char, c: &Config) -> Sample {
    let start = now();
    let mut elapsed = Duration::ZERO;
    let mut reps = 0;
    while elapsed < MIN_ITERATION {
        let t = Instant::now();
        let output = if arm == 'B' {
            prove_candidate(black_box(c))
        } else {
            prove_production(black_box(c))
        };
        drop(black_box(output));
        elapsed += t.elapsed();
        reps += 1;
    }
    Sample {
        arm: arm.to_string(),
        elapsed_seconds: elapsed.as_secs_f64(),
        repetitions: reps,
        seconds_per_proof: elapsed.as_secs_f64() / reps as f64,
        start_unix_seconds: start,
        end_unix_seconds: now(),
    }
}
fn stats(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let variance = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    json!({"median":(sorted[n/2-1]+sorted[n/2])/2.0,"min":sorted[0],"max":sorted[n-1],"cv":variance.sqrt()/mean,"mean":mean,"values":values})
}
fn lookup(b: &Block, arm: &str) -> f64 {
    b.samples
        .iter()
        .find(|s| s.arm == arm)
        .expect("arm")
        .seconds_per_proof
}
fn analysis(blocks: &[Block], c: &Config, u: &Config, sham: bool) -> Value {
    let arms = if c.judged && !sham {
        vec!["A", "B", "U"]
    } else if sham {
        vec!["A", "D"]
    } else {
        vec!["A", "B"]
    };
    let mut data = serde_json::Map::new();
    let mut aggregate = serde_json::Map::new();
    let mut noise = true;
    for arm in arms {
        let s = stats(&blocks.iter().map(|b| lookup(b, arm)).collect::<Vec<_>>());
        noise &= s["cv"].as_f64().expect("cv") <= CV_LIMIT;
        let samples = blocks
            .iter()
            .flat_map(|b| &b.samples)
            .filter(|s| s.arm == arm)
            .collect::<Vec<_>>();
        let elapsed = samples.iter().map(|s| s.elapsed_seconds).sum::<f64>();
        let reps = samples.iter().map(|s| s.repetitions).sum::<usize>();
        let count = if arm == "U" { u.real() } else { c.real() };
        aggregate.insert(arm.into(),json!({"seconds_per_proof":elapsed/reps as f64,"ns_per_real":elapsed*1e9/(reps*count) as f64,"ns_per_padded":elapsed*1e9/(reps*if arm=="U"{u.padded()}else{c.padded()}) as f64}));
        data.insert(arm.into(), s);
    }
    let second = if sham { "D" } else { "B" };
    let r = stats(
        &blocks
            .iter()
            .map(|b| {
                if sham {
                    b.samples[0].seconds_per_proof / b.samples[1].seconds_per_proof
                } else {
                    lookup(b, "A") / lookup(b, second)
                }
            })
            .collect::<Vec<_>>(),
    );
    noise &= r["cv"].as_f64().expect("cv") <= CV_LIMIT;
    let mut result = json!({"arms":data,"aggregate":aggregate,"R":r,"noise_pass":noise});
    let ra = result["aggregate"]["A"]["seconds_per_proof"]
        .as_f64()
        .expect("a")
        / result["aggregate"][second]["seconds_per_proof"]
            .as_f64()
            .expect("b");
    result["R_agg"] = json!(ra);
    result["R_over_f"] = json!(result["R"]["median"].as_f64().expect("r") / c.factor());
    result["R_agg_over_f"] = json!(ra / c.factor());
    if c.judged && !sham {
        let h = stats(
            &blocks
                .iter()
                .map(|b| (lookup(b, "B") / c.real() as f64) / (lookup(b, "U") / u.real() as f64))
                .collect::<Vec<_>>(),
        );
        noise &= h["cv"].as_f64().expect("cv") <= CV_LIMIT;
        result["H"] = h;
        result["H_agg"] = json!(
            result["aggregate"]["B"]["ns_per_real"].as_f64().expect("b")
                / result["aggregate"]["U"]["ns_per_real"].as_f64().expect("u")
        );
    }
    if sham {
        noise &= (0.97..=1.03).contains(&result["R"]["median"].as_f64().expect("sham ratio"));
    }
    result["noise_pass"] = json!(noise);
    result
}
fn cohort(c: &Config, u: &Config, sham: bool) -> Value {
    let mut attempts = Vec::new();
    let max_attempts = if sham || c.judged { 2 } else { 1 };
    for attempt in 0..max_attempts {
        let mut warmups = Vec::new();
        let mut measured = Vec::new();
        for index in 0..WARMUPS + ITERATIONS {
            let order = if sham {
                if index % 2 == 0 { "AD" } else { "DA" }
            } else if c.judged {
                ["ABU", "BAU", "UBA", "UAB", "AUB", "BUA"][index % 6]
            } else if index % 2 == 0 {
                "AB"
            } else {
                "BA"
            };
            let samples = order
                .chars()
                .map(|a| sample(a, if a == 'U' { u } else { c }))
                .collect();
            let block = Block {
                index,
                order: order.into(),
                samples,
            };
            if index < WARMUPS {
                warmups.push(block);
            } else {
                measured.push(block);
            }
        }
        let analysis = analysis(&measured, c, u, sham);
        let pass = analysis["noise_pass"].as_bool().expect("noise");
        println!(
            "{}{} attempt{} {}",
            c.name,
            if sham { " sham" } else { "" },
            attempt + 1,
            analysis
        );
        attempts.push(json!({"attempt":attempt+1,"fresh_warmups":warmups,"measured":measured,"analysis":analysis}));
        if pass {
            break;
        }
    }
    json!({"config":c.name,"judged":c.judged&&!sham,"sham":sham,"real_leaves":c.real(),"padded_leaves":c.padded(),"factor":c.factor(),"selected_attempt":attempts.len(),"attempts":attempts})
}
fn selected(c: &Value) -> &Value {
    &c["attempts"][c["selected_attempt"].as_u64().expect("selected") as usize - 1]["analysis"]
}
pub(super) fn run(path: &str) {
    let d: Value = serde_json::from_slice(
        &std::fs::read("results/check.json").expect("run correctness first"),
    )
    .expect("check JSON");
    assert_eq!(d["correctness"], "pass");
    let configs = configs();
    let u = &configs[4];
    let mut result = json!({"experiment":"EXP-0019","protocol":{"warmups":3,"samples":10,"min_sample_ms":20,"cv_limit":0.03,"max_reruns":1,"exclusive_cpu_authorized":true},"start_unix_seconds":now(),"digests":d["digests"],"cohorts":[]});
    let sham = cohort(&configs[1], u, true);
    result["sham"] = sham;
    std::fs::write(path, serde_json::to_vec_pretty(&result).expect("JSON")).expect("checkpoint");
    for c in &configs {
        let row = cohort(c, u, false);
        result["cohorts"].as_array_mut().expect("cohorts").push(row);
        std::fs::write(path, serde_json::to_vec_pretty(&result).expect("JSON"))
            .expect("checkpoint");
    }
    let noise = selected(&result["sham"])["noise_pass"] == true
        && result["cohorts"]
            .as_array()
            .expect("cohorts")
            .iter()
            .filter(|c| c["judged"] == true)
            .all(|c| selected(c)["noise_pass"] == true);
    let primary = result["cohorts"]
        .as_array()
        .expect("cohorts")
        .iter()
        .filter(|c| c["judged"] == true)
        .collect::<Vec<_>>();
    let valid = primary.iter().all(|c| {
        let s = selected(c);
        (0.9..=1.1).contains(&s["R_over_f"].as_f64().expect("r"))
            && (0.9..=1.1).contains(&s["R_agg_over_f"].as_f64().expect("ra"))
            && s["H"]["median"].as_f64().expect("h") <= 1.05
            && s["H_agg"].as_f64().expect("ha") <= 1.05
    });
    let refuted = primary.iter().any(|c| {
        selected(c)["R"]["median"].as_f64().expect("r")
            <= 1.0 + (c["factor"].as_f64().expect("f") - 1.0) / 2.0
    });
    let verdict = if !noise {
        "inconclusive"
    } else if valid {
        "validated"
    } else if refuted {
        "refuted"
    } else {
        "inconclusive"
    };
    result["noise_pass"] = json!(noise);
    result["verdict"] = json!(verdict);
    result["end_unix_seconds"] = json!(now());
    std::fs::write(path, serde_json::to_vec_pretty(&result).expect("JSON")).expect("final JSON");
    println!("TIMING_TERMINAL: {verdict}");
}
