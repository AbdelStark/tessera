// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Caller-pinned one-request directory Live CLI (RFC-0014/0019/0036).
use crate::{
    CommandOutput,
    directory::Directory,
    files,
    options::{self, Options},
    verify::{self, Execution},
};
use std::{path::Path, time::Duration};
use tv_core::{
    LiveVerifier, ResourceLimits, VerifierConfig,
    params::ParamSet,
    protocol::*,
    verdict::{ErrorCode, RejectReason, Verdict},
};
use tv_hash::{Domain, hash};

const HELP: &str = "tessera-verify live --claims CLAIMS.tcl --weights WEIGHTS.twr --weights-root HEX --statement-digest HEX --bound BITS --connect DIR --request-id N --request-session HEX --prompt-len N --output-len N --manifest-digest HEX [--out VERDICT.json] [--json] [--timeout-ms N] [--max-artifact-bytes N] [--max-memory-bytes N]\nOne-shot RFC-0014 directory exchange, non-hiding protocol/1. The caller owns all request metadata and both pins. No TLS, anchor, ledger or ZK. Test parameters certify nothing. Default message deadlines:600 seconds,3600 for Round2.\n";

fn checked_output(
    out: Option<&str>,
    claims: &str,
    weights: &str,
    dir: &str,
) -> Result<(), ErrorCode> {
    files::output_distinct(out, &[claims, weights, dir])?;
    let canonical = std::fs::canonicalize(dir).map_err(|_| ErrorCode::Io)?;
    if let Some(out) = out {
        let parent = Path::new(out)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        if std::fs::canonicalize(parent)
            .map_err(|_| ErrorCode::Io)?
            .starts_with(canonical)
        {
            return Err(ErrorCode::Usage);
        }
    }
    Ok(())
}

struct Config<'a> {
    claims: &'a str,
    weights: &'a str,
    dir: &'a str,
    out: Option<&'a str>,
    root: [u8; 32],
    pin: [u8; 32],
    bound: FixedBits,
    request: RequestRef,
    manifest: [u8; 32],
    timeout: Option<Duration>,
    limits: ResourceLimits,
}
impl<'a> Config<'a> {
    fn parse(args: &'a [String]) -> Result<Self, ErrorCode> {
        let opts = Options::parse(
            args,
            &[
                "--claims",
                "--weights",
                "--weights-root",
                "--statement-digest",
                "--bound",
                "--connect",
                "--request-id",
                "--request-session",
                "--prompt-len",
                "--output-len",
                "--manifest-digest",
                "--out",
                "--timeout-ms",
                "--max-artifact-bytes",
                "--max-memory-bytes",
            ],
        )?;
        let claims = opts.required("--claims")?;
        let weights = opts.required("--weights")?;
        let dir = opts.required("--connect")?;
        let root = options::digest(opts.required("--weights-root")?)?;
        let pin = options::digest(opts.required("--statement-digest")?)?;
        let bound = FixedBits(options::bound(opts.required("--bound")?)?);
        let request_id = opts.required_number("--request-id")?;
        let session_id = options::hex::<16>(opts.required("--request-session")?)?;
        let prompt_len =
            u32::try_from(opts.required_number("--prompt-len")?).map_err(|_| ErrorCode::Usage)?;
        let output_len =
            u32::try_from(opts.required_number("--output-len")?).map_err(|_| ErrorCode::Usage)?;
        if prompt_len == 0 || output_len == 0 || prompt_len.checked_add(output_len).is_none() {
            return Err(ErrorCode::Usage);
        }
        let manifest = options::digest(opts.required("--manifest-digest")?)?;
        let timeout = opts
            .get("--timeout-ms")
            .map(|_| opts.number("--timeout-ms", 0).map(Duration::from_millis))
            .transpose()?;
        let limits = ResourceLimits {
            max_artifact_bytes: opts.number(
                "--max-artifact-bytes",
                ResourceLimits::default().max_artifact_bytes,
            )?,
            max_memory_bytes: opts.number(
                "--max-memory-bytes",
                ResourceLimits::default().max_memory_bytes,
            )?,
            max_wall_time: timeout.map_or(Duration::from_secs(7200), |t| {
                t.checked_mul(12).unwrap_or(Duration::MAX)
            }),
        };
        Ok(Self {
            claims,
            weights,
            dir,
            out: opts.get("--out"),
            root,
            pin,
            bound,
            request: RequestRef {
                request_id,
                session_id,
                prompt_len,
                output_len,
            },
            manifest,
            timeout,
            limits,
        })
    }
}

pub(crate) fn run(args: &[String], json: bool) -> CommandOutput {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h") {
        return CommandOutput {
            exit_code: 0,
            stdout: HELP.into(),
            stderr: String::new(),
        };
    }
    let c = match Config::parse(args) {
        Ok(c) => c,
        Err(e) => {
            return crate::command_error(
                "live",
                "invalid or missing request policy; see live --help",
                json,
                e.id(),
                e.exit_code(),
            );
        }
    };
    let out = c.out;
    let checked = checked_output(out, c.claims, c.weights, c.dir);
    let mut e = Execution::new();
    let result = (|| {
        checked?;
        e.remaining(c.limits.max_wall_time)?;
        let mut input = files::Inputs {
            remaining: c.limits.max_memory_bytes / 8,
            max_artifact: c.limits.max_artifact_bytes,
        };
        let decode = verify::decode_limits(c.limits);
        let (claims, weights) = e.phase(0, || {
            verify::public_inputs(c.claims, c.weights, &mut input, &decode)
        })?;
        let statement = PublicStatement {
            claim_list_digest: claims.digest(),
            weights_root: c.root,
            weights_epoch: weights.epoch,
            manifest_digest: c.manifest,
            batch: BatchDescriptor {
                requests: vec![c.request.clone()],
            },
            predictor: PredictorSpec::Exact {
                decoding: Decoding::Greedy,
            },
        };
        e.phase(1, || {
            if statement.digest() != c.pin
                || weights.weights_root(|b| hash(Domain::WeightsRoot, b)) != c.root
                || c.request.prompt_len + c.request.output_len > claims.arch.max_context
            {
                return Err(ErrorCode::Usage);
            }
            Ok(())
        })?;
        e.pinned = true;
        let Some(params) = ParamSet::accept(&claims.params) else {
            return Ok(Verdict::Reject(RejectReason::Params {
                name: claims.params,
            }));
        };
        let cfg = VerifierConfig {
            params,
            weights,
            accept_bound: c.bound,
            limits: ResourceLimits {
                max_wall_time: e.remaining(c.limits.max_wall_time)?,
                ..c.limits
            },
        };
        Ok(e.evaluate(|| {
            let mut verifier = LiveVerifier::new(claims, statement, cfg);
            let commit = match verifier.start() {
                Ok(c) => c,
                Err(v) => return v,
            };
            let exchange = (|| {
                let mut d = Directory::new(Path::new(c.dir), input, c.timeout)?;
                d.send(
                    0,
                    SeedCommit::SCHEMA,
                    &commit.to_tce().map_err(ErrorCode::from)?,
                )?;
                let first =
                    Round1::open(&d.receive(1)?, &decode).map_err(|e| ErrorCode::from(e.code()))?;
                let share = ProverSeedShare::open(&d.receive(2)?, &decode)
                    .map_err(|e| ErrorCode::from(e.code()))?;
                let reveal = match verifier.on_round1(&first, &share) {
                    Ok(r) => r,
                    Err(v) => return Ok(v),
                };
                d.send(
                    3,
                    SeedReveal::SCHEMA,
                    &reveal.to_tce().map_err(ErrorCode::from)?,
                )?;
                let second = Round2Response::open(&d.receive(4)?, &decode)
                    .map_err(|e| ErrorCode::from(e.code()))?;
                let queries = match verifier.on_round2(&second) {
                    Ok(q) => q,
                    Err(v) => return Ok(v),
                };
                d.send(
                    5,
                    ColumnQueries::SCHEMA,
                    &queries.to_tce().map_err(ErrorCode::from)?,
                )?;
                let third = Round3Response::open(&d.receive(6)?, &decode)
                    .map_err(|e| ErrorCode::from(e.code()))?;
                Ok(verifier.on_round3(&third))
            })();
            exchange.unwrap_or_else(verify::inconclusive)
        }))
    })();
    verify::finish(
        result.unwrap_or_else(verify::inconclusive),
        e,
        out,
        json,
        checked,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::seal;
    use std::{fs, path::PathBuf};
    use tv_codec::{DecodeLimits, WeightCommitmentPublic, WeightLayoutParams};
    use tv_core::claims::ClaimList;
    use tv_testkit::{Json, hex};
    struct Fixture {
        dir: PathBuf,
        args: Vec<String>,
        claims: Vec<u8>,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
    impl Fixture {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "tessera-live-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            fs::create_dir(&dir).unwrap();
            fs::create_dir(dir.join("exchange")).unwrap();
            let v = tv_testkit::read_vectors("claims/claim-lists.json");
            let bytes = hex(v.get("cases").unwrap().as_array().unwrap()[0]
                .get("hex")
                .unwrap()
                .as_str()
                .unwrap());
            let claims = ClaimList::open(&bytes, &DecodeLimits::MESSAGE).unwrap();
            let p = claims.layout.weights;
            let weights = WeightCommitmentPublic {
                layout: WeightLayoutParams {
                    log2_k: p.log2_k,
                    log2_inv_rate: p.log2_inv_rate,
                    mask_coeffs: p.mask_coeffs,
                    blinding_rows: p.blinding_rows,
                },
                roots: [[3; 32]; 10],
                epoch: 7,
            };
            let root = weights.weights_root(|b| hash(Domain::WeightsRoot, b));
            let s = PublicStatement {
                claim_list_digest: claims.digest(),
                weights_root: root,
                weights_epoch: 7,
                manifest_digest: [19; 32],
                batch: BatchDescriptor {
                    requests: vec![RequestRef {
                        request_id: 23,
                        session_id: [17; 16],
                        prompt_len: 2,
                        output_len: 3,
                    }],
                },
                predictor: PredictorSpec::Exact {
                    decoding: Decoding::Greedy,
                },
            };
            fs::write(dir.join("claims.tcl"), &bytes).unwrap();
            fs::write(
                dir.join("weights.twr"),
                seal(WeightCommitmentPublic::EXPECTED, &weights.encode()).unwrap(),
            )
            .unwrap();
            let h = |v: [u8; 32]| v.iter().map(|b| format!("{b:02x}")).collect::<String>();
            let args = vec![
                "--claims".into(),
                dir.join("claims.tcl").display().to_string(),
                "--weights".into(),
                dir.join("weights.twr").display().to_string(),
                "--weights-root".into(),
                h(root),
                "--statement-digest".into(),
                h(s.digest()),
                "--bound".into(),
                "0".into(),
                "--connect".into(),
                dir.join("exchange").display().to_string(),
                "--request-id".into(),
                "23".into(),
                "--request-session".into(),
                "11".repeat(16),
                "--prompt-len".into(),
                "2".into(),
                "--output-len".into(),
                "3".into(),
                "--manifest-digest".into(),
                "13".repeat(32),
                "--out".into(),
                dir.join("verdict.json").display().to_string(),
                "--json".into(),
            ];
            Self {
                dir,
                args,
                claims: bytes,
            }
        }
        fn set(&self, key: &str, value: String) -> Vec<String> {
            let mut args = self.args.clone();
            let i = args.iter().position(|a| a == key).unwrap();
            args[i + 1] = value;
            args
        }
    }
    fn code(output: CommandOutput, exit: i32, reason: &str) {
        assert_eq!(output.exit_code, exit, "{}", output.stdout);
        assert!(output.stderr.is_empty());
        let j = Json::parse(&output.stdout).unwrap();
        let error = j.get("verdict").unwrap().get("error").unwrap();
        assert_eq!(error.get("code").unwrap().as_str(), Some(reason));
    }
    #[test]
    fn externally_selected_statement_and_epoch_pins_precede_session_side_effects() {
        let f = Fixture::new();
        for (key, value) in [
            ("--statement-digest", "ab".repeat(32)),
            ("--weights-root", "ab".repeat(32)),
            ("--request-id", "24".into()),
            ("--manifest-digest", "ab".repeat(32)),
            ("--request-session", "ab".repeat(16)),
            ("--prompt-len", "3".into()),
        ] {
            code(run(&f.set(key, value), true), 64, "ERR-USAGE");
            assert_eq!(fs::read_dir(f.dir.join("exchange")).unwrap().count(), 0);
        }
    }
    #[test]
    fn epoch_root_check_is_independent_of_the_whole_statement_pin() {
        let f = Fixture::new();
        let mut args = f.set("--weights-root", "ab".repeat(32));
        let claims = ClaimList::open(&f.claims, &DecodeLimits::MESSAGE).unwrap();
        let c = Config::parse(&args).unwrap();
        let expected = PublicStatement {
            claim_list_digest: claims.digest(),
            weights_root: c.root,
            weights_epoch: 7,
            manifest_digest: c.manifest,
            batch: BatchDescriptor {
                requests: vec![c.request.clone()],
            },
            predictor: PredictorSpec::Exact {
                decoding: Decoding::Greedy,
            },
        };
        let pin: String = expected
            .digest()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let i = args.iter().position(|a| a == "--statement-digest").unwrap();
        args[i + 1] = pin;
        args.extend(["--timeout-ms".into(), "1".into()]);
        code(run(&args, true), 64, "ERR-USAGE");
        assert_eq!(fs::read_dir(f.dir.join("exchange")).unwrap().count(), 0);
    }
    #[test]
    fn output_aliases_and_protocol_directory_outputs_are_refused_without_writing() {
        let f = Fixture::new();
        for path in [
            f.dir.join("claims.tcl"),
            f.dir.join("weights.twr"),
            f.dir.join("exchange"),
            f.dir.join("exchange/01-Round1.tce"),
        ] {
            code(
                run(&f.set("--out", path.display().to_string()), true),
                64,
                "ERR-USAGE",
            );
            assert_eq!(fs::read(f.dir.join("claims.tcl")).unwrap(), f.claims);
            assert_eq!(fs::read_dir(f.dir.join("exchange")).unwrap().count(), 0);
        }
    }
    #[test]
    fn zero_time_and_memory_limits_cannot_publish_a_challenge() {
        let f = Fixture::new();
        for (key, reason) in [
            ("--timeout-ms", "ERR-RES-TIME"),
            ("--max-memory-bytes", "ERR-RES-MEMORY"),
        ] {
            let mut args = f.args.clone();
            args.extend([key.into(), "0".into()]);
            code(run(&args, true), 20, reason);
            assert_eq!(fs::read_dir(f.dir.join("exchange")).unwrap().count(), 0);
        }
    }
    #[test]
    fn lengths_are_exact_and_oversized_pinned_requests_never_start() {
        let f = Fixture::new();
        for (key, value) in [
            ("--prompt-len", "0"),
            ("--output-len", "0"),
            ("--prompt-len", "4294967295"),
            ("--output-len", "4294967296"),
            ("--request-id", "18446744073709551616"),
        ] {
            assert!(matches!(
                Config::parse(&f.set(key, value.into())),
                Err(ErrorCode::Usage)
            ));
        }
        let claims = ClaimList::open(&f.claims, &DecodeLimits::MESSAGE).unwrap();
        let mut args = f.set("--output-len", claims.arch.max_context.to_string());
        let c = Config::parse(&args).unwrap();
        let expected = PublicStatement {
            claim_list_digest: claims.digest(),
            weights_root: c.root,
            weights_epoch: 7,
            manifest_digest: c.manifest,
            batch: BatchDescriptor {
                requests: vec![c.request.clone()],
            },
            predictor: PredictorSpec::Exact {
                decoding: Decoding::Greedy,
            },
        };
        let pin: String = expected
            .digest()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let i = args.iter().position(|a| a == "--statement-digest").unwrap();
        args[i + 1] = pin;
        args.extend(["--timeout-ms".into(), "1".into()]);
        code(run(&args, true), 64, "ERR-USAGE");
        assert_eq!(fs::read_dir(f.dir.join("exchange")).unwrap().count(), 0);
    }
}
