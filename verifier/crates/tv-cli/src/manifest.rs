// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's run manifest, `tessera.run-manifest/1` (05 §6 'The verifier's manifest',
//! `spec/schemas/run-manifest-v1.schema.json`), which `tessera-verify verify --run-manifest PATH`
//! writes after its report. It is public-class: the verifier computes only on public and protocol
//! data, so its timings depend on nothing secret, and it names every artifact by its envelope
//! digest. Written by hand, as every JSON document of this crate, so that no crate joins the
//! trusted base.
//!
//! The build is described with what the compiler gives a crate that has no build script, which
//! `cargo xtask check-isolation` refuses in the verifier (build-time code can read any file,
//! RFC-0001 'Enforcement'). The commit and the compiler's version are the build-time variables
//! `TESSERA_GIT_COMMIT` (40 hexadecimal digits) and `TESSERA_RUSTC_VERSION`, which a release
//! build sets (`docs/release/checklist.md`), else null. The profile is `release` when debug assertions are off, the test-only
//! features' own criterion, and the features are the test features the build enables,
//! `tv-core/test-params` when it accepts test parameters (RFC-0001 'Cargo features', OBS-5).

use crate::escape_json;
use std::fmt::Write;
use std::time::{Duration, SystemTime};
use tv_core::observe::CheckEvent;

/// An artifact as the manifest names it: its path, its schema (03 §5's name, or a JSON report's
/// identifier), its class, and its envelope digest once its envelope verifies.
pub(crate) struct Artifact<'a> {
    pub(crate) path: &'a str,
    pub(crate) schema: &'static str,
    pub(crate) class: &'static str,
    pub(crate) digest: Option<[u8; 32]>,
}

/// One `verify` run, as its manifest records it.
pub(crate) struct Run<'a> {
    /// The command line after the program's name.
    pub(crate) arguments: &'a [String],
    pub(crate) started_at: SystemTime,
    /// The claim list's parameter set once it has decoded.
    pub(crate) params: Option<&'a str>,
    /// The claim list, the weight commitment and the proof.
    pub(crate) inputs: &'a [Artifact<'a>],
    /// The report at `--out`, once written.
    pub(crate) outputs: &'a [Artifact<'a>],
    /// Every check that ran, as the log's `check_completed` lines report them.
    pub(crate) checks: &'a [CheckEvent],
    pub(crate) elapsed: Duration,
    pub(crate) exit_code: i32,
    /// The verdict's `REJ-…` or `ERR-…` code; none on accept.
    pub(crate) error_code: Option<&'a str>,
}

fn quoted(text: &str, out: &mut String) {
    out.push('"');
    out.push_str(&escape_json(text));
    out.push('"');
}

fn text_or_null(text: Option<&str>, out: &mut String) {
    match text {
        Some(text) => quoted(text, out),
        None => out.push_str("null"),
    }
}

/// A duration in seconds, as an exact decimal with nine places.
fn seconds(d: Duration, out: &mut String) {
    let _ = write!(out, "{}.{:09}", d.as_secs(), d.subsec_nanos());
}

/// The commit the build was told, `TESSERA_GIT_COMMIT` when it was compiled: 40 hexadecimal
/// digits, in lowercase, else none.
fn git_commit() -> Option<String> {
    option_env!("TESSERA_GIT_COMMIT")
        .filter(|c| c.len() == 40 && c.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(str::to_ascii_lowercase)
}

/// The compiler's version the build was told, `TESSERA_RUSTC_VERSION` when it was compiled, as
/// `rustc --version` prints it; none when unset or empty.
fn rustc() -> Option<&'static str> {
    option_env!("TESSERA_RUSTC_VERSION").filter(|v| !v.is_empty())
}

/// The test features the build enables: `tv-cli` declares none of its own, and `tv-core`'s
/// `test-params` accepts test parameters, which certify nothing.
fn features() -> &'static [&'static str] {
    if tv_core::TEST_PARAMS {
        &["tv-core/test-params"]
    } else {
        &[]
    }
}

fn artifacts(list: &[Artifact<'_>], out: &mut String) {
    out.push('[');
    for (i, a) in list.iter().enumerate() {
        if i != 0 {
            out.push(',');
        }
        out.push_str("{\"path\":");
        quoted(a.path, out);
        out.push_str(",\"schema\":");
        quoted(a.schema, out);
        out.push_str(",\"class\":");
        quoted(a.class, out);
        out.push_str(",\"digest\":");
        match &a.digest {
            Some(d) => {
                out.push('"');
                for b in d {
                    let _ = write!(out, "{b:02x}");
                }
                out.push('"');
            }
            None => out.push_str("null"),
        }
        out.push('}');
    }
    out.push(']');
}

/// The manifest's text, one JSON object and a newline.
pub(crate) fn render(run: &Run<'_>) -> String {
    let mut out = String::new();
    out.push_str(
        "{\"schema\":\"tessera.run-manifest/1\",\"tool\":{\"name\":\"tessera-verify\",\"version\":",
    );
    quoted(env!("CARGO_PKG_VERSION"), &mut out);
    out.push_str(",\"git_commit\":");
    text_or_null(git_commit().as_deref(), &mut out);
    out.push_str(",\"profile\":");
    quoted(
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        &mut out,
    );
    out.push_str(",\"features\":[");
    for (i, feature) in features().iter().enumerate() {
        if i != 0 {
            out.push(',');
        }
        quoted(feature, &mut out);
    }
    out.push_str("]},\"toolchain\":{\"rustc\":");
    text_or_null(rustc(), &mut out);
    out.push_str(
        ",\"cuda_runtime\":null,\"cuda_driver\":null,\"python\":null,\"torch\":null},\"host\":{\"os\":",
    );
    quoted(std::env::consts::OS, &mut out);
    out.push_str(",\"arch\":");
    quoted(std::env::consts::ARCH, &mut out);
    out.push_str(",\"cpu_model\":");
    text_or_null(host::cpu_model().as_deref(), &mut out);
    out.push_str(",\"cores\":");
    match std::thread::available_parallelism() {
        Ok(n) => {
            let _ = write!(out, "{n}");
        }
        Err(_) => out.push_str("null"),
    }
    out.push_str(",\"memory_bytes\":");
    match host::memory_bytes() {
        Some(n) => {
            let _ = write!(out, "{n}");
        }
        None => out.push_str("null"),
    }
    out.push_str(",\"gpus\":[],\"ecc\":null},\"command\":\"verify\",\"arguments\":[");
    for (i, arg) in run.arguments.iter().enumerate() {
        if i != 0 {
            out.push(',');
        }
        quoted(arg, &mut out);
    }
    out.push_str("],\"params\":");
    text_or_null(run.params, &mut out);
    out.push_str(",\"protocol\":\"tessera-protocol/1\",\"inputs\":");
    artifacts(run.inputs, &mut out);
    out.push_str(",\"outputs\":");
    artifacts(run.outputs, &mut out);
    out.push_str(",\"public_seeds\":[],\"started_at\":\"");
    crate::log::timestamp(run.started_at, &mut out);
    out.push_str("\",\"elapsed_seconds\":");
    seconds(run.elapsed, &mut out);
    out.push_str(",\"metrics\":[");
    for (i, check) in run.checks.iter().enumerate() {
        if i != 0 {
            out.push(',');
        }
        // One `tessera_verifier_seconds` sample per check that ran (05 §4).
        out.push_str("{\"name\":\"tessera_verifier_seconds\",\"labels\":{\"check\":");
        quoted(check.kind.name(), &mut out);
        out.push_str("},\"type\":\"histogram\",\"count\":1");
        for key in ["sum", "min", "max"] {
            let _ = write!(out, ",\"{key}\":");
            seconds(check.duration, &mut out);
        }
        out.push('}');
    }
    let _ = write!(out, "],\"exit_code\":{},\"error_code\":", run.exit_code);
    text_or_null(run.error_code, &mut out);
    out.push_str("}\n");
    out
}

/// What the platform reports of the host without a crate: Linux's `/proc`, macOS's `sysctl`;
/// none elsewhere or when it does not answer. Each answer is read up to [`host::CAP`] bytes, so
/// the probe allocates a few KiB whatever the platform writes.
mod host {
    use std::io::Read;

    /// The most bytes read from one file or one child's output.
    pub(super) const CAP: u64 = 4096;

    /// The complete lines among the first `cap` bytes of `reader`: a longer input loses the line
    /// the cap cuts. `None` when they are not UTF-8.
    #[cfg_attr(not(any(test, target_os = "linux")), allow(dead_code))]
    pub(super) fn complete_lines(reader: impl Read, cap: u64) -> Option<String> {
        let mut bytes = Vec::new();
        reader.take(cap).read_to_end(&mut bytes).ok()?;
        if u64::try_from(bytes.len()).ok()? == cap {
            let end = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
            bytes.truncate(end);
        }
        String::from_utf8(bytes).ok()
    }

    /// All of `reader`, trimmed, when it holds at most `cap` bytes, and is UTF-8 and not empty.
    #[cfg_attr(not(any(test, target_os = "macos")), allow(dead_code))]
    pub(super) fn whole(reader: impl Read, cap: u64) -> Option<String> {
        let mut bytes = Vec::new();
        reader
            .take(cap.checked_add(1)?)
            .read_to_end(&mut bytes)
            .ok()?;
        if u64::try_from(bytes.len()).ok()? > cap {
            return None;
        }
        let text = String::from_utf8(bytes).ok()?;
        let text = text.trim();
        (!text.is_empty()).then(|| text.to_string())
    }

    #[cfg(target_os = "linux")]
    fn proc(file: &str) -> Option<String> {
        complete_lines(std::fs::File::open(file).ok()?, CAP)
    }

    /// `sysctl -n name`, from its absolute path, its stdout read up to the cap; the pipe is then
    /// closed, so a child that writes more stops on it.
    #[cfg(target_os = "macos")]
    fn sysctl(name: &str) -> Option<String> {
        use std::process::{Command, Stdio};
        let mut child = Command::new("/usr/sbin/sysctl")
            .args(["-n", name])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let text = child.stdout.take().and_then(|out| whole(out, CAP));
        let status = child.wait().ok()?;
        text.filter(|_| status.success())
    }

    /// The first `model name` of `/proc/cpuinfo`, or `machdep.cpu.brand_string`.
    pub(super) fn cpu_model() -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            proc("/proc/cpuinfo")?.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                let value = value.trim();
                (key.trim() == "model name" && !value.is_empty()).then(|| value.to_string())
            })
        }
        #[cfg(target_os = "macos")]
        {
            sysctl("machdep.cpu.brand_string")
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            None
        }
    }

    /// `MemTotal` of `/proc/meminfo`, which counts kibibytes, or `hw.memsize`.
    pub(super) fn memory_bytes() -> Option<u64> {
        #[cfg(target_os = "linux")]
        {
            proc("/proc/meminfo")?.lines().find_map(|line| {
                let kib = line.strip_prefix("MemTotal:")?.trim().strip_suffix("kB")?;
                kib.trim().parse::<u64>().ok()?.checked_mul(1024)
            })
        }
        #[cfg(target_os = "macos")]
        {
            sysctl("hw.memsize")?.parse().ok()
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            None
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Counts what is read from it, so the cap is seen to bound the reading itself.
        struct Counted<R>(R, u64);
        impl<R: Read> Read for Counted<R> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let n = self.0.read(buf)?;
                self.1 += n as u64;
                Ok(n)
            }
        }

        #[test]
        fn an_endless_answer_is_read_up_to_the_cap() {
            let mut endless = Counted(std::io::repeat(b'7'), 0);
            assert_eq!(whole(&mut endless, CAP), None);
            assert_eq!(endless.1, CAP + 1);
            let mut endless = Counted(std::io::repeat(b'7'), 0);
            assert_eq!(complete_lines(&mut endless, CAP).as_deref(), Some(""));
            assert_eq!(endless.1, CAP);
            assert_eq!(
                whole(&b" 51539607552\n"[..], CAP).as_deref(),
                Some("51539607552")
            );
            assert_eq!(whole(&b"\n"[..], CAP), None);
            assert_eq!(whole(&[0xff][..], CAP), None);
            let exact = "8".repeat(4096);
            assert_eq!(
                whole(exact.as_bytes(), CAP).as_deref(),
                Some(exact.as_str())
            );
        }

        #[test]
        fn a_long_file_keeps_its_complete_leading_lines() {
            // A many-core /proc/cpuinfo: its first block's lines, never the one the cap cuts.
            let block = "processor\t: 0\nmodel name\t: Some CPU @ 3.00GHz\nflags\t\t: fpu vme\n";
            let file = block.repeat(1000);
            let mut read = Counted(file.as_bytes(), 0);
            let lines = complete_lines(&mut read, CAP).unwrap();
            assert_eq!(read.1, CAP);
            assert!(lines.len() <= 4096 && lines.ends_with('\n') && file.starts_with(&lines));
            assert!(
                lines.lines().all(|l| block.lines().any(|b| b == l)),
                "{lines}"
            );
            assert_eq!(
                complete_lines(block.as_bytes(), CAP).as_deref(),
                Some(block)
            );
        }
    }
}
