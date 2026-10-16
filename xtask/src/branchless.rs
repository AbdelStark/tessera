// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cargo xtask check-branchless`: the field's operations on secret values stay branch-free in the
//! machine code x86_64 runs (INV-ARCH-2).
//!
//! The source is branch-free by construction, but on x86_64 LLVM turns selects inside loops into
//! conditional jumps when it judges them predictable (#242): carries of secret values would then
//! decide branches. `tessera-field` marks its selects unpredictable. This gate compiles the
//! crate's `codegen-probe` loops (`src/probe.rs`) for `x86_64-unknown-linux-gnu` with the
//! pinned toolchain and the release profile, and reads each loop's assembly. A loop needs two
//! conditional jumps of its own, its empty-input check and its back-edge; any more is a branch
//! on the data. A toolchain update that changes either count fails here, to be looked at.
//!
//! On x86_64 Linux the target is the host's. Elsewhere it needs its standard library:
//! `rustup target add x86_64-unknown-linux-gnu`.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The target whose code generation the gate reads.
const TARGET: &str = "x86_64-unknown-linux-gnu";

/// The probes of `tessera-field`'s `codegen-probe` feature, each a loop over one operation.
pub(crate) const PROBES: [&str; 13] = [
    "fp_add",
    "fp_sub",
    "fp_mul",
    "fp_neg",
    "fp_mul_small",
    "fp_from_u64",
    "fp_from_i64",
    "fp_to_centered",
    "fp_butterfly",
    "fp2_add",
    "fp2_sub",
    "fp2_mul",
    "fp2_dot",
];

/// The conditional jumps a loop needs of its own: the empty-input check and the back-edge.
pub(crate) const LOOP_JUMPS: usize = 2;

/// Each probe's conditional jumps, by name, in the order they appear in `asm`: every `j…`
/// instruction but `jmp` between a probe's label and the end of its function.
pub(crate) fn jumps(asm: &str) -> BTreeMap<String, Vec<String>> {
    let mut found: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut current: Option<String> = None;
    for line in asm.lines() {
        if let Some(name) = probe_label(line) {
            found.insert(name.clone(), Vec::new());
            current = Some(name);
            continue;
        }
        let trimmed = line.trim();
        if trimmed.starts_with(".cfi_endproc") || trimmed.starts_with(".Lfunc_end") {
            current = None;
            continue;
        }
        let Some(name) = &current else { continue };
        let Some(mnemonic) = trimmed.split_whitespace().next() else {
            continue;
        };
        if line.starts_with(char::is_whitespace)
            && mnemonic.starts_with('j')
            && mnemonic != "jmp"
            && mnemonic.chars().all(|c| c.is_ascii_lowercase())
        {
            found
                .entry(name.clone())
                .or_default()
                .push(mnemonic.to_owned());
        }
    }
    found
}

/// The probe a label line opens: `_ZN13tessera_field5probe<len><name>17h<16 hex>E:` in Rust's
/// legacy mangling, for a name of [`PROBES`].
fn probe_label(line: &str) -> Option<String> {
    let rest = line.strip_prefix("_ZN13tessera_field5probe")?;
    let digits = rest.chars().take_while(char::is_ascii_digit).count();
    let len: usize = rest[..digits].parse().ok()?;
    let name = rest.get(digits..digits + len)?;
    let tail = rest.get(digits + len..)?;
    let hash = tail.strip_prefix("17h")?.strip_suffix("E:")?;
    (hash.len() == 16 && hash.chars().all(|c| c.is_ascii_hexdigit()) && PROBES.contains(&name))
        .then(|| name.to_owned())
}

/// What is wrong with `asm`: a probe missing from it, or a probe with more conditional jumps
/// than its loop needs, named with its jumps.
pub(crate) fn violations(asm: &str) -> Vec<String> {
    let found = jumps(asm);
    let mut problems = Vec::new();
    for probe in PROBES {
        match found.get(probe) {
            None => problems.push(format!("{probe}: not in the assembly")),
            Some(jumps) if jumps.len() > LOOP_JUMPS => problems.push(format!(
                "{probe}: {} conditional jumps where its loop needs {LOOP_JUMPS}: {}",
                jumps.len(),
                jumps.join(" ")
            )),
            Some(_) => {}
        }
    }
    problems
}

/// The newest `tessera_field-*.s` in `dir`.
fn newest_assembly(dir: &Path) -> io::Result<PathBuf> {
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("tessera_field-") && name.ends_with(".s") {
            let modified = entry.metadata()?.modified()?;
            if newest.as_ref().is_none_or(|(time, _)| modified > *time) {
                newest = Some((modified, entry.path()));
            }
        }
    }
    newest
        .map(|(_, path)| path)
        .ok_or_else(|| io::Error::other(format!("no tessera_field assembly in {}", dir.display())))
}

/// Compiles the probes for x86_64 and checks each loop's conditional jumps.
pub(crate) fn run(repo: &Path) -> io::Result<bool> {
    let cargo: OsString = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let target_dir = repo.join("target").join("branchless");
    let output = Command::new(cargo)
        .current_dir(repo)
        .args([
            "rustc",
            "--locked",
            "--release",
            "-p",
            "tessera-field",
            "--lib",
        ])
        .args([
            "--features",
            "codegen-probe",
            "--target",
            TARGET,
            "--target-dir",
        ])
        .arg(&target_dir)
        .args(["--", "--emit", "asm"])
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let hint = match stderr.contains("can't find crate for `core`")
            || stderr.contains("target may not be installed")
        {
            true => format!("\n      install its standard library: rustup target add {TARGET}"),
            false => String::new(),
        };
        println!("FAIL  the probes do not compile for {TARGET}{hint}");
        println!("      {}", stderr.trim().replace('\n', "\n      "));
        return Ok(false);
    }
    let asm_path = newest_assembly(&target_dir.join(TARGET).join("release").join("deps"))?;
    let problems = violations(&fs::read_to_string(&asm_path)?);
    if problems.is_empty() {
        println!(
            "ok    {} field operations compile branch-free in x86_64 loops (INV-ARCH-2)",
            PROBES.len()
        );
        return Ok(true);
    }
    println!("FAIL  field operations branch on the data on x86_64 (INV-ARCH-2, #242)");
    for problem in problems {
        println!("      {problem}");
    }
    println!("      assembly: {}", asm_path.display());
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A probe's function in the assembly's shape: its label, `body`, and the end of it.
    fn function(name: &str, body: &str) -> String {
        format!(
            "\t.type\tsym,@function\n_ZN13tessera_field5probe{}{name}17h0123456789abcdefE:\n\t.cfi_startproc\n{body}\t.cfi_endproc\n",
            name.len()
        )
    }

    /// A loop with only its own two conditional jumps.
    const LOOP: &str = "\ttestq\t%rdx, %rdx\n\tje\t.LBB0_3\n.LBB0_2:\n\tcmovbq\t%rax, %r13\n\taddq\t$8, %rcx\n\tcmpq\t%rcx, %rdx\n\tjne\t.LBB0_2\n.LBB0_3:\n\tjmp\t.LBB0_4\n\tretq\n";

    fn assembly(bodies: &[(&str, &str)]) -> String {
        PROBES
            .iter()
            .map(|&probe| {
                let body = bodies
                    .iter()
                    .find(|(name, _)| *name == probe)
                    .map_or(LOOP, |(_, body)| body);
                function(probe, body)
            })
            .collect()
    }

    #[test]
    fn loops_with_only_their_own_jumps_pass() {
        let asm = assembly(&[]);
        assert_eq!(violations(&asm), Vec::<String>::new());
        let found = jumps(&asm);
        assert_eq!(found.len(), PROBES.len());
        assert_eq!(found["fp2_mul"], ["je", "jne"], "jmp is not conditional");
    }

    /// The code #242 found: the borrow and carry corrections of a product as jumps.
    #[test]
    fn a_jump_on_a_carry_fails() {
        let branchy = "\ttestq\t%rdx, %rdx\n\tje\t.LBB0_3\n.LBB0_2:\n\tsubq\t%r11, %rax\n\tjb\t.LBB0_4\n\txorl\t%esi, %esi\n.LBB0_4:\n\taddq\t%rsi, %rdx\n\tjae\t.LBB0_6\n.LBB0_6:\n\tcmpq\t%rcx, %rdx\n\tjne\t.LBB0_2\n.LBB0_3:\n\tretq\n";
        let problems = violations(&assembly(&[("fp2_mul", branchy)]));
        assert_eq!(
            problems,
            ["fp2_mul: 4 conditional jumps where its loop needs 2: je jb jae jne"]
        );
    }

    #[test]
    fn a_missing_probe_fails() {
        let asm: String = PROBES
            .iter()
            .filter(|&&probe| probe != "fp_neg")
            .map(|&probe| function(probe, LOOP))
            .collect();
        assert_eq!(violations(&asm), ["fp_neg: not in the assembly"]);
    }

    /// Labels of other functions, of mangled names that are not probes, and jumps outside any
    /// probe count for nothing.
    #[test]
    fn only_probe_functions_are_read() {
        let mut asm = assembly(&[]);
        asm.push_str("_ZN13tessera_field5probe6fp_add17hnothexadecimalE:\n\tjb\t.L1\n");
        asm.push_str("_ZN13tessera_field4ext23mul17h0123456789abcdefE:\n\tjb\t.L1\n\tjae\t.L2\n");
        asm.push_str("\tjb\t.L3\n");
        assert_eq!(violations(&asm), Vec::<String>::new());
        assert_eq!(
            probe_label("_ZN13tessera_field5probe7fp2_mul17h0123456789abcdefE:").as_deref(),
            Some("fp2_mul")
        );
        assert_eq!(
            probe_label("_ZN13tessera_field5probe7fp3_mul17h0123456789abcdefE:"),
            None
        );
    }
}
