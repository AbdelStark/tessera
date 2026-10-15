// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compiles `kernels/*.cu` into fatbins for `sm_90` (H100, H200) and `sm_100` (B200) when the
//! `gpu` feature is on (RFC-0001 'Language boundaries'). Without the feature this does nothing,
//! so every machine builds the crate as a stub. With the feature, a missing or older CUDA
//! toolkit is a build error that names the requirement.
//!
//! `TESSERA_GPU_SKIP_KERNELS=1` skips kernel compilation and builds the host code against empty
//! kernel images. It exists so that CI can type-check and lint the host code on machines without
//! CUDA; such a build cannot load any kernel, and `Device::new` reports it.

// A build script reports through cargo directives on stdout and errors on stderr.
#![allow(clippy::print_stderr)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};

/// The minimum CUDA toolkit version (RFC-0001, 09 §8): the first with `sm_100` support.
const MIN_CUDA: (u32, u32) = (12, 8);
/// Device architectures every fatbin contains.
const ARCHITECTURES: [&str; 2] = ["90", "100"];

fn main() {
    println!("cargo::rerun-if-changed=kernels");
    for var in ["NVCC", "CUDA_HOME", "CUDA_PATH", "TESSERA_GPU_SKIP_KERNELS"] {
        println!("cargo::rerun-if-env-changed={var}");
    }
    println!("cargo::rustc-check-cfg=cfg(tessera_gpu_kernels_skipped)");
    if env::var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let kernels = kernel_sources();
    if env::var("TESSERA_GPU_SKIP_KERNELS").is_ok_and(|v| v == "1") {
        for source in &kernels {
            fs::write(fatbin_path(&out, source), []).expect("write an empty kernel image");
        }
        println!("cargo::rustc-cfg=tessera_gpu_kernels_skipped");
        println!("cargo::warning=TESSERA_GPU_SKIP_KERNELS=1: CUDA kernels were not compiled");
        return;
    }

    let nvcc = find_nvcc().unwrap_or_else(|| {
        fail("no nvcc found (looked at $NVCC, $CUDA_HOME/bin, $CUDA_PATH/bin, /usr/local/cuda/bin and PATH)")
    });
    let version = nvcc_version(&nvcc)
        .unwrap_or_else(|| fail(&format!("could not read the version of {}", nvcc.display())));
    if version < MIN_CUDA {
        fail(&format!(
            "{} is CUDA {}.{}",
            nvcc.display(),
            version.0,
            version.1
        ));
    }
    for source in &kernels {
        compile(&nvcc, source, &fatbin_path(&out, source));
    }
}

/// Every `.cu` file under `kernels/`, sorted so the build is deterministic.
fn kernel_sources() -> Vec<PathBuf> {
    let mut sources: Vec<PathBuf> = fs::read_dir("kernels")
        .expect("kernels/ exists")
        .map(|entry| entry.expect("readable kernels/ entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "cu"))
        .collect();
    sources.sort();
    sources
}

fn fatbin_path(out: &Path, source: &Path) -> PathBuf {
    let stem = source.file_stem().expect("kernel files have a name");
    out.join(stem).with_extension("fatbin")
}

fn find_nvcc() -> Option<PathBuf> {
    if let Some(explicit) = env::var_os("NVCC") {
        return Some(PathBuf::from(explicit));
    }
    let mut candidates: Vec<PathBuf> = ["CUDA_HOME", "CUDA_PATH"]
        .iter()
        .filter_map(env::var_os)
        .map(|root| PathBuf::from(root).join("bin/nvcc"))
        .collect();
    candidates.push(PathBuf::from("/usr/local/cuda/bin/nvcc"));
    if let Some(path) = env::var_os("PATH") {
        candidates.extend(env::split_paths(&path).map(|dir| dir.join("nvcc")));
    }
    candidates.into_iter().find(|candidate| candidate.is_file())
}

/// The `release X.Y` reported by `nvcc --version`.
fn nvcc_version(nvcc: &Path) -> Option<(u32, u32)> {
    let output = Command::new(nvcc).arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let release = text.split("release ").nth(1)?;
    let version = release.split([',', ' ', '\n']).next()?;
    let (major, minor) = version.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

fn compile(nvcc: &Path, source: &Path, output: &Path) {
    let mut cmd = Command::new(nvcc);
    cmd.args(["--fatbin", "-O3", "-std=c++17", "--expt-relaxed-constexpr"]);
    for arch in ARCHITECTURES {
        cmd.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
    }
    cmd.arg("-o").arg(output).arg(source);
    let status = cmd
        .status()
        .unwrap_or_else(|e| fail(&format!("cannot run {}: {e}", nvcc.display())));
    if !status.success() {
        fail(&format!("nvcc failed to compile {}", source.display()));
    }
}

fn fail(detail: &str) -> ! {
    eprintln!(
        "error: the `gpu` feature of tessera-gpu needs the CUDA toolkit {}.{} or later (nvcc): \
         {detail}.\nInstall CUDA {}.{}+ and put nvcc on PATH or set CUDA_HOME, or build without \
         the `gpu` feature. To type-check the host code without CUDA, set \
         TESSERA_GPU_SKIP_KERNELS=1.",
        MIN_CUDA.0, MIN_CUDA.1, MIN_CUDA.0, MIN_CUDA.1
    );
    exit(1);
}
