# Reproducible release builds

[09 §9](../spec/09-release-and-versioning.md#9-release-process) requires two
independent builders to agree before publication. The
[release workflow](../../.github/workflows/release.yml) builds `tessera-verify`
for Linux x86-64 and aarch64 on four separate hosted machines: two native
builders per architecture. It compares the complete binary bytes and their
build records, then writes `SHA256SUMS` and includes the hashes in draft notes.
A missing builder, empty binary, different byte, or different provenance fails
the assembly job. The two architectures have different binaries and hashes.

## Rehearse a release

Run the workflow from the Actions page, or:

```sh
gh workflow run release.yml --ref main -f release_tag=verify-v0.0.1-rc
```

Download the `release-verify-v0.0.1-rc` artifact from that run. It contains both
binaries, per-target build records, `SHA256SUMS`, a committed source archive,
`release.json` and `notes.md`. This special dry-run tag may differ from the
manifest version; the notes record the actual package version. Dispatches and
pull requests create workflow artifacts only: no tags or GitHub releases.
Pull requests changing the release machinery run this same rehearsal.

## Reproduce on a second machine

Use a clean Linux machine with Git, Bash and Docker, on the native architecture
of the binary being checked. Check out the exact commit in `release.json`.
Choose an output directory outside the checkout:

```sh
bash .github/scripts/build_verifier.sh x86_64-unknown-linux-gnu /tmp/tessera-release
```

For a native ARM machine, substitute `aarch64-unknown-linux-gnu`. Compare the
binary's SHA-256 with the corresponding line of the release's `SHA256SUMS`.
Keep the second builder's record and comparison as evidence in the completed
release checklist. Two hosted jobs establish reproduction across distinct
machines; a separately administered builder adds independent operator evidence.

The script refuses a dirty checkout and builds a `git archive HEAD`, excluding
untracked files and local caches. It uses the Rust 1.95.0 Bookworm image pinned
by its multiarchitecture manifest digest in the script. Docker verifies that
digest; the script checks the toolchain against `verifier/rust-toolchain.toml`.
Updating the toolchain requires updating and requalifying the container pin.
The container has an empty Cargo home and target directory, uses `--locked`,
sets `SOURCE_DATE_EPOCH` to the commit time, remaps source/registry/target paths,
and fixes locale, timezone and CPU baseline. No test-only features are enabled.
The run manifest's `TESSERA_GIT_COMMIT` and `TESSERA_RUSTC_VERSION` are embedded.
Network access downloads the lockfile's dependencies; the build does not use a
mutable container tag, host Cargo configuration, or a shared build cache.

## Create a draft

After the freeze and gates in the [release checklist](checklist.md), push a
`vX.Y.Z` or `verify-vX.Y.Z` tag matching its train's workspace package version.
The workflow also accepts matching `-rc` or `-rc.N` versions. A prover tag
produces a source archive and notes from the prover changelog; a verifier tag
additionally requires both independent builds for each architecture. Notes use
the version's changelog section, falling back to `Unreleased` only during rehearsal.
Tag pushes require the version's release section to exist.
Only tag pushes can create GitHub releases, always as drafts. A failed comparison
cannot reach the draft job. No package registries are contacted for publication.

Before publishing the draft, link the completed checklist, confirm the protocol
versions supported, and attach the release-only gate results and required
reviews. Reproducibility does not establish protocol soundness, completeness of
release qualification, or the independence of the compiler and container supply
chain. A rerun against an existing draft refuses to overwrite it; inspect and
update that draft deliberately after reviewing its artifacts.
