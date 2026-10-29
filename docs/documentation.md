# Building the documentation artifact

The [documentation workflow](../.github/workflows/docs.yml) builds one private
artifact from the specification, RFCs, guides, JSON-schema reference and both
Rust workspaces. It runs on pull requests, `main` pushes and manual dispatch.
It uploads the site without deploying it or enabling hosting (#176).

## Local build

Install [mdBook 0.5.4](https://github.com/rust-lang/mdBook/releases/tag/v0.5.4),
the version pinned by SHA-256 in CI. Use the repository's pinned Rust toolchain.
From the repository root, with a fresh `.docs-build` directory:

```sh
python3 .github/scripts/docs.py prepare
mdbook build .docs-build
export CARGO_TARGET_DIR="$PWD/.docs-build/target/prover"
RUSTDOCFLAGS='-D warnings' cargo doc -p tracing --locked
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked
cargo test --doc -p tessera-prover --locked
export CARGO_TARGET_DIR="$PWD/.docs-build/target/verifier"
(cd verifier && RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked)
(cd verifier && cargo test --doc -p tv-core --locked)
unset CARGO_TARGET_DIR
python3 .github/scripts/docs.py assemble
python3 -m http.server 8000 --bind 127.0.0.1 --directory .docs-build/site
```

Open `http://127.0.0.1:8000/`. Alternatively, download the `documentation`
artifact from a successful Actions run, extract it and serve that directory.
The artifact needs no Rust installation to browse. Rustdoc's crate and module
navigation and mdBook's chapter navigation/search are both included.

`prepare` refuses an existing work directory. For another build, move the old
`.docs-build` aside or choose a fresh path with `--work-dir PATH`, pass that
path to `mdbook build`, and put both Cargo target directories under it. Do not
reuse another branch's rustdoc directory: Cargo can leave obsolete pages behind.
The [mdBook configuration](book.toml) is copied into the staging directory.

## Sources and scope

The staging script reads Git's index and copies canonical Markdown into the
same relative paths. It generates navigation and directory landing pages, with
separate sections for the specification, RFCs, guides and API reference. Draft
RFC status is preserved verbatim. Add new Markdown under `docs/` and stage it
with Git; it joins the book automatically. Generated output is gitignored.
Untracked files, local credentials and symlink chapters are never included.

The API reference links `tessera-prover` and `tv-core` as tier-B Rust interfaces
([02 §1](spec/02-public-api.md#1-stability-tiers)), and links the remaining crate
docs as internal APIs. Both workspaces build independently in fresh target
directories. Their root API examples compile as doctests; examples requiring
caller-owned model/proof inputs are explicitly `no_run`. Other crate doctests
remain in the quality workflow.

Links to tracked non-Markdown resources copy those resources into the artifact.
Research and agent material outside the book links to the exact source commit
on GitHub, where private-repository access is still required. The assembly step
resolves explicit cross-crate Rust paths against the generated module indexes,
including public aliases. Associated items resolve only to existing, uniquely
matching anchors on their owner page, following redirects. Unknown or ambiguous
references remain errors. Inherited `tracing` links resolve to generated local
pages;
its unavailable `instrument` macro and `valuable` feature references point to
pinned upstream docs. Build tracing and its dependencies first so inherited links
resolve against their actual module documentation. Unrecognized URL schemes,
including unresolved Rust paths such as `crate::Span`, fail validation. Hidden-trait method links resolve to the implementation's real
anchor. Rustdoc may request an implementor registry without emitting one for an
empty trait: assembly supplies an empty registry only when the page declares
no implementations. A missing registry for a nonempty trait remains an error.
Every local repaired target and anchor is checked. These repairs preserve the
generated API content and do not suppress link-check failures.

The upload includes hidden paths because linked workflow sources live under
`.github/`. Only the checked site tree is uploaded; neither staging sources nor
Cargo target directories are included. CI downloads and checks the extracted
artifact again, so packaging omissions fail the job too.

## Link checks and failure controls

Assembly checks every local HTML `href` and `src`, including images, API links,
percent-encoded fragments and rustdoc redirects. Rustdoc's source line-range
links require every selected line and its source-highlighting script. A missing
chapter cannot fall back to a GitHub link. Unknown resources, escaping paths,
redirect cycles, missing files and missing anchors fail the build with their
source page and target. External URLs are preserved and are not fetched: private
GitHub authentication and transient network responses must not decide whether a
local artifact is internally complete.

Run the gate's adversarial tests or recheck an assembled site with:

```sh
python3 -m unittest discover -s .github/scripts
python3 .github/scripts/docs.py check
```

The tests plant missing pages, anchors and images, invalid source ranges,
redirect cycles and escaped paths. They also exercise the tracked-file boundary,
untracked credential refusal, symlink refusal and missing API/chapter failures.
Successful docs or doctests do not establish a protocol security claim; release
reviews and the [release checklist](release/checklist.md) still apply.
