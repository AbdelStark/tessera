# Nightly runs

[07 §2](../spec/07-testing-strategy.md#2-test-pyramid) schedules work too slow for every pull request: deep property tests, an hour of fuzzing and the 8B CPU end-to-end subset. The nightly also runs the gates that [07 §9](../spec/07-testing-strategy.md#9-ci-gates) defers from pull requests during the fast-build phase. [`.github/workflows/nightly.yml`](../../.github/workflows/nightly.yml) runs it every night at 03:17 UTC, on `main`, and on demand from the Actions tab.

## The jobs

| Job | What it runs | Time |
|---|---|---|
| `soundness calibration` | The cheating provers over `F65537` against the analytic rates ([soundness-calibration.md](soundness-calibration.md)) | two steps of at most 10 minutes |
| `property tests at 65,536 cases` | Every test of the crates that use `proptest` (`tessera-field`, `tessera-codec`, `tessera-compiler`, `tessera-protocol`) with `PROPTEST_CASES=65536` | about 9 minutes on an Apple M-series laptop; 60 at most |
| `fuzzing` | [`fuzz.yml`](../../.github/workflows/fuzz.yml): each of the verifier's six targets for ten minutes under AddressSanitizer, an hour in all ([verifier/fuzz/README.md](../../verifier/fuzz/README.md)) | an hour across six runners |
| `deferred quality gates` | [`quality.yml`](../../.github/workflows/quality.yml), whose jobs are listed below this table | about 20 minutes, the prover's full suite the longest |
| `report a failure` | When a scheduled run fails: opens an issue labeled `type:bug` and `area:infra`, titled "ci: the nightly run failed", or comments on the one still open. It holds the run's link and the last 120 lines of each failed job's log, or a line saying why a log could not be read | a minute |

The deferred quality gates are:
- the dependency policy (`cargo xtask check-deps`);
- the MSRV build;
- the macOS build;
- the GPU host checks without CUDA;
- each workspace's complete suite, doctests included, and the verifier's `test-params` suite;
- rustdoc with `RUSTDOCFLAGS=-D warnings`.

Before the nightly ran them, two broken intra-doc links stayed on `main` unseen (#492).

`PROPTEST_CASES` overrides every test's `ProptestConfig::with_cases`, since proptest 1.11 applies the environment to each test's own configuration (`contextualize_config`). A test that sets its cases in code therefore runs the nightly's count too.

## Running it

```sh
# The property tests at the nightly's depth, locally.
PROPTEST_CASES=65536 cargo test --locked -p tessera-field -p tessera-codec -p tessera-compiler -p tessera-protocol
# Every nightly job, on GitHub: the Actions tab, "nightly", "Run workflow", or
gh workflow run nightly.yml
# The failure report, checked alone: every other job is skipped, one fails on purpose, and the
# report opens the issue, or comments on the open one.
gh workflow run nightly.yml -f fail=true
```

A failing fuzz target keeps its input as the `fuzz-<target>-failure` artifact of the run ([verifier/fuzz/README.md](../../verifier/fuzz/README.md) says how to reproduce it).

## Not yet nightly

- **The 8B CPU end-to-end subset and the 8B range-assertion run** of [07 §8](../spec/07-testing-strategy.md#8-ml-specific-tests) need Llama-3.1-8B's integer model, which #93 produces from the gated checkpoint.
- **Hypothesis.** The Python package's property tests set `max_examples` per test, which a profile does not override; they run at that depth in every pull request.
