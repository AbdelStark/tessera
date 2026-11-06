# Verifier campaign for #185

Status: run 2026-11-06 on `20df783`: 24.004 clean CPU-hours, every target at its budget,
no failure ([result](#result-2026-11-06)). #185's other gates (#182, second-maintainer review) are open.

The campaign uses the commit adding this plan, the pinned `nightly-2026-01-15`,
`cargo-fuzz` 0.13.2, AddressSanitizer and all nine target binaries. `campaign.py` records
the actual source revision, binary SHA-256 values and every initial seed digest.
No source change is allowed while it runs. Public conformance and producer-integration
fixtures are distinct from independent mathematical oracles; test-small certifies nothing.

**Amendment, 2026-11-06 (toolchain).** On macOS 26 the pinned `nightly-2026-01-15`'s
AddressSanitizer runtime deadlocks before `main` in every target: shadow-memory setup calls
`get_dyld_hdr`, whose dyld iteration allocates and re-enters `AsanInitFromRtl` on its own spin
lock. Workers print nothing and the controller kills them as overdue, so no input runs and no
CPU time is credited. The pin moves to `nightly-2026-11-06` (rustc `249c1e4e3`), whose runtime
starts on this platform; the sanitizer, targets, oracles, budgets and limits are unchanged. The
campaign restarts from the commit carrying this amendment, and no earlier run is credited.

## Fixed completion criterion

Each of decode, envelope, protocol, multiproof, inclusion, sumcheck and audit must
accumulate at least two measured process CPU-hours; claims at least four and verify
at least six. The total must therefore be at least 24 CPU-hours with no panic,
sanitizer failure, resource-limit failure or accepted known-false mutation. Requested
wall durations, compilation, controller time and interrupted/failed workers do not
earn clean CPU-hour credit. `wait4` user plus system CPU usage is the accounting source.
This is finite fuzz evidence, not a proof that every input is safe or every mutation false.

At most eight workers run concurrently. Each worker chunk has at most 600 seconds
requested wall time, a 10-second per-input limit and 1024 MiB RSS limit. Decode and
envelope inputs cap at 4096 bytes (their prefix oracles have quadratic work), claims
at 65,536 and the other targets at 2 MiB. The new claims/verify binaries additionally
cap Rust allocations to 64 MiB beyond their current live baseline. Policy memory/time
controls and the existing isolated allocator/deadline tests are separate checks.

The claims target exercises payload/envelope parsing and semantic validation under
both protocol selections. Its fixed controls require the exact protocol, architecture,
layout and parameter refusals. The complete NI target uses two immutable public caller
contexts: the recorded driver integration fixture and the tamper corpus base. It runs
both the typed and byte entry points; structured controls demand exact rejection,
and every declared NI tamper case demands its predeclared verdict. Arbitrary raw
acceptance is not automatically a forgery: a different nonce may still be valid.
Accepted raw proofs must instead return the pinned public report fields.

## Reproduction

From `verifier/fuzz/`, after the source and plan are committed:

```sh
cargo test --features test-params --test complete_oracles --locked
python3 seeds.py /tmp/tessera-fuzz-seeds
cargo fuzz build --features test-params --sanitizer address
python3 campaign.py --binaries target/aarch64-apple-darwin/release \
  --seeds /tmp/tessera-fuzz-seeds --out /tmp/tessera-fuzz-campaign
```

Use the actual target triple on another platform. Fuzz builds enable debug assertions;
the test-only feature's production release guard remains unchanged. The runner refuses
existing output directories. All workers stop on a finding/failure; retain its artifact,
command and log, reproduce it, and distinguish a harness/environment failure from a
verifier defect. A failed campaign cannot be reported as a clean 24-hour campaign.

## Preflight

The three complete-oracle tests pass, including all 201 declared NI-only tamper
cases. Four compiled production-check omissions were caught by their own assertions;
each restored control passed ([records](preflight.json)). The standalone fuzz tests,
feature-enabled clippy, `tv-core` tests, formatting, repository lint and isolation gate
pass locally. AddressSanitizer binaries build on the pinned nightly. These checks earn
no campaign CPU-hour credit. An injected worker exit 7 stopped the controller with
nonzero status and zero clean CPU credit. Second-maintainer review has not occurred.

The adversarial review checks the harness boundary: caller inputs are initialized
from original public fixtures before fuzz data, never extracted from the candidate;
known-false controls compare exact reasons, and raw acceptance only pins report
fields (it does not establish independent soundness). The zero-root control preserves
the cross-multiplied identity so the denominator check alone must reject. No production
acceptance, transcript, secret-handling or release-feature logic changes. All fixture
files are runtime data, as in the existing driver allocation tests, and are hashed in
the campaign manifest. The existing malformed-byte, allocation and deadline tests
remain enabled.

## Evidence to commit after completion

The report must include source/toolchain/platform, measured per-target CPU times,
execution/coverage statistics, raw terminal summaries, binary/seed identities, minimized
retained corpus and a faithful list of failures/limitations. Preserve any finding and
its replay command; fixing verifier defects is a separate issue/PR. Until that evidence
exists and the prerequisites/review gates are met, #185 remains open.

## Result, 2026-11-06

| | |
|---|---|
| Source | `main` at `20df7835e921bfe6d04067a0fe5b5b103791bded`, frozen in a worktree for the whole run |
| Toolchain | `rustc 1.101.0-nightly (249c1e4e3 2026-11-05)` (`nightly-2026-11-06`), `cargo-fuzz 0.13.2`, `--sanitizer address`, `--features test-params` |
| Platform | Apple M4 Max (16 cores, 48 GiB), macOS 26.6.2 (25G83), `aarch64-apple-darwin` |
| Window | 2026-11-06 17:30:09 to 23:41:28 UTC, eight workers, 600-second chunks |
| Controller | `python3 -I campaign.py --binaries target/aarch64-apple-darwin/release --seeds SEEDS --out OUT --workers 8`, exit 0, `state: complete` |

CPU is `wait4` user plus system time of chunks that exited 0; every chunk did.

| Target | Chunks | Clean CPU-h | Budget | Executions | Coverage `cov` / `ft` | Peak RSS (MiB) | Corpus (minimized) | Artifacts |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| decode | 13 | 2.001 | 2 | 261,453,634 | 1,362 / 6,097 | 971 | 831 (397) | 0 |
| envelope | 13 | 2.000 | 2 | 387,780,831 | 1,516 / 4,594 | 779 | 1,087 (332) | 0 |
| protocol | 12 | 2.000 | 2 | 522,255,402 | 3,072 / 8,852 | 734 | 2,981 (743) | 0 |
| multiproof | 13 | 2.001 | 2 | 111,557,304 | 1,164 / 5,650 | 748 | 5,326 (846) | 0 |
| inclusion | 13 | 2.000 | 2 | 213,734,232 | 672 / 3,567 | 565 | 4,424 (535) | 0 |
| sumcheck | 13 | 2.001 | 2 | 45,494,202 | 7,897 / 20,863 | 594 | 6,892 (1,537) | 0 |
| audit | 13 | 2.001 | 2 | 186,704,147 | 1,663 / 4,627 | 626 | 805 (304) | 0 |
| claims | 24 | 4.000 | 4 | 5,371,238 | 5,181 / 16,764 | 541 | 2,057 (670) | 0 |
| verify | 36 | 6.001 | 6 | 1,489,713 | 15,347 / 33,278 | 966 | 2,907 (587) | 0 |
| **total** | 150 | **24.004** | 24 | 1,735,840,703 | | | | **0** |

Executions sum `stat::number_of_executed_units` over chunks; coverage is the largest final `cov`/`ft` of any chunk (seeds included); peak RSS is the largest `ru_maxrss`, ASan shadow included.

No panic, sanitizer report, timeout, RSS-limit or allocation-cap failure, and no known-false case accepted: each target asserts its oracles in-process, so any of these ends the chunk nonzero, and none did. The criterion above is met for this campaign.

### Evidence in this directory

- [status.json](status.json): the controller's record (SHA-256 `2cc8a4af…d939`): every chunk's command, seed, exit code, CPU, wall time and RSS, and the manifest digest.
- [identities.json](identities.json): per target, the binary's SHA-256, the seed count and the SHA-256 of its canonical seed map; the runtime fixtures' digests; the manifest's SHA-256 (`e8f939b7…7d3e`, equal to `status.json`'s). The 3.7 MB manifest itself is not committed.
- [summaries.txt](summaries.txt): the final libFuzzer lines (`DONE`, `Done`, `stat::`) of all 150 chunks; [logs.tar.bz2](logs.tar.bz2) the complete logs.
- [corpus-min.tar.bz2](corpus-min.tar.bz2): the retained corpus of the eight smaller targets after `-merge=1` with the campaign's binaries and limits. Seeds are excluded: `seeds.py` regenerates them.
- [verify-corpus.sha256](verify-corpus.sha256): the SHA-256 of each of verify's 587 minimized inputs. They hold 150.6 MB (17.8 MB with bzip2, 2 MB inputs mutated from the two NI fixtures); they are kept out of git, outside this repository, until a maintainer chooses where large corpora live.

Replaying every minimized corpus (`-runs=0`, same limits) through the campaign binaries exits 0 for all nine targets, and verify's inputs match their listed digests.

### Limitations

- Finite fuzz evidence under `test-params` and AddressSanitizer on one platform; not a proof that every input is safe, and `test-small` certifies nothing.
- Raw-acceptance of a mutated `verify` input checks the pinned report fields, not independent soundness (see above).
- Peak RSS reached 971 MiB on decode against the 1,024 MiB limit: ASan's shadow and decode's 13,052 seeds leave 5% headroom. A later run with more seeds may need the limit reviewed rather than a finding treated as a defect.
- The previous run on `nightly-2026-01-15` earned no credit (amendment above); this run is the only one counted.
