# Verifier campaign for #185

Status: planned; no CPU-hour result or release qualification is claimed yet.

The campaign uses the commit adding this plan, the pinned `nightly-2026-01-15`,
`cargo-fuzz` 0.13.2, AddressSanitizer and all nine target binaries. `campaign.py` records
the actual source revision, binary SHA-256 values and every initial seed digest.
No source change is allowed while it runs. Public conformance and producer-integration
fixtures are distinct from independent mathematical oracles; test-small certifies nothing.

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
