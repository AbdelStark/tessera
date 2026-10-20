# RFC-0051: Soundness Target Exit Code

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-20
- Target milestone: v0.3
- Supersedes, on acceptance:
  - [02 §2.4](../spec/02-public-api.md#24-exit-codes) 'Exit codes': exit code 10 when `tessera params check` or `tessera-verify params` reports a missed soundness target;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-calculator) 'Soundness calculator': the exit status of `tessera params check` and `tessera-verify params`.
- Origin: [#375](https://github.com/AbdelStark/tessera/issues/375)

## Summary

`tessera params check` and `tessera-verify params` compute and print a parameter set's soundness report ([RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-calculator)). Previously, [02 §2.4](../spec/02-public-api.md#24-exit-codes) defined no exit code for a missed soundness target, so both commands exited 0 even when `meets_targets` was `false`. This RFC assigns exit code 10 when any row of the soundness report misses its target (`meets_targets == Some(false)`), while still writing the full report to stdout. Parameter sets without targets (`test-small`, `meets_targets == None`) and reports where every row meets its target (`meets_targets == Some(true)`) continue to exit 0.

## Motivation

- **CI and script gating.** [RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-calculator) requires CI to run both `tessera params check` and `tessera-verify params` and fail if any bound misses its target. When a missed target exits 0, scripts and CI jobs must parse stdout instead of checking the process exit status.
- **Tier-A contract.** Exit codes are tier A ([02 §1](../spec/02-public-api.md#1-stability-tiers)), and row 10 of [02 §2.4](../spec/02-public-api.md#24-exit-codes) is already the non-error rejection code (`Reject` on `tessera-verify`, `Not used` on `tessera`).
- **A missed target is a negative verdict, not an `ERR-*` error.** Under [02 §2](../spec/02-public-api.md#2-command-line-interface), an `ERR-*` failure with `--json` prints `{"command", "error": {"code", "message"}}` in place of the command's output. When a soundness target is missed, the caller still needs the soundness report itself on stdout (showing which rows have `pass: false` and their margins), paired with a non-zero exit status.

## Goals

- Give `tessera params check` and `tessera-verify params` a distinct non-zero exit code (10) when any soundness row misses its target, without suppressing the report on stdout.
- Keep exit 0 when every row meets its target (`live-64` and `ni-100` at the default 405B statement) or when the parameter set has no targets (`test-small`).

## Non-Goals

- Changing the `tessera.soundness/1` JSON schema or the human-readable table format.
- Adding an `ERR-*` error code for a missed soundness target.

## Proposed Design

### Exit code 10 in 02 §2.4

Row 10 of [02 §2.4](../spec/02-public-api.md#24-exit-codes) is amended to:

| Code | `tessera-verify` meaning | `tessera` meaning |
|---|---|---|
| 10 | Reject: the proof is invalid, `Û` exceeds the bound, or a soundness target is missed | Soundness target missed (`tessera params check`) |

### `tessera params check` and `tessera-verify params`

After computing `SoundnessReport` and writing the human-readable table or `tessera.soundness/1` JSON object to stdout:

- If `report.meets_targets()` is `Some(true)` or `None` (`test-small`), the command writes nothing to stderr and exits 0.
- If `report.meets_targets()` is `Some(false)` (at least one row has `pass == Some(false)`), the command writes nothing to stderr and exits 10.
- Malformed arguments or degenerate statement sizes continue to fail before printing a report and exit 64 with `ERR-USAGE`.
- An I/O failure when writing to stdout continues to exit 20 with `ERR-RES-IO`.

## Alternatives Considered

1. **Raise `ERR-USAGE` (exit 64) or a new `ERR-*` code when a target is missed.** Rejected: under [02 §2](../spec/02-public-api.md#2-command-line-interface), an `ERR-*` failure replaces `--json` stdout with an error envelope `{"command", "error": ...}` and writes `error: ERR-...` to stderr. A soundness report whose row misses its target is a valid evaluation whose report must still be emitted on stdout.
2. **Introduce a new exit code (such as 11).** Rejected: [02 §2.4](../spec/02-public-api.md#24-exit-codes) already reserves 10 for a completed check that rejects the statement or parameters, and `tessera` had no other use for 10.

## Drawbacks

- Callers that invoke `tessera params check` or `tessera-verify params` on intentionally undersized parameters must expect exit status 10.

## Migration / Rollout

v0.3. Updates `tessera-cli` and `tv-cli` in lockstep with [02 §2.4](../spec/02-public-api.md#24-exit-codes) and [RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-calculator).

## Testing Strategy

- `tessera params check --params ni-100 --lookups 1125899906842624` (and `--lookups 1125899902648320`, where `N + T = 2^50`) exits 10 and still prints the full report to stdout (both table and `--json`), with empty stderr.
- `tessera-verify params --params ni-100 --lookups 1125899906842624` (and `--lookups 1125899902648320`) exits 10 and still prints the full report to stdout (both table and `--json`), with empty stderr.
- `live-64`, `ni-100` and `test-small` at the default 405B statement exit 0 on both CLIs.

## Open Questions

None.

## References

- [02 §2.4](../spec/02-public-api.md#24-exit-codes), [04 §1](../spec/04-error-model.md#1-principles), [RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-calculator).
- [#375](https://github.com/AbdelStark/tessera/issues/375), [#103](https://github.com/AbdelStark/tessera/issues/103), [#110](https://github.com/AbdelStark/tessera/issues/110).
