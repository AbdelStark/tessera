// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera-verify`'s log lines (05 §2 'The verifier's lines', §5 'The verifier's spans'),
//! written by the CLI itself so that no logging crate joins the trusted base (RFC-0001, DEP-1).
//!
//! - **Options.** `--log-format {human,json}` and `--log-level {error,warn,info,debug,trace}`,
//!   read by [`scan`] before any command's own options, so that a refused option is logged in the
//!   format asked for; an invalid value of either is refused in the human format.
//! - **Events.** `tessera.tv_cli.command_failed` (error, no span), `tessera.tv_cli.check_completed`
//!   (debug, in `verify{proof_digest}` and `check{kind}`) and `tessera.tv_cli.verified` (info, in
//!   `verify`). Their fields are integers or the text of fixed identifiers, codes and digests, and
//!   each message is a literal template whose `{name}` stands for the field `name`.
//! - **Human lines** are the message with its fields substituted, today's lines: no level prefix
//!   at `error` and `warn`, whose messages carry their own, and no unnamed field appended. At
//!   `debug` and `trace` the level and the span path come first. With `--json` the human format
//!   writes no line.
//! - **JSON lines** are `tessera.log-event/1` (`spec/schemas/log-event-v1.schema.json`): `ts` in
//!   UTC to the microsecond, `level`, `target` (this module, which emits every event), `event`,
//!   `span_path` and `fields`, whose `message` is the substituted template. The JSON format writes
//!   every line its level admits, with or without `--json`.

use crate::escape_json;
use std::fmt::Write;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tv_core::observe::{CheckEvent, CheckKind, CheckOutcome, Observer};
use tv_core::verdict::Verdict;

/// Where stderr's lines go: today's human lines, or `tessera.log-event/1` objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    Human,
    Json,
}

/// An event's level, from the least detailed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Level {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Level {
    const ALL: [Self; 5] = [
        Self::Error,
        Self::Warn,
        Self::Info,
        Self::Debug,
        Self::Trace,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

/// The options every command takes: `--json`, `--log-format` and `--log-level` (02 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Flags {
    /// `--json`: stdout carries one JSON result.
    pub(crate) json: bool,
    pub(crate) format: Format,
    /// The most detailed level written.
    pub(crate) level: Level,
}

impl Flags {
    /// `--json` as given, the human format at `info`: what a refused log option is written in.
    pub(crate) const fn human(json: bool) -> Self {
        Self {
            json,
            format: Format::Human,
            level: Level::Info,
        }
    }

    /// Whether a line of `level` is written.
    fn admits(self, level: Level) -> bool {
        level <= self.level && (self.format == Format::Json || !self.json)
    }
}

/// `args` without `--log-format` and `--log-level` and their values, and what they select.
/// `None` stands for an argument too large to keep; as a value it is invalid.
///
/// # Errors
/// The refusal's detail, for a missing or invalid value or an option given twice.
pub(crate) fn scan(
    args: Vec<Option<String>>,
    json: bool,
) -> Result<(Flags, Vec<Option<String>>), &'static str> {
    const FORMAT: &str = "invalid or missing `--log-format` value (expected human or json)";
    const LEVEL: &str =
        "invalid or missing `--log-level` value (expected error, warn, info, debug or trace)";
    let mut format = None;
    let mut level = None;
    let mut rest = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_deref() {
            Some("--log-format") => {
                if format.is_some() {
                    return Err("`--log-format` specified more than once");
                }
                format = Some(match args.next().flatten().as_deref() {
                    Some("human") => Format::Human,
                    Some("json") => Format::Json,
                    _ => return Err(FORMAT),
                });
            }
            Some("--log-level") => {
                if level.is_some() {
                    return Err("`--log-level` specified more than once");
                }
                let value = args.next().flatten();
                level = Some(
                    Level::ALL
                        .into_iter()
                        .find(|l| value.as_deref() == Some(l.name()))
                        .ok_or(LEVEL)?,
                );
            }
            _ => rest.push(arg),
        }
    }
    Ok((
        Flags {
            json,
            format: format.unwrap_or(Format::Human),
            level: level.unwrap_or(Level::Info),
        },
        rest,
    ))
}

/// A field's value: an integer, or the text of a fixed identifier, code or digest.
#[derive(Clone, Copy)]
enum Value<'a> {
    Int(u64),
    Text(&'a str),
}

impl Value<'_> {
    fn human(self, out: &mut String) {
        match self {
            Self::Int(n) => {
                let _ = write!(out, "{n}");
            }
            Self::Text(s) => out.push_str(s),
        }
    }

    fn json(self, out: &mut String) {
        match self {
            Self::Int(n) => {
                let _ = write!(out, "{n}");
            }
            Self::Text(s) => quoted(s, out),
        }
    }
}

type Fields<'a> = [(&'static str, Value<'a>)];

struct Span<'a> {
    name: &'static str,
    fields: &'a Fields<'a>,
}

struct Event<'a> {
    level: Level,
    name: &'static str,
    spans: &'a [Span<'a>],
    fields: &'a Fields<'a>,
    message: &'static str,
}

fn quoted(text: &str, out: &mut String) {
    out.push('"');
    out.push_str(&escape_json(text));
    out.push('"');
}

/// `template` with each `{name}` replaced by the field `name`'s value.
fn substitute(template: &str, fields: &Fields<'_>, out: &mut String) {
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else {
            out.push_str(&rest[open..]);
            return;
        };
        let name = &after[..close];
        match fields.iter().find(|(key, _)| *key == name) {
            Some((_, value)) => value.human(out),
            None => out.push_str(&rest[open..open + close + 2]),
        }
        rest = &after[close + 1..];
    }
    out.push_str(rest);
}

fn fields_json(fields: &Fields<'_>, out: &mut String) {
    for (i, (key, value)) in fields.iter().enumerate() {
        if i != 0 {
            out.push(',');
        }
        quoted(key, out);
        out.push(':');
        value.json(out);
    }
}

fn human(event: &Event<'_>, out: &mut String) {
    if event.level >= Level::Debug {
        out.push_str(event.level.name());
        for (i, span) in event.spans.iter().enumerate() {
            out.push(if i == 0 { ' ' } else { ':' });
            out.push_str(span.name);
            if !span.fields.is_empty() {
                out.push('{');
                for (j, (key, value)) in span.fields.iter().enumerate() {
                    if j != 0 {
                        out.push(' ');
                    }
                    out.push_str(key);
                    out.push('=');
                    value.human(out);
                }
                out.push('}');
            }
        }
        out.push_str(": ");
    }
    substitute(event.message, event.fields, out);
    out.push('\n');
}

fn json(event: &Event<'_>, at: SystemTime, out: &mut String) {
    out.push_str("{\"ts\":\"");
    timestamp(at, out);
    out.push_str("\",\"level\":\"");
    out.push_str(event.level.name());
    out.push_str("\",\"target\":");
    quoted(module_path!(), out);
    out.push_str(",\"event\":");
    quoted(event.name, out);
    out.push_str(",\"span_path\":[");
    for (i, span) in event.spans.iter().enumerate() {
        if i != 0 {
            out.push(',');
        }
        out.push_str("{\"name\":");
        quoted(span.name, out);
        out.push_str(",\"fields\":{");
        fields_json(span.fields, out);
        out.push_str("}}");
    }
    out.push_str("],\"fields\":{");
    fields_json(event.fields, out);
    if !event.fields.is_empty() {
        out.push(',');
    }
    let mut message = String::new();
    substitute(event.message, event.fields, &mut message);
    out.push_str("\"message\":");
    quoted(&message, out);
    out.push_str("}}\n");
}

/// Appends `event`'s line to `out` when `flags` admit it, `at` being when the event happened
/// (05 §2: the lines reach stderr when the command ends, each with the time of its event).
fn emit(flags: Flags, event: &Event<'_>, at: SystemTime, out: &mut String) {
    if !flags.admits(event.level) {
        return;
    }
    match flags.format {
        Format::Human => human(event, out),
        Format::Json => json(event, at, out),
    }
}

/// The last instant RFC 3339's four-digit year can write, 9999-12-31T23:59:59 UTC.
const LAST_SECOND: u64 = 253_402_300_799;

/// `at` in UTC, RFC 3339, to the microsecond: `YYYY-MM-DDTHH:MM:SS.ffffffZ`, 27 characters.
/// A time before 1970 is written as 1970's first instant and one after 9999 as its last.
pub(crate) fn timestamp(at: SystemTime, out: &mut String) {
    let since = at.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
    let (seconds, micros) = if since.as_secs() > LAST_SECOND {
        (LAST_SECOND, 999_999)
    } else {
        (since.as_secs(), since.subsec_micros())
    };
    let (year, month, day) = civil(seconds / 86_400);
    let second = seconds % 86_400;
    let _ = write!(
        out,
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{micros:06}Z",
        second / 3600,
        second % 3600 / 60,
        second % 60
    );
}

/// The proleptic Gregorian date `days` after 1970-01-01, by eras of 400 years (146,097 days),
/// each counted from 1 March so that the leap day ends its year.
fn civil(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468; // days from 0000-03-01
    let era = z / 146_097;
    let day_of_era = z % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153; // 0 for March
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = era * 400 + year_of_era + u64::from(month <= 2);
    (year, month, day)
}

/// `tessera.tv_cli.command_failed`: a command whose options are refused, `error ({code}):
/// {detail}`, with no span. `command` is `tessera-verify` for no command.
pub(crate) fn command_failed(flags: Flags, command: &str, code: &str, detail: &str) -> String {
    let mut out = String::new();
    emit(
        flags,
        &Event {
            level: Level::Error,
            name: "tessera.tv_cli.command_failed",
            spans: &[],
            fields: &[
                ("command", Value::Text(command)),
                ("code", Value::Text(code)),
                ("detail", Value::Text(detail)),
            ],
            message: "error ({code}): {detail}",
        },
        SystemTime::now(),
        &mut out,
    );
    out
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// The `verify` span of `tessera-verify verify` and `live`, from the options' parsing to the
/// verdict, and the checks it saw. It receives tv-core's reports as an [`Observer`] and joins two
/// consecutive reports of one check whose first passed into one check: the CLI's own reading and
/// decoding of the inputs with the driver's limits in step 1, and in a live session a message's
/// decoding with the driver's checks of it. Each check is written once it is complete, with the
/// time its last report arrived: when the check completed.
pub(crate) struct Trail {
    flags: Flags,
    lines: String,
    /// The NI proof's envelope digest, in lowercase hexadecimal, once its envelope verifies.
    proof_digest: Option<String>,
    /// The check whose next report may still join it, and when it completed.
    pending: Option<(CheckEvent, SystemTime)>,
    completed: Vec<CheckEvent>,
}

impl Trail {
    pub(crate) fn new(flags: Flags) -> Self {
        Self {
            flags,
            lines: String::new(),
            proof_digest: None,
            pending: None,
            completed: Vec::new(),
        }
    }

    /// Records the `verify` span's `proof_digest`.
    pub(crate) fn record_proof(&mut self, digest: &[u8; 32]) {
        let mut hex = String::with_capacity(64);
        for b in digest {
            let _ = write!(hex, "{b:02x}");
        }
        self.proof_digest = Some(hex);
    }

    /// Reports a check the CLI itself ran.
    pub(crate) fn check(&mut self, kind: CheckKind, outcome: CheckOutcome, duration: Duration) {
        self.check_completed(CheckEvent {
            kind,
            outcome,
            duration,
        });
    }

    /// Takes `event`, which completed `at`: it joins the pending check when both are one check
    /// and the pending one passed, and completes it again; otherwise the pending check is written
    /// and `event` waits for its own next report.
    fn report(&mut self, event: CheckEvent, at: SystemTime) {
        if let Some((pending, completed)) = &mut self.pending
            && pending.kind == event.kind
            && pending.outcome == CheckOutcome::Pass
        {
            pending.outcome = event.outcome;
            pending.duration = pending.duration.saturating_add(event.duration);
            *completed = at;
            return;
        }
        self.flush();
        self.pending = Some((event, at));
    }

    /// Writes the check still pending; every check is then in [`checks`](Self::checks).
    pub(crate) fn flush(&mut self) {
        let Some((event, at)) = self.pending.take() else {
            return;
        };
        self.completed.push(event);
        let digest = self.proof_digest.as_deref().map(Value::Text);
        let verify = digest.map(|d| [("proof_digest", d)]);
        let spans = [
            Span {
                name: "verify",
                fields: verify.as_ref().map_or(&[], |f| &f[..]),
            },
            Span {
                name: "check",
                fields: &[("kind", Value::Text(event.kind.name()))],
            },
        ];
        emit(
            self.flags,
            &Event {
                level: Level::Debug,
                name: "tessera.tv_cli.check_completed",
                spans: &spans,
                fields: &[
                    ("step", Value::Int(u64::from(event.kind.step()))),
                    ("outcome", Value::Text(event.outcome.name())),
                    ("duration_ns", Value::Int(nanos(event.duration))),
                ],
                message: "step {step} {outcome} in {duration_ns} ns",
            },
            at,
            &mut self.lines,
        );
    }

    /// Every check that ran, joined as written, once [`flush`](Self::flush) has run.
    pub(crate) fn checks(&self) -> &[CheckEvent] {
        &self.completed
    }

    /// `tessera.tv_cli.verified`, which ends `verify` and `live`: the verdict's status, its code
    /// unless it accepts, the exit code and the time since the command started.
    pub(crate) fn verified(&mut self, verdict: &Verdict, exit_code: i32, elapsed: Duration) {
        self.flush();
        let (status, code) = match verdict {
            Verdict::Accept(_) => ("accept", None),
            Verdict::Reject(reason) => ("reject", Some(reason.code())),
            Verdict::Inconclusive(error) => ("inconclusive", Some(error.code.id())),
        };
        let exit = Value::Int(u64::try_from(exit_code).unwrap_or(u64::MAX));
        let elapsed = Value::Int(nanos(elapsed));
        let with_code;
        let without_code;
        let fields: &Fields<'_> = match code {
            Some(code) => {
                with_code = [
                    ("status", Value::Text(status)),
                    ("code", Value::Text(code)),
                    ("exit_code", exit),
                    ("duration_ns", elapsed),
                ];
                &with_code
            }
            None => {
                without_code = [
                    ("status", Value::Text(status)),
                    ("exit_code", exit),
                    ("duration_ns", elapsed),
                ];
                &without_code
            }
        };
        let digest = self.proof_digest.as_deref().map(Value::Text);
        let verify = digest.map(|d| [("proof_digest", d)]);
        emit(
            self.flags,
            &Event {
                level: Level::Info,
                name: "tessera.tv_cli.verified",
                spans: &[Span {
                    name: "verify",
                    fields: verify.as_ref().map_or(&[], |f| &f[..]),
                }],
                fields,
                message: "verify: completed in {duration_ns} ns; protocol/1, anchor/ledger/ZK unchecked",
            },
            SystemTime::now(),
            &mut self.lines,
        );
    }

    /// The lines written, for stderr.
    pub(crate) fn into_lines(self) -> String {
        self.lines
    }
}

impl Observer for Trail {
    /// tv-core reports each check as it completes, so its time is now.
    fn check_completed(&mut self, event: CheckEvent) {
        self.report(event, SystemTime::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(seconds: u64, micros: u32) -> String {
        let mut out = String::new();
        timestamp(
            UNIX_EPOCH + Duration::from_secs(seconds) + Duration::from_micros(micros.into()),
            &mut out,
        );
        out
    }

    /// Instants and their dates from Python's `datetime.fromtimestamp(t, timezone.utc)`, an
    /// oracle independent of this arithmetic.
    #[test]
    fn timestamps_are_utc_rfc3339_to_the_microsecond() {
        for (seconds, micros, text) in [
            (0, 0, "1970-01-01T00:00:00.000000Z"),
            (68_169_599, 1, "1972-02-28T23:59:59.000001Z"),
            (68_169_600, 0, "1972-02-29T00:00:00.000000Z"),
            (951_782_400, 999_999, "2000-02-29T00:00:00.999999Z"),
            (951_868_799, 0, "2000-02-29T23:59:59.000000Z"),
            (4_107_542_400, 0, "2100-03-01T00:00:00.000000Z"),
            (4_107_456_000, 0, "2100-02-28T00:00:00.000000Z"),
            (1_793_016_000, 123_456, "2026-10-26T12:00:00.123456Z"),
            (1_709_164_799, 500_000, "2024-02-28T23:59:59.500000Z"),
            (253_402_300_799, 999_999, "9999-12-31T23:59:59.999999Z"),
        ] {
            assert_eq!(at(seconds, micros), text, "{seconds}");
        }
        assert_eq!(at(u64::MAX / 4, 0), "9999-12-31T23:59:59.999999Z");
        let mut out = String::new();
        timestamp(UNIX_EPOCH - Duration::from_secs(1), &mut out);
        assert_eq!(out, "1970-01-01T00:00:00.000000Z");
        // Every day of 400 years is the day after the one before it.
        let mut previous = civil(0);
        for days in 1..146_097 {
            let (y, m, d) = civil(days);
            let next = if d > 1 {
                (y, m, d) == (previous.0, previous.1, previous.2 + 1)
            } else if m > 1 {
                (y, m) == (previous.0, previous.1 + 1)
            } else {
                (y, m) == (previous.0 + 1, 1) && previous.1 == 12 && previous.2 == 31
            };
            assert!(next, "{days}: {previous:?} then {:?}", (y, m, d));
            let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
            let length = [
                31,
                if leap { 29 } else { 28 },
                31,
                30,
                31,
                30,
                31,
                31,
                30,
                31,
                30,
                31,
            ];
            assert!(d <= length[usize::try_from(m - 1).unwrap()], "{y}-{m}-{d}");
            previous = (y, m, d);
        }
    }

    /// The `ts` of each JSON line, in order.
    fn stamps(lines: &str) -> Vec<String> {
        lines
            .lines()
            .map(|l| {
                let j = tv_testkit::Json::parse(l).unwrap();
                j.get("ts")
                    .and_then(tv_testkit::Json::as_str)
                    .unwrap()
                    .to_string()
            })
            .collect()
    }

    fn stamp(at: SystemTime) -> String {
        let mut out = String::new();
        timestamp(at, &mut out);
        out
    }

    /// A check's line carries the time its report arrived, not the time it is written: the
    /// first check's `ts` precedes the second check's start, and both precede the writing.
    #[test]
    fn a_line_carries_the_time_of_its_event() {
        let pause = || std::thread::sleep(Duration::from_millis(3));
        let mut trail = Trail::new(flags(Format::Json, Level::Debug, false));
        let before_first = SystemTime::now();
        trail.check(
            CheckKind::Input,
            CheckOutcome::Pass,
            Duration::from_nanos(1),
        );
        let after_first = SystemTime::now();
        pause();
        let second_starts = SystemTime::now();
        trail.check(
            CheckKind::Params,
            CheckOutcome::Pass,
            Duration::from_nanos(1),
        );
        let after_second = SystemTime::now();
        pause();
        let written = SystemTime::now();
        trail.flush();
        pause();
        trail.verified(
            &Verdict::Inconclusive(tv_core::VerifierError {
                code: tv_core::ErrorCode::Io,
            }),
            20,
            Duration::ZERO,
        );
        let after_verified = SystemTime::now();
        let ts = stamps(&trail.into_lines());
        assert_eq!(ts.len(), 3);
        assert!(
            stamp(before_first) <= ts[0] && ts[0] <= stamp(after_first),
            "{ts:?}"
        );
        assert!(ts[0] < stamp(second_starts), "{ts:?}");
        assert!(
            stamp(second_starts) <= ts[1] && ts[1] <= stamp(after_second),
            "{ts:?}"
        );
        assert!(ts[1] < stamp(written), "{ts:?}");
        assert!(
            stamp(written) < ts[2] && ts[2] <= stamp(after_verified),
            "{ts:?}"
        );
    }

    /// With controlled clocks: a joined check takes the time of its last report, and writing the
    /// lines later changes no `ts`.
    #[test]
    fn a_joined_check_completes_with_its_last_report() {
        let mut trail = Trail::new(flags(Format::Json, Level::Debug, false));
        let at = |s: u64| UNIX_EPOCH + Duration::from_secs(s);
        let event = |kind, outcome| CheckEvent {
            kind,
            outcome,
            duration: Duration::from_nanos(1),
        };
        // The CLI's step 1, then the driver's, then step 2 and a step 3 that rejects.
        trail.report(event(CheckKind::Input, CheckOutcome::Pass), at(100));
        trail.report(event(CheckKind::Input, CheckOutcome::Pass), at(160));
        trail.report(event(CheckKind::Params, CheckOutcome::Pass), at(220));
        trail.report(event(CheckKind::Claims, CheckOutcome::Reject), at(280));
        trail.flush();
        assert_eq!(
            stamps(&trail.into_lines()),
            [stamp(at(160)), stamp(at(220)), stamp(at(280))]
        );
    }

    /// The reports `live` hands the trail in an accepted session, call by call: the CLI's step
    /// 1, `start`'s step 2, then per message its opening (step 1, twice for Round1 and the
    /// prover's share) and its call (steps 1 to 6, 10 or 13), as tv-core's corpus test proves
    /// each call reports them. The log is 1, 2, 1–6, 1–10, 1–13: one step 1 per message.
    #[test]
    fn a_live_session_writes_one_step_one_per_message() {
        let mut trail = Trail::new(flags(Format::Json, Level::Debug, false));
        let calls: [&[usize]; 9] = [
            &[1],
            &[2],
            &[1],
            &[1],
            &[1, 2, 3, 4, 5, 6],
            &[1],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            &[1],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        ];
        for steps in calls {
            for &step in steps {
                let (outcome, ns) = if step == 5 {
                    (CheckOutcome::NotApplicable, 0)
                } else {
                    (CheckOutcome::Pass, 10)
                };
                trail.check(CheckKind::ALL[step - 1], outcome, Duration::from_nanos(ns));
            }
        }
        trail.flush();
        let written: Vec<(u8, u64)> = trail
            .checks()
            .iter()
            .map(|e| (e.kind.step(), u64::try_from(e.duration.as_nanos()).unwrap()))
            .collect();
        let mut expected = vec![(1, 10), (2, 10)];
        for (last, first) in [(6, 30), (10, 20), (13, 20)] {
            expected.push((1, first));
            expected.extend((2..=last).map(|s| (s, if s == 5 { 0 } else { 10 })));
        }
        assert_eq!(written, expected);
        let lines = trail.into_lines();
        assert_eq!(lines.lines().count(), 31);
        let step5: Vec<&str> = lines
            .lines()
            .filter(|l| l.contains("\"step\":5,"))
            .collect();
        assert_eq!(step5.len(), 3);
        assert!(
            step5.iter().all(|l| l.contains("\"duration_ns\":0,")),
            "{step5:?}"
        );
    }

    #[test]
    fn templates_substitute_named_fields_only() {
        let fields = [("a", Value::Int(7)), ("b", Value::Text("x{y}"))];
        for (template, expected) in [
            ("{a} and {b}", "7 and x{y}"),
            ("{c}", "{c}"),
            ("open { and {a}", "open { and {a}"),
            ("{a}{a}", "77"),
            ("no fields", "no fields"),
            ("unclosed {a", "unclosed {a"),
        ] {
            let mut out = String::new();
            substitute(template, &fields, &mut out);
            assert_eq!(out, expected, "{template}");
        }
    }

    fn flags(format: Format, level: Level, json: bool) -> Flags {
        Flags {
            json,
            format,
            level,
        }
    }

    #[test]
    fn a_level_admits_itself_and_every_less_detailed_one() {
        for (rank, selected) in Level::ALL.into_iter().enumerate() {
            for (other, level) in Level::ALL.into_iter().enumerate() {
                let human = flags(Format::Human, selected, false);
                let json = flags(Format::Json, selected, true);
                assert_eq!(human.admits(level), other <= rank, "{selected:?} {level:?}");
                assert_eq!(json.admits(level), other <= rank, "{selected:?} {level:?}");
                // With --json the human format writes nothing at any level.
                assert!(!flags(Format::Human, selected, true).admits(level));
            }
        }
    }

    #[test]
    fn human_debug_lines_put_the_level_and_span_path_first() {
        let mut trail = Trail::new(flags(Format::Human, Level::Trace, false));
        trail.check(
            CheckKind::Input,
            CheckOutcome::Pass,
            Duration::from_nanos(5),
        );
        trail.record_proof(&[0xab; 32]);
        trail.check(
            CheckKind::Input,
            CheckOutcome::Pass,
            Duration::from_nanos(6),
        );
        trail.check(
            CheckKind::Params,
            CheckOutcome::Reject,
            Duration::from_nanos(7),
        );
        trail.verified(
            &Verdict::Inconclusive(tv_core::VerifierError {
                code: tv_core::ErrorCode::Io,
            }),
            20,
            Duration::from_nanos(99),
        );
        let digest = "ab".repeat(32);
        assert_eq!(
            trail.checks(),
            [
                CheckEvent {
                    kind: CheckKind::Input,
                    outcome: CheckOutcome::Pass,
                    duration: Duration::from_nanos(11)
                },
                CheckEvent {
                    kind: CheckKind::Params,
                    outcome: CheckOutcome::Reject,
                    duration: Duration::from_nanos(7)
                }
            ]
        );
        assert_eq!(
            trail.into_lines(),
            format!(
                "debug verify{{proof_digest={digest}}}:check{{kind=input}}: step 1 pass in 11 ns\n\
                 debug verify{{proof_digest={digest}}}:check{{kind=params}}: step 2 reject in 7 ns\n\
                 verify: completed in 99 ns; protocol/1, anchor/ledger/ZK unchecked\n"
            )
        );
        let mut out = String::new();
        emit(
            flags(Format::Human, Level::Debug, false),
            &Event {
                level: Level::Debug,
                name: "tessera.tv_cli.test",
                spans: &[],
                fields: &[("a", Value::Int(1))],
                message: "a is {a}",
            },
            SystemTime::now(),
            &mut out,
        );
        assert_eq!(out, "debug: a is 1\n");
    }

    #[test]
    fn consecutive_reports_join_only_after_a_pass() {
        let mut trail = Trail::new(flags(Format::Json, Level::Error, false));
        let ns = Duration::from_nanos;
        for (kind, outcome, n) in [
            (CheckKind::Input, CheckOutcome::Pass, 1),
            (CheckKind::Input, CheckOutcome::Pass, 2),
            (CheckKind::Input, CheckOutcome::Reject, 4),
            (CheckKind::Input, CheckOutcome::Pass, 8),
            (CheckKind::Params, CheckOutcome::Pass, 16),
            (CheckKind::Input, CheckOutcome::Inconclusive, 32),
        ] {
            trail.check(kind, outcome, ns(n));
        }
        trail.flush();
        let joined: Vec<_> = trail
            .checks()
            .iter()
            .map(|e| (e.kind, e.outcome, e.duration))
            .collect();
        assert_eq!(
            joined,
            [
                (CheckKind::Input, CheckOutcome::Reject, ns(7)),
                (CheckKind::Input, CheckOutcome::Pass, ns(8)),
                (CheckKind::Params, CheckOutcome::Pass, ns(16)),
                (CheckKind::Input, CheckOutcome::Inconclusive, ns(32)),
            ]
        );
        // At error, no check or verified line is written.
        assert_eq!(trail.into_lines(), "");
    }

    #[test]
    fn json_lines_escape_text_and_end_with_the_message() {
        let mut out = String::new();
        emit(
            flags(Format::Json, Level::Info, true),
            &Event {
                level: Level::Error,
                name: "tessera.tv_cli.command_failed",
                spans: &[],
                fields: &[("detail", Value::Text("a \"b\"\n"))],
                message: "error: {detail}",
            },
            SystemTime::now(),
            &mut out,
        );
        let (ts, rest) = out
            .strip_prefix("{\"ts\":\"")
            .and_then(|r| r.split_at_checked(27))
            .unwrap();
        assert!(ts.ends_with('Z'), "{ts}");
        assert_eq!(
            rest,
            "\",\"level\":\"error\",\"target\":\"tv_cli::log\",\"event\":\"tessera.tv_cli.command_failed\",\"span_path\":[],\"fields\":{\"detail\":\"a \\\"b\\\"\\n\",\"message\":\"error: a \\\"b\\\"\\n\"}}\n"
        );
        assert!(tv_testkit::Json::parse(&out).is_ok(), "{out}");
    }

    #[test]
    fn scan_removes_the_log_options_wherever_they_are() {
        let args = |list: &[&str]| -> Vec<Option<String>> {
            list.iter().map(|s| Some(s.to_string())).collect()
        };
        let (f, rest) = scan(
            args(&[
                "--log-level",
                "trace",
                "verify",
                "--x",
                "1",
                "--log-format",
                "json",
            ]),
            true,
        )
        .unwrap();
        assert_eq!(f, flags(Format::Json, Level::Trace, true));
        assert_eq!(rest, args(&["verify", "--x", "1"]));
        let (f, rest) = scan(args(&["tables"]), false).unwrap();
        assert_eq!(f, Flags::human(false));
        assert_eq!(rest, args(&["tables"]));
        for bad in [
            &["--log-format"][..],
            &["--log-format", "Human"],
            &["--log-level", "info", "--log-level", "info"],
            &["--log-format", "json", "--log-format", "human"],
            &["--log-level", "--log-format", "json"],
        ] {
            assert!(scan(args(bad), false).is_err(), "{bad:?}");
        }
        // An argument too large to keep is no value.
        assert!(scan(vec![Some("--log-level".into()), None], false).is_err());
    }
}
