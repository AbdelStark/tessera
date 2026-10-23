// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The command line's `tracing` subscriber (05 §2, §5): one line per event on stderr, human by
//! default, JSON lines with `--log-format json` (`tessera.log-event/1`,
//! `spec/schemas/log-event-v1.schema.json`).
//!
//! Events and spans come from `tessera_codec::event!` and `span!`, so every field is public data
//! (05 §3 OBS-1) and every message a template. A `{name}` in the template stands for the field
//! `name`; both formats substitute it. A human line is the message, after `error: ` or `warn: `
//! at those levels and after the level and span path at `debug` and `trace`, followed by the
//! fields the template does not name. A JSON line holds `ts`, `level`, `target`, `event`,
//! `span_path` (the spans from the root, each with its fields) and `fields`, the event's typed
//! fields with the substituted `message`.

use std::io::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt as _};
use tracing_subscriber::registry::LookupSpan;

/// The format of stderr's lines.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum LogFormat {
    /// One readable line per event.
    #[default]
    Human,
    /// One JSON object per line, `tessera.log-event/1`.
    Json,
}

/// The most detailed level written (05 §2). `trace` events exist in debug builds only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum LogLevel {
    /// Failed operations.
    Error,
    /// Degraded but valid states.
    Warn,
    /// Phase boundaries and results.
    #[default]
    Info,
    /// Per-stage detail.
    Debug,
    /// Per-tile detail.
    Trace,
}

impl From<LogLevel> for LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => LevelFilter::ERROR,
            LogLevel::Warn => LevelFilter::WARN,
            LogLevel::Info => LevelFilter::INFO,
            LogLevel::Debug => LevelFilter::DEBUG,
            LogLevel::Trace => LevelFilter::TRACE,
        }
    }
}

/// Installs the subscriber for the rest of the process. A second call keeps the first.
pub(crate) fn install(format: LogFormat, level: LogLevel) {
    let subscriber = tracing_subscriber::registry()
        .with(LevelFilter::from(level))
        .with(Lines { format });
    let _ = tracing::subscriber::set_global_default(subscriber);
}

/// The fields of an event or a span, by name, as JSON values: integers and booleans as such,
/// `Public` values as their `Display` text.
#[derive(Clone, Debug, Default)]
struct Fields(Map<String, Value>);

impl Visit for Fields {
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().to_owned(), json!(value));
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().to_owned(), json!(value));
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.0.insert(field.name().to_owned(), json!(value));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_owned(), json!(value));
    }
    // `Public` values arrive as `tracing`'s `DisplayValue`, whose `Debug` is their `Display`.
    fn record_debug(&mut self, field: &Field, value: &dyn core::fmt::Debug) {
        self.0
            .insert(field.name().to_owned(), json!(format!("{value:?}")));
    }
}

struct Lines {
    format: LogFormat,
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Lines {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut fields = Fields::default();
        attrs.record(&mut fields);
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(fields);
        }
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id)
            && let Some(fields) = span.extensions_mut().get_mut::<Fields>()
        {
            values.record(fields);
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        let spans: Vec<(&'static str, Map<String, Value>)> = ctx
            .event_scope(event)
            .map(|scope| {
                scope
                    .from_root()
                    .map(|span| {
                        let fields = span.extensions().get::<Fields>().cloned();
                        (span.name(), fields.unwrap_or_default().0)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let metadata = event.metadata();
        let line = match self.format {
            LogFormat::Human => human(*metadata.level(), &spans, fields.0),
            LogFormat::Json => json_line(
                SystemTime::now(),
                *metadata.level(),
                metadata.target(),
                metadata.name(),
                &spans,
                fields.0,
            ),
        };
        let _ = std::io::stderr().lock().write_all(line.as_bytes());
    }
}

/// A field's text in a message: a string as it is, any other value as JSON.
fn text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The template with every `{name}` of a field replaced by its text, and the names it used.
fn render(template: &str, fields: &Map<String, Value>) -> (String, Vec<String>) {
    let mut out = String::new();
    let mut used = Vec::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        match after.find('}') {
            Some(close) if fields.contains_key(&after[..close]) => {
                let name = &after[..close];
                out.push_str(&text(&fields[name]));
                used.push(name.to_owned());
                rest = &after[close + 1..];
            }
            _ => {
                out.push('{');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    (out, used)
}

/// The event's message, and the fields other than `message` that it does not name.
fn message(mut fields: Map<String, Value>) -> (String, Map<String, Value>) {
    let template = match fields.remove("message") {
        Some(Value::String(s)) => s,
        _ => String::new(),
    };
    let (rendered, used) = render(&template, &fields);
    for name in used {
        fields.remove(&name);
    }
    (rendered, fields)
}

fn pairs(fields: &Map<String, Value>) -> String {
    fields
        .iter()
        .map(|(k, v)| format!("{k}={}", text(v)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn human(
    level: Level,
    spans: &[(&'static str, Map<String, Value>)],
    fields: Map<String, Value>,
) -> String {
    let (rendered, rest) = message(fields);
    let mut line = match level {
        Level::ERROR => format!("error: {rendered}"),
        Level::WARN => format!("warn: {rendered}"),
        Level::INFO => rendered,
        _ => {
            let path = spans
                .iter()
                .map(|(name, fields)| match fields.is_empty() {
                    true => (*name).to_owned(),
                    false => format!("{name}{{{}}}", pairs(fields)),
                })
                .collect::<Vec<_>>()
                .join(":");
            let level = level.as_str().to_ascii_lowercase();
            match path.is_empty() {
                true => format!("{level}: {rendered}"),
                false => format!("{level} {path}: {rendered}"),
            }
        }
    };
    if !rest.is_empty() {
        line.push(' ');
        line.push_str(&pairs(&rest));
    }
    line.push('\n');
    line
}

fn json_line(
    now: SystemTime,
    level: Level,
    target: &str,
    event: &str,
    spans: &[(&'static str, Map<String, Value>)],
    fields: Map<String, Value>,
) -> String {
    let template = fields.get("message").cloned();
    let (rendered, _) = message(fields.clone());
    let mut fields = fields;
    if template.is_some() {
        fields.insert("message".to_owned(), json!(rendered));
    }
    let span_path: Vec<Value> = spans
        .iter()
        .map(|(name, fields)| json!({"name": name, "fields": fields}))
        .collect();
    let object = json!({
        "ts": rfc3339(now),
        "level": level.as_str().to_ascii_lowercase(),
        "target": target,
        "event": event,
        "span_path": span_path,
        "fields": fields,
    });
    format!("{object}\n")
}

/// UTC, to the microsecond: `YYYY-MM-DDTHH:MM:SS.ffffffZ`.
fn rfc3339(now: SystemTime) -> String {
    let since = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = since.as_secs();
    let (days, of_day) = (secs / 86_400, secs % 86_400);
    let (year, month, day) = civil(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:06}Z",
        of_day / 3600,
        of_day / 60 % 60,
        of_day % 60,
        since.subsec_micros()
    )
}

/// The proleptic Gregorian date `days` after 1970-01-01 (Hinnant's `civil_from_days`).
fn civil(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn fields(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap()
    }

    #[test]
    fn templates_name_fields_and_leave_other_braces() {
        let f = fields(json!({"dir": "out", "rows": 3}));
        assert_eq!(
            render("wrote {rows} rows to {dir}", &f).0,
            "wrote 3 rows to out"
        );
        assert_eq!(render("{missing} {", &f).0, "{missing} {");
        assert_eq!(render("{dir}{dir}", &f).0, "outout");
    }

    #[test]
    fn human_lines_keep_the_command_line_contract() {
        let f = |v| fields(v);
        // 02 §2: errors are `error: ERR-...: what happened`.
        let error =
            f(json!({"code": "ERR-USAGE", "context": "no plan", "message": "{code}: {context}"}));
        assert_eq!(
            human(Level::ERROR, &[], error),
            "error: ERR-USAGE: no plan\n"
        );
        let info = f(json!({"dir": "out", "message": "tessera compile: wrote to {dir}"}));
        assert_eq!(
            human(Level::INFO, &[], info),
            "tessera compile: wrote to out\n"
        );
        let extra = f(json!({"rows": 2, "message": "done"}));
        assert_eq!(human(Level::WARN, &[], extra), "warn: done rows=2\n");
        let spans = [
            ("prove", fields(json!({"batch": "ab"}))),
            ("stage", fields(json!({"id": 1}))),
        ];
        let debug = f(json!({"message": "rounds"}));
        assert_eq!(
            human(Level::DEBUG, &spans, debug),
            "debug prove{batch=ab}:stage{id=1}: rounds\n"
        );
    }

    #[test]
    fn json_lines_hold_every_field() {
        let at = UNIX_EPOCH + Duration::from_micros(1_761_211_769_000_042);
        let spans = [("stage", fields(json!({"id": 2})))];
        let line = json_line(
            at,
            Level::INFO,
            "tessera_prover::cpu",
            "tessera.prover.stage_proved",
            &spans,
            fields(json!({"rounds": 8, "message": "{rounds} rounds"})),
        );
        let value: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(
            value,
            json!({
                "ts": "2025-10-23T09:29:29.000042Z",
                "level": "info",
                "target": "tessera_prover::cpu",
                "event": "tessera.prover.stage_proved",
                "span_path": [{"name": "stage", "fields": {"id": 2}}],
                "fields": {"rounds": 8, "message": "8 rounds"},
            })
        );
        assert!(line.ends_with('\n') && !line[..line.len() - 1].contains('\n'));
    }

    #[test]
    fn dates_follow_the_gregorian_calendar() {
        assert_eq!(rfc3339(UNIX_EPOCH), "1970-01-01T00:00:00.000000Z");
        let leap = UNIX_EPOCH + Duration::from_secs(951_782_400); // 2000-02-29
        assert_eq!(rfc3339(leap), "2000-02-29T00:00:00.000000Z");
        let end = UNIX_EPOCH + Duration::from_secs(4_102_444_799); // 2099-12-31T23:59:59
        assert_eq!(rfc3339(end), "2099-12-31T23:59:59.000000Z");
    }
}
