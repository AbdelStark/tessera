// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Telemetry that accepts public data only (05 §2, §3 OBS-1, §5).
//!
//! [`event!`](crate::event) and [`span!`](crate::span) are the only way Tessera's crates emit
//! `tracing` events and spans: `cargo xtask lint` refuses `tracing`'s own macros elsewhere, and a
//! `tracing` dependency outside this crate and the CLI's subscriber. They take:
//!
//! - **a name**: an event's is `tessera.<crate>.<event>` in snake case, given without its
//!   `tessera.` prefix, and a span's one of 05 §5's;
//! - **fields**, each a [`Field`]: a [`Public`] value, cleared for telemetry by whoever wraps
//!   it, an integer, a `bool`, or a `&'static str`, a fixed identifier. A [`Secret`](crate::Secret)
//!   and a [`SecretBuf`](crate::SecretBuf) are not fields, and neither is a `String` built at
//!   run time, so they cannot reach a subscriber;
//! - **for an event, a message template**: a string literal, recorded as the `message` field and
//!   never formatted by `tracing`. A `{name}` in it stands for the field of that name; the
//!   subscriber of `tessera-cli` substitutes it. The template itself cannot capture a variable.
//!
//! `trace` events are compiled out of release builds (`tracing`'s `release_max_level_debug`).
//! With no subscriber installed, as in a library user's process, events and spans cost a branch.

use core::fmt;

use crate::Public;

/// What [`span!`](crate::span) returns; `.entered()` makes it current until dropped.
pub use tracing::Span;

/// A value an event or span may carry (OBS-1).
pub trait Field {
    /// What `tracing` records.
    type Value<'a>: tracing::Value
    where
        Self: 'a;

    /// The value to record.
    fn value(&self) -> Self::Value<'_>;
}

macro_rules! by_value {
    ($as:ty: $($t:ty),*) => {
        $(
            impl Field for $t {
                type Value<'a> = $as;

                fn value(&self) -> $as {
                    <$as>::from(*self)
                }
            }
        )*
    };
}

by_value!(u64: u8, u16, u32, u64);
by_value!(i64: i8, i16, i32, i64);
by_value!(bool: bool);

impl Field for usize {
    type Value<'a> = u64;

    fn value(&self) -> u64 {
        // usize is at most 64 bits on every supported target (09 §8).
        *self as u64
    }
}

impl Field for &'static str {
    type Value<'a> = &'static str;

    fn value(&self) -> &'static str {
        self
    }
}

impl<T: fmt::Display> Field for Public<T> {
    type Value<'a>
        = tracing::field::DisplayValue<&'a T>
    where
        T: 'a;

    fn value(&self) -> tracing::field::DisplayValue<&T> {
        tracing::field::display(&self.0)
    }
}

/// The crate's events and spans, through [`event!`](crate::event) and [`span!`](crate::span)
/// only. Not part of the public interface.
#[doc(hidden)]
pub mod __private {
    pub use tracing;
}

/// Emits a `tracing` event of public data (05 §2): a level, a name `"<crate>.<event>"`, fields,
/// then a message template.
///
/// ```
/// use tessera_codec::{Public, event};
/// let dir = std::path::Path::new("out");
/// event!(INFO, "codec.example_written", dir = Public(dir.display()), rows = 3u32;
///        "wrote {rows} rows to {dir}");
/// ```
#[macro_export]
macro_rules! event {
    ($level:ident, $name:literal $(, $key:ident = $value:expr)* $(,)? ; $template:literal) => {{
        // The one sanctioned use of `tracing::event!` (`cargo xtask lint`, 05 §3 OBS-1).
        $crate::telemetry::__private::tracing::event!(
            name: ::core::concat!("tessera.", $name),
            $crate::telemetry::__private::tracing::Level::$level,
            $( $key = $crate::telemetry::Field::value(&$value), )*
            message = $template
        )
    }};
}

/// A `tracing` span of public data (05 §5): a level, one of 05 §5's names, fields. Enter it
/// with `.entered()`.
///
/// ```
/// use tessera_codec::span;
/// let _stage = span!(DEBUG, "stage", id = 1u16).entered();
/// ```
#[macro_export]
macro_rules! span {
    ($level:ident, $name:literal $(, $key:ident = $value:expr)* $(,)?) => {{
        // The one sanctioned use of `tracing::span!` (`cargo xtask lint`, 05 §3 OBS-1).
        $crate::telemetry::__private::tracing::span!(
            $crate::telemetry::__private::tracing::Level::$level,
            $name,
            $( $key = $crate::telemetry::Field::value(&$value) ),*
        )
    }};
}

#[cfg(test)]
mod tests {
    use tracing::level_filters::{LevelFilter, STATIC_MAX_LEVEL};

    /// 05 §2: trace-level events vanish from release builds, which have no debug assertions.
    #[test]
    fn trace_events_compile_out_of_release_builds() {
        let expected = match cfg!(debug_assertions) {
            true => LevelFilter::TRACE,
            false => LevelFilter::DEBUG,
        };
        assert_eq!(STATIC_MAX_LEVEL, expected);
    }
}
