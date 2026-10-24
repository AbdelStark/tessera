// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public phase boundaries for the NI driver's optional observer (05 §5).
use std::time::{Duration, Instant};

/// A driver check boundary; no proof messages or witness values are exposed.
#[derive(Clone, Copy, Debug)]
pub enum CheckEvent {
    /// Enter a check of a fixed public kind.
    Started {
        /// Fixed driver phase identifier.
        kind: &'static str,
    },
    /// Leave a check, including on early rejection or operational failure.
    Finished {
        /// The same kind as its start event.
        kind: &'static str,
        /// Wall time spent inside this check.
        duration: Duration,
        /// Whether control reached the next check or successful return.
        completed: bool,
    },
}

pub(super) struct Checks<'a> {
    observer: &'a mut dyn FnMut(CheckEvent),
    active: Option<(&'static str, Instant)>,
}
impl<'a> Checks<'a> {
    pub(super) fn new(observer: &'a mut dyn FnMut(CheckEvent)) -> Self {
        let mut checks = Self {
            observer,
            active: None,
        };
        checks.next("input");
        checks
    }
    pub(super) fn next(&mut self, kind: &'static str) {
        self.finish(true);
        (self.observer)(CheckEvent::Started { kind });
        self.active = Some((kind, Instant::now()));
    }
    pub(super) fn finish(&mut self, completed: bool) {
        if let Some((kind, start)) = self.active.take() {
            (self.observer)(CheckEvent::Finished {
                kind,
                duration: start.elapsed(),
                completed,
            });
        }
    }
}
impl Drop for Checks<'_> {
    fn drop(&mut self) {
        self.finish(false);
    }
}
