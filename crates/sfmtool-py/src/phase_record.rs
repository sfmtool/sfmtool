// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A `Progress` sink that keeps every phase a call closed, for a binding
//! that hands them back to Python as a list.
//!
//! A binding that offers this makes a [`PhaseRecord`], runs its step with a
//! detailed `Progress::to` over a closure that calls [`PhaseRecord::sink`],
//! and turns the record into a list of dicts with [`PhaseRecord::to_list`]. Each phase is
//! kept when its [`Event::Leave`] arrives, so nothing is matched against its
//! [`Event::Enter`]: the start is the time of the leave less the phase's own
//! duration, read against the moment the record was made.
//!
//! A message is given to the phase it was emitted inside: the next phase to
//! close at one depth less than the message's. That holds for a call that
//! reports from one thread at a time; a message emitted outside every phase
//! is not kept.

use std::sync::Mutex;
use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::progress::{Event, Level};

/// One closed phase.
struct Closed {
    name: &'static str,
    depth: u8,
    note: Option<String>,
    start_s: f64,
    duration_s: f64,
    messages: Vec<(Level, String)>,
}

#[derive(Default)]
struct Kept {
    closed: Vec<Closed>,
    /// Messages not yet given to a phase, with the depth each was emitted at.
    pending: Vec<(u8, Level, String)>,
}

/// Every phase one call closed, with its note, its start and duration and the
/// messages emitted inside it.
pub(crate) struct PhaseRecord {
    started: Instant,
    kept: Mutex<Kept>,
}

impl PhaseRecord {
    /// An empty record whose offsets count from now.
    pub(crate) fn new() -> Self {
        PhaseRecord {
            started: Instant::now(),
            kept: Mutex::new(Kept::default()),
        }
    }

    /// Keep what `event` says about a phase or a message; counts, fractions
    /// and statuses are not kept.
    pub(crate) fn sink(&self, event: Event<'_>) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        match event {
            Event::Leave {
                phase,
                depth,
                took,
                note,
            } => {
                let end = self.started.elapsed();
                let inside = depth.saturating_add(1);
                let mut messages = Vec::new();
                kept.pending.retain(|(d, level, text)| {
                    if *d == inside {
                        messages.push((*level, text.clone()));
                        false
                    } else {
                        true
                    }
                });
                kept.closed.push(Closed {
                    name: phase,
                    depth,
                    note: note.map(str::to_string),
                    start_s: end.saturating_sub(took).as_secs_f64(),
                    duration_s: took.as_secs_f64(),
                    messages,
                });
            }
            Event::Message { level, depth, text } => {
                kept.pending.push((depth, level, text.to_string()));
            }
            Event::Enter { .. }
            | Event::Status { .. }
            | Event::Count { .. }
            | Event::Fraction { .. } => {}
        }
    }

    /// The phases as a list of dicts, by start, a phase before the phases
    /// inside it: ``name``, ``depth`` (0 for a phase the call opened at its
    /// top), ``note`` (``None`` when the phase was given none), ``start_s``
    /// and ``duration_s`` (seconds, the start from when the record was made)
    /// and ``messages`` (a list of dicts with ``level``, ``"info"`` or
    /// ``"warn"``, and ``text``).
    pub(crate) fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        let mut order: Vec<&Closed> = kept.closed.iter().collect();
        order.sort_by(|a, b| a.start_s.total_cmp(&b.start_s).then(a.depth.cmp(&b.depth)));
        let list = PyList::empty(py);
        for phase in order {
            let d = PyDict::new(py);
            d.set_item("name", phase.name)?;
            d.set_item("depth", phase.depth)?;
            d.set_item("note", phase.note.as_deref())?;
            d.set_item("start_s", phase.start_s)?;
            d.set_item("duration_s", phase.duration_s)?;
            let messages = PyList::empty(py);
            for (level, text) in &phase.messages {
                let m = PyDict::new(py);
                m.set_item(
                    "level",
                    match level {
                        Level::Info => "info",
                        Level::Warn => "warn",
                    },
                )?;
                m.set_item("text", text)?;
                messages.append(m)?;
            }
            d.set_item("messages", messages)?;
            list.append(d)?;
        }
        Ok(list)
    }
}
