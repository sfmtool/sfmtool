// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A `Progress` sink that keeps every phase a call closed, for a binding
//! that hands them back to Python as a list.
//!
//! A binding that offers this makes a [`PhaseRecord`], runs its step with a
//! detailed `Progress::to` over a closure that calls [`PhaseRecord::sink`],
//! and turns the record into a list of dicts with [`PhaseRecord::to_list`].
//! The keeping is `sfmtool_progress::PhaseLog`'s: each phase takes a slot
//! when it is entered, the slot open at its depth, and fills it when it
//! closes, so the phases come out in entry order, a phase before the phases
//! inside it, and each start is the time of the enter. A message goes to the
//! phase open one depth above it. That holds for a call that reports from one
//! thread at a time; a message emitted outside every phase is not kept.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::progress::{Event, Level, PhaseLog};

/// Every phase one call closed, with its note, its start and duration and the
/// messages emitted inside it.
pub(crate) struct PhaseRecord {
    log: PhaseLog,
}

impl PhaseRecord {
    /// An empty record whose starts count from now.
    pub(crate) fn new() -> Self {
        PhaseRecord {
            log: PhaseLog::new(),
        }
    }

    /// Keep what `event` says about a phase or a message; counts, fractions
    /// and statuses are not kept.
    pub(crate) fn sink(&self, event: Event<'_>) {
        self.log.record(event);
    }

    /// The phases as a list of dicts, in the order they were entered, a phase
    /// before the phases inside it: ``name``, ``depth`` (0 for a phase the
    /// call opened at its top), ``note`` (``None`` when the phase was given
    /// none), ``start_s`` (seconds from when the record was made to the
    /// phase's entry), ``duration_s`` (seconds) and ``messages`` (a list of
    /// dicts with ``level``, ``"info"`` or ``"warn"``, and ``text``).
    pub(crate) fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for phase in self.log.phases() {
            let d = PyDict::new(py);
            d.set_item("name", phase.name)?;
            d.set_item("depth", phase.depth)?;
            d.set_item("note", phase.note.as_deref())?;
            d.set_item("start_s", phase.start.as_secs_f64())?;
            d.set_item("duration_s", phase.took.as_secs_f64())?;
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
