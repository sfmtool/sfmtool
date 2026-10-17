// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`PhaseLog`]: a sink that keeps every phase one call opened and closed, for
//! a caller that wants the whole list afterwards rather than a live panel.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::{Event, Level};

/// One phase a [`PhaseLog`] kept.
#[derive(Debug, Clone, PartialEq)]
pub struct LoggedPhase {
    /// The name the phase was opened with.
    pub name: &'static str,
    /// Its depth, 0 for a phase opened at the top of the call.
    pub depth: u8,
    /// What [`crate::Phase::note`] said it did, if anything.
    pub note: Option<String>,
    /// When its [`Event::Enter`] arrived, counted from when the log was made.
    pub start: Duration,
    /// The duration its [`Event::Leave`] carried.
    pub took: Duration,
    /// The messages emitted inside it and not inside a phase within it.
    pub messages: Vec<(Level, String)>,
}

/// A phase entered, and whether it has closed.
struct Slot {
    phase: LoggedPhase,
    closed: bool,
}

#[derive(Default)]
struct Kept {
    /// Every phase entered, in the order entered.
    slots: Vec<Slot>,
    /// The slot of the phase open at each depth.
    open: Vec<Option<usize>>,
}

/// Every phase one call opened and closed, in the order they were opened,
/// with each one's note, start, duration and messages.
///
/// Feed it every event with [`Self::record`], from the sink closure given to
/// [`crate::Progress::to`]; read the phases with [`Self::phases`]. A phase
/// takes a slot when its [`Event::Enter`] arrives, the slot open at its
/// depth, and fills it when its [`Event::Leave`] arrives, so the phases come
/// out in entry order, each after the phase it is inside, and each start is
/// the time of the enter. A message goes to the phase open one depth above
/// it.
///
/// One slot per depth is what a call that reports from one thread at a time
/// needs. Under phases opened concurrently on several threads at one depth, a
/// leave closes whichever phase of its name entered there last, and a phase
/// can be lost; such a call wants a collector that keeps its events instead.
/// A phase that never closes, one whose guard was cancelled, is not kept,
/// and neither is a leave with no enter seen for it nor a message outside
/// every phase.
///
/// ```
/// use sfmtool_progress::{Event, PhaseLog, Progress};
///
/// let log = PhaseLog::new();
/// let sink = |event: Event<'_>| log.record(event);
/// let progress = Progress::to(&sink).detailed(true);
/// {
///     let outer = progress.phase("solve");
///     let _inner = outer.detail_phase("linearise");
/// }
/// let names: Vec<_> = log.phases().iter().map(|p| (p.name, p.depth)).collect();
/// assert_eq!(names, [("solve", 0), ("linearise", 1)]);
/// ```
pub struct PhaseLog {
    started: Instant,
    kept: Mutex<Kept>,
}

impl Default for PhaseLog {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseLog {
    /// An empty log whose starts count from now.
    pub fn new() -> Self {
        PhaseLog {
            started: Instant::now(),
            kept: Mutex::new(Kept::default()),
        }
    }

    /// Keep what `event` says about a phase or a message. Statuses, counts
    /// and fractions are not kept.
    pub fn record(&self, event: Event<'_>) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        let kept = &mut *kept;
        match event {
            Event::Enter { phase, depth } => {
                let start = self.started.elapsed();
                let d = usize::from(depth);
                // Whatever was open deeper than this has closed or was
                // abandoned.
                kept.open.truncate(d);
                kept.open.resize(d + 1, None);
                kept.open[d] = Some(kept.slots.len());
                kept.slots.push(Slot {
                    phase: LoggedPhase {
                        name: phase,
                        depth,
                        note: None,
                        start,
                        took: Duration::ZERO,
                        messages: Vec::new(),
                    },
                    closed: false,
                });
            }
            Event::Leave {
                phase,
                depth,
                took,
                note,
            } => {
                let d = usize::from(depth);
                let Some(index) = kept.open.get(d).copied().flatten() else {
                    return;
                };
                let slot = &mut kept.slots[index];
                if slot.phase.name != phase {
                    return;
                }
                slot.phase.took = took;
                slot.phase.note = note.map(str::to_string);
                slot.closed = true;
                kept.open[d] = None;
            }
            Event::Message { level, depth, text } => {
                let Some(above) = usize::from(depth).checked_sub(1) else {
                    return;
                };
                if let Some(index) = kept.open.get(above).copied().flatten() {
                    kept.slots[index]
                        .phase
                        .messages
                        .push((level, text.to_string()));
                }
            }
            Event::Status { .. } | Event::Count { .. } | Event::Fraction { .. } => {}
        }
    }

    /// The phases that closed, in the order they were entered: a phase
    /// before every phase inside it.
    pub fn phases(&self) -> Vec<LoggedPhase> {
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        kept.slots
            .iter()
            .filter(|slot| slot.closed)
            .map(|slot| slot.phase.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{progress_info, progress_note, Progress};

    fn enter(phase: &'static str, depth: u8) -> Event<'static> {
        Event::Enter { phase, depth }
    }

    fn leave(phase: &'static str, depth: u8, took: Duration) -> Event<'static> {
        Event::Leave {
            phase,
            depth,
            took,
            note: None,
        }
    }

    /// A child that reports a longer duration than its parent, which a start
    /// read as the close less the duration would put before its parent, still
    /// comes after it, and each start is the time of the enter.
    #[test]
    fn a_phase_comes_after_its_parent_and_starts_at_its_enter() {
        let log = PhaseLog::new();
        log.record(enter("outer", 0));
        log.record(enter("first", 1));
        log.record(enter("deep", 2));
        log.record(leave("deep", 2, Duration::from_secs(30)));
        log.record(leave("first", 1, Duration::from_secs(20)));
        std::thread::sleep(Duration::from_millis(50));
        log.record(enter("second", 1));
        log.record(leave("second", 1, Duration::from_secs(40)));
        log.record(leave("outer", 0, Duration::from_millis(1)));

        let phases = log.phases();
        let names: Vec<_> = phases.iter().map(|p| (p.name, p.depth)).collect();
        assert_eq!(
            names,
            [("outer", 0), ("first", 1), ("deep", 2), ("second", 1)]
        );
        for pair in phases.windows(2) {
            assert!(pair[0].start <= pair[1].start, "{phases:#?}");
        }
        // The starts are the enters', not the leaves' less their durations
        // (which would put `outer` after the rest and `second` at zero), and
        // the durations are the leaves'.
        assert!(phases[3].start >= Duration::from_millis(50), "{phases:#?}");
        assert_eq!(phases[0].took, Duration::from_millis(1));
        assert_eq!(phases[2].took, Duration::from_secs(30));
    }

    /// Every phase follows the nearest phase before it one depth up, which is
    /// the phase it was opened inside; messages go to the phase they were
    /// emitted in; a cancelled phase is not kept.
    #[test]
    fn a_progress_reports_its_tree_in_entry_order() {
        let log = PhaseLog::new();
        let sink = |event: Event<'_>| log.record(event);
        let progress = Progress::to(&sink).detailed(true);
        {
            let mut step = progress.phase("step");
            progress_info!(step, "inside the step");
            for i in 0..3 {
                let mut stage = step.detail_phase("stage");
                progress_note!(stage, "stage {i}");
                let inner = stage.detail_phase("inner");
                progress_info!(inner, "inside inner {i}");
            }
            step.phase("abandoned").cancel();
            progress_note!(step, "done");
        }
        let phases = log.phases();
        let names: Vec<_> = phases.iter().map(|p| p.name).collect();
        assert_eq!(
            names,
            ["step", "stage", "inner", "stage", "inner", "stage", "inner"]
        );
        for (k, phase) in phases.iter().enumerate().skip(1) {
            let parent = phases[..k]
                .iter()
                .rev()
                .find(|p| p.depth + 1 == phase.depth)
                .expect("a parent before it");
            assert!(parent.start <= phase.start);
        }
        assert_eq!(phases[0].note.as_deref(), Some("done"));
        assert_eq!(
            phases[0].messages,
            [(Level::Info, "inside the step".to_string())]
        );
        assert_eq!(phases[1].note.as_deref(), Some("stage 0"));
        assert!(phases[1].messages.is_empty());
        assert_eq!(
            phases[6].messages,
            [(Level::Info, "inside inner 2".to_string())]
        );
    }

    /// A leave with no enter seen, and a message outside every phase, are
    /// not kept.
    #[test]
    fn what_has_no_open_phase_is_not_kept() {
        let log = PhaseLog::new();
        log.record(Event::Message {
            level: Level::Warn,
            depth: 0,
            text: "outside",
        });
        log.record(leave("never entered", 0, Duration::from_millis(5)));
        assert!(log.phases().is_empty());
    }
}
