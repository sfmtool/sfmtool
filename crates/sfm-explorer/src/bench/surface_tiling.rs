// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Track View's Surface Tiling Mode: covering a surface with patches by
//! alternating *Duplicate* with a move of the copy in the 3D viewer.
//!
//! See `specs/gui/track-view.md` § "Surface Tiling Mode". The mode changes two
//! things, and both are made of the bench's existing steps:
//!
//! - **A duplicate in the mode** ([`AppState::duplicate_for_surface_tiling`])
//!   is *Duplicate*, then the copy's `out` observations dropped
//!   ([`AppState::drop_out_bench_observations`]), then every verdict still
//!   pinned on the copy handed back to the thresholds
//!   ([`AppState::unpin_bench_verdicts`]). Three versions, three Action Log
//!   rows, as the same three steps taken by hand would be.
//! - **A move of the patch in the 3D viewer in the mode** starts a
//!   [`TilingSequence`] ([`AppState::start_surface_tiling`]): Fit, Find
//!   matches by geometry, Fit, Grid Plane Normal, Fit and Commit, each through
//!   the starter its button calls, so each pushes its own version and writes
//!   its own row. [`AppState::drive_surface_tiling`] runs once per frame and
//!   starts the next step once the one before it has finished and the track's
//!   live evaluation has caught up with it, which is when a person pressing
//!   the buttons in turn would see *Evaluated* and press the next.
//!
//! **A sequence stops rather than guessing.** A step refused or failed, the
//! item no longer focused, or a version on the node that the sequence did not
//! push (an undo, a pin, a threshold) each end it with a failed Bench row
//! naming the step it stopped at and why. A second move of the patch while the
//! sequence waits between steps starts it again from the first Fit.

#[cfg(test)]
mod tests;

use sfmtool_core::bench::{EditableTrack, ItemId, Verdict};

use crate::action_log::Kind;
use crate::bench::live::Evaluation;
use crate::bench::{FocusedItem, NormalStep, SplitSettings};
use crate::document::VersionSerial;
use crate::scene::ReconId;
use crate::state::AppState;

/// The label of Track View's box, and what the sentences that point at it
/// quote.
pub(crate) const SURFACE_TILING_LABEL: &str = "Surface Tiling Mode";

/// One step of a [`TilingSequence`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TilingStep {
    /// Track View's *Fit* ([`AppState::start_bench_fit`]).
    Fit,
    /// The observation table's *Find matches by geometry*
    /// ([`AppState::start_bench_geometry_search`]), from [`search_row`].
    SearchGeometry,
    /// Track View's *Grid Plane Normal* ([`AppState::start_bench_normal`]),
    /// cut as the *per axis* and *overlap* boxes stood when the sequence
    /// started.
    GridPlaneNormal,
    /// Track View's *Commit* ([`AppState::commit_bench_track`]).
    Commit,
}

impl TilingStep {
    /// The step's name, as its button or menu entry reads.
    fn name(self) -> &'static str {
        match self {
            TilingStep::Fit => "Fit",
            TilingStep::SearchGeometry => "Find matches by geometry",
            TilingStep::GridPlaneNormal => "Grid Plane Normal",
            TilingStep::Commit => "Commit",
        }
    }
}

/// The steps a move of the patch runs in Surface Tiling Mode, in order.
///
/// The first Fit lets the sightings follow the moved patch and triangulates
/// it where they agree; the geometry search then has a placed patch to project
/// into every camera; the second Fit takes the cameras it added; Grid Plane
/// Normal turns the patch to the surface under it; the third Fit settles the
/// sightings on the turned patch; and the commit writes the point.
const TILING_STEPS: [TilingStep; 6] = [
    TilingStep::Fit,
    TilingStep::SearchGeometry,
    TilingStep::Fit,
    TilingStep::GridPlaneNormal,
    TilingStep::Fit,
    TilingStep::Commit,
];

/// A Surface Tiling Mode sequence under way on one bench item.
#[derive(Debug, Clone)]
pub(crate) struct TilingSequence {
    /// The node whose bench holds the item.
    node: ReconId,
    /// The item, by the ID a rename keeps.
    item: ItemId,
    /// How *Grid Plane Normal* cuts the patch.
    split: SplitSettings,
    /// The position in [`TILING_STEPS`] of the step to start next, or of the
    /// step running while `waiting_on` is set.
    next: usize,
    /// The background task the sequence started and has not yet seen end.
    waiting_on: Option<u64>,
    /// The node's version after the last step the sequence took, or after the
    /// move that started it. Any other version at the cursor is a step the
    /// sequence did not take, and ends it.
    serial: VersionSerial,
}

impl TilingSequence {
    /// The step the sequence is running or about to start.
    fn step(&self) -> TilingStep {
        TILING_STEPS[self.next]
    }
}

/// What one frame of [`AppState::drive_surface_tiling`] did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TilingDriven {
    /// Whether anything happened, so the frame knows to draw again.
    pub(crate) changed: bool,
    /// The node a commit gave a new version, whose panel caches now describe
    /// a value it no longer holds.
    pub(crate) committed: Option<ReconId>,
}

/// The observation the sequence's geometry search starts from: the row the
/// track's stored bitmap is the render of ([`crate::bench::reference_in_use`]),
/// else the row the reference-view rule picked at the last evaluation, else
/// the first `in` row with a place in its photograph.
///
/// The search projects the patch into every camera and compares each against
/// a reference appearance built from the named row first and then every `in`
/// row, so the named row is the one whose appearance leads. The reference is
/// the sighting the Fit just before rendered the patch from, which the rule
/// chose as the best view of it, so it is the appearance a person would right
/// click. `None` when no row has a place to read an appearance at.
fn search_row(track: &EditableTrack) -> Option<usize> {
    let placed = |row: usize| {
        track
            .observations
            .get(row)
            .is_some_and(|o| o.site().is_some())
    };
    crate::bench::reference_in_use(track)
        .filter(|&row| placed(row))
        .or_else(|| crate::bench::reference_view_pick(track).filter(|&row| placed(row)))
        .or_else(|| {
            track
                .observations
                .iter()
                .position(|o| o.verdict == Verdict::In && o.site().is_some())
        })
}

impl AppState {
    /// Drop the `out` observations of the track called `label`
    /// ([`sfmtool_core::bench::EditableTrack::drop_out_observations`]), as
    /// one version.
    ///
    /// The step a Surface Tiling Mode duplicate takes after the copy is made.
    /// A track with no `out` observation is a no-effect row, and one whose
    /// every observation is `out` is refused, since a track with none is not
    /// one. The selected rows are cleared, the rows left being renumbered.
    fn drop_out_bench_observations(&mut self, id: ReconId, label: &str) -> Result<(), String> {
        let (index, bench, track) = self.bench_step_target(id, label)?;
        let Some((next, _)) = track.drop_out_observations() else {
            self.no_effect(format!(
                "Dropped the out observations of {label}: no effect, none is out"
            ));
            return Ok(());
        };
        if next.observations.is_empty() {
            return Err(format!(
                "Cannot drop the out observations of {label}: every one of its {} \
                 observations is out, and that would leave it none",
                track.observations.len()
            ));
        }
        let dropped = track.observations.len() - next.observations.len();
        let text = format!(
            "Dropped {dropped} out {} of {label}, leaving {} in",
            if dropped == 1 {
                "observation"
            } else {
                "observations"
            },
            next.observations.len()
        );
        let bench = super::install(&bench, label, next)?;
        self.clear_bench_rows(id);
        self.push_bench_step(index, bench, text);
        Ok(())
    }

    /// *Duplicate* in Surface Tiling Mode: duplicate the track called
    /// `label`, drop the copy's `out` observations, and hand every verdict
    /// still pinned on it back to the thresholds, reporting the copy's label.
    ///
    /// The three steps are the ones a person takes by hand, each its own
    /// version and row, and the copy is the focused item from the first of
    /// them on. A step with nothing to do is skipped. A refusal after the
    /// duplicate leaves the copy on the bench as far as it got and is returned
    /// for the caller to log, as a refused duplicate is.
    pub(crate) fn duplicate_for_surface_tiling(
        &mut self,
        id: ReconId,
        label: &str,
    ) -> Result<String, String> {
        let copy = self.duplicate_bench_item(id, label)?;
        let has_out = self
            .bench_track(id, &copy)
            .is_some_and(|track| track.observations.iter().any(|o| o.verdict == Verdict::Out));
        if has_out {
            self.drop_out_bench_observations(id, &copy)?;
        }
        let pinned: Vec<usize> = self
            .bench_track(id, &copy)
            .map(|track| {
                track
                    .observations
                    .iter()
                    .enumerate()
                    .filter_map(|(row, o)| o.pinned.then_some(row))
                    .collect()
            })
            .unwrap_or_default();
        if !pinned.is_empty() {
            self.unpin_bench_verdicts(id, &copy, &pinned)?;
        }
        Ok(copy)
    }

    /// One released gesture over the 3D viewer's bench figure that edits the
    /// patch of the item called `label` ([`Self::edit_bench_patch`]), and, in
    /// Surface Tiling Mode, the sequence after it.
    ///
    /// `tiling` is `None` with the mode off, and otherwise how the sequence's
    /// *Grid Plane Normal* cuts the patch. The sequence starts only when the
    /// edit changed the patch: a release that put it back where it was is a
    /// no-effect row and nothing more, as it is with the mode off.
    pub(crate) fn edit_bench_patch_from_viewer(
        &mut self,
        id: ReconId,
        label: &str,
        edit: &crate::bench::PatchEdit,
        tiling: Option<SplitSettings>,
    ) -> Result<(), String> {
        let edited = self.edit_bench_patch(id, label, edit)?;
        if let (true, Some(split)) = (edited.changed, tiling) {
            self.start_surface_tiling(id, label, split);
        }
        Ok(())
    }

    /// Start a Surface Tiling Mode sequence on the item called `label`, the
    /// patch of which a gesture in the 3D viewer has just moved.
    ///
    /// The sequence's first step starts on the next
    /// [`Self::drive_surface_tiling`]. A sequence already under way is
    /// replaced, so a second move between its steps starts it again from the
    /// first Fit. `split` is how its *Grid Plane Normal* cuts the patch.
    fn start_surface_tiling(&mut self, id: ReconId, label: &str, split: SplitSettings) {
        let (Ok(index), Some(item)) = (
            self.node_index(id),
            self.bench(id).and_then(|bench| bench.id(label)),
        ) else {
            return;
        };
        let serial = self.scene[index].history.current_version().serial;
        let restarted = self.surface_tiling.take().is_some();
        let steps = TILING_STEPS.map(TilingStep::name).join(", ");
        self.action_log.record(
            Kind::Bench,
            format!(
                "Surface tiling {label} {} the move: {steps}",
                if restarted {
                    "again, from the start, after"
                } else {
                    "after"
                }
            ),
        );
        self.surface_tiling = Some(TilingSequence {
            node: id,
            item,
            split,
            next: 0,
            waiting_on: None,
            serial,
        });
    }

    /// End the sequence under way, if there is one, with a failed row saying
    /// `why`.
    pub(crate) fn stop_surface_tiling(&mut self, why: &str) {
        let Some(sequence) = self.surface_tiling.take() else {
            return;
        };
        let label = self
            .bench(sequence.node)
            .and_then(|bench| bench.label_of(sequence.item))
            .unwrap_or("the item")
            .to_string();
        self.action_log.fail(
            Kind::Bench,
            format!(
                "Surface tiling of {label} stopped at {} (step {} of {}): {why}",
                sequence.step().name(),
                sequence.next + 1,
                TILING_STEPS.len()
            ),
        );
    }

    /// What Track View says beside its box about the sequence under way on
    /// the item called `label` of `id`'s bench, or `None` when there is none.
    pub(crate) fn surface_tiling_status(&self, id: ReconId, label: &str) -> Option<String> {
        let sequence = self.surface_tiling.as_ref()?;
        let item = self.bench(id)?.id(label)?;
        if sequence.node != id || sequence.item != item {
            return None;
        }
        Some(format!(
            "Step {} of {}: {}",
            sequence.next + 1,
            TILING_STEPS.len(),
            sequence.step().name()
        ))
    }

    /// Move the sequence under way on by what has happened since the last
    /// frame: see that the step it waits on has ended, check that nothing
    /// else has changed the item, and start the next step once the item's
    /// evaluation has caught up.
    ///
    /// Run once per frame after the background task is polled and the live
    /// evaluation driven, so a step that ended and the evaluation that
    /// followed it are both seen in the frame they landed.
    pub(crate) fn drive_surface_tiling(&mut self) -> TilingDriven {
        let mut driven = TilingDriven::default();
        let Some(sequence) = self.surface_tiling.clone() else {
            return driven;
        };
        let label = match self.sequence_label(&sequence) {
            Ok(label) => label,
            Err(why) => {
                self.stop_surface_tiling(&why);
                driven.changed = true;
                return driven;
            }
        };
        let index = self
            .node_index(sequence.node)
            .expect("checked by the label");

        if let Some(task) = sequence.waiting_on {
            if self
                .background_task()
                .is_some_and(|running| running.id == task)
            {
                return driven;
            }
            let outcome = self
                .last_background_task
                .as_ref()
                .filter(|last| last.id == task)
                .map(|last| last.outcome.clone());
            match outcome {
                Some(Ok(_)) => {}
                Some(Err(why)) => {
                    self.stop_surface_tiling(&format!("it did not finish: {why}"));
                    driven.changed = true;
                    return driven;
                }
                None => {
                    self.stop_surface_tiling("it ended with no report");
                    driven.changed = true;
                    return driven;
                }
            }
            let serial = self.scene[index].history.current_version().serial;
            let running = self.surface_tiling.as_mut().expect("cloned above");
            running.waiting_on = None;
            running.next += 1;
            running.serial = serial;
            driven.changed = true;
        } else if self.scene[index].history.current_version().serial != sequence.serial {
            self.stop_surface_tiling(&format!(
                "{label} was changed by a step the sequence did not take"
            ));
            driven.changed = true;
            return driven;
        }

        // Wait for the evaluation of the track as it now stands, as a person
        // waits for *Evaluated* before pressing the next button: a geometry
        // search's candidates join `out` and unmeasured, and the evaluation is
        // what lets the bars take them in before the Fit after it.
        if self.bench_evaluation(sequence.node, &label) == Some(Evaluation::Evaluating) {
            return driven;
        }
        driven.changed = true;
        let step = self.surface_tiling.as_ref().expect("still running").step();
        let id = sequence.node;
        let started = match step {
            TilingStep::Fit => self.start_bench_fit(id, &label),
            TilingStep::SearchGeometry => {
                let row = self.bench_track(id, &label).and_then(|t| search_row(t));
                match row {
                    Some(row) => self.start_bench_geometry_search(id, &label, row),
                    None => Err(format!(
                        "no observation of {label} has a place in its photograph to search from"
                    )),
                }
            }
            TilingStep::GridPlaneNormal => {
                let split = self.surface_tiling.as_ref().expect("still running").split;
                self.start_bench_normal(id, &label, NormalStep::GridPlane(split))
            }
            TilingStep::Commit => {
                match self.commit_bench_track(id, &label) {
                    Err(why) => {
                        // The row the Commit button writes for its refusal.
                        self.action_log.fail(Kind::Edit, why.clone());
                        self.stop_surface_tiling(&why);
                    }
                    Ok(written) => {
                        self.surface_tiling = None;
                        if written.changed {
                            driven.committed = Some(id);
                        }
                    }
                }
                return driven;
            }
        };
        match started {
            Err(why) => self.stop_surface_tiling(&why),
            Ok(()) => match self.background_task().map(|task| task.id) {
                Some(task) => {
                    self.surface_tiling
                        .as_mut()
                        .expect("still running")
                        .waiting_on = Some(task);
                }
                None => self.stop_surface_tiling("it started no task to wait on"),
            },
        }
        driven
    }

    /// The label of the sequence's item, or why the sequence cannot go on:
    /// its node closed, the item off the bench, or another item focused.
    fn sequence_label(&self, sequence: &TilingSequence) -> Result<String, String> {
        let label = self
            .bench(sequence.node)
            .and_then(|bench| bench.label_of(sequence.item))
            .ok_or_else(|| "the item is no longer on the bench".to_string())?
            .to_string();
        let focused = FocusedItem {
            node: sequence.node,
            item: sequence.item,
        };
        if self.focused_item != Some(focused) {
            return Err(format!("{label} is no longer the focused item"));
        }
        Ok(label)
    }
}
