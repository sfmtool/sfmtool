// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Measurements on the bench: putting a stored measurement on, following its
//! origin, the commit that writes it back, and the direct delete of a stored
//! one.
//!
//! See `specs/gui/bench.md` and `specs/gui/measurement-view.md`. The item and
//! its pure steps are core's (`sfmtool_core::bench::measurement`,
//! `specs/core/bench/editable-measurement.md`); what this module adds is what
//! the rest of [`crate::bench`] adds for a track: the version each step is
//! pushed as, its Action Log row, and following the origin through the node's
//! version graph.

use std::sync::Arc;

use sfmtool_core::bench::{self, BenchItem, EditableMeasurement, ItemKind};

use crate::action_log::{version_step_text, Kind};
use crate::document::MeasurementMap;
use crate::measurements::MeasurementRef;
use crate::scene::{ReconId, SceneNode};
use crate::state::edits::version_before;
use crate::state::AppState;

#[cfg(test)]
mod tests;

/// What putting a stored measurement on the bench left
/// ([`AppState::put_measurement_on_bench`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PutMeasurement {
    /// The label of the measurement's item: the one the step put on, or the
    /// one already on the bench from that row.
    pub(crate) label: String,
    /// Whether the step put the item on, rather than finding it there.
    pub(crate) made: bool,
}

/// The row a measurement commit wrote, in the version the commit produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MeasurementCommitted {
    /// The row written, or the row that already held the item where the
    /// commit wrote nothing.
    pub(crate) measurement: MeasurementRef,
    /// Whether a stored row was replaced, rather than one appended.
    pub(crate) replaced: bool,
    /// Whether a version was pushed.
    pub(crate) changed: bool,
}

/// The sentence a track step names a measurement item with when it was asked
/// for a track.
pub(crate) fn not_a_track(label: &str) -> String {
    format!("{label} is a measurement, not a track.")
}

/// The sentence a measurement step names a track with when it was asked for a
/// measurement.
pub(crate) fn not_a_measurement(label: &str) -> String {
    format!("{label} is a track, not a measurement.")
}

impl AppState {
    /// The kind of the focused item, read at its node's cursor, or `None`
    /// when nothing is focused or the item is not on that bench there.
    pub(crate) fn focused_item_kind(&self) -> Option<ItemKind> {
        let focused = self.focused_item()?;
        let bench = self.bench(focused.node)?;
        let label = bench.label_of(focused.item)?;
        Some(bench.get(label)?.kind())
    }

    /// The label the focused item carries on `id`'s bench when it is a track,
    /// or `None`: what Track View and every track gesture act on.
    ///
    /// Track View's *Edit* box is ticked exactly when this is `Some`, so a
    /// focused measurement leaves the box clear.
    pub(crate) fn focused_track_label(&self, id: ReconId) -> Option<&str> {
        let label = self.focused_item_label(id)?;
        self.bench(id)?.track(label).map(|_| label)
    }

    /// The label the focused item carries on `id`'s bench when it is a
    /// measurement, or `None`: what Measurement View's edited mode shows.
    pub(crate) fn focused_measurement_label(&self, id: ReconId) -> Option<&str> {
        let label = self.focused_item_label(id)?;
        self.bench(id)?.measurement(label).map(|_| label)
    }

    /// The measurement called `label` on `id`'s bench, at that node's cursor.
    pub(crate) fn bench_measurement(
        &self,
        id: ReconId,
        label: &str,
    ) -> Option<&Arc<EditableMeasurement>> {
        self.bench(id)?.measurement(label)
    }

    /// The row the measurement item's origin names in the version `node`
    /// stands at, or `None` when it has no origin or the row is gone.
    ///
    /// The origin names a version by the serial the viewer minted for it and
    /// a row of one kind in it, so the walk is the node's own version graph
    /// over each step's measurement map
    /// ([`crate::document::History::follow_measurement`]), the way a track's
    /// origin is followed over the point maps ([`Self::resolved_origin`]).
    pub(crate) fn resolved_measurement_origin(
        &self,
        node: &SceneNode,
        item: &EditableMeasurement,
    ) -> Option<u32> {
        let origin = item.origin?;
        let from = node
            .history
            .all_serials()
            .into_iter()
            .find(|serial| serial.as_u64() == origin.version)?;
        let to = node.history.current_version().serial;
        let index = node
            .history
            .follow_measurement(from, to, origin.kind, origin.index)
            .ok()?;
        ((index as usize) < origin.kind.count(node.edited().measurements())).then_some(index)
    }

    /// The focused measurement item's origin followed to the cursor, as a
    /// stored measurement of the item's node, or `None` when no measurement
    /// is focused, it has no origin, or its row is gone.
    ///
    /// While a measurement item is focused, the selected measurement is this
    /// or none.
    pub(crate) fn focused_measurement_origin(&self) -> Option<MeasurementRef> {
        let focused = *self.focused_item()?;
        let node = self.node(focused.node)?;
        let bench = node.history.current_bench();
        let item = bench.measurement(bench.label_of(focused.item)?)?;
        let index = self.resolved_measurement_origin(node, item)?;
        Some(MeasurementRef {
            recon: focused.node,
            kind: item.kind(),
            index,
        })
    }

    /// The label of the item on `measurement`'s node's bench that came from
    /// that stored measurement: its origin, followed to the cursor, is that
    /// row. `None` when no item did.
    pub(crate) fn bench_item_from_measurement(
        &self,
        measurement: MeasurementRef,
    ) -> Option<String> {
        let node = self.node(measurement.recon)?;
        node.history
            .current_bench()
            .entries()
            .iter()
            .find(|entry| match &entry.item {
                BenchItem::Measurement(item) => {
                    item.kind() == measurement.kind
                        && self.resolved_measurement_origin(node, item) == Some(measurement.index)
                }
                BenchItem::Track(_) => false,
            })
            .map(|entry| entry.label.clone())
    }

    /// The image and pixel of endpoint `side` of the focused measurement item
    /// on `id`'s bench, or `None` where no measurement is focused there or
    /// that endpoint is not placed.
    pub(crate) fn focused_item_endpoint(
        &self,
        id: ReconId,
        side: crate::measurements::EndpointSide,
    ) -> Option<(crate::scene::ImageRef, [f64; 2])> {
        let label = self.focused_measurement_label(id)?;
        let item = self.bench_measurement(id, label)?;
        let endpoint = item.as_distance()?.endpoints[side.index()]?;
        Some((
            crate::scene::ImageRef::new(id, endpoint.image as usize),
            endpoint.pixel,
        ))
    }

    /// Select endpoint `side` of the focused measurement item on `id`'s
    /// bench, as a click on its block in Measurement View's edited mode does:
    /// the endpoint's image, with its pixel for Image Detail to reveal, then
    /// the endpoint, as [`Self::select_endpoint`] does for a stored row. An
    /// endpoint whose image is past the image set selects the endpoint alone,
    /// and one not placed selects nothing.
    pub(crate) fn select_item_endpoint(
        &mut self,
        id: ReconId,
        side: crate::measurements::EndpointSide,
    ) {
        let Some((image, [x, y])) = self.focused_item_endpoint(id, side) else {
            return;
        };
        let in_set = self
            .node(id)
            .is_some_and(|node| image.index() < node.recon().image_set.images.len());
        if in_set {
            self.reveal_in_image(image, [x as f32, y as f32]);
        }
        if self.selected_endpoint != Some(side) {
            self.selected_endpoint = Some(side);
            let label = self.focused_item_label(id).unwrap_or_default().to_string();
            self.action_log.record_run(
                Kind::Selection,
                "measurement endpoint",
                format!("Selected endpoint {} of {label}", side.letter()),
            );
        }
    }

    /// Put the stored measurement `measurement` on its node's bench and focus
    /// it: a double-click on its Measurements row, *Edit on Bench* in the
    /// row's menu, ticking Measurement View's *Edit* box, and the wire's
    /// `put_measurement_on_bench`.
    ///
    /// The measurement is selected first. Where an item on the bench already
    /// came from that row, that item is focused and no version is pushed, so
    /// that case is not refused while the node is busy, and its focus writes
    /// the `Editing` row. Otherwise the item is put on in one version (`Put
    /// distance #2 on the bench`), labelled by its kind and row with the
    /// collision suffix where another item holds that, and focused with no
    /// row of its own.
    ///
    /// Allowed on a view-only bench: putting a measurement on the bench is
    /// how it is viewed there, and every step that edits it is refused.
    pub(crate) fn put_measurement_on_bench(
        &mut self,
        measurement: MeasurementRef,
    ) -> Result<PutMeasurement, String> {
        let index = self.node_index(measurement.recon)?;
        if self.distance_row(measurement).is_none() {
            let node = &self.scene[index];
            let count = measurement.kind.count(node.edited().measurements());
            return Err(format!(
                "{} stores {count} {} measurement{}, so there is no {}.",
                node.label,
                measurement.kind.name(),
                if count == 1 { "" } else { "s" },
                measurement.lower_title()
            ));
        }
        self.select_measurement(measurement);
        if let Some(label) = self.bench_item_from_measurement(measurement) {
            self.focus_bench_item(measurement.recon, &label)?;
            return Ok(PutMeasurement { label, made: false });
        }
        if let Some(why) = self.busy_refusal(measurement.recon) {
            return Err(why);
        }
        let node = &self.scene[index];
        let serial = node.history.current_version().serial;
        let (next, label) = bench::put_measurement(
            node.history.current_bench(),
            node.edited(),
            measurement.kind,
            measurement.index,
            serial.as_u64(),
        )
        .map_err(|e| format!("Cannot put that measurement on the bench: {e}"))?;
        let title = measurement.lower_title();
        let text = if label == title {
            format!("Put {title} on the bench")
        } else {
            format!("Put {title} on the bench as {label}")
        };
        self.push_bench_step(index, next, text);
        self.focus_put_item(measurement.recon, &label);
        Ok(PutMeasurement { label, made: true })
    }

    /// Tick (`true`) or clear Measurement View's *Edit* box on `id`.
    ///
    /// Ticked, the selected measurement of `id` is put on the bench and
    /// focused ([`Self::put_measurement_on_bench`]); with none selected, the
    /// most recently focused measurement item still on a bench is focused,
    /// and with none of those the tick is refused naming the ways in.
    /// Cleared, the item is unfocused and stays on the bench, as clearing
    /// Track View's box leaves a track.
    pub(crate) fn set_measurement_editing(&mut self, id: ReconId, on: bool) -> Result<(), String> {
        if !on {
            self.unfocus_bench_item();
            return Ok(());
        }
        match self.selected_measurement.filter(|m| m.recon == id) {
            Some(measurement) => self.put_measurement_on_bench(measurement).map(|_| ()),
            None => {
                let (node, label) = self
                    .most_recent_item_of(ItemKind::Measurement)
                    .ok_or_else(nothing_to_edit)?;
                self.focus_bench_item(node, &label)
            }
        }
    }

    /// Why committing the measurement `label` on `id`'s bench is refused, or
    /// `None`: the node busy or a camera model proposal open on it, a
    /// view-only bench, and then core's refusal of the item it would commit
    /// (`commit_measurement_preconditions`).
    ///
    /// What greys Measurement View's *Commit*, so the button and the step give
    /// one sentence.
    pub(crate) fn measurement_commit_refusal(&self, id: ReconId, label: &str) -> Option<String> {
        if let Some(why) = self
            .bench_edit_refusal(id)
            .or_else(|| self.proposal_refusal(id))
        {
            return Some(why);
        }
        let node = self.node(id)?;
        let Some(item) = node.history.current_bench().measurement(label) else {
            return Some(format!("Nothing on the bench is called {label}."));
        };
        let origin = self.resolved_measurement_origin(node, item);
        bench::commit_measurement_preconditions(node.edited(), item, origin)
            .err()
            .map(|e| commit_refused(label, &e))
    }

    /// Write the measurement called `label` into the node's reconstruction:
    /// *Commit* in Measurement View's toolbar, and the wire's
    /// `commit_bench_measurement`.
    ///
    /// The item's origin is followed to the cursor first: where it names a
    /// row, that row is replaced, and where it names none (the item has no
    /// origin, or its row was deleted), a row is appended. It is one ordinary
    /// edit, as a track's commit is: one version stating the value and the
    /// bench, one Action Log row of kind `Edit`, and the node marked dirty.
    /// The value shares the base's points, tracks and patch bitmaps
    /// ([`sfmtool_core::EditedReconstruction::with_measurements`]), and no
    /// point index moves.
    ///
    /// The item stays on the bench and focused, its origin re-seated on the
    /// row written, which is selected. A replaced row is seated at the
    /// version the commit was computed from, by the row it held there; an
    /// appended one by the row it takes, which that version's identity map
    /// carries forward. A commit onto a row that already holds exactly what
    /// the item holds pushes no version and writes the no-effect row, and the
    /// row is still selected.
    pub(crate) fn commit_bench_measurement(
        &mut self,
        id: ReconId,
        label: &str,
    ) -> Result<MeasurementCommitted, String> {
        if let Some(why) = self.bench_edit_refusal(id) {
            return Err(why);
        }
        if let Some(why) = self.proposal_refusal(id) {
            return Err(why);
        }
        let index = self.node_index(id)?;
        let node = &self.scene[index];
        let bench = Arc::clone(node.history.current_bench());
        let item = match bench.get(label) {
            Some(BenchItem::Measurement(item)) => Arc::clone(item),
            Some(BenchItem::Track(_)) => return Err(not_a_measurement(label)),
            None => return Err(format!("Nothing on the bench is called {label}.")),
        };
        let origin = self.resolved_measurement_origin(node, &item);
        let serial = node.history.current_version().serial;
        let node_label = node.label.clone();
        let commit = bench::commit_measurement(node.edited(), &item, origin)
            .map_err(|e| commit_refused(label, &e))?;
        let written = MeasurementRef {
            recon: id,
            kind: commit.kind,
            index: commit.index,
        };
        if !commit.wrote {
            self.no_effect(commit.label(&node_label));
            self.select_measurement(written);
            return Ok(MeasurementCommitted {
                measurement: written,
                replaced: commit.replaced,
                changed: false,
            });
        }
        let seated = item.with_origin(serial.as_u64(), commit.index);
        let next_bench = bench
            .replace(label, BenchItem::Measurement(Arc::new(seated)))
            .map_err(|e| format!("Cannot install that commit: {e}"))?;
        let text = commit.label(&node_label);
        let node = &mut self.scene[index];
        let pushed = node.history.push_pair(
            Some(commit.value),
            Arc::new(next_bench),
            None,
            crate::document::PointMap::Removed(Vec::new()),
            text.clone(),
            None,
        );
        let parent = version_before(node, pushed);
        self.action_log
            .record(Kind::Edit, version_step_text(&text, parent, pushed));
        self.settle_focused_item(id);
        self.select_measurement(written);
        Ok(MeasurementCommitted {
            measurement: written,
            replaced: commit.replaced,
            changed: true,
        })
    }

    /// Delete the stored measurement `measurement`: *Delete* in a
    /// Measurements row's menu, and the wire's `delete_measurement`.
    ///
    /// A direct edit of the reconstruction, as deleting a point is, not a
    /// bench step: one version (`Deleted distance #2 in dino`), one `Edit`
    /// row, and the node marked dirty, refused while a background task or a
    /// camera model proposal holds the node. It is allowed on a view-only
    /// bench, since it edits no item. The rows after it move down by one, and
    /// the version carries that as its measurement map, which the selected
    /// measurement follows here and every measurement item's origin follows
    /// through the version graph. An item whose origin was the deleted row
    /// stays on the bench with no row to replace, so committing it adds the
    /// measurement back.
    pub(crate) fn delete_measurement(&mut self, measurement: MeasurementRef) -> Result<(), String> {
        let index = self.node_index(measurement.recon)?;
        if let Some(why) = self.edit_refusal(measurement.recon) {
            return Err(why);
        }
        let node = &self.scene[index];
        let (value, rows) =
            bench::delete_measurement(node.edited(), measurement.kind, measurement.index)
                .map_err(|e| format!("Cannot delete that measurement: {e}"))?;
        let text = format!("Deleted {} in {}", measurement.lower_title(), node.label);
        let node = &mut self.scene[index];
        let bench = Arc::clone(node.history.current_bench());
        let serial = node.history.push_renumbering(
            Some(value),
            bench,
            None,
            crate::document::PointMap::Removed(Vec::new()),
            MeasurementMap::of(measurement.kind, rows.clone()),
            text.clone(),
            None,
        );
        let parent = version_before(node, serial);
        self.follow_measurement_rows(measurement.recon, &rows);
        self.action_log
            .record(Kind::Edit, version_step_text(&text, parent, serial));
        Ok(())
    }
}

/// A commit refusal in the panel's and the wire's words.
fn commit_refused(label: &str, refusal: &bench::MeasurementCommitRefusal) -> String {
    format!("Cannot commit {label}: {refusal}.")
}

/// Measurement View's ticked box refused with no measurement selected and no
/// measurement item to go back to: the ways in.
pub(crate) fn nothing_to_edit() -> String {
    "Nothing to edit: select a measurement, double-click a row in the Scene tree's \
     Measurements group, or double-click an item in its Bench Measurements group."
        .to_string()
}
