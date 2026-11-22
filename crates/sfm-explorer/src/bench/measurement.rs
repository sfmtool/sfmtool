// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Measurements on the bench: putting a stored measurement on, following its
//! origin, the steps that set an item's label, its stated distance and an
//! endpoint, the commit that writes it back, and the direct delete of a
//! stored one.
//!
//! See `specs/gui/bench.md` and `specs/gui/measurement-view.md`. The item and
//! its pure steps are core's (`sfmtool_core::bench::measurement`,
//! `specs/core/bench/editable-measurement.md`); what this module adds is what
//! the rest of [`crate::bench`] adds for a track: the version each step is
//! pushed as, its Action Log row, and following the origin through the node's
//! version graph.

use std::sync::Arc;

use sfmtool_core::bench::{self, BenchItem, EditableMeasurement, ItemKind};
use sfmtool_core::LengthUnit;

use crate::action_log::{version_step_text, Kind};
use crate::document::MeasurementMap;
use crate::measurements::{EndpointSide, MeasurementRef};
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

/// What setting a measurement item's stated distance left
/// ([`AppState::set_bench_measurement_distance`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StatedDistanceSet {
    /// The distance and unit the item states afterwards.
    pub(crate) distance: (f64, LengthUnit),
    /// Whether a version was pushed.
    pub(crate) changed: bool,
}

/// How a step that moves an endpoint was asked for, which says how its
/// Edit History label reads ([`AppState::set_bench_measurement_endpoint`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EndpointMove {
    /// A row of the *Switch image…* popup: to the same place on the surface
    /// in another photograph, named by the two images.
    Switch,
    /// An image and a pixel given outright, named by the pixel: the wire's
    /// `set_bench_measurement_endpoint`.
    #[cfg(any(test, feature = "mcp"))]
    Set,
}

/// The unit a bare number typed as a stated distance is in: the
/// reconstruction's `world_space_unit`, or metres where it names none.
pub(crate) fn bare_unit(world_space_unit: Option<&str>) -> LengthUnit {
    world_space_unit
        .and_then(LengthUnit::from_name)
        .unwrap_or(LengthUnit::Metre)
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

    /// The node, its bench and the measurement a step on one item acts on:
    /// refused while a background task holds the node and on a view-only
    /// bench ([`Self::bench_edit_refusal`]), then for a label that names no
    /// item or names a track.
    fn measurement_step_target(
        &self,
        id: ReconId,
        label: &str,
    ) -> Result<(usize, Arc<bench::Bench>, Arc<EditableMeasurement>), String> {
        if let Some(why) = self.bench_edit_refusal(id) {
            return Err(why);
        }
        let index = self.node_index(id)?;
        let bench = Arc::clone(self.scene[index].history.current_bench());
        let item = match bench.get(label) {
            Some(BenchItem::Measurement(item)) => Arc::clone(item),
            Some(BenchItem::Track(_)) => return Err(not_a_measurement(label)),
            None => return Err(format!("Nothing on the bench is called {label}.")),
        };
        Ok((index, bench, item))
    }

    /// Replace the measurement `label` on the bench at `index` with `next`,
    /// as one bench step labelled `text`. The item keeps its ID.
    fn push_measurement_step(
        &mut self,
        index: usize,
        bench: &bench::Bench,
        label: &str,
        next: EditableMeasurement,
        text: String,
    ) -> Result<(), String> {
        let next_bench = bench
            .replace(label, BenchItem::Measurement(Arc::new(next)))
            .map_err(|e| format!("Cannot install that step: {e}"))?;
        self.push_bench_step(index, next_bench, text);
        Ok(())
    }

    /// Set the measurement's own label of the item `label` on `id`'s bench
    /// to `to`: the label field in Measurement View's edited mode, and the
    /// wire's `set_bench_measurement_label`.
    ///
    /// One bench step (`Set the label of distance #2 to "paw width"`), which
    /// leaves the reconstruction alone. A label the item holds already pushes
    /// no version and writes the no-effect row. Refused while a background
    /// task holds the node and on a view-only bench. Whether a version was
    /// pushed comes back.
    pub(crate) fn set_bench_measurement_label(
        &mut self,
        id: ReconId,
        label: &str,
        to: &str,
    ) -> Result<bool, String> {
        let (index, bench, item) = self.measurement_step_target(id, label)?;
        if item.label == to {
            self.no_effect(format!(
                "Set the label of {label} to \"{to}\": no effect, it has that label already"
            ));
            return Ok(false);
        }
        let next = bench::set_measurement_label(&item, to);
        let text = format!("Set the label of {label} to \"{to}\"");
        self.push_measurement_step(index, &bench, label, next, text)?;
        Ok(true)
    }

    /// Set the stated distance of the item `label` on `id`'s bench to what
    /// `text` says: the stated distance field in Measurement View's edited
    /// mode, and the wire's `set_bench_measurement_distance`.
    ///
    /// `text` is a number with an optional unit suffix, read by core's
    /// `parse_stated_distance` as `sfm xform --scale-by-measurements` reads a
    /// distance; a bare number is in the reconstruction's `world_space_unit`,
    /// or in metres where it has none. One bench step (`Set the stated
    /// distance of distance #2 to 4.2 cm`). A distance the item states
    /// already, as the file would store it, pushes no version and writes the
    /// no-effect row. Refused while a background task holds the node, on a
    /// view-only bench, and for text that does not read or a value that is not
    /// finite and positive, in core's sentence. What the item states
    /// afterwards, and whether a version was pushed, come back.
    pub(crate) fn set_bench_measurement_distance(
        &mut self,
        id: ReconId,
        label: &str,
        text: &str,
    ) -> Result<StatedDistanceSet, String> {
        let (index, bench, item) = self.measurement_step_target(id, label)?;
        let bare = bare_unit(
            self.scene[index]
                .recon()
                .metadata
                .world_space_unit
                .as_deref(),
        );
        let refused = |e: bench::StatedDistanceRefusal| {
            format!("Cannot set the stated distance of {label}: {e}.")
        };
        let (distance, unit) = bench::parse_stated_distance(text, bare).map_err(refused)?;
        let stated = format!(
            "{} {}",
            crate::measurements::length_number(distance),
            unit.name()
        );
        let held = item.as_distance().and_then(|d| d.distance);
        if held.is_some_and(|(d, u)| u == unit && d as f32 == distance as f32) {
            self.no_effect(format!(
                "Set the stated distance of {label} to {stated}: no effect, it states that already"
            ));
            return Ok(StatedDistanceSet {
                distance: held.expect("just compared"),
                changed: false,
            });
        }
        let next = bench::set_stated_distance(&item, distance, unit).map_err(refused)?;
        let text = format!("Set the stated distance of {label} to {stated}");
        self.push_measurement_step(index, &bench, label, next, text)?;
        Ok(StatedDistanceSet {
            distance: (distance, unit),
            changed: true,
        })
    }

    /// Move endpoint `side` of the item `label` on `id`'s bench to pixel
    /// `pixel` of image `image`: a row of Measurement View's *Switch image…*
    /// popup (`EndpointMove::Switch`), and the wire's
    /// `set_bench_measurement_endpoint` (`EndpointMove::Set`).
    ///
    /// One bench step, made by core's `set_measurement_endpoint`, which leaves
    /// the reconstruction alone: `Moved endpoint A of distance #2 from
    /// IMG_0004.jpg to IMG_0011.jpg` for a switch, `Moved endpoint A of
    /// distance #2 to IMG_0004.jpg (412.3, 230.5)` for a set, and `Set
    /// endpoint B of distance #2 to IMG_0050.jpg (88.0, 301.5)` for a set of
    /// an endpoint not placed. An endpoint at that image and pixel already,
    /// as the file would store the pixel, pushes no version and writes the
    /// no-effect row. Refused while a background task holds the node, on a
    /// view-only bench, for a switch of an endpoint not placed, and for an
    /// image past the image set or a pixel outside its image, in core's
    /// sentence. Whether a version was pushed comes back.
    pub(crate) fn set_bench_measurement_endpoint(
        &mut self,
        id: ReconId,
        label: &str,
        side: EndpointSide,
        image: u32,
        pixel: [f64; 2],
        how: EndpointMove,
    ) -> Result<bool, String> {
        let (index, bench, item) = self.measurement_step_target(id, label)?;
        let letter = side.letter();
        let held = item.as_distance().and_then(|d| d.endpoints[side.index()]);
        let recon = self.scene[index].recon();
        let name = |image: u32| crate::measurements::image_name(recon, image);
        #[cfg(any(test, feature = "mcp"))]
        let place = format!("{} ({:.1}, {:.1})", name(image), pixel[0], pixel[1]);
        let text = match (how, held) {
            (EndpointMove::Switch, None) => {
                return Err(format!(
                    "Cannot switch endpoint {letter} of {label}: it is not placed, so it has \
                     no image to switch from."
                ))
            }
            (EndpointMove::Switch, Some(from)) => format!(
                "Moved endpoint {letter} of {label} from {} to {}",
                name(from.image),
                name(image)
            ),
            #[cfg(any(test, feature = "mcp"))]
            (EndpointMove::Set, Some(_)) => {
                format!("Moved endpoint {letter} of {label} to {place}")
            }
            #[cfg(any(test, feature = "mcp"))]
            (EndpointMove::Set, None) => format!("Set endpoint {letter} of {label} to {place}"),
        };
        let as_stored = |p: [f64; 2]| [p[0] as f32, p[1] as f32];
        if held.is_some_and(|e| e.image == image && as_stored(e.pixel) == as_stored(pixel)) {
            self.no_effect(format!("{text}: no effect, the endpoint is there already"));
            return Ok(false);
        }
        let endpoint = bench::MeasurementEndpoint { image, pixel };
        let next = bench::set_measurement_endpoint(
            self.scene[index].edited(),
            &item,
            side.index(),
            endpoint,
        )
        .map_err(|e| format!("Cannot move endpoint {letter} of {label}: {e}."))?;
        self.push_measurement_step(index, &bench, label, next, text)?;
        Ok(true)
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
