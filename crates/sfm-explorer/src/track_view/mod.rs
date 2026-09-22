// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Track View panel: one 3D point's track, looked at or worked on.
//!
//! See `specs/gui/track-view.md`. The panel's first row is the **Edit**
//! checkbox, and it is a reading of the bench rather than a setting: it is
//! ticked exactly while the selected node's bench has an active track
//! ([`crate::bench::active_track_label`]). Below it the panel draws one of two
//! bodies:
//!
//! - [`view`], with the box clear: the selected point's committed track, read
//!   from the reconstruction and never from the bench.
//! - [`edit`], with the box ticked: the bench's active track and the steps that
//!   act on it.
//!
//! The two bodies keep their own state, because what each caches is disjoint:
//! one caches thumbnails and patch tiles per image of a committed point, the
//! other caches tiles, painting and the commit refusal against a bench track's
//! `Arc`. What this module adds is the checkbox, the dispatch on it, the
//! selection notice that says the selection and the edited track have parted,
//! and the line about the bench under view mode's empty state.
//!
//! The panel decides nothing. Ticking the box, clearing it and the notice's two
//! buttons land in [`TrackViewResponse`], and the dock applies them through the
//! `AppState` bench steps, for the reason every panel's response works that
//! way: the panel holds `&AppState` while it draws.

pub(crate) mod edit;
pub(crate) mod view;

#[cfg(test)]
mod tests;

use crate::platform::{self, GestureEvent};
use crate::scene::SceneNode;
use crate::state::AppState;

pub(crate) use edit::{TrackEdit, TrackEditResponse};
pub(crate) use view::{PointTrackView, PointTrackViewResponse};

/// The checkbox's label, and what the sentences that point at it quote.
pub(crate) const EDIT_LABEL: &str = "Edit";

/// What one frame of Track View asks the dock to do.
///
/// Only the mode that was drawn reports: `view` is `Some` in view mode and
/// `edit` in edit mode, and both are `None` with no reconstruction selected.
#[derive(Default)]
pub(crate) struct TrackViewResponse {
    /// The box was ticked (`true`, from view mode) or cleared (`false`, from
    /// edit mode), or the selection notice's *View* was pressed (`false`).
    pub set_edit: Option<bool>,
    /// The selection notice's *Edit it*: put the selected point on the bench.
    pub edit_selected_point: bool,
    /// What view mode reported, when it was the mode drawn.
    pub view: Option<PointTrackViewResponse>,
    /// What edit mode reported, when it was the mode drawn.
    pub edit: Option<TrackEditResponse>,
}

/// Track View panel state: the two bodies it is made of.
pub(crate) struct TrackView {
    /// View mode.
    view: PointTrackView,
    /// Edit mode.
    edit: TrackEdit,
}

impl Default for TrackView {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackView {
    pub(crate) fn new() -> Self {
        Self {
            view: PointTrackView::new(),
            edit: TrackEdit::new(),
        }
    }

    /// Edit mode's state, for the tests that read what it drew.
    #[cfg(test)]
    pub(crate) fn edit_body(&self) -> &TrackEdit {
        &self.edit
    }

    /// Select one observation row of the edited track from outside the panel.
    /// See [`TrackEdit::select_row`].
    pub(crate) fn select_row(
        &mut self,
        id: crate::scene::ReconId,
        label: &str,
        observation: usize,
    ) {
        self.edit.select_row(id, label, observation);
    }

    /// The one observation row selected on `label`'s track of `id`, when
    /// exactly one is. See [`TrackEdit::selected_row`].
    pub(crate) fn selected_row(&self, id: crate::scene::ReconId, label: &str) -> Option<usize> {
        self.edit.selected_row(id, label)
    }

    /// Whether edit mode's *Lock* box is ticked, which Image Detail reads a
    /// track-stage dot drag by. See [`TrackEdit::lock`].
    pub(crate) fn lock(&self) -> bool {
        self.edit.lock()
    }

    /// Drop everything either body cached for a reconstruction that has left
    /// the scene, or whose value an edit has replaced.
    pub(crate) fn forget_recon(&mut self, id: crate::scene::ReconId) {
        self.view.forget_recon(id);
        self.edit.forget_recon(id);
    }

    /// Draw the panel and report what the user did with it.
    pub(crate) fn show(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        gesture_events: &[GestureEvent],
        scroll_input: &platform::ScrollInput,
    ) -> TrackViewResponse {
        let mut response = TrackViewResponse::default();
        let Some(node) = crate::scene::selected_node(&state.scene, state.selected_recon) else {
            ui.centered_and_justified(|ui| {
                ui.label("No reconstruction loaded");
            });
            return response;
        };
        let id = node.id;
        let bench = node.history.current_bench();
        let active = crate::bench::active_track_label(bench);
        let editing = active.is_some();
        // A selection the version no longer holds is no selection: the view
        // body filters the same way, so the box and the body agree.
        let selected_point = state
            .selected_point_in(id)
            .filter(|&index| node.edited().point(index as u32).is_some());

        // The box. Every transition it makes is a bench step, so a task holding
        // the node greys it with the node's own sentence; with nothing to edit
        // and nothing to put on, it says what the ways in are.
        let refusal = state
            .busy_refusal(id)
            .or_else(|| (!editing && selected_point.is_none()).then(nothing_to_edit));
        let mut checked = editing;
        let hint = if editing {
            "Stop editing this track: it stays on the bench, and the panel shows the \
             selected point's committed track"
        } else {
            "Put the selected point's track on the bench and edit it"
        };
        let checkbox = ui.add_enabled(
            refusal.is_none(),
            egui::Checkbox::new(&mut checked, EDIT_LABEL),
        );
        let checkbox = match &refusal {
            Some(why) => checkbox.on_disabled_hover_text(why),
            None => checkbox.on_hover_text(hint),
        };
        if checkbox.changed() {
            response.set_edit = Some(checked);
        }
        ui.separator();

        if let Some(label) = active {
            let track = bench.track(label).expect("the active label names a track");
            self.show_selection_notice(ui, state, node, track, selected_point, &mut response);
            response.edit = Some(self.edit.show(ui, state));
        } else {
            let note = bench_note(bench.len());
            let point_id = selected_point
                .map(|index| crate::scene::point_id(node, index))
                .unwrap_or_default();
            response.view = Some(self.view.show(
                ui,
                node.edited(),
                id,
                &point_id,
                selected_point,
                state.hovered_image_in(id),
                &state.sift_cache,
                &state.full_res_cache,
                gesture_events,
                scroll_input,
                Some(&note),
            ));
        }
        response
    }

    /// The line drawn in edit mode when the node has a selected point that is
    /// not the one the edited track came from, followed to the cursor.
    ///
    /// Editing is sticky: selecting another point moves the selection and
    /// leaves the panel on the item, so this line is what says the two have
    /// parted, and its two buttons are the two things wanted next.
    fn show_selection_notice(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        node: &SceneNode,
        track: &sfmtool_core::bench::EditableTrack,
        selected_point: Option<usize>,
        response: &mut TrackViewResponse,
    ) {
        let Some(point) = selected_point else {
            return;
        };
        if state.resolved_origin(node, track) == Some(point as u32) {
            return;
        }
        let busy = state.busy_refusal(node.id);
        ui.horizontal_wrapped(|ui| {
            ui.label(format!(
                "Selected: {}, not the track being edited.",
                crate::scene::point_id(node, point)
            ));
            let view = ui.add_enabled(busy.is_none(), egui::Button::new(VIEW_LABEL));
            let view = match &busy {
                Some(why) => view.on_disabled_hover_text(why),
                None => view
                    .on_hover_text("Stop editing, and show the selected point's committed track"),
            };
            if view.clicked() {
                response.set_edit = Some(false);
            }
            let edit_it = ui.add_enabled(busy.is_none(), egui::Button::new(EDIT_IT_LABEL));
            let edit_it = match &busy {
                Some(why) => edit_it.on_disabled_hover_text(why),
                None => edit_it.on_hover_text("Put the selected point on the bench and edit it"),
            };
            if edit_it.clicked() {
                response.edit_selected_point = true;
            }
        });
        ui.separator();
    }
}

/// The selection notice's button that clears *Edit*.
pub(crate) const VIEW_LABEL: &str = "View";

/// The selection notice's button that puts the selected point on the bench.
pub(crate) const EDIT_IT_LABEL: &str = "Edit it";

/// The ticked box's refusal with no point selected and nothing active: the
/// three ways in, the last quoted from the Image Detail entry's own constant.
pub(crate) fn nothing_to_edit() -> String {
    format!(
        "Nothing to edit: select a point, double-click an item in the Scene tree's \
         Bench groups, or right-click a pixel in Image Detail and choose \"{}\".",
        crate::image_detail::START_CLUSTER_LABEL
    )
}

/// The line under view mode's empty state: what the bench holds when it holds
/// anything, and otherwise the way to start a track from a pixel.
fn bench_note(items: usize) -> String {
    match items {
        0 => format!(
            "To work on a track from a pixel: right-click it in the Image Detail panel \
             and choose \"{}\".",
            crate::image_detail::START_CLUSTER_LABEL
        ),
        1 => "1 item on the bench: double-click it in the Scene tree's Bench groups to edit \
              it."
        .to_string(),
        n => format!(
            "{n} items on the bench: double-click one in the Scene tree's Bench groups to \
             edit it."
        ),
    }
}
