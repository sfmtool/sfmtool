// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Track View panel: one 3D point's track, looked at or worked on.
//!
//! See `specs/gui/track-view.md`. The panel's first row is the **Edit**
//! checkbox, and it is a reading of the focused item rather than a setting: it
//! is ticked exactly while the focused item is on the selected node's bench
//! ([`AppState::focused_item_label`]). Below it the panel draws one of two
//! bodies:
//!
//! - [`view`], with the box clear: the selected point's committed track, read
//!   from the reconstruction and never from the bench.
//! - [`edit`], with the box ticked: the focused item and the steps that act on
//!   it.
//!
//! The two bodies keep their own state, because what each caches is disjoint:
//! one caches thumbnails and patch tiles per image of a committed point, the
//! other caches tiles, painting and the commit refusal against a bench track's
//! `Arc`. What this module adds is the checkbox, the dispatch on it, and the
//! line about the bench under view mode's empty state.
//!
//! While an item is focused, the selected point is its origin or no point
//! (`AppState::select_point` unfocuses the item on any other), so edit mode
//! never shows one track while the selection names another.
//!
//! The panel decides nothing. Ticking the box and clearing it land in
//! [`TrackViewResponse`], and the dock applies them through
//! `AppState`, for the reason every panel's response works that way: the panel
//! holds `&AppState` while it draws.

pub(crate) mod edit;
mod header_buttons;
pub(crate) mod view;

#[cfg(test)]
mod tests;

use crate::platform::{self, GestureEvent};
use crate::state::AppState;

pub(crate) use edit::{TrackEdit, TrackEditResponse};
pub(crate) use view::{PointTrackView, PointTrackViewResponse};

/// The checkbox's label, and what the sentences that point at it quote.
pub(crate) const EDIT_LABEL: &str = "Edit";

/// The mark both headers draw left of the point's ID when the point is at
/// infinity (U+221E), so a direction is told from a place before its numbers
/// are read.
pub(crate) const INFINITY: &str = "\u{221e}";

/// The hover text on [`INFINITY`].
pub(crate) const INFINITY_HOVER: &str =
    "At infinity: the point is a direction, not a place, and its three numbers are a unit \
     bearing";

/// Draw [`INFINITY`] with its hover text, in the size of the ID beside it.
pub(crate) fn infinity_mark(ui: &mut egui::Ui) {
    ui.label(egui::RichText::new(INFINITY).strong())
        .on_hover_text(INFINITY_HOVER);
}

/// What one frame of Track View asks the dock to do.
///
/// Only the mode that was drawn reports: `view` is `Some` in view mode and
/// `edit` in edit mode, and both are `None` with no reconstruction selected.
#[derive(Default)]
pub(crate) struct TrackViewResponse {
    /// The box was ticked (`true`, from view mode) or cleared (`false`, from
    /// edit mode).
    pub set_edit: Option<bool>,
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
        let focused = state.focused_item_label(id);
        let editing = focused.is_some();
        // A selection the version no longer holds is no selection: the view
        // body filters the same way, so the box and the body agree.
        let selected_point = state
            .selected_point_in(id)
            .filter(|&index| node.edited().point(index as u32).is_some());

        // The box. Only a tick over a point not yet on the bench is a bench
        // step, so only that is greyed by a task holding the node, with the
        // node's own sentence; focusing an item already on the bench and
        // unfocusing push no version. With no point selected the tick focuses
        // the most recent item still on a bench, and only with none is the
        // box greyed, saying what the ways in are.
        let recent = state.most_recent_item();
        let refusal = if editing {
            None
        } else {
            match selected_point {
                None => recent.is_none().then(nothing_to_edit),
                Some(point) => put_refusal(state, crate::scene::PointRef::new(id, point)),
            }
        };
        let mut checked = editing;
        let hint = match (editing, selected_point, &recent) {
            (true, _, _) => "Stop editing this track: it stays on the bench, and the panel \
                             shows the point it came from, when it has one"
                .to_string(),
            (false, None, Some((_, label))) => again_hint(label),
            (false, _, _) => "Put the selected point's track on the bench and edit it".to_string(),
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

        if editing {
            response.edit = Some(self.edit.show(ui, state));
        } else {
            let recent_label = recent.as_ref().map(|(_, label)| label.as_str());
            let note = bench_note(bench.len(), recent_label);
            let point_id = selected_point
                .map(|index| crate::scene::point_id(node, index))
                .unwrap_or_default();
            response.view = Some(self.view.show(
                ui,
                node.edited(),
                id,
                node.display_thumbnails.as_ref(),
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
}

/// Why ticking *Edit* over `point` is refused, or `None`: a task holding the
/// node, when the tick would put the point on the bench. A point an item on
/// the bench already came from is focused instead, which no task refuses.
fn put_refusal(state: &AppState, point: crate::scene::PointRef) -> Option<String> {
    if state.bench_item_from_point(point).is_some() {
        return None;
    }
    state.busy_refusal(point.recon)
}

/// The ticked box's refusal with no point selected and no item to go back to:
/// the three ways in, the last quoted from the Image Detail entry's own
/// constant.
pub(crate) fn nothing_to_edit() -> String {
    format!(
        "Nothing to edit: select a point, double-click an item in the Scene tree's \
         Bench groups, or right-click a pixel in Image Detail and choose \"{}\".",
        crate::image_detail::START_CLUSTER_LABEL
    )
}

/// The box's hover text with no point selected, naming the item a tick
/// focuses, so a panel with no room to show it elsewhere still says what the
/// tick does.
pub(crate) fn again_hint(label: &str) -> String {
    format!("Tick to edit {label} again")
}

/// The line under view mode's empty state: the item a tick of *Edit* goes back
/// to when there is one, else what the bench holds when it holds anything, and
/// otherwise the way to start a track from a pixel.
fn bench_note(items: usize, recent: Option<&str>) -> String {
    if let Some(label) = recent {
        return format!(
            "Tick {EDIT_LABEL} to edit {label} again, or double-click an item in the Scene \
             tree's Bench groups."
        );
    }
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
