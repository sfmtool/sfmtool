// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The panel's *Filter the 3D view* line and what it publishes to the 3D view.
//!
//! See `specs/gui/track-table.md` § "Filtering the 3D view" and
//! [`crate::filter_3d`]. The box and its *Dim* / *Hide* choice are panel state
//! beside the query. While the box is ticked and the panel is open, the panel
//! publishes the row list's points as a [`KeptPoints`] set, built again only
//! when the row list is computed again, for the version the rows were computed
//! for. While the panel is open behind another tab it keeps the row list and
//! the table job current for the filter's sake ([`TrackTablePanel::after_dock`]).

use std::sync::Arc;

use crate::filter_3d::{Filter3d, Filter3dMode, KeptPoints};
use crate::scene::{ReconId, SceneNode};
use crate::state::AppState;

use super::TrackTablePanel;

/// The label of the box, which is also its name in `get_widgets`.
pub(super) const FILTER_3D: &str = "Filter the 3D view";
/// The label of the choice that dims the rejected points.
pub(super) const DIM: &str = "Dim";
/// The label of the choice that hides them.
pub(super) const HIDE: &str = "Hide";

/// The box's hover text, naming the reconstruction it filters.
fn filter_hover(label: &str) -> String {
    format!(
        "Apply the bounds and the selected-image box to {label}'s points in the 3D Viewer: \
         the points this table does not list are dimmed or hidden there, and the stats line \
         says how many are shown. The selected and hovered points are always drawn in full. \
         After an edit every point is drawn until the new rows are listed."
    )
}

impl TrackTablePanel {
    /// The second header line: the box and the two choices.
    pub(super) fn show_filter_line(&mut self, ui: &mut egui::Ui, node: &SceneNode) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.query.filter_3d, FILTER_3D)
                .on_hover_text(filter_hover(&node.label));
            ui.radio_value(&mut self.query.filter_3d_mode, Filter3dMode::Dim, DIM)
                .on_hover_text(
                    "Draw the points the table does not list faintly, behind none of the \
                     listed ones. They can still be picked.",
                );
            ui.radio_value(&mut self.query.filter_3d_mode, Filter3dMode::Hide, HIDE)
                .on_hover_text(
                    "Leave the points the table does not list out of the 3D Viewer: they are \
                     not drawn and cannot be picked.",
                );
        });
    }

    /// Say what the 3D view should show for the node `id`, which the panel
    /// shows: nothing while the box is clear, else the row list's points when
    /// the rows are of the version at the node's cursor, and no set while
    /// they are not, so the 3D view never filters by another version's
    /// indexes.
    pub(super) fn publish_filter_3d(&mut self, state: &mut AppState, id: ReconId) {
        if !self.query.filter_3d {
            self.kept = None;
            state.filter_3d = None;
            return;
        }
        let document = state
            .node(id)
            .map(|node| node.history.current_version().document_serial);
        let current = self
            .rows
            .inputs
            .as_ref()
            .filter(|inputs| {
                inputs.node == id && Some(inputs.document) == document && self.rows.error.is_none()
            })
            .map(|inputs| inputs.document);
        let kept = current.map(|document| {
            let recomputed = self.rows.recomputed;
            let built = self
                .kept
                .as_ref()
                .filter(|(node, at, _)| *node == id && *at == recomputed)
                .map(|(_, _, kept)| Arc::clone(kept));
            let kept = built.unwrap_or_else(|| {
                let kept = Arc::new(KeptPoints::from_points(&self.rows.rows));
                self.kept = Some((id, recomputed, Arc::clone(&kept)));
                kept
            });
            (document, kept)
        });
        state.filter_3d = Some(Filter3d {
            node: id,
            mode: self.query.filter_3d_mode,
            kept,
        });
    }

    /// Called by the frame after the dock has drawn, with whether the panel
    /// is open in it.
    ///
    /// A closed panel filters nothing, though its box stays ticked for when
    /// it opens again. An open panel that did not draw this frame, behind
    /// another tab of its node, still owes the 3D view its filter: so while
    /// the box is ticked it computes the row list again when its inputs
    /// change and asks for the columns its query needs, as a drawn one does,
    /// and publishes the set.
    pub(crate) fn after_dock(&mut self, state: &mut AppState, open: bool) {
        let drawn = std::mem::take(&mut self.drawn_this_frame);
        if !open || !self.query.filter_3d {
            self.kept = None;
            state.filter_3d = None;
            return;
        }
        if drawn {
            return;
        }
        let Some(id) =
            crate::scene::selected_node(&state.scene, state.selected_recon).map(|n| n.id)
        else {
            state.filter_3d = None;
            return;
        };
        let refreshed = self.refresh_rows(state, id);
        self.scroll_to_top |= refreshed.reordered;
        state.want_track_table_columns(
            id,
            self.query.needed_computed(),
            self.query.ordering_computed(),
        );
        self.publish_filter_3d(state, id);
    }
}
