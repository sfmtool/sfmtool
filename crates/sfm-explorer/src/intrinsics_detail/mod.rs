// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Camera Intrinsics panel — what the selected camera *is*.
//!
//! The viewer can show where a camera sits and what it saw; this panel is the
//! one place that says what it **is**: which model, what focal length, how far
//! the principal point sits from the image centre, how many degrees the frame
//! subtends — and, when an image is selected too, that image's pose in the
//! frame a user would paste into their own code.
//!
//! The panel is a detail view of a selection, like its two dock neighbours, and
//! it is driven by [`crate::state::AppState::selected_camera`]. This module
//! owns the panel state and the [`IntrinsicsDetail::show`] entry point that
//! orchestrates one frame; the work lives in seven children:
//!
//! - [`derived`] — the per-camera derived report ([`sfmtool_core::camera::report`]),
//!   computed once per camera and cached.
//! - [`header`] — the identity line, its `Switch Model…` button and its
//!   `Copy ▾` menu.
//! - [`proposal`] — the camera model proposal's strip: its controls, the
//!   fit's report, Apply and Cancel. While a proposal is open on the camera
//!   shown, the parameter and derived tables show the current camera and the
//!   proposed one side by side.
//! - [`parameters`] — the parameter table, the derived table and `K`.
//! - [`mod@projection_plot`] — the radial map, the residual and the domain
//!   the model can be held to.
//! - [`extrinsics`] — the selected image's pose, the node-transform toggle and
//!   the rig block.
//! - [`mod@format`] — the number, matrix and vector spellings the tables share.

use std::collections::HashMap;

use crate::camera_model_proposal::CameraModelProposal;
use crate::scene::{CameraRef, ImageRef, SceneNode};

mod derived;
mod extrinsics;
mod format;
mod header;
mod parameters;
mod projection_plot;
pub(crate) mod proposal;

pub(crate) use proposal::ProposalAction;

#[cfg(test)]
mod tests;

use derived::Derived;

/// Which frame the extrinsics block reads the pose in.
///
/// A node can carry a similarity from an in-GUI `Align to…`, so what the
/// viewport draws is not what the file holds. The panel defaults to what the
/// file holds — that is what a user comparing against `sfm inspect` expects —
/// and offers the other explicitly rather than silently picking one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PoseFrame {
    /// The pose as stored in the `.sfmr`.
    #[default]
    Stored,
    /// The stored pose put through the node's transform.
    NodeTransform,
}

/// Camera Intrinsics panel state.
pub struct IntrinsicsDetail {
    /// The derived report per camera. Keyed by [`CameraRef`], so a second
    /// reconstruction reusing camera index 0 gets its own entry; dropped
    /// wholesale by [`IntrinsicsDetail::forget_recon`] when a node leaves.
    ///
    /// Bounded by the number of *distinct* intrinsics across the loaded nodes,
    /// which is small even for a per-image-intrinsics solve, so there is no
    /// eviction policy beyond that.
    derived: HashMap<CameraRef, Derived>,
    /// The derived report of the open proposal's camera, keyed by the
    /// camera and the proposal's generation, so the current camera's entry
    /// above is never served for the proposed one, nor the other way round.
    proposed_derived: Option<(CameraRef, u64, Derived)>,
    /// Which frame the extrinsics block is showing. Sticky across selections:
    /// a user who asked for the transformed pose is comparing nodes, and
    /// resetting it under them on every click would undo the comparison.
    pose_frame: PoseFrame,
}

/// Response from the Camera Intrinsics panel.
///
/// The panel is nearly read-only — it reports navigation, the image name in
/// the extrinsics header, and the one edit it offers: `Switch Model…`, which
/// the dock turns into a camera model proposal, and the proposal's Apply and
/// Cancel. A change of a proposal's controls is made on the proposal in place.
pub struct IntrinsicsDetailResponse {
    /// The user clicked the image name in the extrinsics header.
    pub select_image: Option<ImageRef>,
    /// The user clicked `Switch Model…` for this camera.
    pub(crate) switch_model: Option<CameraRef>,
    /// The user ended the proposal shown: Apply, or Cancel or `Escape`.
    pub(crate) proposal: Option<(CameraRef, ProposalAction)>,
    /// Whether the pointer is currently inside the panel.
    pub has_pointer: bool,
}

impl IntrinsicsDetail {
    pub fn new() -> Self {
        Self {
            derived: HashMap::new(),
            proposed_derived: None,
            pose_frame: PoseFrame::default(),
        }
    }

    /// Show the panel for `node`'s `selected_camera`, with `selected_image`
    /// driving the extrinsics block.
    ///
    /// Both indices are local to `node`: the dock resolves them out of
    /// [`crate::state::AppState`], which by the selection coupling guarantees
    /// that a selected image's camera is the selected camera. `proposal` is the
    /// open camera model proposal, drawn when it is for the camera shown, and
    /// `switch_refusal` is why `Switch Model…` is greyed for that camera.
    pub(crate) fn show(
        &mut self,
        ui: &mut egui::Ui,
        node: &SceneNode,
        selected_camera: Option<usize>,
        selected_image: Option<usize>,
        proposal: Option<&mut CameraModelProposal>,
        switch_refusal: Option<&str>,
    ) -> IntrinsicsDetailResponse {
        let mut response = IntrinsicsDetailResponse {
            select_image: None,
            switch_model: None,
            proposal: None,
            has_pointer: false,
        };

        let panel_rect = ui.available_rect_before_wrap();
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            if panel_rect.contains(pos) {
                response.has_pointer = true;
            }
        }

        // A stale index left over from a node that has since shrunk is the
        // same nothing as no selection at all.
        let index = selected_camera.filter(|&i| i < node.recon().image_table.cameras.len());
        let Some(index) = index else {
            show_empty_state(ui);
            return response;
        };
        let reference = CameraRef::new(node.id, index);
        let camera = &node.recon().image_table.cameras[index];
        let derived = self
            .derived
            .entry(reference)
            .or_insert_with(|| Derived::compute(camera));

        // The image the extrinsics block describes. Filtered by the camera it
        // uses as well as by range: the coupling invariant already guarantees
        // it, and a panel that would otherwise print one lens's `K` beside
        // another lens's pose should not depend on that guarantee holding.
        let image = selected_image
            .filter(|&i| i < node.recon().image_table.images.len())
            .filter(|&i| node.recon().image_table.images[i].camera_index as usize == index);
        let pose = image.map(|i| extrinsics::Pose::resolve(node, i, camera, self.pose_frame));
        let mut proposal = proposal.filter(|p| p.camera_ref() == reference);
        // Greyed while this camera's own proposal is open below the button: a
        // second open would be the one already there.
        let switch_refusal = match (switch_refusal, &proposal) {
            (Some(why), _) => Some(why.to_string()),
            (None, Some(_)) => Some("The proposal for this camera is open below.".to_string()),
            (None, None) => None,
        };
        let proposed_derived = &mut self.proposed_derived;

        egui::ScrollArea::vertical()
            .id_salt("intrinsics_detail")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if header::show_header(
                    ui,
                    node,
                    index,
                    camera,
                    pose.as_ref(),
                    switch_refusal.as_deref(),
                ) {
                    response.switch_model = Some(reference);
                }
                ui.separator();
                let mut proposed_camera = None;
                if let Some(proposal) = proposal.as_deref_mut() {
                    if let Some(action) = proposal::show_proposal(ui, proposal) {
                        response.proposal = Some((reference, action));
                    }
                    ui.separator();
                    if let Ok(refit) = &proposal.fit {
                        let generation = proposal.generation;
                        let stale = !matches!(
                            proposed_derived,
                            Some((r, g, _)) if *r == reference && *g == generation
                        );
                        if stale {
                            *proposed_derived =
                                Some((reference, generation, Derived::compute(&refit.camera)));
                        }
                        proposed_camera = Some(refit.camera.clone());
                    }
                }
                match (proposed_camera.as_ref(), proposed_derived.as_ref()) {
                    (Some(proposed), Some((_, _, proposed_report))) => {
                        let aspect_row = parameters::has_two_focal_lengths(camera)
                            || parameters::has_two_focal_lengths(proposed);
                        ui.columns(2, |columns| {
                            columns[0].label(egui::RichText::new("Current").weak());
                            parameters::show_parameters(&mut columns[0], camera, "current");
                            columns[1].label(egui::RichText::new("Proposed").weak());
                            parameters::show_parameters(&mut columns[1], proposed, "proposed");
                        });
                        ui.separator();
                        ui.columns(2, |columns| {
                            parameters::show_derived(
                                &mut columns[0],
                                camera,
                                derived,
                                "current",
                                aspect_row,
                            );
                            parameters::show_derived(
                                &mut columns[1],
                                proposed,
                                proposed_report,
                                "proposed",
                                aspect_row,
                            );
                        });
                    }
                    _ => {
                        parameters::show_parameters(ui, camera, "current");
                        ui.separator();
                        parameters::show_derived(ui, camera, derived, "current", false);
                    }
                }
                ui.separator();
                parameters::show_k(ui, camera);
                ui.separator();
                projection_plot::show_projection_plot(ui, camera, derived);
                if let (Some(image), Some(pose)) = (image, pose) {
                    ui.separator();
                    let picked =
                        extrinsics::show_extrinsics(ui, node, image, &pose, &mut self.pose_frame);
                    if picked {
                        response.select_image = Some(ImageRef::new(node.id, image));
                    }
                }
            });

        // `Escape` cancels the proposal while the pointer is over the panel
        // showing it, so the key keeps its meaning everywhere else.
        if response.has_pointer
            && response.proposal.is_none()
            && proposal.is_some()
            && !ui.ctx().egui_wants_keyboard_input()
            && ui.input(|i| i.key_pressed(egui::Key::Escape))
        {
            response.proposal = Some((reference, ProposalAction::Cancel));
        }

        response
    }

    /// Drop everything cached for a reconstruction that has left the scene.
    pub fn forget_recon(&mut self, id: crate::scene::ReconId) {
        self.derived.retain(|camera, _| camera.recon != id);
        if self
            .proposed_derived
            .as_ref()
            .is_some_and(|(camera, _, _)| camera.recon == id)
        {
            self.proposed_derived = None;
        }
    }
}

/// The "nothing selected" state.
///
/// The second line names the discoverable route rather than only the direct
/// one: most users reach a set of intrinsics through an image they were
/// already looking at, not by going to the tree to pick a lens.
fn show_empty_state(ui: &mut egui::Ui) {
    ui.centered_and_justified(|ui| {
        ui.vertical_centered(|ui| {
            ui.label("No camera selected");
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(
                    "Select a camera under Camera Intrinsics in the Scene panel, \
                     or select an image.",
                )
                .weak(),
            );
        });
    });
}
