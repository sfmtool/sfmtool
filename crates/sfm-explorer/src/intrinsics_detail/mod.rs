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
//! orchestrates one frame; the work lives in six children:
//!
//! - [`derived`] — the per-camera derived report ([`sfmtool_core::camera::report`]),
//!   computed once per camera and cached.
//! - [`header`] — the identity line and its `Copy ▾` menu.
//! - [`parameters`] — the parameter table, the derived table and `K`.
//! - [`mod@projection_plot`] — the radial map, the residual and the domain
//!   the model can be held to.
//! - [`extrinsics`] — the selected image's pose, the node-transform toggle and
//!   the rig block.
//! - [`mod@format`] — the number, matrix and vector spellings the tables share.

use std::collections::HashMap;

use crate::scene::{CameraRef, ImageRef, SceneNode};

mod derived;
mod extrinsics;
mod format;
mod header;
mod parameters;
mod projection_plot;

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
    /// Which frame the extrinsics block is showing. Sticky across selections:
    /// a user who asked for the transformed pose is comparing nodes, and
    /// resetting it under them on every click would undo the comparison.
    pose_frame: PoseFrame,
}

/// Response from the Camera Intrinsics panel.
///
/// The panel is nearly read-only — it reports only navigation, and the one
/// navigation it offers is the image name in the extrinsics header.
pub struct IntrinsicsDetailResponse {
    /// The user clicked the image name in the extrinsics header.
    pub select_image: Option<ImageRef>,
    /// Whether the pointer is currently inside the panel.
    pub has_pointer: bool,
}

impl IntrinsicsDetail {
    pub fn new() -> Self {
        Self {
            derived: HashMap::new(),
            pose_frame: PoseFrame::default(),
        }
    }

    /// Show the panel for `node`'s `selected_camera`, with `selected_image`
    /// driving the extrinsics block.
    ///
    /// Both indices are local to `node`: the dock resolves them out of
    /// [`crate::state::AppState`], which by the selection coupling guarantees
    /// that a selected image's camera is the selected camera.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        node: &SceneNode,
        selected_camera: Option<usize>,
        selected_image: Option<usize>,
    ) -> IntrinsicsDetailResponse {
        let mut response = IntrinsicsDetailResponse {
            select_image: None,
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
        let index = selected_camera.filter(|&i| i < node.recon.cameras.len());
        let Some(index) = index else {
            show_empty_state(ui);
            return response;
        };
        let reference = CameraRef::new(node.id, index);
        let camera = &node.recon.cameras[index];
        let derived = self
            .derived
            .entry(reference)
            .or_insert_with(|| Derived::compute(camera));

        // The image the extrinsics block describes. Filtered by the camera it
        // uses as well as by range: the coupling invariant already guarantees
        // it, and a panel that would otherwise print one lens's `K` beside
        // another lens's pose should not depend on that guarantee holding.
        let image = selected_image
            .filter(|&i| i < node.recon.images.len())
            .filter(|&i| node.recon.images[i].camera_index as usize == index);
        let pose = image.map(|i| extrinsics::Pose::resolve(node, i, camera, self.pose_frame));

        egui::ScrollArea::vertical()
            .id_salt("intrinsics_detail")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                header::show_header(ui, node, index, camera, pose.as_ref());
                ui.separator();
                parameters::show_parameters(ui, camera);
                ui.separator();
                parameters::show_derived(ui, camera, derived);
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

        response
    }

    /// Drop everything cached for a reconstruction that has left the scene.
    pub fn forget_recon(&mut self, id: crate::scene::ReconId) {
        self.derived.retain(|camera, _| camera.recon != id);
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
