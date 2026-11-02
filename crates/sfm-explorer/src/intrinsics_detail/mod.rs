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
//! orchestrates one frame; the work lives in nine children:
//!
//! - [`derived`] — the per-camera derived report ([`sfmtool_core::camera::report`]),
//!   computed once per camera and cached.
//! - [`header`] — the identity line, its `Switch Model…` button and its
//!   `Copy ▾` menu.
//! - [`proposal`] — the camera model proposal's strip: its controls, the
//!   fit's report, Apply and Cancel. While a proposal is open on the camera
//!   shown, the parameter and derived tables show the current camera and the
//!   proposed one side by side.
//! - [`parameters`] — the parameter table, with the statistics of the
//!   camera's per-image values beside the parameters they adjust, the derived
//!   table and `K`.
//! - [`histogram`] — the histogram of each per-image component's values that
//!   the parameter table draws beside its statistics, and its bins, which MCP
//!   reports too.
//! - [`mod@projection_plot`] — the radial map, the residual and the domain
//!   the model can be held to; while a proposal is open, both cameras drawn
//!   against each other, the change plot and the observation rug.
//! - [`plot`] — what the projection plot and the histograms share: the
//!   palette, the linear scale and the 1/2/5 tick ladder.
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
pub(crate) mod histogram;
mod parameters;
mod plot;
mod projection_plot;
pub(crate) mod proposal;

pub(crate) use proposal::ProposalAction;

#[cfg(test)]
mod tests;

use derived::{Derived, ProposedReport};

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
pub(crate) struct IntrinsicsDetail {
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
    proposed_derived: Option<(CameraRef, u64, ProposedReport)>,
    /// Whether a field was being edited when the last frame ended. egui drops
    /// the keyboard focus on an unmodified `Escape` as the next frame begins,
    /// before any widget draws, so that frame can no longer see the field the
    /// key was for; this is how it still knows.
    editing_last_frame: bool,
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
pub(crate) struct IntrinsicsDetailResponse {
    /// The user clicked the image name in the extrinsics header, or a bin of
    /// a per-image histogram, which selects one of the bin's images.
    pub(crate) select_image: Option<ImageRef>,
    /// The image a click on the per-image histogram bin under the pointer
    /// would select, for the other panels to highlight.
    pub(crate) hovered_image: Option<ImageRef>,
    /// The user clicked `Switch Model…` for this camera.
    pub(crate) switch_model: Option<CameraRef>,
    /// The user ended the proposal shown: Apply, or Cancel or `Escape`.
    pub(crate) proposal: Option<(CameraRef, ProposalAction)>,
    /// Whether the pointer is currently inside the panel.
    pub(crate) has_pointer: bool,
}

impl IntrinsicsDetail {
    pub(crate) fn new() -> Self {
        Self {
            derived: HashMap::new(),
            proposed_derived: None,
            editing_last_frame: false,
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
            hovered_image: None,
            switch_model: None,
            proposal: None,
            has_pointer: false,
        };

        // Read before anything is drawn: an `Escape` that closes the Model
        // dropdown closes it during this frame, and by the end of it nothing
        // would show that the key was the dropdown's. A field's editing has
        // already been ended by egui as this frame began, so it is read off
        // the last frame instead.
        let escape_is_ours = !egui::Popup::is_any_open(ui.ctx())
            && !ui.ctx().egui_wants_keyboard_input()
            && !self.editing_last_frame;
        self.editing_last_frame = false;
        let panel_rect = ui.available_rect_before_wrap();
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            if panel_rect.contains(pos) {
                response.has_pointer = true;
            }
        }

        // A stale index left over from a node that has since shrunk is the
        // same nothing as no selection at all.
        let index = selected_camera.filter(|&i| i < node.recon().image_set.cameras.len());
        let Some(index) = index else {
            show_empty_state(ui);
            return response;
        };
        let reference = CameraRef::new(node.id, index);
        // The panel shows and edits the shared camera, the lens all its images
        // share; the selected image's pose block below reads the image's own.
        let camera = &node.recon().image_set.cameras[index];
        let derived = self
            .derived
            .entry(reference)
            .or_insert_with(|| Derived::compute(camera));

        // The image the extrinsics block describes. Filtered by the camera it
        // uses as well as by range: the coupling invariant already guarantees
        // it, and a panel that would otherwise print one lens's `K` beside
        // another lens's pose should not depend on that guarantee holding.
        let image = selected_image
            .filter(|&i| i < node.recon().image_set.images.len())
            .filter(|&i| node.recon().image_set.images[i].camera_index as usize == index);
        let pose = image.map(|i| {
            let intrinsics = node.recon().image_set.image_intrinsics(i);
            extrinsics::Pose::resolve(node, i, &intrinsics, self.pose_frame)
        });
        let mut proposal = proposal.filter(|p| p.camera_ref() == reference);
        // Greyed while this camera's own proposal is open below the button: a
        // second open would be the one already there.
        let switch_refusal = match (switch_refusal, &proposal) {
            (Some(why), _) => Some(why.to_string()),
            (None, Some(_)) => Some("The proposal for this camera is open below.".to_string()),
            (None, None) => None,
        };
        let proposed_derived = &mut self.proposed_derived;
        // The statistics of the camera's per-image values, for the table to
        // state beside the shared parameters they adjust. One pass over the
        // camera's images, cheap enough to take every frame.
        let per_image =
            sfmtool_core::reconstruction::image_adjustment_statistics::image_adjustment_statistics(
                &node.recon().image_set,
                index,
            )
            .ok()
            .map(|statistics| parameters::PerImage {
                flags: node.recon().image_set.camera_per_image_adjustments(index),
                statistics,
                images: &node.recon().image_set.images,
            });
        let mut histograms = histogram::HistogramResponse::default();

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
                // Which per-image parameters the proposed camera keeps, which
                // it refits to a moved domain end and which it drops, for its
                // table.
                let mut proposed_note = None;
                // The fit's largest angle and where its monotonicity
                // constraint bound, for the change plot.
                let mut fit_angles = None;
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
                            *proposed_derived = Some((
                                reference,
                                generation,
                                ProposedReport::compute(camera, &refit.camera),
                            ));
                        }
                        proposed_camera = Some(refit.camera.clone());
                        proposed_note = parameters::proposed_per_image_note(
                            &sfmtool_core::reconstruction::switch_camera_model::varied_image_adjustments(
                                &node.recon().image_set,
                                index,
                            ),
                            &proposal.inputs.drop_image_adjustments,
                            if proposal.radial_corrections.is_some() {
                                &[sfmtool_core::reconstruction::bundle_adjust::ImageAdjustmentParameter::RadialBspline]
                            } else {
                                &[]
                            },
                        );
                        let monotone = refit.monotone_constraint;
                        fit_angles = Some((
                            refit.theta_fit_deg,
                            monotone.range_deg.filter(|_| monotone.active),
                        ));
                    }
                }
                match (proposed_camera.as_ref(), proposed_derived.as_ref()) {
                    (Some(proposed), Some((_, _, proposed_report))) => {
                        let aspect_row = parameters::has_two_focal_lengths(camera)
                            || parameters::has_two_focal_lengths(proposed);
                        histograms = compare_tables(
                            ui,
                            [(camera, derived), (proposed, &proposed_report.derived)],
                            per_image.as_ref(),
                            proposed_note.as_deref(),
                            image,
                            aspect_row,
                        );
                    }
                    _ => {
                        histograms = parameters::show_parameters(
                            ui,
                            camera,
                            per_image.as_ref(),
                            None,
                            image,
                            "current",
                        );
                        ui.separator();
                        parameters::show_derived(ui, camera, derived, "current", false);
                    }
                }
                ui.separator();
                parameters::show_k(ui, camera);
                ui.separator();
                let comparison = match (
                    proposed_camera.as_ref(),
                    proposed_derived.as_ref(),
                    fit_angles,
                ) {
                    (Some(proposed), Some((_, _, report)), Some((theta_fit_deg, monotone_deg))) => {
                        Some(projection_plot::Comparison {
                            proposed,
                            derived: &report.derived,
                            change: &report.change,
                            theta_fit_deg,
                            monotone_deg,
                        })
                    }
                    _ => None,
                };
                projection_plot::show_projection_plot(
                    ui,
                    camera,
                    derived,
                    comparison.as_ref(),
                    proposal.as_deref().map(|p| p.rug()),
                );
                if let (Some(image), Some(pose)) = (image, pose) {
                    ui.separator();
                    let picked =
                        extrinsics::show_extrinsics(ui, node, image, &pose, &mut self.pose_frame);
                    if picked {
                        response.select_image = Some(ImageRef::new(node.id, image));
                    }
                }
            });

        if let Some(picked) = histograms.select {
            response.select_image = Some(ImageRef::new(node.id, picked));
        }
        response.hovered_image = histograms
            .hovered
            .map(|hovered| ImageRef::new(node.id, hovered));

        // `Escape` cancels the proposal while the pointer is over the panel
        // showing it, so the key keeps its meaning everywhere else, and only
        // when no popup was open and no field was being edited as the frame
        // began: then the key closes those and nothing else.
        if response.has_pointer
            && response.proposal.is_none()
            && proposal.is_some()
            && escape_is_ours
            && ui.input(|i| i.key_pressed(egui::Key::Escape))
        {
            response.proposal = Some((reference, ProposalAction::Cancel));
        }
        self.editing_last_frame = ui.ctx().egui_wants_keyboard_input();

        response
    }

    /// Drop everything cached for a reconstruction that has left the scene.
    pub(crate) fn forget_recon(&mut self, id: crate::scene::ReconId) {
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

/// The current camera's tables and the proposed camera's, under `Current` and
/// `Proposed`: side by side when the panel is wide enough for both at their
/// natural width, and one above the other otherwise, so neither runs into the
/// other.
///
/// The width each needs is measured in an invisible sizing pass, since a
/// table's widest row depends on the model and its numbers. The statistics of
/// the camera's per-image values, `per_image`, are the images' and not either
/// lens's, and are stated once, in the current camera's table, with their
/// histograms, whose response is returned; the proposed camera's table says
/// instead which of them the proposal keeps and which it drops,
/// `proposed_note` ([`parameters::proposed_per_image_note`]).
/// `selected_image` as for [`parameters::show_parameters`].
fn compare_tables(
    ui: &mut egui::Ui,
    cameras: [(&sfmtool_core::camera::CameraIntrinsics, &Derived); 2],
    per_image: Option<&parameters::PerImage<'_>>,
    proposed_note: Option<&str>,
    selected_image: Option<usize>,
    aspect_row: bool,
) -> histogram::HistogramResponse {
    let per_image_of = |k: usize| if k == 0 { per_image } else { None };
    let note_of = |k: usize| if k == 0 { None } else { proposed_note };
    const SALTS: [&str; 2] = ["current", "proposed"];
    const HEADINGS: [&str; 2] = ["Current", "Proposed"];
    let needed = cameras
        .iter()
        .zip(SALTS)
        .enumerate()
        .map(|(k, ((camera, derived), salt))| {
            let mut sizing = ui.new_child(
                egui::UiBuilder::new()
                    .id_salt(("intrinsics_measure", salt))
                    .sizing_pass()
                    .invisible(),
            );
            parameters::show_parameters(
                &mut sizing,
                camera,
                per_image_of(k),
                note_of(k),
                None,
                salt,
            );
            parameters::show_derived(&mut sizing, camera, derived, salt, aspect_row);
            sizing.min_rect().width()
        })
        .fold(0.0f32, f32::max);
    let gap = ui.spacing().item_spacing.x;
    let mut response = histogram::HistogramResponse::default();
    let mut keep = |shown: histogram::HistogramResponse| {
        response.select = response.select.or(shown.select);
        response.hovered = response.hovered.or(shown.hovered);
    };
    if 2.0 * needed + gap <= ui.available_width() {
        ui.columns(2, |columns| {
            for (k, (camera, _)) in cameras.iter().enumerate() {
                columns[k].label(egui::RichText::new(HEADINGS[k]).weak());
                keep(parameters::show_parameters(
                    &mut columns[k],
                    camera,
                    per_image_of(k),
                    note_of(k),
                    selected_image,
                    SALTS[k],
                ));
            }
        });
        ui.separator();
        ui.columns(2, |columns| {
            for (k, (camera, derived)) in cameras.iter().enumerate() {
                parameters::show_derived(&mut columns[k], camera, derived, SALTS[k], aspect_row);
            }
        });
    } else {
        for (k, (camera, derived)) in cameras.iter().enumerate() {
            if k > 0 {
                ui.separator();
            }
            ui.label(egui::RichText::new(HEADINGS[k]).weak());
            keep(parameters::show_parameters(
                ui,
                camera,
                per_image_of(k),
                note_of(k),
                selected_image,
                SALTS[k],
            ));
            ui.separator();
            parameters::show_derived(ui, camera, derived, SALTS[k], aspect_row);
        }
    }
    response
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
