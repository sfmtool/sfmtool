// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The small dialog the Edit menu's `Bundle Adjust...` opens, and the gate the
//! menu entry itself reads.
//!
//! See `specs/gui/edits/bundle-adjust.md`. The adjustment takes one decision
//! from the user -- whether the shared focal is released -- and that is the
//! whole dialog: a checkbox, `Run` and `Cancel`. Everything else about the
//! solve is the core function's defaults.
//!
//! The gate is here rather than in the menu because the edit reads it too, so
//! the entry and the edit cannot disagree about when the adjustment can run.

use sfmtool_core::reconstruction::bundle_adjust::focal_is_releasable;
use sfmtool_core::EditedReconstruction;

use crate::scene::ReconId;

#[cfg(test)]
mod tests;

/// Why the adjustment cannot run on this value, or `None` when it can.
///
/// The two reasons are the ones a caller can see without solving anything: the
/// observations carry no pixel to reproject against, and the posed images do
/// not share one lens. Both are the core function's own refusals, checked here
/// so the menu entry can say them before it is clicked.
pub(crate) fn refusal(edited: &EditedReconstruction) -> Option<String> {
    if !edited.has_keypoints() {
        return Some(
            "This reconstruction's observations are .sift feature indexes with no inline \
             keypoints, and the adjustment needs a pixel per observation."
                .to_string(),
        );
    }
    let table = &edited.base.image_table;
    let mut lenses: Vec<u32> = table
        .images
        .iter()
        .filter(|image| {
            image.quaternion_wxyz.coords.iter().all(|c| c.is_finite())
                && image.translation_xyz.iter().all(|c| c.is_finite())
        })
        .map(|image| image.camera_index)
        .collect();
    lenses.sort_unstable();
    lenses.dedup();
    match lenses.len() {
        0 => Some("No image of this reconstruction carries a pose.".to_string()),
        1 => None,
        n => Some(format!(
            "The adjustment solves one shared camera, and these images are taken through {n}."
        )),
    }
}

/// Why the focal cannot be released on this value's camera, or `None` when it
/// can. The checkbox carries this as its disabled hover text.
pub(crate) fn focal_refusal(edited: &EditedReconstruction) -> Option<String> {
    let table = &edited.base.image_table;
    let camera = table.cameras.first()?;
    if focal_is_releasable(camera) {
        return None;
    }
    Some(format!(
        "The adjustment's focal column is not exact for a {} camera, so its focal stays \
         where it is.",
        camera.model_name()
    ))
}

/// What the user asked for, once they pressed `Run`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BundleAdjustAnswer {
    /// The node to adjust.
    pub recon: ReconId,
    /// Whether to release the shared focal length.
    pub release_focal: bool,
}

/// The dialog, and what it remembers while it is up.
#[derive(Default)]
pub struct BundleAdjustPrompt {
    pending: Option<Pending>,
}

/// The question being asked: which node, what the checkbox says, and why it is
/// disabled when it is.
struct Pending {
    recon: ReconId,
    label: String,
    release_focal: bool,
    focal_refusal: Option<String>,
}

impl BundleAdjustPrompt {
    /// Ask about `recon`.
    ///
    /// Idempotent while the dialog is already up, so a menu item racing itself
    /// cannot stack two of them. The checkbox starts **clear**: a focal that
    /// moves is a different claim about the capture than a pose that does, and
    /// the default should be the smaller one.
    pub fn ask(&mut self, recon: ReconId, label: String, focal_refusal: Option<String>) {
        if self.pending.is_none() {
            self.pending = Some(Pending {
                recon,
                label,
                release_focal: false,
                focal_refusal,
            });
        }
    }

    /// Draw one frame, returning the answer on the frame `Run` is pressed.
    ///
    /// Enter runs and Escape cancels, which is the create-point prompt's
    /// vocabulary: this is a step in a gesture rather than a window to leave
    /// lying open.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<BundleAdjustAnswer> {
        let pending = self.pending.as_mut()?;
        let mut run = false;
        let mut cancel = false;
        let mut still_open = true;

        egui::Window::new("Bundle Adjust")
            .open(&mut still_open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(format!(
                    "Refine every pose and point of {} against its observations.",
                    pending.label
                ));
                ui.add_space(8.0);
                ui.add_enabled_ui(pending.focal_refusal.is_none(), |ui| {
                    ui.checkbox(&mut pending.release_focal, "Release focal length")
                        .on_disabled_hover_text(pending.focal_refusal.clone().unwrap_or_default())
                        .on_hover_text(
                            "Solve the shared focal length along with the poses and the \
                             points, instead of holding it where it is.",
                        );
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Run").clicked() {
                        run = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
                run |= ui.input(|i| i.key_pressed(egui::Key::Enter));
                cancel |= ui.input(|i| i.key_pressed(egui::Key::Escape));
            });

        let answer = run.then_some(BundleAdjustAnswer {
            recon: pending.recon,
            release_focal: pending.release_focal,
        });
        if answer.is_some() || cancel || !still_open {
            self.pending = None;
        }
        answer
    }
}
