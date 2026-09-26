// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The small dialog that `Bundle Adjust...` on a reconstruction's Scene Graph
//! context menu opens, and the gate the menu entry itself reads.
//!
//! See `specs/gui/edits/bundle-adjust.md`. The adjustment takes its decisions
//! about the lens from the user -- whether the cameras' focal lengths are
//! released, whether with them any lens distortion the adjustment can free,
//! and whether the spline cameras are refitted to another coefficient count
//! first -- and that is the whole dialog: two checkboxes, the coefficients
//! row, `Run` and `Cancel`. Everything else about the solve is the core
//! function's defaults.
//!
//! The gate is here rather than in the menu because the edit reads it too, so
//! the entry and the edit cannot disagree about when the adjustment can run.

use sfmtool_core::reconstruction::bundle_adjust::{distortion_is_releasable, focal_is_releasable};
use sfmtool_core::EditedReconstruction;

use crate::scene::ReconId;

#[cfg(test)]
mod tests;

/// Why the adjustment cannot run on this value, or `None` when it can.
///
/// The two reasons are the ones a caller can see without solving anything: the
/// observations carry no pixel to reproject against, and no image carries a
/// pose. Both are the core function's own refusals, checked here so the menu
/// entry can say them before it is clicked. How many cameras the posed images
/// use is not a reason: the adjustment solves each of them.
pub(crate) fn refusal(edited: &EditedReconstruction) -> Option<String> {
    if !edited.has_keypoints() {
        return Some(
            "This reconstruction's observations are .sift feature indexes with no inline \
             keypoints, and the adjustment needs a pixel per observation."
                .to_string(),
        );
    }
    (edited.posed_lens_count() == 0)
        .then(|| "No image of this reconstruction carries a pose.".to_string())
}

/// Why the focal cannot be released on this value, or `None` when it can. The
/// checkbox carries this as its disabled hover text.
///
/// The release reaches every camera the posed images use, and the core function
/// refuses it when any of them has a model its focal column is not exact for, so
/// the checkbox is greyed unless every one of them passes and the reason names
/// the first that does not.
pub(crate) fn focal_refusal(edited: &EditedReconstruction) -> Option<String> {
    let table = &edited.base.image_table;
    let (index, camera) = edited
        .posed_lenses()
        .into_iter()
        .map(|c| (c, &table.cameras[c as usize]))
        .find(|(_, camera)| !focal_is_releasable(camera))?;
    Some(format!(
        "The adjustment's focal column is not exact for camera {index}, a {} camera, so no \
         focal length can be released.",
        camera.model_name()
    ))
}

/// Why the lens distortion cannot be released on this value, or `None` when it
/// can. The checkbox carries this as its disabled hover text.
///
/// The release reaches the cameras the posed images use whose model has
/// distortion the adjustment can free (`k1` on `SIMPLE_RADIAL_FISHEYE`, the
/// spline on the spline models), so the checkbox is live when at least one
/// does; the core function refuses the release when none does. A camera of any
/// other model keeps its distortion.
pub(crate) fn distortion_refusal(edited: &EditedReconstruction) -> Option<String> {
    let table = &edited.base.image_table;
    let any = edited
        .posed_lenses()
        .into_iter()
        .any(|c| distortion_is_releasable(&table.cameras[c as usize]));
    (!any).then(|| {
        "No camera of this reconstruction has lens distortion the adjustment can release. \
         It releases k1 on SIMPLE_RADIAL_FISHEYE and the spline on SFMTOOL_FISHEYE and \
         SFMTOOL_PINHOLE; switch a camera to one of those first."
            .to_string()
    })
}

/// The distinct spline coefficient counts of the cameras the posed images use
/// that are spline models, ascending; empty when none is.
///
/// The coefficients control shows them as what the count is now, and is live
/// only when there is at least one.
pub(crate) fn spline_coeff_counts(edited: &EditedReconstruction) -> Vec<usize> {
    let table = &edited.base.image_table;
    let mut counts: Vec<usize> = edited
        .posed_lenses()
        .into_iter()
        .filter_map(|c| table.cameras[c as usize].model.radial_spline())
        .map(|(bspline, _, _)| bspline.len())
        .collect();
    counts.sort_unstable();
    counts.dedup();
    counts
}

/// What the dialog reads off the node when it opens: why each checkbox is
/// greyed, and the spline coefficient counts the node holds.
#[derive(Debug, Clone, Default)]
pub(crate) struct BundleAdjustGates {
    /// [`focal_refusal`].
    pub(crate) focal_refusal: Option<String>,
    /// [`distortion_refusal`].
    pub(crate) distortion_refusal: Option<String>,
    /// [`spline_coeff_counts`].
    pub(crate) spline_coeff_counts: Vec<usize>,
}

impl BundleAdjustGates {
    /// Every gate, read off `edited`.
    pub(crate) fn of(edited: &EditedReconstruction) -> Self {
        Self {
            focal_refusal: focal_refusal(edited),
            distortion_refusal: distortion_refusal(edited),
            spline_coeff_counts: spline_coeff_counts(edited),
        }
    }
}

/// What the user asked for, once they pressed `Run`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BundleAdjustAnswer {
    /// The node to adjust.
    pub recon: ReconId,
    /// Whether to release each camera's focal length.
    pub release_focal: bool,
    /// Whether to release each camera's lens distortion, where its model has
    /// one the adjustment can free. Only ever true together with
    /// `release_focal`.
    pub release_distortion: bool,
    /// The coefficient count every spline camera is refitted to before the
    /// solve, or `None` to keep each count. Only ever set together with
    /// `release_distortion`, and never to the one count every spline camera
    /// already has.
    pub spline_coeff_count: Option<usize>,
}

/// The dialog, and what it remembers while it is up.
#[derive(Default)]
pub struct BundleAdjustPrompt {
    pending: Option<Pending>,
}

/// The question being asked: which node, what the controls say, and why each
/// is disabled when it is.
struct Pending {
    recon: ReconId,
    label: String,
    gates: BundleAdjustGates,
    release_focal: bool,
    release_distortion: bool,
    /// Keep each spline camera's coefficient count, which is the default.
    keep_coeffs: bool,
    /// The count the control shows, and the one asked for once `keep_coeffs`
    /// is clear.
    coeff_count: usize,
}

impl Pending {
    /// The count the answer carries: `None` unless the distortion is released,
    /// the count is not kept, and it differs from a single count every spline
    /// camera already has.
    fn spline_coeff_count(&self) -> Option<usize> {
        let asked = self.release_focal
            && self.release_distortion
            && !self.keep_coeffs
            && !self.gates.spline_coeff_counts.is_empty();
        (asked && self.gates.spline_coeff_counts != [self.coeff_count]).then_some(self.coeff_count)
    }
}

impl BundleAdjustPrompt {
    /// Ask about `recon`.
    ///
    /// Idempotent while the dialog is already up, so a menu item racing itself
    /// cannot stack two of them. Both checkboxes start **clear**: a lens that
    /// moves is a different claim about the capture than a pose that does, and
    /// the default should be the smaller one. The coefficient count starts at
    /// **keep**, showing the node's largest count.
    pub(crate) fn ask(&mut self, recon: ReconId, label: String, gates: BundleAdjustGates) {
        if self.pending.is_none() {
            let coeff_count = gates
                .spline_coeff_counts
                .last()
                .copied()
                .unwrap_or(sfmtool_core::camera::refit_intrinsics::DEFAULT_COEFF_COUNT);
            self.pending = Some(Pending {
                recon,
                label,
                gates,
                release_focal: false,
                release_distortion: false,
                keep_coeffs: true,
                coeff_count,
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
                ui.add_enabled_ui(pending.gates.focal_refusal.is_none(), |ui| {
                    ui.checkbox(&mut pending.release_focal, "Release focal length")
                        .on_disabled_hover_text(
                            pending.gates.focal_refusal.clone().unwrap_or_default(),
                        )
                        .on_hover_text(
                            "Solve each camera's focal length along with the poses and the \
                             points, instead of holding them where they are.",
                        );
                });
                // Neither k1 nor the spline can change the scale at the centre of
                // the image, which is the focal's job, so the distortion is
                // released only with the focal.
                if !pending.release_focal {
                    pending.release_distortion = false;
                }
                let distortion_why = pending.gates.distortion_refusal.clone().or_else(|| {
                    (!pending.release_focal).then(|| {
                        "The lens distortion is released only together with the focal length."
                            .to_string()
                    })
                });
                ui.add_enabled_ui(distortion_why.is_none(), |ui| {
                    ui.checkbox(&mut pending.release_distortion, "Release lens distortion")
                        .on_disabled_hover_text(distortion_why.unwrap_or_default())
                        .on_hover_text(
                            "Solve each camera's lens distortion along with its focal length: \
                             k1 on SIMPLE_RADIAL_FISHEYE, the spline on SFMTOOL_FISHEYE and \
                             SFMTOOL_PINHOLE. Cameras of other models keep theirs.",
                        );
                });
                spline_coeffs_row(ui, pending);
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

        let answer = run.then(|| BundleAdjustAnswer {
            recon: pending.recon,
            release_focal: pending.release_focal,
            release_distortion: pending.release_focal && pending.release_distortion,
            spline_coeff_count: pending.spline_coeff_count(),
        });
        if answer.is_some() || cancel || !still_open {
            self.pending = None;
        }
        answer
    }
}

/// The "Spline coefficients" row under the distortion checkbox: a `Keep`
/// checkbox, the count, and the count(s) the node's spline cameras have now.
///
/// Live only while the distortion is released and a spline camera is in the
/// solve: a new count is a refit the solve then corrects, and without the
/// release it would only approximate the old curve. Editing the count clears
/// `Keep`.
fn spline_coeffs_row(ui: &mut egui::Ui, pending: &mut Pending) {
    let why = if pending.gates.spline_coeff_counts.is_empty() {
        Some("No camera of this reconstruction is a spline model (SFMTOOL_FISHEYE, SFMTOOL_PINHOLE).")
    } else if !pending.release_distortion {
        Some("The coefficient count changes only while the lens distortion is released.")
    } else {
        None
    };
    let now = pending
        .gates
        .spline_coeff_counts
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    ui.add_enabled_ui(why.is_none(), |ui| {
        ui.horizontal(|ui| {
            ui.label("Spline coefficients");
            ui.checkbox(&mut pending.keep_coeffs, "Keep")
                .on_hover_text("Keep each spline camera's coefficient count.");
            let range = sfmtool_core::reconstruction::bundle_adjust::SPLINE_COEFF_COUNT_RANGE;
            let count = ui
                .add(egui::DragValue::new(&mut pending.coeff_count).range(range))
                .on_hover_text(
                    "Refit each spline camera to this many coefficients over its whole domain \
                     before the solve; the solve then fits them to the observations.",
                );
            if count.changed() {
                pending.keep_coeffs = false;
            }
            if !now.is_empty() {
                ui.label(format!("now {now}"));
            }
        })
        .response
        .on_disabled_hover_text(why.unwrap_or_default());
    });
}
