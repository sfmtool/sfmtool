// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The small dialog that `Bundle Adjust...` on a reconstruction's Scene Graph
//! context menu opens, and the gate the menu entry itself reads.
//!
//! See `specs/gui/edits/bundle-adjust.md`. The adjustment takes its decisions
//! about the lens from the user, camera by camera -- whether each camera's
//! focal length is released, and whether with it the lens distortion the
//! adjustment can free. That is the whole dialog: one row of two checkboxes
//! per camera, `Run` and `Cancel`. Everything else about the solve is the core
//! function's defaults. A spline camera's coefficient count and domain are not
//! the adjustment's to change: they are a refit of that camera, the Camera
//! Intrinsics panel's `Refit spline…` (`crate::refit_spline_prompt`).
//!
//! The gate is here rather than in the menu because the edit reads it too, so
//! the entry and the edit cannot disagree about when the adjustment can run.

use sfmtool_core::reconstruction::bundle_adjust::{
    distortion_is_releasable, focal_is_releasable, CameraRelease,
};
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

/// One camera the posed images use, as its row in the dialog shows it: which
/// camera, how many posed images it takes, and why either of its checkboxes is
/// greyed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CameraGate {
    /// The camera's index in the camera table.
    pub(crate) camera: usize,
    /// Its model's name.
    pub(crate) camera_model: &'static str,
    /// The posed images taken through it.
    pub(crate) images: usize,
    /// Why its focal cannot be released, or `None` when it can. The focal
    /// checkbox carries this as its disabled hover text.
    pub(crate) focal_refusal: Option<String>,
    /// Why its lens distortion cannot be released, or `None` when its model
    /// has some the adjustment can free. The distortion checkbox carries this
    /// as its disabled hover text.
    pub(crate) distortion_refusal: Option<String>,
}

/// One [`CameraGate`] per camera the posed images use, in table order.
///
/// Each release is decided on that camera's own model, as the core function
/// decides it, so a checkbox is greyed exactly when the core function would
/// refuse the release it asks for, and the hover text names the camera and its
/// model.
pub(crate) fn camera_gates(edited: &EditedReconstruction) -> Vec<CameraGate> {
    let table = &edited.base.image_table;
    let posed: Vec<u32> = table
        .images
        .iter()
        .filter(|image| {
            image.quaternion_wxyz.coords.iter().all(|c| c.is_finite())
                && image.translation_xyz.iter().all(|c| c.is_finite())
        })
        .map(|image| image.camera_index)
        .collect();
    edited
        .posed_lenses()
        .into_iter()
        .map(|c| {
            let camera = &table.cameras[c as usize];
            let camera_model = camera.model_name();
            CameraGate {
                camera: c as usize,
                camera_model,
                images: posed.iter().filter(|&&k| k == c).count(),
                focal_refusal: (!focal_is_releasable(camera)).then(|| {
                    format!(
                        "The adjustment's focal column is not exact for camera {c}, a \
                         {camera_model} camera, so its focal length cannot be released."
                    )
                }),
                distortion_refusal: (!distortion_is_releasable(camera)).then(|| {
                    format!(
                        "Camera {c}, a {camera_model} camera, has no lens distortion the \
                         adjustment can release. It releases k1 on SIMPLE_RADIAL_FISHEYE and the spline on \
                         SFMTOOL_FISHEYE and SFMTOOL_PINHOLE; switch the camera to one of those \
                         first."
                    )
                }),
            }
        })
        .collect()
}

/// What the dialog reads off the node when it opens: its cameras and why each
/// checkbox is greyed.
#[derive(Debug, Clone, Default)]
pub(crate) struct BundleAdjustGates {
    /// The length of the node's camera table, which the answer's release list
    /// is sized to.
    pub(crate) camera_count: usize,
    /// [`camera_gates`].
    pub(crate) cameras: Vec<CameraGate>,
}

impl BundleAdjustGates {
    /// Every gate, read off `edited`.
    pub(crate) fn of(edited: &EditedReconstruction) -> Self {
        Self {
            camera_count: edited.base.image_table.cameras.len(),
            cameras: camera_gates(edited),
        }
    }
}

/// What the user asked for, once they pressed `Run`.
#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustAnswer {
    /// The node to adjust.
    pub recon: ReconId,
    /// What each camera releases, one entry per camera in the node's camera
    /// table: held for a camera the dialog showed no row for, and never a
    /// release the camera's row had greyed. A distortion release only ever
    /// comes with the focal of the same camera.
    pub releases: Vec<CameraRelease>,
}

/// The dialog, and what it remembers while it is up.
#[derive(Default)]
pub struct BundleAdjustPrompt {
    pending: Option<Pending>,
}

/// One camera row's two checkboxes, as the user left them.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct RowState {
    focal: bool,
    distortion: bool,
}

/// The question being asked: which node, what the controls say, and why each
/// is disabled when it is.
struct Pending {
    recon: ReconId,
    label: String,
    gates: BundleAdjustGates,
    /// One per [`BundleAdjustGates::cameras`], in its order.
    rows: Vec<RowState>,
}

impl Pending {
    /// What camera row `j` releases: what its checkboxes say, less anything its
    /// model cannot release, and the distortion only with the focal.
    fn release(&self, j: usize) -> CameraRelease {
        let gate = &self.gates.cameras[j];
        let row = self.rows[j];
        let focal = row.focal && gate.focal_refusal.is_none();
        CameraRelease {
            focal,
            distortion: focal && row.distortion && gate.distortion_refusal.is_none(),
        }
    }

    /// The release list the answer carries: one entry per camera in the table,
    /// held where no row stands for it.
    fn releases(&self) -> Vec<CameraRelease> {
        let mut releases = vec![CameraRelease::HELD; self.gates.camera_count];
        for (j, gate) in self.gates.cameras.iter().enumerate() {
            if let Some(slot) = releases.get_mut(gate.camera) {
                *slot = self.release(j);
            }
        }
        releases
    }
}

impl BundleAdjustPrompt {
    /// Ask about `recon`.
    ///
    /// Idempotent while the dialog is already up, so a menu item racing itself
    /// cannot stack two of them. Every checkbox starts **clear**: a lens that
    /// moves is a different claim about the capture than a pose that does, and
    /// the default should be the smaller one.
    pub(crate) fn ask(&mut self, recon: ReconId, label: String, gates: BundleAdjustGates) {
        if self.pending.is_none() {
            let rows = vec![RowState::default(); gates.cameras.len()];
            self.pending = Some(Pending {
                recon,
                label,
                gates,
                rows,
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
                camera_rows(ui, pending);
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
            releases: pending.releases(),
        });
        if answer.is_some() || cancel || !still_open {
            self.pending = None;
        }
        answer
    }
}

/// The camera rows: per camera the posed images use, its index, model and
/// image count, and its "Release focal length" and "Release lens distortion"
/// checkboxes.
///
/// A checkbox the camera's model cannot take is greyed, with the reason as its
/// hover text. The distortion checkbox is also greyed while the same row's
/// focal is clear, and cleared when the focal is: neither k1 nor the spline
/// can change the scale at the centre of the image, which is the focal's job.
fn camera_rows(ui: &mut egui::Ui, pending: &mut Pending) {
    egui::Grid::new("bundle_adjust_cameras")
        .num_columns(3)
        .spacing([12.0, 4.0])
        .show(ui, |ui| {
            for (gate, row) in pending.gates.cameras.iter().zip(pending.rows.iter_mut()) {
                let plural = if gate.images == 1 { "" } else { "s" };
                ui.label(format!(
                    "Camera {}  {}  {} image{plural}",
                    gate.camera, gate.camera_model, gate.images
                ));
                ui.add_enabled_ui(gate.focal_refusal.is_none(), |ui| {
                    ui.checkbox(&mut row.focal, "Release focal length")
                        .on_disabled_hover_text(gate.focal_refusal.clone().unwrap_or_default())
                        .on_hover_text(
                            "Solve this camera's focal length along with the poses and the \
                             points, instead of holding it where it is.",
                        );
                });
                if !row.focal {
                    row.distortion = false;
                }
                let distortion_why = gate.distortion_refusal.clone().or_else(|| {
                    (!row.focal || gate.focal_refusal.is_some()).then(|| {
                        format!(
                            "The lens distortion of camera {} is released only together with its \
                             focal length.",
                            gate.camera
                        )
                    })
                });
                ui.add_enabled_ui(distortion_why.is_none(), |ui| {
                    ui.checkbox(&mut row.distortion, "Release lens distortion")
                        .on_disabled_hover_text(distortion_why.unwrap_or_default())
                        .on_hover_text(
                            "Solve this camera's lens distortion along with its focal length: \
                             k1 on SIMPLE_RADIAL_FISHEYE, the spline on SFMTOOL_FISHEYE and \
                             SFMTOOL_PINHOLE.",
                        );
                });
                ui.end_row();
            }
        });
}
