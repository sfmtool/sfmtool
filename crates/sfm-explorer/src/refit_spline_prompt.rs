// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The small dialog the Camera Intrinsics panel's `Refit spline…` opens: a
//! spline camera's coefficient count and domain end, and the refit that gives
//! it them.
//!
//! See `specs/gui/edits/switch-camera-model.md`. The refit is a switch of the
//! camera to its own spline model, so what the dialog answers becomes a
//! [`crate::state::edits::SwitchCameraModelRequest`] with no model named. It
//! asks two things -- `Coefficients` and `Spline domain (°)` -- and shows the
//! outermost keypoint of the camera's images beside the domain, with a button
//! that sets the domain to its angle, because a circular fisheye is trimmed to
//! its image circle by choice rather than by default.

use sfmtool_core::camera::refit_intrinsics::{spline_domain_deg, SPLINE_COEFF_COUNT_RANGE};
use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::reconstruction::outermost_keypoint::{outermost_keypoints, KeypointReach};
use sfmtool_core::EditedReconstruction;

use crate::scene::ReconId;

#[cfg(test)]
mod tests;

/// Why `camera` has no spline to refit, or `None` when it has one. The
/// header's `Refit spline…` button carries this as its disabled hover text.
pub(crate) fn refit_spline_refusal(camera: &CameraIntrinsics) -> Option<String> {
    camera.model.radial_spline().is_none().then(|| {
        format!(
            "A {} camera has no spline to refit; only SFMTOOL_FISHEYE and SFMTOOL_PINHOLE \
             carry one. Switch the camera to one of those first (sfm xform --camera-model, \
             or the MCP tool switch_camera_model).",
            camera.model_name()
        )
    })
}

/// How far out the photographs of one camera reach: its outermost keypoint, by
/// incidence angle under the camera's model, among the observations and among
/// the features detected in the images' `.sift` files.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct KeypointExtent {
    /// The outermost observation.
    pub(crate) observed: Option<KeypointReach>,
    /// The outermost detected feature, `None` when no `.sift` file is readable.
    pub(crate) detected: Option<KeypointReach>,
}

impl KeypointExtent {
    /// The camera `camera` of `edited`'s base value, over its observations and,
    /// when `read_sift_files`, its images' `.sift` files.
    pub(crate) fn of(edited: &EditedReconstruction, camera: usize, read_sift_files: bool) -> Self {
        outermost_keypoints(&edited.base, &[camera], read_sift_files)
            .into_iter()
            .next()
            .map(|o| KeypointExtent {
                observed: o.observed,
                detected: o.detected,
            })
            .unwrap_or_default()
    }

    /// The angle the domain row's button sets: the detected keypoint's, or the
    /// observed one's when nothing was detected.
    pub(crate) fn suggested_domain_deg(&self) -> Option<f64> {
        self.detected.or(self.observed).map(|r| r.theta_deg)
    }

    /// The sentence the domain row shows, e.g. `outermost keypoint: 229.7 px,
    /// 101.2° observed; 244.1 px, 107.9° detected`; `None` with no keypoint.
    pub(crate) fn describe(&self) -> Option<String> {
        let part = |r: &KeypointReach, source: &str| {
            format!("{:.1} px, {:.1}° {source}", r.radius_px, r.theta_deg)
        };
        let parts: Vec<String> = [
            self.observed.as_ref().map(|r| part(r, "observed")),
            self.detected.as_ref().map(|r| part(r, "detected")),
        ]
        .into_iter()
        .flatten()
        .collect();
        (!parts.is_empty()).then(|| format!("outermost keypoint: {}", parts.join("; ")))
    }
}

/// What the dialog reads off the camera when it opens.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RefitSplineGates {
    /// The camera's index in the node's camera table.
    pub(crate) camera: usize,
    /// Its model's name.
    pub(crate) camera_model: &'static str,
    /// Its spline's coefficient count now.
    pub(crate) coeff_count: usize,
    /// Where its spline's domain ends now, as an incidence angle in degrees.
    pub(crate) domain_deg: f64,
    /// How far out its images reach.
    pub(crate) keypoint_extent: KeypointExtent,
}

impl RefitSplineGates {
    /// The gates of camera `camera` of `edited`, or the reason it has none: no
    /// such camera, or no spline. Reads the images' `.sift` files for the
    /// detected keypoint, once, when the dialog opens.
    pub(crate) fn of(edited: &EditedReconstruction, camera: usize) -> Result<Self, String> {
        let cameras = &edited.base.image_table.cameras;
        let intrinsics = cameras.get(camera).ok_or_else(|| {
            format!(
                "Camera {camera} does not exist; the reconstruction has {} camera(s).",
                cameras.len()
            )
        })?;
        if let Some(why) = refit_spline_refusal(intrinsics) {
            return Err(why);
        }
        let (bspline, _, _) = intrinsics
            .model
            .radial_spline()
            .expect("the refusal above covers a camera with no spline");
        Ok(Self {
            camera,
            camera_model: intrinsics.model_name(),
            coeff_count: bspline.len(),
            domain_deg: spline_domain_deg(intrinsics).unwrap_or(f64::NAN),
            keypoint_extent: KeypointExtent::of(edited, camera, true),
        })
    }
}

/// What the user asked for, once they pressed `Apply`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RefitSplineAnswer {
    /// The node the camera belongs to.
    pub(crate) recon: ReconId,
    /// The camera's index in its camera table.
    pub(crate) camera: usize,
    /// The coefficient count to refit to.
    pub(crate) coeff_count: usize,
    /// The domain end to refit on, in degrees, or `None` to keep the camera's
    /// own exactly, which is what an untouched field means.
    pub(crate) spline_domain_deg: Option<f64>,
}

impl RefitSplineAnswer {
    /// The switch this answer asks for: the camera to its own model.
    pub(crate) fn request(&self) -> crate::state::edits::SwitchCameraModelRequest {
        crate::state::edits::SwitchCameraModelRequest {
            camera: self.camera,
            camera_model: None,
            coeff_count: Some(self.coeff_count),
            spline_domain_deg: self.spline_domain_deg,
            theta_fit_deg: None,
        }
    }
}

/// The dialog, and what it remembers while it is up.
#[derive(Default)]
pub struct RefitSplinePrompt {
    pending: Option<Pending>,
}

/// The question being asked, and what the two fields say.
struct Pending {
    recon: ReconId,
    label: String,
    gates: RefitSplineGates,
    coeff_count: usize,
    domain_deg: f64,
}

impl Pending {
    /// The domain the answer carries: `None` while the field still shows the
    /// camera's own, so an untouched field keeps the domain bit for bit.
    fn spline_domain_deg(&self) -> Option<f64> {
        ((self.domain_deg - self.gates.domain_deg).abs() > 1e-9).then_some(self.domain_deg)
    }

    /// The domain row's button: the domain set to the outermost keypoint's
    /// angle, detected where there is one and observed otherwise. Nothing
    /// happens with no keypoint to take it from.
    fn use_outermost_keypoint(&mut self) {
        if let Some(deg) = self.gates.keypoint_extent.suggested_domain_deg() {
            self.domain_deg = deg;
        }
    }

    fn answer(&self) -> RefitSplineAnswer {
        RefitSplineAnswer {
            recon: self.recon,
            camera: self.gates.camera,
            coeff_count: self.coeff_count,
            spline_domain_deg: self.spline_domain_deg(),
        }
    }
}

impl RefitSplinePrompt {
    /// Ask about one camera of `recon`, with both fields showing what the
    /// camera has now.
    ///
    /// Idempotent while the dialog is already up, so a button racing itself
    /// cannot stack two of them.
    pub(crate) fn ask(&mut self, recon: ReconId, label: String, gates: RefitSplineGates) {
        if self.pending.is_none() {
            self.pending = Some(Pending {
                recon,
                label,
                coeff_count: gates.coeff_count,
                domain_deg: gates.domain_deg,
                gates,
            });
        }
    }

    /// Draw one frame, returning the answer on the frame `Apply` is pressed.
    ///
    /// Enter applies and Escape cancels, as the Bundle Adjust dialog does.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<RefitSplineAnswer> {
        let pending = self.pending.as_mut()?;
        let mut apply = false;
        let mut cancel = false;
        let mut still_open = true;

        egui::Window::new("Refit Spline")
            .open(&mut still_open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(format!(
                    "Refit the spline of camera {} of {}, a {}, over its whole domain.",
                    pending.gates.camera, pending.label, pending.gates.camera_model
                ));
                ui.add_space(8.0);
                coeffs_row(ui, pending);
                domain_row(ui, pending);
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Apply").clicked() {
                        apply = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
                apply |= ui.input(|i| i.key_pressed(egui::Key::Enter));
                cancel |= ui.input(|i| i.key_pressed(egui::Key::Escape));
            });

        let answer = apply.then(|| pending.answer());
        if answer.is_some() || cancel || !still_open {
            self.pending = None;
        }
        answer
    }
}

/// The `Coefficients` row: the count, bounded by the counts a spline with a
/// curve takes, and the count the camera has now.
fn coeffs_row(ui: &mut egui::Ui, pending: &mut Pending) {
    ui.horizontal(|ui| {
        ui.label("Coefficients");
        ui.add(egui::DragValue::new(&mut pending.coeff_count).range(SPLINE_COEFF_COUNT_RANGE))
            .on_hover_text(
                "Refit the spline to this many coefficients over its whole domain. The refit \
                 is the closest monotone curve to the current one; bundle adjustment with the \
                 lens distortion released then fits the coefficients to the observations.",
            );
        ui.label(format!("now {}", pending.gates.coeff_count));
    });
}

/// The `Spline domain (°)` row: the domain end, the one the camera has now,
/// and under them the outermost keypoint with a button that sets the domain to
/// it.
///
/// The default is the camera's own domain. The outermost keypoint is offered,
/// so a circular fisheye can be trimmed to its image circle by choice. The
/// button takes the detected keypoint, and where no `.sift` file could be read
/// the observed one, which the text labels as observed.
fn domain_row(ui: &mut egui::Ui, pending: &mut Pending) {
    let max = if pending.gates.camera_model == "SFMTOOL_PINHOLE" {
        89.9
    } else {
        180.0
    };
    ui.horizontal(|ui| {
        ui.label("Spline domain (°)");
        ui.add(
            egui::DragValue::new(&mut pending.domain_deg)
                .range(0.1..=max)
                .speed(0.1)
                .fixed_decimals(1),
        )
        .on_hover_text(
            "Refit the spline on a domain ending at this incidence angle, over the whole of \
             it. Past the domain the model is a straight line bundle adjustment cannot bend.",
        );
        ui.label(format!("now {:.1}°", pending.gates.domain_deg));
    });
    if let Some(text) = pending.gates.keypoint_extent.describe() {
        ui.horizontal(|ui| {
            ui.label(text);
            if let Some(deg) = pending.gates.keypoint_extent.suggested_domain_deg() {
                if ui
                    .small_button(format!("Use {deg:.1}°"))
                    .on_hover_text(
                        "Set the domain to the outermost keypoint's angle: the detected one, or \
                         the observed one where no .sift file could be read.",
                    )
                    .clicked()
                {
                    pending.use_outermost_keypoint();
                }
            }
        });
    }
}
