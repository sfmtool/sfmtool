// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The small dialog that `Bundle Adjust...` on a reconstruction's Scene Graph
//! context menu opens, and the gate the menu entry itself reads.
//!
//! See `specs/gui/edits/bundle-adjust.md`. The adjustment takes its decisions
//! about the lens from the user, camera by camera -- whether each camera's
//! focal length is released, and whether with it the lens distortion the
//! adjustment can free -- and whether the spline cameras are refitted to
//! another coefficient count first. That is the whole dialog: one row of two
//! checkboxes per camera, the coefficients and domain rows, `Run` and
//! `Cancel`. Everything else about the solve is the core function's defaults.
//!
//! The gate is here rather than in the menu because the edit reads it too, so
//! the entry and the edit cannot disagree about when the adjustment can run.

use sfmtool_core::reconstruction::bundle_adjust::{
    distortion_is_releasable, focal_is_releasable, spline_domain_deg, CameraRelease,
};
use sfmtool_core::reconstruction::outermost_keypoint::{outermost_keypoints, KeypointReach};
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
    pub(crate) model: &'static str,
    /// The posed images taken through it.
    pub(crate) images: usize,
    /// Why its focal cannot be released, or `None` when it can. The focal
    /// checkbox carries this as its disabled hover text.
    pub(crate) focal_refusal: Option<String>,
    /// Why its lens distortion cannot be released, or `None` when its model
    /// has some the adjustment can free. The distortion checkbox carries this
    /// as its disabled hover text.
    pub(crate) distortion_refusal: Option<String>,
    /// Whether it is a spline model, which the coefficient and domain rows
    /// refit.
    pub(crate) spline: bool,
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
            let model = camera.model_name();
            CameraGate {
                camera: c as usize,
                model,
                images: posed.iter().filter(|&&k| k == c).count(),
                focal_refusal: (!focal_is_releasable(camera)).then(|| {
                    format!(
                        "The adjustment's focal column is not exact for camera {c}, a {model} \
                         camera, so its focal length cannot be released."
                    )
                }),
                distortion_refusal: (!distortion_is_releasable(camera)).then(|| {
                    format!(
                        "Camera {c}, a {model} camera, has no lens distortion the adjustment can \
                         release. It releases k1 on SIMPLE_RADIAL_FISHEYE and the spline on \
                         SFMTOOL_FISHEYE and SFMTOOL_PINHOLE; switch the camera to one of those \
                         first."
                    )
                }),
                spline: camera.model.radial_spline().is_some(),
            }
        })
        .collect()
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

/// The camera-table indexes of the cameras the posed images use that are
/// spline models.
fn spline_cameras(edited: &EditedReconstruction) -> Vec<usize> {
    let table = &edited.base.image_table;
    edited
        .posed_lenses()
        .into_iter()
        .map(|c| c as usize)
        .filter(|&c| table.cameras[c].model.radial_spline().is_some())
        .collect()
}

/// The distinct spline domain ends, in degrees, of the cameras the posed images
/// use that are spline models, ascending; empty when none is. The domain row
/// shows them as what the domain is now.
pub(crate) fn spline_domains_deg(edited: &EditedReconstruction) -> Vec<f64> {
    let table = &edited.base.image_table;
    let mut domains: Vec<f64> = spline_cameras(edited)
        .into_iter()
        .filter_map(|c| spline_domain_deg(&table.cameras[c]))
        .collect();
    domains.sort_unstable_by(f64::total_cmp);
    domains.dedup_by(|a, b| (*a - *b).abs() <= 1e-9);
    domains
}

/// How far out the photographs of the node's spline cameras reach: the
/// outermost keypoint, by incidence angle, over all of those cameras, among the
/// observations and among the features detected in the images' `.sift` files.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct KeypointExtent {
    /// The outermost observation.
    pub(crate) observed: Option<KeypointReach>,
    /// The outermost detected feature, `None` when no `.sift` file is readable.
    pub(crate) detected: Option<KeypointReach>,
}

impl KeypointExtent {
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

/// The node's [`KeypointExtent`], each keypoint measured under its own camera's
/// model, over the base value's observations and, when `read_sift_files`, its
/// images' `.sift` files.
pub(crate) fn spline_keypoint_extent(
    edited: &EditedReconstruction,
    read_sift_files: bool,
) -> KeypointExtent {
    let cameras = spline_cameras(edited);
    if cameras.is_empty() {
        return KeypointExtent::default();
    }
    let outer = |a: Option<KeypointReach>, b: Option<KeypointReach>| match (a, b) {
        (Some(a), Some(b)) => Some(if b.theta_deg > a.theta_deg { b } else { a }),
        (a, b) => a.or(b),
    };
    outermost_keypoints(&edited.base, &cameras, read_sift_files)
        .into_iter()
        .fold(KeypointExtent::default(), |acc, c| KeypointExtent {
            observed: outer(acc.observed, c.observed),
            detected: outer(acc.detected, c.detected),
        })
}

/// What the dialog reads off the node when it opens: its cameras and why each
/// checkbox is greyed, the spline coefficient counts and domains the node
/// holds, and how far out its photographs reach.
#[derive(Debug, Clone, Default)]
pub(crate) struct BundleAdjustGates {
    /// The length of the node's camera table, which the answer's release list
    /// is sized to.
    pub(crate) camera_count: usize,
    /// [`camera_gates`].
    pub(crate) cameras: Vec<CameraGate>,
    /// [`spline_coeff_counts`].
    pub(crate) spline_coeff_counts: Vec<usize>,
    /// [`spline_domains_deg`].
    pub(crate) spline_domains_deg: Vec<f64>,
    /// [`spline_keypoint_extent`].
    pub(crate) keypoint_extent: KeypointExtent,
}

impl BundleAdjustGates {
    /// Every gate, read off `edited`. This reads the images' `.sift` files for
    /// the detected keypoint, once, when the dialog opens; the positions alone
    /// are a small read (see `specs/gui/edits/bundle-adjust.md`).
    pub(crate) fn of(edited: &EditedReconstruction) -> Self {
        Self {
            camera_count: edited.base.image_table.cameras.len(),
            cameras: camera_gates(edited),
            spline_coeff_counts: spline_coeff_counts(edited),
            spline_domains_deg: spline_domains_deg(edited),
            keypoint_extent: spline_keypoint_extent(edited, true),
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
    /// The coefficient count every spline camera is refitted to before the
    /// solve, or `None` to keep each count. Only ever set while some spline
    /// camera releases its distortion, and never to the one count every spline
    /// camera already has.
    pub spline_coeff_count: Option<usize>,
    /// The incidence angle, in degrees, every spline camera's domain is moved
    /// to before the solve, or `None` to keep each domain. Set under the same
    /// conditions as `spline_coeff_count`.
    pub spline_domain_deg: Option<f64>,
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
    /// Keep each spline camera's coefficient count, which is the default.
    keep_coeffs: bool,
    /// The count the control shows, and the one asked for once `keep_coeffs`
    /// is clear.
    coeff_count: usize,
    /// Keep each spline camera's domain, which is the default.
    keep_domain: bool,
    /// The domain end the control shows, in degrees, and the one asked for
    /// once `keep_domain` is clear.
    domain_deg: f64,
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

    /// Whether some spline camera releases its distortion, which is what the
    /// coefficient and domain rows need: a refit is only an approximation of
    /// the old curve until the solve fits it to the observations.
    fn spline_distortion_released(&self) -> bool {
        self.gates
            .cameras
            .iter()
            .enumerate()
            .any(|(j, gate)| gate.spline && self.release(j).distortion)
    }

    /// The domain the answer carries, under the rule the count follows.
    fn spline_domain_deg(&self) -> Option<f64> {
        let asked = self.spline_distortion_released()
            && !self.keep_domain
            && !self.gates.spline_domains_deg.is_empty();
        let unchanged = matches!(
            self.gates.spline_domains_deg.as_slice(),
            [only] if (only - self.domain_deg).abs() <= 1e-9
        );
        (asked && !unchanged).then_some(self.domain_deg)
    }

    /// The domain row's button: the domain set to the outermost keypoint's
    /// angle, detected where there is one and observed otherwise, and `Keep`
    /// cleared. Nothing happens with no keypoint to take it from.
    fn use_outermost_keypoint(&mut self) {
        if let Some(deg) = self.gates.keypoint_extent.suggested_domain_deg() {
            self.domain_deg = deg;
            self.keep_domain = false;
        }
    }

    /// The count the answer carries: `None` unless the distortion is released,
    /// the count is not kept, and it differs from a single count every spline
    /// camera already has.
    fn spline_coeff_count(&self) -> Option<usize> {
        let asked = self.spline_distortion_released()
            && !self.keep_coeffs
            && !self.gates.spline_coeff_counts.is_empty();
        (asked && self.gates.spline_coeff_counts != [self.coeff_count]).then_some(self.coeff_count)
    }
}

impl BundleAdjustPrompt {
    /// Ask about `recon`.
    ///
    /// Idempotent while the dialog is already up, so a menu item racing itself
    /// cannot stack two of them. Every checkbox starts **clear**: a lens that
    /// moves is a different claim about the capture than a pose that does, and
    /// the default should be the smaller one. The coefficient count and the
    /// spline domain start at **keep**, showing the node's largest count and
    /// domain.
    pub(crate) fn ask(&mut self, recon: ReconId, label: String, gates: BundleAdjustGates) {
        if self.pending.is_none() {
            let coeff_count = gates
                .spline_coeff_counts
                .last()
                .copied()
                .unwrap_or(sfmtool_core::camera::refit_intrinsics::DEFAULT_COEFF_COUNT);
            let domain_deg = gates.spline_domains_deg.last().copied().unwrap_or(90.0);
            let rows = vec![RowState::default(); gates.cameras.len()];
            self.pending = Some(Pending {
                recon,
                label,
                gates,
                rows,
                keep_coeffs: true,
                coeff_count,
                keep_domain: true,
                domain_deg,
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
                spline_coeffs_row(ui, pending);
                spline_domain_row(ui, pending);
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
            spline_coeff_count: pending.spline_coeff_count(),
            spline_domain_deg: pending.spline_domain_deg(),
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
                    gate.camera, gate.model, gate.images
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

/// The "Spline coefficients" row under the camera rows: a `Keep`
/// checkbox, the count, and the count(s) the node's spline cameras have now.
///
/// Live only while a spline camera in the solve releases its distortion: a new count is a refit the solve then corrects, and without the
/// release it would only approximate the old curve. Editing the count clears
/// `Keep`.
fn spline_coeffs_row(ui: &mut egui::Ui, pending: &mut Pending) {
    let why = spline_row_refusal(pending, "coefficient count");
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
                    "Refit each spline camera that releases its distortion to this many \
                     coefficients over its whole domain before the solve; the solve then fits \
                     them to the observations.",
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

/// Why the spline rows are disabled, or `None` while they are live: the same
/// two reasons for the coefficient count and the domain.
fn spline_row_refusal(pending: &Pending, what: &str) -> Option<String> {
    if pending.gates.spline_coeff_counts.is_empty() {
        Some(
            "No camera of this reconstruction is a spline model (SFMTOOL_FISHEYE, \
             SFMTOOL_PINHOLE)."
                .to_string(),
        )
    } else if !pending.spline_distortion_released() {
        Some(format!(
            "The {what} changes only while a spline camera's lens distortion is released."
        ))
    } else {
        None
    }
}

/// The "Spline domain (°)" row under the coefficients row: a `Keep` checkbox,
/// the domain end, the domain(s) the node's spline cameras have now, and under
/// them the outermost keypoint with a button that sets the domain to it.
///
/// The default domain is the model's own reach, the far image corner, and the
/// row does not change it: the outermost keypoint is offered, so a circular
/// fisheye can be trimmed to its image circle by choice. The button takes the
/// detected keypoint, and where no `.sift` file could be read the observed one,
/// which the text labels as observed.
fn spline_domain_row(ui: &mut egui::Ui, pending: &mut Pending) {
    let why = spline_row_refusal(pending, "spline domain");
    let now = pending
        .gates
        .spline_domains_deg
        .iter()
        .map(|d| format!("{d:.1}°"))
        .collect::<Vec<_>>()
        .join(", ");
    ui.add_enabled_ui(why.is_none(), |ui| {
        ui.horizontal(|ui| {
            ui.label("Spline domain (°)");
            ui.checkbox(&mut pending.keep_domain, "Keep")
                .on_hover_text("Keep each spline camera's domain.");
            let domain = ui
                .add(
                    egui::DragValue::new(&mut pending.domain_deg)
                        .range(1.0..=180.0)
                        .speed(0.1)
                        .fixed_decimals(1),
                )
                .on_hover_text(
                    "Refit each spline camera on a domain ending at this incidence angle, \
                     over the whole of it, before the solve. Past the domain the model is \
                     a straight line the solve cannot bend.",
                );
            if domain.changed() {
                pending.keep_domain = false;
            }
            if !now.is_empty() {
                ui.label(format!("now {now}"));
            }
        })
        .response
        .on_disabled_hover_text(why.clone().unwrap_or_default());
        if let Some(text) = pending.gates.keypoint_extent.describe() {
            ui.horizontal(|ui| {
                ui.label(text);
                if let Some(deg) = pending.gates.keypoint_extent.suggested_domain_deg() {
                    if ui
                        .small_button(format!("Use {deg:.1}°"))
                        .on_hover_text(
                            "Set the domain to the outermost keypoint's angle: the detected \
                             one, or the observed one where no .sift file could be read.",
                        )
                        .clicked()
                    {
                        pending.use_outermost_keypoint();
                    }
                }
            });
        }
    });
}
