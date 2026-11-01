// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The small dialog that `Bundle Adjust...` on a reconstruction's Scene Graph
//! context menu opens, and the gate the menu entry itself reads.
//!
//! See `specs/gui/edits/bundle-adjust.md`. The adjustment takes its decisions
//! about the lens from the user, camera by camera -- whether each camera's
//! focal length is released, whether with it the lens distortion the
//! adjustment can free, whether its principal point is, whether each of its
//! images' focal length and principal point are, and on an `SFMTOOL_PINHOLE`
//! camera whether each image's radial correction and rolling-shutter rates are,
//! with the scanlines the rates are read on -- one about the points: whether a
//! point may cross between a position and a direction -- and the trim schedule.
//! That is the whole dialog: two lines per camera, of three shared and four
//! per-image checkboxes and the Scanlines choice, the crossing checkbox,
//! the schedule's text box with its presets, `Run` and `Cancel`. Everything
//! else about the solve is the core function's defaults. A spline
//! camera's coefficient count and domain are not the adjustment's to change:
//! they are a refit of that camera, made from the Camera Intrinsics panel's
//! `Switch Model…` proposal (`crate::camera_model_proposal`).
//!
//! The gate is here rather than in the menu because the edit reads it too, so
//! the entry and the edit cannot disagree about when the adjustment can run.

use std::collections::HashMap;

use sfmtool_core::camera::RollingShutterScanlines;
use sfmtool_core::geometry::{BaSchedule, FreePointPolicy, DEFAULT_SCHEDULE};
use sfmtool_core::reconstruction::bundle_adjust::{
    check_schedule, distortion_is_releasable, focal_is_releasable,
    image_radial_bspline_is_releasable, image_rolling_shutter_is_releasable, CameraRelease,
    RollingShutterScanlinesChoice,
};
use sfmtool_core::{CameraModel, EditedReconstruction};

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

/// The schedules the preset drop-down fills the text box with, by the name it
/// lists them under: the core function's default, which trims outliers, and
/// one round whose trim never removes an observation, under a loss scale wide
/// enough that the solve is close to least squares.
pub(crate) const SCHEDULE_PRESETS: [(&str, &[BaSchedule]); 2] = [
    ("Default (trims outliers)", &DEFAULT_SCHEDULE),
    (
        "Keep every observation",
        &[BaSchedule {
            trim_px: f64::INFINITY,
            loss_scale: 1000.0,
        }],
    ),
];

/// `schedule` as the text box writes it: each round `(trim_px, loss_scale)`,
/// separated by spaces, with an unbounded trim written `inf`. The default
/// schedule reads `(50, 5) (12, 2) (4, 1)`.
pub(crate) fn schedule_text(schedule: &[BaSchedule]) -> String {
    schedule
        .iter()
        .map(|round| format!("({}, {})", round.trim_px, round.loss_scale))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The schedule the text box states, or why it states none, as the sentence
/// the greyed `Run` button carries.
///
/// The text is a sequence of `(trim_px, loss_scale)` rounds, with spaces or
/// commas between them; `inf` is an unbounded trim. Once the text reads as
/// numbers, whether they are a schedule is [`check_schedule`]'s question, so
/// the dialog refuses what the core function refuses, in its words.
pub(crate) fn parse_schedule(text: &str) -> Result<Vec<BaSchedule>, String> {
    let mut rounds = Vec::new();
    let mut rest = text.trim();
    while !rest.is_empty() {
        let n = rounds.len() + 1;
        let Some(open) = rest.strip_prefix('(') else {
            return Err(format!(
                "Round {n} does not start with '('. Write each round as (trim px, loss scale)."
            ));
        };
        let Some(close) = open.find(')') else {
            return Err(format!("Round {n} has no closing ')'."));
        };
        let inner = &open[..close];
        let number = |field: &str| field.trim().parse::<f64>().ok();
        let round = match inner.split(',').collect::<Vec<_>>()[..] {
            [trim, loss] => number(trim).zip(number(loss)),
            _ => None,
        };
        let Some((trim_px, loss_scale)) = round else {
            return Err(format!(
                "Round {n}, ({inner}), is not two numbers (trim px, loss scale)."
            ));
        };
        rounds.push(BaSchedule {
            trim_px,
            loss_scale,
        });
        rest = open[close + 1..].trim_start();
        rest = rest.strip_prefix(',').unwrap_or(rest).trim_start();
    }
    check_schedule(&rounds).map_err(|e| {
        let why = e.to_string();
        let mut chars = why.chars();
        let first: String = chars
            .next()
            .into_iter()
            .flat_map(char::to_uppercase)
            .collect();
        format!("{first}{}.", chars.as_str())
    })?;
    Ok(rounds)
}

/// One camera the posed images use, as its row in the dialog shows it: which
/// camera, how many posed images it takes, why its focal, distortion,
/// per-image radial or per-image rolling-shutter checkbox is greyed, and the
/// scanlines setting it stores. Its principal-point checkboxes are never
/// greyed for the model, since every model can release its principal point,
/// shared or per image; its per-image focal checkbox is greyed for the focal's
/// reason.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CameraGate {
    /// The camera's index among the image set's cameras.
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
    /// Why its images' radial correction cannot be released, or `None` when it
    /// can: on an `SFMTOOL_PINHOLE` whose spline domain end is positive and
    /// finite. The per-image radial checkbox carries this as its disabled
    /// hover text.
    pub(crate) image_radial_bspline_refusal: Option<String>,
    /// Why its images' rolling-shutter rates cannot be released, or `None` on
    /// an `SFMTOOL_PINHOLE`. The per-image rolling-shutter checkbox carries
    /// this as its disabled hover text.
    pub(crate) image_rolling_shutter_refusal: Option<String>,
    /// The scanlines setting the camera stores, which the row's Scanlines
    /// choice opens on; `None` where it stores none, and the choice opens on
    /// `Unknown`.
    pub(crate) scanlines: Option<RollingShutterScanlines>,
}

/// One [`CameraGate`] per camera the posed images use, in camera index order.
///
/// Each release is decided on that camera's own model, as the core function
/// decides it, so a checkbox is greyed exactly when the core function would
/// refuse the release it asks for, and the hover text names the camera and its
/// model.
pub(crate) fn camera_gates(edited: &EditedReconstruction) -> Vec<CameraGate> {
    let image_set = &edited.base.image_set;
    let posed: Vec<u32> = image_set
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
            // The shared camera: what a release solves is the lens all the
            // camera's images share, each image's adjustment held on top.
            let camera = &image_set.cameras[c as usize];
            let camera_model = camera.model_name();
            let pinhole = matches!(camera.model, CameraModel::SfmtoolPinhole { .. });
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
                image_radial_bspline_refusal: (!image_radial_bspline_is_releasable(camera)).then(
                    || {
                        if pinhole {
                            format!(
                                "The spline of camera {c} has a domain end that is not positive \
                                 and finite, or a single coefficient, so its images' radial \
                                 distortion cannot be released."
                            )
                        } else {
                            format!(
                                "Camera {c}, a {camera_model} camera, cannot release its \
                                 images' radial distortion: the per-image radial correction is \
                                 defined on SFMTOOL_PINHOLE alone. Switch the camera to \
                                 SFMTOOL_PINHOLE first."
                            )
                        }
                    },
                ),
                image_rolling_shutter_refusal: (!image_rolling_shutter_is_releasable(camera))
                    .then(|| {
                        format!(
                            "Camera {c}, a {camera_model} camera, cannot release its images' \
                             rolling shutter: the rolling-shutter rates are defined on \
                             SFMTOOL_PINHOLE alone. Switch the camera to SFMTOOL_PINHOLE first."
                        )
                    }),
                scanlines: image_set
                    .camera_per_image_adjustments(c as usize)
                    .rolling_shutter_scanlines,
            }
        })
        .collect()
}

/// What the dialog reads off the node when it opens: its cameras and why each
/// checkbox is greyed.
#[derive(Debug, Clone, Default)]
pub(crate) struct BundleAdjustGates {
    /// How many cameras the node's image set holds, which the answer's release list
    /// is sized to.
    pub(crate) camera_count: usize,
    /// [`camera_gates`].
    pub(crate) cameras: Vec<CameraGate>,
}

impl BundleAdjustGates {
    /// Every gate, read off `edited`.
    pub(crate) fn of(edited: &EditedReconstruction) -> Self {
        Self {
            camera_count: edited.base.image_set.cameras.len(),
            cameras: camera_gates(edited),
        }
    }
}

/// What the user asked for, once they pressed `Run`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BundleAdjustAnswer {
    /// The node to adjust.
    pub(crate) recon: ReconId,
    /// What each camera releases, one entry per camera in the node's image
    /// set: held for a camera the dialog showed no row for, and never a
    /// release the camera's row had greyed. A distortion release only ever
    /// comes with the focal of the same camera.
    pub(crate) releases: Vec<CameraRelease>,
    /// Whether the free points cross between a position and a direction: the
    /// core function's default unless the user cleared the checkbox.
    pub(crate) free_points: FreePointPolicy,
    /// The trim schedule the text box stated, already checked by
    /// [`check_schedule`].
    pub(crate) schedule: Vec<BaSchedule>,
    /// The scanlines each camera's rolling-shutter rates are solved on, one
    /// entry per camera in the node's image set: the row's Scanlines choice
    /// for a camera whose rates the row releases, and `None` for every other
    /// camera.
    pub(crate) rolling_shutter_scanlines: Vec<Option<RollingShutterScanlinesChoice>>,
}

impl BundleAdjustAnswer {
    /// The options the adjustment runs with: the answer's releases, free-point
    /// policy, schedule and scanlines, and the core function's defaults for
    /// everything else, the per-image prior's widths among them.
    pub(crate) fn options(self) -> sfmtool_core::BundleAdjustOptions {
        sfmtool_core::BundleAdjustOptions {
            releases: self.releases,
            free_points: self.free_points,
            schedule: self.schedule,
            rolling_shutter_scanlines: self.rolling_shutter_scanlines,
            ..sfmtool_core::BundleAdjustOptions::default()
        }
    }
}

/// The dialog, what it remembers while it is up, and the schedule last run on
/// each node.
#[derive(Default)]
pub(crate) struct BundleAdjustPrompt {
    pending: Option<Pending>,
    /// The schedule of the last adjustment started on each node this session,
    /// from the dialog or the wire, which the dialog opens with on that node.
    /// A node with no entry opens with the core function's default.
    last_schedule: HashMap<ReconId, Vec<BaSchedule>>,
}

/// One camera row's seven checkboxes and its Scanlines choice, as the user
/// left them. A ticked per-image focal or principal-point box shows the shared
/// box of its kind ticked and greyed, without changing what the shared box
/// itself holds, so clearing the per-image box gives the shared box back as the
/// user left it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RowState {
    focal: bool,
    distortion: bool,
    principal_point: bool,
    image_focal: bool,
    image_principal_point: bool,
    image_radial_bspline: bool,
    image_rolling_shutter: bool,
    /// The Scanlines choice, kept while the rolling-shutter box is clear, so
    /// that ticking the box again shows what was chosen.
    scanlines: RollingShutterScanlinesChoice,
}

impl Default for RowState {
    /// Every checkbox clear, and the Scanlines choice `Unknown`, which is what
    /// a camera that stores no setting opens on.
    fn default() -> Self {
        Self {
            focal: false,
            distortion: false,
            principal_point: false,
            image_focal: false,
            image_principal_point: false,
            image_radial_bspline: false,
            image_rolling_shutter: false,
            scanlines: RollingShutterScanlinesChoice::Unknown,
        }
    }
}

impl RowState {
    /// A row as the dialog opens it for `gate`'s camera: every checkbox clear,
    /// and the Scanlines choice on the camera's stored setting, or `Unknown`
    /// where it stores none, since an absent setting means the scanlines are
    /// not known.
    fn opening(gate: &CameraGate) -> Self {
        Self {
            scanlines: gate
                .scanlines
                .map_or(RollingShutterScanlinesChoice::Unknown, Into::into),
            ..Self::default()
        }
    }
}

/// The question being asked: which node, what the controls say, and why each
/// is disabled when it is.
struct Pending {
    recon: ReconId,
    label: String,
    gates: BundleAdjustGates,
    /// One per [`BundleAdjustGates::cameras`], in its order.
    rows: Vec<RowState>,
    /// The crossing checkbox, ticked to start with: the core function's
    /// default.
    free_points_cross: bool,
    /// The schedule's text box, as the user left it.
    schedule_text: String,
}

impl Pending {
    /// What camera row `j` releases: what its checkboxes say, less anything its
    /// model cannot release, and the distortion only with the focal, shared or
    /// per image. The principal point is released on any model, with or
    /// without the focal. A per-image focal or principal point is passed as it
    /// is, without the shared one of its kind, which it already moves. The
    /// radial correction and the rolling shutter per image are passed where
    /// the camera can take them.
    fn release(&self, j: usize) -> CameraRelease {
        let gate = &self.gates.cameras[j];
        let row = self.rows[j];
        let focal_ok = gate.focal_refusal.is_none();
        let image_focal = row.image_focal && focal_ok;
        let focal = row.focal && focal_ok && !image_focal;
        CameraRelease {
            focal,
            distortion: (focal || image_focal)
                && row.distortion
                && gate.distortion_refusal.is_none(),
            principal_point: row.principal_point && !row.image_principal_point,
            image_focal,
            image_principal_point: row.image_principal_point,
            image_radial_bspline: row.image_radial_bspline
                && gate.image_radial_bspline_refusal.is_none(),
            image_rolling_shutter: row.image_rolling_shutter
                && gate.image_rolling_shutter_refusal.is_none(),
        }
    }

    /// The release list the answer carries: one entry per camera in the image set,
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

    /// The scanlines list the answer carries: one entry per camera in the image
    /// set, the row's Scanlines choice where the row releases the camera's
    /// rolling shutter, and `None` elsewhere.
    fn scanlines(&self) -> Vec<Option<RollingShutterScanlinesChoice>> {
        let mut scanlines = vec![None; self.gates.camera_count];
        for (j, gate) in self.gates.cameras.iter().enumerate() {
            if let Some(slot) = scanlines.get_mut(gate.camera) {
                *slot = self
                    .release(j)
                    .image_rolling_shutter
                    .then_some(self.rows[j].scanlines);
            }
        }
        scanlines
    }
}

impl BundleAdjustPrompt {
    /// Ask about `recon`.
    ///
    /// Idempotent while the dialog is already up, so a menu item racing itself
    /// cannot stack two of them. Every checkbox starts **clear**: a lens that
    /// moves is a different claim about the capture than a pose that does, and
    /// the default should be the smaller one. Each Scanlines choice starts on
    /// its camera's stored setting, or `Unknown` where it stores none. The
    /// schedule starts as the one last run on `recon` this session, or the
    /// core function's default.
    pub(crate) fn ask(&mut self, recon: ReconId, label: String, gates: BundleAdjustGates) {
        if self.pending.is_none() {
            let rows = gates.cameras.iter().map(RowState::opening).collect();
            let schedule = self
                .last_schedule
                .get(&recon)
                .map_or(&DEFAULT_SCHEDULE[..], Vec::as_slice);
            self.pending = Some(Pending {
                recon,
                label,
                gates,
                rows,
                free_points_cross: sfmtool_core::BundleAdjustOptions::default()
                    .free_points
                    .cross,
                schedule_text: schedule_text(schedule),
            });
        }
    }

    /// Remember `schedule` as the one last run on `recon`, so the dialog opens
    /// with it there. Called when an adjustment starts, whoever asked for it.
    pub(crate) fn remember_schedule(&mut self, recon: ReconId, schedule: &[BaSchedule]) {
        self.last_schedule.insert(recon, schedule.to_vec());
    }

    /// What the open dialog's Schedule box holds, for a test that opens the
    /// dialog through the state.
    #[cfg(test)]
    pub(crate) fn schedule_box(&self) -> Option<&str> {
        self.pending.as_ref().map(|p| p.schedule_text.as_str())
    }

    /// Draw one frame, returning the answer on the frame `Run` is pressed.
    ///
    /// Enter runs and Escape cancels, which is the create-point prompt's
    /// vocabulary: this is a step in a gesture rather than a window to leave
    /// lying open.
    pub(crate) fn show(&mut self, ctx: &egui::Context) -> Option<BundleAdjustAnswer> {
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
                ui.checkbox(
                    &mut pending.free_points_cross,
                    "Store each point as a position or a direction by its rays",
                )
                .on_hover_text(
                    "Solve every point in inverse depth, and at the end store it as a \
                     position where its rays give it a depth and as a direction where they \
                     do not, so a point can cross between the two. Cleared, every point \
                     keeps the representation it has.",
                );
                ui.add_space(8.0);
                schedule_row(ui, &mut pending.schedule_text);
                ui.add_space(8.0);
                let refusal = parse_schedule(&pending.schedule_text).err();
                ui.horizontal(|ui| {
                    let button = ui
                        .add_enabled(refusal.is_none(), egui::Button::new("Run"))
                        .on_disabled_hover_text(refusal.unwrap_or_default());
                    if button.clicked() {
                        run = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
                run |= ui.input(|i| i.key_pressed(egui::Key::Enter));
                cancel |= ui.input(|i| i.key_pressed(egui::Key::Escape));
            });

        // Enter runs only what `Run` would: a box that states no schedule
        // leaves the dialog up.
        let schedule = parse_schedule(&pending.schedule_text).ok();
        let answer = schedule.filter(|_| run).map(|schedule| BundleAdjustAnswer {
            recon: pending.recon,
            releases: pending.releases(),
            free_points: FreePointPolicy {
                cross: pending.free_points_cross,
            },
            schedule,
            rolling_shutter_scanlines: pending.scanlines(),
        });
        if answer.is_some() || cancel || !still_open {
            self.pending = None;
        }
        answer
    }
}

/// The schedule row: its label, the text box, and the preset drop-down that
/// fills the box. The drop-down names the preset the box states, or `Custom`.
fn schedule_row(ui: &mut egui::Ui, text: &mut String) {
    ui.horizontal(|ui| {
        // The box is `labelled_by` the label, which names it; the drop-down
        // has no text beside it, so it has a name of its own.
        let label = ui.label("Schedule");
        ui.add(egui::TextEdit::singleline(text).desired_width(180.0))
            .labelled_by(label.id)
            .on_hover_text(
                "The adjustment's rounds, each (trim px, loss scale). Before a round, every \
                 observation whose reprojection error is at or over its trim is left out of \
                 it, and the round solves the rest under a soft-L1 loss of that scale. A \
                 trim of inf leaves out only observations that do not project; a loss scale \
                 much larger than the errors makes the solve close to least squares.",
            );
        let stated = parse_schedule(text).ok();
        let selected = SCHEDULE_PRESETS
            .iter()
            .find(|(_, preset)| stated.as_deref() == Some(*preset))
            .map_or("Custom", |(name, _)| *name);
        let presets = egui::ComboBox::from_id_salt("bundle_adjust_schedule_preset")
            .selected_text(selected)
            .show_ui(ui, |ui| {
                for (name, preset) in SCHEDULE_PRESETS {
                    if ui.selectable_label(selected == name, name).clicked() {
                        *text = schedule_text(preset);
                    }
                }
            })
            .response;
        crate::widget_name::set_widget_name(&presets, "Schedule preset");
    });
}

/// The camera rows: per camera the posed images use, two lines. The first
/// holds its index, model and image count, and its "Release focal length",
/// "Release lens distortion" and "Release principal point" checkboxes; the
/// second, under them, its "Release focal length per image", "Release radial
/// distortion per image", "Release principal point per image" and "Release
/// rolling shutter per image" checkboxes, and the Scanlines choice while the
/// rolling shutter is ticked.
///
/// A checkbox the camera's model cannot take is greyed, with the reason as its
/// hover text. The distortion checkbox is also greyed while the same row's
/// focal is released neither shared nor per image, and cleared when it is not:
/// neither k1 nor the spline can change the scale at the centre of the image,
/// which is the focal's job. The principal-point checkboxes are enabled on
/// every model and need no focal. A ticked per-image focal or principal-point
/// box shows the shared box of its kind ticked and greyed, since releasing a
/// value per image moves the camera's value to the images' mean.
fn camera_rows(ui: &mut egui::Ui, pending: &mut Pending) {
    egui::Grid::new("bundle_adjust_cameras")
        .num_columns(6)
        .spacing([12.0, 4.0])
        .show(ui, |ui| {
            for (gate, row) in pending.gates.cameras.iter().zip(pending.rows.iter_mut()) {
                let plural = if gate.images == 1 { "" } else { "s" };
                ui.label(format!(
                    "Camera {}  {}  {} image{plural}",
                    gate.camera, gate.camera_model, gate.images
                ));
                let image_focal = row.image_focal && gate.focal_refusal.is_none();
                shared_checkbox(
                    ui,
                    &mut row.focal,
                    image_focal,
                    "Release focal length",
                    gate.focal_refusal.as_deref(),
                    "Solve this camera's focal length along with the poses and the points, \
                     instead of holding it where it is.",
                    "focal length",
                );
                let focal = (row.focal || row.image_focal) && gate.focal_refusal.is_none();
                if !focal {
                    row.distortion = false;
                }
                let distortion_why = gate.distortion_refusal.clone().or_else(|| {
                    (!focal).then(|| {
                        format!(
                            "The lens distortion of camera {} is released only together with its \
                             focal length, shared or per image.",
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
                shared_checkbox(
                    ui,
                    &mut row.principal_point,
                    row.image_principal_point,
                    "Release principal point",
                    None,
                    "Solve this camera's principal point, the pixel its optical axis passes \
                     through, along with the poses and the points. With a narrow field of view \
                     the data barely tell it from a turn of every image, so check how far it \
                     moved.",
                    "principal point",
                );
                ui.end_row();

                // The per-image releases: the focal, the radial distortion and
                // the principal point each under the shared release of its
                // kind, then the rolling shutter, which has none.
                ui.label("");
                ui.add_enabled_ui(gate.focal_refusal.is_none(), |ui| {
                    ui.checkbox(&mut row.image_focal, "Release focal length per image")
                        .on_disabled_hover_text(gate.focal_refusal.clone().unwrap_or_default())
                        .on_hover_text(
                            "Solve a focal length for each image of this camera, a small scale \
                             of the camera's that autofocus changes from one photograph to the \
                             next. Each image is held toward the mean of the camera's images by \
                             a prior whose width the solve derives, and the camera's focal \
                             length moves to that mean.",
                        );
                });
                let radial_why = gate.image_radial_bspline_refusal.as_deref();
                ui.add_enabled_ui(radial_why.is_none(), |ui| {
                    ui.checkbox(
                        &mut row.image_radial_bspline,
                        "Release radial distortion per image",
                    )
                    .on_disabled_hover_text(radial_why.unwrap_or_default())
                    .on_hover_text(
                        "Solve a radial correction for each image of this camera, a small \
                         change to the camera's radial distortion for that photograph, added \
                         to the camera's spline. Each image is held toward the mean of the \
                         camera's images by a prior whose width the solve derives; that mean \
                         is held through each round and then moved into the camera's spline.",
                    );
                });
                ui.checkbox(
                    &mut row.image_principal_point,
                    "Release principal point per image",
                )
                .on_hover_text(
                    "Solve a principal point for each image of this camera, a small offset of \
                     the camera's that a stabilised lens changes from one photograph to the \
                     next. Each image is held toward the mean of the camera's images by a prior \
                     whose width the solve derives, and the camera's principal point moves to \
                     that mean.",
                );
                let shutter_why = gate.image_rolling_shutter_refusal.as_deref();
                ui.add_enabled_ui(shutter_why.is_none(), |ui| {
                    ui.checkbox(
                        &mut row.image_rolling_shutter,
                        "Release rolling shutter per image",
                    )
                    .on_disabled_hover_text(shutter_why.unwrap_or_default())
                    .on_hover_text(
                        "Solve each image's rolling-shutter rates: how far this camera turned \
                         while its sensor read the image out, one scanline at a time. Each \
                         image is held toward the mean of the camera's images by a prior whose \
                         width the solve derives, and that mean is held where it starts.",
                    );
                });
                if row.image_rolling_shutter && shutter_why.is_none() {
                    scanlines_choice(ui, gate.camera, &mut row.scanlines);
                }
                ui.end_row();
            }
        });
}

/// The Scanlines choice of camera `camera`'s row: `Rows`, `Columns` or
/// `Unknown`, named by its `Scanlines` label.
fn scanlines_choice(ui: &mut egui::Ui, camera: usize, choice: &mut RollingShutterScanlinesChoice) {
    const CHOICES: [(RollingShutterScanlinesChoice, &str); 3] = [
        (RollingShutterScanlinesChoice::Rows, "Rows"),
        (RollingShutterScanlinesChoice::Columns, "Columns"),
        (RollingShutterScanlinesChoice::Unknown, "Unknown"),
    ];
    let text = |c: RollingShutterScanlinesChoice| {
        CHOICES
            .iter()
            .find(|(k, _)| *k == c)
            .map_or("Unknown", |(_, text)| *text)
    };
    ui.horizontal(|ui| {
        // `labelled_by` the label, which names the drop-down for a screen
        // reader and in `get_widgets`.
        let label = ui.label("Scanlines");
        egui::ComboBox::from_id_salt(("bundle_adjust_scanlines", camera))
            .selected_text(text(*choice))
            .show_ui(ui, |ui| {
                for (value, text) in CHOICES {
                    ui.selectable_value(choice, value, text);
                }
            })
            .response
            .labelled_by(label.id)
            .on_hover_text(
                "Whether this camera's sensor reads its image one row or one column at a time. \
                 Unknown solves the adjustment once with each, side by side, stops the one \
                 that falls behind once it cannot catch up, and keeps the one with the lower \
                 cost, which takes up to twice as long.",
            );
    });
}

/// A shared release's checkbox: the user's own `value`, or, while the
/// per-image release of its kind is ticked (`implied`), shown ticked and
/// greyed with the reason as its hover text. `refusal` greys it for the
/// camera's model instead.
fn shared_checkbox(
    ui: &mut egui::Ui,
    value: &mut bool,
    implied: bool,
    text: &str,
    refusal: Option<&str>,
    hover: &str,
    what: &str,
) {
    if implied {
        let mut ticked = true;
        ui.add_enabled_ui(false, |ui| {
            ui.checkbox(&mut ticked, text)
                .on_disabled_hover_text(format!(
                    "Released with the {what} per image: the camera's {what} moves to the mean \
                 of its images'."
                ));
        });
        return;
    }
    ui.add_enabled_ui(refusal.is_none(), |ui| {
        ui.checkbox(value, text)
            .on_disabled_hover_text(refusal.unwrap_or_default())
            .on_hover_text(hover);
    });
}
