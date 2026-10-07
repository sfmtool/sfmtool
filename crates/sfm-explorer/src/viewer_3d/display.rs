// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The 3D viewport's display controls as one list of fields: the name each has
//! on the wire, the word the HUD labels it with, the range its slider allows,
//! and the Action Log text a change to it records.
//!
//! Two surfaces change these values. The HUD draws its checkboxes and sliders
//! from [`Field`](crate::viewer_3d::display::Field), and the MCP tools
//! `get_viewer_3d_display` and `set_viewer_3d_display` read and write the same
//! fields through it. Both record a change with
//! [`Field::record`](crate::viewer_3d::display::Field::record), so the row a
//! person's click leaves and the row an agent's call leaves for one control
//! read the same, and the range the tool accepts is the range the slider is
//! built with.
//!
//! Every value lives on `AppState` except `maintain_z_up`, which is navigation
//! state on [`Viewer3D`] beside the camera it turns. The field of view is not
//! here: it is part of the view, which `set_view` sets and the view block
//! reports.
//!
//! See `specs/gui/viewport-hud.md` and `specs/gui/mcp-server.md` §
//! "`get_viewer_3d_display` / `set_viewer_3d_display`".

use eframe::egui;

use crate::action_log::{ActionLog, Kind};
use crate::state::AppState;

use super::Viewer3D;

// The items marked `allow(dead_code)` without the `mcp` feature are the ones
// only the MCP tools use: the list of every field, reading and writing a field
// through it, the snapshot of all of them and its diff, and the range check and
// wording of a refusal. The HUD uses the rest in every build.

/// What a slider allows and how it shows its value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SliderRange {
    pub(crate) min: f32,
    pub(crate) max: f32,
    /// Decimals the slider shows, and rounds the value it holds to.
    pub(crate) decimals: usize,
    pub(crate) logarithmic: bool,
    /// Written after the number in the Action Log text, `" px"` or empty.
    pub(crate) unit: &'static str,
}

impl SliderRange {
    const fn linear(min: f32, max: f32, decimals: usize) -> Self {
        Self {
            min,
            max,
            decimals,
            logarithmic: false,
            unit: "",
        }
    }

    const fn logarithmic(min: f32, max: f32, decimals: usize) -> Self {
        Self {
            min,
            max,
            decimals,
            logarithmic: true,
            unit: "",
        }
    }

    const fn with_unit(self, unit: &'static str) -> Self {
        Self { unit, ..self }
    }

    /// The HUD's slider over `value`, without its label.
    pub(crate) fn slider<'a>(&self, value: &'a mut f32) -> egui::Slider<'a> {
        egui::Slider::new(value, self.min..=self.max)
            .logarithmic(self.logarithmic)
            .fixed_decimals(self.decimals)
    }

    /// Whether `value` is one the slider can hold: finite and inside the
    /// range, ends included.
    ///
    /// The ends are compared as the decimals they are written as, not as the
    /// `f32` they are stored in: `0.001_f32` widens to `0.0010000000474974513`,
    /// which is above the `0.001` a caller sends, so comparing against the
    /// widened `f32` would refuse the bottom of the Scene slider.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn contains(&self, value: f64) -> bool {
        let end = |end: f32| egui::emath::round_to_decimals(f64::from(end), self.decimals);
        value.is_finite() && value >= end(self.min) && value <= end(self.max)
    }

    /// `value` rounded to the decimals the slider shows, as `f32`.
    ///
    /// The slider rounds the value it holds to its shown decimals on every
    /// frame it is drawn, so a value set outside it at finer precision would be
    /// changed the next time the HUD is open. Rounding on the way in, with the
    /// same function the slider uses, keeps what was set and what the HUD shows
    /// the same number. Every range's ends are whole at its decimals, so a
    /// value inside the range stays inside it.
    pub(crate) fn round(&self, value: f64) -> f32 {
        egui::emath::round_to_decimals(value, self.decimals) as f32
    }

    /// `value` moved to the nearest end of the range when it is outside it,
    /// then [rounded](Self::round): the value the slider would hold after the
    /// next frame it is drawn.
    ///
    /// For a value the viewer computes rather than one a person or an agent
    /// chose, such as the scene scale measured from the points at load. Stored
    /// as it is, such a value could be one `set_viewer_3d_display` refuses, and
    /// the HUD would change it the first time it drew it.
    pub(crate) fn clamp_round(&self, value: f64) -> f32 {
        self.round(value.clamp(f64::from(self.min), f64::from(self.max)))
    }

    /// `0.05 to 5`, `1 to 16 px`: the range as a refusal lists it.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn describe(&self) -> String {
        format!("{} to {}{}", self.min, self.max, self.unit)
    }
}

/// A checkbox or a slider.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Control {
    Checkbox,
    Slider(SliderRange),
}

/// One field's value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum FieldValue {
    Flag(bool),
    Number(f32),
}

/// The point-size slider, log₂ of the multiplier, shared by patch size.
const SIZE_LOG2: SliderRange = SliderRange::linear(-3.0, 3.0, 1);
/// A fraction from nothing to all, for opacity and the edge cutoff.
const FRACTION: SliderRange = SliderRange::linear(0.0, 1.0, 2);
/// Frustum and target size, as multiples of the scene scale.
const MULTIPLIER: SliderRange = SliderRange::logarithmic(0.05, 5.0, 2);

/// One display control of the 3D viewport.
///
/// [`Field::ALL`] is the document's order: the Layers toggles, the Size
/// sliders, the Patches sliders, the Camera section's checkbox, the Advanced
/// sliders and the Debug toggles, which is the order the HUD draws them in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Field {
    ShowPoints,
    ShowCameraImages,
    ShowGrid,
    ShowPatches,
    ShowPointsAtInfinity,
    ShowTargetIndicator,
    PointSizeLog2,
    InfinityPointPx,
    LengthScale,
    PatchOpacity,
    PatchSizeLog2,
    PatchAlphaCutoff,
    MaintainZUp,
    EdlLineThickness,
    FrustumSizeMultiplier,
    TargetSizeMultiplier,
    TargetFogMultiplier,
    ShowControlsHelp,
    ShowFps,
}

impl Field {
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) const ALL: [Field; 19] = [
        Field::ShowPoints,
        Field::ShowCameraImages,
        Field::ShowGrid,
        Field::ShowPatches,
        Field::ShowPointsAtInfinity,
        Field::ShowTargetIndicator,
        Field::PointSizeLog2,
        Field::InfinityPointPx,
        Field::LengthScale,
        Field::PatchOpacity,
        Field::PatchSizeLog2,
        Field::PatchAlphaCutoff,
        Field::MaintainZUp,
        Field::EdlLineThickness,
        Field::FrustumSizeMultiplier,
        Field::TargetSizeMultiplier,
        Field::TargetFogMultiplier,
        Field::ShowControlsHelp,
        Field::ShowFps,
    ];

    /// The name on the wire, which is the name of the field it is stored in.
    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            Field::ShowPoints => "show_points",
            Field::ShowCameraImages => "show_camera_images",
            Field::ShowGrid => "show_grid",
            Field::ShowPatches => "show_patches",
            Field::ShowPointsAtInfinity => "show_points_at_infinity",
            Field::ShowTargetIndicator => "show_target_indicator",
            Field::PointSizeLog2 => "point_size_log2",
            Field::InfinityPointPx => "infinity_point_px",
            Field::LengthScale => "length_scale",
            Field::PatchOpacity => "patch_opacity",
            Field::PatchSizeLog2 => "patch_size_log2",
            Field::PatchAlphaCutoff => "patch_alpha_cutoff",
            Field::MaintainZUp => "maintain_z_up",
            Field::EdlLineThickness => "edl_line_thickness",
            Field::FrustumSizeMultiplier => "frustum_size_multiplier",
            Field::TargetSizeMultiplier => "target_size_multiplier",
            Field::TargetFogMultiplier => "target_fog_multiplier",
            Field::ShowControlsHelp => "show_controls_help",
            Field::ShowFps => "show_fps",
        }
    }

    /// The word the Action Log entry opens with, which is also the run a
    /// repeat of the same control folds under. For a checkbox it is the
    /// checkbox's own label.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Field::ShowPoints => "Points",
            Field::ShowCameraImages => "Camera Images",
            Field::ShowGrid => "Grid",
            Field::ShowPatches => "Patches",
            Field::ShowPointsAtInfinity => "Points at ∞",
            Field::ShowTargetIndicator => "Target indicator",
            Field::PointSizeLog2 => "Point size",
            Field::InfinityPointPx => "∞ point size",
            Field::LengthScale => "Scene scale",
            Field::PatchOpacity => "Patch opacity",
            Field::PatchSizeLog2 => "Patch size",
            Field::PatchAlphaCutoff => "Patch edge cutoff",
            Field::MaintainZUp => super::MAINTAIN_Z_UP_LABEL,
            Field::EdlLineThickness => "EDL width",
            Field::FrustumSizeMultiplier => "Frustum size",
            Field::TargetSizeMultiplier => "Target size",
            Field::TargetFogMultiplier => "Target fog",
            Field::ShowControlsHelp => "Controls help",
            Field::ShowFps => "Frame rate",
        }
    }

    /// The widget the HUD draws for it, and for a slider its range.
    pub(crate) fn control(self) -> Control {
        let slider = match self {
            Field::ShowPoints
            | Field::ShowCameraImages
            | Field::ShowGrid
            | Field::ShowPatches
            | Field::ShowPointsAtInfinity
            | Field::ShowTargetIndicator
            | Field::MaintainZUp
            | Field::ShowControlsHelp
            | Field::ShowFps => return Control::Checkbox,
            Field::PointSizeLog2 | Field::PatchSizeLog2 => SIZE_LOG2,
            Field::InfinityPointPx => SliderRange::linear(1.0, 16.0, 1).with_unit(" px"),
            Field::LengthScale => SliderRange::logarithmic(0.001, 100.0, 3),
            Field::PatchOpacity | Field::PatchAlphaCutoff => FRACTION,
            Field::EdlLineThickness => SliderRange::linear(0.5, 8.0, 1),
            Field::FrustumSizeMultiplier | Field::TargetSizeMultiplier => MULTIPLIER,
            Field::TargetFogMultiplier => SliderRange::logarithmic(0.5, 100.0, 1),
        };
        Control::Slider(slider)
    }

    /// The slider's range, for a field the HUD draws as a slider.
    ///
    /// # Panics
    ///
    /// On a checkbox field, which is a mistake in the caller and not a value.
    pub(crate) fn range(self) -> SliderRange {
        match self.control() {
            Control::Slider(range) => range,
            Control::Checkbox => panic!("{} is a checkbox, not a slider", self.wire_name()),
        }
    }

    /// The Action Log text for the field at `value`: `Grid off`,
    /// `Point size 1.5`, `∞ point size 3.0 px`.
    pub(crate) fn text(self, value: FieldValue) -> String {
        let label = self.label();
        match (value, self.control()) {
            (FieldValue::Flag(on), _) => format!("{label} {}", if on { "on" } else { "off" }),
            (FieldValue::Number(number), Control::Slider(range)) => {
                let decimals = range.decimals;
                format!("{label} {number:.decimals$}{}", range.unit)
            }
            // A number for a checkbox is not a value this list produces.
            (FieldValue::Number(number), Control::Checkbox) => format!("{label} {number}"),
        }
    }

    /// Record the field's new value as one `Display` entry, folding into the
    /// newest entry when that one is a value of the same control.
    ///
    /// The one place an entry for these controls is written, whether a HUD
    /// widget or `set_viewer_3d_display` changed the value.
    pub(crate) fn record(self, log: &mut ActionLog, value: FieldValue) {
        log.record_run(Kind::Display, self.label(), self.text(value));
    }

    /// Record the field's value after a HUD widget's frame, when the widget
    /// changed it and a person was on the widget.
    ///
    /// The gate is [`ActionLog::is_action`]: an `egui::Slider` rounds and
    /// clamps the value it is handed on every frame it is drawn and reports
    /// that as a change, so a change with nobody on the widget is not an
    /// action and records nothing.
    pub(crate) fn record_widget(
        self,
        log: &mut ActionLog,
        response: &egui::Response,
        value: FieldValue,
    ) {
        if ActionLog::is_action(response) {
            self.record(log, value);
        }
    }

    /// The field's value as it stands.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn read(self, state: &AppState, viewer: &Viewer3D) -> FieldValue {
        use FieldValue::{Flag, Number};
        match self {
            Field::ShowPoints => Flag(state.show_points),
            Field::ShowCameraImages => Flag(state.show_camera_images),
            Field::ShowGrid => Flag(state.show_grid),
            Field::ShowPatches => Flag(state.show_patches),
            Field::ShowPointsAtInfinity => Flag(state.show_points_at_infinity),
            Field::ShowTargetIndicator => Flag(state.show_target_indicator),
            Field::PointSizeLog2 => Number(state.point_size_log2),
            Field::InfinityPointPx => Number(state.infinity_point_px),
            Field::LengthScale => Number(state.length_scale),
            Field::PatchOpacity => Number(state.patch_opacity),
            Field::PatchSizeLog2 => Number(state.patch_size_log2),
            Field::PatchAlphaCutoff => Number(state.patch_alpha_cutoff),
            Field::MaintainZUp => Flag(viewer.maintain_z_up),
            Field::EdlLineThickness => Number(state.edl_line_thickness),
            Field::FrustumSizeMultiplier => Number(state.frustum_size_multiplier),
            Field::TargetSizeMultiplier => Number(state.target_size_multiplier),
            Field::TargetFogMultiplier => Number(state.target_fog_multiplier),
            Field::ShowControlsHelp => Flag(state.show_controls_help),
            Field::ShowFps => Flag(state.show_fps),
        }
    }

    /// Store `value` in the field. A value of the other kind is ignored: the
    /// parse that builds a change gives each field the kind its control takes.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn write(self, state: &mut AppState, viewer: &mut Viewer3D, value: FieldValue) {
        match value {
            FieldValue::Flag(on) => {
                let slot = match self {
                    Field::ShowPoints => &mut state.show_points,
                    Field::ShowCameraImages => &mut state.show_camera_images,
                    Field::ShowGrid => &mut state.show_grid,
                    Field::ShowPatches => &mut state.show_patches,
                    Field::ShowPointsAtInfinity => &mut state.show_points_at_infinity,
                    Field::ShowTargetIndicator => &mut state.show_target_indicator,
                    Field::MaintainZUp => &mut viewer.maintain_z_up,
                    Field::ShowControlsHelp => &mut state.show_controls_help,
                    Field::ShowFps => &mut state.show_fps,
                    _ => return,
                };
                *slot = on;
            }
            FieldValue::Number(number) => {
                let slot = match self {
                    Field::PointSizeLog2 => &mut state.point_size_log2,
                    Field::InfinityPointPx => &mut state.infinity_point_px,
                    Field::LengthScale => &mut state.length_scale,
                    Field::PatchOpacity => &mut state.patch_opacity,
                    Field::PatchSizeLog2 => &mut state.patch_size_log2,
                    Field::PatchAlphaCutoff => &mut state.patch_alpha_cutoff,
                    Field::EdlLineThickness => &mut state.edl_line_thickness,
                    Field::FrustumSizeMultiplier => &mut state.frustum_size_multiplier,
                    Field::TargetSizeMultiplier => &mut state.target_size_multiplier,
                    Field::TargetFogMultiplier => &mut state.target_fog_multiplier,
                    _ => return,
                };
                *slot = number;
            }
        }
    }
}

/// Set the scene scale to `seed`, the scale measured from the loaded points,
/// held to the range and decimals of its slider.
///
/// The viewer writes `length_scale` itself when a node arrives or a transform
/// changes, and the measured value can fall outside the slider: a scene in
/// millimetres measures in the hundreds. Holding it to the slider keeps
/// `get_viewer_3d_display` from reporting a value `set_viewer_3d_display` would
/// refuse, and keeps the value it reports equal to the one the HUD shows. It
/// records nothing: no one chose the value.
pub(crate) fn seed_length_scale(state: &mut AppState, seed: f32) {
    state.length_scale = Field::LengthScale.range().clamp_round(f64::from(seed));
}

/// Every field's value at one moment, in [`Field::ALL`]'s order.
///
/// It exists to be diffed by [`record_viewer_3d_display_changes`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(not(feature = "mcp"), allow(dead_code))]
pub(crate) struct Viewer3dDisplay([FieldValue; Field::ALL.len()]);

#[cfg_attr(not(feature = "mcp"), allow(dead_code))]
impl Viewer3dDisplay {
    /// Copy every field as it stands.
    pub(crate) fn snapshot(state: &AppState, viewer: &Viewer3D) -> Self {
        Self(Field::ALL.map(|field| field.read(state, viewer)))
    }

    /// Each field with its value, in [`Field::ALL`]'s order.
    pub(crate) fn fields(&self) -> impl Iterator<Item = (Field, FieldValue)> + '_ {
        Field::ALL.into_iter().zip(self.0.iter().copied())
    }
}

/// Record what changed between two snapshots, one [`Kind::Display`] entry per
/// field that differs, through [`Field::record`]. A field that did not change
/// records nothing, and each field is its own run, so a change to three fields
/// leaves three rows.
#[cfg_attr(not(feature = "mcp"), allow(dead_code))]
pub(crate) fn record_viewer_3d_display_changes(
    log: &mut ActionLog,
    before: &Viewer3dDisplay,
    after: &Viewer3dDisplay,
) {
    for ((field, was), (_, now)) in before.fields().zip(after.fields()) {
        if was != now {
            field.record(log, now);
        }
    }
}
