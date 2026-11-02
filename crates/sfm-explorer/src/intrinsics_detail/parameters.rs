// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The three tables: the model's stored parameters, what they mean, and `K`.
//!
//! The parameter table is in `parameter_names()` order — declaration order,
//! which is the order `sfm inspect` prints and which a `BTreeMap` cannot offer:
//! lexicographic order separates related terms and puts `bspline_c10` before
//! `bspline_c2`. The whole point of the table is that it can be diffed against
//! the CLI, so the order and the six decimals are both load-bearing. The
//! spline models' coefficient count and angular domain end are the exception:
//! a count prints as an integer and an angle in degrees
//! ([`format::parameter`]).

use sfmtool_core::camera::{CameraIntrinsics, CameraModel, RollingShutterScanlines};
use sfmtool_core::reconstruction::image_adjustment_statistics::{
    ComponentStatistics, ImageAdjustmentStatistics,
};
use sfmtool_core::reconstruction::PerImageAdjustments;

use super::derived::Derived;
use super::format;

/// What the parameter table states of a camera's per-image parameters: which
/// it varies, with its scanlines setting, and their statistics over its
/// images.
pub(super) struct PerImage {
    pub(super) flags: PerImageAdjustments,
    pub(super) statistics: ImageAdjustmentStatistics,
}

/// Table 1: the model's parameters, in declaration order, and beside each
/// shared parameter the camera also varies per image the statistics of its
/// images' values (`per_image`).
///
/// A camera that varies nothing per image, or a table given no `per_image`
/// (the proposed camera's), is the parameters alone. Otherwise a line under
/// the heading names what varies per image, what does not and what the model
/// cannot carry ([`per_image_summary`]), and each component the camera varies
/// gets a `per image` row of its own ([`statistics_rows`]).
///
/// `id_salt` tells two tables in one frame apart: the current camera's and,
/// while a proposal is open, the proposed one's beside it.
pub(super) fn show_parameters(
    ui: &mut egui::Ui,
    camera: &CameraIntrinsics,
    per_image: Option<&PerImage>,
    id_salt: &str,
) {
    ui.label(egui::RichText::new("Parameters").strong());
    let per_image = per_image.filter(|p| p.flags.any());
    if let Some(per_image) = per_image {
        ui.add(
            egui::Label::new(
                egui::RichText::new(per_image_summary(camera, per_image))
                    .weak()
                    .small(),
            )
            .wrap(),
        );
    }
    let rows = per_image
        .map(|p| statistics_rows(camera, p))
        .unwrap_or_default();
    let parameters = camera.parameters();
    let anchors = anchors(&parameters);
    egui::Grid::new(("intrinsics_parameters", id_salt))
        .num_columns(2)
        .spacing([12.0, 2.0])
        .striped(true)
        .show(ui, |ui| {
            for (k, (name, value)) in parameters.iter().enumerate() {
                ui.monospace(name.as_ref());
                ui.monospace(format::parameter(name.as_ref(), *value));
                ui.end_row();
                for row in rows.iter().filter(|r| anchors.row_of(r.anchor) == Some(k)) {
                    show_statistics_row(ui, row);
                }
            }
            // The rows with no shared parameter of their own to sit under.
            for row in rows.iter().filter(|r| anchors.row_of(r.anchor).is_none()) {
                show_statistics_row(ui, row);
            }
        });
}

/// The shared parameter a `per image` row sits under, or `None` for a row
/// drawn after the table's last parameter: the radial correction, which
/// adjusts the whole spline, and the rates, which no shared parameter is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Anchor {
    Focal,
    PrincipalPointX,
    PrincipalPointY,
}

/// The table rows the anchors sit under: the last focal length (the scale
/// applies to both on a two-focal model) and each principal-point coordinate.
struct Anchors {
    focal: Option<usize>,
    principal_point_x: Option<usize>,
    principal_point_y: Option<usize>,
}

impl Anchors {
    fn row_of(&self, anchor: Option<Anchor>) -> Option<usize> {
        match anchor? {
            Anchor::Focal => self.focal,
            Anchor::PrincipalPointX => self.principal_point_x,
            Anchor::PrincipalPointY => self.principal_point_y,
        }
    }
}

fn anchors(parameters: &[(std::borrow::Cow<'static, str>, f64)]) -> Anchors {
    let position = |name: &str| parameters.iter().position(|(n, _)| n == name);
    Anchors {
        focal: parameters
            .iter()
            .rposition(|(name, _)| name.starts_with("focal_length")),
        principal_point_x: position("principal_point_x"),
        principal_point_y: position("principal_point_y"),
    }
}

/// One `per image` row: what it sits under, its label, its cells and its
/// tooltip.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StatisticsRow {
    anchor: Option<Anchor>,
    pub(super) label: String,
    pub(super) cells: Vec<String>,
    tooltip: String,
}

fn show_statistics_row(ui: &mut egui::Ui, row: &StatisticsRow) {
    ui.add(egui::Label::new(egui::RichText::new(&row.label).monospace().weak()).extend())
        .on_hover_text(&row.tooltip);
    // One label that wraps between the cells and never inside one, its
    // spaces within a cell non-breaking, so at a narrow panel a row takes more
    // lines rather than running past the edge. A label rather than a wrapping
    // layout of one widget per cell, because a grid row centres its cells on
    // the last frame's height, and only a widget laid out at its full height
    // is placed where that height says.
    let text = row
        .cells
        .iter()
        .map(|cell| cell.replace(' ', "\u{a0}"))
        .collect::<Vec<_>>()
        .join("  ");
    ui.add(egui::Label::new(egui::RichText::new(text).monospace()).wrap());
    ui.end_row();
}

/// Whether the model can carry a radial correction and rolling-shutter rates
/// per image: only `SFMTOOL_PINHOLE` can.
fn carries_lens_corrections(camera: &CameraIntrinsics) -> bool {
    matches!(camera.model, CameraModel::SfmtoolPinhole { .. })
}

/// The line under the heading, `Per image, over 17 images: focal, principal
/// point. Not varied: radial correction. Not on OPENCV: …`, which tells a
/// parameter the camera does not vary, and one its model cannot carry, apart
/// from one it varies whose values are all zero.
pub(super) fn per_image_summary(camera: &CameraIntrinsics, per_image: &PerImage) -> String {
    let flags = per_image.flags;
    let lens = carries_lens_corrections(camera);
    let mut varied = Vec::new();
    let mut shared = Vec::new();
    let mut absent = Vec::new();
    for (name, on, carried) in [
        ("focal", flags.focal, true),
        ("principal point", flags.principal_point, true),
        ("radial correction", flags.radial_bspline, lens),
        ("rolling-shutter rates", flags.rolling_shutter, lens),
    ] {
        if on {
            varied.push(name);
        } else if carried {
            shared.push(name);
        } else {
            absent.push(name);
        }
    }
    let n = per_image.statistics.images.len();
    let images = if n == 1 {
        "1 image".to_string()
    } else {
        format!("{n} images")
    };
    let mut text = format!("Per image, over {images}: {}.", varied.join(", "));
    if !shared.is_empty() {
        text.push_str(&format!(" Not varied: {}.", shared.join(", ")));
    }
    if !absent.is_empty() {
        text.push_str(&format!(
            " Not on {}: {}.",
            camera.model.model_name(),
            absent.join(", ")
        ));
    }
    text
}

/// The `per image` rows of a camera, in the order the table draws them: the
/// focal's, the two principal-point offsets', the radial correction's two
/// coefficients and the scanlines setting with the three rates.
pub(super) fn statistics_rows(
    camera: &CameraIntrinsics,
    per_image: &PerImage,
) -> Vec<StatisticsRow> {
    let statistics = &per_image.statistics;
    let n = statistics.images.len();
    let short_side = f64::from(camera.width.min(camera.height));
    let over = format!(
        "Over {n} of the camera's images, its posed ones, or all of them where none is \
         posed: the mean, the standard deviation, the smallest and the largest value, and \
         how many images lie more than two standard deviations from the mean."
    );
    let mut rows = Vec::new();
    let row =
        |anchor: Option<Anchor>, label: &str, s: &ComponentStatistics, unit: Unit, what: &str| {
            StatisticsRow {
                anchor,
                label: format!("  per image {label}"),
                cells: cells(s, unit, n),
                tooltip: format!("{what}. {over}"),
            }
        };
    if let Some(focal) = &statistics.focal {
        rows.push(row(
            Some(Anchor::Focal),
            "focal",
            focal,
            Unit::Percent,
            "Each image's focal scale s, in percent: the image's focal is the camera's times \
             1 + s. The mean is the geometric mean of 1 + s, less one, the scale a drop of the \
             per-image focal folds into the camera, and the standard deviation is that of \
             ln(1 + s)",
        ));
    }
    if let Some([x, y]) = &statistics.principal_point {
        let unit = Unit::Pixels(short_side);
        let what = "Each image's principal-point offset from the camera's, in pixels: the stored \
                    offset times the short side, min(width, height)";
        rows.push(row(Some(Anchor::PrincipalPointX), "Δcx", x, unit, what));
        rows.push(row(Some(Anchor::PrincipalPointY), "Δcy", y, unit, what));
    }
    if let Some([e_0, e_1]) = &statistics.radial_bspline {
        let what = "Each image's radial correction coefficient, as stored, in units of the \
                    normalized radius, added to the camera's spline";
        rows.push(row(None, "e0", e_0, Unit::Number, what));
        rows.push(row(None, "e1", e_1, Unit::Number, what));
    }
    if let Some(rates) = &statistics.rolling_shutter {
        let scanlines = match per_image.flags.rolling_shutter_scanlines {
            Some(RollingShutterScanlines::Rows) => "rows",
            Some(RollingShutterScanlines::Columns) => "columns",
            None => "not known",
        };
        rows.push(StatisticsRow {
            anchor: None,
            label: "  scanlines".to_string(),
            cells: vec![scanlines.to_string()],
            tooltip: "Whether the sensor reads out one row or one column at a time: the camera \
                      setting its images' rolling-shutter rates are read on."
                .to_string(),
        });
        let what = "Each image's rolling-shutter rate, in milliradians per short side of \
                    readout: the stored rate, in radians per short side, times 1000, as Image \
                    Detail's legend states one image's";
        for (label, s) in ["ωx", "ωy", "ωz"].iter().zip(rates) {
            rows.push(row(None, label, s, Unit::MilliradiansPerShortSide, what));
        }
    }
    rows
}

/// How a `per image` row's numbers read.
#[derive(Debug, Clone, Copy)]
enum Unit {
    /// A fraction, shown as a percentage.
    Percent,
    /// A fraction of a short side of this many pixels, shown in pixels.
    Pixels(f64),
    /// A number as stored.
    Number,
    /// Radians per short side of readout, as stored, shown in milliradians,
    /// the unit Image Detail's legend uses, named in a cell of its own.
    MilliradiansPerShortSide,
}

/// `mean +0.120%`, `sd 0.400%`, `min −0.700%`, `max +0.950%`, the unit where
/// it is a cell of its own, and `1 of 17 past 2 sd`.
fn cells(s: &ComponentStatistics, unit: Unit, images: usize) -> Vec<String> {
    let (scale, decimals, suffix) = match unit {
        Unit::Percent => (100.0, 3, "%"),
        Unit::Pixels(short_side) => (short_side, 3, " px"),
        Unit::MilliradiansPerShortSide => (1000.0, 3, ""),
        Unit::Number => (1.0, 6, ""),
    };
    let signed = |v: f64| format!("{}{suffix}", format::signed(scale * v, decimals));
    let mut cells = vec![
        format!("mean {}", signed(s.mean)),
        format!("sd {:.decimals$}{suffix}", scale * s.std),
        format!("min {}", signed(s.min)),
        format!("max {}", signed(s.max)),
    ];
    if let Unit::MilliradiansPerShortSide = unit {
        cells.push("mrad per short side".to_string());
    }
    cells.push(format!("{} of {images} past 2 sd", s.beyond_two_std));
    cells
}

/// Table 2: what the parameters *mean*.
///
/// `id_salt` as for [`show_parameters`]. `aspect_row` shows the aspect row
/// whatever the model, so two tables side by side keep their rows in line: a
/// single-focal model reads `single focal` there.
pub(super) fn show_derived(
    ui: &mut egui::Ui,
    camera: &CameraIntrinsics,
    derived: &Derived,
    id_salt: &str,
    aspect_row: bool,
) {
    ui.label(egui::RichText::new("Derived").strong());
    egui::Grid::new(("intrinsics_derived", id_salt))
        .num_columns(2)
        .spacing([12.0, 2.0])
        .striped(true)
        .show(ui, |ui| {
            let (fx, fy) = camera.focal_lengths();
            let unit = focal_unit(camera);
            ui.label("fx, fy");
            ui.monospace(format!("{fx:.3}, {fy:.3} {unit}"))
                .on_hover_text(focal_unit_note(camera));
            ui.end_row();

            // Hidden for a model that carries a single focal length: there is
            // no aspect ratio to report when the file cannot express one, and
            // a row reading 1.0000 for every SIMPLE_RADIAL in the world would
            // be noise.
            if has_two_focal_lengths(camera) && fx != 0.0 {
                ui.label("aspect fy/fx");
                ui.monospace(format!("{:.4}", fy / fx));
                ui.end_row();
            } else if aspect_row {
                ui.label("aspect fy/fx");
                ui.monospace("single focal");
                ui.end_row();
            }

            let (cx, cy) = camera.principal_point();
            let (dx, dy) = (
                cx - f64::from(camera.width) / 2.0,
                cy - f64::from(camera.height) / 2.0,
            );
            let half_diagonal = f64::from(camera.width).hypot(f64::from(camera.height)) / 2.0;
            ui.label("principal point offset");
            let percent = if half_diagonal > 0.0 {
                format!(
                    " · {:.2}% of half-diagonal",
                    100.0 * dx.hypot(dy) / half_diagonal
                )
            } else {
                String::new()
            };
            wrapped(ui, format!("({dx:+.3}, {dy:+.3}) px{percent}")).on_hover_text(
                "From the image centre (w/2, h/2), which is generally not the principal point",
            );
            ui.end_row();

            if let Some(fov) = &derived.fov {
                for (label, value) in [
                    ("horizontal FOV", fov.horizontal),
                    ("vertical FOV", fov.vertical),
                    ("diagonal FOV", fov.diagonal),
                ] {
                    ui.label(label);
                    ui.monospace(format!("{value:.1}°"));
                    ui.end_row();
                }
                ui.label("max off-axis angle");
                ui.monospace(format!("{:.1}°", fov.max_off_axis))
                    .on_hover_text(
                        "The largest incidence angle over the four image corners — a half-angle, \
                         so roughly half the diagonal span. For an equirectangular panorama the \
                         corners are the poles, so it reads 90° whatever the panorama covers.",
                    );
                ui.end_row();
            }

            if let Some(mm) = derived.equiv_35mm {
                ui.label("35 mm equivalent");
                ui.monospace(format!("{mm:.1} mm")).on_hover_text(
                    "f_px · 43.267 / diagonal_px — sensor-independent by construction, since \
                         both are in pixels. Undefined for a model whose focal length is pixels \
                         per radian.",
                );
                ui.end_row();
            }

            ui.label("distortion");
            match &derived.max_distortion {
                // A bound that excluded nothing is not worth qualifying the
                // row with: it is the same statement as "over the image", one
                // clause longer.
                Some(extent) => match extent.limit_deg.filter(|_| extent.excluded > 0) {
                    Some(limit) => {
                        let (dropped, total) = (extent.excluded, extent.total());
                        wrapped(
                            ui,
                            format!("yes — max {:.1} px inside {limit:.1}°", extent.max_px),
                        )
                        .on_hover_text(format!(
                            "The largest |model − ideal| displacement over a grid of the image, \
                             taken inside {limit:.1}° off-axis. Past that angle this model's own \
                             inverse stops inverting its distortion polynomial and slews toward \
                             the identity ray, so neither map describes the lens any more — \
                             {dropped} of the grid's {total} nodes look further out than that \
                             and are excluded. For a circular fisheye those nodes are the black \
                             corners outside the lens's image circle.",
                        ));
                    }
                    None => {
                        wrapped(
                            ui,
                            format!("yes — max {:.1} px over the image", extent.max_px),
                        )
                        .on_hover_text(
                            "The largest |model − ideal| displacement over a grid of the \
                                 image. This model is trustworthy at every angle its frame \
                                 reaches, so the whole rectangle counts.",
                        );
                    }
                },
                None => {
                    ui.monospace("none").on_hover_text(
                        "Every distortion coefficient is zero, so this model is exactly its own \
                         ideal map.",
                    );
                }
            }
            ui.end_row();
        });
}

/// A monospace value that wraps at the panel's right edge, so a long one (the
/// principal point offset, the distortion) takes a second line in a narrow
/// panel rather than running past the edge.
fn wrapped(ui: &mut egui::Ui, text: String) -> egui::Response {
    ui.add(egui::Label::new(egui::RichText::new(text).monospace()).wrap())
}

/// Table 3: `K`, and the two sentences that keep it from being pasted into the
/// wrong frame.
pub(super) fn show_k(ui: &mut egui::Ui, camera: &CameraIntrinsics) {
    ui.label(egui::RichText::new("K").strong());
    let k = camera.intrinsic_matrix();
    let rows: Vec<Vec<f64>> = (0..3)
        .map(|r| (0..3).map(|c| k[(r, c)]).collect())
        .collect();
    format::matrix_grid(ui, "intrinsics_k", &rows);
    ui.label(
        egui::RichText::new(
            "K is in the optical frame (+Z forward, +Y down). Stored poses are canonical \
             (−Z forward, +Y up), so the projection matrix is P = K · S · [R|t] with \
             S = diag(1, −1, −1) — not K · [R|t].",
        )
        .weak()
        .small(),
    );
}

/// `px`, or `px/rad` for the models whose focal length is pixels per radian.
///
/// Mislabelling this is the single easiest way to make a fisheye's focal length
/// look absurd: 129 px would be a catastrophic lens and 129 px/rad is a 200°
/// fisheye on a 480-pixel frame.
pub(super) fn focal_unit(camera: &CameraIntrinsics) -> &'static str {
    if camera.model.needs_ray_path() {
        "px/rad"
    } else {
        "px"
    }
}

/// The tooltip behind [`focal_unit`].
fn focal_unit_note(camera: &CameraIntrinsics) -> &'static str {
    if camera.model.needs_ray_path() {
        "This model maps an incidence angle to a radius, so its focal length is pixels per radian."
    } else {
        "This model maps an image-plane coordinate to a pixel, so its focal length is pixels."
    }
}

/// Whether the model carries `fx` and `fy` separately.
///
/// Read off `parameter_names()` rather than compared numerically: a two-focal
/// model whose two values happen to be equal still has an aspect ratio the file
/// can express, and a single-focal model never does.
pub(super) fn has_two_focal_lengths(camera: &CameraIntrinsics) -> bool {
    camera
        .model
        .parameter_names()
        .iter()
        .any(|name| name == "focal_length_y")
}
