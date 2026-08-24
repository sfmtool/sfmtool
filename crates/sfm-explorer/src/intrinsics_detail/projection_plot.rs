// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The projection plot: what the lens does to an angle, drawn.
//!
//! Two stacked plots sharing an x axis of incidence angle `θ`. The upper one
//! is the radial map `r(θ)` in pixels with the family's ideal map dashed
//! behind it; the lower one is the residual `Δr = r − r_ref`, which exists
//! because a 12-pixel departure from a 700-pixel curve is invisible on the
//! first. Both are hand-painted with [`egui::Painter`]: the crate has no
//! plotting dependency and should not gain one for a reference curve, an
//! azimuth band and half a dozen labelled rules, none of which a general
//! plotting widget would have shortened.
//!
//! # The shaded region is the point
//!
//! The x axis runs to the frame's own corner angle, which on `kerry_park`'s
//! circular fisheye is 150.5° — and past 84.1° of that the `k1..k4` polynomial
//! is not describing a lens any more but folding, because the image
//! rectangle's corners are outside the lens's image circle where nothing
//! constrained it. Plotting that stretch as though it were the same kind of
//! fact as the rest would be the plot's own version of the number phase 4 had
//! to hedge in a tooltip.
//!
//! So everything past [`sfmtool_core::camera::report::trustworthy_max_theta_deg`]
//! is drawn as extrapolation and says so, three ways at once:
//!
//! - the region is washed over and bounded by a labelled rule carrying the
//!   angle, so the eye separates measurement from extrapolation before reading
//!   anything;
//! - the model's curve continues into it **dotted** rather than solid, so a
//!   reader following the curve is told again at the moment they cross;
//! - both y axes are scaled to the trustworthy samples alone, so the fold does
//!   not flatten the part of the plot that means something. The dotted curve
//!   then dives out through the frame, clipped, which is a fair picture of what
//!   the polynomial is doing.
//!
//! The axis is *not* cut short at the bound. How much of a frame falls outside
//! the lens's modelled domain is exactly the thing a reader of this panel wants
//! to know about a circular fisheye, and it is only visible if the axis still
//! reaches the corner.

use egui::{Align2, Color32, FontId, Pos2, Rect, Shape, Stroke, StrokeKind};
use sfmtool_core::camera::report;
use sfmtool_core::camera::{CameraIntrinsics, CameraModel};

use super::derived::{Derived, ProfileSample};

/// Width reserved left of the plots for their y-axis labels.
const LEFT_GUTTER: f32 = 52.0;
/// Width reserved right of the plots, so the last x tick's label has somewhere
/// to sit.
const RIGHT_PAD: f32 = 14.0;
/// Height of the row above each plot carrying its y-axis title.
const TITLE_HEIGHT: f32 = 14.0;
/// Height of the radial map.
const RADIAL_HEIGHT: f32 = 150.0;
/// Height of the residual, which needs less: it is read for its shape and its
/// extremes, not for absolute values.
const RESIDUAL_HEIGHT: f32 = 96.0;
/// Height of the shared x axis's tick labels, under the lower plot.
const AXIS_HEIGHT: f32 = 16.0;
/// Vertical gap between the two plots, over and above the lower one's title.
const PLOT_GAP: f32 = 8.0;

/// Roughly how many ticks each axis aims for before the 1/2/5 ladder rounds
/// the step to something a person would have chosen.
const X_TICKS: usize = 8;
const Y_TICKS: usize = 4;

/// Headroom past the frame's corner angle, so the last sample is not painted
/// on the plot's own border.
const X_MARGIN: f64 = 1.05;

/// Headroom above the largest plotted value.
const Y_MARGIN: f64 = 1.06;

/// Two labelled edge rules closer together than this are one edge rule: on a
/// square frame with a centred principal point the horizontal and vertical
/// mid-edges are the same angle, and two rules on one line reading `h edge`
/// and `v edge` would be a drawing artefact rather than a fact about the lens.
const EDGE_MERGE_DEG: f64 = 0.1;

/// Dash and gap of the ideal map's stroke.
const DASH: (f32, f32) = (5.0, 4.0);

/// The palette, resolved per frame against the viewer's theme.
struct Palette {
    axis: Color32,
    grid: Color32,
    text: Color32,
    model: Color32,
    ideal: Color32,
    band: Color32,
    extrapolated_wash: Color32,
    extrapolated_rule: Color32,
}

impl Palette {
    fn resolve(visuals: &egui::Visuals) -> Self {
        if visuals.dark_mode {
            Self {
                axis: Color32::from_gray(110),
                grid: Color32::from_gray(58),
                text: visuals.text_color(),
                model: Color32::from_rgb(0x6c, 0xb6, 0xff),
                ideal: Color32::from_rgb(0xe0, 0xa8, 0x4c),
                band: Color32::from_rgba_unmultiplied(0x6c, 0xb6, 0xff, 44),
                extrapolated_wash: Color32::from_rgba_unmultiplied(0xff, 0x8a, 0x5c, 26),
                extrapolated_rule: Color32::from_rgb(0xff, 0x9d, 0x76),
            }
        } else {
            Self {
                axis: Color32::from_gray(120),
                grid: Color32::from_gray(206),
                text: visuals.text_color(),
                model: Color32::from_rgb(0x14, 0x5c, 0xb0),
                ideal: Color32::from_rgb(0x9a, 0x62, 0x00),
                band: Color32::from_rgba_unmultiplied(0x14, 0x5c, 0xb0, 40),
                extrapolated_wash: Color32::from_rgba_unmultiplied(0xc0, 0x4a, 0x10, 26),
                extrapolated_rule: Color32::from_rgb(0xa8, 0x40, 0x0c),
            }
        }
    }
}

/// A labelled vertical rule shared by both plots.
struct Marker {
    theta_deg: f64,
    label: String,
}

/// One linear map from a data interval to a screen interval.
#[derive(Clone, Copy)]
struct Scale {
    lo: f64,
    hi: f64,
    screen_lo: f32,
    screen_hi: f32,
}

impl Scale {
    fn at(&self, value: f64) -> f32 {
        let t = if self.hi > self.lo {
            (value - self.lo) / (self.hi - self.lo)
        } else {
            0.5
        };
        self.screen_lo + (self.screen_hi - self.screen_lo) * t as f32
    }
}

/// The projection plot, or a one-line statement of why there is none.
pub(super) fn show_projection_plot(
    ui: &mut egui::Ui,
    camera: &CameraIntrinsics,
    derived: &Derived,
) {
    ui.label(egui::RichText::new("Projection").strong());

    let samples = &derived.profile.samples;
    let x_max = samples
        .last()
        .map(|s| s.theta_deg)
        .into_iter()
        .chain(derived.fov.map(|fov| fov.max_off_axis))
        .fold(0.0_f64, f64::max)
        * X_MARGIN;
    if samples.len() < 2 || x_max <= 0.0 || x_max.is_nan() {
        ui.label(
            egui::RichText::new(
                "No projection to plot — this camera has no image for an angle to span.",
            )
            .weak(),
        );
        return;
    }

    let limit = derived.trustworthy_max_theta;
    // How many leading samples the model still vouches for. The extrapolated
    // polyline restarts on the last of them, so the two halves join.
    let trusted = match limit {
        Some(limit) => samples.iter().take_while(|s| s.theta_deg <= limit).count(),
        None => samples.len(),
    };

    let width = ui.available_width().max(LEFT_GUTTER + RIGHT_PAD + 80.0);
    let height =
        TITLE_HEIGHT + RADIAL_HEIGHT + PLOT_GAP + TITLE_HEIGHT + RESIDUAL_HEIGHT + AXIS_HEIGHT;
    let (rect, _response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());

    let left = rect.left() + LEFT_GUTTER;
    let right = rect.right() - RIGHT_PAD;
    let upper_top = rect.top() + TITLE_HEIGHT;
    let upper = Rect::from_min_max(
        Pos2::new(left, upper_top),
        Pos2::new(right, upper_top + RADIAL_HEIGHT),
    );
    let lower_top = upper.bottom() + PLOT_GAP + TITLE_HEIGHT;
    let lower = Rect::from_min_max(
        Pos2::new(left, lower_top),
        Pos2::new(right, lower_top + RESIDUAL_HEIGHT),
    );

    let palette = Palette::resolve(ui.visuals());
    let x = Scale {
        lo: 0.0,
        hi: x_max,
        screen_lo: left,
        screen_hi: right,
    };
    let trusted_samples = &samples[..trusted.max(1).min(samples.len())];
    let radial_y = radial_scale(trusted_samples, &upper);
    let residual_y = residual_scale(trusted_samples, &lower);

    let painter = ui.painter().clone();
    let markers = markers(camera, derived, x_max);

    for (plot, y, title) in [
        (&upper, &radial_y, "r (px)"),
        (&lower, &residual_y, "Δr (px)"),
    ] {
        draw_frame(&painter, plot, y, &palette, title);
    }
    draw_x_axis(&painter, &upper, &lower, &x, &palette);

    if let Some(limit) = limit {
        for plot in [&upper, &lower] {
            let start = x.at(limit).max(plot.left());
            if start < plot.right() {
                painter.rect_filled(
                    Rect::from_min_max(
                        Pos2::new(start, plot.top()),
                        Pos2::new(plot.right(), plot.bottom()),
                    ),
                    0.0,
                    palette.extrapolated_wash,
                );
                painter.line_segment(
                    [
                        Pos2::new(start, plot.top()),
                        Pos2::new(start, plot.bottom()),
                    ],
                    Stroke::new(1.5, palette.extrapolated_rule),
                );
            }
        }
        // Right-aligned at the plot's edge rather than beside the rule: the
        // rule can land anywhere, and beside it the label collides with
        // whichever marker the frame's own geometry happens to put nearby.
        painter.text(
            Pos2::new(upper.right() - 4.0, upper.top() + 3.0),
            Align2::RIGHT_TOP,
            format!("extrapolated past {limit:.1}°"),
            FontId::proportional(10.0),
            palette.extrapolated_rule,
        );
    }

    draw_markers(&painter, &markers, &upper, &lower, &x, &palette);
    draw_curves(
        &painter,
        samples,
        trusted,
        derived.profile.band_visible,
        &upper,
        &lower,
        &x,
        &radial_y,
        &residual_y,
        &palette,
    );

    if !camera.has_distortion() {
        draw_no_distortion_banner(&painter, camera, &lower, &palette);
    }

    legend(ui, camera, derived);
}

/// The upper plot's y range: zero to the largest radius any trustworthy sample
/// reaches, on either curve or at either edge of the band.
fn radial_scale(samples: &[ProfileSample], plot: &Rect) -> Scale {
    let hi = samples
        .iter()
        .flat_map(|s| [s.radius_px, s.reference_px, s.band_px.1])
        .fold(0.0_f64, f64::max)
        * Y_MARGIN;
    Scale {
        lo: 0.0,
        hi: hi.max(1.0),
        screen_lo: plot.bottom(),
        screen_hi: plot.top(),
    }
}

/// The lower plot's y range: the trustworthy residuals, always including zero
/// so the zero line is on the plot even for a lens that only ever magnifies.
fn residual_scale(samples: &[ProfileSample], plot: &Rect) -> Scale {
    let mut lo = 0.0_f64;
    let mut hi = 0.0_f64;
    for s in samples {
        for value in [
            s.radius_px - s.reference_px,
            s.band_px.0 - s.reference_px,
            s.band_px.1 - s.reference_px,
        ] {
            lo = lo.min(value);
            hi = hi.max(value);
        }
    }
    // A model with no distortion collapses to a point; give it a symmetric
    // range so the zero line lands in the middle rather than on a border.
    if hi - lo < 1e-9 {
        lo = -1.0;
        hi = 1.0;
    }
    let pad = (hi - lo) * (Y_MARGIN - 1.0);
    Scale {
        lo: lo - pad,
        hi: hi + pad,
        screen_lo: plot.bottom(),
        screen_hi: plot.top(),
    }
}

/// Border, horizontal gridlines with their labels, and the y-axis title.
fn draw_frame(painter: &egui::Painter, plot: &Rect, y: &Scale, palette: &Palette, title: &str) {
    painter.rect_stroke(
        *plot,
        0.0,
        Stroke::new(1.0, palette.axis),
        StrokeKind::Inside,
    );
    painter.text(
        Pos2::new(plot.left(), plot.top() - 2.0),
        Align2::LEFT_BOTTOM,
        title,
        FontId::proportional(10.0),
        palette.text,
    );

    for value in ticks(y.lo, y.hi, Y_TICKS) {
        let py = y.at(value);
        if py < plot.top() || py > plot.bottom() {
            continue;
        }
        painter.line_segment(
            [Pos2::new(plot.left(), py), Pos2::new(plot.right(), py)],
            Stroke::new(1.0, palette.grid),
        );
        painter.text(
            Pos2::new(plot.left() - 4.0, py),
            Align2::RIGHT_CENTER,
            format_tick(value),
            FontId::proportional(9.0),
            palette.text,
        );
    }
}

/// The shared x axis: one ladder of degree ticks, gridded on both plots and
/// labelled once, under the lower one.
fn draw_x_axis(painter: &egui::Painter, upper: &Rect, lower: &Rect, x: &Scale, palette: &Palette) {
    for value in ticks(x.lo, x.hi, X_TICKS) {
        let px = x.at(value);
        if px < upper.left() || px > upper.right() {
            continue;
        }
        for plot in [upper, lower] {
            painter.line_segment(
                [Pos2::new(px, plot.top()), Pos2::new(px, plot.bottom())],
                Stroke::new(1.0, palette.grid),
            );
        }
        painter.text(
            Pos2::new(px, lower.bottom() + 2.0),
            Align2::CENTER_TOP,
            format!("{value:.0}°"),
            FontId::proportional(9.0),
            palette.text,
        );
    }
    painter.text(
        Pos2::new(upper.right(), lower.bottom() + 2.0),
        Align2::RIGHT_TOP,
        "θ off-axis",
        FontId::proportional(9.0),
        palette.text,
    );
}

/// The labelled rules: the frame's own edges and corner, a spline model's
/// domain end, and the 90° a perspective model's projective divide runs into.
fn markers(camera: &CameraIntrinsics, derived: &Derived, x_max: f64) -> Vec<Marker> {
    let mut markers = Vec::new();
    if let Some(fov) = derived.fov {
        let w = f64::from(camera.width);
        let h = f64::from(camera.height);
        let edge = |a: (f64, f64), b: (f64, f64)| {
            report::off_axis_angle_deg(camera, a.0, a.1)
                .max(report::off_axis_angle_deg(camera, b.0, b.1))
        };
        // The wider of each opposed pair: with an off-centre principal point
        // the left and right mid-edges are different angles, and the frame
        // reaches the larger of them.
        let horizontal = edge((0.0, h / 2.0), (w, h / 2.0));
        let vertical = edge((w / 2.0, 0.0), (w / 2.0, h));
        if (horizontal - vertical).abs() < EDGE_MERGE_DEG {
            markers.push(Marker {
                theta_deg: horizontal,
                label: "edge".to_string(),
            });
        } else {
            markers.push(Marker {
                theta_deg: horizontal,
                label: "h edge".to_string(),
            });
            markers.push(Marker {
                theta_deg: vertical,
                label: "v edge".to_string(),
            });
        }
        markers.push(Marker {
            theta_deg: fov.max_off_axis,
            label: "corner".to_string(),
        });
    }

    // Past the spline's domain end `δ` is held constant, so the radial map
    // continues linearly rather than following the fitted curve — a lens whose
    // frame reaches past this rule is being extrapolated, benignly but
    // visibly.
    let spline_end = match &camera.model {
        CameraModel::SfmtoolFisheye {
            bspline_theta_max, ..
        } => Some(bspline_theta_max.to_degrees()),
        CameraModel::SfmtoolPinhole {
            bspline_rho_max, ..
        } => Some(bspline_rho_max.atan().to_degrees()),
        _ => None,
    };
    if let Some(theta_deg) = spline_end {
        markers.push(Marker {
            theta_deg,
            label: "spline domain".to_string(),
        });
    }

    // Only drawn when the frame gets near it. A 40° lens would otherwise spend
    // half the axis on angles its image does not contain in order to show a
    // rule about a ray it never sees.
    if !camera.model.needs_ray_path() && 90.0 <= x_max {
        markers.push(Marker {
            theta_deg: 90.0,
            label: "90° asymptote".to_string(),
        });
    }

    markers.retain(|m| m.theta_deg > 0.0 && m.theta_deg <= x_max);
    markers.sort_by(|a, b| a.theta_deg.total_cmp(&b.theta_deg));
    markers
}

/// Draw the rules across both plots, labelling them in the upper one.
fn draw_markers(
    painter: &egui::Painter,
    markers: &[Marker],
    upper: &Rect,
    lower: &Rect,
    x: &Scale,
    palette: &Palette,
) {
    for (i, marker) in markers.iter().enumerate() {
        let px = x.at(marker.theta_deg);
        if px < upper.left() || px > upper.right() {
            continue;
        }
        for plot in [upper, lower] {
            painter.extend(Shape::dashed_line(
                &[Pos2::new(px, plot.top()), Pos2::new(px, plot.bottom())],
                Stroke::new(1.0, palette.axis),
                3.0,
                3.0,
            ));
        }
        // Two rows, alternating, because the corner and the edge rules of a
        // fisheye can land within a few pixels of each other.
        let row = (i % 2) as f32 * 13.0;
        let near_right = px > upper.right() - 70.0;
        let (align, dx) = if near_right {
            (Align2::RIGHT_TOP, -3.0)
        } else {
            (Align2::LEFT_TOP, 3.0)
        };
        painter.text(
            Pos2::new(px + dx, upper.top() + 19.0 + row),
            align,
            format!("{} {:.1}°", marker.label, marker.theta_deg),
            FontId::proportional(9.0),
            palette.text,
        );
    }
}

/// The band, the ideal map, and the model's own curve, on both plots.
#[allow(clippy::too_many_arguments)]
fn draw_curves(
    painter: &egui::Painter,
    samples: &[ProfileSample],
    trusted: usize,
    band_visible: bool,
    upper: &Rect,
    lower: &Rect,
    x: &Scale,
    radial_y: &Scale,
    residual_y: &Scale,
    palette: &Palette,
) {
    // Clipped, because past the trustworthy bound the dotted curve dives out
    // of a frame scaled to the part that means something — which is the
    // picture, not a defect to design around.
    let upper_painter = painter.with_clip_rect(*upper);
    let lower_painter = painter.with_clip_rect(*lower);

    if band_visible {
        for (plot, y, offset) in [
            (&upper_painter, radial_y, false),
            (&lower_painter, residual_y, true),
        ] {
            // One quad per column rather than a single closed path: each quad
            // is convex, which `convex_polygon` needs and a min/max envelope
            // in general is not.
            for pair in samples.windows(2) {
                let (a, b) = (&pair[0], &pair[1]);
                let base = |s: &ProfileSample, value: f64| {
                    if offset {
                        value - s.reference_px
                    } else {
                        value
                    }
                };
                let (ax, bx) = (x.at(a.theta_deg), x.at(b.theta_deg));
                plot.add(Shape::convex_polygon(
                    vec![
                        Pos2::new(ax, y.at(base(a, a.band_px.0))),
                        Pos2::new(ax, y.at(base(a, a.band_px.1))),
                        Pos2::new(bx, y.at(base(b, b.band_px.1))),
                        Pos2::new(bx, y.at(base(b, b.band_px.0))),
                    ],
                    palette.band,
                    Stroke::NONE,
                ));
            }
        }
    }

    // The residual's zero line: the thing the lower plot is read against.
    lower_painter.line_segment(
        [
            Pos2::new(lower.left(), residual_y.at(0.0)),
            Pos2::new(lower.right(), residual_y.at(0.0)),
        ],
        Stroke::new(1.0, palette.axis),
    );

    let ideal: Vec<Pos2> = samples
        .iter()
        .map(|s| Pos2::new(x.at(s.theta_deg), radial_y.at(s.reference_px)))
        .collect();
    upper_painter.extend(Shape::dashed_line(
        &ideal,
        Stroke::new(1.5, palette.ideal),
        DASH.0,
        DASH.1,
    ));

    let radial: Vec<Pos2> = samples
        .iter()
        .map(|s| Pos2::new(x.at(s.theta_deg), radial_y.at(s.radius_px)))
        .collect();
    let residual: Vec<Pos2> = samples
        .iter()
        .map(|s| {
            Pos2::new(
                x.at(s.theta_deg),
                residual_y.at(s.radius_px - s.reference_px),
            )
        })
        .collect();

    for (plot, points) in [(&upper_painter, &radial), (&lower_painter, &residual)] {
        let split = trusted.min(points.len());
        if split >= 2 {
            plot.add(Shape::line(
                points[..split].to_vec(),
                Stroke::new(1.8, palette.model),
            ));
        }
        // Restart on the last trustworthy point so the two halves join.
        let tail = &points[split.saturating_sub(1)..];
        if tail.len() >= 2 {
            plot.extend(Shape::dotted_line(tail, palette.model, 3.0, 1.0));
        }
    }
}

/// The banner across the residual plot for a model that is exactly its own
/// reference.
fn draw_no_distortion_banner(
    painter: &egui::Painter,
    camera: &CameraIntrinsics,
    lower: &Rect,
    palette: &Palette,
) {
    let what = if camera.model.is_equirectangular() {
        "its own reference map"
    } else if camera.model.is_fisheye() {
        "an equidistant fisheye"
    } else {
        "a pinhole"
    };
    painter.text(
        lower.center(),
        Align2::CENTER_CENTER,
        format!("No distortion — this model is exactly {what}"),
        FontId::proportional(11.0),
        palette.text,
    );
}

/// The key, and the sentence that says what the shaded region is.
fn legend(ui: &mut egui::Ui, camera: &CameraIntrinsics, derived: &Derived) {
    let ideal = if camera.model.is_equirectangular() {
        "ideal (itself)"
    } else if camera.model.is_fisheye() {
        "ideal r = f·θ"
    } else {
        "ideal r = f·tan θ"
    };
    // Words rather than sample glyphs: egui's default font has no box-drawing
    // or geometric-shape coverage, so a `──`/`▨` key renders as tofu.
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new(format!("solid: model · dashed: {ideal}")).small());
        if derived.profile.band_visible {
            ui.label(
                egui::RichText::new("· band: spread over 32 azimuths")
                    .small()
                    .weak(),
            )
            .on_hover_text(
                "The model's radius is not the same at every azimuth — `fx ≠ fy`, or live \
                 tangential or thin-prism terms. A single-azimuth curve hides decentring \
                 distortion completely, which is what the band is for.",
            );
        }
    });
    if let Some(limit) = derived.trustworthy_max_theta {
        let corner = derived.fov.map_or(limit, |fov| fov.max_off_axis);
        ui.label(
            egui::RichText::new(format!(
                "Shaded past {limit:.1}°: this model's own inverse stops inverting its \
                 distortion polynomial there, so neither curve describes the lens beyond it. \
                 The frame reaches {corner:.1}°.",
            ))
            .small()
            .weak(),
        );
    }
}

/// Round `span / count` down to a 1, 2 or 5 times a power of ten, and return
/// the multiples of it that fall inside `[lo, hi]`.
///
/// The 1/2/5 ladder rather than an even division so the labels read as numbers
/// a person would have picked — 20°, not 18.8°.
fn ticks(lo: f64, hi: f64, count: usize) -> Vec<f64> {
    let span = hi - lo;
    if span <= 0.0 || span.is_nan() || count == 0 {
        return Vec::new();
    }
    let rough = span / count as f64;
    let decade = 10.0_f64.powf(rough.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|m| m * decade)
        .find(|s| *s >= rough)
        .unwrap_or(decade * 10.0);
    let first = (lo / step).ceil() * step;
    let mut out = Vec::new();
    let mut value = first;
    while value <= hi + step * 1e-9 && out.len() <= count * 3 {
        out.push(value);
        value += step;
    }
    out
}

/// A y-axis tick label: enough decimals to tell adjacent ticks apart, and no
/// more.
fn format_tick(value: f64) -> String {
    let magnitude = value.abs();
    // Spelled out because the residual's zero tick arrives as `-0.0`
    // (`ceil(lo/step)·step` with a negative `lo`), and `{:.0}` renders that as
    // `-0`.
    if value == 0.0 {
        return "0".to_string();
    }
    if magnitude >= 100.0 {
        format!("{value:.0}")
    } else if magnitude >= 1.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

#[cfg(test)]
mod tests;
