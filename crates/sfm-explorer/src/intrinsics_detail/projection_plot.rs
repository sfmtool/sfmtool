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
//!
//! # While a proposal is open
//!
//! The plot compares the current camera with the proposed one
//! ([`Comparison`]): both radial maps and both residuals are drawn, the
//! current camera in a neutral stroke and the proposed one in the accent, each
//! with its own extrapolated region shaded in its own colour. A third plot,
//! `Change (px)`, draws the proposed radius minus the current one with its
//! azimuth band, a rule at the fit's largest angle and a mark on the θ axis
//! where the monotonicity constraint bound. Under the shared axis an
//! observation rug puts one tick at the incidence angle of each observation of
//! the camera's images, under the current camera.

use egui::{Align2, Color32, FontId, Pos2, Rect, Shape, Stroke, StrokeKind, Vec2};
use sfmtool_core::camera::refit_intrinsics::spline_domain_deg;
use sfmtool_core::camera::report;
use sfmtool_core::camera::CameraIntrinsics;

use super::derived::{ChangeProfile, Derived, ProfileSample};

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
/// Height of the change plot, read the way the residual is.
const CHANGE_HEIGHT: f32 = 96.0;
/// Height of the shared x axis's tick labels, under the lowest plot.
const AXIS_HEIGHT: f32 = 16.0;
/// Height of the observation rug, under the axis's labels.
const RUG_HEIGHT: f32 = 14.0;
/// Vertical gap between two plots, over and above the lower one's title.
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

/// Font size of the rules' labels and the axes' tick labels.
const SMALL_FONT: f32 = 9.0;

/// Font size of the plots' titles and the extrapolated regions' labels.
const LABEL_FONT: f32 = 10.0;

/// Line height of a row of rule labels.
const LABEL_ROW: f32 = 13.0;

/// Rows of rule labels the upper plot holds before a label that fits on none
/// of them is left off, its rule still drawn.
const LABEL_ROWS: usize = 6;

/// The palette, resolved per frame against the viewer's theme.
struct Palette {
    axis: Color32,
    grid: Color32,
    text: Color32,
    /// The camera's curve, and while a proposal is open the proposed camera's.
    model: Color32,
    ideal: Color32,
    band: Color32,
    extrapolated_wash: Color32,
    extrapolated_rule: Color32,
    /// The current camera's curve while a proposal is open.
    current: Color32,
    current_band: Color32,
    current_wash: Color32,
    /// The proposed camera's extrapolated region.
    proposed_wash: Color32,
    /// Where the monotonicity constraint bound.
    monotone: Color32,
    /// The plate under each label: the panel's own colour, nearly opaque.
    plate: Color32,
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
                current: Color32::from_gray(205),
                current_band: Color32::from_rgba_unmultiplied(205, 205, 205, 40),
                current_wash: Color32::from_rgba_unmultiplied(205, 205, 205, 30),
                proposed_wash: Color32::from_rgba_unmultiplied(0x6c, 0xb6, 0xff, 22),
                monotone: Color32::from_rgb(0xd0, 0x8c, 0xff),
                plate: plate(visuals),
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
                current: Color32::from_gray(70),
                current_band: Color32::from_rgba_unmultiplied(70, 70, 70, 36),
                current_wash: Color32::from_rgba_unmultiplied(70, 70, 70, 26),
                proposed_wash: Color32::from_rgba_unmultiplied(0x14, 0x5c, 0xb0, 20),
                monotone: Color32::from_rgb(0x7a, 0x2c, 0xb0),
                plate: plate(visuals),
            }
        }
    }
}

/// The plate under the plots' labels: the panel's fill, nearly opaque.
fn plate(visuals: &egui::Visuals) -> Color32 {
    let fill = visuals.panel_fill;
    Color32::from_rgba_unmultiplied(fill.r(), fill.g(), fill.b(), 215)
}

/// What the plot draws beside the current camera while a proposal with a
/// fitted camera is open on it.
pub(super) struct Comparison<'a> {
    /// The proposed camera.
    pub(crate) proposed: &'a CameraIntrinsics,
    /// Its derived report, cached by the proposal's generation.
    pub(crate) derived: &'a Derived,
    /// The change from the current camera to it, cached with the report.
    pub(crate) change: &'a ChangeProfile,
    /// The largest angle the fit sampled, in degrees.
    pub(crate) theta_fit_deg: f64,
    /// The smallest and largest angle at which the monotonicity constraint
    /// held the fit's slope at its floor, or `None` when it did not bind.
    pub(crate) monotone_deg: Option<[f64; 2]>,
}

/// A labelled vertical rule shared by every plot.
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

/// How one camera's curves are drawn: its samples, how many of them it
/// vouches for, and its colours.
struct Curve<'a> {
    samples: &'a [ProfileSample],
    trusted: usize,
    band_visible: bool,
    stroke: Color32,
    /// The solid stroke's width. While a proposal is open the current camera
    /// is drawn wider, under the proposed one, so where the two agree it shows
    /// as a grey edge either side of the blue rather than vanishing under it.
    width: f32,
    band: Color32,
}

/// A label of the plots, laid out and placed, waiting to be painted after the
/// curves so none of them runs over it.
struct PlotLabel {
    /// Its top-left corner.
    at: Pos2,
    galley: std::sync::Arc<egui::Galley>,
    color: Color32,
}

/// Paint the plots' labels over everything else in them, each on a plate of
/// the panel's own colour, so a curve that passes under a label does not cut
/// through its glyphs.
fn paint_labels(painter: &egui::Painter, labels: Vec<PlotLabel>, plate: Color32) {
    for label in labels {
        let rect = Rect::from_min_size(label.at, label.galley.size()).expand(1.5);
        painter.rect_filled(rect, 2.0, plate);
        painter.galley(label.at, label.galley, label.color);
    }
}

/// An extrapolated region: where it starts and how it is coloured and named.
struct Wash {
    limit_deg: f64,
    fill: Color32,
    rule: Color32,
    label: String,
}

/// How many leading samples a model still vouches for: those at or inside
/// `limit`, or all of them with no limit.
fn trusted_count(samples: &[ProfileSample], limit: Option<f64>) -> usize {
    match limit {
        Some(limit) => samples.iter().take_while(|s| s.theta_deg <= limit).count(),
        None => samples.len(),
    }
}

/// The projection plot, or a one-line statement of why there is none.
///
/// `compare` is the open proposal's camera, drawn against this one; `rug` is
/// the incidence angles of the camera's observations under this camera,
/// ascending, drawn while a proposal is open whether or not its fit
/// succeeded.
pub(super) fn show_projection_plot(
    ui: &mut egui::Ui,
    camera: &CameraIntrinsics,
    derived: &Derived,
    compare: Option<&Comparison<'_>>,
    rug: Option<&[f64]>,
) {
    ui.label(egui::RichText::new("Projection").strong());

    let samples = &derived.profile.samples;
    let mut x_max = samples
        .last()
        .map(|s| s.theta_deg)
        .into_iter()
        .chain(derived.fov.map(|fov| fov.max_off_axis))
        .fold(0.0_f64, f64::max);
    if let Some(c) = compare {
        x_max = c
            .derived
            .profile
            .samples
            .last()
            .map(|s| s.theta_deg)
            .into_iter()
            .chain(c.derived.fov.map(|fov| fov.max_off_axis))
            .fold(x_max, f64::max);
    }
    let x_max = x_max * X_MARGIN;
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
    let trusted = trusted_count(samples, limit);
    let proposed_limit = compare.and_then(|c| c.derived.trustworthy_max_theta);

    let width = ui.available_width().max(LEFT_GUTTER + RIGHT_PAD + 80.0);
    let change_height = if compare.is_some() {
        PLOT_GAP + TITLE_HEIGHT + CHANGE_HEIGHT
    } else {
        0.0
    };
    let rug_height = if rug.is_some() { RUG_HEIGHT } else { 0.0 };
    let height = TITLE_HEIGHT
        + RADIAL_HEIGHT
        + PLOT_GAP
        + TITLE_HEIGHT
        + RESIDUAL_HEIGHT
        + change_height
        + AXIS_HEIGHT
        + rug_height;
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
    let change_rect = compare.map(|_| {
        let top = lower.bottom() + PLOT_GAP + TITLE_HEIGHT;
        Rect::from_min_max(Pos2::new(left, top), Pos2::new(right, top + CHANGE_HEIGHT))
    });
    let bottom = change_rect.unwrap_or(lower);
    let rug_rect = rug.map(|_| {
        let top = bottom.bottom() + AXIS_HEIGHT;
        Rect::from_min_max(
            Pos2::new(left, top),
            Pos2::new(right, top + RUG_HEIGHT - 2.0),
        )
    });

    let palette = Palette::resolve(ui.visuals());
    let x = Scale {
        lo: 0.0,
        hi: x_max,
        screen_lo: left,
        screen_hi: right,
    };
    let trusted_samples = &samples[..trusted.max(1).min(samples.len())];
    let proposed_curve = compare.map(|c| {
        let samples = &c.derived.profile.samples;
        Curve {
            samples,
            trusted: trusted_count(samples, proposed_limit),
            band_visible: c.derived.profile.band_visible,
            stroke: palette.model,
            width: 1.8,
            band: palette.band,
        }
    });
    let proposed_trusted: &[ProfileSample] = proposed_curve
        .as_ref()
        .map_or(&[], |c| &c.samples[..c.trusted.min(c.samples.len())]);
    let radial_y = radial_scale(trusted_samples.iter().chain(proposed_trusted), &upper);
    let residual_y = residual_scale(trusted_samples.iter().chain(proposed_trusted), &lower);

    let painter = ui.painter().clone();
    let mut markers = markers(camera, derived, x_max);
    if let Some(c) = compare {
        markers.extend(comparison_markers(c, x_max));
        markers.sort_by(|a, b| a.theta_deg.total_cmp(&b.theta_deg));
    }

    let mut plots = vec![upper, lower];
    let mut titles = vec!["r (px)", "Δr (px)"];
    let mut scales = vec![radial_y, residual_y];
    let change_bound = [limit, proposed_limit]
        .into_iter()
        .flatten()
        .reduce(f64::min);
    if let (Some(c), Some(change_rect)) = (compare, change_rect) {
        plots.push(change_rect);
        titles.push("Change (px)");
        scales.push(change_scale(c.change, change_bound, &change_rect));
    }
    for ((plot, y), title) in plots.iter().zip(&scales).zip(&titles) {
        draw_frame(&painter, plot, y, &palette, title);
    }
    draw_x_axis(&painter, &plots, &bottom, &x, &palette);

    let mut washes = Vec::new();
    match (compare.is_some(), limit) {
        (false, Some(limit)) => washes.push(Wash {
            limit_deg: limit,
            fill: palette.extrapolated_wash,
            rule: palette.extrapolated_rule,
            label: format!("extrapolated past {limit:.1}°"),
        }),
        (true, Some(limit)) => washes.push(Wash {
            limit_deg: limit,
            fill: palette.current_wash,
            rule: palette.current,
            label: format!("current extrapolated past {limit:.1}°"),
        }),
        (_, None) => {}
    }
    if let Some(limit) = proposed_limit {
        washes.push(Wash {
            limit_deg: limit,
            fill: palette.proposed_wash,
            rule: palette.model,
            label: format!("proposed extrapolated past {limit:.1}°"),
        });
    }
    // Every label of the plots, painted last so no curve runs over one.
    let mut labels: Vec<PlotLabel> = Vec::new();
    let mut label_top = upper.top() + 3.0;
    for wash in &washes {
        for plot in &plots {
            let start = x.at(wash.limit_deg).max(plot.left());
            if start < plot.right() {
                painter.rect_filled(
                    Rect::from_min_max(
                        Pos2::new(start, plot.top()),
                        Pos2::new(plot.right(), plot.bottom()),
                    ),
                    0.0,
                    wash.fill,
                );
                painter.line_segment(
                    [
                        Pos2::new(start, plot.top()),
                        Pos2::new(start, plot.bottom()),
                    ],
                    Stroke::new(1.5, wash.rule),
                );
            }
        }
        // Right-aligned at the plot's edge rather than beside the rule: the
        // rule can land anywhere, and beside it the label collides with
        // whichever marker the frame's own geometry happens to put nearby.
        let galley = painter.layout_no_wrap(
            wash.label.clone(),
            FontId::proportional(LABEL_FONT),
            wash.rule,
        );
        labels.push(PlotLabel {
            at: Pos2::new(upper.right() - 4.0 - galley.size().x, label_top),
            galley,
            color: wash.rule,
        });
        label_top += LABEL_ROW;
    }

    labels.extend(draw_markers(
        &painter,
        &markers,
        &plots,
        &upper,
        &x,
        &palette,
        label_top + 3.0,
    ));
    let current_curve = Curve {
        samples,
        trusted,
        band_visible: derived.profile.band_visible,
        stroke: if compare.is_some() {
            palette.current
        } else {
            palette.model
        },
        width: if compare.is_some() { 4.0 } else { 1.8 },
        band: if compare.is_some() {
            palette.current_band
        } else {
            palette.band
        },
    };
    draw_ideal(&painter, samples, &upper, &x, &radial_y, &palette);
    draw_zero_line(&painter, &lower, &residual_y, &palette);
    for curve in std::iter::once(&current_curve).chain(proposed_curve.as_ref()) {
        draw_curve(&painter, curve, &upper, &lower, &x, &radial_y, &residual_y);
    }
    if let (Some(c), Some(change_rect)) = (compare, change_rect) {
        labels.extend(draw_change(
            &painter,
            c,
            change_bound,
            &change_rect,
            &x,
            &scales[2],
            &palette,
        ));
    }

    let undistorted =
        !camera.has_distortion() && compare.is_none_or(|c| !c.proposed.has_distortion());
    if undistorted {
        draw_no_distortion_banner(&painter, camera, &lower, &palette);
    }
    if let (Some(rug), Some(rug_rect)) = (rug, rug_rect) {
        draw_rug(&painter, rug, limit, &rug_rect, &x, &palette);
    }
    paint_labels(&painter, labels, palette.plate);

    legend(ui, camera, derived, compare, rug);
}

/// The upper plot's y range: zero to the largest radius any of `samples`
/// reaches, on either curve or at either edge of the band.
fn radial_scale<'a>(samples: impl IntoIterator<Item = &'a ProfileSample>, plot: &Rect) -> Scale {
    let hi = samples
        .into_iter()
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

/// The lower plot's y range: the residuals of `samples`, always including zero
/// so the zero line is on the plot even for a lens that only ever magnifies.
fn residual_scale<'a>(samples: impl IntoIterator<Item = &'a ProfileSample>, plot: &Rect) -> Scale {
    let values = samples.into_iter().flat_map(|s| {
        [
            s.radius_px - s.reference_px,
            s.band_px.0 - s.reference_px,
            s.band_px.1 - s.reference_px,
        ]
    });
    zero_including_scale(values, plot)
}

/// The change plot's y range: the change and its band at the angles inside
/// `bound`, always including zero, which is the line a fit is read against.
fn change_scale(change: &ChangeProfile, bound: Option<f64>, plot: &Rect) -> Scale {
    let values = change
        .samples
        .iter()
        .filter(|s| bound.is_none_or(|bound| s.theta_deg <= bound))
        .flat_map(|s| [s.centre_px, s.band_px.0, s.band_px.1]);
    zero_including_scale(values, plot)
}

/// A y range over `values` that always includes zero, padded by the margin.
fn zero_including_scale(values: impl IntoIterator<Item = f64>, plot: &Rect) -> Scale {
    let mut lo = 0.0_f64;
    let mut hi = 0.0_f64;
    for value in values {
        if value.is_finite() {
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
        FontId::proportional(LABEL_FONT),
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
            FontId::proportional(SMALL_FONT),
            palette.text,
        );
    }
}

/// The shared x axis: one ladder of degree ticks, gridded on every plot and
/// labelled once, under the lowest one.
///
/// A tick label that would run into the axis's own title, at the right end,
/// is left off; its gridline is still drawn.
fn draw_x_axis(
    painter: &egui::Painter,
    plots: &[Rect],
    bottom: &Rect,
    x: &Scale,
    palette: &Palette,
) {
    let font = FontId::proportional(SMALL_FONT);
    let title = painter.layout_no_wrap("θ off-axis".to_string(), font.clone(), palette.text);
    let title_rect = Rect::from_min_size(
        Pos2::new(bottom.right() - title.size().x, bottom.bottom() + 2.0),
        title.size(),
    );
    for value in ticks(x.lo, x.hi, X_TICKS) {
        let px = x.at(value);
        if px < bottom.left() || px > bottom.right() {
            continue;
        }
        for plot in plots {
            painter.line_segment(
                [Pos2::new(px, plot.top()), Pos2::new(px, plot.bottom())],
                Stroke::new(1.0, palette.grid),
            );
        }
        let label = painter.layout_no_wrap(format!("{value:.0}°"), font.clone(), palette.text);
        let at = Pos2::new(px - label.size().x / 2.0, bottom.bottom() + 2.0);
        if Rect::from_min_size(at, label.size()).intersects(title_rect.expand(3.0)) {
            continue;
        }
        painter.galley(at, label, palette.text);
    }
    painter.galley(title_rect.min, title, palette.text);
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

    // Past the spline's domain end `δ` continues along its end tangent, so the
    // radial map continues linearly rather than following the fitted curve — a
    // lens whose frame reaches past this rule is being extrapolated, benignly
    // but visibly.
    if let Some(theta_deg) = spline_domain_deg(camera) {
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

/// The rules a comparison adds: the fit's largest angle, and the proposed
/// camera's spline domain end where it has one.
fn comparison_markers(compare: &Comparison<'_>, x_max: f64) -> Vec<Marker> {
    let mut markers = vec![Marker {
        theta_deg: compare.theta_fit_deg,
        label: "fit".to_string(),
    }];
    if let Some(theta_deg) = spline_domain_deg(compare.proposed) {
        markers.push(Marker {
            theta_deg,
            label: "proposed spline domain".to_string(),
        });
    }
    markers.retain(|m| m.theta_deg > 0.0 && m.theta_deg <= x_max);
    markers
}

/// Where a label of `size` goes beside the rule at `px`: on the first of
/// [`LABEL_ROWS`] rows from `top` where it overlaps no label placed before it,
/// right of the rule unless that runs past `bounds`, then left of it. `None`
/// when no row has room.
fn place_label(
    placed: &mut Vec<Rect>,
    size: Vec2,
    px: f32,
    top: f32,
    bounds: (f32, f32),
) -> Option<Pos2> {
    let mut x0 = px + 3.0;
    if x0 + size.x > bounds.1 - 2.0 {
        x0 = px - 3.0 - size.x;
    }
    let x0 = x0.max(bounds.0 + 2.0);
    for row in 0..LABEL_ROWS {
        let at = Pos2::new(x0, top + row as f32 * LABEL_ROW);
        let rect = Rect::from_min_size(at, size);
        if placed
            .iter()
            .all(|other| !other.expand(2.0).intersects(rect))
        {
            placed.push(rect);
            return Some(at);
        }
    }
    None
}

/// Draw the rules across every plot, and return their labels, placed in the
/// upper plot from `label_top` down, for [`paint_labels`] to paint last.
fn draw_markers(
    painter: &egui::Painter,
    markers: &[Marker],
    plots: &[Rect],
    upper: &Rect,
    x: &Scale,
    palette: &Palette,
    label_top: f32,
) -> Vec<PlotLabel> {
    let mut placed = Vec::new();
    let mut labels = Vec::new();
    for marker in markers {
        let px = x.at(marker.theta_deg);
        if px < upper.left() || px > upper.right() {
            continue;
        }
        for plot in plots {
            painter.extend(Shape::dashed_line(
                &[Pos2::new(px, plot.top()), Pos2::new(px, plot.bottom())],
                Stroke::new(1.0, palette.axis),
                3.0,
                3.0,
            ));
        }
        let galley = painter.layout_no_wrap(
            format!("{} {:.1}°", marker.label, marker.theta_deg),
            FontId::proportional(SMALL_FONT),
            palette.text,
        );
        if let Some(at) = place_label(
            &mut placed,
            galley.size(),
            px,
            label_top,
            (upper.left(), upper.right()),
        ) {
            labels.push(PlotLabel {
                at,
                galley,
                color: palette.text,
            });
        }
    }
    labels
}

/// The family's ideal map, dashed, on the upper plot.
fn draw_ideal(
    painter: &egui::Painter,
    samples: &[ProfileSample],
    upper: &Rect,
    x: &Scale,
    radial_y: &Scale,
    palette: &Palette,
) {
    let ideal: Vec<Pos2> = samples
        .iter()
        .map(|s| Pos2::new(x.at(s.theta_deg), radial_y.at(s.reference_px)))
        .collect();
    painter.with_clip_rect(*upper).extend(Shape::dashed_line(
        &ideal,
        Stroke::new(1.5, palette.ideal),
        DASH.0,
        DASH.1,
    ));
}

/// A plot's zero line: the thing it is read against.
fn draw_zero_line(painter: &egui::Painter, plot: &Rect, y: &Scale, palette: &Palette) {
    painter.with_clip_rect(*plot).line_segment(
        [
            Pos2::new(plot.left(), y.at(0.0)),
            Pos2::new(plot.right(), y.at(0.0)),
        ],
        Stroke::new(1.0, palette.axis),
    );
}

/// One camera's band and curve, on the radial plot and the residual.
fn draw_curve(
    painter: &egui::Painter,
    curve: &Curve<'_>,
    upper: &Rect,
    lower: &Rect,
    x: &Scale,
    radial_y: &Scale,
    residual_y: &Scale,
) {
    // Clipped, because past the trustworthy bound the dotted curve dives out
    // of a frame scaled to the part that means something — which is the
    // picture, not a defect to design around.
    let upper_painter = painter.with_clip_rect(*upper);
    let lower_painter = painter.with_clip_rect(*lower);
    let samples = curve.samples;

    if curve.band_visible {
        for (plot, y, offset) in [
            (&upper_painter, radial_y, false),
            (&lower_painter, residual_y, true),
        ] {
            let base = |s: &ProfileSample, value: f64| {
                if offset {
                    value - s.reference_px
                } else {
                    value
                }
            };
            draw_band(
                plot,
                samples
                    .iter()
                    .map(|s| (s.theta_deg, base(s, s.band_px.0), base(s, s.band_px.1))),
                x,
                y,
                curve.band,
            );
        }
    }

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
        solid_then_dotted(plot, points, curve.trusted, curve.stroke, curve.width);
    }
}

/// A band between two edges, one quad per column rather than a single closed
/// path: each quad is convex, which `convex_polygon` needs and a min/max
/// envelope in general is not.
fn draw_band(
    painter: &egui::Painter,
    columns: impl Iterator<Item = (f64, f64, f64)>,
    x: &Scale,
    y: &Scale,
    fill: Color32,
) {
    let columns: Vec<(f64, f64, f64)> = columns.collect();
    for pair in columns.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let (ax, bx) = (x.at(a.0), x.at(b.0));
        painter.add(Shape::convex_polygon(
            vec![
                Pos2::new(ax, y.at(a.1)),
                Pos2::new(ax, y.at(a.2)),
                Pos2::new(bx, y.at(b.2)),
                Pos2::new(bx, y.at(b.1)),
            ],
            fill,
            Stroke::NONE,
        ));
    }
}

/// A polyline solid over its first `trusted` points and dotted after them,
/// the dotted half restarting on the last solid point so the two join.
fn solid_then_dotted(
    painter: &egui::Painter,
    points: &[Pos2],
    trusted: usize,
    color: Color32,
    width: f32,
) {
    let split = trusted.min(points.len());
    if split >= 2 {
        painter.add(Shape::line(
            points[..split].to_vec(),
            Stroke::new(width, color),
        ));
    }
    let tail = &points[split.saturating_sub(1)..];
    if tail.len() >= 2 {
        painter.extend(Shape::dotted_line(tail, color, 3.0, 1.0));
    }
}

/// The change plot's band, its centre line, and the mark on its θ axis where
/// the monotonicity constraint bound, returning that mark's label for
/// [`paint_labels`] to paint last.
///
/// The centre line is solid up to `bound`, the smaller of the two cameras'
/// trusted bounds, and dotted past it: there the current camera is
/// extrapolating, and the line shows how far the proposed one departs from
/// where the current one was going rather than a fitting error.
fn draw_change(
    painter: &egui::Painter,
    compare: &Comparison<'_>,
    bound: Option<f64>,
    plot: &Rect,
    x: &Scale,
    y: &Scale,
    palette: &Palette,
) -> Option<PlotLabel> {
    let clipped = painter.with_clip_rect(*plot);
    let samples = &compare.change.samples;
    if compare.change.band_visible(bound) {
        draw_band(
            &clipped,
            samples
                .iter()
                .map(|s| (s.theta_deg, s.band_px.0, s.band_px.1)),
            x,
            y,
            palette.band,
        );
    }
    draw_zero_line(painter, plot, y, palette);
    let points: Vec<Pos2> = samples
        .iter()
        .map(|s| Pos2::new(x.at(s.theta_deg), y.at(s.centre_px)))
        .collect();
    let trusted = samples
        .iter()
        .take_while(|s| bound.is_none_or(|bound| s.theta_deg <= bound))
        .count();
    solid_then_dotted(&clipped, &points, trusted, palette.model, 1.8);

    if let Some([from, to]) = compare.monotone_deg {
        let (a, b) = (x.at(from), x.at(to));
        // At least a few pixels wide, so a constraint that bound at one angle
        // is still a mark rather than a hairline under the frame.
        let centre = 0.5 * (a + b);
        let half = (0.5 * (b - a)).max(2.0);
        let (a, b) = (
            (centre - half).max(plot.left()),
            (centre + half).min(plot.right()),
        );
        if a < b {
            clipped.rect_filled(
                Rect::from_min_max(
                    Pos2::new(a, plot.bottom() - 5.0),
                    Pos2::new(b, plot.bottom()),
                ),
                0.0,
                palette.monotone,
            );
            let galley = painter.layout_no_wrap(
                "monotone bound".to_string(),
                FontId::proportional(SMALL_FONT),
                palette.monotone,
            );
            let size = galley.size();
            let mut x0 = b + 3.0;
            if x0 + size.x > plot.right() - 2.0 {
                x0 = a - 3.0 - size.x;
            }
            let x0 = x0.max(plot.left() + 2.0);
            return Some(PlotLabel {
                at: Pos2::new(x0, plot.bottom() - 4.0 - size.y),
                galley,
                color: palette.monotone,
            });
        }
    }
    None
}

/// The observation rug: one tick per observation at its incidence angle, a
/// screen column at a time so a dense camera is not ten thousand strokes. A
/// column's opacity grows with its count, and a column past the current
/// camera's trusted bound is drawn in that bound's colour.
fn draw_rug(
    painter: &egui::Painter,
    angles_deg: &[f64],
    limit: Option<f64>,
    rect: &Rect,
    x: &Scale,
    palette: &Palette,
) {
    for (column, count, past) in rug_columns(angles_deg, limit, x) {
        let color = if past {
            palette.extrapolated_rule
        } else {
            palette.text
        };
        let alpha = (90.0 + 50.0 * (count as f32).ln_1p()).min(255.0) as u8;
        let px = column as f32 + 0.5;
        if px < rect.left() || px > rect.right() {
            continue;
        }
        painter.line_segment(
            [Pos2::new(px, rect.top()), Pos2::new(px, rect.bottom())],
            Stroke::new(1.0, color.gamma_multiply(f32::from(alpha) / 255.0)),
        );
    }
}

/// The rug's screen columns: each column's x, how many observations fall in
/// it, and whether they are past `limit`. In ascending x.
fn rug_columns(angles_deg: &[f64], limit: Option<f64>, x: &Scale) -> Vec<(i32, usize, bool)> {
    let mut columns: Vec<(i32, usize, bool)> = Vec::new();
    for &theta in angles_deg {
        if !theta.is_finite() {
            continue;
        }
        let column = x.at(theta).floor() as i32;
        let past = limit.is_some_and(|limit| theta > limit);
        match columns.last_mut() {
            Some((c, n, p)) if *c == column && *p == past => *n += 1,
            _ => columns.push((column, 1, past)),
        }
    }
    columns
}

/// How many of `angles_deg` lie past `limit`.
pub(super) fn past_bound(angles_deg: &[f64], limit: Option<f64>) -> usize {
    limit.map_or(0, |limit| angles_deg.iter().filter(|&&t| t > limit).count())
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

/// The key, and the sentences that say what the shaded regions and the rug
/// are.
fn legend(
    ui: &mut egui::Ui,
    camera: &CameraIntrinsics,
    derived: &Derived,
    compare: Option<&Comparison<'_>>,
    rug: Option<&[f64]>,
) {
    let ideal = if camera.model.is_equirectangular() {
        "ideal (itself)"
    } else if camera.model.is_fisheye() {
        "ideal r = f·θ"
    } else {
        "ideal r = f·tan θ"
    };
    // Words rather than sample glyphs: egui's default font has no box-drawing
    // or geometric-shape coverage, so a `──`/`▨` key renders as tofu.
    // One label rather than a row of them, so a narrow panel wraps it as one
    // sentence rather than running a wrapped part into the one before it.
    let mut key = match compare {
        Some(_) => format!("grey: current · blue: proposed · dashed: {ideal} (current)"),
        None => format!("solid: model · dashed: {ideal}"),
    };
    let band =
        derived.profile.band_visible || compare.is_some_and(|c| c.derived.profile.band_visible);
    if band {
        key.push_str(" · band: spread over 32 azimuths");
    }
    let key = ui.label(egui::RichText::new(key).small());
    if band {
        key.on_hover_text(
            "The model's radius is not the same at every azimuth — `fx ≠ fy`, or live \
             tangential or thin-prism terms. A single-azimuth curve hides decentring \
             distortion completely, which is what the band is for.",
        );
    }
    let corner = derived.fov.map(|fov| fov.max_off_axis);
    match (compare, derived.trustworthy_max_theta) {
        (None, Some(limit)) => {
            ui.label(
                egui::RichText::new(format!(
                    "Shaded past {limit:.1}°: this model's own inverse stops inverting its \
                     distortion polynomial there, so neither curve describes the lens beyond it. \
                     The frame reaches {:.1}°.",
                    corner.unwrap_or(limit)
                ))
                .small()
                .weak(),
            );
        }
        (Some(_), Some(limit)) => {
            ui.label(
                egui::RichText::new(format!(
                    "Grey shading past {limit:.1}°: the current model's own inverse stops \
                     inverting its distortion polynomial there. The frame reaches {:.1}°.",
                    corner.unwrap_or(limit)
                ))
                .small()
                .weak(),
            );
        }
        _ => {}
    }
    if let Some(c) = compare {
        let proposed = match c.derived.trustworthy_max_theta {
            Some(limit) => format!("Blue shading past {limit:.1}°: the proposed model's bound."),
            None => "The proposed model is trusted at every angle the frame reaches.".to_string(),
        };
        ui.label(egui::RichText::new(proposed).small().weak());
        ui.label(
            egui::RichText::new(format!(
                "Change: the proposed radius minus the current one, averaged over 32 azimuths, \
                 with their spread as the band. The fit sampled angles up to {:.1}°.",
                c.theta_fit_deg
            ))
            .small()
            .weak(),
        );
    }
    if let Some(rug) = rug {
        let sentence = match derived.trustworthy_max_theta {
            Some(limit) => format!(
                "Rug: {} observations of this camera's images, at their angle under the current \
                 model; {} past {limit:.1}°.",
                rug.len(),
                past_bound(rug, Some(limit))
            ),
            None => format!(
                "Rug: {} observations of this camera's images, at their angle under the current \
                 model.",
                rug.len()
            ),
        };
        ui.label(egui::RichText::new(sentence).small().weak());
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
