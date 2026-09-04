// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The angular reference grid: two axes through the principal point, their tick
//! ladder, the optional iso-angle rings, and the dashed contours where the
//! model stops describing a lens.
//!
//! # The axes are sampled, not drawn straight
//!
//! Each axis is a polyline through densely sampled `ray_to_pixel` results —
//! `ray(α) = (sin α, 0, −cos α)` for the horizontal one, `(0, sin ε, −cos ε)`
//! for the vertical, in the canonical camera frame (−Z forward, +Y up, +X
//! right), so a positive angle is to the right and up with no negation
//! anywhere. Sampling stops at the first angle the model refuses, or the first
//! that leaves the frame by more than [`OUT_OF_FRAME_FRACTION`] of its
//! diagonal.
//!
//! Whether an axis bends is a property of the model, not of how violent its
//! distortion is. A purely **radial** model — which is most of the registry,
//! and both checked-in fixtures — moves every point along its own radius from
//! the principal point, so a line *through* the principal point stays exactly
//! straight however violent its distortion is: on `kerry_park`'s
//! `OPENCV_FISHEYE` the sampled vertical axis is straight to under a hundredth
//! of a pixel. What its distortion does to these axes is bunch the **ticks**
//! along them — evenly spaced angles landing at unevenly spaced pixels — and
//! that is the reading the ticks are for. Sampling rather than drawing two
//! straight lines is still the right implementation, because it is what makes
//! the tangential and thin-prism models' real curvature show up without a
//! special case; it just is not what a `SIMPLE_RADIAL` shows you.
//!
//! # The ladder is a function of the zoom
//!
//! The panel zooms to 32×, so a tick ladder fixed at load time would be
//! unreadable at one end and empty at the other. [`tick_ladder`] picks the
//! finest step that still keeps adjacent ticks [`MIN_TICK_SPACING_PX`] apart at
//! the current scale. The whole geometry is rebuilt when the scale crosses a
//! half-octave bucket rather than every frame, so a pinch gesture does not
//! resample two polylines sixty times a second.

use egui::{Align2, Color32, Pos2, Rect, Shape, Stroke};
use sfmtool_core::camera::intrinsics::SplineRadial;
use sfmtool_core::camera::CameraIntrinsics;

use super::{haloed_line, haloed_text, ink, View, DEGREE, MINUS, REFERENCE_ALPHA};

/// Angular ladder the ticks and rings are drawn on, coarsest last.
const LADDER_DEG: [f64; 7] = [1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 45.0];

/// Minimum spacing between adjacent ticks, in panel pixels. Below this the
/// labels collide and the ladder is telling the reader nothing they can use.
const MIN_TICK_SPACING_PX: f32 = 48.0;

/// How far past the frame's own boundary an axis or ring keeps being sampled,
/// as a fraction of the image diagonal. A little overshoot shows that the axis
/// runs out of *frame* rather than out of *model*; more than that is drawing
/// off-image geometry nobody asked for.
const OUT_OF_FRAME_FRACTION: f64 = 0.05;

/// Where the sweep gives up, in degrees off-axis. Past 180° there is no ray
/// left to look at: an equirectangular panorama's horizontal axis has come all
/// the way round, and no other model reaches this far.
const MAX_SWEEP_DEG: f64 = 180.0;

/// Target spacing between polyline samples, in panel pixels.
const SAMPLE_SPACING_PX: f32 = 4.0;

/// Bounds on the angular sampling step, in degrees, so neither a very long lens
/// at 32× nor a very short one fitted to the panel produces an unreasonable
/// number of samples.
const STEP_BOUNDS_DEG: (f64, f64) = (0.05, 5.0);

/// Dash and gap of the domain contours, in panel pixels.
const DASH: (f32, f32) = (5.0, 4.0);

/// Which of the two axes a sample belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Axis {
    /// Swept in azimuth: `+` is to the right of the principal point.
    Horizontal,
    /// Swept in elevation: `+` is above it.
    Vertical,
}

impl Axis {
    /// The canonical-frame ray this axis looks along at `angle_deg`.
    ///
    /// `−Z` forward, `+Y` up, `+X` right, so `(sin α, 0, −cos α)` with `α > 0`
    /// projects to the right of the principal point and `(0, sin ε, −cos ε)`
    /// with `ε > 0` projects above it — the reason the labels can be signed and
    /// read correctly with no negation applied anywhere.
    fn ray(self, angle_deg: f64) -> [f64; 3] {
        let (sin, cos) = angle_deg.to_radians().sin_cos();
        match self {
            Axis::Horizontal => [sin, 0.0, -cos],
            Axis::Vertical => [0.0, sin, -cos],
        }
    }
}

/// One tick on an axis: where it lands and what it says.
pub(super) struct Tick {
    pub at: [f64; 2],
    pub axis: Axis,
    pub label: String,
}

/// A closed iso-angle contour, broken into the runs that stayed on the frame.
pub(super) struct Contour {
    /// Runs of consecutive in-frame samples. A circular fisheye's wide rings
    /// leave the frame at the corners and come back, so one contour is several
    /// polylines rather than one.
    pub runs: Vec<Vec<[f64; 2]>>,
    pub label: String,
    /// Where the one label goes, in image pixels.
    ///
    /// On the up-and-left diagonal rather than at the top, which is where the
    /// spec puts it: the top of a `30°` ring is exactly where the vertical
    /// axis's own `+30°` tick lands, and the two labels come out on top of each
    /// other saying the same thing twice.
    pub label_at: [f64; 2],
}

/// The whole reference grid for one camera at one zoom bucket.
pub(super) struct AxisGeometry {
    /// The half-octave scale bucket this was sampled at, and whether the rings
    /// were built — together, the cache key.
    pub bucket: i32,
    pub has_rings: bool,
    /// The two axes, each sample flagged with whether the model still describes
    /// a lens at that angle — solid where it does, dashed where it does not.
    pub horizontal: Vec<([f64; 2], bool)>,
    pub vertical: Vec<([f64; 2], bool)>,
    pub ticks: Vec<Tick>,
    pub rings: Vec<Contour>,
    /// The contour at [`super::CameraLayer::limit_deg`], past which the model
    /// is extrapolating rather than describing a lens.
    pub trustworthy_edge: Option<Contour>,
    /// The contour at a spline model's domain end, past which its correction is
    /// held constant and the radial map continues linearly.
    pub spline_domain: Option<Contour>,
}

/// The scale bucket a view falls in: half-octaves, so a pinch gesture crosses a
/// handful of buckets over the panel's whole 1× – 32× range rather than
/// rebuilding the geometry on every frame of it.
pub(super) fn scale_bucket(scale: f32) -> i32 {
    if scale > 0.0 {
        (scale.log2() * 2.0).round() as i32
    } else {
        0
    }
}

impl AxisGeometry {
    /// Sample the grid for `camera` at `scale` panel pixels per image pixel.
    pub(super) fn compute(
        camera: &CameraIntrinsics,
        scale: f32,
        has_rings: bool,
        limit_deg: Option<f64>,
    ) -> Self {
        let step = sample_step_deg(camera, scale);
        let horizontal = axis_polyline(camera, Axis::Horizontal, step, limit_deg);
        let vertical = axis_polyline(camera, Axis::Vertical, step, limit_deg);

        // Ticks and rings are *numbers* — "this pixel is 60° off axis" — so
        // they stop where the model stops being able to make that claim. The
        // axes themselves carry on dashed, and the dashed contour at the bound
        // says where the change happened; what does not happen is a confident
        // `−120°` printed on a fold.
        let reach = clamp_reach(axis_reaches(camera, scale), limit_deg);
        let ladder_deg = tick_ladder(camera, scale, reach);
        let mut ticks = Vec::new();
        for (axis, reach) in [(Axis::Horizontal, reach.0), (Axis::Vertical, reach.1)] {
            for (angle_deg, at) in tick_positions(camera, axis, ladder_deg, reach) {
                // Zero is labelled once, on the horizontal axis: both axes cross
                // there, and it is also where the principal-point reticle sits.
                if angle_deg == 0.0 && axis == Axis::Vertical {
                    continue;
                }
                ticks.push(Tick {
                    at,
                    axis,
                    label: signed_angle(angle_deg),
                });
            }
        }

        let ring_max = reach.0 .1.max(reach.1 .1);
        let rings = if has_rings {
            let mut rings = Vec::new();
            let mut angle = ladder_deg;
            while angle <= ring_max {
                rings.extend(contour(camera, angle, scale, format!("{angle:.0}{DEGREE}")));
                angle += ladder_deg;
            }
            rings
        } else {
            Vec::new()
        };

        Self {
            bucket: scale_bucket(scale),
            has_rings,
            horizontal,
            vertical,
            ticks,
            rings,
            trustworthy_edge: limit_deg.and_then(|limit| {
                contour(
                    camera,
                    limit,
                    scale,
                    format!("extrapolated past {limit:.1}{DEGREE}"),
                )
            }),
            spline_domain: spline_domain_deg(camera).and_then(|angle| {
                contour(
                    camera,
                    angle,
                    scale,
                    format!("spline domain {angle:.1}{DEGREE}"),
                )
            }),
        }
    }

    /// Paint the grid, beneath the features.
    pub(super) fn draw(&self, painter: &egui::Painter, view: &View, panel: Rect, axes: bool) {
        let reference = ink(REFERENCE_ALPHA);
        // The two domain contours are drawn whether or not the axes are, since
        // they qualify the arrow field as much as the grid.
        for contour in [&self.trustworthy_edge, &self.spline_domain]
            .into_iter()
            .flatten()
        {
            draw_dashed_contour(painter, contour, view, panel, reference);
        }
        if !axes {
            return;
        }
        for polyline in [&self.horizontal, &self.vertical] {
            for (run, trusted) in flagged_runs(polyline) {
                let points = panel_points(&run, view);
                if trusted {
                    haloed_line(painter, points, 1.0, reference);
                } else {
                    dashed(painter, &points, reference);
                }
            }
        }
        for ring in &self.rings {
            draw_contour(painter, ring, view, panel, reference);
        }
        for tick in &self.ticks {
            draw_tick(painter, tick, view, panel, reference);
        }
    }
}

/// Map an image-pixel polyline into panel coordinates.
fn panel_points(points: &[[f64; 2]], view: &View) -> Vec<Pos2> {
    points.iter().map(|p| view.at(*p)).collect()
}

/// Split a flagged polyline into maximal runs sharing a flag, each run
/// overlapping its neighbour by one point so the solid and dashed halves meet
/// rather than leaving a gap at the boundary.
///
/// At most three runs by construction: extrapolated tail, trusted middle,
/// extrapolated tail.
fn flagged_runs(points: &[([f64; 2], bool)]) -> Vec<(Vec<[f64; 2]>, bool)> {
    let mut runs: Vec<(Vec<[f64; 2]>, bool)> = Vec::new();
    let mut start = 0;
    let mut cut = |from: usize, to: usize| {
        runs.push((
            points[from..to].iter().map(|(p, _)| *p).collect(),
            points[from].1,
        ));
    };
    for i in 1..points.len() {
        if points[i].1 != points[start].1 {
            cut(start, i + 1);
            start = i;
        }
    }
    if start < points.len() {
        cut(start, points.len());
    }
    runs
}

/// Paint a dashed polyline in near-white over a dashed dark halo.
fn dashed(painter: &egui::Painter, points: &[Pos2], color: Color32) {
    if points.len() < 2 {
        return;
    }
    painter.extend(Shape::dashed_line(
        points,
        Stroke::new(1.0 + super::HALO_WIDTH, super::HALO),
        DASH.0,
        DASH.1,
    ));
    painter.extend(Shape::dashed_line(
        points,
        Stroke::new(1.0, color),
        DASH.0,
        DASH.1,
    ));
}

/// Clamp an axis reach to the angles the model can still be held to.
fn clamp_reach(
    reach: ((f64, f64), (f64, f64)),
    limit_deg: Option<f64>,
) -> ((f64, f64), (f64, f64)) {
    let Some(limit) = limit_deg else {
        return reach;
    };
    let clamp = |(lo, hi): (f64, f64)| (lo.max(-limit), hi.min(limit));
    (clamp(reach.0), clamp(reach.1))
}

/// The angular step that puts consecutive samples about
/// [`SAMPLE_SPACING_PX`] apart, from the camera's own pixels per radian.
fn sample_step_deg(camera: &CameraIntrinsics, scale: f32) -> f64 {
    let (fx, fy) = camera.focal_lengths();
    let focal = fx.max(fy).max(1.0);
    let step = (f64::from(SAMPLE_SPACING_PX) / (focal * f64::from(scale.max(1e-3)))).to_degrees();
    step.clamp(STEP_BOUNDS_DEG.0, STEP_BOUNDS_DEG.1)
}

/// How far past the frame a sample may stray before the sweep gives up.
fn out_of_frame(camera: &CameraIntrinsics, p: (f64, f64)) -> bool {
    let (w, h) = (f64::from(camera.width), f64::from(camera.height));
    let margin = OUT_OF_FRAME_FRACTION * w.hypot(h);
    p.0 < -margin || p.1 < -margin || p.0 > w + margin || p.1 > h + margin
}

/// How far each axis reaches each way, as `((h⁻, h⁺), (v⁻, v⁺))` in degrees —
/// the negative ends being negative.
pub(super) fn axis_reaches(camera: &CameraIntrinsics, scale: f32) -> ((f64, f64), (f64, f64)) {
    let step = sample_step_deg(camera, scale);
    let reach = |axis| {
        (
            sweep(camera, axis, step, -1.0).0,
            sweep(camera, axis, step, 1.0).0,
        )
    };
    (reach(Axis::Horizontal), reach(Axis::Vertical))
}

/// Walk one direction along `axis` until the model refuses the ray, the pixel
/// leaves the frame, or the map **folds**, returning how far it got and the
/// pixels it passed through.
///
/// The fold check is what keeps a circular fisheye's axis readable. On
/// `kerry_park` the `k1..k4` polynomial turns over near 130° and the radius
/// crashes from 191 px back to 6 px, so without it the polyline doubles back
/// through everything it already drew and a `−120°` tick lands between the
/// `−60°` and `−30°` ones. An axis is a scale only where it is monotone; past
/// the turn there is no reading to take, so the sweep stops.
fn sweep(camera: &CameraIntrinsics, axis: Axis, step: f64, sign: f64) -> (f64, Vec<[f64; 2]>) {
    let (cx, cy) = camera.principal_point();
    let mut points = Vec::new();
    let mut angle = 0.0_f64;
    let mut reached = 0.0_f64;
    let mut previous_radius = -1.0_f64;
    while angle <= MAX_SWEEP_DEG {
        let Some(p) = camera.ray_to_pixel(axis.ray(sign * angle)) else {
            break;
        };
        if out_of_frame(camera, p) {
            break;
        }
        let radius = (p.0 - cx).hypot(p.1 - cy);
        if radius < previous_radius {
            break;
        }
        previous_radius = radius;
        points.push([p.0, p.1]);
        reached = angle;
        angle += step;
    }
    (sign * reached, points)
}

/// One axis as a run of samples from its negative end to its positive one, each
/// flagged with whether the model still describes a lens there.
///
/// Inside `limit_deg` the axis is a measurement and is drawn solid; outside it
/// the model is extrapolating and the same polyline continues dashed, so a
/// reader following it out towards the frame's edge is told where the claim
/// stopped. The plot does the same thing to its curve.
fn axis_polyline(
    camera: &CameraIntrinsics,
    axis: Axis,
    step: f64,
    limit_deg: Option<f64>,
) -> Vec<([f64; 2], bool)> {
    let flag = |points: Vec<[f64; 2]>, sign: f64| {
        points
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                let angle = sign * i as f64 * step;
                (p, limit_deg.is_none_or(|limit| angle.abs() <= limit))
            })
            .collect::<Vec<_>>()
    };
    let (_, negative) = sweep(camera, axis, step, -1.0);
    let (_, positive) = sweep(camera, axis, step, 1.0);
    let mut out = flag(negative, -1.0);
    out.reverse();
    // Both sweeps start at 0°, so the join would otherwise be doubled.
    out.pop();
    out.extend(flag(positive, 1.0));
    out
}

/// Every ladder multiple within `reach`, paired with where it projects.
fn tick_positions(
    camera: &CameraIntrinsics,
    axis: Axis,
    ladder_deg: f64,
    reach: (f64, f64),
) -> Vec<(f64, [f64; 2])> {
    let first = (reach.0 / ladder_deg).ceil() as i64;
    let last = (reach.1 / ladder_deg).floor() as i64;
    (first..=last)
        .filter_map(|i| {
            let angle_deg = i as f64 * ladder_deg;
            let (u, v) = camera.ray_to_pixel(axis.ray(angle_deg))?;
            Some((angle_deg, [u, v]))
        })
        .collect()
}

/// The finest ladder step whose adjacent ticks are still
/// [`MIN_TICK_SPACING_PX`] apart at this scale, or the coarsest step when even
/// that is too dense.
///
/// The **finest** end of the ladder, not the coarsest: the coarsest step always
/// clears the spacing, so picking it would put three ticks on a zoomed-in long
/// lens. As many ticks as fit is the reading the ladder exists for.
pub(super) fn tick_ladder(
    camera: &CameraIntrinsics,
    scale: f32,
    reach: ((f64, f64), (f64, f64)),
) -> f64 {
    for candidate in LADDER_DEG {
        let spacing = [(Axis::Horizontal, reach.0), (Axis::Vertical, reach.1)]
            .into_iter()
            .flat_map(|(axis, reach)| {
                tick_positions(camera, axis, candidate, reach)
                    .windows(2)
                    .map(|pair| (pair[0].1[0] - pair[1].1[0]).hypot(pair[0].1[1] - pair[1].1[1]))
                    .collect::<Vec<_>>()
            })
            .fold(f64::INFINITY, f64::min);
        // An infinite minimum means fewer than two ticks anywhere, which no
        // coarser step is going to improve on.
        if spacing * f64::from(scale) >= f64::from(MIN_TICK_SPACING_PX) {
            return candidate;
        }
    }
    LADDER_DEG[LADDER_DEG.len() - 1]
}

/// A signed angle label. `+` to the right and `+` upward, per the frame
/// convention the axes are sampled in.
fn signed_angle(angle_deg: f64) -> String {
    if angle_deg < 0.0 {
        format!("{MINUS}{:.0}{DEGREE}", -angle_deg)
    } else if angle_deg > 0.0 {
        format!("+{angle_deg:.0}{DEGREE}")
    } else {
        format!("0{DEGREE}")
    }
}

/// The iso-angle contour at `theta_deg`, as the runs of it that stay on frame,
/// or `None` when none of it does.
fn contour(
    camera: &CameraIntrinsics,
    theta_deg: f64,
    scale: f32,
    label: String,
) -> Option<Contour> {
    let (sin_theta, cos_theta) = theta_deg.to_radians().sin_cos();
    // A step that puts samples about `SAMPLE_SPACING_PX` apart along a ring of
    // this radius, from the camera's own scale rather than a fixed count: a 1°
    // ring is a few pixels across and a 90° one spans the frame.
    let (fx, fy) = camera.focal_lengths();
    let radius_px = (fx.max(fy) * theta_deg.to_radians()).max(1.0);
    let step_deg = (f64::from(SAMPLE_SPACING_PX) / (radius_px * f64::from(scale.max(1e-3))))
        .to_degrees()
        .clamp(0.5, 15.0);

    let mut runs: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut run: Vec<[f64; 2]> = Vec::new();
    let mut azimuth = 0.0_f64;
    while azimuth <= 360.0 {
        let (sin_phi, cos_phi) = azimuth.to_radians().sin_cos();
        let ray = [sin_theta * cos_phi, sin_theta * sin_phi, -cos_theta];
        match camera.ray_to_pixel(ray) {
            Some(p) if !out_of_frame(camera, p) => run.push([p.0, p.1]),
            _ => {
                if run.len() > 1 {
                    runs.push(std::mem::take(&mut run));
                } else {
                    run.clear();
                }
            }
        }
        azimuth += step_deg;
    }
    if run.len() > 1 {
        runs.push(run);
    }
    if runs.is_empty() {
        return None;
    }
    // The label sits up and to the left, off both axes, rather than at the
    // contour's top where the vertical axis's own tick for the same angle is.
    let (sin_phi, cos_phi) = LABEL_AZIMUTH_DEG.to_radians().sin_cos();
    let label_at = camera
        .ray_to_pixel([sin_theta * cos_phi, sin_theta * sin_phi, -cos_theta])
        .filter(|p| !out_of_frame(camera, *p))
        .map(|p| [p.0, p.1])
        .or_else(|| {
            runs.iter()
                .flatten()
                .copied()
                .min_by(|a, b| a[1].total_cmp(&b[1]))
        })?;
    Some(Contour {
        runs,
        label,
        label_at,
    })
}

/// Azimuth the one contour label is placed at: up and to the left, where
/// neither axis runs.
const LABEL_AZIMUTH_DEG: f64 = 135.0;

/// Paint one contour as solid runs, labelled once.
fn draw_contour(
    painter: &egui::Painter,
    contour: &Contour,
    view: &View,
    panel: Rect,
    color: Color32,
) {
    for run in &contour.runs {
        haloed_line(painter, panel_points(run, view), 1.0, color);
    }
    if let Some(top) = contour_top(contour, view, panel) {
        haloed_text(
            painter,
            top - egui::vec2(0.0, 3.0),
            Align2::CENTER_BOTTOM,
            contour.label.clone(),
            ink(255),
        );
    }
}

/// Paint one contour dashed — the treatment the two domain edges get, so a
/// reader crossing one is told that what is outside it is not a measurement.
fn draw_dashed_contour(
    painter: &egui::Painter,
    contour: &Contour,
    view: &View,
    panel: Rect,
    color: Color32,
) {
    for run in &contour.runs {
        let points = panel_points(run, view);
        if points.len() < 2 {
            continue;
        }
        painter.extend(Shape::dashed_line(
            &points,
            Stroke::new(1.0 + super::HALO_WIDTH, super::HALO),
            DASH.0,
            DASH.1,
        ));
        painter.extend(Shape::dashed_line(
            &points,
            Stroke::new(1.0, color),
            DASH.0,
            DASH.1,
        ));
    }
    if let Some(top) = contour_top(contour, view, panel) {
        haloed_text(
            painter,
            top - egui::vec2(0.0, 3.0),
            Align2::CENTER_BOTTOM,
            contour.label.clone(),
            ink(255),
        );
    }
}

/// Where a contour's one label goes in panel coordinates, or `None` when that
/// point is off the panel.
fn contour_top(contour: &Contour, view: &View, panel: Rect) -> Option<Pos2> {
    let at = view.at(contour.label_at);
    panel.contains(at).then_some(at)
}

/// Paint one tick: a short stroke across its axis, and its label clear of it.
fn draw_tick(painter: &egui::Painter, tick: &Tick, view: &View, panel: Rect, color: Color32) {
    let at = view.at(tick.at);
    if !panel.contains(at) {
        return;
    }
    let (across, label_offset, anchor) = match tick.axis {
        Axis::Horizontal => (
            egui::vec2(0.0, 4.0),
            egui::vec2(0.0, 6.0),
            Align2::CENTER_TOP,
        ),
        Axis::Vertical => (
            egui::vec2(4.0, 0.0),
            egui::vec2(6.0, 0.0),
            Align2::LEFT_CENTER,
        ),
    };
    super::haloed_segment(painter, at - across, at + across, 1.0, color);
    haloed_text(
        painter,
        at + label_offset,
        anchor,
        tick.label.clone(),
        ink(255),
    );
}

/// The incidence angle at which a spline model's basis ends, in degrees.
///
/// The two spline models differ only in the radial coordinate the basis acts
/// on, so the conversion is per-coordinate: an incidence angle is already the
/// answer, and a normalized image-plane radius is `tan θ`. `None` for every
/// other model, and for a spline whose domain end is degenerate — the kernels
/// short-circuit those to the exact base arithmetic, so there is no domain to
/// mark.
fn spline_domain_deg(camera: &CameraIntrinsics) -> Option<f64> {
    let (_, domain_end, coordinate) = camera.model.radial_spline()?;
    if !(domain_end.is_finite() && domain_end > 0.0) {
        return None;
    }
    Some(match coordinate {
        SplineRadial::IncidenceAngle => domain_end.to_degrees(),
        SplineRadial::ImagePlaneRadius => domain_end.atan().to_degrees(),
    })
}
