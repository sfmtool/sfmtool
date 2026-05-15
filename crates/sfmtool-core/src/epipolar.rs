// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Epipolar geometry primitives.
//!
//! Shared utilities for computing epipoles from fundamental matrices,
//! used by both stereo rectification and polar sweep matching.

use nalgebra::{Matrix3, Point3, Vector3};
use rayon::prelude::*;

use crate::rotation::skew_symmetric;
use crate::{CameraIntrinsics, RigidTransform};

/// Compute the fundamental matrix F from two camera poses and intrinsics.
///
/// `F = K2^{-T} [t_rel]_x R_rel K1^{-1}`
///
/// where `R_rel = R2 @ R1^T` and `t_rel = t2 - R_rel @ t1`.
///
/// The fundamental matrix relates corresponding points in pixel coordinates:
/// `p2^T F p1 = 0`.
///
/// # Parameters
///
/// * `k1`, `k2` - 3x3 intrinsic matrices.
/// * `r1`, `r2` - 3x3 cam_from_world rotation matrices.
/// * `t1`, `t2` - cam_from_world translation vectors.
#[allow(clippy::too_many_arguments)]
pub fn compute_fundamental_matrix(
    k1: &Matrix3<f64>,
    r1: &Matrix3<f64>,
    t1: &Vector3<f64>,
    k2: &Matrix3<f64>,
    r2: &Matrix3<f64>,
    t2: &Vector3<f64>,
) -> Matrix3<f64> {
    let r_rel = r2 * r1.transpose();
    let t_rel = t2 - r_rel * t1;
    let t_skew = skew_symmetric(&t_rel);
    let e = t_skew * r_rel;

    let k2_inv = k2
        .try_inverse()
        .expect("Intrinsic matrix K2 must be invertible");
    let k1_inv = k1
        .try_inverse()
        .expect("Intrinsic matrix K1 must be invertible");

    k2_inv.transpose() * e * k1_inv
}

/// Compute a single epipole from a fundamental matrix via SVD null space.
///
/// Finds the null space of `F^T` (the right epipole, i.e. the projection
/// of camera 1's center into image 2). To get the left epipole (null space
/// of `F`), pass `&f.transpose()`.
///
/// Returns `(epipole_xy, is_at_infinity)` where `is_at_infinity` is true
/// when `|w| < 1e-10`. The `epipole_xy` is the dehomogenized `[x/w, y/w]`
/// coordinates (undefined when at infinity, returns `[0.0, 0.0]`).
pub fn compute_epipole(f: &Matrix3<f64>) -> ([f64; 2], bool) {
    let ft = f.transpose();
    let svd = ft.svd(true, true);
    let v_t = svd.v_t.expect("SVD failed to compute V^T");

    // Null space is the last row of V^T (smallest singular value)
    let w = v_t[(2, 2)];
    let is_at_infinity = w.abs() < 1e-10;

    if is_at_infinity {
        ([0.0, 0.0], true)
    } else {
        ([v_t[(2, 0)] / w, v_t[(2, 1)] / w], false)
    }
}

/// Compute both epipoles from a 3x3 fundamental matrix.
///
/// The epipoles are the null spaces of `F` and `F^T` respectively:
/// - `e1`: null space of `F` (left epipole, projection of camera 2's center into image 1)
/// - `e2`: null space of `F^T` (right epipole, projection of camera 1's center into image 2)
///
/// Returns `Some((e1, e2))` where each is `[x, y]` in pixel coordinates, or
/// `None` if either epipole is at infinity (homogeneous w ≈ 0).
pub fn compute_epipole_pair(f: &Matrix3<f64>) -> Option<([f64; 2], [f64; 2])> {
    let (e2, e2_inf) = compute_epipole(f);
    if e2_inf {
        return None;
    }

    let (e1, e1_inf) = compute_epipole(&f.transpose());
    if e1_inf {
        return None;
    }

    Some((e1, e2))
}

/// Minimum baseline length (scene units) below which the epipolar plane is
/// degenerate; [`plot_epipolar_curve`] returns an empty polyline in that case.
const MIN_BASELINE: f64 = 1e-9;
/// Floor for `anchor_depth`, to keep `ln(anchor)` finite.
const MIN_ANCHOR: f64 = 1e-12;
/// One octave per bracketing step (Phase 1 seed/walk).
const LOG_STEP: f64 = std::f64::consts::LN_2;
/// Maximum number of octave-doubling steps when searching for / extending an
/// in-image bracket. ±24 octaves covers a 1.6×10⁷ dynamic range either side of
/// `anchor_depth` — enough for any plausible reconstruction scale.
const BRACKET_MAX_STEPS: usize = 24;
/// Bisection tolerance in log-depth units (~0.1% relative in λ).
const BRACKET_LOG_TOL: f64 = 1e-3;

/// Controls how an epipolar curve is sampled into a polyline.
#[derive(Debug, Clone)]
pub struct EpipolarCurveOptions {
    /// Maximum allowed perpendicular distance (pixels) from a segment's
    /// midpoint to its chord before the segment is further subdivided.
    pub curvature_tolerance: f64,
    /// Hard cap on the number of vertices in a single polyline. Stops
    /// subdivision once reached, even if the tolerance is not met.
    pub max_vertices: usize,
}

impl Default for EpipolarCurveOptions {
    fn default() -> Self {
        Self {
            curvature_tolerance: 0.5,
            max_vertices: 256,
        }
    }
}

/// Plot the epipolar curve in camera 2's image for pixel `p1` in camera 1.
///
/// Back-projects `p1` through `cam1`'s full model, brackets the depth interval
/// over which the world ray's reprojection stays inside camera 2's image, and
/// adaptively subdivides that interval until every chord lies within
/// `curvature_tolerance` pixels of the true curve. Returns an empty polyline
/// when the baseline is degenerate or no in-image interval is found within
/// `±BRACKET_MAX_STEPS` octaves of `anchor_depth`.
///
/// The returned polyline is fully inside `[0, cam2.width) × [0, cam2.height)`;
/// no further image-rectangle clipping is needed at draw time. `pose1`/`pose2`
/// are `cam_from_world` transforms. See `specs/core/epipolar-curves.md` for
/// the algorithm.
///
/// `anchor_depth` is the seed for Phase-1 bracketing (typically the
/// reconstructed depth of the observed track, otherwise the baseline length
/// `‖C2 − C1‖`). The seed search is wide enough that exact accuracy doesn't
/// matter; an order-of-magnitude estimate is fine.
pub fn plot_epipolar_curve(
    p1: [f64; 2],
    cam1: &CameraIntrinsics,
    pose1: &RigidTransform,
    cam2: &CameraIntrinsics,
    pose2: &RigidTransform,
    anchor_depth: f64,
    opts: &EpipolarCurveOptions,
) -> Vec<[f64; 2]> {
    let c1 = pose1.inverse_translation_origin();
    let c2 = pose2.inverse_translation_origin();
    if (c2 - c1).norm() < MIN_BASELINE {
        return Vec::new();
    }

    // Back-projected ray of p1 in world coordinates.
    let d1_cam = Vector3::from(cam1.pixel_to_ray(p1[0], p1[1]));
    let d1_world = pose1.rotation.inverse().rotate_vector(&d1_cam);

    let width = cam2.width as f64;
    let height = cam2.height as f64;
    let project = |log_lambda: f64| -> Option<[f64; 2]> {
        let lambda = log_lambda.exp();
        let world = Point3::from(c1.coords + lambda * d1_world);
        let xc = pose2.transform_point(&world);
        if xc.z <= 0.0 {
            return None;
        }
        let (u, v) = cam2.ray_to_pixel([xc.x, xc.y, xc.z])?;
        if !(0.0..width).contains(&u) || !(0.0..height).contains(&v) {
            return None;
        }
        Some([u, v])
    };

    let log_anchor = anchor_depth.abs().max(MIN_ANCHOR).ln();

    // Phase 1 (bracket in log-depth — robust to wide dynamic range): find an
    // in-image seed, then bisect each side to find the in-image boundaries.
    let log_seed = match find_inimage_seed(log_anchor, &project) {
        Some(s) => s,
        None => return Vec::new(),
    };
    let log_in = bisect_boundary(log_seed, -LOG_STEP, &project);
    let log_out = bisect_boundary(log_seed, LOG_STEP, &project);

    let p_in = match project(log_in) {
        Some(p) => p,
        None => return Vec::new(),
    };
    if (log_out - log_in).abs() < BRACKET_LOG_TOL {
        return vec![p_in];
    }
    let p_out = match project(log_out) {
        Some(p) => p,
        None => return vec![p_in],
    };

    // Phase 2 (subdivide in t = 1/λ — the natural projective parameter): the
    // pixel-space curve is roughly affine in 1/λ (exactly so for perspective
    // projection), so arithmetic-mean midpoints in `t` give balanced
    // coverage even when the bracket spans many orders of magnitude with
    // one endpoint at the "infinity cap". A log-depth midpoint would collapse
    // toward the larger-λ endpoint and miss the curve's mid-region entirely.
    let t_in = (-log_in).exp();
    let t_out = (-log_out).exp();
    let project_t = |t: f64| -> Option<[f64; 2]> { project(-t.ln()) };
    let tol2 = opts.curvature_tolerance.max(0.0).powi(2);
    subdivide_worst_first(
        t_in,
        p_in,
        t_out,
        p_out,
        &project_t,
        tol2,
        opts.max_vertices,
    )
}

/// Phase 1 seed search: probe `log_anchor`, then walk outward in ±octaves
/// until some probe lands in-image (or `BRACKET_MAX_STEPS` is exhausted).
fn find_inimage_seed(log_anchor: f64, project: &impl Fn(f64) -> Option<[f64; 2]>) -> Option<f64> {
    if project(log_anchor).is_some() {
        return Some(log_anchor);
    }
    for k in 1..=BRACKET_MAX_STEPS {
        let step = k as f64 * LOG_STEP;
        let up = log_anchor + step;
        if project(up).is_some() {
            return Some(up);
        }
        let down = log_anchor - step;
        if project(down).is_some() {
            return Some(down);
        }
    }
    None
}

/// Walk outward from `log_seed` in steps of `dir_step` (signed, magnitude
/// `LOG_STEP`) until the predicate flips to out-of-image, then bisect to
/// `BRACKET_LOG_TOL`. Returns the in-image side of the converged bracket.
/// If `BRACKET_MAX_STEPS` is exhausted without flipping, returns the farthest
/// in-image probe (effectively treating the boundary as at infinity).
fn bisect_boundary(
    log_seed: f64,
    dir_step: f64,
    project: &impl Fn(f64) -> Option<[f64; 2]>,
) -> f64 {
    let mut log_in_side = log_seed;
    let mut log_out_side: Option<f64> = None;
    for k in 1..=BRACKET_MAX_STEPS {
        let probe = log_seed + (k as f64) * dir_step;
        if project(probe).is_some() {
            log_in_side = probe;
        } else {
            log_out_side = Some(probe);
            break;
        }
    }
    let mut log_out_side = match log_out_side {
        Some(v) => v,
        None => return log_in_side,
    };
    while (log_out_side - log_in_side).abs() > BRACKET_LOG_TOL {
        let mid = 0.5 * (log_in_side + log_out_side);
        if project(mid).is_some() {
            log_in_side = mid;
        } else {
            log_out_side = mid;
        }
    }
    log_in_side
}

/// Squared perpendicular distance from `p_m` to the line through `p_a` and
/// `p_b`. When `p_a` and `p_b` coincide, falls back to `|p_m − p_a|²`.
fn chord_deviation_sq(p_a: [f64; 2], p_b: [f64; 2], p_m: [f64; 2]) -> f64 {
    let dx = p_b[0] - p_a[0];
    let dy = p_b[1] - p_a[1];
    let chord_len2 = dx * dx + dy * dy;
    if chord_len2 < 1e-12 {
        let qx = p_m[0] - p_a[0];
        let qy = p_m[1] - p_a[1];
        qx * qx + qy * qy
    } else {
        let cross = (p_m[0] - p_a[0]) * dy - (p_m[1] - p_a[1]) * dx;
        (cross * cross) / chord_len2
    }
}

/// Cached subdivision candidate for a gap between two vertices.
///
/// The `f64` parameter `t` is opaque to this layer — Phase 2 uses `t = 1/λ`
/// so that arithmetic midpoints correspond to natural projective splits, but
/// the data structure only requires that `t` be a totally-ordered number that
/// `project_t` knows how to interpret.
#[derive(Debug, Clone, Copy)]
struct GapCandidate {
    /// Midpoint position (parameter value and pixel). `None` when the midpoint
    /// projection fails the in-image predicate, in which case the gap is
    /// considered final (we don't try to split across a disconnected region).
    midpoint: Option<(f64, [f64; 2])>,
    /// Squared chord deviation at the cached midpoint, or 0.0 when the
    /// midpoint is unprojectable. A value ≤ `tol2` accepts the gap.
    dev2: f64,
}

/// Phase 2: worst-first midpoint subdivision in the parameter space chosen
/// by the caller (typically `t = 1/λ`).
///
/// Maintains a sorted list of polyline vertices and a parallel list of gap
/// candidates. Each iteration picks the gap with the largest cached deviation
/// and splits it, until every gap is within tolerance or the vertex cap is
/// reached. Splitting the worst gap first means a tight `max_vertices` budget
/// is spent where curvature is highest, rather than biased to one end like a
/// depth-first traversal.
fn subdivide_worst_first(
    t_in: f64,
    p_in: [f64; 2],
    t_out: f64,
    p_out: [f64; 2],
    project_t: &impl Fn(f64) -> Option<[f64; 2]>,
    tol2: f64,
    max_vertices: usize,
) -> Vec<[f64; 2]> {
    let max_v = max_vertices.max(2);
    let mut verts: Vec<(f64, [f64; 2])> = Vec::with_capacity(max_v.min(64));
    verts.push((t_in, p_in));
    verts.push((t_out, p_out));
    let mut gaps: Vec<GapCandidate> = Vec::with_capacity(max_v.min(64));
    gaps.push(eval_gap(t_in, p_in, t_out, p_out, project_t));

    while verts.len() < max_v {
        // Find the gap with maximum deviation above tolerance.
        let mut best_i: Option<usize> = None;
        let mut best_dev2 = tol2;
        for (i, g) in gaps.iter().enumerate() {
            if g.midpoint.is_some() && g.dev2 > best_dev2 {
                best_dev2 = g.dev2;
                best_i = Some(i);
            }
        }
        let Some(i) = best_i else {
            break;
        };
        // Already filtered to midpoint.is_some().
        let (t_m, p_m) = gaps[i].midpoint.unwrap();
        let (t_a, p_a) = verts[i];
        let (t_b, p_b) = verts[i + 1];
        verts.insert(i + 1, (t_m, p_m));
        gaps[i] = eval_gap(t_a, p_a, t_m, p_m, project_t);
        gaps.insert(i + 1, eval_gap(t_m, p_m, t_b, p_b, project_t));
    }

    verts.into_iter().map(|(_, p)| p).collect()
}

/// Evaluate the gap candidate (cached midpoint + chord deviation) for one
/// pair of adjacent vertices.
fn eval_gap(
    t_a: f64,
    p_a: [f64; 2],
    t_b: f64,
    p_b: [f64; 2],
    project_t: &impl Fn(f64) -> Option<[f64; 2]>,
) -> GapCandidate {
    let t_m = 0.5 * (t_a + t_b);
    match project_t(t_m) {
        Some(p_m) => GapCandidate {
            midpoint: Some((t_m, p_m)),
            dev2: chord_deviation_sq(p_a, p_b, p_m),
        },
        None => GapCandidate {
            midpoint: None,
            dev2: 0.0,
        },
    }
}

/// Batch form of [`plot_epipolar_curve`]: one polyline per input pixel,
/// parallelized over points.
///
/// `anchor_depths` must have the same length as `points1` — one seed per
/// curve. Use the reconstructed depth of the observed track when available,
/// falling back to baseline length `‖C2 − C1‖` for un-triangulated features.
/// Output order matches `points1`.
pub fn plot_epipolar_curves_batch(
    points1: &[[f64; 2]],
    anchor_depths: &[f64],
    cam1: &CameraIntrinsics,
    pose1: &RigidTransform,
    cam2: &CameraIntrinsics,
    pose2: &RigidTransform,
    opts: &EpipolarCurveOptions,
) -> Vec<Vec<[f64; 2]>> {
    assert_eq!(
        points1.len(),
        anchor_depths.len(),
        "points1 and anchor_depths must have the same length"
    );
    points1
        .par_iter()
        .zip(anchor_depths.par_iter())
        .map(|(&p1, &anchor)| plot_epipolar_curve(p1, cam1, pose1, cam2, pose2, anchor, opts))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CameraModel, RotQuaternion};
    use approx::assert_relative_eq;

    fn pinhole_cam() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::Pinhole {
                focal_length_x: 500.0,
                focal_length_y: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
            },
            width: 640,
            height: 480,
        }
    }

    fn fisheye_cam() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::OpenCVFisheye {
                focal_length_x: 300.0,
                focal_length_y: 300.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.05,
                radial_distortion_k2: -0.01,
                radial_distortion_k3: 0.002,
                radial_distortion_k4: 0.0,
            },
            width: 640,
            height: 480,
        }
    }

    #[test]
    fn epipolar_curve_pinhole_satisfies_fundamental_constraint() {
        let cam = pinhole_cam();
        let pose1 = RigidTransform::identity();
        let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(1.0, 0.0, 0.0));
        let p1 = [400.0, 305.0];
        let curve = plot_epipolar_curve(
            p1,
            &cam,
            &pose1,
            &cam,
            &pose2,
            5.0,
            &EpipolarCurveOptions::default(),
        );
        assert!(curve.len() >= 2);

        let k = cam.intrinsic_matrix();
        let f = compute_fundamental_matrix(
            &k,
            &pose1.to_rotation_matrix(),
            &pose1.translation,
            &k,
            &pose2.to_rotation_matrix(),
            &pose2.translation,
        );
        let p1h = Vector3::new(p1[0], p1[1], 1.0);
        for q in &curve {
            let p2h = Vector3::new(q[0], q[1], 1.0);
            let c = (p2h.transpose() * f * p1h)[(0, 0)];
            assert!(c.abs() < 1e-6, "epipolar constraint = {c}");
        }
    }

    #[test]
    fn epipolar_curve_pinhole_is_two_vertices() {
        // A pinhole epipolar "curve" is exactly a straight line, so the
        // adaptive subdivision should accept the initial chord immediately.
        let cam = pinhole_cam();
        let pose1 = RigidTransform::identity();
        let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.7, 0.2, 0.0));
        let curve = plot_epipolar_curve(
            [400.0, 305.0],
            &cam,
            &pose1,
            &cam,
            &pose2,
            5.0,
            &EpipolarCurveOptions::default(),
        );
        assert_eq!(curve.len(), 2, "pinhole curve should not be subdivided");
    }

    #[test]
    fn epipolar_curve_vertices_are_inside_image_rect() {
        let cam = fisheye_cam();
        let pose1 = RigidTransform::identity();
        let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.5, 0.0, 0.0));
        let curve = plot_epipolar_curve(
            [380.0, 250.0],
            &cam,
            &pose1,
            &cam,
            &pose2,
            5.0,
            &EpipolarCurveOptions::default(),
        );
        assert!(!curve.is_empty());
        for q in &curve {
            assert!(
                q[0] >= 0.0 && q[0] < cam.width as f64,
                "vertex u={} outside [0, {})",
                q[0],
                cam.width
            );
            assert!(
                q[1] >= 0.0 && q[1] < cam.height as f64,
                "vertex v={} outside [0, {})",
                q[1],
                cam.height
            );
        }
    }

    /// Squared distance from `q` to the polyline `curve`, evaluated against
    /// every segment (not just vertices). Returns `f64::INFINITY` for a curve
    /// with fewer than 2 vertices.
    fn polyline_distance(curve: &[[f64; 2]], q: [f64; 2]) -> f64 {
        if curve.len() < 2 {
            return f64::INFINITY;
        }
        let mut best = f64::INFINITY;
        for w in curve.windows(2) {
            let a = w[0];
            let b = w[1];
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let len2 = dx * dx + dy * dy;
            let d2 = if len2 < 1e-12 {
                (q[0] - a[0]).powi(2) + (q[1] - a[1]).powi(2)
            } else {
                let t = (((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / len2).clamp(0.0, 1.0);
                let px = a[0] + t * dx;
                let py = a[1] + t * dy;
                (q[0] - px).powi(2) + (q[1] - py).powi(2)
            };
            if d2 < best {
                best = d2;
            }
        }
        best.sqrt()
    }

    #[test]
    fn epipolar_curve_fisheye_subdivides_under_tight_tolerance() {
        // The chord-deviation criterion should produce more than two vertices
        // for any non-trivial fisheye curve when the tolerance is well below
        // the curve's natural sagitta. 0.01 px ensures subdivision triggers
        // regardless of how mild the curvature happens to be.
        let cam = fisheye_cam();
        let pose1 = RigidTransform::identity();
        let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.5, 0.3, 0.0));
        let opts = EpipolarCurveOptions {
            curvature_tolerance: 0.01,
            max_vertices: 256,
        };
        let curve = plot_epipolar_curve([180.0, 360.0], &cam, &pose1, &cam, &pose2, 5.0, &opts);
        assert!(
            curve.len() > 2,
            "fisheye curve at 0.01px tolerance should subdivide; got {} vertices",
            curve.len()
        );
    }

    #[test]
    fn epipolar_curve_passes_through_true_correspondence_fisheye() {
        let cam = fisheye_cam();
        let pose1 = RigidTransform::identity();
        let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.5, 0.1, 0.0));

        let x_world = Point3::new(0.3, -0.2, 4.0);
        let xc1 = pose1.transform_point(&x_world);
        let xc2 = pose2.transform_point(&x_world);
        let (u1, v1) = cam.ray_to_pixel([xc1.x, xc1.y, xc1.z]).unwrap();
        let (u2, v2) = cam.ray_to_pixel([xc2.x, xc2.y, xc2.z]).unwrap();

        let curve = plot_epipolar_curve(
            [u1, v1],
            &cam,
            &pose1,
            &cam,
            &pose2,
            4.0,
            &EpipolarCurveOptions::default(),
        );
        assert!(!curve.is_empty());
        // Measure to the polyline (segments), not just vertices — the adaptive
        // sampler emits sparse vertices where the curve is locally straight,
        // so a nearest-vertex test would be overly strict.
        let d = polyline_distance(&curve, [u2, v2]);
        assert!(
            d < 1.0,
            "polyline is {d}px from true match (curve has {} vertices)",
            curve.len()
        );
    }

    #[test]
    fn epipolar_curve_empty_for_zero_baseline() {
        let cam = pinhole_cam();
        let pose = RigidTransform::identity();
        let curve = plot_epipolar_curve(
            [400.0, 300.0],
            &cam,
            &pose,
            &cam,
            &pose,
            1.0,
            &Default::default(),
        );
        assert!(curve.is_empty());
    }

    #[test]
    fn epipolar_curve_empty_when_ray_entirely_behind_cam2() {
        // Camera 2 rotated 180° about Y and pushed behind camera 1: the
        // back-projected ray of (320,240) — straight down +Z in world — sits
        // entirely behind camera 2, so the in-image predicate is never true.
        let cam = pinhole_cam();
        let pose1 = RigidTransform::identity();
        let pose2 = RigidTransform::new(
            RotQuaternion::from_axis_angle(Vector3::y(), std::f64::consts::PI).unwrap(),
            Vector3::new(0.0, 0.0, -10.0),
        );
        let curve = plot_epipolar_curve(
            [320.0, 240.0],
            &cam,
            &pose1,
            &cam,
            &pose2,
            5.0,
            &EpipolarCurveOptions::default(),
        );
        assert!(
            curve.is_empty(),
            "expected empty polyline, got {} vertices",
            curve.len()
        );
    }

    #[test]
    fn epipolar_curve_respects_max_vertices_cap() {
        let cam = fisheye_cam();
        let pose1 = RigidTransform::identity();
        let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.5, 0.0, 0.0));
        let opts = EpipolarCurveOptions {
            curvature_tolerance: 0.001, // unreachable tightness
            max_vertices: 8,
        };
        let curve = plot_epipolar_curve([120.0, 360.0], &cam, &pose1, &cam, &pose2, 5.0, &opts);
        assert!(!curve.is_empty());
        assert!(
            curve.len() <= opts.max_vertices,
            "vertex count {} exceeds cap {}",
            curve.len(),
            opts.max_vertices
        );
    }

    #[test]
    fn epipolar_curves_batch_matches_scalar() {
        let cam = pinhole_cam();
        let pose1 = RigidTransform::identity();
        let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.3, 0.4, 0.0));
        let pts = [[400.0, 300.0], [200.0, 150.0], [500.0, 400.0]];
        let anchors = [5.0, 3.0, 8.0];
        let opts = EpipolarCurveOptions::default();
        let batch = plot_epipolar_curves_batch(&pts, &anchors, &cam, &pose1, &cam, &pose2, &opts);
        assert_eq!(batch.len(), pts.len());
        for (i, (&p1, &anchor)) in pts.iter().zip(anchors.iter()).enumerate() {
            let scalar = plot_epipolar_curve(p1, &cam, &pose1, &cam, &pose2, anchor, &opts);
            assert_eq!(batch[i], scalar);
        }
    }

    #[test]
    fn test_fundamental_matrix_identity_cameras() {
        // Two cameras at same pose: R_rel = I, t_rel = 0
        // E = [0]_x * I = 0, so F = 0
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r = Matrix3::identity();
        let t = Vector3::zeros();

        let f = compute_fundamental_matrix(&k, &r, &t, &k, &r, &t);
        assert_relative_eq!(f, Matrix3::zeros(), epsilon = 1e-10);
    }

    #[test]
    fn test_fundamental_matrix_lateral_baseline() {
        // Camera 1 at origin, camera 2 translated along X axis
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);

        let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

        // F should not be zero
        assert!(f.norm() > 1e-10);

        // F should be rank 2 (det = 0)
        assert_relative_eq!(f.determinant(), 0.0, epsilon = 1e-10);

        // Epipolar constraint: for a 3D point, its projections satisfy p2^T F p1 = 0
        // Point at (0, 0, 10): projects to (320, 240) in cam1, (370, 240) in cam2
        let p1 = Vector3::new(320.0, 240.0, 1.0);
        let p2 = Vector3::new(370.0, 240.0, 1.0);
        let epipolar_constraint = p2.transpose() * f * p1;
        assert_relative_eq!(epipolar_constraint[(0, 0)], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_fundamental_matrix_vertical_baseline() {
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(0.0, 1.0, 0.0);

        let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

        assert!(f.norm() > 1e-10);
        assert_relative_eq!(f.determinant(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_epipole_at_infinity() {
        // F = [[0,0,0],[0,0,-1],[0,1,0]]
        // Null space of F is [1,0,0] (at infinity), so pair should return None.
        let f = Matrix3::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]);
        assert!(compute_epipole_pair(&f).is_none());
    }

    #[test]
    fn test_epipole_from_pure_translation() {
        // Pure translation: P1 = [I|0], P2 = [I|t] with t = (2, 3, 1).
        // F = [t]_x is skew-symmetric, so both null(F) and null(F^T) are t.
        // Both epipoles dehomogenize to t/t_z = (2, 3).
        let f = Matrix3::from_row_slice(&[0.0, -1.0, 3.0, 1.0, 0.0, -2.0, -3.0, 2.0, 0.0]);

        let (e1, e2) = compute_epipole_pair(&f).expect("both epipoles should be finite");
        assert!((e1[0] - 2.0).abs() < 1e-6, "e1.x = {}", e1[0]);
        assert!((e1[1] - 3.0).abs() < 1e-6, "e1.y = {}", e1[1]);
        assert!((e2[0] - 2.0).abs() < 1e-6, "e2.x = {}", e2[0]);
        assert!((e2[1] - 3.0).abs() < 1e-6, "e2.y = {}", e2[1]);
    }

    #[test]
    fn test_epipole_from_diagonal_translation() {
        // Pure translation with t = (5, -4, 2).
        // Both epipoles = t/t_z = (2.5, -2).
        let f = Matrix3::from_row_slice(&[0.0, -2.0, -4.0, 2.0, 0.0, -5.0, 4.0, 5.0, 0.0]);

        let (e1, e2) = compute_epipole_pair(&f).expect("both epipoles should be finite");
        assert!((e1[0] - 2.5).abs() < 1e-6, "e1.x = {}", e1[0]);
        assert!((e1[1] - (-2.0)).abs() < 1e-6, "e1.y = {}", e1[1]);
        assert!((e2[0] - 2.5).abs() < 1e-6, "e2.x = {}", e2[0]);
        assert!((e2[1] - (-2.0)).abs() < 1e-6, "e2.y = {}", e2[1]);
    }

    #[test]
    fn test_single_epipole_lateral_motion() {
        // For pure lateral motion, epipole is at infinity.
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);
        let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

        let (_epipole, is_at_infinity) = compute_epipole(&f);
        assert!(is_at_infinity);
    }

    #[test]
    fn test_fundamental_equals_essential_when_k_identity() {
        // When K = I, F = E = [t]_x R_rel
        let k = Matrix3::identity();
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);

        let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

        // E = [t_rel]_x * R_rel
        //   = [t2]_x * I
        //   = skew(t2)
        let e = skew_symmetric(&t2);

        // F should equal E when K = I
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (f[(i, j)] - e[(i, j)]).abs() < 1e-10,
                    "F[{i},{j}] = {} != E[{i},{j}] = {}",
                    f[(i, j)],
                    e[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_essential_matrix_epipolar_constraint() {
        // Verify p2^T E p1 = 0 for a known 3D point
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r1 = Matrix3::identity();
        let t1 = Vector3::zeros();
        let r2 = Matrix3::identity();
        let t2 = Vector3::new(1.0, 0.0, 0.0);

        let f = compute_fundamental_matrix(&k, &r1, &t1, &k, &r2, &t2);

        // 3D point at (2, 3, 10):
        // cam1: K * (2, 3, 10) / 10 = K * (0.2, 0.3, 1) = (420, 390, 1) (homogeneous)
        // cam2: K * (2+1, 3, 10) / 10 = K * (0.3, 0.3, 1) = (470, 390, 1)
        let p1 = Vector3::new(500.0 * 0.2 + 320.0, 500.0 * 0.3 + 240.0, 1.0);
        let p2 = Vector3::new(500.0 * 0.3 + 320.0, 500.0 * 0.3 + 240.0, 1.0);

        let constraint = p2.transpose() * f * p1;
        assert!(
            constraint[(0, 0)].abs() < 1e-6,
            "Epipolar constraint p2^T F p1 should be ≈ 0, got {}",
            constraint[(0, 0)]
        );
    }

    #[test]
    fn test_epipole_from_90_degree_rotation() {
        // Camera 2 rotated 90° around Y from camera 1
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r1 = Matrix3::identity();
        let t1 = Vector3::zeros();

        let angle = std::f64::consts::FRAC_PI_2;
        let r2 = Matrix3::new(
            angle.cos(),
            0.0,
            angle.sin(),
            0.0,
            1.0,
            0.0,
            -angle.sin(),
            0.0,
            angle.cos(),
        );
        let t2 = Vector3::new(1.0, 0.0, 0.0);

        let f = compute_fundamental_matrix(&k, &r1, &t1, &k, &r2, &t2);

        // F should be rank 2
        assert!(f.norm() > 1e-10);
        assert!(f.determinant().abs() < 1e-6, "F should be rank 2");
    }

    #[test]
    fn test_single_epipole_forward_motion() {
        // For forward motion along Z, epipole is at principal point.
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(0.0, 0.0, 1.0);
        let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

        let (epipole, is_at_infinity) = compute_epipole(&f);
        assert!(!is_at_infinity);
        assert!((epipole[0] - 320.0).abs() < 1.0);
        assert!((epipole[1] - 240.0).abs() < 1.0);
    }
}
