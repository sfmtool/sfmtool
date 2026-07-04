// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Epipolar geometry primitives.
//!
//! Shared utilities for computing epipoles from fundamental matrices,
//! used by both stereo rectification and polar sweep matching.

use nalgebra::{Matrix3, Point3, Vector3};
use rayon::prelude::*;

use crate::geometry::rotation::skew_symmetric;
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
/// Returns `None` when either intrinsic matrix is singular (e.g. a degenerate
/// camera with a zero focal length), in which case no fundamental matrix is
/// defined. Callers should treat this as "these two views cannot be related"
/// rather than crashing.
///
/// **Convention.** This is pure `K`-matrix pixel algebra: the poses must be
/// in the COLMAP/OpenCV **optical** camera convention (+Z forward, y down),
/// where `K · p_cam` dehomogenizes to the pixel. A caller holding canonical
/// `.sfmr` poses (−Z forward, +Y up) must pre-multiply each pose by
/// `S = diag(1, −1, −1)` first (`r → S·r`, `t → S·t`; see
/// [`crate::geometry::convention`]) — feeding canonical poses directly would
/// conjugate the essential matrix by `S` and yield a wrong pixel-space `F`.
///
/// # Parameters
///
/// * `k1`, `k2` - 3x3 intrinsic matrices.
/// * `r1`, `r2` - 3x3 cam_from_world rotation matrices (optical convention).
/// * `t1`, `t2` - cam_from_world translation vectors (optical convention).
#[allow(clippy::too_many_arguments)]
pub fn compute_fundamental_matrix(
    k1: &Matrix3<f64>,
    r1: &Matrix3<f64>,
    t1: &Vector3<f64>,
    k2: &Matrix3<f64>,
    r2: &Matrix3<f64>,
    t2: &Vector3<f64>,
) -> Option<Matrix3<f64>> {
    let r_rel = r2 * r1.transpose();
    let t_rel = t2 - r_rel * t1;
    let t_skew = skew_symmetric(&t_rel);
    let e = t_skew * r_rel;

    let k2_inv = k2.try_inverse()?;
    let k1_inv = k1.try_inverse()?;

    Some(k2_inv.transpose() * e * k1_inv)
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
        // Cheirality: a point in front of a canonical camera has z < 0.
        if xc.z >= 0.0 {
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
mod tests;
