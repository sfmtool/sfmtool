// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Trimmed pose-only resection refinement.
//!
//! Refines a single world-to-camera pose against known 3D points by minimizing
//! pixel reprojection error with iterative trimming — the robust companion to
//! the minimal [`crate::geometry::absolute_pose`] estimator. A plain L2 warm-up
//! is dragged by junk correspondences' leverage and a robust loss has near-zero
//! gradient when every residual starts as a large outlier, so from a decent
//! init we instead refit L2 on the best-fitting fraction each round.
//!
//! Canonical camera frame throughout (the camera looks along `−Z`; a point in
//! front has `z < 0`). Each Levenberg–Marquardt step is taken over a local
//! `SO(3) × ℝ³` perturbation of the pose with an analytic Jacobian: the
//! projection derivative from
//! [`crate::CameraIntrinsics::ray_to_pixel_with_jacobian`] (a central
//! difference for models without an analytic one) composed with the exact
//! `−[R·X]ₓ` rotation and identity translation blocks.

use nalgebra::{Matrix6, UnitQuaternion, Vector3, Vector6};

use crate::camera::PixelJacobian;
use crate::CameraIntrinsics;

/// A point behind the camera / outside the model domain contributes this pixel
/// residual per component — large enough to be trimmed, finite so the normal
/// equations stay well-posed.
const INVALID_RESIDUAL: f64 = 1e6;

/// Result of [`refine_absolute_pose`].
#[derive(Clone, Debug)]
pub struct PoseRefinement {
    /// Refined world-to-camera rotation (canonical convention).
    pub rotation: UnitQuaternion<f64>,
    /// Refined world-to-camera translation.
    pub translation: Vector3<f64>,
    /// Fraction of the supplied observations within `inlier_px` after refinement.
    pub inlier_fraction: f64,
}

/// Six-parameter pose: `[rvec (3), t (3)]` with `R = exp(rvec)`.
type Params = [f64; 6];

fn rot_of(p: &Params) -> UnitQuaternion<f64> {
    UnitQuaternion::from_scaled_axis(Vector3::new(p[0], p[1], p[2]))
}

/// Reprojection residual norm per observation for pose `p`.
fn residual_norms(
    cam: &CameraIntrinsics,
    uv: &[[f64; 2]],
    points: &[[f64; 3]],
    p: &Params,
) -> Vec<f64> {
    let r = rot_of(p);
    let t = Vector3::new(p[3], p[4], p[5]);
    uv.iter()
        .zip(points.iter())
        .map(|(o, x)| {
            let c = r * Vector3::new(x[0], x[1], x[2]) + t;
            match cam.ray_to_pixel([c.x, c.y, c.z]) {
                Some((u, v)) => (u - o[0]).hypot(v - o[1]),
                None => INVALID_RESIDUAL,
            }
        })
        .collect()
}

/// Sum of squared pixel reprojection residuals over `idx` for pose `(r, t)`.
/// A point outside the model domain contributes `INVALID_RESIDUAL²` (matching
/// the `(INVALID_RESIDUAL, 0)` residual the Jacobian assembly assigns it).
fn cost(
    cam: &CameraIntrinsics,
    uv: &[[f64; 2]],
    points: &[[f64; 3]],
    idx: &[usize],
    r: &UnitQuaternion<f64>,
    t: &Vector3<f64>,
) -> f64 {
    idx.iter()
        .map(|&i| {
            let x = points[i];
            let c = r * Vector3::new(x[0], x[1], x[2]) + t;
            match cam.ray_to_pixel([c.x, c.y, c.z]) {
                Some((u, v)) => (u - uv[i][0]).powi(2) + (v - uv[i][1]).powi(2),
                None => INVALID_RESIDUAL * INVALID_RESIDUAL,
            }
        })
        .sum()
}

/// Projected pixel and the 2×3 projection Jacobian `∂(u, v)/∂p_cam` at a
/// camera-frame point. Analytic for the perspective family; a central
/// difference of `ray_to_pixel` for fisheye / equirectangular models, which
/// have no analytic Jacobian yet. `None` when the point is outside the model
/// domain (behind the camera / non-invertible).
fn project_with_jac(
    cam: &CameraIntrinsics,
    p_cam: Vector3<f64>,
    analytic: bool,
) -> Option<PixelJacobian> {
    if analytic {
        return cam.ray_to_pixel_with_jacobian([p_cam.x, p_cam.y, p_cam.z]);
    }
    let uv = cam.ray_to_pixel([p_cam.x, p_cam.y, p_cam.z])?;
    let h = 1e-6;
    let mut j = [[0.0f64; 3]; 2];
    for c in 0..3 {
        let mut pp = p_cam;
        let mut pm = p_cam;
        pp[c] += h;
        pm[c] -= h;
        let (up, vp) = cam.ray_to_pixel([pp.x, pp.y, pp.z])?;
        let (um, vm) = cam.ray_to_pixel([pm.x, pm.y, pm.z])?;
        j[0][c] = (up - um) / (2.0 * h);
        j[1][c] = (vp - vm) / (2.0 * h);
    }
    Some((uv, j))
}

/// Full pose Jacobian `Jᵢ = Jπ · [ −[R·X]ₓ | I₃ ]` (2×6): the 2×3 projection
/// block `jp` composed with the camera-point-to-pose block. The rotation block
/// `∂p_cam/∂δθ = −[R·X]ₓ` is the local (left) `SO(3)` perturbation; the
/// translation block is the identity.
fn compose_pose_jacobian(jp: &[[f64; 3]; 2], rot_pt: &Vector3<f64>) -> [[f64; 6]; 2] {
    let (a, b, c) = (rot_pt.x, rot_pt.y, rot_pt.z);
    // −[rot_pt]ₓ (columns are ∂p_cam/∂δθ).
    let nskew = [[0.0, c, -b], [-c, 0.0, a], [b, -a, 0.0]];
    let mut ji = [[0.0f64; 6]; 2];
    for row in 0..2 {
        for col in 0..3 {
            ji[row][col] = (0..3).map(|k| jp[row][k] * nskew[k][col]).sum();
            ji[row][3 + col] = jp[row][col]; // I₃ translation block
        }
    }
    ji
}

/// Levenberg–Marquardt over the six pose degrees of freedom (local `SO(3) × ℝ³`
/// perturbation), minimizing pixel reprojection error on the observation subset
/// `idx`. The Jacobian is analytic: the projection block from
/// [`CameraIntrinsics::ray_to_pixel_with_jacobian`] (a central difference for
/// models without an analytic one) composed with the exact `−[R·X]ₓ` rotation
/// and identity translation blocks.
fn lm_fit(
    cam: &CameraIntrinsics,
    uv: &[[f64; 2]],
    points: &[[f64; 3]],
    idx: &[usize],
    p0: Params,
    max_iter: usize,
) -> Params {
    if idx.len() < 3 {
        return p0;
    }
    let mut r = rot_of(&p0);
    let mut t = Vector3::new(p0[3], p0[4], p0[5]);
    let analytic = cam.model.supports_pixel_jacobian();
    let mut lambda = 1e-3;
    let mut prev = cost(cam, uv, points, idx, &r, &t);

    for _ in 0..max_iter {
        // Assemble the normal equations JᵀJ (6×6) and Jᵀr (6) over `idx`.
        let mut jtj = Matrix6::<f64>::zeros();
        let mut jtr = Vector6::<f64>::zeros();
        for &i in idx {
            let x = points[i];
            let rot_pt = r * Vector3::new(x[0], x[1], x[2]);
            let (res, jp) = match project_with_jac(cam, rot_pt + t, analytic) {
                Some(((u, v), jp)) => ([u - uv[i][0], v - uv[i][1]], jp),
                // Outside the domain: a large finite residual with a zero
                // Jacobian row, so it is penalized but does not steer the step.
                None => ([INVALID_RESIDUAL, 0.0], [[0.0; 3]; 2]),
            };
            let ji = compose_pose_jacobian(&jp, &rot_pt);
            for a in 0..2 {
                for col in 0..6 {
                    jtr[col] += ji[a][col] * res[a];
                    for col2 in 0..6 {
                        jtj[(col, col2)] += ji[a][col] * ji[a][col2];
                    }
                }
            }
        }

        let mut improved = false;
        for _ in 0..12 {
            let mut damped = jtj;
            for d in 0..6 {
                damped[(d, d)] += lambda * jtj[(d, d)].max(1e-12);
            }
            let Some(delta) = damped.lu().solve(&(-jtr)) else {
                lambda *= 4.0;
                continue;
            };
            // Apply the local perturbation: R ← exp([δθ]ₓ)·R, t ← t + δt.
            let dtheta = Vector3::new(delta[0], delta[1], delta[2]);
            let r_cand = UnitQuaternion::from_scaled_axis(dtheta) * r;
            let t_cand = t + Vector3::new(delta[3], delta[4], delta[5]);
            let new_cost = cost(cam, uv, points, idx, &r_cand, &t_cand);
            if new_cost < prev {
                r = r_cand;
                t = t_cand;
                prev = new_cost;
                lambda = (lambda * 0.5).max(1e-12);
                improved = true;
                break;
            }
            lambda *= 4.0;
            if lambda > 1e12 {
                break;
            }
        }
        if !improved {
            break;
        }
    }

    let axis = r.scaled_axis();
    [axis.x, axis.y, axis.z, t.x, t.y, t.z]
}

/// numpy-compatible linear-interpolation quantile of `values` at `q ∈ [0, 1]`.
fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = q * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        s[lo]
    } else {
        let frac = pos - lo as f64;
        s[lo] + frac * (s[hi] - s[lo])
    }
}

/// Refine a world-to-camera pose against 2D-3D correspondences by trimmed
/// pixel-reprojection least squares.
///
/// - `uv`: observed pixels (full image coordinates), one per correspondence.
/// - `points`: world points (canonical frame), one per correspondence.
/// - `init_rotation` / `init_translation`: the pose to refine from.
/// - `trim_rounds`: L2 refits on the best-fitting `keep_fraction` each round.
/// - `keep_fraction`: fraction of observations retained each trim round (0.6).
/// - `inlier_px`: final-inlier pixel threshold; a last refit runs on the
///   `< inlier_px` residuals when at least six qualify. The returned
///   `inlier_fraction` is the share of all supplied observations within it.
#[allow(clippy::too_many_arguments)]
pub fn refine_absolute_pose(
    cam: &CameraIntrinsics,
    uv: &[[f64; 2]],
    points: &[[f64; 3]],
    init_rotation: &UnitQuaternion<f64>,
    init_translation: &Vector3<f64>,
    trim_rounds: usize,
    keep_fraction: f64,
    inlier_px: f64,
) -> PoseRefinement {
    let n = uv.len();
    let axis = init_rotation.scaled_axis();
    let mut p: Params = [
        axis.x,
        axis.y,
        axis.z,
        init_translation.x,
        init_translation.y,
        init_translation.z,
    ];

    for _ in 0..trim_rounds {
        let rn = residual_norms(cam, uv, points, &p);
        let thresh = quantile(&rn, keep_fraction);
        let keep: Vec<usize> = (0..n).filter(|&i| rn[i] <= thresh).collect();
        p = lm_fit(cam, uv, points, &keep, p, 30);
    }

    let mut rn = residual_norms(cam, uv, points, &p);
    let inliers: Vec<usize> = (0..n).filter(|&i| rn[i] < inlier_px).collect();
    if inliers.len() >= 6 {
        p = lm_fit(cam, uv, points, &inliers, p, 30);
        rn = residual_norms(cam, uv, points, &p);
    }

    let frac = if n == 0 {
        0.0
    } else {
        rn.iter().filter(|&&r| r < inlier_px).count() as f64 / n as f64
    };
    PoseRefinement {
        rotation: rot_of(&p),
        translation: Vector3::new(p[3], p[4], p[5]),
        inlier_fraction: frac,
    }
}

#[cfg(test)]
mod tests;
