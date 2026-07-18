// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Rotation-locked resection: solve a camera's translation against known
//! world points when its world-to-camera rotation is already fixed.
//!
//! See `specs/core/rotation-locked-resection.md`. With the rotation fixed the
//! problem is linear in the three translation components: each observation's
//! unit ray `r_k = pixel_to_ray(uv_k)` must be parallel to `R·X_k + t`, giving
//! the cross-product rows `[r_k]ₓ · t = −[r_k]ₓ · R·X_k`. Working in ray space
//! keeps the mechanism camera-model-agnostic — fisheye and equirectangular
//! observations resect through the same equations, `pixel_to_ray` absorbing
//! the model.
//!
//! The solve is trimmed iteratively reweighted least squares: a least-squares
//! solve over the current observation set, then a re-gate keeping observations
//! in front of the canonical camera (`(R·X_k + t)_z < 0`; the camera looks
//! along `−Z`) with pixel residual (through `ray_to_pixel`) below
//! `max_error_px`, repeated for three rounds or until the kept set is stable.
//! Fewer than `min_inliers` survivors at any round fails the resection.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use crate::CameraIntrinsics;

/// A point behind the camera / outside the model domain reports this pixel
/// residual norm — large enough to never pass the gate, finite so the output
/// stays well-formed.
const INVALID_RESIDUAL: f64 = 1e6;

/// Trim rounds: least-squares solve + residual gate per round.
const TRIM_ROUNDS: usize = 3;

/// Relative ridge added to the normal matrix only when the plain 3×3 solve
/// fails (exactly parallel rays): a degenerate bundle still returns the
/// least-squares translation — conditioning is the caller's concern.
const RIDGE: f64 = 1e-12;

/// Result of [`resect_translation`].
#[derive(Clone, Debug)]
pub struct TranslationResection {
    /// World-to-camera translation (canonical convention, `x_cam = R·X + t`).
    pub translation: Vector3<f64>,
    /// Per-input-observation survivor mask (in front and within the gate).
    pub inliers: Vec<bool>,
    /// Per-input-observation pixel residual norm at the final translation.
    /// Behind-camera / out-of-domain observations report [`INVALID_RESIDUAL`].
    pub residual_norms: Vec<f64>,
}

/// Least-squares translation over `kept`: with unit rays the stacked
/// cross-product rows reduce to the normal equations
/// `Σ (I − r·rᵀ) · t = −Σ (I − r·rᵀ) · R·X` (each `[r]ₓᵀ[r]ₓ = I − r·rᵀ`).
fn solve_ls(
    rays: &[Vector3<f64>],
    rot_pts: &[Vector3<f64>],
    kept: &[bool],
) -> Option<Vector3<f64>> {
    let mut m = Matrix3::<f64>::zeros();
    let mut b = Vector3::<f64>::zeros();
    for i in 0..rays.len() {
        if !kept[i] {
            continue;
        }
        let proj = Matrix3::identity() - rays[i] * rays[i].transpose();
        m += proj;
        b -= proj * rot_pts[i];
    }
    if let Some(t) = m.lu().solve(&b) {
        if t.iter().all(|v| v.is_finite()) {
            return Some(t);
        }
    }
    // Rank-deficient normal matrix (exactly parallel rays): a small relative
    // ridge picks a finite least-squares solution.
    let mut ridged = m;
    let scale = m.trace().abs().max(1.0);
    for d in 0..3 {
        ridged[(d, d)] += RIDGE * scale;
    }
    let t = ridged.lu().solve(&b)?;
    t.iter().all(|v| v.is_finite()).then_some(t)
}

/// Pixel residual norm of one observation at translation `t`, or
/// [`INVALID_RESIDUAL`] when the point is behind the canonical camera
/// (`z ≥ 0`) or outside the model domain.
fn residual_norm(
    cam: &CameraIntrinsics,
    uv: &[f64; 2],
    rot_pt: &Vector3<f64>,
    t: &Vector3<f64>,
) -> f64 {
    let c = rot_pt + t;
    if c.z >= 0.0 {
        return INVALID_RESIDUAL;
    }
    match cam.ray_to_pixel([c.x, c.y, c.z]) {
        Some((u, v)) => (u - uv[0]).hypot(v - uv[1]),
        None => INVALID_RESIDUAL,
    }
}

/// Resect a camera's translation against known world points with the
/// world-to-camera rotation locked.
///
/// - `uv`: observed pixels (full image coordinates), one per correspondence.
/// - `points`: world points (canonical frame), one per correspondence.
/// - `rotation`: the fixed world-to-camera rotation (`x_cam = R·X + t`).
/// - `max_error_px`: trim gate on the pixel residual norm.
/// - `min_inliers`: fewer survivors than this at any round fails.
///
/// Returns `None` when the observation counts disagree or the survivor set
/// falls below `min_inliers`. See `specs/core/rotation-locked-resection.md`.
pub fn resect_translation(
    cam: &CameraIntrinsics,
    rotation: &UnitQuaternion<f64>,
    points: &[[f64; 3]],
    uv: &[[f64; 2]],
    max_error_px: f64,
    min_inliers: usize,
) -> Option<TranslationResection> {
    let n = points.len();
    if uv.len() != n || n < min_inliers.max(1) {
        return None;
    }

    // Unit rays and rotated world points, computed once. A non-finite or
    // zero-length ray (or non-finite point) can never be an inlier and must
    // not poison the normal equations: excluded from every kept set.
    let mut rays = Vec::with_capacity(n);
    let mut rot_pts = Vec::with_capacity(n);
    let mut valid = vec![true; n];
    for i in 0..n {
        let r = cam.pixel_to_ray(uv[i][0], uv[i][1]);
        let mut ray = Vector3::new(r[0], r[1], r[2]);
        let norm = ray.norm();
        if !norm.is_finite() || norm < 1e-12 {
            valid[i] = false;
            ray = Vector3::zeros();
        } else {
            ray /= norm;
        }
        let x = points[i];
        let rot_pt = rotation * Vector3::new(x[0], x[1], x[2]);
        if !rot_pt.iter().all(|v| v.is_finite()) {
            valid[i] = false;
        }
        rays.push(ray);
        rot_pts.push(rot_pt);
    }

    // Trimmed IRLS: solve over the kept set, re-gate, repeat until stable.
    let mut kept = valid.clone();
    if kept.iter().filter(|&&k| k).count() < min_inliers {
        return None;
    }
    let mut translation = Vector3::zeros();
    for _ in 0..TRIM_ROUNDS {
        translation = solve_ls(&rays, &rot_pts, &kept)?;
        let mut new_kept = vec![false; n];
        let mut survivors = 0usize;
        for i in 0..n {
            if !valid[i] {
                continue;
            }
            let keep = residual_norm(cam, &uv[i], &rot_pts[i], &translation) < max_error_px;
            new_kept[i] = keep;
            survivors += keep as usize;
        }
        if survivors < min_inliers {
            return None;
        }
        let stable = new_kept == kept;
        kept = new_kept;
        if stable {
            break;
        }
    }

    let residual_norms: Vec<f64> = (0..n)
        .map(|i| {
            if valid[i] {
                residual_norm(cam, &uv[i], &rot_pts[i], &translation)
            } else {
                INVALID_RESIDUAL
            }
        })
        .collect();
    Some(TranslationResection {
        translation,
        inliers: kept,
        residual_norms,
    })
}

#[cfg(test)]
mod tests;
