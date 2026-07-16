// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Batched reprojection residuals for images sharing one camera model.
//!
//! Composes per-image world-to-camera poses with the model-general
//! [`CameraIntrinsics::ray_to_pixel`] projection (canonical camera frame: the
//! camera looks along `−Z`, a point in front has `z < 0`). This is the forward
//! reprojection model used by resection scoring, acceptance gates, and
//! pose-only refinement — not an optimizer.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use rayon::prelude::*;

use crate::CameraIntrinsics;

/// Per-observation reprojection residual `(proj − observed)` in pixels for a
/// set of images sharing one camera model.
///
/// - `quats_wxyz`: `n_img * 4`, world-to-camera rotation per image (WXYZ).
/// - `translations`: `n_img * 3`, world-to-camera translation per image.
/// - `points`: `n_pt * 3`, world points (canonical frame). A non-finite point
///   is treated as invalid.
/// - `uv`: `n_obs * 2`, observed pixels.
/// - `obs_img` / `obs_pt`: per-observation image and point indices.
/// - `invalid_residual`: magnitude assigned (on the x component, y = 0) when a
///   point is behind the camera or outside the model's valid domain
///   ([`CameraIntrinsics::ray_to_pixel`] returns `None`) — the observation is
///   kept as a large-residual outlier rather than dropped, so trims/inlier
///   counts see it. Pass `f64::INFINITY` to exclude by norm, or a large finite
///   value (e.g. `1e6`) to keep it finite for a least-squares residual.
///
/// Returns a flat `Vec` of length `n_obs * 2` (`dx, dy` per observation).
#[allow(clippy::too_many_arguments)]
pub fn reprojection_residuals(
    cam: &CameraIntrinsics,
    quats_wxyz: &[f64],
    translations: &[f64],
    points: &[f64],
    uv: &[f64],
    obs_img: &[u32],
    obs_pt: &[u32],
    invalid_residual: f64,
) -> Vec<f64> {
    let n_obs = obs_img.len();
    assert_eq!(obs_pt.len(), n_obs, "obs_img and obs_pt length mismatch");
    assert_eq!(uv.len(), n_obs * 2, "uv must be n_obs * 2");

    let n_img = quats_wxyz.len() / 4;
    let quats: Vec<UnitQuaternion<f64>> = (0..n_img)
        .map(|i| {
            let o = i * 4;
            UnitQuaternion::from_quaternion(Quaternion::new(
                quats_wxyz[o],
                quats_wxyz[o + 1],
                quats_wxyz[o + 2],
                quats_wxyz[o + 3],
            ))
        })
        .collect();

    let mut out = vec![0.0f64; n_obs * 2];
    out.par_chunks_mut(2).enumerate().for_each(|(k, r)| {
        let im = obs_img[k] as usize;
        let pt = obs_pt[k] as usize;
        let x = points[pt * 3];
        let y = points[pt * 3 + 1];
        let z = points[pt * 3 + 2];
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            r[0] = invalid_residual;
            r[1] = 0.0;
            return;
        }
        let t = im * 3;
        let cam_pt = quats[im] * Vector3::new(x, y, z)
            + Vector3::new(translations[t], translations[t + 1], translations[t + 2]);
        match cam.ray_to_pixel([cam_pt.x, cam_pt.y, cam_pt.z]) {
            Some((u, v)) => {
                r[0] = u - uv[k * 2];
                r[1] = v - uv[k * 2 + 1];
            }
            None => {
                r[0] = invalid_residual;
                r[1] = 0.0;
            }
        }
    });
    out
}

/// Fraction of observations whose reprojection residual norm is below
/// `threshold_px`. `residuals` is the flat `(dx, dy)` output of
/// [`reprojection_residuals`].
pub fn inlier_fraction(residuals: &[f64], threshold_px: f64) -> f64 {
    let n = residuals.len() / 2;
    if n == 0 {
        return 0.0;
    }
    let count = (0..n)
        .filter(|&k| residuals[2 * k].hypot(residuals[2 * k + 1]) < threshold_px)
        .count();
    count as f64 / n as f64
}

#[cfg(test)]
mod tests;
