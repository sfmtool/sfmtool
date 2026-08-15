// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Small numeric primitives shared across the geometry kernels.
//!
//! Every item here previously existed as two or more byte-identical private
//! copies in sibling modules. They are RNG and threshold primitives feeding
//! RANSAC sampling and robust statistics, so a copy that drifts changes
//! reconstruction results without failing to compile — the reason they are
//! centralized rather than left co-located.

use nalgebra::Matrix3;

use crate::camera::{CameraIntrinsics, CameraModel};

/// SplitMix64 step: advance `state` and return the mixed output.
///
/// The deterministic RANSAC samplers seed one of these per kernel, so the
/// exact bit mixing is part of every sampling-dependent result.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Nearest rotation to `m` by polar decomposition (`U Vᵀ` from the SVD).
///
/// The conjugate-homography and pose-verification callers recover `R` only up
/// to scale *including sign*, so `M ≈ −R` must come back as `R`, not as the
/// (distant) proper projection of `−R`. `None` for a non-finite or degenerate
/// input.
pub(crate) fn polar_rotation(m: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = m.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let p = u * v_t;
    if !p.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(if p.determinant() < 0.0 { -p } else { p })
}

/// Rotation angle of `r` in radians.
pub(crate) fn rotation_angle(r: &Matrix3<f64>) -> f64 {
    (((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)).acos()
}

/// The camera at focal `f` — identity for every model but the two
/// single-focal, distortion-free ones: `SIMPLE_PINHOLE` and
/// `EQUIDISTANT_FISHEYE`.
///
/// Focal optimization is gated on exactly those two models, so no other camera
/// ever sees a moved focal; this matches the bundle adjustment's focal
/// handling.
pub(crate) fn cam_at(cam: &CameraIntrinsics, f: f64) -> CameraIntrinsics {
    let mut out = cam.clone();
    match &mut out.model {
        CameraModel::SimplePinhole { focal_length, .. }
        | CameraModel::EquidistantFisheye { focal_length, .. } => *focal_length = f,
        _ => {}
    }
    out
}
