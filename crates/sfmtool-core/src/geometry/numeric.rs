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

/// The camera at focal `f` — identity for every model but the five the focal
/// release admits: `SIMPLE_PINHOLE`, `EQUIDISTANT_FISHEYE`,
/// `SIMPLE_RADIAL_FISHEYE`, `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`, whose
/// projections all multiply `f` onto a distorted coordinate that does not
/// itself read `f` (for the last two, the dimensionless radial spline rides on
/// the ray's own radial coordinate).
///
/// Focal optimization is gated on exactly those five models, so no other
/// camera ever sees a moved focal; this matches the bundle adjustment's focal
/// handling.
pub(crate) fn cam_at(cam: &CameraIntrinsics, f: f64) -> CameraIntrinsics {
    let mut out = cam.clone();
    match &mut out.model {
        CameraModel::SimplePinhole { focal_length, .. }
        | CameraModel::EquidistantFisheye { focal_length, .. }
        | CameraModel::SimpleRadialFisheye { focal_length, .. }
        | CameraModel::SfmtoolFisheye { focal_length, .. }
        | CameraModel::SfmtoolPinhole { focal_length, .. } => *focal_length = f,
        _ => {}
    }
    out
}

/// The camera at focal `f` and radial coefficient `k1` — [`cam_at`] plus the
/// one distortion parameter the bundle adjustment can release, which exists
/// only on `SIMPLE_RADIAL_FISHEYE`. `k1` is ignored for every other model
/// (`opt_k1` is gated on that one).
pub(crate) fn cam_with(cam: &CameraIntrinsics, f: f64, k1: f64) -> CameraIntrinsics {
    let mut out = cam_at(cam, f);
    if let CameraModel::SimpleRadialFisheye {
        radial_distortion_k1,
        ..
    } = &mut out.model
    {
        *radial_distortion_k1 = k1;
    }
    out
}

/// The camera at focal `f` and spline coefficients `bspline` — [`cam_at`]
/// plus the coefficient vector the bundle adjustment's spline release moves,
/// which exists on the two sfmtool spline models, `SFMTOOL_FISHEYE` and
/// `SFMTOOL_PINHOLE`. `bspline` is ignored for every other model
/// (`opt_bspline` is gated on those two). The sibling of [`cam_with`]: the two
/// never apply together, because no model carries both a `k1` and a spline.
pub(crate) fn cam_with_bspline(
    cam: &CameraIntrinsics,
    f: f64,
    bspline: &[f64],
) -> CameraIntrinsics {
    let mut out = cam_at(cam, f);
    let coeffs = match &mut out.model {
        CameraModel::SfmtoolFisheye {
            bspline: coeffs, ..
        }
        | CameraModel::SfmtoolPinhole {
            bspline: coeffs, ..
        } => coeffs,
        _ => return out,
    };
    coeffs.clear();
    coeffs.extend_from_slice(bspline);
    out
}
