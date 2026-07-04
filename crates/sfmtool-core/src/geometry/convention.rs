// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Coordinate-convention conversion primitives (COLMAP/OpenCV ↔ canonical).
//!
//! The canonical `.sfmr` convention (see `specs/formats/sfmr-file-format.md`
//! § "Coordinate System Conventions") is a right-handed **Z-up** world with
//! cameras looking down **−Z**, image plane +X right / +Y up (OpenGL-style).
//! COLMAP/OpenCV uses +Z-forward / Y-down cameras and typically −Y-up worlds.
//!
//! This module is the single source of truth for the conversion math
//! (`specs/drafts/zup-camera-convention-migration.md` §1, design decision D2).
//! Two fixed proper rotations (det = +1) define every conversion:
//!
//! ```text
//! S = diag(1, −1, −1)      camera-frame flip: 180° about camera X (S·S = I)
//! W = [[1, 0, 0],          world canonicalization: (x, y, z) → (x, z, −y);
//!      [0, 0, 1],          maps COLMAP's typical −Y-up worlds to +Z-up
//!      [0, −1, 0]]
//! ```
//!
//! For world-to-camera poses `(R, t)` (`p_cam = R·p_world + t`):
//!
//! - COLMAP → canonical: `R' = S·R·Wᵀ`, `t' = S·t`
//! - canonical → COLMAP: `R = S·R'·W`, `t = S·t'`
//! - relative / rig-relative poses (`cam2_from_cam1`, `sensor_from_rig`):
//!   `R' = S·R·S`, `t' = S·t` (W cancels; the conversion is its own inverse)
//! - world points / directions / normals / patch half-vectors: `X' = W·X`
//!   (finite xyz only; a homogeneous `w` is carried through unchanged for
//!   points at infinity)

use nalgebra::{Matrix3, Vector3};

use crate::geometry::RotQuaternion;

/// The camera-frame flip `S = diag(1, −1, −1)`: a 180° rotation about the
/// camera X axis mapping between OpenCV (+Z forward, Y down) and canonical
/// (−Z forward, Y up) camera frames. Proper rotation, involutive (`S·S = I`).
pub fn s_matrix() -> Matrix3<f64> {
    Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0))
}

/// The world canonicalization rotation `W: (x, y, z) → (x, z, −y)`, a −90°
/// rotation about world X. Maps COLMAP's typical −Y-up worlds to +Z-up.
/// Proper rotation; its inverse is `Wᵀ: (x, y, z) → (x, −z, y)`.
pub fn w_matrix() -> Matrix3<f64> {
    Matrix3::new(
        1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, //
        0.0, -1.0, 0.0,
    )
}

/// Convert a COLMAP-convention world-to-camera pose to the canonical
/// convention: `R' = S·R·Wᵀ`, `t' = S·t` (plan §1, import direction).
pub fn pose_colmap_to_canonical(
    rotation: &RotQuaternion,
    translation: &Vector3<f64>,
) -> (RotQuaternion, Vector3<f64>) {
    let r = rotation.to_rotation_matrix();
    let r_new = s_matrix() * r * w_matrix().transpose();
    (
        RotQuaternion::from_rotation_matrix(r_new),
        s_matrix() * translation,
    )
}

/// Convert a canonical-convention world-to-camera pose back to COLMAP:
/// `R = S·R'·W`, `t = S·t'` (plan §1, export direction; inverse of
/// [`pose_colmap_to_canonical`]).
pub fn pose_canonical_to_colmap(
    rotation: &RotQuaternion,
    translation: &Vector3<f64>,
) -> (RotQuaternion, Vector3<f64>) {
    let r = rotation.to_rotation_matrix();
    let r_new = s_matrix() * r * w_matrix();
    (
        RotQuaternion::from_rotation_matrix(r_new),
        s_matrix() * translation,
    )
}

/// Conjugate a relative pose (`cam2_from_cam1` or rig `sensor_from_rig`) by
/// the camera flip `S`: `R' = S·R·S`, `t' = S·t` (plan §1 — the world
/// rotation `W` cancels for camera-to-camera poses). Involutive: applying it
/// twice returns the original pose, so the same function converts in both
/// directions.
pub fn relative_pose_conjugate_s(
    rotation: &RotQuaternion,
    translation: &Vector3<f64>,
) -> (RotQuaternion, Vector3<f64>) {
    let r = rotation.to_rotation_matrix();
    let r_new = s_matrix() * r * s_matrix();
    (
        RotQuaternion::from_rotation_matrix(r_new),
        s_matrix() * translation,
    )
}

/// Rotate a world-space vector by `W`: `(x, y, z) → (x, z, −y)`.
///
/// Applies to finite point coordinates, infinity directions, normals, and
/// patch `u`/`v` half-vectors on COLMAP → canonical import (plan §1). For
/// homogeneous `xyzw` points, rotate the xyz part and carry `w` unchanged.
pub fn world_rotate_w(v: &Vector3<f64>) -> Vector3<f64> {
    Vector3::new(v.x, v.z, -v.y)
}

/// Rotate a world-space vector by `W⁻¹ = Wᵀ`: `(x, y, z) → (x, −z, y)`.
///
/// The canonical → COLMAP (export) counterpart of [`world_rotate_w`].
pub fn world_rotate_w_inverse(v: &Vector3<f64>) -> Vector3<f64> {
    Vector3::new(v.x, -v.z, v.y)
}

#[cfg(test)]
mod tests;
