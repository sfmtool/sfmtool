// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Coordinate-convention conversion primitives (COLMAP/OpenCV ↔ canonical).
//!
//! The canonical `.sfmr` convention (see `specs/formats/sfmr-file-format.md`
//! § "Coordinate System Conventions") is a right-handed **Z-up** world with
//! cameras looking down **−Z**, image plane +X right / +Y up (OpenGL-style).
//! COLMAP/OpenCV uses +Z-forward / Y-down cameras and typically −Y-up worlds.
//!
//! This module is the single source of truth for the conversion math (see
//! `specs/formats/sfmr-file-format.md` § "Conversions happen at the I/O
//! boundary" for the import/export formulas and invariants).
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

use sfmr_format::SfmrData;

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

/// Flip only the camera frame of a world-to-camera pose by the camera flip
/// `S`: `R' = S·R`, `t' = S·t`. The world frame is left untouched (no `W`).
///
/// Involutive (`S·S = I`). Used for in-pipeline pycolmap round trips (plan
/// §1, design decision D3) where a reconstruction is exported to OpenCV
/// camera frames and re-imported within one operation, leaving the world
/// frame fixed. Distinct from [`relative_pose_conjugate_s`] (`S·R·S`), which
/// is for rig-relative / `cam2_from_cam1` poses.
pub fn flip_camera_pose_s(
    rotation: &RotQuaternion,
    translation: &Vector3<f64>,
) -> (RotQuaternion, Vector3<f64>) {
    let r = rotation.to_rotation_matrix();
    let r_new = s_matrix() * r;
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

/// Upgrade a whole loaded [`SfmrData`] from the COLMAP convention (`.sfmr`
/// version ≤ 4) to the canonical convention, in place (design decision D1).
///
/// Applies exactly the plan-§1 import conversion:
///
/// - camera poses (`quaternions_wxyz` / `translations_xyz`):
///   [`pose_colmap_to_canonical`] (`R' = S·R·Wᵀ`, `t' = S·t`);
/// - rig sensor poses (`sensor_from_rig`), when `rig_frame_data` is present:
///   [`relative_pose_conjugate_s`] (`R' = S·R·S`, `t' = S·t` — `W` cancels
///   for rig-relative poses);
/// - world points (`positions_xyzw`): `xyz → W·xyz` with the homogeneous `w`
///   carried through unchanged, so `w = 0` infinity directions rotate by `W`
///   too;
/// - per-point normals and patch `u`/`v` half-vectors: `W` rotation (exact on
///   `f32` — `W` only permutes and negates components).
///
/// Depth statistics are left untouched: the stored per-image depth was
/// COLMAP camera-space `+z`, and the canonical depth is `−z'` with
/// `z' = −z`, so every depth value (and histogram) is invariant under the
/// conversion.
///
/// Content hashes describe the **stored** bytes, so integrity verification
/// ([`sfmr_format::verify_sfmr`], which re-reads the file) is unaffected by
/// this in-memory conversion; a converted reconstruction saved back to disk
/// is a new version-5 file with freshly computed hashes.
pub fn sfmr_data_colmap_to_canonical(data: &mut SfmrData) {
    // Camera poses.
    for i in 0..data.quaternions_wxyz.nrows() {
        let q = RotQuaternion::from_wxyz_array([
            data.quaternions_wxyz[[i, 0]],
            data.quaternions_wxyz[[i, 1]],
            data.quaternions_wxyz[[i, 2]],
            data.quaternions_wxyz[[i, 3]],
        ]);
        let t = Vector3::new(
            data.translations_xyz[[i, 0]],
            data.translations_xyz[[i, 1]],
            data.translations_xyz[[i, 2]],
        );
        let (q_new, t_new) = pose_colmap_to_canonical(&q, &t);
        for (k, &v) in q_new.to_wxyz_array().iter().enumerate() {
            data.quaternions_wxyz[[i, k]] = v;
        }
        for k in 0..3 {
            data.translations_xyz[[i, k]] = t_new[k];
        }
    }

    // Rig sensor poses (rig-relative: S-conjugation only).
    if let Some(rig) = data.rig_frame_data.as_mut() {
        for i in 0..rig.sensor_quaternions_wxyz.nrows() {
            let q = RotQuaternion::from_wxyz_array([
                rig.sensor_quaternions_wxyz[[i, 0]],
                rig.sensor_quaternions_wxyz[[i, 1]],
                rig.sensor_quaternions_wxyz[[i, 2]],
                rig.sensor_quaternions_wxyz[[i, 3]],
            ]);
            let t = Vector3::new(
                rig.sensor_translations_xyz[[i, 0]],
                rig.sensor_translations_xyz[[i, 1]],
                rig.sensor_translations_xyz[[i, 2]],
            );
            let (q_new, t_new) = relative_pose_conjugate_s(&q, &t);
            for (k, &v) in q_new.to_wxyz_array().iter().enumerate() {
                rig.sensor_quaternions_wxyz[[i, k]] = v;
            }
            for k in 0..3 {
                rig.sensor_translations_xyz[[i, k]] = t_new[k];
            }
        }
    }

    // World points: W on xyz, w carried through (also rotates w = 0 infinity
    // directions).
    for i in 0..data.positions_xyzw.nrows() {
        let y = data.positions_xyzw[[i, 1]];
        let z = data.positions_xyzw[[i, 2]];
        data.positions_xyzw[[i, 1]] = z;
        data.positions_xyzw[[i, 2]] = -y;
    }

    // World-space direction arrays: W rotation, exact on f32.
    let rotate_w_f32_rows = |arr: &mut ndarray::Array2<f32>| {
        for i in 0..arr.nrows() {
            let y = arr[[i, 1]];
            let z = arr[[i, 2]];
            arr[[i, 1]] = z;
            arr[[i, 2]] = -y;
        }
    };
    if let Some(normals) = data.normals_xyz.as_mut() {
        rotate_w_f32_rows(normals);
    }
    if let Some(u) = data.patch_u_halfvec_xyz.as_mut() {
        rotate_w_f32_rows(u);
    }
    if let Some(v) = data.patch_v_halfvec_xyz.as_mut() {
        rotate_w_f32_rows(v);
    }
    // `normal_confidence` is deliberately untouched: it is a scalar per point,
    // frame-independent, so no convention change affects it.
}

#[cfg(test)]
mod tests;
