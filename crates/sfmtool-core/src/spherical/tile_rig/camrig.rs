// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Conversion between [`SphericalTileRig`] and the `.camrig` file format.
//!
//! A spherical tile rig is a co-centric rig of `n` identical pinhole tiles,
//! so it maps onto `.camrig` with no loss: one shared camera in the pool,
//! all-zero translations, and one `sensor_from_rig` quaternion per tile. The
//! per-tile rotation already encodes the tile's look direction and tangent
//! basis, and the shared pool camera carries the tile intrinsics, from which
//! `patch_size` and `half_fov_rad` are recovered. `rig_attributes` carries
//! the remaining derived scalars (`centre`, the `measured_*` angles,
//! `atlas_cols`) so [`from_camrig`](SphericalTileRig::from_camrig) can rebuild
//! the rig via [`from_parts`](SphericalTileRig::from_parts) without re-running
//! the sphere-point relaxer. `rig_attributes` also stores informational copies
//! of `patch_size` and `half_fov_rad`, but those are not consumed on read —
//! the shared camera is authoritative.

use std::path::Path;

use camrig_format::{CamRigCamera, CamRigContentHash, CamRigData, CamRigError, CamRigMetadata};
use nalgebra::{Matrix3, Quaternion, Rotation3, UnitQuaternion};
use ndarray::Array2;
use sfmr_format::SfmrCamera;

use crate::camera::intrinsics::{CameraIntrinsics, CameraModel};

use super::{SphericalTileRig, SphericalTileRigError, SphericalTileRigParts};

/// zstd level for `.camrig` files written from a tile rig. The data (one
/// camera, all-zero translations, near-uniform quaternions) compresses well
/// regardless, so a mid-range level keeps writes fast.
const CAMRIG_ZSTD_LEVEL: i32 = 3;

/// Errors from converting a `.camrig` file into a [`SphericalTileRig`].
#[derive(Debug)]
pub enum CamRigConversionError {
    /// The `.camrig` data has a `rig_type` other than `"spherical_tiles"`.
    NotSphericalTiles(String),
    /// The `.camrig` data is structurally inconsistent with a
    /// `spherical_tiles` rig (failed [`CamRigData::validate`], or violates a
    /// rig-specific invariant: not a single shared camera, not co-centric, or
    /// a shared camera that is not a square centred pinhole tile camera).
    Invalid(String),
    /// A required `rig_attributes` field is missing or has the wrong type.
    BadAttribute(String),
    /// Reconstructing the rig from its parts failed.
    Rig(SphericalTileRigError),
    /// Reading or writing the `.camrig` file failed.
    File(CamRigError),
}

impl std::fmt::Display for CamRigConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSphericalTiles(t) => {
                write!(f, "rig_type is {t:?}, expected \"spherical_tiles\"")
            }
            Self::Invalid(m) => write!(f, "invalid spherical_tiles .camrig data: {m}"),
            Self::BadAttribute(m) => write!(f, "bad rig_attributes: {m}"),
            Self::Rig(e) => write!(f, "{e}"),
            Self::File(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CamRigConversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::File(e) => Some(e),
            _ => None,
        }
    }
}

impl From<SphericalTileRigError> for CamRigConversionError {
    fn from(e: SphericalTileRigError) -> Self {
        Self::Rig(e)
    }
}

impl From<CamRigError> for CamRigConversionError {
    fn from(e: CamRigError) -> Self {
        Self::File(e)
    }
}

impl SphericalTileRig {
    /// Build the `.camrig` representation of this rig.
    ///
    /// Produces a `spherical_tiles` rig with one shared pinhole camera (from
    /// [`tile_camera`](Self::tile_camera)), all-zero translations (the rig is
    /// co-centric), and one `sensor_from_rig` quaternion per tile. `name` is
    /// the human-readable rig name stored in the file metadata.
    ///
    /// `rig_attributes` also records `patch_size` and `half_fov_rad`, but
    /// those are informational only: [`from_camrig`](Self::from_camrig)
    /// recovers them from the shared camera, which is authoritative.
    pub fn to_camrig(&self, name: &str) -> CamRigData {
        let n = self.len();

        let camera = camrig_camera_from_intrinsics(&self.tile_camera());

        let mut quat = Vec::with_capacity(n * 4);
        for i in 0..n {
            quat.extend_from_slice(&sensor_from_rig_quat(&self.tile_rotation(i)));
        }
        let quaternions_wxyz =
            Array2::from_shape_vec((n, 4), quat).expect("n*4 values reshape to (n, 4)");
        let translations_xyz = Array2::zeros((n, 3));

        let rig_attributes = serde_json::json!({
            "centre": self.centre(),
            "half_fov_rad": self.half_fov_rad(),
            "measured_max_nn_angle": self.measured_max_nn_angle(),
            "measured_max_coverage_angle": self.measured_max_coverage_angle(),
            "patch_size": self.patch_size(),
            "atlas_cols": self.atlas_cols(),
        });

        CamRigData {
            metadata: CamRigMetadata {
                version: 1,
                name: name.to_string(),
                sensor_count: n as u32,
                camera_count: 1,
                rig_type: "spherical_tiles".to_string(),
                rig_attributes,
            },
            content_hash: CamRigContentHash::default(),
            cameras: vec![camera],
            sensor_image_patterns: Vec::new(),
            camera_indexes: vec![0; n],
            quaternions_wxyz,
            translations_xyz,
        }
    }

    /// Reconstruct a rig from its `.camrig` representation.
    ///
    /// The inverse of [`to_camrig`](Self::to_camrig). The per-tile look
    /// directions and tangent bases come from the `sensor_from_rig`
    /// quaternions; `patch_size` and `half_fov_rad` are recovered from the
    /// shared pool camera (the authoritative tile intrinsics); `centre`, the
    /// `measured_*` angles, and `atlas_cols` come from `rig_attributes`.
    ///
    /// A `spherical_tiles` rig is co-centric and uses a single shared pinhole
    /// tile camera, so this rejects any `data` that violates those invariants
    /// — non-zero translations, a camera count other than one, or a shared
    /// camera that is not a square centred pinhole — rather than silently
    /// dropping the divergent values.
    pub fn from_camrig(data: &CamRigData) -> Result<Self, CamRigConversionError> {
        if data.metadata.rig_type != "spherical_tiles" {
            return Err(CamRigConversionError::NotSphericalTiles(
                data.metadata.rig_type.clone(),
            ));
        }

        // Guarantees the array shapes and pool indices are self-consistent
        // before we index quaternion rows below.
        data.validate()
            .map_err(|e| CamRigConversionError::Invalid(e.to_string()))?;

        // spherical_tiles invariants: one shared pinhole camera, co-centric.
        if data.cameras.len() != 1 {
            return Err(CamRigConversionError::Invalid(format!(
                "expected exactly 1 shared camera, got {}",
                data.cameras.len()
            )));
        }
        if data.camera_indexes.iter().any(|&ci| ci != 0) {
            return Err(CamRigConversionError::Invalid(
                "every sensor must reference the single shared camera (index 0)".into(),
            ));
        }
        if let Some((i, _)) = data
            .translations_xyz
            .iter()
            .enumerate()
            .find(|(_, &t)| t != 0.0)
        {
            return Err(CamRigConversionError::Invalid(format!(
                "rig must be co-centric (all-zero translations); sensor {} is offset",
                i / 3
            )));
        }

        // The shared camera is the authoritative tile intrinsics; patch_size
        // and half_fov_rad are recovered from it, not from the informational
        // rig_attributes copies.
        let (patch_size, half_fov_rad) = tile_params_from_camera(&data.cameras[0])?;

        let attrs = &data.metadata.rig_attributes;
        let centre = attr_vec3(attrs, "centre")?;
        let measured_max_nn_angle = attr_f64(attrs, "measured_max_nn_angle")?;
        let measured_max_coverage_angle = attr_f64(attrs, "measured_max_coverage_angle")?;
        let atlas_cols = attr_u32(attrs, "atlas_cols")?;

        let n = data.quaternions_wxyz.nrows();
        let mut directions = Vec::with_capacity(n);
        let mut bases = Vec::with_capacity(n);
        for i in 0..n {
            let row = data.quaternions_wxyz.row(i);
            let (direction, basis) = world_from_tile_parts(&[row[0], row[1], row[2], row[3]]);
            directions.push(direction);
            bases.push(basis);
        }

        Self::from_parts(SphericalTileRigParts {
            centre,
            directions,
            bases,
            half_fov_rad,
            measured_max_nn_angle,
            measured_max_coverage_angle,
            patch_size,
            atlas_cols,
        })
        .map_err(CamRigConversionError::from)
    }

    /// Write this rig to a `.camrig` file at `path`.
    pub fn write_camrig(&self, path: &Path, name: &str) -> Result<(), CamRigConversionError> {
        let data = self.to_camrig(name);
        camrig_format::write_camrig(path, &data, CAMRIG_ZSTD_LEVEL)?;
        Ok(())
    }

    /// Read a rig from a `.camrig` file at `path`.
    pub fn read_camrig(path: &Path) -> Result<Self, CamRigConversionError> {
        let data = camrig_format::read_camrig(path)?;
        Self::from_camrig(&data)
    }
}

/// Convert a tile's `CameraIntrinsics` into a `.camrig` pool camera. The two
/// types are structurally identical to `sfmr_format::SfmrCamera`, so the
/// existing `CameraIntrinsics -> SfmrCamera` conversion does the work.
fn camrig_camera_from_intrinsics(cam: &CameraIntrinsics) -> CamRigCamera {
    let s = SfmrCamera::from(cam);
    CamRigCamera {
        model: s.model,
        width: s.width,
        height: s.height,
        parameters: s.parameters,
    }
}

/// Recover `(patch_size, half_fov_rad)` from the shared pool camera.
///
/// Inverse of [`SphericalTileRig::tile_camera`]: a tile camera is a square
/// pinhole with `width == height == patch_size`, equal focal lengths
/// `f = (patch_size / 2) / tan(half_fov_rad)`, and a centred principal point.
/// Rejects any camera that does not match that shape — the camera is the
/// authoritative tile intrinsics, so a malformed one is a malformed rig.
fn tile_params_from_camera(cam: &CamRigCamera) -> Result<(u32, f64), CamRigConversionError> {
    let sfmr = SfmrCamera {
        model: cam.model.clone(),
        width: cam.width,
        height: cam.height,
        parameters: cam.parameters.clone(),
    };
    let intrinsics = CameraIntrinsics::try_from(&sfmr)
        .map_err(|e| CamRigConversionError::Invalid(format!("shared camera: {e}")))?;

    let CameraModel::Pinhole {
        focal_length_x,
        focal_length_y,
        principal_point_x,
        principal_point_y,
    } = intrinsics.model
    else {
        return Err(CamRigConversionError::Invalid(
            "shared camera must use the PINHOLE model".into(),
        ));
    };

    if intrinsics.width != intrinsics.height {
        return Err(CamRigConversionError::Invalid(format!(
            "shared camera must be square; got {}x{}",
            intrinsics.width, intrinsics.height
        )));
    }
    let patch_size = intrinsics.width;
    if patch_size == 0 {
        return Err(CamRigConversionError::Invalid(
            "shared camera has zero width/height".into(),
        ));
    }
    let half = patch_size as f64 / 2.0;

    if !focal_length_x.is_finite() || focal_length_x <= 0.0 {
        return Err(CamRigConversionError::Invalid(
            "shared camera focal length must be finite and > 0".into(),
        ));
    }
    if (focal_length_x - focal_length_y).abs() > 1e-6 * focal_length_x {
        return Err(CamRigConversionError::Invalid(format!(
            "shared camera must have equal fx/fy; got {focal_length_x} and {focal_length_y}"
        )));
    }
    if (principal_point_x - half).abs() > 1e-6 * half
        || (principal_point_y - half).abs() > 1e-6 * half
    {
        return Err(CamRigConversionError::Invalid(format!(
            "shared camera principal point must be centred at ({half}, {half}); \
             got ({principal_point_x}, {principal_point_y})"
        )));
    }

    Ok((patch_size, (half / focal_length_x).atan()))
}

/// WXYZ quaternion of `R_sensor_from_rig` for a tile, given its
/// `R_world_from_tile` as a column-major 3x3 (`[e_right | e_up | direction]`).
///
/// The rig frame is the world frame, so `R_rig_from_sensor = R_world_from_tile`
/// and the stored `sensor_from_rig` rotation is its transpose.
fn sensor_from_rig_quat(world_from_tile_cols: &[f64; 9]) -> [f64; 4] {
    let r_world_from_tile = Matrix3::from_column_slice(world_from_tile_cols);
    let rot = Rotation3::from_matrix_unchecked(r_world_from_tile.transpose());
    let q = UnitQuaternion::from_rotation_matrix(&rot);
    [q.w, q.i, q.j, q.k]
}

/// Inverse of [`sensor_from_rig_quat`]: recover a tile's `(direction, basis)`
/// from its WXYZ `sensor_from_rig` quaternion. `basis` is `(e_right, e_up)`
/// flattened, matching `SphericalTileRig::bases`.
fn world_from_tile_parts(wxyz: &[f64; 4]) -> ([f64; 3], [f64; 6]) {
    let q = UnitQuaternion::from_quaternion(Quaternion::new(wxyz[0], wxyz[1], wxyz[2], wxyz[3]));
    // q is R_sensor_from_rig; R_world_from_tile = transpose.
    let m = q.to_rotation_matrix().matrix().transpose();
    let e_right = [m[(0, 0)], m[(1, 0)], m[(2, 0)]];
    let e_up = [m[(0, 1)], m[(1, 1)], m[(2, 1)]];
    let direction = [m[(0, 2)], m[(1, 2)], m[(2, 2)]];
    (
        direction,
        [
            e_right[0], e_right[1], e_right[2], e_up[0], e_up[1], e_up[2],
        ],
    )
}

fn attr_f64(attrs: &serde_json::Value, key: &str) -> Result<f64, CamRigConversionError> {
    attrs
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            CamRigConversionError::BadAttribute(format!("missing or non-numeric {key:?}"))
        })
}

fn attr_u32(attrs: &serde_json::Value, key: &str) -> Result<u32, CamRigConversionError> {
    attrs
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| {
            CamRigConversionError::BadAttribute(format!("missing or out-of-range {key:?}"))
        })
}

fn attr_vec3(attrs: &serde_json::Value, key: &str) -> Result<[f64; 3], CamRigConversionError> {
    let arr = attrs
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| CamRigConversionError::BadAttribute(format!("missing array {key:?}")))?;
    if arr.len() != 3 {
        return Err(CamRigConversionError::BadAttribute(format!(
            "{key:?} must have 3 elements, got {}",
            arr.len()
        )));
    }
    let mut out = [0.0; 3];
    for (i, e) in arr.iter().enumerate() {
        out[i] = e.as_f64().ok_or_else(|| {
            CamRigConversionError::BadAttribute(format!("{key:?}[{i}] is non-numeric"))
        })?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
