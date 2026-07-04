// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Data types for the `.camrig` file format.

use std::collections::HashMap;

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::archive_io::ArchiveIoError;
use crate::pattern::{count_frame_fields, validate_pattern};

/// Errors that can occur when reading or writing `.camrig` files.
#[derive(Error, Debug)]
pub enum CamRigError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{operation} '{path}': {source}")]
    IoPath {
        operation: &'static str,
        path: std::path::PathBuf,
        source: std::io::Error,
    },

    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Hash verification failed: {0}")]
    HashMismatch(String),
}

impl From<ArchiveIoError> for CamRigError {
    fn from(e: ArchiveIoError) -> Self {
        match e {
            ArchiveIoError::Io(e) => CamRigError::Io(e),
            ArchiveIoError::Zip(e) => CamRigError::Zip(e),
            ArchiveIoError::Json(e) => CamRigError::Json(e),
            ArchiveIoError::InvalidFormat(s) => CamRigError::InvalidFormat(s),
            ArchiveIoError::ShapeMismatch(s) => CamRigError::ShapeMismatch(s),
        }
    }
}

/// Current `.camrig` format version. [`crate::write_camrig`] always writes
/// this version; [`crate::read_camrig`] accepts any version up to it.
///
/// Version 2 made the canonical camera convention normative for the stored
/// `sensor_from_rig` poses (sensors looking down −Z with +Y up — see
/// `specs/formats/camrig-file-format.md` § "Versioning and migration"),
/// mirroring the `.sfmr` version 5 bump. The bump is purely semantic: no
/// member was added, removed, or renamed. Version 1 files hold
/// COLMAP-convention sensor poses and are upgraded on load by S-conjugation
/// ([`s_conjugate_sensor_pose`]).
pub const CAMRIG_FORMAT_VERSION: u32 = 2;

/// Conjugate a `sensor_from_rig` pose with the camera-frame flip
/// `S = diag(1, −1, −1)`: `R' = S·R·S`, `t' = S·t`.
///
/// In quaternion terms, conjugating by the 180°-about-X rotation negates the
/// y and z components: `(w, x, y, z) → (w, x, −y, −z)`; the translation's y
/// and z flip likewise. Rig-relative poses never touch the world frame, so
/// the `.sfmr` world canonicalization `W` does not apply. Involutive, so the
/// same function maps COLMAP ↔ canonical in both directions.
///
/// This is a local copy of the relative-pose case of
/// `sfmtool_core::geometry::convention::relative_pose_conjugate_s`
/// (the single source of truth for the convention math): `camrig-format`
/// sits below `sfmtool-core` in the crate graph and cannot depend on it, and
/// the operation is an exact component permutation with no rotation-matrix
/// round trip, so duplicating it here is loss-free.
pub fn s_conjugate_sensor_pose(quaternion_wxyz: &mut [f64; 4], translation_xyz: &mut [f64; 3]) {
    quaternion_wxyz[2] = -quaternion_wxyz[2];
    quaternion_wxyz[3] = -quaternion_wxyz[3];
    translation_xyz[1] = -translation_xyz[1];
    translation_xyz[2] = -translation_xyz[2];
}

/// Left-multiply a `sensor_from_rig` pose by the camera-frame flip
/// `S = diag(1, −1, −1)`: `R' = S·R`, `t' = S·t`.
///
/// In quaternion terms, pre-multiplying by the 180°-about-X rotation maps
/// `(w, x, y, z) → (−x, w, −z, y)`; the translation's y and z flip.
///
/// This is the transport for a **world-anchored** sensor pose — one whose rig
/// frame *is* the reconstruction world (a `spherical_tiles` rig; see
/// [`CamRigData::upgrade_sensor_poses_from_v1`]). Only the sensor's camera
/// frame flips by `S`; the world frame is left fixed, so unlike the
/// body-anchored [`s_conjugate_sensor_pose`] there is no right `S` factor (and
/// unlike a `.sfmr` world pose no `W` factor — the `.camrig` world is never
/// canonicalized). Involutive (`S·S = I`), so it maps COLMAP ↔ canonical in
/// both directions.
pub fn s_premultiply_sensor_pose(quaternion_wxyz: &mut [f64; 4], translation_xyz: &mut [f64; 3]) {
    let [w, x, y, z] = *quaternion_wxyz;
    *quaternion_wxyz = [-x, w, -z, y];
    translation_xyz[1] = -translation_xyz[1];
    translation_xyz[2] = -translation_xyz[2];
}

/// A camera in the shared camera pool (`cameras/metadata.json.zst`).
///
/// Structurally identical to `sfmr_format::SfmrCamera`: a COLMAP camera
/// model name, image dimensions, and a name→value parameter map whose keys
/// depend on `model`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CamRigCamera {
    /// COLMAP camera model name, e.g. `"PINHOLE"`, `"OPENCV_FISHEYE"`.
    pub model: String,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Named model parameters; the key set is determined by `model`.
    pub parameters: HashMap<String, f64>,
}

/// Top-level rig metadata from `metadata.json.zst`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CamRigMetadata {
    /// Format version number. `1` (COLMAP-convention sensor poses, upgraded
    /// on load) or [`CAMRIG_FORMAT_VERSION`] (canonical convention).
    pub version: u32,
    /// Human-readable rig name. May be empty.
    pub name: String,
    /// Number of sensors in the rig (`>= 1`).
    pub sensor_count: u32,
    /// Number of distinct cameras in the pool (`>= 1`).
    pub camera_count: u32,
    /// Hint describing how the rig was generated, e.g. `"generic"`,
    /// `"spherical_tiles"`, `"cubemap"`, `"fisheye_360"`, `"stereo_pair"`.
    pub rig_type: String,
    /// Free-form, `rig_type`-specific attributes. Not load-bearing.
    pub rig_attributes: serde_json::Value,
}

/// Content integrity hashes from `content_hash.json.zst`.
///
/// All hash values are 32-character lowercase hex strings.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CamRigContentHash {
    /// XXH128 hex digest of the uncompressed `metadata.json`.
    pub metadata_xxh128: String,
    /// XXH128 hex digest of concatenated member digests (see spec).
    pub content_xxh128: String,
}

/// Columnar camera-rig data, mirroring the `.camrig` file layout.
///
/// `S` denotes `metadata.sensor_count`.
#[derive(Debug, Clone)]
pub struct CamRigData {
    pub metadata: CamRigMetadata,
    pub content_hash: CamRigContentHash,

    /// Shared camera pool. Length `metadata.camera_count`.
    pub cameras: Vec<CamRigCamera>,

    /// Per-sensor image file pattern (see the format spec). Either empty
    /// (geometry-only rig, not backed by workspace images) or length `S`.
    pub sensor_image_patterns: Vec<String>,

    /// Camera pool index per sensor. Length `S`; each value `< camera_count`.
    pub camera_indexes: Vec<u32>,

    /// `sensor_from_rig` rotation per sensor as WXYZ unit quaternions.
    /// Shape `(S, 4)`.
    pub quaternions_wxyz: Array2<f64>,

    /// `sensor_from_rig` translation per sensor as XYZ vectors.
    /// Shape `(S, 3)`.
    pub translations_xyz: Array2<f64>,
}

impl CamRigData {
    /// Number of sensors in the rig.
    pub fn sensor_count(&self) -> usize {
        self.camera_indexes.len()
    }

    /// Whether the rig is geometry-only (no per-sensor image file patterns).
    pub fn is_anonymous(&self) -> bool {
        self.sensor_image_patterns.is_empty()
    }

    /// S-conjugate every `sensor_from_rig` pose in place (COLMAP ↔
    /// canonical; see [`s_conjugate_sensor_pose`]).
    pub fn s_conjugate_sensor_poses(&mut self) {
        self.map_sensor_poses(s_conjugate_sensor_pose);
    }

    /// Upgrade version-1 `sensor_from_rig` poses (COLMAP convention) to the
    /// canonical convention in place, dispatching on how the rig is anchored.
    ///
    /// Most rigs are **body-anchored**: the rig frame is a reference-sensor
    /// frame, so both frames flip under the convention change and the poses
    /// S-conjugate ([`s_conjugate_sensor_pose`]). A `spherical_tiles` rig is
    /// **world-anchored** — its rig frame *is* the reconstruction world, which
    /// the convention change leaves fixed — so only the sensor frame flips and
    /// the poses transport by a left `S` multiply ([`s_premultiply_sensor_pose`]).
    /// Applying S-conjugation to a world-anchored rig would leave every sensor
    /// rotated 180° about the world X axis (the wrong hemisphere), yet still
    /// pass structural validation.
    pub fn upgrade_sensor_poses_from_v1(&mut self) {
        if self.metadata.rig_type == "spherical_tiles" {
            self.map_sensor_poses(s_premultiply_sensor_pose);
        } else {
            self.map_sensor_poses(s_conjugate_sensor_pose);
        }
    }

    /// Apply a per-pose transform to every `sensor_from_rig` pose in place.
    fn map_sensor_poses(&mut self, f: fn(&mut [f64; 4], &mut [f64; 3])) {
        for i in 0..self.quaternions_wxyz.nrows() {
            let mut q = [
                self.quaternions_wxyz[[i, 0]],
                self.quaternions_wxyz[[i, 1]],
                self.quaternions_wxyz[[i, 2]],
                self.quaternions_wxyz[[i, 3]],
            ];
            let mut t = [
                self.translations_xyz[[i, 0]],
                self.translations_xyz[[i, 1]],
                self.translations_xyz[[i, 2]],
            ];
            f(&mut q, &mut t);
            for (k, &v) in q.iter().enumerate() {
                self.quaternions_wxyz[[i, k]] = v;
            }
            for (k, &v) in t.iter().enumerate() {
                self.translations_xyz[[i, k]] = v;
            }
        }
    }

    /// Check the structural constraints from the format spec
    /// (*Data ordering and constraints*).
    ///
    /// Run on both write and read: a file whose content hashes are valid but
    /// whose tables are inconsistent — an out-of-range camera index, a
    /// non-unit quaternion, a count that disagrees with a table length —
    /// must be rejected before a downstream consumer can panic on it.
    pub fn validate(&self) -> Result<(), CamRigError> {
        let m = &self.metadata;
        let s = self.camera_indexes.len();

        if m.version == 0 || m.version > CAMRIG_FORMAT_VERSION {
            return Err(CamRigError::InvalidFormat(format!(
                "unsupported format version {} (expected 1 to {CAMRIG_FORMAT_VERSION})",
                m.version
            )));
        }
        if s == 0 {
            return Err(CamRigError::InvalidFormat(
                "rig must have at least one sensor".into(),
            ));
        }
        if self.cameras.is_empty() {
            return Err(CamRigError::InvalidFormat(
                "rig must have at least one camera".into(),
            ));
        }
        if m.sensor_count as usize != s {
            return Err(CamRigError::ShapeMismatch(format!(
                "metadata.sensor_count {} != camera_indexes length {s}",
                m.sensor_count
            )));
        }
        if m.camera_count as usize != self.cameras.len() {
            return Err(CamRigError::ShapeMismatch(format!(
                "metadata.camera_count {} != cameras length {}",
                m.camera_count,
                self.cameras.len()
            )));
        }
        if self.quaternions_wxyz.shape() != [s, 4] {
            return Err(CamRigError::ShapeMismatch(format!(
                "quaternions_wxyz shape {:?} != [{s}, 4]",
                self.quaternions_wxyz.shape()
            )));
        }
        if self.translations_xyz.shape() != [s, 3] {
            return Err(CamRigError::ShapeMismatch(format!(
                "translations_xyz shape {:?} != [{s}, 3]",
                self.translations_xyz.shape()
            )));
        }
        if !self.sensor_image_patterns.is_empty() && self.sensor_image_patterns.len() != s {
            return Err(CamRigError::ShapeMismatch(format!(
                "sensor_image_patterns length {} is neither 0 (geometry-only) \
                 nor sensor_count {s}",
                self.sensor_image_patterns.len()
            )));
        }
        // Each image pattern must be a structurally valid pattern (see
        // `validate_pattern`), and a multi-sensor rig additionally requires a
        // frame field in every pattern. A frame-field-less pattern groups
        // frames positionally, which is well-defined only for a single-sensor
        // rig: with more sensors, frames are paired across sensors by frame
        // index. See the format spec, *How `.camrig` files fit into
        // workspaces*. Geometry-only rigs (empty patterns) are exempt.
        for (i, pattern) in self.sensor_image_patterns.iter().enumerate() {
            validate_pattern(pattern)
                .map_err(|msg| CamRigError::InvalidFormat(format!("sensor {i} {msg}")))?;
            if s > 1 && count_frame_fields(pattern) == 0 {
                return Err(CamRigError::InvalidFormat(format!(
                    "sensor {i} image pattern '{pattern}' has no frame field \
                     (%d / %0Nd); a rig with more than one sensor requires a \
                     frame field in every image pattern"
                )));
            }
        }
        for (i, &ci) in self.camera_indexes.iter().enumerate() {
            if ci as usize >= self.cameras.len() {
                return Err(CamRigError::InvalidFormat(format!(
                    "sensor {i} references camera {ci}, out of range [0, {})",
                    self.cameras.len()
                )));
            }
        }
        // Camera dimensions must be positive: a zero width or height is a
        // degenerate camera that yields a zero-aspect-ratio division in
        // consumers that scale intrinsics to an actual image size.
        for (i, camera) in self.cameras.iter().enumerate() {
            if camera.width == 0 || camera.height == 0 {
                return Err(CamRigError::InvalidFormat(format!(
                    "camera {i} has a non-positive dimension {}x{}; width and \
                     height must both be greater than zero",
                    camera.width, camera.height
                )));
            }
        }
        for i in 0..s {
            let q = self.quaternions_wxyz.row(i);
            let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            if (norm - 1.0).abs() > 1e-6 {
                return Err(CamRigError::InvalidFormat(format!(
                    "sensor {i} quaternion is not unit length (norm {norm})"
                )));
            }
        }

        Ok(())
    }
}

/// Open a file, mapping an I/O failure to a [`CamRigError::IoPath`] that names
/// the path. Shared by the read and verify paths.
pub(crate) fn open_file(path: &std::path::Path) -> Result<std::fs::File, CamRigError> {
    std::fs::File::open(path).map_err(|e| CamRigError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })
}
