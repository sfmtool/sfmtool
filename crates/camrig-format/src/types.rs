// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Data types for the `.camrig` file format.

use std::collections::HashMap;

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::archive_io::ArchiveIoError;

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
    /// Format version number. Must be `1`.
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

        if m.version != 1 {
            return Err(CamRigError::InvalidFormat(format!(
                "unsupported format version {} (expected 1)",
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
        for (i, &ci) in self.camera_indexes.iter().enumerate() {
            if ci as usize >= self.cameras.len() {
                return Err(CamRigError::InvalidFormat(format!(
                    "sensor {i} references camera {ci}, out of range [0, {})",
                    self.cameras.len()
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
