// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Data types for the `.camrig` file format.

use std::collections::BTreeMap;

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use thiserror::Error;

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

    /// A file of a version this build no longer reads, below
    /// [`CAMRIG_OLDEST_READ_VERSION`]. The message names the last release
    /// that reads it, [`CAMRIG_LAST_RELEASE_READING_OLDER_VERSIONS`], so the
    /// user can upgrade the workspace with that release.
    #[error(
        "{} is a version {version} .camrig file, and this sfmtool reads .camrig files of \
         version {CAMRIG_OLDEST_READ_VERSION} or later. sfmtool \
         {CAMRIG_LAST_RELEASE_READING_OLDER_VERSIONS} is the last release that reads version \
         {version}: install it, run `sfm ws upgrade` on the workspace, then use the file with \
         this release.",
        path.display()
    )]
    VersionNoLongerRead {
        /// The file.
        path: std::path::PathBuf,
        /// Its `metadata.version`.
        version: u32,
    },
}

sfmtool_archive_io::impl_from_archive_io_error!(CamRigError);

/// Current `.camrig` format version. [`crate::write_camrig`] always writes
/// this version; [`crate::read_camrig`] accepts every version from
/// [`CAMRIG_OLDEST_READ_VERSION`] to this one. See
/// `specs/formats/camrig-file-format.md` § "Format versions".
pub const CAMRIG_FORMAT_VERSION: u32 = 2;

/// The oldest `.camrig` format version a reader accepts. A file of an older
/// version is refused with [`CamRigError::VersionNoLongerRead`].
pub const CAMRIG_OLDEST_READ_VERSION: u32 = 2;

/// The last sfmtool release whose readers accept the `.camrig` versions below
/// [`CAMRIG_OLDEST_READ_VERSION`], which a refusal of such a file names.
pub const CAMRIG_LAST_RELEASE_READING_OLDER_VERSIONS: &str = "0.4.1";

/// A camera in the shared camera pool (`cameras/metadata.json.zst`).
///
/// Structurally identical to `sfmtool_sfmr_format::SfmrCamera`: a COLMAP camera
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
    pub parameters: BTreeMap<String, f64>,
}

/// Top-level rig metadata from `metadata.json.zst`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CamRigMetadata {
    /// Format version. The current version is [`CAMRIG_FORMAT_VERSION`].
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

        if !(CAMRIG_OLDEST_READ_VERSION..=CAMRIG_FORMAT_VERSION).contains(&m.version) {
            return Err(CamRigError::InvalidFormat(format!(
                "unsupported .camrig format version {} (this build reads versions \
                 {CAMRIG_OLDEST_READ_VERSION} to {CAMRIG_FORMAT_VERSION})",
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
