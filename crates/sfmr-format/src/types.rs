// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Data types for the `.sfmr` file format.

use std::path::PathBuf;

use ndarray::{Array1, Array2, Array4};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

use crate::archive_io::ArchiveIoError;

/// Errors that can occur when reading or writing `.sfmr` files.
#[derive(Error, Debug)]
pub enum SfmrError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{operation} '{path}': {source}")]
    IoPath {
        operation: &'static str,
        path: PathBuf,
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

impl From<ArchiveIoError> for SfmrError {
    fn from(e: ArchiveIoError) -> Self {
        match e {
            ArchiveIoError::Io(e) => SfmrError::Io(e),
            ArchiveIoError::Zip(e) => SfmrError::Zip(e),
            ArchiveIoError::Json(e) => SfmrError::Json(e),
            ArchiveIoError::InvalidFormat(s) => SfmrError::InvalidFormat(s),
            ArchiveIoError::ShapeMismatch(s) => SfmrError::ShapeMismatch(s),
        }
    }
}

/// Camera intrinsics as stored in the `.sfmr` JSON format.
///
/// This mirrors the on-disk JSON representation in `cameras/metadata.json.zst`,
/// where parameters are stored as a flat string-keyed map (e.g., `"focal_length_x"`,
/// `"radial_distortion_k1"`, etc.) that varies by camera model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SfmrCamera {
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub parameters: HashMap<String, f64>,
}

impl SfmrCamera {
    /// Returns `(fx, fy, cx, cy)` for any COLMAP camera model.
    ///
    /// Models with a shared focal length (`focal_length`) return it as both
    /// `fx` and `fy`. Models with separate focal lengths use `focal_length_x`
    /// and `focal_length_y`. Panics if neither convention is present.
    pub fn pinhole_params(&self) -> (f64, f64, f64, f64) {
        let (fx, fy) = if let Some(&f) = self.parameters.get("focal_length") {
            (f, f)
        } else {
            (
                self.parameters["focal_length_x"],
                self.parameters["focal_length_y"],
            )
        };
        let cx = self.parameters["principal_point_x"];
        let cy = self.parameters["principal_point_y"];
        (fx, fy, cx, cy)
    }
}

/// Workspace contents configuration (mirrors `.sfm-workspace.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceContents {
    pub feature_tool: String,
    pub feature_type: String,
    pub feature_options: serde_json::Value,
    pub feature_prefix_dir: String,
}

/// Workspace metadata embedded in the `.sfmr` top-level metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceMetadata {
    pub absolute_path: String,
    pub relative_path: String,
    pub contents: WorkspaceContents,
}

/// Top-level reconstruction metadata from `metadata.json.zst`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SfmrMetadata {
    pub version: u32,
    pub operation: String,
    pub tool: String,
    pub tool_version: String,
    pub tool_options: HashMap<String, serde_json::Value>,
    pub workspace: WorkspaceMetadata,
    pub timestamp: String,
    pub image_count: u32,
    pub points3d_count: u32,
    pub observation_count: u32,
    pub camera_count: u32,
    /// Number of rig definitions. Present only when rig data exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rig_count: Option<u32>,
    /// Total sensors across all rigs. Present only when rig data exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sensor_count: Option<u32>,
    /// Number of frames (temporal instants). Present only when rig data exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_count: Option<u32>,
}

/// Content integrity hashes from `content_hash.json.zst`.
///
/// All hash values are plain 32-character lowercase hex strings.
/// The `rigs_xxh128` and `frames_xxh128` fields are only present when
/// the `.sfmr` file contains rig/frame data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentHash {
    pub metadata_xxh128: String,
    pub cameras_xxh128: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rigs_xxh128: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_xxh128: Option<String>,
    pub images_xxh128: String,
    pub points3d_xxh128: String,
    pub tracks_xxh128: String,
    pub content_xxh128: String,
}

/// Statistics for observed points in a single image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedDepthStats {
    pub count: u32,
    pub min_z: Option<f64>,
    pub max_z: Option<f64>,
    pub median_z: Option<f64>,
    pub mean_z: Option<f64>,
}

/// Per-image depth statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageDepthStats {
    pub histogram_min_z: Option<f64>,
    pub histogram_max_z: Option<f64>,
    pub observed: ObservedDepthStats,
}

/// Depth statistics for the entire reconstruction from `images/depth_statistics.json.zst`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthStatistics {
    pub num_histogram_buckets: u32,
    pub images: Vec<ImageDepthStats>,
}

/// A single rig definition in the `rigs/metadata.json.zst`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigDefinition {
    pub name: String,
    pub sensor_count: u32,
    pub sensor_offset: u32,
    pub ref_sensor_name: String,
    pub sensor_names: Vec<String>,
}

/// Rig metadata from `rigs/metadata.json.zst`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigsMetadata {
    pub rig_count: u32,
    pub sensor_count: u32,
    pub rigs: Vec<RigDefinition>,
}

/// Frames metadata from `frames/metadata.json.zst`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FramesMetadata {
    pub frame_count: u32,
}

/// Optional rig and frame data stored in the `.sfmr` file.
///
/// When `None`, the reconstruction has no multi-camera rigs and every
/// camera is implicitly a single-sensor rig with identity `sensor_from_rig`.
#[derive(Debug, Clone)]
pub struct RigFrameData {
    // Rigs
    pub rigs_metadata: RigsMetadata,
    /// `(S,)` camera intrinsics index per sensor.
    pub sensor_camera_indexes: Array1<u32>,
    /// `(S, 4)` WXYZ quaternions for `sensor_from_rig` rotation.
    pub sensor_quaternions_wxyz: Array2<f64>,
    /// `(S, 3)` XYZ translations for `sensor_from_rig`.
    pub sensor_translations_xyz: Array2<f64>,

    // Frames
    pub frames_metadata: FramesMetadata,
    /// `(F,)` rig definition index per frame.
    pub rig_indexes: Array1<u32>,
    /// `(N,)` global sensor index per image.
    pub image_sensor_indexes: Array1<u32>,
    /// `(N,)` frame index per image.
    pub image_frame_indexes: Array1<u32>,
}

/// Columnar reconstruction data, mirroring the `.sfmr` file layout.
///
/// Each field corresponds to a file in the archive. This is the primary
/// type for I/O — it maps directly to/from numpy arrays on the Python side.
///
/// The `workspace_dir` field is populated by [`crate::read_sfmr`] when
/// reading from a file path, using the workspace resolution strategy from
/// the spec. It is `None` when resolution fails or when the struct is
/// constructed programmatically.
pub struct SfmrData {
    /// Resolved workspace directory path (populated on read, `None` if unresolved).
    pub workspace_dir: Option<PathBuf>,
    pub metadata: SfmrMetadata,
    pub content_hash: ContentHash,
    pub cameras: Vec<SfmrCamera>,

    // Rigs and frames (optional)
    /// Rig definitions and frame groupings. `None` when no multi-camera rigs.
    pub rig_frame_data: Option<RigFrameData>,

    // Images
    pub image_names: Vec<String>,
    /// `(N,)` camera index per image.
    pub camera_indexes: Array1<u32>,
    /// `(N, 4)` WXYZ quaternions (world-to-camera rotation).
    pub quaternions_wxyz: Array2<f64>,
    /// `(N, 3)` XYZ translations (world-to-camera).
    pub translations_xyz: Array2<f64>,
    /// `N` x 16-byte XXH128 hashes identifying feature extraction tool.
    pub feature_tool_hashes: Vec<[u8; 16]>,
    /// `N` x 16-byte XXH128 hashes of `.sift` file contents.
    pub sift_content_hashes: Vec<[u8; 16]>,
    /// `(N, 128, 128, 3)` RGB thumbnails of the source images.
    pub thumbnails_y_x_rgb: Array4<u8>,

    // Points3D
    /// `(P, 3)` 3D point positions in world coordinates.
    pub positions_xyz: Array2<f64>,
    /// `(P, 3)` RGB colors (0-255).
    pub colors_rgb: Array2<u8>,
    /// `(P,)` RMS reprojection errors in pixels.
    pub reprojection_errors: Array1<f32>,
    /// `(P, 3)` estimated surface normals (unit vectors).
    pub estimated_normals_xyz: Array2<f32>,

    // Tracks
    /// `(M,)` image index per observation.
    pub image_indexes: Array1<u32>,
    /// `(M,)` feature index per observation.
    pub feature_indexes: Array1<u32>,
    /// `(M,)` 3D point index per observation.
    pub points3d_indexes: Array1<u32>,
    /// `(P,)` number of observations per 3D point.
    pub observation_counts: Array1<u32>,

    // Depth statistics
    pub depth_statistics: DepthStatistics,
    /// `(N, 128)` depth histogram counts per image.
    pub observed_depth_histogram_counts: Array2<u32>,
}
