// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Data types for the `.sift` file format.

use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::archive_io::ArchiveIoError;

/// Errors that can occur when reading or writing `.sift` files.
#[derive(Error, Debug)]
pub enum SiftError {
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

impl From<ArchiveIoError> for SiftError {
    fn from(e: ArchiveIoError) -> Self {
        match e {
            ArchiveIoError::Io(e) => SiftError::Io(e),
            ArchiveIoError::Zip(e) => SiftError::Zip(e),
            ArchiveIoError::Json(e) => SiftError::Json(e),
            ArchiveIoError::InvalidFormat(s) => SiftError::InvalidFormat(s),
            ArchiveIoError::ShapeMismatch(s) => SiftError::ShapeMismatch(s),
        }
    }
}

/// Feature tool metadata from `feature_tool_metadata.json.zst`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureToolMetadata {
    /// Tool name, e.g. "colmap" or "opencv".
    pub feature_tool: String,
    /// Feature type, e.g. "sift".
    pub feature_type: String,
    /// Tool-specific options as a JSON object.
    pub feature_options: serde_json::Value,
}

/// Image and feature metadata from `metadata.json.zst`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiftMetadata {
    /// Format version number.
    pub version: u32,
    /// Image filename without directory.
    pub image_name: String,
    /// XXH128 hex digest of the image file bytes.
    pub image_file_xxh128: String,
    /// Size of the image file in bytes.
    pub image_file_size: u64,
    /// Image width in pixels.
    pub image_width: u32,
    /// Image height in pixels.
    pub image_height: u32,
    /// Number of SIFT features.
    pub feature_count: u32,
}

/// Content integrity hashes from `content_hash.json.zst`.
///
/// All hash values are plain 32-character lowercase hex strings.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SiftContentHash {
    /// XXH128 hex digest of the uncompressed `metadata.json`.
    pub metadata_xxh128: String,
    /// XXH128 hex digest of the uncompressed `feature_tool_metadata.json`.
    pub feature_tool_xxh128: String,
    /// XXH128 hex digest of concatenated hash digests (see spec).
    pub content_xxh128: String,
}

/// Columnar SIFT feature data, mirroring the `.sift` file layout.
///
/// Each array field corresponds to a binary file in the archive.
pub struct SiftData {
    pub feature_tool_metadata: FeatureToolMetadata,
    pub metadata: SiftMetadata,
    pub content_hash: SiftContentHash,

    /// Feature positions as (x, y) pairs in COLMAP convention.
    /// Shape: `(feature_count, 2)`, dtype: `f32`.
    /// Pixel center of upper-left pixel is `(0.5, 0.5)`.
    pub positions_xy: Array2<f32>,

    /// Affine shape matrices `[[a11, a12], [a21, a22]]`.
    /// Shape: `(feature_count, 2, 2)`, dtype: `f32`.
    pub affine_shapes: Array3<f32>,

    /// 128-dimensional SIFT descriptors.
    /// Shape: `(feature_count, 128)`, dtype: `u8`.
    pub descriptors: Array2<u8>,

    /// 128×128 RGB thumbnail of the source image.
    /// Shape: `(128, 128, 3)`, dtype: `u8`.
    pub thumbnail_y_x_rgb: Array3<u8>,
}