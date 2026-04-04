// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Data types for the `.matches` file format.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::archive_io::ArchiveIoError;

/// Errors that can occur when reading or writing `.matches` files.
#[derive(Error, Debug)]
pub enum MatchesError {
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

impl From<ArchiveIoError> for MatchesError {
    fn from(e: ArchiveIoError) -> Self {
        match e {
            ArchiveIoError::Io(e) => MatchesError::Io(e),
            ArchiveIoError::Zip(e) => MatchesError::Zip(e),
            ArchiveIoError::Json(e) => MatchesError::Json(e),
            ArchiveIoError::InvalidFormat(s) => MatchesError::InvalidFormat(s),
            ArchiveIoError::ShapeMismatch(s) => MatchesError::ShapeMismatch(s),
        }
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

/// Workspace metadata embedded in the `.matches` top-level metadata.
///
/// Same structure as in `.sfmr` files — identifies the workspace and
/// feature extraction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceMetadata {
    pub absolute_path: String,
    pub relative_path: String,
    pub contents: WorkspaceContents,
}

/// Top-level metadata from `metadata.json.zst`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchesMetadata {
    pub version: u32,
    /// Type of matching used (e.g., "exhaustive", "sequential", "vocab_tree",
    /// "spatial", "transitive", "custom").
    pub matching_method: String,
    /// Tool that produced the matches (e.g., "colmap").
    pub matching_tool: String,
    /// Version string of the matching tool.
    pub matching_tool_version: String,
    /// Method-specific parameters. Contents depend on `matching_method`
    /// and `matching_tool`.
    pub matching_options: HashMap<String, serde_json::Value>,
    /// Workspace reference for relocatability.
    pub workspace: WorkspaceMetadata,
    /// ISO 8601 timestamp with timezone.
    pub timestamp: String,
    /// Number of images referenced.
    pub image_count: u32,
    /// Number of image pairs with matches.
    pub image_pair_count: u32,
    /// Total number of matches across all pairs.
    pub match_count: u32,
    /// Whether the optional two-view geometries section is present.
    pub has_two_view_geometries: bool,
}

/// Content integrity hashes from `content_hash.json.zst`.
///
/// All hash values are plain 32-character lowercase hex strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchesContentHash {
    pub metadata_xxh128: String,
    pub images_xxh128: String,
    pub image_pairs_xxh128: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub two_view_geometries_xxh128: Option<String>,
    pub content_xxh128: String,
}

/// Two-view geometry configuration type (COLMAP semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TwoViewGeometryConfig {
    Undefined,
    Degenerate,
    Calibrated,
    Uncalibrated,
    Planar,
    PlanarOrPanoramic,
    Panoramic,
    Multiple,
    WatermarkClean,
    WatermarkBad,
}

impl TwoViewGeometryConfig {
    /// Convert to the canonical string representation used in the file format.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Undefined => "undefined",
            Self::Degenerate => "degenerate",
            Self::Calibrated => "calibrated",
            Self::Uncalibrated => "uncalibrated",
            Self::Planar => "planar",
            Self::PlanarOrPanoramic => "planar_or_panoramic",
            Self::Panoramic => "panoramic",
            Self::Multiple => "multiple",
            Self::WatermarkClean => "watermark_clean",
            Self::WatermarkBad => "watermark_bad",
        }
    }
}

impl FromStr for TwoViewGeometryConfig {
    type Err = MatchesError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "undefined" => Ok(Self::Undefined),
            "degenerate" => Ok(Self::Degenerate),
            "calibrated" => Ok(Self::Calibrated),
            "uncalibrated" => Ok(Self::Uncalibrated),
            "planar" => Ok(Self::Planar),
            "planar_or_panoramic" => Ok(Self::PlanarOrPanoramic),
            "panoramic" => Ok(Self::Panoramic),
            "multiple" => Ok(Self::Multiple),
            "watermark_clean" => Ok(Self::WatermarkClean),
            "watermark_bad" => Ok(Self::WatermarkBad),
            _ => Err(MatchesError::InvalidFormat(format!(
                "Unknown TwoViewGeometryConfig: {s:?}"
            ))),
        }
    }
}

impl fmt::Display for TwoViewGeometryConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Two-view geometries section metadata from `two_view_geometries/metadata.json.zst`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TvgMetadata {
    pub image_pair_count: u32,
    pub inlier_count: u32,
    pub verification_tool: String,
    pub verification_options: HashMap<String, serde_json::Value>,
}

/// Optional two-view geometry data.
pub struct TwoViewGeometryData {
    pub metadata: TvgMetadata,
    /// Unique config type strings that appear in this file.
    pub config_types: Vec<TwoViewGeometryConfig>,
    /// `(P,)` index into `config_types` for each pair.
    pub config_indexes: Array1<u8>,
    /// `(P,)` number of geometrically verified inlier matches per pair.
    pub inlier_counts: Array1<u32>,
    /// `(I, 2)` inlier feature index pairs, flat concatenation across all pairs.
    pub inlier_feature_indexes: Array2<u32>,
    /// `(P, 3, 3)` fundamental matrices, row-major.
    pub f_matrices: Array3<f64>,
    /// `(P, 3, 3)` essential matrices, row-major.
    pub e_matrices: Array3<f64>,
    /// `(P, 3, 3)` homography matrices, row-major.
    pub h_matrices: Array3<f64>,
    /// `(P, 4)` relative rotation quaternions in WXYZ format.
    pub quaternions_wxyz: Array2<f64>,
    /// `(P, 3)` relative translation vectors.
    pub translations_xyz: Array2<f64>,
}

/// Primary data structure for `.matches` files.
///
/// Each field corresponds to a file in the archive. This is the primary
/// type for I/O.
pub struct MatchesData {
    pub metadata: MatchesMetadata,
    pub content_hash: MatchesContentHash,

    // Images
    /// Image paths relative to workspace directory (POSIX format).
    pub image_names: Vec<String>,
    /// `N` x 16-byte XXH128 hashes identifying feature extraction tool.
    pub feature_tool_hashes: Vec<[u8; 16]>,
    /// `N` x 16-byte XXH128 hashes of `.sift` file contents.
    pub sift_content_hashes: Vec<[u8; 16]>,
    /// `(N,)` feature count per image as used during matching.
    pub feature_counts: Array1<u32>,

    // Image pairs
    /// `(P, 2)` image index pairs, `idx_i < idx_j`, sorted lexicographically.
    pub image_index_pairs: Array2<u32>,
    /// `(P,)` number of matches per pair. `sum == match_count`.
    pub match_counts: Array1<u32>,
    /// `(M, 2)` feature index pairs, flat concatenation across all image pairs.
    pub match_feature_indexes: Array2<u32>,
    /// `(M,)` L2 descriptor distance per match.
    pub match_descriptor_distances: Array1<f32>,

    // Optional two-view geometries
    pub two_view_geometries: Option<TwoViewGeometryData>,
}