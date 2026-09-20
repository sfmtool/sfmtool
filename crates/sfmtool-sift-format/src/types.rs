// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Data types for the `.sift` file format.

use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use sfmtool_archive_io::ArchiveIoError;

/// Edge length, in pixels, of the square RGB thumbnail every `.sift` carries.
///
/// The format pins this: it is written into the archive entry's *name*
/// (`thumbnail_y_x_rgb.<size>.<size>.3.uint8.zst`), so a reader locates the entry by
/// a string that embeds the size. Changing it changes the on-disk format and
/// makes existing files unreadable — this constant exists so that the several
/// places which must agree cannot drift apart, not because the value is
/// adjustable.
///
/// `.sfmr` carries the same thumbnails, copied out of the `.sift` files, and so
/// declares the same edge as `sfmtool_sfmr_format::THUMBNAIL_SIZE`. The two crates are
/// independent — neither depends on the other — so the agreement is enforced by
/// a compile-time assertion in `sfmtool-core`, the first crate that sees both.
pub const THUMBNAIL_SIZE: usize = 128;

/// Length of one SIFT descriptor, in bytes, which is the width of every
/// descriptor row a `.sift` file holds.
///
/// The format pins this the way it pins [`THUMBNAIL_SIZE`]: the width is
/// written into the archive entry's *name*
/// (`features/descriptors.<count>.<dim>.uint8.zst`), so a reader locates the
/// entry by a string that embeds it. A reader therefore never has to be told
/// how wide a descriptor is, and a consumer laying several images' descriptors
/// out end to end can size the buffer before it opens a file.
pub const DESCRIPTOR_DIM: usize = 128;

/// Archive entry name for the thumbnail.
///
/// The extent is part of the name, so [`read`](crate::read),
/// [`write`](crate::write) and [`verify`](crate::verify) must all spell it the
/// same way or a lookup fails and a content hash goes wrong. (This crate has no
/// `entries` module of the kind `sfmtool-sfmr-format` and `sfmtool-matches-format` carry; the
/// four below are the names whose text depends on a value declared elsewhere.)
pub(crate) fn thumbnail_entry_name() -> String {
    format!("thumbnail_y_x_rgb.{THUMBNAIL_SIZE}.{THUMBNAIL_SIZE}.3.uint8.zst")
}

/// Archive entry name for the keypoint centres of a `count`-feature file.
pub(crate) fn positions_entry_name(count: usize) -> String {
    format!("features/positions_xy.{count}.2.float32.zst")
}

/// Archive entry name for the affine shapes of a `count`-feature file.
pub(crate) fn affine_shapes_entry_name(count: usize) -> String {
    format!("features/affine_shapes.{count}.2.2.float32.zst")
}

/// Archive entry name for the descriptors of a `count`-feature file.
pub(crate) fn descriptors_entry_name(count: usize) -> String {
    format!("features/descriptors.{count}.{DESCRIPTOR_DIM}.uint8.zst")
}

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

/// Newest `.sift` format version this build understands.
pub const SIFT_FORMAT_VERSION: u32 = 1;

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

    /// Square RGB thumbnail of the source image, [`THUMBNAIL_SIZE`] on a side.
    /// Shape: `(THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3)`, dtype: `u8`.
    pub thumbnail_y_x_rgb: Array3<u8>,
}

/// Every feature of one `.sift` file, laid out flat, and the hashes that say
/// which archive they came out of.
///
/// The three columns are one entry per feature in feature order and are all the
/// same length, which the entry-size checks of the read guarantee. Descriptors
/// are row-major, [`DESCRIPTOR_DIM`] bytes to the feature, so a consumer
/// concatenating several images' descriptors copies a slice rather than a row
/// at a time.
pub struct SiftFeatures {
    /// The file's own content hashes, for a consumer recording which archive it
    /// read.
    pub content_hash: SiftContentHash,

    /// Feature positions as (x, y) pairs in COLMAP convention. Pixel center of
    /// the upper-left pixel is `(0.5, 0.5)`.
    pub positions_xy: Vec<[f32; 2]>,

    /// Affine shape matrices `[[a11, a12], [a21, a22]]`, one per feature.
    pub affine_shapes: Vec<[[f32; 2]; 2]>,

    /// Descriptors, `positions_xy.len() * DESCRIPTOR_DIM` bytes.
    pub descriptors: Vec<u8>,
}
