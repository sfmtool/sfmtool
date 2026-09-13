// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Current `.kdf` wire-format version.
pub const KDF_FORMAT_VERSION: u32 = 2;

/// Errors from persistent forest I/O, validation, and resource accounting.
#[derive(thiserror::Error, Debug)]
pub enum KdfError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid KDF format: {0}")]
    InvalidFormat(String),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("integrity mismatch: {0}")]
    Integrity(String),
    #[error("resource limit exceeded: {0}")]
    ResourceLimit(String),
    #[error("scalar type mismatch: file contains {file}, caller requested {requested}")]
    ScalarType {
        file: String,
        requested: &'static str,
    },
    #[error("invalid query: {0}")]
    InvalidQuery(String),
    #[error("source file is missing: {0}")]
    MissingSource(PathBuf),
}

impl From<sfmtool_archive_io::ArchiveIoError> for KdfError {
    fn from(value: sfmtool_archive_io::ArchiveIoError) -> Self {
        use sfmtool_archive_io::ArchiveIoError;
        match value {
            ArchiveIoError::Io(e) => Self::Io(e),
            ArchiveIoError::Zip(e) => Self::Zip(e),
            ArchiveIoError::Json(e) => Self::Json(e),
            ArchiveIoError::InvalidFormat(e) => Self::InvalidFormat(e),
            ArchiveIoError::ShapeMismatch(e) => Self::ShapeMismatch(e),
        }
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for f32 {}
}

/// Scalar types admitted by version 2 of the format.
pub trait KdfScalar:
    sealed::Sealed + bytemuck::Pod + Copy + Send + Sync + PartialEq + std::fmt::Debug + 'static
{
    /// Wire-format scalar name.
    const TYPE_NAME: &'static str;
    /// Positive zero for unused split fields.
    const ZERO: Self;
    /// Whether a stored or query coordinate is finite.
    fn is_finite(self) -> bool;
    /// Total coordinate order, including signed zero for floating point.
    fn total_cmp(self, other: Self) -> Ordering;
}

impl KdfScalar for u8 {
    const TYPE_NAME: &'static str = "uint8";
    const ZERO: Self = 0;
    fn is_finite(self) -> bool {
        true
    }
    fn total_cmp(self, other: Self) -> Ordering {
        self.cmp(&other)
    }
}

impl KdfScalar for f32 {
    const TYPE_NAME: &'static str = "float32";
    const ZERO: Self = 0.0;
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn total_cmp(self, other: Self) -> Ordering {
        f32::total_cmp(&self, &other)
    }
}

/// Export controls for the single-corpus version-2 layout.
#[derive(Clone, Copy, Debug)]
pub struct KdfWriteOptions {
    pub target_descriptor_block_bytes: usize,
    pub target_chunk_bytes: usize,
    pub compression_level: i32,
    pub origin_block_rows: usize,
}

impl Default for KdfWriteOptions {
    fn default() -> Self {
        Self {
            target_descriptor_block_bytes: 2 << 10,
            target_chunk_bytes: 1 << 20,
            compression_level: 3,
            origin_block_rows: 131_072,
        }
    }
}

/// Limits for lazy opening, decoding, caching, and query scratch.
#[derive(Clone, Debug)]
pub struct LazyKdForestOptions {
    pub max_address_map_bytes: usize,
    pub max_leaf_features: usize,
    pub cache_bytes: usize,
    pub max_in_flight_bytes: usize,
    /// Maximum compressed buffer held for any one entry read.
    pub max_compressed_bytes: usize,
    pub max_metadata_bytes: usize,
    pub max_chunk_bytes: usize,
    pub query_workers: usize,
}

impl Default for LazyKdForestOptions {
    fn default() -> Self {
        Self {
            max_address_map_bytes: 256 << 20,
            max_leaf_features: 1_048_576,
            cache_bytes: 256 << 20,
            max_in_flight_bytes: 64 << 20,
            max_compressed_bytes: 64 << 20,
            max_metadata_bytes: 64 << 20,
            max_chunk_bytes: 64 << 20,
            query_workers: 1,
        }
    }
}

/// One source-image feature associated with an original corpus feature ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FeatureOrigin {
    pub image_index: u32,
    pub image_feature_index: u32,
}

/// One SIFT feature's image-space center and affine footprint.
///
/// Rows are `[x, y]`, `[a11, a12]`, `[a21, a22]`. The wire representation is
/// therefore exactly one row-major `3 x 2` float32 array per corpus feature.
pub type FeatureGeometry = [[f32; 2]; 3];

/// Embedded workspace settings used only by explicit source verification.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KdfWorkspaceContents {
    pub feature_tool: String,
    pub feature_type: String,
    pub feature_options: serde_json::Value,
    pub feature_prefix_dir: String,
}

/// Workspace location recorded at export time.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KdfWorkspaceMetadata {
    pub absolute_path: String,
    pub relative_path: String,
    pub contents: KdfWorkspaceContents,
}

/// Optional image/SIFT provenance for every corpus feature.
#[derive(Clone, Debug)]
pub struct KdfSiftSources {
    pub workspace: KdfWorkspaceMetadata,
    pub image_names: Vec<String>,
    pub feature_tool_hashes: Vec<[u8; 16]>,
    pub sift_content_hashes: Vec<[u8; 16]>,
    pub origins: Vec<FeatureOrigin>,
    pub geometry: Vec<FeatureGeometry>,
}

/// Lazily available image table; it never opens a referenced SIFT file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KdfImageTable {
    pub names: Vec<String>,
    pub feature_tool_hashes: Vec<[u8; 16]>,
    pub sift_content_hashes: Vec<[u8; 16]>,
}

/// An arena node supplied to the format writer.
#[derive(Clone, Copy, Debug)]
pub enum KdfNode<S: KdfScalar> {
    Internal {
        split_dimension: u16,
        split: S,
        left: u32,
        right: u32,
    },
    Leaf {
        start: u32,
        len: u32,
    },
}

/// One tree topology and its leaf-order feature permutation.
#[derive(Clone, Debug)]
pub struct KdfTree<S: KdfScalar> {
    pub nodes: Vec<KdfNode<S>>,
    pub feature_ids: Vec<u32>,
}

/// Complete neutral input to the format writer.
#[derive(Clone, Debug)]
pub struct KdfForestData<'a, S: KdfScalar> {
    pub vectors: &'a [S],
    pub feature_count: usize,
    pub dimension: usize,
    pub trees: Vec<KdfTree<S>>,
    pub provenance: Option<serde_json::Value>,
    /// Order the corpus is stored in: row `r` holds feature
    /// `descriptor_order[r]`. Must be a permutation of `0..feature_count`.
    ///
    /// `None` means tree 0's leaf order, which makes tree 0's leaves contiguous
    /// and leaves every other tree scattered. Any permutation is valid — the
    /// stored row map is what a reader follows — so which one to choose is a
    /// writer policy question with no effect on the wire format.
    pub descriptor_order: Option<&'a [u32]>,
}

/// Disk address plus the logical ID needed by deterministic queue ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeAddress {
    pub chunk: u32,
    pub local: u32,
    pub logical: u32,
}

/// A validated decoded node.
#[derive(Clone, Copy, Debug)]
pub enum DecodedNode<S: KdfScalar> {
    Internal {
        split_dimension: u16,
        split: S,
        left: NodeAddress,
        right: NodeAddress,
    },
    Leaf {
        start: u32,
        len: u32,
    },
}

/// A validated decoded tree chunk.
#[derive(Clone, Debug)]
pub struct DecodedTreeChunk<S: KdfScalar> {
    pub logical_node_ids: Vec<u32>,
    pub nodes: Vec<DecodedNode<S>>,
    pub feature_ids: Vec<u32>,
    pub decoded_bytes: usize,
}

/// One leaf copied out of its tree-chunk pin. Descriptors are fetched from the
/// descriptor corpus by feature ID after that pin is released.
#[derive(Clone, Debug)]
pub struct DecodedLeaf {
    pub feature_ids: Vec<u32>,
}

/// Monotonic I/O/cache counters for one open handle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KdfIoStats {
    pub read_calls: u64,
    pub compressed_bytes: u64,
    pub decoded_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub duplicate_load_waits: u64,
    pub resident_bytes: usize,
    pub peak_resident_bytes: usize,
    pub in_flight_bytes: usize,
    pub peak_in_flight_bytes: usize,
    pub address_map_bytes: usize,
}

/// Successful full verification counts, useful to audit what was actually read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Verification {
    pub trees: usize,
    pub chunks: usize,
    pub descriptor_blocks: usize,
    pub geometry_blocks: usize,
    pub origin_blocks: usize,
    pub features: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Metadata {
    pub format: String,
    pub version: u32,
    pub scalar_type: String,
    pub metric: String,
    pub feature_count: u32,
    pub dimension: u16,
    pub node_kinds: Vec<String>,
    pub target_chunk_bytes: u64,
    pub trees: Vec<TreeMetadata>,
    pub feature_source: String,
    pub descriptor_block_rows: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin_block_rows: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace: Option<KdfWorkspaceMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct TreeMetadata {
    pub root: Option<[u32; 2]>,
    pub chunks: Vec<ChunkMetadata>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ChunkMetadata {
    pub node_count: u32,
    pub feature_count: u32,
    pub decoded_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ContentHash {
    pub metadata_xxh128: String,
    pub chunks_xxh128: Vec<Vec<String>>,
    pub content_xxh128: String,
    pub storage_rows_xxh128: String,
    pub descriptor_blocks_xxh128: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geometry_blocks_xxh128: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images_xxh128: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origins_xxh128: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ImagesMetadata {
    pub image_count: u32,
}

/// Node columns packed into every chunk's node array.
pub(crate) const NODE_COLUMNS: usize = 10;

/// Name of a tree chunk's grouped topology entry.
///
/// One entry carries a chunk's node columns, splits and feature IDs,
/// concatenated in that order. The counts are in the name so a reader still
/// knows the exact decoded length before decompressing.
///
/// Vectors are deliberately **not** in here. Version 2 stores exactly one
/// independently blocked corpus used by every tree.
pub(crate) fn chunk_entry_name<S: KdfScalar>(
    tree: usize,
    chunk: usize,
    nodes: usize,
    features: usize,
) -> String {
    format!(
        "trees/{tree}/chunks/{chunk}/chunk.{nodes}.{features}.{}.zst",
        S::TYPE_NAME
    )
}

/// Byte ranges of the three arrays inside a chunk's topology entry.
///
/// Returned as end offsets so a reader can split one decoded buffer without
/// repeating the arithmetic at each call site.
pub(crate) struct ChunkSpans {
    pub nodes_end: usize,
    pub splits_end: usize,
    pub total: usize,
}

pub(crate) fn chunk_spans<S: KdfScalar>(nodes: usize, features: usize) -> ChunkSpans {
    let scalar = std::mem::size_of::<S>();
    let nodes_end = NODE_COLUMNS * nodes * 4;
    let splits_end = nodes_end + nodes * scalar;
    ChunkSpans {
        nodes_end,
        splits_end,
        total: splits_end + features * 4,
    }
}

/// Name of the descriptor corpus container.
///
/// `.frames` rather than `.zst`: the entry holds one independent zstd frame per
/// descriptor block, not a single frame, so the extension says so instead of
/// implying a whole-entry decompression that would fail.
pub(crate) fn corpus_entry_name<S: KdfScalar>(features: usize, dimension: usize) -> String {
    format!(
        "features/corpus.{features}.{dimension}.{}.frames",
        S::TYPE_NAME
    )
}

/// Name of the array giving each corpus frame's start, plus a final end offset.
pub(crate) fn block_offsets_entry_name(count: usize) -> String {
    format!("features/block_offsets.{count}.uint64.zst")
}

/// Name of the optional SIFT geometry container.
pub(crate) fn geometry_entry_name(features: usize) -> String {
    format!("features/geometry.{features}.3.2.float32.frames")
}

/// Frame boundaries for the geometry container. Geometry uses the descriptor
/// row order and block row count, but its compressed frames have their own
/// lengths and therefore need an independent offset table.
pub(crate) fn geometry_block_offsets_entry_name(count: usize) -> String {
    format!("features/geometry_block_offsets.{count}.uint64.zst")
}
