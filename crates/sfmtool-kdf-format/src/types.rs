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
    /// Filesystem or stream I/O failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The ZIP container could not be read or written.
    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),
    /// JSON metadata could not be encoded or decoded.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// The file or writer input violates a KDF structural or semantic rule.
    #[error("invalid KDF format: {0}")]
    InvalidFormat(String),
    /// An array or buffer has a different shape from the declared shape.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    /// A stored or source-content digest does not match the data.
    #[error("integrity mismatch: {0}")]
    Integrity(String),
    /// A requested allocation or declared file section exceeds a configured limit.
    #[error("resource limit exceeded: {0}")]
    ResourceLimit(String),
    /// The caller opened a file using a different scalar type from the stored corpus.
    #[error("scalar type mismatch: file contains {file}, caller requested {requested}")]
    ScalarType {
        /// Scalar type named by the file metadata.
        file: String,
        /// Scalar type selected through the [`KdfFile`](crate::KdfFile) type parameter.
        requested: &'static str,
    },
    /// A query supplied an invalid feature ID, tree address, or output shape.
    #[error("invalid query: {0}")]
    InvalidQuery(String),
    /// A SIFT file required by explicit source verification does not exist.
    #[error("source file is missing: {0}")]
    MissingSource(PathBuf),
    /// The caller asked a write to stop, and it did, so nothing was written.
    ///
    /// Carried as a variant with a `From` so that `progress.check_cancel()?`
    /// propagates out of a write by the same mechanism as every other failure.
    #[error("{0}")]
    Cancelled(#[from] sfmtool_progress::Cancelled),
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
    /// Target decoded size in bytes of one descriptor block.
    ///
    /// A block contains at least one complete descriptor row, so this target may
    /// be exceeded when one row is larger. Must be positive; defaults to 2 KiB.
    pub target_descriptor_block_bytes: usize,
    /// Target decoded size in bytes of one packed tree chunk.
    ///
    /// An indivisible subtree may exceed the target. Must be positive; defaults
    /// to 1 MiB. See the [format specification](../../../specs/formats/kdf-file-format.md#chunk-independence-and-integrity).
    pub target_chunk_bytes: usize,
    /// Zstandard compression level used for every compressed frame.
    ///
    /// This affects stored size and write time, but not the decoded content hash.
    /// The default is 3.
    pub compression_level: i32,
    /// Number of feature-origin rows in each origin block.
    ///
    /// Must be in `1..=u32::MAX`; defaults to 131,072 rows. This has no effect
    /// for a generic corpus without [`KdfSiftSources`].
    pub origin_block_rows: usize,
    /// Whether a file already at the destination is replaced. Defaults to
    /// `false`, which refuses one.
    ///
    /// Replacing costs nothing extra and loses nothing: the archive is streamed
    /// into a temporary sibling either way and renamed over the destination only
    /// once it is whole, so the file that is there stays exactly as it was until
    /// that instant and survives a write that fails or is cancelled. What the
    /// default buys is the caller who did not mean to write over a corpus at
    /// all, which is why rebuilding an index in place asks for it by name.
    pub replace_existing: bool,
}

impl Default for KdfWriteOptions {
    fn default() -> Self {
        Self {
            target_descriptor_block_bytes: 2 << 10,
            target_chunk_bytes: 1 << 20,
            compression_level: 3,
            origin_block_rows: 131_072,
            replace_existing: false,
        }
    }
}

/// Limits for lazy opening, decoding, caching, and query scratch.
#[derive(Clone, Debug)]
pub struct LazyKdForestOptions {
    /// Maximum bytes used by the eagerly loaded feature-ID-to-storage-row map.
    /// Defaults to 256 MiB.
    pub max_address_map_bytes: usize,
    /// Maximum number of feature IDs that a leaf accessor or query may copy.
    /// Defaults to 1,048,576 features.
    pub max_leaf_features: usize,
    /// Total decoded-byte budget for cached tree, descriptor, geometry, and origin blocks.
    ///
    /// Must be positive and large enough for the largest declared cache item;
    /// defaults to 256 MiB.
    pub cache_bytes: usize,
    /// Maximum decoded bytes being loaded concurrently across cache misses.
    ///
    /// Must be positive and large enough for the largest declared cache item;
    /// defaults to 64 MiB.
    pub max_in_flight_bytes: usize,
    /// Maximum compressed bytes buffered for any one entry or frame read.
    ///
    /// Must be positive; defaults to 64 MiB. Open-time metadata reads apply both
    /// this compressed-byte budget and the decoded
    /// [`max_metadata_bytes`](Self::max_metadata_bytes) budget, except the
    /// integrity directory may use the metadata budget for its compressed frame.
    pub max_compressed_bytes: usize,
    /// Maximum decoded metadata bytes and approximate ZIP-directory bookkeeping bytes.
    ///
    /// Must be positive; defaults to 64 MiB.
    pub max_metadata_bytes: usize,
    /// Maximum decoded size in bytes of any single cache item declared by the file.
    ///
    /// This covers tree chunks and descriptor, geometry, and origin blocks. Must
    /// be positive; defaults to 64 MiB.
    pub max_chunk_bytes: usize,
    /// Number of independent archive readers, and the query worker count used by the core lazy forest.
    ///
    /// Must be positive; defaults to one worker.
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
    /// Zero-based row in [`KdfSiftSources::image_names`].
    pub image_index: u32,
    /// Zero-based feature row in the referenced image's SIFT file.
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
    /// Feature extractor identifier recorded by the source workspace.
    pub feature_tool: String,
    /// Feature representation identifier recorded by the source workspace.
    pub feature_type: String,
    /// Uninterpreted feature extractor settings from the source workspace.
    pub feature_options: serde_json::Value,
    /// Workspace-relative POSIX directory containing per-image SIFT files.
    pub feature_prefix_dir: String,
}

/// Workspace location recorded at export time.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KdfWorkspaceMetadata {
    /// Absolute workspace path at export time, used as a fallback during source verification.
    pub absolute_path: String,
    /// POSIX path from the KDF parent to the workspace, tried before the absolute path.
    pub relative_path: String,
    /// Embedded feature settings needed to locate and interpret source SIFT files.
    pub contents: KdfWorkspaceContents,
}

/// Optional image/SIFT provenance for every corpus feature.
#[derive(Clone, Debug)]
pub struct KdfSiftSources {
    /// Source workspace location and feature settings.
    pub workspace: KdfWorkspaceMetadata,
    /// Unique workspace-relative POSIX image paths, indexed by [`FeatureOrigin::image_index`].
    pub image_names: Vec<String>,
    /// Per-image SIFT feature-tool XXH128 digests, stored as 16 little-endian bytes.
    pub feature_tool_hashes: Vec<[u8; 16]>,
    /// Per-image SIFT content XXH128 digests, stored as 16 little-endian bytes.
    pub sift_content_hashes: Vec<[u8; 16]>,
    /// One source mapping per corpus feature, in original feature-ID order.
    pub origins: Vec<FeatureOrigin>,
    /// One image-space geometry row per corpus feature, in original feature-ID order.
    pub geometry: Vec<FeatureGeometry>,
}

/// Lazily available image table; it never opens a referenced SIFT file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KdfImageTable {
    /// Unique workspace-relative POSIX image paths.
    pub names: Vec<String>,
    /// Per-image SIFT feature-tool XXH128 digests, as 16 little-endian bytes.
    pub feature_tool_hashes: Vec<[u8; 16]>,
    /// Per-image SIFT content XXH128 digests, as 16 little-endian bytes.
    pub sift_content_hashes: Vec<[u8; 16]>,
}

/// An arena node supplied to the format writer.
#[derive(Clone, Copy, Debug)]
pub enum KdfNode<S: KdfScalar> {
    /// Binary partition with arena-index children.
    Internal {
        /// Zero-based vector coordinate used for the split.
        split_dimension: u16,
        /// Coordinate dividing the left (`<=`) and right (`>=`) subtrees.
        split: S,
        /// Arena index of the left child.
        left: u32,
        /// Arena index of the right child.
        right: u32,
    },
    /// Contiguous range in the tree's leaf-order feature permutation.
    Leaf {
        /// Start row in [`KdfTree::feature_ids`].
        start: u32,
        /// Number of feature IDs owned by this leaf; must be positive.
        len: u32,
    },
}

/// One tree topology and its leaf-order feature permutation.
#[derive(Clone, Debug)]
pub struct KdfTree<S: KdfScalar> {
    /// Arena nodes with the root at index zero.
    pub nodes: Vec<KdfNode<S>>,
    /// Feature IDs addressed by leaf `start` and `len` ranges.
    pub feature_ids: Vec<u32>,
}

/// Complete neutral input to the format writer.
#[derive(Clone, Debug)]
pub struct KdfForestData<'a, S: KdfScalar> {
    /// Row-major `feature_count * dimension` descriptor scalars.
    pub vectors: &'a [S],
    /// Number of descriptor rows and valid feature IDs.
    pub feature_count: usize,
    /// Scalars per descriptor row, in `1..=u16::MAX`.
    pub dimension: usize,
    /// Nonempty collection of trees, each containing every feature ID exactly once.
    pub trees: Vec<KdfTree<S>>,
    /// Uninterpreted producer information embedded in metadata.
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
    /// Zero-based chunk index within a tree.
    pub chunk: u32,
    /// Zero-based node index within the chunk.
    pub local: u32,
    /// Stable tree-wide node ID used for deterministic traversal ties.
    pub logical: u32,
}

/// A validated decoded node.
#[derive(Clone, Copy, Debug)]
pub enum DecodedNode<S: KdfScalar> {
    /// Binary partition with validated on-disk child addresses.
    Internal {
        /// Zero-based vector coordinate used for the split.
        split_dimension: u16,
        /// Coordinate dividing the left (`<=`) and right (`>=`) subtrees.
        split: S,
        /// Validated address of the left child.
        left: NodeAddress,
        /// Validated address of the right child.
        right: NodeAddress,
    },
    /// Contiguous range in its decoded chunk's feature-ID array.
    Leaf {
        /// Start row in the decoded chunk's feature-ID array.
        start: u32,
        /// Number of feature IDs owned by the leaf.
        len: u32,
    },
}

/// A validated decoded tree chunk.
#[derive(Clone, Debug)]
pub struct DecodedTreeChunk<S: KdfScalar> {
    /// Tree-wide logical ID for each node in local-node order.
    pub logical_node_ids: Vec<u32>,
    /// Validated nodes in local-node order.
    pub nodes: Vec<DecodedNode<S>>,
    /// Original feature IDs owned by the chunk's leaves.
    pub feature_ids: Vec<u32>,
    /// Decoded cache charge in bytes for this chunk's stored arrays.
    pub decoded_bytes: usize,
}

/// One leaf copied out of its tree-chunk pin. Descriptors are fetched from the
/// descriptor corpus by feature ID after that pin is released.
#[derive(Clone, Debug)]
pub struct DecodedLeaf {
    /// Original feature IDs copied from the leaf's chunk-local range.
    pub feature_ids: Vec<u32>,
}

/// I/O/cache counters and live byte gauges for one open handle.
///
/// Counters accumulate until [`KdfFile::reset_io_stats`](crate::KdfFile::reset_io_stats);
/// live gauges survive a reset and peak gauges restart at their current values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KdfIoStats {
    /// Successful cache-miss loads since open or the last reset.
    pub read_calls: u64,
    /// Compressed bytes read by those successful loads; open-time metadata is excluded.
    pub compressed_bytes: u64,
    /// Decoded bytes produced by those successful loads.
    pub decoded_bytes: u64,
    /// Cache lookups satisfied by a resident entry.
    pub cache_hits: u64,
    /// Cache lookups that became the loader for a nonresident entry.
    pub cache_misses: u64,
    /// Resident entries removed to admit another decoded item.
    pub evictions: u64,
    /// Wait episodes caused by another thread loading the same key.
    ///
    /// One request can contribute more than once if it wakes and waits again.
    pub duplicate_load_waits: u64,
    /// Decoded bytes currently resident across all cache shards.
    pub resident_bytes: usize,
    /// Sum of each shard's resident-byte high-water mark since open or reset.
    ///
    /// Shards may peak at different times, so this is a budget-oriented bound,
    /// not necessarily a simultaneous global maximum.
    pub peak_resident_bytes: usize,
    /// Decoded bytes currently reserved by cache-miss loads in progress.
    pub in_flight_bytes: usize,
    /// High-water mark of concurrent decoded-byte reservations since open or reset.
    pub peak_in_flight_bytes: usize,
    /// Bytes occupied by the eagerly loaded feature-ID-to-storage-row map.
    pub address_map_bytes: usize,
}

/// Successful full verification counts, useful to audit what was actually read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Verification {
    /// Trees traversed or identified by the verifier.
    pub trees: usize,
    /// Distinct tree chunks read; source-only verification reports zero.
    pub chunks: usize,
    /// Descriptor blocks covered by verification.
    pub descriptor_blocks: usize,
    /// Geometry blocks covered by verification, or zero for a generic corpus.
    pub geometry_blocks: usize,
    /// Feature-origin blocks covered by verification, or zero for a generic corpus.
    pub origin_blocks: usize,
    /// Corpus features covered by verification.
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
