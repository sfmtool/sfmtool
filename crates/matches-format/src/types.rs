// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Data types for the `.matches` file format.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use sfmtool_archive_io::ArchiveIoError;

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

/// Current `.matches` format version. [`crate::write_matches`] always writes
/// this version; [`crate::read_matches`] accepts any version up to it.
///
/// Version 6 gives the cluster backbone one place for its members' geometry:
/// `clusters/` gains a mandatory `member_positions.{K}.2.float32` and
/// `member_affine_shapes.{K}.2.2.float32` pair, and
/// `cluster_patches/member_affines.{K}.2.3.float64` is **removed**. There is
/// now exactly one keypoint position and one affine shape per member in a
/// cluster file, and its **content is the file's stage**: a matcher output
/// holds the detections, copied verbatim from the `.sift` rows
/// `member_features` names; a `sfm cluster-patches` output holds the
/// refinement, downcast to `f32` at the write boundary, for every member the
/// cascade measured, and the untouched detection for every member it never
/// fitted. Nothing is `NaN`:
/// [`ClusterPatchData::member_status`] is the sole authority on what a row
/// means and on exclusion, and the rejected-but-measured members keep their
/// measurement so a consumer can re-gate without re-running.
///
/// `cluster_patches/` therefore holds the vetting evidence and nothing else —
/// statuses, ZNCC, shift and the warp-consistency residual — while the
/// geometry it produced is read through
/// [`MatchesData::member_positions`] / [`MatchesData::member_affine_shapes`],
/// the same accessors a bare backbone answers.
///
/// The `f32` storage is deliberate: the refinement's fit precision is on the
/// order of 0.01 px, an `f32` position quantizes to 1.5e-5..6.1e-5 px on real
/// captures, and the structure-free focal vote cannot distinguish the
/// downcast from its own seed-to-seed spread. The kernel still computes in
/// `f64`; only the write boundary rounds.
///
/// **A cluster-backbone file below version 6 is refused on read.** Its
/// geometry lives only in the `.sift` files, which this crate does not open,
/// so there is nothing to upgrade from: regenerate the backbone with
/// `sfm match --cluster` (and re-run `sfm cluster-patches` if it was
/// enriched). Pairwise files are untouched by the bump and keep their own
/// version compatibility.
///
/// Version 5 gave `cluster_patches/member_affines` its absolute affine shape,
/// and version 4 gave it the absolute keypoint position and added the
/// mandatory `images/image_dims.{N}.2.uint32` array. Both of those affine
/// semantics are now history — version 6 removed the member they described,
/// and every cluster file below version 6 is refused. What survives for a
/// **pairwise** file is `image_dims`: version ≤ 3 pairwise files load with
/// [`MatchesData::image_dims`] as `None`.
///
/// Version 3 introduced the cluster backbone: a file stores exactly one of
/// the `image_pairs/` or `clusters/` sections as its correspondence backbone,
/// plus the optional `cluster_patches/` enrichment (requires `clusters/`).
/// Version ≤ 2 files always store the pairwise backbone and never have
/// clusters, so they load unchanged.
///
/// Version 2 made the canonical camera convention normative for the stored
/// two-view relative poses (`cam2_from_cam1` with cameras looking down −Z,
/// +Y up — see `specs/formats/matches-file-format.md` § "Coordinate
/// Conventions"). The bump is purely semantic: no member was added, removed,
/// or renamed. Version 1 files hold COLMAP-convention relative poses and are
/// upgraded on load by S-conjugation ([`s_conjugate_relative_pose`]); the
/// pixel-space F/E/H matrices are identical in both versions.
pub const MATCHES_FORMAT_VERSION: u32 = 6;

/// Conjugate a relative camera pose (`cam2_from_cam1`) with the camera-frame
/// flip `S = diag(1, −1, −1)`: `R' = S·R·S`, `t' = S·t`.
///
/// In quaternion terms, conjugating by the 180°-about-X rotation negates the
/// y and z components: `(w, x, y, z) → (w, x, −y, −z)`; the translation's y
/// and z flip likewise. Involutive, so the same function maps COLMAP ↔
/// canonical in both directions.
///
/// This is a local copy of the relative-pose case of
/// `sfmtool_core::geometry::convention::relative_pose_conjugate_s`
/// (the single source of truth for the convention math): `matches-format`
/// sits below `sfmtool-core` in the crate graph and cannot depend on it, and
/// the operation is an exact component permutation with no rotation-matrix
/// round trip, so duplicating it here is loss-free.
pub fn s_conjugate_relative_pose(quaternion_wxyz: &mut [f64; 4], translation_xyz: &mut [f64; 3]) {
    quaternion_wxyz[2] = -quaternion_wxyz[2];
    quaternion_wxyz[3] = -quaternion_wxyz[3];
    translation_xyz[1] = -translation_xyz[1];
    translation_xyz[2] = -translation_xyz[2];
}

/// Top-level metadata from `metadata.json.zst`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchesMetadata {
    /// Format version. The current version is [`MATCHES_FORMAT_VERSION`];
    /// version 1 files (COLMAP-convention relative poses) upgrade on load.
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
    pub matching_options: BTreeMap<String, serde_json::Value>,
    /// Workspace reference for relocatability.
    pub workspace: WorkspaceMetadata,
    /// ISO 8601 timestamp with timezone.
    pub timestamp: String,
    /// Number of images referenced.
    pub image_count: u32,
    /// Number of image pairs with matches. Present exactly when the file
    /// stores the pairwise backbone (`image_pairs/`); cluster-bearing files
    /// carry `cluster_count` / `cluster_member_count` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_pair_count: Option<u32>,
    /// Total number of matches across all pairs. Present exactly when the
    /// file stores the pairwise backbone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub match_count: Option<u32>,
    /// Number of clusters. Present exactly when the file stores the cluster
    /// backbone (`clusters/`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster_count: Option<u32>,
    /// Total number of cluster members. Present exactly when the file stores
    /// the cluster backbone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster_member_count: Option<u32>,
    /// Whether the optional two-view geometries section is present (pairwise
    /// backbone only).
    pub has_two_view_geometries: bool,
    /// Whether the file stores the cluster backbone (`clusters/`) instead of
    /// the pairwise backbone (`image_pairs/`). Absent in version ≤ 2 files
    /// (always pairwise), hence the serde default.
    #[serde(default)]
    pub has_clusters: bool,
    /// Whether the optional `cluster_patches/` enrichment section is present
    /// (requires `has_clusters`).
    #[serde(default)]
    pub has_cluster_patches: bool,
}

/// Content integrity hashes from `content_hash.json.zst`.
///
/// All hash values are plain 32-character lowercase hex strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchesContentHash {
    pub metadata_xxh128: String,
    pub images_xxh128: String,
    /// Present exactly when the file stores the pairwise backbone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_pairs_xxh128: Option<String>,
    /// Present exactly when the file stores the cluster backbone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clusters_xxh128: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster_patches_xxh128: Option<String>,
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
    pub verification_options: BTreeMap<String, serde_json::Value>,
}

/// Optional two-view geometry data.
#[derive(Debug)]
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
    /// `(P, 4)` relative rotation quaternions in WXYZ format
    /// (`cam2_from_cam1`, canonical camera convention — see
    /// [`MATCHES_FORMAT_VERSION`]).
    pub quaternions_wxyz: Array2<f64>,
    /// `(P, 3)` relative translation vectors (`cam2_from_cam1`, canonical
    /// camera convention).
    pub translations_xyz: Array2<f64>,
}

impl TwoViewGeometryData {
    /// S-conjugate every stored relative pose in place (COLMAP ↔ canonical;
    /// see [`s_conjugate_relative_pose`]). The pixel-space F/E/H matrices are
    /// convention-independent and are not touched.
    pub fn s_conjugate_poses(&mut self) {
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
            s_conjugate_relative_pose(&mut q, &mut t);
            for (k, &v) in q.iter().enumerate() {
                self.quaternions_wxyz[[i, k]] = v;
            }
            for (k, &v) in t.iter().enumerate() {
                self.translations_xyz[[i, k]] = v;
            }
        }
    }
}

/// The pairwise correspondence backbone (`image_pairs/` section).
#[derive(Debug)]
pub struct PairsData {
    /// `(P, 2)` image index pairs, `idx_i < idx_j`, sorted lexicographically.
    pub image_index_pairs: Array2<u32>,
    /// `(P,)` number of matches per pair. `sum == match_count`.
    pub match_counts: Array1<u32>,
    /// `(M, 2)` feature index pairs, flat concatenation across all image pairs.
    pub match_feature_indexes: Array2<u32>,
    /// `(M,)` L2 descriptor distance per match.
    pub match_descriptor_distances: Array1<f32>,
}

/// The cluster correspondence backbone (`clusters/` section): groups of SIFT
/// features across images that are likely co-observations of one surface
/// point, in CSR layout. Cluster `c` owns members
/// `cluster_starts[c]..cluster_starts[c+1]`.
#[derive(Debug)]
pub struct ClustersData {
    /// `(C+1,)` CSR offsets into the member arrays. `cluster_starts[0] == 0`,
    /// non-decreasing, final value equals the member count `M`.
    pub cluster_starts: Array1<u32>,
    /// `(M,)` index into `images/names.json.zst` per member.
    pub member_images: Array1<u32>,
    /// `(M,)` feature index in that image's `.sift` file per member.
    pub member_features: Array1<u32>,
    /// `(M, 2)` the member's keypoint position in source-image pixels **at
    /// this file's stage**: the detection, copied verbatim (same `f32` bits,
    /// no dtype round trip) from row `member_features[k]` of its image's
    /// `.sift` `features/positions_xy`, in a matcher output; the refined
    /// absolute position in a `sfm cluster-patches` output — for the members
    /// its cascade measured, with the detection left in place for those it
    /// never fitted. Never `NaN`: [`ClusterPatchData::member_status`] is what
    /// says which a row is, and which members stand.
    ///
    /// Mandatory since format version 6 — [`crate::write_matches`] requires
    /// it, and a cluster file below version 6 is refused on read — so `None`
    /// only for a pairwise file or a [`MatchesData`] built in memory. Present
    /// exactly when [`Self::member_affine_shapes`] is: the two are one
    /// member's geometry and are written and read together.
    pub member_positions: Option<Array2<f32>>,
    /// `(M, 2, 2)` the member's affine shape at the same stage as
    /// [`Self::member_positions`], under the same rules — the map from the
    /// detector's canonical unit frame onto the member's image pixels, so its
    /// column norms are the member's image-space extent. The detector's own
    /// shape (verbatim from `features/affine_shapes`) at the detection stage;
    /// the refined absolute shape `S = W·S_ref` where the cascade measured
    /// one, with the reference member's own row holding `S_ref`, so the
    /// reference→member warp is recoverable as `W = S·S_ref⁻¹`.
    pub member_affine_shapes: Option<Array3<f32>>,
    /// Matcher options recorded in `clusters/metadata.json.zst`
    /// (e.g. `d`, `alpha`, `min_size`, `preset`).
    pub matcher_options: serde_json::Value,
}

/// Sentinel in `ClusterPatchData::reference_members` for a cluster that
/// could not be refined (no usable reference member).
pub const CLUSTER_REFERENCE_UNREFINABLE: u32 = u32::MAX;

/// Per-member status in the `cluster_patches/` section
/// (`member_status` discriminants).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClusterMemberStatus {
    /// The cluster's reference member (identity affine, ZNCC 1.0).
    Reference = 0,
    /// Refined and vetted successfully.
    Kept = 1,
    /// Rejected: achieved ZNCC below the acceptance threshold.
    RejectedLowZncc = 2,
    /// Rejected: translation drifted too far from the SIFT seed.
    RejectedShift = 3,
    /// Outscored by another kept member in the same image, or shares the
    /// reference's image.
    DuplicateImage = 4,
    /// Not evaluated: degenerate shape, template/seed support out of frame,
    /// or the cluster itself was unrefinable.
    NotEvaluated = 5,
    /// Rejected: the member's own patch scored a keypoint position
    /// uncertainty above the localizability threshold (excluded before
    /// reference selection and refinement).
    RejectedUnlocalizable = 6,
}

impl ClusterMemberStatus {
    /// The canonical lowercase name used in metadata JSON (e.g. the
    /// cluster-selection provenance) and the spec's status listing.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Kept => "kept",
            Self::RejectedLowZncc => "rejected_low_zncc",
            Self::RejectedShift => "rejected_shift",
            Self::DuplicateImage => "duplicate_image",
            Self::NotEvaluated => "not_evaluated",
            Self::RejectedUnlocalizable => "rejected_unlocalizable",
        }
    }

    /// Decode a stored discriminant; `None` when out of range.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Reference),
            1 => Some(Self::Kept),
            2 => Some(Self::RejectedLowZncc),
            3 => Some(Self::RejectedShift),
            4 => Some(Self::DuplicateImage),
            5 => Some(Self::NotEvaluated),
            6 => Some(Self::RejectedUnlocalizable),
            _ => None,
        }
    }
}

impl FromStr for ClusterMemberStatus {
    type Err = MatchesError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "reference" => Ok(Self::Reference),
            "kept" => Ok(Self::Kept),
            "rejected_low_zncc" => Ok(Self::RejectedLowZncc),
            "rejected_shift" => Ok(Self::RejectedShift),
            "duplicate_image" => Ok(Self::DuplicateImage),
            "not_evaluated" => Ok(Self::NotEvaluated),
            "rejected_unlocalizable" => Ok(Self::RejectedUnlocalizable),
            _ => Err(MatchesError::InvalidFormat(format!(
                "Unknown ClusterMemberStatus: {s:?}"
            ))),
        }
    }
}

impl fmt::Display for ClusterMemberStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Optional cluster-patch enrichment (`cluster_patches/` section; requires
/// the cluster backbone). Arrays parallel the clusters' member arrays.
///
/// This section is the vetting evidence — which member is each cluster's
/// reference, what became of every other, and the signals behind those
/// verdicts. The geometry the refinement produced is not here: it goes into
/// the backbone's [`ClustersData::member_positions`] /
/// [`ClustersData::member_affine_shapes`], which a cluster file carries at
/// every stage, so there is one position and one shape per member in a file
/// and [`Self::member_status`] says what each one means.
#[derive(Debug)]
pub struct ClusterPatchData {
    /// `(C,)` global member index of each cluster's reference member;
    /// [`CLUSTER_REFERENCE_UNREFINABLE`] when the cluster could not be
    /// refined.
    pub reference_members: Array1<u32>,
    /// `(M,)` [`ClusterMemberStatus`] discriminants — the authority on what
    /// each of the backbone's geometry rows means: `Reference`, `Kept`,
    /// `RejectedLowZncc` and `RejectedShift` rows are the refinement's own
    /// measurement, and the rest are the detection it never displaced.
    pub member_status: Array1<u8>,
    /// `(M,)` achieved windowed ZNCC vs the reference (NaN where not
    /// evaluated).
    pub member_zncc: Array1<f32>,
    /// `(M,)` translation drift in pixels from the SIFT seed (NaN where not
    /// evaluated).
    pub member_shift_px: Array1<f32>,
    /// `(M,)` warp-consistency residual: the member's relative misfit
    /// against the jointly-fitted weak-perspective factorization of all
    /// cluster warps (`‖M_k·T_c − J‖_F / ‖J‖_F`; lower = more consistent, 0
    /// = perfect; NaN where the member did not participate). A signal, not
    /// a gate — consumers pick their own threshold, mirroring how
    /// `member_zncc` enables re-vetting. See
    /// `specs/core/patch/cluster-warp-consistency.md`.
    pub member_consistency_residual: Array1<f32>,
    /// Refinement options recorded in `cluster_patches/metadata.json.zst`.
    pub refine_options: serde_json::Value,
}

/// Primary data structure for `.matches` files.
///
/// Each field corresponds to a file in the archive. This is the primary
/// type for I/O. Exactly one of `image_pairs` / `clusters` is present (the
/// correspondence backbone); `cluster_patches` requires `clusters`, and
/// `two_view_geometries` requires `image_pairs`.
#[derive(Debug)]
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
    /// `(N, 2)` per-image pixel dimensions (width, height). Mandatory since
    /// format version 4 — [`crate::write_matches`] requires it — and `None`
    /// only for version ≤ 3 files loaded from disk, which never stored it.
    pub image_dims: Option<Array2<u32>>,

    /// Pairwise backbone. Exactly one of `image_pairs` / `clusters` is `Some`.
    pub image_pairs: Option<PairsData>,
    /// Cluster backbone. Exactly one of `image_pairs` / `clusters` is `Some`.
    pub clusters: Option<ClustersData>,
    /// Optional cluster-patch enrichment; requires `clusters`.
    pub cluster_patches: Option<ClusterPatchData>,
    /// Optional two-view geometries; requires `image_pairs`.
    pub two_view_geometries: Option<TwoViewGeometryData>,
}

/// Why a file states no single image resolution — see
/// [`MatchesData::shared_image_dims`].
///
/// Each variant is a property of the file, so a caller that needs one shared
/// camera can report which of the three situations it met without re-walking
/// `image_dims` itself.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum SharedDimsError {
    /// The file names no images, so there is nothing to read a resolution
    /// from.
    #[error("this .matches file names no images")]
    NoImages,

    /// The file records no `image_dims` at all (format version ≤ 3, which
    /// never stored them).
    #[error("this .matches file stores no image dimensions")]
    NoDimensions,

    /// Two images carry different dimensions.
    #[error("{image} is {}x{}, not the {}x{} of the first image", found.0, found.1, expected.0, expected.1)]
    Mixed {
        /// The first image's dimensions, which every image must share.
        expected: (u32, u32),
        /// The dimensions of the first image that disagrees.
        found: (u32, u32),
        /// That image's name (or `image {index}` when the name is missing).
        image: String,
    },
}

impl MatchesData {
    /// The one `(width, height)` every image of this file carries.
    ///
    /// The uniformity reading lives here, once, so that a caller estimating a
    /// single shared camera never has to take `image_dims[0]` and hope: a
    /// file whose images differ in resolution states no such pair, and says
    /// which image is the first to disagree ([`SharedDimsError`]).
    pub fn shared_image_dims(&self) -> Result<(u32, u32), SharedDimsError> {
        if self.image_names.is_empty() {
            return Err(SharedDimsError::NoImages);
        }
        let dims = self
            .image_dims
            .as_ref()
            .ok_or(SharedDimsError::NoDimensions)?;
        let mut rows = dims.rows().into_iter();
        let first = rows.next().ok_or(SharedDimsError::NoImages)?;
        let expected = (first[0], first[1]);
        for (i, row) in rows.enumerate() {
            let found = (row[0], row[1]);
            if found != expected {
                return Err(SharedDimsError::Mixed {
                    expected,
                    found,
                    // The dimension rows are parallel to the image table; `i`
                    // counts from the second row, which is image 1.
                    image: self
                        .image_names
                        .get(i + 1)
                        .cloned()
                        .unwrap_or_else(|| format!("image {}", i + 1)),
                });
            }
        }
        Ok(expected)
    }
}
