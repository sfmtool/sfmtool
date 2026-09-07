// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Data types for the `.sfmr` file format.

use std::path::PathBuf;

use ndarray::{Array1, Array2, Array4};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

use sfmtool_archive_io::ArchiveIoError;

/// Edge length, in pixels, of the square RGB thumbnails a `.sfmr` carries, one
/// per image.
///
/// The format pins this: it is written into the archive entry's *name*
/// (`images/thumbnails_y_x_rgb.<image_count>.<size>.<size>.3.uint8.zst`) and into the
/// `images` section metadata as `thumbnail_size`, so a reader locates the entry
/// by a string that embeds the size. Changing it changes the on-disk format and
/// makes existing files unreadable — this constant exists so that the several
/// places which must agree cannot drift apart, not because the value is
/// adjustable.
///
/// These thumbnails are copied verbatim out of the per-image `.sift` files, so
/// this must equal `sift_format::THUMBNAIL_SIZE`. The two crates are
/// independent — neither depends on the other — so the agreement is enforced by
/// a compile-time assertion in `sfmtool-core`, the first crate that sees both.
pub const THUMBNAIL_SIZE: usize = 128;

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
    pub parameters: BTreeMap<String, f64>,
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
    pub tool_options: BTreeMap<String, serde_json::Value>,
    pub workspace: WorkspaceMetadata,
    pub timestamp: String,
    pub image_count: u32,
    /// Number of points (finite and at infinity combined).
    ///
    /// The `points3d_count` alias accepts the version 1 field name on read.
    #[serde(alias = "points3d_count")]
    pub point_count: u32,
    /// Number of points at infinity (rows of `positions_xyzw` with `w = 0`).
    ///
    /// Absent in version 1 files (which have no infinity points); `serde(default)`
    /// supplies `0` in that case.
    #[serde(default)]
    pub infinity_point_count: u32,
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
    /// Physical unit of 3D world-space coordinates (point positions and camera
    /// translations). One of `"mm"`, `"cm"`, `"m"`, `"in"`, `"ft"`. Absent when
    /// the reconstruction is in arbitrary (unscaled) units — the default after
    /// an SfM solve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_space_unit: Option<String>,
    /// Observation source (format version 4+):
    /// [`FEATURE_SOURCE_SIFT_FILES`] — observations reference external `.sift`
    /// files via `feature_indexes`, optionally alongside an inline copy of the
    /// coordinate in `tracks/keypoints_xy`; or [`FEATURE_SOURCE_EMBEDDED_PATCHES`]
    /// — per-observation keypoints are stored inline in `tracks/keypoints_xy`
    /// and there is no `.sift` link at all.
    /// Legacy version 1–3 files have no key and read as `sift_files`.
    #[serde(default = "default_feature_source")]
    pub feature_source: String,
}

/// Validate per-observation keypoints: every `(u, v)` must be finite and lie
/// within `[0, width) × [0, height)` of the image's camera intrinsics. Returns a
/// descriptive message on the first violation. Used on both read and verify,
/// wherever the column is present. Index arrays are bounds-checked so malformed
/// input yields an error rather than a panic.
pub fn validate_keypoints(
    keypoints: &Array2<f32>,
    image_indexes: &[u32],
    camera_indexes: &[u32],
    cameras: &[SfmrCamera],
) -> Result<(), String> {
    if keypoints.nrows() != image_indexes.len() {
        return Err(format!(
            "keypoints_xy rows {} != observation count {}",
            keypoints.nrows(),
            image_indexes.len()
        ));
    }
    for j in 0..keypoints.nrows() {
        let (u, v) = (keypoints[[j, 0]], keypoints[[j, 1]]);
        if !u.is_finite() || !v.is_finite() {
            return Err(format!("keypoints_xy row {j} is not finite: ({u}, {v})"));
        }
        let img = image_indexes[j] as usize;
        let cam = *camera_indexes
            .get(img)
            .ok_or_else(|| format!("keypoints_xy row {j}: image index {img} out of range"))?
            as usize;
        let c = cameras
            .get(cam)
            .ok_or_else(|| format!("keypoints_xy row {j}: camera index {cam} out of range"))?;
        if !(u >= 0.0 && u < c.width as f32 && v >= 0.0 && v < c.height as f32) {
            return Err(format!(
                "keypoints_xy row {j} = ({u}, {v}) is outside image bounds \
                 [0, {}) x [0, {})",
                c.width, c.height
            ));
        }
    }
    Ok(())
}

/// What a solve owns of one point -- the meaning behind a
/// `points3d/point_constraints` code.
///
/// The stored column is numeric, and a number is only as self-describing as the
/// legend beside it: `points3d/metadata.json` carries `point_constraint_names`,
/// and a stored code is an index into *that* list rather than a number the file
/// format fixes. The legend is a file-level concern, though: a reader resolves
/// every code through it and hands back the **canonical** numbering this enum
/// defines ([`POINT_CONSTRAINT_FREE`], [`POINT_CONSTRAINT_RANGED`],
/// [`POINT_CONSTRAINT_HELD`]), and a writer states that same canonical legend.
/// So in memory there is exactly one numbering, and a consumer of
/// [`SfmrData::point_constraints`] reads it with the constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PointConstraint {
    /// The solve owns the point outright.
    Free = 0,
    /// The caller owns the point's distance from a reference image and the
    /// solve owns its direction.
    Ranged = 1,
    /// The caller owns the point's coordinate outright.
    Held = 2,
}

impl PointConstraint {
    /// Every constraint this format defines, in canonical order: a constraint's
    /// position here is its [`Self::code`], and this is the legend a writer
    /// states and a reader normalises onto.
    pub const ALL: [PointConstraint; 3] = [Self::Free, Self::Ranged, Self::Held];

    /// The canonical legend, one name per entry of [`Self::ALL`].
    pub const NAMES: [&'static str; 3] = ["free", "ranged", "held"];

    /// The canonical code for this constraint: its index in [`Self::ALL`].
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// The legend name for this constraint.
    pub const fn name(self) -> &'static str {
        Self::NAMES[self.code() as usize]
    }

    /// The constraint a legend name denotes, or `None` for a name this format
    /// does not define.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == name)
    }
}

/// In-memory [`SfmrData::point_constraints`] code for [`PointConstraint::Free`].
pub const POINT_CONSTRAINT_FREE: u8 = PointConstraint::Free.code();
/// In-memory [`SfmrData::point_constraints`] code for [`PointConstraint::Ranged`].
pub const POINT_CONSTRAINT_RANGED: u8 = PointConstraint::Ranged.code();
/// In-memory [`SfmrData::point_constraints`] code for [`PointConstraint::Held`].
pub const POINT_CONSTRAINT_HELD: u8 = PointConstraint::Held.code();
/// `points3d/constraint_reference_images` value for a row that names no image:
/// every free and held point, and a ranged point at an infinite distance, which
/// needs no reference.
pub const NO_REFERENCE_IMAGE: u32 = u32::MAX;

/// The constraint each legend name denotes, in the order given -- the legend a
/// stored code indexes.
///
/// A legend has to name at least one constraint, name only constraints this
/// format defines, and name each of them once: a repeat would give one
/// constraint two codes and leave the file saying which of them a row means
/// only by accident. Read and verify both resolve their codes through a legend
/// this accepted, so a file states its own numbering and a reader normalises it
/// away rather than assuming one.
pub(crate) fn parse_point_constraint_names<S: AsRef<str>>(
    names: &[S],
) -> Result<Vec<PointConstraint>, String> {
    if names.is_empty() {
        return Err("point_constraint_names is empty, so it names no constraint at all".into());
    }
    let mut legend = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        let name = name.as_ref();
        let constraint = PointConstraint::from_name(name).ok_or_else(|| {
            format!(
                "point_constraint_names[{i}] is {name:?}, not one of {:?}",
                PointConstraint::NAMES
            )
        })?;
        if legend.contains(&constraint) {
            return Err(format!(
                "point_constraint_names[{i}] repeats {name:?}, which already has a code"
            ));
        }
        legend.push(constraint);
    }
    Ok(legend)
}

/// The `point_constraint_names` legend of a `points3d/metadata.json` that flags
/// the constraint columns present.
///
/// The legend is part of that metadata entry, which is inside the `points3d`
/// section hash, so it is covered by the same integrity envelope as the column
/// it describes.
pub(crate) fn read_point_constraint_legend(
    points3d_meta: &serde_json::Value,
) -> Result<Vec<PointConstraint>, String> {
    let value = points3d_meta.get("point_constraint_names").ok_or_else(|| {
        "points3d/metadata.json says has_point_constraints but carries no \
         point_constraint_names to read the codes through"
            .to_string()
    })?;
    let names: Vec<String> = value
        .as_array()
        .and_then(|items| {
            items
                .iter()
                .map(|v| Some(v.as_str()?.to_string()))
                .collect::<Option<Vec<String>>>()
        })
        .ok_or_else(|| {
            format!(
                "points3d/metadata.json's point_constraint_names is {value}, not a list of names"
            )
        })?;
    parse_point_constraint_names(&names)
}

/// The constraint each code names, resolved through `legend`.
///
/// A code past the legend's end is the one thing a legend cannot explain, so it
/// is rejected here rather than guessed at. The reader passes the file's legend
/// and so normalises the column onto [`PointConstraint::ALL`]; the writer passes
/// [`PointConstraint::ALL`] itself, which is the same call refusing a code
/// outside the canonical numbering it is about to state.
pub(crate) fn resolve_point_constraints(
    codes: &[u8],
    legend: &[PointConstraint],
) -> Result<Vec<PointConstraint>, String> {
    codes
        .iter()
        .enumerate()
        .map(|(p, &code)| {
            legend.get(code as usize).copied().ok_or_else(|| {
                format!(
                    "points3d/point_constraints row {p} is {code}, past the {} names its \
                     legend gives",
                    legend.len()
                )
            })
        })
        .collect()
}

/// Validate the per-point constraint triple against the positions it annotates.
///
/// The three columns are present together or absent together, so this takes the
/// three `Option`s and reports the first violation as a message the caller wraps
/// in its own error type. Read, write and verify all route through it, so the
/// rules are stated once.
///
/// The constraints arrive resolved through the file's legend
/// ([`resolve_point_constraints`]), so what a stored code meant is settled
/// before any of these rules is applied.
///
/// With all three absent every point is free and there is nothing to check.
pub(crate) fn validate_point_constraints(
    point_constraints: Option<&[PointConstraint]>,
    constraint_distances: Option<&[f64]>,
    constraint_reference_images: Option<&[u32]>,
    positions_xyzw: &Array2<f64>,
    point_count: usize,
    image_count: usize,
) -> Result<(), String> {
    let present = [
        point_constraints.is_some(),
        constraint_distances.is_some(),
        constraint_reference_images.is_some(),
    ];
    if present.iter().all(|&p| !p) {
        return Ok(());
    }
    if !present.iter().all(|&p| p) {
        return Err(format!(
            "points3d/point_constraints, points3d/constraint_distances and \
             points3d/constraint_reference_images are present together or absent \
             together (point_constraints={}, constraint_distances={}, \
             constraint_reference_images={})",
            present[0], present[1], present[2]
        ));
    }
    let (point_constraints, constraint_distances, constraint_reference_images) = (
        point_constraints.unwrap(),
        constraint_distances.unwrap(),
        constraint_reference_images.unwrap(),
    );
    for (name, len) in [
        ("point_constraints", point_constraints.len()),
        ("constraint_distances", constraint_distances.len()),
        (
            "constraint_reference_images",
            constraint_reference_images.len(),
        ),
    ] {
        if len != point_count {
            return Err(format!(
                "points3d/{name} len {len} != point_count {point_count}"
            ));
        }
    }
    let has_w = positions_xyzw.nrows() == point_count && positions_xyzw.ncols() == 4;
    for p in 0..point_count {
        let (k, r, c) = (
            point_constraints[p],
            constraint_distances[p],
            constraint_reference_images[p],
        );
        let name = k.name();
        match k {
            PointConstraint::Free | PointConstraint::Held => {
                if !r.is_nan() {
                    return Err(format!(
                        "points3d/constraint_distances row {p} is {r} on a {name} \
                         point, which carries no distance (expected NaN)"
                    ));
                }
                if c != NO_REFERENCE_IMAGE {
                    return Err(format!(
                        "points3d/constraint_reference_images row {p} names image {c} on a \
                         {name} point, which is measured from nothing"
                    ));
                }
            }
            PointConstraint::Ranged => {
                if r.is_nan() || r <= 0.0 {
                    return Err(format!(
                        "points3d/constraint_distances row {p} is {r} on a ranged point, \
                         which needs a strictly positive distance (+inf for a direction)"
                    ));
                }
                if r.is_finite() {
                    if c as usize >= image_count {
                        return Err(format!(
                            "points3d/constraint_reference_images row {p} = {c} is past the \
                             {image_count} images, and a finite distance is measured from \
                             one of them"
                        ));
                    }
                } else if c != NO_REFERENCE_IMAGE {
                    return Err(format!(
                        "points3d/constraint_reference_images row {p} names image {c} on a \
                         point at an infinite distance, which is measured from nothing"
                    ));
                }
                if has_w {
                    let w_is_zero = positions_xyzw[[p, 3]] == 0.0;
                    if w_is_zero != r.is_infinite() {
                        return Err(format!(
                            "positions_xyzw row {p} has w = {} but its distance is {r}: a \
                             ranged point is a direction exactly at an infinite distance",
                            positions_xyzw[[p, 3]]
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Current `.sfmr` format version. [`crate::write_sfmr`] always writes this
/// version; [`crate::read_sfmr`] accepts any version up to it.
///
/// Version 7 added the optional per-point constraint triple
/// `points3d/point_constraints`, `points3d/constraint_distances` and
/// `points3d/constraint_reference_images`, flagged together by
/// `points3d/metadata.json`'s `has_point_constraints` and read through that same
/// entry's `point_constraint_names` legend (see [`SfmrData::point_constraints`]
/// and [`PointConstraint`]). Older files carry neither the flag nor the arrays
/// and read as `None`, which is every point free.
///
/// Version 6 added the optional per-observation array
/// `tracks/observation_confidence`, flagged by `tracks/metadata.json`'s
/// `has_observation_confidence` (see [`SfmrData::observation_confidence`]).
/// Older files carry neither the flag nor the array and read as `None`.
///
/// Version 5 made the canonical coordinate convention normative (right-handed
/// Z-up world, cameras looking down −Z with +Y up — see
/// `specs/formats/sfmr-file-format.md` § "Coordinate System Conventions").
/// The bump is purely semantic: no array was added, removed, or renamed.
/// Version ≤ 4 files hold COLMAP-convention poses and world data; this crate
/// reads them structurally and **preserves the stored version number** so the
/// consumer can apply the COLMAP→canonical conversion — that conversion lives
/// in `sfmtool-core` (`SfmrReconstruction::load`), which owns the `S`/`W`
/// convention math (`geometry::convention`) that this lower-level crate
/// cannot depend on.
pub const SFMR_FORMAT_VERSION: u32 = 7;

/// The first `.sfmr` version whose stored poses and world data are in the
/// canonical convention (right-handed Z-up world, cameras looking down −Z).
///
/// A file stored **below** this version holds COLMAP-convention content and
/// must be converted on load; a file at or above it is already canonical and
/// must be loaded untouched. This is a fixed fact about the format's history,
/// so it is a constant of its own and never [`SFMR_FORMAT_VERSION`]: gating the
/// conversion on the *current* version re-applies it to every already-canonical
/// file the moment the format version is bumped for an unrelated reason (it
/// was, in version 6, and every version-5 file then loaded with its cameras
/// flipped by `S` — see `SfmrReconstruction::load` in `sfmtool-core`, the one
/// consumer of this constant).
pub const SFMR_CANONICAL_CONVENTION_VERSION: u32 = 5;

/// `feature_source` value: observations reference external `.sift` files.
pub const FEATURE_SOURCE_SIFT_FILES: &str = "sift_files";
/// `feature_source` value: per-observation keypoints stored inline in the
/// `.sfmr`, with no `.sift` companion.
pub const FEATURE_SOURCE_EMBEDDED_PATCHES: &str = "embedded_patches";

fn default_feature_source() -> String {
    FEATURE_SOURCE_SIFT_FILES.to_string()
}

/// Content integrity hashes from `content_hash.json.zst`.
///
/// All hash values are plain 32-character lowercase hex strings.
/// The `rigs_xxh128` and `frames_xxh128` fields are only present when the
/// `.sfmr` file contains the corresponding optional section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentHash {
    pub metadata_xxh128: String,
    pub cameras_xxh128: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rigs_xxh128: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_xxh128: Option<String>,
    pub images_xxh128: String,
    /// Hash of the `points3d/` section. Covers the optional per-point patch
    /// frame arrays (`patch_u_halfvec_xyz`, `patch_v_halfvec_xyz`, `patch_bitmaps_y_x_rgba`) when
    /// present.
    pub points3d_xxh128: String,
    pub tracks_xxh128: String,
    pub content_xxh128: String,
}

/// Statistics for observed points in a single image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedDepthStats {
    /// Number of observed finite points with positive depth.
    pub count: u32,
    /// Number of observed points at infinity (`w == 0`). Absent in v1 files.
    #[serde(default)]
    pub infinity_count: u32,
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
    /// `Some` in a `sift_files` file; `None` in an `embedded_patches` file.
    pub feature_tool_hashes: Option<Vec<[u8; 16]>>,
    /// `N` x 16-byte XXH128 hashes of `.sift` file contents.
    /// `Some` in a `sift_files` file; `None` in an `embedded_patches` file.
    pub sift_content_hashes: Option<Vec<[u8; 16]>>,
    /// `N` x 16-byte XXH128 hashes of the source image file bytes (the same
    /// value the `.sift` records as `image_file_xxh128`). `Some` in an
    /// `embedded_patches` file (the direct image-identity link that substitutes
    /// for the `.sift`-mediated one); `None` in a `sift_files` file.
    pub image_file_hashes: Option<Vec<[u8; 16]>>,
    /// `(N, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3)` RGB thumbnails of the source
    /// images (see [`THUMBNAIL_SIZE`]).
    pub thumbnails_y_x_rgb: Array4<u8>,

    // Points3D
    /// `(P, 4)` homogeneous 3D point positions in world coordinates.
    ///
    /// Each row is `[x, y, z, w]`: `w != 0` is a finite point at
    /// `(x/w, y/w, z/w)`; `w == 0` is a point at infinity whose direction is
    /// `(x, y, z)`.
    pub positions_xyzw: Array2<f64>,
    /// `(P, 3)` RGB colors (0-255).
    pub colors_rgb: Array2<u8>,
    /// `(P,)` RMS reprojection errors in pixels.
    pub reprojection_errors: Array1<f32>,
    /// Optional `(P, 3)` surface normals (unit vectors; the default mean-viewing
    /// estimate leaves `(0, 0, 0)` rows for `w == 0` points). `None` when the
    /// reconstruction carries no normals at all.
    ///
    /// On disk this is `points3d/normals_xyz` in format version 3+ (present only
    /// when `points3d/metadata.json`'s `has_normals` is `true`); version 1 and 2
    /// files always store it under the legacy name `estimated_normals_xyz`,
    /// which the reader accepts and maps onto this field.
    pub normals_xyz: Option<Array2<f32>>,
    /// Optional `(P,)` per-point confidence in the stored normal: `0` means the
    /// matching `normals_xyz` row carries no data-derived support (a
    /// placeholder), `255` means fully data-derived, and intermediate values are
    /// reserved as a graded scale for future writers. Consumers must treat it
    /// monotonically rather than switching on exact codes. `None` when the
    /// reconstruction carries no confidence information at all — which is *not*
    /// the same as "all confident".
    ///
    /// On disk this is `points3d/normal_confidence` (version 5+, present only
    /// when `points3d/metadata.json`'s `has_normal_confidence` is `true`). The
    /// writer passes it through untouched.
    pub normal_confidence: Option<Array1<u8>>,

    // Per-point solve constraints (optional, version 7+): what an adjustment
    // owns of each point, and what a caller-owned distance is measured from.
    // The three columns are present together or absent together, and absent is
    // "every point free", which is what every file below version 7 is. They
    // annotate `positions_xyzw` without changing its meaning: `w = 0` is still
    // a direction and `w != 0` still a finite point, whatever the constraint.
    /// `(P,)` constraint per point in the canonical numbering:
    /// [`POINT_CONSTRAINT_FREE`], [`POINT_CONSTRAINT_RANGED`] or
    /// [`POINT_CONSTRAINT_HELD`]. `None` when the file carries no constraints,
    /// which is every point free.
    ///
    /// On disk this is `points3d/point_constraints` (version 7+, present only
    /// when `points3d/metadata.json`'s `has_point_constraints` is `true`), where
    /// a code indexes that entry's `point_constraint_names` legend instead. The
    /// reader resolves every code through the legend the file carries and hands
    /// back the canonical numbering, and the writer states the canonical legend,
    /// so the file's own numbering never reaches a consumer.
    pub point_constraints: Option<Array1<u8>>,
    /// `(P,)` distance a ranged point sits at from its reference: a positive
    /// world-unit distance, or `+inf` for a direction (which needs no
    /// reference). `NaN` on every free and held row. `None` together with
    /// [`Self::point_constraints`].
    ///
    /// On disk this is `points3d/constraint_distances` (version 7+).
    pub constraint_distances: Option<Array1<f64>>,
    /// `(P,)` image index a finite distance is measured from -- the distance
    /// runs from that image's camera centre at whatever pose the reader holds.
    /// [`NO_REFERENCE_IMAGE`] on every row that names no image: free, held, and
    /// ranged at an infinite distance. `None` together with
    /// [`Self::point_constraints`].
    ///
    /// The file carries the single-image reference only. An adjustment can also
    /// measure a distance from the mean of several camera centres, which is a
    /// call-time construct of the kernel rather than stored state.
    ///
    /// On disk this is `points3d/constraint_reference_images` (version 7+).
    pub constraint_reference_images: Option<Array1<u32>>,

    // Per-point oriented-patch ("surfel") frame (optional, version 3+), stored
    // alongside the other `points3d/` arrays. A patch is centred on its 3D point
    // with outward normal `u × v`, so only the in-plane frame is stored: two
    // half-extent vectors `u` and `v` that span the patch corner
    // `(center + s·u + t·v, w)` for `(s, t) ∈ [-1, 1]²`, each carrying the
    // in-plane orientation and half-size. The offset is homogeneous, so this is
    // defined for finite points (planar surfels) and points at infinity alike
    // (a patch of directions tangent to the sphere, whose outward normal is fixed
    // at `normalize(-d)` for direction `d`, so `u × v` points along `-d`). A
    // point with no patch stores all-zero rows (a row is "present" iff its `u` is
    // non-zero), independent of finiteness. `patch_u_halfvec_xyz` and
    // `patch_v_halfvec_xyz` are both present or both `None`; bitmaps require them.
    /// `(P, 3)` in-plane half-extent vector `u`. `None` when no patch frame.
    pub patch_u_halfvec_xyz: Option<Array2<f32>>,
    /// `(P, 3)` in-plane half-extent vector `v`. `None` when no patch frame.
    pub patch_v_halfvec_xyz: Option<Array2<f32>>,
    /// `(P, R, R, 4)` RGBA patch textures, one `R×R` bitmap per point, stored
    /// like image thumbnails. The alpha channel holds a per-pixel confidence.
    /// `None` when no patch bitmaps.
    pub patch_bitmaps_y_x_rgba: Option<Array4<u8>>,

    // Tracks
    /// `(M,)` image index per observation.
    pub image_indexes: Array1<u32>,
    /// `(M,)` feature index per observation (index into the per-image `.sift`).
    /// `Some` in a `sift_files` file; `None` in an `embedded_patches` file.
    pub feature_indexes: Option<Array1<u32>>,
    /// `(M, 2)` sub-pixel `(u, v)` keypoint per observation, in image pixel
    /// coordinates. Always `Some` in an `embedded_patches` file, where it *is*
    /// the observation coordinate. Optional in a `sift_files` file, where it is
    /// an inline copy of the position the `feature_indexes` resolve to in the
    /// `.sift` companions, so a consumer that has only the `.sfmr` still reads
    /// where each observation sits.
    pub keypoints_xy: Option<Array2<f32>>,
    /// Optional `(M,)` per-observation confidence in that observation's
    /// photometric sharpness *relative to its track's consensus*: how well this
    /// observation's image content resolves the detail the rest of the track
    /// agrees on.
    ///
    /// `0` means **no data-derived support** — nothing measured this
    /// observation (the writer had no estimator, or the track was too short or
    /// unscored to measure against). It is not a claim that the observation is
    /// bad. Measured values occupy `1..=255`: `1` is maximally soft relative to
    /// the track consensus, `255` fully sharp. Consumers must treat the value
    /// monotonically (higher = sharper) rather than switching on exact codes.
    /// `None` means no information at all, which is *not* the same as "every
    /// observation is sharp".
    ///
    /// On disk this is `tracks/observation_confidence` (version 6+, present
    /// only when `tracks/metadata.json`'s `has_observation_confidence` is
    /// `true`). It is defined for both `sift_files` and `embedded_patches`
    /// files and requires no other array to be present — it rates observations,
    /// which always exist. The writer passes it through untouched, so a writer
    /// that supplies it is responsible for keeping it parallel to the
    /// observations.
    pub observation_confidence: Option<Array1<u8>>,
    /// `(M,)` point index per observation.
    pub point_indexes: Array1<u32>,
    /// `(P,)` number of observations per 3D point.
    pub observation_counts: Array1<u32>,

    // Depth statistics
    pub depth_statistics: DepthStatistics,
    /// `(N, 128)` depth histogram counts per image.
    pub observed_depth_histogram_counts: Array2<u32>,
}
