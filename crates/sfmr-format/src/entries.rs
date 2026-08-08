// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Archive entry names — the single source of truth for `.sfmr` ZIP paths.
//!
//! Every entry in a `.sfmr` archive encodes its shape and dtype in its own name
//! (`tracks/keypoints_xy.{observation_count}.2.float32.zst`), so the name is not
//! decoration: [`read`](crate::read) looks entries up by exact name,
//! [`write`](crate::write) creates them, and [`verify`](crate::verify) rehashes
//! them. A name written three times is a name that can disagree three ways, and
//! a mismatch surfaces only as a failed lookup or a wrong content hash.
//!
//! One function per entry, so the template exists once. Callers pass the
//! dimension tokens; the function owns the layout.
//!
//! Dimension parameters are `impl Display` rather than `usize` because the same
//! conceptual count reaches these functions in different integer types — a
//! `usize` local in [`read`](crate::read) and [`write`](crate::write), a `u32`
//! deserialized from section metadata in [`verify`](crate::verify). The token is
//! only ever interpolated, so widening the parameter is preferable to `as usize`
//! casts at the call sites. The trade-off is that `Display` is wider than the
//! intent — it would also accept a float or a signed value — so it documents
//! less than a concrete type would.
//!
//! ## Version-dependent names
//!
//! Three entries were renamed across format versions, and both the reader and
//! the verifier have to accept either spelling. Those functions take the
//! version predicate rather than making each caller re-derive the mapping:
//!
//! | Function | `true` (legacy) | `false` (current) |
//! |---|---|---|
//! | [`points3d_positions`] | `positions_xyz` (v1, Euclidean) | `positions_xyzw` (v2+, homogeneous) |
//! | [`points3d_normals`] | `estimated_normals_xyz` (pre-v3) | `normals_xyz` (v3+) |
//! | [`tracks_point_indexes`] | `points3d_indexes` (v1) | `point_indexes` (v2+) |
//!
//! The writer only ever emits the current spelling, so it passes `false`.
//!
//! ## How the tests use this module
//!
//! `tests::entry_names_are_pinned` calls these functions and compares each
//! result against a literal spelled out in `tests.rs`. Everywhere else, tests
//! that name an archive entry spell it literally rather than calling in here —
//! routing them through this module would make them agree with whatever these
//! functions produce, which is the property under test.
//!
//! That pin covers spelling and shape only. Which count a given entry is sized
//! by is decided at the call sites, not here, so a caller passing
//! `observation_count` where `point_count` belongs is invisible to it;
//! `tests::archive_entry_names_pin_call_sites` pins the written archive's
//! listing to cover that.

/// `metadata.json.zst` — the top-level `.sfmr` metadata.
///
/// Top-level entries carry no section prefix and are not part of any section
/// hash; they sit at the archive root alongside the section directories.
pub(crate) fn metadata() -> &'static str {
    "metadata.json.zst"
}

/// `content_hash.json.zst` — the top-level per-section digest record.
pub(crate) fn content_hash() -> &'static str {
    "content_hash.json.zst"
}

/// `cameras/metadata.json.zst` — the camera-intrinsics array.
pub(crate) fn cameras_metadata() -> &'static str {
    "cameras/metadata.json.zst"
}

/// `rigs/metadata.json.zst` — rig descriptors (optional section).
pub(crate) fn rigs_metadata() -> &'static str {
    "rigs/metadata.json.zst"
}

/// `rigs/sensor_camera_indexes` — camera index per rig sensor.
pub(crate) fn rigs_sensor_camera_indexes(sensor_count: impl std::fmt::Display) -> String {
    format!("rigs/sensor_camera_indexes.{sensor_count}.uint32.zst")
}

/// `rigs/sensor_quaternions_wxyz` — sensor-from-rig rotation per sensor.
pub(crate) fn rigs_sensor_quaternions_wxyz(sensor_count: impl std::fmt::Display) -> String {
    format!("rigs/sensor_quaternions_wxyz.{sensor_count}.4.float64.zst")
}

/// `rigs/sensor_translations_xyz` — sensor-from-rig translation per sensor.
pub(crate) fn rigs_sensor_translations_xyz(sensor_count: impl std::fmt::Display) -> String {
    format!("rigs/sensor_translations_xyz.{sensor_count}.3.float64.zst")
}

/// `frames/metadata.json.zst` — frame-section metadata (optional section).
pub(crate) fn frames_metadata() -> &'static str {
    "frames/metadata.json.zst"
}

/// `frames/image_frame_indexes` — frame index per image.
pub(crate) fn frames_image_frame_indexes(image_count: impl std::fmt::Display) -> String {
    format!("frames/image_frame_indexes.{image_count}.uint32.zst")
}

/// `frames/image_sensor_indexes` — rig-sensor index per image.
pub(crate) fn frames_image_sensor_indexes(image_count: impl std::fmt::Display) -> String {
    format!("frames/image_sensor_indexes.{image_count}.uint32.zst")
}

/// `frames/rig_indexes` — rig index per frame.
pub(crate) fn frames_rig_indexes(frame_count: impl std::fmt::Display) -> String {
    format!("frames/rig_indexes.{frame_count}.uint32.zst")
}

/// `images/metadata.json.zst` — image-section metadata.
pub(crate) fn images_metadata() -> &'static str {
    "images/metadata.json.zst"
}

/// `images/names.json.zst` — image file names.
pub(crate) fn images_names() -> &'static str {
    "images/names.json.zst"
}

/// `images/depth_statistics.json.zst` — per-image observed-depth summary.
pub(crate) fn images_depth_statistics() -> &'static str {
    "images/depth_statistics.json.zst"
}

/// `images/camera_indexes` — camera index per image.
pub(crate) fn images_camera_indexes(image_count: impl std::fmt::Display) -> String {
    format!("images/camera_indexes.{image_count}.uint32.zst")
}

/// `images/quaternions_wxyz` — world-from-camera rotation per image.
pub(crate) fn images_quaternions_wxyz(image_count: impl std::fmt::Display) -> String {
    format!("images/quaternions_wxyz.{image_count}.4.float64.zst")
}

/// `images/translations_xyz` — world-from-camera translation per image.
pub(crate) fn images_translations_xyz(image_count: impl std::fmt::Display) -> String {
    format!("images/translations_xyz.{image_count}.3.float64.zst")
}

/// `images/image_file_hashes` — XXH128 of each source image file.
pub(crate) fn images_image_file_hashes(image_count: impl std::fmt::Display) -> String {
    format!("images/image_file_hashes.{image_count}.uint128.zst")
}

/// `images/feature_tool_hashes` — XXH128 identifying the feature tool per image.
pub(crate) fn images_feature_tool_hashes(image_count: impl std::fmt::Display) -> String {
    format!("images/feature_tool_hashes.{image_count}.uint128.zst")
}

/// `images/sift_content_hashes` — XXH128 of each linked `.sift` file.
pub(crate) fn images_sift_content_hashes(image_count: impl std::fmt::Display) -> String {
    format!("images/sift_content_hashes.{image_count}.uint128.zst")
}

/// `images/thumbnails_y_x_rgb` — fixed 128×128 RGB thumbnail per image.
pub(crate) fn images_thumbnails_y_x_rgb(image_count: impl std::fmt::Display) -> String {
    format!("images/thumbnails_y_x_rgb.{image_count}.128.128.3.uint8.zst")
}

/// `images/observed_depth_histogram_counts` — per-image depth histogram.
pub(crate) fn images_observed_depth_histogram_counts(
    image_count: impl std::fmt::Display,
    num_buckets: impl std::fmt::Display,
) -> String {
    format!("images/observed_depth_histogram_counts.{image_count}.{num_buckets}.uint32.zst")
}

/// `points3d/metadata.json.zst` — point-section metadata.
pub(crate) fn points3d_metadata() -> &'static str {
    "points3d/metadata.json.zst"
}

/// `points3d/colors_rgb` — RGB colour per point.
pub(crate) fn points3d_colors_rgb(point_count: impl std::fmt::Display) -> String {
    format!("points3d/colors_rgb.{point_count}.3.uint8.zst")
}

/// `points3d/reprojection_errors` — mean reprojection error per point.
pub(crate) fn points3d_reprojection_errors(point_count: impl std::fmt::Display) -> String {
    format!("points3d/reprojection_errors.{point_count}.float32.zst")
}

/// `points3d/normal_confidence` — normal confidence per point.
pub(crate) fn points3d_normal_confidence(point_count: impl std::fmt::Display) -> String {
    format!("points3d/normal_confidence.{point_count}.uint8.zst")
}

/// `points3d/patch_u_halfvec_xyz` — patch-frame `u` half-axis per point.
pub(crate) fn points3d_patch_u_halfvec_xyz(point_count: impl std::fmt::Display) -> String {
    format!("points3d/patch_u_halfvec_xyz.{point_count}.3.float32.zst")
}

/// `points3d/patch_v_halfvec_xyz` — patch-frame `v` half-axis per point.
pub(crate) fn points3d_patch_v_halfvec_xyz(point_count: impl std::fmt::Display) -> String {
    format!("points3d/patch_v_halfvec_xyz.{point_count}.3.float32.zst")
}

/// `points3d/patch_bitmaps_y_x_rgba` — square RGBA patch bitmap per point.
///
/// `r` is the patch edge length in pixels, which appears **twice** in the name
/// (the bitmap is square).
pub(crate) fn points3d_patch_bitmaps_y_x_rgba(
    point_count: impl std::fmt::Display,
    r: impl std::fmt::Display,
) -> String {
    format!("points3d/patch_bitmaps_y_x_rgba.{point_count}.{r}.{r}.4.uint8.zst")
}

/// `points3d/positions_xyz` (v1) or `points3d/positions_xyzw` (v2+).
///
/// Version 1 stored Euclidean 3-vectors; version 2 switched to homogeneous
/// 4-vectors so points at infinity are representable. The dtype and the
/// component count both change with the spelling, which is why this is one
/// function rather than two names sharing a suffix.
pub(crate) fn points3d_positions(is_v1: bool, point_count: impl std::fmt::Display) -> String {
    if is_v1 {
        format!("points3d/positions_xyz.{point_count}.3.float64.zst")
    } else {
        format!("points3d/positions_xyzw.{point_count}.4.float64.zst")
    }
}

/// `points3d/estimated_normals_xyz` (pre-v3) or `points3d/normals_xyz` (v3+).
///
/// A pure rename — the shape and dtype are unchanged.
pub(crate) fn points3d_normals(is_pre_v3: bool, point_count: impl std::fmt::Display) -> String {
    if is_pre_v3 {
        format!("points3d/estimated_normals_xyz.{point_count}.3.float32.zst")
    } else {
        format!("points3d/normals_xyz.{point_count}.3.float32.zst")
    }
}

/// `tracks/metadata.json.zst` — track-section metadata.
pub(crate) fn tracks_metadata() -> &'static str {
    "tracks/metadata.json.zst"
}

/// `tracks/image_indexes` — observing image per observation.
pub(crate) fn tracks_image_indexes(observation_count: impl std::fmt::Display) -> String {
    format!("tracks/image_indexes.{observation_count}.uint32.zst")
}

/// `tracks/feature_indexes` — feature index within the observing image.
pub(crate) fn tracks_feature_indexes(observation_count: impl std::fmt::Display) -> String {
    format!("tracks/feature_indexes.{observation_count}.uint32.zst")
}

/// `tracks/keypoints_xy` — inline keypoint per observation (`embedded_patches`).
pub(crate) fn tracks_keypoints_xy(observation_count: impl std::fmt::Display) -> String {
    format!("tracks/keypoints_xy.{observation_count}.2.float32.zst")
}

/// `tracks/observation_confidence` — per-observation confidence (v6+, optional).
pub(crate) fn tracks_observation_confidence(observation_count: impl std::fmt::Display) -> String {
    format!("tracks/observation_confidence.{observation_count}.uint8.zst")
}

/// `tracks/observation_counts` — observation count per point.
///
/// Sized by `point_count`, not `observation_count` — it is the per-point CSR
/// run length over the observation arrays.
pub(crate) fn tracks_observation_counts(point_count: impl std::fmt::Display) -> String {
    format!("tracks/observation_counts.{point_count}.uint32.zst")
}

/// `tracks/points3d_indexes` (v1) or `tracks/point_indexes` (v2+).
///
/// A pure rename, tracking the `points3d` → `point` vocabulary change.
pub(crate) fn tracks_point_indexes(
    is_v1: bool,
    observation_count: impl std::fmt::Display,
) -> String {
    if is_v1 {
        format!("tracks/points3d_indexes.{observation_count}.uint32.zst")
    } else {
        format!("tracks/point_indexes.{observation_count}.uint32.zst")
    }
}
