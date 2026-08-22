// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sfmr` file reading.

use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array4};

use sfmtool_archive_io::{read_binary_array, read_json_entry, read_uint128_array};

use crate::entries;
use crate::types::*;

const WORKSPACE_MARKER: &str = ".sfm-workspace.json";

/// Read only the top-level metadata from a `.sfmr` file (fast, no binary data).
pub fn read_sfmr_metadata(path: &Path) -> Result<SfmrMetadata, SfmrError> {
    let file = std::fs::File::open(path).map_err(|e| SfmrError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut archive = zip::ZipArchive::new(file)?;
    Ok(read_json_entry(&mut archive, entries::metadata())?)
}

/// Read only the content-integrity hashes from a `.sfmr` file.
///
/// Decompresses just `content_hash.json.zst`, so it is cheap enough to scan a
/// directory of `.sfmr` files (e.g. to resolve a `pt3d_<hash>_<index>` Point ID
/// by its `content_xxh128` prefix) without loading any reconstruction data.
pub fn read_sfmr_content_hash(path: &Path) -> Result<ContentHash, SfmrError> {
    let file = std::fs::File::open(path).map_err(|e| SfmrError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut archive = zip::ZipArchive::new(file)?;
    Ok(read_json_entry(&mut archive, entries::content_hash())?)
}

/// Read a complete `.sfmr` file into columnar data.
pub fn read_sfmr(path: &Path) -> Result<SfmrData, SfmrError> {
    let file = std::fs::File::open(path).map_err(|e| SfmrError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut archive = zip::ZipArchive::new(file)?;

    // Top-level metadata
    let mut metadata: SfmrMetadata = read_json_entry(&mut archive, entries::metadata())?;
    let content_hash: ContentHash = read_json_entry(&mut archive, entries::content_hash())?;

    // Reject versions newer than this build understands; their layout is unknown
    // so reading with current-version assumptions would misparse silently.
    if metadata.version > SFMR_FORMAT_VERSION {
        return Err(SfmrError::InvalidFormat(format!(
            "unsupported .sfmr format version {} (this build supports up to \
             {SFMR_FORMAT_VERSION})",
            metadata.version
        )));
    }

    let image_count = metadata.image_count as usize;
    let point_count = metadata.point_count as usize;
    let observation_count = metadata.observation_count as usize;
    // Version 1 stored Euclidean positions and `points3d_indexes`; version 2
    // stores homogeneous positions and `point_indexes`. The reader accepts both
    // and upgrades version 1 to the version 2 in-memory model.
    let is_v1 = metadata.version < 2;
    // Version 3 renamed `points3d/estimated_normals_xyz` to
    // `points3d/normals_xyz` and added the optional per-point patch frame in the
    // points3d section. Versions 1 and 2 carry the legacy normals name and no
    // patch frame.
    let is_pre_v3 = metadata.version < 3;

    // Cameras
    let cameras: Vec<SfmrCamera> = read_json_entry(&mut archive, entries::cameras_metadata())?;
    if cameras.len() != metadata.camera_count as usize {
        return Err(SfmrError::InvalidFormat(format!(
            "Camera count mismatch: metadata says {}, got {}",
            metadata.camera_count,
            cameras.len()
        )));
    }

    // Cross-check section metadata
    let images_meta: serde_json::Value = read_json_entry(&mut archive, entries::images_metadata())?;
    if images_meta.get("image_count").and_then(|v| v.as_u64()) != Some(image_count as u64) {
        return Err(SfmrError::InvalidFormat(
            "images/metadata.json.zst image_count doesn't match top-level metadata".into(),
        ));
    }

    let points3d_meta: serde_json::Value =
        read_json_entry(&mut archive, entries::points3d_metadata())?;
    // Accept either the version 2 key (`point_count`) or the version 1 key
    // (`points3d_count`).
    let section_point_count = points3d_meta
        .get("point_count")
        .or_else(|| points3d_meta.get("points3d_count"))
        .and_then(|v| v.as_u64());
    if section_point_count != Some(point_count as u64) {
        return Err(SfmrError::InvalidFormat(
            "points3d/metadata.json.zst point count doesn't match top-level metadata".into(),
        ));
    }

    let tracks_meta: serde_json::Value = read_json_entry(&mut archive, entries::tracks_metadata())?;
    if tracks_meta
        .get("observation_count")
        .and_then(|v| v.as_u64())
        != Some(observation_count as u64)
    {
        return Err(SfmrError::InvalidFormat(
            "tracks/metadata.json.zst observation_count doesn't match top-level metadata".into(),
        ));
    }

    // Images
    let image_names: Vec<String> = read_json_entry(&mut archive, entries::images_names())?;
    if image_names.len() != image_count {
        return Err(SfmrError::ShapeMismatch(format!(
            "image names count {} != image_count {image_count}",
            image_names.len()
        )));
    }

    let camera_indexes_vec: Vec<u32> = read_binary_array(
        &mut archive,
        &entries::images_camera_indexes(image_count),
        image_count,
    )?;
    let camera_indexes = Array1::from_vec(camera_indexes_vec);

    let quaternions_vec: Vec<f64> = read_binary_array(
        &mut archive,
        &entries::images_quaternions_wxyz(image_count),
        image_count * 4,
    )?;
    let quaternions_wxyz = Array2::from_shape_vec((image_count, 4), quaternions_vec)
        .map_err(|e| SfmrError::ShapeMismatch(format!("quaternions reshape: {e}")))?;

    let translations_vec: Vec<f64> = read_binary_array(
        &mut archive,
        &entries::images_translations_xyz(image_count),
        image_count * 3,
    )?;
    let translations_xyz = Array2::from_shape_vec((image_count, 3), translations_vec)
        .map_err(|e| SfmrError::ShapeMismatch(format!("translations reshape: {e}")))?;

    // Observation source (version 4+). Legacy v1–v3 files have no key and
    // `serde(default)` reads them as `sift_files`. Reject any unrecognized value
    // rather than silently treating it as `sift_files`.
    match metadata.feature_source.as_str() {
        FEATURE_SOURCE_SIFT_FILES | FEATURE_SOURCE_EMBEDDED_PATCHES => {}
        other => {
            return Err(SfmrError::InvalidFormat(format!(
                "unknown feature_source {other:?} (expected {FEATURE_SOURCE_SIFT_FILES:?} \
                 or {FEATURE_SOURCE_EMBEDDED_PATCHES:?})"
            )));
        }
    }
    // The top-level `feature_source` is the authoritative discriminator for which
    // per-observation / per-image columns exist (a file is wholly one mode — the
    // spec's "no mixing"). The `tracks/metadata.json` `has_feature_indexes` /
    // `has_keypoints_xy` flags mirror it for section-local readers and are
    // cross-checked in `verify_sfmr`; they intentionally do not gate reading here
    // the way the independent `points3d` `has_*` flags do.
    let is_embedded = metadata.feature_source == FEATURE_SOURCE_EMBEDDED_PATCHES;

    // A `sift_files` file links to `.sift` via per-image tool/content hashes; an
    // `embedded_patches` file substitutes the direct image-bytes hash instead.
    let (feature_tool_hashes, sift_content_hashes, image_file_hashes) = if is_embedded {
        let image_file_hashes = read_uint128_array(
            &mut archive,
            &entries::images_image_file_hashes(image_count),
            image_count,
        )?;
        (None, None, Some(image_file_hashes))
    } else {
        let feature_tool_hashes = read_uint128_array(
            &mut archive,
            &entries::images_feature_tool_hashes(image_count),
            image_count,
        )?;
        let sift_content_hashes = read_uint128_array(
            &mut archive,
            &entries::images_sift_content_hashes(image_count),
            image_count,
        )?;
        (Some(feature_tool_hashes), Some(sift_content_hashes), None)
    };

    // Thumbnails
    let thumbnails_vec: Vec<u8> = read_binary_array(
        &mut archive,
        &entries::images_thumbnails_y_x_rgb(image_count),
        image_count * THUMBNAIL_SIZE * THUMBNAIL_SIZE * 3,
    )?;
    let thumbnails_y_x_rgb = Array4::from_shape_vec(
        (image_count, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3),
        thumbnails_vec,
    )
    .map_err(|e| SfmrError::ShapeMismatch(format!("thumbnails reshape: {e}")))?;

    // Depth statistics
    let depth_statistics: DepthStatistics =
        read_json_entry(&mut archive, entries::images_depth_statistics())?;
    let num_buckets = depth_statistics.num_histogram_buckets as usize;

    let histogram_vec: Vec<u32> = read_binary_array(
        &mut archive,
        &entries::images_observed_depth_histogram_counts(image_count, num_buckets),
        image_count * num_buckets,
    )?;
    let observed_depth_histogram_counts =
        Array2::from_shape_vec((image_count, num_buckets), histogram_vec)
            .map_err(|e| SfmrError::ShapeMismatch(format!("histogram reshape: {e}")))?;

    // Points3D positions: version 1 stored Euclidean `(P, 3)`; version 2 stores
    // homogeneous `(P, 4)`. A version 1 file is upgraded by appending a `w = 1`
    // column, since every version 1 point is finite.
    let positions_xyzw = if is_v1 {
        let positions_vec: Vec<f64> = read_binary_array(
            &mut archive,
            &entries::points3d_positions(true, point_count),
            point_count * 3,
        )?;
        let mut xyzw = Array2::<f64>::zeros((point_count, 4));
        for i in 0..point_count {
            xyzw[[i, 0]] = positions_vec[i * 3];
            xyzw[[i, 1]] = positions_vec[i * 3 + 1];
            xyzw[[i, 2]] = positions_vec[i * 3 + 2];
            xyzw[[i, 3]] = 1.0;
        }
        xyzw
    } else {
        let positions_vec: Vec<f64> = read_binary_array(
            &mut archive,
            &entries::points3d_positions(false, point_count),
            point_count * 4,
        )?;
        Array2::from_shape_vec((point_count, 4), positions_vec)
            .map_err(|e| SfmrError::ShapeMismatch(format!("positions reshape: {e}")))?
    };

    let colors_vec: Vec<u8> = read_binary_array(
        &mut archive,
        &entries::points3d_colors_rgb(point_count),
        point_count * 3,
    )?;
    let colors_rgb = Array2::from_shape_vec((point_count, 3), colors_vec)
        .map_err(|e| SfmrError::ShapeMismatch(format!("colors reshape: {e}")))?;

    let reprojection_vec: Vec<f32> = read_binary_array(
        &mut archive,
        &entries::points3d_reprojection_errors(point_count),
        point_count,
    )?;
    let reprojection_errors = Array1::from_vec(reprojection_vec);

    // Normals are optional in version 3 (flagged by `has_normals`, which
    // defaults to `false` when absent — like the patch flags). Versions 1 and 2
    // always carry them and have no flag.
    let has_normals = is_pre_v3
        || points3d_meta
            .get("has_normals")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
    let normals_xyz = if has_normals {
        let normals_name = entries::points3d_normals(is_pre_v3, point_count);
        let normals_vec: Vec<f32> =
            read_binary_array(&mut archive, &normals_name, point_count * 3)?;
        Some(
            Array2::from_shape_vec((point_count, 3), normals_vec)
                .map_err(|e| SfmrError::ShapeMismatch(format!("normals reshape: {e}")))?,
        )
    } else {
        None
    };

    // Optional per-point normal confidence (version 5+). Absent means "no
    // confidence information", so an older file — which never carries the flag
    // nor the array — simply reads as `None`.
    let normal_confidence = if points3d_meta
        .get("has_normal_confidence")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        let confidence_vec: Vec<u8> = read_binary_array(
            &mut archive,
            &entries::points3d_normal_confidence(point_count),
            point_count,
        )?;
        Some(Array1::from_vec(confidence_vec))
    } else {
        None
    };

    // Optional per-point patch frame (version 3+), stored beside the normals.
    // Takes the entry name rather than a field stem, so the name comes from
    // `entries` like every other read. The stem for the error message is
    // recovered from the name instead of being passed a second time.
    let read_vec3 = |archive: &mut zip::ZipArchive<std::fs::File>,
                     entry: &str|
     -> Result<Array2<f32>, SfmrError> {
        let v: Vec<f32> = read_binary_array(archive, entry, point_count * 3)?;
        let field = entry
            .rsplit_once('/')
            .map_or(entry, |(_, leaf)| leaf)
            .split('.')
            .next()
            .unwrap_or(entry);
        Array2::from_shape_vec((point_count, 3), v)
            .map_err(|e| SfmrError::ShapeMismatch(format!("{field} reshape: {e}")))
    };
    let has_uv_frames = points3d_meta
        .get("has_uv_frames")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let (patch_u_halfvec_xyz, patch_v_halfvec_xyz) = if has_uv_frames {
        (
            Some(read_vec3(
                &mut archive,
                &entries::points3d_patch_u_halfvec_xyz(point_count),
            )?),
            Some(read_vec3(
                &mut archive,
                &entries::points3d_patch_v_halfvec_xyz(point_count),
            )?),
        )
    } else {
        (None, None)
    };
    let patch_bitmaps_y_x_rgba = if has_uv_frames
        && points3d_meta
            .get("has_patch_bitmaps")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    {
        let r = points3d_meta
            .get("patch_bitmap_resolution")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                SfmrError::InvalidFormat(
                    "points3d/metadata.json.zst has_patch_bitmaps but no patch_bitmap_resolution"
                        .into(),
                )
            })? as usize;
        let bitmaps_vec: Vec<u8> = read_binary_array(
            &mut archive,
            &entries::points3d_patch_bitmaps_y_x_rgba(point_count, r),
            point_count * r * r * 4,
        )?;
        Some(
            Array4::from_shape_vec((point_count, r, r, 4), bitmaps_vec)
                .map_err(|e| SfmrError::ShapeMismatch(format!("patch bitmaps reshape: {e}")))?,
        )
    } else {
        None
    };

    // Tracks
    let image_indexes_vec: Vec<u32> = read_binary_array(
        &mut archive,
        &entries::tracks_image_indexes(observation_count),
        observation_count,
    )?;
    let image_indexes = Array1::from_vec(image_indexes_vec);

    // A `sift_files` file references `.sift` features by index; an
    // `embedded_patches` file carries the sub-pixel `(u, v)` coordinate inline.
    let (feature_indexes, keypoints_xy) = if is_embedded {
        let kp_vec: Vec<f32> = read_binary_array(
            &mut archive,
            &entries::tracks_keypoints_xy(observation_count),
            observation_count * 2,
        )?;
        let keypoints_xy = Array2::from_shape_vec((observation_count, 2), kp_vec)
            .map_err(|e| SfmrError::ShapeMismatch(format!("keypoints_xy reshape: {e}")))?;
        validate_keypoints(
            &keypoints_xy,
            image_indexes.as_slice().unwrap(),
            camera_indexes.as_slice().unwrap(),
            &cameras,
        )
        .map_err(SfmrError::InvalidFormat)?;
        (None, Some(keypoints_xy))
    } else {
        let feature_indexes_vec: Vec<u32> = read_binary_array(
            &mut archive,
            &entries::tracks_feature_indexes(observation_count),
            observation_count,
        )?;
        (Some(Array1::from_vec(feature_indexes_vec)), None)
    };

    // Optional per-observation confidence (version 6+). Absent means "no
    // information at all", so an older file — which never carries the flag nor
    // the array — simply reads as `None`. Independent of `feature_source`.
    let observation_confidence = if tracks_meta
        .get("has_observation_confidence")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        let confidence_vec: Vec<u8> = read_binary_array(
            &mut archive,
            &entries::tracks_observation_confidence(observation_count),
            observation_count,
        )?;
        Some(Array1::from_vec(confidence_vec))
    } else {
        None
    };

    // Version 1 named this array `points3d_indexes`; version 2 renames it to
    // `point_indexes`.
    let point_indexes_name = entries::tracks_point_indexes(is_v1, observation_count);
    let point_indexes_vec: Vec<u32> =
        read_binary_array(&mut archive, &point_indexes_name, observation_count)?;
    let point_indexes = Array1::from_vec(point_indexes_vec);

    let observation_counts_vec: Vec<u32> = read_binary_array(
        &mut archive,
        &entries::tracks_observation_counts(point_count),
        point_count,
    )?;
    let observation_counts = Array1::from_vec(observation_counts_vec);

    // Rigs and frames (optional — only present in rig-aware reconstructions)
    let rig_frame_data = if archive.index_for_name(entries::rigs_metadata()).is_some() {
        let rigs_metadata: RigsMetadata = read_json_entry(&mut archive, entries::rigs_metadata())?;
        let sensor_count = rigs_metadata.sensor_count as usize;

        // Validate rig_count matches rigs array length
        if rigs_metadata.rigs.len() != rigs_metadata.rig_count as usize {
            return Err(SfmrError::InvalidFormat(format!(
                "rigs/metadata.json.zst rig_count {} doesn't match rigs array length {}",
                rigs_metadata.rig_count,
                rigs_metadata.rigs.len()
            )));
        }

        let sensor_camera_indexes_vec: Vec<u32> = read_binary_array(
            &mut archive,
            &entries::rigs_sensor_camera_indexes(sensor_count),
            sensor_count,
        )?;
        let sensor_camera_indexes = Array1::from_vec(sensor_camera_indexes_vec);

        let sensor_quaternions_vec: Vec<f64> = read_binary_array(
            &mut archive,
            &entries::rigs_sensor_quaternions_wxyz(sensor_count),
            sensor_count * 4,
        )?;
        let sensor_quaternions_wxyz =
            Array2::from_shape_vec((sensor_count, 4), sensor_quaternions_vec).map_err(|e| {
                SfmrError::ShapeMismatch(format!("sensor quaternions reshape: {e}"))
            })?;

        let sensor_translations_vec: Vec<f64> = read_binary_array(
            &mut archive,
            &entries::rigs_sensor_translations_xyz(sensor_count),
            sensor_count * 3,
        )?;
        let sensor_translations_xyz =
            Array2::from_shape_vec((sensor_count, 3), sensor_translations_vec).map_err(|e| {
                SfmrError::ShapeMismatch(format!("sensor translations reshape: {e}"))
            })?;

        // Frames
        let frames_metadata: FramesMetadata =
            read_json_entry(&mut archive, entries::frames_metadata())?;
        let frame_count = frames_metadata.frame_count as usize;

        let rig_indexes_vec: Vec<u32> = read_binary_array(
            &mut archive,
            &entries::frames_rig_indexes(frame_count),
            frame_count,
        )?;
        let rig_indexes = Array1::from_vec(rig_indexes_vec);

        let image_sensor_indexes_vec: Vec<u32> = read_binary_array(
            &mut archive,
            &entries::frames_image_sensor_indexes(image_count),
            image_count,
        )?;
        let image_sensor_indexes = Array1::from_vec(image_sensor_indexes_vec);

        let image_frame_indexes_vec: Vec<u32> = read_binary_array(
            &mut archive,
            &entries::frames_image_frame_indexes(image_count),
            image_count,
        )?;
        let image_frame_indexes = Array1::from_vec(image_frame_indexes_vec);

        Some(RigFrameData {
            rigs_metadata,
            sensor_camera_indexes,
            sensor_quaternions_wxyz,
            sensor_translations_xyz,
            frames_metadata,
            rig_indexes,
            image_sensor_indexes,
            image_frame_indexes,
        })
    } else {
        None
    };

    // The arrays above are upgraded to the current structural layout, but
    // `metadata.version` deliberately keeps the **stored** version: version ≤ 4
    // files hold COLMAP-convention poses/points, and the consumer that can see
    // the convention math (`SfmrReconstruction::load` in `sfmtool-core`) gates
    // the COLMAP→canonical upgrade on this value (see [`SFMR_FORMAT_VERSION`]).
    // Recompute `infinity_point_count` from the `w` column so it is correct
    // regardless of the source version (version 1 has none). `feature_source`
    // already defaulted to `sift_files` for pre-v4 files on deserialization.
    metadata.infinity_point_count = (0..point_count)
        .filter(|&i| positions_xyzw[[i, 3]] == 0.0)
        .count() as u32;

    // Resolve workspace directory (best-effort, None on failure)
    let workspace_dir = resolve_workspace_dir(path, &metadata).ok();

    Ok(SfmrData {
        workspace_dir,
        metadata,
        content_hash,
        cameras,
        rig_frame_data,
        image_names,
        camera_indexes,
        quaternions_wxyz,
        translations_xyz,
        feature_tool_hashes,
        sift_content_hashes,
        image_file_hashes,
        thumbnails_y_x_rgb,
        positions_xyzw,
        colors_rgb,
        reprojection_errors,
        normals_xyz,
        normal_confidence,
        patch_u_halfvec_xyz,
        patch_v_halfvec_xyz,
        patch_bitmaps_y_x_rgba,
        image_indexes,
        feature_indexes,
        keypoints_xy,
        observation_confidence,
        point_indexes,
        observation_counts,
        depth_statistics,
        observed_depth_histogram_counts,
    })
}

/// Resolve the workspace directory for a `.sfmr` file.
///
/// Strategy (from the spec):
/// 1. Try `workspace.relative_path` from the `.sfmr` file's directory
/// 2. Fall back to `workspace.absolute_path`
/// 3. Fall back to searching upward from the `.sfmr` file for `.sfm-workspace.json`
pub fn resolve_workspace_dir(
    sfmr_path: &Path,
    metadata: &SfmrMetadata,
) -> Result<PathBuf, SfmrError> {
    let sfmr_dir = dunce::canonicalize(sfmr_path.parent().unwrap_or_else(|| Path::new(".")))
        .unwrap_or_else(|_| sfmr_path.parent().unwrap_or(Path::new(".")).to_path_buf());

    // Strategy 1: Try relative path from .sfmr file's directory
    let relative_path = &metadata.workspace.relative_path;
    if !relative_path.is_empty() {
        let candidate = sfmr_dir.join(relative_path);
        if let Ok(candidate) = dunce::canonicalize(&candidate) {
            if candidate.join(WORKSPACE_MARKER).exists() {
                return Ok(candidate);
            }
        }
    }

    // Strategy 2: Fall back to absolute path
    let absolute_path = &metadata.workspace.absolute_path;
    if !absolute_path.is_empty() {
        let candidate = PathBuf::from(absolute_path);
        if candidate.exists() && candidate.join(WORKSPACE_MARKER).exists() {
            return Ok(candidate);
        }
    }

    // Strategy 3: Search upward from the .sfmr file for .sfm-workspace.json
    if let Some(candidate) = find_workspace_for_path(&sfmr_dir) {
        return Ok(candidate);
    }

    Err(SfmrError::InvalidFormat(format!(
        "Could not resolve workspace directory for {}. Tried:\n\
         1. Relative path from .sfmr: {}\n\
         2. Absolute path: {}\n\
         3. Searching upward from .sfmr file: no workspace found",
        sfmr_path.display(),
        if relative_path.is_empty() {
            "not specified"
        } else {
            relative_path
        },
        if absolute_path.is_empty() {
            "not specified"
        } else {
            absolute_path
        },
    )))
}

/// Search upward from a path for a directory containing `.sfm-workspace.json`.
fn find_workspace_for_path(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        if current.join(WORKSPACE_MARKER).exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}
