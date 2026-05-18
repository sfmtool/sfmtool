// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sfmr` file reading.

use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array4};

use crate::archive_io::{read_binary_array, read_json_entry, read_uint128_array};

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
    Ok(read_json_entry(&mut archive, "metadata.json.zst")?)
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
    let mut metadata: SfmrMetadata = read_json_entry(&mut archive, "metadata.json.zst")?;
    let content_hash: ContentHash = read_json_entry(&mut archive, "content_hash.json.zst")?;

    let image_count = metadata.image_count as usize;
    let point_count = metadata.point_count as usize;
    let observation_count = metadata.observation_count as usize;
    // Version 1 stored Euclidean positions and `points3d_indexes`; version 2
    // stores homogeneous positions and `point_indexes`. The reader accepts both
    // and upgrades version 1 to the version 2 in-memory model.
    let is_v1 = metadata.version < 2;

    // Cameras
    let cameras: Vec<SfmrCamera> = read_json_entry(&mut archive, "cameras/metadata.json.zst")?;
    if cameras.len() != metadata.camera_count as usize {
        return Err(SfmrError::InvalidFormat(format!(
            "Camera count mismatch: metadata says {}, got {}",
            metadata.camera_count,
            cameras.len()
        )));
    }

    // Cross-check section metadata
    let images_meta: serde_json::Value = read_json_entry(&mut archive, "images/metadata.json.zst")?;
    if images_meta.get("image_count").and_then(|v| v.as_u64()) != Some(image_count as u64) {
        return Err(SfmrError::InvalidFormat(
            "images/metadata.json.zst image_count doesn't match top-level metadata".into(),
        ));
    }

    let points3d_meta: serde_json::Value =
        read_json_entry(&mut archive, "points3d/metadata.json.zst")?;
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

    let tracks_meta: serde_json::Value = read_json_entry(&mut archive, "tracks/metadata.json.zst")?;
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
    let image_names: Vec<String> = read_json_entry(&mut archive, "images/names.json.zst")?;
    if image_names.len() != image_count {
        return Err(SfmrError::ShapeMismatch(format!(
            "image names count {} != image_count {image_count}",
            image_names.len()
        )));
    }

    let camera_indexes_vec: Vec<u32> = read_binary_array(
        &mut archive,
        &format!("images/camera_indexes.{image_count}.uint32.zst"),
        image_count,
    )?;
    let camera_indexes = Array1::from_vec(camera_indexes_vec);

    let quaternions_vec: Vec<f64> = read_binary_array(
        &mut archive,
        &format!("images/quaternions_wxyz.{image_count}.4.float64.zst"),
        image_count * 4,
    )?;
    let quaternions_wxyz = Array2::from_shape_vec((image_count, 4), quaternions_vec)
        .map_err(|e| SfmrError::ShapeMismatch(format!("quaternions reshape: {e}")))?;

    let translations_vec: Vec<f64> = read_binary_array(
        &mut archive,
        &format!("images/translations_xyz.{image_count}.3.float64.zst"),
        image_count * 3,
    )?;
    let translations_xyz = Array2::from_shape_vec((image_count, 3), translations_vec)
        .map_err(|e| SfmrError::ShapeMismatch(format!("translations reshape: {e}")))?;

    let feature_tool_hashes = read_uint128_array(
        &mut archive,
        &format!("images/feature_tool_hashes.{image_count}.uint128.zst"),
        image_count,
    )?;

    let sift_content_hashes = read_uint128_array(
        &mut archive,
        &format!("images/sift_content_hashes.{image_count}.uint128.zst"),
        image_count,
    )?;

    // Thumbnails
    let thumbnails_vec: Vec<u8> = read_binary_array(
        &mut archive,
        &format!("images/thumbnails_y_x_rgb.{image_count}.128.128.3.uint8.zst"),
        image_count * 128 * 128 * 3,
    )?;
    let thumbnails_y_x_rgb = Array4::from_shape_vec((image_count, 128, 128, 3), thumbnails_vec)
        .map_err(|e| SfmrError::ShapeMismatch(format!("thumbnails reshape: {e}")))?;

    // Depth statistics
    let depth_statistics: DepthStatistics =
        read_json_entry(&mut archive, "images/depth_statistics.json.zst")?;
    let num_buckets = depth_statistics.num_histogram_buckets as usize;

    let histogram_vec: Vec<u32> = read_binary_array(
        &mut archive,
        &format!("images/observed_depth_histogram_counts.{image_count}.{num_buckets}.uint32.zst"),
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
            &format!("points3d/positions_xyz.{point_count}.3.float64.zst"),
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
            &format!("points3d/positions_xyzw.{point_count}.4.float64.zst"),
            point_count * 4,
        )?;
        Array2::from_shape_vec((point_count, 4), positions_vec)
            .map_err(|e| SfmrError::ShapeMismatch(format!("positions reshape: {e}")))?
    };

    let colors_vec: Vec<u8> = read_binary_array(
        &mut archive,
        &format!("points3d/colors_rgb.{point_count}.3.uint8.zst"),
        point_count * 3,
    )?;
    let colors_rgb = Array2::from_shape_vec((point_count, 3), colors_vec)
        .map_err(|e| SfmrError::ShapeMismatch(format!("colors reshape: {e}")))?;

    let reprojection_vec: Vec<f32> = read_binary_array(
        &mut archive,
        &format!("points3d/reprojection_errors.{point_count}.float32.zst"),
        point_count,
    )?;
    let reprojection_errors = Array1::from_vec(reprojection_vec);

    let normals_vec: Vec<f32> = read_binary_array(
        &mut archive,
        &format!("points3d/estimated_normals_xyz.{point_count}.3.float32.zst"),
        point_count * 3,
    )?;
    let estimated_normals_xyz = Array2::from_shape_vec((point_count, 3), normals_vec)
        .map_err(|e| SfmrError::ShapeMismatch(format!("normals reshape: {e}")))?;

    // Tracks
    let image_indexes_vec: Vec<u32> = read_binary_array(
        &mut archive,
        &format!("tracks/image_indexes.{observation_count}.uint32.zst"),
        observation_count,
    )?;
    let image_indexes = Array1::from_vec(image_indexes_vec);

    let feature_indexes_vec: Vec<u32> = read_binary_array(
        &mut archive,
        &format!("tracks/feature_indexes.{observation_count}.uint32.zst"),
        observation_count,
    )?;
    let feature_indexes = Array1::from_vec(feature_indexes_vec);

    // Version 1 named this array `points3d_indexes`; version 2 renames it to
    // `point_indexes`.
    let point_indexes_name = if is_v1 {
        format!("tracks/points3d_indexes.{observation_count}.uint32.zst")
    } else {
        format!("tracks/point_indexes.{observation_count}.uint32.zst")
    };
    let point_indexes_vec: Vec<u32> =
        read_binary_array(&mut archive, &point_indexes_name, observation_count)?;
    let point_indexes = Array1::from_vec(point_indexes_vec);

    let observation_counts_vec: Vec<u32> = read_binary_array(
        &mut archive,
        &format!("tracks/observation_counts.{point_count}.uint32.zst"),
        point_count,
    )?;
    let observation_counts = Array1::from_vec(observation_counts_vec);

    // Rigs and frames (optional — only present in rig-aware reconstructions)
    let rig_frame_data = if archive.index_for_name("rigs/metadata.json.zst").is_some() {
        let rigs_metadata: RigsMetadata = read_json_entry(&mut archive, "rigs/metadata.json.zst")?;
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
            &format!("rigs/sensor_camera_indexes.{sensor_count}.uint32.zst"),
            sensor_count,
        )?;
        let sensor_camera_indexes = Array1::from_vec(sensor_camera_indexes_vec);

        let sensor_quaternions_vec: Vec<f64> = read_binary_array(
            &mut archive,
            &format!("rigs/sensor_quaternions_wxyz.{sensor_count}.4.float64.zst"),
            sensor_count * 4,
        )?;
        let sensor_quaternions_wxyz =
            Array2::from_shape_vec((sensor_count, 4), sensor_quaternions_vec).map_err(|e| {
                SfmrError::ShapeMismatch(format!("sensor quaternions reshape: {e}"))
            })?;

        let sensor_translations_vec: Vec<f64> = read_binary_array(
            &mut archive,
            &format!("rigs/sensor_translations_xyz.{sensor_count}.3.float64.zst"),
            sensor_count * 3,
        )?;
        let sensor_translations_xyz =
            Array2::from_shape_vec((sensor_count, 3), sensor_translations_vec).map_err(|e| {
                SfmrError::ShapeMismatch(format!("sensor translations reshape: {e}"))
            })?;

        // Frames
        let frames_metadata: FramesMetadata =
            read_json_entry(&mut archive, "frames/metadata.json.zst")?;
        let frame_count = frames_metadata.frame_count as usize;

        let rig_indexes_vec: Vec<u32> = read_binary_array(
            &mut archive,
            &format!("frames/rig_indexes.{frame_count}.uint32.zst"),
            frame_count,
        )?;
        let rig_indexes = Array1::from_vec(rig_indexes_vec);

        let image_sensor_indexes_vec: Vec<u32> = read_binary_array(
            &mut archive,
            &format!("frames/image_sensor_indexes.{image_count}.uint32.zst"),
            image_count,
        )?;
        let image_sensor_indexes = Array1::from_vec(image_sensor_indexes_vec);

        let image_frame_indexes_vec: Vec<u32> = read_binary_array(
            &mut archive,
            &format!("frames/image_frame_indexes.{image_count}.uint32.zst"),
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

    // Upgrade version-1 metadata to the version-2 model: the in-memory data is
    // always version 2. Recompute `infinity_point_count` from the `w` column so
    // it is correct regardless of the source version (version 1 has none).
    metadata.version = 2;
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
        thumbnails_y_x_rgb,
        positions_xyzw,
        colors_rgb,
        reprojection_errors,
        estimated_normals_xyz,
        image_indexes,
        feature_indexes,
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
