// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sfmr` file integrity verification.

use std::path::Path;

use xxhash_rust::xxh3::Xxh3;

use crate::archive_io::{format_hash, read_zst_entry};
use crate::types::*;

/// Verify integrity of a `.sfmr` file using content hashes.
///
/// Returns `Ok((true, []))` if all hashes match, `Ok((false, errors))` with
/// details if verification fails. Returns `Err` only for I/O errors.
pub fn verify_sfmr(path: &Path) -> Result<(bool, Vec<String>), SfmrError> {
    let file = std::fs::File::open(path).map_err(|e| SfmrError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut errors = Vec::new();

    // Read stored hashes
    let content_hash_bytes = read_zst_entry(&mut archive, "content_hash.json.zst")?;
    let stored: ContentHash = serde_json::from_slice(&content_hash_bytes)?;

    // Read metadata for counts
    let metadata_raw = read_zst_entry(&mut archive, "metadata.json.zst")?;
    let metadata: SfmrMetadata = serde_json::from_slice(&metadata_raw)?;
    let image_count = metadata.image_count as usize;
    let points3d_count = metadata.points3d_count as usize;
    let observation_count = metadata.observation_count as usize;

    let mut section_digests: Vec<u128> = Vec::with_capacity(5);

    // === Metadata hash (raw bytes, not re-serialized) ===
    let metadata_hash = xxhash_rust::xxh3::xxh3_128(&metadata_raw);
    if format_hash(metadata_hash) != stored.metadata_xxh128 {
        errors.push(format!(
            "Metadata hash mismatch: computed {}, stored {}",
            format_hash(metadata_hash),
            stored.metadata_xxh128
        ));
    }
    section_digests.push(metadata_hash);

    // === Cameras hash ===
    let cameras_raw = read_zst_entry(&mut archive, "cameras/metadata.json.zst")?;
    let cameras_hash = xxhash_rust::xxh3::xxh3_128(&cameras_raw);
    if format_hash(cameras_hash) != stored.cameras_xxh128 {
        errors.push(format!(
            "Cameras hash mismatch: computed {}, stored {}",
            format_hash(cameras_hash),
            stored.cameras_xxh128
        ));
    }
    section_digests.push(cameras_hash);

    // === Rigs hash (optional, lexicographic path order) ===
    let has_rigs = archive.index_for_name("rigs/metadata.json.zst").is_some();
    if has_rigs {
        let mut rigs_hasher = Xxh3::new();

        // Read rigs metadata first to get sensor_count
        let rigs_meta_raw = read_zst_entry(&mut archive, "rigs/metadata.json.zst")?;
        let rigs_meta: RigsMetadata = serde_json::from_slice(&rigs_meta_raw)?;
        let sensor_count = rigs_meta.sensor_count;

        // rigs/metadata.json
        rigs_hasher.update(&rigs_meta_raw);
        // rigs/sensor_camera_indexes
        rigs_hasher.update(&read_zst_entry(
            &mut archive,
            &format!("rigs/sensor_camera_indexes.{sensor_count}.uint32.zst"),
        )?);
        // rigs/sensor_quaternions_wxyz
        rigs_hasher.update(&read_zst_entry(
            &mut archive,
            &format!("rigs/sensor_quaternions_wxyz.{sensor_count}.4.float64.zst"),
        )?);
        // rigs/sensor_translations_xyz
        rigs_hasher.update(&read_zst_entry(
            &mut archive,
            &format!("rigs/sensor_translations_xyz.{sensor_count}.3.float64.zst"),
        )?);

        let rigs_hash = rigs_hasher.digest128();
        if let Some(stored_rigs) = &stored.rigs_xxh128 {
            if &format_hash(rigs_hash) != stored_rigs {
                errors.push(format!(
                    "Rigs hash mismatch: computed {}, stored {}",
                    format_hash(rigs_hash),
                    stored_rigs
                ));
            }
        }
        section_digests.push(rigs_hash);

        // === Frames hash (lexicographic path order) ===
        let mut frames_hasher = Xxh3::new();

        let frames_meta_raw = read_zst_entry(&mut archive, "frames/metadata.json.zst")?;
        let frames_meta: FramesMetadata = serde_json::from_slice(&frames_meta_raw)?;
        let frame_count = frames_meta.frame_count;

        // frames/image_frame_indexes
        frames_hasher.update(&read_zst_entry(
            &mut archive,
            &format!("frames/image_frame_indexes.{image_count}.uint32.zst"),
        )?);
        // frames/image_sensor_indexes
        frames_hasher.update(&read_zst_entry(
            &mut archive,
            &format!("frames/image_sensor_indexes.{image_count}.uint32.zst"),
        )?);
        // frames/metadata.json
        frames_hasher.update(&frames_meta_raw);
        // frames/rig_indexes
        frames_hasher.update(&read_zst_entry(
            &mut archive,
            &format!("frames/rig_indexes.{frame_count}.uint32.zst"),
        )?);

        let frames_hash = frames_hasher.digest128();
        if let Some(stored_frames) = &stored.frames_xxh128 {
            if &format_hash(frames_hash) != stored_frames {
                errors.push(format!(
                    "Frames hash mismatch: computed {}, stored {}",
                    format_hash(frames_hash),
                    stored_frames
                ));
            }
        }
        section_digests.push(frames_hash);
    }

    // === Images hash (lexicographic path order) ===
    let mut images_hasher = Xxh3::new();

    // Read depth_statistics first to get num_buckets
    let depth_stats_raw = read_zst_entry(&mut archive, "images/depth_statistics.json.zst")?;
    let depth_stats: DepthStatistics = serde_json::from_slice(&depth_stats_raw)?;
    let num_buckets = depth_stats.num_histogram_buckets;

    // images/camera_indexes
    images_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("images/camera_indexes.{image_count}.uint32.zst"),
    )?);
    // images/depth_statistics.json
    images_hasher.update(&depth_stats_raw);
    // images/feature_tool_hashes
    images_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("images/feature_tool_hashes.{image_count}.uint128.zst"),
    )?);
    // images/metadata.json
    images_hasher.update(&read_zst_entry(&mut archive, "images/metadata.json.zst")?);
    // images/names.json
    images_hasher.update(&read_zst_entry(&mut archive, "images/names.json.zst")?);
    // images/observed_depth_histogram_counts
    images_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("images/observed_depth_histogram_counts.{image_count}.{num_buckets}.uint32.zst"),
    )?);
    // images/quaternions_wxyz
    images_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("images/quaternions_wxyz.{image_count}.4.float64.zst"),
    )?);
    // images/sift_content_hashes
    images_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("images/sift_content_hashes.{image_count}.uint128.zst"),
    )?);
    // images/thumbnails_y_x_rgb
    images_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("images/thumbnails_y_x_rgb.{image_count}.128.128.3.uint8.zst"),
    )?);
    // images/translations_xyz
    images_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("images/translations_xyz.{image_count}.3.float64.zst"),
    )?);

    let images_hash = images_hasher.digest128();
    if format_hash(images_hash) != stored.images_xxh128 {
        errors.push(format!(
            "Images hash mismatch: computed {}, stored {}",
            format_hash(images_hash),
            stored.images_xxh128
        ));
    }
    section_digests.push(images_hash);

    // === Points3D hash (lexicographic path order) ===
    let mut points3d_hasher = Xxh3::new();

    // points3d/colors_rgb
    points3d_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("points3d/colors_rgb.{points3d_count}.3.uint8.zst"),
    )?);
    // points3d/estimated_normals_xyz
    points3d_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("points3d/estimated_normals_xyz.{points3d_count}.3.float32.zst"),
    )?);
    // points3d/metadata.json
    points3d_hasher.update(&read_zst_entry(&mut archive, "points3d/metadata.json.zst")?);
    // points3d/positions_xyz
    points3d_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("points3d/positions_xyz.{points3d_count}.3.float64.zst"),
    )?);
    // points3d/reprojection_errors
    points3d_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("points3d/reprojection_errors.{points3d_count}.float32.zst"),
    )?);

    let points3d_hash = points3d_hasher.digest128();
    if format_hash(points3d_hash) != stored.points3d_xxh128 {
        errors.push(format!(
            "Points3D hash mismatch: computed {}, stored {}",
            format_hash(points3d_hash),
            stored.points3d_xxh128
        ));
    }
    section_digests.push(points3d_hash);

    // === Tracks hash (lexicographic path order) ===
    let mut tracks_hasher = Xxh3::new();

    // tracks/feature_indexes
    tracks_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("tracks/feature_indexes.{observation_count}.uint32.zst"),
    )?);
    // tracks/image_indexes
    let track_image_indexes_raw = read_zst_entry(
        &mut archive,
        &format!("tracks/image_indexes.{observation_count}.uint32.zst"),
    )?;
    tracks_hasher.update(&track_image_indexes_raw);
    // tracks/metadata.json
    tracks_hasher.update(&read_zst_entry(&mut archive, "tracks/metadata.json.zst")?);
    // tracks/observation_counts
    let track_obs_counts_raw = read_zst_entry(
        &mut archive,
        &format!("tracks/observation_counts.{points3d_count}.uint32.zst"),
    )?;
    tracks_hasher.update(&track_obs_counts_raw);
    // tracks/points3d_indexes
    let track_points3d_indexes_raw = read_zst_entry(
        &mut archive,
        &format!("tracks/points3d_indexes.{observation_count}.uint32.zst"),
    )?;
    tracks_hasher.update(&track_points3d_indexes_raw);

    let tracks_hash = tracks_hasher.digest128();
    if format_hash(tracks_hash) != stored.tracks_xxh128 {
        errors.push(format!(
            "Tracks hash mismatch: computed {}, stored {}",
            format_hash(tracks_hash),
            stored.tracks_xxh128
        ));
    }
    section_digests.push(tracks_hash);

    // === Validate track sorting ===
    if observation_count > 1 {
        let p3d_idxs: &[u32] = bytemuck::cast_slice(&track_points3d_indexes_raw);
        let img_idxs: &[u32] = bytemuck::cast_slice(&track_image_indexes_raw);
        for i in 0..observation_count - 1 {
            if p3d_idxs[i] > p3d_idxs[i + 1]
                || (p3d_idxs[i] == p3d_idxs[i + 1] && img_idxs[i] > img_idxs[i + 1])
            {
                errors.push(format!(
                    "Tracks not sorted lexicographically by (points3d_indexes, image_indexes) at index {i}"
                ));
                break;
            }
        }
    }

    // === Validate observation_counts ===
    if points3d_count > 0 {
        let obs_counts: &[u32] = bytemuck::cast_slice(&track_obs_counts_raw);
        let obs_sum: u64 = obs_counts.iter().map(|&c| c as u64).sum();
        if obs_sum != observation_count as u64 {
            errors.push(format!(
                "Sum of observation_counts ({obs_sum}) != observation_count ({observation_count})"
            ));
        }
        if obs_counts.iter().any(|&c| c < 1) {
            errors.push("observation_counts contains values < 1".into());
        }
    }

    // === Overall content hash ===
    let all_digests_bytes: Vec<u8> = section_digests
        .iter()
        .flat_map(|d| d.to_be_bytes())
        .collect();
    let content_hash_value = xxhash_rust::xxh3::xxh3_128(&all_digests_bytes);
    if format_hash(content_hash_value) != stored.content_xxh128 {
        errors.push(format!(
            "Overall content hash mismatch: computed {}, stored {}",
            format_hash(content_hash_value),
            stored.content_xxh128
        ));
    }

    Ok((errors.is_empty(), errors))
}