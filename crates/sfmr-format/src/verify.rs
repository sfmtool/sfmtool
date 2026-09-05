// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sfmr` file integrity verification.

use std::path::Path;

use xxhash_rust::xxh3::Xxh3;

use crate::entries;
use crate::types::*;
use sfmtool_archive_io::{format_hash, raw_to_u32, read_zst_entry};

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
    let content_hash_bytes = read_zst_entry(&mut archive, entries::content_hash())?;
    let stored: ContentHash = serde_json::from_slice(&content_hash_bytes)?;

    // Read metadata for counts
    let metadata_raw = read_zst_entry(&mut archive, entries::metadata())?;
    let metadata: SfmrMetadata = serde_json::from_slice(&metadata_raw)?;
    let image_count = metadata.image_count as usize;
    let point_count = metadata.point_count as usize;
    let observation_count = metadata.observation_count as usize;
    // Version 1 stored Euclidean `positions_xyz` and `points3d_indexes`;
    // version 2 stores homogeneous `positions_xyzw` and `point_indexes`.
    let is_v1 = metadata.version < 2;
    // Version 3 renamed `points3d/estimated_normals_xyz` to
    // `points3d/normals_xyz` and added the optional per-point patch frame in the
    // points3d section.
    let is_pre_v3 = metadata.version < 3;
    // Version 4 added `embedded_patches`: per-observation keypoints inline and a
    // direct image-bytes hash, in place of the `.sift`-link arrays.
    match metadata.feature_source.as_str() {
        FEATURE_SOURCE_SIFT_FILES | FEATURE_SOURCE_EMBEDDED_PATCHES => {}
        other => errors.push(format!(
            "unknown feature_source {other:?} (expected {FEATURE_SOURCE_SIFT_FILES:?} \
             or {FEATURE_SOURCE_EMBEDDED_PATCHES:?})"
        )),
    }
    let is_embedded = metadata.feature_source == FEATURE_SOURCE_EMBEDDED_PATCHES;

    // A version newer than this build understands has an unknown layout; report it
    // and stop rather than emit confusing per-file hash mismatches.
    if metadata.version > SFMR_FORMAT_VERSION {
        errors.push(format!(
            "unsupported .sfmr format version {} (this build supports up to \
             {SFMR_FORMAT_VERSION})",
            metadata.version
        ));
        return Ok((false, errors));
    }

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
    let cameras_raw = read_zst_entry(&mut archive, entries::cameras_metadata())?;
    // Parsed lazily for the embedded-patches keypoint bounds check below.
    let cameras: Vec<SfmrCamera> = serde_json::from_slice(&cameras_raw).unwrap_or_default();
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
    let has_rigs = archive.index_for_name(entries::rigs_metadata()).is_some();
    if has_rigs {
        let mut rigs_hasher = Xxh3::new();

        // Read rigs metadata first to get sensor_count
        let rigs_meta_raw = read_zst_entry(&mut archive, entries::rigs_metadata())?;
        let rigs_meta: RigsMetadata = serde_json::from_slice(&rigs_meta_raw)?;
        let sensor_count = rigs_meta.sensor_count;

        // rigs/metadata.json
        rigs_hasher.update(&rigs_meta_raw);
        // rigs/sensor_camera_indexes
        rigs_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::rigs_sensor_camera_indexes(sensor_count),
        )?);
        // rigs/sensor_quaternions_wxyz
        rigs_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::rigs_sensor_quaternions_wxyz(sensor_count),
        )?);
        // rigs/sensor_translations_xyz
        rigs_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::rigs_sensor_translations_xyz(sensor_count),
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

        let frames_meta_raw = read_zst_entry(&mut archive, entries::frames_metadata())?;
        let frames_meta: FramesMetadata = serde_json::from_slice(&frames_meta_raw)?;
        let frame_count = frames_meta.frame_count;

        // frames/image_frame_indexes
        frames_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::frames_image_frame_indexes(image_count),
        )?);
        // frames/image_sensor_indexes
        frames_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::frames_image_sensor_indexes(image_count),
        )?);
        // frames/metadata.json
        frames_hasher.update(&frames_meta_raw);
        // frames/rig_indexes
        frames_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::frames_rig_indexes(frame_count),
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
    let depth_stats_raw = read_zst_entry(&mut archive, entries::images_depth_statistics())?;
    let depth_stats: DepthStatistics = serde_json::from_slice(&depth_stats_raw)?;
    let num_buckets = depth_stats.num_histogram_buckets;

    // images/camera_indexes (captured for the keypoint bounds check below)
    let camera_indexes_raw =
        read_zst_entry(&mut archive, &entries::images_camera_indexes(image_count))?;
    images_hasher.update(&camera_indexes_raw);
    // images/depth_statistics.json
    images_hasher.update(&depth_stats_raw);
    // images/feature_tool_hashes (sift_files) or image_file_hashes
    // (embedded_patches) — same lexicographic slot, mirrors the writer.
    if is_embedded {
        images_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::images_image_file_hashes(image_count),
        )?);
    } else {
        images_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::images_feature_tool_hashes(image_count),
        )?);
    }
    // images/metadata.json
    images_hasher.update(&read_zst_entry(&mut archive, entries::images_metadata())?);
    // images/names.json
    images_hasher.update(&read_zst_entry(&mut archive, entries::images_names())?);
    // images/observed_depth_histogram_counts
    images_hasher.update(&read_zst_entry(
        &mut archive,
        &entries::images_observed_depth_histogram_counts(image_count, num_buckets),
    )?);
    // images/quaternions_wxyz
    images_hasher.update(&read_zst_entry(
        &mut archive,
        &entries::images_quaternions_wxyz(image_count),
    )?);
    // images/sift_content_hashes (sift_files only)
    if !is_embedded {
        images_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::images_sift_content_hashes(image_count),
        )?);
    }
    // images/thumbnails_y_x_rgb
    images_hasher.update(&read_zst_entry(
        &mut archive,
        &entries::images_thumbnails_y_x_rgb(image_count),
    )?);
    // images/translations_xyz
    images_hasher.update(&read_zst_entry(
        &mut archive,
        &entries::images_translations_xyz(image_count),
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
    // The optional per-point patch frame (`patch_*`) lives in this section.
    let points3d_meta_raw = read_zst_entry(&mut archive, entries::points3d_metadata())?;
    let points3d_meta: serde_json::Value = serde_json::from_slice(&points3d_meta_raw)?;
    // Normals are optional in version 3 (default `false`); versions 1 and 2
    // always carry them.
    let has_normals = is_pre_v3
        || points3d_meta
            .get("has_normals")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
    // Normal confidence is optional from version 5 (default `false`).
    let has_normal_confidence = points3d_meta
        .get("has_normal_confidence")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let has_uv_frames = points3d_meta
        .get("has_uv_frames")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let has_patch_bitmaps = points3d_meta
        .get("has_patch_bitmaps")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let patch_bitmap_r = points3d_meta
        .get("patch_bitmap_resolution")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let mut points3d_hasher = Xxh3::new();

    // points3d/colors_rgb
    points3d_hasher.update(&read_zst_entry(
        &mut archive,
        &entries::points3d_colors_rgb(point_count),
    )?);
    // points3d/metadata.json
    points3d_hasher.update(&points3d_meta_raw);
    // points3d/normal_confidence (optional, version 5+; sorts before normals_xyz)
    if has_normal_confidence {
        points3d_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::points3d_normal_confidence(point_count),
        )?);
    }
    // points3d/normals_xyz (optional; named estimated_normals_xyz in versions 1-2)
    if has_normals {
        let normals_name = entries::points3d_normals(is_pre_v3, point_count);
        points3d_hasher.update(&read_zst_entry(&mut archive, &normals_name)?);
    }
    // Optional patch frame, in lexicographic order: bitmaps, u, v.
    if has_uv_frames {
        if has_patch_bitmaps {
            points3d_hasher.update(&read_zst_entry(
                &mut archive,
                &entries::points3d_patch_bitmaps_y_x_rgba(point_count, patch_bitmap_r),
            )?);
        }
        points3d_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::points3d_patch_u_halfvec_xyz(point_count),
        )?);
        points3d_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::points3d_patch_v_halfvec_xyz(point_count),
        )?);
    }
    // points3d/positions_xyz (version 1) or positions_xyzw (version 2)
    let positions_name = entries::points3d_positions(is_v1, point_count);
    let positions_raw = read_zst_entry(&mut archive, &positions_name)?;
    points3d_hasher.update(&positions_raw);
    // points3d/reprojection_errors
    points3d_hasher.update(&read_zst_entry(
        &mut archive,
        &entries::points3d_reprojection_errors(point_count),
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

    // === Validate homogeneous positions (version 2 only) ===
    // No coordinate may be NaN/infinite; a w=0 row must be a non-zero
    // direction; and the count of w=0 rows must match infinity_point_count.
    if !is_v1 && point_count > 0 {
        if positions_raw.len() == point_count * 32 {
            let read_f64 = |off: usize| -> f64 {
                f64::from_le_bytes(positions_raw[off..off + 8].try_into().unwrap())
            };
            let mut infinity_count = 0u32;
            for i in 0..point_count {
                let base = i * 32;
                let x = read_f64(base);
                let y = read_f64(base + 8);
                let z = read_f64(base + 16);
                let w = read_f64(base + 24);
                if !x.is_finite() || !y.is_finite() || !z.is_finite() || !w.is_finite() {
                    errors.push(format!(
                        "positions_xyzw row {i} contains a NaN or infinite value"
                    ));
                } else if w == 0.0 {
                    infinity_count += 1;
                    if x == 0.0 && y == 0.0 && z == 0.0 {
                        errors.push(format!(
                            "positions_xyzw row {i} has w=0 but a zero direction"
                        ));
                    }
                }
            }
            if infinity_count != metadata.infinity_point_count {
                errors.push(format!(
                    "infinity_point_count ({}) != number of w=0 points ({infinity_count})",
                    metadata.infinity_point_count
                ));
            }
        } else {
            errors.push(format!(
                "positions_xyzw byte length {} != point_count {point_count} * 32",
                positions_raw.len()
            ));
        }
    }

    // === Tracks hash (lexicographic path order) ===
    let mut tracks_hasher = Xxh3::new();

    // tracks/metadata.json is read ahead of the section — its `has_*` flags say
    // which optional columns the archive carries — but fed to the hasher below in
    // its own lexicographic slot, after `keypoints_xy`.
    let tracks_meta_raw = read_zst_entry(&mut archive, entries::tracks_metadata())?;
    let tracks_meta: serde_json::Value =
        serde_json::from_slice(&tracks_meta_raw).unwrap_or(serde_json::Value::Null);
    // The inline keypoint column is the coordinate in an `embedded_patches` file
    // and an optional copy of it in a `sift_files` one; `has_keypoints_xy` is what
    // says whether the entry exists.
    let has_keypoints_xy = is_embedded
        || tracks_meta
            .get("has_keypoints_xy")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
    // Per-observation confidence is optional from version 6 (default `false`).
    let has_observation_confidence = tracks_meta
        .get("has_observation_confidence")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // tracks/feature_indexes (sift_files only; sorts before image_indexes)
    if !is_embedded {
        tracks_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::tracks_feature_indexes(observation_count),
        )?);
    }
    // tracks/image_indexes
    let track_image_indexes_raw = read_zst_entry(
        &mut archive,
        &entries::tracks_image_indexes(observation_count),
    )?;
    tracks_hasher.update(&track_image_indexes_raw);
    // tracks/keypoints_xy (optional; sorts after image_indexes)
    let keypoints_raw = if has_keypoints_xy {
        let raw = read_zst_entry(
            &mut archive,
            &entries::tracks_keypoints_xy(observation_count),
        )?;
        tracks_hasher.update(&raw);
        Some(raw)
    } else {
        None
    };
    // tracks/metadata.json
    tracks_hasher.update(&tracks_meta_raw);
    // tracks/observation_confidence (optional, version 6+; sorts after
    // metadata.json, before observation_counts)
    if has_observation_confidence {
        tracks_hasher.update(&read_zst_entry(
            &mut archive,
            &entries::tracks_observation_confidence(observation_count),
        )?);
    }
    // tracks/observation_counts
    let track_obs_counts_raw = read_zst_entry(
        &mut archive,
        &entries::tracks_observation_counts(point_count),
    )?;
    tracks_hasher.update(&track_obs_counts_raw);
    // tracks/points3d_indexes (version 1) or point_indexes (version 2)
    let point_indexes_name = entries::tracks_point_indexes(is_v1, observation_count);
    let track_point_indexes_raw = read_zst_entry(&mut archive, &point_indexes_name)?;
    tracks_hasher.update(&track_point_indexes_raw);

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
        let p3d_idxs = raw_to_u32(&track_point_indexes_raw);
        let img_idxs = raw_to_u32(&track_image_indexes_raw);
        for i in 0..observation_count - 1 {
            if p3d_idxs[i] > p3d_idxs[i + 1]
                || (p3d_idxs[i] == p3d_idxs[i + 1] && img_idxs[i] > img_idxs[i + 1])
            {
                errors.push(format!(
                    "Tracks not sorted lexicographically by (point_indexes, image_indexes) at index {i}"
                ));
                break;
            }
        }
    }

    // === Validate observation_counts ===
    if point_count > 0 {
        let obs_counts = raw_to_u32(&track_obs_counts_raw);
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

    // === Validate feature_source vs tracks-metadata flags (v4) ===
    // `has_feature_indexes` is true exactly for a `sift_files` file.
    // `has_keypoints_xy` is a presence flag rather than a mode flag: an
    // `embedded_patches` file must set it (the column is its coordinate), while a
    // `sift_files` file sets it only when it carries the optional inline copy.
    // Only enforced when the flags are present (v4 writers always emit them; an
    // upgraded pre-v4 file has none).
    if let Some(hfi) = tracks_meta
        .get("has_feature_indexes")
        .and_then(|v| v.as_bool())
    {
        if hfi == is_embedded {
            errors.push(format!(
                "tracks/metadata.json has_feature_indexes={hfi} contradicts \
                 feature_source (embedded={is_embedded})"
            ));
        }
    }
    if let Some(hk) = tracks_meta
        .get("has_keypoints_xy")
        .and_then(|v| v.as_bool())
    {
        if is_embedded && !hk {
            errors.push(
                "tracks/metadata.json has_keypoints_xy=false contradicts \
                 feature_source \"embedded_patches\", whose keypoints_xy is the \
                 observation coordinate"
                    .to_string(),
            );
        }
    }

    // === Validate the inline keypoints (finite + within image bounds) ===
    // Each keypoint row is two little-endian f32s (u, v).
    const KEYPOINT_ROW_BYTES: usize = 2 * std::mem::size_of::<f32>();
    if let Some(kp_raw) = &keypoints_raw {
        if kp_raw.len() == observation_count * KEYPOINT_ROW_BYTES {
            let floats: Vec<f32> = kp_raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let kp = ndarray::Array2::from_shape_vec((observation_count, 2), floats).unwrap();
            let img_idx = raw_to_u32(&track_image_indexes_raw);
            let cam_idx = raw_to_u32(&camera_indexes_raw);
            if let Err(e) = validate_keypoints(&kp, img_idx.as_ref(), cam_idx.as_ref(), &cameras) {
                errors.push(e);
            }
        } else {
            errors.push(format!(
                "keypoints_xy byte length {} != observation_count {observation_count} * \
                 {KEYPOINT_ROW_BYTES}",
                kp_raw.len()
            ));
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
