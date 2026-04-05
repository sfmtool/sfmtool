// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sfmr` file writing.

use std::borrow::Cow;
use std::path::Path;

use xxhash_rust::xxh3::Xxh3;
use zip::ZipWriter;

use crate::archive_io::{format_hash, write_binary_entry, write_json_entry};

use crate::depth_stats::{compute_depth_statistics, DepthStatsResult};
use crate::types::*;

/// Options for writing a `.sfmr` file.
#[derive(Debug, Clone)]
pub struct WriteOptions {
    /// Zstandard compression level for all entries. Default is 3.
    pub zstd_level: i32,
    /// If true, skip recomputing depth statistics and use the values from `SfmrData` as-is.
    /// Default is false (depth statistics are always recomputed for trustworthiness).
    pub skip_recompute_depth_stats: bool,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            zstd_level: 3,
            skip_recompute_depth_stats: false,
        }
    }
}

/// Write columnar data to a `.sfmr` file.
///
/// Recomputes depth statistics (normals, histograms, per-image stats) by default
/// to ensure they are consistent with the current poses, points, and tracks.
/// Use [`write_sfmr_with_options`] to skip recomputation if needed.
///
/// Sorts tracks by `(points3d_indexes, image_indexes)` if not already sorted.
/// Computes content hashes and writes all section metadata files.
/// The `content_hash` field in `data` is ignored on write (recomputed).
pub fn write_sfmr(path: &Path, data: &mut SfmrData) -> Result<(), SfmrError> {
    write_sfmr_with_options(path, data, &WriteOptions::default())
}

/// Write columnar data to a `.sfmr` file with explicit options.
///
/// Sorts tracks by `(points3d_indexes, image_indexes)` if not already sorted.
/// See [`WriteOptions`] for available options. By default, depth statistics
/// are recomputed from the reconstruction data to ensure trustworthiness.
pub fn write_sfmr_with_options(
    path: &Path,
    data: &mut SfmrData,
    options: &WriteOptions,
) -> Result<(), SfmrError> {
    // Ensure tracks are sorted by (points3d_indexes, image_indexes)
    ensure_tracks_sorted(data);

    // Recompute depth statistics unless explicitly skipped
    let recomputed: Option<DepthStatsResult>;
    let (depth_statistics, estimated_normals_xyz, observed_depth_histogram_counts) =
        if options.skip_recompute_depth_stats {
            (
                &data.depth_statistics,
                Cow::Borrowed(&data.estimated_normals_xyz),
                Cow::Borrowed(&data.observed_depth_histogram_counts),
            )
        } else {
            recomputed = Some(compute_depth_statistics(
                &data.quaternions_wxyz,
                &data.translations_xyz,
                &data.positions_xyz,
                &data.image_indexes,
                &data.points3d_indexes,
            )?);
            let r = recomputed.as_ref().unwrap();
            (
                &r.depth_statistics,
                Cow::Borrowed(&r.estimated_normals_xyz),
                Cow::Borrowed(&r.observed_depth_histogram_counts),
            )
        };

    let image_count = data.metadata.image_count as usize;
    let points3d_count = data.metadata.points3d_count as usize;
    let observation_count = data.metadata.observation_count as usize;
    let num_buckets = depth_statistics.num_histogram_buckets as usize;

    // Validate dimensions (use possibly-recomputed depth stats)
    validate_dimensions_with(
        data,
        &estimated_normals_xyz,
        &observed_depth_histogram_counts,
        depth_statistics,
        image_count,
        points3d_count,
        observation_count,
        num_buckets,
    )?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| SfmrError::IoPath {
            operation: "Failed to create parent directory",
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    let file = std::fs::File::create(path).map_err(|e| SfmrError::IoPath {
        operation: "Failed to create file",
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut zip = ZipWriter::new(file);
    let has_rigs = data.rig_frame_data.is_some();
    let mut section_digests: Vec<u128> = Vec::with_capacity(if has_rigs { 7 } else { 5 });

    // === Top-level metadata ===
    let metadata_bytes = write_json_entry(
        &mut zip,
        "metadata.json.zst",
        &data.metadata,
        options.zstd_level,
    )?;
    let metadata_hash = xxhash_rust::xxh3::xxh3_128(&metadata_bytes);
    section_digests.push(metadata_hash);

    // === Cameras ===
    let cameras_bytes = write_json_entry(
        &mut zip,
        "cameras/metadata.json.zst",
        &data.cameras,
        options.zstd_level,
    )?;
    let cameras_hash = xxhash_rust::xxh3::xxh3_128(&cameras_bytes);
    section_digests.push(cameras_hash);

    // === Rigs (optional, hashed in lexicographic path order) ===
    let rigs_hash: Option<u128>;
    let frames_hash: Option<u128>;
    if let Some(rf) = &data.rig_frame_data {
        let sensor_count = rf.rigs_metadata.sensor_count as usize;
        let frame_count = rf.frames_metadata.frame_count as usize;

        let mut rigs_hasher = Xxh3::new();

        // rigs/metadata.json
        let bytes = write_json_entry(
            &mut zip,
            "rigs/metadata.json.zst",
            &rf.rigs_metadata,
            options.zstd_level,
        )?;
        rigs_hasher.update(&bytes);

        // rigs/sensor_camera_indexes
        let bytes = write_binary_entry(
            &mut zip,
            &format!("rigs/sensor_camera_indexes.{sensor_count}.uint32.zst"),
            bytemuck::cast_slice(rf.sensor_camera_indexes.as_slice().unwrap()),
            options.zstd_level,
        )?;
        rigs_hasher.update(&bytes);

        // rigs/sensor_quaternions_wxyz
        let bytes = write_binary_entry(
            &mut zip,
            &format!("rigs/sensor_quaternions_wxyz.{sensor_count}.4.float64.zst"),
            bytemuck::cast_slice(rf.sensor_quaternions_wxyz.as_slice().unwrap()),
            options.zstd_level,
        )?;
        rigs_hasher.update(&bytes);

        // rigs/sensor_translations_xyz
        let bytes = write_binary_entry(
            &mut zip,
            &format!("rigs/sensor_translations_xyz.{sensor_count}.3.float64.zst"),
            bytemuck::cast_slice(rf.sensor_translations_xyz.as_slice().unwrap()),
            options.zstd_level,
        )?;
        rigs_hasher.update(&bytes);

        let rigs_digest = rigs_hasher.digest128();
        section_digests.push(rigs_digest);
        rigs_hash = Some(rigs_digest);

        // === Frames (hashed in lexicographic path order) ===
        let mut frames_hasher = Xxh3::new();

        // frames/image_frame_indexes
        let bytes = write_binary_entry(
            &mut zip,
            &format!("frames/image_frame_indexes.{image_count}.uint32.zst"),
            bytemuck::cast_slice(rf.image_frame_indexes.as_slice().unwrap()),
            options.zstd_level,
        )?;
        frames_hasher.update(&bytes);

        // frames/image_sensor_indexes
        let bytes = write_binary_entry(
            &mut zip,
            &format!("frames/image_sensor_indexes.{image_count}.uint32.zst"),
            bytemuck::cast_slice(rf.image_sensor_indexes.as_slice().unwrap()),
            options.zstd_level,
        )?;
        frames_hasher.update(&bytes);

        // frames/metadata.json
        let bytes = write_json_entry(
            &mut zip,
            "frames/metadata.json.zst",
            &rf.frames_metadata,
            options.zstd_level,
        )?;
        frames_hasher.update(&bytes);

        // frames/rig_indexes
        let bytes = write_binary_entry(
            &mut zip,
            &format!("frames/rig_indexes.{frame_count}.uint32.zst"),
            bytemuck::cast_slice(rf.rig_indexes.as_slice().unwrap()),
            options.zstd_level,
        )?;
        frames_hasher.update(&bytes);

        let frames_digest = frames_hasher.digest128();
        section_digests.push(frames_digest);
        frames_hash = Some(frames_digest);
    } else {
        rigs_hash = None;
        frames_hash = None;
    }

    // === Images (hashed in lexicographic path order) ===
    let mut images_hasher = Xxh3::new();

    // images/camera_indexes
    let bytes = write_binary_entry(
        &mut zip,
        &format!("images/camera_indexes.{image_count}.uint32.zst"),
        bytemuck::cast_slice(data.camera_indexes.as_slice().unwrap()),
        options.zstd_level,
    )?;
    images_hasher.update(&bytes);

    // images/depth_statistics.json
    let bytes = write_json_entry(
        &mut zip,
        "images/depth_statistics.json.zst",
        depth_statistics,
        options.zstd_level,
    )?;
    images_hasher.update(&bytes);

    // images/feature_tool_hashes
    let hash_bytes: Vec<u8> = data
        .feature_tool_hashes
        .iter()
        .flat_map(|h| h.iter().copied())
        .collect();
    let bytes = write_binary_entry(
        &mut zip,
        &format!("images/feature_tool_hashes.{image_count}.uint128.zst"),
        &hash_bytes,
        options.zstd_level,
    )?;
    images_hasher.update(&bytes);

    // images/metadata.json
    let images_meta = serde_json::json!({"image_count": image_count, "thumbnail_size": 128});
    let bytes = write_json_entry(
        &mut zip,
        "images/metadata.json.zst",
        &images_meta,
        options.zstd_level,
    )?;
    images_hasher.update(&bytes);

    // images/names.json
    let bytes = write_json_entry(
        &mut zip,
        "images/names.json.zst",
        &data.image_names,
        options.zstd_level,
    )?;
    images_hasher.update(&bytes);

    // images/observed_depth_histogram_counts
    let bytes = write_binary_entry(
        &mut zip,
        &format!("images/observed_depth_histogram_counts.{image_count}.{num_buckets}.uint32.zst"),
        bytemuck::cast_slice(observed_depth_histogram_counts.as_slice().unwrap()),
        options.zstd_level,
    )?;
    images_hasher.update(&bytes);

    // images/quaternions_wxyz
    let bytes = write_binary_entry(
        &mut zip,
        &format!("images/quaternions_wxyz.{image_count}.4.float64.zst"),
        bytemuck::cast_slice(data.quaternions_wxyz.as_slice().unwrap()),
        options.zstd_level,
    )?;
    images_hasher.update(&bytes);

    // images/sift_content_hashes
    let hash_bytes: Vec<u8> = data
        .sift_content_hashes
        .iter()
        .flat_map(|h| h.iter().copied())
        .collect();
    let bytes = write_binary_entry(
        &mut zip,
        &format!("images/sift_content_hashes.{image_count}.uint128.zst"),
        &hash_bytes,
        options.zstd_level,
    )?;
    images_hasher.update(&bytes);

    // images/thumbnails_y_x_rgb
    let bytes = write_binary_entry(
        &mut zip,
        &format!("images/thumbnails_y_x_rgb.{image_count}.128.128.3.uint8.zst"),
        data.thumbnails_y_x_rgb.as_slice().unwrap(),
        options.zstd_level,
    )?;
    images_hasher.update(&bytes);

    // images/translations_xyz
    let bytes = write_binary_entry(
        &mut zip,
        &format!("images/translations_xyz.{image_count}.3.float64.zst"),
        bytemuck::cast_slice(data.translations_xyz.as_slice().unwrap()),
        options.zstd_level,
    )?;
    images_hasher.update(&bytes);

    let images_hash = images_hasher.digest128();
    section_digests.push(images_hash);

    // === Points3D (hashed in lexicographic path order) ===
    let mut points3d_hasher = Xxh3::new();

    // points3d/colors_rgb
    let bytes = write_binary_entry(
        &mut zip,
        &format!("points3d/colors_rgb.{points3d_count}.3.uint8.zst"),
        data.colors_rgb.as_slice().unwrap(),
        options.zstd_level,
    )?;
    points3d_hasher.update(&bytes);

    // points3d/estimated_normals_xyz
    let bytes = write_binary_entry(
        &mut zip,
        &format!("points3d/estimated_normals_xyz.{points3d_count}.3.float32.zst"),
        bytemuck::cast_slice(estimated_normals_xyz.as_slice().unwrap()),
        options.zstd_level,
    )?;
    points3d_hasher.update(&bytes);

    // points3d/metadata.json
    let points3d_meta = serde_json::json!({"points3d_count": points3d_count});
    let bytes = write_json_entry(
        &mut zip,
        "points3d/metadata.json.zst",
        &points3d_meta,
        options.zstd_level,
    )?;
    points3d_hasher.update(&bytes);

    // points3d/positions_xyz
    let bytes = write_binary_entry(
        &mut zip,
        &format!("points3d/positions_xyz.{points3d_count}.3.float64.zst"),
        bytemuck::cast_slice(data.positions_xyz.as_slice().unwrap()),
        options.zstd_level,
    )?;
    points3d_hasher.update(&bytes);

    // points3d/reprojection_errors
    let bytes = write_binary_entry(
        &mut zip,
        &format!("points3d/reprojection_errors.{points3d_count}.float32.zst"),
        bytemuck::cast_slice(data.reprojection_errors.as_slice().unwrap()),
        options.zstd_level,
    )?;
    points3d_hasher.update(&bytes);

    let points3d_hash = points3d_hasher.digest128();
    section_digests.push(points3d_hash);

    // === Tracks (hashed in lexicographic path order) ===
    let mut tracks_hasher = Xxh3::new();

    // tracks/feature_indexes
    let bytes = write_binary_entry(
        &mut zip,
        &format!("tracks/feature_indexes.{observation_count}.uint32.zst"),
        bytemuck::cast_slice(data.feature_indexes.as_slice().unwrap()),
        options.zstd_level,
    )?;
    tracks_hasher.update(&bytes);

    // tracks/image_indexes
    let bytes = write_binary_entry(
        &mut zip,
        &format!("tracks/image_indexes.{observation_count}.uint32.zst"),
        bytemuck::cast_slice(data.image_indexes.as_slice().unwrap()),
        options.zstd_level,
    )?;
    tracks_hasher.update(&bytes);

    // tracks/metadata.json
    let tracks_meta = serde_json::json!({"observation_count": observation_count});
    let bytes = write_json_entry(
        &mut zip,
        "tracks/metadata.json.zst",
        &tracks_meta,
        options.zstd_level,
    )?;
    tracks_hasher.update(&bytes);

    // tracks/observation_counts
    let bytes = write_binary_entry(
        &mut zip,
        &format!("tracks/observation_counts.{points3d_count}.uint32.zst"),
        bytemuck::cast_slice(data.observation_counts.as_slice().unwrap()),
        options.zstd_level,
    )?;
    tracks_hasher.update(&bytes);

    // tracks/points3d_indexes
    let bytes = write_binary_entry(
        &mut zip,
        &format!("tracks/points3d_indexes.{observation_count}.uint32.zst"),
        bytemuck::cast_slice(data.points3d_indexes.as_slice().unwrap()),
        options.zstd_level,
    )?;
    tracks_hasher.update(&bytes);

    let tracks_hash = tracks_hasher.digest128();
    section_digests.push(tracks_hash);

    // === Content hash ===
    let all_digests_bytes: Vec<u8> = section_digests
        .iter()
        .flat_map(|d| d.to_be_bytes())
        .collect();
    let content_hash_value = xxhash_rust::xxh3::xxh3_128(&all_digests_bytes);

    let content_hash = ContentHash {
        metadata_xxh128: format_hash(metadata_hash),
        cameras_xxh128: format_hash(cameras_hash),
        rigs_xxh128: rigs_hash.map(format_hash),
        frames_xxh128: frames_hash.map(format_hash),
        images_xxh128: format_hash(images_hash),
        points3d_xxh128: format_hash(points3d_hash),
        tracks_xxh128: format_hash(tracks_hash),
        content_xxh128: format_hash(content_hash_value),
    };
    write_json_entry(
        &mut zip,
        "content_hash.json.zst",
        &content_hash,
        options.zstd_level,
    )?;

    zip.finish()?;
    Ok(())
}

/// Check if tracks are sorted by `(points3d_indexes, image_indexes)` and sort
/// them in-place if not. This is a no-op when tracks are already sorted.
fn ensure_tracks_sorted(data: &mut SfmrData) {
    let n = data.points3d_indexes.len();
    if n <= 1 {
        return;
    }

    let p3d = data.points3d_indexes.as_slice().unwrap();
    let img = data.image_indexes.as_slice().unwrap();

    // Check if already sorted
    let sorted = p3d
        .windows(2)
        .zip(img.windows(2))
        .all(|(p, i)| p[0] < p[1] || (p[0] == p[1] && i[0] <= i[1]));
    if sorted {
        return;
    }

    // Build permutation indices and sort by (points3d_index, image_index)
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_unstable_by(|&a, &b| p3d[a].cmp(&p3d[b]).then_with(|| img[a].cmp(&img[b])));

    // Apply permutation to all three track arrays
    let reorder = |arr: &mut ndarray::Array1<u32>, perm: &[usize]| {
        let old: Vec<u32> = arr.as_slice().unwrap().to_vec();
        for (i, &pi) in perm.iter().enumerate() {
            arr[i] = old[pi];
        }
    };
    reorder(&mut data.points3d_indexes, &perm);
    reorder(&mut data.image_indexes, &perm);
    reorder(&mut data.feature_indexes, &perm);
}

#[allow(clippy::too_many_arguments)]
fn validate_dimensions_with(
    data: &SfmrData,
    estimated_normals_xyz: &ndarray::Array2<f32>,
    observed_depth_histogram_counts: &ndarray::Array2<u32>,
    depth_statistics: &DepthStatistics,
    image_count: usize,
    points3d_count: usize,
    observation_count: usize,
    num_buckets: usize,
) -> Result<(), SfmrError> {
    macro_rules! check {
        ($cond:expr, $msg:expr) => {
            if !($cond) {
                return Err(SfmrError::ShapeMismatch($msg.into()));
            }
        };
    }

    check!(
        data.cameras.len() == data.metadata.camera_count as usize,
        format!(
            "cameras count {} != camera_count {}",
            data.cameras.len(),
            data.metadata.camera_count
        )
    );
    check!(
        data.image_names.len() == image_count,
        format!(
            "image_names count {} != image_count {image_count}",
            data.image_names.len()
        )
    );
    check!(
        data.camera_indexes.len() == image_count,
        format!(
            "camera_indexes len {} != image_count {image_count}",
            data.camera_indexes.len()
        )
    );
    check!(
        data.quaternions_wxyz.shape() == [image_count, 4],
        format!(
            "quaternions_wxyz shape {:?} != [{image_count}, 4]",
            data.quaternions_wxyz.shape()
        )
    );
    check!(
        data.translations_xyz.shape() == [image_count, 3],
        format!(
            "translations_xyz shape {:?} != [{image_count}, 3]",
            data.translations_xyz.shape()
        )
    );
    check!(
        data.feature_tool_hashes.len() == image_count,
        format!(
            "feature_tool_hashes len {} != image_count {image_count}",
            data.feature_tool_hashes.len()
        )
    );
    check!(
        data.sift_content_hashes.len() == image_count,
        format!(
            "sift_content_hashes len {} != image_count {image_count}",
            data.sift_content_hashes.len()
        )
    );
    check!(
        data.thumbnails_y_x_rgb.shape() == [image_count, 128, 128, 3],
        format!(
            "thumbnails_y_x_rgb shape {:?} != [{image_count}, 128, 128, 3]",
            data.thumbnails_y_x_rgb.shape()
        )
    );
    check!(
        data.positions_xyz.shape() == [points3d_count, 3],
        format!(
            "positions_xyz shape {:?} != [{points3d_count}, 3]",
            data.positions_xyz.shape()
        )
    );
    check!(
        data.colors_rgb.shape() == [points3d_count, 3],
        format!(
            "colors_rgb shape {:?} != [{points3d_count}, 3]",
            data.colors_rgb.shape()
        )
    );
    check!(
        data.reprojection_errors.len() == points3d_count,
        format!(
            "reprojection_errors len {} != points3d_count {points3d_count}",
            data.reprojection_errors.len()
        )
    );
    check!(
        estimated_normals_xyz.shape() == [points3d_count, 3],
        format!(
            "estimated_normals_xyz shape {:?} != [{points3d_count}, 3]",
            estimated_normals_xyz.shape()
        )
    );
    check!(
        data.image_indexes.len() == observation_count,
        format!(
            "image_indexes len {} != observation_count {observation_count}",
            data.image_indexes.len()
        )
    );
    check!(
        data.feature_indexes.len() == observation_count,
        format!(
            "feature_indexes len {} != observation_count {observation_count}",
            data.feature_indexes.len()
        )
    );
    check!(
        data.points3d_indexes.len() == observation_count,
        format!(
            "points3d_indexes len {} != observation_count {observation_count}",
            data.points3d_indexes.len()
        )
    );
    check!(
        data.observation_counts.len() == points3d_count,
        format!(
            "observation_counts len {} != points3d_count {points3d_count}",
            data.observation_counts.len()
        )
    );
    check!(
        observed_depth_histogram_counts.shape() == [image_count, num_buckets],
        format!(
            "observed_depth_histogram_counts shape {:?} != [{image_count}, {num_buckets}]",
            observed_depth_histogram_counts.shape()
        )
    );
    check!(
        depth_statistics.images.len() == image_count,
        format!(
            "depth_statistics.images len {} != image_count {image_count}",
            depth_statistics.images.len()
        )
    );

    // Validate observation_counts sum
    if points3d_count > 0 {
        let obs_sum: u64 = data.observation_counts.iter().map(|&c| c as u64).sum();
        if obs_sum != observation_count as u64 {
            return Err(SfmrError::ShapeMismatch(format!(
                "sum of observation_counts ({obs_sum}) != observation_count ({observation_count})"
            )));
        }
        if data.observation_counts.iter().any(|&c| c < 1) {
            return Err(SfmrError::ShapeMismatch(
                "observation_counts contains values < 1".into(),
            ));
        }
    }

    // Validate rig/frame data if present
    if let Some(rf) = &data.rig_frame_data {
        let sensor_count = rf.rigs_metadata.sensor_count as usize;
        let frame_count = rf.frames_metadata.frame_count as usize;

        check!(
            rf.rigs_metadata.rigs.len() == rf.rigs_metadata.rig_count as usize,
            format!(
                "rigs array length {} != rig_count {}",
                rf.rigs_metadata.rigs.len(),
                rf.rigs_metadata.rig_count
            )
        );
        for rig in &rf.rigs_metadata.rigs {
            check!(
                rig.sensor_names.len() == rig.sensor_count as usize,
                format!(
                    "rig '{}': sensor_names length {} != sensor_count {}",
                    rig.name,
                    rig.sensor_names.len(),
                    rig.sensor_count
                )
            );
            check!(
                rig.sensor_names.contains(&rig.ref_sensor_name),
                format!(
                    "rig '{}': ref_sensor_name '{}' not found in sensor_names {:?}",
                    rig.name, rig.ref_sensor_name, rig.sensor_names
                )
            );
        }
        check!(
            rf.sensor_camera_indexes.len() == sensor_count,
            format!(
                "sensor_camera_indexes len {} != sensor_count {sensor_count}",
                rf.sensor_camera_indexes.len()
            )
        );
        check!(
            rf.sensor_quaternions_wxyz.shape() == [sensor_count, 4],
            format!(
                "sensor_quaternions_wxyz shape {:?} != [{sensor_count}, 4]",
                rf.sensor_quaternions_wxyz.shape()
            )
        );
        check!(
            rf.sensor_translations_xyz.shape() == [sensor_count, 3],
            format!(
                "sensor_translations_xyz shape {:?} != [{sensor_count}, 3]",
                rf.sensor_translations_xyz.shape()
            )
        );
        check!(
            rf.rig_indexes.len() == frame_count,
            format!(
                "rig_indexes len {} != frame_count {frame_count}",
                rf.rig_indexes.len()
            )
        );
        check!(
            rf.image_sensor_indexes.len() == image_count,
            format!(
                "image_sensor_indexes len {} != image_count {image_count}",
                rf.image_sensor_indexes.len()
            )
        );
        check!(
            rf.image_frame_indexes.len() == image_count,
            format!(
                "image_frame_indexes len {} != image_count {image_count}",
                rf.image_frame_indexes.len()
            )
        );
    }

    Ok(())
}
