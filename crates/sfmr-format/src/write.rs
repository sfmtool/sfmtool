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

/// Squared-norm below which a stored estimated normal counts as "missing".
///
/// A set normal is a unit vector (norm² ≈ 1); the zero vector is the
/// initializer kept for points whose normal was never estimated and for
/// degenerate / infinity points, so anything this close to zero is treated as
/// absent and filled from the recomputed mean-viewing normals.
const MISSING_NORMAL_NORM_SQ: f32 = 1e-6;

/// Merge the stored normals with the geometry-recomputed mean-viewing ones,
/// *keeping* every stored normal that is present and filling only the missing
/// (zero) rows from the recompute. Falls back to the recomputed set wholesale if
/// the stored array's shape doesn't match (e.g. a dict-built `SfmrData` that
/// never carried normals). Returns a borrow when no copy is needed.
fn merge_preserving_normals<'a>(
    stored: &'a ndarray::Array2<f32>,
    recomputed: &'a ndarray::Array2<f32>,
) -> Cow<'a, ndarray::Array2<f32>> {
    if stored.dim() != recomputed.dim() {
        return Cow::Borrowed(recomputed);
    }
    let mut merged: Option<ndarray::Array2<f32>> = None;
    for i in 0..stored.nrows() {
        let x = stored[[i, 0]];
        let y = stored[[i, 1]];
        let z = stored[[i, 2]];
        if x * x + y * y + z * z <= MISSING_NORMAL_NORM_SQ {
            let out = merged.get_or_insert_with(|| stored.clone());
            out[[i, 0]] = recomputed[[i, 0]];
            out[[i, 1]] = recomputed[[i, 1]];
            out[[i, 2]] = recomputed[[i, 2]];
        }
    }
    match merged {
        Some(out) => Cow::Owned(out),
        None => Cow::Borrowed(stored),
    }
}

/// Write columnar data to a `.sfmr` file.
///
/// Recomputes depth statistics (normals, histograms, per-image stats) by default
/// to ensure they are consistent with the current poses, points, and tracks.
/// Use [`write_sfmr_with_options`] to skip recomputation if needed.
///
/// Sorts tracks by `(point_indexes, image_indexes)` if not already sorted.
/// Computes content hashes and writes all section metadata files.
/// Always writes format version 3; the `content_hash` field in `data` is
/// ignored on write (recomputed).
pub fn write_sfmr(path: &Path, data: &mut SfmrData) -> Result<(), SfmrError> {
    write_sfmr_with_options(path, data, &WriteOptions::default())
}

/// Write columnar data to a `.sfmr` file with explicit options.
///
/// Sorts tracks by `(point_indexes, image_indexes)` if not already sorted.
/// See [`WriteOptions`] for available options. By default, depth statistics
/// are recomputed from the reconstruction data to ensure trustworthiness.
pub fn write_sfmr_with_options(
    path: &Path,
    data: &mut SfmrData,
    options: &WriteOptions,
) -> Result<(), SfmrError> {
    // Ensure tracks are sorted by (point_indexes, image_indexes)
    ensure_tracks_sorted(data);

    // Always emit format version 3, and keep `infinity_point_count` consistent
    // with the actual `w` column.
    data.metadata.version = 3;
    data.metadata.infinity_point_count = (0..data.positions_xyzw.shape()[0])
        .filter(|&i| data.positions_xyzw[[i, 3]] == 0.0)
        .count() as u32;

    // Recompute depth statistics unless explicitly skipped
    let recomputed: Option<DepthStatsResult>;
    let (depth_statistics, normals_xyz, observed_depth_histogram_counts) =
        if options.skip_recompute_depth_stats {
            (
                &data.depth_statistics,
                data.normals_xyz.as_ref().map(Cow::Borrowed),
                Cow::Borrowed(&data.observed_depth_histogram_counts),
            )
        } else {
            recomputed = Some(compute_depth_statistics(
                &data.quaternions_wxyz,
                &data.translations_xyz,
                &data.positions_xyzw,
                &data.image_indexes,
                &data.point_indexes,
            )?);
            let r = recomputed.as_ref().unwrap();
            // Depth statistics and histograms always come from the recompute so
            // they track the current geometry (e.g. after a bundle adjust). The
            // estimated normals, however, are *preserved* from the input — only
            // the missing ones (the zero vector, the initializer used before any
            // normal is set and the value left for degenerate/infinity points)
            // are filled in from the recomputed mean-viewing normals. This keeps
            // normals a consumer has set (e.g. `sfm xform --refine-normals`)
            // instead of silently overwriting them, while a freshly imported
            // reconstruction — whose normals start all-zero — still gets a full
            // set computed on its first write. A reconstruction that carries no
            // normals at all (`None`) opts out entirely: none are written.
            let normals = data
                .normals_xyz
                .as_ref()
                .map(|n| merge_preserving_normals(n, &r.mean_viewing_normals_xyz));
            (
                &r.depth_statistics,
                normals,
                Cow::Borrowed(&r.observed_depth_histogram_counts),
            )
        };

    let image_count = data.metadata.image_count as usize;
    let point_count = data.metadata.point_count as usize;
    let observation_count = data.metadata.observation_count as usize;
    let num_buckets = depth_statistics.num_histogram_buckets as usize;

    // Validate dimensions (use possibly-recomputed depth stats)
    validate_dimensions_with(
        data,
        normals_xyz.as_deref(),
        &observed_depth_histogram_counts,
        depth_statistics,
        image_count,
        point_count,
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
    // The optional per-point patch frame (`patch_u_halfvec_xyz`,
    // `patch_v_halfvec_xyz`, and an optional `patch_bitmaps_y_x_rgba`) lives in
    // this section, beside the normal.
    validate_patch_dimensions(data, point_count)?;
    let mut points3d_hasher = Xxh3::new();

    // points3d/colors_rgb
    let bytes = write_binary_entry(
        &mut zip,
        &format!("points3d/colors_rgb.{point_count}.3.uint8.zst"),
        data.colors_rgb.as_slice().unwrap(),
        options.zstd_level,
    )?;
    points3d_hasher.update(&bytes);

    // points3d/metadata.json (records which optional per-point arrays are present)
    let patch_bitmap_resolution = data.patch_bitmaps_y_x_rgba.as_ref().map(|b| b.shape()[1]);
    let points3d_meta = serde_json::json!({
        "point_count": point_count,
        "has_normals": normals_xyz.is_some(),
        "has_uv_frames": data.patch_u_halfvec_xyz.is_some(),
        "has_patch_bitmaps": data.patch_bitmaps_y_x_rgba.is_some(),
        "patch_bitmap_resolution": patch_bitmap_resolution,
    });
    let bytes = write_json_entry(
        &mut zip,
        "points3d/metadata.json.zst",
        &points3d_meta,
        options.zstd_level,
    )?;
    points3d_hasher.update(&bytes);

    // points3d/normals_xyz (optional; named estimated_normals_xyz in versions 1-2)
    if let Some(normals_xyz) = &normals_xyz {
        let bytes = write_binary_entry(
            &mut zip,
            &format!("points3d/normals_xyz.{point_count}.3.float32.zst"),
            bytemuck::cast_slice(normals_xyz.as_slice().unwrap()),
            options.zstd_level,
        )?;
        points3d_hasher.update(&bytes);
    }

    // Optional patch frame, in lexicographic order: bitmaps, u, v.
    if let Some(bitmaps) = &data.patch_bitmaps_y_x_rgba {
        let r = bitmaps.shape()[1];
        let bytes = write_binary_entry(
            &mut zip,
            &format!("points3d/patch_bitmaps_y_x_rgba.{point_count}.{r}.{r}.4.uint8.zst"),
            bitmaps.as_slice().unwrap(),
            options.zstd_level,
        )?;
        points3d_hasher.update(&bytes);
    }
    if let Some(u) = &data.patch_u_halfvec_xyz {
        let bytes = write_binary_entry(
            &mut zip,
            &format!("points3d/patch_u_halfvec_xyz.{point_count}.3.float32.zst"),
            bytemuck::cast_slice(u.as_slice().unwrap()),
            options.zstd_level,
        )?;
        points3d_hasher.update(&bytes);
    }
    if let Some(v) = &data.patch_v_halfvec_xyz {
        let bytes = write_binary_entry(
            &mut zip,
            &format!("points3d/patch_v_halfvec_xyz.{point_count}.3.float32.zst"),
            bytemuck::cast_slice(v.as_slice().unwrap()),
            options.zstd_level,
        )?;
        points3d_hasher.update(&bytes);
    }

    // points3d/positions_xyzw
    let bytes = write_binary_entry(
        &mut zip,
        &format!("points3d/positions_xyzw.{point_count}.4.float64.zst"),
        bytemuck::cast_slice(data.positions_xyzw.as_slice().unwrap()),
        options.zstd_level,
    )?;
    points3d_hasher.update(&bytes);

    // points3d/reprojection_errors
    let bytes = write_binary_entry(
        &mut zip,
        &format!("points3d/reprojection_errors.{point_count}.float32.zst"),
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
        &format!("tracks/observation_counts.{point_count}.uint32.zst"),
        bytemuck::cast_slice(data.observation_counts.as_slice().unwrap()),
        options.zstd_level,
    )?;
    tracks_hasher.update(&bytes);

    // tracks/point_indexes
    let bytes = write_binary_entry(
        &mut zip,
        &format!("tracks/point_indexes.{observation_count}.uint32.zst"),
        bytemuck::cast_slice(data.point_indexes.as_slice().unwrap()),
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

/// Validate the optional per-point patch frame arrays: `patch_u_halfvec_xyz`
/// and `patch_v_halfvec_xyz` must be present together and shaped `(P, 3)`;
/// bitmaps require the frame and must be shaped `(P, R, R, 3)`.
fn validate_patch_dimensions(data: &SfmrData, point_count: usize) -> Result<(), SfmrError> {
    let check = |name: &str, got: &[usize]| -> Result<(), SfmrError> {
        if got != [point_count, 3] {
            return Err(SfmrError::ShapeMismatch(format!(
                "points3d/{name} shape {got:?} != [{point_count}, 3]"
            )));
        }
        Ok(())
    };
    if data.patch_u_halfvec_xyz.is_some() != data.patch_v_halfvec_xyz.is_some() {
        return Err(SfmrError::ShapeMismatch(
            "patch_u_halfvec_xyz and patch_v_halfvec_xyz must be present together".into(),
        ));
    }
    if let Some(u) = &data.patch_u_halfvec_xyz {
        check("patch_u_halfvec_xyz", u.shape())?;
    }
    if let Some(v) = &data.patch_v_halfvec_xyz {
        check("patch_v_halfvec_xyz", v.shape())?;
    }
    if let Some(b) = &data.patch_bitmaps_y_x_rgba {
        if data.patch_u_halfvec_xyz.is_none() {
            return Err(SfmrError::ShapeMismatch(
                "patch_bitmaps_y_x_rgba requires the patch frame (patch_u/v_halfvec_xyz)".into(),
            ));
        }
        let s = b.shape();
        if s.len() != 4 || s[0] != point_count || s[1] != s[2] || s[3] != 4 {
            return Err(SfmrError::ShapeMismatch(format!(
                "points3d/patch_bitmaps_y_x_rgba shape {s:?} != [{point_count}, R, R, 4]"
            )));
        }
    }
    Ok(())
}

/// Check if tracks are sorted by `(point_indexes, image_indexes)` and sort
/// them in-place if not. This is a no-op when tracks are already sorted.
fn ensure_tracks_sorted(data: &mut SfmrData) {
    let n = data.point_indexes.len();
    if n <= 1 {
        return;
    }

    let p3d = data.point_indexes.as_slice().unwrap();
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
    reorder(&mut data.point_indexes, &perm);
    reorder(&mut data.image_indexes, &perm);
    reorder(&mut data.feature_indexes, &perm);
}

#[allow(clippy::too_many_arguments)]
fn validate_dimensions_with(
    data: &SfmrData,
    normals_xyz: Option<&ndarray::Array2<f32>>,
    observed_depth_histogram_counts: &ndarray::Array2<u32>,
    depth_statistics: &DepthStatistics,
    image_count: usize,
    point_count: usize,
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
        data.positions_xyzw.shape() == [point_count, 4],
        format!(
            "positions_xyzw shape {:?} != [{point_count}, 4]",
            data.positions_xyzw.shape()
        )
    );
    check!(
        data.colors_rgb.shape() == [point_count, 3],
        format!(
            "colors_rgb shape {:?} != [{point_count}, 3]",
            data.colors_rgb.shape()
        )
    );
    check!(
        data.reprojection_errors.len() == point_count,
        format!(
            "reprojection_errors len {} != point_count {point_count}",
            data.reprojection_errors.len()
        )
    );
    if let Some(normals_xyz) = normals_xyz {
        check!(
            normals_xyz.shape() == [point_count, 3],
            format!(
                "normals_xyz shape {:?} != [{point_count}, 3]",
                normals_xyz.shape()
            )
        );
    }
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
        data.point_indexes.len() == observation_count,
        format!(
            "point_indexes len {} != observation_count {observation_count}",
            data.point_indexes.len()
        )
    );
    check!(
        data.observation_counts.len() == point_count,
        format!(
            "observation_counts len {} != point_count {point_count}",
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
    if point_count > 0 {
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
