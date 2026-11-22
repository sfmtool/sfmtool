// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sfmr` file reading.

use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3, Array4};

use sfmtool_archive_io::{read_binary_array, read_json_entry, read_uint128_array};

use crate::entries::{self, ReadingColumn};
use crate::float16::{scores_from_deficits, widen_float16};
use crate::image_readings::image_readings_presence_error;
use crate::measurements::{
    read_measurement_unit_legend, resolve_measurement_units, validate_distance_measurements,
    DistanceMeasurementsData, MeasurementsData,
};
use crate::observation_readings::{
    observation_reading_options, observation_readings_flagged, ReadingColumnsBuilder,
};
use crate::types::*;
use crate::{validate_image_readings, ImageReadingColumns, ObservationReadingColumns};

fn open_archive(path: &Path) -> Result<zip::ZipArchive<std::fs::File>, SfmrError> {
    let file = std::fs::File::open(path).map_err(|e| SfmrError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(zip::ZipArchive::new(file)?)
}

/// Refuse a `metadata.version` this build does not read: one below
/// [`SFMR_OLDEST_READ_VERSION`] that a release wrote, with
/// [`SfmrError::VersionNoLongerRead`], which names `path` and the last
/// release that reads it; and one of 0, which no writer has ever produced, or
/// newer than [`SFMR_FORMAT_VERSION`], whose layout is unknown so reading it
/// with current-version assumptions would misparse silently.
pub(crate) fn check_version(version: u32, path: &Path) -> Result<(), SfmrError> {
    if (1..SFMR_OLDEST_READ_VERSION).contains(&version) {
        return Err(SfmrError::VersionNoLongerRead {
            path: path.to_path_buf(),
            version,
        });
    }
    if !(SFMR_OLDEST_READ_VERSION..=SFMR_FORMAT_VERSION).contains(&version) {
        return Err(SfmrError::InvalidFormat(format!(
            "unsupported .sfmr format version {version} (this build reads versions \
             {SFMR_OLDEST_READ_VERSION} to {SFMR_FORMAT_VERSION})"
        )));
    }
    Ok(())
}

/// The keys a file's `metadata.json` holds that [`SfmrMetadata`] fills in
/// when a value built in memory leaves them out.
const REQUIRED_DEFAULTED_KEYS: [&str; 2] = ["feature_source", "infinity_point_count"];

/// Parse `metadata.json` from its JSON value, checking its version with
/// [`check_version`] first, so that a file of a version this build does not
/// read is refused for its version rather than for a key that version lacks,
/// and then that it holds every key a file requires.
pub(crate) fn parse_metadata(
    value: serde_json::Value,
    path: &Path,
) -> Result<SfmrMetadata, SfmrError> {
    if let Some(version) = value.get("version").and_then(serde_json::Value::as_u64) {
        check_version(u32::try_from(version).unwrap_or(u32::MAX), path)?;
    }
    if let Some(key) = REQUIRED_DEFAULTED_KEYS
        .iter()
        .find(|key| value.get(**key).is_none())
    {
        return Err(SfmrError::InvalidFormat(format!(
            "metadata.json.zst has no {key}"
        )));
    }
    let metadata: SfmrMetadata = serde_json::from_value(value)?;
    if metadata.version < MEASUREMENTS_FORMAT_VERSION
        && metadata.distance_measurement_count.is_some()
    {
        return Err(SfmrError::InvalidFormat(format!(
            "version {} .sfmr file must not carry distance_measurement_count (introduced in \
             version {MEASUREMENTS_FORMAT_VERSION})",
            metadata.version
        )));
    }
    if metadata.distance_measurement_count == Some(0) {
        return Err(SfmrError::InvalidFormat(
            "metadata.json.zst distance_measurement_count must be >= 1 when present".into(),
        ));
    }
    Ok(metadata)
}

/// Read the top-level metadata of the archive at `path`: `metadata.json`
/// ([`parse_metadata`]), with the write timestamp put back from
/// `written.json`, which a file keeps outside the content digest.
fn read_metadata<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    path: &Path,
) -> Result<SfmrMetadata, SfmrError> {
    let value: serde_json::Value = read_json_entry(archive, entries::metadata())?;
    let mut metadata = parse_metadata(value, path)?;
    if metadata.version < MEASUREMENTS_FORMAT_VERSION && has_measurements_entry(archive)? {
        return Err(SfmrError::InvalidFormat(format!(
            "version {} .sfmr file must not carry measurements/ entries (introduced in version \
             {MEASUREMENTS_FORMAT_VERSION})",
            metadata.version
        )));
    }
    let record: WriteRecord = read_json_entry(archive, entries::written())?;
    metadata.timestamp = record.timestamp;
    Ok(metadata)
}

/// Read only the top-level metadata from a `.sfmr` file (fast, no binary data).
///
/// A file of a version this build does not read is refused, as [`read_sfmr`]
/// refuses it.
pub fn read_sfmr_metadata(path: &Path) -> Result<SfmrMetadata, SfmrError> {
    read_metadata(&mut open_archive(path)?, path)
}

/// Read the top-level metadata and the image names of a `.sfmr` file, and no
/// binary section.
///
/// For a caller that needs only which images a file records, and in which
/// workspace, without decompressing its points and tracks: two JSON entries
/// rather than the whole archive. The version is checked, and the names
/// against the metadata's `image_count`, as [`read_sfmr`] checks them.
pub fn read_sfmr_image_names(path: &Path) -> Result<(SfmrMetadata, Vec<String>), SfmrError> {
    let mut archive = open_archive(path)?;
    let metadata = read_metadata(&mut archive, path)?;
    let names: Vec<String> = read_json_entry(&mut archive, entries::images_names())?;
    if names.len() != metadata.image_count as usize {
        return Err(SfmrError::ShapeMismatch(format!(
            "image names count {} != image_count {}",
            names.len(),
            metadata.image_count
        )));
    }
    Ok((metadata, names))
}

/// Read only the content-integrity hashes from a `.sfmr` file.
///
/// Decompresses just `content_hash.json.zst`, so it is cheap enough to scan a
/// directory of `.sfmr` files (e.g. to resolve a `pt3d_<hash>_<index>` Point ID
/// by its `content_xxh128` prefix) without loading any reconstruction data,
/// and `metadata.json.zst` for the version, so that a file of a version this
/// build does not read is refused, as [`read_sfmr`] refuses it.
pub fn read_sfmr_content_hash(path: &Path) -> Result<ContentHash, SfmrError> {
    let mut archive = open_archive(path)?;
    let value: serde_json::Value = read_json_entry(&mut archive, entries::metadata())?;
    let metadata = parse_metadata(value, path)?;
    if metadata.version < MEASUREMENTS_FORMAT_VERSION && has_measurements_entry(&archive)? {
        return Err(SfmrError::InvalidFormat(format!(
            "version {} .sfmr file must not carry measurements/ entries (introduced in version \
             {MEASUREMENTS_FORMAT_VERSION})",
            metadata.version
        )));
    }
    let content_hash: ContentHash = read_json_entry(&mut archive, entries::content_hash())?;
    if metadata.version < MEASUREMENTS_FORMAT_VERSION && content_hash.measurements_xxh128.is_some()
    {
        return Err(SfmrError::InvalidFormat(format!(
            "version {} .sfmr file must not carry measurements_xxh128 (introduced in version \
             {MEASUREMENTS_FORMAT_VERSION})",
            metadata.version
        )));
    }
    if declares_measurements(&metadata) != content_hash.measurements_xxh128.is_some() {
        return Err(SfmrError::InvalidFormat(
            "content_hash.json.zst carries measurements_xxh128 exactly when metadata.json.zst \
             carries a measurement count (distance_measurement_count), and this file has one \
             without the other"
                .into(),
        ));
    }
    Ok(content_hash)
}

/// Read a complete `.sfmr` file into columnar data.
///
/// Every rule of the format is checked. The per-image intrinsic adjustments
/// (the focal scales, the principal-point offsets, the radial B-spline
/// corrections and the rolling-shutter rates) and the cameras'
/// `rolling_shutter_scanlines` settings are read as stored.
pub fn read_sfmr(path: &Path) -> Result<SfmrData, SfmrError> {
    let mut archive = open_archive(path)?;

    let (mut metadata, content_hash) = read_top_level(&mut archive, path)?;
    let image_count = metadata.image_count as usize;
    let point_count = metadata.point_count as usize;
    let cameras = read_cameras(&mut archive, &metadata)?;
    let (images_meta, points3d_meta, tracks_meta) = read_section_metadata(&mut archive, &metadata)?;
    let images = read_images_section(&mut archive, &metadata, &images_meta, &cameras)?;
    let points = read_points_section(&mut archive, &metadata, &points3d_meta)?;
    let tracks = read_tracks_section(
        &mut archive,
        &metadata,
        &tracks_meta,
        &images.camera_indexes,
        &cameras,
        points.patch_u_halfvec_xyz.is_some(),
    )?;
    let rig_frame_data = read_rig_frames(&mut archive, image_count)?;
    let measurements = read_measurements_section(
        &mut archive,
        &metadata,
        &content_hash,
        &images.camera_indexes,
        &cameras,
    )?;

    // `infinity_point_count` is recomputed from the `w` column, so the value
    // describes the points it was read with.
    metadata.infinity_point_count = (0..point_count)
        .filter(|&i| points.positions_xyzw[[i, 3]] == 0.0)
        .count() as u32;
    let workspace_dir = resolve_workspace_dir(path, &metadata).ok();

    Ok(SfmrData {
        workspace_dir,
        metadata,
        content_hash,
        cameras,
        rig_frame_data,
        image_names: images.image_names,
        camera_indexes: images.camera_indexes,
        quaternions_wxyz: images.quaternions_wxyz,
        translations_xyz: images.translations_xyz,
        feature_tool_hashes: images.feature_tool_hashes,
        sift_content_hashes: images.sift_content_hashes,
        image_file_hashes: images.image_file_hashes,
        thumbnails_y_x_rgb: images.thumbnails_y_x_rgb,
        focal_adjustments: images.focal_adjustments,
        principal_point_adjustments_xy: images.principal_point_adjustments_xy,
        radial_bspline_adjustments: images.radial_bspline_adjustments,
        rolling_shutter_rates_xyz: images.rolling_shutter_rates_xyz,
        image_readings: Some(images.image_readings),
        depth_statistics: images.depth_statistics,
        observed_depth_histogram_counts: images.observed_depth_histogram_counts,
        positions_xyzw: points.positions_xyzw,
        colors_rgb: points.colors_rgb,
        reprojection_errors: points.reprojection_errors,
        normals_xyz: points.normals_xyz,
        normal_confidence: points.normal_confidence,
        point_constraints: points.point_constraints,
        constraint_distances: points.constraint_distances,
        constraint_reference_images: points.constraint_reference_images,
        patch_u_halfvec_xyz: points.patch_u_halfvec_xyz,
        patch_v_halfvec_xyz: points.patch_v_halfvec_xyz,
        patch_bitmaps_y_x_rgba: points.patch_bitmaps_y_x_rgba,
        image_indexes: tracks.image_indexes,
        feature_indexes: tracks.feature_indexes,
        keypoints_xy: tracks.keypoints_xy,
        observation_confidence: tracks.observation_confidence,
        observation_readings: tracks.observation_readings,
        point_indexes: tracks.point_indexes,
        observation_counts: tracks.observation_counts,
        reference_observations: tracks.reference_observations,
        measurements,
    })
}

// This call precedes all section reads, including their count checks.
fn read_top_level(
    archive: &mut zip::ZipArchive<std::fs::File>,
    path: &Path,
) -> Result<(SfmrMetadata, ContentHash), SfmrError> {
    // The version is checked first, before any entry whose layout depends on it.
    let metadata = read_metadata(archive, path)?;
    let content_hash: ContentHash = read_json_entry(archive, entries::content_hash())?;
    validate_world_space_unit(metadata.world_space_unit.as_deref())
        .map_err(SfmrError::InvalidFormat)?;

    Ok((metadata, content_hash))
}

fn read_cameras(
    archive: &mut zip::ZipArchive<std::fs::File>,
    metadata: &SfmrMetadata,
) -> Result<Vec<SfmrCamera>, SfmrError> {
    let cameras: Vec<SfmrCamera> = read_json_entry(archive, entries::cameras_metadata())?;
    if cameras.len() != metadata.camera_count as usize {
        return Err(SfmrError::InvalidFormat(format!(
            "Camera count mismatch: metadata says {}, got {}",
            metadata.camera_count,
            cameras.len()
        )));
    }
    Ok(cameras)
}

// Check all three section counts before reading any image columns.
fn read_section_metadata(
    archive: &mut zip::ZipArchive<std::fs::File>,
    metadata: &SfmrMetadata,
) -> Result<(serde_json::Value, serde_json::Value, serde_json::Value), SfmrError> {
    let image_count = metadata.image_count as usize;
    let point_count = metadata.point_count as usize;
    let observation_count = metadata.observation_count as usize;
    // Cross-check section metadata
    let images_meta: serde_json::Value = read_json_entry(archive, entries::images_metadata())?;
    if images_meta.get("image_count").and_then(|v| v.as_u64()) != Some(image_count as u64) {
        return Err(SfmrError::InvalidFormat(
            "images/metadata.json.zst image_count doesn't match top-level metadata".into(),
        ));
    }

    let points3d_meta: serde_json::Value = read_json_entry(archive, entries::points3d_metadata())?;
    let section_point_count = points3d_meta.get("point_count").and_then(|v| v.as_u64());
    if section_point_count != Some(point_count as u64) {
        return Err(SfmrError::InvalidFormat(
            "points3d/metadata.json.zst point count doesn't match top-level metadata".into(),
        ));
    }

    let tracks_meta: serde_json::Value = read_json_entry(archive, entries::tracks_metadata())?;
    if tracks_meta
        .get("observation_count")
        .and_then(|v| v.as_u64())
        != Some(observation_count as u64)
    {
        return Err(SfmrError::InvalidFormat(
            "tracks/metadata.json.zst observation_count doesn't match top-level metadata".into(),
        ));
    }

    Ok((images_meta, points3d_meta, tracks_meta))
}

struct ImagesSection {
    image_names: Vec<String>,
    camera_indexes: Array1<u32>,
    quaternions_wxyz: Array2<f64>,
    translations_xyz: Array2<f64>,
    feature_tool_hashes: Option<Vec<[u8; 16]>>,
    sift_content_hashes: Option<Vec<[u8; 16]>>,
    image_file_hashes: Option<Vec<[u8; 16]>>,
    thumbnails_y_x_rgb: Option<Array4<u8>>,
    focal_adjustments: Option<Array1<f64>>,
    principal_point_adjustments_xy: Option<Array2<f64>>,
    radial_bspline_adjustments: Option<Array2<f64>>,
    rolling_shutter_rates_xyz: Option<Array2<f64>>,
    image_readings: ImageReadingColumns,
    depth_statistics: DepthStatistics,
    observed_depth_histogram_counts: Array2<u32>,
}

fn read_images_section(
    archive: &mut zip::ZipArchive<std::fs::File>,
    metadata: &SfmrMetadata,
    images_meta: &serde_json::Value,
    cameras: &[SfmrCamera],
) -> Result<ImagesSection, SfmrError> {
    let image_count = metadata.image_count as usize;
    // Images
    let image_names: Vec<String> = read_json_entry(archive, entries::images_names())?;
    if image_names.len() != image_count {
        return Err(SfmrError::ShapeMismatch(format!(
            "image names count {} != image_count {image_count}",
            image_names.len()
        )));
    }
    sfmtool_archive_io::validate_image_names(&image_names).map_err(SfmrError::InvalidFormat)?;

    let camera_indexes_vec: Vec<u32> = read_binary_array(
        archive,
        &entries::images_camera_indexes(image_count),
        image_count,
    )?;
    let camera_indexes = Array1::from_vec(camera_indexes_vec);

    let quaternions_vec: Vec<f64> = read_binary_array(
        archive,
        &entries::images_quaternions_wxyz(image_count),
        image_count * 4,
    )?;
    let quaternions_wxyz = Array2::from_shape_vec((image_count, 4), quaternions_vec)
        .map_err(|e| SfmrError::ShapeMismatch(format!("quaternions reshape: {e}")))?;

    let translations_vec: Vec<f64> = read_binary_array(
        archive,
        &entries::images_translations_xyz(image_count),
        image_count * 3,
    )?;
    let translations_xyz = Array2::from_shape_vec((image_count, 3), translations_vec)
        .map_err(|e| SfmrError::ShapeMismatch(format!("translations reshape: {e}")))?;

    // Observation source. Reject any unrecognized value rather than silently
    // treating it as `sift_files`.
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
    // identity columns exist (a file links to `.sift` or carries its own image
    // hash, never both). `tracks/metadata.json`'s `has_feature_indexes` mirrors it
    // for section-local readers and is cross-checked in `verify_sfmr`; it
    // intentionally does not gate reading here the way the independent `points3d`
    // `has_*` flags do. `has_keypoints_xy` is different: a `sift_files` file
    // older than `SIFT_FILES_KEYPOINTS_FORMAT_VERSION` may lack the inline
    // keypoint column, so that flag is what says whether the entry is there, and
    // it does gate the read in `read_tracks_section`.
    let is_embedded = metadata.feature_source == FEATURE_SOURCE_EMBEDDED_PATCHES;

    // A `sift_files` file links to `.sift` via per-image tool/content hashes; an
    // `embedded_patches` file substitutes the direct image-bytes hash instead.
    let (feature_tool_hashes, sift_content_hashes, image_file_hashes) = if is_embedded {
        let image_file_hashes = read_uint128_array(
            archive,
            &entries::images_image_file_hashes(image_count),
            image_count,
        )?;
        (None, None, Some(image_file_hashes))
    } else {
        let feature_tool_hashes = read_uint128_array(
            archive,
            &entries::images_feature_tool_hashes(image_count),
            image_count,
        )?;
        let sift_content_hashes = read_uint128_array(
            archive,
            &entries::images_sift_content_hashes(image_count),
            image_count,
        )?;
        (Some(feature_tool_hashes), Some(sift_content_hashes), None)
    };

    // Thumbnails, optional behind `has_thumbnails`.
    let thumbnails_y_x_rgb =
        if images_meta_has_thumbnails(images_meta).map_err(SfmrError::InvalidFormat)? {
            let thumbnails_vec: Vec<u8> = read_binary_array(
                archive,
                &entries::images_thumbnails_y_x_rgb(image_count),
                image_count * THUMBNAIL_SIZE * THUMBNAIL_SIZE * 3,
            )?;
            Some(
                Array4::from_shape_vec(
                    (image_count, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3),
                    thumbnails_vec,
                )
                .map_err(|e| SfmrError::ShapeMismatch(format!("thumbnails reshape: {e}")))?,
            )
        } else {
            None
        };

    let AdjustmentColumns {
        focal_adjustments,
        principal_point_adjustments_xy,
        radial_bspline_adjustments,
        rolling_shutter_rates_xyz,
    } = read_per_image_adjustments(archive, metadata, cameras, &camera_indexes)?;
    let image_readings = read_image_readings(archive, metadata, images_meta)?;

    let depth_statistics: DepthStatistics = read_json_entry(archive, entries::depth_statistics())?;
    let num_buckets = depth_statistics.num_histogram_buckets as usize;

    let histogram_vec: Vec<u32> = read_binary_array(
        archive,
        &entries::observed_depth_histogram_counts(image_count, num_buckets),
        image_count * num_buckets,
    )?;
    let observed_depth_histogram_counts =
        Array2::from_shape_vec((image_count, num_buckets), histogram_vec)
            .map_err(|e| SfmrError::ShapeMismatch(format!("histogram reshape: {e}")))?;

    Ok(ImagesSection {
        image_names,
        camera_indexes,
        quaternions_wxyz,
        translations_xyz,
        feature_tool_hashes,
        sift_content_hashes,
        image_file_hashes,
        thumbnails_y_x_rgb,
        focal_adjustments,
        principal_point_adjustments_xy,
        radial_bspline_adjustments,
        rolling_shutter_rates_xyz,
        image_readings,
        depth_statistics,
        observed_depth_histogram_counts,
    })
}

/// The image readings: both columns and the `image_to_gray` key of
/// `images/metadata.json`, all three required, checked by
/// [`validate_image_readings`]. The sharpness is stored as `float16` and
/// widened to `f32` exactly.
fn read_image_readings(
    archive: &mut zip::ZipArchive<std::fs::File>,
    metadata: &SfmrMetadata,
    images_meta: &serde_json::Value,
) -> Result<ImageReadingColumns, SfmrError> {
    let image_count = metadata.image_count as usize;
    let sharpness_name = entries::images_sharpness(image_count);
    let tensors_name = entries::images_gradient_structure_tensors_xx_yy_xy(image_count);
    let image_to_gray = images_meta.get("image_to_gray");
    let stored = [
        archive.index_for_name(&sharpness_name).is_some(),
        archive.index_for_name(&tensors_name).is_some(),
        image_to_gray.is_some(),
    ];
    if let Some(e) = image_readings_presence_error(stored) {
        return Err(SfmrError::InvalidFormat(e));
    }
    let sharpness = widen_float16(&read_binary_array::<_, u16>(
        archive,
        &sharpness_name,
        image_count,
    )?);
    let tensors: Vec<f32> = read_binary_array(archive, &tensors_name, image_count * 3)?;
    let readings = ImageReadingColumns {
        // Present, as the presence check above found.
        image_to_gray: image_to_gray.cloned().unwrap_or_default(),
        sharpness: Array1::from_vec(sharpness),
        gradient_structure_tensors_xx_yy_xy: Array2::from_shape_vec((image_count, 3), tensors)
            .map_err(|e| SfmrError::ShapeMismatch(format!("{tensors_name} reshape: {e}")))?,
    };
    validate_image_readings(&readings, image_count).map_err(SfmrError::InvalidFormat)?;
    Ok(readings)
}

/// The four per-image adjustment columns, each `None` where absent.
struct AdjustmentColumns {
    focal_adjustments: Option<Array1<f64>>,
    principal_point_adjustments_xy: Option<Array2<f64>>,
    radial_bspline_adjustments: Option<Array2<f64>>,
    rolling_shutter_rates_xyz: Option<Array2<f64>>,
}

/// The per-image adjustment columns (the focal, the principal point, the
/// radial B-spline correction and the rolling-shutter rates), each present
/// exactly when some camera lists its parameter in `per_image_adjustments`,
/// checked against the format's rules. An entry that no camera's key accounts
/// for is refused, so a column is never silently ignored.
fn read_per_image_adjustments(
    archive: &mut zip::ZipArchive<std::fs::File>,
    metadata: &SfmrMetadata,
    cameras: &[SfmrCamera],
    camera_indexes: &Array1<u32>,
) -> Result<AdjustmentColumns, SfmrError> {
    let image_count = metadata.image_count as usize;
    let names = [
        entries::images_focal_adjustments(image_count),
        entries::images_principal_point_adjustments_xy(image_count),
        entries::images_radial_bspline_adjustments(image_count),
        entries::images_rolling_shutter_rates_xyz(image_count),
    ];
    let stored = names
        .each_ref()
        .map(|name| archive.index_for_name(name).is_some());
    let mut listed = [false; 4];
    for (c, camera) in cameras.iter().enumerate() {
        let flags = parse_per_image_adjustment_names(&camera.per_image_adjustments)
            .map_err(|e| SfmrError::InvalidFormat(format!("camera {c}: {e}")))?;
        for (l, f) in listed.iter_mut().zip(flags) {
            *l |= f;
        }
    }
    // An entry without a listing camera is read anyway, so that the shared
    // rules below name it rather than this reader skipping it unseen.
    let focal_adjustments = if listed[0] || stored[0] {
        let values: Vec<f64> = read_binary_array(archive, &names[0], image_count)?;
        Some(Array1::from_vec(values))
    } else {
        None
    };
    let mut read_rows = |k: usize, width: usize| -> Result<Option<Array2<f64>>, SfmrError> {
        if !(listed[k] || stored[k]) {
            return Ok(None);
        }
        let values: Vec<f64> = read_binary_array(archive, &names[k], image_count * width)?;
        Array2::from_shape_vec((image_count, width), values)
            .map(Some)
            .map_err(|e| SfmrError::ShapeMismatch(format!("{} reshape: {e}", names[k])))
    };
    let principal_point_adjustments_xy = read_rows(1, 2)?;
    let radial_bspline_adjustments = read_rows(2, 2)?;
    let rolling_shutter_rates_xyz = read_rows(3, 3)?;
    validate_per_image_adjustments(
        cameras,
        camera_indexes.as_slice().unwrap(),
        focal_adjustments.as_ref(),
        principal_point_adjustments_xy.as_ref(),
        radial_bspline_adjustments.as_ref(),
        rolling_shutter_rates_xyz.as_ref(),
    )
    .map_err(SfmrError::InvalidFormat)?;
    Ok(AdjustmentColumns {
        focal_adjustments,
        principal_point_adjustments_xy,
        radial_bspline_adjustments,
        rolling_shutter_rates_xyz,
    })
}

struct PointsSection {
    positions_xyzw: Array2<f64>,
    colors_rgb: Array2<u8>,
    reprojection_errors: Array1<f32>,
    normals_xyz: Option<Array2<f32>>,
    normal_confidence: Option<Array1<u8>>,
    point_constraints: Option<Array1<u8>>,
    constraint_distances: Option<Array1<f64>>,
    constraint_reference_images: Option<Array1<u32>>,
    patch_u_halfvec_xyz: Option<Array2<f32>>,
    patch_v_halfvec_xyz: Option<Array2<f32>>,
    patch_bitmaps_y_x_rgba: Option<Array4<u8>>,
}

fn read_points_section(
    archive: &mut zip::ZipArchive<std::fs::File>,
    metadata: &SfmrMetadata,
    points3d_meta: &serde_json::Value,
) -> Result<PointsSection, SfmrError> {
    let image_count = metadata.image_count as usize;
    let point_count = metadata.point_count as usize;
    let positions_vec: Vec<f64> = read_binary_array(
        archive,
        &entries::points3d_positions(point_count),
        point_count * 4,
    )?;
    let positions_xyzw = Array2::from_shape_vec((point_count, 4), positions_vec)
        .map_err(|e| SfmrError::ShapeMismatch(format!("positions reshape: {e}")))?;

    let colors_vec: Vec<u8> = read_binary_array(
        archive,
        &entries::points3d_colors_rgb(point_count),
        point_count * 3,
    )?;
    let colors_rgb = Array2::from_shape_vec((point_count, 3), colors_vec)
        .map_err(|e| SfmrError::ShapeMismatch(format!("colors reshape: {e}")))?;

    let reprojection_vec: Vec<f32> = read_binary_array(
        archive,
        &entries::points3d_reprojection_errors(point_count),
        point_count,
    )?;
    let reprojection_errors = Array1::from_vec(reprojection_vec);

    // The normals, their confidence, and the patch frame and bitmaps are each
    // flagged in the section metadata, `false` when the key is absent.
    let flag = |key: &str| {
        points3d_meta
            .get(key)
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    };
    let has_normals = flag("has_normals");
    let has_normal_confidence = flag("has_normal_confidence");
    let has_uv_frames = flag("has_uv_frames");
    let has_patch_bitmaps = flag("has_patch_bitmaps");
    // The writer refuses a combination that breaks a presence rule, and so
    // does the reader, before reading any optional column.
    OptionalPresence {
        embedded_patches: metadata.feature_source == FEATURE_SOURCE_EMBEDDED_PATCHES,
        normals: has_normals,
        normal_confidence: has_normal_confidence,
        patch_frame: has_uv_frames,
        patch_bitmaps: has_patch_bitmaps,
    }
    .check()
    .map_err(SfmrError::InvalidFormat)?;
    // A file with the patch frame stores only the normals the frame does not
    // give, read below once the frame is.
    let derives_normals = has_normals && has_uv_frames;
    let normals_xyz = if has_normals && !derives_normals {
        let normals_name = entries::points3d_normals(point_count);
        let normals_vec: Vec<f32> = read_binary_array(archive, &normals_name, point_count * 3)?;
        Some(
            Array2::from_shape_vec((point_count, 3), normals_vec)
                .map_err(|e| SfmrError::ShapeMismatch(format!("normals reshape: {e}")))?,
        )
    } else {
        None
    };

    // Optional per-point normal confidence. Absent means "no confidence
    // information", and reads as `None`.
    let normal_confidence = if has_normal_confidence {
        let confidence_vec: Vec<u8> = read_binary_array(
            archive,
            &entries::points3d_normal_confidence(point_count),
            point_count,
        )?;
        Some(Array1::from_vec(confidence_vec))
    } else {
        None
    };

    // Optional per-point constraint triple, flagged as one set and read through
    // the same metadata entry's `point_constraint_names` legend. Absent means
    // every point is free, and reads as `None`.
    let (point_constraints, constraint_distances, constraint_reference_images) = if points3d_meta
        .get("has_point_constraints")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        let legend =
            read_point_constraint_legend(points3d_meta).map_err(SfmrError::InvalidFormat)?;
        let point_constraints: Vec<u8> = read_binary_array(
            archive,
            &entries::points3d_point_constraints(point_count),
            point_count,
        )?;
        let constraint_distances: Vec<f64> = read_binary_array(
            archive,
            &entries::points3d_constraint_distances(point_count),
            point_count,
        )?;
        let constraint_reference_images: Vec<u32> = read_binary_array(
            archive,
            &entries::points3d_constraint_reference_images(point_count),
            point_count,
        )?;
        // The legend settles what each code means; the triple's own rules are
        // then checked on the constraints it named. The column is handed back on
        // the canonical numbering rather than the file's, so a consumer reads it
        // with the constants and the file's legend stops at this boundary.
        let resolved = resolve_point_constraints(&point_constraints, &legend)
            .map_err(SfmrError::InvalidFormat)?;
        validate_point_constraints(
            Some(&resolved),
            Some(&constraint_distances),
            Some(&constraint_reference_images),
            &positions_xyzw,
            point_count,
            image_count,
        )
        .map_err(SfmrError::InvalidFormat)?;
        (
            Some(Array1::from_vec(
                resolved.iter().map(|k| k.code()).collect(),
            )),
            Some(Array1::from_vec(constraint_distances)),
            Some(Array1::from_vec(constraint_reference_images)),
        )
    } else {
        (None, None, None)
    };

    // Optional per-point patch frame, stored beside the normals.
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
    let (patch_u_halfvec_xyz, patch_v_halfvec_xyz) = if has_uv_frames {
        (
            Some(read_vec3(
                archive,
                &entries::points3d_patch_u_halfvec_xyz(point_count),
            )?),
            Some(read_vec3(
                archive,
                &entries::points3d_patch_v_halfvec_xyz(point_count),
            )?),
        )
    } else {
        (None, None)
    };
    let patch_bitmaps_y_x_rgba = if has_patch_bitmaps {
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
            archive,
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

    // With the patch frame, the normal of a finite point whose frame gives one
    // is derived, and only the others are stored.
    let mut normals_xyz = normals_xyz;
    let w = positions_xyzw.column(3).to_vec();
    let frame = match (&patch_u_halfvec_xyz, &patch_v_halfvec_xyz) {
        (Some(u), Some(v)) => Some(crate::patch_normals::PatchFrameRows {
            w: &w,
            u: u.as_slice().expect("read in standard layout"),
            v: v.as_slice().expect("read in standard layout"),
        }),
        _ => None,
    };
    let underived = frame
        .as_ref()
        .filter(|_| derives_normals)
        .map(|f| f.underived().len());
    let names: Vec<String> = archive
        .file_names()
        .map(|n| n.map(|n| n.into_owned()))
        .collect::<Result<_, _>>()?;
    if let Some(e) = crate::patch_normals::normals_entries_error(
        point_count,
        underived,
        names.iter().map(String::as_str),
    ) {
        return Err(SfmrError::InvalidFormat(e));
    }
    if let (Some(frame), Some(k)) = (&frame, underived) {
        let stored: Vec<f32> = if k > 0 {
            read_binary_array(archive, &entries::points3d_underived_normals(k), k * 3)?
        } else {
            Vec::new()
        };
        let values = frame.rebuild(&stored).map_err(SfmrError::InvalidFormat)?;
        normals_xyz = Some(
            Array2::from_shape_vec((point_count, 3), values)
                .map_err(|e| SfmrError::ShapeMismatch(format!("normals reshape: {e}")))?,
        );
    }

    Ok(PointsSection {
        positions_xyzw,
        colors_rgb,
        reprojection_errors,
        normals_xyz,
        normal_confidence,
        point_constraints,
        constraint_distances,
        constraint_reference_images,
        patch_u_halfvec_xyz,
        patch_v_halfvec_xyz,
        patch_bitmaps_y_x_rgba,
    })
}

struct TracksSection {
    image_indexes: Array1<u32>,
    feature_indexes: Option<Array1<u32>>,
    keypoints_xy: Option<Array2<f32>>,
    observation_confidence: Option<Array1<u8>>,
    observation_readings: Option<ObservationReadingColumns>,
    point_indexes: Array1<u32>,
    observation_counts: Array1<u32>,
    reference_observations: Option<Array1<i32>>,
}

/// `has_patch_frame` is whether the points section carries the patch frame,
/// which is what says whether `reference_observations` is there, with or
/// without the bitmaps.
fn read_tracks_section(
    archive: &mut zip::ZipArchive<std::fs::File>,
    metadata: &SfmrMetadata,
    tracks_meta: &serde_json::Value,
    camera_indexes: &Array1<u32>,
    cameras: &[SfmrCamera],
    has_patch_frame: bool,
) -> Result<TracksSection, SfmrError> {
    let observation_count = metadata.observation_count as usize;
    let point_count = metadata.point_count as usize;
    let is_embedded = metadata.feature_source == FEATURE_SOURCE_EMBEDDED_PATCHES;
    // Tracks
    let image_indexes_vec: Vec<u32> = read_binary_array(
        archive,
        &entries::tracks_image_indexes(observation_count),
        observation_count,
    )?;
    let image_indexes = Array1::from_vec(image_indexes_vec);

    // A `sift_files` file references `.sift` features by index; an
    // `embedded_patches` file carries the sub-pixel `(u, v)` coordinate inline.
    let feature_indexes = if is_embedded {
        None
    } else {
        let feature_indexes_vec: Vec<u32> = read_binary_array(
            archive,
            &entries::tracks_feature_indexes(observation_count),
            observation_count,
        )?;
        Some(Array1::from_vec(feature_indexes_vec))
    };

    // The inline keypoint column: every observation's pixel, in either mode.
    // Only a `sift_files` file older than `SIFT_FILES_KEYPOINTS_FORMAT_VERSION`
    // may lack it, flagged by `has_keypoints_xy`; a newer one that says it has
    // none is refused.
    let has_keypoints_xy = is_embedded
        || tracks_meta
            .get("has_keypoints_xy")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
    if !has_keypoints_xy && metadata.version >= SIFT_FILES_KEYPOINTS_FORMAT_VERSION {
        return Err(SfmrError::InvalidFormat(format!(
            "a version {} sift_files file must carry tracks/keypoints_xy, and \
             tracks/metadata.json does not set has_keypoints_xy",
            metadata.version
        )));
    }
    let keypoints_xy = if has_keypoints_xy {
        let kp_vec: Vec<f32> = read_binary_array(
            archive,
            &entries::tracks_keypoints_xy(observation_count),
            observation_count * 2,
        )?;
        let keypoints_xy = Array2::from_shape_vec((observation_count, 2), kp_vec)
            .map_err(|e| SfmrError::ShapeMismatch(format!("keypoints_xy reshape: {e}")))?;
        validate_keypoints(
            &keypoints_xy,
            image_indexes.as_slice().unwrap(),
            camera_indexes.as_slice().unwrap(),
            cameras,
        )
        .map_err(SfmrError::InvalidFormat)?;
        Some(keypoints_xy)
    } else {
        None
    };

    // Optional per-observation confidence. Absent means "no information at
    // all", and reads as `None`. Independent of `feature_source`.
    let observation_confidence = if tracks_meta
        .get("has_observation_confidence")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        let confidence_vec: Vec<u8> = read_binary_array(
            archive,
            &entries::tracks_observation_confidence(observation_count),
            observation_count,
        )?;
        Some(Array1::from_vec(confidence_vec))
    } else {
        None
    };

    let observation_readings = read_observation_readings(archive, tracks_meta, observation_count)?;

    let point_indexes_name = entries::tracks_point_indexes(observation_count);
    let point_indexes_vec: Vec<u32> =
        read_binary_array(archive, &point_indexes_name, observation_count)?;
    let point_indexes = Array1::from_vec(point_indexes_vec);

    let observation_counts_vec: Vec<u32> = read_binary_array(
        archive,
        &entries::tracks_observation_counts(point_count),
        point_count,
    )?;
    let observation_counts = Array1::from_vec(observation_counts_vec);

    // The reference observations, present exactly with the patch frame.
    let reference_observations = if !has_patch_frame {
        None
    } else {
        let values: Vec<i32> = read_binary_array(
            archive,
            &entries::tracks_reference_observations(point_count),
            point_count,
        )?;
        validate_reference_observations(&values, observation_counts.as_slice().unwrap())
            .map_err(SfmrError::InvalidFormat)?;
        Some(Array1::from_vec(values))
    };

    Ok(TracksSection {
        image_indexes,
        feature_indexes,
        keypoints_xy,
        observation_confidence,
        observation_readings,
        point_indexes,
        observation_counts,
        reference_observations,
    })
}

/// The eight observation-reading columns, where `tracks/metadata.json` flags
/// them (`has_observation_readings`), with the options recorded beside them.
/// Absent in a file without the flag.
///
/// The five self-similarity value columns are `float16`, widened to `f32`
/// exactly, and the two scores are read from their stored `float16` ZNCC
/// deficits `w` as `1 − w` in `f32`.
fn read_observation_readings(
    archive: &mut zip::ZipArchive<std::fs::File>,
    tracks_meta: &serde_json::Value,
    observation_count: usize,
) -> Result<Option<ObservationReadingColumns>, SfmrError> {
    if !observation_readings_flagged(tracks_meta) {
        return Ok(None);
    }
    let options = observation_reading_options(tracks_meta).map_err(SfmrError::InvalidFormat)?;
    let mut builder = ReadingColumnsBuilder::default();
    for column in ReadingColumn::ALL {
        let len = observation_count * column.width();
        let name = column.entry(observation_count);
        if column.element_size() == 1 {
            builder.push_flags(read_binary_array(archive, &name, len)?);
        } else {
            let bits = read_binary_array::<_, u16>(archive, &name, len)?;
            let values = if column.stores_deficit() {
                scores_from_deficits(&bits)
            } else {
                widen_float16(&bits)
            };
            builder.push_f32(column, values);
        }
    }
    builder
        .finish(observation_count, options)
        .map(Some)
        .map_err(SfmrError::InvalidFormat)
}

fn read_rig_frames(
    archive: &mut zip::ZipArchive<std::fs::File>,
    image_count: usize,
) -> Result<Option<RigFrameData>, SfmrError> {
    // Rigs and frames (optional — only present in rig-aware reconstructions)
    let rig_frame_data = if archive.index_for_name(entries::rigs_metadata()).is_some() {
        let rigs_metadata: RigsMetadata = read_json_entry(archive, entries::rigs_metadata())?;
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
            archive,
            &entries::rigs_sensor_camera_indexes(sensor_count),
            sensor_count,
        )?;
        let sensor_camera_indexes = Array1::from_vec(sensor_camera_indexes_vec);

        let sensor_quaternions_vec: Vec<f64> = read_binary_array(
            archive,
            &entries::rigs_sensor_quaternions_wxyz(sensor_count),
            sensor_count * 4,
        )?;
        let sensor_quaternions_wxyz =
            Array2::from_shape_vec((sensor_count, 4), sensor_quaternions_vec).map_err(|e| {
                SfmrError::ShapeMismatch(format!("sensor quaternions reshape: {e}"))
            })?;

        let sensor_translations_vec: Vec<f64> = read_binary_array(
            archive,
            &entries::rigs_sensor_translations_xyz(sensor_count),
            sensor_count * 3,
        )?;
        let sensor_translations_xyz =
            Array2::from_shape_vec((sensor_count, 3), sensor_translations_vec).map_err(|e| {
                SfmrError::ShapeMismatch(format!("sensor translations reshape: {e}"))
            })?;

        // Frames
        let frames_metadata: FramesMetadata = read_json_entry(archive, entries::frames_metadata())?;
        let frame_count = frames_metadata.frame_count as usize;

        let rig_indexes_vec: Vec<u32> = read_binary_array(
            archive,
            &entries::frames_rig_indexes(frame_count),
            frame_count,
        )?;
        let rig_indexes = Array1::from_vec(rig_indexes_vec);

        let image_sensor_indexes_vec: Vec<u32> = read_binary_array(
            archive,
            &entries::frames_image_sensor_indexes(image_count),
            image_count,
        )?;
        let image_sensor_indexes = Array1::from_vec(image_sensor_indexes_vec);

        let image_frame_indexes_vec: Vec<u32> = read_binary_array(
            archive,
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
    } else if has_frames_entry(archive)? {
        return Err(SfmrError::InvalidFormat(
            "the frames/ section is present without a rigs/ section; the two must be \
             present together"
                .into(),
        ));
    } else {
        None
    };

    Ok(rig_frame_data)
}

/// Whether any archive entry belongs to the `frames/` section.
pub(crate) fn has_frames_entry<R: std::io::Read + std::io::Seek>(
    archive: &zip::ZipArchive<R>,
) -> Result<bool, zip::result::ZipError> {
    for name in archive.file_names() {
        if name?.starts_with(entries::frames_section_prefix()) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether any archive entry's path starts with `prefix`.
pub(crate) fn has_entry_under<R: std::io::Read + std::io::Seek>(
    archive: &zip::ZipArchive<R>,
    prefix: &str,
) -> Result<bool, zip::result::ZipError> {
    for name in archive.file_names() {
        if name?.starts_with(prefix) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether any archive entry belongs to the `measurements/` section.
pub(crate) fn has_measurements_entry<R: std::io::Read + std::io::Seek>(
    archive: &zip::ZipArchive<R>,
) -> Result<bool, zip::result::ZipError> {
    has_entry_under(archive, entries::measurements_section_prefix())
}

/// Whether the top-level metadata carries the count of any kind of
/// measurement, and so says the `measurements/` section is present.
pub(crate) fn declares_measurements(metadata: &SfmrMetadata) -> bool {
    metadata.distance_measurement_count.is_some()
}

/// Read and validate the optional `measurements/` section: each kind it
/// holds, then the section's presence against `measurements_xxh128`.
fn read_measurements_section(
    archive: &mut zip::ZipArchive<std::fs::File>,
    metadata: &SfmrMetadata,
    content_hash: &ContentHash,
    camera_indexes: &Array1<u32>,
    cameras: &[SfmrCamera],
) -> Result<MeasurementsData, SfmrError> {
    if metadata.version < MEASUREMENTS_FORMAT_VERSION {
        if has_measurements_entry(archive)? {
            return Err(SfmrError::InvalidFormat(format!(
                "version {} .sfmr file must not carry measurements/ entries (introduced in \
                 version {MEASUREMENTS_FORMAT_VERSION})",
                metadata.version
            )));
        }
        if content_hash.measurements_xxh128.is_some() {
            return Err(SfmrError::InvalidFormat(format!(
                "version {} .sfmr file must not carry measurements_xxh128 (introduced in \
                 version {MEASUREMENTS_FORMAT_VERSION})",
                metadata.version
            )));
        }
        return Ok(MeasurementsData::default());
    }

    let measurements = MeasurementsData {
        distance: read_distance_measurements(archive, metadata, camera_indexes, cameras)?,
    };
    match (
        measurements.is_empty(),
        content_hash.measurements_xxh128.is_some(),
    ) {
        (true, true) => Err(SfmrError::InvalidFormat(
            "content_hash.json.zst sets measurements_xxh128, but the file has no measurements/ \
             section (no kind of measurement)"
                .into(),
        )),
        (false, false) => Err(SfmrError::InvalidFormat(
            "the measurements/ section is present, but content_hash.json.zst has no \
             measurements_xxh128"
                .into(),
        )),
        _ => Ok(measurements),
    }
}

/// Read and validate the distance kind of the `measurements/` section, under
/// `measurements/distance/`, normalizing unit codes through the file's
/// `unit_names` legend onto the canonical [`LengthUnit`](crate::LengthUnit)
/// order. `None` when the file has no distance measurements.
fn read_distance_measurements(
    archive: &mut zip::ZipArchive<std::fs::File>,
    metadata: &SfmrMetadata,
    camera_indexes: &Array1<u32>,
    cameras: &[SfmrCamera],
) -> Result<Option<DistanceMeasurementsData>, SfmrError> {
    if archive
        .index_for_name(entries::distance_measurements_metadata())
        .is_none()
    {
        if has_entry_under(archive, entries::distance_measurements_prefix())? {
            return Err(SfmrError::InvalidFormat(
                "measurements/distance/ entries are present without \
                 measurements/distance/metadata.json.zst"
                    .into(),
            ));
        }
        if metadata.distance_measurement_count.is_some() {
            return Err(SfmrError::InvalidFormat(
                "metadata.json.zst sets distance_measurement_count, but the file has no \
                 measurements/distance/ entries"
                    .into(),
            ));
        }
        return Ok(None);
    }

    let Some(top_count) = metadata.distance_measurement_count else {
        return Err(SfmrError::InvalidFormat(
            "measurements/distance/ is present, but metadata.json.zst has no \
             distance_measurement_count"
                .into(),
        ));
    };

    let meas_meta: serde_json::Value =
        read_json_entry(archive, entries::distance_measurements_metadata())?;
    let section_count = meas_meta
        .get("distance_measurement_count")
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| {
            SfmrError::InvalidFormat(
                "measurements/distance/metadata.json.zst has no u32 distance_measurement_count"
                    .into(),
            )
        })?;
    if section_count == 0 {
        return Err(SfmrError::InvalidFormat(
            "measurements/distance/metadata.json.zst distance_measurement_count must be >= 1"
                .into(),
        ));
    }
    if section_count != top_count {
        return Err(SfmrError::InvalidFormat(format!(
            "measurements/distance/metadata.json.zst distance_measurement_count \
             ({section_count}) doesn't match top-level metadata distance_measurement_count \
             ({top_count})"
        )));
    }

    let legend = read_measurement_unit_legend(&meas_meta).map_err(SfmrError::InvalidFormat)?;
    let n = section_count as usize;

    let distances_vec: Vec<f32> =
        read_binary_array(archive, &entries::distance_measurements_distances(n), n)?;
    let distances = Array1::from_vec(distances_vec);

    let endpoints_vec: Vec<f32> = read_binary_array(
        archive,
        &entries::distance_measurements_endpoints_xy(n),
        n * 4,
    )?;
    let endpoints_xy = Array3::from_shape_vec((n, 2, 2), endpoints_vec).map_err(|e| {
        SfmrError::ShapeMismatch(format!("measurements/distance/endpoints_xy reshape: {e}"))
    })?;

    let image_indexes_vec: Vec<u32> = read_binary_array(
        archive,
        &entries::distance_measurements_image_indexes(n),
        n * 2,
    )?;
    let image_indexes = Array2::from_shape_vec((n, 2), image_indexes_vec).map_err(|e| {
        SfmrError::ShapeMismatch(format!("measurements/distance/image_indexes reshape: {e}"))
    })?;

    let labels: Vec<String> = read_json_entry(archive, entries::distance_measurements_labels())?;

    let units_raw: Vec<u8> =
        read_binary_array(archive, &entries::distance_measurements_units(n), n)?;
    let resolved_units =
        resolve_measurement_units(&units_raw, &legend).map_err(SfmrError::InvalidFormat)?;
    let units = Array1::from_vec(resolved_units.iter().map(|u| u.code()).collect());

    let measurements = DistanceMeasurementsData {
        distances,
        units,
        image_indexes,
        endpoints_xy,
        labels,
    };
    validate_distance_measurements(
        &measurements,
        camera_indexes.as_slice().expect("read in standard layout"),
        cameras,
    )
    .map_err(SfmrError::InvalidFormat)?;

    Ok(Some(measurements))
}

/// Whether `images/metadata.json` says the thumbnail entry is present: its
/// `has_thumbnails` key, which is required, or a message when the key is
/// missing or not a boolean.
pub(crate) fn images_meta_has_thumbnails(images_meta: &serde_json::Value) -> Result<bool, String> {
    images_meta
        .get("has_thumbnails")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| "images/metadata.json.zst has no boolean has_thumbnails".to_string())
}

/// Resolve the workspace directory for a `.sfmr` file.
///
/// Strategy (from the spec), the one
/// [`sfmtool_archive_io::resolve_workspace_dir`] holds for every format that
/// records a workspace:
/// 1. Try `workspace.relative_path` from the `.sfmr` file's directory
/// 2. Fall back to `workspace.absolute_path`
/// 3. Fall back to searching upward from the `.sfmr` file for `.sfm-workspace.json`
///
/// A recorded directory counts only when it holds `.sfm-workspace.json`. An
/// empty `relative_path` or `absolute_path` means none was recorded, and its
/// step is skipped.
pub fn resolve_workspace_dir(
    sfmr_path: &Path,
    metadata: &SfmrMetadata,
) -> Result<PathBuf, SfmrError> {
    let relative_path = &metadata.workspace.relative_path;
    let absolute_path = &metadata.workspace.absolute_path;
    if let Some(dir) =
        sfmtool_archive_io::resolve_workspace_dir(sfmr_path, relative_path, absolute_path)
    {
        return Ok(dir);
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
