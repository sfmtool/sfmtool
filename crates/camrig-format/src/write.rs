// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.camrig` file writing.

use std::path::Path;

use zip::ZipWriter;

use crate::archive_io::{format_hash, write_binary_entry, write_json_entry};
use crate::types::*;

/// Write columnar camera-rig data to a `.camrig` file.
///
/// Validates the structural constraints (see [`CamRigData::validate`]) before
/// writing, so a `.camrig` file on disk is always well-formed. Computes
/// content hashes automatically; the `content_hash` field in `data` is
/// ignored on write and recomputed from the actual data.
pub fn write_camrig(path: &Path, data: &CamRigData, zstd_level: i32) -> Result<(), CamRigError> {
    data.validate()?;
    write_camrig_unchecked(path, data, zstd_level)
}

/// Write a `.camrig` file without validating structural constraints.
///
/// Used by tests to produce deliberately-invalid files; `write_camrig` is the
/// public entry point and validates first.
pub(crate) fn write_camrig_unchecked(
    path: &Path,
    data: &CamRigData,
    zstd_level: i32,
) -> Result<(), CamRigError> {
    let s = data.sensor_count();

    let file = std::fs::File::create(path).map_err(|e| CamRigError::IoPath {
        operation: "Failed to create file",
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut zip = ZipWriter::new(file);

    // metadata.json
    let metadata_bytes =
        write_json_entry(&mut zip, "metadata.json.zst", &data.metadata, zstd_level)?;
    let metadata_xxh128 = xxhash_rust::xxh3::xxh3_128(&metadata_bytes);

    let mut digests: Vec<u8> = Vec::new();
    let push = |digests: &mut Vec<u8>, bytes: &[u8]| {
        digests.extend_from_slice(&xxhash_rust::xxh3::xxh3_128(bytes).to_be_bytes());
    };
    push(&mut digests, &metadata_bytes);

    // cameras/metadata.json
    let cameras_bytes = write_json_entry(
        &mut zip,
        "cameras/metadata.json.zst",
        &data.cameras,
        zstd_level,
    )?;
    push(&mut digests, &cameras_bytes);

    // sensors/image_file_patterns.json
    let patterns_bytes = write_json_entry(
        &mut zip,
        "sensors/image_file_patterns.json.zst",
        &data.sensor_image_patterns,
        zstd_level,
    )?;
    push(&mut digests, &patterns_bytes);

    // sensors/camera_indexes
    let cam_idx_bytes: &[u8] = bytemuck::cast_slice(&data.camera_indexes);
    write_binary_entry(
        &mut zip,
        &format!("sensors/camera_indexes.{s}.uint32.zst"),
        cam_idx_bytes,
        zstd_level,
    )?;
    push(&mut digests, cam_idx_bytes);

    // sensors/quaternions_wxyz — `as_standard_layout` makes a contiguous copy
    // only if the caller handed us a non-contiguous array; the common case
    // (a freshly built array) borrows without copying.
    let quaternions = data.quaternions_wxyz.as_standard_layout();
    let quat_bytes: &[u8] = bytemuck::cast_slice(quaternions.as_slice().unwrap());
    write_binary_entry(
        &mut zip,
        &format!("sensors/quaternions_wxyz.{s}.4.float64.zst"),
        quat_bytes,
        zstd_level,
    )?;
    push(&mut digests, quat_bytes);

    // sensors/translations_xyz
    let translations = data.translations_xyz.as_standard_layout();
    let trans_bytes: &[u8] = bytemuck::cast_slice(translations.as_slice().unwrap());
    write_binary_entry(
        &mut zip,
        &format!("sensors/translations_xyz.{s}.3.float64.zst"),
        trans_bytes,
        zstd_level,
    )?;
    push(&mut digests, trans_bytes);

    // content_hash.json
    let content_xxh128 = xxhash_rust::xxh3::xxh3_128(&digests);
    let content_hash = CamRigContentHash {
        metadata_xxh128: format_hash(metadata_xxh128),
        content_xxh128: format_hash(content_xxh128),
    };
    write_json_entry(&mut zip, "content_hash.json.zst", &content_hash, zstd_level)?;

    zip.finish()?;
    Ok(())
}
