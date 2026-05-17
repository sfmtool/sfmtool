// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.camrig` file reading.

use std::path::Path;

use ndarray::Array2;
use zip::ZipArchive;

use crate::archive_io::{read_binary_array, read_json_entry};
use crate::types::*;

/// Read only metadata from a `.camrig` file (fast, no binary data).
pub fn read_camrig_metadata(
    path: &Path,
) -> Result<(CamRigMetadata, CamRigContentHash), CamRigError> {
    let file = open_file(path)?;
    let mut archive = ZipArchive::new(file)?;
    let metadata: CamRigMetadata = read_json_entry(&mut archive, "metadata.json.zst")?;
    let content_hash: CamRigContentHash = read_json_entry(&mut archive, "content_hash.json.zst")?;
    Ok((metadata, content_hash))
}

/// Read a complete `.camrig` file into columnar data.
///
/// Enforces the structural constraints from the spec (see
/// [`CamRigData::validate`]) — a file with valid content hashes but
/// inconsistent tables (out-of-range camera index, non-unit quaternion,
/// mismatched counts) is rejected here rather than panicking a downstream
/// consumer.
pub fn read_camrig(path: &Path) -> Result<CamRigData, CamRigError> {
    let data = read_camrig_unchecked(path)?;
    data.validate()?;
    Ok(data)
}

/// Read a `.camrig` file without enforcing structural constraints.
///
/// Decompresses and parses every member but does not call
/// [`CamRigData::validate`]. Used by [`crate::verify_camrig`] so it can
/// report structural problems as findings rather than as a hard error.
pub(crate) fn read_camrig_unchecked(path: &Path) -> Result<CamRigData, CamRigError> {
    let file = open_file(path)?;
    let mut archive = ZipArchive::new(file)?;

    let metadata: CamRigMetadata = read_json_entry(&mut archive, "metadata.json.zst")?;
    let content_hash: CamRigContentHash = read_json_entry(&mut archive, "content_hash.json.zst")?;
    let cameras: Vec<CamRigCamera> = read_json_entry(&mut archive, "cameras/metadata.json.zst")?;
    let sensor_image_patterns: Vec<String> =
        read_json_entry(&mut archive, "sensors/image_file_patterns.json.zst")?;

    let s = metadata.sensor_count as usize;

    let camera_indexes: Vec<u32> = read_binary_array(
        &mut archive,
        &format!("sensors/camera_indexes.{s}.uint32.zst"),
        s,
    )?;

    let quat_vec: Vec<f64> = read_binary_array(
        &mut archive,
        &format!("sensors/quaternions_wxyz.{s}.4.float64.zst"),
        s * 4,
    )?;
    let quaternions_wxyz = Array2::from_shape_vec((s, 4), quat_vec)
        .map_err(|e| CamRigError::ShapeMismatch(format!("quaternions_wxyz reshape: {e}")))?;

    let trans_vec: Vec<f64> = read_binary_array(
        &mut archive,
        &format!("sensors/translations_xyz.{s}.3.float64.zst"),
        s * 3,
    )?;
    let translations_xyz = Array2::from_shape_vec((s, 3), trans_vec)
        .map_err(|e| CamRigError::ShapeMismatch(format!("translations_xyz reshape: {e}")))?;

    Ok(CamRigData {
        metadata,
        content_hash,
        cameras,
        sensor_image_patterns,
        camera_indexes,
        quaternions_wxyz,
        translations_xyz,
    })
}
