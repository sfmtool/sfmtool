// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.camrig` file reading.

use std::path::Path;

use ndarray::Array2;
use zip::ZipArchive;

use crate::types::*;
use sfmtool_archive_io::{read_binary_array, read_json_entry};

/// Refuse a `metadata.version` this build does not read: one below
/// [`CAMRIG_OLDEST_READ_VERSION`], with [`CamRigError::VersionNoLongerRead`],
/// which names `path` and the last release that reads it; and one of 0, which
/// no writer has ever produced, or newer than [`CAMRIG_FORMAT_VERSION`], whose
/// layout is unknown so reading it with current-version assumptions would
/// misparse silently.
fn check_version(version: u32, path: &Path) -> Result<(), CamRigError> {
    if (1..CAMRIG_OLDEST_READ_VERSION).contains(&version) {
        return Err(CamRigError::VersionNoLongerRead {
            path: path.to_path_buf(),
            version,
        });
    }
    if !(CAMRIG_OLDEST_READ_VERSION..=CAMRIG_FORMAT_VERSION).contains(&version) {
        return Err(CamRigError::InvalidFormat(format!(
            "unsupported .camrig format version {version} (this build reads versions \
             {CAMRIG_OLDEST_READ_VERSION} to {CAMRIG_FORMAT_VERSION})"
        )));
    }
    Ok(())
}

/// Parse `metadata.json` from its JSON value, checking its version with
/// [`check_version`] first, so that a file of a version this build does not
/// read is refused for its version rather than for a key that version lacks.
pub(crate) fn parse_metadata(
    value: serde_json::Value,
    path: &Path,
) -> Result<CamRigMetadata, CamRigError> {
    if let Some(version) = value.get("version").and_then(serde_json::Value::as_u64) {
        check_version(u32::try_from(version).unwrap_or(u32::MAX), path)?;
    }
    Ok(serde_json::from_value(value)?)
}

/// Read only metadata from a `.camrig` file (fast, no binary data).
///
/// A file of a version this build does not read is refused, as
/// [`read_camrig`] refuses it.
pub fn read_camrig_metadata(
    path: &Path,
) -> Result<(CamRigMetadata, CamRigContentHash), CamRigError> {
    let file = open_file(path)?;
    let mut archive = ZipArchive::new(file)?;
    let metadata = parse_metadata(read_json_entry(&mut archive, "metadata.json.zst")?, path)?;
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
///
/// A file of a version below [`CAMRIG_OLDEST_READ_VERSION`] is refused with
/// [`CamRigError::VersionNoLongerRead`], and one of version 0 or newer than
/// [`CAMRIG_FORMAT_VERSION`] as unsupported, before anything that depends on
/// the version is parsed.
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

    let metadata = parse_metadata(read_json_entry(&mut archive, "metadata.json.zst")?, path)?;
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
