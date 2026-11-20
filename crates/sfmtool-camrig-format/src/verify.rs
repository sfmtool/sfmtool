// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.camrig` file integrity verification.

use std::path::Path;

use crate::types::*;
use sfmtool_archive_io::{format_hash, read_zst_entry};

/// Verify a `.camrig` file: content-hash integrity *and* structural validity.
///
/// Recomputes the stored content hashes, then checks the structural
/// constraints from the spec (*Data ordering and constraints*; see
/// [`CamRigData::validate`]). A file can be byte-intact yet structurally
/// invalid — an out-of-range camera index, a non-unit quaternion, a count
/// that disagrees with a table length — and both classes of problem are
/// reported.
///
/// A `metadata.version` the readers refuse (below
/// [`CAMRIG_OLDEST_READ_VERSION`], 0 or newer than [`CAMRIG_FORMAT_VERSION`])
/// is reported as the one error, with the readers' message, and nothing else
/// is checked.
///
/// Returns `Ok((true, []))` if everything passes, `Ok((false, errors))` with
/// details otherwise. Returns `Err` for I/O errors or a malformed archive
/// (one that cannot be opened, or that is missing or cannot decode a
/// required member).
pub fn verify_camrig(path: &Path) -> Result<(bool, Vec<String>), CamRigError> {
    let file = open_file(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut errors = Vec::new();

    // Read metadata before anything else so a version the readers refuse stops
    // without producing misleading diagnostics.
    let metadata_raw = read_zst_entry(&mut archive, "metadata.json.zst")?;
    let metadata = match crate::read::parse_metadata(serde_json::from_slice(&metadata_raw)?, path) {
        Ok(metadata) => metadata,
        Err(e @ (CamRigError::VersionNoLongerRead { .. } | CamRigError::InvalidFormat(_))) => {
            return Ok((false, vec![e.to_string()]));
        }
        Err(e) => return Err(e),
    };
    let s = metadata.sensor_count;

    let content_hash_raw = read_zst_entry(&mut archive, "content_hash.json.zst")?;
    let stored: CamRigContentHash = serde_json::from_slice(&content_hash_raw)?;

    // metadata_xxh128
    let metadata_hash = xxhash_rust::xxh3::xxh3_128(&metadata_raw);
    if format_hash(metadata_hash) != stored.metadata_xxh128 {
        errors.push(format!(
            "Metadata hash mismatch: computed {}, stored {}",
            format_hash(metadata_hash),
            stored.metadata_xxh128
        ));
    }

    // content_xxh128 — hash of hashes over the ordered member list.
    let members = [
        "metadata.json.zst".to_string(),
        "cameras/metadata.json.zst".to_string(),
        "sensors/image_file_patterns.json.zst".to_string(),
        format!("sensors/camera_indexes.{s}.uint32.zst"),
        format!("sensors/quaternions_wxyz.{s}.4.float64.zst"),
        format!("sensors/translations_xyz.{s}.3.float64.zst"),
    ];
    let mut digests = sfmtool_archive_io::SectionDigests::new();
    for name in &members {
        let raw = read_zst_entry(&mut archive, name)?;
        digests.push_bytes(&raw);
    }
    let content_hash = digests.finish();
    if format_hash(content_hash) != stored.content_xxh128 {
        errors.push(format!(
            "Content hash mismatch: computed {}, stored {}",
            format_hash(content_hash),
            stored.content_xxh128
        ));
    }

    // Structural validity. This re-opens and re-decompresses the archive —
    // a redundant pass, but verify is not a hot path.
    match crate::read::read_camrig_unchecked(path) {
        Ok(data) => {
            if let Err(e) = data.validate() {
                errors.push(format!("Structural validation failed: {e}"));
            }
        }
        Err(e) => {
            errors.push(format!("Could not parse archive contents: {e}"));
        }
    }

    Ok((errors.is_empty(), errors))
}
