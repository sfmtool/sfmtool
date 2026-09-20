// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sift` file integrity verification.

use std::path::Path;

use sfmtool_archive_io::{format_hash, read_zst_entry};

use crate::types::*;

/// Verify integrity of a `.sift` file using content hashes.
///
/// Returns `Ok((true, []))` if all hashes match, `Ok((false, errors))` with
/// details if verification fails. Returns `Err` only for I/O errors.
pub fn verify_sift(path: &Path) -> Result<(bool, Vec<String>), SiftError> {
    let file = std::fs::File::open(path).map_err(|e| SiftError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut errors = Vec::new();

    // Read stored hashes
    let content_hash_bytes = read_zst_entry(&mut archive, "content_hash.json.zst")?;
    let stored: SiftContentHash = serde_json::from_slice(&content_hash_bytes)?;

    // Read metadata for feature_count
    let metadata_raw = read_zst_entry(&mut archive, "metadata.json.zst")?;
    let metadata: SiftMetadata = serde_json::from_slice(&metadata_raw)?;
    let feature_count = metadata.feature_count as usize;

    // Read feature tool metadata (raw bytes for hashing)
    let feature_tool_raw = read_zst_entry(&mut archive, "feature_tool_metadata.json.zst")?;

    // Verify metadata_xxh128
    let metadata_hash = xxhash_rust::xxh3::xxh3_128(&metadata_raw);
    if format_hash(metadata_hash) != stored.metadata_xxh128 {
        errors.push(format!(
            "Metadata hash mismatch: computed {}, stored {}",
            format_hash(metadata_hash),
            stored.metadata_xxh128
        ));
    }

    // Verify feature_tool_xxh128
    let feature_tool_hash = xxhash_rust::xxh3::xxh3_128(&feature_tool_raw);
    if format_hash(feature_tool_hash) != stored.feature_tool_xxh128 {
        errors.push(format!(
            "Feature tool hash mismatch: computed {}, stored {}",
            format_hash(feature_tool_hash),
            stored.feature_tool_xxh128
        ));
    }

    // Recompute content hash from individual digests
    let mut content_hash_digests = sfmtool_archive_io::SectionDigests::new();

    // 1. feature_tool_metadata.json
    content_hash_digests.push_bytes(&feature_tool_raw);

    // 2. metadata.json
    content_hash_digests.push_bytes(&metadata_raw);

    // 3. features/positions_xy
    let pos_raw = read_zst_entry(&mut archive, &positions_entry_name(feature_count))?;
    content_hash_digests.push_bytes(&pos_raw);

    // 4. features/affine_shapes
    let shape_raw = read_zst_entry(&mut archive, &affine_shapes_entry_name(feature_count))?;
    content_hash_digests.push_bytes(&shape_raw);

    // 5. features/descriptors
    let desc_raw = read_zst_entry(&mut archive, &descriptors_entry_name(feature_count))?;
    content_hash_digests.push_bytes(&desc_raw);

    // 6. thumbnail_y_x_rgb
    let thumb_raw = read_zst_entry(&mut archive, &thumbnail_entry_name())?;
    content_hash_digests.push_bytes(&thumb_raw);

    let content_hash = content_hash_digests.finish();
    if format_hash(content_hash) != stored.content_xxh128 {
        errors.push(format!(
            "Content hash mismatch: computed {}, stored {}",
            format_hash(content_hash),
            stored.content_xxh128
        ));
    }

    Ok((errors.is_empty(), errors))
}
