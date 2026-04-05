// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sift` file writing.

use std::path::Path;

use zip::ZipWriter;

use crate::archive_io::{format_hash, write_binary_entry, write_json_entry};

use crate::types::*;

/// Write columnar data to a `.sift` file.
///
/// Computes content hashes automatically. The `content_hash` field in `data`
/// is ignored on write (recomputed from the actual data).
pub fn write_sift(path: &Path, data: &SiftData, zstd_level: i32) -> Result<(), SiftError> {
    let feature_count = data.metadata.feature_count as usize;

    // Validate dimensions
    validate_dimensions(data, feature_count)?;

    let file = std::fs::File::create(path).map_err(|e| SiftError::IoPath {
        operation: "Failed to create file",
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut zip = ZipWriter::new(file);

    // Encode metadata to bytes and track for hashing
    let feature_tool_bytes = write_json_entry(
        &mut zip,
        "feature_tool_metadata.json.zst",
        &data.feature_tool_metadata,
        zstd_level,
    )?;
    let feature_tool_xxh128 = xxhash_rust::xxh3::xxh3_128(&feature_tool_bytes);

    let metadata_bytes =
        write_json_entry(&mut zip, "metadata.json.zst", &data.metadata, zstd_level)?;
    let metadata_xxh128 = xxhash_rust::xxh3::xxh3_128(&metadata_bytes);

    // Accumulate hash digests for content hash
    let mut content_hash_digests: Vec<u8> = Vec::new();
    content_hash_digests
        .extend_from_slice(&xxhash_rust::xxh3::xxh3_128(&feature_tool_bytes).to_be_bytes());
    content_hash_digests
        .extend_from_slice(&xxhash_rust::xxh3::xxh3_128(&metadata_bytes).to_be_bytes());

    // Position data
    let pos_bytes = write_binary_entry(
        &mut zip,
        &format!("features/positions_xy.{feature_count}.2.float32.zst"),
        bytemuck::cast_slice(data.positions_xy.as_slice().unwrap()),
        zstd_level,
    )?;
    content_hash_digests.extend_from_slice(&xxhash_rust::xxh3::xxh3_128(&pos_bytes).to_be_bytes());

    // Affine shape data
    let shape_bytes = write_binary_entry(
        &mut zip,
        &format!("features/affine_shapes.{feature_count}.2.2.float32.zst"),
        bytemuck::cast_slice(data.affine_shapes.as_slice().unwrap()),
        zstd_level,
    )?;
    content_hash_digests
        .extend_from_slice(&xxhash_rust::xxh3::xxh3_128(&shape_bytes).to_be_bytes());

    // Descriptor data
    let desc_bytes = write_binary_entry(
        &mut zip,
        &format!("features/descriptors.{feature_count}.128.uint8.zst"),
        data.descriptors.as_slice().unwrap(),
        zstd_level,
    )?;
    content_hash_digests.extend_from_slice(&xxhash_rust::xxh3::xxh3_128(&desc_bytes).to_be_bytes());

    // Thumbnail data
    let thumb_bytes = write_binary_entry(
        &mut zip,
        "thumbnail_y_x_rgb.128.128.3.uint8.zst",
        data.thumbnail_y_x_rgb.as_slice().unwrap(),
        zstd_level,
    )?;
    content_hash_digests
        .extend_from_slice(&xxhash_rust::xxh3::xxh3_128(&thumb_bytes).to_be_bytes());

    // Compute and write content hash
    let content_xxh128 = xxhash_rust::xxh3::xxh3_128(&content_hash_digests);
    let content_hash = SiftContentHash {
        metadata_xxh128: format_hash(metadata_xxh128),
        feature_tool_xxh128: format_hash(feature_tool_xxh128),
        content_xxh128: format_hash(content_xxh128),
    };
    write_json_entry(&mut zip, "content_hash.json.zst", &content_hash, zstd_level)?;

    zip.finish()?;
    Ok(())
}

fn validate_dimensions(data: &SiftData, feature_count: usize) -> Result<(), SiftError> {
    if data.positions_xy.shape() != [feature_count, 2] {
        return Err(SiftError::ShapeMismatch(format!(
            "positions_xy shape {:?} != [{feature_count}, 2]",
            data.positions_xy.shape()
        )));
    }
    if data.affine_shapes.shape() != [feature_count, 2, 2] {
        return Err(SiftError::ShapeMismatch(format!(
            "affine_shapes shape {:?} != [{feature_count}, 2, 2]",
            data.affine_shapes.shape()
        )));
    }
    if data.descriptors.shape() != [feature_count, 128] {
        return Err(SiftError::ShapeMismatch(format!(
            "descriptors shape {:?} != [{feature_count}, 128]",
            data.descriptors.shape()
        )));
    }
    if data.thumbnail_y_x_rgb.shape() != [128, 128, 3] {
        return Err(SiftError::ShapeMismatch(format!(
            "thumbnail_y_x_rgb shape {:?} != [128, 128, 3]",
            data.thumbnail_y_x_rgb.shape()
        )));
    }
    Ok(())
}
