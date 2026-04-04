// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sift` file reading.

use std::io::{Read, Seek};
use std::path::Path;

use ndarray::{Array2, Array3};
use zip::ZipArchive;

use crate::archive_io::{read_binary_array, read_json_entry};

use crate::types::*;

fn open_file(path: &Path) -> Result<std::fs::File, SiftError> {
    std::fs::File::open(path).map_err(|e| SiftError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })
}

/// Read a complete `.sift` file into columnar data.
pub fn read_sift(path: &Path) -> Result<SiftData, SiftError> {
    let file = open_file(path)?;
    let mut archive = ZipArchive::new(file)?;
    read_sift_from_archive(&mut archive, None)
}

/// Read the first `count` features from a `.sift` file.
///
/// If `count` exceeds `feature_count`, returns all features.
pub fn read_sift_partial(path: &Path, count: usize) -> Result<SiftData, SiftError> {
    let file = open_file(path)?;
    let mut archive = ZipArchive::new(file)?;
    read_sift_from_archive(&mut archive, Some(count))
}

/// Read only feature positions from a `.sift` file.
///
/// Returns `(N, 2)` positions as `Vec<[f32; 2]>`. Reads only the positions
/// binary entry — skips descriptors, affine shapes, and thumbnail.
/// If `count` exceeds `feature_count`, returns all features.
pub fn read_sift_positions(path: &Path, count: usize) -> Result<Vec<[f32; 2]>, SiftError> {
    let file = open_file(path)?;
    let mut archive = ZipArchive::new(file)?;

    let metadata: SiftMetadata = read_json_entry(&mut archive, "metadata.json.zst")?;
    let total = metadata.feature_count as usize;
    let read_count = count.min(total);

    let positions = read_partial_f32_array(
        &mut archive,
        &format!("features/positions_xy.{total}.2.float32.zst"),
        read_count,
        2,
    )?;

    let mut result = Vec::with_capacity(read_count);
    for i in 0..positions.nrows() {
        result.push([positions[[i, 0]], positions[[i, 1]]]);
    }
    Ok(result)
}

/// Read only metadata from a `.sift` file (fast, no binary data).
pub fn read_sift_metadata(
    path: &Path,
) -> Result<(FeatureToolMetadata, SiftMetadata, SiftContentHash), SiftError> {
    let file = open_file(path)?;
    let mut archive = ZipArchive::new(file)?;

    let feature_tool_metadata: FeatureToolMetadata =
        read_json_entry(&mut archive, "feature_tool_metadata.json.zst")?;
    let metadata: SiftMetadata = read_json_entry(&mut archive, "metadata.json.zst")?;
    let content_hash: SiftContentHash = read_json_entry(&mut archive, "content_hash.json.zst")?;

    Ok((feature_tool_metadata, metadata, content_hash))
}

/// Internal: read sift data from an open archive, optionally truncating to `max_count` features.
fn read_sift_from_archive<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    max_count: Option<usize>,
) -> Result<SiftData, SiftError> {
    // Metadata
    let feature_tool_metadata: FeatureToolMetadata =
        read_json_entry(archive, "feature_tool_metadata.json.zst")?;
    let metadata: SiftMetadata = read_json_entry(archive, "metadata.json.zst")?;
    let content_hash: SiftContentHash = read_json_entry(archive, "content_hash.json.zst")?;

    let total = metadata.feature_count as usize;
    let read_count = match max_count {
        Some(c) => c.min(total),
        None => total,
    };

    // Positions: (N, 2) f32
    let positions_xy = if read_count == total {
        let pos_vec: Vec<f32> = read_binary_array(
            archive,
            &format!("features/positions_xy.{total}.2.float32.zst"),
            total * 2,
        )?;
        Array2::from_shape_vec((total, 2), pos_vec)
            .map_err(|e| SiftError::ShapeMismatch(format!("positions_xy reshape: {e}")))?
    } else {
        read_partial_f32_array(
            archive,
            &format!("features/positions_xy.{total}.2.float32.zst"),
            read_count,
            2,
        )?
    };

    // Affine shapes: (N, 2, 2) f32
    let affine_shapes = if read_count == total {
        let shape_vec: Vec<f32> = read_binary_array(
            archive,
            &format!("features/affine_shapes.{total}.2.2.float32.zst"),
            total * 4,
        )?;
        Array3::from_shape_vec((total, 2, 2), shape_vec)
            .map_err(|e| SiftError::ShapeMismatch(format!("affine_shapes reshape: {e}")))?
    } else {
        let flat = read_partial_f32_array(
            archive,
            &format!("features/affine_shapes.{total}.2.2.float32.zst"),
            read_count,
            4,
        )?;
        flat.into_shape_with_order((read_count, 2, 2))
            .map_err(|e| SiftError::ShapeMismatch(format!("affine_shapes reshape: {e}")))?
    };

    // Descriptors: (N, 128) u8
    let descriptors = if read_count == total {
        let desc_vec: Vec<u8> = read_binary_array(
            archive,
            &format!("features/descriptors.{total}.128.uint8.zst"),
            total * 128,
        )?;
        Array2::from_shape_vec((total, 128), desc_vec)
            .map_err(|e| SiftError::ShapeMismatch(format!("descriptors reshape: {e}")))?
    } else {
        read_partial_u8_array(
            archive,
            &format!("features/descriptors.{total}.128.uint8.zst"),
            read_count,
            128,
        )?
    };

    // Thumbnail: (128, 128, 3) u8 — always read in full regardless of max_count
    let thumb_vec: Vec<u8> = read_binary_array(
        archive,
        "thumbnail_y_x_rgb.128.128.3.uint8.zst",
        128 * 128 * 3,
    )?;
    let thumbnail_y_x_rgb = Array3::from_shape_vec((128, 128, 3), thumb_vec)
        .map_err(|e| SiftError::ShapeMismatch(format!("thumbnail_y_x_rgb reshape: {e}")))?;

    Ok(SiftData {
        feature_tool_metadata,
        metadata,
        content_hash,
        positions_xy,
        affine_shapes,
        descriptors,
        thumbnail_y_x_rgb,
    })
}

/// Read the first `count` items from a zst-compressed f32 binary entry.
/// Each item has `cols` f32 values.
fn read_partial_f32_array<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
    count: usize,
    cols: usize,
) -> Result<Array2<f32>, SiftError> {
    let bytes = crate::archive_io::read_zst_entry(archive, name)?;
    let item_bytes = count * cols * std::mem::size_of::<f32>();
    let read_bytes = item_bytes.min(bytes.len());
    let slice: &[f32] = bytemuck::cast_slice(&bytes[..read_bytes]);
    let actual_count = slice.len() / cols;
    Array2::from_shape_vec((actual_count, cols), slice.to_vec())
        .map_err(|e| SiftError::ShapeMismatch(format!("partial f32 reshape: {e}")))
}

/// Read the first `count` items from a zst-compressed u8 binary entry.
/// Each item has `cols` u8 values.
fn read_partial_u8_array<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
    count: usize,
    cols: usize,
) -> Result<Array2<u8>, SiftError> {
    let bytes = crate::archive_io::read_zst_entry(archive, name)?;
    let item_bytes = count * cols;
    let read_bytes = item_bytes.min(bytes.len());
    let slice = &bytes[..read_bytes];
    let actual_count = slice.len() / cols;
    Array2::from_shape_vec((actual_count, cols), slice.to_vec())
        .map_err(|e| SiftError::ShapeMismatch(format!("partial u8 reshape: {e}")))
}