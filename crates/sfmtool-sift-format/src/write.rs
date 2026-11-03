// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sift` file writing.

use std::path::Path;

use zip::ZipWriter;

use sfmtool_archive_io::{format_hash, write_binary_entry, write_json_entry};

use crate::types::*;

/// Write columnar data to a `.sift` file.
///
/// Computes content hashes automatically. The `content_hash` field in `data`
/// is ignored on write (recomputed from the actual data). Checks `data` with
/// [`validate_sift_data`] first, so a write never produces a file this crate
/// cannot read back, and a rejected write leaves no file behind.
pub fn write_sift(path: &Path, data: &SiftData, zstd_level: i32) -> Result<(), SiftError> {
    validate_sift_data(data)?;
    write_sift_any_version(path, data, zstd_level)
}

/// Check that `data` is a `.sift` file's content, without writing anything.
///
/// Rejects a `metadata.version` that [`crate::read_sift`] would reject
/// ([`SiftError::InvalidFormat`]), including a version 2 file without the two
/// image readings and a version 1 file with either; `feature_options` that is
/// not a JSON object, a version 2 file whose `feature_options` carry no valid
/// `image_to_gray` object, a version 1 file whose `feature_options` carry one,
/// and a reading that is not finite or, for the sharpness, `Exx` and `Eyy`, is
/// negative ([`SiftError::InvalidMetadata`]); and an array whose shape does
/// not match `metadata.feature_count` or, for the thumbnail, the
/// [`THUMBNAIL_SIZE`] square ([`SiftError::ShapeMismatch`]). A writer that
/// parses its metadata from JSON parses it with
/// [`SiftMetadata::from_json_for_write`] and
/// [`FeatureToolMetadata::from_json_for_write`], which refuse a key the
/// format does not define.
///
/// The `image_to_gray` check is of the object's shape: a string `formula`, an
/// optional integer `quantize_bits` from 1 to [`MAX_QUANTIZE_BITS`], and no
/// other key. Whether the formula follows the grammar is checked by the
/// writer that evaluates it.
///
/// [`write_sift`] calls it; a caller that must tell a caller's mistake from a
/// failed write, such as the Python binding, which raises `ValueError` for
/// one and `OSError` for the other, calls it first.
pub fn validate_sift_data(data: &SiftData) -> Result<(), SiftError> {
    crate::read::check_metadata(&data.metadata)?;
    if !data.feature_tool_metadata.feature_options.is_object() {
        return Err(SiftError::InvalidMetadata(format!(
            "feature_options must be a JSON object, got {}",
            data.feature_tool_metadata.feature_options
        )));
    }
    check_version_content(&data.feature_tool_metadata, &data.metadata)?;
    validate_dimensions(data, data.metadata.feature_count as usize)
}

/// The rules on `feature_options.image_to_gray` and the two readings that
/// depend on the version: required and valid from version 2, absent before.
/// [`validate_sift_data`] and [`crate::verify_sift`] apply them.
pub(crate) fn check_version_content(
    feature_tool_metadata: &FeatureToolMetadata,
    metadata: &SiftMetadata,
) -> Result<(), SiftError> {
    let invalid = |message: String| Err(SiftError::InvalidMetadata(message));
    let image_to_gray = feature_tool_metadata.feature_options.get("image_to_gray");
    if metadata.version < SIFT_READINGS_VERSION {
        if image_to_gray.is_some() {
            return invalid(format!(
                "version {} feature_options carry 'image_to_gray', which belongs to version {SIFT_READINGS_VERSION}",
                metadata.version
            ));
        }
        return Ok(());
    }
    let Some(image_to_gray) = image_to_gray else {
        return invalid(format!(
            "version {} feature_options must carry 'image_to_gray'",
            metadata.version
        ));
    };
    validate_image_to_gray(image_to_gray).map_err(SiftError::InvalidMetadata)?;
    if let Some(sharpness) = metadata.sharpness {
        if !sharpness.is_finite() || sharpness < 0.0 {
            return invalid(format!(
                "sharpness must be finite and not negative, got {sharpness}"
            ));
        }
    }
    if let Some([xx, yy, xy]) = metadata.gradient_structure_tensor_xx_yy_xy {
        if !(xx.is_finite() && yy.is_finite() && xy.is_finite()) || xx < 0.0 || yy < 0.0 {
            return invalid(format!(
                "gradient_structure_tensor_xx_yy_xy must be finite with Exx and Eyy not negative, got [{xx}, {yy}, {xy}]"
            ));
        }
    }
    Ok(())
}

/// Check the shape of an `image_to_gray` object, returning what is wrong: a
/// JSON object with a non-empty string `formula`, an optional integer
/// `quantize_bits` from 1 to [`MAX_QUANTIZE_BITS`], and no other key.
///
/// Whether the formula follows the grammar is checked by the code that parses
/// it. [`validate_sift_data`] applies this to a version 2 file's
/// `feature_options.image_to_gray`, and the `.sfmr` format crate to the
/// `image_to_gray` of an `.sfmr` file's image readings.
pub fn validate_image_to_gray(value: &serde_json::Value) -> Result<(), String> {
    let Some(map) = value.as_object() else {
        return Err(format!("image_to_gray must be a JSON object, got {value}"));
    };
    if let Some(key) = map
        .keys()
        .find(|k| k.as_str() != "formula" && k.as_str() != "quantize_bits")
    {
        return Err(format!(
            "image_to_gray has an unknown key {key:?} (only 'formula' and 'quantize_bits' are defined)"
        ));
    }
    match map.get("formula") {
        Some(serde_json::Value::String(formula)) if !formula.trim().is_empty() => {}
        Some(other) => {
            return Err(format!(
                "image_to_gray 'formula' must be a non-empty string, got {other}"
            ))
        }
        None => return Err("image_to_gray has no 'formula'".to_owned()),
    }
    if let Some(bits) = map.get("quantize_bits") {
        if !bits
            .as_u64()
            .is_some_and(|b| (1..=MAX_QUANTIZE_BITS).contains(&b))
        {
            return Err(format!(
                "image_to_gray 'quantize_bits' must be an integer from 1 to {MAX_QUANTIZE_BITS}, got {bits}"
            ));
        }
    }
    Ok(())
}

/// [`write_sift`] without the version check, so the tests can produce the
/// files a reader must reject.
pub(crate) fn write_sift_any_version(
    path: &Path,
    data: &SiftData,
    zstd_level: i32,
) -> Result<(), SiftError> {
    let feature_count = data.metadata.feature_count as usize;

    // Validate dimensions before opening anything, so a rejected write leaves
    // no file behind.
    validate_dimensions(data, feature_count)?;

    // Through a temporary file renamed over the target, so a failed write
    // leaves the previous file intact rather than a truncated one. See
    // `sfmtool_archive_io::write_atomically`.
    sfmtool_archive_io::write_atomically(path, |file| write_sift_into(file, data, zstd_level))
}

/// Stream the archive into `writer`. The whole of the write but the choosing of
/// where it lands.
fn write_sift_into<W: std::io::Write + std::io::Seek>(
    writer: W,
    data: &SiftData,
    zstd_level: i32,
) -> Result<(), SiftError> {
    let feature_count = data.metadata.feature_count as usize;
    let mut zip = ZipWriter::new(writer);

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
    let mut content_hash_digests = sfmtool_archive_io::SectionDigests::new();
    content_hash_digests.push_bytes(&feature_tool_bytes);
    content_hash_digests.push_bytes(&metadata_bytes);

    // Position data
    let pos_bytes: &[u8] = bytemuck::cast_slice(data.positions_xy.as_slice().unwrap());
    write_binary_entry(
        &mut zip,
        &positions_entry_name(feature_count),
        pos_bytes,
        zstd_level,
    )?;
    content_hash_digests.push_bytes(pos_bytes);

    // Affine shape data
    let shape_bytes: &[u8] = bytemuck::cast_slice(data.affine_shapes.as_slice().unwrap());
    write_binary_entry(
        &mut zip,
        &affine_shapes_entry_name(feature_count),
        shape_bytes,
        zstd_level,
    )?;
    content_hash_digests.push_bytes(shape_bytes);

    // Descriptor data
    let desc_bytes: &[u8] = data.descriptors.as_slice().unwrap();
    write_binary_entry(
        &mut zip,
        &descriptors_entry_name(feature_count),
        desc_bytes,
        zstd_level,
    )?;
    content_hash_digests.push_bytes(desc_bytes);

    // Thumbnail data
    let thumb_bytes: &[u8] = data.thumbnail_y_x_rgb.as_slice().unwrap();
    write_binary_entry(&mut zip, &thumbnail_entry_name(), thumb_bytes, zstd_level)?;
    content_hash_digests.push_bytes(thumb_bytes);

    // Compute and write content hash
    let content_xxh128 = content_hash_digests.finish();
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
    if data.descriptors.shape() != [feature_count, DESCRIPTOR_DIM] {
        return Err(SiftError::ShapeMismatch(format!(
            "descriptors shape {:?} != [{feature_count}, {DESCRIPTOR_DIM}]",
            data.descriptors.shape()
        )));
    }
    if data.thumbnail_y_x_rgb.shape() != [THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3] {
        return Err(SiftError::ShapeMismatch(format!(
            "thumbnail_y_x_rgb shape {:?} != [{THUMBNAIL_SIZE}, {THUMBNAIL_SIZE}, 3]",
            data.thumbnail_y_x_rgb.shape()
        )));
    }
    Ok(())
}
