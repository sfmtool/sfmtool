// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared ZIP + zstd I/O utilities for `.sift` and `.sfmr` file formats.
//!
//! Both formats use the same container structure: a ZIP archive with
//! zstandard-compressed entries containing JSON metadata and binary arrays.

use std::io::{Read, Seek, Write};

use zip::write::SimpleFileOptions;
use zip::{ZipArchive, ZipWriter};

/// Errors that can occur during archive I/O operations.
#[derive(thiserror::Error, Debug)]
pub enum ArchiveIoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
}

// ── Reading ─────────────────────────────────────────────────────────────

/// Decompress a zstandard-compressed entry from a ZIP archive,
/// returning the raw decompressed bytes.
pub fn read_zst_entry<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<Vec<u8>, ArchiveIoError> {
    let mut entry = archive.by_name(name)?;
    let mut compressed = Vec::new();
    entry.read_to_end(&mut compressed)?;
    let mut decompressed = Vec::new();
    zstd::stream::copy_decode(&compressed[..], &mut decompressed).map_err(|e| {
        ArchiveIoError::InvalidFormat(format!("zstd decompression failed for {name}: {e}"))
    })?;
    Ok(decompressed)
}

/// Read and parse a zstandard-compressed JSON entry from a ZIP archive.
pub fn read_json_entry<R: Read + Seek, T: serde::de::DeserializeOwned>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<T, ArchiveIoError> {
    let bytes = read_zst_entry(archive, name)?;
    serde_json::from_slice(&bytes).map_err(|e| e.into())
}

/// Read a binary array from a zstandard-compressed ZIP entry.
///
/// Validates that the decompressed size matches `expected_len * size_of::<T>()`.
pub fn read_binary_array<R: Read + Seek, T: bytemuck::Pod>(
    archive: &mut ZipArchive<R>,
    name: &str,
    expected_len: usize,
) -> Result<Vec<T>, ArchiveIoError> {
    let bytes = read_zst_entry(archive, name)?;
    let expected_bytes = expected_len * std::mem::size_of::<T>();
    if bytes.len() != expected_bytes {
        return Err(ArchiveIoError::ShapeMismatch(format!(
            "{name}: expected {expected_bytes} bytes ({expected_len} elements), got {} bytes",
            bytes.len()
        )));
    }
    Ok(bytemuck::cast_slice::<u8, T>(&bytes).to_vec())
}

/// Read uint128 hashes (16 bytes each) from a zstandard-compressed entry.
#[allow(dead_code)]
pub fn read_uint128_array<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
    count: usize,
) -> Result<Vec<[u8; 16]>, ArchiveIoError> {
    let bytes = read_zst_entry(archive, name)?;
    let expected = count * 16;
    if bytes.len() != expected {
        return Err(ArchiveIoError::ShapeMismatch(format!(
            "{name}: expected {expected} bytes ({count} hashes), got {} bytes",
            bytes.len()
        )));
    }
    let mut hashes = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(16) {
        let mut hash = [0u8; 16];
        hash.copy_from_slice(chunk);
        hashes.push(hash);
    }
    Ok(hashes)
}

// ── Writing ─────────────────────────────────────────────────────────────

/// Compress bytes with zstandard at the given level.
pub fn zstd_compress(data: &[u8], level: i32) -> Result<Vec<u8>, ArchiveIoError> {
    zstd::bulk::compress(data, level)
        .map_err(|e| ArchiveIoError::InvalidFormat(format!("zstd compression failed: {e}")))
}

/// Write a zstandard-compressed JSON entry to a ZIP archive.
///
/// Returns the uncompressed JSON bytes (for hashing).
pub fn write_json_entry<W: Write + Seek>(
    zip: &mut ZipWriter<W>,
    name: &str,
    value: &impl serde::Serialize,
    zstd_level: i32,
) -> Result<Vec<u8>, ArchiveIoError> {
    let json_bytes = serde_json::to_vec(value)?;
    let compressed = zstd_compress(&json_bytes, zstd_level)?;
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file(name, options)?;
    zip.write_all(&compressed)?;
    Ok(json_bytes)
}

/// Write a zstandard-compressed binary entry to a ZIP archive.
///
/// Returns the uncompressed data (for hashing).
pub fn write_binary_entry<W: Write + Seek>(
    zip: &mut ZipWriter<W>,
    name: &str,
    data: &[u8],
    zstd_level: i32,
) -> Result<Vec<u8>, ArchiveIoError> {
    let compressed = zstd_compress(data, zstd_level)?;
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file(name, options)?;
    zip.write_all(&compressed)?;
    Ok(data.to_vec())
}

// ── Hashing ─────────────────────────────────────────────────────────────

/// Format an XXH128 digest as a 32-character lowercase hex string.
pub fn format_hash(digest: u128) -> String {
    format!("{:032x}", digest)
}