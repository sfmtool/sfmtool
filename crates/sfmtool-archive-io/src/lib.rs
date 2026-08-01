// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared ZIP + zstd container I/O for sfmtool's archive-based file formats.
//!
//! `.sift`, `.sfmr`, `.matches` and `.camrig` all use the same container
//! structure: a ZIP archive (STORE method) whose entries are zstandard-
//! compressed JSON metadata and binary columnar arrays, with per-section
//! XXH128 content hashes computed over the *uncompressed* bytes.
//!
//! Each format crate owns its own schema, validation rules and error type;
//! this crate owns only the container primitives they share.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "sfmtool-archive-io requires a little-endian target (binary arrays are stored as little-endian)"
);

use std::borrow::Cow;
use std::io::{Read, Seek, Write};

use xxhash_rust::xxh3::Xxh3;
use zip::write::SimpleFileOptions;
use zip::{ZipArchive, ZipWriter};

/// Errors that can occur during archive I/O operations.
///
/// Each format crate converts this into its own public error type, so callers
/// of `read_sfmr` / `write_matches` / … never see it directly.
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
    Ok(cast_or_copy(&bytes, expected_len))
}

/// Reinterpret `bytes` as `expected_len` values of `T`, copying once.
///
/// Fast path: when the buffer is already aligned for `T` (the common case,
/// since the allocator over-aligns sizeable allocations), borrow it and copy
/// once via `to_vec` — exactly what the original code did. Only when the buffer
/// lands on an address `cast_slice` would reject (which used to panic) do we
/// route through a freshly aligned `Vec<T>`.
///
/// Split out from [`read_binary_array`] so the misaligned branch is reachable
/// from a test: the alignment of a decompressed `Vec<u8>` is the allocator's
/// choice, so driving that branch through the archive path is not possible.
///
/// `bytes.len()` must equal `expected_len * size_of::<T>()`; callers check that
/// first so they can attach the entry name to the error.
fn cast_or_copy<T: bytemuck::Pod>(bytes: &[u8], expected_len: usize) -> Vec<T> {
    debug_assert_eq!(bytes.len(), expected_len * std::mem::size_of::<T>());
    if bytes.is_empty() {
        return Vec::new();
    }
    match bytemuck::try_cast_slice::<u8, T>(bytes) {
        Ok(slice) => slice.to_vec(),
        Err(_) => {
            let mut out: Vec<T> = vec![T::zeroed(); expected_len];
            bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(bytes);
            out
        }
    }
}

/// Read uint128 hashes (16 bytes each) from a zstandard-compressed entry.
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

// ── Raw buffer reinterpretation ─────────────────────────────────────────

/// Reinterpret a freshly decompressed byte buffer as `u32` values.
///
/// [`read_zst_entry`] returns a `Vec<u8>` whose start address is only
/// guaranteed to be 1-aligned, so `bytemuck::cast_slice::<u8, u32>` panics when
/// the buffer is not 4-aligned. Borrow the buffer directly when it is already
/// aligned (the common case, no copy), and fall back to copying through a
/// freshly aligned `Vec<u32>` only when it is not. Any trailing bytes that do
/// not form a whole `u32` (truncated/corrupt entry) are dropped; structural
/// checks downstream then catch the mismatch.
///
/// Used by the verifiers, which validate straight from the raw bytes they
/// hashed rather than re-reading through [`read_binary_array`].
pub fn raw_to_u32(raw: &[u8]) -> Cow<'_, [u32]> {
    let size = std::mem::size_of::<u32>();
    let n = raw.len() / size;
    let trimmed = &raw[..n * size];
    match bytemuck::try_cast_slice::<u8, u32>(trimmed) {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => {
            let mut out = vec![0u32; n];
            bytemuck::cast_slice_mut::<u32, u8>(&mut out).copy_from_slice(trimmed);
            Cow::Owned(out)
        }
    }
}

/// [`raw_to_u32`] for `f64` entries (8-byte alignment fallback included).
pub fn raw_to_f64(raw: &[u8]) -> Cow<'_, [f64]> {
    let size = std::mem::size_of::<f64>();
    let n = raw.len() / size;
    let trimmed = &raw[..n * size];
    match bytemuck::try_cast_slice::<u8, f64>(trimmed) {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => {
            let mut out = vec![0f64; n];
            bytemuck::cast_slice_mut::<f64, u8>(&mut out).copy_from_slice(trimmed);
            Cow::Owned(out)
        }
    }
}

// ── Writing ─────────────────────────────────────────────────────────────

/// Compress bytes with zstandard at the given level.
pub fn zstd_compress(data: &[u8], level: i32) -> Result<Vec<u8>, ArchiveIoError> {
    zstd::bulk::compress(data, level)
        .map_err(|e| ArchiveIoError::InvalidFormat(format!("zstd compression failed: {e}")))
}

/// Write a zstandard-compressed JSON entry to a ZIP archive.
///
/// Returns the uncompressed JSON bytes (for hashing). Unlike the binary
/// entries, these bytes are freshly serialized here and the caller has no
/// other handle on them, so returning them costs nothing.
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
/// The caller already owns `data` and can hash it directly, so nothing is
/// returned — this avoids cloning large binary buffers (whole uncompressed
/// columns: positions, tracks, thumbnails, patch bitmaps).
///
/// Callers folding the entry into a rolling section hash should prefer
/// [`write_binary_entry_hashed`].
pub fn write_binary_entry<W: Write + Seek>(
    zip: &mut ZipWriter<W>,
    name: &str,
    data: &[u8],
    zstd_level: i32,
) -> Result<(), ArchiveIoError> {
    let compressed = zstd_compress(data, zstd_level)?;
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file(name, options)?;
    zip.write_all(&compressed)?;
    Ok(())
}

/// [`write_binary_entry`], folding the uncompressed bytes into `hasher`.
///
/// Section hashes are computed over the uncompressed bytes of each entry in
/// the order they are written, so writing and hashing belong together: keeping
/// them in one call makes it impossible to write an entry and forget to hash
/// it, or to hash in an order that does not match the write order.
pub fn write_binary_entry_hashed<W: Write + Seek>(
    zip: &mut ZipWriter<W>,
    name: &str,
    data: &[u8],
    zstd_level: i32,
    hasher: &mut Xxh3,
) -> Result<(), ArchiveIoError> {
    write_binary_entry(zip, name, data, zstd_level)?;
    hasher.update(data);
    Ok(())
}

// ── Hashing ─────────────────────────────────────────────────────────────

/// Format an XXH128 digest as a 32-character lowercase hex string.
pub fn format_hash(digest: u128) -> String {
    format!("{:032x}", digest)
}

#[cfg(test)]
mod tests;
