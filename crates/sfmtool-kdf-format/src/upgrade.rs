// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Rewrite an older `.kdf` version as the current one: [`upgrade_kdf`].

use std::collections::HashMap;
use std::path::Path;

use sfmtool_archive_io::{format_hash, write_json_entry};
use sfmtool_progress::Progress;
use xxhash_rust::xxh3::xxh3_128;
use zip::{ZipArchive, ZipWriter};

use crate::read::{compose_content_hash, parse_metadata, read_bounded_json_raw};
use crate::types::*;
use crate::{read_kdf_version, KdfOpenOptions};

/// The zstd level the two rewritten JSON entries are written at, the
/// writer's default.
const ZSTD_LEVEL: i32 = 3;

/// Rewrite the `.kdf` file at `path` as the current version when it is of an
/// older version this build reads, and leave it as it is when it is current.
///
/// Version 3 differs from version 4 only in `metadata.json`, which gains the
/// `chunk_layout` legend, so only `metadata.json.zst` and
/// `content_hash.json.zst` are rewritten. Every other entry's stored bytes
/// are copied unchanged, and with them every section digest but
/// `metadata_xxh128`, which is computed over the new metadata, and
/// `content_xxh128`, which is folded again from the section digests.
///
/// The file is first checked as [`KdfFile::open`](crate::KdfFile::open)
/// checks it, by [`read_kdf_version`]. It is then written to a temporary
/// sibling and renamed over `path` once it is whole, so a rewrite that fails
/// or is cancelled leaves the file as it was. Copying a large index takes
/// seconds, so the rewrite counts the entries it copies in `progress` and
/// stops between entries when asked to.
///
/// ```no_run
/// use sfmtool_progress::Progress;
///
/// sfmtool_kdf_format::upgrade_kdf("kerry-sift-index.kdf".as_ref(), &Progress::none())?;
/// # Ok::<(), sfmtool_kdf_format::KdfError>(())
/// ```
///
/// # Errors
///
/// [`KdfError`] for a file [`read_kdf_version`] refuses or one that cannot
/// be written, and [`KdfError::Cancelled`] when `progress` asks it to stop.
pub fn upgrade_kdf(path: &Path, progress: &Progress<'_>) -> Result<(), KdfError> {
    if read_kdf_version(path)? == KDF_FORMAT_VERSION {
        return Ok(());
    }
    sfmtool_archive_io::write_atomically(path, |out| {
        // Opened inside the write, so the handle is closed before the
        // temporary file is renamed over `path`.
        let mut source = ZipArchive::new(std::fs::File::open(path)?)?;
        let mut entries = HashMap::new();
        for i in 0..source.len() {
            let entry = source.by_index_raw(i)?;
            entries.insert(entry.name()?.into_owned(), entry.compressed_size());
        }
        let limit = KdfOpenOptions::default().max_metadata_bytes;
        let raw = read_bounded_json_raw(&mut source, &entries, "metadata.json.zst", limit)?;
        let mut metadata = parse_metadata(&raw)?;
        metadata.version = KDF_FORMAT_VERSION;
        let raw = read_bounded_json_raw(&mut source, &entries, "content_hash.json.zst", limit)?;
        let mut hashes: ContentHash = serde_json::from_slice(&raw)?;

        let mut zip = ZipWriter::new(out);
        let metadata_raw = write_json_entry(&mut zip, "metadata.json.zst", &metadata, ZSTD_LEVEL)?;
        hashes.metadata_xxh128 = format_hash(xxh3_128(&metadata_raw));
        hashes.content_xxh128 = format_hash(compose_content_hash(&hashes)?);
        let total = source.len() as u64;
        for i in 0..source.len() {
            progress.check_cancel()?;
            let entry = source.by_index_raw(i)?;
            let name = entry.name()?;
            if name != "metadata.json.zst" && name != "content_hash.json.zst" {
                zip.raw_copy_file(entry)?;
            }
            progress.count(i as u64 + 1, Some(total), "entry");
        }
        write_json_entry(&mut zip, "content_hash.json.zst", &hashes, ZSTD_LEVEL)?;
        zip.finish()?;
        Ok(())
    })
}
