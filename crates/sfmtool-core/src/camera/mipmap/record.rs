// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The record each mip level file carries in an APP15 segment, and the facts
//! about an original that a record is checked against.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use super::location::MipmapLocation;
use super::MipmapError;

/// The bytes an APP15 segment's body starts with when it holds a record,
/// before the JSON.
pub(super) const RECORD_SIGNATURE: &[u8] = b"sfmtool-mipmap\0";

/// The record version this build writes and reads.
pub const MIP_LEVEL_RECORD_VERSION: u32 = 1;

/// The encoder named in every record. Diagnostic only: no reader compares it.
pub(super) const ENCODER: &str = "jpeg-encoder 0.7.1";

/// What a mip level file says about itself and the original it was made
/// from: the APP15 segment `sfmtool-mipmap\0` followed by this as JSON.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MipLevelRecord {
    /// [`MIP_LEVEL_RECORD_VERSION`] when written by this build.
    pub version: u32,
    /// The original's path relative to its workspace root, with `/`
    /// separators.
    pub source: String,
    /// The original's length in bytes.
    pub source_size: u64,
    /// The original's modification time in nanoseconds since the Unix
    /// epoch, or `null` where the file system gives none.
    pub source_modified_ns: Option<i64>,
    /// The XXH128 of the original's bytes, as 32 lowercase hex digits.
    pub source_xxh128: String,
    /// The original's size, from its header.
    pub source_width: u32,
    pub source_height: u32,
    /// The pyramid level the file holds: 0 for `-mip0`.
    pub level: u32,
    /// The workspace's `out_of_core_long_side` when the file was made, for
    /// diagnosis; no reader compares it.
    pub out_of_core_long_side: u32,
    /// The region test's answer for the original.
    pub region_readable_original: bool,
    /// The level's size, the file's frame.
    pub width: u32,
    pub height: u32,
    /// `K`, the coarsest level the mipmap stores.
    pub stored_levels: u32,
    /// The JPEG quality the file is encoded at, the number in its name.
    pub jpeg_quality: u8,
    /// 1 for grey, 3 for colour.
    pub components: u8,
    /// The restart interval in MCUs.
    pub restart_interval_mcus: u16,
    /// The encoder and its version.
    pub encoder: String,
    /// The sfmtool version that wrote the file.
    pub sfmtool_version: String,
}

impl MipLevelRecord {
    /// The APP15 segment body: the signature, then the JSON.
    pub(super) fn to_segment(&self) -> Vec<u8> {
        let mut body = RECORD_SIGNATURE.to_vec();
        body.extend(serde_json::to_vec(self).expect("a record always serializes"));
        body
    }

    /// The record in an APP15 segment body, or why it holds none.
    pub(super) fn from_segment(body: &[u8]) -> Result<Self, String> {
        let json = body
            .strip_prefix(RECORD_SIGNATURE)
            .ok_or_else(|| "its APP15 segment is not an sfmtool-mipmap record".to_string())?;
        let value: serde_json::Value =
            serde_json::from_slice(json).map_err(|e| format!("its record does not parse: {e}"))?;
        match value.get("version").and_then(|v| v.as_u64()) {
            Some(v) if v == u64::from(MIP_LEVEL_RECORD_VERSION) => {}
            Some(v) => return Err(format!("its record is of unknown version {v}")),
            None => return Err("its record has no version".to_string()),
        }
        serde_json::from_value(value).map_err(|e| format!("its record does not parse: {e}"))
    }
}

/// An original as the records of its mipmap describe it: its relative path,
/// header size, length and modification time, and its hash, computed at most
/// once for the life of this value.
///
/// A caller keeps one per original and spelling of its path, so checking
/// every file of a mipmap, or the same mipmap again, hashes the original at
/// most once.
#[derive(Debug)]
pub struct MipmapSource {
    path: PathBuf,
    relative: String,
    width: u32,
    height: u32,
    size: u64,
    modified_ns: Option<i64>,
    hash: OnceLock<Result<String, String>>,
    hashes: AtomicUsize,
}

/// A file's modification time in nanoseconds since the Unix epoch.
pub(super) fn modified_ns(meta: &std::fs::Metadata) -> Option<i64> {
    let modified = meta.modified().ok()?;
    match modified.duration_since(UNIX_EPOCH) {
        Ok(after) => i64::try_from(after.as_nanos()).ok(),
        Err(before) => i64::try_from(before.duration().as_nanos()).ok().map(|n| -n),
    }
}

impl MipmapSource {
    /// Read the facts about `photograph`, whose mipmap is at `location`: its
    /// header size, length and modification time. The hash waits until a
    /// check needs it.
    pub fn read(photograph: &Path, location: &MipmapLocation) -> Result<Self, MipmapError> {
        let meta = std::fs::metadata(photograph).map_err(|error| MipmapError::Io {
            path: photograph.to_path_buf(),
            error,
        })?;
        let (width, height) =
            crate::camera::image::image_dimensions(photograph).map_err(|error| {
                MipmapError::Source {
                    path: photograph.to_path_buf(),
                    error,
                }
            })?;
        Ok(Self {
            path: photograph.to_path_buf(),
            relative: location.relative().to_string(),
            width,
            height,
            size: meta.len(),
            modified_ns: modified_ns(&meta),
            hash: OnceLock::new(),
            hashes: AtomicUsize::new(0),
        })
    }

    /// The facts of an original whose bytes the caller has read, with the
    /// hash computed from them, so a build reads the original once.
    pub(super) fn from_bytes(
        photograph: &Path,
        relative: &str,
        meta: &std::fs::Metadata,
        bytes: &[u8],
        (width, height): (u32, u32),
    ) -> Self {
        let hash = OnceLock::new();
        let _ = hash.set(Ok(xxh128_hex(bytes)));
        Self {
            path: photograph.to_path_buf(),
            relative: relative.to_string(),
            width,
            height,
            size: bytes.len() as u64,
            modified_ns: modified_ns(meta),
            hash,
            hashes: AtomicUsize::new(0),
        }
    }

    /// The original's path relative to its workspace root.
    pub fn relative(&self) -> &str {
        &self.relative
    }

    /// The original's size from its header.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// How many times this value has read the original to hash it: 0 or 1.
    pub fn hashes(&self) -> usize {
        self.hashes.load(Ordering::Relaxed)
    }

    fn hash(&self) -> Result<&str, String> {
        self.hash
            .get_or_init(|| {
                self.hashes.fetch_add(1, Ordering::Relaxed);
                crate::features::sift::xxh128_of_file(&self.path)
                    .map(|(hash, _)| hash)
                    .map_err(|e| format!("the original could not be hashed: {e}"))
            })
            .as_deref()
            .map_err(Clone::clone)
    }

    /// The record's facts written for these: tests 2 to 4 of when a mip
    /// level file is current, so far as they concern the original. `Err`
    /// says why the file is stale.
    pub(crate) fn check(&self, record: &MipLevelRecord) -> Result<(), String> {
        // Test 2: the same path, or one that differs only in letter case,
        // which the hash then decides.
        let case_only = if record.source == self.relative {
            false
        } else if record.source.to_lowercase() == self.relative.to_lowercase() {
            true
        } else {
            return Err(format!(
                "it was made from {}, not {}",
                record.source, self.relative
            ));
        };
        // Test 3.
        if (record.source_width, record.source_height) != (self.width, self.height) {
            return Err(format!(
                "it was made from a {} x {} original, and this one is {} x {}",
                record.source_width, record.source_height, self.width, self.height
            ));
        }
        // Test 4.
        if record.source_size != self.size {
            return Err(format!(
                "it was made from an original of {} bytes, and this one has {}",
                record.source_size, self.size
            ));
        }
        let same_time =
            record.source_modified_ns.is_some() && record.source_modified_ns == self.modified_ns;
        if same_time && !case_only {
            return Ok(());
        }
        if self.hash()? == record.source_xxh128 {
            Ok(())
        } else {
            Err(format!(
                "it was made from an original whose bytes differ from {}",
                self.relative
            ))
        }
    }

    /// The record fields that describe the original.
    pub(super) fn record_fields(&self) -> Result<(String, u64, Option<i64>, String), String> {
        Ok((
            self.relative.clone(),
            self.size,
            self.modified_ns,
            self.hash()?.to_string(),
        ))
    }
}

/// The XXH128 of `bytes` as 32 lowercase hex digits, as `.sift` files record
/// an image's.
fn xxh128_hex(bytes: &[u8]) -> String {
    format!("{:032x}", xxhash_rust::xxh3::xxh3_128(bytes))
}
