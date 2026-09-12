// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Size accounting for a `.kdf`, for comparing descriptor layouts.
//!
//! The two version-1 layouts trade the same bytes against each other: tree-local
//! storage keeps one vector copy per tree, shared storage keeps one copy plus a
//! feature-ID-to-row map. Which is smaller is a question about a particular
//! corpus's compression in the order the file stored it, not one arithmetic
//! settles, so it has to be measured — and measuring it means splitting a file
//! into the parts that differ rather than comparing two totals.
//!
//! [`kdf_summary`] does that without decoding a single payload: the ZIP central
//! directory already carries every entry's compressed and uncompressed size, so
//! the whole accounting is one directory pass plus `metadata.json.zst`.

use std::collections::BTreeMap;
use std::path::Path;

use zip::ZipArchive;

use crate::types::{KdfError, Metadata, NODE_COLUMNS};

/// One bucket of entries, named by the role its entries play.
///
/// `decoded_bytes` is what the arrays occupy in memory, `compressed_bytes` the
/// zstd frame stored on disk, excluding ZIP headers. Their ratio is the
/// compression the corpus actually achieved in the order this file stored it,
/// which is the number the format spec's size projections could only estimate
/// from proxy orderings.
///
/// The decoded size comes from the shape in each entry's name, not from the ZIP
/// directory: entries are STORE-wrapped zstd frames, so the directory's
/// "uncompressed" size is the frame length and would make every ratio 100%.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KdfSection {
    /// Role of the entries in this bucket, e.g. `"tree_vectors"`.
    pub section: String,
    /// How many ZIP entries fell into it.
    pub entries: u64,
    /// Stored (zstd-compressed) bytes, excluding ZIP headers.
    pub compressed_bytes: u64,
    /// Uncompressed bytes.
    pub decoded_bytes: u64,
}

/// What a `.kdf` holds and what each part of it costs.
#[derive(Clone, Debug)]
pub struct KdfSummary {
    pub feature_count: u64,
    pub dimension: u32,
    pub scalar_type: String,
    pub tree_count: usize,
    /// `"tree_local"` or `"shared"`.
    pub descriptor_storage: String,
    /// Descriptor rows per shared block; `None` in tree-local layout.
    pub descriptor_block_rows: Option<u32>,
    pub target_chunk_bytes: u64,
    /// Chunks in each tree, in tree order.
    pub chunks_per_tree: Vec<usize>,
    /// Nodes in each tree, in tree order.
    pub nodes_per_tree: Vec<u64>,
    /// Whether the file carries an image table and per-feature origins.
    pub has_sources: bool,
    /// Size of the file itself, ZIP headers and central directory included.
    pub file_bytes: u64,
    /// Bytes in every entry's payload, which excludes those headers. The
    /// difference against `file_bytes` is the container's own overhead — a real
    /// cost at high entry counts, and one no per-section total can show.
    pub payload_compressed_bytes: u64,
    pub payload_decoded_bytes: u64,
    /// Per-role buckets, in a stable order by section name.
    pub sections: Vec<KdfSection>,
}

/// Decoded size of an array entry, from the shape its name encodes.
///
/// **The ZIP directory cannot answer this.** Entries are STORE-compressed
/// wrappers around an independent zstd frame, so both sizes the directory
/// carries are the *frame* length; taking `uncompressed_size` for the decoded
/// size reports a compression ratio of exactly 100% for every entry in every
/// file, which is wrong in a way that looks entirely plausible.
///
/// The names carry the shape instead, which is what they are for. Every array
/// entry ends `{...}.{counts}.{dtype}.zst`, so the decoded length is the product
/// of the decimal counts and the scalar width. Returns `None` for `.json.zst`
/// entries, whose size their name does not describe.
fn scalar_width(name: &str) -> Option<u64> {
    match name {
        "uint8" => Some(1),
        "float32" => Some(4),
        "uint32" => Some(4),
        "uint64" => Some(8),
        "uint128" => Some(16),
        _ => None,
    }
}

fn decoded_bytes_from_name(name: &str) -> Option<u64> {
    let file = name.rsplit('/').next()?;
    let mut parts: Vec<&str> = file.split('.').collect();
    let suffix = parts.pop()?;

    // `chunk.{M}.{P}[.{D}].{scalar}.zst` is the one heterogeneous entry: ten
    // uint32 node columns, then one split per node, then one uint32 per feature.
    // The generic product-of-counts rule below cannot express a sum of arrays.
    if parts.first() == Some(&"chunk") && suffix == "zst" {
        let width = scalar_width(parts.pop()?)?;
        let counts: Vec<u64> = parts[1..]
            .iter()
            .map(|t| t.parse().ok())
            .collect::<Option<_>>()?;
        let [nodes, features] = counts.as_slice() else {
            return None;
        };
        let columns = NODE_COLUMNS as u64;
        return columns
            .checked_mul(*nodes)?
            .checked_mul(4)?
            .checked_add(nodes.checked_mul(width)?)?
            .checked_add(features.checked_mul(4)?);
    }

    // `corpus.{N}.{D}.{scalar}.frames` holds one zstd frame per descriptor
    // block, so it must be sized from the name rather than decoded: decoding it
    // would expand the entire corpus to learn a number the name already gives.
    if parts.first() == Some(&"corpus") && suffix == "frames" {
        let width = scalar_width(parts.pop()?)?;
        let counts: Vec<u64> = parts[1..]
            .iter()
            .map(|t| t.parse().ok())
            .collect::<Option<_>>()?;
        let [features, dimension] = counts.as_slice() else {
            return None;
        };
        return features.checked_mul(*dimension)?.checked_mul(width);
    }

    if suffix != "zst" {
        return None;
    }
    let width = scalar_width(parts.pop()?)?;
    // Trailing decimal tokens are the shape; the leading token is the stem.
    let mut elements: u64 = 1;
    let mut saw_count = false;
    for token in parts.iter().skip(1) {
        let count: u64 = token.parse().ok()?;
        elements = elements.checked_mul(count)?;
        saw_count = true;
    }
    if !saw_count {
        return None;
    }
    elements.checked_mul(width)
}

/// Classify an entry by the role its name encodes.
///
/// Names are structural in this format (`trees/{t}/chunks/{c}/...`), so the role
/// is a prefix-and-suffix question that needs no metadata lookup.
fn section_of(name: &str) -> &'static str {
    if name == "metadata.json.zst" {
        "metadata"
    } else if name == "content_hash.json.zst" {
        "content_hash"
    } else if name.starts_with("images/") {
        "images"
    } else if name.starts_with("origins/") {
        "origins"
    } else if name.starts_with("features/storage_rows.") {
        "shared_row_map"
    } else if name.starts_with("features/block_offsets.") {
        "shared_block_offsets"
    } else if name.starts_with("features/corpus.") {
        "shared_vectors"
    } else if name.starts_with("trees/") {
        // A chunk's three integer arrays share one entry; its vectors do not, so
        // `tree_vectors` is still exactly what the shared layout removes T-1
        // copies of, and `tree_chunks` is identical between the two layouts.
        match name.rsplit('/').next().unwrap_or("") {
            n if n.starts_with("vectors.") => "tree_vectors",
            _ => "tree_chunks",
        }
    } else {
        "other"
    }
}

/// Account for a `.kdf`'s size without decoding its payloads.
///
/// Reads the ZIP central directory and `metadata.json.zst` only, so the cost is
/// independent of corpus size: a 5 GB file summarizes as fast as a 5 KB one.
///
/// This is deliberately not generic over the scalar type. The accounting never
/// touches a vector, and making the caller name the scalar type up front would
/// defeat one of the uses — finding out what an unfamiliar file contains.
///
/// `max_metadata_bytes` bounds the one payload it does decode, matching the
/// limit [`crate::LazyKdForestOptions`] applies at open.
///
/// # Example
///
/// ```no_run
/// # fn main() -> Result<(), sfmtool_kdf_format::KdfError> {
/// let summary = sfmtool_kdf_format::kdf_summary("corpus.kdf".as_ref(), 64 << 20)?;
/// for section in &summary.sections {
///     println!("{:>18}  {:>12}", section.section, section.compressed_bytes);
/// }
/// # Ok(())
/// # }
/// ```
pub fn kdf_summary(path: &Path, max_metadata_bytes: usize) -> Result<KdfSummary, KdfError> {
    let file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let mut archive = ZipArchive::new(file)?;

    let mut buckets: BTreeMap<&'static str, KdfSection> = BTreeMap::new();
    let mut payload_compressed = 0u64;
    let mut payload_decoded = 0u64;
    let mut metadata_index = None;
    let mut json_entries: Vec<(usize, String)> = Vec::new();
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        if entry.is_dir() {
            return Err(KdfError::InvalidFormat(format!(
                "directory ZIP entry is forbidden: {}",
                entry.name()
            )));
        }
        let name = entry.name().to_string();
        if name == "metadata.json.zst" {
            metadata_index = Some(i);
        }
        // `entry.size()` is the stored zstd frame, not the decoded array; see
        // `decoded_bytes_from_name`. JSON entries carry no shape in their name,
        // so they are decoded after this pass — there are at most four of them
        // and they are small.
        let compressed = entry.compressed_size();
        let decoded = match decoded_bytes_from_name(&name) {
            Some(bytes) => bytes,
            None => {
                json_entries.push((i, name.clone()));
                0
            }
        };
        payload_compressed += compressed;
        payload_decoded += decoded;
        let key = section_of(&name);
        let bucket = buckets.entry(key).or_insert_with(|| KdfSection {
            section: key.to_string(),
            entries: 0,
            compressed_bytes: 0,
            decoded_bytes: 0,
        });
        bucket.entries += 1;
        bucket.compressed_bytes += compressed;
        bucket.decoded_bytes += decoded;
    }

    // Decode the JSON entries for their true size, so a per-section ratio means
    // the same thing everywhere in the table.
    for (i, name) in &json_entries {
        let mut entry = archive.by_index(*i)?;
        if entry.size() > max_metadata_bytes as u64 {
            return Err(KdfError::ResourceLimit(format!(
                "{name} is over the {max_metadata_bytes}-byte limit"
            )));
        }
        let mut frame = Vec::new();
        std::io::Read::read_to_end(&mut entry, &mut frame)?;
        let decoded = zstd::bulk::decompress(&frame, max_metadata_bytes)?.len() as u64;
        payload_decoded += decoded;
        if let Some(bucket) = buckets.get_mut(section_of(name)) {
            bucket.decoded_bytes += decoded;
        }
    }

    let index = metadata_index
        .ok_or_else(|| KdfError::InvalidFormat("metadata.json.zst is missing".into()))?;
    let metadata: Metadata = {
        let mut entry = archive.by_index(index)?;
        if entry.size() > max_metadata_bytes as u64 {
            return Err(KdfError::ResourceLimit(format!(
                "metadata.json.zst decodes to {} bytes, over the {max_metadata_bytes}-byte limit",
                entry.size()
            )));
        }
        let mut frame = Vec::new();
        std::io::Read::read_to_end(&mut entry, &mut frame)?;
        serde_json::from_slice(&zstd::bulk::decompress(&frame, max_metadata_bytes)?)?
    };

    Ok(KdfSummary {
        feature_count: metadata.feature_count as u64,
        dimension: metadata.dimension as u32,
        scalar_type: metadata.scalar_type.clone(),
        tree_count: metadata.trees.len(),
        descriptor_storage: metadata.descriptor_storage.clone(),
        descriptor_block_rows: metadata.descriptor_block_rows,
        target_chunk_bytes: metadata.target_chunk_bytes,
        chunks_per_tree: metadata.trees.iter().map(|t| t.chunks.len()).collect(),
        nodes_per_tree: metadata
            .trees
            .iter()
            .map(|t| t.chunks.iter().map(|c| c.node_count as u64).sum())
            .collect(),
        has_sources: metadata.workspace.is_some(),
        file_bytes,
        payload_compressed_bytes: payload_compressed,
        payload_decoded_bytes: payload_decoded,
        sections: buckets.into_values().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_shapes_are_sized_without_decoding() {
        assert_eq!(
            decoded_bytes_from_name("features/block_offsets.303001.uint64.zst"),
            Some(303001 * 8)
        );
        assert_eq!(
            decoded_bytes_from_name("trees/0/chunks/0/chunk.18446744073709551615.1.uint8.zst"),
            None
        );
    }

    #[test]
    fn metadata_expansion_is_bounded() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.kdf");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
        zip.start_file(
            "metadata.json.zst",
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored),
        )
        .unwrap();
        zip.write_all(&zstd::bulk::compress(&vec![b' '; 1 << 20], 1).unwrap())
            .unwrap();
        zip.finish().unwrap();
        assert!(kdf_summary(&path, 1024).is_err());
    }
}
