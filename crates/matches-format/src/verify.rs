// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.matches` file integrity verification.
//!
//! [`verify_matches`] is an orchestrator. It settles the questions that decide
//! whether the rest of the file is interpretable at all — the format version,
//! then [`structure_errors`] — and hands each present section to its own
//! `verify_*_section` function, which rehashes that section's entries,
//! compares the digest against the stored record, and appends whatever
//! findings the section's raw arrays support.
//!
//! Two orderings are load-bearing, and they are why the split is by section
//! rather than by kind of check. Within a section, entries feed one hasher in
//! lexicographic path order, matching the order [`write`](crate::write) hashed
//! them in; across sections, the digests accumulate in the canonical order
//! (metadata, images, backbone, cluster patches, two-view geometries) and are
//! hashed together into the overall content hash. So a section function owns
//! one hasher from open to digest, and the orchestrator owns the sequence —
//! neither can be reordered without changing what a valid file hashes to.
//!
//! Errors accumulate in a single vector rather than short-circuiting, so a
//! damaged file reports everything that is wrong with it at once. The call
//! order above is therefore also the reported order.

use std::io::Seek;
use std::path::Path;

use xxhash_rust::xxh3::Xxh3;
use zip::ZipArchive;

use crate::entries;
use crate::types::*;
use sfmtool_archive_io::{format_hash, raw_to_f32, raw_to_u32, read_zst_entry};

/// Check the backbone rule and metadata flag / summary-count / zip-entry
/// consistency. Returns the errors found; when non-empty the caller reports
/// them and stops (section hashing assumes a structurally coherent file).
fn structure_errors(metadata: &MatchesMetadata, entry_names: &[String]) -> Vec<String> {
    let mut errors = Vec::new();
    let has_prefix = |prefix: &str| entry_names.iter().any(|n| n.starts_with(prefix));
    let has_entry = |name: &str| entry_names.iter().any(|n| n == name);

    if metadata.version < 3 && (metadata.has_clusters || metadata.has_cluster_patches) {
        errors.push(format!(
            "version {} file claims clusters/cluster_patches (introduced in version 3)",
            metadata.version
        ));
    }
    if metadata.has_cluster_patches && !metadata.has_clusters {
        errors.push("has_cluster_patches requires has_clusters (cluster_patches requires the clusters section)".into());
    }
    if metadata.has_two_view_geometries && metadata.has_clusters {
        errors.push(
            "two_view_geometries requires the image_pairs backbone, but this file stores clusters"
                .into(),
        );
    }

    // Per-image dimensions are mandatory since version 4 and never stored
    // before it.
    let has_dims = has_prefix(entries::images_image_dims_prefix());
    if metadata.version >= 4 && !has_dims {
        errors.push(
            "version 4+ file is missing images/image_dims (mandatory since version 4)".into(),
        );
    }
    if metadata.version < 4 && has_dims {
        errors.push(format!(
            "version {} file contains images/image_dims (introduced in version 4)",
            metadata.version
        ));
    }

    // The cluster backbone's member detections are mandatory from version 6
    // and never stored before it.
    for (prefix, name) in [
        (
            entries::clusters_member_positions_prefix(),
            "member_positions",
        ),
        (
            entries::clusters_member_affine_shapes_prefix(),
            "member_affine_shapes",
        ),
    ] {
        let present = has_prefix(prefix);
        if metadata.has_clusters && metadata.version >= 6 && !present {
            errors.push(format!(
                "version 6+ cluster file is missing clusters/{name} (mandatory since version 6)"
            ));
        }
        if metadata.version < 6 && present {
            errors.push(format!(
                "version {} file contains clusters/{name} (introduced in version 6)",
                metadata.version
            ));
        }
    }

    if metadata.has_clusters {
        if metadata.cluster_count.is_none() || metadata.cluster_member_count.is_none() {
            errors.push(
                "cluster-bearing file requires metadata.cluster_count and \
                 metadata.cluster_member_count"
                    .into(),
            );
        }
        if metadata.image_pair_count.is_some() || metadata.match_count.is_some() {
            errors.push(
                "cluster-bearing file must not set metadata.image_pair_count / match_count".into(),
            );
        }
        if has_prefix("image_pairs/") {
            errors.push(
                "file stores clusters but contains image_pairs/ entries (exactly one backbone \
                 must be present)"
                    .into(),
            );
        }
        if !has_entry(entries::clusters_metadata()) {
            errors.push(
                "file claims has_clusters but has no clusters/ section (no backbone present)"
                    .into(),
            );
        }
        if metadata.has_cluster_patches && !has_entry(entries::cluster_patches_metadata()) {
            errors
                .push("file claims has_cluster_patches but has no cluster_patches/ section".into());
        }
        if !metadata.has_cluster_patches && has_prefix("cluster_patches/") {
            errors.push(
                "file contains cluster_patches/ entries but has_cluster_patches is false".into(),
            );
        }
    } else {
        if metadata.image_pair_count.is_none() || metadata.match_count.is_none() {
            errors.push(
                "pairwise file requires metadata.image_pair_count and metadata.match_count".into(),
            );
        }
        if metadata.cluster_count.is_some() || metadata.cluster_member_count.is_some() {
            errors.push(
                "pairwise file must not set metadata.cluster_count / cluster_member_count".into(),
            );
        }
        if has_prefix("clusters/") {
            errors.push(
                "file stores image_pairs but contains clusters/ entries (exactly one backbone \
                 must be present)"
                    .into(),
            );
        }
        if has_prefix("cluster_patches/") {
            errors.push("file contains cluster_patches/ entries but stores no clusters".into());
        }
        if !has_entry(entries::image_pairs_metadata()) {
            errors.push(
                "file has no image_pairs/ section and does not claim clusters (no backbone \
                 present)"
                    .into(),
            );
        }
    }

    errors
}

/// Verify integrity of a `.matches` file using content hashes and
/// structural constraints.
///
/// Returns `Ok((true, []))` if all checks pass, `Ok((false, errors))` with
/// details if verification fails. Returns `Err` only for I/O errors.
pub fn verify_matches(path: &Path) -> Result<(bool, Vec<String>), MatchesError> {
    let file = std::fs::File::open(path).map_err(|e| MatchesError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut errors = Vec::new();

    // Read stored hashes
    let content_hash_bytes = read_zst_entry(&mut archive, entries::content_hash())?;
    let stored: MatchesContentHash = serde_json::from_slice(&content_hash_bytes)?;

    // Read metadata for counts
    let metadata_raw = read_zst_entry(&mut archive, entries::metadata())?;
    let metadata: MatchesMetadata = serde_json::from_slice(&metadata_raw)?;

    // A version newer than this build understands has unknown semantics;
    // report it and stop rather than emit confusing per-file findings.
    if metadata.version > MATCHES_FORMAT_VERSION {
        errors.push(format!(
            "unsupported .matches format version {} (this build supports up to \
             {MATCHES_FORMAT_VERSION})",
            metadata.version
        ));
        return Ok((false, errors));
    }

    // Backbone / flag / entry consistency. A file that fails these is
    // structurally incoherent, so report and stop before section hashing
    // (which assumes the flagged sections exist).
    let entry_names: Vec<String> = archive.file_names().map(String::from).collect();
    let structural = structure_errors(&metadata, &entry_names);
    if !structural.is_empty() {
        errors.extend(structural);
        return Ok((false, errors));
    }

    // Present-section digests in the canonical order: metadata, images,
    // pairs, clusters, cluster_patches, two_view_geometries.
    let mut section_digests: Vec<u128> = Vec::with_capacity(6);

    // === Metadata hash ===
    let metadata_hash = xxhash_rust::xxh3::xxh3_128(&metadata_raw);
    check_section_hash(
        "Metadata",
        "metadata",
        metadata_hash,
        Some(&stored.metadata_xxh128),
        &mut errors,
    );
    section_digests.push(metadata_hash);

    // === Images ===
    let (images_hash, feature_counts_raw) =
        verify_images_section(&mut archive, &metadata, &stored, &mut errors)?;
    section_digests.push(images_hash);

    // === The correspondence backbone, and what it carries ===
    if !metadata.has_clusters {
        let pairs_hash = verify_image_pairs_section(
            &mut archive,
            &metadata,
            &feature_counts_raw,
            &stored,
            &mut errors,
        )?;
        section_digests.push(pairs_hash);
    } else {
        let (clusters_hash, backbone) = verify_clusters_section(
            &mut archive,
            &metadata,
            &feature_counts_raw,
            &stored,
            &mut errors,
        )?;
        section_digests.push(clusters_hash);

        if metadata.has_cluster_patches {
            let cp_hash =
                verify_cluster_patches_section(&mut archive, &backbone, &stored, &mut errors)?;
            section_digests.push(cp_hash);
        }
    }

    // === Two-view geometries (optional) ===
    if metadata.has_two_view_geometries {
        let tvg_hash =
            verify_two_view_geometries_section(&mut archive, &metadata, &stored, &mut errors)?;
        section_digests.push(tvg_hash);
    }

    // === Overall content hash ===
    let all_digests_bytes: Vec<u8> = section_digests
        .iter()
        .flat_map(|d| d.to_be_bytes())
        .collect();
    let content_hash_value = xxhash_rust::xxh3::xxh3_128(&all_digests_bytes);
    if format_hash(content_hash_value) != stored.content_xxh128 {
        errors.push(format!(
            "Overall content hash mismatch: computed {}, stored {}",
            format_hash(content_hash_value),
            stored.content_xxh128
        ));
    }

    Ok((errors.is_empty(), errors))
}

/// Compare one section's computed digest against `content_hash.json.zst`.
///
/// `label` opens the mismatch message ("Images hash mismatch: …"); `section`
/// names the section in the record, for the file that carries a section the
/// record has no digest for. A section whose stored digest is not optional
/// passes `Some` unconditionally and never reaches the second case.
fn check_section_hash(
    label: &str,
    section: &str,
    computed: u128,
    stored: Option<&str>,
    errors: &mut Vec<String>,
) {
    match stored {
        Some(stored) => {
            if format_hash(computed) != stored {
                errors.push(format!(
                    "{label} hash mismatch: computed {}, stored {stored}",
                    format_hash(computed)
                ));
            }
        }
        None => {
            errors.push(format!(
                "File has {section} but content_hash has no {section}_xxh128"
            ));
        }
    }
}

/// Hash `images/` and check the per-image dimensions.
///
/// Returns the section digest and the raw `images/feature_counts` bytes, which
/// both backbones' feature-index bounds checks read.
fn verify_images_section<R: std::io::Read + Seek>(
    archive: &mut ZipArchive<R>,
    metadata: &MatchesMetadata,
    stored: &MatchesContentHash,
    errors: &mut Vec<String>,
) -> Result<(u128, Vec<u8>), MatchesError> {
    let image_count = metadata.image_count as usize;
    let mut images_hasher = Xxh3::new();

    // images/feature_counts
    let feature_counts_raw = read_zst_entry(archive, &entries::images_feature_counts(image_count))?;
    images_hasher.update(&feature_counts_raw);

    // images/feature_tool_hashes
    images_hasher.update(&read_zst_entry(
        archive,
        &entries::images_feature_tool_hashes(image_count),
    )?);

    // images/image_dims (version 4+ only; structure_errors gated presence)
    if metadata.version >= 4 {
        let dims_raw = read_zst_entry(archive, &entries::images_image_dims(image_count))?;
        images_hasher.update(&dims_raw);
        if dims_raw.len() != image_count * 8 {
            errors.push(format!(
                "image_dims byte length {} != expected {} ({image_count} uint32 pairs)",
                dims_raw.len(),
                image_count * 8
            ));
        } else if let Some((k, _)) = raw_to_u32(&dims_raw)
            .iter()
            .enumerate()
            .find(|(_, &v)| v == 0)
        {
            errors.push(format!(
                "image_dims[{}] has a zero {} (every dimension must be >= 1)",
                k / 2,
                if k % 2 == 0 { "width" } else { "height" }
            ));
        }
    }

    // images/metadata.json
    images_hasher.update(&read_zst_entry(archive, entries::images_metadata())?);

    // images/names.json
    images_hasher.update(&read_zst_entry(archive, entries::images_names())?);

    // images/sift_content_hashes
    images_hasher.update(&read_zst_entry(
        archive,
        &entries::images_sift_content_hashes(image_count),
    )?);

    let images_hash = images_hasher.digest128();
    check_section_hash(
        "Images",
        "images",
        images_hash,
        Some(&stored.images_xxh128),
        errors,
    );
    Ok((images_hash, feature_counts_raw))
}

/// The `image_pairs/` raw entry bytes the section's checks read back.
struct PairsRaw {
    image_index_pairs: Vec<u8>,
    match_counts: Vec<u8>,
    match_feature_indexes: Vec<u8>,
}

/// Hash `image_pairs/` and check the pair table it stores.
fn verify_image_pairs_section<R: std::io::Read + Seek>(
    archive: &mut ZipArchive<R>,
    metadata: &MatchesMetadata,
    feature_counts_raw: &[u8],
    stored: &MatchesContentHash,
    errors: &mut Vec<String>,
) -> Result<u128, MatchesError> {
    let image_count = metadata.image_count as usize;
    let pair_count = metadata.image_pair_count.unwrap_or(0) as usize;
    let match_count = metadata.match_count.unwrap_or(0) as usize;
    let mut pairs_hasher = Xxh3::new();

    // image_pairs/image_index_pairs
    let pairs_raw = read_zst_entry(archive, &entries::image_pairs_image_index_pairs(pair_count))?;
    pairs_hasher.update(&pairs_raw);

    // image_pairs/match_counts
    let match_counts_raw = read_zst_entry(archive, &entries::image_pairs_match_counts(pair_count))?;
    pairs_hasher.update(&match_counts_raw);

    // image_pairs/match_descriptor_distances
    pairs_hasher.update(&read_zst_entry(
        archive,
        &entries::image_pairs_match_descriptor_distances(match_count),
    )?);

    // image_pairs/match_feature_indexes
    let match_fi_raw = read_zst_entry(
        archive,
        &entries::image_pairs_match_feature_indexes(match_count),
    )?;
    pairs_hasher.update(&match_fi_raw);

    // image_pairs/metadata.json
    pairs_hasher.update(&read_zst_entry(archive, entries::image_pairs_metadata())?);

    let pairs_hash = pairs_hasher.digest128();
    check_section_hash(
        "Image pairs",
        "image_pairs",
        pairs_hash,
        stored.image_pairs_xxh128.as_deref(),
        errors,
    );

    let raw = PairsRaw {
        image_index_pairs: pairs_raw,
        match_counts: match_counts_raw,
        match_feature_indexes: match_fi_raw,
    };

    // Structural validation on raw data
    check_pair_ordering(&raw.image_index_pairs, pair_count, image_count, errors);
    check_match_counts(&raw.match_counts, match_count, errors);
    check_match_feature_bounds(
        &raw,
        feature_counts_raw,
        pair_count,
        match_count,
        image_count,
        errors,
    );

    Ok(pairs_hash)
}

/// Validate pair sorting (idx_i < idx_j, lexicographic order).
fn check_pair_ordering(
    pairs_raw: &[u8],
    pair_count: usize,
    image_count: usize,
    errors: &mut Vec<String>,
) {
    if pair_count > 0 {
        let pair_idxs = raw_to_u32(pairs_raw);
        for k in 0..pair_count {
            let idx_i = pair_idxs[k * 2];
            let idx_j = pair_idxs[k * 2 + 1];
            if idx_i >= idx_j {
                errors.push(format!(
                    "image_index_pairs[{k}] = ({idx_i}, {idx_j}): idx_i must be < idx_j"
                ));
                break;
            }
            if idx_i as usize >= image_count || idx_j as usize >= image_count {
                errors.push(format!(
                    "image_index_pairs[{k}] = ({idx_i}, {idx_j}): index out of bounds (image_count = {image_count})"
                ));
                break;
            }
            if k > 0 {
                let prev_i = pair_idxs[(k - 1) * 2];
                let prev_j = pair_idxs[(k - 1) * 2 + 1];
                if (idx_i, idx_j) <= (prev_i, prev_j) {
                    errors.push(format!(
                        "image_index_pairs not sorted at index {k}: ({prev_i}, {prev_j}) >= ({idx_i}, {idx_j})"
                    ));
                    break;
                }
            }
        }
    }
}

/// Validate the per-pair match counts against the file's match total.
fn check_match_counts(match_counts_raw: &[u8], match_count: usize, errors: &mut Vec<String>) {
    if !match_counts_raw.is_empty() {
        let counts = raw_to_u32(match_counts_raw);
        let sum: u64 = counts.iter().map(|&c| c as u64).sum();
        if sum != match_count as u64 {
            errors.push(format!(
                "Sum of match_counts ({sum}) != match_count ({match_count})"
            ));
        }
        if counts.iter().any(|&c| c < 1) {
            errors.push("match_counts contains values < 1".into());
        }
    }
}

/// Validate that every matched feature index is within its image's feature
/// count, walking the CSR-style match runs pair by pair.
fn check_match_feature_bounds(
    raw: &PairsRaw,
    feature_counts_raw: &[u8],
    pair_count: usize,
    match_count: usize,
    image_count: usize,
    errors: &mut Vec<String>,
) {
    if pair_count > 0 && match_count > 0 {
        let pair_idxs = raw_to_u32(&raw.image_index_pairs);
        let counts = raw_to_u32(&raw.match_counts);
        let match_fi = raw_to_u32(&raw.match_feature_indexes);
        let feature_counts = raw_to_u32(feature_counts_raw);

        let mut offset: usize = 0;
        'outer: for k in 0..pair_count {
            let idx_i = pair_idxs[k * 2] as usize;
            let idx_j = pair_idxs[k * 2 + 1] as usize;
            if idx_i >= image_count || idx_j >= image_count {
                break;
            }
            let fc_i = feature_counts[idx_i];
            let fc_j = feature_counts[idx_j];
            let c = counts[k] as usize;
            for m in offset..offset + c {
                let fi = match_fi[m * 2];
                let fj = match_fi[m * 2 + 1];
                if fi >= fc_i {
                    errors.push(format!(
                        "match_feature_indexes[{m}][0] = {fi} >= feature_counts[{idx_i}] = {fc_i}"
                    ));
                    break 'outer;
                }
                if fj >= fc_j {
                    errors.push(format!(
                        "match_feature_indexes[{m}][1] = {fj} >= feature_counts[{idx_j}] = {fc_j}"
                    ));
                    break 'outer;
                }
            }
            offset += c;
        }
    }
}

/// The cluster backbone as `verify` holds it: the raw entry bytes, the counts
/// they are sized by, and whether they proved self-consistent.
///
/// `consistent` is the gate the cluster-patch checks share with the backbone's
/// own: once a length or CSR check has failed, the arrays cannot be indexed by
/// cluster or member without going out of bounds, so every later check that
/// would do so is skipped.
struct ClusterBackbone {
    cluster_count: usize,
    member_count: usize,
    cluster_starts: Vec<u8>,
    member_images: Vec<u8>,
    member_features: Vec<u8>,
    /// `None` before format version 6, which introduced the entry.
    member_positions: Option<Vec<u8>>,
    /// `None` before format version 6, which introduced the entry.
    member_affine_shapes: Option<Vec<u8>>,
    consistent: bool,
}

/// Hash `clusters/` and check the CSR backbone it stores.
///
/// Returns the section digest and the backbone, which the optional
/// `cluster_patches/` section indexes into.
fn verify_clusters_section<R: std::io::Read + Seek>(
    archive: &mut ZipArchive<R>,
    metadata: &MatchesMetadata,
    feature_counts_raw: &[u8],
    stored: &MatchesContentHash,
    errors: &mut Vec<String>,
) -> Result<(u128, ClusterBackbone), MatchesError> {
    let image_count = metadata.image_count as usize;
    let cluster_count = metadata.cluster_count.unwrap_or(0) as usize;
    let member_count = metadata.cluster_member_count.unwrap_or(0) as usize;
    let mut clusters_hasher = Xxh3::new();

    // clusters/cluster_starts
    let starts_raw = read_zst_entry(archive, &entries::clusters_cluster_starts(cluster_count))?;
    clusters_hasher.update(&starts_raw);

    // clusters/member_affine_shapes (version 6+ only; structure_errors
    // gated presence)
    let member_shapes_raw = if metadata.version >= 6 {
        let raw = read_zst_entry(
            archive,
            &entries::clusters_member_affine_shapes(member_count),
        )?;
        clusters_hasher.update(&raw);
        Some(raw)
    } else {
        None
    };

    // clusters/member_features
    let member_features_raw =
        read_zst_entry(archive, &entries::clusters_member_features(member_count))?;
    clusters_hasher.update(&member_features_raw);

    // clusters/member_images
    let member_images_raw =
        read_zst_entry(archive, &entries::clusters_member_images(member_count))?;
    clusters_hasher.update(&member_images_raw);

    // clusters/member_positions (version 6+ only)
    let member_positions_raw = if metadata.version >= 6 {
        let raw = read_zst_entry(archive, &entries::clusters_member_positions(member_count))?;
        clusters_hasher.update(&raw);
        Some(raw)
    } else {
        None
    };

    // clusters/metadata.json
    let clusters_meta_raw = read_zst_entry(archive, entries::clusters_metadata())?;
    clusters_hasher.update(&clusters_meta_raw);

    let clusters_hash = clusters_hasher.digest128();
    check_section_hash(
        "Clusters",
        "clusters",
        clusters_hash,
        stored.clusters_xxh128.as_deref(),
        errors,
    );

    // Cross-check clusters section metadata counts
    let clusters_meta: serde_json::Value = serde_json::from_slice(&clusters_meta_raw)?;
    if clusters_meta.get("cluster_count").and_then(|v| v.as_u64()) != Some(cluster_count as u64) {
        errors.push(
            "clusters/metadata.json.zst cluster_count doesn't match top-level metadata".into(),
        );
    }
    if clusters_meta.get("member_count").and_then(|v| v.as_u64()) != Some(member_count as u64) {
        errors.push(
            "clusters/metadata.json.zst member_count doesn't match top-level metadata".into(),
        );
    }

    let mut backbone = ClusterBackbone {
        cluster_count,
        member_count,
        cluster_starts: starts_raw,
        member_images: member_images_raw,
        member_features: member_features_raw,
        member_positions: member_positions_raw,
        member_affine_shapes: member_shapes_raw,
        consistent: true,
    };
    check_clusters_structure(&mut backbone, feature_counts_raw, image_count, errors);

    Ok((clusters_hash, backbone))
}

/// Check the cluster backbone's array lengths, its CSR offsets and its members'
/// image and feature indexes, recording in `backbone.consistent` whether the
/// arrays can be indexed by cluster and member.
fn check_clusters_structure(
    backbone: &mut ClusterBackbone,
    feature_counts_raw: &[u8],
    image_count: usize,
    errors: &mut Vec<String>,
) {
    let cluster_count = backbone.cluster_count;
    let member_count = backbone.member_count;
    let starts_raw = &backbone.cluster_starts;
    let member_images_raw = &backbone.member_images;
    let member_features_raw = &backbone.member_features;

    let mut clusters_ok = true;
    if starts_raw.len() != (cluster_count + 1) * 4 {
        errors.push(format!(
            "cluster_starts byte length {} != expected {} ({} uint32 values)",
            starts_raw.len(),
            (cluster_count + 1) * 4,
            cluster_count + 1
        ));
        clusters_ok = false;
    }
    if member_images_raw.len() != member_count * 4 {
        errors.push(format!(
            "member_images byte length {} != expected {} ({member_count} uint32 values)",
            member_images_raw.len(),
            member_count * 4
        ));
        clusters_ok = false;
    }
    if member_features_raw.len() != member_count * 4 {
        errors.push(format!(
            "member_features byte length {} != expected {} ({member_count} uint32 values)",
            member_features_raw.len(),
            member_count * 4
        ));
        clusters_ok = false;
    }
    // (M, 2) and (M, 2, 2) float32: 8 and 16 bytes per member.
    for (name, raw, expected) in [
        (
            "member_positions",
            &backbone.member_positions,
            member_count * 8,
        ),
        (
            "member_affine_shapes",
            &backbone.member_affine_shapes,
            member_count * 16,
        ),
    ] {
        if let Some(raw) = raw {
            if raw.len() != expected {
                errors.push(format!(
                    "{name} byte length {} != expected {expected}",
                    raw.len()
                ));
                clusters_ok = false;
            }
        }
    }

    let starts = raw_to_u32(starts_raw);
    let member_images = raw_to_u32(member_images_raw);
    let member_features = raw_to_u32(member_features_raw);

    if clusters_ok {
        if starts[0] != 0 {
            errors.push(format!("cluster_starts[0] = {} != 0", starts[0]));
            clusters_ok = false;
        }
        for c in 0..cluster_count {
            if starts[c + 1] < starts[c] {
                errors.push(format!(
                    "cluster_starts not non-decreasing at cluster {c}: {} > {}",
                    starts[c],
                    starts[c + 1]
                ));
                clusters_ok = false;
                break;
            }
            if starts[c + 1] - starts[c] < 2 {
                errors.push(format!(
                    "cluster {c} has {} members; every cluster must have >= 2",
                    starts[c + 1] - starts[c]
                ));
                clusters_ok = false;
                break;
            }
        }
        if clusters_ok && starts[cluster_count] as usize != member_count {
            errors.push(format!(
                "cluster_starts final value {} != member count {member_count}",
                starts[cluster_count]
            ));
            clusters_ok = false;
        }
    }

    if clusters_ok {
        let feature_counts = raw_to_u32(feature_counts_raw);
        for k in 0..member_count {
            let img = member_images[k];
            if img as usize >= image_count || img as usize >= feature_counts.len() {
                errors.push(format!(
                    "member_images[{k}] = {img} >= image_count {image_count}"
                ));
                break;
            }
            let feat = member_features[k];
            let fc = feature_counts[img as usize];
            if feat >= fc {
                errors.push(format!(
                    "member_features[{k}] = {feat} >= feature_counts[{img}] = {fc}"
                ));
                break;
            }
        }
    }

    backbone.consistent = clusters_ok;
}

/// The `cluster_patches/` raw entry bytes the section's checks read back.
struct ClusterPatchRaw {
    member_consistency_residual: Vec<u8>,
    member_shift_px: Vec<u8>,
    member_status: Vec<u8>,
    member_zncc: Vec<u8>,
    reference_members: Vec<u8>,
}

/// Hash `cluster_patches/` and check the refinement record it stores against
/// the cluster backbone it enriches.
fn verify_cluster_patches_section<R: std::io::Read + Seek>(
    archive: &mut ZipArchive<R>,
    backbone: &ClusterBackbone,
    stored: &MatchesContentHash,
    errors: &mut Vec<String>,
) -> Result<u128, MatchesError> {
    let cluster_count = backbone.cluster_count;
    let member_count = backbone.member_count;
    let mut cp_hasher = Xxh3::new();

    // cluster_patches/member_consistency_residual
    let consistency_raw = read_zst_entry(
        archive,
        &entries::cluster_patches_member_consistency_residual(member_count),
    )?;
    cp_hasher.update(&consistency_raw);

    // cluster_patches/member_shift_px
    let shift_raw = read_zst_entry(
        archive,
        &entries::cluster_patches_member_shift_px(member_count),
    )?;
    cp_hasher.update(&shift_raw);

    // cluster_patches/member_status
    let status_raw = read_zst_entry(
        archive,
        &entries::cluster_patches_member_status(member_count),
    )?;
    cp_hasher.update(&status_raw);

    // cluster_patches/member_zncc
    let zncc_raw = read_zst_entry(archive, &entries::cluster_patches_member_zncc(member_count))?;
    cp_hasher.update(&zncc_raw);

    // cluster_patches/metadata.json
    let cp_meta_raw = read_zst_entry(archive, entries::cluster_patches_metadata())?;
    cp_hasher.update(&cp_meta_raw);

    // cluster_patches/reference_members
    let refs_raw = read_zst_entry(
        archive,
        &entries::cluster_patches_reference_members(cluster_count),
    )?;
    cp_hasher.update(&refs_raw);

    let cp_hash = cp_hasher.digest128();
    check_section_hash(
        "Cluster patches",
        "cluster_patches",
        cp_hash,
        stored.cluster_patches_xxh128.as_deref(),
        errors,
    );

    // Cross-check cluster_patches section metadata counts
    let cp_meta: serde_json::Value = serde_json::from_slice(&cp_meta_raw)?;
    if cp_meta.get("cluster_count").and_then(|v| v.as_u64()) != Some(cluster_count as u64) {
        errors.push(
            "cluster_patches/metadata.json.zst cluster_count doesn't match top-level \
             metadata"
                .into(),
        );
    }
    if cp_meta.get("member_count").and_then(|v| v.as_u64()) != Some(member_count as u64) {
        errors.push(
            "cluster_patches/metadata.json.zst member_count doesn't match top-level \
             metadata"
                .into(),
        );
    }

    let raw = ClusterPatchRaw {
        member_consistency_residual: consistency_raw,
        member_shift_px: shift_raw,
        member_status: status_raw,
        member_zncc: zncc_raw,
        reference_members: refs_raw,
    };

    // Structural validation on raw cluster-patch data
    let cp_ok = check_cluster_patch_lengths(&raw, backbone, errors);
    if cp_ok {
        check_member_statuses(&raw.member_status, errors);
    }
    if cp_ok && backbone.consistent {
        check_cluster_references(&raw, backbone, errors);
    }

    Ok(cp_hash)
}

/// Check every `cluster_patches/` array against the count it is sized by.
///
/// Returns whether they can be indexed by member (and, for
/// `reference_members`, by cluster).
fn check_cluster_patch_lengths(
    raw: &ClusterPatchRaw,
    backbone: &ClusterBackbone,
    errors: &mut Vec<String>,
) -> bool {
    let member_count = backbone.member_count;
    let cluster_count = backbone.cluster_count;

    let mut cp_ok = true;
    for (name, raw_len, expected) in [
        (
            "member_consistency_residual",
            raw.member_consistency_residual.len(),
            member_count * 4,
        ),
        (
            "member_shift_px",
            raw.member_shift_px.len(),
            member_count * 4,
        ),
        ("member_status", raw.member_status.len(), member_count),
        ("member_zncc", raw.member_zncc.len(), member_count * 4),
        (
            "reference_members",
            raw.reference_members.len(),
            cluster_count * 4,
        ),
    ] {
        if raw_len != expected {
            errors.push(format!(
                "{name} byte length {raw_len} != expected {expected}"
            ));
            cp_ok = false;
        }
    }
    cp_ok
}

/// Check that every member status is a [`ClusterMemberStatus`] discriminant.
fn check_member_statuses(status_raw: &[u8], errors: &mut Vec<String>) {
    for (k, &status) in status_raw.iter().enumerate() {
        if ClusterMemberStatus::from_u8(status).is_none() {
            errors.push(format!(
                "member_status[{k}] = {status} is not a valid ClusterMemberStatus \
                 discriminant"
            ));
            break;
        }
    }
}

/// Check each cluster's reference member: that it lies in the cluster, that it
/// is marked as the reference, that its affine shape is invertible, and that
/// no image is covered twice within the cluster.
fn check_cluster_references(
    raw: &ClusterPatchRaw,
    backbone: &ClusterBackbone,
    errors: &mut Vec<String>,
) {
    let starts = raw_to_u32(&backbone.cluster_starts);
    let member_images = raw_to_u32(&backbone.member_images);
    let status_raw = &raw.member_status;
    let refs = raw_to_u32(&raw.reference_members);

    'clusters: for c in 0..backbone.cluster_count {
        let start = starts[c];
        let end = starts[c + 1];

        let reference = refs[c];
        if reference != CLUSTER_REFERENCE_UNREFINABLE {
            if reference < start || reference >= end {
                errors.push(format!(
                    "reference_members[{c}] = {reference} outside cluster member \
                     range [{start}, {end})"
                ));
                break;
            }
            if status_raw[reference as usize] != ClusterMemberStatus::Reference as u8 {
                errors.push(format!(
                    "reference_members[{c}] = {reference} has status {}, expected {} \
                     (reference)",
                    status_raw[reference as usize],
                    ClusterMemberStatus::Reference as u8
                ));
                break;
            }
            // A cluster's reference member carries `S_ref`, its
            // own detector affine shape, in the backbone's shape
            // array. The value is not checkable without the
            // `.sift` data, but it must be invertible — the
            // relative warp `W = S·S_ref⁻¹` is recovered through
            // it. Only version 6+ files reach here with the array
            // present; older cluster files carry other semantics
            // and are refused by the reader.
            if let Some(raw) = &backbone.member_affine_shapes {
                let shapes = raw_to_f32(raw);
                let base = reference as usize * 4;
                let det = f64::from(shapes[base]) * f64::from(shapes[base + 3])
                    - f64::from(shapes[base + 1]) * f64::from(shapes[base + 2]);
                if !det.is_finite() || det == 0.0 {
                    errors.push(format!(
                        "member_affine_shapes[{reference}] (cluster {c}'s reference \
                         member) is singular (det {det})"
                    ));
                    break;
                }
            }
        }

        // At most one Reference/Kept member per (cluster, image).
        let mut covered: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for m in start as usize..end as usize {
            let status = status_raw[m];
            if status == ClusterMemberStatus::Reference as u8
                || status == ClusterMemberStatus::Kept as u8
            {
                let img = member_images[m];
                if let Some(prev) = covered.insert(img, m) {
                    errors.push(format!(
                        "cluster {c}: members {prev} and {m} are both reference/kept \
                         for image {img}"
                    ));
                    break 'clusters;
                }
            }
        }
    }
}

/// Hash `two_view_geometries/` and check its per-pair counts.
fn verify_two_view_geometries_section<R: std::io::Read + Seek>(
    archive: &mut ZipArchive<R>,
    metadata: &MatchesMetadata,
    stored: &MatchesContentHash,
    errors: &mut Vec<String>,
) -> Result<u128, MatchesError> {
    let pair_count = metadata.image_pair_count.unwrap_or(0) as usize;
    let mut tvg_hasher = Xxh3::new();

    // Read TVG metadata for inlier_count
    let tvg_meta_raw = read_zst_entry(archive, entries::two_view_geometries_metadata())?;
    let tvg_meta: TvgMetadata = serde_json::from_slice(&tvg_meta_raw)?;
    let inlier_count = tvg_meta.inlier_count as usize;

    if tvg_meta.image_pair_count as usize != pair_count {
        errors.push(format!(
            "TVG image_pair_count {} != pair_count {pair_count}",
            tvg_meta.image_pair_count
        ));
    }

    // Hash all TVG files in lexicographic order

    // two_view_geometries/config_indexes
    tvg_hasher.update(&read_zst_entry(
        archive,
        &entries::two_view_geometries_config_indexes(pair_count),
    )?);

    // two_view_geometries/config_types.json
    tvg_hasher.update(&read_zst_entry(
        archive,
        entries::two_view_geometries_config_types(),
    )?);

    // two_view_geometries/e_matrices
    tvg_hasher.update(&read_zst_entry(
        archive,
        &entries::two_view_geometries_e_matrices(pair_count),
    )?);

    // two_view_geometries/f_matrices
    tvg_hasher.update(&read_zst_entry(
        archive,
        &entries::two_view_geometries_f_matrices(pair_count),
    )?);

    // two_view_geometries/h_matrices
    tvg_hasher.update(&read_zst_entry(
        archive,
        &entries::two_view_geometries_h_matrices(pair_count),
    )?);

    // two_view_geometries/inlier_counts
    let inlier_counts_raw = read_zst_entry(
        archive,
        &entries::two_view_geometries_inlier_counts(pair_count),
    )?;
    tvg_hasher.update(&inlier_counts_raw);

    // two_view_geometries/inlier_feature_indexes
    tvg_hasher.update(&read_zst_entry(
        archive,
        &entries::two_view_geometries_inlier_feature_indexes(inlier_count),
    )?);

    // two_view_geometries/metadata.json
    tvg_hasher.update(&tvg_meta_raw);

    // two_view_geometries/quaternions_wxyz
    tvg_hasher.update(&read_zst_entry(
        archive,
        &entries::two_view_geometries_quaternions_wxyz(pair_count),
    )?);

    // two_view_geometries/translations_xyz
    tvg_hasher.update(&read_zst_entry(
        archive,
        &entries::two_view_geometries_translations_xyz(pair_count),
    )?);

    let tvg_hash = tvg_hasher.digest128();
    check_section_hash(
        "TVG",
        "two_view_geometries",
        tvg_hash,
        stored.two_view_geometries_xxh128.as_deref(),
        errors,
    );

    // Validate inlier_counts sum
    if !inlier_counts_raw.is_empty() {
        let inlier_counts = raw_to_u32(&inlier_counts_raw);
        let inlier_sum: u64 = inlier_counts.iter().map(|&c| c as u64).sum();
        if inlier_sum != inlier_count as u64 {
            errors.push(format!(
                "Sum of inlier_counts ({inlier_sum}) != inlier_count ({inlier_count})"
            ));
        }
    }

    Ok(tvg_hash)
}
