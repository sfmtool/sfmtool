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

use std::borrow::Cow;
use std::io::Seek;
use std::path::Path;

use xxhash_rust::xxh3::Xxh3;
use zip::ZipArchive;

use crate::cells::{
    cell_entry_errors, cell_entry_names, normalize_cell_statuses, not_kept_with_readings,
    read_member_cell_status_legend,
};
use crate::entries;
use crate::types::*;
use crate::validate::{
    check_backbone_counts, cluster_errors, cluster_patch_errors, config_index_errors,
    image_dims_errors, inlier_errors, pair_errors, ClusterSlices,
};
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

    errors.extend(check_backbone_counts(metadata));

    if metadata.has_clusters {
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
        if metadata.has_two_view_geometries && !has_entry(entries::two_view_geometries_metadata()) {
            errors.push(
                "file claims has_two_view_geometries but has no two_view_geometries/ section"
                    .into(),
            );
        }
    }
    if !metadata.has_two_view_geometries && has_prefix("two_view_geometries/") {
        errors.push(
            "file contains two_view_geometries/ entries but has_two_view_geometries is false"
                .into(),
        );
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
    let entry_names = archive
        .file_names()
        .map(|name| name.map(String::from))
        .collect::<Result<Vec<String>, _>>()?;
    let structural = structure_errors(&metadata, &entry_names);
    if !structural.is_empty() {
        errors.extend(structural);
        return Ok((false, errors));
    }

    // Present-section digests in the canonical order: metadata, images,
    // pairs, clusters, cluster_patches, two_view_geometries.
    let mut section_digests = sfmtool_archive_io::SectionDigests::new();

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
    let mut pairs = None;
    if !metadata.has_clusters {
        let (pairs_hash, pairs_raw) = verify_image_pairs_section(
            &mut archive,
            &metadata,
            &feature_counts_raw,
            &stored,
            &mut errors,
        )?;
        section_digests.push(pairs_hash);
        pairs = pairs_raw;
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
            let cp_hash = verify_cluster_patches_section(
                &mut archive,
                metadata.version,
                &backbone,
                &stored,
                &mut errors,
            )?;
            section_digests.push(cp_hash);
        }
    }

    // === Two-view geometries (optional) ===
    if metadata.has_two_view_geometries {
        let tvg_hash = verify_two_view_geometries_section(
            &mut archive,
            &metadata,
            pairs.as_ref(),
            &stored,
            &mut errors,
        )?;
        section_digests.push(tvg_hash);
    }

    // === Overall content hash ===
    let content_hash_value = section_digests.finish();
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
        } else {
            image_dims_errors(&raw_to_u32(&dims_raw), errors);
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
///
/// Returns the section digest and, when every array is the length its count
/// calls for, the raw pair table, which the two-view geometries' inlier check
/// reads.
fn verify_image_pairs_section<R: std::io::Read + Seek>(
    archive: &mut ZipArchive<R>,
    metadata: &MatchesMetadata,
    feature_counts_raw: &[u8],
    stored: &MatchesContentHash,
    errors: &mut Vec<String>,
) -> Result<(u128, Option<PairsRaw>), MatchesError> {
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

    // Structural validation on raw data. The length checks run first and gate
    // the rest, because everything below indexes by pair, by match or by
    // image: an array that is not the length its declared count calls for
    // cannot be walked. `pair_errors` extends that gate itself: the match
    // counts are the CSR run lengths its feature-bounds walk advances
    // through, so it walks only when they sum to `match_count`.
    let lengths_ok = check_pair_array_lengths(
        &raw,
        feature_counts_raw,
        pair_count,
        match_count,
        image_count,
        errors,
    );
    if !lengths_ok {
        return Ok((pairs_hash, None));
    }
    pair_errors(
        &raw_to_u32(&raw.image_index_pairs),
        &raw_to_u32(&raw.match_counts),
        &raw_to_u32(&raw.match_feature_indexes),
        &raw_to_u32(feature_counts_raw),
        image_count,
        errors,
    );

    Ok((pairs_hash, Some(raw)))
}

/// Check every array the pair-table checks index against the count it is sized
/// by: the three `image_pairs/` entries, plus the `images/feature_counts` the
/// bounds walk resolves each pair's two images through.
///
/// Returns whether they can be indexed by pair, by match and by image. This is
/// the pairwise backbone's counterpart to the cluster backbone's `consistent`
/// flag and plays the same role — once an array is not the length its declared
/// count calls for, no later check may walk it, and the length error itself is
/// what tells the reader why those findings are absent.
fn check_pair_array_lengths(
    raw: &PairsRaw,
    feature_counts_raw: &[u8],
    pair_count: usize,
    match_count: usize,
    image_count: usize,
    errors: &mut Vec<String>,
) -> bool {
    let mut pairs_ok = true;
    // The (P, 2) and (M, 2) uint32 tables are 8 bytes per row; the flat
    // per-pair and per-image uint32 arrays are 4 bytes per element.
    for (name, raw_len, expected, count, unit) in [
        (
            "image_index_pairs",
            raw.image_index_pairs.len(),
            pair_count * 8,
            pair_count,
            "uint32 pairs",
        ),
        (
            "match_counts",
            raw.match_counts.len(),
            pair_count * 4,
            pair_count,
            "uint32 values",
        ),
        (
            "match_feature_indexes",
            raw.match_feature_indexes.len(),
            match_count * 8,
            match_count,
            "uint32 pairs",
        ),
        (
            "feature_counts",
            feature_counts_raw.len(),
            image_count * 4,
            image_count,
            "uint32 values",
        ),
    ] {
        if raw_len != expected {
            errors.push(format!(
                "{name} byte length {raw_len} != expected {expected} ({count} {unit})"
            ));
            pairs_ok = false;
        }
    }
    pairs_ok
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

/// The cluster backbone's raw entries read as the values they hold, borrowed
/// where the bytes are aligned, for the content checks.
struct RawClusterSlices<'a> {
    cluster_starts: Cow<'a, [u32]>,
    member_images: Cow<'a, [u32]>,
    member_features: Cow<'a, [u32]>,
    member_positions: Option<Cow<'a, [f32]>>,
    member_affine_shapes: Option<Cow<'a, [f32]>>,
}

impl<'a> RawClusterSlices<'a> {
    fn new(backbone: &'a ClusterBackbone) -> Self {
        RawClusterSlices {
            cluster_starts: raw_to_u32(&backbone.cluster_starts),
            member_images: raw_to_u32(&backbone.member_images),
            member_features: raw_to_u32(&backbone.member_features),
            member_positions: backbone.member_positions.as_deref().map(raw_to_f32),
            member_affine_shapes: backbone.member_affine_shapes.as_deref().map(raw_to_f32),
        }
    }

    fn view(&self) -> ClusterSlices<'_> {
        ClusterSlices {
            cluster_starts: &self.cluster_starts,
            member_images: &self.member_images,
            member_features: &self.member_features,
            member_positions: self.member_positions.as_deref(),
            member_affine_shapes: self.member_affine_shapes.as_deref(),
        }
    }
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
    let starts_raw = read_zst_entry(
        archive,
        &entries::clusters_cluster_starts(cluster_count + 1),
    )?;
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

    if clusters_ok {
        clusters_ok = cluster_errors(
            &RawClusterSlices::new(backbone).view(),
            &raw_to_u32(feature_counts_raw),
            image_count,
            errors,
        );
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
///
/// The `member_status` codes are resolved through the file's legend (the
/// canonical one for a `version` 6 file, which carries none) before the
/// per-cluster checks read them, so those checks see the canonical numbering
/// whatever order the file stated its legend in.
fn verify_cluster_patches_section<R: std::io::Read + Seek>(
    archive: &mut ZipArchive<R>,
    version: u32,
    backbone: &ClusterBackbone,
    stored: &MatchesContentHash,
    errors: &mut Vec<String>,
) -> Result<u128, MatchesError> {
    let cluster_count = backbone.cluster_count;
    let member_count = backbone.member_count;
    let mut cp_hasher = Xxh3::new();

    // The section's metadata says whether the per-cell columns are present, so
    // it is read before them; it is hashed in its own place below.
    let cp_meta_raw = read_zst_entry(archive, entries::cluster_patches_metadata())?;
    let cp_meta: serde_json::Value = serde_json::from_slice(&cp_meta_raw)?;

    // cluster_patches/member_cell_* (optional, version 8). Every entry under
    // that name is hashed, in name order, whatever its shape, as the writer's
    // lexicographic order would have taken it; they sort before
    // member_consistency_residual. A wrong or unpaired entry is reported
    // below, not left out of the digest.
    let cell_legend = match read_member_cell_status_legend(&cp_meta, version) {
        Ok(legend) => legend,
        Err(e) => {
            errors.push(e);
            None
        }
    };
    let entry_names = archive
        .file_names()
        .map(|name| name.map(String::from))
        .collect::<Result<Vec<String>, _>>()?;
    let mut cell_names: Vec<&String> = entry_names
        .iter()
        .filter(|n| n.starts_with("cluster_patches/member_cell_"))
        .collect();
    cell_names.sort();
    let mut cell_raw: Vec<(String, Vec<u8>)> = Vec::with_capacity(cell_names.len());
    for name in cell_names {
        let raw = read_zst_entry(archive, name)?;
        cp_hasher.update(&raw);
        cell_raw.push((name.clone(), raw));
    }
    let has_cell_legend = cp_meta
        .get(crate::cells::MEMBER_CELL_STATUS_NAMES_KEY)
        .is_some();
    let cell_entries_errors = cell_entry_errors(
        entry_names.iter().map(String::as_str),
        member_count,
        has_cell_legend,
    );
    let cells_ok = cell_entries_errors.is_empty();
    errors.extend(cell_entries_errors);

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

    // The legend the member_status codes index. Without a readable one the
    // codes say nothing, so the checks that read them are skipped.
    let status_legend = match read_member_status_legend(&cp_meta, version) {
        Ok(legend) => Some(legend),
        Err(e) => {
            errors.push(e);
            None
        }
    };

    let mut raw = ClusterPatchRaw {
        member_consistency_residual: consistency_raw,
        member_shift_px: shift_raw,
        member_status: status_raw,
        member_zncc: zncc_raw,
        reference_members: refs_raw,
    };

    // Structural validation on raw cluster-patch data
    let cp_ok = check_cluster_patch_lengths(&raw, backbone, errors);
    let statuses_ok = match &status_legend {
        Some(legend) if cp_ok => match normalize_member_statuses(&mut raw.member_status, legend) {
            Ok(()) => true,
            Err(e) => {
                errors.push(e);
                false
            }
        },
        _ => false,
    };
    if statuses_ok && backbone.consistent {
        cluster_patch_errors(
            &RawClusterSlices::new(backbone).view(),
            &raw_to_u32(&raw.reference_members),
            &raw.member_status,
            errors,
        );
    }
    if let (true, Some(legend)) = (cells_ok, &cell_legend) {
        check_member_cells(
            &cell_raw,
            member_count,
            legend,
            statuses_ok.then_some(raw.member_status.as_slice()),
            errors,
        );
    }

    Ok(cp_hash)
}

/// Check the per-cell columns, all four present at the member count's shape:
/// each sized by the member count, every status code within the file's
/// legend, and, when the member statuses could be read (`member_status`, in
/// the canonical numbering), no reading on a member that is not `kept`.
fn check_member_cells(
    cell_raw: &[(String, Vec<u8>)],
    member_count: usize,
    legend: &[u8],
    member_status: Option<&[u8]>,
    errors: &mut Vec<String>,
) {
    let bytes = |name: &str| -> &[u8] {
        cell_raw
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, raw)| raw.as_slice())
            .unwrap_or(&[])
    };
    let [(_, iterations_name), (_, shift_name), (_, status_name), (_, zncc_name)] =
        cell_entry_names(member_count);
    let mut ok = true;
    for (name, expected) in [
        (&iterations_name, member_count),
        (&shift_name, member_count * 18 * 4),
        (&status_name, member_count * 9),
        (&zncc_name, member_count * 9 * 4),
    ] {
        let len = bytes(name).len();
        if len != expected {
            errors.push(format!("{name} byte length {len} != expected {expected}"));
            ok = false;
        }
    }
    if !ok {
        return;
    }
    let mut status = bytes(&status_name).to_vec();
    if let Err(e) = normalize_cell_statuses(&mut status, legend) {
        errors.push(e);
        return;
    }
    if let Some(member_status) = member_status {
        let floats = |name: &str| -> Vec<f32> {
            bytes(name)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect()
        };
        if let Some(e) = not_kept_with_readings(
            member_status,
            &status,
            &floats(&shift_name),
            &floats(&zncc_name),
            bytes(&iterations_name),
        ) {
            errors.push(e);
        }
    }
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

/// Hash `two_view_geometries/` and check its per-pair counts, its config
/// indexes and, when `pairs` holds the pair table at its declared lengths, that
/// every inlier is one of its pair's candidate matches.
fn verify_two_view_geometries_section<R: std::io::Read + Seek>(
    archive: &mut ZipArchive<R>,
    metadata: &MatchesMetadata,
    pairs: Option<&PairsRaw>,
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
    let config_indexes_raw = read_zst_entry(
        archive,
        &entries::two_view_geometries_config_indexes(pair_count),
    )?;
    tvg_hasher.update(&config_indexes_raw);

    // two_view_geometries/config_types.json
    let config_types_raw = read_zst_entry(archive, entries::two_view_geometries_config_types())?;
    tvg_hasher.update(&config_types_raw);

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
    let inlier_fi_raw = read_zst_entry(
        archive,
        &entries::two_view_geometries_inlier_feature_indexes(inlier_count),
    )?;
    tvg_hasher.update(&inlier_fi_raw);

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

    // Every config index names an entry of config_types.
    let config_types: Vec<serde_json::Value> = serde_json::from_slice(&config_types_raw)?;
    config_index_errors(&config_indexes_raw, config_types.len(), errors);

    // The inlier arrays are walked by pair and by inlier, so they are checked
    // against their declared counts first, as the pair table's are.
    let mut lengths_ok = true;
    for (name, raw_len, expected, count, unit) in [
        (
            "inlier_counts",
            inlier_counts_raw.len(),
            pair_count * 4,
            pair_count,
            "uint32 values",
        ),
        (
            "inlier_feature_indexes",
            inlier_fi_raw.len(),
            inlier_count * 8,
            inlier_count,
            "uint32 pairs",
        ),
    ] {
        if raw_len != expected {
            errors.push(format!(
                "{name} byte length {raw_len} != expected {expected} ({count} {unit})"
            ));
            lengths_ok = false;
        }
    }
    if lengths_ok {
        let candidates = pairs.map(|pairs| {
            (
                raw_to_u32(&pairs.match_counts),
                raw_to_u32(&pairs.match_feature_indexes),
            )
        });
        inlier_errors(
            &raw_to_u32(&inlier_counts_raw),
            &raw_to_u32(&inlier_fi_raw),
            candidates
                .as_ref()
                .map(|(counts, indexes)| (&counts[..], &indexes[..])),
            errors,
        );
    }

    Ok(tvg_hash)
}
