// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.matches` file integrity verification.

use std::borrow::Cow;
use std::path::Path;

use xxhash_rust::xxh3::Xxh3;

use crate::archive_io::{format_hash, read_zst_entry};
use crate::types::*;

/// Reinterpret a freshly decompressed byte buffer as `u32` values.
///
/// `read_zst_entry` returns a `Vec<u8>` whose start address is only guaranteed
/// to be 1-aligned, so `bytemuck::cast_slice::<u8, u32>` panics when the buffer
/// is not 4-aligned. Borrow the buffer directly when it is already aligned (the
/// common case, no copy), and fall back to copying through a freshly aligned
/// `Vec<u32>` only when it is not. Any trailing bytes that do not form a whole
/// `u32` (truncated/corrupt entry) are dropped; structural checks downstream
/// then catch the mismatch.
fn raw_to_u32(raw: &[u8]) -> Cow<'_, [u32]> {
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
fn raw_to_f64(raw: &[u8]) -> Cow<'_, [f64]> {
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
    let has_dims = has_prefix("images/image_dims.");
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
        if !has_entry("clusters/metadata.json.zst") {
            errors.push(
                "file claims has_clusters but has no clusters/ section (no backbone present)"
                    .into(),
            );
        }
        if metadata.has_cluster_patches && !has_entry("cluster_patches/metadata.json.zst") {
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
        if !has_entry("image_pairs/metadata.json.zst") {
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
    let content_hash_bytes = read_zst_entry(&mut archive, "content_hash.json.zst")?;
    let stored: MatchesContentHash = serde_json::from_slice(&content_hash_bytes)?;

    // Read metadata for counts
    let metadata_raw = read_zst_entry(&mut archive, "metadata.json.zst")?;
    let metadata: MatchesMetadata = serde_json::from_slice(&metadata_raw)?;
    let image_count = metadata.image_count as usize;

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
    if format_hash(metadata_hash) != stored.metadata_xxh128 {
        errors.push(format!(
            "Metadata hash mismatch: computed {}, stored {}",
            format_hash(metadata_hash),
            stored.metadata_xxh128
        ));
    }
    section_digests.push(metadata_hash);

    // === Images hash (lexicographic path order) ===
    let mut images_hasher = Xxh3::new();

    // images/feature_counts
    let feature_counts_raw = read_zst_entry(
        &mut archive,
        &format!("images/feature_counts.{image_count}.uint32.zst"),
    )?;
    images_hasher.update(&feature_counts_raw);

    // images/feature_tool_hashes
    images_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("images/feature_tool_hashes.{image_count}.uint128.zst"),
    )?);

    // images/image_dims (version 4+ only; structure_errors gated presence)
    if metadata.version >= 4 {
        let dims_raw = read_zst_entry(
            &mut archive,
            &format!("images/image_dims.{image_count}.2.uint32.zst"),
        )?;
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
    images_hasher.update(&read_zst_entry(&mut archive, "images/metadata.json.zst")?);

    // images/names.json
    images_hasher.update(&read_zst_entry(&mut archive, "images/names.json.zst")?);

    // images/sift_content_hashes
    images_hasher.update(&read_zst_entry(
        &mut archive,
        &format!("images/sift_content_hashes.{image_count}.uint128.zst"),
    )?);

    let images_hash = images_hasher.digest128();
    if format_hash(images_hash) != stored.images_xxh128 {
        errors.push(format!(
            "Images hash mismatch: computed {}, stored {}",
            format_hash(images_hash),
            stored.images_xxh128
        ));
    }
    section_digests.push(images_hash);

    if !metadata.has_clusters {
        // === Image pairs hash (lexicographic path order) ===
        let pair_count = metadata.image_pair_count.unwrap_or(0) as usize;
        let match_count = metadata.match_count.unwrap_or(0) as usize;
        let mut pairs_hasher = Xxh3::new();

        // image_pairs/image_index_pairs
        let pairs_raw = read_zst_entry(
            &mut archive,
            &format!("image_pairs/image_index_pairs.{pair_count}.2.uint32.zst"),
        )?;
        pairs_hasher.update(&pairs_raw);

        // image_pairs/match_counts
        let match_counts_raw = read_zst_entry(
            &mut archive,
            &format!("image_pairs/match_counts.{pair_count}.uint32.zst"),
        )?;
        pairs_hasher.update(&match_counts_raw);

        // image_pairs/match_descriptor_distances
        pairs_hasher.update(&read_zst_entry(
            &mut archive,
            &format!("image_pairs/match_descriptor_distances.{match_count}.float32.zst"),
        )?);

        // image_pairs/match_feature_indexes
        let match_fi_raw = read_zst_entry(
            &mut archive,
            &format!("image_pairs/match_feature_indexes.{match_count}.2.uint32.zst"),
        )?;
        pairs_hasher.update(&match_fi_raw);

        // image_pairs/metadata.json
        pairs_hasher.update(&read_zst_entry(
            &mut archive,
            "image_pairs/metadata.json.zst",
        )?);

        let pairs_hash = pairs_hasher.digest128();
        match &stored.image_pairs_xxh128 {
            Some(stored_pairs) => {
                if &format_hash(pairs_hash) != stored_pairs {
                    errors.push(format!(
                        "Image pairs hash mismatch: computed {}, stored {}",
                        format_hash(pairs_hash),
                        stored_pairs
                    ));
                }
            }
            None => {
                errors
                    .push("File has image_pairs but content_hash has no image_pairs_xxh128".into());
            }
        }
        section_digests.push(pairs_hash);

        // === Structural validation on raw data ===

        // Validate pair sorting (idx_i < idx_j, lexicographic order)
        if pair_count > 0 {
            let pair_idxs = raw_to_u32(&pairs_raw);
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

        // Validate match_counts sum
        if !match_counts_raw.is_empty() {
            let counts = raw_to_u32(&match_counts_raw);
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

        // Validate feature index bounds
        if pair_count > 0 && match_count > 0 {
            let pair_idxs = raw_to_u32(&pairs_raw);
            let counts = raw_to_u32(&match_counts_raw);
            let match_fi = raw_to_u32(&match_fi_raw);
            let feature_counts = raw_to_u32(&feature_counts_raw);

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
    } else {
        // === Clusters hash (lexicographic path order) ===
        let cluster_count = metadata.cluster_count.unwrap_or(0) as usize;
        let member_count = metadata.cluster_member_count.unwrap_or(0) as usize;
        let mut clusters_hasher = Xxh3::new();

        // clusters/cluster_starts
        let starts_raw = read_zst_entry(
            &mut archive,
            &format!("clusters/cluster_starts.{}.uint32.zst", cluster_count + 1),
        )?;
        clusters_hasher.update(&starts_raw);

        // clusters/member_features
        let member_features_raw = read_zst_entry(
            &mut archive,
            &format!("clusters/member_features.{member_count}.uint32.zst"),
        )?;
        clusters_hasher.update(&member_features_raw);

        // clusters/member_images
        let member_images_raw = read_zst_entry(
            &mut archive,
            &format!("clusters/member_images.{member_count}.uint32.zst"),
        )?;
        clusters_hasher.update(&member_images_raw);

        // clusters/metadata.json
        let clusters_meta_raw = read_zst_entry(&mut archive, "clusters/metadata.json.zst")?;
        clusters_hasher.update(&clusters_meta_raw);

        let clusters_hash = clusters_hasher.digest128();
        match &stored.clusters_xxh128 {
            Some(stored_clusters) => {
                if &format_hash(clusters_hash) != stored_clusters {
                    errors.push(format!(
                        "Clusters hash mismatch: computed {}, stored {}",
                        format_hash(clusters_hash),
                        stored_clusters
                    ));
                }
            }
            None => {
                errors.push("File has clusters but content_hash has no clusters_xxh128".into());
            }
        }
        section_digests.push(clusters_hash);

        // Cross-check clusters section metadata counts
        let clusters_meta: serde_json::Value = serde_json::from_slice(&clusters_meta_raw)?;
        if clusters_meta.get("cluster_count").and_then(|v| v.as_u64()) != Some(cluster_count as u64)
        {
            errors.push(
                "clusters/metadata.json.zst cluster_count doesn't match top-level metadata".into(),
            );
        }
        if clusters_meta.get("member_count").and_then(|v| v.as_u64()) != Some(member_count as u64) {
            errors.push(
                "clusters/metadata.json.zst member_count doesn't match top-level metadata".into(),
            );
        }

        // === Structural validation on raw cluster data ===
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

        let starts = raw_to_u32(&starts_raw);
        let member_images = raw_to_u32(&member_images_raw);
        let member_features = raw_to_u32(&member_features_raw);

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
            let feature_counts = raw_to_u32(&feature_counts_raw);
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

        // === Cluster patches (optional, lexicographic path order) ===
        if metadata.has_cluster_patches {
            let mut cp_hasher = Xxh3::new();

            // cluster_patches/member_affines
            let affines_raw = read_zst_entry(
                &mut archive,
                &format!("cluster_patches/member_affines.{member_count}.2.3.float64.zst"),
            )?;
            cp_hasher.update(&affines_raw);

            // cluster_patches/member_consistency_residual
            let consistency_raw = read_zst_entry(
                &mut archive,
                &format!("cluster_patches/member_consistency_residual.{member_count}.float32.zst"),
            )?;
            cp_hasher.update(&consistency_raw);

            // cluster_patches/member_shift_px
            let shift_raw = read_zst_entry(
                &mut archive,
                &format!("cluster_patches/member_shift_px.{member_count}.float32.zst"),
            )?;
            cp_hasher.update(&shift_raw);

            // cluster_patches/member_status
            let status_raw = read_zst_entry(
                &mut archive,
                &format!("cluster_patches/member_status.{member_count}.uint8.zst"),
            )?;
            cp_hasher.update(&status_raw);

            // cluster_patches/member_zncc
            let zncc_raw = read_zst_entry(
                &mut archive,
                &format!("cluster_patches/member_zncc.{member_count}.float32.zst"),
            )?;
            cp_hasher.update(&zncc_raw);

            // cluster_patches/metadata.json
            let cp_meta_raw = read_zst_entry(&mut archive, "cluster_patches/metadata.json.zst")?;
            cp_hasher.update(&cp_meta_raw);

            // cluster_patches/reference_members
            let refs_raw = read_zst_entry(
                &mut archive,
                &format!("cluster_patches/reference_members.{cluster_count}.uint32.zst"),
            )?;
            cp_hasher.update(&refs_raw);

            let cp_hash = cp_hasher.digest128();
            match &stored.cluster_patches_xxh128 {
                Some(stored_cp) => {
                    if &format_hash(cp_hash) != stored_cp {
                        errors.push(format!(
                            "Cluster patches hash mismatch: computed {}, stored {}",
                            format_hash(cp_hash),
                            stored_cp
                        ));
                    }
                }
                None => {
                    errors.push(
                        "File has cluster_patches but content_hash has no cluster_patches_xxh128"
                            .into(),
                    );
                }
            }
            section_digests.push(cp_hash);

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

            // === Structural validation on raw cluster-patch data ===
            let mut cp_ok = true;
            for (name, raw_len, expected) in [
                ("member_affines", affines_raw.len(), member_count * 48),
                (
                    "member_consistency_residual",
                    consistency_raw.len(),
                    member_count * 4,
                ),
                ("member_shift_px", shift_raw.len(), member_count * 4),
                ("member_status", status_raw.len(), member_count),
                ("member_zncc", zncc_raw.len(), member_count * 4),
                ("reference_members", refs_raw.len(), cluster_count * 4),
            ] {
                if raw_len != expected {
                    errors.push(format!(
                        "{name} byte length {raw_len} != expected {expected}"
                    ));
                    cp_ok = false;
                }
            }

            if cp_ok {
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

            if cp_ok && clusters_ok {
                let refs = raw_to_u32(&refs_raw);
                'clusters: for c in 0..cluster_count {
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
                        // Version 4+: reference rows are identity | x_ref —
                        // check the leading 2×2 exactly (the last column,
                        // the reference keypoint's absolute position, is not
                        // checkable without the `.sift` data). Version ≤ 3
                        // files stored identity | 0 and are already rejected
                        // by the reader; verification leaves them untouched.
                        if metadata.version >= 4 {
                            let affines = raw_to_f64(&affines_raw);
                            let base = reference as usize * 6;
                            let identity = affines[base] == 1.0
                                && affines[base + 1] == 0.0
                                && affines[base + 3] == 0.0
                                && affines[base + 4] == 1.0;
                            if !identity {
                                errors.push(format!(
                                    "member_affines[{reference}] (cluster {c}'s reference row) \
                                     must have an identity leading 2x2 block"
                                ));
                                break;
                            }
                        }
                    }

                    // At most one Reference/Kept member per (cluster, image).
                    let mut covered: std::collections::HashMap<u32, usize> =
                        std::collections::HashMap::new();
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
        }
    }

    // === Two-view geometries (optional) ===
    if metadata.has_two_view_geometries {
        let pair_count = metadata.image_pair_count.unwrap_or(0) as usize;
        let mut tvg_hasher = Xxh3::new();

        // Read TVG metadata for inlier_count
        let tvg_meta_raw = read_zst_entry(&mut archive, "two_view_geometries/metadata.json.zst")?;
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
            &mut archive,
            &format!("two_view_geometries/config_indexes.{pair_count}.uint8.zst"),
        )?);

        // two_view_geometries/config_types.json
        tvg_hasher.update(&read_zst_entry(
            &mut archive,
            "two_view_geometries/config_types.json.zst",
        )?);

        // two_view_geometries/e_matrices
        tvg_hasher.update(&read_zst_entry(
            &mut archive,
            &format!("two_view_geometries/e_matrices.{pair_count}.3.3.float64.zst"),
        )?);

        // two_view_geometries/f_matrices
        tvg_hasher.update(&read_zst_entry(
            &mut archive,
            &format!("two_view_geometries/f_matrices.{pair_count}.3.3.float64.zst"),
        )?);

        // two_view_geometries/h_matrices
        tvg_hasher.update(&read_zst_entry(
            &mut archive,
            &format!("two_view_geometries/h_matrices.{pair_count}.3.3.float64.zst"),
        )?);

        // two_view_geometries/inlier_counts
        let inlier_counts_raw = read_zst_entry(
            &mut archive,
            &format!("two_view_geometries/inlier_counts.{pair_count}.uint32.zst"),
        )?;
        tvg_hasher.update(&inlier_counts_raw);

        // two_view_geometries/inlier_feature_indexes
        tvg_hasher.update(&read_zst_entry(
            &mut archive,
            &format!("two_view_geometries/inlier_feature_indexes.{inlier_count}.2.uint32.zst"),
        )?);

        // two_view_geometries/metadata.json
        tvg_hasher.update(&tvg_meta_raw);

        // two_view_geometries/quaternions_wxyz
        tvg_hasher.update(&read_zst_entry(
            &mut archive,
            &format!("two_view_geometries/quaternions_wxyz.{pair_count}.4.float64.zst"),
        )?);

        // two_view_geometries/translations_xyz
        tvg_hasher.update(&read_zst_entry(
            &mut archive,
            &format!("two_view_geometries/translations_xyz.{pair_count}.3.float64.zst"),
        )?);

        let tvg_hash = tvg_hasher.digest128();
        if let Some(stored_tvg) = &stored.two_view_geometries_xxh128 {
            if &format_hash(tvg_hash) != stored_tvg {
                errors.push(format!(
                    "TVG hash mismatch: computed {}, stored {}",
                    format_hash(tvg_hash),
                    stored_tvg
                ));
            }
        } else {
            errors.push(
                "File has two_view_geometries but content_hash has no two_view_geometries_xxh128"
                    .into(),
            );
        }
        section_digests.push(tvg_hash);

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
