// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.matches` file integrity verification.

use std::path::Path;

use xxhash_rust::xxh3::Xxh3;

use crate::archive_io::{format_hash, read_zst_entry};
use crate::types::*;

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
    let pair_count = metadata.image_pair_count as usize;
    let match_count = metadata.match_count as usize;

    let mut section_digests: Vec<u128> = Vec::with_capacity(4);

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

    // === Image pairs hash (lexicographic path order) ===
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
    if format_hash(pairs_hash) != stored.image_pairs_xxh128 {
        errors.push(format!(
            "Image pairs hash mismatch: computed {}, stored {}",
            format_hash(pairs_hash),
            stored.image_pairs_xxh128
        ));
    }
    section_digests.push(pairs_hash);

    // === Structural validation on raw data ===

    // Validate pair sorting (idx_i < idx_j, lexicographic order)
    if pair_count > 0 {
        let pair_idxs: &[u32] = bytemuck::cast_slice(&pairs_raw);
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
        let counts: &[u32] = bytemuck::cast_slice(&match_counts_raw);
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
        let pair_idxs: &[u32] = bytemuck::cast_slice(&pairs_raw);
        let counts: &[u32] = bytemuck::cast_slice(&match_counts_raw);
        let match_fi: &[u32] = bytemuck::cast_slice(&match_fi_raw);
        let feature_counts: &[u32] = bytemuck::cast_slice(&feature_counts_raw);

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

    // === Two-view geometries (optional) ===
    if metadata.has_two_view_geometries {
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
            let inlier_counts: &[u32] = bytemuck::cast_slice(&inlier_counts_raw);
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