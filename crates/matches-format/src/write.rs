// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.matches` file writing.

use std::collections::HashMap;
use std::path::Path;

use xxhash_rust::xxh3::Xxh3;
use zip::ZipWriter;

use crate::archive_io::{format_hash, write_binary_entry, write_json_entry};
use crate::types::*;

/// Write match data to a `.matches` file.
///
/// Validates dimensions and structural constraints, computes content hashes,
/// and writes all sections. The `content_hash` field in `data` is ignored
/// on write (recomputed).
pub fn write_matches(path: &Path, data: &MatchesData, zstd_level: i32) -> Result<(), MatchesError> {
    let image_count = data.metadata.image_count as usize;
    let pair_count = data.metadata.image_pair_count as usize;
    let match_count = data.metadata.match_count as usize;

    validate_dimensions(data, image_count, pair_count, match_count)?;
    validate_constraints(data, image_count)?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| MatchesError::IoPath {
            operation: "Failed to create parent directory",
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    let file = std::fs::File::create(path).map_err(|e| MatchesError::IoPath {
        operation: "Failed to create file",
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut zip = ZipWriter::new(file);
    let has_tvg = data.two_view_geometries.is_some();
    let mut section_digests: Vec<u128> = Vec::with_capacity(if has_tvg { 4 } else { 3 });

    // === Top-level metadata ===
    let metadata_bytes =
        write_json_entry(&mut zip, "metadata.json.zst", &data.metadata, zstd_level)?;
    let metadata_hash = xxhash_rust::xxh3::xxh3_128(&metadata_bytes);
    section_digests.push(metadata_hash);

    // === Images (hashed in lexicographic path order) ===
    let mut images_hasher = Xxh3::new();

    // images/feature_counts
    let bytes = write_binary_entry(
        &mut zip,
        &format!("images/feature_counts.{image_count}.uint32.zst"),
        bytemuck::cast_slice(data.feature_counts.as_slice().unwrap()),
        zstd_level,
    )?;
    images_hasher.update(&bytes);

    // images/feature_tool_hashes
    let hash_bytes: Vec<u8> = data
        .feature_tool_hashes
        .iter()
        .flat_map(|h| h.iter().copied())
        .collect();
    let bytes = write_binary_entry(
        &mut zip,
        &format!("images/feature_tool_hashes.{image_count}.uint128.zst"),
        &hash_bytes,
        zstd_level,
    )?;
    images_hasher.update(&bytes);

    // images/metadata.json
    let images_meta = serde_json::json!({"image_count": image_count});
    let bytes = write_json_entry(
        &mut zip,
        "images/metadata.json.zst",
        &images_meta,
        zstd_level,
    )?;
    images_hasher.update(&bytes);

    // images/names.json
    let bytes = write_json_entry(
        &mut zip,
        "images/names.json.zst",
        &data.image_names,
        zstd_level,
    )?;
    images_hasher.update(&bytes);

    // images/sift_content_hashes
    let hash_bytes: Vec<u8> = data
        .sift_content_hashes
        .iter()
        .flat_map(|h| h.iter().copied())
        .collect();
    let bytes = write_binary_entry(
        &mut zip,
        &format!("images/sift_content_hashes.{image_count}.uint128.zst"),
        &hash_bytes,
        zstd_level,
    )?;
    images_hasher.update(&bytes);

    let images_hash = images_hasher.digest128();
    section_digests.push(images_hash);

    // === Image pairs (hashed in lexicographic path order) ===
    let mut pairs_hasher = Xxh3::new();

    // image_pairs/image_index_pairs
    let bytes = write_binary_entry(
        &mut zip,
        &format!("image_pairs/image_index_pairs.{pair_count}.2.uint32.zst"),
        bytemuck::cast_slice(data.image_index_pairs.as_slice().unwrap()),
        zstd_level,
    )?;
    pairs_hasher.update(&bytes);

    // image_pairs/match_counts
    let bytes = write_binary_entry(
        &mut zip,
        &format!("image_pairs/match_counts.{pair_count}.uint32.zst"),
        bytemuck::cast_slice(data.match_counts.as_slice().unwrap()),
        zstd_level,
    )?;
    pairs_hasher.update(&bytes);

    // image_pairs/match_descriptor_distances
    let bytes = write_binary_entry(
        &mut zip,
        &format!("image_pairs/match_descriptor_distances.{match_count}.float32.zst"),
        bytemuck::cast_slice(data.match_descriptor_distances.as_slice().unwrap()),
        zstd_level,
    )?;
    pairs_hasher.update(&bytes);

    // image_pairs/match_feature_indexes
    let bytes = write_binary_entry(
        &mut zip,
        &format!("image_pairs/match_feature_indexes.{match_count}.2.uint32.zst"),
        bytemuck::cast_slice(data.match_feature_indexes.as_slice().unwrap()),
        zstd_level,
    )?;
    pairs_hasher.update(&bytes);

    // image_pairs/metadata.json
    let pairs_meta =
        serde_json::json!({"image_pair_count": pair_count, "match_count": match_count});
    let bytes = write_json_entry(
        &mut zip,
        "image_pairs/metadata.json.zst",
        &pairs_meta,
        zstd_level,
    )?;
    pairs_hasher.update(&bytes);

    let pairs_hash = pairs_hasher.digest128();
    section_digests.push(pairs_hash);

    // === Two-view geometries (optional, hashed in lexicographic path order) ===
    let tvg_hash: Option<u128> = if let Some(tvg) = &data.two_view_geometries {
        let inlier_count = tvg.metadata.inlier_count as usize;
        let mut tvg_hasher = Xxh3::new();

        // two_view_geometries/config_indexes
        let bytes = write_binary_entry(
            &mut zip,
            &format!("two_view_geometries/config_indexes.{pair_count}.uint8.zst"),
            tvg.config_indexes.as_slice().unwrap(),
            zstd_level,
        )?;
        tvg_hasher.update(&bytes);

        // two_view_geometries/config_types.json
        let config_type_strings: Vec<&str> = tvg.config_types.iter().map(|c| c.as_str()).collect();
        let bytes = write_json_entry(
            &mut zip,
            "two_view_geometries/config_types.json.zst",
            &config_type_strings,
            zstd_level,
        )?;
        tvg_hasher.update(&bytes);

        // two_view_geometries/e_matrices
        let bytes = write_binary_entry(
            &mut zip,
            &format!("two_view_geometries/e_matrices.{pair_count}.3.3.float64.zst"),
            bytemuck::cast_slice(tvg.e_matrices.as_slice().unwrap()),
            zstd_level,
        )?;
        tvg_hasher.update(&bytes);

        // two_view_geometries/f_matrices
        let bytes = write_binary_entry(
            &mut zip,
            &format!("two_view_geometries/f_matrices.{pair_count}.3.3.float64.zst"),
            bytemuck::cast_slice(tvg.f_matrices.as_slice().unwrap()),
            zstd_level,
        )?;
        tvg_hasher.update(&bytes);

        // two_view_geometries/h_matrices
        let bytes = write_binary_entry(
            &mut zip,
            &format!("two_view_geometries/h_matrices.{pair_count}.3.3.float64.zst"),
            bytemuck::cast_slice(tvg.h_matrices.as_slice().unwrap()),
            zstd_level,
        )?;
        tvg_hasher.update(&bytes);

        // two_view_geometries/inlier_counts
        let bytes = write_binary_entry(
            &mut zip,
            &format!("two_view_geometries/inlier_counts.{pair_count}.uint32.zst"),
            bytemuck::cast_slice(tvg.inlier_counts.as_slice().unwrap()),
            zstd_level,
        )?;
        tvg_hasher.update(&bytes);

        // two_view_geometries/inlier_feature_indexes
        let bytes = write_binary_entry(
            &mut zip,
            &format!("two_view_geometries/inlier_feature_indexes.{inlier_count}.2.uint32.zst"),
            bytemuck::cast_slice(tvg.inlier_feature_indexes.as_slice().unwrap()),
            zstd_level,
        )?;
        tvg_hasher.update(&bytes);

        // two_view_geometries/metadata.json
        let bytes = write_json_entry(
            &mut zip,
            "two_view_geometries/metadata.json.zst",
            &tvg.metadata,
            zstd_level,
        )?;
        tvg_hasher.update(&bytes);

        // two_view_geometries/quaternions_wxyz
        let bytes = write_binary_entry(
            &mut zip,
            &format!("two_view_geometries/quaternions_wxyz.{pair_count}.4.float64.zst"),
            bytemuck::cast_slice(tvg.quaternions_wxyz.as_slice().unwrap()),
            zstd_level,
        )?;
        tvg_hasher.update(&bytes);

        // two_view_geometries/translations_xyz
        let bytes = write_binary_entry(
            &mut zip,
            &format!("two_view_geometries/translations_xyz.{pair_count}.3.float64.zst"),
            bytemuck::cast_slice(tvg.translations_xyz.as_slice().unwrap()),
            zstd_level,
        )?;
        tvg_hasher.update(&bytes);

        let digest = tvg_hasher.digest128();
        section_digests.push(digest);
        Some(digest)
    } else {
        None
    };

    // === Content hash ===
    let all_digests_bytes: Vec<u8> = section_digests
        .iter()
        .flat_map(|d| d.to_be_bytes())
        .collect();
    let content_hash_value = xxhash_rust::xxh3::xxh3_128(&all_digests_bytes);

    let content_hash = MatchesContentHash {
        metadata_xxh128: format_hash(metadata_hash),
        images_xxh128: format_hash(images_hash),
        image_pairs_xxh128: format_hash(pairs_hash),
        two_view_geometries_xxh128: tvg_hash.map(format_hash),
        content_xxh128: format_hash(content_hash_value),
    };
    write_json_entry(&mut zip, "content_hash.json.zst", &content_hash, zstd_level)?;

    zip.finish()?;
    Ok(())
}

fn validate_dimensions(
    data: &MatchesData,
    image_count: usize,
    pair_count: usize,
    match_count: usize,
) -> Result<(), MatchesError> {
    macro_rules! check {
        ($cond:expr, $msg:expr) => {
            if !($cond) {
                return Err(MatchesError::ShapeMismatch($msg.into()));
            }
        };
    }

    // Images
    check!(
        data.image_names.len() == image_count,
        format!(
            "image_names count {} != image_count {image_count}",
            data.image_names.len()
        )
    );
    check!(
        data.feature_tool_hashes.len() == image_count,
        format!(
            "feature_tool_hashes len {} != image_count {image_count}",
            data.feature_tool_hashes.len()
        )
    );
    check!(
        data.sift_content_hashes.len() == image_count,
        format!(
            "sift_content_hashes len {} != image_count {image_count}",
            data.sift_content_hashes.len()
        )
    );
    check!(
        data.feature_counts.len() == image_count,
        format!(
            "feature_counts len {} != image_count {image_count}",
            data.feature_counts.len()
        )
    );

    // Image pairs
    check!(
        data.image_index_pairs.shape() == [pair_count, 2],
        format!(
            "image_index_pairs shape {:?} != [{pair_count}, 2]",
            data.image_index_pairs.shape()
        )
    );
    check!(
        data.match_counts.len() == pair_count,
        format!(
            "match_counts len {} != pair_count {pair_count}",
            data.match_counts.len()
        )
    );
    check!(
        data.match_feature_indexes.shape() == [match_count, 2],
        format!(
            "match_feature_indexes shape {:?} != [{match_count}, 2]",
            data.match_feature_indexes.shape()
        )
    );
    check!(
        data.match_descriptor_distances.len() == match_count,
        format!(
            "match_descriptor_distances len {} != match_count {match_count}",
            data.match_descriptor_distances.len()
        )
    );

    // Match counts sum
    let match_sum: u64 = data.match_counts.iter().map(|&c| c as u64).sum();
    check!(
        match_sum == match_count as u64,
        format!("sum of match_counts ({match_sum}) != match_count ({match_count})")
    );

    // All match_counts >= 1
    if data.match_counts.iter().any(|&c| c < 1) {
        return Err(MatchesError::ShapeMismatch(
            "match_counts contains values < 1 (pairs with zero matches should not be stored)"
                .into(),
        ));
    }

    // TVG dimensions
    if let Some(tvg) = &data.two_view_geometries {
        let inlier_count = tvg.metadata.inlier_count as usize;

        check!(
            tvg.metadata.image_pair_count as usize == pair_count,
            format!(
                "TVG image_pair_count {} != pair_count {pair_count}",
                tvg.metadata.image_pair_count
            )
        );
        check!(
            tvg.config_indexes.len() == pair_count,
            format!(
                "config_indexes len {} != pair_count {pair_count}",
                tvg.config_indexes.len()
            )
        );
        check!(
            tvg.inlier_counts.len() == pair_count,
            format!(
                "inlier_counts len {} != pair_count {pair_count}",
                tvg.inlier_counts.len()
            )
        );
        check!(
            tvg.inlier_feature_indexes.shape() == [inlier_count, 2],
            format!(
                "inlier_feature_indexes shape {:?} != [{inlier_count}, 2]",
                tvg.inlier_feature_indexes.shape()
            )
        );
        check!(
            tvg.f_matrices.shape() == [pair_count, 3, 3],
            format!(
                "f_matrices shape {:?} != [{pair_count}, 3, 3]",
                tvg.f_matrices.shape()
            )
        );
        check!(
            tvg.e_matrices.shape() == [pair_count, 3, 3],
            format!(
                "e_matrices shape {:?} != [{pair_count}, 3, 3]",
                tvg.e_matrices.shape()
            )
        );
        check!(
            tvg.h_matrices.shape() == [pair_count, 3, 3],
            format!(
                "h_matrices shape {:?} != [{pair_count}, 3, 3]",
                tvg.h_matrices.shape()
            )
        );
        check!(
            tvg.quaternions_wxyz.shape() == [pair_count, 4],
            format!(
                "quaternions_wxyz shape {:?} != [{pair_count}, 4]",
                tvg.quaternions_wxyz.shape()
            )
        );
        check!(
            tvg.translations_xyz.shape() == [pair_count, 3],
            format!(
                "translations_xyz shape {:?} != [{pair_count}, 3]",
                tvg.translations_xyz.shape()
            )
        );

        // Inlier counts sum
        let inlier_sum: u64 = tvg.inlier_counts.iter().map(|&c| c as u64).sum();
        check!(
            inlier_sum == inlier_count as u64,
            format!("sum of inlier_counts ({inlier_sum}) != inlier_count ({inlier_count})")
        );

        // Config indexes valid
        let num_types = tvg.config_types.len();
        for (k, &idx) in tvg.config_indexes.iter().enumerate() {
            if idx as usize >= num_types {
                return Err(MatchesError::ShapeMismatch(format!(
                    "config_indexes[{k}] = {idx} >= config_types length {num_types}"
                )));
            }
        }
    }

    Ok(())
}

fn validate_constraints(data: &MatchesData, image_count: usize) -> Result<(), MatchesError> {
    let pair_count = data.metadata.image_pair_count as usize;

    // Pairs sorted with idx_i < idx_j
    for k in 0..pair_count {
        let idx_i = data.image_index_pairs[[k, 0]];
        let idx_j = data.image_index_pairs[[k, 1]];
        if idx_i >= idx_j {
            return Err(MatchesError::InvalidFormat(format!(
                "image_index_pairs[{k}] = ({idx_i}, {idx_j}): idx_i must be < idx_j"
            )));
        }
        if idx_i as usize >= image_count || idx_j as usize >= image_count {
            return Err(MatchesError::InvalidFormat(format!(
                "image_index_pairs[{k}] = ({idx_i}, {idx_j}): index out of bounds (image_count = {image_count})"
            )));
        }
        if k > 0 {
            let prev_i = data.image_index_pairs[[k - 1, 0]];
            let prev_j = data.image_index_pairs[[k - 1, 1]];
            if (idx_i, idx_j) <= (prev_i, prev_j) {
                return Err(MatchesError::InvalidFormat(format!(
                    "image_index_pairs not sorted at index {k}: ({prev_i}, {prev_j}) >= ({idx_i}, {idx_j})"
                )));
            }
        }
    }

    // Feature index bounds
    let mut offset: usize = 0;
    for k in 0..pair_count {
        let idx_i = data.image_index_pairs[[k, 0]] as usize;
        let idx_j = data.image_index_pairs[[k, 1]] as usize;
        let count = data.match_counts[k] as usize;
        let fc_i = data.feature_counts[idx_i];
        let fc_j = data.feature_counts[idx_j];
        for m in offset..offset + count {
            let fi = data.match_feature_indexes[[m, 0]];
            let fj = data.match_feature_indexes[[m, 1]];
            if fi >= fc_i {
                return Err(MatchesError::InvalidFormat(format!(
                    "match_feature_indexes[{m}][0] = {fi} >= feature_counts[{idx_i}] = {fc_i}"
                )));
            }
            if fj >= fc_j {
                return Err(MatchesError::InvalidFormat(format!(
                    "match_feature_indexes[{m}][1] = {fj} >= feature_counts[{idx_j}] = {fc_j}"
                )));
            }
        }
        offset += count;
    }

    // Validate inlier subset constraint
    if let Some(tvg) = &data.two_view_geometries {
        let mut match_offset: usize = 0;
        let mut inlier_offset: usize = 0;
        for k in 0..pair_count {
            let mc = data.match_counts[k] as usize;
            let ic = tvg.inlier_counts[k] as usize;

            // Build set of candidate matches for this pair
            let mut candidates: HashMap<(u32, u32), ()> = HashMap::with_capacity(mc);
            for m in match_offset..match_offset + mc {
                let fi = data.match_feature_indexes[[m, 0]];
                let fj = data.match_feature_indexes[[m, 1]];
                candidates.insert((fi, fj), ());
            }

            // Check each inlier is in the candidate set
            for i in inlier_offset..inlier_offset + ic {
                let fi = tvg.inlier_feature_indexes[[i, 0]];
                let fj = tvg.inlier_feature_indexes[[i, 1]];
                if !candidates.contains_key(&(fi, fj)) {
                    return Err(MatchesError::InvalidFormat(format!(
                        "inlier ({fi}, {fj}) at index {i} for pair {k} is not in candidate matches"
                    )));
                }
            }

            match_offset += mc;
            inlier_offset += ic;
        }
    }

    Ok(())
}
