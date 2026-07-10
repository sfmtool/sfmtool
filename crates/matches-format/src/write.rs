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
/// and writes all sections. Always writes the current format version
/// ([`MATCHES_FORMAT_VERSION`]); the caller must supply two-view relative
/// poses in the canonical camera convention. The `content_hash` field in
/// `data` is ignored on write (recomputed).
///
/// Exactly one of `image_pairs` / `clusters` must be present (the
/// correspondence backbone); `cluster_patches` requires `clusters`, and
/// `two_view_geometries` requires `image_pairs`. The metadata `has_*` flags
/// and summary counts must be consistent with the supplied sections.
pub fn write_matches(path: &Path, data: &MatchesData, zstd_level: i32) -> Result<(), MatchesError> {
    let image_count = data.metadata.image_count as usize;

    validate_structure(data)?;
    validate_dimensions(data, image_count)?;
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
    // Present-section digests, accumulated in the canonical order: metadata,
    // images, pairs, clusters, cluster_patches, two_view_geometries.
    let mut section_digests: Vec<u128> = Vec::with_capacity(6);

    // === Top-level metadata (always emitted at the current format version) ===
    let mut metadata = data.metadata.clone();
    metadata.version = MATCHES_FORMAT_VERSION;
    let metadata_bytes = write_json_entry(&mut zip, "metadata.json.zst", &metadata, zstd_level)?;
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

    // === Image pairs (backbone alternative, hashed in lexicographic path order) ===
    let pairs_hash: Option<u128> = if let Some(pairs) = &data.image_pairs {
        let pair_count = pairs.image_index_pairs.nrows();
        let match_count = pairs.match_feature_indexes.nrows();
        let mut pairs_hasher = Xxh3::new();

        // image_pairs/image_index_pairs
        let bytes = write_binary_entry(
            &mut zip,
            &format!("image_pairs/image_index_pairs.{pair_count}.2.uint32.zst"),
            bytemuck::cast_slice(pairs.image_index_pairs.as_slice().unwrap()),
            zstd_level,
        )?;
        pairs_hasher.update(&bytes);

        // image_pairs/match_counts
        let bytes = write_binary_entry(
            &mut zip,
            &format!("image_pairs/match_counts.{pair_count}.uint32.zst"),
            bytemuck::cast_slice(pairs.match_counts.as_slice().unwrap()),
            zstd_level,
        )?;
        pairs_hasher.update(&bytes);

        // image_pairs/match_descriptor_distances
        let bytes = write_binary_entry(
            &mut zip,
            &format!("image_pairs/match_descriptor_distances.{match_count}.float32.zst"),
            bytemuck::cast_slice(pairs.match_descriptor_distances.as_slice().unwrap()),
            zstd_level,
        )?;
        pairs_hasher.update(&bytes);

        // image_pairs/match_feature_indexes
        let bytes = write_binary_entry(
            &mut zip,
            &format!("image_pairs/match_feature_indexes.{match_count}.2.uint32.zst"),
            bytemuck::cast_slice(pairs.match_feature_indexes.as_slice().unwrap()),
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

        let digest = pairs_hasher.digest128();
        section_digests.push(digest);
        Some(digest)
    } else {
        None
    };

    // === Clusters (backbone alternative, hashed in lexicographic path order) ===
    let clusters_hash: Option<u128> = if let Some(clusters) = &data.clusters {
        let cluster_count = clusters.cluster_starts.len() - 1;
        let member_count = clusters.member_images.len();
        let mut clusters_hasher = Xxh3::new();

        // clusters/cluster_starts
        let bytes = write_binary_entry(
            &mut zip,
            &format!("clusters/cluster_starts.{}.uint32.zst", cluster_count + 1),
            bytemuck::cast_slice(clusters.cluster_starts.as_slice().unwrap()),
            zstd_level,
        )?;
        clusters_hasher.update(&bytes);

        // clusters/member_features
        let bytes = write_binary_entry(
            &mut zip,
            &format!("clusters/member_features.{member_count}.uint32.zst"),
            bytemuck::cast_slice(clusters.member_features.as_slice().unwrap()),
            zstd_level,
        )?;
        clusters_hasher.update(&bytes);

        // clusters/member_images
        let bytes = write_binary_entry(
            &mut zip,
            &format!("clusters/member_images.{member_count}.uint32.zst"),
            bytemuck::cast_slice(clusters.member_images.as_slice().unwrap()),
            zstd_level,
        )?;
        clusters_hasher.update(&bytes);

        // clusters/metadata.json
        let clusters_meta = serde_json::json!({
            "cluster_count": cluster_count,
            "member_count": member_count,
            "matcher_options": clusters.matcher_options,
        });
        let bytes = write_json_entry(
            &mut zip,
            "clusters/metadata.json.zst",
            &clusters_meta,
            zstd_level,
        )?;
        clusters_hasher.update(&bytes);

        let digest = clusters_hasher.digest128();
        section_digests.push(digest);
        Some(digest)
    } else {
        None
    };

    // === Cluster patches (optional, hashed in lexicographic path order) ===
    let cluster_patches_hash: Option<u128> = if let Some(cp) = &data.cluster_patches {
        let cluster_count = cp.reference_members.len();
        let member_count = cp.member_status.len();
        let mut cp_hasher = Xxh3::new();

        // cluster_patches/member_affines
        let bytes = write_binary_entry(
            &mut zip,
            &format!("cluster_patches/member_affines.{member_count}.2.3.float64.zst"),
            bytemuck::cast_slice(cp.member_affines.as_slice().unwrap()),
            zstd_level,
        )?;
        cp_hasher.update(&bytes);

        // cluster_patches/member_shift_px
        let bytes = write_binary_entry(
            &mut zip,
            &format!("cluster_patches/member_shift_px.{member_count}.float32.zst"),
            bytemuck::cast_slice(cp.member_shift_px.as_slice().unwrap()),
            zstd_level,
        )?;
        cp_hasher.update(&bytes);

        // cluster_patches/member_status
        let bytes = write_binary_entry(
            &mut zip,
            &format!("cluster_patches/member_status.{member_count}.uint8.zst"),
            cp.member_status.as_slice().unwrap(),
            zstd_level,
        )?;
        cp_hasher.update(&bytes);

        // cluster_patches/member_zncc
        let bytes = write_binary_entry(
            &mut zip,
            &format!("cluster_patches/member_zncc.{member_count}.float32.zst"),
            bytemuck::cast_slice(cp.member_zncc.as_slice().unwrap()),
            zstd_level,
        )?;
        cp_hasher.update(&bytes);

        // cluster_patches/metadata.json
        let cp_meta = serde_json::json!({
            "cluster_count": cluster_count,
            "member_count": member_count,
            "refine_options": cp.refine_options,
        });
        let bytes = write_json_entry(
            &mut zip,
            "cluster_patches/metadata.json.zst",
            &cp_meta,
            zstd_level,
        )?;
        cp_hasher.update(&bytes);

        // cluster_patches/reference_members
        let bytes = write_binary_entry(
            &mut zip,
            &format!("cluster_patches/reference_members.{cluster_count}.uint32.zst"),
            bytemuck::cast_slice(cp.reference_members.as_slice().unwrap()),
            zstd_level,
        )?;
        cp_hasher.update(&bytes);

        let digest = cp_hasher.digest128();
        section_digests.push(digest);
        Some(digest)
    } else {
        None
    };

    // === Two-view geometries (optional, hashed in lexicographic path order) ===
    let tvg_hash: Option<u128> = if let Some(tvg) = &data.two_view_geometries {
        let pair_count = tvg.config_indexes.len();
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
        image_pairs_xxh128: pairs_hash.map(format_hash),
        clusters_xxh128: clusters_hash.map(format_hash),
        cluster_patches_xxh128: cluster_patches_hash.map(format_hash),
        two_view_geometries_xxh128: tvg_hash.map(format_hash),
        content_xxh128: format_hash(content_hash_value),
    };
    write_json_entry(&mut zip, "content_hash.json.zst", &content_hash, zstd_level)?;

    zip.finish()?;
    Ok(())
}

/// Validate the backbone rule, section dependencies, and metadata flag /
/// summary-count consistency.
fn validate_structure(data: &MatchesData) -> Result<(), MatchesError> {
    macro_rules! invalid {
        ($cond:expr, $msg:expr) => {
            if $cond {
                return Err(MatchesError::InvalidFormat($msg.into()));
            }
        };
    }

    let has_pairs = data.image_pairs.is_some();
    let has_clusters = data.clusters.is_some();

    invalid!(
        has_pairs && has_clusters,
        "exactly one of image_pairs / clusters must be present (both supplied)"
    );
    invalid!(
        !has_pairs && !has_clusters,
        "exactly one of image_pairs / clusters must be present (neither supplied)"
    );
    invalid!(
        data.cluster_patches.is_some() && !has_clusters,
        "cluster_patches requires the clusters section"
    );
    invalid!(
        data.two_view_geometries.is_some() && !has_pairs,
        "two_view_geometries requires the image_pairs section"
    );

    // Metadata flags must agree with the supplied sections.
    invalid!(
        data.metadata.has_clusters != has_clusters,
        format!(
            "metadata.has_clusters ({}) doesn't match clusters presence ({has_clusters})",
            data.metadata.has_clusters
        )
    );
    invalid!(
        data.metadata.has_cluster_patches != data.cluster_patches.is_some(),
        format!(
            "metadata.has_cluster_patches ({}) doesn't match cluster_patches presence ({})",
            data.metadata.has_cluster_patches,
            data.cluster_patches.is_some()
        )
    );
    invalid!(
        data.metadata.has_two_view_geometries != data.two_view_geometries.is_some(),
        format!(
            "metadata.has_two_view_geometries ({}) doesn't match two_view_geometries presence ({})",
            data.metadata.has_two_view_geometries,
            data.two_view_geometries.is_some()
        )
    );

    // Backbone-specific summary counts: pairwise files carry
    // image_pair_count / match_count, cluster files carry cluster_count /
    // cluster_member_count — never both.
    if has_pairs {
        invalid!(
            data.metadata.image_pair_count.is_none() || data.metadata.match_count.is_none(),
            "pairwise file requires metadata.image_pair_count and metadata.match_count"
        );
        invalid!(
            data.metadata.cluster_count.is_some() || data.metadata.cluster_member_count.is_some(),
            "pairwise file must not set metadata.cluster_count / cluster_member_count"
        );
    } else {
        invalid!(
            data.metadata.cluster_count.is_none() || data.metadata.cluster_member_count.is_none(),
            "cluster-bearing file requires metadata.cluster_count and metadata.cluster_member_count"
        );
        invalid!(
            data.metadata.image_pair_count.is_some() || data.metadata.match_count.is_some(),
            "cluster-bearing file must not set metadata.image_pair_count / match_count"
        );
    }

    Ok(())
}

fn validate_dimensions(data: &MatchesData, image_count: usize) -> Result<(), MatchesError> {
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
    if let Some(pairs) = &data.image_pairs {
        let pair_count = data.metadata.image_pair_count.unwrap_or(0) as usize;
        let match_count = data.metadata.match_count.unwrap_or(0) as usize;

        check!(
            pairs.image_index_pairs.shape() == [pair_count, 2],
            format!(
                "image_index_pairs shape {:?} != [{pair_count}, 2]",
                pairs.image_index_pairs.shape()
            )
        );
        check!(
            pairs.match_counts.len() == pair_count,
            format!(
                "match_counts len {} != pair_count {pair_count}",
                pairs.match_counts.len()
            )
        );
        check!(
            pairs.match_feature_indexes.shape() == [match_count, 2],
            format!(
                "match_feature_indexes shape {:?} != [{match_count}, 2]",
                pairs.match_feature_indexes.shape()
            )
        );
        check!(
            pairs.match_descriptor_distances.len() == match_count,
            format!(
                "match_descriptor_distances len {} != match_count {match_count}",
                pairs.match_descriptor_distances.len()
            )
        );

        // Match counts sum
        let match_sum: u64 = pairs.match_counts.iter().map(|&c| c as u64).sum();
        check!(
            match_sum == match_count as u64,
            format!("sum of match_counts ({match_sum}) != match_count ({match_count})")
        );

        // All match_counts >= 1
        if pairs.match_counts.iter().any(|&c| c < 1) {
            return Err(MatchesError::ShapeMismatch(
                "match_counts contains values < 1 (pairs with zero matches should not be stored)"
                    .into(),
            ));
        }
    }

    // Clusters
    if let Some(clusters) = &data.clusters {
        let cluster_count = data.metadata.cluster_count.unwrap_or(0) as usize;
        let member_count = data.metadata.cluster_member_count.unwrap_or(0) as usize;

        check!(
            clusters.cluster_starts.len() == cluster_count + 1,
            format!(
                "cluster_starts len {} != cluster_count + 1 ({})",
                clusters.cluster_starts.len(),
                cluster_count + 1
            )
        );
        check!(
            clusters.member_images.len() == member_count,
            format!(
                "member_images len {} != cluster_member_count {member_count}",
                clusters.member_images.len()
            )
        );
        check!(
            clusters.member_features.len() == member_count,
            format!(
                "member_features len {} != cluster_member_count {member_count}",
                clusters.member_features.len()
            )
        );
    }

    // Cluster patches
    if let Some(cp) = &data.cluster_patches {
        let cluster_count = data.metadata.cluster_count.unwrap_or(0) as usize;
        let member_count = data.metadata.cluster_member_count.unwrap_or(0) as usize;

        check!(
            cp.reference_members.len() == cluster_count,
            format!(
                "reference_members len {} != cluster_count {cluster_count}",
                cp.reference_members.len()
            )
        );
        check!(
            cp.member_status.len() == member_count,
            format!(
                "member_status len {} != cluster_member_count {member_count}",
                cp.member_status.len()
            )
        );
        check!(
            cp.member_affines.shape() == [member_count, 2, 3],
            format!(
                "member_affines shape {:?} != [{member_count}, 2, 3]",
                cp.member_affines.shape()
            )
        );
        check!(
            cp.member_zncc.len() == member_count,
            format!(
                "member_zncc len {} != cluster_member_count {member_count}",
                cp.member_zncc.len()
            )
        );
        check!(
            cp.member_shift_px.len() == member_count,
            format!(
                "member_shift_px len {} != cluster_member_count {member_count}",
                cp.member_shift_px.len()
            )
        );
    }

    // TVG dimensions
    if let Some(tvg) = &data.two_view_geometries {
        let pair_count = data.metadata.image_pair_count.unwrap_or(0) as usize;
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
    if let Some(pairs) = &data.image_pairs {
        validate_pairs_constraints(data, pairs, image_count)?;
    }
    if let Some(clusters) = &data.clusters {
        validate_clusters_constraints(data, clusters, image_count)?;
        if let Some(cp) = &data.cluster_patches {
            validate_cluster_patches_constraints(clusters, cp)?;
        }
    }
    Ok(())
}

fn validate_pairs_constraints(
    data: &MatchesData,
    pairs: &PairsData,
    image_count: usize,
) -> Result<(), MatchesError> {
    let pair_count = pairs.image_index_pairs.nrows();

    // Pairs sorted with idx_i < idx_j
    for k in 0..pair_count {
        let idx_i = pairs.image_index_pairs[[k, 0]];
        let idx_j = pairs.image_index_pairs[[k, 1]];
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
            let prev_i = pairs.image_index_pairs[[k - 1, 0]];
            let prev_j = pairs.image_index_pairs[[k - 1, 1]];
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
        let idx_i = pairs.image_index_pairs[[k, 0]] as usize;
        let idx_j = pairs.image_index_pairs[[k, 1]] as usize;
        let count = pairs.match_counts[k] as usize;
        let fc_i = data.feature_counts[idx_i];
        let fc_j = data.feature_counts[idx_j];
        for m in offset..offset + count {
            let fi = pairs.match_feature_indexes[[m, 0]];
            let fj = pairs.match_feature_indexes[[m, 1]];
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
            let mc = pairs.match_counts[k] as usize;
            let ic = tvg.inlier_counts[k] as usize;

            // Build set of candidate matches for this pair
            let mut candidates: HashMap<(u32, u32), ()> = HashMap::with_capacity(mc);
            for m in match_offset..match_offset + mc {
                let fi = pairs.match_feature_indexes[[m, 0]];
                let fj = pairs.match_feature_indexes[[m, 1]];
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

fn validate_clusters_constraints(
    data: &MatchesData,
    clusters: &ClustersData,
    image_count: usize,
) -> Result<(), MatchesError> {
    let cluster_count = clusters.cluster_starts.len() - 1;
    let member_count = clusters.member_images.len();

    if clusters.cluster_starts[0] != 0 {
        return Err(MatchesError::InvalidFormat(format!(
            "cluster_starts[0] = {} != 0",
            clusters.cluster_starts[0]
        )));
    }
    for c in 0..cluster_count {
        let start = clusters.cluster_starts[c];
        let end = clusters.cluster_starts[c + 1];
        if end < start {
            return Err(MatchesError::InvalidFormat(format!(
                "cluster_starts not non-decreasing at cluster {c}: {start} > {end}"
            )));
        }
        if end - start < 2 {
            return Err(MatchesError::InvalidFormat(format!(
                "cluster {c} has {} members; every cluster must have >= 2",
                end - start
            )));
        }
    }
    if clusters.cluster_starts[cluster_count] as usize != member_count {
        return Err(MatchesError::InvalidFormat(format!(
            "cluster_starts final value {} != member count {member_count}",
            clusters.cluster_starts[cluster_count]
        )));
    }

    for k in 0..member_count {
        let img = clusters.member_images[k];
        if img as usize >= image_count {
            return Err(MatchesError::InvalidFormat(format!(
                "member_images[{k}] = {img} >= image_count {image_count}"
            )));
        }
        let feat = clusters.member_features[k];
        let fc = data.feature_counts[img as usize];
        if feat >= fc {
            return Err(MatchesError::InvalidFormat(format!(
                "member_features[{k}] = {feat} >= feature_counts[{img}] = {fc}"
            )));
        }
    }

    Ok(())
}

fn validate_cluster_patches_constraints(
    clusters: &ClustersData,
    cp: &ClusterPatchData,
) -> Result<(), MatchesError> {
    let cluster_count = clusters.cluster_starts.len() - 1;

    for (k, &status) in cp.member_status.iter().enumerate() {
        if ClusterMemberStatus::from_u8(status).is_none() {
            return Err(MatchesError::InvalidFormat(format!(
                "member_status[{k}] = {status} is not a valid ClusterMemberStatus discriminant"
            )));
        }
    }

    for c in 0..cluster_count {
        let start = clusters.cluster_starts[c];
        let end = clusters.cluster_starts[c + 1];

        let reference = cp.reference_members[c];
        if reference != CLUSTER_REFERENCE_UNREFINABLE {
            if reference < start || reference >= end {
                return Err(MatchesError::InvalidFormat(format!(
                    "reference_members[{c}] = {reference} outside cluster member range [{start}, {end})"
                )));
            }
            if cp.member_status[reference as usize] != ClusterMemberStatus::Reference as u8 {
                return Err(MatchesError::InvalidFormat(format!(
                    "reference_members[{c}] = {reference} has status {}, expected {} (reference)",
                    cp.member_status[reference as usize],
                    ClusterMemberStatus::Reference as u8
                )));
            }
        }

        // At most one Reference/Kept member per (cluster, image).
        let mut covered_images: HashMap<u32, usize> = HashMap::new();
        for m in start as usize..end as usize {
            let status = cp.member_status[m];
            if status == ClusterMemberStatus::Reference as u8
                || status == ClusterMemberStatus::Kept as u8
            {
                let img = clusters.member_images[m];
                if let Some(prev) = covered_images.insert(img, m) {
                    return Err(MatchesError::InvalidFormat(format!(
                        "cluster {c}: members {prev} and {m} are both reference/kept for image {img}"
                    )));
                }
            }
        }
    }

    Ok(())
}
