// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.matches` file writing.

use std::path::Path;

use xxhash_rust::xxh3::Xxh3;
use zip::ZipWriter;

use crate::cells::{ClusterCellStatus, MEMBER_CELL_STATUS_NAMES_KEY};
use crate::entries;
use crate::types::*;
use crate::validate::{content_errors, validate_dimensions, validate_structure};
use sfmtool_archive_io::{
    format_hash, row_major_bytes, write_binary_entry_hashed, write_json_entry,
};

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
/// `image_dims` is mandatory (format version 4): `(N, 2)` width/height with
/// every value ≥ 1. A cluster-backbone file additionally requires the
/// members' detections (format version 6): `member_positions` `(M, 2)` and
/// `member_affine_shapes` `(M, 2, 2)`. A `cluster_patches` section's
/// `member_status` holds canonical [`ClusterMemberStatus`] discriminants, and
/// the writer states the canonical `member_status_names` legend beside them
/// (format version 7). Its optional per-cell columns
/// ([`ClusterPatchData::member_cells`]) hold canonical [`ClusterCellStatus`]
/// discriminants, are sized by the member count, carry readings only for
/// `kept` members, and are written with the canonical
/// `member_cell_status_names` legend (format version 8).
pub fn write_matches(path: &Path, data: &MatchesData, zstd_level: i32) -> Result<(), MatchesError> {
    let image_count = data.metadata.image_count as usize;

    validate_structure(data)?;
    validate_dimensions(data, image_count)?;
    if let Some(error) = content_errors(data).into_iter().next() {
        return Err(MatchesError::InvalidFormat(error));
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| MatchesError::IoPath {
            operation: "Failed to create parent directory",
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    // Through a temporary file renamed over the target, so a failed write
    // leaves the previous file intact rather than a truncated one. See
    // `sfmtool_archive_io::write_atomically`.
    sfmtool_archive_io::write_atomically(path, |file| write_matches_into(file, data, zstd_level))
}

/// Write a clusters file: a `.matches` file with the cluster backbone and no
/// pairs, cluster patches or two-view geometries.
///
/// This is [`write_matches`] for a cluster matcher's output. It takes only the
/// values the caller supplies ([`ClustersMatchesData`]) and fills in the rest
/// of the top-level metadata: the image count from `image_names`, the cluster
/// count from `cluster_starts`, the member count from `member_images`, and the
/// `has_*` flags. [`write_matches`] then checks the result as it checks any
/// file, so the same errors reach the caller: a `cluster_starts` that is not a
/// CSR offset array over the members, a `member_images` entry at or past the
/// image count, a `member_features` entry at or past its image's
/// `feature_counts` entry, and image or member arrays whose lengths disagree.
///
/// ```no_run
/// # fn example(data: sfmtool_matches_format::ClustersMatchesData) {
/// let path = std::path::Path::new("matches/clusters.matches");
/// sfmtool_matches_format::write_clusters_matches(path, data, 3).unwrap();
/// # }
/// ```
pub fn write_clusters_matches(
    path: &Path,
    data: ClustersMatchesData,
    zstd_level: i32,
) -> Result<(), MatchesError> {
    let count = |what: &str, n: usize| {
        u32::try_from(n)
            .map_err(|_| MatchesError::ShapeMismatch(format!("{what} {n} does not fit in a u32")))
    };
    let image_count = count("image count", data.image_names.len())?;
    // An empty `cluster_starts` reads as zero clusters, which `write_matches`
    // then refuses because the array lacks its leading 0.
    let cluster_count = count(
        "cluster count",
        data.clusters.cluster_starts.len().saturating_sub(1),
    )?;
    let member_count = count("member count", data.clusters.member_images.len())?;

    let matches = MatchesData {
        metadata: MatchesMetadata {
            version: MATCHES_FORMAT_VERSION,
            matching_method: data.matching_method,
            matching_tool: data.matching_tool,
            matching_tool_version: data.matching_tool_version,
            matching_options: data.matching_options,
            workspace: data.workspace,
            timestamp: data.timestamp,
            image_count,
            image_pair_count: None,
            match_count: None,
            cluster_count: Some(cluster_count),
            cluster_member_count: Some(member_count),
            has_two_view_geometries: false,
            has_clusters: true,
            has_cluster_patches: false,
        },
        // Recomputed by the write.
        content_hash: MatchesContentHash {
            metadata_xxh128: String::new(),
            images_xxh128: String::new(),
            image_pairs_xxh128: None,
            clusters_xxh128: None,
            cluster_patches_xxh128: None,
            two_view_geometries_xxh128: None,
            content_xxh128: String::new(),
        },
        image_names: data.image_names,
        feature_tool_hashes: data.feature_tool_hashes,
        sift_content_hashes: data.sift_content_hashes,
        feature_counts: data.feature_counts,
        image_dims: Some(data.image_dims),
        image_pairs: None,
        clusters: Some(data.clusters),
        cluster_patches: None,
        two_view_geometries: None,
    };
    write_matches(path, &matches, zstd_level)
}

/// Stream the archive into `writer`. The whole of the write but the choosing of
/// where it lands.
fn write_matches_into<W: std::io::Write + std::io::Seek>(
    writer: W,
    data: &MatchesData,
    zstd_level: i32,
) -> Result<(), MatchesError> {
    let image_count = data.metadata.image_count as usize;
    let mut zip = ZipWriter::new(writer);
    // Present-section digests, accumulated in the canonical order: metadata,
    // images, pairs, clusters, cluster_patches, two_view_geometries.
    let mut section_digests = sfmtool_archive_io::SectionDigests::new();

    // === Top-level metadata (always emitted at the current format version) ===
    let mut metadata = data.metadata.clone();
    metadata.version = MATCHES_FORMAT_VERSION;
    let metadata_bytes = write_json_entry(&mut zip, entries::metadata(), &metadata, zstd_level)?;
    let metadata_hash = xxhash_rust::xxh3::xxh3_128(&metadata_bytes);
    section_digests.push(metadata_hash);

    let images_hash = write_images(&mut zip, data, image_count, zstd_level)?;
    section_digests.push(images_hash);

    let pairs_hash = write_image_pairs(&mut zip, data, zstd_level)?;
    if let Some(hash) = pairs_hash {
        section_digests.push(hash);
    }
    let clusters_hash = write_clusters(&mut zip, data, zstd_level)?;
    if let Some(hash) = clusters_hash {
        section_digests.push(hash);
    }
    let cluster_patches_hash = write_cluster_patches(&mut zip, data, zstd_level)?;
    if let Some(hash) = cluster_patches_hash {
        section_digests.push(hash);
    }
    let tvg_hash = write_two_view_geometries(&mut zip, data, zstd_level)?;
    if let Some(hash) = tvg_hash {
        section_digests.push(hash);
    }

    // === Content hash ===
    let content_hash_value = section_digests.finish();

    let content_hash = MatchesContentHash {
        metadata_xxh128: format_hash(metadata_hash),
        images_xxh128: format_hash(images_hash),
        image_pairs_xxh128: pairs_hash.map(format_hash),
        clusters_xxh128: clusters_hash.map(format_hash),
        cluster_patches_xxh128: cluster_patches_hash.map(format_hash),
        two_view_geometries_xxh128: tvg_hash.map(format_hash),
        content_xxh128: format_hash(content_hash_value),
    };
    write_json_entry(&mut zip, entries::content_hash(), &content_hash, zstd_level)?;

    zip.finish()?;
    Ok(())
}

fn write_images<W: std::io::Write + std::io::Seek>(
    zip: &mut ZipWriter<W>,
    data: &MatchesData,
    image_count: usize,
    zstd_level: i32,
) -> Result<u128, MatchesError> {
    // === Images (hashed in lexicographic path order) ===
    let mut images_hasher = Xxh3::new();

    // images/feature_counts
    write_binary_entry_hashed(
        zip,
        &entries::images_feature_counts(image_count),
        &row_major_bytes(&data.feature_counts),
        zstd_level,
        &mut images_hasher,
    )?;

    // images/feature_tool_hashes
    let hash_bytes: Vec<u8> = data
        .feature_tool_hashes
        .iter()
        .flat_map(|h| h.iter().copied())
        .collect();
    write_binary_entry_hashed(
        zip,
        &entries::images_feature_tool_hashes(image_count),
        &hash_bytes,
        zstd_level,
        &mut images_hasher,
    )?;

    // images/image_dims (mandatory since format version 4; validated Some
    // by validate_dimensions)
    let image_dims = data.image_dims.as_ref().expect("validated Some");
    write_binary_entry_hashed(
        zip,
        &entries::images_image_dims(image_count),
        &row_major_bytes(image_dims),
        zstd_level,
        &mut images_hasher,
    )?;

    // images/metadata.json
    let images_meta = serde_json::json!({"image_count": image_count});
    let bytes = write_json_entry(zip, entries::images_metadata(), &images_meta, zstd_level)?;
    images_hasher.update(&bytes);

    // images/names.json
    let bytes = write_json_entry(zip, entries::images_names(), &data.image_names, zstd_level)?;
    images_hasher.update(&bytes);

    // images/sift_content_hashes
    let hash_bytes: Vec<u8> = data
        .sift_content_hashes
        .iter()
        .flat_map(|h| h.iter().copied())
        .collect();
    write_binary_entry_hashed(
        zip,
        &entries::images_sift_content_hashes(image_count),
        &hash_bytes,
        zstd_level,
        &mut images_hasher,
    )?;

    Ok(images_hasher.digest128())
}

fn write_image_pairs<W: std::io::Write + std::io::Seek>(
    zip: &mut ZipWriter<W>,
    data: &MatchesData,
    zstd_level: i32,
) -> Result<Option<u128>, MatchesError> {
    // === Image pairs (backbone alternative, hashed in lexicographic path order) ===
    let pairs_hash = if let Some(pairs) = &data.image_pairs {
        let pair_count = pairs.image_index_pairs.nrows();
        let match_count = pairs.match_feature_indexes.nrows();
        let mut pairs_hasher = Xxh3::new();

        // image_pairs/image_index_pairs
        write_binary_entry_hashed(
            zip,
            &entries::image_pairs_image_index_pairs(pair_count),
            &row_major_bytes(&pairs.image_index_pairs),
            zstd_level,
            &mut pairs_hasher,
        )?;

        // image_pairs/match_counts
        write_binary_entry_hashed(
            zip,
            &entries::image_pairs_match_counts(pair_count),
            &row_major_bytes(&pairs.match_counts),
            zstd_level,
            &mut pairs_hasher,
        )?;

        // image_pairs/match_descriptor_distances
        write_binary_entry_hashed(
            zip,
            &entries::image_pairs_match_descriptor_distances(match_count),
            &row_major_bytes(&pairs.match_descriptor_distances),
            zstd_level,
            &mut pairs_hasher,
        )?;

        // image_pairs/match_feature_indexes
        write_binary_entry_hashed(
            zip,
            &entries::image_pairs_match_feature_indexes(match_count),
            &row_major_bytes(&pairs.match_feature_indexes),
            zstd_level,
            &mut pairs_hasher,
        )?;

        // image_pairs/metadata.json
        let pairs_meta =
            serde_json::json!({"image_pair_count": pair_count, "match_count": match_count});
        let bytes = write_json_entry(
            zip,
            entries::image_pairs_metadata(),
            &pairs_meta,
            zstd_level,
        )?;
        pairs_hasher.update(&bytes);

        let digest = pairs_hasher.digest128();
        Some(digest)
    } else {
        None
    };
    Ok(pairs_hash)
}

fn write_clusters<W: std::io::Write + std::io::Seek>(
    zip: &mut ZipWriter<W>,
    data: &MatchesData,
    zstd_level: i32,
) -> Result<Option<u128>, MatchesError> {
    // === Clusters (backbone alternative, hashed in lexicographic path order) ===
    let clusters_hash = if let Some(clusters) = &data.clusters {
        let cluster_count = clusters.cluster_starts.len() - 1;
        let member_count = clusters.member_images.len();
        let mut clusters_hasher = Xxh3::new();

        // clusters/cluster_starts
        write_binary_entry_hashed(
            zip,
            &entries::clusters_cluster_starts(cluster_count),
            &row_major_bytes(&clusters.cluster_starts),
            zstd_level,
            &mut clusters_hasher,
        )?;

        // clusters/member_affine_shapes (mandatory since format version 6;
        // validated Some by validate_dimensions)
        let member_affine_shapes = clusters
            .member_affine_shapes
            .as_ref()
            .expect("validated Some");
        write_binary_entry_hashed(
            zip,
            &entries::clusters_member_affine_shapes(member_count),
            &row_major_bytes(member_affine_shapes),
            zstd_level,
            &mut clusters_hasher,
        )?;

        // clusters/member_features
        write_binary_entry_hashed(
            zip,
            &entries::clusters_member_features(member_count),
            &row_major_bytes(&clusters.member_features),
            zstd_level,
            &mut clusters_hasher,
        )?;

        // clusters/member_images
        write_binary_entry_hashed(
            zip,
            &entries::clusters_member_images(member_count),
            &row_major_bytes(&clusters.member_images),
            zstd_level,
            &mut clusters_hasher,
        )?;

        // clusters/member_positions (mandatory since format version 6)
        let member_positions = clusters.member_positions.as_ref().expect("validated Some");
        write_binary_entry_hashed(
            zip,
            &entries::clusters_member_positions(member_count),
            &row_major_bytes(member_positions),
            zstd_level,
            &mut clusters_hasher,
        )?;

        // clusters/metadata.json
        let clusters_meta = serde_json::json!({
            "cluster_count": cluster_count,
            "member_count": member_count,
            "matcher_options": clusters.matcher_options,
        });
        let bytes = write_json_entry(
            zip,
            entries::clusters_metadata(),
            &clusters_meta,
            zstd_level,
        )?;
        clusters_hasher.update(&bytes);

        let digest = clusters_hasher.digest128();
        Some(digest)
    } else {
        None
    };
    Ok(clusters_hash)
}

fn write_cluster_patches<W: std::io::Write + std::io::Seek>(
    zip: &mut ZipWriter<W>,
    data: &MatchesData,
    zstd_level: i32,
) -> Result<Option<u128>, MatchesError> {
    // === Cluster patches (optional, hashed in lexicographic path order) ===
    let cluster_patches_hash = if let Some(cp) = &data.cluster_patches {
        let cluster_count = cp.reference_members.len();
        let member_count = cp.member_status.len();
        let mut cp_hasher = Xxh3::new();

        // cluster_patches/member_cell_* (optional, version 8): they sort
        // before member_consistency_residual, so they are hashed first.
        if let Some(cells) = &cp.member_cells {
            write_binary_entry_hashed(
                zip,
                &entries::cluster_patches_member_cell_iterations(member_count),
                &row_major_bytes(&cells.iterations),
                zstd_level,
                &mut cp_hasher,
            )?;
            write_binary_entry_hashed(
                zip,
                &entries::cluster_patches_member_cell_shift_px(member_count),
                &row_major_bytes(&cells.shift_px),
                zstd_level,
                &mut cp_hasher,
            )?;
            write_binary_entry_hashed(
                zip,
                &entries::cluster_patches_member_cell_status(member_count),
                &row_major_bytes(&cells.status),
                zstd_level,
                &mut cp_hasher,
            )?;
            write_binary_entry_hashed(
                zip,
                &entries::cluster_patches_member_cell_zncc(member_count),
                &row_major_bytes(&cells.zncc),
                zstd_level,
                &mut cp_hasher,
            )?;
        }

        // cluster_patches/member_consistency_residual
        write_binary_entry_hashed(
            zip,
            &entries::cluster_patches_member_consistency_residual(member_count),
            &row_major_bytes(&cp.member_consistency_residual),
            zstd_level,
            &mut cp_hasher,
        )?;

        // cluster_patches/member_shift_px
        write_binary_entry_hashed(
            zip,
            &entries::cluster_patches_member_shift_px(member_count),
            &row_major_bytes(&cp.member_shift_px),
            zstd_level,
            &mut cp_hasher,
        )?;

        // cluster_patches/member_status
        write_binary_entry_hashed(
            zip,
            &entries::cluster_patches_member_status(member_count),
            &row_major_bytes(&cp.member_status),
            zstd_level,
            &mut cp_hasher,
        )?;

        // cluster_patches/member_zncc
        write_binary_entry_hashed(
            zip,
            &entries::cluster_patches_member_zncc(member_count),
            &row_major_bytes(&cp.member_zncc),
            zstd_level,
            &mut cp_hasher,
        )?;

        // cluster_patches/metadata.json, with the whole canonical legend the
        // member_status codes index (validated against it above).
        let mut cp_meta = serde_json::json!({
            "cluster_count": cluster_count,
            "member_count": member_count,
            "member_status_names": ClusterMemberStatus::NAMES,
            "refine_options": cp.refine_options,
        });
        // The per-cell columns' legend, present exactly when they are.
        if cp.member_cells.is_some() {
            cp_meta[MEMBER_CELL_STATUS_NAMES_KEY] = serde_json::json!(ClusterCellStatus::NAMES);
        }
        let bytes = write_json_entry(
            zip,
            entries::cluster_patches_metadata(),
            &cp_meta,
            zstd_level,
        )?;
        cp_hasher.update(&bytes);

        // cluster_patches/reference_members
        write_binary_entry_hashed(
            zip,
            &entries::cluster_patches_reference_members(cluster_count),
            &row_major_bytes(&cp.reference_members),
            zstd_level,
            &mut cp_hasher,
        )?;

        let digest = cp_hasher.digest128();
        Some(digest)
    } else {
        None
    };
    Ok(cluster_patches_hash)
}

fn write_two_view_geometries<W: std::io::Write + std::io::Seek>(
    zip: &mut ZipWriter<W>,
    data: &MatchesData,
    zstd_level: i32,
) -> Result<Option<u128>, MatchesError> {
    // === Two-view geometries (optional, hashed in lexicographic path order) ===
    let tvg_hash = if let Some(tvg) = &data.two_view_geometries {
        let pair_count = tvg.config_indexes.len();
        let inlier_count = tvg.metadata.inlier_count as usize;
        let mut tvg_hasher = Xxh3::new();

        // two_view_geometries/config_indexes
        write_binary_entry_hashed(
            zip,
            &entries::two_view_geometries_config_indexes(pair_count),
            &row_major_bytes(&tvg.config_indexes),
            zstd_level,
            &mut tvg_hasher,
        )?;

        // two_view_geometries/config_types.json
        let config_type_strings: Vec<&str> = tvg.config_types.iter().map(|c| c.as_str()).collect();
        let bytes = write_json_entry(
            zip,
            entries::two_view_geometries_config_types(),
            &config_type_strings,
            zstd_level,
        )?;
        tvg_hasher.update(&bytes);

        // two_view_geometries/e_matrices
        write_binary_entry_hashed(
            zip,
            &entries::two_view_geometries_e_matrices(pair_count),
            &row_major_bytes(&tvg.e_matrices),
            zstd_level,
            &mut tvg_hasher,
        )?;

        // two_view_geometries/f_matrices
        write_binary_entry_hashed(
            zip,
            &entries::two_view_geometries_f_matrices(pair_count),
            &row_major_bytes(&tvg.f_matrices),
            zstd_level,
            &mut tvg_hasher,
        )?;

        // two_view_geometries/h_matrices
        write_binary_entry_hashed(
            zip,
            &entries::two_view_geometries_h_matrices(pair_count),
            &row_major_bytes(&tvg.h_matrices),
            zstd_level,
            &mut tvg_hasher,
        )?;

        // two_view_geometries/inlier_counts
        write_binary_entry_hashed(
            zip,
            &entries::two_view_geometries_inlier_counts(pair_count),
            &row_major_bytes(&tvg.inlier_counts),
            zstd_level,
            &mut tvg_hasher,
        )?;

        // two_view_geometries/inlier_feature_indexes
        write_binary_entry_hashed(
            zip,
            &entries::two_view_geometries_inlier_feature_indexes(inlier_count),
            &row_major_bytes(&tvg.inlier_feature_indexes),
            zstd_level,
            &mut tvg_hasher,
        )?;

        // two_view_geometries/metadata.json
        let bytes = write_json_entry(
            zip,
            entries::two_view_geometries_metadata(),
            &tvg.metadata,
            zstd_level,
        )?;
        tvg_hasher.update(&bytes);

        // two_view_geometries/quaternions_wxyz
        write_binary_entry_hashed(
            zip,
            &entries::two_view_geometries_quaternions_wxyz(pair_count),
            &row_major_bytes(&tvg.quaternions_wxyz),
            zstd_level,
            &mut tvg_hasher,
        )?;

        // two_view_geometries/translations_xyz
        write_binary_entry_hashed(
            zip,
            &entries::two_view_geometries_translations_xyz(pair_count),
            &row_major_bytes(&tvg.translations_xyz),
            zstd_level,
            &mut tvg_hasher,
        )?;

        let digest = tvg_hasher.digest128();
        Some(digest)
    } else {
        None
    };
    Ok(tvg_hash)
}
