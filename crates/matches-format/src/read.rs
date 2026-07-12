// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.matches` file reading.

use std::io::Seek;
use std::path::Path;

use ndarray::{Array1, Array2, Array3};
use zip::ZipArchive;

use crate::archive_io::{read_binary_array, read_json_entry, read_uint128_array};
use crate::types::*;

/// Read only the top-level metadata from a `.matches` file (fast, no binary data).
pub fn read_matches_metadata(path: &Path) -> Result<MatchesMetadata, MatchesError> {
    let file = std::fs::File::open(path).map_err(|e| MatchesError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut archive = zip::ZipArchive::new(file)?;
    Ok(read_json_entry(&mut archive, "metadata.json.zst")?)
}

/// Read a complete `.matches` file into columnar data.
///
/// The file stores exactly one correspondence backbone: `image_pairs/`
/// (pairwise, all versions) or `clusters/` (version 3+, indicated by the
/// metadata `has_clusters` flag). The optional `cluster_patches/` section
/// requires `clusters/`; the optional `two_view_geometries/` section
/// requires `image_pairs/`.
///
/// Version 1 files store their two-view relative poses in the COLMAP camera
/// convention and are upgraded to the canonical convention on load by
/// S-conjugating every pose ([`s_conjugate_relative_pose`]); the pixel-space
/// F/E/H matrices are identical in both versions and are left untouched.
/// Content hashes cover the stored bytes ([`crate::verify_matches`] re-reads
/// the file), so integrity verification is unaffected by the in-memory
/// upgrade; a re-written file is a new current-version file with new hashes.
///
/// Version ≤ 3 files never store `images/image_dims`, so they load with
/// [`MatchesData::image_dims`] as `None`. A version ≤ 3 file that carries a
/// `cluster_patches/` section is rejected: its `member_affines` last column
/// holds the affine translation `t`, which cannot be upgraded to the
/// version-4 absolute-position semantics (`p = A·x_ref + t`) without the
/// referenced `.sift` keypoint positions — regenerate the file with
/// `sfm cluster-patches` from its cluster backbone source.
pub fn read_matches(path: &Path) -> Result<MatchesData, MatchesError> {
    let file = std::fs::File::open(path).map_err(|e| MatchesError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut archive = zip::ZipArchive::new(file)?;

    // Top-level metadata
    let mut metadata: MatchesMetadata = read_json_entry(&mut archive, "metadata.json.zst")?;
    let content_hash: MatchesContentHash = read_json_entry(&mut archive, "content_hash.json.zst")?;

    // Reject versions newer than this build understands; their semantics are
    // unknown so reading with current-version assumptions would mislead.
    if metadata.version > MATCHES_FORMAT_VERSION {
        return Err(MatchesError::InvalidFormat(format!(
            "unsupported .matches format version {} (this build supports up to \
             {MATCHES_FORMAT_VERSION})",
            metadata.version
        )));
    }
    // Version ≤ 2 files always store the pairwise backbone; clusters arrived
    // with version 3.
    if metadata.version < 3 && (metadata.has_clusters || metadata.has_cluster_patches) {
        return Err(MatchesError::InvalidFormat(format!(
            "version {} file claims clusters/cluster_patches (introduced in version 3)",
            metadata.version
        )));
    }
    if metadata.has_cluster_patches && !metadata.has_clusters {
        return Err(MatchesError::InvalidFormat(
            "has_cluster_patches requires has_clusters (cluster_patches requires the clusters \
             section)"
                .into(),
        ));
    }
    if metadata.has_two_view_geometries && metadata.has_clusters {
        return Err(MatchesError::InvalidFormat(
            "two_view_geometries requires the image_pairs backbone, but this file stores clusters"
                .into(),
        ));
    }
    // Version ≤ 3 cluster-patch files store the affine translation `t` in
    // the member_affines last column; version 4 stores the absolute refined
    // keypoint position `p = A·x_ref + t`. The upgrade needs the referenced
    // `.sift` positions, which the reader does not have — refuse the file
    // (its cluster-backbone source still loads; regenerate the enrichment).
    if metadata.version < 4 && metadata.has_cluster_patches {
        return Err(MatchesError::InvalidFormat(format!(
            "version {} cluster-patch file: member_affines stores the affine translation, not \
             the absolute keypoint position introduced in version 4, and cannot be upgraded on \
             load — regenerate with `sfm cluster-patches` from the cluster backbone file",
            metadata.version
        )));
    }
    let needs_convention_upgrade = metadata.version < 2;
    let stored_version = metadata.version;

    let image_count = metadata.image_count as usize;

    // Cross-check images section metadata
    let images_meta: serde_json::Value = read_json_entry(&mut archive, "images/metadata.json.zst")?;
    if images_meta.get("image_count").and_then(|v| v.as_u64()) != Some(image_count as u64) {
        return Err(MatchesError::InvalidFormat(
            "images/metadata.json.zst image_count doesn't match top-level metadata".into(),
        ));
    }

    // === Images ===
    let image_names: Vec<String> = read_json_entry(&mut archive, "images/names.json.zst")?;
    if image_names.len() != image_count {
        return Err(MatchesError::ShapeMismatch(format!(
            "image names count {} != image_count {image_count}",
            image_names.len()
        )));
    }

    let feature_tool_hashes = read_uint128_array(
        &mut archive,
        &format!("images/feature_tool_hashes.{image_count}.uint128.zst"),
        image_count,
    )?;

    let sift_content_hashes = read_uint128_array(
        &mut archive,
        &format!("images/sift_content_hashes.{image_count}.uint128.zst"),
        image_count,
    )?;

    let feature_counts_vec: Vec<u32> = read_binary_array(
        &mut archive,
        &format!("images/feature_counts.{image_count}.uint32.zst"),
        image_count,
    )?;
    let feature_counts = Array1::from_vec(feature_counts_vec);

    // Per-image dimensions: mandatory since version 4; version ≤ 3 files
    // never stored them.
    let image_dims = if stored_version >= 4 {
        let dims_vec: Vec<u32> = read_binary_array(
            &mut archive,
            &format!("images/image_dims.{image_count}.2.uint32.zst"),
            image_count * 2,
        )?;
        Some(
            Array2::from_shape_vec((image_count, 2), dims_vec)
                .map_err(|e| MatchesError::ShapeMismatch(format!("image_dims reshape: {e}")))?,
        )
    } else {
        None
    };

    // === Backbone: image pairs XOR clusters ===
    let image_pairs = if metadata.has_clusters {
        if metadata.image_pair_count.is_some() || metadata.match_count.is_some() {
            return Err(MatchesError::InvalidFormat(
                "cluster-bearing file must not set metadata.image_pair_count / match_count".into(),
            ));
        }
        None
    } else {
        if metadata.cluster_count.is_some() || metadata.cluster_member_count.is_some() {
            return Err(MatchesError::InvalidFormat(
                "pairwise file must not set metadata.cluster_count / cluster_member_count".into(),
            ));
        }
        Some(read_pairs_section(&mut archive, &metadata)?)
    };

    let clusters = if metadata.has_clusters {
        Some(read_clusters_section(&mut archive, &metadata)?)
    } else {
        None
    };

    // === Cluster patches (optional) ===
    let cluster_patches = if metadata.has_cluster_patches {
        Some(read_cluster_patches_section(&mut archive, &metadata)?)
    } else {
        None
    };

    // === Two-view geometries (optional) ===
    let two_view_geometries = if metadata.has_two_view_geometries {
        let pair_count = metadata.image_pair_count.unwrap_or(0) as usize;
        let mut tvg = read_tvg_section(&mut archive, pair_count)?;
        // Version 1 → 2 upgrade: relative poses were stored in the COLMAP
        // camera convention; S-conjugate them to canonical. F/E/H are
        // pixel-space and unchanged.
        if needs_convention_upgrade {
            tvg.s_conjugate_poses();
        }
        Some(tvg)
    } else {
        None
    };

    // Older files are upgraded in memory; a re-written file is a new
    // current-version file.
    if metadata.version < MATCHES_FORMAT_VERSION {
        metadata.version = MATCHES_FORMAT_VERSION;
    }

    Ok(MatchesData {
        metadata,
        content_hash,
        image_names,
        feature_tool_hashes,
        sift_content_hashes,
        feature_counts,
        image_dims,
        image_pairs,
        clusters,
        cluster_patches,
        two_view_geometries,
    })
}

fn read_pairs_section<R: std::io::Read + Seek>(
    archive: &mut ZipArchive<R>,
    metadata: &MatchesMetadata,
) -> Result<PairsData, MatchesError> {
    let pair_count = metadata.image_pair_count.ok_or_else(|| {
        MatchesError::InvalidFormat(
            "pairwise file requires metadata.image_pair_count and metadata.match_count".into(),
        )
    })? as usize;
    let match_count = metadata.match_count.ok_or_else(|| {
        MatchesError::InvalidFormat(
            "pairwise file requires metadata.image_pair_count and metadata.match_count".into(),
        )
    })? as usize;

    // Cross-check pairs section metadata
    let pairs_meta: serde_json::Value = read_json_entry(archive, "image_pairs/metadata.json.zst")?;
    if pairs_meta.get("image_pair_count").and_then(|v| v.as_u64()) != Some(pair_count as u64) {
        return Err(MatchesError::InvalidFormat(
            "image_pairs/metadata.json.zst image_pair_count doesn't match top-level metadata"
                .into(),
        ));
    }
    if pairs_meta.get("match_count").and_then(|v| v.as_u64()) != Some(match_count as u64) {
        return Err(MatchesError::InvalidFormat(
            "image_pairs/metadata.json.zst match_count doesn't match top-level metadata".into(),
        ));
    }

    let image_index_pairs_vec: Vec<u32> = read_binary_array(
        archive,
        &format!("image_pairs/image_index_pairs.{pair_count}.2.uint32.zst"),
        pair_count * 2,
    )?;
    let image_index_pairs = Array2::from_shape_vec((pair_count, 2), image_index_pairs_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("image_index_pairs reshape: {e}")))?;

    let match_counts_vec: Vec<u32> = read_binary_array(
        archive,
        &format!("image_pairs/match_counts.{pair_count}.uint32.zst"),
        pair_count,
    )?;
    let match_counts = Array1::from_vec(match_counts_vec);

    let match_feature_indexes_vec: Vec<u32> = read_binary_array(
        archive,
        &format!("image_pairs/match_feature_indexes.{match_count}.2.uint32.zst"),
        match_count * 2,
    )?;
    let match_feature_indexes = Array2::from_shape_vec((match_count, 2), match_feature_indexes_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("match_feature_indexes reshape: {e}")))?;

    let match_descriptor_distances_vec: Vec<f32> = read_binary_array(
        archive,
        &format!("image_pairs/match_descriptor_distances.{match_count}.float32.zst"),
        match_count,
    )?;
    let match_descriptor_distances = Array1::from_vec(match_descriptor_distances_vec);

    Ok(PairsData {
        image_index_pairs,
        match_counts,
        match_feature_indexes,
        match_descriptor_distances,
    })
}

fn read_clusters_section<R: std::io::Read + Seek>(
    archive: &mut ZipArchive<R>,
    metadata: &MatchesMetadata,
) -> Result<ClustersData, MatchesError> {
    let cluster_count = metadata.cluster_count.ok_or_else(|| {
        MatchesError::InvalidFormat(
            "cluster-bearing file requires metadata.cluster_count and \
             metadata.cluster_member_count"
                .into(),
        )
    })? as usize;
    let member_count = metadata.cluster_member_count.ok_or_else(|| {
        MatchesError::InvalidFormat(
            "cluster-bearing file requires metadata.cluster_count and \
             metadata.cluster_member_count"
                .into(),
        )
    })? as usize;

    // Cross-check clusters section metadata
    let clusters_meta: serde_json::Value = read_json_entry(archive, "clusters/metadata.json.zst")?;
    if clusters_meta.get("cluster_count").and_then(|v| v.as_u64()) != Some(cluster_count as u64) {
        return Err(MatchesError::InvalidFormat(
            "clusters/metadata.json.zst cluster_count doesn't match top-level metadata".into(),
        ));
    }
    if clusters_meta.get("member_count").and_then(|v| v.as_u64()) != Some(member_count as u64) {
        return Err(MatchesError::InvalidFormat(
            "clusters/metadata.json.zst member_count doesn't match top-level metadata".into(),
        ));
    }
    let matcher_options = clusters_meta
        .get("matcher_options")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let cluster_starts_vec: Vec<u32> = read_binary_array(
        archive,
        &format!("clusters/cluster_starts.{}.uint32.zst", cluster_count + 1),
        cluster_count + 1,
    )?;
    let cluster_starts = Array1::from_vec(cluster_starts_vec);

    let member_images_vec: Vec<u32> = read_binary_array(
        archive,
        &format!("clusters/member_images.{member_count}.uint32.zst"),
        member_count,
    )?;
    let member_images = Array1::from_vec(member_images_vec);

    let member_features_vec: Vec<u32> = read_binary_array(
        archive,
        &format!("clusters/member_features.{member_count}.uint32.zst"),
        member_count,
    )?;
    let member_features = Array1::from_vec(member_features_vec);

    Ok(ClustersData {
        cluster_starts,
        member_images,
        member_features,
        matcher_options,
    })
}

fn read_cluster_patches_section<R: std::io::Read + Seek>(
    archive: &mut ZipArchive<R>,
    metadata: &MatchesMetadata,
) -> Result<ClusterPatchData, MatchesError> {
    let cluster_count = metadata.cluster_count.unwrap_or(0) as usize;
    let member_count = metadata.cluster_member_count.unwrap_or(0) as usize;

    // Cross-check cluster_patches section metadata
    let cp_meta: serde_json::Value = read_json_entry(archive, "cluster_patches/metadata.json.zst")?;
    if cp_meta.get("cluster_count").and_then(|v| v.as_u64()) != Some(cluster_count as u64) {
        return Err(MatchesError::InvalidFormat(
            "cluster_patches/metadata.json.zst cluster_count doesn't match top-level metadata"
                .into(),
        ));
    }
    if cp_meta.get("member_count").and_then(|v| v.as_u64()) != Some(member_count as u64) {
        return Err(MatchesError::InvalidFormat(
            "cluster_patches/metadata.json.zst member_count doesn't match top-level metadata"
                .into(),
        ));
    }
    let refine_options = cp_meta
        .get("refine_options")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let reference_members_vec: Vec<u32> = read_binary_array(
        archive,
        &format!("cluster_patches/reference_members.{cluster_count}.uint32.zst"),
        cluster_count,
    )?;
    let reference_members = Array1::from_vec(reference_members_vec);

    let member_status_vec: Vec<u8> = read_binary_array(
        archive,
        &format!("cluster_patches/member_status.{member_count}.uint8.zst"),
        member_count,
    )?;
    let member_status = Array1::from_vec(member_status_vec);

    let member_affines_vec: Vec<f64> = read_binary_array(
        archive,
        &format!("cluster_patches/member_affines.{member_count}.2.3.float64.zst"),
        member_count * 6,
    )?;
    let member_affines = Array3::from_shape_vec((member_count, 2, 3), member_affines_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("member_affines reshape: {e}")))?;

    let member_zncc_vec: Vec<f32> = read_binary_array(
        archive,
        &format!("cluster_patches/member_zncc.{member_count}.float32.zst"),
        member_count,
    )?;
    let member_zncc = Array1::from_vec(member_zncc_vec);

    let member_shift_px_vec: Vec<f32> = read_binary_array(
        archive,
        &format!("cluster_patches/member_shift_px.{member_count}.float32.zst"),
        member_count,
    )?;
    let member_shift_px = Array1::from_vec(member_shift_px_vec);

    let member_consistency_vec: Vec<f32> = read_binary_array(
        archive,
        &format!("cluster_patches/member_consistency_residual.{member_count}.float32.zst"),
        member_count,
    )?;
    let member_consistency_residual = Array1::from_vec(member_consistency_vec);

    Ok(ClusterPatchData {
        reference_members,
        member_status,
        member_affines,
        member_zncc,
        member_shift_px,
        member_consistency_residual,
        refine_options,
    })
}

fn read_tvg_section<R: std::io::Read + Seek>(
    archive: &mut ZipArchive<R>,
    pair_count: usize,
) -> Result<TwoViewGeometryData, MatchesError> {
    let tvg_metadata: TvgMetadata =
        read_json_entry(archive, "two_view_geometries/metadata.json.zst")?;
    let inlier_count = tvg_metadata.inlier_count as usize;

    if tvg_metadata.image_pair_count as usize != pair_count {
        return Err(MatchesError::InvalidFormat(format!(
            "TVG image_pair_count {} != pair_count {pair_count}",
            tvg_metadata.image_pair_count
        )));
    }

    // config_types.json
    let config_type_strings: Vec<String> =
        read_json_entry(archive, "two_view_geometries/config_types.json.zst")?;
    let config_types: Vec<TwoViewGeometryConfig> = config_type_strings
        .iter()
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;

    // config_indexes
    let config_indexes_vec: Vec<u8> = read_binary_array(
        archive,
        &format!("two_view_geometries/config_indexes.{pair_count}.uint8.zst"),
        pair_count,
    )?;
    let config_indexes = Array1::from_vec(config_indexes_vec);

    // Validate config_indexes bounds
    for (k, &idx) in config_indexes.iter().enumerate() {
        if idx as usize >= config_types.len() {
            return Err(MatchesError::InvalidFormat(format!(
                "config_indexes[{k}] = {idx} >= config_types length {}",
                config_types.len()
            )));
        }
    }

    // inlier_counts
    let inlier_counts_vec: Vec<u32> = read_binary_array(
        archive,
        &format!("two_view_geometries/inlier_counts.{pair_count}.uint32.zst"),
        pair_count,
    )?;
    let inlier_counts = Array1::from_vec(inlier_counts_vec);

    // inlier_feature_indexes
    let inlier_fi_vec: Vec<u32> = read_binary_array(
        archive,
        &format!("two_view_geometries/inlier_feature_indexes.{inlier_count}.2.uint32.zst"),
        inlier_count * 2,
    )?;
    let inlier_feature_indexes = Array2::from_shape_vec((inlier_count, 2), inlier_fi_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("inlier_feature_indexes reshape: {e}")))?;

    // Matrices
    let f_vec: Vec<f64> = read_binary_array(
        archive,
        &format!("two_view_geometries/f_matrices.{pair_count}.3.3.float64.zst"),
        pair_count * 9,
    )?;
    let f_matrices = Array3::from_shape_vec((pair_count, 3, 3), f_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("f_matrices reshape: {e}")))?;

    let e_vec: Vec<f64> = read_binary_array(
        archive,
        &format!("two_view_geometries/e_matrices.{pair_count}.3.3.float64.zst"),
        pair_count * 9,
    )?;
    let e_matrices = Array3::from_shape_vec((pair_count, 3, 3), e_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("e_matrices reshape: {e}")))?;

    let h_vec: Vec<f64> = read_binary_array(
        archive,
        &format!("two_view_geometries/h_matrices.{pair_count}.3.3.float64.zst"),
        pair_count * 9,
    )?;
    let h_matrices = Array3::from_shape_vec((pair_count, 3, 3), h_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("h_matrices reshape: {e}")))?;

    // Quaternions and translations
    let quat_vec: Vec<f64> = read_binary_array(
        archive,
        &format!("two_view_geometries/quaternions_wxyz.{pair_count}.4.float64.zst"),
        pair_count * 4,
    )?;
    let quaternions_wxyz = Array2::from_shape_vec((pair_count, 4), quat_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("quaternions reshape: {e}")))?;

    let trans_vec: Vec<f64> = read_binary_array(
        archive,
        &format!("two_view_geometries/translations_xyz.{pair_count}.3.float64.zst"),
        pair_count * 3,
    )?;
    let translations_xyz = Array2::from_shape_vec((pair_count, 3), trans_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("translations reshape: {e}")))?;

    Ok(TwoViewGeometryData {
        metadata: tvg_metadata,
        config_types,
        config_indexes,
        inlier_counts,
        inlier_feature_indexes,
        f_matrices,
        e_matrices,
        h_matrices,
        quaternions_wxyz,
        translations_xyz,
    })
}
