// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.matches` file reading.

use std::path::Path;

use ndarray::{Array1, Array2, Array3};

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
pub fn read_matches(path: &Path) -> Result<MatchesData, MatchesError> {
    let file = std::fs::File::open(path).map_err(|e| MatchesError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut archive = zip::ZipArchive::new(file)?;

    // Top-level metadata
    let metadata: MatchesMetadata = read_json_entry(&mut archive, "metadata.json.zst")?;
    let content_hash: MatchesContentHash = read_json_entry(&mut archive, "content_hash.json.zst")?;

    let image_count = metadata.image_count as usize;
    let pair_count = metadata.image_pair_count as usize;
    let match_count = metadata.match_count as usize;

    // Cross-check images section metadata
    let images_meta: serde_json::Value = read_json_entry(&mut archive, "images/metadata.json.zst")?;
    if images_meta.get("image_count").and_then(|v| v.as_u64()) != Some(image_count as u64) {
        return Err(MatchesError::InvalidFormat(
            "images/metadata.json.zst image_count doesn't match top-level metadata".into(),
        ));
    }

    // Cross-check pairs section metadata
    let pairs_meta: serde_json::Value =
        read_json_entry(&mut archive, "image_pairs/metadata.json.zst")?;
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

    // === Image pairs ===
    let image_index_pairs_vec: Vec<u32> = read_binary_array(
        &mut archive,
        &format!("image_pairs/image_index_pairs.{pair_count}.2.uint32.zst"),
        pair_count * 2,
    )?;
    let image_index_pairs = Array2::from_shape_vec((pair_count, 2), image_index_pairs_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("image_index_pairs reshape: {e}")))?;

    let match_counts_vec: Vec<u32> = read_binary_array(
        &mut archive,
        &format!("image_pairs/match_counts.{pair_count}.uint32.zst"),
        pair_count,
    )?;
    let match_counts = Array1::from_vec(match_counts_vec);

    let match_feature_indexes_vec: Vec<u32> = read_binary_array(
        &mut archive,
        &format!("image_pairs/match_feature_indexes.{match_count}.2.uint32.zst"),
        match_count * 2,
    )?;
    let match_feature_indexes = Array2::from_shape_vec((match_count, 2), match_feature_indexes_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("match_feature_indexes reshape: {e}")))?;

    let match_descriptor_distances_vec: Vec<f32> = read_binary_array(
        &mut archive,
        &format!("image_pairs/match_descriptor_distances.{match_count}.float32.zst"),
        match_count,
    )?;
    let match_descriptor_distances = Array1::from_vec(match_descriptor_distances_vec);

    // === Two-view geometries (optional) ===
    let two_view_geometries = if metadata.has_two_view_geometries {
        let tvg_metadata: TvgMetadata =
            read_json_entry(&mut archive, "two_view_geometries/metadata.json.zst")?;
        let inlier_count = tvg_metadata.inlier_count as usize;

        if tvg_metadata.image_pair_count as usize != pair_count {
            return Err(MatchesError::InvalidFormat(format!(
                "TVG image_pair_count {} != pair_count {pair_count}",
                tvg_metadata.image_pair_count
            )));
        }

        // config_types.json
        let config_type_strings: Vec<String> =
            read_json_entry(&mut archive, "two_view_geometries/config_types.json.zst")?;
        let config_types: Vec<TwoViewGeometryConfig> = config_type_strings
            .iter()
            .map(|s| s.parse())
            .collect::<Result<_, _>>()?;

        // config_indexes
        let config_indexes_vec: Vec<u8> = read_binary_array(
            &mut archive,
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
            &mut archive,
            &format!("two_view_geometries/inlier_counts.{pair_count}.uint32.zst"),
            pair_count,
        )?;
        let inlier_counts = Array1::from_vec(inlier_counts_vec);

        // inlier_feature_indexes
        let inlier_fi_vec: Vec<u32> = read_binary_array(
            &mut archive,
            &format!("two_view_geometries/inlier_feature_indexes.{inlier_count}.2.uint32.zst"),
            inlier_count * 2,
        )?;
        let inlier_feature_indexes = Array2::from_shape_vec((inlier_count, 2), inlier_fi_vec)
            .map_err(|e| {
                MatchesError::ShapeMismatch(format!("inlier_feature_indexes reshape: {e}"))
            })?;

        // Matrices
        let f_vec: Vec<f64> = read_binary_array(
            &mut archive,
            &format!("two_view_geometries/f_matrices.{pair_count}.3.3.float64.zst"),
            pair_count * 9,
        )?;
        let f_matrices = Array3::from_shape_vec((pair_count, 3, 3), f_vec)
            .map_err(|e| MatchesError::ShapeMismatch(format!("f_matrices reshape: {e}")))?;

        let e_vec: Vec<f64> = read_binary_array(
            &mut archive,
            &format!("two_view_geometries/e_matrices.{pair_count}.3.3.float64.zst"),
            pair_count * 9,
        )?;
        let e_matrices = Array3::from_shape_vec((pair_count, 3, 3), e_vec)
            .map_err(|e| MatchesError::ShapeMismatch(format!("e_matrices reshape: {e}")))?;

        let h_vec: Vec<f64> = read_binary_array(
            &mut archive,
            &format!("two_view_geometries/h_matrices.{pair_count}.3.3.float64.zst"),
            pair_count * 9,
        )?;
        let h_matrices = Array3::from_shape_vec((pair_count, 3, 3), h_vec)
            .map_err(|e| MatchesError::ShapeMismatch(format!("h_matrices reshape: {e}")))?;

        // Quaternions and translations
        let quat_vec: Vec<f64> = read_binary_array(
            &mut archive,
            &format!("two_view_geometries/quaternions_wxyz.{pair_count}.4.float64.zst"),
            pair_count * 4,
        )?;
        let quaternions_wxyz = Array2::from_shape_vec((pair_count, 4), quat_vec)
            .map_err(|e| MatchesError::ShapeMismatch(format!("quaternions reshape: {e}")))?;

        let trans_vec: Vec<f64> = read_binary_array(
            &mut archive,
            &format!("two_view_geometries/translations_xyz.{pair_count}.3.float64.zst"),
            pair_count * 3,
        )?;
        let translations_xyz = Array2::from_shape_vec((pair_count, 3), trans_vec)
            .map_err(|e| MatchesError::ShapeMismatch(format!("translations reshape: {e}")))?;

        Some(TwoViewGeometryData {
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
    } else {
        None
    };

    Ok(MatchesData {
        metadata,
        content_hash,
        image_names,
        feature_tool_hashes,
        sift_content_hashes,
        feature_counts,
        image_index_pairs,
        match_counts,
        match_feature_indexes,
        match_descriptor_distances,
        two_view_geometries,
    })
}
