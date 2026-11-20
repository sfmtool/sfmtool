// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.matches` file reading.

use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3, Array4};

use crate::cells::{
    cell_entry_errors, normalize_cell_statuses, read_member_cell_status_legend, MemberCellData,
};
use crate::entries;
use crate::types::*;
use crate::validate::{check_backbone_counts, content_errors};
use sfmtool_archive_io::{read_json_entry, DecodedEntries};

fn open_archive(path: &Path) -> Result<zip::ZipArchive<std::fs::File>, MatchesError> {
    let file = std::fs::File::open(path).map_err(|e| MatchesError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(zip::ZipArchive::new(file)?)
}

/// Refuse a `metadata.version` this build does not read: one below
/// [`MATCHES_OLDEST_READ_VERSION`] that a release wrote, with
/// [`MatchesError::VersionNoLongerRead`], which names `path` and the last
/// release that reads it; and one of 0, which no writer has ever produced, or
/// newer than [`MATCHES_FORMAT_VERSION`], whose layout is unknown so reading it
/// with current-version assumptions would misparse silently.
fn check_version(version: u32, path: &Path) -> Result<(), MatchesError> {
    if (1..MATCHES_OLDEST_READ_VERSION).contains(&version) {
        return Err(MatchesError::VersionNoLongerRead {
            path: path.to_path_buf(),
            version,
        });
    }
    if !(MATCHES_OLDEST_READ_VERSION..=MATCHES_FORMAT_VERSION).contains(&version) {
        return Err(MatchesError::InvalidFormat(format!(
            "unsupported .matches format version {version} (this build reads versions \
             {MATCHES_OLDEST_READ_VERSION} to {MATCHES_FORMAT_VERSION})"
        )));
    }
    Ok(())
}

/// The keys a file's `metadata.json` holds that [`MatchesMetadata`] fills in
/// when a value built in memory leaves them out.
const REQUIRED_DEFAULTED_KEYS: [&str; 2] = ["has_clusters", "has_cluster_patches"];

/// Parse `metadata.json` from its JSON value, checking its version with
/// [`check_version`] first, so that a file of a version this build does not
/// read is refused for its version rather than for a key that version lacks,
/// and then that it holds every key a file requires.
pub(crate) fn parse_metadata(
    value: serde_json::Value,
    path: &Path,
) -> Result<MatchesMetadata, MatchesError> {
    if let Some(version) = value.get("version").and_then(serde_json::Value::as_u64) {
        check_version(u32::try_from(version).unwrap_or(u32::MAX), path)?;
    }
    if let Some(key) = REQUIRED_DEFAULTED_KEYS
        .iter()
        .find(|key| value.get(**key).is_none())
    {
        return Err(MatchesError::InvalidFormat(format!(
            "metadata.json.zst has no {key}"
        )));
    }
    Ok(serde_json::from_value(value)?)
}

/// Read only the top-level metadata from a `.matches` file (fast, no binary data).
///
/// A file of a version this build does not read is refused, as
/// [`read_matches`] refuses it.
pub fn read_matches_metadata(path: &Path) -> Result<MatchesMetadata, MatchesError> {
    let mut archive = open_archive(path)?;
    parse_metadata(read_json_entry(&mut archive, entries::metadata())?, path)
}

/// The workspace directory a `.matches` file records, or `None` when none of
/// the three lookups finds one.
///
/// The rule is [`sfmtool_archive_io::resolve_workspace_dir`], the one the
/// `.sfmr` reader and `.kdf` source verification use:
/// 1. `workspace.relative_path` joined to the `.matches` file's directory;
/// 2. `workspace.absolute_path`;
/// 3. the nearest directory at or above the `.matches` file's directory that
///    holds `.sfm-workspace.json`.
///
/// A recorded directory counts only when it holds `.sfm-workspace.json`. An
/// empty `relative_path` or `absolute_path` means none was recorded, and its
/// step is skipped. A file that names no workspace is still a valid file, so
/// `None` is not an error here; each caller that needs the workspace reports
/// its absence in its own terms.
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use sfmtool_matches_format::{read_matches_metadata, resolve_workspace_dir};
/// let path = Path::new("/projects/ws/matches/pairs.matches");
/// let metadata = read_matches_metadata(path)?;
/// let workspace = resolve_workspace_dir(path, &metadata);
/// # Ok::<(), sfmtool_matches_format::MatchesError>(())
/// ```
pub fn resolve_workspace_dir(matches_path: &Path, metadata: &MatchesMetadata) -> Option<PathBuf> {
    sfmtool_archive_io::resolve_workspace_dir(
        matches_path,
        &metadata.workspace.relative_path,
        &metadata.workspace.absolute_path,
    )
}

/// Read the top-level metadata and the image names of a `.matches` file, and
/// no binary section.
///
/// For a caller that needs to know which images a file covers, in which order,
/// without decompressing its correspondences: two JSON entries rather than the
/// whole archive. The version is checked, and the names against the
/// metadata's `image_count` and against the image-name rule, as
/// [`read_matches`] checks them.
pub fn read_matches_image_names(
    path: &Path,
) -> Result<(MatchesMetadata, Vec<String>), MatchesError> {
    let mut archive = open_archive(path)?;
    let metadata = parse_metadata(read_json_entry(&mut archive, entries::metadata())?, path)?;
    let image_names: Vec<String> = read_json_entry(&mut archive, entries::images_names())?;
    if image_names.len() != metadata.image_count as usize {
        return Err(MatchesError::ShapeMismatch(format!(
            "image names count {} != image_count {}",
            image_names.len(),
            metadata.image_count
        )));
    }
    sfmtool_archive_io::validate_image_names(&image_names).map_err(MatchesError::InvalidFormat)?;
    Ok((metadata, image_names))
}

/// Read a complete `.matches` file into columnar data.
///
/// The file stores exactly one correspondence backbone: `image_pairs/`
/// (pairwise) or `clusters/` (indicated by the metadata `has_clusters` flag).
/// The optional `cluster_patches/` section requires `clusters/`; the optional
/// `two_view_geometries/` section requires `image_pairs/`.
///
/// A file of a version below [`MATCHES_OLDEST_READ_VERSION`] is refused with
/// [`MatchesError::VersionNoLongerRead`], and one of version 0 or newer than
/// [`MATCHES_FORMAT_VERSION`] as unsupported, before anything that depends on
/// the version is parsed.
///
/// `cluster_patches/member_status` codes are resolved through the file's
/// `member_status_names` legend and returned in the canonical
/// [`ClusterMemberStatus`] numbering. A code past the legend's end, or a
/// legend naming a status the format does not define, is rejected.
///
/// The data read is held to the rules [`crate::write_matches`] holds its input
/// to: every `image_dims` value at least 1, pairs sorted and within the image
/// count, match and inlier counts that sum to their totals, every feature
/// index within its image's feature count, every inlier one of its pair's
/// candidate matches, every config index naming a config type, a well-formed
/// `cluster_starts`, members within the image and feature counts, member
/// geometry free of `NaN`, and each cluster's reference and kept members as
/// the format defines them. A file that breaks one is refused with the first
/// finding as [`MatchesError::InvalidFormat`].
pub fn read_matches(path: &Path) -> Result<MatchesData, MatchesError> {
    let mut archive = open_archive(path)?;
    // A full read consumes every entry the file holds, so they are all
    // decompressed in one batch rather than one at a time; the section readers
    // below then look up the entries this file's metadata says exist.
    let decoded = DecodedEntries::read_all(&mut archive)?;

    // Top-level metadata
    let metadata = parse_metadata(decoded.json_entry(entries::metadata())?, path)?;
    let content_hash: MatchesContentHash = decoded.json_entry(entries::content_hash())?;

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

    let image_count = metadata.image_count as usize;

    // Cross-check images section metadata
    let images_meta: serde_json::Value = decoded.json_entry(entries::images_metadata())?;
    if images_meta.get("image_count").and_then(|v| v.as_u64()) != Some(image_count as u64) {
        return Err(MatchesError::InvalidFormat(
            "images/metadata.json.zst image_count doesn't match top-level metadata".into(),
        ));
    }

    // === Images ===
    let image_names: Vec<String> = decoded.json_entry(entries::images_names())?;
    if image_names.len() != image_count {
        return Err(MatchesError::ShapeMismatch(format!(
            "image names count {} != image_count {image_count}",
            image_names.len()
        )));
    }

    let feature_tool_hashes = decoded.uint128_array(
        &entries::images_feature_tool_hashes(image_count),
        image_count,
    )?;

    let sift_content_hashes = decoded.uint128_array(
        &entries::images_sift_content_hashes(image_count),
        image_count,
    )?;

    let feature_counts_vec: Vec<u32> =
        decoded.binary_array(&entries::images_feature_counts(image_count), image_count)?;
    let feature_counts = Array1::from_vec(feature_counts_vec);

    let dims_vec: Vec<u32> =
        decoded.binary_array(&entries::images_image_dims(image_count), image_count * 2)?;
    let image_dims = Array2::from_shape_vec((image_count, 2), dims_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("image_dims reshape: {e}")))?;

    // === Backbone: image pairs XOR clusters ===
    // Historically the reader rejected forbidden counts before missing
    // required counts, while the writer and verifier report required counts
    // first. Pop from the shared, required-first list to preserve that reader
    // behavior when malformed metadata violates both halves at once.
    if let Some(error) = check_backbone_counts(&metadata).pop() {
        return Err(MatchesError::InvalidFormat(error));
    }
    let image_pairs = if metadata.has_clusters {
        None
    } else {
        Some(read_pairs_section(&decoded, &metadata)?)
    };

    let clusters = if metadata.has_clusters {
        Some(read_clusters_section(&decoded, &metadata)?)
    } else {
        None
    };

    // === Cluster patches (optional) ===
    let cluster_patches = if metadata.has_cluster_patches {
        Some(read_cluster_patches_section(&decoded, &metadata)?)
    } else {
        None
    };

    // === Two-view geometries (optional) ===
    let two_view_geometries = if metadata.has_two_view_geometries {
        let pair_count = metadata.image_pair_count.unwrap_or(0) as usize;
        Some(read_tvg_section(&decoded, pair_count)?)
    } else {
        None
    };

    let data = MatchesData {
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
    };
    // The content rules the writer holds a file to. Only damage or a hand
    // edit makes a file that breaks one, and refusing it here keeps its
    // out-of-range values from the code that indexes by them.
    if let Some(error) = content_errors(&data).into_iter().next() {
        return Err(MatchesError::InvalidFormat(error));
    }
    Ok(data)
}

fn read_pairs_section(
    decoded: &DecodedEntries,
    metadata: &MatchesMetadata,
) -> Result<PairsData, MatchesError> {
    // `read_matches` checks the backbone count invariant before dispatching.
    let pair_count = metadata
        .image_pair_count
        .expect("backbone counts validated") as usize;
    let match_count = metadata.match_count.expect("backbone counts validated") as usize;

    // Cross-check pairs section metadata
    let pairs_meta: serde_json::Value = decoded.json_entry(entries::image_pairs_metadata())?;
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

    let image_index_pairs_vec: Vec<u32> = decoded.binary_array(
        &entries::image_pairs_image_index_pairs(pair_count),
        pair_count * 2,
    )?;
    let image_index_pairs = Array2::from_shape_vec((pair_count, 2), image_index_pairs_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("image_index_pairs reshape: {e}")))?;

    let match_counts_vec: Vec<u32> =
        decoded.binary_array(&entries::image_pairs_match_counts(pair_count), pair_count)?;
    let match_counts = Array1::from_vec(match_counts_vec);

    let match_feature_indexes_vec: Vec<u32> = decoded.binary_array(
        &entries::image_pairs_match_feature_indexes(match_count),
        match_count * 2,
    )?;
    let match_feature_indexes = Array2::from_shape_vec((match_count, 2), match_feature_indexes_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("match_feature_indexes reshape: {e}")))?;

    let match_descriptor_distances_vec: Vec<f32> = decoded.binary_array(
        &entries::image_pairs_match_descriptor_distances(match_count),
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

fn read_clusters_section(
    decoded: &DecodedEntries,
    metadata: &MatchesMetadata,
) -> Result<ClustersData, MatchesError> {
    // `read_matches` checks the backbone count invariant before dispatching.
    let cluster_count = metadata.cluster_count.expect("backbone counts validated") as usize;
    let member_count = metadata
        .cluster_member_count
        .expect("backbone counts validated") as usize;

    // Cross-check clusters section metadata
    let clusters_meta: serde_json::Value = decoded.json_entry(entries::clusters_metadata())?;
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

    let offset_count = cluster_count + 1;
    let cluster_starts_vec: Vec<u32> = decoded.binary_array(
        &entries::clusters_cluster_starts(offset_count),
        offset_count,
    )?;
    let cluster_starts = Array1::from_vec(cluster_starts_vec);

    let member_images_vec: Vec<u32> =
        decoded.binary_array(&entries::clusters_member_images(member_count), member_count)?;
    let member_images = Array1::from_vec(member_images_vec);

    let member_features_vec: Vec<u32> = decoded.binary_array(
        &entries::clusters_member_features(member_count),
        member_count,
    )?;
    let member_features = Array1::from_vec(member_features_vec);

    // The members' geometry.
    let positions_vec: Vec<f32> = decoded.binary_array(
        &entries::clusters_member_positions(member_count),
        member_count * 2,
    )?;
    let member_positions = Array2::from_shape_vec((member_count, 2), positions_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("member_positions reshape: {e}")))?;

    let shapes_vec: Vec<f32> = decoded.binary_array(
        &entries::clusters_member_affine_shapes(member_count),
        member_count * 4,
    )?;
    let member_affine_shapes = Array3::from_shape_vec((member_count, 2, 2), shapes_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("member_affine_shapes reshape: {e}")))?;
    let (member_positions, member_affine_shapes) =
        (Some(member_positions), Some(member_affine_shapes));

    Ok(ClustersData {
        cluster_starts,
        member_images,
        member_features,
        member_positions,
        member_affine_shapes,
        matcher_options,
    })
}

/// Read `cluster_patches/`, resolving `member_status` through the file's legend.
fn read_cluster_patches_section(
    decoded: &DecodedEntries,
    metadata: &MatchesMetadata,
) -> Result<ClusterPatchData, MatchesError> {
    let cluster_count = metadata.cluster_count.unwrap_or(0) as usize;
    let member_count = metadata.cluster_member_count.unwrap_or(0) as usize;

    // Cross-check cluster_patches section metadata
    let cp_meta: serde_json::Value = decoded.json_entry(entries::cluster_patches_metadata())?;
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
    let status_legend = read_member_status_legend(&cp_meta).map_err(MatchesError::InvalidFormat)?;

    let reference_members_vec: Vec<u32> = decoded.binary_array(
        &entries::cluster_patches_reference_members(cluster_count),
        cluster_count,
    )?;
    let reference_members = Array1::from_vec(reference_members_vec);

    let mut member_status_vec: Vec<u8> = decoded.binary_array(
        &entries::cluster_patches_member_status(member_count),
        member_count,
    )?;
    // The file's own numbering stops here: everything above the reader holds
    // the canonical `ClusterMemberStatus` discriminants.
    normalize_member_statuses(&mut member_status_vec, &status_legend)
        .map_err(MatchesError::InvalidFormat)?;
    let member_status = Array1::from_vec(member_status_vec);

    let member_zncc_vec: Vec<f32> = decoded.binary_array(
        &entries::cluster_patches_member_zncc(member_count),
        member_count,
    )?;
    let member_zncc = Array1::from_vec(member_zncc_vec);

    let member_shift_px_vec: Vec<f32> = decoded.binary_array(
        &entries::cluster_patches_member_shift_px(member_count),
        member_count,
    )?;
    let member_shift_px = Array1::from_vec(member_shift_px_vec);

    let member_consistency_vec: Vec<f32> = decoded.binary_array(
        &entries::cluster_patches_member_consistency_residual(member_count),
        member_count,
    )?;
    let member_consistency_residual = Array1::from_vec(member_consistency_vec);

    let member_cells = read_member_cells(decoded, &cp_meta, member_count)?;

    Ok(ClusterPatchData {
        reference_members,
        member_status,
        member_zncc,
        member_shift_px,
        member_consistency_residual,
        member_cells,
        refine_options,
    })
}

/// Read the optional per-cell columns of `cluster_patches/`, resolving
/// `member_cell_status` through the file's own `member_cell_status_names`
/// legend.
///
/// The legend and the four columns are present together or absent together;
/// a file with neither has no cells and reads as `None`.
fn read_member_cells(
    decoded: &DecodedEntries,
    cp_meta: &serde_json::Value,
    member_count: usize,
) -> Result<Option<MemberCellData>, MatchesError> {
    let legend = read_member_cell_status_legend(cp_meta).map_err(MatchesError::InvalidFormat)?;
    if let Some(error) = cell_entry_errors(decoded.names(), member_count, legend.is_some())
        .into_iter()
        .next()
    {
        return Err(MatchesError::InvalidFormat(error));
    }
    let Some(legend) = legend else {
        return Ok(None);
    };

    let iterations: Vec<u8> = decoded.binary_array(
        &entries::cluster_patches_member_cell_iterations(member_count),
        member_count,
    )?;
    let shift_px: Vec<f32> = decoded.binary_array(
        &entries::cluster_patches_member_cell_shift_px(member_count),
        member_count * 18,
    )?;
    let mut status: Vec<u8> = decoded.binary_array(
        &entries::cluster_patches_member_cell_status(member_count),
        member_count * 9,
    )?;
    let zncc: Vec<f32> = decoded.binary_array(
        &entries::cluster_patches_member_cell_zncc(member_count),
        member_count * 9,
    )?;
    // As for member_status, the file's own numbering stops here.
    normalize_cell_statuses(&mut status, &legend).map_err(MatchesError::InvalidFormat)?;
    let reshape = |name: &str, e: ndarray::ShapeError| {
        MatchesError::ShapeMismatch(format!("{name} reshape: {e}"))
    };
    let cells = MemberCellData {
        shift_px: Array4::from_shape_vec((member_count, 3, 3, 2), shift_px)
            .map_err(|e| reshape("member_cell_shift_px", e))?,
        zncc: Array3::from_shape_vec((member_count, 3, 3), zncc)
            .map_err(|e| reshape("member_cell_zncc", e))?,
        status: Array3::from_shape_vec((member_count, 3, 3), status)
            .map_err(|e| reshape("member_cell_status", e))?,
        iterations: Array1::from_vec(iterations),
    };
    Ok(Some(cells))
}

fn read_tvg_section(
    decoded: &DecodedEntries,
    pair_count: usize,
) -> Result<TwoViewGeometryData, MatchesError> {
    let tvg_metadata: TvgMetadata = decoded.json_entry(entries::two_view_geometries_metadata())?;
    let inlier_count = tvg_metadata.inlier_count as usize;

    if tvg_metadata.image_pair_count as usize != pair_count {
        return Err(MatchesError::InvalidFormat(format!(
            "TVG image_pair_count {} != pair_count {pair_count}",
            tvg_metadata.image_pair_count
        )));
    }

    // config_types.json
    let config_type_strings: Vec<String> =
        decoded.json_entry(entries::two_view_geometries_config_types())?;
    let config_types: Vec<TwoViewGeometryConfig> = config_type_strings
        .iter()
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;

    // config_indexes
    let config_indexes_vec: Vec<u8> = decoded.binary_array(
        &entries::two_view_geometries_config_indexes(pair_count),
        pair_count,
    )?;
    let config_indexes = Array1::from_vec(config_indexes_vec);

    // inlier_counts
    let inlier_counts_vec: Vec<u32> = decoded.binary_array(
        &entries::two_view_geometries_inlier_counts(pair_count),
        pair_count,
    )?;
    let inlier_counts = Array1::from_vec(inlier_counts_vec);

    // inlier_feature_indexes
    let inlier_fi_vec: Vec<u32> = decoded.binary_array(
        &entries::two_view_geometries_inlier_feature_indexes(inlier_count),
        inlier_count * 2,
    )?;
    let inlier_feature_indexes = Array2::from_shape_vec((inlier_count, 2), inlier_fi_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("inlier_feature_indexes reshape: {e}")))?;

    // Matrices
    let f_vec: Vec<f64> = decoded.binary_array(
        &entries::two_view_geometries_f_matrices(pair_count),
        pair_count * 9,
    )?;
    let f_matrices = Array3::from_shape_vec((pair_count, 3, 3), f_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("f_matrices reshape: {e}")))?;

    let e_vec: Vec<f64> = decoded.binary_array(
        &entries::two_view_geometries_e_matrices(pair_count),
        pair_count * 9,
    )?;
    let e_matrices = Array3::from_shape_vec((pair_count, 3, 3), e_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("e_matrices reshape: {e}")))?;

    let h_vec: Vec<f64> = decoded.binary_array(
        &entries::two_view_geometries_h_matrices(pair_count),
        pair_count * 9,
    )?;
    let h_matrices = Array3::from_shape_vec((pair_count, 3, 3), h_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("h_matrices reshape: {e}")))?;

    // Quaternions and translations
    let quat_vec: Vec<f64> = decoded.binary_array(
        &entries::two_view_geometries_quaternions_wxyz(pair_count),
        pair_count * 4,
    )?;
    let quaternions_wxyz = Array2::from_shape_vec((pair_count, 4), quat_vec)
        .map_err(|e| MatchesError::ShapeMismatch(format!("quaternions reshape: {e}")))?;

    let trans_vec: Vec<f64> = decoded.binary_array(
        &entries::two_view_geometries_translations_xyz(pair_count),
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
