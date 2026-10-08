// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `.matches` file I/O.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

use sfmtool_matches_format::{
    ClusterPatchData, ClustersData, MatchesContentHash, MatchesData, MatchesMetadata,
    MemberCellData, PairsData, TvgMetadata, TwoViewGeometryConfig, TwoViewGeometryData,
};

use crate::helpers::{
    dtype_name, get_item, get_optional_item, py_to_serde, py_to_u128_bytes, serde_to_py,
    u128_bytes_to_py,
};

/// Convert MatchesData to a Python dict.
pub fn matches_data_to_py(py: Python<'_>, data: MatchesData) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    dict.set_item("metadata", serde_to_py(py, &data.metadata)?)?;
    dict.set_item("content_hash", serde_to_py(py, &data.content_hash)?)?;
    dict.set_item("image_names", &data.image_names)?;
    dict.set_item(
        "feature_tool_hashes",
        u128_bytes_to_py(py, &data.feature_tool_hashes)?,
    )?;
    dict.set_item(
        "sift_content_hashes",
        u128_bytes_to_py(py, &data.sift_content_hashes)?,
    )?;
    dict.set_item("feature_counts", data.feature_counts.into_pyarray(py))?;
    // Present for every version-4+ file (the writer requires it); absent
    // only when loading a version <= 3 file, which never stored dims.
    if let Some(image_dims) = data.image_dims {
        dict.set_item("image_dims", image_dims.into_pyarray(py))?;
    }

    if let Some(pairs) = data.image_pairs {
        dict.set_item(
            "image_index_pairs",
            pairs.image_index_pairs.into_pyarray(py),
        )?;
        dict.set_item("match_counts", pairs.match_counts.into_pyarray(py))?;
        dict.set_item(
            "match_feature_indexes",
            pairs.match_feature_indexes.into_pyarray(py),
        )?;
        dict.set_item(
            "match_descriptor_distances",
            pairs.match_descriptor_distances.into_pyarray(py),
        )?;
    }

    if let Some(clusters) = data.clusters {
        dict.set_item("has_clusters", true)?;
        dict.set_item("cluster_starts", clusters.cluster_starts.into_pyarray(py))?;
        dict.set_item("member_images", clusters.member_images.into_pyarray(py))?;
        dict.set_item("member_features", clusters.member_features.into_pyarray(py))?;
        // The backbone's stage geometry; present for every version-6+ file
        // (the writer requires it), absent for a version <= 5 file.
        if let Some(member_positions) = clusters.member_positions {
            dict.set_item("member_positions", member_positions.into_pyarray(py))?;
        }
        if let Some(member_affine_shapes) = clusters.member_affine_shapes {
            dict.set_item(
                "member_affine_shapes",
                member_affine_shapes.into_pyarray(py),
            )?;
        }
        dict.set_item(
            "matcher_options",
            serde_to_py(py, &clusters.matcher_options)?,
        )?;
    } else {
        dict.set_item("has_clusters", false)?;
    }

    if let Some(cp) = data.cluster_patches {
        dict.set_item("has_cluster_patches", true)?;
        dict.set_item("reference_members", cp.reference_members.into_pyarray(py))?;
        dict.set_item("member_status", cp.member_status.into_pyarray(py))?;
        dict.set_item("member_zncc", cp.member_zncc.into_pyarray(py))?;
        dict.set_item("member_shift_px", cp.member_shift_px.into_pyarray(py))?;
        dict.set_item(
            "member_consistency_residual",
            cp.member_consistency_residual.into_pyarray(py),
        )?;
        // The piecewise refinement's per-cell columns (format version 8),
        // present together when the file carries them, absent otherwise.
        if let Some(cells) = cp.member_cells {
            dict.set_item("member_cell_shift_px", cells.shift_px.into_pyarray(py))?;
            dict.set_item("member_cell_zncc", cells.zncc.into_pyarray(py))?;
            dict.set_item("member_cell_status", cells.status.into_pyarray(py))?;
            dict.set_item("member_cell_iterations", cells.iterations.into_pyarray(py))?;
        }
        dict.set_item("refine_options", serde_to_py(py, &cp.refine_options)?)?;
    } else {
        dict.set_item("has_cluster_patches", false)?;
    }

    if let Some(tvg) = data.two_view_geometries {
        dict.set_item("has_two_view_geometries", true)?;
        dict.set_item("tvg_metadata", serde_to_py(py, &tvg.metadata)?)?;
        let config_strs: Vec<&str> = tvg.config_types.iter().map(|c| c.as_str()).collect();
        dict.set_item("config_types", config_strs)?;
        dict.set_item("config_indexes", tvg.config_indexes.into_pyarray(py))?;
        dict.set_item("inlier_counts", tvg.inlier_counts.into_pyarray(py))?;
        dict.set_item(
            "inlier_feature_indexes",
            tvg.inlier_feature_indexes.into_pyarray(py),
        )?;
        dict.set_item("f_matrices", tvg.f_matrices.into_pyarray(py))?;
        dict.set_item("e_matrices", tvg.e_matrices.into_pyarray(py))?;
        dict.set_item("h_matrices", tvg.h_matrices.into_pyarray(py))?;
        dict.set_item("quaternions_wxyz", tvg.quaternions_wxyz.into_pyarray(py))?;
        dict.set_item("translations_xyz", tvg.translations_xyz.into_pyarray(py))?;
    } else {
        dict.set_item("has_two_view_geometries", false)?;
    }

    Ok(dict.into())
}

/// Read a complete .matches file, returning a dict with numpy arrays and metadata.
#[pyfunction]
pub fn read_matches(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let data = sfmtool_matches_format::read_matches(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    matches_data_to_py(py, data)
}

/// Read only metadata from a .matches file (fast, no binary data).
#[pyfunction]
pub fn read_matches_metadata(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let metadata = sfmtool_matches_format::read_matches_metadata(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    serde_to_py(py, &metadata)
}

/// The per-cell columns of a `write_matches` dict: the four
/// `member_cell_*` keys together, or none of them (a missing key and `None`
/// alike).
fn member_cells_from_py(data: &Bound<'_, PyDict>) -> PyResult<Option<MemberCellData>> {
    let keys = [
        "member_cell_shift_px",
        "member_cell_zncc",
        "member_cell_status",
        "member_cell_iterations",
    ];
    let items = keys
        .iter()
        .map(|key| get_optional_item(data, key))
        .collect::<PyResult<Vec<_>>>()?;
    let present = items.iter().filter(|item| item.is_some()).count();
    if present == 0 {
        return Ok(None);
    }
    if present != keys.len() {
        let missing: Vec<&str> = keys
            .iter()
            .zip(&items)
            .filter(|(_, item)| item.is_none())
            .map(|(key, _)| *key)
            .collect();
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "the member_cell_* columns are written together; missing {missing:?}"
        )));
    }
    let item = |i: usize| items[i].as_ref().expect("checked present");
    // A column of the wrong dtype or rank names its key, so the caller can
    // tell which of the four it handed over wrong.
    let wrong_type = |i: usize, expected: &str| {
        let actual = dtype_name(item(i)).unwrap_or_else(|_| "unknown".to_string());
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{} must be a {expected} array, got dtype {actual}",
            keys[i]
        ))
    };
    let shift_px: PyReadonlyArray4<f32> = item(0)
        .extract()
        .map_err(|_| wrong_type(0, "(M, 3, 3, 2) float32"))?;
    let zncc: PyReadonlyArray3<f32> = item(1)
        .extract()
        .map_err(|_| wrong_type(1, "(M, 3, 3) float32"))?;
    let status: PyReadonlyArray3<u8> = item(2)
        .extract()
        .map_err(|_| wrong_type(2, "(M, 3, 3) uint8"))?;
    let iterations: PyReadonlyArray1<u8> =
        item(3).extract().map_err(|_| wrong_type(3, "(M,) uint8"))?;
    Ok(Some(MemberCellData {
        shift_px: shift_px.as_array().as_standard_layout().into_owned(),
        zncc: zncc.as_array().as_standard_layout().into_owned(),
        status: status.as_array().as_standard_layout().into_owned(),
        iterations: iterations.as_array().as_standard_layout().into_owned(),
    }))
}

/// Extract an optional boolean flag from the dict; a missing key or an
/// explicit `None` both mean `false` (so pre-cluster pairwise dicts keep
/// working unchanged).
fn get_flag(data: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    Ok(match get_optional_item(data, key)? {
        Some(v) => v.extract()?,
        None => false,
    })
}

/// Write a .matches file from a dict of numpy arrays and metadata.
///
/// The dict should have the same keys as returned by `read_matches`.
/// The `content_hash` key is ignored (recomputed on write).
#[pyfunction]
#[pyo3(signature = (path, data, zstd_level=3))]
pub fn write_matches(
    py: Python<'_>,
    path: PathBuf,
    data: &Bound<'_, PyDict>,
    zstd_level: i32,
) -> PyResult<()> {
    let metadata: MatchesMetadata = py_to_serde(py, &get_item(data, "metadata")?)?;

    let image_names: Vec<String> = get_item(data, "image_names")?.extract()?;
    let feature_tool_hashes = py_to_u128_bytes(&get_item(data, "feature_tool_hashes")?)?;
    let sift_content_hashes = py_to_u128_bytes(&get_item(data, "sift_content_hashes")?)?;
    let feature_counts: PyReadonlyArray1<u32> = get_item(data, "feature_counts")?.extract()?;
    // Mandatory since format version 4: (N, 2) per-image width/height.
    let image_dims: PyReadonlyArray2<u32> = get_item(data, "image_dims")?.extract()?;

    // Backbone: clusters when the has_clusters flag is set, pairs otherwise.
    let has_clusters = get_flag(data, "has_clusters")?;
    let (image_pairs, clusters) = if has_clusters {
        let cluster_starts: PyReadonlyArray1<u32> = get_item(data, "cluster_starts")?.extract()?;
        let member_images: PyReadonlyArray1<u32> = get_item(data, "member_images")?.extract()?;
        let member_features: PyReadonlyArray1<u32> =
            get_item(data, "member_features")?.extract()?;
        let matcher_options: serde_json::Value =
            py_to_serde(py, &get_item(data, "matcher_options")?)?;
        // The backbone's stage geometry, mandatory since format version 6.
        // Extracted optionally so a dict read back from a version <= 5 file
        // reaches `write_matches`' own message naming the regeneration, rather
        // than a bare KeyError.
        let member_positions = match get_optional_item(data, "member_positions")? {
            Some(v) => {
                let arr: PyReadonlyArray2<f32> = v.extract()?;
                Some(arr.as_array().as_standard_layout().into_owned())
            }
            None => None,
        };
        let member_affine_shapes = match get_optional_item(data, "member_affine_shapes")? {
            Some(v) => {
                let arr: PyReadonlyArray3<f32> = v.extract()?;
                Some(arr.as_array().as_standard_layout().into_owned())
            }
            None => None,
        };
        (
            None,
            Some(ClustersData {
                cluster_starts: cluster_starts.as_array().as_standard_layout().into_owned(),
                member_images: member_images.as_array().as_standard_layout().into_owned(),
                member_features: member_features.as_array().as_standard_layout().into_owned(),
                member_positions,
                member_affine_shapes,
                matcher_options,
            }),
        )
    } else {
        let image_index_pairs: PyReadonlyArray2<u32> =
            get_item(data, "image_index_pairs")?.extract()?;
        let match_counts: PyReadonlyArray1<u32> = get_item(data, "match_counts")?.extract()?;
        let match_feature_indexes: PyReadonlyArray2<u32> =
            get_item(data, "match_feature_indexes")?.extract()?;
        let match_descriptor_distances: PyReadonlyArray1<f32> =
            get_item(data, "match_descriptor_distances")?.extract()?;
        (
            Some(PairsData {
                image_index_pairs: image_index_pairs
                    .as_array()
                    .as_standard_layout()
                    .into_owned(),
                match_counts: match_counts.as_array().as_standard_layout().into_owned(),
                match_feature_indexes: match_feature_indexes
                    .as_array()
                    .as_standard_layout()
                    .into_owned(),
                match_descriptor_distances: match_descriptor_distances
                    .as_array()
                    .as_standard_layout()
                    .into_owned(),
            }),
            None,
        )
    };

    // Cluster patches (optional, requires clusters)
    let cluster_patches = if get_flag(data, "has_cluster_patches")? {
        let reference_members: PyReadonlyArray1<u32> =
            get_item(data, "reference_members")?.extract()?;
        let member_status: PyReadonlyArray1<u8> = get_item(data, "member_status")?.extract()?;
        let member_zncc: PyReadonlyArray1<f32> = get_item(data, "member_zncc")?.extract()?;
        let member_shift_px: PyReadonlyArray1<f32> =
            get_item(data, "member_shift_px")?.extract()?;
        let member_consistency_residual: PyReadonlyArray1<f32> =
            get_item(data, "member_consistency_residual")?.extract()?;
        let refine_options: serde_json::Value =
            py_to_serde(py, &get_item(data, "refine_options")?)?;
        let member_cells = member_cells_from_py(data)?;
        Some(ClusterPatchData {
            reference_members: reference_members
                .as_array()
                .as_standard_layout()
                .into_owned(),
            member_status: member_status.as_array().as_standard_layout().into_owned(),
            member_zncc: member_zncc.as_array().as_standard_layout().into_owned(),
            member_shift_px: member_shift_px.as_array().as_standard_layout().into_owned(),
            member_consistency_residual: member_consistency_residual
                .as_array()
                .as_standard_layout()
                .into_owned(),
            member_cells,
            refine_options,
        })
    } else {
        None
    };

    // TVG
    let has_tvg: bool = get_item(data, "has_two_view_geometries")?.extract()?;
    let two_view_geometries = if has_tvg {
        let tvg_metadata: TvgMetadata = py_to_serde(py, &get_item(data, "tvg_metadata")?)?;
        let config_type_strs: Vec<String> = get_item(data, "config_types")?.extract()?;
        let config_types: Vec<TwoViewGeometryConfig> = config_type_strs
            .iter()
            .map(|s| {
                s.parse()
                    .map_err(|e: sfmtool_matches_format::MatchesError| {
                        pyo3::exceptions::PyValueError::new_err(e.to_string())
                    })
            })
            .collect::<PyResult<_>>()?;
        let config_indexes: PyReadonlyArray1<u8> = get_item(data, "config_indexes")?.extract()?;
        let inlier_counts: PyReadonlyArray1<u32> = get_item(data, "inlier_counts")?.extract()?;
        let inlier_feature_indexes: PyReadonlyArray2<u32> =
            get_item(data, "inlier_feature_indexes")?.extract()?;
        let f_matrices: PyReadonlyArray3<f64> = get_item(data, "f_matrices")?.extract()?;
        let e_matrices: PyReadonlyArray3<f64> = get_item(data, "e_matrices")?.extract()?;
        let h_matrices: PyReadonlyArray3<f64> = get_item(data, "h_matrices")?.extract()?;
        let quaternions_wxyz: PyReadonlyArray2<f64> =
            get_item(data, "quaternions_wxyz")?.extract()?;
        let translations_xyz: PyReadonlyArray2<f64> =
            get_item(data, "translations_xyz")?.extract()?;

        Some(TwoViewGeometryData {
            metadata: tvg_metadata,
            config_types,
            config_indexes: config_indexes.as_array().as_standard_layout().into_owned(),
            inlier_counts: inlier_counts.as_array().as_standard_layout().into_owned(),
            inlier_feature_indexes: inlier_feature_indexes
                .as_array()
                .as_standard_layout()
                .into_owned(),
            f_matrices: f_matrices.as_array().as_standard_layout().into_owned(),
            e_matrices: e_matrices.as_array().as_standard_layout().into_owned(),
            h_matrices: h_matrices.as_array().as_standard_layout().into_owned(),
            quaternions_wxyz: quaternions_wxyz
                .as_array()
                .as_standard_layout()
                .into_owned(),
            translations_xyz: translations_xyz
                .as_array()
                .as_standard_layout()
                .into_owned(),
        })
    } else {
        None
    };

    let matches_data = MatchesData {
        metadata,
        content_hash: MatchesContentHash {
            metadata_xxh128: String::new(),
            images_xxh128: String::new(),
            image_pairs_xxh128: None,
            clusters_xxh128: None,
            cluster_patches_xxh128: None,
            two_view_geometries_xxh128: None,
            content_xxh128: String::new(),
        },
        image_names,
        feature_tool_hashes,
        sift_content_hashes,
        feature_counts: feature_counts.as_array().as_standard_layout().into_owned(),
        image_dims: Some(image_dims.as_array().as_standard_layout().into_owned()),
        image_pairs,
        clusters,
        cluster_patches,
        two_view_geometries,
    };

    sfmtool_matches_format::write_matches(&path, &matches_data, zstd_level)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Verify integrity of a .matches file.
///
/// Returns a tuple (is_valid, error_messages).
#[pyfunction]
pub fn verify_matches(path: PathBuf) -> PyResult<(bool, Vec<String>)> {
    sfmtool_matches_format::verify_matches(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_matches, m)?)?;
    m.add_function(wrap_pyfunction!(read_matches_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(write_matches, m)?)?;
    m.add_function(wrap_pyfunction!(verify_matches, m)?)?;
    Ok(())
}
