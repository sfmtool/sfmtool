// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `.matches` file I/O.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

use matches_format::{
    ClusterPatchData, ClustersData, MatchesContentHash, MatchesData, MatchesMetadata, PairsData,
    TvgMetadata, TwoViewGeometryConfig, TwoViewGeometryData,
};

use crate::helpers::{
    get_item, get_optional_item, py_to_serde, py_to_u128_bytes, serde_to_py, u128_bytes_to_py,
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
        dict.set_item("member_affines", cp.member_affines.into_pyarray(py))?;
        dict.set_item("member_zncc", cp.member_zncc.into_pyarray(py))?;
        dict.set_item("member_shift_px", cp.member_shift_px.into_pyarray(py))?;
        dict.set_item(
            "member_consistency_residual",
            cp.member_consistency_residual.into_pyarray(py),
        )?;
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
    let data = matches_format::read_matches(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    matches_data_to_py(py, data)
}

/// Read only metadata from a .matches file (fast, no binary data).
#[pyfunction]
pub fn read_matches_metadata(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let metadata = matches_format::read_matches_metadata(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    serde_to_py(py, &metadata)
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

    // Backbone: clusters when the has_clusters flag is set, pairs otherwise.
    let has_clusters = get_flag(data, "has_clusters")?;
    let (image_pairs, clusters) = if has_clusters {
        let cluster_starts: PyReadonlyArray1<u32> = get_item(data, "cluster_starts")?.extract()?;
        let member_images: PyReadonlyArray1<u32> = get_item(data, "member_images")?.extract()?;
        let member_features: PyReadonlyArray1<u32> =
            get_item(data, "member_features")?.extract()?;
        let matcher_options: serde_json::Value =
            py_to_serde(py, &get_item(data, "matcher_options")?)?;
        (
            None,
            Some(ClustersData {
                cluster_starts: cluster_starts.as_array().to_owned(),
                member_images: member_images.as_array().to_owned(),
                member_features: member_features.as_array().to_owned(),
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
                image_index_pairs: image_index_pairs.as_array().to_owned(),
                match_counts: match_counts.as_array().to_owned(),
                match_feature_indexes: match_feature_indexes.as_array().to_owned(),
                match_descriptor_distances: match_descriptor_distances.as_array().to_owned(),
            }),
            None,
        )
    };

    // Cluster patches (optional, requires clusters)
    let cluster_patches = if get_flag(data, "has_cluster_patches")? {
        let reference_members: PyReadonlyArray1<u32> =
            get_item(data, "reference_members")?.extract()?;
        let member_status: PyReadonlyArray1<u8> = get_item(data, "member_status")?.extract()?;
        let member_affines: PyReadonlyArray3<f64> = get_item(data, "member_affines")?.extract()?;
        let member_zncc: PyReadonlyArray1<f32> = get_item(data, "member_zncc")?.extract()?;
        let member_shift_px: PyReadonlyArray1<f32> =
            get_item(data, "member_shift_px")?.extract()?;
        let member_consistency_residual: PyReadonlyArray1<f32> =
            get_item(data, "member_consistency_residual")?.extract()?;
        let refine_options: serde_json::Value =
            py_to_serde(py, &get_item(data, "refine_options")?)?;
        Some(ClusterPatchData {
            reference_members: reference_members.as_array().to_owned(),
            member_status: member_status.as_array().to_owned(),
            member_affines: member_affines.as_array().to_owned(),
            member_zncc: member_zncc.as_array().to_owned(),
            member_shift_px: member_shift_px.as_array().to_owned(),
            member_consistency_residual: member_consistency_residual.as_array().to_owned(),
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
                s.parse().map_err(|e: matches_format::MatchesError| {
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
            config_indexes: config_indexes.as_array().to_owned(),
            inlier_counts: inlier_counts.as_array().to_owned(),
            inlier_feature_indexes: inlier_feature_indexes.as_array().to_owned(),
            f_matrices: f_matrices.as_array().to_owned(),
            e_matrices: e_matrices.as_array().to_owned(),
            h_matrices: h_matrices.as_array().to_owned(),
            quaternions_wxyz: quaternions_wxyz.as_array().to_owned(),
            translations_xyz: translations_xyz.as_array().to_owned(),
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
        feature_counts: feature_counts.as_array().to_owned(),
        image_pairs,
        clusters,
        cluster_patches,
        two_view_geometries,
    };

    matches_format::write_matches(&path, &matches_data, zstd_level)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Verify integrity of a .matches file.
///
/// Returns a tuple (is_valid, error_messages).
#[pyfunction]
pub fn verify_matches(path: PathBuf) -> PyResult<(bool, Vec<String>)> {
    matches_format::verify_matches(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_matches, m)?)?;
    m.add_function(wrap_pyfunction!(read_matches_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(write_matches, m)?)?;
    m.add_function(wrap_pyfunction!(verify_matches, m)?)?;
    Ok(())
}
