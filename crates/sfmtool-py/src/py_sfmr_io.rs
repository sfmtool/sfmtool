// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `.sfmr` file I/O.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::Path;

use sfmr_format::{self, ContentHash, DepthStatistics, SfmrData, SfmrMetadata, WriteOptions};

use crate::helpers::{
    extract_cameras_as_sfmr, extract_rig_frame_data, get_item, py_to_serde, py_to_u128_bytes,
    rig_frame_data_to_py, serde_to_py, u128_bytes_to_py,
};
use crate::PyCameraIntrinsics;

/// Read a complete .sfmr file, returning a dict with numpy arrays and metadata.
///
/// Returns a dict with keys:
///   metadata, content_hash, cameras, depth_statistics (dicts/lists),
///   image_names (list[str]),
///   feature_tool_hashes, sift_content_hashes (list[bytes], 16 bytes each),
///   camera_indexes, quaternions_wxyz, translations_xyz, positions_xyz,
///   colors_rgb, reprojection_errors, estimated_normals_xyz,
///   image_indexes, feature_indexes, points3d_indexes, observation_counts,
///   observed_depth_histogram_counts (numpy arrays).
#[pyfunction]
pub fn read_sfmr(py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
    let data = sfmr_format::read_sfmr(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);

    // Resolved workspace directory (None if resolution failed)
    match &data.workspace_dir {
        Some(dir) => {
            let s = dir.to_string_lossy();
            // Strip Windows extended-length path prefix (\\?\) for Python compat
            #[cfg(target_os = "windows")]
            let s = s.strip_prefix(r"\\?\").unwrap_or(&s);
            dict.set_item("workspace_dir", s)?
        }
        None => dict.set_item("workspace_dir", py.None())?,
    }

    // JSON-like metadata
    dict.set_item("metadata", serde_to_py(py, &data.metadata)?)?;
    dict.set_item("content_hash", serde_to_py(py, &data.content_hash)?)?;
    // Convert SfmrCamera → CameraIntrinsics for typed Python API
    let cameras: Vec<PyCameraIntrinsics> = data
        .cameras
        .iter()
        .map(|c| {
            let inner = sfmtool_core::CameraIntrinsics::try_from(c)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyCameraIntrinsics { inner })
        })
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("cameras", PyList::new(py, cameras)?)?;
    dict.set_item("depth_statistics", serde_to_py(py, &data.depth_statistics)?)?;

    // String list
    dict.set_item("image_names", &data.image_names)?;

    // uint128 hashes as list[bytes]
    dict.set_item(
        "feature_tool_hashes",
        u128_bytes_to_py(py, &data.feature_tool_hashes)?,
    )?;
    dict.set_item(
        "sift_content_hashes",
        u128_bytes_to_py(py, &data.sift_content_hashes)?,
    )?;

    // Numpy arrays (ownership transferred)
    dict.set_item("camera_indexes", data.camera_indexes.into_pyarray(py))?;
    dict.set_item("quaternions_wxyz", data.quaternions_wxyz.into_pyarray(py))?;
    dict.set_item("translations_xyz", data.translations_xyz.into_pyarray(py))?;
    dict.set_item("positions_xyz", data.positions_xyz.into_pyarray(py))?;
    dict.set_item("colors_rgb", data.colors_rgb.into_pyarray(py))?;
    dict.set_item(
        "reprojection_errors",
        data.reprojection_errors.into_pyarray(py),
    )?;
    dict.set_item(
        "estimated_normals_xyz",
        data.estimated_normals_xyz.into_pyarray(py),
    )?;
    dict.set_item("image_indexes", data.image_indexes.into_pyarray(py))?;
    dict.set_item("feature_indexes", data.feature_indexes.into_pyarray(py))?;
    dict.set_item("points3d_indexes", data.points3d_indexes.into_pyarray(py))?;
    dict.set_item(
        "observation_counts",
        data.observation_counts.into_pyarray(py),
    )?;
    dict.set_item(
        "observed_depth_histogram_counts",
        data.observed_depth_histogram_counts.into_pyarray(py),
    )?;
    dict.set_item(
        "thumbnails_y_x_rgb",
        data.thumbnails_y_x_rgb.into_pyarray(py),
    )?;

    // Rig/frame data (optional)
    match &data.rig_frame_data {
        Some(rf) => dict.set_item("rig_frame_data", rig_frame_data_to_py(py, rf)?)?,
        None => dict.set_item("rig_frame_data", py.None())?,
    }

    Ok(dict.into())
}

/// Read only the top-level metadata from a .sfmr file (fast, no binary data).
#[pyfunction]
pub fn read_sfmr_metadata(py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
    let metadata = sfmr_format::read_sfmr_metadata(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    serde_to_py(py, &metadata)
}

/// Parse a Python dict into an `SfmrData` struct.
///
/// This is the shared parsing logic used by both `write_sfmr` and
/// `SfmrReconstruction.from_data`.
pub(crate) fn parse_sfmr_data_from_dict(
    py: Python<'_>,
    data: &Bound<'_, PyDict>,
    skip_recompute_depth_stats: bool,
) -> PyResult<SfmrData> {
    let metadata: SfmrMetadata = py_to_serde(py, &get_item(data, "metadata")?)?;
    let cameras = extract_cameras_as_sfmr(&get_item(data, "cameras")?)?;
    let image_names: Vec<String> = get_item(data, "image_names")?.extract()?;

    let camera_indexes: PyReadonlyArray1<u32> = get_item(data, "camera_indexes")?.extract()?;
    let quaternions_wxyz: PyReadonlyArray2<f64> = get_item(data, "quaternions_wxyz")?.extract()?;
    let translations_xyz: PyReadonlyArray2<f64> = get_item(data, "translations_xyz")?.extract()?;
    let positions_xyz: PyReadonlyArray2<f64> = get_item(data, "positions_xyz")?.extract()?;
    let colors_rgb: PyReadonlyArray2<u8> = get_item(data, "colors_rgb")?.extract()?;
    let reprojection_errors: PyReadonlyArray1<f32> =
        get_item(data, "reprojection_errors")?.extract()?;
    let image_indexes: PyReadonlyArray1<u32> = get_item(data, "image_indexes")?.extract()?;
    let feature_indexes: PyReadonlyArray1<u32> = get_item(data, "feature_indexes")?.extract()?;
    let points3d_indexes: PyReadonlyArray1<u32> = get_item(data, "points3d_indexes")?.extract()?;
    let observation_counts: PyReadonlyArray1<u32> =
        get_item(data, "observation_counts")?.extract()?;

    let feature_tool_hashes = py_to_u128_bytes(&get_item(data, "feature_tool_hashes")?)?;
    let sift_content_hashes = py_to_u128_bytes(&get_item(data, "sift_content_hashes")?)?;
    let thumbnails_y_x_rgb: PyReadonlyArray4<u8> =
        get_item(data, "thumbnails_y_x_rgb")?.extract()?;

    // Depth-related fields are only required when skipping recomputation
    let (depth_statistics, estimated_normals_xyz, observed_depth_histogram_counts) =
        if skip_recompute_depth_stats {
            let ds: DepthStatistics = py_to_serde(py, &get_item(data, "depth_statistics")?)?;
            let en: PyReadonlyArray2<f32> = get_item(data, "estimated_normals_xyz")?.extract()?;
            let ohc: PyReadonlyArray2<u32> =
                get_item(data, "observed_depth_histogram_counts")?.extract()?;
            (ds, en.as_array().to_owned(), ohc.as_array().to_owned())
        } else {
            let image_count = metadata.image_count as usize;
            let points3d_count = metadata.points3d_count as usize;
            (
                DepthStatistics {
                    num_histogram_buckets: 128,
                    images: Vec::new(),
                },
                ndarray::Array2::<f32>::zeros((points3d_count, 3)),
                ndarray::Array2::<u32>::zeros((image_count, 128)),
            )
        };

    // Extract optional rig/frame data from the dict
    let rig_frame_data = extract_rig_frame_data(py, data)?;

    Ok(SfmrData {
        workspace_dir: None,
        metadata,
        content_hash: ContentHash {
            metadata_xxh128: String::new(),
            cameras_xxh128: String::new(),
            rigs_xxh128: None,
            frames_xxh128: None,
            images_xxh128: String::new(),
            points3d_xxh128: String::new(),
            tracks_xxh128: String::new(),
            content_xxh128: String::new(),
        },
        cameras,
        rig_frame_data,
        image_names,
        camera_indexes: camera_indexes.as_array().to_owned(),
        quaternions_wxyz: quaternions_wxyz.as_array().to_owned(),
        translations_xyz: translations_xyz.as_array().to_owned(),
        feature_tool_hashes,
        sift_content_hashes,
        thumbnails_y_x_rgb: thumbnails_y_x_rgb.as_array().to_owned(),
        positions_xyz: positions_xyz.as_array().to_owned(),
        colors_rgb: colors_rgb.as_array().to_owned(),
        reprojection_errors: reprojection_errors.as_array().to_owned(),
        estimated_normals_xyz,
        image_indexes: image_indexes.as_array().to_owned(),
        feature_indexes: feature_indexes.as_array().to_owned(),
        points3d_indexes: points3d_indexes.as_array().to_owned(),
        observation_counts: observation_counts.as_array().to_owned(),
        depth_statistics,
        observed_depth_histogram_counts,
    })
}

/// Write a .sfmr file from a dict of numpy arrays and metadata.
///
/// The dict should have the same keys as returned by `read_sfmr`.
/// The `content_hash` key is ignored (recomputed on write).
#[pyfunction]
#[pyo3(signature = (path, data, zstd_level=3, skip_recompute_depth_stats=false))]
pub fn write_sfmr(
    py: Python<'_>,
    path: &str,
    data: &Bound<'_, PyDict>,
    zstd_level: i32,
    skip_recompute_depth_stats: bool,
) -> PyResult<()> {
    let mut sfmr_data = parse_sfmr_data_from_dict(py, data, skip_recompute_depth_stats)?;

    let options = WriteOptions {
        zstd_level,
        skip_recompute_depth_stats,
    };
    sfmr_format::write_sfmr_with_options(Path::new(path), &mut sfmr_data, &options)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Verify integrity of a .sfmr file.
///
/// Returns a tuple (is_valid, error_messages).
#[pyfunction]
pub fn verify_sfmr(path: &str) -> PyResult<(bool, Vec<String>)> {
    sfmr_format::verify_sfmr(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}