// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `.sfmr` file I/O.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;

use sfmr_format::{self, ContentHash, DepthStatistics, SfmrData, SfmrMetadata, WriteOptions};

use crate::helpers::{
    extract_cameras_as_sfmr, extract_rig_frame_data, get_item, get_optional_item, py_to_serde,
    py_to_u128_bytes, rig_frame_data_to_py, serde_to_py, u128_bytes_to_py,
};
use crate::PyCameraIntrinsics;

/// Read a complete .sfmr file, returning a dict with numpy arrays and metadata.
///
/// KNOWN LIMITATION (convention upgrade not applied): unlike
/// `SfmrReconstruction::load`, this low-level dict reader does **not** apply the
/// version ≤ 4 → 5 COLMAP→canonical convention upgrade. A pre-v5 file is
/// returned with its stored COLMAP-convention poses/points (Y-down/+Z-forward
/// cameras, un-rotated world) and its stored `metadata["version"]`, while every
/// other loader returns canonical data. Worse, `write_sfmr` (below) restamps the
/// version to the current one unconditionally, so a read→write round trip of a
/// pre-v5 file through this dict API permanently mislabels COLMAP-convention data
/// as canonical v5 and the upgrade in `SfmrReconstruction::load` never fires
/// again. Prefer `SfmrReconstruction` for pose/point data; use this dict API only
/// for v5 files or when you handle the convention yourself. (Tracked as a known
/// bug in the zup-completion review.)
///
/// Returns a dict with keys:
///   metadata, content_hash, cameras, depth_statistics (dicts/lists),
///   image_names (list[str]),
///   feature_tool_hashes, sift_content_hashes (list[bytes], 16 bytes each),
///   camera_indexes, quaternions_wxyz, translations_xyz, positions_xyzw,
///   colors_rgb, reprojection_errors, normals_xyz,
///   image_indexes, feature_indexes, point_indexes, observation_counts,
///   observed_depth_histogram_counts (numpy arrays).
///
/// `positions_xyzw` is the homogeneous `(P, 4)` point array.
#[pyfunction]
pub fn read_sfmr(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let data = sfmr_format::read_sfmr(&path)
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

    // uint128 hashes as list[bytes] — mode-dependent, `None` when absent.
    match &data.feature_tool_hashes {
        Some(v) => dict.set_item("feature_tool_hashes", u128_bytes_to_py(py, v)?)?,
        None => dict.set_item("feature_tool_hashes", py.None())?,
    }
    match &data.sift_content_hashes {
        Some(v) => dict.set_item("sift_content_hashes", u128_bytes_to_py(py, v)?)?,
        None => dict.set_item("sift_content_hashes", py.None())?,
    }
    match &data.image_file_hashes {
        Some(v) => dict.set_item("image_file_hashes", u128_bytes_to_py(py, v)?)?,
        None => dict.set_item("image_file_hashes", py.None())?,
    }

    // Numpy arrays (ownership transferred)
    dict.set_item("camera_indexes", data.camera_indexes.into_pyarray(py))?;
    dict.set_item("quaternions_wxyz", data.quaternions_wxyz.into_pyarray(py))?;
    dict.set_item("translations_xyz", data.translations_xyz.into_pyarray(py))?;
    dict.set_item("positions_xyzw", data.positions_xyzw.into_pyarray(py))?;
    dict.set_item("colors_rgb", data.colors_rgb.into_pyarray(py))?;
    dict.set_item(
        "reprojection_errors",
        data.reprojection_errors.into_pyarray(py),
    )?;
    // Normals are optional: emit the array when present, else `None`.
    match data.normals_xyz {
        Some(n) => dict.set_item("normals_xyz", n.into_pyarray(py))?,
        None => dict.set_item("normals_xyz", py.None())?,
    }
    dict.set_item("image_indexes", data.image_indexes.into_pyarray(py))?;
    // feature_indexes (sift_files) / keypoints_xy (embedded_patches) are
    // mode-dependent: emit the present one, `None` for the absent one.
    match data.feature_indexes {
        Some(f) => dict.set_item("feature_indexes", f.into_pyarray(py))?,
        None => dict.set_item("feature_indexes", py.None())?,
    }
    match data.keypoints_xy {
        Some(k) => dict.set_item("keypoints_xy", k.into_pyarray(py))?,
        None => dict.set_item("keypoints_xy", py.None())?,
    }
    dict.set_item("point_indexes", data.point_indexes.into_pyarray(py))?;
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
pub fn read_sfmr_metadata(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let metadata = sfmr_format::read_sfmr_metadata(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    serde_to_py(py, &metadata)
}

/// Read the overall content hash (`content_xxh128`) of a .sfmr file.
///
/// Decompresses only `content_hash.json.zst`, so it is cheap enough to scan a
/// directory of `.sfmr` files to resolve a `pt3d_<hash>_<index>` Point ID.
#[pyfunction]
pub fn read_sfmr_content_hash(path: PathBuf) -> PyResult<String> {
    let content_hash = sfmr_format::read_sfmr_content_hash(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(content_hash.content_xxh128)
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
    let positions_xyzw: PyReadonlyArray2<f64> = get_item(data, "positions_xyzw")?.extract()?;
    let colors_rgb: PyReadonlyArray2<u8> = get_item(data, "colors_rgb")?.extract()?;
    let reprojection_errors: PyReadonlyArray1<f32> =
        get_item(data, "reprojection_errors")?.extract()?;
    let image_indexes: PyReadonlyArray1<u32> = get_item(data, "image_indexes")?.extract()?;
    let point_indexes: PyReadonlyArray1<u32> = get_item(data, "point_indexes")?.extract()?;
    let observation_counts: PyReadonlyArray1<u32> =
        get_item(data, "observation_counts")?.extract()?;

    // Mode-dependent columns: exactly one of feature_indexes / keypoints_xy and
    // one of the per-image hash sets is present (the others None or absent).
    let feature_indexes = match get_optional_item(data, "feature_indexes")? {
        Some(v) => Some(v.extract::<PyReadonlyArray1<u32>>()?.as_array().to_owned()),
        None => None,
    };
    let keypoints_xy = match get_optional_item(data, "keypoints_xy")? {
        Some(v) => Some(v.extract::<PyReadonlyArray2<f32>>()?.as_array().to_owned()),
        None => None,
    };
    let feature_tool_hashes = match get_optional_item(data, "feature_tool_hashes")? {
        Some(v) => Some(py_to_u128_bytes(&v)?),
        None => None,
    };
    let sift_content_hashes = match get_optional_item(data, "sift_content_hashes")? {
        Some(v) => Some(py_to_u128_bytes(&v)?),
        None => None,
    };
    let image_file_hashes = match get_optional_item(data, "image_file_hashes")? {
        Some(v) => Some(py_to_u128_bytes(&v)?),
        None => None,
    };
    let thumbnails_y_x_rgb: PyReadonlyArray4<u8> =
        get_item(data, "thumbnails_y_x_rgb")?.extract()?;

    // Depth-related fields are only required when skipping recomputation.
    // Normals are optional: a missing or `None` `normals_xyz` means no normals.
    let (depth_statistics, normals_xyz, observed_depth_histogram_counts) =
        if skip_recompute_depth_stats {
            let ds: DepthStatistics = py_to_serde(py, &get_item(data, "depth_statistics")?)?;
            let normals: Option<ndarray::Array2<f32>> = match data.get_item("normals_xyz")? {
                Some(v) if !v.is_none() => {
                    let en: PyReadonlyArray2<f32> = v.extract()?;
                    Some(en.as_array().to_owned())
                }
                _ => None,
            };
            let ohc: PyReadonlyArray2<u32> =
                get_item(data, "observed_depth_histogram_counts")?.extract()?;
            (ds, normals, ohc.as_array().to_owned())
        } else {
            let image_count = metadata.image_count as usize;
            let point_count = metadata.point_count as usize;
            // Default to an all-zero set so the write-time mean-viewing recompute
            // fills them; pass `normals_xyz=None` explicitly to opt out.
            let normals = match data.get_item("normals_xyz")? {
                Some(v) if v.is_none() => None,
                _ => Some(ndarray::Array2::<f32>::zeros((point_count, 3))),
            };
            (
                DepthStatistics {
                    num_histogram_buckets: 128,
                    images: Vec::new(),
                },
                normals,
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
        image_file_hashes,
        thumbnails_y_x_rgb: thumbnails_y_x_rgb.as_array().to_owned(),
        positions_xyzw: positions_xyzw.as_array().to_owned(),
        colors_rgb: colors_rgb.as_array().to_owned(),
        reprojection_errors: reprojection_errors.as_array().to_owned(),
        normals_xyz,
        // The dict-based columnar API does not carry patch data; the
        // patch-aware path is `SfmrReconstruction`.
        patch_u_halfvec_xyz: None,
        patch_v_halfvec_xyz: None,
        patch_bitmaps_y_x_rgba: None,
        image_indexes: image_indexes.as_array().to_owned(),
        feature_indexes,
        keypoints_xy,
        point_indexes: point_indexes.as_array().to_owned(),
        observation_counts: observation_counts.as_array().to_owned(),
        depth_statistics,
        observed_depth_histogram_counts,
    })
}

/// Write a .sfmr file from a dict of numpy arrays and metadata.
///
/// The dict should have the same keys as returned by `read_sfmr`.
/// The `content_hash` key is ignored (recomputed on write).
///
/// KNOWN LIMITATION: this always writes the current [`SFMR_FORMAT_VERSION`]
/// regardless of the `metadata["version"]` in the dict, and assumes the arrays
/// are already in the canonical convention — it applies no conversion. Writing a
/// dict read from a pre-v5 file via `read_sfmr` (which does not upgrade) therefore
/// stamps COLMAP-convention data as canonical v5. See `read_sfmr` above.
#[pyfunction]
#[pyo3(signature = (path, data, zstd_level=3, skip_recompute_depth_stats=false))]
pub fn write_sfmr(
    py: Python<'_>,
    path: PathBuf,
    data: &Bound<'_, PyDict>,
    zstd_level: i32,
    skip_recompute_depth_stats: bool,
) -> PyResult<()> {
    let mut sfmr_data = parse_sfmr_data_from_dict(py, data, skip_recompute_depth_stats)?;

    let options = WriteOptions {
        zstd_level,
        skip_recompute_depth_stats,
    };
    sfmr_format::write_sfmr_with_options(&path, &mut sfmr_data, &options)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Verify integrity of a .sfmr file.
///
/// Returns a tuple (is_valid, error_messages).
#[pyfunction]
pub fn verify_sfmr(path: PathBuf) -> PyResult<(bool, Vec<String>)> {
    sfmr_format::verify_sfmr(&path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_sfmr, m)?)?;
    m.add_function(wrap_pyfunction!(read_sfmr_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(read_sfmr_content_hash, m)?)?;
    m.add_function(wrap_pyfunction!(write_sfmr, m)?)?;
    m.add_function(wrap_pyfunction!(verify_sfmr, m)?)?;
    Ok(())
}
