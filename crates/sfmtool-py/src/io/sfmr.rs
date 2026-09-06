// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `.sfmr` file I/O.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;

use sfmr_format::{self, ContentHash, DepthStatistics, SfmrData, SfmrMetadata, WriteOptions};

use crate::helpers::{
    dtype_name, extract_cameras_as_sfmr, extract_rig_frame_data, get_item, get_optional_item,
    py_to_serde, py_to_u128_bytes, rig_frame_data_to_py, serde_to_py, u128_bytes_to_py,
};
use crate::PyCameraIntrinsics;

/// Extract an optional numpy array from a data dict: `None` for a missing key
/// or an explicit `None`, otherwise the array copied into standard (C) layout.
///
/// `expected` describes the dtype and rank the caller wants (`"a 2D float32
/// array"`), and is reported alongside the offending key and the value's actual
/// type/dtype, so a wrong-dtype column is named rather than surfacing as a bare
/// numpy conversion failure. Shape constraints beyond the rank are the caller's.
fn optional_array<'py, T, D>(
    data: &Bound<'py, PyDict>,
    key: &str,
    expected: &str,
) -> PyResult<Option<ndarray::Array<T, D>>>
where
    T: numpy::Element + Clone,
    D: ndarray::Dimension,
{
    let Some(value) = get_optional_item(data, key)? else {
        return Ok(None);
    };
    let array = value
        .extract::<numpy::PyReadonlyArray<'py, T, D>>()
        .map_err(|_| {
            let actual_type = value
                .get_type()
                .qualname()
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            let actual_dtype = dtype_name(&value).unwrap_or_else(|_| "unknown".to_string());
            pyo3::exceptions::PyTypeError::new_err(format!(
                "'{key}' must be {expected}, got {actual_type} with dtype {actual_dtype}"
            ))
        })?;
    Ok(Some(array.as_array().as_standard_layout().into_owned()))
}

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
///   feature_tool_hashes, sift_content_hashes, image_file_hashes
///   (`list[bytes]`, 16 bytes each),
///   camera_indexes, quaternions_wxyz, translations_xyz, positions_xyzw,
///   colors_rgb, reprojection_errors, normals_xyz, normal_confidence,
///   patch_u_halfvec_xyz, patch_v_halfvec_xyz, patch_bitmaps_y_x_rgba,
///   image_indexes, feature_indexes, keypoints_xy, observation_confidence,
///   point_indexes, observation_counts, observed_depth_histogram_counts,
///   thumbnails_y_x_rgb (numpy arrays).
///
/// `positions_xyzw` is the homogeneous `(P, 4)` point array. Every optional
/// column is emitted as `None` when the file does not carry it: the normals and
/// their `(P,)` uint8 `normal_confidence`, the per-point patch frame
/// (`(P, 3)` float32 `patch_u_halfvec_xyz` / `patch_v_halfvec_xyz` and the
/// `(P, R, R, 4)` uint8 `patch_bitmaps_y_x_rgba`), the mode-dependent
/// observation columns, and the `(M,)` uint8 `observation_confidence`.
///
/// Everything `write_sfmr` stores is emitted here, so
/// `write_sfmr(out, read_sfmr(path))` round-trips a file of either observation
/// source — including an `embedded_patches` one, whose write validation demands
/// the patch frame.
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
    match data.normal_confidence {
        Some(c) => dict.set_item("normal_confidence", c.into_pyarray(py))?,
        None => dict.set_item("normal_confidence", py.None())?,
    }
    // The per-point patch frame: `u` and `v` are present or absent together and
    // the bitmaps require them, so all three ride along as a set.
    match data.patch_u_halfvec_xyz {
        Some(u) => dict.set_item("patch_u_halfvec_xyz", u.into_pyarray(py))?,
        None => dict.set_item("patch_u_halfvec_xyz", py.None())?,
    }
    match data.patch_v_halfvec_xyz {
        Some(v) => dict.set_item("patch_v_halfvec_xyz", v.into_pyarray(py))?,
        None => dict.set_item("patch_v_halfvec_xyz", py.None())?,
    }
    match data.patch_bitmaps_y_x_rgba {
        Some(b) => dict.set_item("patch_bitmaps_y_x_rgba", b.into_pyarray(py))?,
        None => dict.set_item("patch_bitmaps_y_x_rgba", py.None())?,
    }
    dict.set_item("image_indexes", data.image_indexes.into_pyarray(py))?;
    // feature_indexes is the sift_files column and keypoints_xy the
    // embedded_patches one, except that a sift_files file may also carry
    // keypoints_xy inline: emit each present column, `None` for an absent one.
    match data.feature_indexes {
        Some(f) => dict.set_item("feature_indexes", f.into_pyarray(py))?,
        None => dict.set_item("feature_indexes", py.None())?,
    }
    match data.keypoints_xy {
        Some(k) => dict.set_item("keypoints_xy", k.into_pyarray(py))?,
        None => dict.set_item("keypoints_xy", py.None())?,
    }
    match data.observation_confidence {
        Some(c) => dict.set_item("observation_confidence", c.into_pyarray(py))?,
        None => dict.set_item("observation_confidence", py.None())?,
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

    // Mode-dependent columns: one of the per-image hash sets is present (the
    // other None or absent), and so is feature_indexes or keypoints_xy — though
    // a sift_files reconstruction may carry both, the second as its inline copy.
    let feature_indexes = match get_optional_item(data, "feature_indexes")? {
        Some(v) => Some(
            v.extract::<PyReadonlyArray1<u32>>()?
                .as_array()
                .as_standard_layout()
                .into_owned(),
        ),
        None => None,
    };
    let keypoints_xy = match get_optional_item(data, "keypoints_xy")? {
        Some(v) => Some(
            v.extract::<PyReadonlyArray2<f32>>()?
                .as_array()
                .as_standard_layout()
                .into_owned(),
        ),
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
                    Some(en.as_array().as_standard_layout().into_owned())
                }
                _ => None,
            };
            let ohc: PyReadonlyArray2<u32> =
                get_item(data, "observed_depth_histogram_counts")?.extract()?;
            (
                ds,
                normals,
                ohc.as_array().as_standard_layout().into_owned(),
            )
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

    // Per-point confidence in the normal, and the per-observation photometric
    // confidence: independent optional columns, passed through untouched.
    let normal_confidence =
        optional_array::<u8, ndarray::Ix1>(data, "normal_confidence", "a 1D uint8 array")?;
    let observation_confidence =
        optional_array::<u8, ndarray::Ix1>(data, "observation_confidence", "a 1D uint8 array")?;

    // The per-point patch frame. `patch_u_halfvec_xyz`/`patch_v_halfvec_xyz`
    // must be present together and the bitmaps require them; those cross-array
    // rules and the row counts are the format writer's to enforce, so only the
    // per-array dtype and trailing extent are checked here.
    let patch_u_halfvec_xyz =
        optional_array::<f32, ndarray::Ix2>(data, "patch_u_halfvec_xyz", "a 2D float32 array")?;
    let patch_v_halfvec_xyz =
        optional_array::<f32, ndarray::Ix2>(data, "patch_v_halfvec_xyz", "a 2D float32 array")?;
    for (key, halfvec) in [
        ("patch_u_halfvec_xyz", &patch_u_halfvec_xyz),
        ("patch_v_halfvec_xyz", &patch_v_halfvec_xyz),
    ] {
        if let Some(arr) = halfvec {
            if arr.shape()[1] != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "'{key}' must have shape (P, 3), got shape {:?}",
                    arr.shape()
                )));
            }
        }
    }
    let patch_bitmaps_y_x_rgba =
        optional_array::<u8, ndarray::Ix4>(data, "patch_bitmaps_y_x_rgba", "a 4D uint8 array")?;
    if let Some(arr) = &patch_bitmaps_y_x_rgba {
        let shape = arr.shape();
        if shape[1] != shape[2] || shape[3] != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "'patch_bitmaps_y_x_rgba' must have shape (P, R, R, 4), got shape {shape:?}"
            )));
        }
    }

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
        camera_indexes: camera_indexes.as_array().as_standard_layout().into_owned(),
        quaternions_wxyz: quaternions_wxyz
            .as_array()
            .as_standard_layout()
            .into_owned(),
        translations_xyz: translations_xyz
            .as_array()
            .as_standard_layout()
            .into_owned(),
        feature_tool_hashes,
        sift_content_hashes,
        image_file_hashes,
        thumbnails_y_x_rgb: thumbnails_y_x_rgb
            .as_array()
            .as_standard_layout()
            .into_owned(),
        positions_xyzw: positions_xyzw.as_array().as_standard_layout().into_owned(),
        colors_rgb: colors_rgb.as_array().as_standard_layout().into_owned(),
        reprojection_errors: reprojection_errors
            .as_array()
            .as_standard_layout()
            .into_owned(),
        normals_xyz,
        normal_confidence,
        patch_u_halfvec_xyz,
        patch_v_halfvec_xyz,
        patch_bitmaps_y_x_rgba,
        image_indexes: image_indexes.as_array().as_standard_layout().into_owned(),
        feature_indexes,
        keypoints_xy,
        observation_confidence,
        point_indexes: point_indexes.as_array().as_standard_layout().into_owned(),
        observation_counts: observation_counts
            .as_array()
            .as_standard_layout()
            .into_owned(),
        depth_statistics,
        observed_depth_histogram_counts,
    })
}

/// Write a .sfmr file from a dict of numpy arrays and metadata.
///
/// The dict should have the same keys as returned by `read_sfmr`.
/// The `content_hash` key is ignored (recomputed on write).
///
/// Every optional column `read_sfmr` emits is read back here, so a dict that
/// came from `read_sfmr` writes out whatever the source file carried: the
/// normals and their `normal_confidence`, the per-observation
/// `observation_confidence`, and the per-point patch frame
/// (`patch_u_halfvec_xyz`, `patch_v_halfvec_xyz` and the optional
/// `patch_bitmaps_y_x_rgba`). A missing or `None` value means the column is
/// absent — which is what a `sift_files` dict carries for the patch frame,
/// while an `embedded_patches` dict must carry it, since the format requires
/// the frame there.
///
/// KNOWN LIMITATION: this always writes the current
/// [`sfmr_format::SFMR_FORMAT_VERSION`] regardless of the `metadata["version"]`
/// in the dict, and assumes the arrays
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
