// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `.sfmr` file I/O.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;

use sfmtool_sfmr_format::{
    self, ContentHash, DepthStatistics, SfmrData, SfmrMetadata, WriteOptions,
};

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
///   point_constraints, constraint_distances, constraint_reference_images,
///   patch_u_halfvec_xyz, patch_v_halfvec_xyz, patch_bitmaps_y_x_rgba,
///   image_indexes, feature_indexes, keypoints_xy, observation_confidence,
///   observation_readings (a dict of the eight reading columns and their
///   `"options"`, or `None`), point_indexes, observation_counts, reference_observations,
///   observed_depth_histogram_counts, thumbnails_y_x_rgb (numpy arrays;
///   `thumbnails_y_x_rgb` is `None` for a file without thumbnails),
///   focal_adjustments, principal_point_adjustments_xy,
///   radial_bspline_adjustments, rolling_shutter_rates_xyz (numpy arrays or
///   `None`), per_image_adjustments (`list[list[str]]`, one per camera),
///   rolling_shutter_scanlines (`list[str | None]`, one per camera).
///
/// The per-image intrinsic adjustments (version 13+): `focal_adjustments` is
/// the `(N,)` float64 focal scale `s` of each image (its focal is its camera's
/// times `1 + s`), `principal_point_adjustments_xy` the `(N, 2)` float64
/// principal-point offset of each image in its camera's short side,
/// `min(width, height)`, and `per_image_adjustments` names, per camera, which
/// of `"focal"`, `"principal_point"`, `"radial_bspline"` and
/// `"rolling_shutter"` vary per image. A column is `None` exactly when no
/// camera names its parameter.
///
/// Version 14 adds `radial_bspline_adjustments`, the `(N, 2)` float64
/// per-image radial B-spline correction in units of the normalised radius,
/// `rolling_shutter_rates_xyz`, the `(N, 3)` float64 per-image rolling-shutter
/// rates in radians per short side of readout, and `rolling_shutter_scanlines`,
/// each camera's `"ROWS"`, `"COLUMNS"` or `None`. All three read as stored.
///
/// `positions_xyzw` is the homogeneous `(P, 4)` point array. Every optional
/// column is emitted as `None` when the file does not carry it: the normals and
/// their `(P,)` uint8 `normal_confidence`, the per-point constraint triple
/// (`(P,)` uint8 `point_constraints`, `(P,)` float64 `constraint_distances` and
/// `(P,)` uint32 `constraint_reference_images`, which the archive names
/// `points3d/point_constraints`, `points3d/constraint_distances` and
/// `points3d/constraint_reference_images`), the per-point patch frame
/// (`(P, 3)` float32 `patch_u_halfvec_xyz` / `patch_v_halfvec_xyz` and the
/// `(P, R, R, 4)` uint8 `patch_bitmaps_y_x_rgba`), the mode-dependent
/// observation columns, the `(M,)` uint8 `observation_confidence`, and the
/// `(P,)` int32 `reference_observations`, which is present exactly with the
/// patch frame (`-1` throughout for a file below version 12).
///
/// Everything `write_sfmr` stores is emitted here, so
/// `write_sfmr(out, read_sfmr(path))` round-trips a file of either observation
/// source — including an `embedded_patches` one, whose write validation demands
/// the patch frame.
#[pyfunction]
pub fn read_sfmr(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let data = sfmtool_sfmr_format::read_sfmr(&path)
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
    // Per camera, the parameters that vary per image: the cameras'
    // `per_image_adjustments` key, which a `CameraIntrinsics` does not carry.
    let per_image_adjustments: Vec<Vec<String>> = data
        .cameras
        .iter()
        .map(|c| c.per_image_adjustments.clone())
        .collect();
    dict.set_item("per_image_adjustments", per_image_adjustments)?;
    // Per camera, its `rolling_shutter_scanlines` setting or `None`.
    let rolling_shutter_scanlines: Vec<Option<&str>> = data
        .cameras
        .iter()
        .map(|c| c.rolling_shutter_scanlines.map(|s| s.as_str()))
        .collect();
    dict.set_item("rolling_shutter_scanlines", rolling_shutter_scanlines)?;
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
    // The per-point constraint triple: present together or absent together,
    // absent meaning every point is free. The column is numeric -- a
    // million-point reconstruction is not a million strings -- and in the
    // canonical numbering `POINT_CONSTRAINT_NAMES` labels, whatever legend the
    // file itself stored it on.
    match data.point_constraints {
        Some(k) => dict.set_item("point_constraints", k.into_pyarray(py))?,
        None => dict.set_item("point_constraints", py.None())?,
    }
    match data.constraint_distances {
        Some(r) => dict.set_item("constraint_distances", r.into_pyarray(py))?,
        None => dict.set_item("constraint_distances", py.None())?,
    }
    match data.constraint_reference_images {
        Some(c) => dict.set_item("constraint_reference_images", c.into_pyarray(py))?,
        None => dict.set_item("constraint_reference_images", py.None())?,
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
    match data.observation_readings {
        Some(r) => dict.set_item(
            "observation_readings",
            crate::readings::columns_to_py(py, r)?,
        )?,
        None => dict.set_item("observation_readings", py.None())?,
    }
    dict.set_item("point_indexes", data.point_indexes.into_pyarray(py))?;
    dict.set_item(
        "observation_counts",
        data.observation_counts.into_pyarray(py),
    )?;
    // Present exactly with the patch frame; `-1` throughout for a file below
    // version 12 that has one.
    match data.reference_observations {
        Some(r) => dict.set_item("reference_observations", r.into_pyarray(py))?,
        None => dict.set_item("reference_observations", py.None())?,
    }
    dict.set_item(
        "observed_depth_histogram_counts",
        data.observed_depth_histogram_counts.into_pyarray(py),
    )?;
    dict.set_item(
        "thumbnails_y_x_rgb",
        data.thumbnails_y_x_rgb.map(|t| t.into_pyarray(py)),
    )?;
    dict.set_item(
        "focal_adjustments",
        data.focal_adjustments.map(|a| a.into_pyarray(py)),
    )?;
    dict.set_item(
        "principal_point_adjustments_xy",
        data.principal_point_adjustments_xy
            .map(|a| a.into_pyarray(py)),
    )?;
    dict.set_item(
        "radial_bspline_adjustments",
        data.radial_bspline_adjustments.map(|a| a.into_pyarray(py)),
    )?;
    dict.set_item(
        "rolling_shutter_rates_xyz",
        data.rolling_shutter_rates_xyz.map(|a| a.into_pyarray(py)),
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
    let metadata = sfmtool_sfmr_format::read_sfmr_metadata(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    serde_to_py(py, &metadata)
}

/// Read the overall content hash (`content_xxh128`) of a .sfmr file.
///
/// Decompresses only `content_hash.json.zst`, so it is cheap enough to scan a
/// directory of `.sfmr` files to resolve a `pt3d_<hash>_<index>` Point ID.
#[pyfunction]
pub fn read_sfmr_content_hash(path: PathBuf) -> PyResult<String> {
    let content_hash = sfmtool_sfmr_format::read_sfmr_content_hash(&path)
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
    let mut cameras = extract_cameras_as_sfmr(&get_item(data, "cameras")?)?;
    // The cameras' `per_image_adjustments` key, one list per camera; a dict
    // without it varies nothing per image. The names and the columns are
    // the format writer's to check.
    if let Some(value) = get_optional_item(data, "per_image_adjustments")? {
        let lists: Vec<Vec<String>> = value.extract()?;
        if lists.len() != cameras.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "'per_image_adjustments' has {} entries, one per camera is {}",
                lists.len(),
                cameras.len()
            )));
        }
        for (camera, names) in cameras.iter_mut().zip(lists) {
            camera.per_image_adjustments = names;
        }
    }
    // The cameras' `rolling_shutter_scanlines` key, one `"ROWS"`,
    // `"COLUMNS"` or `None` per camera; a dict without it records none.
    if let Some(value) = get_optional_item(data, "rolling_shutter_scanlines")? {
        let settings: Vec<Option<String>> = value.extract()?;
        if settings.len() != cameras.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "'rolling_shutter_scanlines' has {} entries, one per camera is {}",
                settings.len(),
                cameras.len()
            )));
        }
        for (c, (camera, setting)) in cameras.iter_mut().zip(settings).enumerate() {
            camera.rolling_shutter_scanlines = setting
                .as_deref()
                .map(sfmtool_sfmr_format::RollingShutterScanlines::parse)
                .transpose()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("camera {c}: {e}")))?;
        }
    }
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
    let thumbnails_y_x_rgb = match get_optional_item(data, "thumbnails_y_x_rgb")? {
        Some(v) => Some(
            v.extract::<PyReadonlyArray4<u8>>()?
                .as_array()
                .as_standard_layout()
                .into_owned(),
        ),
        None => None,
    };

    // The normals do not depend on whether the depth statistics are recomputed:
    // an array is written as given, an explicit `None` means no normals, and a
    // missing key is an all-zero set, every row of which the writer fills with
    // a mean-viewing normal (it fills zero rows in both modes).
    let point_count = positions_xyzw.as_array().nrows();
    let normals_xyz = if data.contains("normals_xyz")? {
        optional_array::<f32, ndarray::Ix2>(data, "normals_xyz", "a 2D float32 array")?
    } else {
        Some(ndarray::Array2::<f32>::zeros((point_count, 3)))
    };
    if let Some(normals) = &normals_xyz {
        if normals.shape() != [point_count, 3] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "'normals_xyz' must have shape ({point_count}, 3), got shape {:?}",
                normals.shape()
            )));
        }
    }

    // The depth statistics are read from the dict only when the write keeps
    // them; otherwise the write recomputes them and these are placeholders.
    let (depth_statistics, observed_depth_histogram_counts) = if skip_recompute_depth_stats {
        let ds: DepthStatistics = py_to_serde(py, &get_item(data, "depth_statistics")?)?;
        let ohc: PyReadonlyArray2<u32> =
            get_item(data, "observed_depth_histogram_counts")?.extract()?;
        (ds, ohc.as_array().as_standard_layout().into_owned())
    } else {
        let image_count = metadata.image_count as usize;
        (
            DepthStatistics {
                num_histogram_buckets: 128,
                images: Vec::new(),
            },
            ndarray::Array2::<u32>::zeros((image_count, 128)),
        )
    };

    // Per-point confidence in the normal, and the per-observation photometric
    // confidence: independent optional columns, passed through untouched.
    let normal_confidence =
        optional_array::<u8, ndarray::Ix1>(data, "normal_confidence", "a 1D uint8 array")?;
    let observation_confidence =
        optional_array::<u8, ndarray::Ix1>(data, "observation_confidence", "a 1D uint8 array")?;
    // The eight observation-reading columns, a dict or `None`; their row
    // counts are the format writer's to check.
    let observation_readings = match get_optional_item(data, "observation_readings")? {
        Some(v) => crate::readings::columns_from_py(&v, None)?,
        None => None,
    };

    // The per-point constraint triple. Its cross-array rules -- present
    // together, constraint codes, a finite distance naming a real image, a ranged row's
    // `w` agreeing with its distance -- are the format writer's to enforce, so
    // only the per-array dtype and rank are checked here.
    let point_constraints =
        optional_array::<u8, ndarray::Ix1>(data, "point_constraints", "a 1D uint8 array")?;
    let constraint_distances =
        optional_array::<f64, ndarray::Ix1>(data, "constraint_distances", "a 1D float64 array")?;
    let constraint_reference_images = optional_array::<u32, ndarray::Ix1>(
        data,
        "constraint_reference_images",
        "a 1D uint32 array",
    )?;

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

    // The reference observations, present exactly with the patch frame. A dict
    // with a frame and no column (one built before version 12) states that no
    // point names one, so every row is `-1`; the range checks are the format
    // writer's.
    let reference_observations =
        optional_array::<i32, ndarray::Ix1>(data, "reference_observations", "a 1D int32 array")?
            .or_else(|| {
                patch_u_halfvec_xyz.as_ref().map(|_| {
                    ndarray::Array1::from_elem(
                        observation_counts.as_array().len(),
                        sfmtool_sfmr_format::NO_REFERENCE_OBSERVATION,
                    )
                })
            });

    // The per-image intrinsic adjustment columns, each `None` or absent where
    // no camera varies its parameter.
    let focal_adjustments =
        optional_array::<f64, ndarray::Ix1>(data, "focal_adjustments", "a 1D float64 array")?;
    let principal_point_adjustments_xy = optional_array::<f64, ndarray::Ix2>(
        data,
        "principal_point_adjustments_xy",
        "a 2D float64 array",
    )?;
    let radial_bspline_adjustments = optional_array::<f64, ndarray::Ix2>(
        data,
        "radial_bspline_adjustments",
        "a 2D float64 array",
    )?;
    let rolling_shutter_rates_xyz = optional_array::<f64, ndarray::Ix2>(
        data,
        "rolling_shutter_rates_xyz",
        "a 2D float64 array",
    )?;

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
            derived_xxh128: None,
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
        thumbnails_y_x_rgb,
        focal_adjustments,
        principal_point_adjustments_xy,
        radial_bspline_adjustments,
        rolling_shutter_rates_xyz,
        positions_xyzw: positions_xyzw.as_array().as_standard_layout().into_owned(),
        colors_rgb: colors_rgb.as_array().as_standard_layout().into_owned(),
        reprojection_errors: reprojection_errors
            .as_array()
            .as_standard_layout()
            .into_owned(),
        normals_xyz,
        normal_confidence,
        point_constraints,
        constraint_distances,
        constraint_reference_images,
        patch_u_halfvec_xyz,
        patch_v_halfvec_xyz,
        patch_bitmaps_y_x_rgba,
        image_indexes: image_indexes.as_array().as_standard_layout().into_owned(),
        feature_indexes,
        keypoints_xy,
        observation_confidence,
        observation_readings,
        point_indexes: point_indexes.as_array().as_standard_layout().into_owned(),
        observation_counts: observation_counts
            .as_array()
            .as_standard_layout()
            .into_owned(),
        reference_observations,
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
/// `observation_confidence`, the constraint triple (`point_constraints`,
/// `constraint_distances`, `constraint_reference_images` -- written only when some point is not
/// free, since an all-free set is the same statement as none at all), and the
/// per-point patch frame
/// (`patch_u_halfvec_xyz`, `patch_v_halfvec_xyz` and the optional
/// `patch_bitmaps_y_x_rgba`). A missing or `None` value means the column is
/// absent — which is what a `sift_files` dict carries for the patch frame,
/// while an `embedded_patches` dict must carry it, since the format requires
/// the frame there. A missing `per_image_adjustments` means no camera varies
/// anything per image, and a missing `rolling_shutter_scanlines` that no
/// camera has a scanlines setting.
///
/// `skip_recompute_depth_stats` decides only where the depth statistics come
/// from: by default they are recomputed from the geometry, and with `True`
/// the dict's `depth_statistics` and `observed_depth_histogram_counts` are
/// written as given. The normals are handled the same way in both modes. A
/// `normals_xyz` array is written as given, except that each zero row is
/// filled with the point's mean-viewing normal; an explicit `None` writes no
/// normals; and a missing key writes a mean-viewing normal for every point. A
/// caller that moves or rotates the points in the dict updates `normals_xyz`
/// to match, or deletes the key to have them computed again.
///
/// KNOWN LIMITATION: this always writes the current
/// [`sfmtool_sfmr_format::SFMR_FORMAT_VERSION`] regardless of the `metadata["version"]`
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
    sfmtool_sfmr_format::write_sfmr_with_options(&path, &mut sfmr_data, &options)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Verify integrity of a .sfmr file.
///
/// Returns a tuple (is_valid, error_messages).
#[pyfunction]
pub fn verify_sfmr(path: PathBuf) -> PyResult<(bool, Vec<String>)> {
    sfmtool_sfmr_format::verify_sfmr(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The name of each `point_constraints` code, in code order, so a consumer
    // can label a column without hard-coding the numbering: a code indexes a
    // legend on disk, and this is the legend every column reaching Python is on.
    m.add(
        "POINT_CONSTRAINT_NAMES",
        pyo3::types::PyTuple::new(m.py(), sfmtool_sfmr_format::PointConstraint::NAMES)?,
    )?;
    m.add_function(wrap_pyfunction!(read_sfmr, m)?)?;
    m.add_function(wrap_pyfunction!(read_sfmr_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(read_sfmr_content_hash, m)?)?;
    m.add_function(wrap_pyfunction!(write_sfmr, m)?)?;
    m.add_function(wrap_pyfunction!(verify_sfmr, m)?)?;
    Ok(())
}
