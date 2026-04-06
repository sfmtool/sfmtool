// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for COLMAP SQLite database I/O.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;

use sfmr_colmap::colmap_db;

use crate::helpers::{extract_cameras_as_sfmr, extract_optional_3x3_matrix, get_item};
use crate::py_matches_io::matches_data_to_py;

/// Create a COLMAP SQLite database from reconstruction data.
///
/// Args:
///   db_path: Path where the database file should be created.
///   data: Dict with keys:
///     cameras: list of camera dicts or CameraIntrinsics objects
///     image_names: list[str]
///     camera_indexes: numpy array (N,) uint32
///     quaternions_wxyz: numpy array (N,4) float64
///     translations_xyz: numpy array (N,3) float64
///     keypoints_per_image: list of (K,2) float32/float64 arrays
///     descriptors_per_image: list of (K,128) uint8 arrays
///     descriptor_dim: int (default 128)
///     pose_priors: optional list of dicts with keys:
///       position: (3,) float64, position_covariance: (3,3) float64,
///       coordinate_system: int
///     two_view_geometries: optional list of dicts with keys:
///       image_idx1: int, image_idx2: int, matches: (M,2) uint32 array,
///       config: int, f_matrix: optional (3,3) float64,
///       e_matrix: optional (3,3) float64, h_matrix: optional (3,3) float64,
///       qvec_wxyz: optional (4,) float64, tvec: optional (3,) float64
///
/// Returns:
///   list[int]: Database image IDs (1-based), indexed by 0-based image index.
#[pyfunction]
pub fn write_colmap_db(
    py: Python<'_>,
    db_path: PathBuf,
    data: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let cameras = extract_cameras_as_sfmr(&get_item(data, "cameras")?)?;
    let image_names: Vec<String> = get_item(data, "image_names")?.extract()?;
    let camera_indexes: PyReadonlyArray1<u32> = get_item(data, "camera_indexes")?.extract()?;
    let quaternions_wxyz: PyReadonlyArray2<f64> = get_item(data, "quaternions_wxyz")?.extract()?;
    let translations_xyz: PyReadonlyArray2<f64> = get_item(data, "translations_xyz")?.extract()?;

    // Extract per-image keypoint arrays: list of (K,2) float64
    let kp_list_obj = get_item(data, "keypoints_per_image")?;
    let kp_py_list = kp_list_obj.downcast::<PyList>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("keypoints_per_image must be a list")
    })?;
    let mut keypoints_per_image: Vec<Vec<[f64; 2]>> = Vec::with_capacity(kp_py_list.len());
    for item in kp_py_list.iter() {
        let arr: PyReadonlyArray2<f64> = item.extract()?;
        let view = arr.as_array();
        let k = view.shape()[0];
        let mut kps = Vec::with_capacity(k);
        for i in 0..k {
            kps.push([view[[i, 0]], view[[i, 1]]]);
        }
        keypoints_per_image.push(kps);
    }

    // Extract per-image descriptor arrays: list of (K,D) uint8
    let desc_list_obj = get_item(data, "descriptors_per_image")?;
    let desc_py_list = desc_list_obj.downcast::<PyList>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("descriptors_per_image must be a list")
    })?;
    let mut descriptors_per_image: Vec<Vec<u8>> = Vec::with_capacity(desc_py_list.len());
    for item in desc_py_list.iter() {
        let arr: PyReadonlyArray2<u8> = item.extract()?;
        let view = arr.as_array();
        // Flatten to contiguous u8 vec
        descriptors_per_image.push(view.iter().copied().collect());
    }

    // Descriptor dimensionality (default 128)
    let descriptor_dim: u32 = data
        .get_item("descriptor_dim")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(128);

    // Convert numpy arrays to Rust vecs
    let cam_idx_slice = camera_indexes.as_slice()?;
    let quat_view = quaternions_wxyz.as_array();
    let trans_view = translations_xyz.as_array();
    let n = image_names.len();
    let quats: Vec<[f64; 4]> = (0..n)
        .map(|i| {
            [
                quat_view[[i, 0]],
                quat_view[[i, 1]],
                quat_view[[i, 2]],
                quat_view[[i, 3]],
            ]
        })
        .collect();
    let trans: Vec<[f64; 3]> = (0..n)
        .map(|i| [trans_view[[i, 0]], trans_view[[i, 1]], trans_view[[i, 2]]])
        .collect();

    // Extract optional pose priors
    let pose_priors: Option<Vec<colmap_db::PosePrior>> = if let Some(priors_obj) =
        data.get_item("pose_priors")?
    {
        let priors_list = priors_obj
            .downcast::<PyList>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("pose_priors must be a list"))?;
        let mut priors = Vec::with_capacity(priors_list.len());
        for item in priors_list.iter() {
            let dict = item.downcast::<PyDict>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("pose_prior entries must be dicts")
            })?;
            let pos: PyReadonlyArray1<f64> = get_item(dict, "position")?.extract()?;
            let pos_slice = pos.as_slice()?;
            let cov: PyReadonlyArray2<f64> = get_item(dict, "position_covariance")?.extract()?;
            let cov_view = cov.as_array();
            let coord_sys: i32 = get_item(dict, "coordinate_system")?.extract()?;

            let mut cov_arr = [0.0f64; 9];
            for r in 0..3 {
                for c in 0..3 {
                    cov_arr[r * 3 + c] = cov_view[[r, c]];
                }
            }

            priors.push(colmap_db::PosePrior {
                position: [pos_slice[0], pos_slice[1], pos_slice[2]],
                position_covariance: cov_arr,
                coordinate_system: coord_sys,
            });
        }
        Some(priors)
    } else {
        None
    };

    // Extract optional two-view geometries
    let two_view_geometries: Option<Vec<colmap_db::TwoViewGeometry>> = if let Some(tvg_obj) =
        data.get_item("two_view_geometries")?
    {
        let tvg_list = tvg_obj.downcast::<PyList>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("two_view_geometries must be a list")
        })?;
        let mut tvgs = Vec::with_capacity(tvg_list.len());
        for item in tvg_list.iter() {
            let dict = item.downcast::<PyDict>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("two_view_geometry entries must be dicts")
            })?;

            let image_idx1: u32 = get_item(dict, "image_idx1")?.extract()?;
            let image_idx2: u32 = get_item(dict, "image_idx2")?.extract()?;
            let config: i32 = get_item(dict, "config")?.extract()?;

            // Matches: (M,2) uint32 array, flatten to interleaved
            let matches: Vec<u32> = if let Some(matches_obj) = dict.get_item("matches")? {
                let arr: PyReadonlyArray2<u32> = matches_obj.extract()?;
                let view = arr.as_array();
                let m = view.shape()[0];
                let mut flat = Vec::with_capacity(m * 2);
                for i in 0..m {
                    flat.push(view[[i, 0]]);
                    flat.push(view[[i, 1]]);
                }
                flat
            } else {
                vec![]
            };

            let f_matrix = extract_optional_3x3_matrix(dict, "f_matrix")?;
            let e_matrix = extract_optional_3x3_matrix(dict, "e_matrix")?;
            let h_matrix = extract_optional_3x3_matrix(dict, "h_matrix")?;

            let qvec_wxyz: Option<[f64; 4]> = if let Some(q_obj) = dict.get_item("qvec_wxyz")? {
                let arr: PyReadonlyArray1<f64> = q_obj.extract()?;
                let s = arr.as_slice()?;
                Some([s[0], s[1], s[2], s[3]])
            } else {
                None
            };

            let tvec: Option<[f64; 3]> = if let Some(t_obj) = dict.get_item("tvec")? {
                let arr: PyReadonlyArray1<f64> = t_obj.extract()?;
                let s = arr.as_slice()?;
                Some([s[0], s[1], s[2]])
            } else {
                None
            };

            tvgs.push(colmap_db::TwoViewGeometry {
                image_idx1,
                image_idx2,
                matches,
                config: match config {
                    0 => colmap_db::TwoViewGeometryConfig::Undefined,
                    1 => colmap_db::TwoViewGeometryConfig::Degenerate,
                    2 => colmap_db::TwoViewGeometryConfig::Calibrated,
                    3 => colmap_db::TwoViewGeometryConfig::Uncalibrated,
                    4 => colmap_db::TwoViewGeometryConfig::Planar,
                    5 => colmap_db::TwoViewGeometryConfig::PlanarOrPanoramic,
                    6 => colmap_db::TwoViewGeometryConfig::Panoramic,
                    7 => colmap_db::TwoViewGeometryConfig::Multiple,
                    8 => colmap_db::TwoViewGeometryConfig::WatermarkClean,
                    9 => colmap_db::TwoViewGeometryConfig::WatermarkBad,
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid two-view geometry config: {config}"
                        )));
                    }
                },
                f_matrix,
                e_matrix,
                h_matrix,
                qvec_wxyz,
                tvec,
            });
        }
        Some(tvgs)
    } else {
        None
    };

    let write_data = colmap_db::ColmapDbWriteData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: cam_idx_slice,
        quaternions_wxyz: &quats,
        translations_xyz: &trans,
        keypoints_per_image: &keypoints_per_image,
        descriptors_per_image: &descriptors_per_image,
        descriptor_dim,
        pose_priors: pose_priors.as_deref(),
        two_view_geometries: two_view_geometries.as_deref(),
        rigs: None,
        frames: None,
    };

    let image_ids = colmap_db::write_colmap_db(&db_path, &write_data)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    Ok(PyList::new(py, &image_ids)?.into())
}

/// Read matches and two-view geometries from a COLMAP SQLite database.
///
/// Returns a dict with the same structure as `read_matches`, with placeholder
/// metadata. Descriptor distances are set to 0.0 (not stored in the DB).
///
/// Args:
///   db_path: Path to the COLMAP database file.
///   include_tvg: Whether to read the two_view_geometries table.
///
/// Returns:
///   Dict with matches data (same format as read_matches output).
#[pyfunction]
#[pyo3(signature = (db_path, include_tvg=true))]
pub fn read_colmap_db_matches(
    py: Python<'_>,
    db_path: PathBuf,
    include_tvg: bool,
) -> PyResult<Py<PyAny>> {
    let data = colmap_db::read_colmap_db_matches(&db_path, include_tvg)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    matches_data_to_py(py, data)
}
