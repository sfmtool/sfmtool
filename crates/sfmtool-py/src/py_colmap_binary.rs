// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for COLMAP binary reconstruction I/O.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::Path;

use sfmr_colmap::colmap_io;
use sfmr_format::SfmrCamera;

use crate::helpers::{extract_cameras_as_sfmr, get_item};
use crate::PyCameraIntrinsics;

/// Read a COLMAP binary reconstruction from a directory.
///
/// The directory must contain `cameras.bin`, `images.bin`, and `points3D.bin`.
///
/// Returns a dict with keys:
///   cameras (list[CameraIntrinsics]),
///   image_names (list[str]),
///   camera_indexes (N,) uint32,
///   quaternions_wxyz (N,4) float64,
///   translations_xyz (N,3) float64,
///   keypoints_per_image (list of (K,3) float64 arrays: x, y, point3d_idx or -1),
///   positions_xyz (P,3) float64,
///   colors_rgb (P,3) uint8,
///   reprojection_errors (P,) float64,
///   track_image_indexes (M,) uint32,
///   track_feature_indexes (M,) uint32,
///   track_point3d_indexes (M,) uint32,
///   observation_counts (P,) uint32.
#[pyfunction]
pub fn read_colmap_binary(py: Python<'_>, dir: &str) -> PyResult<Py<PyAny>> {
    let recon = colmap_io::read_colmap_binary(Path::new(dir))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);

    // Cameras as CameraIntrinsics objects
    let cameras: Vec<PyCameraIntrinsics> = recon
        .cameras
        .iter()
        .map(|c| {
            let inner = sfmtool_core::CameraIntrinsics::try_from(c)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyCameraIntrinsics { inner })
        })
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("cameras", PyList::new(py, cameras)?)?;

    dict.set_item("image_names", &recon.image_names)?;

    // Rig/frame data (if rigs.bin and frames.bin exist) - extract before consuming camera_indexes
    let rig_frame_py = if let (Some(rigs), Some(frames)) = (&recon.rigs, &recon.frames) {
        Some(colmap_rigs_frames_to_py(
            py,
            rigs,
            frames,
            &recon.camera_indexes,
        )?)
    } else {
        None
    };

    // Camera indexes
    let n = recon.image_names.len();
    let camera_indexes = ndarray::Array1::from_vec(recon.camera_indexes).into_pyarray(py);
    dict.set_item("camera_indexes", camera_indexes)?;

    // Quaternions (N,4)
    let mut quat_data = Vec::with_capacity(n * 4);
    for q in &recon.quaternions_wxyz {
        quat_data.extend_from_slice(q);
    }
    let quaternions = ndarray::Array2::from_shape_vec((n, 4), quat_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    dict.set_item("quaternions_wxyz", quaternions)?;

    // Translations (N,3)
    let mut trans_data = Vec::with_capacity(n * 3);
    for t in &recon.translations_xyz {
        trans_data.extend_from_slice(t);
    }
    let translations = ndarray::Array2::from_shape_vec((n, 3), trans_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    dict.set_item("translations_xyz", translations)?;

    // Keypoints per image: list of (K,3) arrays where columns are [x, y, point3d_idx_or_neg1]
    let kp_list: Vec<_> = recon
        .keypoints_per_image
        .iter()
        .map(|kps| {
            let k = kps.len();
            let mut data = Vec::with_capacity(k * 3);
            for kp in kps {
                data.push(kp.x);
                data.push(kp.y);
                data.push(match kp.point3d_index {
                    Some(idx) => idx as f64,
                    None => -1.0,
                });
            }
            ndarray::Array2::from_shape_vec((k, 3), data)
                .expect("keypoint array shape is k*3 by construction")
                .into_pyarray(py)
        })
        .collect();
    dict.set_item("keypoints_per_image", PyList::new(py, &kp_list)?)?;

    // Points3D
    let p = recon.positions_xyz.len();
    let mut pos_data = Vec::with_capacity(p * 3);
    for pt in &recon.positions_xyz {
        pos_data.extend_from_slice(pt);
    }
    let positions = ndarray::Array2::from_shape_vec((p, 3), pos_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    dict.set_item("positions_xyz", positions)?;

    let mut color_data = Vec::with_capacity(p * 3);
    for c in &recon.colors_rgb {
        color_data.extend_from_slice(c);
    }
    let colors = ndarray::Array2::from_shape_vec((p, 3), color_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    dict.set_item("colors_rgb", colors)?;

    let errors = ndarray::Array1::from_vec(
        recon
            .reprojection_errors
            .iter()
            .map(|&e| e as f32)
            .collect(),
    )
    .into_pyarray(py);
    dict.set_item("reprojection_errors", errors)?;

    // Flatten tracks into parallel arrays + observation_counts
    let mut track_image_indexes = Vec::new();
    let mut track_feature_indexes = Vec::new();
    let mut track_point3d_indexes = Vec::new();
    let mut observation_counts = Vec::with_capacity(p);

    for (pt_idx, track) in recon.tracks.iter().enumerate() {
        observation_counts.push(track.len() as u32);
        for &(img_idx, feat_idx) in track {
            track_image_indexes.push(img_idx);
            track_feature_indexes.push(feat_idx);
            track_point3d_indexes.push(pt_idx as u32);
        }
    }

    dict.set_item(
        "track_image_indexes",
        ndarray::Array1::from_vec(track_image_indexes).into_pyarray(py),
    )?;
    dict.set_item(
        "track_feature_indexes",
        ndarray::Array1::from_vec(track_feature_indexes).into_pyarray(py),
    )?;
    dict.set_item(
        "track_point3d_indexes",
        ndarray::Array1::from_vec(track_point3d_indexes).into_pyarray(py),
    )?;
    dict.set_item(
        "observation_counts",
        ndarray::Array1::from_vec(observation_counts).into_pyarray(py),
    )?;

    if let Some(rig_frame_dict) = rig_frame_py {
        dict.set_item("rig_frame_data", rig_frame_dict)?;
    }

    Ok(dict.into())
}

/// Convert COLMAP rig and frame data to a Python rig_frame_data dict
/// matching the sfmr format.
fn colmap_rigs_frames_to_py<'py>(
    py: Python<'py>,
    rigs: &[colmap_io::ColmapRig],
    frames: &[colmap_io::ColmapFrame],
    camera_indexes: &[u32],
) -> PyResult<Py<PyAny>> {
    use std::collections::HashMap;

    // Check if all rigs are trivial (single sensor, no non-ref sensors)
    let all_trivial = rigs.iter().all(|r| r.non_ref_sensors.is_empty());
    if all_trivial {
        return Ok(py.None());
    }

    let num_images = camera_indexes.len();

    // Build rig definitions and sensor arrays
    // Sort by rig_id for deterministic ordering
    let mut sorted_rigs: Vec<&colmap_io::ColmapRig> = rigs.iter().collect();
    sorted_rigs.sort_by_key(|r| r.rig_id);

    let mut rig_id_to_index: HashMap<u32, usize> = HashMap::new();
    let mut rig_defs = Vec::new();
    let mut all_sensor_camera_indexes: Vec<u32> = Vec::new();
    let mut all_sensor_quaternions: Vec<f64> = Vec::new();
    let mut all_sensor_translations: Vec<f64> = Vec::new();
    let mut global_sensor_offset: usize = 0;

    // Map (rig_id, sensor_type, sensor_camera_index) -> global sensor index.
    // Sensor IDs are already remapped to 0-based camera indexes by read_colmap_binary.
    let mut sensor_key_to_global: HashMap<(u32, i32, u32), usize> = HashMap::new();

    for (rig_idx, rig) in sorted_rigs.iter().enumerate() {
        rig_id_to_index.insert(rig.rig_id, rig_idx);

        let mut sensor_names = Vec::new();
        let mut sensor_count: usize = 0;

        // Reference sensor first
        if let Some(ref ref_sensor) = rig.ref_sensor {
            let key = (rig.rig_id, ref_sensor.sensor_type as i32, ref_sensor.id);
            sensor_key_to_global.insert(key, global_sensor_offset + sensor_count);
            all_sensor_camera_indexes.push(ref_sensor.id);
            all_sensor_quaternions.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            all_sensor_translations.extend_from_slice(&[0.0, 0.0, 0.0]);
            sensor_names.push(format!("sensor{}", sensor_count));
            sensor_count += 1;
        }

        // Non-reference sensors
        for non_ref in &rig.non_ref_sensors {
            let key = (
                rig.rig_id,
                non_ref.sensor.sensor_type as i32,
                non_ref.sensor.id,
            );
            sensor_key_to_global.insert(key, global_sensor_offset + sensor_count);
            all_sensor_camera_indexes.push(non_ref.sensor.id);

            if let Some((quat, trans)) = non_ref.sensor_from_rig {
                all_sensor_quaternions.extend_from_slice(&quat);
                all_sensor_translations.extend_from_slice(&trans);
            } else {
                all_sensor_quaternions.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
                all_sensor_translations.extend_from_slice(&[0.0, 0.0, 0.0]);
            }
            sensor_names.push(format!("sensor{}", sensor_count));
            sensor_count += 1;
        }

        let rig_def = PyDict::new(py);
        rig_def.set_item("name", format!("rig{}", rig_idx))?;
        rig_def.set_item("sensor_count", sensor_count)?;
        rig_def.set_item("sensor_offset", global_sensor_offset)?;
        rig_def.set_item("ref_sensor_name", "sensor0")?;
        rig_def.set_item("sensor_names", &sensor_names)?;
        rig_defs.push(rig_def);

        global_sensor_offset += sensor_count;
    }

    let total_sensors = global_sensor_offset;

    // Build rigs_metadata dict
    let rigs_metadata = PyDict::new(py);
    rigs_metadata.set_item("rig_count", sorted_rigs.len())?;
    rigs_metadata.set_item("sensor_count", total_sensors)?;
    rigs_metadata.set_item("rigs", PyList::new(py, &rig_defs)?)?;

    // Build frame arrays.
    // Frame data_ids have already been remapped by read_colmap_binary:
    // - data_id.sensor_id is a 0-based camera index
    // - data_id.data_id is a 0-based sorted image index
    let mut sorted_frames: Vec<&colmap_io::ColmapFrame> = frames.iter().collect();
    sorted_frames.sort_by_key(|f| f.frame_id);
    let num_frames = sorted_frames.len();

    let mut rig_indexes = Vec::with_capacity(num_frames);
    let mut image_sensor_indexes = vec![0u32; num_images];
    let mut image_frame_indexes = vec![0u32; num_images];

    for (frame_idx, frame) in sorted_frames.iter().enumerate() {
        let rig_idx = *rig_id_to_index.get(&frame.rig_id).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Frame {} references unknown rig_id {}",
                frame.frame_id, frame.rig_id
            ))
        })?;
        rig_indexes.push(rig_idx as u32);

        for data_id in &frame.data_ids {
            let img_idx = data_id.data_id as usize;
            if img_idx >= num_images {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Frame {} data_id references image index {} but only {} images exist",
                    frame.frame_id, img_idx, num_images
                )));
            }
            let sensor_key = (frame.rig_id, data_id.sensor_type as i32, data_id.sensor_id);
            let global_sensor_idx = *sensor_key_to_global.get(&sensor_key).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Frame {} data_id references unknown sensor (rig_id={}, type={}, id={})",
                    frame.frame_id, frame.rig_id, sensor_key.1, data_id.sensor_id
                ))
            })?;
            image_sensor_indexes[img_idx] = global_sensor_idx as u32;
            image_frame_indexes[img_idx] = frame_idx as u32;
        }
    }

    let result = PyDict::new(py);
    result.set_item("rigs_metadata", rigs_metadata)?;
    result.set_item(
        "sensor_camera_indexes",
        ndarray::Array1::from_vec(all_sensor_camera_indexes).into_pyarray(py),
    )?;
    let s = total_sensors;
    result.set_item(
        "sensor_quaternions_wxyz",
        ndarray::Array2::from_shape_vec((s, 4), all_sensor_quaternions)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py),
    )?;
    result.set_item(
        "sensor_translations_xyz",
        ndarray::Array2::from_shape_vec((s, 3), all_sensor_translations)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py),
    )?;

    let frames_metadata = PyDict::new(py);
    frames_metadata.set_item("frame_count", num_frames)?;
    result.set_item("frames_metadata", frames_metadata)?;
    result.set_item(
        "rig_indexes",
        ndarray::Array1::from_vec(rig_indexes).into_pyarray(py),
    )?;
    result.set_item(
        "image_sensor_indexes",
        ndarray::Array1::from_vec(image_sensor_indexes).into_pyarray(py),
    )?;
    result.set_item(
        "image_frame_indexes",
        ndarray::Array1::from_vec(image_frame_indexes).into_pyarray(py),
    )?;

    Ok(result.into())
}

/// Write a COLMAP binary reconstruction to a directory.
///
/// Creates `cameras.bin`, `images.bin`, and `points3D.bin` in the given directory.
///
/// Args:
///   dir: Output directory path
///   data: Dict with keys matching the output of `read_colmap_binary`, plus
///     `keypoints_per_image` as list of (K,2) float64 arrays (x, y positions).
#[pyfunction]
pub fn write_colmap_binary(dir: &str, data: &Bound<'_, PyDict>) -> PyResult<()> {
    let cameras: Vec<SfmrCamera> = extract_cameras_as_sfmr(&get_item(data, "cameras")?)?;
    let image_names: Vec<String> = get_item(data, "image_names")?.extract()?;

    let camera_indexes: PyReadonlyArray1<u32> = get_item(data, "camera_indexes")?.extract()?;
    let quaternions_wxyz: PyReadonlyArray2<f64> = get_item(data, "quaternions_wxyz")?.extract()?;
    let translations_xyz: PyReadonlyArray2<f64> = get_item(data, "translations_xyz")?.extract()?;
    let positions_xyz: PyReadonlyArray2<f64> = get_item(data, "positions_xyz")?.extract()?;
    let colors_rgb: PyReadonlyArray2<u8> = get_item(data, "colors_rgb")?.extract()?;
    let reprojection_errors: PyReadonlyArray1<f32> =
        get_item(data, "reprojection_errors")?.extract()?;
    let track_image_indexes: PyReadonlyArray1<u32> =
        get_item(data, "track_image_indexes")?.extract()?;
    let track_feature_indexes: PyReadonlyArray1<u32> =
        get_item(data, "track_feature_indexes")?.extract()?;
    let track_point3d_indexes: PyReadonlyArray1<u32> =
        get_item(data, "track_point3d_indexes")?.extract()?;

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

    // Convert 2D arrays to slices of fixed-size arrays
    let cam_idx_slice = camera_indexes.as_slice()?;
    let track_img_slice = track_image_indexes.as_slice()?;
    let track_feat_slice = track_feature_indexes.as_slice()?;
    let track_pt_slice = track_point3d_indexes.as_slice()?;

    // Convert 2D ndarray views to Vec of fixed-size arrays
    let quat_view = quaternions_wxyz.as_array();
    let trans_view = translations_xyz.as_array();
    let pos_view = positions_xyz.as_array();
    let col_view = colors_rgb.as_array();
    let err_view = reprojection_errors.as_array();

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

    let p = pos_view.shape()[0];
    let positions: Vec<[f64; 3]> = (0..p)
        .map(|i| [pos_view[[i, 0]], pos_view[[i, 1]], pos_view[[i, 2]]])
        .collect();
    let colors: Vec<[u8; 3]> = (0..p)
        .map(|i| [col_view[[i, 0]], col_view[[i, 1]], col_view[[i, 2]]])
        .collect();
    let errors: Vec<f64> = (0..p).map(|i| err_view[i] as f64).collect();

    // Extract optional rig/frame data
    let (colmap_rigs, colmap_frames) = extract_colmap_rigs_frames(data, &quats, &trans)?;

    let write_data = colmap_io::ColmapWriteData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: cam_idx_slice,
        quaternions_wxyz: &quats,
        translations_xyz: &trans,
        positions_xyz: &positions,
        colors_rgb: &colors,
        reprojection_errors: &errors,
        track_image_indexes: track_img_slice,
        track_feature_indexes: track_feat_slice,
        track_point3d_indexes: track_pt_slice,
        keypoints_per_image: &keypoints_per_image,
        rigs: colmap_rigs.as_deref(),
        frames: colmap_frames.as_deref(),
    };

    colmap_io::write_colmap_binary(Path::new(dir), &write_data)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Extract optional rig_frame_data from the Python dict and convert to ColmapRig/ColmapFrame.
///
/// Returns (None, None) if the dict has no "rig_frame_data" key.
#[allow(clippy::type_complexity)]
fn extract_colmap_rigs_frames(
    data: &Bound<'_, PyDict>,
    image_quats: &[[f64; 4]],
    image_trans: &[[f64; 3]],
) -> PyResult<(
    Option<Vec<colmap_io::ColmapRig>>,
    Option<Vec<colmap_io::ColmapFrame>>,
)> {
    use colmap_io::{
        ColmapDataId, ColmapFrame, ColmapRig, ColmapRigSensor, ColmapSensor, ColmapSensorType,
    };

    let rfd = match data.get_item("rig_frame_data")? {
        Some(v) => v,
        None => return Ok((None, None)),
    };
    let rfd = rfd
        .downcast::<PyDict>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("rig_frame_data must be a dict"))?;

    // Extract arrays
    let sensor_camera_indexes: PyReadonlyArray1<u32> =
        get_item(rfd, "sensor_camera_indexes")?.extract()?;
    let sensor_quats: PyReadonlyArray2<f64> =
        get_item(rfd, "sensor_quaternions_wxyz")?.extract()?;
    let sensor_trans: PyReadonlyArray2<f64> =
        get_item(rfd, "sensor_translations_xyz")?.extract()?;
    let rig_indexes: PyReadonlyArray1<u32> = get_item(rfd, "rig_indexes")?.extract()?;
    let image_sensor_indexes: PyReadonlyArray1<u32> =
        get_item(rfd, "image_sensor_indexes")?.extract()?;
    let image_frame_indexes: PyReadonlyArray1<u32> =
        get_item(rfd, "image_frame_indexes")?.extract()?;

    let sensor_cam_slice = sensor_camera_indexes.as_slice()?;
    let sensor_quat_view = sensor_quats.as_array();
    let sensor_trans_view = sensor_trans.as_array();
    let rig_idx_slice = rig_indexes.as_slice()?;
    let img_sensor_slice = image_sensor_indexes.as_slice()?;
    let img_frame_slice = image_frame_indexes.as_slice()?;

    // Extract rigs_metadata
    let rigs_metadata = get_item(rfd, "rigs_metadata")?;
    let rigs_metadata = rigs_metadata
        .downcast::<PyDict>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("rigs_metadata must be a dict"))?;
    let rig_defs_list = get_item(rigs_metadata, "rigs")?;
    let rig_defs_list = rig_defs_list
        .downcast::<PyList>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("rigs must be a list"))?;

    // Build ColmapRig for each rig definition
    let mut colmap_rigs = Vec::with_capacity(rig_defs_list.len());
    for (rig_idx, rig_def_obj) in rig_defs_list.iter().enumerate() {
        let rig_def = rig_def_obj
            .downcast::<PyDict>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("rig def must be a dict"))?;
        let sensor_count: usize = get_item(rig_def, "sensor_count")?.extract()?;
        let sensor_offset: usize = get_item(rig_def, "sensor_offset")?.extract()?;

        // Reference sensor (first in the rig) — camera_index is 0-based, camera_id is 1-based
        if sensor_offset >= sensor_cam_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Rig {} sensor_offset {} out of bounds (sensor_camera_indexes length {})",
                rig_idx,
                sensor_offset,
                sensor_cam_slice.len()
            )));
        }
        let ref_camera_id = sensor_cam_slice[sensor_offset] + 1;
        let ref_sensor = Some(ColmapSensor {
            sensor_type: ColmapSensorType::Camera,
            id: ref_camera_id,
        });

        // Non-reference sensors
        let mut non_ref_sensors = Vec::with_capacity(sensor_count.saturating_sub(1));
        for j in 1..sensor_count {
            let gs_idx = sensor_offset + j;
            if gs_idx >= sensor_cam_slice.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Rig {} sensor index {} out of bounds (sensor_camera_indexes length {})",
                    rig_idx,
                    gs_idx,
                    sensor_cam_slice.len()
                )));
            }
            let camera_id = sensor_cam_slice[gs_idx] + 1;
            let quat = [
                sensor_quat_view[[gs_idx, 0]],
                sensor_quat_view[[gs_idx, 1]],
                sensor_quat_view[[gs_idx, 2]],
                sensor_quat_view[[gs_idx, 3]],
            ];
            let trans = [
                sensor_trans_view[[gs_idx, 0]],
                sensor_trans_view[[gs_idx, 1]],
                sensor_trans_view[[gs_idx, 2]],
            ];

            // Check if pose is identity (no sensor_from_rig)
            let is_identity = (quat[0] - 1.0).abs() < 1e-12
                && quat[1].abs() < 1e-12
                && quat[2].abs() < 1e-12
                && quat[3].abs() < 1e-12
                && trans[0].abs() < 1e-12
                && trans[1].abs() < 1e-12
                && trans[2].abs() < 1e-12;

            non_ref_sensors.push(ColmapRigSensor {
                sensor: ColmapSensor {
                    sensor_type: ColmapSensorType::Camera,
                    id: camera_id,
                },
                sensor_from_rig: if is_identity {
                    None
                } else {
                    Some((quat, trans))
                },
            });
        }

        colmap_rigs.push(ColmapRig {
            rig_id: rig_idx as u32,
            ref_sensor,
            non_ref_sensors,
        });
    }

    // Build ColmapFrame for each frame
    let num_frames = rig_idx_slice.len();
    let num_images = img_sensor_slice.len();

    // Build per-frame image lists: frame_idx -> [(global_sensor_idx, image_idx), ...]
    let mut frame_images: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_frames];
    for img_idx in 0..num_images {
        let frame_idx = img_frame_slice[img_idx] as usize;
        let gs_idx = img_sensor_slice[img_idx] as usize;
        if frame_idx >= num_frames {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Image {} has frame index {} but only {} frames exist",
                img_idx, frame_idx, num_frames
            )));
        }
        frame_images[frame_idx].push((gs_idx, img_idx));
    }

    // For each rig, find the reference sensor's global sensor offset
    let mut rig_ref_offsets: Vec<usize> = Vec::with_capacity(rig_defs_list.len());
    for rig_def_obj in rig_defs_list.iter() {
        let rig_def = rig_def_obj
            .downcast::<PyDict>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("rig def must be a dict"))?;
        let offset: usize = get_item(rig_def, "sensor_offset")?.extract()?;
        rig_ref_offsets.push(offset);
    }

    let mut colmap_frames = Vec::with_capacity(num_frames);
    for frame_idx in 0..num_frames {
        let rig_idx = rig_idx_slice[frame_idx] as usize;
        if rig_idx >= rig_ref_offsets.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Frame {} references rig index {} but only {} rigs exist",
                frame_idx,
                rig_idx,
                rig_ref_offsets.len()
            )));
        }
        let ref_gs_idx = rig_ref_offsets[rig_idx];

        // Find reference sensor's image to get rig_from_world pose
        let ref_img_idx = frame_images[frame_idx]
            .iter()
            .find(|(gs_idx, _)| *gs_idx == ref_gs_idx)
            .map(|(_, img_idx)| *img_idx);

        let (quat, trans) = match ref_img_idx {
            Some(idx) => (image_quats[idx], image_trans[idx]),
            None => {
                // Fallback: use first image in the frame
                if let Some(&(_, idx)) = frame_images[frame_idx].first() {
                    (image_quats[idx], image_trans[idx])
                } else {
                    ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                }
            }
        };

        let mut data_ids: Vec<ColmapDataId> = Vec::with_capacity(frame_images[frame_idx].len());
        for &(gs_idx, img_idx) in &frame_images[frame_idx] {
            if gs_idx >= sensor_cam_slice.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Frame {} references sensor index {} but only {} sensors exist",
                    frame_idx,
                    gs_idx,
                    sensor_cam_slice.len()
                )));
            }
            let camera_id = sensor_cam_slice[gs_idx] + 1; // 1-based
            data_ids.push(ColmapDataId {
                sensor_type: ColmapSensorType::Camera,
                sensor_id: camera_id,
                data_id: (img_idx as u64) + 1, // 1-based image_id
            });
        }

        colmap_frames.push(ColmapFrame {
            frame_id: frame_idx as u32,
            rig_id: rig_idx as u32,
            quaternion_wxyz: quat,
            translation_xyz: trans,
            data_ids,
        });
    }

    Ok((Some(colmap_rigs), Some(colmap_frames)))
}
