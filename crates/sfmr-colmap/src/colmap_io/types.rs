// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Types for COLMAP binary I/O.

use std::collections::HashMap;

use thiserror::Error;

use sfmr_format::SfmrCamera;

/// Errors that can occur when reading or writing COLMAP binary files.
#[derive(Error, Debug)]
pub enum ColmapIoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{operation} '{path}': {source}")]
    IoPath {
        operation: &'static str,
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("Unknown camera model ID: {0}")]
    UnknownModelId(i32),
    #[error("Unknown camera model name: {0}")]
    UnknownModelName(String),
    #[error("Unexpected end of file in {0}")]
    UnexpectedEof(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// COLMAP sensor type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColmapSensorType {
    Camera = 0,
    Imu = 1,
}

impl ColmapSensorType {
    pub fn from_i32(v: i32) -> Result<Self, ColmapIoError> {
        match v {
            0 => Ok(Self::Camera),
            1 => Ok(Self::Imu),
            _ => Err(ColmapIoError::InvalidData(format!(
                "Unknown sensor type: {}",
                v
            ))),
        }
    }
}

/// A COLMAP sensor identifier (type + id).
#[derive(Debug, Clone)]
pub struct ColmapSensor {
    pub sensor_type: ColmapSensorType,
    pub id: u32,
}

/// A non-reference sensor in a rig, with optional sensor_from_rig pose.
#[derive(Debug, Clone)]
pub struct ColmapRigSensor {
    pub sensor: ColmapSensor,
    /// sensor_from_rig transform. WXYZ quaternion + XYZ translation.
    pub sensor_from_rig: Option<([f64; 4], [f64; 3])>,
}

/// A COLMAP rig definition.
#[derive(Debug, Clone)]
pub struct ColmapRig {
    pub rig_id: u32,
    pub ref_sensor: Option<ColmapSensor>,
    pub non_ref_sensors: Vec<ColmapRigSensor>,
}

/// A data_id entry in a COLMAP frame (sensor + image mapping).
#[derive(Debug, Clone)]
pub struct ColmapDataId {
    pub sensor_type: ColmapSensorType,
    pub sensor_id: u32,
    pub data_id: u64,
}

/// A COLMAP frame (temporal instant with rig pose and image assignments).
#[derive(Debug, Clone)]
pub struct ColmapFrame {
    pub frame_id: u32,
    pub rig_id: u32,
    /// rig_from_world: WXYZ quaternion
    pub quaternion_wxyz: [f64; 4],
    /// rig_from_world: XYZ translation
    pub translation_xyz: [f64; 3],
    pub data_ids: Vec<ColmapDataId>,
}

/// A parsed COLMAP reconstruction with 0-based indices.
pub struct ColmapReconstruction {
    pub cameras: Vec<SfmrCamera>,
    pub image_names: Vec<String>,
    pub camera_indexes: Vec<u32>,
    pub quaternions_wxyz: Vec<[f64; 4]>,
    pub translations_xyz: Vec<[f64; 3]>,
    /// Per-image 2D keypoints: (x, y, point3d_0based_index_or_None)
    pub keypoints_per_image: Vec<Vec<Keypoint2D>>,
    pub positions_xyz: Vec<[f64; 3]>,
    pub colors_rgb: Vec<[u8; 3]>,
    pub reprojection_errors: Vec<f64>,
    /// Track observations per 3D point: list of (image_index, feature_index), 0-based
    pub tracks: Vec<Vec<(u32, u32)>>,
    /// Rig definitions from rigs.bin (None if no rigs.bin exists)
    pub rigs: Option<Vec<ColmapRig>>,
    /// Frame definitions from frames.bin (None if no frames.bin exists)
    pub frames: Option<Vec<ColmapFrame>>,
}

/// A 2D keypoint observation in an image.
pub struct Keypoint2D {
    pub x: f64,
    pub y: f64,
    /// 0-based index into positions_xyz, None if unobserved
    pub point3d_index: Option<u32>,
}

/// Data needed to write COLMAP binary files.
pub struct ColmapWriteData<'a> {
    pub cameras: &'a [SfmrCamera],
    pub image_names: &'a [String],
    pub camera_indexes: &'a [u32],
    pub quaternions_wxyz: &'a [[f64; 4]],
    pub translations_xyz: &'a [[f64; 3]],
    pub positions_xyz: &'a [[f64; 3]],
    pub colors_rgb: &'a [[u8; 3]],
    pub reprojection_errors: &'a [f64],
    /// Flat track observations, all 0-based.
    pub track_image_indexes: &'a [u32],
    pub track_feature_indexes: &'a [u32],
    pub track_point3d_indexes: &'a [u32],
    /// Per-image 2D keypoint positions (x, y)
    pub keypoints_per_image: &'a [Vec<[f64; 2]>],
    /// Optional rig definitions (written to rigs.bin)
    pub rigs: Option<&'a [ColmapRig]>,
    /// Optional frame definitions (written to frames.bin)
    pub frames: Option<&'a [ColmapFrame]>,
}

/// Camera model definitions: (model_id, model_name, num_params, param_names).
const CAMERA_MODELS: &[(i32, &str, usize, &[&str])] = &[
    (
        0,
        "SIMPLE_PINHOLE",
        3,
        &["focal_length", "principal_point_x", "principal_point_y"],
    ),
    (
        1,
        "PINHOLE",
        4,
        &[
            "focal_length_x",
            "focal_length_y",
            "principal_point_x",
            "principal_point_y",
        ],
    ),
    (
        2,
        "SIMPLE_RADIAL",
        4,
        &[
            "focal_length",
            "principal_point_x",
            "principal_point_y",
            "radial_distortion_k1",
        ],
    ),
    (
        3,
        "RADIAL",
        5,
        &[
            "focal_length",
            "principal_point_x",
            "principal_point_y",
            "radial_distortion_k1",
            "radial_distortion_k2",
        ],
    ),
    (
        4,
        "OPENCV",
        8,
        &[
            "focal_length_x",
            "focal_length_y",
            "principal_point_x",
            "principal_point_y",
            "radial_distortion_k1",
            "radial_distortion_k2",
            "tangential_distortion_p1",
            "tangential_distortion_p2",
        ],
    ),
    (
        5,
        "OPENCV_FISHEYE",
        8,
        &[
            "focal_length_x",
            "focal_length_y",
            "principal_point_x",
            "principal_point_y",
            "radial_distortion_k1",
            "radial_distortion_k2",
            "radial_distortion_k3",
            "radial_distortion_k4",
        ],
    ),
    (
        6,
        "FULL_OPENCV",
        12,
        &[
            "focal_length_x",
            "focal_length_y",
            "principal_point_x",
            "principal_point_y",
            "radial_distortion_k1",
            "radial_distortion_k2",
            "tangential_distortion_p1",
            "tangential_distortion_p2",
            "radial_distortion_k3",
            "radial_distortion_k4",
            "radial_distortion_k5",
            "radial_distortion_k6",
        ],
    ),
    (
        8,
        "SIMPLE_RADIAL_FISHEYE",
        4,
        &[
            "focal_length",
            "principal_point_x",
            "principal_point_y",
            "radial_distortion_k1",
        ],
    ),
    (
        9,
        "RADIAL_FISHEYE",
        5,
        &[
            "focal_length",
            "principal_point_x",
            "principal_point_y",
            "radial_distortion_k1",
            "radial_distortion_k2",
        ],
    ),
    (
        10,
        "THIN_PRISM_FISHEYE",
        12,
        &[
            "focal_length_x",
            "focal_length_y",
            "principal_point_x",
            "principal_point_y",
            "radial_distortion_k1",
            "radial_distortion_k2",
            "tangential_distortion_p1",
            "tangential_distortion_p2",
            "radial_distortion_k3",
            "radial_distortion_k4",
            "thin_prism_sx1",
            "thin_prism_sy1",
        ],
    ),
    (
        11,
        "RAD_TAN_THIN_PRISM_FISHEYE",
        16,
        &[
            "focal_length_x",
            "focal_length_y",
            "principal_point_x",
            "principal_point_y",
            "radial_distortion_k0",
            "radial_distortion_k1",
            "radial_distortion_k2",
            "radial_distortion_k3",
            "radial_distortion_k4",
            "radial_distortion_k5",
            "tangential_distortion_p0",
            "tangential_distortion_p1",
            "thin_prism_s0",
            "thin_prism_s1",
            "thin_prism_s2",
            "thin_prism_s3",
        ],
    ),
];

/// Look up the COLMAP integer model ID from a model name string.
pub fn colmap_model_id(model_name: &str) -> Result<i32, ColmapIoError> {
    CAMERA_MODELS
        .iter()
        .find(|(_, name, _, _)| *name == model_name)
        .map(|(id, _, _, _)| *id)
        .ok_or_else(|| ColmapIoError::UnknownModelName(model_name.to_string()))
}

/// Look up the model name string from a COLMAP integer model ID.
pub(crate) fn colmap_model_name(model_id: i32) -> Result<&'static str, ColmapIoError> {
    CAMERA_MODELS
        .iter()
        .find(|(id, _, _, _)| *id == model_id)
        .map(|(_, name, _, _)| *name)
        .ok_or(ColmapIoError::UnknownModelId(model_id))
}

/// Return the number of parameters for a given camera model name.
pub(crate) fn colmap_num_params(model_name: &str) -> Result<usize, ColmapIoError> {
    CAMERA_MODELS
        .iter()
        .find(|(_, name, _, _)| *name == model_name)
        .map(|(_, _, n, _)| *n)
        .ok_or_else(|| ColmapIoError::UnknownModelName(model_name.to_string()))
}

/// Convert an `SfmrCamera`'s named parameters to a positional array
/// matching the COLMAP binary format ordering.
pub fn camera_params_to_array(camera: &SfmrCamera) -> Result<Vec<f64>, ColmapIoError> {
    let entry = CAMERA_MODELS
        .iter()
        .find(|(_, name, _, _)| *name == camera.model)
        .ok_or_else(|| ColmapIoError::UnknownModelName(camera.model.clone()))?;
    let param_names = entry.3;
    let mut out = Vec::with_capacity(param_names.len());
    for &pname in param_names {
        let val = camera.parameters.get(pname).ok_or_else(|| {
            ColmapIoError::InvalidData(format!(
                "Camera model {} missing parameter '{}'",
                camera.model, pname
            ))
        })?;
        out.push(*val);
    }
    Ok(out)
}

/// Convert a positional parameter array to a named `HashMap` for a given camera model.
pub(crate) fn camera_params_from_array(
    model_name: &str,
    params: &[f64],
) -> Result<HashMap<String, f64>, ColmapIoError> {
    let entry = CAMERA_MODELS
        .iter()
        .find(|(_, name, _, _)| *name == model_name)
        .ok_or_else(|| ColmapIoError::UnknownModelName(model_name.to_string()))?;
    let param_names = entry.3;
    if params.len() != param_names.len() {
        return Err(ColmapIoError::InvalidData(format!(
            "Model {} expects {} params, got {}",
            model_name,
            param_names.len(),
            params.len()
        )));
    }
    let map = param_names
        .iter()
        .zip(params.iter())
        .map(|(&name, &val)| (name.to_string(), val))
        .collect();
    Ok(map)
}
