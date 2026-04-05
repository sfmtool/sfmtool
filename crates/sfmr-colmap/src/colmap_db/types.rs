// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Types for COLMAP SQLite database I/O.

use std::collections::HashMap;

use thiserror::Error;

use sfmr_format::SfmrCamera;

/// Errors that can occur when working with COLMAP databases.
#[derive(Error, Debug)]
pub enum ColmapDbError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Unknown camera model name: {0}")]
    UnknownModelName(String),
    #[error("Missing camera parameter '{param}' for model '{model}'")]
    MissingCameraParam { model: String, param: String },
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("Invalid image IDs for pair encoding: {0}")]
    InvalidPairId(String),
}

/// Data needed to write a COLMAP database from a reconstruction.
pub struct ColmapDbWriteData<'a> {
    pub cameras: &'a [SfmrCamera],
    pub image_names: &'a [String],
    pub camera_indexes: &'a [u32],
    pub quaternions_wxyz: &'a [[f64; 4]],
    pub translations_xyz: &'a [[f64; 3]],
    /// Per-image keypoint positions (x, y). Each inner vec corresponds to one image.
    pub keypoints_per_image: &'a [Vec<[f64; 2]>],
    /// Per-image descriptors. Each inner vec is a flat row-major array of u8
    /// with shape (num_keypoints, descriptor_dim). `descriptor_dim` is typically 128.
    pub descriptors_per_image: &'a [Vec<u8>],
    /// Descriptor dimensionality (number of columns). Typically 128 for SIFT.
    pub descriptor_dim: u32,
    /// Optional pose priors for each image. Must be same length as image_names if provided.
    pub pose_priors: Option<&'a [PosePrior]>,
    /// Optional two-view geometries to write.
    pub two_view_geometries: Option<&'a [TwoViewGeometry]>,
    /// Optional rig definitions. When provided, frames must also be provided.
    pub rigs: Option<&'a [DbRig]>,
    /// Optional frame definitions. Each frame references a rig by index into `rigs`.
    pub frames: Option<&'a [DbFrame]>,
}

/// A pose prior for an image in the COLMAP database.
#[derive(Debug, Clone)]
pub struct PosePrior {
    /// Camera center position in world coordinates (x, y, z).
    pub position: [f64; 3],
    /// 3x3 position covariance matrix, stored row-major.
    pub position_covariance: [f64; 9],
    /// Coordinate system: 0 = UNDEFINED, 1 = WGS84, 2 = CARTESIAN.
    pub coordinate_system: i32,
}

/// Two-view geometry configuration values matching COLMAP's enum.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoViewGeometryConfig {
    Undefined = 0,
    Degenerate = 1,
    Calibrated = 2,
    Uncalibrated = 3,
    Planar = 4,
    PlanarOrPanoramic = 5,
    Panoramic = 6,
    Multiple = 7,
    WatermarkClean = 8,
    WatermarkBad = 9,
}

/// A two-view geometry entry for the COLMAP database.
#[derive(Debug, Clone)]
pub struct TwoViewGeometry {
    /// First image index (0-based, into image_names).
    pub image_idx1: u32,
    /// Second image index (0-based, into image_names).
    pub image_idx2: u32,
    /// Inlier match indices: pairs of (feature_idx_in_img1, feature_idx_in_img2).
    /// Stored as a flat array: [idx1_a, idx2_a, idx1_b, idx2_b, ...].
    pub matches: Vec<u32>,
    /// Configuration type.
    pub config: TwoViewGeometryConfig,
    /// Fundamental matrix (3x3), row-major. None if not available.
    pub f_matrix: Option<[f64; 9]>,
    /// Essential matrix (3x3), row-major. None if not available.
    pub e_matrix: Option<[f64; 9]>,
    /// Homography matrix (3x3), row-major. None if not available.
    pub h_matrix: Option<[f64; 9]>,
    /// Relative rotation quaternion in WXYZ order. None if not available.
    pub qvec_wxyz: Option<[f64; 4]>,
    /// Relative translation vector. None if not available.
    pub tvec: Option<[f64; 3]>,
}

// ── Rig and frame types ─────────────────────────────────────────────────

/// Sensor type enum matching COLMAP's `SensorType`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DbSensorType {
    Camera = 0,
    Imu = 1,
}

/// A sensor identifier (type + id).
#[derive(Debug, Clone)]
pub struct DbSensor {
    pub sensor_type: DbSensorType,
    pub id: u32,
}

/// A non-reference sensor in a rig, with optional `sensor_from_rig` transform.
#[derive(Debug, Clone)]
pub struct DbRigSensor {
    pub sensor: DbSensor,
    /// Optional `sensor_from_rig` pose: WXYZ quaternion + XYZ translation.
    pub sensor_from_rig: Option<([f64; 4], [f64; 3])>,
}

/// A rig definition for the COLMAP database.
#[derive(Debug, Clone)]
pub struct DbRig {
    /// Reference sensor (defines the rig origin).
    pub ref_sensor: DbSensor,
    /// Non-reference sensors with optional extrinsic poses.
    pub sensors: Vec<DbRigSensor>,
}

/// A data_id entry in a frame (maps a sensor to an image).
#[derive(Debug, Clone)]
pub struct DbFrameDataId {
    pub sensor_type: DbSensorType,
    pub sensor_id: u32,
    /// The image_id (0-based index into image_names) this sensor maps to.
    pub data_id: u32,
}

/// A frame (temporal instant linking a rig to its images).
#[derive(Debug, Clone)]
pub struct DbFrame {
    /// Index of the rig in the rigs slice (0-based).
    pub rig_index: u32,
    /// Data IDs mapping sensors to images for this frame.
    pub data_ids: Vec<DbFrameDataId>,
}

// ── Matches-format interop types ────────────────────────────────────────

/// Mapping between 0-based image indexes and 1-based COLMAP database image IDs.
///
/// Returned by [`write_colmap_db_features`] to ensure consistent ID mapping
/// across separate feature and match writing steps.
pub struct ImageIdMap {
    /// DB image_id for each 0-based image index (index → db_id).
    pub index_to_db_id: Vec<i64>,
    /// 0-based image index for each DB image_id (db_id → index).
    pub db_id_to_index: HashMap<i64, usize>,
}

impl ImageIdMap {
    /// Build from a vec of DB image IDs indexed by 0-based image index.
    pub fn from_db_ids(db_ids: Vec<i64>) -> Self {
        let db_id_to_index: HashMap<i64, usize> = db_ids
            .iter()
            .enumerate()
            .map(|(idx, &db_id)| (db_id, idx))
            .collect();
        Self {
            index_to_db_id: db_ids,
            db_id_to_index,
        }
    }
}

/// Data needed to populate a COLMAP database with images, cameras, and features.
pub struct ColmapDbFeatureData<'a> {
    pub cameras: &'a [SfmrCamera],
    pub image_names: &'a [String],
    pub camera_indexes: &'a [u32],
    /// Per-image keypoint positions (x, y).
    pub keypoints_per_image: &'a [Vec<[f64; 2]>],
    /// Per-image descriptors (flat row-major u8, shape num_keypoints × descriptor_dim).
    pub descriptors_per_image: &'a [Vec<u8>],
    /// Descriptor dimensionality. Typically 128 for SIFT.
    pub descriptor_dim: u32,
    /// Optional pose priors for each image.
    pub pose_priors: Option<&'a [PosePrior]>,
    /// Optional rig definitions.
    pub rigs: Option<&'a [DbRig]>,
    /// Optional frame definitions.
    pub frames: Option<&'a [DbFrame]>,
}

// ── TwoViewGeometryConfig conversions ───────────────────────────────────

impl TwoViewGeometryConfig {
    /// Convert from COLMAP integer to config enum.
    pub fn from_colmap_int(v: i32) -> Result<Self, ColmapDbError> {
        match v {
            0 => Ok(Self::Undefined),
            1 => Ok(Self::Degenerate),
            2 => Ok(Self::Calibrated),
            3 => Ok(Self::Uncalibrated),
            4 => Ok(Self::Planar),
            5 => Ok(Self::PlanarOrPanoramic),
            6 => Ok(Self::Panoramic),
            7 => Ok(Self::Multiple),
            8 => Ok(Self::WatermarkClean),
            9 => Ok(Self::WatermarkBad),
            _ => Err(ColmapDbError::InvalidData(format!(
                "Unknown TwoViewGeometryConfig integer: {v}"
            ))),
        }
    }
}

/// Convert from matches-format config to COLMAP DB config.
impl From<matches_format::TwoViewGeometryConfig> for TwoViewGeometryConfig {
    fn from(c: matches_format::TwoViewGeometryConfig) -> Self {
        match c {
            matches_format::TwoViewGeometryConfig::Undefined => Self::Undefined,
            matches_format::TwoViewGeometryConfig::Degenerate => Self::Degenerate,
            matches_format::TwoViewGeometryConfig::Calibrated => Self::Calibrated,
            matches_format::TwoViewGeometryConfig::Uncalibrated => Self::Uncalibrated,
            matches_format::TwoViewGeometryConfig::Planar => Self::Planar,
            matches_format::TwoViewGeometryConfig::PlanarOrPanoramic => Self::PlanarOrPanoramic,
            matches_format::TwoViewGeometryConfig::Panoramic => Self::Panoramic,
            matches_format::TwoViewGeometryConfig::Multiple => Self::Multiple,
            matches_format::TwoViewGeometryConfig::WatermarkClean => Self::WatermarkClean,
            matches_format::TwoViewGeometryConfig::WatermarkBad => Self::WatermarkBad,
        }
    }
}

/// Convert from COLMAP DB config to matches-format config.
impl From<TwoViewGeometryConfig> for matches_format::TwoViewGeometryConfig {
    fn from(c: TwoViewGeometryConfig) -> Self {
        match c {
            TwoViewGeometryConfig::Undefined => Self::Undefined,
            TwoViewGeometryConfig::Degenerate => Self::Degenerate,
            TwoViewGeometryConfig::Calibrated => Self::Calibrated,
            TwoViewGeometryConfig::Uncalibrated => Self::Uncalibrated,
            TwoViewGeometryConfig::Planar => Self::Planar,
            TwoViewGeometryConfig::PlanarOrPanoramic => Self::PlanarOrPanoramic,
            TwoViewGeometryConfig::Panoramic => Self::Panoramic,
            TwoViewGeometryConfig::Multiple => Self::Multiple,
            TwoViewGeometryConfig::WatermarkClean => Self::WatermarkClean,
            TwoViewGeometryConfig::WatermarkBad => Self::WatermarkBad,
        }
    }
}
