// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Camera intrinsics and camera model definitions.
//!
//! Provides a typed representation of camera intrinsic parameters for various
//! camera models used in structure-from-motion pipelines (matching COLMAP conventions).
//!
//! [`CameraIntrinsics`] is the computation type. For serialization, convert to/from
//! [`sfmr_format::SfmrCamera`] using the provided `TryFrom` / `From` implementations.

use std::collections::HashMap;
use std::fmt;

use nalgebra::Matrix3;
use sfmr_format::SfmrCamera;

/// Camera model with typed parameters.
///
/// Each variant carries exactly the parameters defined by its COLMAP model.
/// Parameter names match the serialization convention used by [`SfmrCamera`].
#[derive(Debug, Clone, PartialEq)]
pub enum CameraModel {
    Pinhole {
        focal_length_x: f64,
        focal_length_y: f64,
        principal_point_x: f64,
        principal_point_y: f64,
    },
    SimplePinhole {
        focal_length: f64,
        principal_point_x: f64,
        principal_point_y: f64,
    },
    SimpleRadial {
        focal_length: f64,
        principal_point_x: f64,
        principal_point_y: f64,
        radial_distortion_k1: f64,
    },
    Radial {
        focal_length: f64,
        principal_point_x: f64,
        principal_point_y: f64,
        radial_distortion_k1: f64,
        radial_distortion_k2: f64,
    },
    OpenCV {
        focal_length_x: f64,
        focal_length_y: f64,
        principal_point_x: f64,
        principal_point_y: f64,
        radial_distortion_k1: f64,
        radial_distortion_k2: f64,
        tangential_distortion_p1: f64,
        tangential_distortion_p2: f64,
    },
    OpenCVFisheye {
        focal_length_x: f64,
        focal_length_y: f64,
        principal_point_x: f64,
        principal_point_y: f64,
        radial_distortion_k1: f64,
        radial_distortion_k2: f64,
        radial_distortion_k3: f64,
        radial_distortion_k4: f64,
    },
    SimpleRadialFisheye {
        focal_length: f64,
        principal_point_x: f64,
        principal_point_y: f64,
        radial_distortion_k1: f64,
    },
    RadialFisheye {
        focal_length: f64,
        principal_point_x: f64,
        principal_point_y: f64,
        radial_distortion_k1: f64,
        radial_distortion_k2: f64,
    },
    ThinPrismFisheye {
        focal_length_x: f64,
        focal_length_y: f64,
        principal_point_x: f64,
        principal_point_y: f64,
        radial_distortion_k1: f64,
        radial_distortion_k2: f64,
        tangential_distortion_p1: f64,
        tangential_distortion_p2: f64,
        radial_distortion_k3: f64,
        radial_distortion_k4: f64,
        thin_prism_sx1: f64,
        thin_prism_sy1: f64,
    },
    RadTanThinPrismFisheye {
        focal_length_x: f64,
        focal_length_y: f64,
        principal_point_x: f64,
        principal_point_y: f64,
        radial_distortion_k0: f64,
        radial_distortion_k1: f64,
        radial_distortion_k2: f64,
        radial_distortion_k3: f64,
        radial_distortion_k4: f64,
        radial_distortion_k5: f64,
        tangential_distortion_p0: f64,
        tangential_distortion_p1: f64,
        thin_prism_s0: f64,
        thin_prism_s1: f64,
        thin_prism_s2: f64,
        thin_prism_s3: f64,
    },
    FullOpenCV {
        focal_length_x: f64,
        focal_length_y: f64,
        principal_point_x: f64,
        principal_point_y: f64,
        radial_distortion_k1: f64,
        radial_distortion_k2: f64,
        tangential_distortion_p1: f64,
        tangential_distortion_p2: f64,
        radial_distortion_k3: f64,
        radial_distortion_k4: f64,
        radial_distortion_k5: f64,
        radial_distortion_k6: f64,
    },
    /// Equirectangular projection for panoramic imagery.
    ///
    /// Maps longitude and latitude linearly to pixel coordinates. No distortion
    /// parameters — `distort`/`undistort` are identity operations.
    ///
    /// Focal lengths are in pixels per radian. For a standard full-sphere
    /// panorama (360° × 180°): `focal_length_x = width / (2π)`,
    /// `focal_length_y = height / π`, with principal point at `(width/2, height/2)`.
    Equirectangular {
        focal_length_x: f64,
        focal_length_y: f64,
        principal_point_x: f64,
        principal_point_y: f64,
    },
}

/// Threshold below which a distortion coefficient is considered zero.
const DISTORTION_EPS: f64 = 1e-12;

impl CameraModel {
    /// Return the COLMAP model name string for this camera model.
    pub fn model_name(&self) -> &'static str {
        match self {
            CameraModel::Pinhole { .. } => "PINHOLE",
            CameraModel::SimplePinhole { .. } => "SIMPLE_PINHOLE",
            CameraModel::SimpleRadial { .. } => "SIMPLE_RADIAL",
            CameraModel::Radial { .. } => "RADIAL",
            CameraModel::OpenCV { .. } => "OPENCV",
            CameraModel::OpenCVFisheye { .. } => "OPENCV_FISHEYE",
            CameraModel::SimpleRadialFisheye { .. } => "SIMPLE_RADIAL_FISHEYE",
            CameraModel::RadialFisheye { .. } => "RADIAL_FISHEYE",
            CameraModel::ThinPrismFisheye { .. } => "THIN_PRISM_FISHEYE",
            CameraModel::RadTanThinPrismFisheye { .. } => "RAD_TAN_THIN_PRISM_FISHEYE",
            CameraModel::FullOpenCV { .. } => "FULL_OPENCV",
            CameraModel::Equirectangular { .. } => "EQUIRECTANGULAR",
        }
    }

    /// Return whether this camera model has effective distortion.
    ///
    /// Returns `false` for Pinhole/SimplePinhole (no distortion parameters),
    /// and also `false` for distortion-capable models where all distortion
    /// coefficients are zero (below [`DISTORTION_EPS`]).
    pub fn has_distortion(&self) -> bool {
        match self {
            CameraModel::Pinhole { .. }
            | CameraModel::SimplePinhole { .. }
            | CameraModel::Equirectangular { .. } => false,
            CameraModel::SimpleRadial {
                radial_distortion_k1: k1,
                ..
            } => k1.abs() > DISTORTION_EPS,
            CameraModel::Radial {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                ..
            } => k1.abs() > DISTORTION_EPS || k2.abs() > DISTORTION_EPS,
            CameraModel::OpenCV {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                tangential_distortion_p1: p1,
                tangential_distortion_p2: p2,
                ..
            } => {
                k1.abs() > DISTORTION_EPS
                    || k2.abs() > DISTORTION_EPS
                    || p1.abs() > DISTORTION_EPS
                    || p2.abs() > DISTORTION_EPS
            }
            CameraModel::OpenCVFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                ..
            } => {
                k1.abs() > DISTORTION_EPS
                    || k2.abs() > DISTORTION_EPS
                    || k3.abs() > DISTORTION_EPS
                    || k4.abs() > DISTORTION_EPS
            }
            CameraModel::SimpleRadialFisheye {
                radial_distortion_k1: k,
                ..
            } => k.abs() > DISTORTION_EPS,
            CameraModel::RadialFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                ..
            } => k1.abs() > DISTORTION_EPS || k2.abs() > DISTORTION_EPS,
            CameraModel::ThinPrismFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                tangential_distortion_p1: p1,
                tangential_distortion_p2: p2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                thin_prism_sx1: sx1,
                thin_prism_sy1: sy1,
                ..
            } => {
                k1.abs() > DISTORTION_EPS
                    || k2.abs() > DISTORTION_EPS
                    || p1.abs() > DISTORTION_EPS
                    || p2.abs() > DISTORTION_EPS
                    || k3.abs() > DISTORTION_EPS
                    || k4.abs() > DISTORTION_EPS
                    || sx1.abs() > DISTORTION_EPS
                    || sy1.abs() > DISTORTION_EPS
            }
            CameraModel::RadTanThinPrismFisheye {
                radial_distortion_k0: k0,
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                radial_distortion_k5: k5,
                tangential_distortion_p0: p0,
                tangential_distortion_p1: p1,
                thin_prism_s0: s0,
                thin_prism_s1: s1,
                thin_prism_s2: s2,
                thin_prism_s3: s3,
                ..
            } => {
                k0.abs() > DISTORTION_EPS
                    || k1.abs() > DISTORTION_EPS
                    || k2.abs() > DISTORTION_EPS
                    || k3.abs() > DISTORTION_EPS
                    || k4.abs() > DISTORTION_EPS
                    || k5.abs() > DISTORTION_EPS
                    || p0.abs() > DISTORTION_EPS
                    || p1.abs() > DISTORTION_EPS
                    || s0.abs() > DISTORTION_EPS
                    || s1.abs() > DISTORTION_EPS
                    || s2.abs() > DISTORTION_EPS
                    || s3.abs() > DISTORTION_EPS
            }
            CameraModel::FullOpenCV {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                tangential_distortion_p1: p1,
                tangential_distortion_p2: p2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                radial_distortion_k5: k5,
                radial_distortion_k6: k6,
                ..
            } => {
                k1.abs() > DISTORTION_EPS
                    || k2.abs() > DISTORTION_EPS
                    || p1.abs() > DISTORTION_EPS
                    || p2.abs() > DISTORTION_EPS
                    || k3.abs() > DISTORTION_EPS
                    || k4.abs() > DISTORTION_EPS
                    || k5.abs() > DISTORTION_EPS
                    || k6.abs() > DISTORTION_EPS
            }
        }
    }

    /// Returns true for camera models that use a fisheye (equidistant) projection.
    pub fn is_fisheye(&self) -> bool {
        matches!(
            self,
            CameraModel::SimpleRadialFisheye { .. }
                | CameraModel::RadialFisheye { .. }
                | CameraModel::OpenCVFisheye { .. }
                | CameraModel::ThinPrismFisheye { .. }
                | CameraModel::RadTanThinPrismFisheye { .. }
        )
    }

    /// Returns true for the equirectangular projection model.
    pub fn is_equirectangular(&self) -> bool {
        matches!(self, CameraModel::Equirectangular { .. })
    }

    /// Returns true if this model requires the ray-based warp path
    /// (fisheye or equirectangular), as opposed to the perspective
    /// image-plane path.
    pub fn needs_ray_path(&self) -> bool {
        self.is_fisheye() || self.is_equirectangular()
    }
}

/// Camera intrinsic parameters with image dimensions.
///
/// Combines a [`CameraModel`] (which holds the optical parameters) with the
/// image width and height.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraIntrinsics {
    pub model: CameraModel,
    pub width: u32,
    pub height: u32,
}

/// Error type for camera intrinsics conversion failures.
#[derive(Debug, Clone)]
pub enum CameraIntrinsicsError {
    /// The camera model name is not recognized.
    UnknownModel(String),
    /// A required parameter is missing from the parameter map.
    MissingParameter { model: String, parameter: String },
    /// The camera model is not a perspective projection (fisheye or
    /// equirectangular) and cannot be converted to a pinhole.
    UnsupportedModel(String),
}

impl fmt::Display for CameraIntrinsicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CameraIntrinsicsError::UnknownModel(name) => {
                write!(f, "unknown camera model: {name}")
            }
            CameraIntrinsicsError::MissingParameter { model, parameter } => {
                write!(
                    f,
                    "missing parameter '{parameter}' for camera model '{model}'"
                )
            }
            CameraIntrinsicsError::UnsupportedModel(name) => {
                write!(
                    f,
                    "camera model '{name}' is not a perspective projection and cannot be converted to a pinhole"
                )
            }
        }
    }
}

impl std::error::Error for CameraIntrinsicsError {}

impl CameraIntrinsics {
    /// Construct the 3x3 intrinsic matrix K.
    ///
    /// Distortion parameters are not part of K and are ignored.
    /// For single-focal-length models, `fx = fy = f`.
    ///
    /// ```text
    /// K = | fx  0  cx |
    ///     |  0  fy cy |
    ///     |  0   0  1 |
    /// ```
    pub fn intrinsic_matrix(&self) -> Matrix3<f64> {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
    }

    /// Return the COLMAP model name string for this camera model.
    pub fn model_name(&self) -> &'static str {
        self.model.model_name()
    }

    /// Return the focal lengths as `(fx, fy)`.
    ///
    /// For single-focal-length models (SimplePinhole, SimpleRadial, Radial),
    /// both values are the same: `(f, f)`.
    pub fn focal_lengths(&self) -> (f64, f64) {
        match &self.model {
            CameraModel::Pinhole {
                focal_length_x,
                focal_length_y,
                ..
            } => (*focal_length_x, *focal_length_y),
            CameraModel::SimplePinhole { focal_length, .. }
            | CameraModel::SimpleRadial { focal_length, .. }
            | CameraModel::Radial { focal_length, .. }
            | CameraModel::SimpleRadialFisheye { focal_length, .. }
            | CameraModel::RadialFisheye { focal_length, .. } => (*focal_length, *focal_length),
            CameraModel::OpenCV {
                focal_length_x,
                focal_length_y,
                ..
            }
            | CameraModel::OpenCVFisheye {
                focal_length_x,
                focal_length_y,
                ..
            }
            | CameraModel::ThinPrismFisheye {
                focal_length_x,
                focal_length_y,
                ..
            }
            | CameraModel::RadTanThinPrismFisheye {
                focal_length_x,
                focal_length_y,
                ..
            }
            | CameraModel::FullOpenCV {
                focal_length_x,
                focal_length_y,
                ..
            }
            | CameraModel::Equirectangular {
                focal_length_x,
                focal_length_y,
                ..
            } => (*focal_length_x, *focal_length_y),
        }
    }

    /// Return the principal point as `(cx, cy)`.
    pub fn principal_point(&self) -> (f64, f64) {
        match &self.model {
            CameraModel::Pinhole {
                principal_point_x,
                principal_point_y,
                ..
            }
            | CameraModel::SimplePinhole {
                principal_point_x,
                principal_point_y,
                ..
            }
            | CameraModel::SimpleRadial {
                principal_point_x,
                principal_point_y,
                ..
            }
            | CameraModel::Radial {
                principal_point_x,
                principal_point_y,
                ..
            }
            | CameraModel::OpenCV {
                principal_point_x,
                principal_point_y,
                ..
            }
            | CameraModel::OpenCVFisheye {
                principal_point_x,
                principal_point_y,
                ..
            }
            | CameraModel::SimpleRadialFisheye {
                principal_point_x,
                principal_point_y,
                ..
            }
            | CameraModel::RadialFisheye {
                principal_point_x,
                principal_point_y,
                ..
            }
            | CameraModel::ThinPrismFisheye {
                principal_point_x,
                principal_point_y,
                ..
            }
            | CameraModel::RadTanThinPrismFisheye {
                principal_point_x,
                principal_point_y,
                ..
            }
            | CameraModel::FullOpenCV {
                principal_point_x,
                principal_point_y,
                ..
            }
            | CameraModel::Equirectangular {
                principal_point_x,
                principal_point_y,
                ..
            } => (*principal_point_x, *principal_point_y),
        }
    }

    /// Return whether this camera has effective distortion.
    ///
    /// Returns `false` for Pinhole/SimplePinhole models, and also `false` for
    /// distortion-capable models where all distortion coefficients are zero
    /// (below `1e-12`). Delegates to [`CameraModel::has_distortion`].
    pub fn has_distortion(&self) -> bool {
        self.model.has_distortion()
    }
}

// ---------------------------------------------------------------------------
// Conversion: SfmrCamera -> CameraIntrinsics
// ---------------------------------------------------------------------------

/// Helper to extract a required parameter from the hashmap.
fn get_param(
    params: &HashMap<String, f64>,
    model: &str,
    name: &str,
) -> Result<f64, CameraIntrinsicsError> {
    params
        .get(name)
        .copied()
        .ok_or_else(|| CameraIntrinsicsError::MissingParameter {
            model: model.to_string(),
            parameter: name.to_string(),
        })
}

impl TryFrom<&SfmrCamera> for CameraIntrinsics {
    type Error = CameraIntrinsicsError;

    fn try_from(cam: &SfmrCamera) -> Result<Self, Self::Error> {
        let p = &cam.parameters;
        let m = cam.model.as_str();

        let model = match m {
            "PINHOLE" => CameraModel::Pinhole {
                focal_length_x: get_param(p, m, "focal_length_x")?,
                focal_length_y: get_param(p, m, "focal_length_y")?,
                principal_point_x: get_param(p, m, "principal_point_x")?,
                principal_point_y: get_param(p, m, "principal_point_y")?,
            },
            "SIMPLE_PINHOLE" => CameraModel::SimplePinhole {
                focal_length: get_param(p, m, "focal_length")?,
                principal_point_x: get_param(p, m, "principal_point_x")?,
                principal_point_y: get_param(p, m, "principal_point_y")?,
            },
            "SIMPLE_RADIAL" => CameraModel::SimpleRadial {
                focal_length: get_param(p, m, "focal_length")?,
                principal_point_x: get_param(p, m, "principal_point_x")?,
                principal_point_y: get_param(p, m, "principal_point_y")?,
                radial_distortion_k1: get_param(p, m, "radial_distortion_k1")?,
            },
            "RADIAL" => CameraModel::Radial {
                focal_length: get_param(p, m, "focal_length")?,
                principal_point_x: get_param(p, m, "principal_point_x")?,
                principal_point_y: get_param(p, m, "principal_point_y")?,
                radial_distortion_k1: get_param(p, m, "radial_distortion_k1")?,
                radial_distortion_k2: get_param(p, m, "radial_distortion_k2")?,
            },
            "OPENCV" => CameraModel::OpenCV {
                focal_length_x: get_param(p, m, "focal_length_x")?,
                focal_length_y: get_param(p, m, "focal_length_y")?,
                principal_point_x: get_param(p, m, "principal_point_x")?,
                principal_point_y: get_param(p, m, "principal_point_y")?,
                radial_distortion_k1: get_param(p, m, "radial_distortion_k1")?,
                radial_distortion_k2: get_param(p, m, "radial_distortion_k2")?,
                tangential_distortion_p1: get_param(p, m, "tangential_distortion_p1")?,
                tangential_distortion_p2: get_param(p, m, "tangential_distortion_p2")?,
            },
            "OPENCV_FISHEYE" => CameraModel::OpenCVFisheye {
                focal_length_x: get_param(p, m, "focal_length_x")?,
                focal_length_y: get_param(p, m, "focal_length_y")?,
                principal_point_x: get_param(p, m, "principal_point_x")?,
                principal_point_y: get_param(p, m, "principal_point_y")?,
                radial_distortion_k1: get_param(p, m, "radial_distortion_k1")?,
                radial_distortion_k2: get_param(p, m, "radial_distortion_k2")?,
                radial_distortion_k3: get_param(p, m, "radial_distortion_k3")?,
                radial_distortion_k4: get_param(p, m, "radial_distortion_k4")?,
            },
            "SIMPLE_RADIAL_FISHEYE" => CameraModel::SimpleRadialFisheye {
                focal_length: get_param(p, m, "focal_length")?,
                principal_point_x: get_param(p, m, "principal_point_x")?,
                principal_point_y: get_param(p, m, "principal_point_y")?,
                radial_distortion_k1: get_param(p, m, "radial_distortion_k1")?,
            },
            "RADIAL_FISHEYE" => CameraModel::RadialFisheye {
                focal_length: get_param(p, m, "focal_length")?,
                principal_point_x: get_param(p, m, "principal_point_x")?,
                principal_point_y: get_param(p, m, "principal_point_y")?,
                radial_distortion_k1: get_param(p, m, "radial_distortion_k1")?,
                radial_distortion_k2: get_param(p, m, "radial_distortion_k2")?,
            },
            "THIN_PRISM_FISHEYE" => CameraModel::ThinPrismFisheye {
                focal_length_x: get_param(p, m, "focal_length_x")?,
                focal_length_y: get_param(p, m, "focal_length_y")?,
                principal_point_x: get_param(p, m, "principal_point_x")?,
                principal_point_y: get_param(p, m, "principal_point_y")?,
                radial_distortion_k1: get_param(p, m, "radial_distortion_k1")?,
                radial_distortion_k2: get_param(p, m, "radial_distortion_k2")?,
                tangential_distortion_p1: get_param(p, m, "tangential_distortion_p1")?,
                tangential_distortion_p2: get_param(p, m, "tangential_distortion_p2")?,
                radial_distortion_k3: get_param(p, m, "radial_distortion_k3")?,
                radial_distortion_k4: get_param(p, m, "radial_distortion_k4")?,
                thin_prism_sx1: get_param(p, m, "thin_prism_sx1")?,
                thin_prism_sy1: get_param(p, m, "thin_prism_sy1")?,
            },
            "RAD_TAN_THIN_PRISM_FISHEYE" => CameraModel::RadTanThinPrismFisheye {
                focal_length_x: get_param(p, m, "focal_length_x")?,
                focal_length_y: get_param(p, m, "focal_length_y")?,
                principal_point_x: get_param(p, m, "principal_point_x")?,
                principal_point_y: get_param(p, m, "principal_point_y")?,
                radial_distortion_k0: get_param(p, m, "radial_distortion_k0")?,
                radial_distortion_k1: get_param(p, m, "radial_distortion_k1")?,
                radial_distortion_k2: get_param(p, m, "radial_distortion_k2")?,
                radial_distortion_k3: get_param(p, m, "radial_distortion_k3")?,
                radial_distortion_k4: get_param(p, m, "radial_distortion_k4")?,
                radial_distortion_k5: get_param(p, m, "radial_distortion_k5")?,
                tangential_distortion_p0: get_param(p, m, "tangential_distortion_p0")?,
                tangential_distortion_p1: get_param(p, m, "tangential_distortion_p1")?,
                thin_prism_s0: get_param(p, m, "thin_prism_s0")?,
                thin_prism_s1: get_param(p, m, "thin_prism_s1")?,
                thin_prism_s2: get_param(p, m, "thin_prism_s2")?,
                thin_prism_s3: get_param(p, m, "thin_prism_s3")?,
            },
            "FULL_OPENCV" => CameraModel::FullOpenCV {
                focal_length_x: get_param(p, m, "focal_length_x")?,
                focal_length_y: get_param(p, m, "focal_length_y")?,
                principal_point_x: get_param(p, m, "principal_point_x")?,
                principal_point_y: get_param(p, m, "principal_point_y")?,
                radial_distortion_k1: get_param(p, m, "radial_distortion_k1")?,
                radial_distortion_k2: get_param(p, m, "radial_distortion_k2")?,
                tangential_distortion_p1: get_param(p, m, "tangential_distortion_p1")?,
                tangential_distortion_p2: get_param(p, m, "tangential_distortion_p2")?,
                radial_distortion_k3: get_param(p, m, "radial_distortion_k3")?,
                radial_distortion_k4: get_param(p, m, "radial_distortion_k4")?,
                radial_distortion_k5: get_param(p, m, "radial_distortion_k5")?,
                radial_distortion_k6: get_param(p, m, "radial_distortion_k6")?,
            },
            "EQUIRECTANGULAR" => CameraModel::Equirectangular {
                focal_length_x: get_param(p, m, "focal_length_x")?,
                focal_length_y: get_param(p, m, "focal_length_y")?,
                principal_point_x: get_param(p, m, "principal_point_x")?,
                principal_point_y: get_param(p, m, "principal_point_y")?,
            },
            other => return Err(CameraIntrinsicsError::UnknownModel(other.to_string())),
        };

        Ok(CameraIntrinsics {
            model,
            width: cam.width,
            height: cam.height,
        })
    }
}

// ---------------------------------------------------------------------------
// Conversion: CameraIntrinsics -> SfmrCamera
// ---------------------------------------------------------------------------

impl From<&CameraIntrinsics> for SfmrCamera {
    fn from(cam: &CameraIntrinsics) -> Self {
        let mut parameters = HashMap::new();

        match &cam.model {
            CameraModel::Pinhole {
                focal_length_x,
                focal_length_y,
                principal_point_x,
                principal_point_y,
            } => {
                parameters.insert("focal_length_x".to_string(), *focal_length_x);
                parameters.insert("focal_length_y".to_string(), *focal_length_y);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
            }
            CameraModel::SimplePinhole {
                focal_length,
                principal_point_x,
                principal_point_y,
            } => {
                parameters.insert("focal_length".to_string(), *focal_length);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
            }
            CameraModel::SimpleRadial {
                focal_length,
                principal_point_x,
                principal_point_y,
                radial_distortion_k1,
            } => {
                parameters.insert("focal_length".to_string(), *focal_length);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
                parameters.insert("radial_distortion_k1".to_string(), *radial_distortion_k1);
            }
            CameraModel::Radial {
                focal_length,
                principal_point_x,
                principal_point_y,
                radial_distortion_k1,
                radial_distortion_k2,
            } => {
                parameters.insert("focal_length".to_string(), *focal_length);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
                parameters.insert("radial_distortion_k1".to_string(), *radial_distortion_k1);
                parameters.insert("radial_distortion_k2".to_string(), *radial_distortion_k2);
            }
            CameraModel::OpenCV {
                focal_length_x,
                focal_length_y,
                principal_point_x,
                principal_point_y,
                radial_distortion_k1,
                radial_distortion_k2,
                tangential_distortion_p1,
                tangential_distortion_p2,
            } => {
                parameters.insert("focal_length_x".to_string(), *focal_length_x);
                parameters.insert("focal_length_y".to_string(), *focal_length_y);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
                parameters.insert("radial_distortion_k1".to_string(), *radial_distortion_k1);
                parameters.insert("radial_distortion_k2".to_string(), *radial_distortion_k2);
                parameters.insert(
                    "tangential_distortion_p1".to_string(),
                    *tangential_distortion_p1,
                );
                parameters.insert(
                    "tangential_distortion_p2".to_string(),
                    *tangential_distortion_p2,
                );
            }
            CameraModel::OpenCVFisheye {
                focal_length_x,
                focal_length_y,
                principal_point_x,
                principal_point_y,
                radial_distortion_k1,
                radial_distortion_k2,
                radial_distortion_k3,
                radial_distortion_k4,
            } => {
                parameters.insert("focal_length_x".to_string(), *focal_length_x);
                parameters.insert("focal_length_y".to_string(), *focal_length_y);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
                parameters.insert("radial_distortion_k1".to_string(), *radial_distortion_k1);
                parameters.insert("radial_distortion_k2".to_string(), *radial_distortion_k2);
                parameters.insert("radial_distortion_k3".to_string(), *radial_distortion_k3);
                parameters.insert("radial_distortion_k4".to_string(), *radial_distortion_k4);
            }
            CameraModel::SimpleRadialFisheye {
                focal_length,
                principal_point_x,
                principal_point_y,
                radial_distortion_k1,
            } => {
                parameters.insert("focal_length".to_string(), *focal_length);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
                parameters.insert("radial_distortion_k1".to_string(), *radial_distortion_k1);
            }
            CameraModel::RadialFisheye {
                focal_length,
                principal_point_x,
                principal_point_y,
                radial_distortion_k1,
                radial_distortion_k2,
            } => {
                parameters.insert("focal_length".to_string(), *focal_length);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
                parameters.insert("radial_distortion_k1".to_string(), *radial_distortion_k1);
                parameters.insert("radial_distortion_k2".to_string(), *radial_distortion_k2);
            }
            CameraModel::ThinPrismFisheye {
                focal_length_x,
                focal_length_y,
                principal_point_x,
                principal_point_y,
                radial_distortion_k1,
                radial_distortion_k2,
                tangential_distortion_p1,
                tangential_distortion_p2,
                radial_distortion_k3,
                radial_distortion_k4,
                thin_prism_sx1,
                thin_prism_sy1,
            } => {
                parameters.insert("focal_length_x".to_string(), *focal_length_x);
                parameters.insert("focal_length_y".to_string(), *focal_length_y);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
                parameters.insert("radial_distortion_k1".to_string(), *radial_distortion_k1);
                parameters.insert("radial_distortion_k2".to_string(), *radial_distortion_k2);
                parameters.insert(
                    "tangential_distortion_p1".to_string(),
                    *tangential_distortion_p1,
                );
                parameters.insert(
                    "tangential_distortion_p2".to_string(),
                    *tangential_distortion_p2,
                );
                parameters.insert("radial_distortion_k3".to_string(), *radial_distortion_k3);
                parameters.insert("radial_distortion_k4".to_string(), *radial_distortion_k4);
                parameters.insert("thin_prism_sx1".to_string(), *thin_prism_sx1);
                parameters.insert("thin_prism_sy1".to_string(), *thin_prism_sy1);
            }
            CameraModel::RadTanThinPrismFisheye {
                focal_length_x,
                focal_length_y,
                principal_point_x,
                principal_point_y,
                radial_distortion_k0,
                radial_distortion_k1,
                radial_distortion_k2,
                radial_distortion_k3,
                radial_distortion_k4,
                radial_distortion_k5,
                tangential_distortion_p0,
                tangential_distortion_p1,
                thin_prism_s0,
                thin_prism_s1,
                thin_prism_s2,
                thin_prism_s3,
            } => {
                parameters.insert("focal_length_x".to_string(), *focal_length_x);
                parameters.insert("focal_length_y".to_string(), *focal_length_y);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
                parameters.insert("radial_distortion_k0".to_string(), *radial_distortion_k0);
                parameters.insert("radial_distortion_k1".to_string(), *radial_distortion_k1);
                parameters.insert("radial_distortion_k2".to_string(), *radial_distortion_k2);
                parameters.insert("radial_distortion_k3".to_string(), *radial_distortion_k3);
                parameters.insert("radial_distortion_k4".to_string(), *radial_distortion_k4);
                parameters.insert("radial_distortion_k5".to_string(), *radial_distortion_k5);
                parameters.insert(
                    "tangential_distortion_p0".to_string(),
                    *tangential_distortion_p0,
                );
                parameters.insert(
                    "tangential_distortion_p1".to_string(),
                    *tangential_distortion_p1,
                );
                parameters.insert("thin_prism_s0".to_string(), *thin_prism_s0);
                parameters.insert("thin_prism_s1".to_string(), *thin_prism_s1);
                parameters.insert("thin_prism_s2".to_string(), *thin_prism_s2);
                parameters.insert("thin_prism_s3".to_string(), *thin_prism_s3);
            }
            CameraModel::FullOpenCV {
                focal_length_x,
                focal_length_y,
                principal_point_x,
                principal_point_y,
                radial_distortion_k1,
                radial_distortion_k2,
                tangential_distortion_p1,
                tangential_distortion_p2,
                radial_distortion_k3,
                radial_distortion_k4,
                radial_distortion_k5,
                radial_distortion_k6,
            } => {
                parameters.insert("focal_length_x".to_string(), *focal_length_x);
                parameters.insert("focal_length_y".to_string(), *focal_length_y);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
                parameters.insert("radial_distortion_k1".to_string(), *radial_distortion_k1);
                parameters.insert("radial_distortion_k2".to_string(), *radial_distortion_k2);
                parameters.insert(
                    "tangential_distortion_p1".to_string(),
                    *tangential_distortion_p1,
                );
                parameters.insert(
                    "tangential_distortion_p2".to_string(),
                    *tangential_distortion_p2,
                );
                parameters.insert("radial_distortion_k3".to_string(), *radial_distortion_k3);
                parameters.insert("radial_distortion_k4".to_string(), *radial_distortion_k4);
                parameters.insert("radial_distortion_k5".to_string(), *radial_distortion_k5);
                parameters.insert("radial_distortion_k6".to_string(), *radial_distortion_k6);
            }
            CameraModel::Equirectangular {
                focal_length_x,
                focal_length_y,
                principal_point_x,
                principal_point_y,
            } => {
                parameters.insert("focal_length_x".to_string(), *focal_length_x);
                parameters.insert("focal_length_y".to_string(), *focal_length_y);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
            }
        }

        SfmrCamera {
            model: cam.model_name().to_string(),
            width: cam.width,
            height: cam.height,
            parameters,
        }
    }
}

#[cfg(test)]
mod tests;
