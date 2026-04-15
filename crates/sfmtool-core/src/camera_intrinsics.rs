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
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // -----------------------------------------------------------------------
    // Helper: build test instances for each model
    // -----------------------------------------------------------------------

    fn pinhole() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::Pinhole {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
            },
            width: 640,
            height: 480,
        }
    }

    fn simple_pinhole() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::SimplePinhole {
                focal_length: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
            },
            width: 640,
            height: 480,
        }
    }

    fn simple_radial() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::SimpleRadial {
                focal_length: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.1,
            },
            width: 640,
            height: 480,
        }
    }

    fn radial() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::Radial {
                focal_length: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.1,
                radial_distortion_k2: -0.05,
            },
            width: 640,
            height: 480,
        }
    }

    fn opencv() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::OpenCV {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.1,
                radial_distortion_k2: -0.05,
                tangential_distortion_p1: 0.001,
                tangential_distortion_p2: -0.002,
            },
            width: 640,
            height: 480,
        }
    }

    fn opencv_fisheye() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::OpenCVFisheye {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.1,
                radial_distortion_k2: -0.05,
                radial_distortion_k3: 0.01,
                radial_distortion_k4: -0.005,
            },
            width: 640,
            height: 480,
        }
    }

    fn full_opencv() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::FullOpenCV {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.1,
                radial_distortion_k2: -0.05,
                tangential_distortion_p1: 0.001,
                tangential_distortion_p2: -0.002,
                radial_distortion_k3: 0.01,
                radial_distortion_k4: -0.005,
                radial_distortion_k5: 0.002,
                radial_distortion_k6: -0.001,
            },
            width: 640,
            height: 480,
        }
    }

    fn simple_radial_fisheye() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::SimpleRadialFisheye {
                focal_length: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.05,
            },
            width: 640,
            height: 480,
        }
    }

    fn radial_fisheye() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::RadialFisheye {
                focal_length: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.05,
                radial_distortion_k2: -0.02,
            },
            width: 640,
            height: 480,
        }
    }

    fn thin_prism_fisheye() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::ThinPrismFisheye {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.05,
                radial_distortion_k2: -0.02,
                tangential_distortion_p1: 0.001,
                tangential_distortion_p2: -0.001,
                radial_distortion_k3: 0.005,
                radial_distortion_k4: -0.003,
                thin_prism_sx1: 0.001,
                thin_prism_sy1: -0.001,
            },
            width: 640,
            height: 480,
        }
    }

    fn rad_tan_thin_prism_fisheye() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::RadTanThinPrismFisheye {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k0: 0.01,
                radial_distortion_k1: 0.02,
                radial_distortion_k2: -0.01,
                radial_distortion_k3: 0.005,
                radial_distortion_k4: -0.003,
                radial_distortion_k5: 0.001,
                tangential_distortion_p0: 0.001,
                tangential_distortion_p1: -0.001,
                thin_prism_s0: 0.001,
                thin_prism_s1: -0.001,
                thin_prism_s2: 0.0005,
                thin_prism_s3: -0.0005,
            },
            width: 640,
            height: 480,
        }
    }

    fn equirectangular() -> CameraIntrinsics {
        // Equirectangular with same principal point (320, 240) as other test cameras
        CameraIntrinsics {
            model: CameraModel::Equirectangular {
                focal_length_x: 640.0 / (2.0 * std::f64::consts::PI),
                focal_length_y: 480.0 / std::f64::consts::PI,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
            },
            width: 640,
            height: 480,
        }
    }

    fn all_cameras() -> Vec<CameraIntrinsics> {
        vec![
            pinhole(),
            simple_pinhole(),
            simple_radial(),
            radial(),
            opencv(),
            opencv_fisheye(),
            simple_radial_fisheye(),
            radial_fisheye(),
            thin_prism_fisheye(),
            rad_tan_thin_prism_fisheye(),
            full_opencv(),
            equirectangular(),
        ]
    }

    // -----------------------------------------------------------------------
    // Intrinsic matrix: K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    // -----------------------------------------------------------------------

    #[test]
    fn intrinsic_matrix_pinhole() {
        let cam = pinhole();
        let k = cam.intrinsic_matrix();
        assert_relative_eq!(k[(0, 0)], 500.0);
        assert_relative_eq!(k[(1, 1)], 502.0);
        assert_relative_eq!(k[(0, 2)], 320.0);
        assert_relative_eq!(k[(1, 2)], 240.0);
        assert_relative_eq!(k[(0, 1)], 0.0);
        assert_relative_eq!(k[(1, 0)], 0.0);
        assert_relative_eq!(k[(2, 0)], 0.0);
        assert_relative_eq!(k[(2, 1)], 0.0);
        assert_relative_eq!(k[(2, 2)], 1.0);
    }

    // Single-focal models use fx = fy = f

    #[test]
    fn intrinsic_matrix_simple_pinhole() {
        let cam = simple_pinhole();
        let k = cam.intrinsic_matrix();
        assert_relative_eq!(k[(0, 0)], 500.0);
        assert_relative_eq!(k[(1, 1)], 500.0);
        assert_relative_eq!(k[(0, 2)], 320.0);
        assert_relative_eq!(k[(1, 2)], 240.0);
        assert_relative_eq!(k[(2, 2)], 1.0);
    }

    // Distortion parameters do not affect K

    #[test]
    fn intrinsic_matrix_ignores_distortion() {
        // All distortion models should produce the same K when they share fx, fy, cx, cy.
        let cam_opencv = opencv();
        let cam_fisheye = opencv_fisheye();
        let cam_full = full_opencv();

        for cam in [&cam_opencv, &cam_fisheye, &cam_full] {
            let k = cam.intrinsic_matrix();
            assert_relative_eq!(k[(0, 0)], 500.0);
            assert_relative_eq!(k[(1, 1)], 502.0);
            assert_relative_eq!(k[(0, 2)], 320.0);
            assert_relative_eq!(k[(1, 2)], 240.0);
            assert_relative_eq!(k[(2, 2)], 1.0);
            // Off-diagonal zeros
            assert_relative_eq!(k[(0, 1)], 0.0);
            assert_relative_eq!(k[(1, 0)], 0.0);
            assert_relative_eq!(k[(2, 0)], 0.0);
            assert_relative_eq!(k[(2, 1)], 0.0);
        }

        // Single-focal models with distortion
        let cam_sr = simple_radial();
        let cam_r = radial();
        for cam in [&cam_sr, &cam_r] {
            let k = cam.intrinsic_matrix();
            assert_relative_eq!(k[(0, 0)], 500.0);
            assert_relative_eq!(k[(1, 1)], 500.0);
            assert_relative_eq!(k[(0, 2)], 320.0);
            assert_relative_eq!(k[(1, 2)], 240.0);
        }
    }

    // -----------------------------------------------------------------------
    // model_name() returns COLMAP-compatible string for each variant
    // -----------------------------------------------------------------------

    #[test]
    fn model_name_all_variants() {
        let expected = [
            "PINHOLE",
            "SIMPLE_PINHOLE",
            "SIMPLE_RADIAL",
            "RADIAL",
            "OPENCV",
            "OPENCV_FISHEYE",
            "SIMPLE_RADIAL_FISHEYE",
            "RADIAL_FISHEYE",
            "THIN_PRISM_FISHEYE",
            "RAD_TAN_THIN_PRISM_FISHEYE",
            "FULL_OPENCV",
            "EQUIRECTANGULAR",
        ];
        for (cam, name) in all_cameras().iter().zip(expected.iter()) {
            assert_eq!(cam.model_name(), *name);
        }
    }

    // -----------------------------------------------------------------------
    // focal_lengths(): dual-focal models return (fx, fy), single-focal return (f, f)
    // -----------------------------------------------------------------------

    #[test]
    fn focal_lengths_dual_focal() {
        let cam = pinhole();
        assert_eq!(cam.focal_lengths(), (500.0, 502.0));
    }

    #[test]
    fn focal_lengths_single_focal() {
        for cam in [simple_pinhole(), simple_radial(), radial()] {
            let (fx, fy) = cam.focal_lengths();
            assert_relative_eq!(fx, 500.0);
            assert_relative_eq!(fy, 500.0);
        }
    }

    // -----------------------------------------------------------------------
    // principal_point() extracts (cx, cy) from all model variants
    // -----------------------------------------------------------------------

    #[test]
    fn principal_point_all_models() {
        for cam in all_cameras() {
            assert_eq!(cam.principal_point(), (320.0, 240.0));
        }
    }

    // -----------------------------------------------------------------------
    // has_distortion(): false for pure pinhole, true for models with k/p params
    // -----------------------------------------------------------------------

    #[test]
    fn has_distortion_false_for_pinhole_models() {
        assert!(!pinhole().has_distortion());
        assert!(!simple_pinhole().has_distortion());
    }

    #[test]
    fn has_distortion_true_for_distortion_models() {
        assert!(simple_radial().has_distortion());
        assert!(radial().has_distortion());
        assert!(opencv().has_distortion());
        assert!(opencv_fisheye().has_distortion());
        assert!(full_opencv().has_distortion());
    }

    #[test]
    fn has_distortion_false_for_zero_coefficient_models() {
        // Distortion-capable models with all-zero coefficients are effectively pinhole
        let sr = CameraIntrinsics {
            model: CameraModel::SimpleRadial {
                focal_length: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.0,
            },
            width: 640,
            height: 480,
        };
        assert!(!sr.has_distortion());

        let r = CameraIntrinsics {
            model: CameraModel::Radial {
                focal_length: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.0,
                radial_distortion_k2: 0.0,
            },
            width: 640,
            height: 480,
        };
        assert!(!r.has_distortion());

        let cv = CameraIntrinsics {
            model: CameraModel::OpenCV {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.0,
                radial_distortion_k2: 0.0,
                tangential_distortion_p1: 0.0,
                tangential_distortion_p2: 0.0,
            },
            width: 640,
            height: 480,
        };
        assert!(!cv.has_distortion());

        let fe = CameraIntrinsics {
            model: CameraModel::OpenCVFisheye {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.0,
                radial_distortion_k2: 0.0,
                radial_distortion_k3: 0.0,
                radial_distortion_k4: 0.0,
            },
            width: 640,
            height: 480,
        };
        assert!(!fe.has_distortion());

        let full = CameraIntrinsics {
            model: CameraModel::FullOpenCV {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.0,
                radial_distortion_k2: 0.0,
                tangential_distortion_p1: 0.0,
                tangential_distortion_p2: 0.0,
                radial_distortion_k3: 0.0,
                radial_distortion_k4: 0.0,
                radial_distortion_k5: 0.0,
                radial_distortion_k6: 0.0,
            },
            width: 640,
            height: 480,
        };
        assert!(!full.has_distortion());
    }

    #[test]
    fn has_distortion_on_camera_model_directly() {
        // Test CameraModel::has_distortion directly
        assert!(!CameraModel::Pinhole {
            focal_length_x: 500.0,
            focal_length_y: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        }
        .has_distortion());

        assert!(CameraModel::SimpleRadial {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.01,
        }
        .has_distortion());

        // Zero k1 → no effective distortion
        assert!(!CameraModel::SimpleRadial {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.0,
        }
        .has_distortion());
    }

    // -----------------------------------------------------------------------
    // SfmrCamera serialization round-trip preserves all parameters
    // -----------------------------------------------------------------------

    #[test]
    fn sfmr_camera_round_trip_all_models() {
        for cam in all_cameras() {
            let sfmr: SfmrCamera = SfmrCamera::from(&cam);
            let restored = CameraIntrinsics::try_from(&sfmr)
                .unwrap_or_else(|e| panic!("round-trip failed for {}: {e}", cam.model_name()));
            assert_eq!(
                cam,
                restored,
                "round-trip mismatch for {}",
                cam.model_name()
            );
        }
    }

    // -----------------------------------------------------------------------
    // TryFrom rejects unknown models and missing parameters
    // -----------------------------------------------------------------------

    #[test]
    fn try_from_unknown_model() {
        let sfmr = SfmrCamera {
            model: "UNKNOWN_MODEL".to_string(),
            width: 640,
            height: 480,
            parameters: HashMap::new(),
        };
        let err = CameraIntrinsics::try_from(&sfmr).unwrap_err();
        assert!(
            matches!(err, CameraIntrinsicsError::UnknownModel(ref name) if name == "UNKNOWN_MODEL")
        );
    }

    #[test]
    fn try_from_missing_parameter() {
        let mut params = HashMap::new();
        params.insert("focal_length_x".to_string(), 500.0);
        // Missing focal_length_y, principal_point_x, principal_point_y
        let sfmr = SfmrCamera {
            model: "PINHOLE".to_string(),
            width: 640,
            height: 480,
            parameters: params,
        };
        let err = CameraIntrinsics::try_from(&sfmr).unwrap_err();
        assert!(matches!(
            err,
            CameraIntrinsicsError::MissingParameter {
                ref model,
                ref parameter,
            } if model == "PINHOLE" && parameter == "focal_length_y"
        ));
    }

    // -----------------------------------------------------------------------
    // 10. RADIAL uses single focal length (matching COLMAP definition)
    // -----------------------------------------------------------------------
    //
    // COLMAP's RADIAL model has a single focal length parameter, not two.
    // Verify that intrinsic_matrix() correctly uses (f, f) for both fx and fy.

    #[test]
    fn radial_uses_single_focal_length() {
        let cam = radial();
        let k = cam.intrinsic_matrix();
        // fx and fy should both equal the single focal_length parameter
        assert_relative_eq!(k[(0, 0)], k[(1, 1)], epsilon = 1e-12);
        assert_relative_eq!(k[(0, 0)], 500.0);
    }

    // -----------------------------------------------------------------------
    // CameraModel::model_name() delegates correctly
    // -----------------------------------------------------------------------

    #[test]
    fn camera_model_model_name() {
        let cam = pinhole();
        assert_eq!(cam.model.model_name(), "PINHOLE");

        let cam = simple_pinhole();
        assert_eq!(cam.model.model_name(), "SIMPLE_PINHOLE");

        let cam = simple_radial();
        assert_eq!(cam.model.model_name(), "SIMPLE_RADIAL");

        let cam = radial();
        assert_eq!(cam.model.model_name(), "RADIAL");

        let cam = opencv();
        assert_eq!(cam.model.model_name(), "OPENCV");

        let cam = opencv_fisheye();
        assert_eq!(cam.model.model_name(), "OPENCV_FISHEYE");

        let cam = full_opencv();
        assert_eq!(cam.model.model_name(), "FULL_OPENCV");
    }

    // -----------------------------------------------------------------------
    // Debug formatting includes type name, variant, and values
    // -----------------------------------------------------------------------

    #[test]
    fn debug_formatting() {
        let cam = simple_pinhole();
        let debug_str = format!("{cam:?}");
        assert!(debug_str.contains("CameraIntrinsics"));
        assert!(debug_str.contains("SimplePinhole"));
        assert!(debug_str.contains("500"));
        assert!(debug_str.contains("640"));
        assert!(debug_str.contains("480"));
    }

    // -----------------------------------------------------------------------
    // Error messages are human-readable
    // -----------------------------------------------------------------------

    #[test]
    fn error_display_unknown_model() {
        let err = CameraIntrinsicsError::UnknownModel("FANCY".to_string());
        let msg = format!("{err}");
        assert_eq!(msg, "unknown camera model: FANCY");
    }

    #[test]
    fn error_display_missing_parameter() {
        let err = CameraIntrinsicsError::MissingParameter {
            model: "PINHOLE".to_string(),
            parameter: "focal_length_x".to_string(),
        };
        let msg = format!("{err}");
        assert_eq!(
            msg,
            "missing parameter 'focal_length_x' for camera model 'PINHOLE'"
        );
    }

    // -----------------------------------------------------------------------
    // Dual-focal distortion models preserve separate fx, fy
    // -----------------------------------------------------------------------

    #[test]
    fn focal_lengths_dual_focal_distortion_models() {
        for cam in [opencv(), opencv_fisheye(), full_opencv()] {
            let (fx, fy) = cam.focal_lengths();
            assert_relative_eq!(fx, 500.0);
            assert_relative_eq!(fy, 502.0);
        }
    }

    // -----------------------------------------------------------------------
    // is_fisheye(): true for all fisheye variants, false for perspective models
    // -----------------------------------------------------------------------

    #[test]
    fn is_fisheye_true_for_fisheye_models() {
        assert!(simple_radial_fisheye().model.is_fisheye());
        assert!(radial_fisheye().model.is_fisheye());
        assert!(opencv_fisheye().model.is_fisheye());
        assert!(thin_prism_fisheye().model.is_fisheye());
        assert!(rad_tan_thin_prism_fisheye().model.is_fisheye());
    }

    #[test]
    fn is_fisheye_false_for_perspective_models() {
        assert!(!pinhole().model.is_fisheye());
        assert!(!simple_pinhole().model.is_fisheye());
        assert!(!simple_radial().model.is_fisheye());
        assert!(!radial().model.is_fisheye());
        assert!(!opencv().model.is_fisheye());
        assert!(!full_opencv().model.is_fisheye());
        assert!(!equirectangular().model.is_fisheye());
    }

    #[test]
    fn is_equirectangular() {
        assert!(equirectangular().model.is_equirectangular());
        assert!(!pinhole().model.is_equirectangular());
        assert!(!opencv_fisheye().model.is_equirectangular());
    }

    #[test]
    fn equirectangular_has_no_distortion() {
        assert!(!equirectangular().has_distortion());
    }

    // -----------------------------------------------------------------------
    // has_distortion for new fisheye models
    // -----------------------------------------------------------------------

    #[test]
    fn has_distortion_false_for_zero_coefficient_fisheye() {
        // OpenCVFisheye with all-zero coefficients should report no distortion
        let fe = CameraIntrinsics {
            model: CameraModel::OpenCVFisheye {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.0,
                radial_distortion_k2: 0.0,
                radial_distortion_k3: 0.0,
                radial_distortion_k4: 0.0,
            },
            width: 640,
            height: 480,
        };
        assert!(!fe.has_distortion());
    }

    #[test]
    fn has_distortion_true_for_distortion_fisheye_models() {
        assert!(simple_radial_fisheye().has_distortion());
        assert!(radial_fisheye().has_distortion());
        assert!(thin_prism_fisheye().has_distortion());
        assert!(rad_tan_thin_prism_fisheye().has_distortion());
    }
}
