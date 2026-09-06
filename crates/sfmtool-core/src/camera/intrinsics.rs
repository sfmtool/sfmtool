// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Camera intrinsics and camera model definitions.
//!
//! Provides a typed representation of camera intrinsic parameters for various
//! camera models used in structure-from-motion pipelines (matching COLMAP conventions).
//!
//! [`CameraIntrinsics`] is the computation type. For serialization, convert
//! to/from [`sfmr_format::SfmrCamera`] using the provided `TryFrom` / `From`
//! implementations, which live in the private `registry` module along with the one
//! declaration every camera model is derived from. See
//! `specs/core/camera/camera-model-registry.md`.

use std::fmt;

use nalgebra::Matrix3;

use super::distortion::bspline::bspline_is_inactive;

mod registry;

/// Re-exported so a test corpus outside this module — `camera::report`'s, in
/// particular — can assert its own completeness against the registry without
/// the private `registry` module being widened for it.
#[cfg(test)]
pub(crate) use registry::MODEL_COUNT;

/// Camera model with typed parameters.
///
/// Each variant carries exactly the parameters defined by its COLMAP model.
/// Parameter names match the serialization convention used by [`sfmr_format::SfmrCamera`].
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
    /// Distortion-free equidistant (equiangular) fisheye: `θ = r / f`.
    ///
    /// Carries [`CameraModel::SimplePinhole`]'s exact parameter list — one
    /// focal length in pixels per radian and a principal point — under the
    /// equidistant map, so a point at incidence angle `θ` off the optical axis
    /// lands `f·θ` pixels from the principal point. There are no distortion
    /// coefficients: both directions of the projection are closed form and
    /// exact at every `θ` up to π, with no iteration and no wide-angle blend.
    ///
    /// Not a COLMAP model — an sfmtool extension, like
    /// [`CameraModel::Equirectangular`]. The COLMAP carrier is
    /// `SIMPLE_RADIAL_FISHEYE` with `k = 0`, which parameterizes the identical
    /// map; `sfmr-colmap` converts in both directions.
    EquidistantFisheye {
        focal_length: f64,
        principal_point_x: f64,
        principal_point_y: f64,
    },
    /// Equidistant fisheye with a monotone radial spline:
    /// `r(θ) = f·(θ + δ(θ))` with `δ(θ) = Σ cᵢ·Bᵢ(θ)` over a cubic
    /// open-uniform B-spline basis on `[0, bspline_theta_max]`.
    ///
    /// The coefficients are **dimensionless** (θ-units), so the focal length
    /// stays a pure multiplier of an `f`-independent distorted coordinate —
    /// the same property [`CameraModel::EquidistantFisheye`] and
    /// [`CameraModel::SimpleRadialFisheye`] have. The basis omits the first
    /// two functions of the full clamped basis, pinning `δ(0) = 0` and
    /// `δ'(0) = 0`: the spline cannot express a central-scale correction
    /// (that is `f`'s job), only how the lens departs from equidistant away
    /// from the axis. Beyond `bspline_theta_max` the correction is held
    /// constant, so the radial map continues linearly with slope `f`.
    ///
    /// An empty or all-zero `bspline` short-circuits to the exact
    /// [`CameraModel::EquidistantFisheye`] arithmetic, bit for bit, and so
    /// does a `bspline_theta_max` that is not positive and finite (there is no
    /// domain for the basis then, whatever the coefficients say). A valid
    /// non-empty coefficient vector has at least two entries (a cubic basis
    /// needs them); the monotonicity of `θ + δ(θ)` (`1 + δ'(θ) > 0`) is a
    /// construction invariant enforced where splines are fitted, which is
    /// what gives the Newton inverse a guaranteed bracket. Deserialization
    /// rejects both a one-entry coefficient vector and a degenerate
    /// `bspline_theta_max`; only direct construction can produce them.
    ///
    /// Not a COLMAP model — an sfmtool extension. Unlike
    /// [`CameraModel::EquidistantFisheye`] it has **no COLMAP carrier**: no
    /// COLMAP model parameterizes the spline, so `sfmr-colmap` rejects it
    /// with `UnknownModelName` on every export path.
    ///
    /// **Beta:** the parameterization — the basis, the knot layout, the
    /// parameter names — may change, so a `.sfmr` file carrying this model may
    /// need to be regenerated across releases.
    SfmtoolFisheye {
        focal_length: f64,
        principal_point_x: f64,
        principal_point_y: f64,
        /// Domain end of the spline basis in radians of incidence angle;
        /// `δ` is held constant beyond it.
        bspline_theta_max: f64,
        /// Dimensionless spline coefficients `c₀..c_{N−1}`. Serialized as
        /// `bspline_c0..bspline_c{N−1}` in [`sfmr_format::SfmrCamera`] parameters.
        bspline: Vec<f64>,
    },
    /// Pinhole with a monotone radial spline: `r(ρ) = f·(ρ + δ(ρ))` with
    /// `δ(ρ) = Σ cᵢ·Bᵢ(ρ)` over a cubic open-uniform B-spline basis on
    /// `[0, bspline_rho_max]`, where `ρ = √(rx² + ry²)/rz = tan θ` is the
    /// normalized image-plane radius of a ray in front of the camera.
    ///
    /// The perspective sibling of [`CameraModel::SfmtoolFisheye`], sharing its
    /// spline machinery on the pinhole's radial coordinate instead of the
    /// incidence angle. The coefficients are **dimensionless** (`ρ`-units), so
    /// the focal length stays a pure multiplier of an `f`-independent
    /// distorted coordinate, and the basis omits the first two functions of
    /// the full clamped basis, pinning `δ(0) = 0` and `δ'(0) = 0`: the spline
    /// cannot express a central-scale correction (that is `f`'s job), only how
    /// the lens departs from a pinhole away from the axis. Beyond
    /// `bspline_rho_max` the correction is held constant, so the radial map
    /// continues linearly with slope `f` — which matters here because the
    /// domain is the pinhole's, `θ < 90°`, and `ρ` grows without bound toward
    /// it.
    ///
    /// An empty or all-zero `bspline` short-circuits to the exact
    /// [`CameraModel::SimplePinhole`] arithmetic, bit for bit, and so does a
    /// `bspline_rho_max` that is not positive and finite (there is no domain
    /// for the basis then, whatever the coefficients say). A valid non-empty
    /// coefficient vector has at least two entries (a cubic basis needs them);
    /// the monotonicity of `ρ + δ(ρ)` (`1 + δ'(ρ) > 0`) is a construction
    /// invariant enforced where splines are fitted, which is what gives the
    /// Newton inverse a guaranteed bracket. Deserialization rejects both a
    /// one-entry coefficient vector and a degenerate `bspline_rho_max`; only
    /// direct construction can produce them.
    ///
    /// Not a COLMAP model — an sfmtool extension with **no COLMAP carrier**:
    /// no COLMAP model parameterizes the spline, so `sfmr-colmap` rejects it
    /// with `UnknownModelName` on every export path.
    ///
    /// **Beta:** the parameterization — the basis, the knot layout, the
    /// parameter names — may change, so a `.sfmr` file carrying this model may
    /// need to be regenerated across releases.
    SfmtoolPinhole {
        focal_length: f64,
        principal_point_x: f64,
        principal_point_y: f64,
        /// Domain end of the spline basis in normalized image-plane radius;
        /// `δ` is held constant beyond it.
        bspline_rho_max: f64,
        /// Dimensionless spline coefficients `c₀..c_{N−1}`. Serialized as
        /// `bspline_c0..bspline_c{N−1}` in [`sfmr_format::SfmrCamera`] parameters.
        bspline: Vec<f64>,
    },
}

/// Threshold below which a distortion coefficient is considered zero.
const DISTORTION_EPS: f64 = 1e-12;

/// Whether a radial spline distorts anything, for the `has_distortion` arms of
/// both sfmtool spline models.
///
/// Magnitude alone is not enough: the kernels short-circuit an INACTIVE spline
/// (identity coefficients, or a domain end that is not positive and finite) to
/// the exact base-model arithmetic, so live coefficients on a degenerate
/// `d_max` project with no distortion at all and must report none.
fn spline_has_distortion(bspline: &[f64], d_max: f64) -> bool {
    !bspline_is_inactive(bspline, d_max) && bspline.iter().any(|c| c.abs() > DISTORTION_EPS)
}

/// The radial coordinate a spline model's correction `δ` acts on.
///
/// The basis in `camera::distortion::bspline` is arithmetic on a scalar, so
/// one implementation serves both spline models; this says which scalar a
/// given camera hands it. See [`CameraModel::radial_spline`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplineRadial {
    /// The incidence angle `θ = atan2(ρ, rz)` in radians
    /// ([`CameraModel::SfmtoolFisheye`]).
    IncidenceAngle,
    /// The normalized image-plane radius `√(rx² + ry²)/rz = tan θ`
    /// ([`CameraModel::SfmtoolPinhole`]).
    ImagePlaneRadius,
}

impl CameraModel {
    /// The radial spline this model carries — its coefficients, domain end and
    /// the radial coordinate they act on — or `None` for every model without
    /// one.
    ///
    /// The two sfmtool spline models differ only in that coordinate, so the
    /// callers that linearize or rebuild a spline (the bundle adjustment's
    /// `opt_bspline` release and its admission gate) read it here rather than
    /// matching each variant.
    pub fn radial_spline(&self) -> Option<(&[f64], f64, SplineRadial)> {
        match self {
            CameraModel::SfmtoolFisheye {
                bspline,
                bspline_theta_max,
                ..
            } => Some((bspline, *bspline_theta_max, SplineRadial::IncidenceAngle)),
            CameraModel::SfmtoolPinhole {
                bspline,
                bspline_rho_max,
                ..
            } => Some((bspline, *bspline_rho_max, SplineRadial::ImagePlaneRadius)),
            _ => None,
        }
    }

    /// Return whether this camera model has effective distortion.
    ///
    /// Returns `false` for Pinhole/SimplePinhole (no distortion parameters),
    /// and also `false` for distortion-capable models where all distortion
    /// coefficients are zero (below `DISTORTION_EPS`) — and, for the two
    /// spline models [`CameraModel::SfmtoolFisheye`] and
    /// [`CameraModel::SfmtoolPinhole`], whenever the spline is inactive
    /// however live its coefficients are (an empty or too-short coefficient
    /// vector, or a domain end that is not positive and finite).
    pub fn has_distortion(&self) -> bool {
        match self {
            CameraModel::Pinhole { .. }
            | CameraModel::SimplePinhole { .. }
            | CameraModel::Equirectangular { .. }
            | CameraModel::EquidistantFisheye { .. } => false,
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
            CameraModel::SfmtoolFisheye {
                bspline_theta_max,
                bspline,
                ..
            } => spline_has_distortion(bspline, *bspline_theta_max),
            CameraModel::SfmtoolPinhole {
                bspline_rho_max,
                bspline,
                ..
            } => spline_has_distortion(bspline, *bspline_rho_max),
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
            CameraModel::EquidistantFisheye { .. }
                | CameraModel::SfmtoolFisheye { .. }
                | CameraModel::SimpleRadialFisheye { .. }
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

    /// Whether [`CameraIntrinsics::ray_to_pixel_with_jacobian`] can return an
    /// analytic pixel Jacobian for this model. True for the perspective family
    /// (pinhole, polynomial-distortion and [`CameraModel::SfmtoolPinhole`],
    /// whose radial spline enters the family's shared `x_d = x·g(r²)` form as
    /// `g(ρ) = 1 + δ(ρ)/ρ`) and for the θ-map fisheye
    /// trio [`CameraModel::EquidistantFisheye`],
    /// [`CameraModel::SimpleRadialFisheye`] (`θ_d = θ·(1 + k1·θ²)`, `k1 = 0`
    /// for the first) and [`CameraModel::SfmtoolFisheye`]
    /// (`θ_d = θ + δ(θ)`, with the spline's `δ'` in closed form), all of
    /// which differentiate at every `θ`; false for the multi-coefficient
    /// polynomial fisheye models and equirectangular, whose forward map takes
    /// the ray path with no analytic derivative here.
    pub fn supports_pixel_jacobian(&self) -> bool {
        match self {
            CameraModel::EquidistantFisheye { .. }
            | CameraModel::SimpleRadialFisheye { .. }
            | CameraModel::SfmtoolFisheye { .. } => true,
            _ => !self.needs_ray_path(),
        }
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
    /// A parameter is present but carries a value the model cannot accept,
    /// or is a key the model does not define.
    InvalidParameter { model: String, parameter: String },
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
            CameraIntrinsicsError::InvalidParameter { model, parameter } => {
                write!(
                    f,
                    "invalid parameter '{parameter}' for camera model '{model}'"
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
            | CameraModel::EquidistantFisheye { focal_length, .. }
            | CameraModel::SfmtoolFisheye { focal_length, .. }
            | CameraModel::SfmtoolPinhole { focal_length, .. }
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
            | CameraModel::EquidistantFisheye {
                principal_point_x,
                principal_point_y,
                ..
            }
            | CameraModel::SfmtoolFisheye {
                principal_point_x,
                principal_point_y,
                ..
            }
            | CameraModel::SfmtoolPinhole {
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

    /// This camera at focal `f` — a clone for every model but the five the
    /// focal release admits: `SIMPLE_PINHOLE`, `EQUIDISTANT_FISHEYE`,
    /// `SIMPLE_RADIAL_FISHEYE`, `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`, whose
    /// projections all multiply `f` onto a distorted coordinate that does not
    /// itself read `f` (for the last two, the dimensionless radial spline rides
    /// on the ray's own radial coordinate).
    ///
    /// Focal optimization is gated on exactly those five models, so no other
    /// camera ever sees a moved focal; this matches the bundle adjustment's
    /// focal handling.
    pub(crate) fn with_focal(&self, f: f64) -> CameraIntrinsics {
        let mut out = self.clone();
        match &mut out.model {
            CameraModel::SimplePinhole { focal_length, .. }
            | CameraModel::EquidistantFisheye { focal_length, .. }
            | CameraModel::SimpleRadialFisheye { focal_length, .. }
            | CameraModel::SfmtoolFisheye { focal_length, .. }
            | CameraModel::SfmtoolPinhole { focal_length, .. } => *focal_length = f,
            _ => {}
        }
        out
    }

    /// This camera at focal `f` and radial coefficient `k1` —
    /// [`with_focal`](Self::with_focal) plus the one distortion parameter the
    /// bundle adjustment can release, which exists only on
    /// `SIMPLE_RADIAL_FISHEYE`. `k1` is ignored for every other model (`opt_k1`
    /// is gated on that one).
    pub(crate) fn with_focal_k1(&self, f: f64, k1: f64) -> CameraIntrinsics {
        let mut out = self.with_focal(f);
        if let CameraModel::SimpleRadialFisheye {
            radial_distortion_k1,
            ..
        } = &mut out.model
        {
            *radial_distortion_k1 = k1;
        }
        out
    }

    /// This camera at focal `f` and spline coefficients `bspline` —
    /// [`with_focal`](Self::with_focal) plus the coefficient vector the bundle
    /// adjustment's spline release moves, which exists on the two sfmtool
    /// spline models, `SFMTOOL_FISHEYE` and `SFMTOOL_PINHOLE`. `bspline` is
    /// ignored for every other model (`opt_bspline` is gated on those two). The
    /// sibling of [`with_focal_k1`](Self::with_focal_k1): the two never apply
    /// together, because no model carries both a `k1` and a spline.
    pub(crate) fn with_focal_bspline(&self, f: f64, bspline: &[f64]) -> CameraIntrinsics {
        let mut out = self.with_focal(f);
        let coeffs = match &mut out.model {
            CameraModel::SfmtoolFisheye {
                bspline: coeffs, ..
            }
            | CameraModel::SfmtoolPinhole {
                bspline: coeffs, ..
            } => coeffs,
            _ => return out,
        };
        coeffs.clear();
        coeffs.extend_from_slice(bspline);
        out
    }
}

#[cfg(test)]
mod tests;
