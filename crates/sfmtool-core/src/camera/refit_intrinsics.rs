// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Fit a camera of one model to a camera of another.
//!
//! `specs/core/camera/refit-camera-intrinsics.md` is the design. [`refit_camera_intrinsics`] samples rays
//! over the angles where the source camera is trusted, projects each with the
//! source, and chooses the target model's parameters so the target puts every
//! ray as close as it can to the pixel the source gave it. The principal point
//! is copied, not fitted.
//!
//! Two solvers serve the targets:
//!
//! - the spline models (`SFMTOOL_FISHEYE`, `SFMTOOL_PINHOLE`, and
//!   `EQUIDISTANT_FISHEYE` as the spline with no coefficients) are linear in the
//!   focal and the focal-scaled coefficients, so one least-squares solve gives
//!   both, with a penalty on the coefficients' second difference so that the
//!   coefficients no sample reaches are still determined;
//! - the COLMAP polynomial models are fitted by a small Levenberg–Marquardt
//!   solve that starts from the source's parameters copied by name.
//!
//! The function knows only the lens. The reconstruction-level switch that adds
//! the observations is
//! [`switch_camera_model`](crate::reconstruction::switch_camera_model::switch_camera_model).

use std::collections::BTreeMap;
use std::fmt;
use std::ops::RangeInclusive;

use nalgebra::{DMatrix, DVector};
use sfmtool_sfmr_format::SfmrCamera;

use super::distortion::bspline::{
    basis_at, bspline_is_monotone, BSPLINE_SUPPORT, MIN_BSPLINE_COEFFS,
};
use super::intrinsics::{fixed_arity_model_by_name, CameraIntrinsics, CameraModel, SplineRadial};
use super::report::{forward_fold_deg, off_axis_angle_deg, trustworthy_max_theta_deg};
use constrained_lsq::least_squares_with_inequalities;

mod constrained_lsq;

/// The spline coefficient count a caller gets when it names a spline model
/// and no count, and the source has no spline of that model to keep the
/// count of.
pub const DEFAULT_COEFF_COUNT: usize = 8;

/// The largest spline coefficient count the fit accepts. The basis has one
/// knot span per coefficient past the first, so past this the spans are
/// narrower than anything a lens calibration can support.
pub const MAX_COEFF_COUNT: usize = 32;

/// The coefficient counts a spline with a curve takes, which is what a caller
/// offering the count as a choice bounds its field by. The floor is the fewest
/// coefficients a spline is defined with (fewer evaluate as the identity); the
/// ceiling is [`MAX_COEFF_COUNT`]. A target may also have none, which is the
/// base model alone.
pub const SPLINE_COEFF_COUNT_RANGE: RangeInclusive<usize> = MIN_BSPLINE_COEFFS..=MAX_COEFF_COUNT;

/// Incidence angles the fit samples, evenly spaced over `(0, θ_fit]`.
const THETA_SAMPLES: usize = 96;

/// Azimuths the fit samples at each incidence angle. Enough to see a focal
/// aspect or a tangential term, which a single azimuth hides completely.
const AZIMUTHS: usize = 64;

/// Weight of the spline fit's second-difference penalty, per data row and per
/// coefficient. Small enough that on a source the target represents exactly
/// the penalty moves the fit by well under a thousandth of a pixel; any
/// positive weight is enough to determine a coefficient no sample reaches.
const SMOOTHING: f64 = 1e-6;

/// Iteration budget of the polynomial fit.
const LM_MAX_ITERS: usize = 200;

/// The smallest slope a fitted spline's radial map may have, as a fraction of
/// its focal: the fit requires `r′(d) ≥ MIN_SLOPE · f` over the whole domain
/// and along the linear tail.
///
/// A positive floor rather than zero, so the fitted camera is invertible with
/// a margin rather than only just: the inverse's Newton step is `Δr / r′`, and
/// at this floor a pixel of radius is never more than twenty times the angle
/// it is at the centre. It is below what a real lens reaches inside its field:
/// the orthographic projection `r = f·sin θ`, the most compressive of the
/// classical fisheye projections, falls to it only at 87°, and the equisolid
/// projection only at 174°, so the floor binds on a curve with a fold or a
/// near-flat dip, not on an optical design.
pub const MIN_SLOPE: f64 = 0.05;

/// Constraint grid points per knot span. The same density as the dense test in
/// `bspline_is_monotone`, at the same angles, so a fit that holds the floor at
/// every grid point passes that test by construction.
const SLOPE_GRID_PER_SPAN: usize = 64;

/// The model a camera is refitted to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefitTarget {
    /// `SFMTOOL_FISHEYE` with this many spline coefficients.
    SfmtoolFisheye {
        /// The spline's coefficient count, `0` or `2..=MAX_COEFF_COUNT`.
        /// `None` keeps the source's count when the source is an
        /// `SFMTOOL_FISHEYE` with a spline, and is [`DEFAULT_COEFF_COUNT`]
        /// otherwise; see [`RefitTarget::coeff_count_for`].
        coeff_count: Option<usize>,
    },
    /// `SFMTOOL_PINHOLE` with this many spline coefficients.
    SfmtoolPinhole {
        /// The spline's coefficient count, as for `SfmtoolFisheye`.
        coeff_count: Option<usize>,
    },
    /// `EQUIDISTANT_FISHEYE`: the fisheye spline fit with no coefficients.
    EquidistantFisheye,
    /// A COLMAP polynomial model, by its registered name.
    Colmap(&'static str),
}

impl RefitTarget {
    /// The target a model name asks for, case-insensitively.
    ///
    /// `coeff_count` applies to the two spline models, where `None` is
    /// resolved against each source by [`RefitTarget::coeff_count_for`], and
    /// is refused for every other model rather than ignored. `EQUIRECTANGULAR` is not a lens model and is refused as a
    /// target, as is any name the registry does not know.
    pub fn from_name(camera_model: &str, coeff_count: Option<usize>) -> Result<Self, RefitError> {
        let upper = camera_model.trim().to_ascii_uppercase();
        let target = match upper.as_str() {
            "SFMTOOL_FISHEYE" => RefitTarget::SfmtoolFisheye { coeff_count },
            "SFMTOOL_PINHOLE" => RefitTarget::SfmtoolPinhole { coeff_count },
            "EQUIDISTANT_FISHEYE" => RefitTarget::EquidistantFisheye,
            "EQUIRECTANGULAR" => {
                return Err(RefitError::UnknownTarget {
                    camera_model: upper,
                })
            }
            other => match fixed_arity_model_by_name(other) {
                Some((name, _)) => RefitTarget::Colmap(name),
                None => {
                    return Err(RefitError::UnknownTarget {
                        camera_model: upper,
                    })
                }
            },
        };
        if coeff_count.is_some() && target.spline_radial().is_none() {
            return Err(RefitError::CoeffCountNotApplicable {
                camera_model: target.model_name(),
            });
        }
        target.check()?;
        Ok(target)
    }

    /// The model name of the camera the fit produces.
    pub fn model_name(&self) -> &'static str {
        match self {
            RefitTarget::SfmtoolFisheye { .. } => "SFMTOOL_FISHEYE",
            RefitTarget::SfmtoolPinhole { .. } => "SFMTOOL_PINHOLE",
            RefitTarget::EquidistantFisheye => "EQUIDISTANT_FISHEYE",
            RefitTarget::Colmap(name) => name,
        }
    }

    /// The spline coefficient count the caller stated, for the two spline
    /// models.
    pub fn coeff_count(&self) -> Option<usize> {
        match self {
            RefitTarget::SfmtoolFisheye { coeff_count }
            | RefitTarget::SfmtoolPinhole { coeff_count } => *coeff_count,
            _ => None,
        }
    }

    /// The radial coordinate of the target's spline, for the two spline
    /// models.
    fn spline_radial(&self) -> Option<SplineRadial> {
        match self {
            RefitTarget::SfmtoolFisheye { .. } => Some(SplineRadial::IncidenceAngle),
            RefitTarget::SfmtoolPinhole { .. } => Some(SplineRadial::ImagePlaneRadius),
            _ => None,
        }
    }

    /// The coefficient count the target's spline gets when `source` is fitted
    /// to it: the stated count, or with none stated, the source's own count
    /// when the source carries a spline of this model, so a refit that changes
    /// only the domain keeps the camera's count; [`DEFAULT_COEFF_COUNT`] for any
    /// other source. `None` for a target with no spline.
    pub fn coeff_count_for(&self, source: &CameraIntrinsics) -> Option<usize> {
        let radial = self.spline_radial()?;
        Some(
            self.coeff_count()
                .unwrap_or_else(|| match source.model.radial_spline() {
                    Some((bspline, _, source_radial))
                        if source_radial == radial && bspline.len() >= MIN_BSPLINE_COEFFS =>
                    {
                        bspline.len()
                    }
                    _ => DEFAULT_COEFF_COUNT,
                }),
        )
    }

    /// Whether the target is a perspective model, which has no pixel for a ray
    /// at 90° or more off the axis.
    pub fn is_perspective(&self) -> bool {
        matches!(
            self,
            RefitTarget::SfmtoolPinhole { .. }
                | RefitTarget::Colmap(
                    "PINHOLE"
                        | "SIMPLE_PINHOLE"
                        | "SIMPLE_RADIAL"
                        | "RADIAL"
                        | "OPENCV"
                        | "FULL_OPENCV"
                )
        )
    }

    /// Refuse a target no camera can be built for: a spline of one
    /// coefficient (a cubic basis needs two) or more than [`MAX_COEFF_COUNT`],
    /// or a COLMAP name that is not a lens model.
    fn check(&self) -> Result<(), RefitError> {
        if let Some(n) = self.coeff_count() {
            if n == 1 || n > MAX_COEFF_COUNT {
                return Err(RefitError::CoeffCount {
                    camera_model: self.model_name(),
                    count: n,
                });
            }
        }
        if let RefitTarget::Colmap(name) = self {
            if matches!(*name, "EQUIRECTANGULAR" | "EQUIDISTANT_FISHEYE")
                || fixed_arity_model_by_name(name).is_none()
            {
                return Err(RefitError::UnknownTarget {
                    camera_model: name.to_string(),
                });
            }
        }
        Ok(())
    }
}

/// What the caller states about the fit. Every field has a default.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RefitOptions {
    /// The largest incidence angle the fit samples, in degrees. `None` takes the
    /// source's trusted bound, or, for a source with none, the incidence angle
    /// of its far image corner. A value past the trusted bound is refused.
    pub theta_fit_deg: Option<f64>,
    /// Where a spline target's domain ends, as an incidence angle in degrees.
    /// `None` places it at the far image corner, estimated as the corner's
    /// pixel radius over the source's focal. Ignored for other targets.
    pub spline_domain_deg: Option<f64>,
}

/// Where the fit's largest angle came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThetaFitSource {
    /// The source's trusted bound, [`trustworthy_max_theta_deg`].
    TrustedBound,
    /// The largest incidence angle among the observations of the camera's
    /// images; only the reconstruction-level switch has these.
    Observations,
    /// The incidence angle of the source's far image corner, for a lens-only
    /// fit of a source with no trusted bound.
    ImageCorner,
    /// The caller's own value.
    Given,
    /// The whole domain of a spline source refitted as the same spline model,
    /// by [`refit_spline`].
    SplineDomain,
}

impl ThetaFitSource {
    /// The word a report prints for it.
    pub fn as_str(self) -> &'static str {
        match self {
            ThetaFitSource::TrustedBound => "trusted_bound",
            ThetaFitSource::Observations => "observations",
            ThetaFitSource::ImageCorner => "image_corner",
            ThetaFitSource::Given => "given",
            ThetaFitSource::SplineDomain => "spline_domain",
        }
    }
}

/// A part of the source camera the target model has no parameter for.
#[derive(Debug, Clone, PartialEq)]
pub enum DroppedTerm {
    /// The source has two focal lengths and the target one: the fit keeps a
    /// single focal, and the aspect `fy / fx` is lost.
    FocalAspect {
        /// The source's `fy / fx`.
        fy_over_fx: f64,
    },
    /// A non-zero tangential or thin-prism parameter the target does not
    /// carry.
    Parameter {
        /// The source's parameter name.
        name: String,
        /// Its value.
        value: f64,
    },
}

impl fmt::Display for DroppedTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DroppedTerm::FocalAspect { fy_over_fx } => {
                write!(f, "fx/fy aspect {fy_over_fx:.4} dropped (single focal)")
            }
            DroppedTerm::Parameter { name, value } => write!(f, "{name} = {value:.6} dropped"),
        }
    }
}

/// How far the fitted camera and its source reach.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelExtent {
    /// The largest incidence angle, in degrees, the fitted camera gives the
    /// midpoints of the four image edges.
    pub edge_deg: f64,
    /// The largest incidence angle, in degrees, the fitted camera gives the
    /// four image corners.
    pub corner_deg: f64,
    /// The source's trusted bound in degrees, where it has one.
    pub source_trusted_deg: Option<f64>,
    /// The angle in degrees at which the source's forward map stops
    /// increasing, where it does ([`forward_fold_deg`]).
    pub source_fold_deg: Option<f64>,
}

/// What the monotonicity constraint of a spline fit did.
///
/// A spline fit requires the fitted radial map's slope to stay at or above
/// [`MIN_SLOPE`] times its focal at a dense grid of angles over the domain and
/// along the tail. Where the least-squares curve would fall below that, the
/// fit is the closest curve that does not, and it departs from the source
/// there.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MonotoneConstraint {
    /// Whether the constraint changed the fit: at least one grid angle holds
    /// the slope at the floor.
    pub active: bool,
    /// How many grid angles hold the slope at the floor.
    pub active_angles: usize,
    /// The smallest and the largest of those angles, as incidence angles in
    /// degrees, or `None` when the constraint is not active.
    pub range_deg: Option<[f64; 2]>,
}

/// The fitted camera and how well it matches its source.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraIntrinsicsRefit {
    /// The fitted camera: the target model, the source's image size and
    /// principal point.
    pub camera: CameraIntrinsics,
    /// The largest incidence angle the fit sampled, in degrees.
    pub theta_fit_deg: f64,
    /// Where that angle came from.
    pub theta_fit_source: ThetaFitSource,
    /// Where a spline target's domain ends, as an incidence angle in degrees.
    /// `None` for other targets.
    pub spline_domain_deg: Option<f64>,
    /// RMS pixel distance between the source's and the fitted camera's pixel
    /// over the samples.
    pub rms_px: f64,
    /// The largest such distance.
    pub max_px: f64,
    /// RMS over the sampled angles of the difference between the two cameras'
    /// azimuth-averaged radii: the error of the radial profile alone, without
    /// whatever varies with azimuth.
    pub radial_rms_px: f64,
    /// What the target cannot represent, named.
    pub dropped: Vec<DroppedTerm>,
    /// How far the fitted camera and its source reach.
    pub extent: ModelExtent,
    /// What the monotonicity constraint did to a spline fit; inactive for
    /// every other target.
    pub monotone_constraint: MonotoneConstraint,
}

/// Why a fit produced no camera. Each variant names the rule and the value it
/// measured.
#[derive(Debug, Clone, PartialEq)]
pub enum RefitError {
    /// The target model is not one a camera can be refitted to.
    UnknownTarget {
        /// The name as given, upper-cased.
        camera_model: String,
    },
    /// A spline target with a coefficient count the model does not allow.
    CoeffCount {
        /// The target model.
        camera_model: &'static str,
        /// The count asked for.
        count: usize,
    },
    /// A coefficient count given for a model without a spline.
    CoeffCountNotApplicable {
        /// The target model.
        camera_model: &'static str,
    },
    /// The fit's largest angle is not a positive angle of at most 180°.
    ThetaFitInvalid {
        /// The angle asked for, in degrees.
        theta_fit_deg: f64,
    },
    /// The fit's largest angle is past the source's trusted bound.
    BeyondTrustedBound {
        /// The angle asked for, in degrees.
        theta_fit_deg: f64,
        /// The source's trusted bound, in degrees.
        trusted_deg: f64,
    },
    /// A perspective target asked to fit rays at 90° or more.
    PerspectivePast90 {
        /// The fit's largest angle, in degrees.
        theta_fit_deg: f64,
    },
    /// A perspective target for a camera observed at 90° or more.
    ObservationsPast90 {
        /// The largest observed incidence angle, in degrees.
        max_theta_deg: f64,
    },
    /// The source model has no pixel for a ray inside the fit's domain.
    SourceCannotProject {
        /// The first incidence angle it could not project, in degrees.
        theta_deg: f64,
    },
    /// The spline domain asked for is not an angle the target can have.
    SplineDomainInvalid {
        /// The angle asked for, in degrees.
        spline_domain_deg: f64,
    },
    /// The fitted spline is not strictly increasing despite the fit's
    /// monotonicity constraint, so it has no inverse: a failure of the
    /// constrained solve, not of the source.
    NotMonotone,
    /// A fitted polynomial fisheye whose own trusted bound falls short of the
    /// fit's largest angle.
    TrustedBoundShort {
        /// The fitted camera's trusted bound, in degrees.
        trusted_deg: f64,
        /// The fit's largest angle, in degrees.
        theta_fit_deg: f64,
    },
    /// [`refit_spline`] was handed a camera with no spline.
    NotSplineSource {
        /// The source's model.
        camera_model: &'static str,
    },
    /// The samples do not determine the target's parameters.
    Degenerate {
        /// What did not hold.
        reason: &'static str,
    },
}

impl fmt::Display for RefitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RefitError::UnknownTarget { camera_model } => write!(
                f,
                "'{camera_model}' is not a model a camera can be refitted to; the targets are \
                 SFMTOOL_FISHEYE, SFMTOOL_PINHOLE, EQUIDISTANT_FISHEYE and the COLMAP lens models"
            ),
            RefitError::CoeffCount {
                camera_model,
                count,
            } => write!(
                f,
                "{camera_model} takes 0 or 2 to {MAX_COEFF_COUNT} spline coefficients, not {count}"
            ),
            RefitError::CoeffCountNotApplicable { camera_model } => write!(
                f,
                "a coefficient count applies only to SFMTOOL_FISHEYE and SFMTOOL_PINHOLE, \
                 not to {camera_model}"
            ),
            RefitError::ThetaFitInvalid { theta_fit_deg } => write!(
                f,
                "the fit's largest angle must be above 0° and at most 180°, not {theta_fit_deg}°"
            ),
            RefitError::BeyondTrustedBound {
                theta_fit_deg,
                trusted_deg,
            } => write!(
                f,
                "the fit's largest angle {theta_fit_deg:.2}° is past the source's trusted bound \
                 {trusted_deg:.2}°; a fit past it would copy the source's failure into the new model"
            ),
            RefitError::PerspectivePast90 { theta_fit_deg } => write!(
                f,
                "a perspective model has no pixel for a ray at 90° or more, and the fit reaches \
                 {theta_fit_deg:.2}°"
            ),
            RefitError::ObservationsPast90 { max_theta_deg } => write!(
                f,
                "a perspective model has no pixel for a ray at 90° or more, and the camera is \
                 observed at {max_theta_deg:.2}°"
            ),
            RefitError::SourceCannotProject { theta_deg } => write!(
                f,
                "the source model has no pixel for a ray at {theta_deg:.2}°, inside the fit"
            ),
            RefitError::SplineDomainInvalid { spline_domain_deg } => write!(
                f,
                "the spline domain must end above 0° and at most 180° (below 90° for a \
                 perspective model), not {spline_domain_deg}°"
            ),
            RefitError::NotMonotone => write!(
                f,
                "the fitted spline is not strictly increasing even though the fit constrains \
                 it to be, so it has no inverse"
            ),
            RefitError::TrustedBoundShort {
                trusted_deg,
                theta_fit_deg,
            } => write!(
                f,
                "the fitted polynomial is trusted only to {trusted_deg:.2}°, short of the fit's \
                 largest angle {theta_fit_deg:.2}°"
            ),
            RefitError::NotSplineSource { camera_model } => write!(
                f,
                "a {camera_model} camera has no spline; only SFMTOOL_FISHEYE and SFMTOOL_PINHOLE carry one"
            ),
            RefitError::Degenerate { reason } => write!(f, "the fit is degenerate: {reason}"),
        }
    }
}

impl std::error::Error for RefitError {}

/// Fit `target` to `source` and report how well it matches.
///
/// The fit samples rays over `θ ∈ (0, θ_fit]` at 64 azimuths, projects each
/// with the source, and chooses the target's parameters to minimize the pixel
/// distance between the source's pixel and the target's. The principal point
/// and the image size are copied. See [`RefitOptions`] for the defaults of
/// `θ_fit` and of a spline target's domain.
///
/// # Example
///
/// ```
/// use sfmtool_core::camera::refit_intrinsics::{refit_camera_intrinsics, RefitOptions, RefitTarget};
/// use sfmtool_core::{CameraIntrinsics, CameraModel};
///
/// let source = CameraIntrinsics {
///     model: CameraModel::EquidistantFisheye {
///         focal_length: 130.0,
///         principal_point_x: 240.0,
///         principal_point_y: 240.0,
///     },
///     width: 480,
///     height: 480,
/// };
/// let target = RefitTarget::from_name("SFMTOOL_FISHEYE", Some(8)).unwrap();
/// let refit = refit_camera_intrinsics(&source, &target, &RefitOptions::default()).unwrap();
/// assert!(refit.max_px < 1e-6);
/// ```
pub fn refit_camera_intrinsics(
    source: &CameraIntrinsics,
    target: &RefitTarget,
    options: &RefitOptions,
) -> Result<CameraIntrinsicsRefit, RefitError> {
    let (theta_fit_deg, theta_fit_source) = match options.theta_fit_deg {
        Some(theta) => (theta, ThetaFitSource::Given),
        None => match trustworthy_max_theta_deg(source) {
            Some(bound) => (bound, ThetaFitSource::TrustedBound),
            None => (image_corner_deg(source)?, ThetaFitSource::ImageCorner),
        },
    };
    refit_camera_intrinsics_over(
        source,
        target,
        theta_fit_deg,
        theta_fit_source,
        options.spline_domain_deg,
    )
}

/// [`refit_camera_intrinsics`] with the fit's largest angle already resolved, which is
/// what the reconstruction-level switch calls once it has read the
/// observations' extent.
pub(crate) fn refit_camera_intrinsics_over(
    source: &CameraIntrinsics,
    target: &RefitTarget,
    theta_fit_deg: f64,
    theta_fit_source: ThetaFitSource,
    spline_domain_deg: Option<f64>,
) -> Result<CameraIntrinsicsRefit, RefitError> {
    target.check()?;
    if !(theta_fit_deg > 0.0 && theta_fit_deg <= 180.0) {
        return Err(RefitError::ThetaFitInvalid { theta_fit_deg });
    }
    let trusted = trustworthy_max_theta_deg(source);
    if let Some(trusted_deg) = trusted {
        // A hair of slack, so a caller handing back the bound it was told
        // gets the default fit rather than a refusal over a rounding.
        if theta_fit_deg > trusted_deg + 1e-9 {
            return Err(RefitError::BeyondTrustedBound {
                theta_fit_deg,
                trusted_deg,
            });
        }
    }
    if target.is_perspective() && theta_fit_deg >= 90.0 {
        return Err(RefitError::PerspectivePast90 { theta_fit_deg });
    }

    let samples = Samples::new(source, theta_fit_deg.to_radians())?;
    let (cx, cy) = source.principal_point();

    let (camera, domain, constraint) = match target {
        RefitTarget::SfmtoolFisheye { .. } | RefitTarget::SfmtoolPinhole { .. } => {
            let radial = target.spline_radial().expect("a spline target");
            let coeff_count = target
                .coeff_count_for(source)
                .expect("a spline target has a count");
            fit_spline_camera(source, &samples, radial, coeff_count, spline_domain_deg)?
        }
        RefitTarget::EquidistantFisheye => {
            let f = fit_spline(&samples, cx, cy, SplineRadial::IncidenceAngle, 0, 1.0)?.0;
            let camera = CameraIntrinsics {
                model: CameraModel::EquidistantFisheye {
                    focal_length: f,
                    principal_point_x: cx,
                    principal_point_y: cy,
                },
                width: source.width,
                height: source.height,
            };
            (camera, None, MonotoneConstraint::default())
        }
        RefitTarget::Colmap(name) => {
            let camera = fit_colmap(source, name, &samples)?;
            if let Some(trusted_deg) = trustworthy_max_theta_deg(&camera) {
                if trusted_deg < theta_fit_deg - 1e-6 {
                    return Err(RefitError::TrustedBoundShort {
                        trusted_deg,
                        theta_fit_deg,
                    });
                }
            }
            (camera, None, MonotoneConstraint::default())
        }
    };

    measured(
        source,
        camera,
        &samples,
        theta_fit_deg,
        theta_fit_source,
        domain,
        constraint,
    )
}

/// Refit a spline camera as the same spline model with `coeff_count`
/// coefficients, and with its domain ending at `spline_domain_deg` where that
/// is given, or where the source's ends otherwise.
///
/// The fit samples the source over the **whole** new domain, `θ ∈ (0, θ_max]`
/// with `θ_max` the domain end as an incidence angle, rather than over a
/// trusted bound or the observations: a spline model is defined everywhere on
/// its domain and along its linear tail past it, so the source states the
/// curve at every angle the new spline covers, whether the new domain is
/// shorter or longer than the old. The result is the best least-squares
/// description of the source's curve on the new coefficient scheme, focal
/// included. A domain end that is not given is copied exactly
/// (`SFMTOOL_PINHOLE`'s `tan θ` is not taken through degrees and back). Like
/// every spline fit it is constrained to be monotone, which makes it the best
/// description with an inverse; the report's
/// [`monotone_constraint`](CameraIntrinsicsRefit::monotone_constraint) says
/// where that departed from the source.
///
/// Refused for a source without a spline (`NotSplineSource`), a count the
/// model does not allow, and a domain end the model cannot have.
///
/// # Example
///
/// ```
/// use sfmtool_core::camera::refit_intrinsics::refit_spline;
/// use sfmtool_core::{CameraIntrinsics, CameraModel};
///
/// let source = CameraIntrinsics {
///     model: CameraModel::SfmtoolFisheye {
///         focal_length: 130.0,
///         principal_point_x: 240.0,
///         principal_point_y: 240.0,
///         bspline_theta_max: 2.0,
///         bspline: vec![0.0, -0.01, -0.03, -0.05, -0.07, -0.09, -0.11, -0.13],
///     },
///     width: 480,
///     height: 480,
/// };
/// let refit = refit_spline(&source, 12, None).unwrap();
/// assert!(refit.max_px < 0.05, "{}", refit.max_px);
/// ```
pub fn refit_spline(
    source: &CameraIntrinsics,
    coeff_count: usize,
    spline_domain_deg: Option<f64>,
) -> Result<CameraIntrinsicsRefit, RefitError> {
    let Some((_, source_d_max, radial)) = source.model.radial_spline() else {
        return Err(RefitError::NotSplineSource {
            camera_model: source.model_name(),
        });
    };
    let target = match radial {
        SplineRadial::IncidenceAngle => RefitTarget::SfmtoolFisheye {
            coeff_count: Some(coeff_count),
        },
        SplineRadial::ImagePlaneRadius => RefitTarget::SfmtoolPinhole {
            coeff_count: Some(coeff_count),
        },
    };
    target.check()?;
    let d_max = match spline_domain_deg {
        Some(deg) => domain_end_of(radial, deg)?,
        None => source_d_max,
    };
    if !(d_max > 0.0 && d_max.is_finite()) {
        return Err(RefitError::Degenerate {
            reason: "the spline domain has no extent",
        });
    }
    let theta_fit = match radial {
        SplineRadial::IncidenceAngle => d_max,
        SplineRadial::ImagePlaneRadius => d_max.atan(),
    };
    if theta_fit > std::f64::consts::PI {
        return Err(RefitError::SplineDomainInvalid {
            spline_domain_deg: theta_fit.to_degrees(),
        });
    }
    let samples = Samples::new(source, theta_fit)?;
    let (camera, domain_deg, constraint) =
        fit_spline_camera_on(source, &samples, radial, coeff_count, d_max)?;
    measured(
        source,
        camera,
        &samples,
        theta_fit.to_degrees(),
        ThetaFitSource::SplineDomain,
        Some(domain_deg),
        constraint,
    )
}

/// Where a spline camera's domain ends, as an incidence angle in degrees:
/// `bspline_theta_max` itself for `SFMTOOL_FISHEYE`, and the angle whose
/// tangent is `bspline_rho_max` for `SFMTOOL_PINHOLE`. `None` for a camera with
/// no spline. A caller showing the domain as an editable value reads it here,
/// in the unit [`refit_spline`] and [`RefitOptions::spline_domain_deg`] take.
pub fn spline_domain_deg(camera: &CameraIntrinsics) -> Option<f64> {
    camera
        .model
        .radial_spline()
        .map(|(_, d_max, radial)| incidence_angle(radial, d_max).to_degrees())
}

/// The report of a fitted `camera` against its `source` over `samples`.
fn measured(
    source: &CameraIntrinsics,
    camera: CameraIntrinsics,
    samples: &Samples,
    theta_fit_deg: f64,
    theta_fit_source: ThetaFitSource,
    spline_domain_deg: Option<f64>,
    monotone_constraint: MonotoneConstraint,
) -> Result<CameraIntrinsicsRefit, RefitError> {
    let (cx, cy) = source.principal_point();
    let fitted = samples.pixels(&camera).ok_or(RefitError::Degenerate {
        reason: "the fitted camera has no pixel for a sampled ray",
    })?;
    let (rms_px, max_px, radial_rms_px) = samples.errors(&fitted, cx, cy);

    Ok(CameraIntrinsicsRefit {
        dropped: dropped_terms(source, &camera),
        extent: ModelExtent {
            edge_deg: extreme_angle_deg(&camera, &EDGE_MIDPOINTS),
            corner_deg: extreme_angle_deg(&camera, &CORNERS),
            source_trusted_deg: trustworthy_max_theta_deg(source),
            source_fold_deg: forward_fold_deg(source),
        },
        camera,
        theta_fit_deg,
        theta_fit_source,
        spline_domain_deg,
        rms_px,
        max_px,
        radial_rms_px,
        monotone_constraint,
    })
}

/// The incidence angle of the source's far image corner, in degrees, which is
/// the lens-only default of the fit's largest angle for a source with no
/// trusted bound.
pub(crate) fn image_corner_deg(source: &CameraIntrinsics) -> Result<f64, RefitError> {
    if source.width == 0 || source.height == 0 {
        return Err(RefitError::Degenerate {
            reason: "the camera has no image",
        });
    }
    Ok(extreme_angle_deg(source, &CORNERS).min(180.0))
}

/// The four image corners, as fractions of the width and height.
const CORNERS: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

/// The midpoints of the four image edges, as fractions of the width and
/// height.
const EDGE_MIDPOINTS: [[f64; 2]; 4] = [[0.5, 0.0], [0.5, 1.0], [0.0, 0.5], [1.0, 0.5]];

/// The largest incidence angle, in degrees, `camera` gives any of `points`.
fn extreme_angle_deg(camera: &CameraIntrinsics, points: &[[f64; 2]]) -> f64 {
    let (w, h) = (f64::from(camera.width), f64::from(camera.height));
    points
        .iter()
        .map(|[a, b]| off_axis_angle_deg(camera, a * w, b * h))
        .fold(0.0, f64::max)
}

/// The rays the fit samples and the source's pixel for each.
///
/// Ray `k · AZIMUTHS + j` is at incidence angle `theta[k]` and azimuth
/// `2π·j/AZIMUTHS`, in the canonical camera frame (the camera looks along `−Z`).
struct Samples {
    theta: Vec<f64>,
    rays: Vec<[f64; 3]>,
    source: Vec<[f64; 2]>,
}

impl Samples {
    fn new(source: &CameraIntrinsics, theta_fit: f64) -> Result<Self, RefitError> {
        let theta: Vec<f64> = (1..=THETA_SAMPLES)
            .map(|k| theta_fit * k as f64 / THETA_SAMPLES as f64)
            .collect();
        let mut rays = Vec::with_capacity(THETA_SAMPLES * AZIMUTHS);
        for &t in &theta {
            let (sin_t, cos_t) = t.sin_cos();
            for j in 0..AZIMUTHS {
                let phi = std::f64::consts::TAU * j as f64 / AZIMUTHS as f64;
                let (sin_p, cos_p) = phi.sin_cos();
                rays.push([sin_t * cos_p, sin_t * sin_p, -cos_t]);
            }
        }
        let mut pixels = Vec::with_capacity(rays.len());
        for (i, ray) in rays.iter().enumerate() {
            match source.ray_to_pixel(*ray) {
                Some((u, v)) if u.is_finite() && v.is_finite() => pixels.push([u, v]),
                _ => {
                    return Err(RefitError::SourceCannotProject {
                        theta_deg: theta[i / AZIMUTHS].to_degrees(),
                    })
                }
            }
        }
        Ok(Samples {
            theta,
            rays,
            source: pixels,
        })
    }

    /// Every sample's pixel under `camera`, or `None` when one has none.
    fn pixels(&self, camera: &CameraIntrinsics) -> Option<Vec<[f64; 2]>> {
        self.rays
            .iter()
            .map(|ray| {
                camera
                    .ray_to_pixel(*ray)
                    .filter(|(u, v)| u.is_finite() && v.is_finite())
                    .map(|(u, v)| [u, v])
            })
            .collect()
    }

    /// `(rms, max, radial rms)` of `fitted` against the source's pixels, the
    /// radii measured from `(cx, cy)`.
    fn errors(&self, fitted: &[[f64; 2]], cx: f64, cy: f64) -> (f64, f64, f64) {
        let mut sum = 0.0;
        let mut max: f64 = 0.0;
        for (a, b) in self.source.iter().zip(fitted) {
            let d = (a[0] - b[0]).hypot(a[1] - b[1]);
            sum += d * d;
            max = max.max(d);
        }
        let rms = (sum / self.source.len() as f64).sqrt();

        let radius = |p: &[f64; 2]| (p[0] - cx).hypot(p[1] - cy);
        let mut radial = 0.0;
        for k in 0..self.theta.len() {
            let ring = k * AZIMUTHS..(k + 1) * AZIMUTHS;
            let source_mean: f64 =
                self.source[ring.clone()].iter().map(radius).sum::<f64>() / AZIMUTHS as f64;
            let fitted_mean: f64 = fitted[ring].iter().map(radius).sum::<f64>() / AZIMUTHS as f64;
            radial += (source_mean - fitted_mean).powi(2);
        }
        let radial_rms = (radial / self.theta.len() as f64).sqrt();
        (rms, max, radial_rms)
    }
}

/// The base model of a spline family at focal 1 with its principal point at
/// the origin: its pixel for a ray is `d·û`, with `d` the family's radial
/// coordinate and `û` the unit image direction.
fn unit_base(radial: SplineRadial) -> CameraIntrinsics {
    let model = match radial {
        SplineRadial::IncidenceAngle => CameraModel::EquidistantFisheye {
            focal_length: 1.0,
            principal_point_x: 0.0,
            principal_point_y: 0.0,
        },
        SplineRadial::ImagePlaneRadius => CameraModel::SimplePinhole {
            focal_length: 1.0,
            principal_point_x: 0.0,
            principal_point_y: 0.0,
        },
    };
    CameraIntrinsics {
        model,
        width: 1,
        height: 1,
    }
}

/// The spline family's radial coordinate for an incidence angle in radians.
fn radial_coordinate(radial: SplineRadial, theta: f64) -> f64 {
    match radial {
        SplineRadial::IncidenceAngle => theta,
        SplineRadial::ImagePlaneRadius => theta.tan(),
    }
}

/// A spline family's domain end, in its own radial coordinate, for an
/// incidence angle in degrees; refused where the family cannot end there.
fn domain_end_of(radial: SplineRadial, deg: f64) -> Result<f64, RefitError> {
    let valid = deg > 0.0
        && match radial {
            SplineRadial::IncidenceAngle => deg <= 180.0,
            SplineRadial::ImagePlaneRadius => deg < 90.0,
        };
    if !valid {
        return Err(RefitError::SplineDomainInvalid {
            spline_domain_deg: deg,
        });
    }
    Ok(radial_coordinate(radial, deg.to_radians()))
}

/// Fit a spline camera: place the domain, solve, build, and check the result
/// is monotone. Returns the camera, its domain end in degrees and what the
/// monotonicity constraint did.
fn fit_spline_camera(
    source: &CameraIntrinsics,
    samples: &Samples,
    radial: SplineRadial,
    coeff_count: usize,
    spline_domain_deg: Option<f64>,
) -> Result<(CameraIntrinsics, Option<f64>, MonotoneConstraint), RefitError> {
    let (cx, cy) = source.principal_point();
    let d_max = match spline_domain_deg {
        Some(deg) => domain_end_of(radial, deg)?,
        None => {
            // The far corner's pixel radius over the source's focal on the
            // axis, where every model family agrees with its base: the format
            // spec's placement, fixed before the spline is fitted on it.
            let (fx, fy) = source.focal_lengths();
            let f0 = (fx * fy).sqrt();
            let (w, h) = (f64::from(source.width), f64::from(source.height));
            let corner = CORNERS
                .iter()
                .map(|[a, b]| (a * w - cx).hypot(b * h - cy))
                .fold(0.0, f64::max);
            let d = corner / f0;
            match radial {
                SplineRadial::IncidenceAngle => d.min(std::f64::consts::PI),
                SplineRadial::ImagePlaneRadius => d,
            }
        }
    };
    if !(d_max > 0.0 && d_max.is_finite()) {
        return Err(RefitError::Degenerate {
            reason: "the spline domain has no extent",
        });
    }
    let (camera, domain_deg, constraint) =
        fit_spline_camera_on(source, samples, radial, coeff_count, d_max)?;
    Ok((camera, Some(domain_deg), constraint))
}

/// Fit a spline camera on the domain ending at `d_max`, in the family's radial
/// coordinate: solve under the monotonicity constraint, build, and check the
/// result is monotone. Returns the camera, its domain end in degrees and what
/// the constraint did. A failed check refuses: the constraint holds the slope
/// above [`MIN_SLOPE`] at the check's own sample angles, so it fails only where
/// the constrained solve itself went wrong.
fn fit_spline_camera_on(
    source: &CameraIntrinsics,
    samples: &Samples,
    radial: SplineRadial,
    coeff_count: usize,
    d_max: f64,
) -> Result<(CameraIntrinsics, f64, MonotoneConstraint), RefitError> {
    let (cx, cy) = source.principal_point();
    let (f, coeffs, constraint) = fit_spline(samples, cx, cy, radial, coeff_count, d_max)?;
    if coeff_count > 0 && !bspline_is_monotone(&coeffs, d_max, d_max) {
        return Err(RefitError::NotMonotone);
    }
    let model = match radial {
        SplineRadial::IncidenceAngle => CameraModel::SfmtoolFisheye {
            focal_length: f,
            principal_point_x: cx,
            principal_point_y: cy,
            bspline_theta_max: d_max,
            bspline: coeffs,
        },
        SplineRadial::ImagePlaneRadius => CameraModel::SfmtoolPinhole {
            focal_length: f,
            principal_point_x: cx,
            principal_point_y: cy,
            bspline_rho_max: d_max,
            bspline: coeffs,
        },
    };
    let domain_deg = match radial {
        SplineRadial::IncidenceAngle => d_max.to_degrees(),
        SplineRadial::ImagePlaneRadius => d_max.atan().to_degrees(),
    };
    Ok((
        CameraIntrinsics {
            model,
            width: source.width,
            height: source.height,
        },
        domain_deg,
        constraint,
    ))
}

/// Each coefficient's basis value at radial coordinate `d`, as
/// `(coefficient index, value)` pairs: the at most four functions active
/// there, carried along their end tangent past `d_max` exactly as the model's
/// linear tail carries `δ`.
fn coefficient_basis(n: usize, d_max: f64, d: f64) -> [(usize, f64); BSPLINE_SUPPORT] {
    let (first, values, derivatives) = basis_at(n, d_max, d);
    let beyond = (d - d_max).max(0.0);
    let mut out = [(usize::MAX, 0.0); BSPLINE_SUPPORT];
    for j in 0..BSPLINE_SUPPORT {
        let full = first + j;
        // The first two functions of the full basis are the gauge-anchored
        // pair, which carry no coefficient.
        if full >= 2 && full - 2 < n {
            out[j] = (full - 2, values[j] + derivatives[j] * beyond);
        }
    }
    out
}

/// The linear least-squares fit of a spline model's focal and coefficients.
///
/// The model's pixel is `(cx, cy) + f·(d + Σ cᵢ·Bᵢ(d))·û`, which is linear in
/// `x = (f, f·c₀, …, f·c_{N−1})`, so the fit is one solve of the normal
/// equations, `N + 1` unknowns wide, without the constraint below. The second
/// difference of `f·c` is penalized, so a coefficient no sample reaches
/// continues its neighbours along the smoothest curve instead of leaving the
/// solve rank-deficient.
///
/// The radial map must be invertible, so the fit is constrained to keep its
/// slope at or above [`MIN_SLOPE`] times the focal ([`slope_floor_rows`]).
/// Where the unconstrained solution already does, it is the result, unchanged;
/// where it does not, the result is the least-squares optimum under the
/// constraint, the closest invertible curve to the source on this basis.
///
/// Returns `(f, c, constraint)`. `n = 0` fits the focal alone, has no
/// constraint, and leaves `d_max` unused.
fn fit_spline(
    samples: &Samples,
    cx: f64,
    cy: f64,
    radial: SplineRadial,
    n: usize,
    d_max: f64,
) -> Result<(f64, Vec<f64>, MonotoneConstraint), RefitError> {
    let base = unit_base(radial);
    let width = n + 1;
    let penalty_rows = n.saturating_sub(2);
    let data_rows = 2 * samples.rays.len();
    let mut design = DMatrix::<f64>::zeros(data_rows + penalty_rows, width);
    let mut rhs = DVector::<f64>::zeros(data_rows + penalty_rows);

    for (i, ray) in samples.rays.iter().enumerate() {
        let (x, y) = base.ray_to_pixel(*ray).ok_or(RefitError::Degenerate {
            reason: "the target's base model has no pixel for a sampled ray",
        })?;
        let d = radial_coordinate(radial, samples.theta[i / AZIMUTHS]);
        if d <= 0.0 {
            continue;
        }
        let unit = [x / d, y / d];
        let observed = [samples.source[i][0] - cx, samples.source[i][1] - cy];
        for axis in 0..2 {
            let row = 2 * i + axis;
            design[(row, 0)] = d * unit[axis];
            if n > 0 {
                for (c, value) in coefficient_basis(n, d_max, d) {
                    if c != usize::MAX {
                        design[(row, c + 1)] = value * unit[axis];
                    }
                }
            }
            rhs[row] = observed[axis];
        }
    }

    // One row per interior coefficient, `√w·(a_{i−1} − 2a_i + a_{i+1})`, with
    // the weight scaled so the penalty's share of the solve does not change
    // with the number of samples or of coefficients.
    if penalty_rows > 0 {
        let weight = (SMOOTHING * data_rows as f64 / n as f64).sqrt();
        for i in 1..n - 1 {
            let row = data_rows + i - 1;
            design[(row, i)] = weight;
            design[(row, i + 1)] = -2.0 * weight;
            design[(row, i + 2)] = weight;
        }
    }

    // The SVD of the design matrix itself rather than of the normal
    // equations, which square its condition number: an exact source has to
    // come back with exactly-zero coefficients to rounding, not to 1e-11.
    let svd = design.svd(true, true);
    let largest = svd.singular_values.max();
    let smallest = svd.singular_values.min();
    if largest.is_nan() || largest <= 0.0 || smallest <= largest * 1e-12 {
        return Err(RefitError::Degenerate {
            reason: "the samples do not determine every spline coefficient",
        });
    }
    let x = svd
        .solve(&rhs, largest * 1e-14)
        .map_err(|_| RefitError::Degenerate {
            reason: "the spline solve failed",
        })?;

    // The slope floor at every grid angle, `r′(d_k) − MIN_SLOPE·f ≥ 0`. The
    // unconstrained solution is kept exactly when it already holds, so a fit
    // the constraint does not touch is the plain least-squares one.
    let mut constraint = MonotoneConstraint::default();
    let x = if n > 0 {
        let (grid, rows) = slope_floor_rows(n, d_max);
        let slack = &rows * &x;
        if slack.iter().all(|&s| s >= 0.0) {
            x
        } else {
            let v_t = svd.v_t.as_ref().ok_or(RefitError::Degenerate {
                reason: "the spline solve failed",
            })?;
            let solved = least_squares_with_inequalities(&svd.singular_values, v_t, &x, &rows)
                .ok_or(RefitError::Degenerate {
                    reason: "the monotonicity-constrained spline solve did not converge",
                })?;
            let angles: Vec<f64> = grid
                .iter()
                .zip(&solved.active)
                .filter(|(_, &active)| active)
                .map(|(&d, _)| incidence_angle(radial, d).to_degrees())
                .collect();
            if !angles.is_empty() {
                constraint = MonotoneConstraint {
                    active: true,
                    active_angles: angles.len(),
                    range_deg: Some([
                        angles.iter().copied().fold(f64::INFINITY, f64::min),
                        angles.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    ]),
                };
            }
            solved.x
        }
    } else {
        x
    };
    let f = x[0];
    if !(f > 0.0 && f.is_finite()) {
        return Err(RefitError::Degenerate {
            reason: "the fitted focal is not positive",
        });
    }
    let coeffs: Vec<f64> = (0..n).map(|i| x[i + 1] / f).collect();
    if coeffs.iter().any(|c| !c.is_finite()) {
        return Err(RefitError::Degenerate {
            reason: "a fitted coefficient is not finite",
        });
    }
    Ok((f, coeffs, constraint))
}

/// The monotonicity constraint of an `n`-coefficient spline on `[0, d_max]`:
/// the grid of radial coordinates and one row per grid point, in the fit's
/// unknowns `x = (f, f·c₀, …, f·c_{N−1})`.
///
/// The radial map's slope is `r′(d) = f·(1 + Σ cᵢ·Bᵢ′(d)) = x₀ + Σ xᵢ₊₁·Bᵢ′(d)`,
/// linear in `x`, so `r′(d) ≥ MIN_SLOPE·f` is the homogeneous row
/// `(1 − MIN_SLOPE, B₀′(d), …, B_{N−1}′(d))·x ≥ 0`. The grid is
/// [`SLOPE_GRID_PER_SPAN`] points per knot span, computed as
/// `bspline_is_monotone` computes its own; its last point is `d_max`, where the
/// slope is the end tangent the linear tail carries past the domain.
fn slope_floor_rows(n: usize, d_max: f64) -> (Vec<f64>, DMatrix<f64>) {
    let points = SLOPE_GRID_PER_SPAN * (n - 1);
    let grid: Vec<f64> = (0..=points)
        .map(|s| d_max * s as f64 / points as f64)
        .collect();
    let mut rows = DMatrix::<f64>::zeros(grid.len(), n + 1);
    for (k, &d) in grid.iter().enumerate() {
        rows[(k, 0)] = 1.0 - MIN_SLOPE;
        let (first, _, derivatives) = basis_at(n, d_max, d);
        for (j, derivative) in derivatives.iter().enumerate() {
            let full = first + j;
            if full >= 2 && full - 2 < n {
                rows[(k, full - 1)] = *derivative;
            }
        }
    }
    (grid, rows)
}

/// The incidence angle, in radians, of a spline family's radial coordinate.
fn incidence_angle(radial: SplineRadial, d: f64) -> f64 {
    match radial {
        SplineRadial::IncidenceAngle => d,
        SplineRadial::ImagePlaneRadius => d.atan(),
    }
}

/// Whether a parameter name is a principal-point coordinate, which the fit
/// copies rather than solves.
fn is_principal_point(name: &str) -> bool {
    matches!(name, "principal_point_x" | "principal_point_y")
}

/// The starting parameters of a COLMAP target: every parameter the source
/// carries under the same name, the focal translated between one and two
/// values, and zero for the rest.
fn copied_parameters(source: &CameraIntrinsics, names: &[&str]) -> BTreeMap<String, f64> {
    let from = SfmrCamera::from(source).parameters;
    let (fx, fy) = source.focal_lengths();
    names
        .iter()
        .map(|&name| {
            let value = match from.get(name) {
                Some(&v) => v,
                None => match name {
                    "focal_length" => 0.5 * (fx + fy),
                    "focal_length_x" => fx,
                    "focal_length_y" => fy,
                    _ => 0.0,
                },
            };
            (name.to_string(), value)
        })
        .collect()
}

/// A camera of `camera_model` with `parameters`, or `None` where the registry refuses
/// them.
fn build_camera(
    source: &CameraIntrinsics,
    camera_model: &str,
    parameters: BTreeMap<String, f64>,
) -> Option<CameraIntrinsics> {
    CameraIntrinsics::try_from(&SfmrCamera {
        model: camera_model.to_string(),
        width: source.width,
        height: source.height,
        parameters,
    })
    .ok()
}

/// The residual vector of `camera` against the samples, two entries per ray,
/// or `None` when the camera has no pixel for one of them.
fn residuals(samples: &Samples, camera: &CameraIntrinsics) -> Option<Vec<f64>> {
    let mut out = Vec::with_capacity(2 * samples.rays.len());
    for (ray, source) in samples.rays.iter().zip(&samples.source) {
        let (u, v) = camera.ray_to_pixel(*ray)?;
        if !(u.is_finite() && v.is_finite()) {
            return None;
        }
        out.push(u - source[0]);
        out.push(v - source[1]);
    }
    Some(out)
}

/// The Levenberg–Marquardt fit of a COLMAP polynomial model.
///
/// Every parameter but the principal point is free. The start copies the
/// source's same-named parameters, so a target that contains the source's
/// model starts at zero error and stays there; when the copied start cannot
/// project every ray (a coefficient whose meaning changed between the models),
/// the distortion starts from zero instead. The Jacobian is a central
/// difference: the models' own projections are the only statement of what
/// each parameter does, and a second one here would be a second thing to keep
/// in step.
fn fit_colmap(
    source: &CameraIntrinsics,
    camera_model: &'static str,
    samples: &Samples,
) -> Result<CameraIntrinsics, RefitError> {
    let (_, names) = fixed_arity_model_by_name(camera_model).ok_or(RefitError::UnknownTarget {
        camera_model: camera_model.to_string(),
    })?;
    let mut start = copied_parameters(source, names);
    let free: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| !is_principal_point(n))
        .collect();
    let build = |values: &[f64], template: &BTreeMap<String, f64>| {
        let mut parameters = template.clone();
        for (name, v) in free.iter().zip(values) {
            parameters.insert(name.to_string(), *v);
        }
        build_camera(source, camera_model, parameters)
    };

    let evaluate = |values: &[f64], template: &BTreeMap<String, f64>| {
        let camera = build(values, template)?;
        let r = residuals(samples, &camera)?;
        let cost: f64 = r.iter().map(|x| x * x).sum();
        Some((camera, r, cost))
    };

    let mut values: Vec<f64> = free.iter().map(|n| start[*n]).collect();
    let mut state = evaluate(&values, &start);
    if state.is_none() {
        for (name, v) in free.iter().zip(values.iter_mut()) {
            if !name.starts_with("focal_length") {
                *v = 0.0;
                start.insert(name.to_string(), 0.0);
            }
        }
        state = evaluate(&values, &start);
    }
    let (mut camera, mut r, mut cost) = state.ok_or(RefitError::Degenerate {
        reason: "the target has no pixel for a sampled ray at its starting parameters",
    })?;

    let rows = r.len();
    let p = free.len();
    let mut lambda = 1e-3;
    for _ in 0..LM_MAX_ITERS {
        // Zero error is where a target containing the source's model starts,
        // and there is nothing to improve on.
        if cost <= 1e-24 * rows as f64 {
            break;
        }
        let mut jacobian = DMatrix::<f64>::zeros(rows, p);
        for j in 0..p {
            let h = 1e-6 * values[j].abs().max(1.0);
            let mut plus = values.clone();
            plus[j] += h;
            let mut minus = values.clone();
            minus[j] -= h;
            let rp = build(&plus, &start).and_then(|c| residuals(samples, &c));
            let rm = build(&minus, &start).and_then(|c| residuals(samples, &c));
            match (rp, rm) {
                (Some(rp), Some(rm)) => {
                    for i in 0..rows {
                        jacobian[(i, j)] = (rp[i] - rm[i]) / (2.0 * h);
                    }
                }
                (Some(rp), None) => {
                    for i in 0..rows {
                        jacobian[(i, j)] = (rp[i] - r[i]) / h;
                    }
                }
                (None, Some(rm)) => {
                    for i in 0..rows {
                        jacobian[(i, j)] = (r[i] - rm[i]) / h;
                    }
                }
                (None, None) => {}
            }
        }
        let jtj = jacobian.transpose() * &jacobian;
        let jtr = jacobian.transpose() * DVector::from_column_slice(&r);

        let mut improved = false;
        while lambda < 1e16 {
            let mut damped = jtj.clone();
            for j in 0..p {
                damped[(j, j)] += lambda * jtj[(j, j)].max(1e-12);
            }
            let Some(step) = damped.clone().cholesky().map(|c| c.solve(&(-&jtr))) else {
                lambda *= 10.0;
                continue;
            };
            let candidate: Vec<f64> = values.iter().zip(step.iter()).map(|(v, s)| v + s).collect();
            match evaluate(&candidate, &start) {
                Some((c, rc, cc)) if cc < cost => {
                    let relative = (cost - cc) / cost;
                    values = candidate;
                    camera = c;
                    r = rc;
                    cost = cc;
                    lambda = (lambda / 10.0).max(1e-12);
                    improved = relative > 1e-12;
                    break;
                }
                _ => lambda *= 10.0,
            }
        }
        if !improved {
            break;
        }
    }
    Ok(camera)
}

/// What `camera` cannot represent of `source`: a focal aspect where the target
/// keeps one focal, and every non-zero tangential or thin-prism term the
/// target has no parameter for.
fn dropped_terms(source: &CameraIntrinsics, camera: &CameraIntrinsics) -> Vec<DroppedTerm> {
    let mut dropped = Vec::new();
    let target_names = camera.model.parameter_names();
    let (fx, fy) = source.focal_lengths();
    let single_focal = !target_names.iter().any(|n| n == "focal_length_y");
    if single_focal && fx != 0.0 && (fy / fx - 1.0).abs() > 1e-12 {
        dropped.push(DroppedTerm::FocalAspect {
            fy_over_fx: fy / fx,
        });
    }
    for (name, value) in SfmrCamera::from(source).parameters {
        let asymmetric = name.starts_with("tangential_") || name.starts_with("thin_prism_");
        if asymmetric && value.abs() > 1e-12 && !target_names.iter().any(|n| *n == name) {
            dropped.push(DroppedTerm::Parameter { name, value });
        }
    }
    dropped
}

#[cfg(test)]
mod tests;
