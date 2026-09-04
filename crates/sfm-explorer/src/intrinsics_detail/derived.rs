// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-camera derived report: everything the panel shows that is not a
//! stored parameter.
//!
//! All of it comes out of [`sfmtool_core::camera::report`], where it is unit
//! tested without an egui context; nothing here recomputes a quantity the core
//! already defines. What this module adds is the *caching*: the field of view
//! is nine ray round trips and the displacement field a few hundred, and
//! `undistort` is iterative for the OpenCV family, so neither belongs in a
//! per-frame path.

use sfmtool_core::camera::report::{self, FieldOfView};
use sfmtool_core::camera::CameraIntrinsics;

/// Arrows across the image width in the grid the maximum displacement is taken
/// over — the same density the Image Detail overlay layer will default to,
/// so the panel's number and the overlay's legend describe one field.
const FIELD_COLS: usize = 16;

/// Points along the projection plot's curves. Enough that a 400-pixel-wide
/// plot draws a polyline rather than a chain of visible segments, and few
/// enough that the whole profile — this many angles over
/// [`PROFILE_AZIMUTHS`] azimuths — is a few thousand cheap forward
/// projections computed once per camera.
const PROFILE_SAMPLES: usize = 128;

/// Azimuths the plot's band is taken over, per the spec's "min and max over 32
/// azimuths". Azimuth matters only for a model that is not radially symmetric,
/// and sweeping it is the only way decentring distortion shows up at all.
const PROFILE_AZIMUTHS: usize = 32;

/// A band narrower than this is floating-point noise in the kernels' multiply
/// order rather than a lens that treats its azimuths differently.
const BAND_VISIBLE_PX: f64 = 0.05;

/// One camera's derived quantities, computed once per [`crate::scene::CameraRef`].
pub(super) struct Derived {
    /// The angles the frame subtends, or `None` for a camera with no frame.
    pub fov: Option<FieldOfView>,
    /// 35 mm-equivalent focal length in millimetres; `None` for the fisheye and
    /// equirectangular models, whose focal length is pixels per radian.
    pub equiv_35mm: Option<f64>,
    /// How far the lens displaces a pixel, and over what part of the frame
    /// that was measured. `None` when the model carries no distortion at all.
    pub max_distortion: Option<DistortionExtent>,
    /// Where this model stops describing a lens, in degrees off-axis, or
    /// `None` when it describes one at every angle. The plot shades everything
    /// past it; see [`super::projection_plot`].
    pub trustworthy_max_theta: Option<f64>,
    /// The projection plot's curves.
    pub profile: Profile,
}

/// The radial map the projection plot draws, sampled once per camera.
///
/// `radial_profile` **drops** angles the model will not project, so the curve
/// can stop short of the frame's corner angle — on `kerry_park` it ends at
/// 132.7°, past which the folded polynomial refuses the ray outright. The plot
/// draws what is here and lets the axis run to the corner regardless, because
/// where the curve stops is itself worth seeing.
pub(super) struct Profile {
    /// One entry per angle, ascending from 0°.
    pub samples: Vec<ProfileSample>,
    /// Whether the azimuth band is wide enough anywhere to be worth drawing.
    ///
    /// Measured rather than looked up: the spec's condition is "`fx ≠ fy`, or
    /// tangential/thin-prism terms present", and the width of the band *is*
    /// that condition evaluated, without a second per-model table to drift
    /// from the first.
    pub band_visible: bool,
}

/// One angle on the projection plot.
pub(super) struct ProfileSample {
    /// Incidence angle in degrees.
    pub theta_deg: f64,
    /// The model's radius at azimuth 0.
    pub radius_px: f64,
    /// The family's ideal radius at azimuth 0.
    pub reference_px: f64,
    /// Smallest and largest radius over the sampled azimuths.
    pub band_px: (f64, f64),
}

/// The largest `|model − ideal|` displacement the panel found, and the domain
/// it was allowed to look over.
///
/// The maximum bounds its own domain rather than running over the whole image
/// *rectangle*, and the row names the bound it used. On `kerry_park`'s real
/// `OPENCV_FISHEYE` the rectangle's corners are 150° off-axis, outside the
/// lens's image circle, where the `k1..k4` polynomial has folded — and the
/// unrestricted maximum comes out at 272.7 px, twenty times the 13.0 px the
/// lens actually displaces anything. Such a number is honest about two forward
/// maps and misleading about the camera.
pub(super) struct DistortionExtent {
    /// Largest displacement in pixels over the nodes that count.
    pub max_px: f64,
    /// The incidence-angle bound the nodes were filtered to, or `None` when
    /// the whole grid counted.
    pub limit_deg: Option<f64>,
    /// Grid nodes dropped by that filter, and the total the grid produced —
    /// how much of the frame the number is *not* about.
    pub excluded: (usize, usize),
}

impl Derived {
    /// Compute the report for `camera`.
    pub(super) fn compute(camera: &CameraIntrinsics) -> Self {
        let fov = report::field_of_view(camera);
        let equiv_35mm = report::equiv_focal_length_35mm(camera);
        let trustworthy_max_theta = report::trustworthy_max_theta_deg(camera);
        let max_distortion = camera
            .has_distortion()
            .then(|| distortion_extent(camera, trustworthy_max_theta));
        Self {
            fov,
            equiv_35mm,
            max_distortion,
            trustworthy_max_theta,
            profile: profile(camera),
        }
    }
}

/// Sample the radial map at azimuth 0, with the min/max envelope over
/// [`PROFILE_AZIMUTHS`] azimuths around it.
fn profile(camera: &CameraIntrinsics) -> Profile {
    let base = report::radial_profile(camera, 0.0, PROFILE_SAMPLES);
    // The other azimuths are only ever read by index against `base`. That is
    // sound because `radial_profile` samples the same ascending linspace for
    // every azimuth and drops only the tail — the angles a model refuses are
    // the widest ones — so index `i` is the same θ in every one of them, and a
    // short azimuth simply stops contributing past its own end.
    let azimuths: Vec<Vec<report::RadialSample>> = (1..PROFILE_AZIMUTHS)
        .map(|i| {
            let azimuth_deg = 360.0 * i as f64 / PROFILE_AZIMUTHS as f64;
            report::radial_profile(camera, azimuth_deg, PROFILE_SAMPLES)
        })
        .collect();

    let mut band_visible = false;
    let samples = base
        .iter()
        .enumerate()
        .map(|(i, sample)| {
            let mut lo = sample.radius_px;
            let mut hi = sample.radius_px;
            for azimuth in &azimuths {
                if let Some(other) = azimuth.get(i) {
                    lo = lo.min(other.radius_px);
                    hi = hi.max(other.radius_px);
                }
            }
            band_visible |= hi - lo > BAND_VISIBLE_PX;
            ProfileSample {
                theta_deg: sample.theta_deg,
                radius_px: sample.radius_px,
                reference_px: sample.reference_px,
                band_px: (lo, hi),
            }
        })
        .collect();

    Profile {
        samples,
        band_visible,
    }
}

/// The displacement field's maximum, taken over the part of the frame the
/// model can be held to.
fn distortion_extent(camera: &CameraIntrinsics, limit_deg: Option<f64>) -> DistortionExtent {
    // Rows chosen to keep the cells square, so the grid samples the frame
    // evenly rather than more densely along its short side.
    let rows = grid_rows(camera);
    let field = report::distortion_field(camera, FIELD_COLS, rows);
    let total = field.len();
    let inside = |sample: &report::DistortionSample| match limit_deg {
        Some(limit) => sample.theta_deg <= limit,
        None => true,
    };
    let max_px = field
        .iter()
        .filter(|sample| inside(sample))
        .map(|sample| {
            (sample.pixel[0] - sample.reference[0]).hypot(sample.pixel[1] - sample.reference[1])
        })
        .fold(0.0_f64, f64::max);
    let dropped = field.iter().filter(|sample| !inside(sample)).count();
    DistortionExtent {
        max_px,
        // A bound that excluded nothing is not worth qualifying the row with:
        // it is the same statement as "over the image", one clause longer.
        limit_deg: limit_deg.filter(|_| dropped > 0),
        excluded: (dropped, total),
    }
}

/// Grid rows that keep the sampled cells square at [`FIELD_COLS`] across.
fn grid_rows(camera: &CameraIntrinsics) -> usize {
    if camera.width == 0 {
        return FIELD_COLS;
    }
    let ratio = f64::from(camera.height) / f64::from(camera.width);
    ((FIELD_COLS as f64 * ratio).round() as usize).max(1)
}
