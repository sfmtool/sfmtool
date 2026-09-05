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

use sfmtool_core::camera::report::{self, DistortionExtent, FieldOfView};
use sfmtool_core::camera::CameraIntrinsics;

/// Arrows across the image width in the grid the maximum displacement is taken
/// over — the same density the Image Detail overlay layer will default to,
/// so the panel's number and the overlay's legend describe one field. Both go
/// through [`report::distortion_extent`], so "one field" is a shared
/// definition rather than a promise.
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
    /// that was measured. `None` when the model carries no distortion at all,
    /// which is the panel's cue to say so rather than to quote a zero.
    ///
    /// The grid the maximum was taken over comes back with it and is dropped:
    /// the panel quotes the number, the Image Detail overlay draws the arrows,
    /// and both take them from [`report::distortion_extent`].
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

impl Derived {
    /// Compute the report for `camera`.
    pub(super) fn compute(camera: &CameraIntrinsics) -> Self {
        let fov = report::field_of_view(camera);
        let equiv_35mm = report::equiv_focal_length_35mm(camera);
        let trustworthy_max_theta = report::trustworthy_max_theta_deg(camera);
        let max_distortion = camera
            .has_distortion()
            .then(|| report::distortion_extent(camera, FIELD_COLS));
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
