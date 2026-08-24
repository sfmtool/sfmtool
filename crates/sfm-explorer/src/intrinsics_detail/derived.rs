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
}

/// The largest `|model − ideal|` displacement the panel found, and the domain
/// it was allowed to look over.
///
/// Phase 4 reported the maximum over the whole image *rectangle* and explained
/// itself in a tooltip, because it had no way to say anything narrower. It is
/// worth being precise about why that was not good enough: on `kerry_park`'s
/// real `OPENCV_FISHEYE` the rectangle's corners are 150° off-axis, outside
/// the lens's image circle, where the `k1..k4` polynomial has folded — and the
/// maximum came out at 241 px, twenty times the 12 px the lens actually
/// displaces anything. That number was honest about two forward maps and
/// misleading about the camera.
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
        let max_distortion = camera
            .has_distortion()
            .then(|| distortion_extent(camera, report::trustworthy_max_theta_deg(camera)));
        Self {
            fov,
            equiv_35mm,
            max_distortion,
        }
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
