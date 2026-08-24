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
    /// The largest `|model − ideal|` displacement over the sampled grid, and
    /// `None` when the model carries no distortion at all.
    ///
    /// Read the number with the grid in mind: it is a maximum over the whole
    /// image *rectangle*, and for a circular fisheye that rectangle's corners
    /// lie outside the lens's image circle, where the distortion polynomial is
    /// being extrapolated far past anything it was fitted to. The panel says so
    /// in the row's tooltip rather than quietly reporting a plausible-looking
    /// number for a part of the frame that is black.
    pub max_distortion: Option<f64>,
}

impl Derived {
    /// Compute the report for `camera`.
    pub(super) fn compute(camera: &CameraIntrinsics) -> Self {
        let fov = report::field_of_view(camera);
        let equiv_35mm = report::equiv_focal_length_35mm(camera);
        let max_distortion = camera.has_distortion().then(|| {
            // Rows chosen to keep the cells square, so the grid samples the
            // frame evenly rather than more densely along its short side.
            let rows = grid_rows(camera);
            report::distortion_field(camera, FIELD_COLS, rows)
                .iter()
                .map(|sample| {
                    (sample.pixel[0] - sample.reference[0])
                        .hypot(sample.pixel[1] - sample.reference[1])
                })
                .fold(0.0_f64, f64::max)
        });
        Self {
            fov,
            equiv_35mm,
            max_distortion,
        }
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
