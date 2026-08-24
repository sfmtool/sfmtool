// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The layer's hover readout — the cheapest high-value part of the whole
//! overlay, because it turns the photograph into a calibrated protractor.
//!
//! It is not a tooltip of its own. The feature layer already owns one, hit-
//! testing keypoints through its kd-tree, and two tooltips fighting for the
//! cursor would be worse than either; so this returns *text*, and
//! `super::super::overlay` appends it below a painted rule to whatever the
//! feature layer produced. Hovering a keypoint then tells you its 3D point
//! **and** how many degrees off-axis it sits and how far the lens displaced it,
//! which is the natural way to ask whether a suspicious observation is a
//! rim-distortion artefact.
//!
//! With the layer off, nothing here runs and the feature tooltip is byte for
//! byte what it has always been.

use sfmtool_core::camera::report;
use sfmtool_core::camera::CameraIntrinsics;

use super::{DEGREE, MINUS};

/// Below this incidence angle the azimuth is arithmetic on rounding error: the
/// ray is the optical axis, and "which way round the axis" has no answer.
const AZIMUTH_FLOOR_DEG: f64 = 0.05;

/// The layer's contribution to the panel's one tooltip, for the pixel under the
/// cursor — or `None` when the cursor is not on the image at all.
///
/// `limit_deg` is [`report::trustworthy_max_theta_deg`]. Past it the
/// displacement is a fold in a polynomial rather than a lens, so the line says
/// so instead of printing a number that would be read as a measurement.
pub(super) fn readout(
    camera: &CameraIntrinsics,
    limit_deg: Option<f64>,
    pixel: [f64; 2],
) -> Option<String> {
    let [u, v] = pixel;
    if u < 0.0 || v < 0.0 || u > f64::from(camera.width) || v > f64::from(camera.height) {
        return None;
    }

    let ray = camera.pixel_to_ray(u, v);
    let off_axis = report::off_axis_angle_deg(camera, u, v);
    let mut text = format!(
        "pixel  ({}, {})\nray    ({}, {}, {})\noff-axis {off_axis:.1}{DEGREE}",
        decimals(u, 1),
        decimals(v, 1),
        decimals(ray[0], 3),
        decimals(ray[1], 3),
        decimals(ray[2], 3),
    );
    if off_axis >= AZIMUTH_FLOOR_DEG {
        // The same convention `radial_profile` sweeps in: 0° is +X (right),
        // 90° is +Y (up), in the canonical camera frame.
        let azimuth = ray[1].atan2(ray[0]).to_degrees().rem_euclid(360.0);
        text.push_str(&format!("   azimuth {azimuth:.1}{DEGREE}"));
    }

    if camera.has_distortion() {
        text.push('\n');
        text.push_str(&displacement_line(camera, limit_deg, u, v, off_axis));
    }
    Some(text)
}

/// The `distortion` line: the displacement at this exact pixel, or a statement
/// of why there is not one.
fn displacement_line(
    camera: &CameraIntrinsics,
    limit_deg: Option<f64>,
    u: f64,
    v: f64,
    off_axis: f64,
) -> String {
    if let Some(limit) = limit_deg {
        if off_axis > limit {
            return format!("distortion  not modelled past {limit:.1}{DEGREE}");
        }
    }
    match report::displacement_at(camera, u, v) {
        Some(sample) => {
            let displacement = (sample.pixel[0] - sample.reference[0])
                .hypot(sample.pixel[1] - sample.reference[1]);
            format!("distortion  {displacement:.2} px")
        }
        // The model refused the ray outright, which past a fold it does.
        None => "distortion  outside the model's domain".to_owned(),
    }
}

/// A number with `places` decimals, using the typographic minus the rest of the
/// layer's labels use so a column of coordinates lines up.
fn decimals(value: f64, places: usize) -> String {
    let text = format!("{:.*}", places, value.abs());
    // `-0.000` is a rounding artefact rather than a direction; the sign goes
    // with the digits, not with the underlying float.
    if value < 0.0 && text.bytes().any(|b| (b'1'..=b'9').contains(&b)) {
        format!("{MINUS}{text}")
    } else {
        text
    }
}
