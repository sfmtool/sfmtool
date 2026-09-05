// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `OPENCV` and `FULL_OPENCV`: radial + tangential distortion in the
//! normalized image plane, plus the full model's extra radial terms,
//! rational denominator and thin-prism offsets.

/// OpenCV distortion: radial (k1, k2) + tangential (p1, p2).
pub(in crate::camera::distortion) fn distort_opencv(
    x: f64,
    y: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
) -> (f64, f64) {
    let r2 = x * x + y * y;
    let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
    let x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    let y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    (x_d, y_d)
}

/// Full OpenCV distortion: rational radial (k1..k6) + tangential (p1, p2).
#[allow(clippy::too_many_arguments)]
pub(in crate::camera::distortion) fn distort_full_opencv(
    x: f64,
    y: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
    k3: f64,
    k4: f64,
    k5: f64,
    k6: f64,
) -> (f64, f64) {
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6);
    let x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    let y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    (x_d, y_d)
}
