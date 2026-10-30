// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The rolling-shutter step of an image's projection: the displacement
//! [`CameraIntrinsics::rolling_shutter`] adds after the model's distortion and
//! before the focal and principal point, its inverse, and its derivatives.
//!
//! The projection methods in `projection.rs` hand an image whose rates are not
//! zero to the functions here and run their own arithmetic, unchanged, for
//! every other camera, so a camera without rates projects bit for bit as it
//! did before the step existed.
//!
//! Everything here is written in the kernels' optical frame (`+Z` forward,
//! `y` down), where the radial kernels return their distorted point
//! `(x_o, y_o)` and the pixel is `(f·x_o + cx, f·y_o + cy)`. The `.sfmr`
//! camera frame's distorted point is `(x_d, y_d) = (x_o, −y_o)`, so the
//! readout position `t = (v − cy)/m = −f·y_d/m` of rows is `f·y_o/m` here, and
//! the image-plane velocity `(ẋ, ẏ)` of the `.sfmr` frame is `(ẋ, −ẏ)` here.
//! See `specs/core/camera/sfmtool-pinhole-kernels.md` § "The rolling-shutter
//! step".

use crate::camera::{CameraIntrinsics, CameraModel, RollingShutterScanlines};

/// Fixed-point passes the unprojection makes at most. Each pass contracts the
/// error by about `|ω|·(f/m)·(1 + ρ²)`, which is `10⁻³` on a measured phone
/// capture, so three or four passes reach rounding and the cap is only met
/// where the step is far from the identity.
const MAX_INVERSE_ITERATIONS: usize = 50;

/// The unprojection stops once a pass moves the distorted point by less than
/// this, in normalized image-plane units.
const INVERSE_TOLERANCE: f64 = 1e-12;

/// The rolling-shutter step of one image, with what it needs of the image's
/// intrinsics: its rates, its scanlines and `f/m`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RollingShutterStep {
    rates: [f64; 3],
    scanlines: RollingShutterScanlines,
    /// The image's focal over its short side, `f/min(width, height)`: the
    /// readout position per normalized image-plane unit.
    f_over_m: f64,
}

impl CameraIntrinsics {
    /// The rolling-shutter step the projection methods apply, or `None` when
    /// the camera carries no rolling shutter or its rates are all zero.
    ///
    /// # Panics
    ///
    /// Panics when a camera of a model other than `SFMTOOL_PINHOLE` carries
    /// non-zero rates: the step is defined on that model's image plane only.
    pub(crate) fn rolling_shutter_step(&self) -> Option<RollingShutterStep> {
        let rolling_shutter = self.rolling_shutter.as_ref()?;
        if rolling_shutter.is_zero() {
            return None;
        }
        assert!(
            matches!(self.model, CameraModel::SfmtoolPinhole { .. }),
            "a rolling shutter is applied only on an SFMTOOL_PINHOLE camera, not {}",
            self.model_name()
        );
        let m = f64::from(self.width.min(self.height));
        Some(RollingShutterStep {
            rates: rolling_shutter.rates,
            scanlines: rolling_shutter.scanlines,
            f_over_m: self.focal_lengths().0 / m,
        })
    }

    /// The readout position `t`, the optical-frame image-plane velocity `ṗ`
    /// and the readout gradient `∇t = ∂t/∂(x_d, y_d)` at the camera-frame
    /// point `ray`, for a caller that differentiates the step's parameters
    /// itself (the bundle adjustment's focal and spline columns); `None` when
    /// the camera has no active rolling shutter or the ray is behind it.
    pub(crate) fn rolling_shutter_motion(
        &self,
        ray: [f64; 3],
    ) -> Option<(f64, [f64; 2], [f64; 2])> {
        let step = self.rolling_shutter_step()?;
        let (rx, ry, rz) = (ray[0], -ray[1], -ray[2]);
        if rz <= 0.0 {
            return None;
        }
        let (x, y) = (rx / rz, ry / rz);
        let (x_d, y_d) = self.model.distort(x, y);
        Some((
            step.readout(x_d, y_d),
            step.velocity(x, y),
            step.readout_gradient(),
        ))
    }

    /// `∂(x_r, y_r)/∂ω`, the derivative of the rolling-shutter step's moved
    /// point with respect to the rates, at the camera-frame point `ray`, in
    /// the optical frame, read on `scanlines` whatever rates the camera
    /// carries: `t·∂ṗ/∂ω`, with the readout position `t` of the point's
    /// distorted position at this camera's focal. The step is linear in the
    /// rates, so the columns do not depend on them. `None` when the ray is
    /// behind the camera. The bundle adjustment's rate columns are these times
    /// the focal.
    pub(crate) fn rolling_shutter_rate_columns(
        &self,
        scanlines: RollingShutterScanlines,
        ray: [f64; 3],
    ) -> Option<[[f64; 3]; 2]> {
        let (rx, ry, rz) = (ray[0], -ray[1], -ray[2]);
        if rz <= 0.0 {
            return None;
        }
        let (x, y) = (rx / rz, ry / rz);
        let (x_d, y_d) = self.model.distort(x, y);
        let m = f64::from(self.width.min(self.height));
        let step = RollingShutterStep {
            rates: [0.0; 3],
            scanlines,
            f_over_m: self.focal_lengths().0 / m,
        };
        let t = step.readout(x_d, y_d);
        // `ṗ` is linear in `ω`, so its column for one rate is the velocity at
        // that unit rate.
        let columns: [[f64; 2]; 3] = std::array::from_fn(|k| {
            let mut unit = step;
            unit.rates[k] = 1.0;
            unit.velocity(x, y)
        });
        Some([columns.map(|c| t * c[0]), columns.map(|c| t * c[1])])
    }

    /// How far the rolling-shutter step is from the identity at the image's
    /// four corners: the largest Frobenius norm there of
    /// `∂(t·ṗ)/∂(x_d, y_d) = ṗ·∇tᵀ + t·∂ṗ/∂(x, y)·D⁻¹`, the step's Jacobian
    /// with respect to the distorted point less `I`, read with each corner
    /// pixel's normalized position taken as its distorted point, which differs
    /// from it only by the step's own small displacement. It bounds the contraction
    /// of the unprojection's fixed-point iteration there, which converges
    /// while it is under one. `0` for a camera without an active rolling
    /// shutter, and `NaN` where the radial map is singular or a value is not
    /// finite.
    pub(crate) fn rolling_shutter_contraction(&self) -> f64 {
        let Some(step) = self.rolling_shutter_step() else {
            return 0.0;
        };
        let f = self.focal_lengths().0;
        let (cx, cy) = self.principal_point();
        let (w, h) = (f64::from(self.width), f64::from(self.height));
        let mut worst = 0.0f64;
        for (u, v) in [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)] {
            let (x_d, y_d) = ((u - cx) / f, (v - cy) / f);
            let (x, y) = self.model.undistort(x_d, y_d);
            let Some(d) = self.model.distort_jacobian(x, y) else {
                return f64::NAN;
            };
            let det = d[0][0] * d[1][1] - d[0][1] * d[1][0];
            if !(det.is_finite() && det != 0.0) {
                return f64::NAN;
            }
            let d_inv = [
                [d[1][1] / det, -d[0][1] / det],
                [-d[1][0] / det, d[0][0] / det],
            ];
            let t = step.readout(x_d, y_d);
            let p = step.velocity(x, y);
            let g = step.readout_gradient();
            let vj = step.velocity_jacobian(x, y);
            let mut norm2 = 0.0;
            for r in 0..2 {
                for c in 0..2 {
                    let tv = t * (vj[r][0] * d_inv[0][c] + vj[r][1] * d_inv[1][c]);
                    let e = p[r] * g[c] + tv;
                    norm2 += e * e;
                }
            }
            let norm = norm2.sqrt();
            if !norm.is_finite() {
                return f64::NAN;
            }
            worst = worst.max(norm);
        }
        worst
    }

    /// [`Self::ray_to_pixel_with_jacobian`] of a camera with an active
    /// rolling shutter: the derivative of the whole map, the radial 2×2 `D`
    /// composed with the step's `(I + ṗ·∇tᵀ)·D + t·∂ṗ/∂(x, y)`.
    pub(super) fn rolling_shutter_ray_jacobian(
        &self,
        step: RollingShutterStep,
        ray: [f64; 3],
    ) -> Option<super::PixelJacobian> {
        let f = self.focal_lengths().0;
        let (cx, cy) = self.principal_point();
        let [rx, ry, rz] = [ray[0], -ray[1], -ray[2]];
        if rz <= 0.0 {
            return None;
        }
        let x = rx / rz;
        let y = ry / rz;
        if !self.model.forward_projection_invertible(x, y) {
            return None;
        }
        let d = self.model.distort_jacobian(x, y)?;
        let (x_d, y_d) = self.model.distort(x, y);
        let (x_r, y_r) = step.forward(x, y, x_d, y_d);
        let m = step.jacobian(x, y, x_d, y_d, d);
        // ∂(x, y)/∂ray = P·S, as on the perspective path of
        // `ray_to_pixel_with_jacobian`.
        let inv = 1.0 / rz;
        let ps = [[inv, 0.0, rx * inv * inv], [0.0, -inv, ry * inv * inv]];
        let mut jac = [[0.0f64; 3]; 2];
        for c in 0..3 {
            jac[0][c] = f * (m[0][0] * ps[0][c] + m[0][1] * ps[1][c]);
            jac[1][c] = f * (m[1][0] * ps[0][c] + m[1][1] * ps[1][c]);
        }
        Some(((f * x_r + cx, f * y_r + cy), jac))
    }
}

impl RollingShutterStep {
    /// The readout position `t` of the optical-frame distorted point
    /// `(x_d, y_d)`, in the short-side unit: `(v − cy)/m` for rows and
    /// `(u − cx)/m` for columns, read at the pixel the point reaches before
    /// this step.
    #[inline]
    pub(crate) fn readout(&self, x_d: f64, y_d: f64) -> f64 {
        match self.scanlines {
            RollingShutterScanlines::Rows => self.f_over_m * y_d,
            RollingShutterScanlines::Columns => self.f_over_m * x_d,
        }
    }

    /// `∇t = ∂t/∂(x_d, y_d)` in the optical frame.
    #[inline]
    fn readout_gradient(&self) -> [f64; 2] {
        match self.scanlines {
            RollingShutterScanlines::Rows => [0.0, self.f_over_m],
            RollingShutterScanlines::Columns => [self.f_over_m, 0.0],
        }
    }

    /// The image-plane velocity of a fixed point under the camera rotation
    /// `ω`, at the optical-frame undistorted point `(x, y)`, in the optical
    /// frame: the `.sfmr` frame's `(ẋ, ẏ)` at `(x, −y)` with its second
    /// component negated.
    #[inline]
    pub(crate) fn velocity(&self, x: f64, y: f64) -> [f64; 2] {
        let [wx, wy, wz] = self.rates;
        let (xs, ys) = (x, -y);
        let x_dot = wy * (1.0 + xs * xs) - wx * xs * ys + wz * ys;
        let y_dot = -wx * (1.0 + ys * ys) + wy * xs * ys - wz * xs;
        [x_dot, -y_dot]
    }

    /// `∂ṗ/∂(x, y)` in the optical frame, row-major.
    #[inline]
    fn velocity_jacobian(&self, x: f64, y: f64) -> [[f64; 2]; 2] {
        let [wx, wy, wz] = self.rates;
        let (xs, ys) = (x, -y);
        // The `.sfmr` frame's partials at (xs, ys).
        let dxdot_dx = 2.0 * wy * xs - wx * ys;
        let dxdot_dy = -wx * xs + wz;
        let dydot_dx = wy * ys - wz;
        let dydot_dy = -2.0 * wx * ys + wy * xs;
        // ṗ_o = (ẋ, −ẏ) and y_o = −y_s, so ∂/∂y_o = −∂/∂y_s.
        [[dxdot_dx, -dxdot_dy], [-dydot_dx, dydot_dy]]
    }

    /// The step itself: the distorted point `(x_d, y_d)` of the undistorted
    /// point `(x, y)`, both optical, moved to `(x_d, y_d) + t·ṗ(x, y)`.
    #[inline]
    pub(crate) fn forward(&self, x: f64, y: f64, x_d: f64, y_d: f64) -> (f64, f64) {
        let t = self.readout(x_d, y_d);
        let [px, py] = self.velocity(x, y);
        (x_d + t * px, y_d + t * py)
    }

    /// `∂(x_r, y_r)/∂(x, y) = (I + ṗ·∇tᵀ)·D + t·∂ṗ/∂(x, y)` at the undistorted
    /// point `(x, y)` with distorted point `(x_d, y_d)` and radial 2×2 `D`.
    fn jacobian(&self, x: f64, y: f64, x_d: f64, y_d: f64, d: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
        let t = self.readout(x_d, y_d);
        let p = self.velocity(x, y);
        let g = self.readout_gradient();
        let v = self.velocity_jacobian(x, y);
        let a = [
            [1.0 + p[0] * g[0], p[0] * g[1]],
            [p[1] * g[0], 1.0 + p[1] * g[1]],
        ];
        let mut out = [[0.0f64; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                out[r][c] = a[r][0] * d[0][c] + a[r][1] * d[1][c] + t * v[r][c];
            }
        }
        out
    }

    /// The optical-frame distorted point `(x_d, y_d)` that the step moves to
    /// `(x_r, y_r)`: a fixed-point iteration over the displacement around the
    /// model's exact radial inverse,
    /// `(x_d)ₖ₊₁ = (x_r) − t((x_d)ₖ)·ṗ(undistort((x_d)ₖ))`, from `(x_r)`,
    /// until a pass moves the point by less than `1e-12` or after
    /// [`MAX_INVERSE_ITERATIONS`] passes. A pass that does not converge
    /// returns its last iterate, as the radial inverse returns its base
    /// model's inverse rather than failing; a non-finite iterate stops the
    /// iteration and is returned as it is.
    pub(crate) fn invert(&self, model: &CameraModel, x_r: f64, y_r: f64) -> (f64, f64) {
        let (mut x_d, mut y_d) = (x_r, y_r);
        for _ in 0..MAX_INVERSE_ITERATIONS {
            let (x, y) = model.undistort(x_d, y_d);
            let t = self.readout(x_d, y_d);
            let [px, py] = self.velocity(x, y);
            let (nx, ny) = (x_r - t * px, y_r - t * py);
            let step = (nx - x_d).hypot(ny - y_d);
            (x_d, y_d) = (nx, ny);
            if step.is_nan() || step < INVERSE_TOLERANCE {
                break;
            }
        }
        (x_d, y_d)
    }
}

#[cfg(test)]
mod tests;
