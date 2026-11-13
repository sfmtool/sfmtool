// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The fit of one pair's plane at one radius: the normal of the plane through
//! the point, and the pair's free 2D offset, that best explain the measured
//! flow over a disc, with the two σ of the normal.
//!
//! 1. **A coarse grid** of normals about the pair's mean viewing direction,
//!    tilts from 0 to `max_grid_tilt_deg` in `grid_tilt_step_deg` steps at
//!    azimuths `grid_azimuth_step_deg` apart, each with its offset solved in
//!    closed form; the grid's best is the start.
//! 2. **Gauss-Newton** from there over the normal's two tilt angles and the
//!    offset, damped as Levenberg-Marquardt, each sample's residual `r`
//!    weighted by its information `w` and by a Tukey weight on its whitened
//!    length `√(rᵀ w r)` in units of the whitened lengths' robust scale,
//!    the weights recomputed after every step.
//! 3. **σ from the texture**: the tilt block of `H⁻¹`, `H = Σ ρ J̃ᵀ w J̃`
//!    over the disc with the Tukey weights `ρ`, `J̃` the derivative of the
//!    residual with respect to the two tilt angles and the offset; `w` is
//!    already the information of one sample, a correlated square's shared
//!    among its `L²` samples (`super::texture`).
//! 4. **σ from the residuals**: the tilt block of `s² · L² · (Σ J̃ᵀ J̃)⁻¹`
//!    over the samples that carry weight, `s²` their Tukey-weighted residual
//!    variance per flow component.
//!
//! Each σ is the square root of the larger eigenvalue of its tilt
//! covariance, about two orthonormal axes in the fitted plane's own tangent
//! plane, in degrees, and `σ` is the larger of the two. A tilt the disc does
//! not fix, `H` or `Σ J̃ᵀ J̃` singular, gives no fit.
//!
//! The fit runs in `f64`: it sums a few thousand samples into 4×4 normal
//! equations, which costs nothing at that size and keeps the inverse well
//! conditioned near edge-on tilts.

use nalgebra::{Matrix2, Matrix4, Vector3, Vector4};

use crate::numeric::median;

use super::model::{tangent_basis, PairModel};
use super::texture::{Disc, Sym2};
use super::PairwiseFlowNormalOptions;

/// The coarse grid is scored on at most this many of the disc's samples,
/// taken at an even stride; it only picks the start.
const GRID_SAMPLES: usize = 4096;

/// A grid normal is scored only where it predicts the flow at this share of
/// the scored samples that carry information.
const MIN_GRID_COVERAGE: f64 = 0.95;

/// The most Gauss-Newton steps a fit takes.
const MAX_ITERATIONS: u32 = 30;

/// The most damping retries for one step before the fit stops.
const DAMPING_TRIES: u32 = 8;

/// The largest turn of the normal one step takes, radians.
const MAX_TILT_STEP: f64 = 0.2;

/// A step that turns the normal less than this, radians, ends the fit.
const CONVERGED_TILT: f64 = 1e-9;

/// The median of the length of a standard 2D normal vector, `√(2 ln 2)`: the
/// whitened lengths' median over this is their robust scale.
const CHI2_MEDIAN: f64 = 1.177_410_022_515_474_7;

/// An eigenvalue of a tilt block's Schur complement at or under this share
/// of the block's largest diagonal entry makes it singular.
const SINGULAR: f64 = 1e-10;

/// One pair's plane at one radius.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PairFit {
    /// The unit normal, on the side of the pair's cameras.
    pub(crate) normal: [f64; 3],
    /// The larger of the two σ below, degrees, largest axis.
    pub(crate) sigma_deg: f64,
    /// σ from the texture-weighted Hessian, degrees, largest axis.
    pub(crate) sigma_texture_deg: f64,
    /// σ from the fit's residuals, degrees, largest axis.
    pub(crate) sigma_residual_deg: f64,
    /// The pair's free shift, grid samples.
    pub(crate) offset: [f64; 2],
    /// The Tukey-weighted root mean square of the residuals' lengths over the
    /// samples that carry information, grid samples.
    pub(crate) rms_residual: f64,
    /// `σ_I²`, the photometric residual's variance over the disc, in the
    /// flow's grey units squared.
    pub(crate) residual_variance: f64,
}

/// One sample's predicted flow and its derivative with respect to the
/// normal's two tilt angles, or `None` where the pair model has no
/// prediction.
type Prediction = Option<([f64; 2], [[f64; 2]; 2])>;

fn quadratic(w: &Sym2, r: [f64; 2]) -> f64 {
    w[0] * r[0] * r[0] + 2.0 * w[1] * r[0] * r[1] + w[2] * r[1] * r[1]
}

fn carries(w: &Sym2) -> bool {
    w[0] + w[2] > 0.0
}

/// Fit the plane of `model`'s pair to the flow over `disc`; `None` where the
/// disc holds no information, no grid normal predicts its flow, or the fitted
/// tilt is not fixed (a singular Hessian).
pub(crate) fn fit_pair(
    model: &PairModel,
    disc: &Disc,
    options: &PairwiseFlowNormalOptions,
) -> Option<PairFit> {
    if !disc.information.iter().any(carries) {
        return None;
    }
    let x = model.plane.point;
    let mean_view = {
        let sum = (model.from - x).normalize() + (model.to - x).normalize();
        if sum.norm() > 1e-9 {
            sum.normalize()
        } else {
            model.plane.normal
        }
    };
    let start = grid_start(model, disc, mean_view, options)?;
    let mut fit = Gauss::new(model, disc, options.tukey_c, start);
    fit.run();
    fit.finish(mean_view)
}

/// The normals of the coarse grid about `mean_view`.
fn grid_normals(mean_view: Vector3<f64>, options: &PairwiseFlowNormalOptions) -> Vec<Vector3<f64>> {
    let (p, q) = tangent_basis(mean_view, Vector3::x());
    let mut normals = vec![mean_view];
    let step = options.grid_tilt_step_deg;
    let azimuths = (360.0 / options.grid_azimuth_step_deg).ceil().max(1.0) as usize;
    let mut k = 1;
    while (k as f64) * step <= options.max_grid_tilt_deg + 1e-9 {
        let tilt = (k as f64 * step).to_radians();
        for a in 0..azimuths {
            let azimuth = (a as f64 * options.grid_azimuth_step_deg).to_radians();
            let side = p * azimuth.cos() + q * azimuth.sin();
            normals.push(mean_view * tilt.cos() + side * tilt.sin());
        }
        k += 1;
    }
    normals
}

/// The offset that best explains `residuals` (measured less predicted
/// flow, `None` where there is no prediction) under the information, in
/// closed form: `(Σ w)⁻¹ Σ w r`, or the plain mean where `Σ w` is singular.
fn closed_form_offset(residuals: &[Option<[f64; 2]>], information: &[Sym2]) -> [f64; 2] {
    let mut sum = Matrix2::zeros();
    let mut rhs = nalgebra::Vector2::zeros();
    let mut plain = [0.0; 2];
    let mut count = 0.0;
    for (r, w) in residuals.iter().zip(information) {
        let Some(r) = r else { continue };
        let m = Matrix2::new(w[0], w[1], w[1], w[2]);
        sum += m;
        rhs += m * nalgebra::Vector2::new(r[0], r[1]);
        plain[0] += r[0];
        plain[1] += r[1];
        count += 1.0;
    }
    match sum.try_inverse() {
        Some(inverse) if sum.determinant() > 0.0 => {
            let o = inverse * rhs;
            [o[0], o[1]]
        }
        _ if count > 0.0 => [plain[0] / count, plain[1] / count],
        _ => [0.0; 2],
    }
}

/// The best normal of the coarse grid, scored over an even subset of the
/// disc's samples with each normal's offset solved in closed form.
fn grid_start(
    model: &PairModel,
    disc: &Disc,
    mean_view: Vector3<f64>,
    options: &PairwiseFlowNormalOptions,
) -> Option<Vector3<f64>> {
    let stride = disc.positions.len().div_ceil(GRID_SAMPLES).max(1);
    let subset: Vec<usize> = (0..disc.positions.len())
        .step_by(stride)
        .filter(|&k| carries(&disc.information[k]))
        .collect();
    if subset.is_empty() {
        return None;
    }
    let information: Vec<Sym2> = subset.iter().map(|&k| disc.information[k]).collect();
    let mut best: Option<(f64, Vector3<f64>)> = None;
    for n in grid_normals(mean_view, options) {
        let residuals: Vec<Option<[f64; 2]>> = subset
            .iter()
            .map(|&k| {
                let f = model.flow(disc.positions[k], n)?;
                let m = disc.flows[k];
                Some([m[0] - f[0], m[1] - f[1]])
            })
            .collect();
        let predicted = residuals.iter().filter(|r| r.is_some()).count();
        if (predicted as f64) < MIN_GRID_COVERAGE * subset.len() as f64 {
            continue;
        }
        let o = closed_form_offset(&residuals, &information);
        let (mut cost, mut weight) = (0.0, 0.0);
        for (r, w) in residuals.iter().zip(&information) {
            if let Some(r) = r {
                cost += quadratic(w, [r[0] - o[0], r[1] - o[1]]);
                weight += w[0] + w[2];
            }
        }
        let score = cost / weight;
        if score.is_finite() && best.is_none_or(|(b, _)| score < b) {
            best = Some((score, n));
        }
    }
    best.map(|(_, n)| n)
}

/// The Gauss-Newton fit's state.
struct Gauss<'a> {
    model: &'a PairModel,
    disc: &'a Disc,
    tukey_c: f64,
    normal: Vector3<f64>,
    offset: [f64; 2],
    damping: f64,
}

impl<'a> Gauss<'a> {
    fn new(model: &'a PairModel, disc: &'a Disc, tukey_c: f64, start: Vector3<f64>) -> Self {
        let mut fit = Self {
            model,
            disc,
            tukey_c,
            normal: start,
            offset: [0.0; 2],
            damping: 1e-3,
        };
        let residuals = fit.residuals(&fit.predict(start), [0.0; 2]);
        fit.offset = closed_form_offset(&residuals, &disc.information);
        fit
    }

    /// The tangent axes the tilt is stated against at `n`.
    fn basis(&self, n: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
        tangent_basis(n, self.model.plane.x_axis)
    }

    fn predict(&self, n: Vector3<f64>) -> Vec<Prediction> {
        let (a, b) = self.basis(n);
        self.disc
            .positions
            .iter()
            .map(|&q| self.model.flow_and_jacobian(q, n, a, b))
            .collect()
    }

    /// Measured less predicted flow less the offset, per sample.
    fn residuals(&self, predictions: &[Prediction], offset: [f64; 2]) -> Vec<Option<[f64; 2]>> {
        predictions
            .iter()
            .zip(&self.disc.flows)
            .map(|(p, m)| {
                let (f, _) = p.as_ref()?;
                Some([m[0] - f[0] - offset[0], m[1] - f[1] - offset[1]])
            })
            .collect()
    }

    /// Tukey's weight per sample on its whitened residual length, in units
    /// of the lengths' robust scale over the samples that carry information;
    /// 0 for a sample with no prediction.
    fn tukey(&self, residuals: &[Option<[f64; 2]>]) -> Vec<f64> {
        let lengths: Vec<Option<f64>> = residuals
            .iter()
            .zip(&self.disc.information)
            .map(|(r, w)| r.map(|r| quadratic(w, r).max(0.0).sqrt()))
            .collect();
        let carrying: Vec<f64> = lengths
            .iter()
            .zip(&self.disc.information)
            .filter_map(|(d, w)| d.filter(|_| carries(w)))
            .collect();
        let scale = median(&carrying) / CHI2_MEDIAN;
        let cut = self.tukey_c * scale;
        lengths
            .iter()
            .map(|d| match d {
                None => 0.0,
                Some(_) if !(cut.is_finite() && cut > 0.0) => 1.0,
                Some(d) if *d < cut => (1.0 - (d / cut).powi(2)).powi(2),
                Some(_) => 0.0,
            })
            .collect()
    }

    fn cost(&self, residuals: &[Option<[f64; 2]>], weights: &[f64]) -> f64 {
        residuals
            .iter()
            .zip(weights)
            .zip(&self.disc.information)
            .filter_map(|((r, &rho), w)| r.map(|r| rho * quadratic(w, r)))
            .sum()
    }

    /// The weighted normal equations `Σ ρ J̃ᵀ w J̃` and `Σ ρ J̃ᵀ w r`.
    fn normal_equations(
        &self,
        predictions: &[Prediction],
        residuals: &[Option<[f64; 2]>],
        weights: &[f64],
    ) -> (Matrix4<f64>, Vector4<f64>) {
        let mut h = Matrix4::zeros();
        let mut g = Vector4::zeros();
        for ((p, r), (&rho, w)) in predictions
            .iter()
            .zip(residuals)
            .zip(weights.iter().zip(&self.disc.information))
        {
            let (Some((_, j)), Some(r)) = (p, r) else {
                continue;
            };
            if rho == 0.0 {
                continue;
            }
            let jt = jacobian(j);
            let m = Matrix2::new(w[0], w[1], w[1], w[2]) * rho;
            h += jt.transpose() * m * jt;
            g += jt.transpose() * m * nalgebra::Vector2::new(r[0], r[1]);
        }
        (h, g)
    }

    fn run(&mut self) {
        for _ in 0..MAX_ITERATIONS {
            let predictions = self.predict(self.normal);
            let residuals = self.residuals(&predictions, self.offset);
            let weights = self.tukey(&residuals);
            let cost = self.cost(&residuals, &weights);
            let (h, g) = self.normal_equations(&predictions, &residuals, &weights);
            let (a, b) = self.basis(self.normal);
            let predicted_count = residuals.iter().filter(|r| r.is_some()).count();
            let mut accepted = None;
            for _ in 0..DAMPING_TRIES {
                let damped = h + Matrix4::from_diagonal(&h.diagonal()) * self.damping;
                let Some(step) = damped.cholesky().map(|c| c.solve(&g)) else {
                    self.damping *= 10.0;
                    continue;
                };
                let tilt = step[0].hypot(step[1]);
                let shrink = if tilt > MAX_TILT_STEP {
                    MAX_TILT_STEP / tilt
                } else {
                    1.0
                };
                let step = step * shrink;
                let normal = (self.normal + a * step[0] + b * step[1]).normalize();
                let offset = [self.offset[0] + step[2], self.offset[1] + step[3]];
                let trial_predictions = self.predict(normal);
                let trial = self.residuals(&trial_predictions, offset);
                // A step that loses a prediction is not compared on fewer
                // samples.
                let lost = trial.iter().filter(|r| r.is_some()).count() < predicted_count;
                let trial_cost = self.cost(&trial, &weights);
                if !lost && trial_cost <= cost {
                    accepted = Some((normal, offset, tilt * shrink));
                    self.damping = (self.damping / 10.0).max(1e-12);
                    break;
                }
                self.damping *= 10.0;
            }
            let Some((normal, offset, turned)) = accepted else {
                break;
            };
            self.normal = normal;
            self.offset = offset;
            if turned < CONVERGED_TILT {
                break;
            }
        }
    }

    /// The fit at its last normal and offset, with both σ; `None` where
    /// either covariance's tilt block is singular.
    fn finish(&self, mean_view: Vector3<f64>) -> Option<PairFit> {
        let predictions = self.predict(self.normal);
        let residuals = self.residuals(&predictions, self.offset);
        let weights = self.tukey(&residuals);
        let (h, _) = self.normal_equations(&predictions, &residuals, &weights);
        let texture = tilt_variance(&h)?;
        let mut plain = Matrix4::zeros();
        let (mut squared, mut weight) = (0.0, 0.0);
        for ((p, r), (&rho, w)) in predictions
            .iter()
            .zip(&residuals)
            .zip(weights.iter().zip(&self.disc.information))
        {
            let (Some((_, j)), Some(r)) = (p, r) else {
                continue;
            };
            if rho == 0.0 || !carries(w) {
                continue;
            }
            let jt = jacobian(j);
            plain += jt.transpose() * jt;
            squared += rho * (r[0] * r[0] + r[1] * r[1]);
            weight += rho;
        }
        if weight <= 0.0 {
            return None;
        }
        let s2 = squared / (2.0 * weight);
        let residual = s2 * self.disc.l2 * tilt_variance(&plain)?;
        let sigma_texture_deg = texture.sqrt().to_degrees();
        let sigma_residual_deg = residual.sqrt().to_degrees();
        let n = if self.normal.dot(&mean_view) < 0.0 {
            -self.normal
        } else {
            self.normal
        };
        let scale = self.model.level_scale;
        Some(PairFit {
            normal: n.into(),
            sigma_deg: sigma_texture_deg.max(sigma_residual_deg),
            sigma_texture_deg,
            sigma_residual_deg,
            offset: [self.offset[0] * scale, self.offset[1] * scale],
            rms_residual: (squared / weight).sqrt() * scale,
            residual_variance: self.disc.residual_variance,
        })
    }
}

/// `J̃`, the derivative of one sample's residual with respect to the two
/// tilt angles and the offset, from the model's `∂flow/∂(α, β)`: the
/// residual is measured less predicted less offset, so a step `δ` solves
/// `J̃ δ ≈ r` with `J̃ = [J | I]`.
fn jacobian(j: &[[f64; 2]; 2]) -> nalgebra::Matrix2x4<f64> {
    nalgebra::Matrix2x4::new(j[0][0], j[0][1], 1.0, 0.0, j[1][0], j[1][1], 0.0, 1.0)
}

/// The larger eigenvalue of the tilt block of `h⁻¹`, from the Schur
/// complement of the offset block: the variance of the worst-fixed tilt,
/// radians squared; `None` where `h`'s tilt is not fixed.
fn tilt_variance(h: &Matrix4<f64>) -> Option<f64> {
    let tt = h.fixed_view::<2, 2>(0, 0).into_owned();
    let to = h.fixed_view::<2, 2>(0, 2).into_owned();
    let oo = h.fixed_view::<2, 2>(2, 2).into_owned();
    let determinant = oo.determinant();
    if !(determinant.is_finite() && determinant > 0.0) {
        return None;
    }
    let oo_inv = oo.try_inverse()?;
    let schur = tt - to * oo_inv * to.transpose();
    let schur = (schur + schur.transpose()) * 0.5;
    let least = schur.symmetric_eigenvalues().min();
    let bar = SINGULAR * tt[(0, 0)].abs().max(tt[(1, 1)].abs());
    if !(least.is_finite() && least > bar && least > 0.0) {
        return None;
    }
    Some(1.0 / least)
}

#[cfg(test)]
mod tests;
