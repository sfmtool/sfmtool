// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pixel noise a reconstruction's own residuals measure: the RMS per-axis
//! reprojection residual over the observations of its finite points, with
//! gross outliers left out.
//!
//! This is the `σ_px` the point-or-bearing test weights its rays by (see
//! `specs/core/reconstruction/batch-triangulation-api.md` § "Point or bearing"
//! and § "The measured noise level"). It is an RMS rather than a robust spread,
//! because it has to cover pose, lens-model and keypoint error together, and
//! those are heavier-tailed than a Gaussian. A robust spread only sets the
//! gate that decides which residuals are mismatches rather than noise: an
//! observation whose residual is more than [`OUTLIER_GATE`] robust spreads is
//! left out of the RMS.

use ndarray::Array2;

use crate::reconstruction::data::observation_reprojection_residual;
use crate::reconstruction::triangulation::observed_ray;
use crate::reconstruction::{ReconstructionError, SfmrReconstruction};

/// How many robust spreads an observation's residual may be before
/// [`SfmrReconstruction::reprojection_noise`] calls it an outlier and leaves
/// it out of the RMS. The robust spread is `1.4826 · median(|r|)` over the
/// per-axis residual components of the observations made by the same camera.
///
/// The gate is far wider than a Gaussian needs because real residuals are
/// not Gaussian: pose, lens-model and keypoint error give a long tail, and
/// that tail is noise the test has to allow for. On the Kerry Park ground
/// truth `tk117` and a seoul bull `sift_files` solve, both bundle adjusted,
/// the largest residuals sit at 18.8 and 13.1 robust spreads, and a gate of
/// 5 would leave out 3.7% and 3.3% of their observations and lower `σ` by
/// 23% and 20%. At 30 neither loses an observation. What the gate removes is
/// a residual no noise model explains, a mismatched keypoint: on the seoul
/// bull ground truth, four residuals at 34 to 79 spreads (7 to 16 px), apart
/// from the rest of its tail (27 spreads and under), which carried about half
/// of `Σ e²` and raised `σ` from 0.468 to 0.646 px. See
/// `specs/core/reconstruction/batch-triangulation-api.md` § "The measured
/// noise level" for the comparison with other estimators.
pub const OUTLIER_GATE: f64 = 30.0;

/// `1 / Φ⁻¹(3/4)`: the median absolute value of a zero-mean Gaussian sample
/// times this is its standard deviation.
const MAD_TO_SIGMA: f64 = 1.482_602_218_505_602;

/// The RMS per-axis reprojection residual of a reconstruction, over all its
/// cameras and for each one, with outliers left out.
///
/// A residual is the pixel offset `(du, dv)` from an observation to its finite
/// point's projection, and the per-axis RMS over `n` observations is
/// `√(Σ (du² + dv²) / 2n)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReprojectionNoise {
    /// The RMS per-axis residual in pixels over every counted observation, or
    /// `None` when none was counted. Never under the reconstruction's
    /// [`keypoint_resolution_px`]: a residual finer than an `f32` keypoint can
    /// state is round-off.
    pub sigma_px: Option<f64>,
    /// How many observations `sigma_px` is measured over.
    pub observation_count: usize,
    /// How many observations were left out as outliers: their residual's
    /// length is more than [`OUTLIER_GATE`] times their camera's robust
    /// spread.
    pub outlier_count: usize,
    /// The same measure over the observations made by each camera, indexed as
    /// `image_table.cameras`; `None` for a camera with no counted observation.
    /// Each is never under that camera's keypoint resolution.
    pub per_camera_sigma_px: Vec<Option<f64>>,
    /// How many observations each entry of `per_camera_sigma_px` is measured
    /// over.
    pub per_camera_observation_count: Vec<usize>,
}

/// One observation's residual, as the measure reads it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ObservationResidual {
    /// The observing image's camera, an index into `image_table.cameras`.
    pub camera: usize,
    /// The pixel offset from the observation to its point's projection.
    pub residual: [f64; 2],
}

impl SfmrReconstruction {
    /// Measure the reconstruction's reprojection noise from the observations
    /// of its finite points (`w ≠ 0`).
    ///
    /// Each observation's pixel is the inline keypoint when the reconstruction
    /// carries the column, and otherwise its `.sift` feature's position, read
    /// from the workspace for the images that observe a finite point. An
    /// observation is left out when its pixel is not finite, when the camera
    /// model cannot image its point (behind the camera, or outside the model's
    /// domain), or when [`observed_ray`] declines its pixel, since the
    /// point-or-bearing test then gives it no ray either. Points at infinity
    /// are left out because the test this measure feeds is about them: their
    /// residuals are the bearing's, which is the model under question. Of the
    /// rest, an observation whose residual is more than [`OUTLIER_GATE`]
    /// robust spreads of its camera's residuals is counted as an outlier and
    /// left out of the RMS.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sfmtool_core::SfmrReconstruction;
    /// # fn run(recon: &SfmrReconstruction) -> Result<(), Box<dyn std::error::Error>> {
    /// let noise = recon.reprojection_noise()?;
    /// if let Some(sigma) = noise.sigma_px {
    ///     println!(
    ///         "{sigma:.3} px over {} observations, {} excluded as outliers",
    ///         noise.observation_count, noise.outlier_count
    ///     );
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn reprojection_noise(&self) -> Result<ReprojectionNoise, ReconstructionError> {
        let pixels = self.observation_pixels(&self.finite_observing_images())?;
        Ok(self.reprojection_noise_from_pixels(&pixels))
    }

    /// One flag per image: whether it observes a finite point, which is whether
    /// [`Self::reprojection_noise`] reads its pixels.
    pub(crate) fn finite_observing_images(&self) -> Vec<bool> {
        let points = &self.point_set.points;
        let mut wanted = vec![false; self.image_table.images.len()];
        for obs in &self.point_set.tracks {
            if !points[obs.point_index as usize].is_at_infinity() {
                wanted[obs.image_index as usize] = true;
            }
        }
        wanted
    }

    /// The residual of every observation of a finite point that
    /// [`Self::reprojection_noise`] considers, before the outlier gate, from
    /// pixels already read by `observation_pixels` for at least
    /// [`Self::finite_observing_images`].
    pub(crate) fn finite_observation_residuals(
        &self,
        pixels: &Array2<f32>,
    ) -> Vec<ObservationResidual> {
        let images = &self.image_table.images;
        let points = &self.point_set.points;
        let mut out = Vec::new();
        for (row, obs) in self.point_set.tracks.iter().enumerate() {
            let point = &points[obs.point_index as usize];
            if point.is_at_infinity() {
                continue;
            }
            let pixel = [pixels[[row, 0]] as f64, pixels[[row, 1]] as f64];
            if !(pixel[0].is_finite() && pixel[1].is_finite()) {
                continue;
            }
            let image = &images[obs.image_index as usize];
            let camera_index = image.camera_index as usize;
            let camera = &self.image_table.cameras[camera_index];
            // The test gives a pixel observed_ray declines no ray, so it does
            // not count towards the noise the rays are weighted by either. The
            // noise level passed here only scales the weight, which is unused.
            if observed_ray(camera, &image.quaternion_wxyz, pixel, 1.0).is_none() {
                continue;
            }
            let Some(residual) = observation_reprojection_residual(
                &image.quaternion_wxyz,
                &image.translation_xyz,
                camera,
                &point.position,
                false,
                pixel,
            ) else {
                continue;
            };
            if !(residual[0].is_finite() && residual[1].is_finite()) {
                continue;
            }
            out.push(ObservationResidual {
                camera: camera_index,
                residual,
            });
        }
        out
    }

    /// [`Self::reprojection_noise`] from pixels already read by
    /// `observation_pixels` for at least [`Self::finite_observing_images`].
    pub(crate) fn reprojection_noise_from_pixels(&self, pixels: &Array2<f32>) -> ReprojectionNoise {
        let residuals = self.finite_observation_residuals(pixels);
        let mut noise =
            gated_reprojection_noise(&residuals, self.image_table.cameras.len(), OUTLIER_GATE);
        // Keypoints at the exact projections of their points measure round-off,
        // or 0, which would weight every ray of the point-or-bearing test
        // infinitely; the level is never finer than a keypoint can be stored.
        for (sigma, camera) in noise
            .per_camera_sigma_px
            .iter_mut()
            .zip(&self.image_table.cameras)
        {
            *sigma = sigma.map(|s| s.max(camera_keypoint_resolution_px(camera)));
        }
        noise.sigma_px = noise.sigma_px.map(|s| s.max(keypoint_resolution_px(self)));
        noise
    }

    /// [`Self::reprojection_noise`]'s overall `sigma_px`: the RMS per-axis
    /// reprojection residual in pixels over the observations of finite points,
    /// outliers left out, or `None` when there are none.
    pub fn reprojection_noise_px(&self) -> Result<Option<f64>, ReconstructionError> {
        Ok(self.reprojection_noise()?.sigma_px)
    }
}

/// The finest pixel coordinate an `f32` keypoint states in any of `recon`'s
/// cameras: the `f32` machine epsilon times the camera's larger dimension,
/// about 10⁻⁴ px on a 1,000 px image.
///
/// A bound of representation, not a noise floor: a measured noise level under
/// it is round-off. Real captures measure tenths of a pixel, thousands of times
/// above it, so it binds only on exact data, where every consumer of the
/// point-or-bearing test then calls any track with parallax a point.
pub fn keypoint_resolution_px(recon: &SfmrReconstruction) -> f64 {
    recon
        .image_table
        .cameras
        .iter()
        .map(camera_keypoint_resolution_px)
        .fold(f64::from(f32::EPSILON), f64::max)
}

/// [`keypoint_resolution_px`] for one camera.
fn camera_keypoint_resolution_px(camera: &crate::camera::CameraIntrinsics) -> f64 {
    f64::from(camera.width.max(camera.height).max(1)) * f64::from(f32::EPSILON)
}

/// The robust per-axis spread of `components`: `1.4826 · median(|c|)`.
/// `None` for an empty slice.
fn robust_spread(components: &mut [f64]) -> Option<f64> {
    if components.is_empty() {
        return None;
    }
    for c in components.iter_mut() {
        *c = c.abs();
    }
    let mid = components.len() / 2;
    let (_, &mut upper, _) = components.select_nth_unstable_by(mid, f64::total_cmp);
    let median = if components.len() % 2 == 1 {
        upper
    } else {
        let lower = components[..mid]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower + upper)
    };
    Some(MAD_TO_SIGMA * median)
}

/// The gated RMS over `residuals`, with `camera_count` cameras: each camera's
/// residuals are gated at `gate` times that camera's robust spread, and the
/// RMS is taken over what passes, per camera and pooled.
pub(crate) fn gated_reprojection_noise(
    residuals: &[ObservationResidual],
    camera_count: usize,
    gate: f64,
) -> ReprojectionNoise {
    let mut components: Vec<Vec<f64>> = vec![Vec::new(); camera_count];
    for r in residuals {
        components[r.camera].extend_from_slice(&r.residual);
    }
    // A spread of 0 (more than half the components exactly 0, which only
    // synthetic data reaches) gates nothing, rather than everything.
    let limit_sq: Vec<f64> = components
        .iter_mut()
        .map(|c| match robust_spread(c) {
            Some(s) if s > 0.0 => (gate * s).powi(2),
            _ => f64::INFINITY,
        })
        .collect();

    let mut sum_sq = vec![0.0f64; camera_count];
    let mut count = vec![0usize; camera_count];
    let mut outlier_count = 0usize;
    for r in residuals {
        let e2 = r.residual[0] * r.residual[0] + r.residual[1] * r.residual[1];
        if e2 > limit_sq[r.camera] {
            outlier_count += 1;
            continue;
        }
        sum_sq[r.camera] += e2;
        count[r.camera] += 1;
    }

    let rms = |s: f64, n: usize| (n > 0).then(|| (s / (2 * n) as f64).sqrt());
    let total_sq: f64 = sum_sq.iter().sum();
    let total: usize = count.iter().sum();
    ReprojectionNoise {
        sigma_px: rms(total_sq, total),
        observation_count: total,
        outlier_count,
        per_camera_sigma_px: sum_sq
            .iter()
            .zip(&count)
            .map(|(&s, &n)| rms(s, n))
            .collect(),
        per_camera_observation_count: count,
    }
}

#[cfg(test)]
pub(crate) mod tests;
