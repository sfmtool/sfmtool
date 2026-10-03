// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pixel noise a reconstruction's own residuals measure: the RMS per-axis
//! reprojection residual over the observations of its finite points.
//!
//! This is the `σ_px` the point-or-bearing test weights its rays by (see
//! `specs/core/reconstruction/batch-triangulation-api.md` § "Point or bearing"
//! and § "The measured noise level"). It is the RMS rather than a robust spread,
//! because it has to cover pose, lens-model and keypoint error together, and
//! those are heavier-tailed than a Gaussian.

use ndarray::Array2;

use crate::reconstruction::data::observation_reprojection_error;
use crate::reconstruction::{ReconstructionError, SfmrReconstruction};

/// The RMS per-axis reprojection residual of a reconstruction, over all its
/// cameras and for each one.
///
/// A residual is the pixel offset `(du, dv)` from an observation to its finite
/// point's projection, and the per-axis RMS over `n` observations is
/// `√(Σ (du² + dv²) / 2n)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReprojectionNoise {
    /// The RMS per-axis residual in pixels over every counted observation, or
    /// `None` when none was counted.
    pub sigma_px: Option<f64>,
    /// How many observations `sigma_px` is measured over.
    pub observation_count: usize,
    /// The same measure over the observations made by each camera, indexed as
    /// `image_table.cameras`; `None` for a camera with no counted observation.
    pub per_camera_sigma_px: Vec<Option<f64>>,
    /// How many observations each entry of `per_camera_sigma_px` is measured
    /// over.
    pub per_camera_observation_count: Vec<usize>,
}

impl SfmrReconstruction {
    /// Measure the reconstruction's reprojection noise from the observations
    /// of its finite points (`w ≠ 0`).
    ///
    /// Each observation's pixel is the inline keypoint when the reconstruction
    /// carries the column, and otherwise its `.sift` feature's position, read
    /// from the workspace for the images that observe a finite point. An
    /// observation is left out when its pixel is not finite, or when the camera
    /// model cannot image its point (behind the camera, or outside the model's
    /// domain). Points at infinity are left out because the test this measure
    /// feeds is about them: their residuals are the bearing's, which is the
    /// model under question.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sfmtool_core::SfmrReconstruction;
    /// # fn run(recon: &SfmrReconstruction) -> Result<(), Box<dyn std::error::Error>> {
    /// let noise = recon.reprojection_noise()?;
    /// if let Some(sigma) = noise.sigma_px {
    ///     println!("{sigma:.3} px over {} observations", noise.observation_count);
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

    /// [`Self::reprojection_noise`] from pixels already read by
    /// `observation_pixels` for at least [`Self::finite_observing_images`].
    pub(crate) fn reprojection_noise_from_pixels(&self, pixels: &Array2<f32>) -> ReprojectionNoise {
        let images = &self.image_table.images;
        let points = &self.point_set.points;
        let n_cameras = self.image_table.cameras.len();
        let mut sum_sq = vec![0.0f64; n_cameras];
        let mut count = vec![0usize; n_cameras];
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
            let Some(error) = observation_reprojection_error(
                &image.quaternion_wxyz,
                &image.translation_xyz,
                &self.image_table.cameras[camera_index],
                &point.position,
                false,
                pixel,
            ) else {
                continue;
            };
            if !error.is_finite() {
                continue;
            }
            sum_sq[camera_index] += error * error;
            count[camera_index] += 1;
        }

        let rms = |s: f64, n: usize| (n > 0).then(|| (s / (2 * n) as f64).sqrt());
        let total_sq: f64 = sum_sq.iter().sum();
        let total: usize = count.iter().sum();
        ReprojectionNoise {
            sigma_px: rms(total_sq, total),
            observation_count: total,
            per_camera_sigma_px: sum_sq
                .iter()
                .zip(&count)
                .map(|(&s, &n)| rms(s, n))
                .collect(),
            per_camera_observation_count: count,
        }
    }

    /// [`Self::reprojection_noise`]'s overall `sigma_px`: the RMS per-axis
    /// reprojection residual in pixels over the observations of finite points,
    /// or `None` when there are none.
    pub fn reprojection_noise_px(&self) -> Result<Option<f64>, ReconstructionError> {
        Ok(self.reprojection_noise()?.sigma_px)
    }
}

#[cfg(test)]
pub(crate) mod tests;
