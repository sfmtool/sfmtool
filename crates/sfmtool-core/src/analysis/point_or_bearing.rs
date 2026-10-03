// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The point-or-bearing test over a reconstruction: rays and noise weights
//! built from observations at the reconstruction's own poses and lenses, and
//! the test over its stored points.
//!
//! [`observation_ray`] and [`track_rays`] build a ray, its camera centre and its
//! 2×3 noise weight from an `(image, pixel)` observation, through
//! [`observed_ray`]. They take any observation, stored or not, so every caller
//! that holds a reconstruction weights a track's rays alike: the stored-point
//! convenience [`SfmrReconstruction::point_or_bearing_scores`] here, and the
//! tracks that discovery assembles from `.sift` keypoints and the bench from
//! its uncommitted sightings. See
//! `specs/core/reconstruction/batch-triangulation-api.md` § "Over a
//! reconstruction".

use std::borrow::Cow;

use nalgebra::{Matrix2x3, Point3, Vector3};
use ndarray::Array2;
use rayon::prelude::*;

use crate::readable::Readable;
use crate::reconstruction::triangulation::point_or_bearing::{
    bearing_score_batch, fit_point_and_bearing_batch, observed_ray, BearingScore, PointBearingFit,
    PointBearingFitOptions,
};
use crate::reconstruction::{ImageTable, ReconstructionError, SfmrReconstruction};

/// One observation as the point-or-bearing test reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservationRay {
    /// The unit world-frame ray through the pixel.
    pub dir: Vector3<f64>,
    /// The observing image's camera centre.
    pub center: Point3<f64>,
    /// The ray's 2×3 world-frame noise weight, `(1/σ_px)·J·R`.
    pub weight: Matrix2x3<f64>,
}

/// A batch of tracks of rays, in the CSR layout the batch functions take (not
/// the bench's one-track `bench::classify::TrackRays`): track `t` owns
/// `dirs[offsets[t]..offsets[t+1]]` and the matching `centers` and `weights`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RayBatch {
    /// Unit world-frame rays.
    pub dirs: Vec<Vector3<f64>>,
    /// The camera centre of each ray.
    pub centers: Vec<Point3<f64>>,
    /// The 2×3 noise weight of each ray.
    pub weights: Vec<Matrix2x3<f64>>,
    /// `M + 1` track boundaries.
    pub offsets: Vec<usize>,
}

/// The ray, camera centre and noise weight of `pixel` in image `image_index`
/// of `image_table`, at per-axis pixel noise `sigma_px`.
///
/// `None` when the index is past the table, the pixel is not finite, or
/// [`observed_ray`] declines it (a `sigma_px` that is not finite and positive,
/// or a pixel outside the camera model's domain).
pub fn observation_ray(
    image_table: &ImageTable,
    image_index: usize,
    pixel: [f64; 2],
    sigma_px: f64,
) -> Option<ObservationRay> {
    if !(pixel[0].is_finite() && pixel[1].is_finite()) {
        return None;
    }
    let image = image_table.images.get(image_index)?;
    let camera = image_table.cameras.get(image.camera_index as usize)?;
    let ray = observed_ray(camera, &image.quaternion_wxyz, pixel, sigma_px)?;
    Some(ObservationRay {
        dir: ray.dir,
        center: image.camera_center(),
        weight: ray.weight,
    })
}

/// [`observation_ray`] over tracks of `(image_index, pixel)` observations,
/// CSR-style: track `t` owns `observations[offsets[t]..offsets[t+1]]`. The
/// result has one track per input track, in order, holding the rays of the
/// observations that give one; a track whose observations give fewer than two
/// is still there, and the batch functions return `None` for it.
///
/// # Panics
///
/// When `offsets` is not non-decreasing or indexes past `observations`, as the
/// batch functions do for their own offsets. A caller holding offsets from
/// outside the crate validates them first.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::analysis::point_or_bearing::track_rays;
/// use sfmtool_core::reconstruction::triangulation::bearing_score_batch;
/// # fn run(recon: &sfmtool_core::SfmrReconstruction, sigma_px: f64) {
/// // Two candidate tracks, each a run of (image, pixel) sightings.
/// let observations = [
///     (0, [412.0, 230.5]),
///     (3, [398.2, 241.0]),
///     (1, [10.0, 20.0]),
///     (2, [12.5, 19.0]),
/// ];
/// let rays = track_rays(&recon.image_table, &observations, &[0, 2, 4], sigma_px);
/// let scores = bearing_score_batch(&rays.dirs, &rays.centers, &rays.offsets, &rays.weights);
/// # }
/// ```
pub fn track_rays(
    image_table: &ImageTable,
    observations: &[(usize, [f64; 2])],
    offsets: &[usize],
    sigma_px: f64,
) -> RayBatch {
    let m = offsets.len().saturating_sub(1);
    let per_track: Vec<Vec<ObservationRay>> = (0..m)
        .into_par_iter()
        .map(|t| {
            observations[offsets[t]..offsets[t + 1]]
                .iter()
                .filter_map(|&(image, pixel)| observation_ray(image_table, image, pixel, sigma_px))
                .collect()
        })
        .collect();
    let total: usize = per_track.iter().map(Vec::len).sum();
    let mut rays = RayBatch {
        dirs: Vec::with_capacity(total),
        centers: Vec::with_capacity(total),
        weights: Vec::with_capacity(total),
        offsets: Vec::with_capacity(m + 1),
    };
    rays.offsets.push(0);
    for track in per_track {
        for r in track {
            rays.dirs.push(r.dir);
            rays.centers.push(r.center);
            rays.weights.push(r.weight);
        }
        rays.offsets.push(rays.dirs.len());
    }
    rays
}

/// The point-or-bearing test over a set of a reconstruction's points.
#[derive(Debug, Clone, PartialEq)]
pub struct PointOrBearingScores {
    /// The per-axis pixel noise the rays were weighted by: the caller's, or
    /// the reconstruction's [`reprojection_noise_px`](SfmrReconstruction::reprojection_noise_px).
    pub sigma_px: f64,
    /// The points scored, in the order of `scores`: the caller's indexes, or
    /// every point in order.
    pub point_indexes: Vec<usize>,
    /// One entry per entry of `point_indexes`; `None` where fewer than two of
    /// the point's observations give a usable ray.
    pub scores: Vec<Option<BearingScore>>,
    /// The fits, when asked for, aligned as `scores` and `None` in the same
    /// places.
    pub fits: Option<Vec<Option<PointBearingFit>>>,
}

/// Why [`SfmrReconstruction::point_or_bearing_scores`] could not run.
#[derive(Debug)]
pub enum PointOrBearingError {
    /// No `sigma_px` was given and the reconstruction has no observation of a
    /// finite point to measure one from.
    NoNoiseLevel,
    /// The given `sigma_px` is not finite and positive.
    InvalidNoiseLevel(f64),
    /// A point index is past the reconstruction's points.
    PointIndexOutOfRange {
        /// The index asked for.
        index: usize,
        /// How many points the reconstruction has.
        point_count: usize,
    },
    /// Reading the observations' pixels failed.
    Reconstruction(ReconstructionError),
}

impl std::fmt::Display for PointOrBearingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoNoiseLevel => write!(
                f,
                "no sigma_px was given, and the reconstruction has no observation of a finite \
                 point to measure the reprojection noise from"
            ),
            Self::InvalidNoiseLevel(s) => {
                write!(
                    f,
                    "sigma_px must be finite and positive, got {}",
                    Readable(*s)
                )
            }
            Self::PointIndexOutOfRange { index, point_count } => write!(
                f,
                "point index {index} is past the reconstruction's {point_count} points"
            ),
            Self::Reconstruction(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PointOrBearingError {}

impl From<ReconstructionError> for PointOrBearingError {
    fn from(e: ReconstructionError) -> Self {
        Self::Reconstruction(e)
    }
}

impl SfmrReconstruction {
    /// Run the point-or-bearing test on points of this reconstruction.
    ///
    /// For each point in `point_indexes` (every point when `None`), each
    /// observation becomes a ray and its noise weight through [`track_rays`],
    /// from the observation's pixel (the inline keypoint, or the `.sift`
    /// position for a `sift_files` reconstruction without the inline column)
    /// at the per-axis pixel noise `sigma_px`. An observation whose pixel is
    /// not finite, or that [`observed_ray`] declines, gives no ray. The rays
    /// are scored with [`bearing_score_batch`]. Points at infinity are scored
    /// like finite points, since they are the ones a finite verdict would
    /// promote.
    ///
    /// `sigma_px` defaults to [`Self::reprojection_noise_px`], and a `.sift`
    /// file is then read once for both. With `fit` given, the same rays are
    /// also fitted with [`fit_point_and_bearing_batch`] under those options,
    /// each finite point warm-started from its stored position. A report that
    /// sets `Λ` beside `depth_score` fits with `soft_l1_scale: None`, the plain
    /// least squares the score approximates.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sfmtool_core::reconstruction::triangulation::{
    ///     is_finite, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD,
    /// };
    /// use sfmtool_core::SfmrReconstruction;
    /// # fn run(recon: &SfmrReconstruction) -> Result<(), Box<dyn std::error::Error>> {
    /// let result = recon.point_or_bearing_scores(None, None, None)?;
    /// for (&point, score) in result.point_indexes.iter().zip(&result.scores) {
    ///     let finite = score
    ///         .as_ref()
    ///         .is_some_and(|s| is_finite(s, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD));
    ///     println!("point {point}: finite {finite}");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn point_or_bearing_scores(
        &self,
        point_indexes: Option<&[usize]>,
        sigma_px: Option<f64>,
        fit: Option<&PointBearingFitOptions>,
    ) -> Result<PointOrBearingScores, PointOrBearingError> {
        let point_count = self.point_set.points.len();
        let point_indexes: Vec<usize> = match point_indexes {
            Some(idx) => {
                if let Some(&index) = idx.iter().find(|&&i| i >= point_count) {
                    return Err(PointOrBearingError::PointIndexOutOfRange { index, point_count });
                }
                idx.to_vec()
            }
            None => (0..point_count).collect(),
        };
        if let Some(s) = sigma_px {
            if !(s.is_finite() && s > 0.0) {
                return Err(PointOrBearingError::InvalidNoiseLevel(s));
            }
        }

        // The images whose pixels are read: those of the points asked for and,
        // when the noise level is to be measured, those that observe a finite
        // point, so that a `.sift` file is read once for both.
        let mut wanted = match sigma_px {
            Some(_) => vec![false; self.image_table.images.len()],
            None => self.finite_observing_images(),
        };
        for &p in &point_indexes {
            for obs in self.observations_for_point(p) {
                wanted[obs.image_index as usize] = true;
            }
        }
        let pixels: Cow<'_, Array2<f32>> = self.observation_pixels(&wanted)?;
        let sigma_px = match sigma_px {
            Some(s) => s,
            None => self
                .reprojection_noise_from_pixels(&pixels)
                .sigma_px
                .ok_or(PointOrBearingError::NoNoiseLevel)?,
        };

        let mut observations = Vec::new();
        let mut offsets = Vec::with_capacity(point_indexes.len() + 1);
        offsets.push(0);
        for &p in &point_indexes {
            let start = self.point_set.observation_offsets[p];
            for (k, obs) in self.observations_for_point(p).iter().enumerate() {
                let row = start + k;
                observations.push((
                    obs.image_index as usize,
                    [pixels[[row, 0]] as f64, pixels[[row, 1]] as f64],
                ));
            }
            offsets.push(observations.len());
        }
        let rays = track_rays(&self.image_table, &observations, &offsets, sigma_px);

        let scores = bearing_score_batch(&rays.dirs, &rays.centers, &rays.offsets, &rays.weights);
        let fits = fit.map(|options| {
            let starts: Vec<Point3<f64>> = point_indexes
                .iter()
                .map(|&p| {
                    let point = &self.point_set.points[p];
                    if point.is_at_infinity() {
                        Point3::new(f64::NAN, f64::NAN, f64::NAN)
                    } else {
                        point.position
                    }
                })
                .collect();
            fit_point_and_bearing_batch(
                &rays.dirs,
                &rays.centers,
                &rays.offsets,
                &rays.weights,
                Some(&starts),
                None,
                options,
            )
        });
        Ok(PointOrBearingScores {
            sigma_px,
            point_indexes,
            scores,
            fits,
        })
    }
}

#[cfg(test)]
mod tests;
