// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! One round of the pairwise-flow normal: every view rendered on the round's
//! plane, the pairs chosen, each pair's flow and fits at every radius, the
//! combination at every radius, and the round's answer and next plane, as
//! the round's report.
//!
//! The crops' pyramids are built in parallel over crops, and the pairs are
//! read in parallel over pairs with rayon; each pair's reading depends on
//! nothing but its two crops, and the readings are collected and combined in
//! pair order, so the round does not depend on the thread count.

use std::sync::atomic::{AtomicU64, Ordering};

use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::numeric::splitmix64;
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::Progress;

use super::crops::{render_crops, Crop, CropGrid, CropPyramid, PairLevel, ViewSight};
use super::fit::{fit_pair, PairFit};
use super::flow::pair_flow;
use super::model::PairModel;
use super::report::{
    PairwiseFlowCrop, PairwiseFlowFit, PairwiseFlowPair, PairwiseFlowRadius, PairwiseFlowRound,
};
use super::support::{angle_deg, combine, decide, Combination, Decision, PairNormal, RoundKind};
use super::texture::{noise_threshold, window_side, PairSamples};
use super::{NormalError, PairwiseFlowNormalOptions};

/// The seed of the shuffle that chooses which pairs a point with more than
/// `max_pairs` of them reads. Fixed, so the choice is the same on every run.
const PAIR_SEED: u64 = 0x9A12_F10E;

/// What every round of one step shares: the point, the patch's axes and
/// size, and the observations the step can render.
pub(super) struct Point<'a, 'v> {
    /// The point `X`, the patch's centre.
    pub(super) point: Point3<f64>,
    /// The patch's in-plane axes, which set the crops' axes.
    pub(super) axes: [Vector3<f64>; 2],
    /// The patch's half-extent, world units.
    pub(super) half_extent: f64,
    /// How many observations the track holds.
    pub(super) observations: usize,
    /// The observations the step renders: each `in`, the first in its image,
    /// as `(observation, image)`.
    pub(super) attempted: Vec<(usize, u32)>,
    /// Each attempted observation's view.
    pub(super) views: Vec<ProjectedImage<'v>>,
    /// The unit mean of the directions from the point to the attempted
    /// views' camera centres.
    pub(super) mean_view: Vector3<f64>,
    /// The step's options.
    pub(super) options: &'a PairwiseFlowNormalOptions,
}

impl Point<'_, '_> {
    /// Whether the point has exactly two observations, one pair, and so runs
    /// one round read at the single pair's radius.
    pub(super) fn single_pair(&self) -> bool {
        self.attempted.len() == 2
    }

    /// Run one round on the plane of unit normal `plane`, with `more` saying
    /// whether a round may follow it.
    pub(super) fn round(
        &self,
        plane: Vector3<f64>,
        more: bool,
        progress: &Progress<'_>,
    ) -> Result<(PairwiseFlowRound, Decision), NormalError> {
        let options = self.options;
        let (grid, sights, crops) = {
            let _crops = progress.phase("crops");
            let (grid, sights) = CropGrid::new(
                self.point,
                plane,
                self.axes,
                self.half_extent,
                &self.views,
                options,
            );
            let crops: Vec<Option<Crop>> = match &grid {
                Some(grid) => render_crops(grid, &self.views, &sights),
                None => self.views.iter().map(|_| None).collect(),
            };
            (grid, sights, crops)
        };
        let mut crop_rows = vec![PairwiseFlowCrop::NotAttempted; self.observations];
        for (k, &(observation, image)) in self.attempted.iter().enumerate() {
            crop_rows[observation] = match (&crops[k], sights[k]) {
                (Some(crop), _) => PairwiseFlowCrop::Rendered {
                    image,
                    angle_deg: crop.angle_deg,
                    samples_per_pixel: crop.samples_per_pixel,
                    coverage: crop.coverage,
                },
                (None, ViewSight::TooOblique { angle_deg }) => {
                    PairwiseFlowCrop::TooOblique { image, angle_deg }
                }
                (None, _) => PairwiseFlowCrop::NotAttempted,
            };
        }
        let rendered: Vec<usize> = (0..crops.len()).filter(|&k| crops[k].is_some()).collect();
        let pairs: Vec<[usize; 2]> = choose_pairs(rendered.len(), options.max_pairs)
            .into_iter()
            .map(|[a, b]| [rendered[a], rendered[b]])
            .collect();
        let readings = match &grid {
            Some(grid) => self.read_pairs(grid, &crops, &pairs, progress)?,
            None => Vec::new(),
        };

        // Every pair's normal at every radius, facing the point's views.
        let normals: Vec<Vec<Option<PairNormal>>> = readings
            .iter()
            .map(|reading| {
                reading
                    .fits
                    .iter()
                    .map(|fit| {
                        fit.as_ref().map(|f| {
                            let n = Vector3::from(f.normal);
                            let n = if n.dot(&self.mean_view) < 0.0 { -n } else { n };
                            (n, f.sigma_deg)
                        })
                    })
                    .collect()
            })
            .collect();
        let at_radius = |k: usize| -> Vec<Option<PairNormal>> {
            normals.iter().map(|per_pair| per_pair[k]).collect()
        };
        let combinations: Vec<Combination> = (0..options.radii.len())
            .map(|k| combine(&at_radius(k), options.max_sigma_deg, options))
            .collect();
        let index_of = |r: f64| options.radii.iter().position(|&x| x == r);
        let single_fits;
        let kind = if self.single_pair() {
            single_fits = if normals.len() == 1 {
                normals[0].clone()
            } else {
                Vec::new()
            };
            RoundKind::SinglePair {
                fits: &single_fits,
                radius: index_of(options.single_pair_radius)
                    .expect("the options' check keeps the single pair's radius in the list"),
            }
        } else {
            let fallback = index_of(options.fallback_radius)
                .expect("the options' check keeps the fallback radius in the list");
            RoundKind::Several {
                fallback: combine(&at_radius(fallback), f64::INFINITY, options).normal,
            }
        };
        let decision = decide(&combinations, kind, self.mean_view, more, options);

        let pair_rows = pairs
            .iter()
            .zip(&readings)
            .enumerate()
            .map(|(p, (&[a, b], reading))| {
                let (obs_a, image_a) = self.attempted[a];
                let (obs_b, image_b) = self.attempted[b];
                let centres = [a, b].map(|k| crops[k].as_ref().map(|c| c.centre));
                let angle = match centres {
                    [Some(ca), Some(cb)] => {
                        let x = self.point.coords;
                        angle_deg((ca - x).normalize(), (cb - x).normalize())
                    }
                    _ => f64::NAN,
                };
                PairwiseFlowPair {
                    observations: [obs_a, obs_b],
                    images: [image_a, image_b],
                    angle_deg: angle,
                    level: reading.level,
                    half_width: reading.half_width,
                    fits: reading
                        .fits
                        .iter()
                        .zip(&combinations)
                        .map(|(fit, c)| {
                            fit.as_ref().map(|f| PairwiseFlowFit {
                                normal: f.normal,
                                sigma_deg: f.sigma_deg,
                                sigma_texture_deg: f.sigma_texture_deg,
                                sigma_residual_deg: f.sigma_residual_deg,
                                offset: f.offset,
                                rms_residual: f.rms_residual,
                                residual_variance: f.residual_variance,
                                determined: c.determined[p],
                                combined: c.combined[p],
                            })
                        })
                        .collect(),
                }
            })
            .collect();
        let radius_rows = options
            .radii
            .iter()
            .zip(&combinations)
            .map(|(&radius, c)| PairwiseFlowRadius {
                radius,
                normal: c.normal.map(Into::into),
                sigma_deg: c.sigma_deg,
                mean_chi2: c.mean_chi2,
                pairs_fitted: c.fitted,
                pairs_determined: c.determined_count(),
                pairs_combined: c.combined_count(),
            })
            .collect();
        let round = PairwiseFlowRound {
            plane_normal: plane.into(),
            sample_spacing: grid.map_or(f64::NAN, |g| g.plane.spacing),
            samples_per_patch: grid.map_or(f64::NAN, |g| g.samples_per_patch),
            crop_half_width: grid.map_or(0, |g| g.half_width),
            crops: crop_rows,
            pairs: pair_rows,
            radii: radius_rows,
            outcome: decision.outcome,
            chosen_radius: decision.chosen.map(|k| options.radii[k]),
            normal: decision.normal.map(Into::into),
            sigma_deg: decision.sigma_deg,
            from_mean_view_deg: decision.from_mean_view_deg,
            next_plane: decision.next_plane.map(Into::into),
        };
        Ok((round, decision))
    }

    /// Read every pair of `pairs`, indexes into `crops`, on `grid`: each
    /// crop's pyramid down to the deepest level any of its pairs reads, then
    /// each pair's flow and its fit at every radius, in parallel over pairs.
    fn read_pairs(
        &self,
        grid: &CropGrid,
        crops: &[Option<Crop>],
        pairs: &[[usize; 2]],
        progress: &Progress<'_>,
    ) -> Result<Vec<PairReading>, NormalError> {
        let crop = |k: usize| crops[k].as_ref().expect("a pair is of rendered crops");
        let levels: Vec<PairLevel> = pairs
            .iter()
            .map(|&[a, b]| PairLevel::new(crop(a).samples_per_pixel, crop(b).samples_per_pixel))
            .collect();
        let mut deepest: Vec<Option<u32>> = vec![None; crops.len()];
        for (&[a, b], level) in pairs.iter().zip(&levels) {
            for k in [a, b] {
                deepest[k] = Some(deepest[k].map_or(level.level, |d| d.max(level.level)));
            }
        }
        let pyramids: Vec<Option<CropPyramid>> = {
            let _pyramids = progress.phase("pyramids");
            deepest
                .par_iter()
                .enumerate()
                .map(|(k, d)| d.map(|d| CropPyramid::new(crop(k), d)))
                .collect()
        };
        let phase = progress.phase("pairs");
        let total = pairs.len() as u64;
        let done = AtomicU64::new(0);
        pairs
            .par_iter()
            .zip(&levels)
            .map(|(&[a, b], level)| {
                phase.check_cancel()?;
                let pyramid = |k: usize| pyramids[k].as_ref().expect("built for every pair");
                let reading = self.read_pair(
                    grid,
                    [crop(a), crop(b)],
                    [pyramid(a), pyramid(b)],
                    *level,
                    &phase,
                )?;
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                phase.count(n, Some(total), "pair");
                Ok(reading)
            })
            .collect()
    }

    /// One pair's flow at its level and its fit at every radius.
    fn read_pair(
        &self,
        grid: &CropGrid,
        crops: [&Crop; 2],
        pyramids: [&CropPyramid; 2],
        level: PairLevel,
        progress: &Progress<'_>,
    ) -> Result<PairReading, NormalError> {
        let options = self.options;
        let [a, b] = crops;
        let (from, to) = (
            pyramids[0].level(level.level),
            pyramids[1].level(level.level),
        );
        let half_width = from.0.width().min(from.0.height()) / 2;
        let unread = || PairReading {
            level: level.level,
            half_width,
            fits: vec![None; options.radii.len()],
        };
        let Some(flow) = pair_flow(from.0, to.0, &options.flow).map_err(NormalError::Flow)? else {
            return Ok(unread());
        };
        let window = window_side(options.correlation_length_px, &level);
        let samples = PairSamples::new(from, to, &flow, grid.half_width, &level, window);
        let thresholds = [a, b].map(|crop| {
            noise_threshold(
                window,
                crop.samples_per_pixel,
                level.level,
                options.noise_false_alarm,
            )
        });
        let model = PairModel {
            plane: grid.plane,
            from: a.centre,
            to: b.centre,
            level_scale: level.scale(),
        };
        let per_half_extent = self.half_extent / grid.plane.spacing;
        let floor = options.flow_floor_px * level.upsampling;
        let mut fits = Vec::with_capacity(options.radii.len());
        for &radius in &options.radii {
            progress.check_cancel()?;
            fits.push(
                samples
                    .disc(radius * per_half_extent, thresholds, floor)
                    .and_then(|disc| fit_pair(&model, &disc, options)),
            );
        }
        Ok(PairReading {
            level: level.level,
            half_width,
            fits,
        })
    }
}

/// One pair's reading in a round.
pub(super) struct PairReading {
    level: u32,
    half_width: u32,
    fits: Vec<Option<PairFit>>,
}

/// The pairs `[a, b]`, `a < b`, of `count` crops that a round reads, in
/// order: every pair, or where there are more than `max_pairs`, `max_pairs`
/// of them chosen by a shuffle with a fixed seed, so the same count always
/// gives the same choice.
pub(super) fn choose_pairs(count: usize, max_pairs: usize) -> Vec<[usize; 2]> {
    let mut all: Vec<[usize; 2]> = (0..count)
        .flat_map(|a| (a + 1..count).map(move |b| [a, b]))
        .collect();
    if all.len() <= max_pairs {
        return all;
    }
    // A partial Fisher-Yates shuffle: the first `max_pairs` places drawn
    // from the rest.
    let mut state = PAIR_SEED;
    for i in 0..max_pairs {
        let draw = splitmix64(&mut state);
        let j = i + (draw % (all.len() - i) as u64) as usize;
        all.swap(i, j);
    }
    all.truncate(max_pairs);
    all.sort_unstable();
    all
}
