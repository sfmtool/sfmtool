// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The options' check, and the pieces composed as the step will compose them
//! for one round, on synthetic photographs of a textured plane: the crops,
//! each pair's level, flow, texture information and fits.

use nalgebra::Vector3;

use crate::features::optical_flow::OpticalFlowError;

use super::crops::{render_crops, Crop, CropGrid, CropPyramid, PairLevel};
use super::fit::{fit_pair, PairFit};
use super::flow::pair_flow;
use super::model::{tangent_basis, PairModel};
use super::scene::{origin, ring, Scene, HALF_EXTENT};
use super::texture::{noise_threshold, window_side, PairSamples};
use super::{NormalError, PairwiseFlowNormalOptions};

#[test]
fn the_default_options_pass_their_check() {
    assert_eq!(PairwiseFlowNormalOptions::default().check(), Ok(()));
}

#[test]
fn the_check_names_the_first_setting_that_fails() {
    let refused = |change: &dyn Fn(&mut PairwiseFlowNormalOptions)| {
        let mut options = PairwiseFlowNormalOptions::default();
        change(&mut options);
        options.check().unwrap_err()
    };
    let bad = NormalError::BadPairwiseFlowOption;
    assert_eq!(refused(&|o| o.rounds = 0), bad("rounds"));
    assert_eq!(refused(&|o| o.radii.clear()), bad("radii"));
    assert_eq!(refused(&|o| o.radii = vec![1.0, 0.5]), bad("radii"));
    assert_eq!(refused(&|o| o.radii = vec![1.0, 1.0]), bad("radii"));
    assert_eq!(refused(&|o| o.radii = vec![0.0, 1.0]), bad("radii"));
    assert_eq!(
        refused(&|o| o.min_samples_per_patch = 1),
        bad("min_samples_per_patch")
    );
    assert_eq!(
        refused(&|o| o.min_samples_per_patch = 200),
        bad("min_samples_per_patch")
    );
    assert_eq!(
        refused(&|o| o.crop_half_extents = 0.9),
        bad("crop_half_extents")
    );
    assert_eq!(
        refused(&|o| o.max_crop_samples = 40),
        bad("max_crop_samples")
    );
    assert_eq!(
        refused(&|o| o.max_view_angle_deg = 95.0),
        bad("max_view_angle_deg")
    );
    assert_eq!(
        refused(&|o| o.max_grid_tilt_deg = 90.0),
        bad("max_grid_tilt_deg")
    );
    assert_eq!(
        refused(&|o| o.grid_tilt_step_deg = 0.0),
        bad("grid_tilt_step_deg")
    );
    assert_eq!(
        refused(&|o| o.noise_false_alarm = 1.0),
        bad("noise_false_alarm")
    );
    assert_eq!(
        refused(&|o| o.flow_floor_px = f64::NAN),
        bad("flow_floor_px")
    );
    assert_eq!(refused(&|o| o.max_sigma_deg = 0.0), bad("max_sigma_deg"));
    assert_eq!(
        refused(&|o| o.max_mean_chi2 = f64::NAN),
        bad("max_mean_chi2")
    );
    assert_eq!(
        refused(&|o| o.single_pair_radius = -1.0),
        bad("single_pair_radius")
    );
    assert_eq!(refused(&|o| o.fit.refine.resolution = 1), bad("resolution"));
    assert_eq!(
        refused(&|o| o.flow.patch_size = 0),
        NormalError::Flow(OpticalFlowError::PatchSizeZero)
    );
    // An infinite bar on σ or χ² takes every answer, and is allowed.
    let open = PairwiseFlowNormalOptions {
        max_sigma_deg: f64::INFINITY,
        max_mean_chi2: f64::INFINITY,
        ..PairwiseFlowNormalOptions::default()
    };
    assert_eq!(open.check(), Ok(()));
}

/// The normal `n` turned by `deg` toward `toward`.
fn turned(n: Vector3<f64>, deg: f64, toward: Vector3<f64>) -> Vector3<f64> {
    let (a, _) = tangent_basis(n, toward);
    (n * deg.to_radians().cos() + a * deg.to_radians().sin()).normalize()
}

fn angle_deg(a: [f64; 3], b: Vector3<f64>) -> f64 {
    let a = Vector3::from(a);
    a.cross(&b).norm().atan2(a.dot(&b)).to_degrees()
}

/// One round of the pieces on `scene`: the crops rendered on the plane of
/// normal `plane_normal` through the origin, and for each pair of crops the
/// fits at each of `radii` (half-extents) with the pair's crops at its level
/// and its flow.
struct Round {
    grid: CropGrid,
    crops: Vec<Crop>,
}

struct PairReading {
    level: PairLevel,
    model: PairModel,
    samples: PairSamples,
    fits: Vec<Option<PairFit>>,
}

impl Round {
    fn new(scene: &Scene, plane_normal: Vector3<f64>, options: &PairwiseFlowNormalOptions) -> Self {
        let views = scene.views();
        let (grid, sights) = CropGrid::new(
            origin(),
            plane_normal,
            [Vector3::x(), Vector3::y()],
            HALF_EXTENT,
            &views,
            options,
        );
        let grid = grid.unwrap();
        let crops = render_crops(&grid, &views, &sights)
            .into_iter()
            .map(Option::unwrap)
            .collect();
        Self { grid, crops }
    }

    fn pair(
        &self,
        i: usize,
        j: usize,
        radii: &[f64],
        options: &PairwiseFlowNormalOptions,
    ) -> PairReading {
        let (a, b) = (&self.crops[i], &self.crops[j]);
        let level = PairLevel::new(a.samples_per_pixel, b.samples_per_pixel);
        let (pa, pb) = (
            CropPyramid::new(a, level.level),
            CropPyramid::new(b, level.level),
        );
        let (from, to) = (pa.level(level.level), pb.level(level.level));
        let flow = pair_flow(from.0, to.0, &options.flow).unwrap().unwrap();
        let window = window_side(options.correlation_length_px, &level);
        let samples = PairSamples::new(from, to, &flow, self.grid.half_width, &level, window);
        let thresholds = [a, b].map(|crop| {
            noise_threshold(
                window,
                crop.samples_per_pixel,
                level.level,
                options.noise_false_alarm,
            )
        });
        let model = PairModel {
            plane: self.grid.plane,
            from: a.centre,
            to: b.centre,
            level_scale: level.scale(),
        };
        let per_half_extent = HALF_EXTENT / self.grid.plane.spacing;
        let floor = options.flow_floor_px * level.upsampling;
        let fits = radii
            .iter()
            .map(|r| {
                let disc = samples.disc(r * per_half_extent, thresholds, floor)?;
                fit_pair(&model, &disc, options)
            })
            .collect();
        PairReading {
            level,
            model,
            samples,
            fits,
        }
    }
}

/// The plane's true normal in the scenes below, and the cameras: four on a
/// cone of 15° about a direction 20° from it, so the pairs are 21° to 30°
/// apart and every view sees the plane from 5° to 35° off its normal.
fn truth() -> Vector3<f64> {
    Vector3::new(0.25, -0.15, 1.0).normalize()
}

fn cameras() -> Vec<Vector3<f64>> {
    ring(turned(truth(), 20.0, Vector3::y()), 15.0, 4, 20.0)
}

/// Four views of a textured plane, the crops rendered on a plane 10° off the
/// true one, read at 1, 2 and 3 half-extents: every pair's fit lands within
/// a degree of the truth (0.01° to 0.6°), its error within twice its
/// texture σ, which the flow floor keeps honest (0.03 to 1.3 times it), and
/// its σ under the step's 10° bar. This is the draft's measurement on synthetic scenes
/// at contrast 30 with noise of 2 grey levels.
#[test]
fn pair_fits_on_a_textured_plane_land_near_the_true_normal() {
    let options = PairwiseFlowNormalOptions::default();
    let scene = Scene::plane(truth(), &cameras(), 30.0, 2.0, 1);
    let round = Round::new(&scene, turned(truth(), 10.0, Vector3::x()), &options);
    for i in 0..4 {
        for j in i + 1..4 {
            let reading = round.pair(i, j, &[1.0, 2.0, 3.0], &options);
            assert_eq!(reading.level.level, 0);
            for (fit, radius) in reading.fits.iter().zip([1.0, 2.0, 3.0]) {
                let fit = fit
                    .as_ref()
                    .unwrap_or_else(|| panic!("pair {i}-{j} at {radius}"));
                let error = angle_deg(fit.normal, truth());
                assert!(
                    error < 1.0 && error < 2.0 * fit.sigma_texture_deg && fit.sigma_deg < 10.0,
                    "pair {i}-{j} at {radius}: {error}° off, {fit:?}"
                );
            }
        }
    }
}

/// The predicted flow of the true plane carries each crop onto the other:
/// crop `j` read at the end of each sample's predicted flow matches crop `i`
/// far better than crop `j` read where it stands. This is the pair model
/// checked against crops rendered through the camera models.
#[test]
fn the_true_planes_predicted_flow_maps_one_crop_onto_the_other() {
    let options = PairwiseFlowNormalOptions::default();
    let scene = Scene::plane(truth(), &cameras(), 30.0, 0.0, 2);
    let round = Round::new(&scene, turned(truth(), 15.0, Vector3::x()), &options);
    let model = PairModel {
        plane: round.grid.plane,
        from: round.crops[0].centre,
        to: round.crops[2].centre,
        level_scale: 1.0,
    };
    let side = round.grid.side() as usize;
    let half = f64::from(round.grid.half_width);
    let (a, b) = (&round.crops[0], &round.crops[2]);
    // Crop `j` read bilinearly at index coordinates `(x, y)`.
    let read = |x: f64, y: f64| -> Option<f32> {
        if !(x >= 0.0 && y >= 0.0 && x < (side - 1) as f64 && y < (side - 1) as f64) {
            return None;
        }
        let (c, r) = (x as usize, y as usize);
        let at = [
            r * side + c,
            r * side + c + 1,
            (r + 1) * side + c,
            (r + 1) * side + c + 1,
        ];
        if !at.iter().all(|&k| b.valid[k]) {
            return None;
        }
        let (fx, fy) = ((x - c as f64) as f32, (y - r as f64) as f32);
        let d = b.image.data();
        let top = d[at[0]] * (1.0 - fx) + d[at[1]] * fx;
        let bottom = d[at[2]] * (1.0 - fx) + d[at[3]] * fx;
        Some(top * (1.0 - fy) + bottom * fy)
    };
    let (mut moved, mut still, mut count, mut largest) = (0.0, 0.0, 0usize, 0.0f64);
    for r in (side / 4..3 * side / 4).step_by(3) {
        for c in (side / 4..3 * side / 4).step_by(3) {
            let q = [c as f64 + 0.5 - half, r as f64 + 0.5 - half];
            let f = model.flow(q, truth()).unwrap();
            largest = largest.max(f[0].hypot(f[1]));
            let k = r * side + c;
            let (Some(there), Some(here)) = (
                read(c as f64 + f[0], r as f64 + f[1]),
                read(c as f64, r as f64),
            ) else {
                continue;
            };
            if !a.valid[k] {
                continue;
            }
            moved += f64::from((there - a.image.data()[k]).abs());
            still += f64::from((here - a.image.data()[k]).abs());
            count += 1;
        }
    }
    let (moved, still) = (255.0 * moved / count as f64, 255.0 * still / count as f64);
    assert!(largest > 3.0, "{largest}");
    assert!(moved < 0.2 * still, "moved {moved}, still {still}");
}

/// Two views 90° apart, given the true plane: the crops line up, so the
/// measured flow is under a tenth of a sample over the disc, and the pair's
/// normal is within its σ of the truth.
#[test]
fn on_the_true_plane_the_flow_is_near_zero() {
    let options = PairwiseFlowNormalOptions::default();
    let normal = truth();
    let scene = Scene::plane(normal, &ring(normal, 45.0, 2, 0.0), 30.0, 2.0, 3);
    let round = Round::new(&scene, normal, &options);
    let reading = round.pair(0, 1, &[2.0], &options);
    let radius = 2.0 * HALF_EXTENT / round.grid.plane.spacing;
    let mut lengths: Vec<f64> = reading
        .samples
        .positions
        .iter()
        .zip(&reading.samples.flows)
        .filter(|(p, _)| p[0].hypot(p[1]) <= radius)
        .map(|(_, f)| f[0].hypot(f[1]))
        .collect();
    let median = crate::numeric::median_in_place(&mut lengths);
    assert!(median < 0.1, "{median}");
    let fit = reading.fits[0].as_ref().unwrap();
    let error = angle_deg(fit.normal, normal);
    assert!(
        error < fit.sigma_deg.max(0.5),
        "{error}° against σ {}",
        fit.sigma_deg
    );
    // The model's predicted flow for the plane it was rendered on is zero.
    assert_eq!(
        reading
            .model
            .flow([10.0, -5.0], normal)
            .map(|f| f[0].abs() < 1e-9),
        Some(true)
    );
}

/// A plane of constant grey with noise of 2 grey levels: after the
/// significance threshold almost no window keeps any information, so no
/// pair has a fit whose texture σ is under the step's 10° bar.
#[test]
fn a_flat_crop_carries_no_information() {
    let options = PairwiseFlowNormalOptions::default();
    let scene = Scene::plane(truth(), &cameras(), 0.0, 2.0, 4);
    let round = Round::new(&scene, turned(truth(), 10.0, Vector3::x()), &options);
    for (i, j) in [(0, 1), (0, 2), (1, 3)] {
        let reading = round.pair(i, j, &[1.0, 3.0], &options);
        for fit in reading.fits.iter().flatten() {
            assert!(
                fit.sigma_texture_deg > options.max_sigma_deg,
                "pair {i}-{j}: {fit:?}"
            );
        }
    }
}
