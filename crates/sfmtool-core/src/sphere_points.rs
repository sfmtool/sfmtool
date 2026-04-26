// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Generation of evenly-distributed points on the unit sphere via Thomson-style
//! repulsion relaxation.
//!
//! [`evenly_distributed_sphere_points`] samples `n` points uniformly on the
//! sphere and then runs an iterative tangent-projected 1/r² repulsion to push
//! them towards an evenly-distributed configuration. A KD-tree restricts the
//! per-point neighbor search to a finite cutoff, giving O(n log n) per
//! iteration instead of O(n²).
//!
//! All functions return points as a flat `Vec<f32>` of length `3 * n` in
//! row-major order: `[x0, y0, z0, x1, y1, z1, ...]`.

use rand::Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::f32::consts::PI;

use crate::spatial::PointCloud3;

/// Configuration for Thomson-style sphere-point relaxation.
///
/// `step_size` and `cutoff_multiplier` are both expressed in units of the
/// characteristic nearest-neighbor spacing `√(4π/n)`, so good defaults are
/// stable across `n`.
pub struct RelaxConfig {
    /// Number of relaxation iterations.
    pub iterations: usize,
    /// Step length per iteration, as a fraction of the characteristic
    /// nearest-neighbor spacing `√(4π/n)`.
    pub step_size: f32,
    /// Per-point neighbor cutoff radius, as a multiple of the characteristic
    /// nearest-neighbor spacing.
    pub cutoff_multiplier: f32,
}

impl Default for RelaxConfig {
    fn default() -> Self {
        Self {
            iterations: 50,
            step_size: 0.05,
            cutoff_multiplier: 5.0,
        }
    }
}

/// Sample `n` points uniformly on the unit sphere.
///
/// Marsaglia's method: sample three independent N(0, 1) values and normalize.
/// The joint distribution is rotationally symmetric, so the normalized vector
/// is uniform on the sphere. Resamples in the (vanishingly unlikely) case of
/// a near-zero magnitude vector.
///
/// Returns a flat `Vec<f32>` of length `3 * n`.
pub fn random_sphere_points(n: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    let mut points = Vec::with_capacity(3 * n);
    for _ in 0..n {
        loop {
            let x: f32 = rng.sample(StandardNormal);
            let y: f32 = rng.sample(StandardNormal);
            let z: f32 = rng.sample(StandardNormal);
            let norm_sq = x * x + y * y + z * z;
            if norm_sq > 1e-30 {
                let inv_norm = 1.0 / norm_sq.sqrt();
                points.push(x * inv_norm);
                points.push(y * inv_norm);
                points.push(z * inv_norm);
                break;
            }
        }
    }
    points
}

/// Relax points in-place towards an even distribution on the unit sphere.
///
/// At each iteration, each point experiences a 1/r² repulsion from neighbors
/// within `cutoff_multiplier * √(4π/n)`, projected onto the local tangent
/// plane and applied as a fixed-length step of `step_size * √(4π/n)` before
/// being renormalized back onto the unit sphere.
///
/// `points` must have length `3 * n` for some `n`. No-op for `n < 2`.
pub fn relax_sphere_points(points: &mut [f32], config: &RelaxConfig) {
    assert!(
        points.len().is_multiple_of(3),
        "points must be a flat array of 3D coords"
    );
    let n = points.len() / 3;
    if n < 2 {
        return;
    }

    let char_spacing = (4.0 * PI / n as f32).sqrt();
    let cutoff = config.cutoff_multiplier * char_spacing;
    let step_len = config.step_size * char_spacing;

    let mut next = vec![0.0f32; points.len()];

    for _ in 0..config.iterations {
        let cloud = PointCloud3::<f32>::new(points, n);
        let (offsets, indices) = cloud.within_radius(points, n, cutoff);

        next.par_chunks_mut(3).enumerate().for_each(|(i, slot)| {
            let px = points[3 * i];
            let py = points[3 * i + 1];
            let pz = points[3 * i + 2];

            // Sum 1/r² repulsion from neighbors (excluding self).
            // Force direction is (p - q) / |p - q|³ → magnitude 1/r² along the chord.
            let nb_range = offsets[i] as usize..offsets[i + 1] as usize;
            let mut fx = 0.0f32;
            let mut fy = 0.0f32;
            let mut fz = 0.0f32;
            for &j_u32 in &indices[nb_range] {
                let j = j_u32 as usize;
                if j == i {
                    continue;
                }
                let dx = px - points[3 * j];
                let dy = py - points[3 * j + 1];
                let dz = pz - points[3 * j + 2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 < 1e-12 {
                    continue;
                }
                let inv_r3 = 1.0 / (r2 * r2.sqrt());
                fx += dx * inv_r3;
                fy += dy * inv_r3;
                fz += dz * inv_r3;
            }

            // Project force onto tangent plane: f_t = f - (f·p) p
            let dot = fx * px + fy * py + fz * pz;
            let tx = fx - dot * px;
            let ty = fy - dot * py;
            let tz = fz - dot * pz;
            let t_norm = (tx * tx + ty * ty + tz * tz).sqrt();

            if t_norm < 1e-10 {
                slot[0] = px;
                slot[1] = py;
                slot[2] = pz;
            } else {
                let scale = step_len / t_norm;
                let mx = px + tx * scale;
                let my = py + ty * scale;
                let mz = pz + tz * scale;
                let m_norm = (mx * mx + my * my + mz * mz).sqrt();
                slot[0] = mx / m_norm;
                slot[1] = my / m_norm;
                slot[2] = mz / m_norm;
            }
        });

        points.copy_from_slice(&next);
    }
}

/// Generate `n` points evenly distributed on the unit sphere.
///
/// Combines [`random_sphere_points`] (uniform random initialization) with
/// [`relax_sphere_points`] (Thomson-style repulsion relaxation).
pub fn evenly_distributed_sphere_points(n: usize, config: &RelaxConfig) -> Vec<f32> {
    let mut points = random_sphere_points(n);
    relax_sphere_points(&mut points, config);
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_unit_norm(points: &[f32], tol: f32) {
        for chunk in points.chunks_exact(3) {
            let norm = (chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2]).sqrt();
            assert!((norm - 1.0).abs() < tol, "point not unit norm: norm={norm}");
        }
    }

    fn std_dev(values: &[f32]) -> f32 {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        variance.sqrt()
    }

    fn nn_distances(points: &[f32]) -> Vec<f32> {
        let n = points.len() / 3;
        PointCloud3::<f32>::new(points, n).nearest_neighbor_distances()
    }

    #[test]
    fn random_points_have_unit_norm() {
        let points = random_sphere_points(200);
        assert_eq!(points.len(), 600);
        assert_unit_norm(&points, 1e-5);
    }

    #[test]
    fn random_points_cover_sphere() {
        // A simple smoke check: octant occupancy should be roughly balanced.
        let points = random_sphere_points(8000);
        let mut counts = [0usize; 8];
        for chunk in points.chunks_exact(3) {
            let i = (chunk[0] >= 0.0) as usize
                | (((chunk[1] >= 0.0) as usize) << 1)
                | (((chunk[2] >= 0.0) as usize) << 2);
            counts[i] += 1;
        }
        // Uniform expectation per octant is 1000; allow plenty of slack.
        for c in counts {
            assert!(c > 700 && c < 1300, "octant count out of range: {c}");
        }
    }

    #[test]
    fn relaxation_improves_uniformity() {
        let n = 500;
        let initial = random_sphere_points(n);
        let rough_std = std_dev(&nn_distances(&initial));

        let mut relaxed = initial.clone();
        relax_sphere_points(&mut relaxed, &RelaxConfig::default());
        let smooth_std = std_dev(&nn_distances(&relaxed));

        // Random uniform sampling has high NN variance; relaxation should
        // cut it dramatically (well over 2x in practice).
        assert!(
            smooth_std < rough_std * 0.5,
            "relaxation did not cut NN variance enough: rough_std={rough_std}, smooth_std={smooth_std}"
        );
    }

    #[test]
    fn relaxed_points_remain_on_unit_sphere() {
        let mut points = random_sphere_points(300);
        relax_sphere_points(&mut points, &RelaxConfig::default());
        assert_unit_norm(&points, 1e-4);
    }

    #[test]
    fn evenly_distributed_returns_unit_norm_points() {
        let points = evenly_distributed_sphere_points(50, &RelaxConfig::default());
        assert_eq!(points.len(), 150);
        assert_unit_norm(&points, 1e-4);
    }

    #[test]
    fn empty_input_is_a_noop() {
        let mut empty: Vec<f32> = vec![];
        relax_sphere_points(&mut empty, &RelaxConfig::default());
        assert!(empty.is_empty());
    }

    #[test]
    fn single_point_is_a_noop() {
        let mut single = vec![1.0, 0.0, 0.0];
        relax_sphere_points(&mut single, &RelaxConfig::default());
        assert_eq!(single, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn two_points_become_approximately_antipodal() {
        // With fixed step length the algorithm has a limit cycle around
        // antipodal (≈ one step of angular travel), so the threshold here
        // is loose. The point is just to confirm the repulsion drives them
        // into opposite hemispheres from a random start.
        let mut points = random_sphere_points(2);
        let config = RelaxConfig {
            iterations: 500,
            step_size: 0.05,
            cutoff_multiplier: 5.0,
        };
        relax_sphere_points(&mut points, &config);
        let dot = points[0] * points[3] + points[1] * points[4] + points[2] * points[5];
        assert!(
            dot < -0.95,
            "two points did not converge near antipodal: dot={dot}"
        );
    }

    #[test]
    fn zero_iterations_leaves_points_unchanged() {
        let mut points = random_sphere_points(10);
        let snapshot = points.clone();
        let config = RelaxConfig {
            iterations: 0,
            ..Default::default()
        };
        relax_sphere_points(&mut points, &config);
        assert_eq!(points, snapshot);
    }
}
