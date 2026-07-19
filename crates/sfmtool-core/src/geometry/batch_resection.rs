// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Batch pose-only registration against fixed structure.
//!
//! [`resect_images_batch`] is the standalone registration primitive of
//! `specs/core/reconstruction-growth.md`: each requested image is resected
//! independently against the supplied points (no adjustment, no cross-image
//! coupling), parallelized across images. The per-image estimate-then-refine
//! ladder is shared with the growth kernel
//! (`super::reconstruction_growth::resect_one`).

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use rayon::prelude::*;

use crate::geometry::reconstruction_growth::{build_covisibility, per_image_seed, resect_one};
use crate::CameraIntrinsics;

/// Batch-registration P3P consensus floor.
const P3P_MIN_CONSENSUS_BATCH: usize = 8;
/// Covisible-neighbour inits tried by the batch-registration fallback.
const BATCH_FALLBACK_INITS: usize = 2;

/// Options for [`resect_images_batch`].
#[derive(Clone, Debug)]
pub struct ResectOptions {
    /// Skip an image with fewer observations of valid points than this.
    pub min_obs: usize,
    /// Accept an image at or above this all-observation inlier fraction.
    pub accept_gate: f64,
    /// Base seed; each image's RANSAC is seeded as a pure function of
    /// `(seed, image index)`.
    pub seed: u64,
}

impl Default for ResectOptions {
    fn default() -> Self {
        Self {
            min_obs: 8,
            accept_gate: 0.30,
            seed: 0,
        }
    }
}

/// Result of [`resect_images_batch`], aligned with the requested image list.
/// Skipped or failed images carry the identity pose and `accepted = false`.
#[derive(Clone, Debug)]
pub struct BatchResection {
    /// World-to-camera rotations (WXYZ), one per requested image.
    pub quaternions_wxyz: Vec<[f64; 4]>,
    /// World-to-camera translations, one per requested image.
    pub translations: Vec<[f64; 3]>,
    /// All-observation inlier fraction at the 3 px final-inlier bound, one
    /// per requested image.
    pub inlier_fractions: Vec<f64>,
    /// Whether the image cleared `accept_gate`.
    pub accepted: Vec<bool>,
}

/// Pose-only resection of many images against fixed structure, each image
/// independent (no adjustment, no cross-image coupling), parallelized across
/// images. See `specs/core/reconstruction-growth.md`.
///
/// For each image of `image_list`: gather its observations of clusters with a
/// finite `points` row; below `min_obs` skip. Estimate by RANSAC P3P polished
/// by trimmed pose-only refinement on the consensus subset; when the minimal
/// estimate fails, fall back to trimmed refinement initialized from the poses
/// of the image's most-covisible registered neighbours
/// (`posed_quaternions_wxyz` / `posed_translations` / `posed_indexes`; the
/// fallback is unavailable when they are empty). Score by the all-observation
/// inlier fraction at 3 px; accept at or above `accept_gate`.
///
/// Each image's RANSAC is seeded as a pure function of `(seed, image index)`,
/// so the parallel execution is deterministic and a one-image call matches
/// its batch row bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn resect_images_batch(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    camera: &CameraIntrinsics,
    points: &[[f64; 3]],
    image_list: &[u32],
    posed_quaternions_wxyz: &[[f64; 4]],
    posed_translations: &[[f64; 3]],
    posed_indexes: &[u32],
    options: &ResectOptions,
) -> BatchResection {
    let n_obs = cluster_indexes.len();
    assert_eq!(
        image_indexes.len(),
        n_obs,
        "cluster_indexes and image_indexes length mismatch"
    );
    assert_eq!(
        positions_xy.len(),
        n_obs,
        "cluster_indexes and positions_xy length mismatch"
    );
    let n_posed = posed_indexes.len();
    assert_eq!(
        posed_quaternions_wxyz.len(),
        n_posed,
        "posed_quaternions_wxyz and posed_indexes length mismatch"
    );
    assert_eq!(
        posed_translations.len(),
        n_posed,
        "posed_translations and posed_indexes length mismatch"
    );
    assert!(
        cluster_indexes.windows(2).all(|w| w[0] <= w[1]),
        "cluster_indexes must be nondecreasing"
    );
    if let Some(&max_c) = cluster_indexes.iter().max() {
        assert!(
            (max_c as usize) < points.len(),
            "cluster index {max_c} out of range for {} points",
            points.len()
        );
    }

    let n_img = image_indexes
        .iter()
        .chain(image_list)
        .chain(posed_indexes)
        .map(|&i| i as usize + 1)
        .max()
        .unwrap_or(0);
    let n_cl = points.len();

    let mut image_obs: Vec<Vec<usize>> = vec![Vec::new(); n_img];
    for (k, &i) in image_indexes.iter().enumerate() {
        image_obs[i as usize].push(k);
    }
    let covis = build_covisibility(cluster_indexes, image_indexes, n_img, n_cl);

    let mut posed = vec![false; n_img];
    let mut posed_quats = vec![UnitQuaternion::identity(); n_img];
    let mut posed_trans = vec![Vector3::zeros(); n_img];
    for (k, &i) in posed_indexes.iter().enumerate() {
        let q = posed_quaternions_wxyz[k];
        posed_quats[i as usize] =
            UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]));
        posed_trans[i as usize] = Vector3::new(
            posed_translations[k][0],
            posed_translations[k][1],
            posed_translations[k][2],
        );
        posed[i as usize] = true;
    }
    let posed_idx: Vec<u32> = (0..n_img as u32).filter(|&j| posed[j as usize]).collect();

    let results: Vec<(UnitQuaternion<f64>, Vector3<f64>, f64, bool)> = image_list
        .par_iter()
        .map(|&j| {
            let ju = j as usize;
            let mut rows = Vec::new();
            let mut uv = Vec::new();
            let mut world = Vec::new();
            for &k in &image_obs[ju] {
                let c = cluster_indexes[k] as usize;
                if points[c][0].is_finite() {
                    rows.push(k);
                    uv.push(positions_xy[k]);
                    world.push(points[c]);
                }
            }
            if rows.len() < options.min_obs {
                return (UnitQuaternion::identity(), Vector3::zeros(), 0.0, false);
            }
            let bearings: Vec<Vector3<f64>> = uv
                .iter()
                .map(|p| {
                    let d = camera.pixel_to_ray(p[0], p[1]);
                    Vector3::new(d[0], d[1], d[2])
                })
                .collect();
            let inits: Vec<(UnitQuaternion<f64>, Vector3<f64>)> = match &covis {
                Some(c) => c
                    .rank_by_covisibility(j, &posed_idx)
                    .into_iter()
                    .take(BATCH_FALLBACK_INITS)
                    .map(|n| (posed_quats[n as usize], posed_trans[n as usize]))
                    .collect(),
                None => posed_idx
                    .iter()
                    .take(BATCH_FALLBACK_INITS)
                    .map(|&n| (posed_quats[n as usize], posed_trans[n as usize]))
                    .collect(),
            };
            match resect_one(
                camera,
                &uv,
                &world,
                &bearings,
                P3P_MIN_CONSENSUS_BATCH,
                per_image_seed(options.seed, j),
                &inits,
            ) {
                Some(out) => {
                    let accepted = out.inlier_fraction >= options.accept_gate;
                    (out.rotation, out.translation, out.inlier_fraction, accepted)
                }
                None => (UnitQuaternion::identity(), Vector3::zeros(), 0.0, false),
            }
        })
        .collect();

    let mut quaternions_wxyz = Vec::with_capacity(results.len());
    let mut translations = Vec::with_capacity(results.len());
    let mut inlier_fractions = Vec::with_capacity(results.len());
    let mut accepted = Vec::with_capacity(results.len());
    for (r, t, inl, acc) in results {
        let q = r.into_inner();
        quaternions_wxyz.push([q.w, q.i, q.j, q.k]);
        translations.push([t.x, t.y, t.z]);
        inlier_fractions.push(inl);
        accepted.push(acc);
    }
    BatchResection {
        quaternions_wxyz,
        translations,
        inlier_fractions,
        accepted,
    }
}
