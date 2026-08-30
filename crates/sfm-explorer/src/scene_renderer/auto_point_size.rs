// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Scene statistics read off the point cloud on upload: the automatic splat
//! size, the characteristic inter-camera distance, and the bounding sphere.
//!
//! The automatic splat size wants the spacing of the coherent structure, not
//! of the whole cloud. A reconstruction carries a scattered sub-population
//! beside its surfaces (mis-triangulated points strung out along rays through
//! empty space, for one), and those points sit far from any neighbour, so
//! their nearest-neighbour distances land in a long upper tail that drags the
//! plain median of the distance set several-fold above the spacing of the
//! dense structure. Splats sized off that median swell until the surfaces
//! disappear under them.
//!
//! [`compute_auto_point_size`] therefore takes an iteratively trimmed median
//! of the nearest-neighbour distances: take the median, drop every distance
//! above [`NN_ISOLATION_FACTOR`] times it, take the median of what is left,
//! and repeat to a fixpoint. Each pass measures isolation against the cloud's
//! own scale rather than a distance in scene units, and the tail is only ever
//! cut from above, so the sequence of medians is non-increasing and reaches a
//! fixpoint in a handful of passes ([`NN_TRIM_MAX_ITERATIONS`] bounds it). On
//! a tight distribution nothing is above the bar and the trim is a no-op.
//!
//! The factor of 2 is the one constant. Measured on a noisy reconstruction and
//! its hand-cleaned sibling, the plain medians differed by 3.8x while the
//! trimmed medians of the same two clouds agreed to within 1.25x: the trim
//! converges on the spacing the two clouds share.

use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::Point3;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::gpu_types::{FALLBACK_POINT_SIZE, NN_SUBSAMPLE_COUNT};

/// A nearest-neighbour distance above this multiple of the current median is
/// an isolated point by the cloud's own scale, and leaves the set the next
/// median is taken over. See the module documentation.
const NN_ISOLATION_FACTOR: f32 = 2.0;

/// Splat radius per trimmed-median NN distance.  Chosen for visual parity
/// with the previous release on a tight distribution (it drew `1.2 * p40`,
/// and p40 is within a few percent of the median there), so the trim is the
/// only behavioural change on clean clouds.
const SPLAT_RADIUS_FACTOR: f32 = 1.1;

/// Iteration bound on the trim. The median is non-increasing across passes and
/// settles in a handful of them; the bound only caps the pathological case.
const NN_TRIM_MAX_ITERATIONS: usize = 8;

/// Seed for the nearest-neighbour subsample, so a given cloud reports the same
/// splat size on every load.
const NN_SUBSAMPLE_SEED: u64 = 0x0a17_5123;

/// The median of `sorted` after iteratively dropping the isolated distances.
///
/// `sorted` must be sorted ascending and non-empty. Every pass takes the
/// median of the surviving prefix, sets the bar at [`NN_ISOLATION_FACTOR`]
/// times it, and keeps the distances at or below the bar; since the median
/// itself is always kept, the surviving prefix never empties. The loop stops
/// as soon as a pass removes nothing.
fn iteratively_trimmed_median(sorted: &[f32]) -> f32 {
    let mut len = sorted.len();
    let mut median = sorted[len / 2];
    for _ in 0..NN_TRIM_MAX_ITERATIONS {
        let cutoff = median * NN_ISOLATION_FACTOR;
        let kept = sorted[..len].partition_point(|&d| d <= cutoff);
        if kept == len {
            break;
        }
        len = kept;
        median = sorted[len / 2];
    }
    median
}

/// Compute an automatic point size from nearest-neighbor distances.
///
/// Builds a KD-tree of all points, then queries NN distances for a seeded
/// random subsample of up to `NN_SUBSAMPLE_COUNT` points. Returns [`SPLAT_RADIUS_FACTOR`] times the
/// iteratively trimmed median of those distances (see the module
/// documentation) as the splat radius.
pub(super) fn compute_auto_point_size(points: &[sfmtool_core::Point3D]) -> f32 {
    // Points at infinity store a unit direction, not a location, so they would
    // cluster on the unit sphere and skew NN distances: exclude them.
    let positions: Vec<[f32; 3]> = points
        .iter()
        .filter(|p| !p.is_at_infinity())
        .map(|p| {
            [
                p.position.x as f32,
                p.position.y as f32,
                p.position.z as f32,
            ]
        })
        .collect();
    if positions.len() < 2 {
        return FALLBACK_POINT_SIZE;
    }

    // Build KD-tree from finite points (f32 for speed)
    let mut tree: KdTree<f32, 3> = KdTree::with_capacity(positions.len());
    for (i, p) in positions.iter().enumerate() {
        tree.add(p, i as u64);
    }

    // Subsample indices for NN queries
    let mut rng = StdRng::seed_from_u64(NN_SUBSAMPLE_SEED);
    let query_indices: Vec<usize> = if positions.len() <= NN_SUBSAMPLE_COUNT {
        (0..positions.len()).collect()
    } else {
        let mut indices: Vec<usize> = (0..positions.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(NN_SUBSAMPLE_COUNT);
        indices
    };

    // Query nearest neighbor for each subsampled point (k=2: self + nearest)
    let mut nn_distances: Vec<f32> = Vec::with_capacity(query_indices.len());
    for &idx in &query_indices {
        let neighbors = tree.nearest_n::<SquaredEuclidean>(&positions[idx], 2);
        // The first result is the point itself (distance 0); take the second
        if neighbors.len() >= 2 {
            let dist = neighbors[1].distance.sqrt();
            if dist > 0.0 {
                nn_distances.push(dist);
            }
        }
    }

    if nn_distances.is_empty() {
        return FALLBACK_POINT_SIZE;
    }

    nn_distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let plain_median = nn_distances[nn_distances.len() / 2];
    let trimmed_median = iteratively_trimmed_median(&nn_distances);

    let auto_size = trimmed_median * SPLAT_RADIUS_FACTOR;
    log::info!(
        "Auto point size: {:.4} (trimmed median NN dist: {:.4}, plain median: {:.4}, from {} queries over {} finite points)",
        auto_size,
        trimmed_median,
        plain_median,
        nn_distances.len(),
        positions.len()
    );

    auto_size
}

/// Compute a characteristic inter-camera distance from nearest-neighbor distances.
///
/// Builds a KD-tree of all camera centers, queries the NN distance for each
/// camera, and returns the 90th percentile. The high percentile makes the
/// result robust to a few cameras that happen to sit on top of each other
/// (e.g. colocated rig cameras), which would otherwise pull the value to zero.
///
/// Returns `None` if there are fewer than 2 images.
pub(super) fn compute_camera_nn_scale(images: &[sfmtool_core::SfmrImage]) -> Option<f32> {
    if images.len() < 2 {
        return None;
    }

    // Exact-duplicate centers collapse to a single tree entry: kiddo v5's
    // fixed-size leaf buckets panic once more than 32 items share identical
    // coordinates (a rotation-only reconstruction stores every camera at the
    // origin), and duplicates could only contribute the zero distances the
    // `dist > 0.0` filter below discards.
    let mut tree: KdTree<f32, 3> = KdTree::with_capacity(images.len());
    let mut seen = std::collections::HashSet::with_capacity(images.len());
    for (i, img) in images.iter().enumerate() {
        let c = img.camera_center();
        let p = [c.x as f32, c.y as f32, c.z as f32];
        if seen.insert([p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]) {
            tree.add(&p, i as u64);
        }
    }

    let mut nn_distances: Vec<f32> = Vec::with_capacity(images.len());
    for img in images {
        let c = img.camera_center();
        let query = [c.x as f32, c.y as f32, c.z as f32];
        let neighbors = tree.nearest_n::<SquaredEuclidean>(&query, 2);
        if neighbors.len() >= 2 {
            let dist = neighbors[1].distance.sqrt();
            if dist > 0.0 {
                nn_distances.push(dist);
            }
        }
    }

    if nn_distances.is_empty() {
        return None;
    }

    nn_distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let p90 = nn_distances[nn_distances.len() * 9 / 10];

    log::info!(
        "Camera NN scale: {:.4} (p90 of {} NN distances from {} cameras)",
        p90,
        nn_distances.len(),
        images.len()
    );

    Some(p90)
}

/// Compute the bounding sphere (center, radius) for a set of 3D points.
///
/// Uses component-wise median for a robust center, then 80th percentile
/// distance from center as a robust radius. Handles outliers gracefully
/// since percentile-based statistics ignore extreme values.
///
/// Returns `(origin, 1.0)` if fewer than 2 points.
pub(super) fn compute_scene_bounds(points: &[sfmtool_core::Point3D]) -> (Point3<f64>, f64) {
    // Exclude points at infinity: their `position` is a unit direction, not a
    // location, and would pull the center toward the origin and distort the
    // radius (and hence the adaptive clip planes that depend on these bounds).
    let finite: Vec<&sfmtool_core::Point3D> =
        points.iter().filter(|p| !p.is_at_infinity()).collect();
    if finite.len() < 2 {
        return (Point3::origin(), 1.0);
    }

    // Collect coordinates
    let mut xs: Vec<f64> = finite.iter().map(|p| p.position.x).collect();
    let mut ys: Vec<f64> = finite.iter().map(|p| p.position.y).collect();
    let mut zs: Vec<f64> = finite.iter().map(|p| p.position.z).collect();

    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    ys.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    zs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n = xs.len();
    let center = Point3::new(xs[n / 2], ys[n / 2], zs[n / 2]);

    // Compute distances from center and take 80th percentile
    let mut dists: Vec<f64> = finite
        .iter()
        .map(|p| (p.position - center).norm())
        .collect();
    dists.sort_unstable_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());

    let p80 = dists[n * 4 / 5].max(0.1);

    log::info!(
        "Scene bounds: center=[{:.2}, {:.2}, {:.2}], radius={:.2} (from {} points)",
        center.x,
        center.y,
        center.z,
        p80,
        n
    );

    (center, p80)
}

#[cfg(test)]
mod tests;
