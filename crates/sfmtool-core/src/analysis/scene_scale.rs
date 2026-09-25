// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Scene statistics read off a reconstruction: the automatic splat size, the
//! characteristic inter-camera distance, the bounding sphere, and the length
//! scale made from the first two.
//!
//! SfM Explorer reads these off each node's points on upload
//! (`specs/gui/point-cloud-rendering.md` § "Length Scale"), and `sfm web-export`
//! reads the same ones into the scene it writes, so a page sizes its splats and
//! frustums as the viewer does.
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
//! above `NN_ISOLATION_FACTOR` times it, take the median of what is left,
//! and repeat to a fixpoint. Each pass measures isolation against the cloud's
//! own scale rather than a distance in scene units, and the tail is only ever
//! cut from above, so the sequence of medians is non-increasing and reaches a
//! fixpoint in a handful of passes (`NN_TRIM_MAX_ITERATIONS` bounds it). On
//! a tight distribution nothing is above the bar and the trim is a no-op.
//!
//! The factor of 2 is the one constant. Measured on a noisy reconstruction and
//! its hand-cleaned sibling, the plain medians differed by 3.8x while the
//! trimmed medians of the same two clouds agreed to within 1.25x: the trim
//! converges on the spacing the two clouds share.

use std::collections::HashMap;

use nalgebra::Point3;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::spatial::PointCloud3;

/// The splat size [`compute_auto_point_size`] reports for a cloud with fewer
/// than two finite points, or no nonzero nearest-neighbour distance.
pub const FALLBACK_POINT_SIZE: f32 = 0.03;

/// How many points [`compute_auto_point_size`] queries nearest neighbours for,
/// at most: a seeded random subsample when the cloud is larger.
pub const NN_SUBSAMPLE_COUNT: usize = 10_000;

/// Length scale per automatic splat size: [`length_scale`] is this multiple of
/// the splat size, or the inter-camera distance when that is smaller.
pub const LENGTH_SCALE_MULTIPLIER: f32 = 10.0;

/// The length scale of a cloud whose automatic splat size is `auto_point_size`
/// and whose cameras are `camera_nn_scale` apart: [`LENGTH_SCALE_MULTIPLIER`]
/// times the splat size, capped at the camera spacing when there is one.
pub fn length_scale(auto_point_size: f32, camera_nn_scale: Option<f32>) -> f32 {
    let point_scale = LENGTH_SCALE_MULTIPLIER * auto_point_size;
    match camera_nn_scale {
        Some(camera_scale) => point_scale.min(camera_scale),
        None => point_scale,
    }
}

/// Euclidean distance between two 3-D points, accumulated the way the KD-tree
/// accumulates its squared distances so the two agree bit for bit.
fn distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    (0..3)
        .map(|d| (a[d] - b[d]) * (a[d] - b[d]))
        .sum::<f32>()
        .sqrt()
}

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
/// random subsample of up to [`NN_SUBSAMPLE_COUNT`] points. Returns
/// `SPLAT_RADIUS_FACTOR` times the iteratively trimmed median of those distances
/// (see the module documentation) as the splat radius.
pub fn compute_auto_point_size(points: &[crate::Point3D]) -> f32 {
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

    // Build the index over every finite point (f32 for speed).
    let flat: Vec<f32> = positions.iter().flatten().copied().collect();
    let cloud = PointCloud3::<f32>::new(&flat, positions.len());

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

    // Query the two nearest points for each subsampled point: the first is the
    // point itself (or a coincident twin), so the second is the neighbour.
    let query: Vec<f32> = query_indices
        .iter()
        .flat_map(|&idx| positions[idx])
        .collect();
    let neighbors = cloud.nearest_k(&query, query_indices.len(), 2);
    let mut nn_distances: Vec<f32> = Vec::with_capacity(query_indices.len());
    for (&idx, pair) in query_indices.iter().zip(neighbors.chunks(2)) {
        if pair[1] == u32::MAX {
            continue;
        }
        let dist = distance(&positions[idx], &cloud.position(pair[1] as usize));
        if dist > 0.0 {
            nn_distances.push(dist);
        }
    }

    if nn_distances.is_empty() {
        return FALLBACK_POINT_SIZE;
    }

    nn_distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    iteratively_trimmed_median(&nn_distances) * SPLAT_RADIUS_FACTOR
}

/// Compute a characteristic inter-camera distance from nearest-neighbor distances.
///
/// Builds a KD-tree of all camera centers, queries the NN distance for each
/// camera, and returns the 90th percentile. The high percentile makes the
/// result robust to a few cameras that happen to sit on top of each other
/// (e.g. colocated rig cameras), which would otherwise pull the value to zero.
///
/// Returns `None` if there are fewer than 2 images.
pub fn compute_camera_nn_scale(images: &[crate::SfmrImage]) -> Option<f32> {
    if images.len() < 2 {
        return None;
    }

    // Exact-duplicate centers collapse to a single cloud entry, and every
    // image that shares a center reports that entry's distance. Indexing the
    // duplicates instead would make each of them report zero — the distance to
    // its twin — which the `dist.is_finite() && dist > 0.0` filter below then
    // throws away, so a colocated rig pair would contribute nothing at all
    // rather than the spacing to the nearest camera that is somewhere else.
    let mut distinct: Vec<f32> = Vec::with_capacity(images.len() * 3);
    let mut entry_of: HashMap<[u32; 3], usize> = HashMap::with_capacity(images.len());
    let mut per_image: Vec<usize> = Vec::with_capacity(images.len());
    for img in images {
        let c = img.camera_center();
        let p = [c.x as f32, c.y as f32, c.z as f32];
        let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
        let next = entry_of.len();
        let entry = *entry_of.entry(key).or_insert_with(|| {
            distinct.extend_from_slice(&p);
            next
        });
        per_image.push(entry);
    }

    // Every center is its own cloud point, so a cloud of one (a rotation-only
    // reconstruction) reports an infinite distance and drops out here.
    let cloud = PointCloud3::<f32>::new(&distinct, entry_of.len());
    let entry_nn = cloud.nearest_neighbor_distances();
    let mut nn_distances: Vec<f32> = per_image
        .iter()
        .map(|&entry| entry_nn[entry])
        .filter(|d| d.is_finite() && *d > 0.0)
        .collect();

    if nn_distances.is_empty() {
        return None;
    }

    nn_distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    Some(nn_distances[nn_distances.len() * 9 / 10])
}

/// Compute the bounding sphere (center, radius) for a set of 3D points.
///
/// Uses component-wise median for a robust center, then 80th percentile
/// distance from center as a robust radius. Handles outliers gracefully
/// since percentile-based statistics ignore extreme values.
///
/// Returns `(origin, 1.0)` if fewer than 2 points.
pub fn compute_scene_bounds(points: &[crate::Point3D]) -> (Point3<f64>, f64) {
    // Exclude points at infinity: their `position` is a unit direction, not a
    // location, and would pull the center toward the origin and distort the
    // radius (and hence the adaptive clip planes that depend on these bounds).
    let finite: Vec<&crate::Point3D> = points.iter().filter(|p| !p.is_at_infinity()).collect();
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

    (center, dists[n * 4 / 5].max(0.1))
}

#[cfg(test)]
mod tests;
