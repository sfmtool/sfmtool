// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Observation adjacency graph: which reconstructed points are next to each
//! other on the imaged surface.
//!
//! Several per-point operations need the same structure — surfel-normal
//! estimation fits a plane through a point's image-space neighbours,
//! duplicate-point collapse looks for points whose observations coincide in
//! every shared image, aliased-track repair looks for pairs that coincide in
//! one image but sit far apart in another — and differ only in how they filter
//! and read its edges. [`build_observation_adjacency`] builds it in one batch
//! call.
//!
//! Adjacency is decided in image space, in units of a per-point radius the
//! caller supplies (the feature's detection scale, the projected extent of the
//! point's patch, …), so "next to" scales with the feature and not with the
//! scene. Two points are adjacent when their keypoint separation lands in the
//! annulus `[a_lo · r_pq, b_max · r_pq]` (with `r_pq = min(radius_p,
//! radius_q)`) in a majority of the images that see both, the pair shares
//! enough images, and the pair's ranges from those cameras agree — the range
//! test is what separates "next to each other on the surface" from "one behind
//! the other along the viewing ray".
//!
//! See `specs/core/observation-adjacency-graph.md` for the design.

use std::cmp::Ordering;

use rayon::prelude::*;

use crate::numeric::median_in_place;
use crate::spatial::PointCloud2;

/// Guard against a division by zero when a pair radius or a mean range
/// underflows; both quantities are strictly positive in any real input.
const EPS_RADIUS: f64 = 1e-9;
const EPS_RANGE: f64 = 1e-12;

/// Tuning for [`build_observation_adjacency`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservationAdjacencyParams {
    /// Annulus inner edge in pair-radius units. `0.0` admits fully-overlapping
    /// observations (the duplicate-collapse regime).
    pub a_lo: f64,
    /// Annulus outer edge in pair-radius units.
    pub b_max: f64,
    /// A pair must be observed by at least this many images.
    pub min_shared_images: u32,
    /// Fraction of the shared images whose separation must land in the annulus.
    pub majority: f64,
    /// Gate on the median relative range difference. [`f64::INFINITY`] disables
    /// the range vet (the aliased-pair regime, which needs to *see*
    /// range-inconsistent neighbours rather than filter them).
    pub range_tol: f64,
}

impl Default for ObservationAdjacencyParams {
    /// The spec's defaults. `b_max` has no spec default — it is the caller's
    /// statement of how far "next to" reaches — and is 10 radii here, the
    /// value the criterion was calibrated at.
    fn default() -> Self {
        Self {
            a_lo: 1.0,
            b_max: 10.0,
            min_shared_images: 2,
            majority: 0.5,
            range_tol: 0.05,
        }
    }
}

/// Symmetric CSR adjacency over points, with per-directed-edge statistics
/// parallel to [`Self::neighbours`].
///
/// The neighbours of point `p` are `neighbours[offsets[p]..offsets[p + 1]]`,
/// sorted by `(separation_med, neighbour index)`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObservationAdjacency {
    /// CSR row boundaries; length `n_points + 1`.
    pub offsets: Vec<u32>,
    /// Neighbour point indices.
    pub neighbours: Vec<u32>,
    /// Median keypoint separation over the shared images where the pair landed
    /// in the annulus, in pair-radius units.
    pub separation_med: Vec<f32>,
    /// Minimum of the same population.
    pub separation_min: Vec<f32>,
    /// Maximum of the same population.
    pub separation_max: Vec<f32>,
    /// Number of images observing both endpoints.
    pub shared_images: Vec<u32>,
    /// Number of those images where the separation landed in the annulus.
    pub annulus_hits: Vec<u32>,
    /// Median relative range difference over the shared images.
    pub range_mismatch: Vec<f32>,
}

impl ObservationAdjacency {
    /// An edge-free graph over `n_points` points.
    fn empty(n_points: usize) -> Self {
        Self {
            offsets: vec![0; n_points + 1],
            ..Default::default()
        }
    }

    /// Number of points the graph spans.
    pub fn point_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Number of directed edges (twice the number of adjacent pairs).
    pub fn directed_edge_count(&self) -> usize {
        self.neighbours.len()
    }
}

/// A vetted pair and the statistics the vet accumulated for it.
#[derive(Clone, Copy, Debug)]
struct VettedPair {
    p: u32,
    q: u32,
    separation_med: f32,
    separation_min: f32,
    separation_max: f32,
    shared_images: u32,
    annulus_hits: u32,
    range_mismatch: f32,
}

/// One directed edge, laid out so a CSR row can be sorted in place.
#[derive(Clone, Copy, Debug)]
struct DirectedEdge {
    neighbour: u32,
    separation_med: f32,
    separation_min: f32,
    separation_max: f32,
    shared_images: u32,
    annulus_hits: u32,
    range_mismatch: f32,
}

impl DirectedEdge {
    /// A placeholder row entry, overwritten during the scatter.
    const PLACEHOLDER: Self = Self {
        neighbour: 0,
        separation_med: 0.0,
        separation_min: 0.0,
        separation_max: 0.0,
        shared_images: 0,
        annulus_hits: 0,
        range_mismatch: 0.0,
    };
}

/// Build the observation adjacency graph.
///
/// # Arguments
/// * `keypoints_xy` — per observation, the keypoint in its image.
/// * `track_point_indexes` — per observation, the point it belongs to.
/// * `track_image_indexes` — per observation, the image it was seen in.
/// * `radii_px` — per point; a non-positive radius excludes the point.
/// * `point_is_at_infinity` — per point; points at infinity take part in no
///   edges.
/// * `positions` — per point, for the range vet.
/// * `camera_centers` — per image (`-Rᵀ·t`), for the range vet.
/// * `params` — the adjacency criterion.
///
/// Observations whose point or image index is out of range are ignored, as are
/// observations of excluded points. The result is a fixed function of the
/// inputs: parallel work is merged in image (then candidate) order, so it never
/// depends on thread scheduling.
///
/// # Panics
/// If the per-observation or per-point slices disagree on length.
#[allow(clippy::too_many_arguments)]
pub fn build_observation_adjacency(
    keypoints_xy: &[[f64; 2]],
    track_point_indexes: &[u32],
    track_image_indexes: &[u32],
    radii_px: &[f32],
    point_is_at_infinity: &[bool],
    positions: &[[f64; 3]],
    camera_centers: &[[f64; 3]],
    params: &ObservationAdjacencyParams,
) -> ObservationAdjacency {
    let n_obs = keypoints_xy.len();
    assert_eq!(
        track_point_indexes.len(),
        n_obs,
        "track_point_indexes must have one entry per observation"
    );
    assert_eq!(
        track_image_indexes.len(),
        n_obs,
        "track_image_indexes must have one entry per observation"
    );
    let n_points = radii_px.len();
    assert_eq!(
        point_is_at_infinity.len(),
        n_points,
        "point_is_at_infinity must have one entry per point"
    );
    assert_eq!(
        positions.len(),
        n_points,
        "positions must have one entry per point"
    );
    let n_images = camera_centers.len();

    if n_points == 0 || n_obs == 0 || n_images == 0 {
        return ObservationAdjacency::empty(n_points);
    }

    // ── Observation bookkeeping ───────────────────────────────────────────
    // A live observation is one of a point that can take part in edges, in an
    // image that exists.
    let live: Vec<bool> = (0..n_obs)
        .map(|o| {
            let p = track_point_indexes[o] as usize;
            let i = track_image_indexes[o] as usize;
            p < n_points
                && i < n_images
                && !point_is_at_infinity[p]
                && radii_px[p] > 0.0
                && keypoints_xy[o][0].is_finite()
                && keypoints_xy[o][1].is_finite()
        })
        .collect();

    // Live observations grouped by image, and by point, both in observation
    // order (so each point's list is sorted by image only if the input is —
    // it is sorted explicitly below, which the vet's merge join relies on).
    let mut by_image: Vec<Vec<u32>> = vec![Vec::new(); n_images];
    let mut by_point: Vec<Vec<u32>> = vec![Vec::new(); n_points];
    for (o, &is_live) in live.iter().enumerate() {
        if is_live {
            by_image[track_image_indexes[o] as usize].push(o as u32);
            by_point[track_point_indexes[o] as usize].push(o as u32);
        }
    }
    for obs in by_point.iter_mut() {
        obs.sort_unstable_by_key(|&o| track_image_indexes[o as usize]);
    }

    // Range from the observing camera: a convention-free depth proxy, so the
    // range vet never has to know which way the camera frame points.
    let obs_range: Vec<f64> = (0..n_obs)
        .map(|o| {
            if !live[o] {
                return f64::NAN;
            }
            let pos = positions[track_point_indexes[o] as usize];
            let c = camera_centers[track_image_indexes[o] as usize];
            let d = [pos[0] - c[0], pos[1] - c[1], pos[2] - c[2]];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        })
        .collect();

    // ── Pass 1: candidates ────────────────────────────────────────────────
    // One ball query per image; a pair whose separation lands in the annulus in
    // at least one image becomes a candidate.
    let per_image: Vec<Vec<(u32, u32)>> = by_image
        .par_iter()
        .map(|obs| image_candidates(obs, keypoints_xy, track_point_indexes, radii_px, params))
        .collect();

    let mut candidates: Vec<(u32, u32)> = per_image.into_iter().flatten().collect();
    candidates.sort_unstable();
    candidates.dedup();
    if candidates.is_empty() {
        return ObservationAdjacency::empty(n_points);
    }

    // ── Pass 2: vet ───────────────────────────────────────────────────────
    // Every candidate is judged on ALL the images that see both endpoints.
    let vetted: Vec<VettedPair> = candidates
        .par_iter()
        .filter_map(|&(p, q)| {
            vet_pair(
                p,
                q,
                &by_point,
                keypoints_xy,
                track_image_indexes,
                radii_px,
                &obs_range,
                params,
            )
        })
        .collect();

    to_csr(&vetted, n_points)
}

/// Candidate pairs from one image: an annulus hit here makes a pair a
/// candidate, wherever else it is seen.
///
/// The ball query runs per site with that site's own radius `b_max · r_i`.
/// Since the pair threshold is `b_max · min(r_i, r_j) <= b_max · r_i`, a query
/// from either endpoint reaches any admissible partner, so only the `i < j`
/// half of each hit list needs examining.
fn image_candidates(
    obs: &[u32],
    keypoints_xy: &[[f64; 2]],
    track_point_indexes: &[u32],
    radii_px: &[f32],
    params: &ObservationAdjacencyParams,
) -> Vec<(u32, u32)> {
    if obs.len() < 2 || params.b_max.is_nan() || params.b_max < params.a_lo || params.b_max < 0.0 {
        return Vec::new();
    }

    let coords: Vec<f64> = obs
        .iter()
        .flat_map(|&o| {
            let xy = keypoints_xy[o as usize];
            [xy[0], xy[1]]
        })
        .collect();
    let radii: Vec<f64> = obs
        .iter()
        .map(|&o| radii_px[track_point_indexes[o as usize] as usize] as f64)
        .collect();

    let cloud = PointCloud2::<f64>::new(&coords, obs.len());

    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for i in 0..obs.len() {
        let site = [coords[2 * i], coords[2 * i + 1]];
        // Per-site radius, so one query at a time; the tree search dominates.
        // Widened by a hair so the annulus's outer edge lands inside the query
        // whatever boundary convention the tree uses — the exact annulus test
        // below is what decides membership.
        let (_, hits) = cloud.within_radius(&site, 1, params.b_max * radii[i] * (1.0 + 1e-9));
        for &j in &hits {
            let j = j as usize;
            if j <= i {
                continue;
            }
            let p = track_point_indexes[obs[i] as usize];
            let q = track_point_indexes[obs[j] as usize];
            if p == q {
                continue;
            }
            let r_pair = radii[i].min(radii[j]);
            let dx = coords[2 * i] - coords[2 * j];
            let dy = coords[2 * i + 1] - coords[2 * j + 1];
            let d = (dx * dx + dy * dy).sqrt();
            if d >= params.a_lo * r_pair && d <= params.b_max * r_pair {
                pairs.push((p.min(q), p.max(q)));
            }
        }
    }

    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Walk every image observing both endpoints and apply the three criteria.
///
/// Returns `None` when the pair fails support, the majority vote, or the range
/// gate.
#[allow(clippy::too_many_arguments)]
fn vet_pair(
    p: u32,
    q: u32,
    by_point: &[Vec<u32>],
    keypoints_xy: &[[f64; 2]],
    track_image_indexes: &[u32],
    radii_px: &[f32],
    obs_range: &[f64],
    params: &ObservationAdjacencyParams,
) -> Option<VettedPair> {
    let r_pair = (radii_px[p as usize] as f64)
        .min(radii_px[q as usize] as f64)
        .max(EPS_RADIUS);

    let obs_p = &by_point[p as usize];
    let obs_q = &by_point[q as usize];
    let mut separations: Vec<f64> = Vec::new();
    let mut mismatches: Vec<f64> = Vec::new();
    let mut shared_images = 0u32;

    // Merge join over the two image-sorted observation lists. A point observed
    // more than once in one image contributes its first observation there.
    let (mut a, mut b) = (0usize, 0usize);
    while a < obs_p.len() && b < obs_q.len() {
        let (oa, ob) = (obs_p[a] as usize, obs_q[b] as usize);
        let (ia, ib) = (track_image_indexes[oa], track_image_indexes[ob]);
        match ia.cmp(&ib) {
            Ordering::Less => {
                a += 1;
                continue;
            }
            Ordering::Greater => {
                b += 1;
                continue;
            }
            Ordering::Equal => {}
        }

        shared_images += 1;
        let dx = keypoints_xy[oa][0] - keypoints_xy[ob][0];
        let dy = keypoints_xy[oa][1] - keypoints_xy[ob][1];
        let ratio = (dx * dx + dy * dy).sqrt() / r_pair;
        if ratio >= params.a_lo && ratio <= params.b_max {
            separations.push(ratio);
        }
        let (ra, rb) = (obs_range[oa], obs_range[ob]);
        mismatches.push((ra - rb).abs() / (0.5 * (ra + rb)).max(EPS_RANGE));

        // Skip the rest of this image on both sides.
        while a < obs_p.len() && track_image_indexes[obs_p[a] as usize] == ia {
            a += 1;
        }
        while b < obs_q.len() && track_image_indexes[obs_q[b] as usize] == ib {
            b += 1;
        }
    }

    let annulus_hits = separations.len() as u32;
    if shared_images < params.min_shared_images
        || annulus_hits == 0
        || (annulus_hits as f64) < params.majority * shared_images as f64
    {
        return None;
    }
    // `range_tol = INFINITY` disables the gate; a NaN mismatch never passes.
    let range_mismatch = median_in_place(&mut mismatches);
    if range_mismatch.is_nan() || range_mismatch > params.range_tol {
        return None;
    }

    let separation_min = separations.iter().copied().fold(f64::INFINITY, f64::min);
    let separation_max = separations
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let separation_med = median_in_place(&mut separations);

    Some(VettedPair {
        p,
        q,
        separation_med: separation_med as f32,
        separation_min: separation_min as f32,
        separation_max: separation_max as f32,
        shared_images,
        annulus_hits,
        range_mismatch: range_mismatch as f32,
    })
}

/// Scatter the vetted pairs into symmetric CSR, each row sorted by
/// `(separation_med, neighbour index)`.
fn to_csr(vetted: &[VettedPair], n_points: usize) -> ObservationAdjacency {
    let mut offsets = vec![0u32; n_points + 1];
    for pair in vetted {
        offsets[pair.p as usize + 1] += 1;
        offsets[pair.q as usize + 1] += 1;
    }
    for i in 0..n_points {
        offsets[i + 1] += offsets[i];
    }

    let n_directed = 2 * vetted.len();
    let mut rows = vec![DirectedEdge::PLACEHOLDER; n_directed];
    let mut cursor: Vec<u32> = offsets[..n_points].to_vec();
    for pair in vetted {
        for (src, dst) in [(pair.p, pair.q), (pair.q, pair.p)] {
            let slot = &mut cursor[src as usize];
            rows[*slot as usize] = DirectedEdge {
                neighbour: dst,
                separation_med: pair.separation_med,
                separation_min: pair.separation_min,
                separation_max: pair.separation_max,
                shared_images: pair.shared_images,
                annulus_hits: pair.annulus_hits,
                range_mismatch: pair.range_mismatch,
            };
            *slot += 1;
        }
    }

    for p in 0..n_points {
        let (lo, hi) = (offsets[p] as usize, offsets[p + 1] as usize);
        rows[lo..hi].sort_unstable_by(|a, b| {
            a.separation_med
                .partial_cmp(&b.separation_med)
                .unwrap_or(Ordering::Equal)
                .then(a.neighbour.cmp(&b.neighbour))
        });
    }

    let mut out = ObservationAdjacency {
        offsets,
        neighbours: Vec::with_capacity(n_directed),
        separation_med: Vec::with_capacity(n_directed),
        separation_min: Vec::with_capacity(n_directed),
        separation_max: Vec::with_capacity(n_directed),
        shared_images: Vec::with_capacity(n_directed),
        annulus_hits: Vec::with_capacity(n_directed),
        range_mismatch: Vec::with_capacity(n_directed),
    };
    for edge in rows {
        out.neighbours.push(edge.neighbour);
        out.separation_med.push(edge.separation_med);
        out.separation_min.push(edge.separation_min);
        out.separation_max.push(edge.separation_max);
        out.shared_images.push(edge.shared_images);
        out.annulus_hits.push(edge.annulus_hits);
        out.range_mismatch.push(edge.range_mismatch);
    }
    out
}

#[cfg(test)]
mod tests;
