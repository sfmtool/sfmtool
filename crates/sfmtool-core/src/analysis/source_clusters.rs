// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Which clusters of a selection a member's admission never held, banded by
//! feature radius.
//!
//! A member is drawn from a cluster selection and holds a subset of its
//! clusters. The identity between a member observation and a selection member
//! is EXACT and needs no geometry: both carry the image index and the feature
//! index, so a member row is a selection row by `(image, feature)`. This module
//! runs that join, names the clusters the member left behind that at least two
//! of its own frames still see, reads each one's feature radius, and assigns it
//! to a band of radius measured in units of the member's own admission floor.
//!
//! The join is a sorted-key binary search over the selection's rows, not a
//! hash: the keys are `(image, feature)` packed into one integer, sorted with
//! ties in row order, and a member row takes the FIRST selection row that
//! carries its key.
//!
//! See `specs/core/analysis/source-clusters.md` for the design.

use rayon::prelude::*;

/// The cluster selection the member was drawn from, as flat arrays.
#[derive(Debug, Clone, Copy)]
pub struct SourceSelection<'a> {
    /// `n_cluster + 1` CSR boundaries into the member arrays below.
    pub cluster_starts: &'a [u32],
    /// `n_member` image index per selection row.
    pub member_images: &'a [u32],
    /// `n_member` feature index per selection row.
    pub member_features: &'a [u32],
    /// `n_member * 6`, each row's absolute 2x3 affine in row-major order: the
    /// leading 2x2 is the feature's shape and the last column its pixel.
    pub member_affines: &'a [f64],
    /// The refine radius the selection's shapes are expressed against.
    pub refine_radius: f64,
    /// How many images the selection's table names.
    pub n_images: usize,
}

/// The member's own observation identities.
#[derive(Debug, Clone, Copy)]
pub struct MemberIdentity<'a> {
    /// `n_obs` image index per member observation.
    pub obs_image: &'a [u32],
    /// `n_obs` feature index per member observation.
    pub obs_feature: &'a [u32],
}

/// What the join found.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceClusters {
    /// Clusters the selection holds.
    pub n_file_clusters: usize,
    /// Member rows that matched a selection row.
    pub n_rows_matched: usize,
    /// Radius of each cluster the member's admission holds, in cluster order.
    pub admission_radius: Vec<f64>,
    /// The smallest of those, the unit the bands are measured in. `NaN` where
    /// the admission is empty.
    pub admission_floor_px: f64,
    /// Cluster ids at least two of the member's frames see that its admission
    /// never held, ascending.
    pub candidates: Vec<u32>,
    /// Each candidate's radius, in the same order.
    pub candidate_radius: Vec<f64>,
    /// Each candidate's band index, or `-1` past the last band.
    pub candidate_band: Vec<i64>,
    /// Cluster id of each selected observation, in selection row order.
    pub obs_cluster: Vec<u32>,
    /// Image index of each selected observation.
    pub obs_image: Vec<u32>,
    /// Feature index of each selected observation.
    pub obs_feature: Vec<u32>,
    /// `n_selected * 2` pixels of the selected observations.
    pub obs_uv: Vec<f64>,
    /// `n_selected * 4` row-major 2x2 shapes of the selected observations.
    pub obs_shape: Vec<f64>,
}

/// Every cluster the member's frames see that its admission never held, with
/// each one's radius and band.
///
/// `frames` are the member's placed image indices. `band_edges` runs DECREASING
/// in units of the admission floor: `band_edges[k]` is band `k`'s upper bound
/// and `band_edges[k + 1]` its lower one, half open, so band `k` holds the radii
/// in `[floor * band_edges[k + 1], floor * band_edges[k])`. A radius under the
/// last edge falls in no band and reads `-1`.
pub fn source_clusters(
    sel: SourceSelection<'_>,
    member: MemberIdentity<'_>,
    frames: &[u32],
    band_edges: &[f64],
) -> SourceClusters {
    let n_member = sel.member_images.len();
    assert_eq!(
        sel.member_features.len(),
        n_member,
        "member_images/member_features mismatch"
    );
    assert_eq!(
        sel.member_affines.len(),
        n_member * 6,
        "member_affines must be n_member * 6"
    );
    assert_eq!(
        member.obs_image.len(),
        member.obs_feature.len(),
        "obs_image/obs_feature mismatch"
    );
    let n_cl = sel.cluster_starts.len().saturating_sub(1);

    let row_cluster = cluster_of_row(sel.cluster_starts, n_member);
    let row_radius = radius_of_row(sel.member_affines, sel.refine_radius);

    // Cluster radius: the widest of its own members', so "radius" means here
    // what it meant when the admission was drawn.
    let mut cluster_radius = vec![0.0f64; n_cl];
    for (r, &c) in row_cluster.iter().enumerate() {
        let slot = &mut cluster_radius[c as usize];
        if row_radius[r] > *slot {
            *slot = row_radius[r];
        }
    }

    let (admitted, n_rows_matched) = admission_mask(sel, member, &row_cluster, n_cl);

    let mut on_frame = vec![false; sel.n_images];
    for &f in frames {
        if (f as usize) < sel.n_images {
            on_frame[f as usize] = true;
        }
    }
    let keep_row: Vec<bool> = sel
        .member_images
        .par_iter()
        .map(|&i| (i as usize) < sel.n_images && on_frame[i as usize])
        .collect();

    let mut seen = vec![0u32; n_cl];
    for (r, &c) in row_cluster.iter().enumerate() {
        if keep_row[r] {
            seen[c as usize] += 1;
        }
    }

    // A CANDIDATE is a cluster at least two placed frames see that the
    // admission never held: one frame states a bearing and no depth.
    let candidates: Vec<u32> = (0..n_cl)
        .filter(|&c| seen[c] >= 2 && !admitted[c])
        .map(|c| c as u32)
        .collect();
    let candidate_radius: Vec<f64> = candidates
        .iter()
        .map(|&c| cluster_radius[c as usize])
        .collect();

    let admission_radius: Vec<f64> = (0..n_cl)
        .filter(|&c| admitted[c])
        .map(|c| cluster_radius[c])
        .collect();
    let admission_floor_px = admission_radius
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let admission_floor_px = if admission_radius.is_empty() {
        f64::NAN
    } else {
        admission_floor_px
    };

    let candidate_band = assign_bands(&candidate_radius, admission_floor_px, band_edges);

    let mut is_candidate = vec![false; n_cl];
    for &c in &candidates {
        is_candidate[c as usize] = true;
    }
    let selected: Vec<usize> = (0..n_member)
        .filter(|&r| keep_row[r] && is_candidate[row_cluster[r] as usize])
        .collect();

    let mut obs_cluster = Vec::with_capacity(selected.len());
    let mut obs_image = Vec::with_capacity(selected.len());
    let mut obs_feature = Vec::with_capacity(selected.len());
    let mut obs_uv = Vec::with_capacity(selected.len() * 2);
    let mut obs_shape = Vec::with_capacity(selected.len() * 4);
    for &r in &selected {
        obs_cluster.push(row_cluster[r]);
        obs_image.push(sel.member_images[r]);
        obs_feature.push(sel.member_features[r]);
        let a = &sel.member_affines[r * 6..r * 6 + 6];
        obs_uv.push(a[2]);
        obs_uv.push(a[5]);
        obs_shape.extend_from_slice(&[a[0], a[1], a[3], a[4]]);
    }

    SourceClusters {
        n_file_clusters: n_cl,
        n_rows_matched,
        admission_radius,
        admission_floor_px,
        candidates,
        candidate_radius,
        candidate_band,
        obs_cluster,
        obs_image,
        obs_feature,
        obs_uv,
        obs_shape,
    }
}

/// The band index of every radius, or `-1` past the last edge.
///
/// Half open: band `k` holds the radii in `[floor * edges[k + 1],
/// floor * edges[k])`.
pub fn assign_bands(radius: &[f64], floor: f64, edges: &[f64]) -> Vec<i64> {
    radius
        .par_iter()
        .map(|&r| {
            let x = r / floor;
            let mut band = -1i64;
            for k in 0..edges.len().saturating_sub(1) {
                if x < edges[k] && x >= edges[k + 1] {
                    band = k as i64;
                }
            }
            band
        })
        .collect()
}

/// The cluster each selection row belongs to, from the CSR boundaries.
fn cluster_of_row(cluster_starts: &[u32], n_member: usize) -> Vec<u32> {
    let mut out = vec![0u32; n_member];
    for c in 0..cluster_starts.len().saturating_sub(1) {
        let lo = cluster_starts[c] as usize;
        let hi = cluster_starts[c + 1] as usize;
        for slot in out.iter_mut().take(hi.min(n_member)).skip(lo.min(n_member)) {
            *slot = c as u32;
        }
    }
    out
}

/// Each selection row's feature radius: half the refine radius times the sum of
/// the stored affine's two column norms, which is the refine radius times their
/// mean.
fn radius_of_row(affines: &[f64], refine_radius: f64) -> Vec<f64> {
    let half = 0.5 * refine_radius;
    affines
        .par_chunks_exact(6)
        .map(|a| {
            let c0 = (a[0] * a[0] + a[3] * a[3]).sqrt();
            let c1 = (a[1] * a[1] + a[4] * a[4]).sqrt();
            half * (c0 + c1)
        })
        .collect()
}

/// Which clusters the member's admission already holds, and how many member
/// rows found a selection row at all.
///
/// The key is `(image << 32) | feature`, sorted with ties in selection row
/// order; a member row takes the FIRST selection row carrying its key, and that
/// row's cluster is admitted.
fn admission_mask(
    sel: SourceSelection<'_>,
    member: MemberIdentity<'_>,
    row_cluster: &[u32],
    n_cl: usize,
) -> (Vec<bool>, usize) {
    let n_member = sel.member_images.len();
    let mut order: Vec<u32> = (0..n_member as u32).collect();
    let key = |r: usize| ((sel.member_images[r] as u64) << 32) | sel.member_features[r] as u64;
    order.sort_by_key(|&r| key(r as usize));
    let keys: Vec<u64> = order.iter().map(|&r| key(r as usize)).collect();

    let hits: Vec<Option<u32>> = (0..member.obs_image.len())
        .into_par_iter()
        .map(|k| {
            let want = ((member.obs_image[k] as u64) << 32) | member.obs_feature[k] as u64;
            let pos = keys.partition_point(|&x| x < want);
            if pos < keys.len() && keys[pos] == want {
                Some(row_cluster[order[pos] as usize])
            } else {
                None
            }
        })
        .collect();

    let mut admitted = vec![false; n_cl];
    let mut matched = 0usize;
    for hit in hits.into_iter().flatten() {
        admitted[hit as usize] = true;
        matched += 1;
    }
    (admitted, matched)
}

#[cfg(test)]
mod tests;
