// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Background-floor track-cluster matcher.
//!
//! Instead of matching image pairs, this matcher clusters the whole corpus of
//! SIFT descriptors directly: each track's observations form a tight cluster in
//! descriptor space, so a single k-NN query per descriptor over a randomized
//! kd-tree forest ([`crate::features::kdforest`]) recovers candidate co-observations. Each
//! descriptor sets its own membership radius from its *background floor* — its
//! `d`-th-nearest distance marks the shell of unrelated features, and neighbours
//! within `alpha` of that floor are kept as candidates. Clusters are then
//! materialized by density-ordered seeding into a hard partition (one feature
//! per image per cluster), and a derived view expands them into the familiar
//! per-image-pair matches.
//!
//! All distances here are Euclidean L2 (square-rooted): the tuned defaults
//! (`d = 10`, `alpha = 0.8`) were fit in L2 space, while the forest reports
//! squared L2 internally.
//!
//! See `specs/core/track-cluster-matching.md` for the algorithm's design and
//! empirical justification.

#[cfg(test)]
mod tests;

use std::borrow::Cow;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

use crate::features::kdforest::{KdForestParams, KdForestU8};

/// Env-gated stage timing for the matcher. Enabled by setting
/// `SFMTOOL_CLUSTER_TIMING`; one cached bool check plus a few `Instant::now()`
/// per call otherwise. When on, [`background_floor_clusters`] and
/// [`clusters_to_pair_matches`] each print one `CLUSTER_TIMING ...` line to
/// stderr with per-stage wall-clock milliseconds (the SIFT `SIFT_TIMING`
/// precedent).
static CLUSTER_TIMING: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("SFMTOOL_CLUSTER_TIMING").is_some());

/// Tuning for the background-floor matcher. `Default` is the production config.
/// The k-NN query width is derived, not configured: `d + 1` (self + the `d`
/// nearest others), exactly enough that the background rank is the last column.
#[derive(Clone, Debug)]
pub struct BackgroundFloorParams {
    /// Background rank: the `d`-th-nearest distance is the background floor
    /// `B_i = dist[i, d]`. Default 10.
    pub d: usize,
    /// Radius multiplier: keep neighbours within `alpha * B_i`. Default 0.8.
    pub alpha: f32,
    /// Record a cluster only if it spans at least this many images. Default 2.
    pub min_size: usize,
    /// Index build + per-query search budget. Default `KdForestParams::accurate()`.
    pub forest: KdForestParams,
}

impl Default for BackgroundFloorParams {
    fn default() -> Self {
        Self {
            d: 10,
            alpha: 0.8,
            min_size: 2,
            forest: KdForestParams::accurate(),
        }
    }
}

/// Materialized track clusters — the matcher's primary output. CSR layout:
/// cluster `c` owns members `cluster_starts[c] .. cluster_starts[c+1]`. Within a
/// cluster, members are sorted by image index, and a cluster holds at most one
/// feature per image (so member count == image span). Clusters are disjoint (a
/// hard partition of the participating features).
#[derive(Debug)]
pub struct Clusters {
    /// `(C + 1,)` CSR offsets into the member arrays.
    pub cluster_starts: Array1<u32>,
    /// `(M,)` member image index, aligned with `member_features`.
    pub member_images: Array1<u32>,
    /// `(M,)` member feature index (row in that image's `.sift` file).
    pub member_features: Array1<u32>,
}

/// Cross-image matches, in the parallel-array form the `.matches` writer wants.
#[derive(Debug)]
pub struct PairMatches {
    /// `(P, 2)` image-index pairs, each `[i, j]` with `i < j`, sorted ascending
    /// by `(i, j)`.
    pub image_index_pairs: Array2<u32>,
    /// `(P,)` number of matches in each pair; `sum == M`. Aligned to
    /// `image_index_pairs`.
    pub match_counts: Array1<u32>,
    /// `(M, 2)` feature-index pairs `[feat_i, feat_j]`, grouped by pair in the
    /// same order as `image_index_pairs`. `feat_i` indexes image `i`'s `.sift`
    /// rows, `feat_j` indexes image `j`'s.
    pub match_feature_indexes: Array2<u32>,
    /// `(M,)` Euclidean L2 descriptor distance per match, aligned to
    /// `match_feature_indexes`.
    pub match_descriptor_distances: Array1<f32>,
}

/// Errors from the background-floor matcher's input validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterMatchError {
    /// The descriptor corpus has no rows.
    EmptyCorpus,
    /// The corpus has too few descriptors for the background rank to exist.
    CorpusSmallerThanFloor { n: usize, d: usize },
    /// `image_starts` is not a valid CSR offset array over the corpus.
    BadOffsets { n: usize },
}

impl std::fmt::Display for ClusterMatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCorpus => write!(f, "descriptor corpus is empty"),
            Self::CorpusSmallerThanFloor { n, d } => write!(
                f,
                "corpus has {n} descriptors; need more than d ({d}) for the floor"
            ),
            Self::BadOffsets { n } => write!(
                f,
                "image_starts must be non-decreasing, start at 0, and end at N ({n})"
            ),
        }
    }
}

impl std::error::Error for ClusterMatchError {}

/// Euclidean L2 distance between two `u8` descriptor rows (integer squared-L2
/// accumulation, `sqrt` only when reporting — like the forest itself).
fn l2_distance(a: ArrayView1<'_, u8>, b: ArrayView1<'_, u8>) -> f32 {
    let mut acc: i64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = x as i64 - y as i64;
        acc += diff * diff;
    }
    (acc as f64).sqrt() as f32
}

/// Background-floor track-cluster matcher: materialize the clusters.
///
/// `descriptors` is the `(N, D)` corpus of every image's SIFT descriptors,
/// concatenated image by image (uint8, D = 128). `image_starts` is a CSR-style
/// offset array of length `n_images + 1`: image `i` owns rows
/// `image_starts[i] .. image_starts[i+1]`, and row `r` of that image has
/// feature index `r - image_starts[i]`. Returns the materialized clusters —
/// the primary artefact.
pub fn background_floor_clusters(
    descriptors: ArrayView2<'_, u8>,
    image_starts: &[u32],
    params: &BackgroundFloorParams,
) -> Result<Clusters, ClusterMatchError> {
    let n = descriptors.nrows();
    let dim = descriptors.ncols();
    if n == 0 {
        return Err(ClusterMatchError::EmptyCorpus);
    }
    if n <= params.d {
        return Err(ClusterMatchError::CorpusSmallerThanFloor { n, d: params.d });
    }
    let offsets_valid = image_starts.len() >= 2
        && image_starts[0] == 0
        && image_starts.windows(2).all(|w| w[0] <= w[1])
        && *image_starts.last().unwrap() as usize == n;
    if !offsets_valid {
        return Err(ClusterMatchError::BadOffsets { n });
    }
    let n_images = image_starts.len() - 1;

    // Row -> owning image.
    let mut image_of = vec![0u32; n];
    for img in 0..n_images {
        let lo = image_starts[img] as usize;
        let hi = image_starts[img + 1] as usize;
        image_of[lo..hi].fill(img as u32);
    }

    // The query width is derived, not configured: the background rank `d` is
    // the last column of a `d + 1`-wide query (self + the `d` nearest others).
    let k = params.d + 1;

    let corpus: Cow<'_, [u8]> = match descriptors.as_slice() {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(descriptors.iter().copied().collect()),
    };

    let timing = *CLUSTER_TIMING;
    let t = std::time::Instant::now();
    let forest = KdForestU8::build(&corpus, n, dim, params.forest);
    let t_build = t.elapsed();
    let t = std::time::Instant::now();
    // Self-join batch: process the queries in the forest's descriptor-space
    // locality order so consecutive queries hit cache-resident point rows
    // (identical results in identical positions — only the schedule changes).
    let (idx, dist_sq) = forest.search_batch_with_distances_ordered(
        &corpus,
        n,
        k,
        params.forest.max_leaf_checks,
        None,
        forest.locality_order(),
    );
    let t_query = t.elapsed();
    let t = std::time::Instant::now();

    // Per-row within-radius cross-image candidates, with the L2 distance kept
    // alongside for the per-image nearest resolution below. The stride is the
    // full query width `k`: with `alpha >= 1` the floor neighbour itself can
    // pass the radius test, and if the forest misses the self-hit every one of
    // the `k` columns can be a kept candidate, so `d` slots would not suffice.
    let d = params.d;
    let alpha = params.alpha;
    let mut cand = vec![u32::MAX; n * k];
    let mut cand_dist = vec![f32::INFINITY; n * k];
    let mut cand_count = vec![0u32; n];
    cand.par_chunks_mut(k)
        .zip(cand_dist.par_chunks_mut(k))
        .zip(cand_count.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((crow, drow), ccount))| {
            // An infinite floor (forest miss at the background rank) keeps
            // nothing — the radius test below never passes.
            let radius = alpha * dist_sq[i * k + d].sqrt();
            let mut m = 0;
            for c in 0..k {
                let j = idx[i * k + c];
                if j == u32::MAX || j as usize == i || image_of[j as usize] == image_of[i] {
                    continue;
                }
                let dist = dist_sq[i * k + c].sqrt();
                if dist <= radius {
                    crow[m] = j;
                    drow[m] = dist;
                    m += 1;
                }
            }
            *ccount = m as u32;
        });

    let t_radius = t.elapsed();
    let t = std::time::Instant::now();

    // Density-ordered seeding: walk rows densest-first (row index breaks ties,
    // so the order — and the result — is deterministic), claiming members into
    // a hard partition.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.par_sort_unstable_by(|&a, &b| {
        cand_count[b as usize]
            .cmp(&cand_count[a as usize])
            .then(a.cmp(&b))
    });

    let mut claimed = vec![false; n];
    let mut cluster_starts: Vec<u32> = vec![0];
    let mut member_images: Vec<u32> = Vec::new();
    let mut member_features: Vec<u32> = Vec::new();
    // Scratch: (image, distance to seed, row) — at most one entry per image.
    let mut members: Vec<(u32, f32, u32)> = Vec::with_capacity(d + 1);

    for &s in &order {
        let si = s as usize;
        if claimed[si] {
            continue;
        }
        members.clear();
        members.push((image_of[si], 0.0, s));
        let count = cand_count[si] as usize;
        for c in 0..count {
            let j = cand[si * k + c];
            if claimed[j as usize] {
                continue;
            }
            let dist = cand_dist[si * k + c];
            let image = image_of[j as usize];
            // One feature per image: keep the nearest to the seed, breaking
            // distance ties by the smaller row index.
            match members.iter_mut().find(|m| m.0 == image) {
                Some(existing) => {
                    if (dist, j) < (existing.1, existing.2) {
                        existing.1 = dist;
                        existing.2 = j;
                    }
                }
                None => members.push((image, dist, j)),
            }
        }

        if members.len() >= params.min_size {
            members.sort_unstable_by_key(|m| m.0);
            for &(image, _, row) in &members {
                claimed[row as usize] = true;
                member_images.push(image);
                member_features.push(row - image_starts[image as usize]);
            }
            cluster_starts.push(member_images.len() as u32);
        } else {
            claimed[si] = true;
        }
    }

    if timing {
        eprintln!(
            "CLUSTER_TIMING floor build_ms={:.1} query_ms={:.1} radius_ms={:.1} seed_ms={:.1} \
             n={} clusters={}",
            t_build.as_secs_f64() * 1e3,
            t_query.as_secs_f64() * 1e3,
            t_radius.as_secs_f64() * 1e3,
            t.elapsed().as_secs_f64() * 1e3,
            n,
            cluster_starts.len() - 1,
        );
    }

    Ok(Clusters {
        cluster_starts: Array1::from_vec(cluster_starts),
        member_images: Array1::from_vec(member_images),
        member_features: Array1::from_vec(member_features),
    })
}

/// Derived view: expand clusters into one-to-one-per-image-pair cross-image
/// matches (each cluster's C(m,2) pairs, bucketed by image pair). `descriptors`
/// and `image_starts` must be the same arrays the clusters were built from;
/// they supply the L2 match distances.
pub fn clusters_to_pair_matches(
    clusters: &Clusters,
    descriptors: ArrayView2<'_, u8>,
    image_starts: &[u32],
) -> PairMatches {
    let timing = *CLUSTER_TIMING;
    let t0 = std::time::Instant::now();
    let starts = clusters.cluster_starts.as_slice().unwrap();
    let images = clusters.member_images.as_slice().unwrap();
    let features = clusters.member_features.as_slice().unwrap();
    let n_clusters = starts.len().saturating_sub(1);

    // Each cluster's members are sorted by image, so (a, b) with a < b already
    // has img_lo < img_hi.
    let mut edges: Vec<(u32, u32, u32, u32, f32)> = (0..n_clusters)
        .into_par_iter()
        .flat_map_iter(|c| {
            let lo = starts[c] as usize;
            let hi = starts[c + 1] as usize;
            (lo..hi).flat_map(move |a| {
                ((a + 1)..hi).map(move |b| {
                    let row_a = (image_starts[images[a] as usize] + features[a]) as usize;
                    let row_b = (image_starts[images[b] as usize] + features[b]) as usize;
                    let dist = l2_distance(descriptors.row(row_a), descriptors.row(row_b));
                    (images[a], images[b], features[a], features[b], dist)
                })
            })
        })
        .collect();

    edges.par_sort_unstable_by(|x, y| (x.0, x.1, x.2, x.3).cmp(&(y.0, y.1, y.2, y.3)));

    let mut image_index_pairs: Vec<u32> = Vec::new();
    let mut match_counts: Vec<u32> = Vec::new();
    let mut match_feature_indexes: Vec<u32> = Vec::with_capacity(edges.len() * 2);
    let mut match_descriptor_distances: Vec<f32> = Vec::with_capacity(edges.len());

    for &(img_lo, img_hi, feat_lo, feat_hi, dist) in &edges {
        let is_new_pair = match image_index_pairs.rchunks(2).next() {
            Some(last) => last != [img_lo, img_hi],
            None => true,
        };
        if is_new_pair {
            image_index_pairs.extend([img_lo, img_hi]);
            match_counts.push(0);
        }
        *match_counts.last_mut().unwrap() += 1;
        match_feature_indexes.extend([feat_lo, feat_hi]);
        match_descriptor_distances.push(dist);
    }

    let pair_count = match_counts.len();
    let match_count = match_descriptor_distances.len();
    if timing {
        eprintln!(
            "CLUSTER_TIMING pairs total_ms={:.1} pairs={} matches={}",
            t0.elapsed().as_secs_f64() * 1e3,
            pair_count,
            match_count,
        );
    }
    PairMatches {
        image_index_pairs: Array2::from_shape_vec((pair_count, 2), image_index_pairs).unwrap(),
        match_counts: Array1::from_vec(match_counts),
        match_feature_indexes: Array2::from_shape_vec((match_count, 2), match_feature_indexes)
            .unwrap(),
        match_descriptor_distances: Array1::from_vec(match_descriptor_distances),
    }
}
