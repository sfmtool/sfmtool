// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Sparse displacement-neighborhood substrate.
//!
//! Per *realized* covisible image pair, the shared-cluster count and the
//! mean pixel displacement of the pair's shared-cluster keypoints, built in
//! one pass over the clusters — linear in observations, no dense matrix
//! anywhere. See `specs/core/geometry/pose-verification.md` (Substrate).

use std::collections::HashMap;

use super::CovisibilityError;

/// Sampled per-image-pair feature-displacement tables (row-major
/// `(num_images, num_images)`, symmetric, zero diagonal). Present only when
/// positions were supplied at construction.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DisplacementTables {
    /// Mean sampled displacement per pair; `0` where no sample landed.
    pub(super) mean: Vec<f64>,
    /// Samples behind each mean.
    pub(super) count: Vec<u32>,
}

/// Sparse displacement-neighborhood substrate: per *realized* covisible image
/// pair, the shared-cluster count and the mean pixel displacement of the
/// pair's shared-cluster keypoints. Built in one pass over the clusters —
/// each cluster emits its accepted cross-image member pairs, so under the
/// cluster matcher's span cap both time and storage are linear in
/// observations, with no dense matrix anywhere. See
/// `specs/core/geometry/pose-verification.md` (Substrate).
///
/// The shared count matches [`ClusterCovisibility::count`](super::ClusterCovisibility::count) (each cluster
/// votes at most once per pair); the mean displacement averages over *every*
/// accepted cross-image member pair of the shared clusters — exhaustive, not
/// sampled (contrast the seeded one-sample-per-cluster tables behind
/// [`ClusterCovisibility::pair_displacement`](super::ClusterCovisibility::pair_displacement)).
///
/// Serialize with [`Self::to_arrays`] / reload with [`Self::from_arrays`], so
/// one computation serves a multi-stage pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplacementNeighborhood {
    num_images: usize,
    /// CSR row offsets over the adjacency arrays, length `num_images + 1`.
    nbr_starts: Vec<usize>,
    /// Partner image per adjacency entry, ascending within each row.
    nbr_images: Vec<u32>,
    /// Shared-cluster count per adjacency entry.
    nbr_shared: Vec<u32>,
    /// Mean keypoint displacement (pixels) per adjacency entry.
    nbr_mean_disp: Vec<f64>,
}

/// Per-pair accumulator for the neighborhood build.
#[derive(Clone, Copy, Default)]
struct PairAccum {
    shared: u32,
    disp_sum: f64,
    disp_n: u32,
}

impl DisplacementNeighborhood {
    /// Build the substrate from CSR cluster arrays plus per-member positions
    /// (all parallel to `member_images`, pixel units). `member_accepted` is
    /// honored exactly as in [`ClusterCovisibility::from_clusters`](super::ClusterCovisibility::from_clusters): `None`
    /// means every member counts.
    ///
    /// Per cluster: the accepted members' deduplicated image list votes once
    /// per pair into the shared count, and every accepted cross-image member
    /// pair contributes its Euclidean position distance to the pair's mean
    /// displacement. Deterministic — no sampling.
    ///
    /// Positions are the `f32` pairs a `.matches` backbone stores; the
    /// distances and their means are `f64`.
    pub fn from_clusters(
        cluster_starts: &[u32],
        member_images: &[u32],
        member_accepted: Option<&[bool]>,
        num_images: usize,
        positions_xy: &[[f32; 2]],
    ) -> Result<Self, CovisibilityError> {
        let m = member_images.len();
        let csr_valid = !cluster_starts.is_empty()
            && cluster_starts[0] == 0
            && cluster_starts.windows(2).all(|w| w[0] <= w[1])
            && *cluster_starts.last().unwrap() as usize == m;
        if !csr_valid {
            return Err(CovisibilityError::BadClusterStarts { m });
        }
        if let Some(mask) = member_accepted {
            if mask.len() != m {
                return Err(CovisibilityError::MaskNotParallel {
                    members: m,
                    mask: mask.len(),
                });
            }
        }
        if positions_xy.len() != m {
            return Err(CovisibilityError::PositionsNotParallel {
                members: m,
                positions: positions_xy.len(),
            });
        }
        if let Some(&bad) = member_images.iter().find(|&&i| i as usize >= num_images) {
            return Err(CovisibilityError::ImageIndexOutOfRange {
                index: bad,
                num_images,
            });
        }

        let mut pairs: HashMap<(u32, u32), PairAccum> = HashMap::new();
        let mut rows: Vec<usize> = Vec::new();
        let mut span: Vec<u32> = Vec::new();
        for c in 0..cluster_starts.len() - 1 {
            let lo = cluster_starts[c] as usize;
            let hi = cluster_starts[c + 1] as usize;
            rows.clear();
            rows.extend((lo..hi).filter(|&k| member_accepted.is_none_or(|mask| mask[k])));
            // Shared-cluster votes: once per deduplicated image pair.
            span.clear();
            span.extend(rows.iter().map(|&k| member_images[k]));
            span.sort_unstable();
            span.dedup();
            for (a, &i) in span.iter().enumerate() {
                for &j in &span[a + 1..] {
                    pairs.entry((i, j)).or_default().shared += 1;
                }
            }
            // Displacement: every accepted cross-image member pair.
            for (a, &ka) in rows.iter().enumerate() {
                for &kb in &rows[a + 1..] {
                    let (ia, ib) = (member_images[ka], member_images[kb]);
                    if ia == ib {
                        continue;
                    }
                    let d = f64::hypot(
                        positions_xy[ka][0] as f64 - positions_xy[kb][0] as f64,
                        positions_xy[ka][1] as f64 - positions_xy[kb][1] as f64,
                    );
                    let e = pairs.entry((ia.min(ib), ia.max(ib))).or_default();
                    e.disp_sum += d;
                    e.disp_n += 1;
                }
            }
        }

        // Deterministic order despite the hash-map accumulator.
        let mut sorted: Vec<((u32, u32), PairAccum)> = pairs.into_iter().collect();
        sorted.sort_unstable_by_key(|&(k, _)| k);
        Ok(Self::from_sorted_pairs(num_images, &sorted))
    }

    /// Assemble the CSR adjacency from `(i, j) → accum` pairs sorted by key
    /// (`i < j`, unique).
    fn from_sorted_pairs(num_images: usize, sorted: &[((u32, u32), PairAccum)]) -> Self {
        let mut nbr_starts = vec![0usize; num_images + 1];
        for &((i, j), _) in sorted {
            nbr_starts[i as usize + 1] += 1;
            nbr_starts[j as usize + 1] += 1;
        }
        for r in 0..num_images {
            nbr_starts[r + 1] += nbr_starts[r];
        }
        let total = nbr_starts[num_images];
        let mut cursor = nbr_starts.clone();
        let mut nbr_images = vec![0u32; total];
        let mut nbr_shared = vec![0u32; total];
        let mut nbr_mean_disp = vec![0.0f64; total];
        // Keys ascend by (i, j), so both the row-i entries (partner j,
        // ascending) and the row-j entries (partner i, ascending) land in
        // ascending-partner order.
        for &((i, j), acc) in sorted {
            let mean = if acc.disp_n > 0 {
                acc.disp_sum / acc.disp_n as f64
            } else {
                0.0
            };
            for (row, partner) in [(i as usize, j), (j as usize, i)] {
                let at = cursor[row];
                nbr_images[at] = partner;
                nbr_shared[at] = acc.shared;
                nbr_mean_disp[at] = mean;
                cursor[row] += 1;
            }
        }
        Self {
            num_images,
            nbr_starts,
            nbr_images,
            nbr_shared,
            nbr_mean_disp,
        }
    }

    /// Number of images the substrate covers.
    pub fn num_images(&self) -> usize {
        self.num_images
    }

    /// Number of realized (covisible) pairs.
    pub fn num_pairs(&self) -> usize {
        self.nbr_images.len() / 2
    }

    /// `(shared count, mean displacement)` for the pair `(i, j)`; `None` when
    /// the pair is unrealized (or `i == j`). Panics if either index is out of
    /// range.
    pub fn pair(&self, i: u32, j: u32) -> Option<(u32, f64)> {
        assert!(
            (i as usize) < self.num_images && (j as usize) < self.num_images,
            "image index out of range"
        );
        if i == j {
            return None;
        }
        let (lo, hi) = (self.nbr_starts[i as usize], self.nbr_starts[i as usize + 1]);
        let at = lo + self.nbr_images[lo..hi].binary_search(&j).ok()?;
        Some((self.nbr_shared[at], self.nbr_mean_disp[at]))
    }

    /// Image `i`'s realized partners as `(partner, shared count, mean
    /// displacement)`, ascending partner index. Panics if `i` is out of
    /// range.
    pub fn neighbors(&self, i: u32) -> impl Iterator<Item = (u32, u32, f64)> + '_ {
        let i = i as usize;
        assert!(i < self.num_images, "image index out of range");
        let (lo, hi) = (self.nbr_starts[i], self.nbr_starts[i + 1]);
        (lo..hi).map(move |at| {
            (
                self.nbr_images[at],
                self.nbr_shared[at],
                self.nbr_mean_disp[at],
            )
        })
    }

    /// Partners of `i` at or above the `min_shared` shared-cluster floor,
    /// ordered by the displacement key (ties: ascending partner index),
    /// truncated to `k`.
    fn ranked_partners(&self, i: u32, k: usize, min_shared: u32, descending: bool) -> Vec<u32> {
        let mut ranked: Vec<(f64, u32)> = self
            .neighbors(i)
            .filter(|&(_, shared, _)| shared >= min_shared)
            .map(|(j, _, d)| (d, j))
            .collect();
        ranked.sort_by(|a, b| {
            let ord = a.0.total_cmp(&b.0);
            (if descending { ord.reverse() } else { ord }).then(a.1.cmp(&b.1))
        });
        ranked.truncate(k);
        ranked.into_iter().map(|(_, j)| j).collect()
    }

    /// The `k` lowest-mean-displacement partners of `i` with at least
    /// `min_shared` shared clusters (near-duplicate viewpoints; ties break by
    /// ascending partner index). Panics if `i` is out of range.
    pub fn nearest(&self, i: u32, k: usize, min_shared: u32) -> Vec<u32> {
        self.ranked_partners(i, k, min_shared, false)
    }

    /// The `k` highest-mean-displacement partners of `i` with at least
    /// `min_shared` shared clusters (wide-baseline pairs; ties break by
    /// ascending partner index). Panics if `i` is out of range.
    pub fn farthest(&self, i: u32, k: usize, min_shared: u32) -> Vec<u32> {
        self.ranked_partners(i, k, min_shared, true)
    }

    /// Compact serialization: parallel per-pair arrays `(i, j, shared count,
    /// mean displacement)` with `i < j`, sorted by `(i, j)`. Round-trips
    /// through [`Self::from_arrays`].
    pub fn to_arrays(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f64>) {
        let n_pairs = self.num_pairs();
        let mut pi = Vec::with_capacity(n_pairs);
        let mut pj = Vec::with_capacity(n_pairs);
        let mut shared = Vec::with_capacity(n_pairs);
        let mut mean_disp = Vec::with_capacity(n_pairs);
        for i in 0..self.num_images as u32 {
            for (j, s, d) in self.neighbors(i) {
                if j > i {
                    pi.push(i);
                    pj.push(j);
                    shared.push(s);
                    mean_disp.push(d);
                }
            }
        }
        (pi, pj, shared, mean_disp)
    }

    /// Rebuild the substrate from serialized per-pair arrays (any pair
    /// order; each unordered pair at most once, off-diagonal, indexes below
    /// `num_images`). The inverse of [`Self::to_arrays`].
    pub fn from_arrays(
        pair_i: &[u32],
        pair_j: &[u32],
        shared: &[u32],
        mean_disp: &[f64],
        num_images: usize,
    ) -> Result<Self, CovisibilityError> {
        let n = pair_i.len();
        if pair_j.len() != n || shared.len() != n || mean_disp.len() != n {
            return Err(CovisibilityError::PairArraysNotParallel {
                i: n,
                j: pair_j.len(),
                shared: shared.len(),
                mean_disp: mean_disp.len(),
            });
        }
        let mut sorted: Vec<((u32, u32), PairAccum)> = Vec::with_capacity(n);
        for k in 0..n {
            let (i, j) = (pair_i[k], pair_j[k]);
            if i == j {
                return Err(CovisibilityError::BadPair { i, j });
            }
            for &idx in &[i, j] {
                if idx as usize >= num_images {
                    return Err(CovisibilityError::ImageIndexOutOfRange {
                        index: idx,
                        num_images,
                    });
                }
            }
            sorted.push((
                (i.min(j), i.max(j)),
                PairAccum {
                    shared: shared[k],
                    disp_sum: mean_disp[k],
                    disp_n: 1,
                },
            ));
        }
        sorted.sort_unstable_by_key(|&(k, _)| k);
        if let Some(w) = sorted.windows(2).find(|w| w[0].0 == w[1].0) {
            let (i, j) = w[0].0;
            return Err(CovisibilityError::BadPair { i, j });
        }
        Ok(Self::from_sorted_pairs(num_images, &sorted))
    }
}
