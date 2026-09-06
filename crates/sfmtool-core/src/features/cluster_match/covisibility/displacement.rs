// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Sparse displacement-neighborhood substrate.
//!
//! Per *realized* covisible image pair, the shared-cluster count and two mean
//! statistics over the pair's shared-cluster keypoints — the mean displacement
//! *magnitude* and the mean displacement *vector* — built in one pass over the
//! clusters: linear in observations, and stored sparsely.
//! See `specs/core/geometry/pose-verification.md` (Substrate).

use super::{CovisibilityError, MAX_DENSE_IMAGES};

/// Sampled per-image-pair feature-displacement tables (row-major
/// `(num_images, num_images)`, symmetric, zero diagonal). Present only when
/// positions were supplied at construction.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DisplacementTables {
    /// Mean sampled displacement magnitude per pair; `0` where no sample
    /// landed.
    pub(super) mean: Vec<f64>,
    /// Samples behind each mean.
    pub(super) count: Vec<u32>,
}

/// Sparse displacement-neighborhood substrate: per *realized* covisible image
/// pair, the shared-cluster count, the mean pixel displacement magnitude and
/// the mean displacement vector of the pair's shared-cluster keypoints. Built
/// in one pass over the clusters — each cluster emits its accepted cross-image
/// member pairs, so under the cluster matcher's span cap both time and storage
/// are linear in observations; the only dense array is the build's transient
/// pair-slot index (`PairSlots`), which the assembled substrate does not keep.
/// See `specs/core/geometry/pose-verification.md` (Substrate).
///
/// The shared count matches [`ClusterCovisibility::count`](super::ClusterCovisibility::count) (each cluster
/// votes at most once per pair); both means average over *every* accepted
/// cross-image member pair of the shared clusters — exhaustive, not sampled
/// (contrast the seeded one-sample-per-cluster tables behind
/// [`ClusterCovisibility::pair_displacement_magnitude`](super::ClusterCovisibility::pair_displacement_magnitude)).
///
/// **Orientation contract (the vector statistic).** A pair is keyed by its
/// ascending image indexes `(lo, hi)` and its mean vector is oriented to that
/// key: each member pair contributes *(position in image `hi`) − (position in
/// image `lo`)*, so a member pair that arrives in the other order contributes
/// the negated delta. Without that normalization the same physical flow would
/// enter the sum with either sign depending on member order and coherent
/// motion would cancel itself. The accessors report in the key's orientation
/// whatever order their arguments arrive in — [`Self::pair_vector`] answers
/// `(i, j)` and `(j, i)` with the same vector, pointing from the
/// lower-indexed image's keypoint to the higher-indexed image's — and
/// [`Self::neighbors_vector`] reports a row's entries in that same key
/// orientation, *not* relative to the row's own image.
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
    /// Mean keypoint displacement magnitude (pixels) per adjacency entry.
    nbr_mean_magnitude: Vec<f64>,
    /// Mean keypoint displacement vector (pixels) per adjacency entry, in the
    /// pair key's low→high orientation (see the type's orientation contract),
    /// so both of a pair's two entries hold the identical vector.
    nbr_mean_vector: Vec<[f64; 2]>,
}

/// Per-pair accumulator for the neighborhood build. `magnitude_sum` and the
/// component sums `dx_sum` / `dy_sum` accumulate over the same member pairs
/// and divide by the same `disp_n`; the component sums follow the type's
/// low→high orientation contract.
#[derive(Clone, Copy, Default)]
struct PairAccum {
    shared: u32,
    magnitude_sum: f64,
    dx_sum: f64,
    dy_sum: f64,
    disp_n: u32,
}

/// A pair slot no cluster has touched yet. Entry indexes stay well below it:
/// at most `MAX_DENSE_IMAGES * (MAX_DENSE_IMAGES - 1) / 2` (~8.4 M) pairs can
/// ever be realized.
const UNSET: u32 = u32::MAX;

/// Transient dense index from an unordered image pair to its accumulator,
/// backing the neighborhood accumulation.
///
/// `slots` is one `u32` per ordered pair (`4 · num_images²` bytes, the same
/// order as the `counts` matrix [`ClusterCovisibility`](super::ClusterCovisibility)
/// holds permanently: 192 KB at 219 images, 67 MB at the
/// [`MAX_DENSE_IMAGES`] cap) and is dropped when the build assembles its CSR;
/// only *realized* pairs get an accumulator, so `entries` keeps the sparsity
/// a hash map would give while the address computation is one multiply-add
/// and one indexed load instead of a hash.
///
/// **Accumulation-order contract.** The kernel's arithmetic is order
/// sensitive: `magnitude_sum`, `dx_sum` and `dy_sum` are `f64` running sums,
/// so a pair's means are bit-reproducible only while their additions arrive in
/// the same sequence. This index preserves that because it changes the
/// *address* of an accumulator and nothing else -- the clusters are walked in
/// the same order, each cluster's span pairs and member pairs are emitted in
/// the same order, and each emitted pair reaches the same accumulator.
/// `entries` fills in first-touch order (as a hash map's entries are created),
/// and [`DisplacementNeighborhood::from_clusters`] sorts it by key before
/// assembly; the keys are unique unordered pairs, so that sort fixes the
/// sequence [`DisplacementNeighborhood::from_sorted_pairs`] sees regardless of
/// how the entries were ordered before it.
struct PairSlots {
    num_images: usize,
    slots: Vec<u32>,
    entries: Vec<((u32, u32), PairAccum)>,
}

impl PairSlots {
    fn new(num_images: usize) -> Self {
        Self {
            num_images,
            slots: vec![UNSET; num_images * num_images],
            entries: Vec::new(),
        }
    }

    /// The accumulator for the pair `(i, j)`, `i < j`, created on first touch.
    #[inline]
    fn accum(&mut self, i: u32, j: u32) -> &mut PairAccum {
        let cell = i as usize * self.num_images + j as usize;
        let slot = &mut self.slots[cell];
        if *slot == UNSET {
            *slot = self.entries.len() as u32;
            self.entries.push(((i, j), PairAccum::default()));
        }
        &mut self.entries[*slot as usize].1
    }
}

impl DisplacementNeighborhood {
    /// Build the substrate from CSR cluster arrays plus per-member positions
    /// (all parallel to `member_images`, pixel units). `member_accepted` is
    /// honored exactly as in [`ClusterCovisibility::from_clusters`](super::ClusterCovisibility::from_clusters): `None`
    /// means every member counts.
    ///
    /// Per cluster: the accepted members' deduplicated image list votes once
    /// per pair into the shared count, and every accepted cross-image member
    /// pair contributes both its Euclidean position distance to the pair's
    /// mean magnitude and its low→high position delta to the pair's mean
    /// vector (see the type's orientation contract). Deterministic — no
    /// sampling.
    ///
    /// Positions are the `f32` pairs a `.matches` backbone stores; the
    /// distances, the deltas and their means are `f64`.
    ///
    /// `num_images` is bounded by [`MAX_DENSE_IMAGES`] the same way
    /// [`ClusterCovisibility`](super::ClusterCovisibility) is: the assembled
    /// substrate is sparse, but the accumulation indexes pairs through a
    /// transient `num_images²` slot array, so the bound that sizes the count
    /// matrix sizes this too.
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

        // Both statistics address their accumulator through one slot index,
        // whose `num_images²` cell array is what the dense bound sizes.
        if num_images > MAX_DENSE_IMAGES {
            return Err(CovisibilityError::TooManyImages { num_images });
        }

        let mut entries = super::prof::NBR_ACCUM.time(|| {
            let mut pairs = PairSlots::new(num_images);
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
                        pairs.accum(i, j).shared += 1;
                    }
                }
                // Displacement: every accepted cross-image member pair. The
                // magnitude is member-order free; the vector is oriented by
                // image order, so the member sitting in the lower-indexed
                // image is the one subtracted.
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
                        let (k_lo, k_hi) = if ia < ib { (ka, kb) } else { (kb, ka) };
                        let dx = positions_xy[k_hi][0] as f64 - positions_xy[k_lo][0] as f64;
                        let dy = positions_xy[k_hi][1] as f64 - positions_xy[k_lo][1] as f64;
                        let e = pairs.accum(ia.min(ib), ia.max(ib));
                        e.magnitude_sum += d;
                        e.dx_sum += dx;
                        e.dy_sum += dy;
                        e.disp_n += 1;
                    }
                }
            }
            pairs.entries
        });

        // Deterministic order despite the first-touch accumulator order.
        Ok(super::prof::NBR_SORT.time(|| {
            entries.sort_unstable_by_key(|&(k, _)| k);
            Self::from_sorted_pairs(num_images, &entries)
        }))
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
        let mut nbr_mean_magnitude = vec![0.0f64; total];
        let mut nbr_mean_vector = vec![[0.0f64; 2]; total];
        // Keys ascend by (i, j), so both the row-i entries (partner j,
        // ascending) and the row-j entries (partner i, ascending) land in
        // ascending-partner order. The vector is keyed low→high, so both
        // entries of a pair carry it unchanged.
        for &((i, j), acc) in sorted {
            let (magnitude, vector) = if acc.disp_n > 0 {
                let n = acc.disp_n as f64;
                (acc.magnitude_sum / n, [acc.dx_sum / n, acc.dy_sum / n])
            } else {
                (0.0, [0.0, 0.0])
            };
            for (row, partner) in [(i as usize, j), (j as usize, i)] {
                let at = cursor[row];
                nbr_images[at] = partner;
                nbr_shared[at] = acc.shared;
                nbr_mean_magnitude[at] = magnitude;
                nbr_mean_vector[at] = vector;
                cursor[row] += 1;
            }
        }
        Self {
            num_images,
            nbr_starts,
            nbr_images,
            nbr_shared,
            nbr_mean_magnitude,
            nbr_mean_vector,
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

    /// The adjacency slot holding the pair `(i, j)`, or `None` when the pair
    /// is unrealized (or `i == j`). Panics if either index is out of range.
    fn entry(&self, i: u32, j: u32) -> Option<usize> {
        assert!(
            (i as usize) < self.num_images && (j as usize) < self.num_images,
            "image index out of range"
        );
        if i == j {
            return None;
        }
        let (lo, hi) = (self.nbr_starts[i as usize], self.nbr_starts[i as usize + 1]);
        Some(lo + self.nbr_images[lo..hi].binary_search(&j).ok()?)
    }

    /// The adjacency slots of image `i`'s row, ascending partner index.
    /// Panics if `i` is out of range.
    fn row(&self, i: u32) -> std::ops::Range<usize> {
        let i = i as usize;
        assert!(i < self.num_images, "image index out of range");
        self.nbr_starts[i]..self.nbr_starts[i + 1]
    }

    /// `(shared count, mean displacement magnitude)` for the pair `(i, j)`;
    /// `None` when the pair is unrealized (or `i == j`). Panics if either
    /// index is out of range.
    pub fn pair_magnitude(&self, i: u32, j: u32) -> Option<(u32, f64)> {
        let at = self.entry(i, j)?;
        Some((self.nbr_shared[at], self.nbr_mean_magnitude[at]))
    }

    /// `(shared count, mean displacement vector)` for the pair `(i, j)`;
    /// `None` when the pair is unrealized (or `i == j`). The vector is in the
    /// pair key's low→high orientation, so the answer does not depend on the
    /// argument order (see the type's orientation contract). Panics if either
    /// index is out of range.
    pub fn pair_vector(&self, i: u32, j: u32) -> Option<(u32, [f64; 2])> {
        let at = self.entry(i, j)?;
        Some((self.nbr_shared[at], self.nbr_mean_vector[at]))
    }

    /// Image `i`'s realized partners as `(partner, shared count, mean
    /// displacement magnitude)`, ascending partner index. Panics if `i` is out
    /// of range.
    pub fn neighbors_magnitude(&self, i: u32) -> impl Iterator<Item = (u32, u32, f64)> + '_ {
        self.row(i).map(move |at| {
            (
                self.nbr_images[at],
                self.nbr_shared[at],
                self.nbr_mean_magnitude[at],
            )
        })
    }

    /// Image `i`'s realized partners as `(partner, shared count, mean
    /// displacement vector)`, ascending partner index. Every vector is in its
    /// pair key's low→high orientation, *not* relative to `i` (see the type's
    /// orientation contract), so an entry whose partner sits below `i` points
    /// from the partner towards `i`. Panics if `i` is out of range.
    pub fn neighbors_vector(&self, i: u32) -> impl Iterator<Item = (u32, u32, [f64; 2])> + '_ {
        self.row(i).map(move |at| {
            (
                self.nbr_images[at],
                self.nbr_shared[at],
                self.nbr_mean_vector[at],
            )
        })
    }

    /// Partners of `i` at or above the `min_shared` shared-cluster floor,
    /// ordered by the displacement-magnitude key (ties: ascending partner
    /// index), truncated to `k`.
    fn ranked_partners(&self, i: u32, k: usize, min_shared: u32, descending: bool) -> Vec<u32> {
        let mut ranked: Vec<(f64, u32)> = self
            .neighbors_magnitude(i)
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

    /// The `k` lowest-mean-magnitude partners of `i` with at least
    /// `min_shared` shared clusters (near-duplicate viewpoints; ties break by
    /// ascending partner index). Panics if `i` is out of range.
    pub fn nearest(&self, i: u32, k: usize, min_shared: u32) -> Vec<u32> {
        self.ranked_partners(i, k, min_shared, false)
    }

    /// The `k` highest-mean-magnitude partners of `i` with at least
    /// `min_shared` shared clusters (wide-baseline pairs; ties break by
    /// ascending partner index). Panics if `i` is out of range.
    pub fn farthest(&self, i: u32, k: usize, min_shared: u32) -> Vec<u32> {
        self.ranked_partners(i, k, min_shared, true)
    }

    /// Compact serialization: parallel per-pair arrays `(i, j, shared count,
    /// mean displacement magnitude, mean displacement vector)` with `i < j`,
    /// sorted by `(i, j)`. Every vector is oriented `i → j`, which is the
    /// key's own low→high orientation. Round-trips through
    /// [`Self::from_arrays`].
    #[allow(clippy::type_complexity)]
    pub fn to_arrays(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f64>, Vec<[f64; 2]>) {
        let n_pairs = self.num_pairs();
        let mut pi = Vec::with_capacity(n_pairs);
        let mut pj = Vec::with_capacity(n_pairs);
        let mut shared = Vec::with_capacity(n_pairs);
        let mut mean_magnitude = Vec::with_capacity(n_pairs);
        let mut mean_vector = Vec::with_capacity(n_pairs);
        for i in 0..self.num_images as u32 {
            for at in self.row(i) {
                let j = self.nbr_images[at];
                if j > i {
                    pi.push(i);
                    pj.push(j);
                    shared.push(self.nbr_shared[at]);
                    mean_magnitude.push(self.nbr_mean_magnitude[at]);
                    mean_vector.push(self.nbr_mean_vector[at]);
                }
            }
        }
        (pi, pj, shared, mean_magnitude, mean_vector)
    }

    /// Rebuild the substrate from serialized per-pair arrays (any pair
    /// order; each unordered pair at most once, off-diagonal, indexes below
    /// `num_images`). The inverse of [`Self::to_arrays`].
    ///
    /// `mean_vector` is read in each row's own `pair_i[k] → pair_j[k]`
    /// orientation and negated where that row is descending, so it lands in
    /// the key's low→high orientation however the row was written. `None`
    /// leaves every vector zero — the magnitude-only serializations a caller
    /// persisted before the vector existed, and the only input the
    /// pose-verification kernels need.
    pub fn from_arrays(
        pair_i: &[u32],
        pair_j: &[u32],
        shared: &[u32],
        mean_magnitude: &[f64],
        mean_vector: Option<&[[f64; 2]]>,
        num_images: usize,
    ) -> Result<Self, CovisibilityError> {
        let n = pair_i.len();
        if pair_j.len() != n
            || shared.len() != n
            || mean_magnitude.len() != n
            || mean_vector.is_some_and(|v| v.len() != n)
        {
            return Err(CovisibilityError::PairArraysNotParallel {
                i: n,
                j: pair_j.len(),
                shared: shared.len(),
                mean_magnitude: mean_magnitude.len(),
                mean_vector: mean_vector.map(<[[f64; 2]]>::len),
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
            let [dx, dy] = mean_vector.map_or([0.0, 0.0], |v| v[k]);
            let flip = if i < j { 1.0 } else { -1.0 };
            sorted.push((
                (i.min(j), i.max(j)),
                PairAccum {
                    shared: shared[k],
                    magnitude_sum: mean_magnitude[k],
                    dx_sum: flip * dx,
                    dy_sum: flip * dy,
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
