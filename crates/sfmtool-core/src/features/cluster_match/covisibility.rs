// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Cluster covisibility: how many match clusters each pair of images shares.
//!
//! Built from the `clusters/` backbone of a `.matches` file (optionally
//! restricted by a per-member acceptance mask), the symmetric count matrix
//! answers pre-reconstruction grouping queries: greedy mutually-covisible
//! seed groups, candidate ranking by shared-cluster count, and raw-count
//! inspection. This is *cluster* covisibility, computed before any
//! reconstruction exists — distinct from the post-reconstruction
//! shared-3D-track covisibility in `crate::analysis::image_pair_graph`,
//! which requires poses and points.
//!
//! See `specs/core/cluster-covisibility.md` for the design and the seed-group
//! algorithm's determinism contract, `specs/core/covisibility-selection.md`
//! for the selection queries built on top (pair displacement, banded thinning,
//! reach), and `specs/core/pose-verification.md` for the sparse
//! displacement-neighborhood substrate ([`DisplacementNeighborhood`]).

use std::collections::HashMap;

/// Dense-backend image cap. Storage is a row-major `u32` matrix (`4·N²`
/// bytes): 64 MB at this bound, which sits inside the spec's ~4–5 k-image
/// window where dense storage stops being reasonable. Construction errors
/// with [`CovisibilityError::TooManyImages`] above it; a sparse backend
/// behind the same type is the intended remedy when a larger consumer
/// appears.
pub const MAX_DENSE_IMAGES: usize = 4096;

/// Errors from [`ClusterCovisibility::from_clusters`] input validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CovisibilityError {
    /// `num_images` exceeds [`MAX_DENSE_IMAGES`].
    TooManyImages { num_images: usize },
    /// `cluster_starts` is not a valid CSR offset array over the members.
    BadClusterStarts { m: usize },
    /// `member_accepted` is not parallel to `member_images`.
    MaskNotParallel { members: usize, mask: usize },
    /// `positions_xy` is not parallel to `member_images`.
    PositionsNotParallel { members: usize, positions: usize },
    /// A member's image index is out of range.
    ImageIndexOutOfRange { index: u32, num_images: usize },
    /// [`DisplacementNeighborhood::from_arrays`]: the four pair arrays do not
    /// share one length.
    PairArraysNotParallel {
        i: usize,
        j: usize,
        shared: usize,
        mean_disp: usize,
    },
    /// [`DisplacementNeighborhood::from_arrays`]: a diagonal (`i == j`) or
    /// repeated pair.
    BadPair { i: u32, j: u32 },
}

impl std::fmt::Display for CovisibilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyImages { num_images } => write!(
                f,
                "num_images ({num_images}) exceeds the dense covisibility bound \
                 ({MAX_DENSE_IMAGES}); the dense u32 matrix would need {} MB — a sparse \
                 backend is required beyond this",
                4 * num_images * num_images / (1024 * 1024),
            ),
            Self::BadClusterStarts { m } => write!(
                f,
                "cluster_starts must be non-empty, non-decreasing, start at 0, and end at \
                 the member count M ({m})"
            ),
            Self::MaskNotParallel { members, mask } => write!(
                f,
                "member_accepted ({mask}) must be parallel to member_images ({members})"
            ),
            Self::PositionsNotParallel { members, positions } => write!(
                f,
                "positions_xy ({positions}) must be parallel to member_images ({members})"
            ),
            Self::ImageIndexOutOfRange { index, num_images } => write!(
                f,
                "member image index {index} is out of range for {num_images} images"
            ),
            Self::PairArraysNotParallel {
                i,
                j,
                shared,
                mean_disp,
            } => write!(
                f,
                "pair arrays must share one length: i ({i}), j ({j}), shared ({shared}), \
                 mean_disp ({mean_disp})"
            ),
            Self::BadPair { i, j } => write!(
                f,
                "pair ({i}, {j}) is diagonal or repeated — pairs must be distinct \
                 unordered image pairs"
            ),
        }
    }
}

impl std::error::Error for CovisibilityError {}

/// Tuning for [`ClusterCovisibility::seed_groups`].
#[derive(Clone, Debug)]
pub struct SeedGroupParams {
    /// Maximum images per group (default 5). The seed edge always
    /// contributes two images, so values below 2 behave as 2.
    pub group_size: usize,
    /// Minimum shared-cluster count: every within-group pair of a yielded
    /// group has covisibility ≥ this (default 8).
    pub min_shared: u32,
}

impl Default for SeedGroupParams {
    fn default() -> Self {
        Self {
            group_size: 5,
            min_shared: 8,
        }
    }
}

/// Deterministic 64-bit generator (splitmix64) behind the sampled
/// displacement pass. Bounded draws use Lemire's widening multiply; the
/// modulo bias is ~`bound / 2^64` — irrelevant at cluster sizes.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform draw in `[0, bound)`; `bound` must be nonzero.
    fn below(&mut self, bound: usize) -> usize {
        ((self.next_u64() as u128 * bound as u128) >> 64) as usize
    }
}

/// Sampled per-image-pair feature-displacement tables (row-major
/// `(num_images, num_images)`, symmetric, zero diagonal). Present only when
/// positions were supplied at construction.
#[derive(Debug, Clone, PartialEq)]
struct DisplacementTables {
    /// Mean sampled displacement per pair; `0` where no sample landed.
    mean: Vec<f64>,
    /// Samples behind each mean.
    count: Vec<u32>,
}

/// Sparse displacement-neighborhood substrate: per *realized* covisible image
/// pair, the shared-cluster count and the mean pixel displacement of the
/// pair's shared-cluster keypoints. Built in one pass over the clusters —
/// each cluster emits its accepted cross-image member pairs, so under the
/// cluster matcher's span cap both time and storage are linear in
/// observations, with no dense matrix anywhere. See
/// `specs/core/pose-verification.md` (Substrate).
///
/// The shared count matches [`ClusterCovisibility::count`] (each cluster
/// votes at most once per pair); the mean displacement averages over *every*
/// accepted cross-image member pair of the shared clusters — exhaustive, not
/// sampled (contrast the seeded one-sample-per-cluster tables behind
/// [`ClusterCovisibility::pair_displacement`]).
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
    /// honored exactly as in [`ClusterCovisibility::from_clusters`]: `None`
    /// means every member counts.
    ///
    /// Per cluster: the accepted members' deduplicated image list votes once
    /// per pair into the shared count, and every accepted cross-image member
    /// pair contributes its Euclidean position distance to the pair's mean
    /// displacement. Deterministic — no sampling.
    pub fn from_clusters(
        cluster_starts: &[u32],
        member_images: &[u32],
        member_accepted: Option<&[bool]>,
        num_images: usize,
        positions_xy: &[[f64; 2]],
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
                        positions_xy[ka][0] - positions_xy[kb][0],
                        positions_xy[ka][1] - positions_xy[kb][1],
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

/// Symmetric per-image-pair shared-cluster counts (zero diagonal).
///
/// `W[i, j]` = number of clusters with an accepted member in image `i` and
/// an accepted member in image `j`; each cluster contributes at most 1 to
/// any pair, and clusters spanning fewer than 2 accepted images contribute
/// nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterCovisibility {
    num_images: usize,
    /// Row-major `(num_images, num_images)` counts.
    counts: Vec<u32>,
    /// Sampled displacement tables; `None` without construction positions.
    displacement: Option<DisplacementTables>,
    /// Sparse displacement neighborhood; `None` without construction
    /// positions.
    neighborhood: Option<DisplacementNeighborhood>,
}

impl ClusterCovisibility {
    /// Build the count matrix from CSR cluster arrays (the `clusters/`
    /// section layout: cluster `c` owns members
    /// `cluster_starts[c]..cluster_starts[c+1]`).
    ///
    /// `member_accepted` is parallel to `member_images`; `None` means every
    /// member counts. Each cluster's accepted-image list is deduplicated
    /// before counting, so a cluster votes at most once per pair even if the
    /// input holds several members in one image.
    ///
    /// Displacement queries stay unavailable; see
    /// [`Self::from_clusters_with_positions`].
    pub fn from_clusters(
        cluster_starts: &[u32],
        member_images: &[u32],
        member_accepted: Option<&[bool]>,
        num_images: usize,
    ) -> Result<Self, CovisibilityError> {
        Self::from_clusters_with_positions(
            cluster_starts,
            member_images,
            member_accepted,
            num_images,
            None,
            0,
        )
    }

    /// [`Self::from_clusters`] plus optional per-member observation positions
    /// (`positions_xy`, parallel to `member_images`, pixel units), which
    /// enable the displacement queries ([`Self::pair_displacement`],
    /// [`Self::pair_displacement_counts`]) and the isolation-ordered thinning
    /// sweep (see [`Self::thin`]).
    ///
    /// One sampled displacement pass runs at construction: every cluster with
    /// two or more accepted members contributes one seeded uniformly-sampled
    /// distinct-member pair (`seed` drives the sampling; same-image pairs are
    /// skipped, not resampled), and the pair's Euclidean position distance
    /// accumulates into its image pair's mean. The shared-cluster counts are
    /// unchanged by `positions_xy`.
    pub fn from_clusters_with_positions(
        cluster_starts: &[u32],
        member_images: &[u32],
        member_accepted: Option<&[bool]>,
        num_images: usize,
        positions_xy: Option<&[[f64; 2]]>,
        seed: u64,
    ) -> Result<Self, CovisibilityError> {
        if num_images > MAX_DENSE_IMAGES {
            return Err(CovisibilityError::TooManyImages { num_images });
        }
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
        if let Some(pos) = positions_xy {
            if pos.len() != m {
                return Err(CovisibilityError::PositionsNotParallel {
                    members: m,
                    positions: pos.len(),
                });
            }
        }
        if let Some(&bad) = member_images.iter().find(|&&i| i as usize >= num_images) {
            return Err(CovisibilityError::ImageIndexOutOfRange {
                index: bad,
                num_images,
            });
        }

        let mut counts = vec![0u32; num_images * num_images];
        let mut displacement = positions_xy.map(|_| DisplacementTables {
            mean: vec![0.0; num_images * num_images],
            count: vec![0u32; num_images * num_images],
        });
        let mut rng = SplitMix64::new(seed);
        let mut rows: Vec<usize> = Vec::new();
        let mut span: Vec<u32> = Vec::new();
        for c in 0..cluster_starts.len() - 1 {
            let lo = cluster_starts[c] as usize;
            let hi = cluster_starts[c + 1] as usize;
            rows.clear();
            rows.extend((lo..hi).filter(|&k| member_accepted.is_none_or(|mask| mask[k])));
            span.clear();
            span.extend(rows.iter().map(|&k| member_images[k]));
            span.sort_unstable();
            span.dedup();
            for (a, &i) in span.iter().enumerate() {
                for &j in &span[a + 1..] {
                    counts[i as usize * num_images + j as usize] += 1;
                    counts[j as usize * num_images + i as usize] += 1;
                }
            }
            // One uniformly-sampled distinct-member pair per multi-member
            // cluster; a pair landing in one image is skipped, not resampled
            // (the mean displacement tables measure cross-image motion only).
            if let (Some(tables), Some(pos)) = (displacement.as_mut(), positions_xy) {
                if rows.len() >= 2 {
                    let a = rng.below(rows.len());
                    let mut b = rng.below(rows.len() - 1);
                    if b >= a {
                        b += 1;
                    }
                    let (ra, rb) = (rows[a], rows[b]);
                    let (ia, ib) = (member_images[ra] as usize, member_images[rb] as usize);
                    if ia != ib {
                        let d = f64::hypot(pos[ra][0] - pos[rb][0], pos[ra][1] - pos[rb][1]);
                        // Accumulate sums in `mean` (upper triangle); a final
                        // pass divides and mirrors.
                        let key = ia.min(ib) * num_images + ia.max(ib);
                        tables.mean[key] += d;
                        tables.count[key] += 1;
                    }
                }
            }
        }
        if let Some(tables) = displacement.as_mut() {
            for i in 0..num_images {
                for j in (i + 1)..num_images {
                    let (up, lo) = (i * num_images + j, j * num_images + i);
                    let n = tables.count[up];
                    if n > 0 {
                        tables.mean[up] /= n as f64;
                        tables.mean[lo] = tables.mean[up];
                        tables.count[lo] = n;
                    }
                }
            }
        }

        // The sparse displacement neighborhood shares the positioned inputs;
        // a second linear pass keeps the sampled-table RNG stream untouched.
        let neighborhood = match positions_xy {
            Some(pos) => Some(DisplacementNeighborhood::from_clusters(
                cluster_starts,
                member_images,
                member_accepted,
                num_images,
                pos,
            )?),
            None => None,
        };

        Ok(Self {
            num_images,
            counts,
            displacement,
            neighborhood,
        })
    }

    /// Number of images the matrix covers.
    pub fn num_images(&self) -> usize {
        self.num_images
    }

    /// Shared-cluster count for the pair `(i, j)`. Zero on the diagonal.
    /// Panics if either index is out of range.
    pub fn count(&self, i: u32, j: u32) -> u32 {
        assert!((i as usize) < self.num_images && (j as usize) < self.num_images);
        self.counts[i as usize * self.num_images + j as usize]
    }

    /// Image `i`'s row of counts (length [`Self::num_images`]). Panics if
    /// `i` is out of range.
    pub fn row(&self, i: u32) -> &[u32] {
        let i = i as usize;
        assert!(i < self.num_images);
        &self.counts[i * self.num_images..(i + 1) * self.num_images]
    }

    /// `candidates` reordered by descending covisibility with `image` (ties:
    /// ascending index); zero-covisibility candidates are dropped. Panics if
    /// `image` or any candidate is out of range.
    pub fn rank_by_covisibility(&self, image: u32, candidates: &[u32]) -> Vec<u32> {
        let row = self.row(image);
        let mut ranked: Vec<u32> = candidates
            .iter()
            .copied()
            .filter(|&c| {
                assert!((c as usize) < self.num_images);
                row[c as usize] > 0
            })
            .collect();
        ranked.sort_unstable_by(|&a, &b| row[b as usize].cmp(&row[a as usize]).then(a.cmp(&b)));
        ranked
    }

    /// Row-major `(num_images, num_images)` mean sampled feature
    /// displacement per covisible pair (symmetric, `0` where no sample
    /// landed). `None` when constructed without positions.
    pub fn pair_displacement(&self) -> Option<&[f64]> {
        self.displacement.as_ref().map(|t| t.mean.as_slice())
    }

    /// Row-major `(num_images, num_images)` sample counts behind
    /// [`Self::pair_displacement`], for callers that gate on support.
    /// `None` when constructed without positions.
    pub fn pair_displacement_counts(&self) -> Option<&[u32]> {
        self.displacement.as_ref().map(|t| t.count.as_slice())
    }

    /// The sparse displacement-neighborhood substrate (per realized pair:
    /// shared-cluster count + exhaustive mean keypoint displacement, with the
    /// `nearest` / `farthest` / `pair` queries and array serialization).
    /// `None` when constructed without positions. See
    /// `specs/core/pose-verification.md`.
    pub fn displacement_neighborhood(&self) -> Option<&DisplacementNeighborhood> {
        self.neighborhood.as_ref()
    }

    /// The thinning sweep order: decreasing isolation — largest
    /// nearest-covisible-partner displacement first (an image's isolation is
    /// the *smallest* sampled mean displacement to any partner; no sampled
    /// partner means infinitely isolated), ties and the no-positions case
    /// falling back to construction order (ascending index).
    fn sweep_order(&self) -> Vec<u32> {
        let n = self.num_images;
        let mut order: Vec<u32> = (0..n as u32).collect();
        if let Some(tables) = &self.displacement {
            let isolation: Vec<f64> = (0..n)
                .map(|i| {
                    (0..n)
                        .filter(|&j| tables.count[i * n + j] > 0)
                        .map(|j| tables.mean[i * n + j])
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            order.sort_by(|&a, &b| {
                isolation[b as usize]
                    .partial_cmp(&isolation[a as usize])
                    .expect("isolation values are never NaN")
                    .then(a.cmp(&b))
            });
        }
        order
    }

    /// [`Self::thin`] against a precomputed sweep order.
    fn thin_in_order(&self, order: &[u32], tau: f64) -> Vec<u32> {
        let n = self.num_images;
        let mut kept: Vec<u32> = Vec::new();
        for &i in order {
            if kept.is_empty() {
                kept.push(i);
                continue;
            }
            let row = &self.counts[i as usize * n..(i as usize + 1) * n];
            let best = kept
                .iter()
                .map(|&k| row[k as usize])
                .max()
                .expect("kept is non-empty") as f64;
            if tau / 8.0 <= best && best < tau {
                kept.push(i);
            }
        }
        kept.sort_unstable();
        kept
    }

    /// Redundancy-thinned subset (sorted ascending): a greedy sweep in
    /// decreasing isolation (see the spec's Thinning section) keeps an image
    /// only when its best shared-cluster count against the already-kept set
    /// falls in the band `[tau/8, tau)` — images above the band duplicate a
    /// kept viewpoint, images below it are disconnected from the skeleton.
    /// The first swept image is always kept.
    pub fn thin(&self, tau: f64) -> Vec<u32> {
        self.thin_in_order(&self.sweep_order(), tau)
    }

    /// Thin to approximately `target` images: binary-search `tau` (the kept
    /// count grows monotonically with `tau`) over `[1, median row peak]` and
    /// return the subset whose size lands closest to `target` (sorted
    /// ascending; earlier iterations win exact-distance ties).
    pub fn thin_to(&self, target: usize) -> Vec<u32> {
        let n = self.num_images;
        if n == 0 {
            return Vec::new();
        }
        // Median (numpy-style: mean of the middle two for even n) of the
        // per-image peak covisibility.
        let mut peaks: Vec<u32> = (0..n)
            .map(|i| {
                self.counts[i * n..(i + 1) * n]
                    .iter()
                    .copied()
                    .max()
                    .expect("rows are non-empty")
            })
            .collect();
        peaks.sort_unstable();
        let med_peak = if n % 2 == 1 {
            peaks[n / 2] as f64
        } else {
            (peaks[n / 2 - 1] as f64 + peaks[n / 2] as f64) / 2.0
        };

        let order = self.sweep_order();
        let (mut lo, mut hi) = (1.0f64, med_peak);
        let mut best: Option<Vec<u32>> = None;
        for _ in 0..25 {
            let mid = (lo + hi) / 2.0;
            let keep = self.thin_in_order(&order, mid);
            let closer = best
                .as_ref()
                .is_none_or(|b| keep.len().abs_diff(target) < b.len().abs_diff(target));
            let below = keep.len() < target;
            if closer {
                best = Some(keep);
            }
            if below {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        best.expect("25 iterations always produce a candidate")
    }

    /// Fraction of all images sharing at least `min_shared` clusters with at
    /// least one image of `images` (subset members count as reached). An
    /// empty subset reaches nothing (`0.0`). Panics if any index is out of
    /// range.
    pub fn reach(&self, images: &[u32], min_shared: u32) -> f64 {
        let n = self.num_images;
        if images.is_empty() || n == 0 {
            return 0.0;
        }
        let mut connected = vec![false; n];
        for &s in images {
            assert!((s as usize) < n, "subset image index out of range");
            connected[s as usize] = true;
        }
        for (i, slot) in connected.iter_mut().enumerate() {
            *slot = *slot
                || images
                    .iter()
                    .any(|&s| self.counts[i * n + s as usize] >= min_shared);
        }
        connected.iter().filter(|&&c| c).count() as f64 / n as f64
    }

    /// Lazy iterator of greedy mutually-covisible seed groups (see the
    /// spec's Seed-group algorithm): each `next()` scans for the strongest
    /// remaining edge and greedily extends it, so consumers take as many
    /// groups as they need and drop the rest unpaid. Deterministic: the
    /// sequence depends only on the input arrays, groups are disjoint, and
    /// the first `k` groups are identical however many are consumed.
    pub fn seed_groups(&self, params: &SeedGroupParams) -> SeedGroups<'_> {
        SeedGroups {
            covis: self,
            excluded: vec![false; self.num_images],
            params: params.clone(),
        }
    }

    /// One step of the seed-group algorithm against an external exclusion
    /// mask: find the strongest non-excluded edge, greedily extend it, mark
    /// the yielded group excluded, and return it sorted ascending. `None`
    /// when the strongest remaining edge is below `min_shared` (or no edge
    /// remains).
    ///
    /// This is the single implementation the borrowing [`SeedGroups`]
    /// iterator and external lazy iterators (e.g. the Python binding, which
    /// cannot hold a Rust borrow) both drive; `excluded` must have
    /// [`Self::num_images`] entries. Panics otherwise.
    pub fn next_seed_group(
        &self,
        excluded: &mut [bool],
        params: &SeedGroupParams,
    ) -> Option<Vec<u32>> {
        let n = self.num_images;
        assert_eq!(
            excluded.len(),
            n,
            "excluded mask must have num_images entries"
        );

        // 1. Strongest remaining edge; strict > with ascending (i, j)
        //    iteration keeps the lexicographically smallest tie.
        let mut best: Option<(u32, usize, usize)> = None;
        for i in 0..n {
            if excluded[i] {
                continue;
            }
            let row = &self.counts[i * n..(i + 1) * n];
            for j in (i + 1)..n {
                if !excluded[j] && best.is_none_or(|(w, _, _)| row[j] > w) {
                    best = Some((row[j], i, j));
                }
            }
        }
        let (w, i, j) = best?;
        if w < params.min_shared {
            return None;
        }
        let mut group: Vec<u32> = vec![i as u32, j as u32];

        // 2. Greedy extension maximizing the *minimum* shared count vs the
        //    group (mutual covisibility, not hub-and-spokes); strict > with
        //    ascending k keeps the smallest tie.
        while group.len() < params.group_size {
            let mut best_k: Option<(u32, usize)> = None;
            for (k, &k_excluded) in excluded.iter().enumerate() {
                if k_excluded || group.iter().any(|&g| g as usize == k) {
                    continue;
                }
                let min_w = group
                    .iter()
                    .map(|&g| self.counts[k * n + g as usize])
                    .min()
                    .expect("group is never empty");
                if best_k.is_none_or(|(w, _)| min_w > w) {
                    best_k = Some((min_w, k));
                }
            }
            match best_k {
                Some((min_w, k)) if min_w >= params.min_shared => group.push(k as u32),
                _ => break,
            }
        }

        // 3. Yield sorted ascending; exclude from all later consideration.
        group.sort_unstable();
        for &g in &group {
            excluded[g as usize] = true;
        }
        Some(group)
    }
}

/// Lazy seed-group iterator. Borrows the matrix; the only state is the
/// excluded-image mask (no matrix copy). Each `next()` costs one strongest
/// remaining-edge scan plus the group-extension steps.
pub struct SeedGroups<'a> {
    covis: &'a ClusterCovisibility,
    excluded: Vec<bool>,
    params: SeedGroupParams,
}

impl Iterator for SeedGroups<'_> {
    type Item = Vec<u32>;

    fn next(&mut self) -> Option<Vec<u32>> {
        self.covis.next_seed_group(&mut self.excluded, &self.params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesize CSR cluster arrays whose covisibility equals the given
    /// weighted edges: edge `(i, j, w)` becomes `w` two-member clusters.
    fn from_edges(num_images: usize, edges: &[(u32, u32, u32)]) -> ClusterCovisibility {
        let mut starts = vec![0u32];
        let mut images = Vec::new();
        for &(i, j, w) in edges {
            for _ in 0..w {
                images.extend([i.min(j), i.max(j)]);
                starts.push(images.len() as u32);
            }
        }
        ClusterCovisibility::from_clusters(&starts, &images, None, num_images).unwrap()
    }

    #[test]
    fn counts_symmetric_zero_diagonal() {
        // Cluster 0: images {0, 1, 2}; cluster 1: {0, 2}; cluster 2: span 1.
        let cov = ClusterCovisibility::from_clusters(&[0, 3, 5, 6], &[0, 1, 2, 0, 2, 1], None, 4)
            .unwrap();
        assert_eq!(cov.num_images(), 4);
        assert_eq!(cov.count(0, 1), 1);
        assert_eq!(cov.count(0, 2), 2);
        assert_eq!(cov.count(1, 2), 1);
        for i in 0..4 {
            assert_eq!(cov.count(i, i), 0);
            for j in 0..4 {
                assert_eq!(cov.count(i, j), cov.count(j, i));
            }
            // Image 3 appears in no cluster.
            assert_eq!(cov.count(i, 3), 0);
        }
        assert_eq!(cov.row(0), &[0, 1, 2, 0]);
    }

    #[test]
    fn mask_restricts_members() {
        // Masking cluster 0's image-2 member removes its (0,2) and (1,2)
        // votes; cluster 1 still supplies (0,2).
        let cov = ClusterCovisibility::from_clusters(
            &[0, 3, 5],
            &[0, 1, 2, 0, 2],
            Some(&[true, true, false, true, true]),
            3,
        )
        .unwrap();
        assert_eq!(cov.count(0, 1), 1);
        assert_eq!(cov.count(0, 2), 1);
        assert_eq!(cov.count(1, 2), 0);
    }

    #[test]
    fn duplicate_images_in_cluster_count_once() {
        // Two members in image 0 within one cluster: still one vote for (0,1).
        let cov = ClusterCovisibility::from_clusters(&[0, 3], &[0, 0, 1], None, 2).unwrap();
        assert_eq!(cov.count(0, 1), 1);
    }

    #[test]
    fn masked_span_below_two_contributes_nothing() {
        let cov =
            ClusterCovisibility::from_clusters(&[0, 2], &[0, 1], Some(&[true, false]), 2).unwrap();
        assert_eq!(cov.count(0, 1), 0);
    }

    #[test]
    fn validation_errors() {
        // Bad CSR: does not end at M.
        assert_eq!(
            ClusterCovisibility::from_clusters(&[0, 3], &[0, 1], None, 2),
            Err(CovisibilityError::BadClusterStarts { m: 2 })
        );
        // Bad CSR: empty starts.
        assert_eq!(
            ClusterCovisibility::from_clusters(&[], &[], None, 2),
            Err(CovisibilityError::BadClusterStarts { m: 0 })
        );
        // Bad CSR: does not start at 0.
        assert_eq!(
            ClusterCovisibility::from_clusters(&[1, 2], &[0, 1], None, 2),
            Err(CovisibilityError::BadClusterStarts { m: 2 })
        );
        // Mask not parallel.
        assert_eq!(
            ClusterCovisibility::from_clusters(&[0, 2], &[0, 1], Some(&[true]), 2),
            Err(CovisibilityError::MaskNotParallel {
                members: 2,
                mask: 1
            })
        );
        // Image index out of range.
        assert_eq!(
            ClusterCovisibility::from_clusters(&[0, 2], &[0, 5], None, 2),
            Err(CovisibilityError::ImageIndexOutOfRange {
                index: 5,
                num_images: 2
            })
        );
    }

    #[test]
    fn dense_bound_errors() {
        let n = MAX_DENSE_IMAGES + 1;
        let err = ClusterCovisibility::from_clusters(&[0], &[], None, n).unwrap_err();
        assert_eq!(err, CovisibilityError::TooManyImages { num_images: n });
        assert!(err.to_string().contains("dense covisibility bound"));
        // At the bound itself, construction succeeds.
        assert!(ClusterCovisibility::from_clusters(&[0], &[], None, MAX_DENSE_IMAGES).is_ok());
    }

    #[test]
    fn rank_by_covisibility_orders_and_drops_zeros() {
        let cov = from_edges(5, &[(0, 1, 3), (0, 2, 7), (0, 3, 3)]);
        // Descending count; the 3-count tie (1 vs 3) resolves ascending;
        // zero-covisibility candidate 4 is dropped.
        assert_eq!(cov.rank_by_covisibility(0, &[4, 3, 2, 1]), vec![2, 1, 3]);
        // The image itself has zero self-covisibility and is dropped.
        assert_eq!(cov.rank_by_covisibility(0, &[0, 2]), vec![2]);
    }

    #[test]
    fn seed_groups_two_disjoint_triangles() {
        let cov = from_edges(
            6,
            &[
                (0, 1, 10),
                (0, 2, 10),
                (1, 2, 10),
                (3, 4, 9),
                (3, 5, 9),
                (4, 5, 9),
            ],
        );
        let params = SeedGroupParams {
            group_size: 3,
            min_shared: 8,
        };
        let groups: Vec<_> = cov.seed_groups(&params).collect();
        assert_eq!(groups, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn seed_edge_tie_breaks_lexicographically() {
        // Two equal-strength edges: (0, 1) wins over (2, 3).
        let cov = from_edges(4, &[(2, 3, 10), (0, 1, 10)]);
        let params = SeedGroupParams::default();
        let groups: Vec<_> = cov.seed_groups(&params).collect();
        assert_eq!(groups, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn extension_tie_breaks_smallest_k() {
        // Both 2 and 3 extend {0, 1} with min 8; smallest k (2) is added
        // first, then 3 still qualifies (min over {0,1,2} of its edges).
        let cov = from_edges(
            4,
            &[
                (0, 1, 10),
                (0, 2, 8),
                (1, 2, 8),
                (0, 3, 8),
                (1, 3, 8),
                (2, 3, 8),
            ],
        );
        let params = SeedGroupParams {
            group_size: 3,
            min_shared: 8,
        };
        let groups: Vec<_> = cov.seed_groups(&params).collect();
        assert_eq!(groups, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn star_topology_does_not_form_a_group() {
        // Hub 0 strongly connected to 1, 2, 3; no spoke-spoke edges. The
        // minimum-vs-group criterion stops every extension, so only the
        // strongest hub edge pair is ever yielded.
        let cov = from_edges(4, &[(0, 1, 10), (0, 2, 10), (0, 3, 10)]);
        let params = SeedGroupParams {
            group_size: 4,
            min_shared: 8,
        };
        let groups: Vec<_> = cov.seed_groups(&params).collect();
        assert_eq!(groups, vec![vec![0, 1]]);
    }

    #[test]
    fn extension_stops_below_min_shared_but_yields_partial_group() {
        // {0, 1, 2} is mutually strong; 3 attaches to 0 and 1 only weakly.
        let cov = from_edges(4, &[(0, 1, 10), (0, 2, 9), (1, 2, 9), (0, 3, 4), (1, 3, 4)]);
        let params = SeedGroupParams {
            group_size: 4,
            min_shared: 8,
        };
        let groups: Vec<_> = cov.seed_groups(&params).collect();
        assert_eq!(groups, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn iterator_ends_when_strongest_edge_below_min_shared() {
        let cov = from_edges(3, &[(0, 1, 5), (1, 2, 3)]);
        let params = SeedGroupParams::default(); // min_shared = 8
        assert_eq!(cov.seed_groups(&params).count(), 0);
    }

    #[test]
    fn group_size_caps_extension() {
        // Complete graph on 5 images, all edges 10, group_size 3: the first
        // group is {0, 1, 2}, the second {3, 4}.
        let mut edges = Vec::new();
        for i in 0..5u32 {
            for j in (i + 1)..5 {
                edges.push((i, j, 10));
            }
        }
        let cov = from_edges(5, &edges);
        let params = SeedGroupParams {
            group_size: 3,
            min_shared: 8,
        };
        let groups: Vec<_> = cov.seed_groups(&params).collect();
        assert_eq!(groups, vec![vec![0, 1, 2], vec![3, 4]]);
    }

    #[test]
    fn prefix_stability() {
        let mut edges = Vec::new();
        // A deterministic pseudo-random weighted graph on 12 images.
        let mut state = 0x9e3779b9u32;
        for i in 0..12u32 {
            for j in (i + 1)..12 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                edges.push((i, j, state >> 28));
            }
        }
        let cov = from_edges(12, &edges);
        let params = SeedGroupParams {
            group_size: 4,
            min_shared: 3,
        };
        let all: Vec<_> = cov.seed_groups(&params).collect();
        assert!(all.len() >= 2, "fixture must yield multiple groups");
        for k in 1..=all.len() {
            let prefix: Vec<_> = cov.seed_groups(&params).take(k).collect();
            assert_eq!(prefix, all[..k]);
        }
    }

    #[test]
    fn determinism() {
        let cov = from_edges(6, &[(0, 1, 9), (1, 2, 9), (0, 2, 9), (3, 4, 8)]);
        let params = SeedGroupParams::default();
        let a: Vec<_> = cov.seed_groups(&params).collect();
        let b: Vec<_> = cov.seed_groups(&params).collect();
        assert_eq!(a, b);
        assert_eq!(a, vec![vec![0, 1, 2], vec![3, 4]]);
    }

    // ── Selection queries (specs/core/covisibility-selection.md) ──────────

    /// Synthesize positioned two-member clusters: edge `(i, j, w, d)` becomes
    /// `w` clusters, each holding one member in image `i` at the origin and
    /// one in image `j` at distance `d`. Two-member clusters make the sampled
    /// displacement pass deterministic regardless of seed (the pair is
    /// forced), so `mean[i][j] == d` and `count[i][j] == w` exactly.
    fn from_positioned_edges(
        num_images: usize,
        edges: &[(u32, u32, u32, f64)],
    ) -> ClusterCovisibility {
        let mut starts = vec![0u32];
        let mut images = Vec::new();
        let mut positions = Vec::new();
        for &(i, j, w, d) in edges {
            for _ in 0..w {
                images.extend([i, j]);
                positions.extend([[0.0, 0.0], [d, 0.0]]);
                starts.push(images.len() as u32);
            }
        }
        ClusterCovisibility::from_clusters_with_positions(
            &starts,
            &images,
            None,
            num_images,
            Some(&positions),
            0,
        )
        .unwrap()
    }

    #[test]
    fn displacement_exact_on_forced_two_member_samples() {
        // Cluster 0: (0, 1) at distance 5; cluster 1: (0, 1) at distance 10;
        // cluster 2: (0, 2) at distance 17. Two-member clusters force the
        // sample, so the means are exact whatever the seed.
        let starts = [0u32, 2, 4, 6];
        let images = [0u32, 1, 0, 1, 0, 2];
        let positions = [
            [0.0, 0.0],
            [3.0, 4.0],
            [0.0, 0.0],
            [6.0, 8.0],
            [0.0, 0.0],
            [8.0, 15.0],
        ];
        let cov = ClusterCovisibility::from_clusters_with_positions(
            &starts,
            &images,
            None,
            3,
            Some(&positions),
            42,
        )
        .unwrap();
        let mean = cov.pair_displacement().unwrap();
        let count = cov.pair_displacement_counts().unwrap();
        let at = |m: &[f64], i: usize, j: usize| m[i * 3 + j];
        assert_eq!(at(mean, 0, 1), 7.5);
        assert_eq!(at(mean, 0, 2), 17.0);
        assert_eq!(at(mean, 1, 2), 0.0); // no sample landed
        assert_eq!(count[1], 2);
        assert_eq!(count[2], 1);
        assert_eq!(count[5], 0);
        for i in 0..3 {
            assert_eq!(at(mean, i, i), 0.0);
            for j in 0..3 {
                assert_eq!(at(mean, i, j), at(mean, j, i));
                assert_eq!(count[i * 3 + j], count[j * 3 + i]);
            }
        }
        // Positions leave the shared-cluster counts unchanged.
        assert_eq!(cov.count(0, 1), 2);
        assert_eq!(cov.count(0, 2), 1);
    }

    #[test]
    fn displacement_same_image_pairs_skipped() {
        // Cluster 0's two members both sit in image 0: the forced sample is a
        // same-image pair and is skipped, not resampled.
        let cov = ClusterCovisibility::from_clusters_with_positions(
            &[0, 2, 4],
            &[0, 0, 0, 1],
            None,
            2,
            Some(&[[0.0, 0.0], [9.0, 0.0], [0.0, 0.0], [4.0, 0.0]]),
            0,
        )
        .unwrap();
        let count = cov.pair_displacement_counts().unwrap();
        assert_eq!(count.iter().sum::<u32>(), 2); // cluster 1 only, mirrored
        assert_eq!(cov.pair_displacement().unwrap()[1], 4.0);
    }

    #[test]
    fn displacement_unavailable_without_positions() {
        let cov = from_edges(3, &[(0, 1, 5)]);
        assert!(cov.pair_displacement().is_none());
        assert!(cov.pair_displacement_counts().is_none());
    }

    #[test]
    fn displacement_seeded_determinism() {
        // Clusters with more than two members exercise the RNG; identical
        // seeds must reproduce the tables exactly, and every sample lands
        // (all members sit in distinct images), so the total sample count is
        // the multi-member cluster count for any seed.
        let starts = [0u32, 4, 7, 9];
        let images = [0u32, 1, 2, 3, 1, 2, 3, 0, 2];
        let positions: Vec<[f64; 2]> = (0..9).map(|k| [k as f64 * 3.0, k as f64]).collect();
        let build = |seed| {
            ClusterCovisibility::from_clusters_with_positions(
                &starts,
                &images,
                None,
                4,
                Some(&positions),
                seed,
            )
            .unwrap()
        };
        let (a, b, c) = (build(7), build(7), build(8));
        assert_eq!(a.pair_displacement(), b.pair_displacement());
        assert_eq!(a.pair_displacement_counts(), b.pair_displacement_counts());
        for cov in [&a, &c] {
            let total: u32 = cov.pair_displacement_counts().unwrap().iter().sum();
            assert_eq!(total, 2 * 3); // 3 clusters, each mirrored
        }
    }

    #[test]
    fn displacement_sampling_respects_mask() {
        // Cluster of three members; the image-1 member is masked out, so the
        // forced sample is (image 0, image 2).
        let cov = ClusterCovisibility::from_clusters_with_positions(
            &[0, 3],
            &[0, 1, 2],
            Some(&[true, false, true]),
            3,
            Some(&[[0.0, 0.0], [100.0, 0.0], [5.0, 12.0]]),
            0,
        )
        .unwrap();
        let mean = cov.pair_displacement().unwrap();
        let count = cov.pair_displacement_counts().unwrap();
        assert_eq!(mean[2], 13.0); // (0, 2)
        assert_eq!(count[2], 1);
        assert_eq!(count.iter().sum::<u32>(), 2);
    }

    #[test]
    fn positions_not_parallel_error() {
        assert_eq!(
            ClusterCovisibility::from_clusters_with_positions(
                &[0, 2],
                &[0, 1],
                None,
                2,
                Some(&[[0.0, 0.0]]),
                0,
            ),
            Err(CovisibilityError::PositionsNotParallel {
                members: 2,
                positions: 1
            })
        );
    }

    /// An 8-image chain with geometrically decaying covisibility:
    /// `W[i, j] = 128 >> |i - j|` (64 adjacent, 32 at distance 2, …).
    fn chain8() -> ClusterCovisibility {
        let mut edges = Vec::new();
        for i in 0..8u32 {
            for j in (i + 1)..8 {
                edges.push((i, j, 128 >> (j - i)));
            }
        }
        from_edges(8, &edges)
    }

    #[test]
    fn thin_reproduces_band_selection_on_chain() {
        let cov = chain8();
        // Band [8, 64): adjacent images (64) duplicate, distance-2 (32)
        // stays linked — a stride-2 skeleton.
        assert_eq!(cov.thin(64.0), vec![0, 2, 4, 6]);
        // Band [16, 128): every image keeps its adjacent link.
        assert_eq!(cov.thin(128.0), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        // A tau below every count keeps only the first swept image.
        assert_eq!(cov.thin(0.5), vec![0]);
    }

    #[test]
    fn thin_sweeps_in_decreasing_isolation_with_positions() {
        // W: (0,1)=10, (0,2)=10, (1,2)=3. Displacements: d(0,1)=1,
        // d(0,2)=2, d(1,2)=9 → isolation 0:1, 1:1, 2:9 → sweep [2, 0, 1].
        // Band [1, 8): 2 seeds the kept set, 0 duplicates it (10), 1 links
        // through (1,2)=3. Construction order would keep only image 0.
        let edges = [(0, 1, 10, 1.0), (0, 2, 10, 2.0), (1, 2, 3, 9.0)];
        let cov = from_positioned_edges(3, &edges);
        assert_eq!(cov.thin(8.0), vec![1, 2]);
        // The no-positions fallback sweeps construction order.
        let unpositioned = from_edges(3, &[(0, 1, 10), (0, 2, 10), (1, 2, 3)]);
        assert_eq!(unpositioned.thin(8.0), vec![0]);
    }

    #[test]
    fn thin_is_permutation_invariant_with_positions() {
        // Isolations: [1, 1, 2, 4] — images 0 and 1 tie through their shared
        // minimum edge (the global-minimum edge always ties its endpoints,
        // and ties break by index), so the invariance contract is "up to
        // exact ties": the permutation below preserves the index order
        // within the {0, 1} tie class, and the kept set must then relabel
        // exactly.
        let edges = [
            (0u32, 1u32, 10u32, 1.0),
            (0, 2, 10, 2.0),
            (1, 2, 3, 9.0),
            (2, 3, 20, 4.0),
            (1, 3, 6, 7.0),
        ];
        let cov = from_positioned_edges(4, &edges);
        // Relabel via perm[old] = new; perm[0] = 1 < perm[1] = 3.
        let perm = [1u32, 3, 0, 2];
        let permuted_edges: Vec<_> = edges
            .iter()
            .map(|&(i, j, w, d)| (perm[i as usize], perm[j as usize], w, d))
            .collect();
        let cov_p = from_positioned_edges(4, &permuted_edges);
        for tau in [2.0, 8.0, 16.0, 32.0, 64.0] {
            let base = cov.thin(tau);
            let mut mapped: Vec<u32> = base.iter().map(|&i| perm[i as usize]).collect();
            mapped.sort_unstable();
            assert_eq!(cov_p.thin(tau), mapped, "tau = {tau}");
        }
    }

    #[test]
    fn thin_to_hits_requested_sizes() {
        let cov = chain8();
        // Reachable sizes on the chain over tau in (1, median peak]: 2
        // (small tau keeps the weight-1 distance-7 link), 3 (~17), 4
        // (stride 2). thin_to finds each exactly and returns the closest
        // reachable size at the ends of the sweep.
        for target in 2..=4usize {
            assert_eq!(cov.thin_to(target).len(), target, "target = {target}");
        }
        assert_eq!(cov.thin_to(1).len(), 2); // size 1 needs tau <= 1
        assert_eq!(cov.thin_to(8), vec![0, 2, 4, 6]); // saturates at stride 2
    }

    #[test]
    fn reach_exact_fractions() {
        // 5 images; image 4 shares nothing.
        let cov = from_edges(5, &[(0, 1, 8), (1, 2, 7), (0, 3, 9)]);
        assert_eq!(cov.reach(&[0], 8), 3.0 / 5.0); // 0 (member), 1, 3
        assert_eq!(cov.reach(&[1], 8), 2.0 / 5.0); // 1 (member), 0
        assert_eq!(cov.reach(&[], 8), 0.0);
        // A member counts as reached even with zero covisibility.
        assert_eq!(cov.reach(&[4], 8), 1.0 / 5.0);
        // The whole set reaches everything.
        assert_eq!(cov.reach(&[0, 1, 2, 3, 4], 8), 1.0);
    }

    #[test]
    fn reach_respects_min_shared_boundary() {
        let cov = from_edges(5, &[(0, 1, 8), (1, 2, 7), (0, 3, 9)]);
        // Exactly at the bar counts; one below does not.
        assert_eq!(cov.reach(&[1], 7), 3.0 / 5.0); // adds image 2 (7 >= 7)
        assert_eq!(cov.reach(&[0], 9), 2.0 / 5.0); // drops image 1 (8 < 9)
    }

    // ── Displacement neighborhood (specs/core/pose-verification.md) ────────

    /// A small positioned scene: 4 images, clusters mixing spans, one masked
    /// member, one duplicate-image member.
    fn small_scene() -> (Vec<u32>, Vec<u32>, Vec<[f64; 2]>, Vec<bool>) {
        // Cluster 0: images {0, 1, 2}; cluster 1: {0, 1}; cluster 2: {1, 2}
        // with a duplicate member in image 1; cluster 3: {2, 3};
        // cluster 4: {0, 3} but the image-3 member is masked out.
        let starts = vec![0u32, 3, 5, 8, 10, 12];
        let images = vec![0u32, 1, 2, 0, 1, 1, 1, 2, 2, 3, 0, 3];
        let positions: Vec<[f64; 2]> = (0..12).map(|k| [k as f64 * 2.0, k as f64]).collect();
        let mask = vec![
            true, true, true, true, true, true, true, true, true, true, true, false,
        ];
        (starts, images, positions, mask)
    }

    /// Dense reference: brute-force per-pair shared counts and displacement
    /// means straight from the definition.
    fn dense_reference(
        starts: &[u32],
        images: &[u32],
        positions: &[[f64; 2]],
        mask: Option<&[bool]>,
        n: usize,
    ) -> (Vec<Vec<u32>>, Vec<Vec<f64>>) {
        let mut shared = vec![vec![0u32; n]; n];
        let mut sum = vec![vec![0.0f64; n]; n];
        let mut cnt = vec![vec![0u32; n]; n];
        for c in 0..starts.len() - 1 {
            let rows: Vec<usize> = (starts[c] as usize..starts[c + 1] as usize)
                .filter(|&k| mask.is_none_or(|m| m[k]))
                .collect();
            let mut span: Vec<usize> = rows.iter().map(|&k| images[k] as usize).collect();
            span.sort_unstable();
            span.dedup();
            for (a, &i) in span.iter().enumerate() {
                for &j in &span[a + 1..] {
                    shared[i][j] += 1;
                    shared[j][i] += 1;
                }
            }
            for (a, &ka) in rows.iter().enumerate() {
                for &kb in &rows[a + 1..] {
                    let (ia, ib) = (images[ka] as usize, images[kb] as usize);
                    if ia == ib {
                        continue;
                    }
                    let d = f64::hypot(
                        positions[ka][0] - positions[kb][0],
                        positions[ka][1] - positions[kb][1],
                    );
                    sum[ia][ib] += d;
                    sum[ib][ia] += d;
                    cnt[ia][ib] += 1;
                    cnt[ib][ia] += 1;
                }
            }
        }
        let mean = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if cnt[i][j] > 0 {
                            sum[i][j] / cnt[i][j] as f64
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        (shared, mean)
    }

    #[test]
    fn neighborhood_exact_against_dense_reference() {
        let (starts, images, positions, _) = small_scene();
        let nb =
            DisplacementNeighborhood::from_clusters(&starts, &images, None, 4, &positions).unwrap();
        let (shared, mean) = dense_reference(&starts, &images, &positions, None, 4);
        for i in 0..4u32 {
            for j in 0..4u32 {
                let expected = if i != j && shared[i as usize][j as usize] > 0 {
                    Some((shared[i as usize][j as usize], mean[i as usize][j as usize]))
                } else {
                    None
                };
                assert_eq!(nb.pair(i, j), expected, "pair ({i}, {j})");
            }
        }
        // Shared counts agree with the dense ClusterCovisibility matrix.
        let cov = ClusterCovisibility::from_clusters(&starts, &images, None, 4).unwrap();
        for i in 0..4u32 {
            for j in 0..4u32 {
                assert_eq!(
                    nb.pair(i, j).map(|(s, _)| s).unwrap_or(0),
                    cov.count(i, j),
                    "count ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn neighborhood_nearest_farthest_exact() {
        let (starts, images, positions, _) = small_scene();
        let nb =
            DisplacementNeighborhood::from_clusters(&starts, &images, None, 4, &positions).unwrap();
        let (shared, mean) = dense_reference(&starts, &images, &positions, None, 4);
        for i in 0..4u32 {
            for min_shared in [1u32, 2] {
                // Brute-force ranking from the dense reference.
                let mut cands: Vec<(f64, u32)> = (0..4u32)
                    .filter(|&j| j != i && shared[i as usize][j as usize] >= min_shared)
                    .map(|j| (mean[i as usize][j as usize], j))
                    .collect();
                cands.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let want_near: Vec<u32> = cands.iter().map(|&(_, j)| j).collect();
                assert_eq!(nb.nearest(i, 4, min_shared), want_near, "nearest({i})");
                let mut cands_far = cands.clone();
                cands_far.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                let want_far: Vec<u32> = cands_far.iter().map(|&(_, j)| j).collect();
                assert_eq!(nb.farthest(i, 4, min_shared), want_far, "farthest({i})");
                // Truncation to k.
                assert_eq!(
                    nb.nearest(i, 1, min_shared),
                    want_near[..1.min(want_near.len())]
                );
            }
        }
    }

    #[test]
    fn neighborhood_honors_mask() {
        let (starts, images, positions, mask) = small_scene();
        let nb =
            DisplacementNeighborhood::from_clusters(&starts, &images, Some(&mask), 4, &positions)
                .unwrap();
        // Cluster 4's image-3 member is masked, so pair (0, 3) loses its only
        // vote; pair (2, 3) from cluster 3 survives.
        assert_eq!(nb.pair(0, 3), None);
        assert!(nb.pair(2, 3).is_some());
        let (shared, mean) = dense_reference(&starts, &images, &positions, Some(&mask), 4);
        for i in 0..4u32 {
            for j in 0..4u32 {
                let expected = if i != j && shared[i as usize][j as usize] > 0 {
                    Some((shared[i as usize][j as usize], mean[i as usize][j as usize]))
                } else {
                    None
                };
                assert_eq!(nb.pair(i, j), expected, "pair ({i}, {j})");
            }
        }
    }

    #[test]
    fn neighborhood_duplicate_image_members_count_once_but_displace() {
        // Cluster 2 of the small scene holds two members in image 1 and one
        // in image 2: one shared vote for (1, 2), two displacement samples.
        let (starts, images, positions, _) = small_scene();
        let nb =
            DisplacementNeighborhood::from_clusters(&starts, &images, None, 4, &positions).unwrap();
        let (s, d) = nb.pair(1, 2).unwrap();
        assert_eq!(s, 2); // clusters 0 and 2
                          // Cluster 0 contributes |p1 - p2|; cluster 2 contributes
                          // |p5 - p7| and |p6 - p7| (members 5, 6 in image 1; 7 in image 2).
        let dist = |a: usize, b: usize| {
            f64::hypot(
                positions[a][0] - positions[b][0],
                positions[a][1] - positions[b][1],
            )
        };
        let want = (dist(1, 2) + dist(5, 7) + dist(6, 7)) / 3.0;
        assert!((d - want).abs() < 1e-12);
    }

    #[test]
    fn neighborhood_serialization_round_trips() {
        let (starts, images, positions, mask) = small_scene();
        let nb =
            DisplacementNeighborhood::from_clusters(&starts, &images, Some(&mask), 4, &positions)
                .unwrap();
        let (pi, pj, shared, mean_disp) = nb.to_arrays();
        assert_eq!(pi.len(), nb.num_pairs());
        assert!(pi.iter().zip(&pj).all(|(&i, &j)| i < j));
        assert!(pi
            .iter()
            .zip(&pj)
            .collect::<Vec<_>>()
            .windows(2)
            .all(|w| w[0] < w[1]));
        let back = DisplacementNeighborhood::from_arrays(&pi, &pj, &shared, &mean_disp, 4).unwrap();
        assert_eq!(back, nb);
        // Reversed pair order still reloads to the same substrate.
        let rev = |v: &[u32]| -> Vec<u32> { v.iter().rev().copied().collect() };
        let mean_rev: Vec<f64> = mean_disp.iter().rev().copied().collect();
        let back2 = DisplacementNeighborhood::from_arrays(
            &rev(&pj), // also swap i/j: (j, i) normalizes to (i, j)
            &rev(&pi),
            &rev(&shared),
            &mean_rev,
            4,
        )
        .unwrap();
        assert_eq!(back2, nb);
    }

    #[test]
    fn neighborhood_from_arrays_validation() {
        assert_eq!(
            DisplacementNeighborhood::from_arrays(&[0], &[1, 2], &[3], &[1.0], 3),
            Err(CovisibilityError::PairArraysNotParallel {
                i: 1,
                j: 2,
                shared: 1,
                mean_disp: 1
            })
        );
        assert_eq!(
            DisplacementNeighborhood::from_arrays(&[1], &[1], &[3], &[1.0], 3),
            Err(CovisibilityError::BadPair { i: 1, j: 1 })
        );
        // Duplicate unordered pair (0, 1) given as (0, 1) and (1, 0).
        assert_eq!(
            DisplacementNeighborhood::from_arrays(&[0, 1], &[1, 0], &[3, 4], &[1.0, 2.0], 3),
            Err(CovisibilityError::BadPair { i: 0, j: 1 })
        );
        assert_eq!(
            DisplacementNeighborhood::from_arrays(&[0], &[5], &[3], &[1.0], 3),
            Err(CovisibilityError::ImageIndexOutOfRange {
                index: 5,
                num_images: 3
            })
        );
    }

    #[test]
    fn neighborhood_available_through_cluster_covisibility() {
        let (starts, images, positions, _) = small_scene();
        let cov = ClusterCovisibility::from_clusters_with_positions(
            &starts,
            &images,
            None,
            4,
            Some(&positions),
            0,
        )
        .unwrap();
        let nb = cov.displacement_neighborhood().expect("positions supplied");
        let direct =
            DisplacementNeighborhood::from_clusters(&starts, &images, None, 4, &positions).unwrap();
        assert_eq!(nb, &direct);
        // Without positions the substrate is absent.
        let plain = ClusterCovisibility::from_clusters(&starts, &images, None, 4).unwrap();
        assert!(plain.displacement_neighborhood().is_none());
    }
}
