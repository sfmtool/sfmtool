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
//! See `specs/core/features/cluster-covisibility.md` for the design and the seed-group
//! algorithm's determinism contract, `specs/core/features/covisibility-selection.md`
//! for the selection queries built on top (pair displacement, banded thinning,
//! reach), and `specs/core/geometry/pose-verification.md` for the sparse
//! displacement-neighborhood substrate ([`DisplacementNeighborhood`]).

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
    /// `specs/core/geometry/pose-verification.md`.
    pub fn displacement_neighborhood(&self) -> Option<&DisplacementNeighborhood> {
        self.neighborhood.as_ref()
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

mod displacement;
mod selection;

pub use displacement::DisplacementNeighborhood;
use displacement::DisplacementTables;

#[cfg(test)]
mod tests;
