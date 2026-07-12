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
//! algorithm's determinism contract.

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
    /// A member's image index is out of range.
    ImageIndexOutOfRange { index: u32, num_images: usize },
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
            Self::ImageIndexOutOfRange { index, num_images } => write!(
                f,
                "member image index {index} is out of range for {num_images} images"
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

/// Symmetric per-image-pair shared-cluster counts (zero diagonal).
///
/// `W[i, j]` = number of clusters with an accepted member in image `i` and
/// an accepted member in image `j`; each cluster contributes at most 1 to
/// any pair, and clusters spanning fewer than 2 accepted images contribute
/// nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterCovisibility {
    num_images: usize,
    /// Row-major `(num_images, num_images)` counts.
    counts: Vec<u32>,
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
    pub fn from_clusters(
        cluster_starts: &[u32],
        member_images: &[u32],
        member_accepted: Option<&[bool]>,
        num_images: usize,
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
        if let Some(&bad) = member_images.iter().find(|&&i| i as usize >= num_images) {
            return Err(CovisibilityError::ImageIndexOutOfRange {
                index: bad,
                num_images,
            });
        }

        let mut counts = vec![0u32; num_images * num_images];
        let mut span: Vec<u32> = Vec::new();
        for c in 0..cluster_starts.len() - 1 {
            let lo = cluster_starts[c] as usize;
            let hi = cluster_starts[c + 1] as usize;
            span.clear();
            span.extend(
                (lo..hi)
                    .filter(|&k| member_accepted.is_none_or(|mask| mask[k]))
                    .map(|k| member_images[k]),
            );
            span.sort_unstable();
            span.dedup();
            for (a, &i) in span.iter().enumerate() {
                for &j in &span[a + 1..] {
                    counts[i as usize * num_images + j as usize] += 1;
                    counts[j as usize * num_images + i as usize] += 1;
                }
            }
        }

        Ok(Self { num_images, counts })
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
}
