// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

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
    let cov =
        ClusterCovisibility::from_clusters(&[0, 3, 5, 6], &[0, 1, 2, 0, 2, 1], None, 4).unwrap();
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

/// The image lists of a seed-group run, for the expectations that predate
/// the enriched yield.
fn group_images(cov: &ClusterCovisibility, params: &SeedImageGroupParams) -> Vec<Vec<u32>> {
    cov.seed_image_groups(params)
        .map(|g| g.images)
        .collect::<Vec<_>>()
}

#[test]
fn seed_image_groups_two_disjoint_triangles() {
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
    let params = SeedImageGroupParams {
        group_size: 3,
        min_shared: 8,
    };
    let groups: Vec<_> = cov.seed_image_groups(&params).collect();
    assert_eq!(
        groups.iter().map(|g| g.images.clone()).collect::<Vec<_>>(),
        vec![vec![0, 1, 2], vec![3, 4, 5]]
    );
    // Each group reports the founding edge and its weight: the strongest
    // remaining pair at that step, lexicographically smallest on the tie.
    assert_eq!(groups[0].seed_pair, (0, 1));
    assert_eq!(groups[0].seed_shared, 10);
    assert_eq!(groups[1].seed_pair, (3, 4));
    assert_eq!(groups[1].seed_shared, 9);
}

#[test]
fn seed_edge_tie_breaks_lexicographically() {
    // Two equal-strength edges: (0, 1) wins over (2, 3).
    let cov = from_edges(4, &[(2, 3, 10), (0, 1, 10)]);
    let params = SeedImageGroupParams::default();
    let groups: Vec<_> = cov.seed_image_groups(&params).collect();
    assert_eq!(
        groups.iter().map(|g| g.images.clone()).collect::<Vec<_>>(),
        vec![vec![0, 1], vec![2, 3]]
    );
    assert_eq!(groups[0].seed_pair, (0, 1));
    assert_eq!(groups[1].seed_pair, (2, 3));
}

#[test]
fn seed_pair_is_the_groups_maximum_shared_pair() {
    // A weighted graph with a non-trivial tie structure: the founding edge
    // ties with several in-group pairs, and the extension order is decided
    // by the minimum-vs-group criterion rather than by the pair weights, so
    // the invariant is not simply "the first pair happens to be largest".
    let mut edges = Vec::new();
    let mut state = 0x1234_5678u32;
    for i in 0..14u32 {
        for j in (i + 1)..14 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // A small weight alphabet forces many exact ties.
            edges.push((i, j, 4 + (state >> 29)));
        }
    }
    let cov = from_edges(14, &edges);
    let params = SeedImageGroupParams {
        group_size: 5,
        min_shared: 5,
    };
    let groups: Vec<_> = cov.seed_image_groups(&params).collect();
    assert!(groups.len() >= 2, "fixture must yield multiple groups");
    for g in &groups {
        let (i, j) = g.seed_pair;
        assert!(i < j, "seed pair is ordered");
        assert!(g.images.contains(&i) && g.images.contains(&j));
        assert_eq!(cov.count(i, j), g.seed_shared);
        assert!(g.seed_shared >= params.min_shared);
        for a in 0..g.images.len() {
            for b in (a + 1)..g.images.len() {
                assert!(
                    cov.count(g.images[a], g.images[b]) <= g.seed_shared,
                    "pair ({}, {}) exceeds the seed edge",
                    g.images[a],
                    g.images[b]
                );
            }
        }
        // The same invariant restated through the enriched yield: the seed
        // pair's condensed entry is `seed_shared`, and nothing exceeds it.
        assert_eq!(
            g.pair_shared[condensed_index(&g.images, i, j)],
            g.seed_shared
        );
        assert_eq!(g.pair_shared.iter().copied().max(), Some(g.seed_shared));
    }
}

/// The condensed upper-triangle index `SeedImageGroup::pair_shared` and
/// `SeedImageGroup::pair_displacement` are ordered by, for the pair of image
/// indexes `(x, y)` — written from the documented formula rather than by
/// re-deriving the enumeration.
fn condensed_index(images: &[u32], x: u32, y: u32) -> usize {
    let pos = |v: u32| images.iter().position(|&g| g == v).expect("image in group");
    let (a, b) = (pos(x.min(y)), pos(x.max(y)));
    let len = images.len();
    a * (2 * len - a - 1) / 2 + (b - a - 1)
}

#[test]
fn pair_shared_is_the_condensed_upper_triangle() {
    // Six distinct edge weights on four images, so every condensed entry is
    // distinguishable from every other: any ordering mistake shows up.
    let cov = from_edges(
        4,
        &[
            (0, 1, 20),
            (0, 2, 19),
            (0, 3, 18),
            (1, 2, 17),
            (1, 3, 16),
            (2, 3, 15),
        ],
    );
    let params = SeedImageGroupParams {
        group_size: 4,
        min_shared: 8,
    };
    let groups: Vec<_> = cov.seed_image_groups(&params).collect();
    assert_eq!(groups.len(), 1);
    let g = &groups[0];
    assert_eq!(g.images, vec![0, 1, 2, 3]);
    // The documented order: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
    assert_eq!(g.pair_shared, vec![20, 19, 18, 17, 16, 15]);
    // And each entry against a direct count(i, j) at its formula index.
    for a in 0..g.images.len() {
        for b in (a + 1)..g.images.len() {
            let (x, y) = (g.images[a], g.images[b]);
            assert_eq!(
                g.pair_shared[condensed_index(&g.images, x, y)],
                cov.count(x, y),
                "pair ({x}, {y})"
            );
        }
    }
    // Consistency with the founding edge.
    assert_eq!(g.seed_pair, (0, 1));
    assert_eq!(
        g.pair_shared[condensed_index(&g.images, 0, 1)],
        g.seed_shared
    );
    assert_eq!(g.pair_shared.iter().copied().max(), Some(g.seed_shared));
    // No positions were supplied, so there is no displacement to report.
    assert_eq!(g.pair_displacement, None);
}

#[test]
fn pair_displacement_reads_the_neighborhood() {
    // Distinct per-edge displacements over the same four-image graph: each
    // two-member cluster forces its pair, so the neighborhood's exhaustive
    // mean for an edge is exactly that edge's distance.
    let edges = [
        (0u32, 1u32, 20u32, 5.0f32),
        (0, 2, 19, 7.0),
        (0, 3, 18, 9.0),
        (1, 2, 17, 11.0),
        (1, 3, 16, 13.0),
        (2, 3, 15, 3.0),
    ];
    let cov = from_positioned_edges(4, &edges);
    let params = SeedImageGroupParams {
        group_size: 4,
        min_shared: 8,
    };
    let g = cov.seed_image_groups(&params).next().unwrap();
    assert_eq!(g.images, vec![0, 1, 2, 3]);
    let disp = g.pair_displacement.as_ref().expect("positioned build");
    assert_eq!(disp.len(), g.pair_shared.len());
    assert_eq!(*disp, vec![5.0, 7.0, 9.0, 11.0, 13.0, 3.0]);
    // Every entry is the neighborhood's own answer for that pair, and the
    // exhaustive means differ from the sampled tables' storage layout.
    let nb = cov.displacement_neighborhood().expect("positioned build");
    for a in 0..g.images.len() {
        for b in (a + 1)..g.images.len() {
            let (x, y) = (g.images[a], g.images[b]);
            let (shared, mean) = nb.pair(x, y).expect("realized pair");
            assert_eq!(
                disp[condensed_index(&g.images, x, y)],
                mean,
                "pair ({x}, {y})"
            );
            assert_eq!(g.pair_shared[condensed_index(&g.images, x, y)], shared);
        }
    }
}

#[test]
fn pair_displacement_reads_zero_for_an_unrealized_pair() {
    // `min_shared = 0` lets the extension take an image sharing nothing with
    // the group, so the group holds a pair the neighborhood never realized.
    let cov = from_positioned_edges(3, &[(0, 1, 5, 4.0)]);
    let params = SeedImageGroupParams {
        group_size: 3,
        min_shared: 0,
    };
    let g = cov.seed_image_groups(&params).next().unwrap();
    assert_eq!(g.images, vec![0, 1, 2]);
    assert_eq!(g.pair_shared, vec![5, 0, 0]);
    let nb = cov.displacement_neighborhood().expect("positioned build");
    assert_eq!(nb.pair(0, 2), None);
    assert_eq!(g.pair_displacement, Some(vec![4.0, 0.0, 0.0]));
}

#[test]
fn pair_displacement_absent_without_positions() {
    let cov = from_edges(3, &[(0, 1, 10), (0, 2, 10), (1, 2, 10)]);
    let params = SeedImageGroupParams {
        group_size: 3,
        min_shared: 8,
    };
    for g in cov.seed_image_groups(&params) {
        assert_eq!(g.pair_displacement, None);
        assert_eq!(
            g.pair_shared.len(),
            g.images.len() * (g.images.len() - 1) / 2
        );
    }
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
    let params = SeedImageGroupParams {
        group_size: 3,
        min_shared: 8,
    };
    assert_eq!(group_images(&cov, &params), vec![vec![0, 1, 2]]);
}

#[test]
fn star_topology_does_not_form_a_group() {
    // Hub 0 strongly connected to 1, 2, 3; no spoke-spoke edges. The
    // minimum-vs-group criterion stops every extension, so only the
    // strongest hub edge pair is ever yielded.
    let cov = from_edges(4, &[(0, 1, 10), (0, 2, 10), (0, 3, 10)]);
    let params = SeedImageGroupParams {
        group_size: 4,
        min_shared: 8,
    };
    assert_eq!(group_images(&cov, &params), vec![vec![0, 1]]);
}

#[test]
fn extension_stops_below_min_shared_but_yields_partial_group() {
    // {0, 1, 2} is mutually strong; 3 attaches to 0 and 1 only weakly.
    let cov = from_edges(4, &[(0, 1, 10), (0, 2, 9), (1, 2, 9), (0, 3, 4), (1, 3, 4)]);
    let params = SeedImageGroupParams {
        group_size: 4,
        min_shared: 8,
    };
    assert_eq!(group_images(&cov, &params), vec![vec![0, 1, 2]]);
}

#[test]
fn iterator_ends_when_strongest_edge_below_min_shared() {
    let cov = from_edges(3, &[(0, 1, 5), (1, 2, 3)]);
    let params = SeedImageGroupParams::default(); // min_shared = 8
    assert_eq!(cov.seed_image_groups(&params).count(), 0);
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
    let params = SeedImageGroupParams {
        group_size: 3,
        min_shared: 8,
    };
    assert_eq!(group_images(&cov, &params), vec![vec![0, 1, 2], vec![3, 4]]);
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
    let params = SeedImageGroupParams {
        group_size: 4,
        min_shared: 3,
    };
    let all: Vec<_> = cov.seed_image_groups(&params).collect();
    assert!(all.len() >= 2, "fixture must yield multiple groups");
    for k in 1..=all.len() {
        let prefix: Vec<_> = cov.seed_image_groups(&params).take(k).collect();
        assert_eq!(prefix, all[..k]);
    }
}

#[test]
fn determinism() {
    let cov = from_edges(6, &[(0, 1, 9), (1, 2, 9), (0, 2, 9), (3, 4, 8)]);
    let params = SeedImageGroupParams::default();
    let a: Vec<_> = cov.seed_image_groups(&params).collect();
    let b: Vec<_> = cov.seed_image_groups(&params).collect();
    assert_eq!(a, b);
    assert_eq!(
        a.iter().map(|g| g.images.clone()).collect::<Vec<_>>(),
        vec![vec![0, 1, 2], vec![3, 4]]
    );
}

// ── Selection queries (specs/core/features/covisibility-selection.md) ──────────

/// Synthesize positioned two-member clusters: edge `(i, j, w, d)` becomes
/// `w` clusters, each holding one member in image `i` at the origin and
/// one in image `j` at distance `d`. Two-member clusters make the sampled
/// displacement pass deterministic regardless of seed (the pair is
/// forced), so `mean[i][j] == d` and `count[i][j] == w` exactly.
fn from_positioned_edges(num_images: usize, edges: &[(u32, u32, u32, f32)]) -> ClusterCovisibility {
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
    let positions: Vec<[f32; 2]> = (0..9).map(|k| [k as f32 * 3.0, k as f32]).collect();
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

// ── Displacement neighborhood (specs/core/geometry/pose-verification.md) ────────

/// A small positioned scene: 4 images, clusters mixing spans, one masked
/// member, one duplicate-image member.
fn small_scene() -> (Vec<u32>, Vec<u32>, Vec<[f32; 2]>, Vec<bool>) {
    // Cluster 0: images {0, 1, 2}; cluster 1: {0, 1}; cluster 2: {1, 2}
    // with a duplicate member in image 1; cluster 3: {2, 3};
    // cluster 4: {0, 3} but the image-3 member is masked out.
    let starts = vec![0u32, 3, 5, 8, 10, 12];
    let images = vec![0u32, 1, 2, 0, 1, 1, 1, 2, 2, 3, 0, 3];
    let positions: Vec<[f32; 2]> = (0..12).map(|k| [k as f32 * 2.0, k as f32]).collect();
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
    positions: &[[f32; 2]],
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
                    positions[ka][0] as f64 - positions[kb][0] as f64,
                    positions[ka][1] as f64 - positions[kb][1] as f64,
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
    let nb = DisplacementNeighborhood::from_clusters(&starts, &images, Some(&mask), 4, &positions)
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
            positions[a][0] as f64 - positions[b][0] as f64,
            positions[a][1] as f64 - positions[b][1] as f64,
        )
    };
    let want = (dist(1, 2) + dist(5, 7) + dist(6, 7)) / 3.0;
    assert!((d - want).abs() < 1e-12);
}

#[test]
fn neighborhood_serialization_round_trips() {
    let (starts, images, positions, mask) = small_scene();
    let nb = DisplacementNeighborhood::from_clusters(&starts, &images, Some(&mask), 4, &positions)
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

/// A fleet-shaped synthetic scene: `n_clusters` band-diagonal clusters over
/// `n_images` images (each anchors one image and takes the next `span - 1`
/// around the ring, the pair-space shape of a sequential capture), spans
/// 2..=8, one member in eight duplicated into its cluster's anchor image, and
/// pseudo-random pixel positions. Every draw comes off one seeded
/// [`SplitMix64`] stream, so the arrays are the same on every run and every
/// platform.
fn fleet_scene(
    n_images: usize,
    n_clusters: usize,
    seed: u64,
) -> (Vec<u32>, Vec<u32>, Vec<[f32; 2]>, Vec<bool>) {
    let mut rng = SplitMix64::new(seed);
    let mut starts = vec![0u32];
    let mut images: Vec<u32> = Vec::new();
    let mut positions: Vec<[f32; 2]> = Vec::new();
    let mut mask: Vec<bool> = Vec::new();
    for _ in 0..n_clusters {
        let span = 2 + rng.below(7);
        let anchor = rng.below(n_images);
        for t in 0..span {
            images.push(((anchor + t) % n_images) as u32);
        }
        if rng.below(8) == 0 {
            images.push(anchor as u32);
        }
        while positions.len() < images.len() {
            positions.push([
                rng.below(1 << 21) as f32 / 1024.0,
                rng.below(1 << 20) as f32 / 1024.0,
            ]);
            // ~1 member in 16 rejected, so the masked arm exercises clusters
            // that lose members without losing their whole span.
            mask.push(rng.below(16) != 0);
        }
        starts.push(images.len() as u32);
    }
    (starts, images, positions, mask)
}

/// One realized pair with its finished statistics: `((i, j), (shared count,
/// mean displacement))`.
type PairStat = ((u32, u32), (u32, f64));

/// The pre-dense-slot accumulation, kept verbatim as the equality reference:
/// one `HashMap<(u32, u32), _>` entry per realized pair, shared votes from the
/// deduplicated span and displacement sums over every accepted cross-image
/// member pair. Returns per-pair `(shared, mean displacement)` keyed by the
/// ordered pair `i < j`.
fn hashmap_reference(
    starts: &[u32],
    images: &[u32],
    positions: &[[f32; 2]],
    mask: Option<&[bool]>,
) -> std::collections::HashMap<(u32, u32), (u32, f64)> {
    let mut pairs: std::collections::HashMap<(u32, u32), (u32, f64, u32)> =
        std::collections::HashMap::new();
    let mut rows: Vec<usize> = Vec::new();
    let mut span: Vec<u32> = Vec::new();
    for c in 0..starts.len() - 1 {
        let (lo, hi) = (starts[c] as usize, starts[c + 1] as usize);
        rows.clear();
        rows.extend((lo..hi).filter(|&k| mask.is_none_or(|m| m[k])));
        span.clear();
        span.extend(rows.iter().map(|&k| images[k]));
        span.sort_unstable();
        span.dedup();
        for (a, &i) in span.iter().enumerate() {
            for &j in &span[a + 1..] {
                pairs.entry((i, j)).or_default().0 += 1;
            }
        }
        for (a, &ka) in rows.iter().enumerate() {
            for &kb in &rows[a + 1..] {
                let (ia, ib) = (images[ka], images[kb]);
                if ia == ib {
                    continue;
                }
                let d = f64::hypot(
                    positions[ka][0] as f64 - positions[kb][0] as f64,
                    positions[ka][1] as f64 - positions[kb][1] as f64,
                );
                let e = pairs.entry((ia.min(ib), ia.max(ib))).or_default();
                e.1 += d;
                e.2 += 1;
            }
        }
    }
    pairs
        .into_iter()
        .map(|(k, (shared, sum, n))| (k, (shared, if n > 0 { sum / n as f64 } else { 0.0 })))
        .collect()
}

/// The dense pair-slot index must reproduce the hash-map accumulation it
/// replaced to the last bit, at a scale where hash order and first-touch order
/// are thoroughly different: every pair, count, mean (compared by `to_bits`)
/// and the CSR row layout `neighbors` walks.
#[test]
fn neighborhood_matches_hashmap_reference_at_scale() {
    let (starts, images, positions, mask) = fleet_scene(300, 20_000, 0x5eed);
    assert!(
        images.len() > 80_000,
        "scene too small to be a regression net"
    );
    for (arm, m) in [("unmasked", None), ("masked", Some(mask.as_slice()))] {
        let nb =
            DisplacementNeighborhood::from_clusters(&starts, &images, m, 300, &positions).unwrap();
        let want = hashmap_reference(&starts, &images, &positions, m);
        assert!(want.len() > 2000, "{arm}: too few realized pairs to matter");
        assert_eq!(nb.num_pairs(), want.len(), "{arm}: pair count");

        // Serialized pair arrays, exactly: order, keys, counts, and the mean
        // bit pattern (an f64 running sum reordered would differ here).
        let (pi, pj, shared, mean) = nb.to_arrays();
        let mut expect: Vec<PairStat> = want.iter().map(|(&k, &v)| (k, v)).collect();
        expect.sort_unstable_by_key(|&(k, _)| k);
        assert_eq!(pi.len(), expect.len());
        for (k, &((i, j), (s, d))) in expect.iter().enumerate() {
            assert_eq!((pi[k], pj[k]), (i, j), "{arm}: pair {k}");
            assert_eq!(shared[k], s, "{arm}: shared for ({i}, {j})");
            assert_eq!(
                mean[k].to_bits(),
                d.to_bits(),
                "{arm}: mean displacement for ({i}, {j})"
            );
        }

        // CSR structure: each row's partners, ascending, with their stats.
        for i in 0..300u32 {
            let mut want_row: Vec<(u32, u32, f64)> = want
                .iter()
                .filter_map(|(&(a, b), &(s, d))| match (a, b) {
                    (a, b) if a == i => Some((b, s, d)),
                    (a, b) if b == i => Some((a, s, d)),
                    _ => None,
                })
                .collect();
            want_row.sort_unstable_by_key(|&(j, _, _)| j);
            let got: Vec<(u32, u32, f64)> = nb.neighbors(i).collect();
            assert_eq!(got.len(), want_row.len(), "{arm}: row {i} length");
            for (g, w) in got.iter().zip(&want_row) {
                assert_eq!((g.0, g.1), (w.0, w.1), "{arm}: row {i} entry");
                assert_eq!(g.2.to_bits(), w.2.to_bits(), "{arm}: row {i} mean");
            }
        }
    }
}

#[test]
fn neighborhood_edge_scales() {
    // Zero images: the slot array is empty and no cluster can address it.
    let nb = DisplacementNeighborhood::from_clusters(&[0], &[], None, 0, &[]).unwrap();
    assert_eq!((nb.num_images(), nb.num_pairs()), (0, 0));
    assert_eq!(nb.to_arrays(), (vec![], vec![], vec![], vec![]));
    // One image: slots exist, but no cross-image pair ever realizes.
    let nb = DisplacementNeighborhood::from_clusters(
        &[0, 2],
        &[0, 0],
        None,
        1,
        &[[0.0, 0.0], [3.0, 4.0]],
    )
    .unwrap();
    assert_eq!(nb.num_pairs(), 0);
    // Above the dense bound the slot array is refused, as for the matrix.
    assert_eq!(
        DisplacementNeighborhood::from_clusters(&[0], &[], None, MAX_DENSE_IMAGES + 1, &[]),
        Err(CovisibilityError::TooManyImages {
            num_images: MAX_DENSE_IMAGES + 1
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
