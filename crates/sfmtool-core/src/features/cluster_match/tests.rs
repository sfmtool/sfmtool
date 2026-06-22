// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the background-floor track-cluster matcher.

use super::*;
use ndarray::ArrayView2;
use rand::rngs::StdRng;
use rand::RngExt;
use rand::SeedableRng;

const DIM: usize = 128;
const N_IMAGES: usize = 4;
const N_POINTS: usize = 6;
const N_BACKGROUND: usize = 20;

/// Synthetic corpus: `N_POINTS` planted 3-D points observed in every one of
/// `N_IMAGES` images (base descriptor + small jitter), plus `N_BACKGROUND`
/// scattered background descriptors per image. Each image's rows start with
/// the planted observations, so a planted row's feature index equals its
/// point id.
fn synthetic_corpus(seed: u64) -> (Vec<u8>, Vec<u32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let bases: Vec<Vec<u8>> = (0..N_POINTS)
        .map(|_| (0..DIM).map(|_| rng.random_range(0..=u8::MAX)).collect())
        .collect();

    let mut corpus = Vec::new();
    let mut image_starts = vec![0u32];
    for _ in 0..N_IMAGES {
        for base in &bases {
            for &v in base {
                let jitter = rng.random_range(-2i16..=2i16);
                corpus.push((v as i16 + jitter).clamp(0, 255) as u8);
            }
        }
        for _ in 0..N_BACKGROUND {
            for _ in 0..DIM {
                corpus.push(rng.random_range(0..=u8::MAX));
            }
        }
        image_starts.push((corpus.len() / DIM) as u32);
    }
    (corpus, image_starts)
}

/// Exact-search params (single leaf spanning the corpus) so the tests exercise
/// the membership rule, not forest recall.
fn exact_params(n: usize) -> BackgroundFloorParams {
    BackgroundFloorParams {
        d: 16,
        forest: KdForestParams {
            num_trees: 1,
            leaf_size: n,
            max_leaf_checks: n,
            ..KdForestParams::balanced()
        },
        ..BackgroundFloorParams::default()
    }
}

fn corpus_view(corpus: &[u8]) -> ArrayView2<'_, u8> {
    ArrayView2::from_shape((corpus.len() / DIM, DIM), corpus).unwrap()
}

#[test]
fn recovers_planted_points_as_clusters() {
    let (corpus, image_starts) = synthetic_corpus(7);
    let n = corpus.len() / DIM;
    let clusters =
        background_floor_clusters(corpus_view(&corpus), &image_starts, &exact_params(n)).unwrap();

    let starts = clusters.cluster_starts.as_slice().unwrap();
    assert_eq!(starts.len() - 1, N_POINTS, "one cluster per planted point");
    for c in 0..starts.len() - 1 {
        let lo = starts[c] as usize;
        let hi = starts[c + 1] as usize;
        assert_eq!(hi - lo, N_IMAGES, "each point observed in every image");
        // All members of a cluster carry the same planted point id (their
        // feature index), so no cluster mixes two planted points.
        let feats = &clusters.member_features.as_slice().unwrap()[lo..hi];
        assert!(feats.iter().all(|&f| f == feats[0]));
        assert!(
            (feats[0] as usize) < N_POINTS,
            "background row in a cluster"
        );
    }
}

#[test]
fn cluster_invariants_hold() {
    let (corpus, image_starts) = synthetic_corpus(11);
    let n = corpus.len() / DIM;
    let params = exact_params(n);
    let clusters = background_floor_clusters(corpus_view(&corpus), &image_starts, &params).unwrap();

    let starts = clusters.cluster_starts.as_slice().unwrap();
    let images = clusters.member_images.as_slice().unwrap();
    let features = clusters.member_features.as_slice().unwrap();

    assert_eq!(starts[0], 0);
    assert!(starts.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(*starts.last().unwrap() as usize, images.len());
    assert_eq!(images.len(), features.len());

    let mut seen = std::collections::HashSet::new();
    for c in 0..starts.len() - 1 {
        let lo = starts[c] as usize;
        let hi = starts[c + 1] as usize;
        let members = &images[lo..hi];
        // Sorted by image, at most one feature per image.
        assert!(members.windows(2).all(|w| w[0] < w[1]));
        // Spans at least min_size images.
        assert!(hi - lo >= params.min_size);
        // Disjoint: no (image, feature) appears in two clusters.
        for m in lo..hi {
            assert!(
                seen.insert((images[m], features[m])),
                "feature claimed twice"
            );
        }
    }
}

#[test]
fn conversion_expands_each_cluster_pairwise() {
    let (corpus, image_starts) = synthetic_corpus(13);
    let n = corpus.len() / DIM;
    let view = corpus_view(&corpus);
    let clusters = background_floor_clusters(view, &image_starts, &exact_params(n)).unwrap();
    let pairs = clusters_to_pair_matches(&clusters, view, &image_starts);

    // Exactly sum C(m, 2) emitted matches.
    let starts = clusters.cluster_starts.as_slice().unwrap();
    let expected: usize = starts
        .windows(2)
        .map(|w| {
            let m = (w[1] - w[0]) as usize;
            m * (m - 1) / 2
        })
        .sum();
    assert_eq!(pairs.match_feature_indexes.nrows(), expected);
    assert_eq!(pairs.match_descriptor_distances.len(), expected);
    let total: u32 = pairs.match_counts.iter().sum();
    assert_eq!(total as usize, expected);

    // Pairs are sorted ascending with i < j.
    let ip = &pairs.image_index_pairs;
    for p in 0..ip.nrows() {
        assert!(ip[[p, 0]] < ip[[p, 1]]);
        if p > 0 {
            assert!((ip[[p - 1, 0]], ip[[p - 1, 1]]) < (ip[[p, 0]], ip[[p, 1]]));
        }
    }

    // One-to-one per image pair and true (non-squared) L2 distances.
    let mut offset = 0usize;
    for p in 0..ip.nrows() {
        let count = pairs.match_counts[p] as usize;
        let mut lo_seen = std::collections::HashSet::new();
        let mut hi_seen = std::collections::HashSet::new();
        for m in offset..offset + count {
            let feat_lo = pairs.match_feature_indexes[[m, 0]];
            let feat_hi = pairs.match_feature_indexes[[m, 1]];
            assert!(lo_seen.insert(feat_lo), "feat_lo repeats within a pair");
            assert!(hi_seen.insert(feat_hi), "feat_hi repeats within a pair");

            let row_lo = (image_starts[ip[[p, 0]] as usize] + feat_lo) as usize;
            let row_hi = (image_starts[ip[[p, 1]] as usize] + feat_hi) as usize;
            let expected_dist = l2_distance(view.row(row_lo), view.row(row_hi));
            assert_eq!(pairs.match_descriptor_distances[m], expected_dist);
        }
        offset += count;
    }
}

#[test]
fn wide_radius_alpha_keeps_invariants() {
    // `alpha >= 1` widens the radius past the background floor itself, so the
    // candidate buffers must be sized for the full query width `d + 1`, not `d`
    // (with a forest self-miss every column can be a kept candidate). Exercise
    // that path and confirm the cluster invariants still hold.
    let (corpus, image_starts) = synthetic_corpus(23);
    let n = corpus.len() / DIM;
    let params = BackgroundFloorParams {
        alpha: 1.5,
        ..exact_params(n)
    };
    let clusters = background_floor_clusters(corpus_view(&corpus), &image_starts, &params).unwrap();

    let starts = clusters.cluster_starts.as_slice().unwrap();
    let images = clusters.member_images.as_slice().unwrap();
    let features = clusters.member_features.as_slice().unwrap();
    assert_eq!(starts[0], 0);
    assert_eq!(*starts.last().unwrap() as usize, images.len());
    let mut seen = std::collections::HashSet::new();
    for c in 0..starts.len() - 1 {
        let lo = starts[c] as usize;
        let hi = starts[c + 1] as usize;
        assert!(images[lo..hi].windows(2).all(|w| w[0] < w[1]));
        assert!(hi - lo >= params.min_size);
        for m in lo..hi {
            assert!(
                seen.insert((images[m], features[m])),
                "feature claimed twice"
            );
        }
    }
    // The planted points still cluster across all images.
    assert!(starts.len() > N_POINTS);
}

#[test]
fn min_size_filters_clusters_below_threshold() {
    // Each planted point spans exactly N_IMAGES images; require more and no
    // cluster can qualify, so the matcher records nothing.
    let (corpus, image_starts) = synthetic_corpus(29);
    let n = corpus.len() / DIM;
    let params = BackgroundFloorParams {
        min_size: N_IMAGES + 1,
        ..exact_params(n)
    };
    let clusters = background_floor_clusters(corpus_view(&corpus), &image_starts, &params).unwrap();
    assert_eq!(clusters.cluster_starts.as_slice().unwrap(), &[0]);
    assert!(clusters.member_images.is_empty());
    assert!(clusters.member_features.is_empty());
}

#[test]
fn validation_errors() {
    let params = BackgroundFloorParams::default();

    let empty: Vec<u8> = Vec::new();
    let view = ArrayView2::from_shape((0, DIM), &empty).unwrap();
    assert_eq!(
        background_floor_clusters(view, &[0], &params).unwrap_err(),
        ClusterMatchError::EmptyCorpus
    );

    // N <= d: the floor rank does not exist.
    let small = vec![0u8; 10 * DIM];
    let view = ArrayView2::from_shape((10, DIM), &small).unwrap();
    assert_eq!(
        background_floor_clusters(view, &[0, 5, 10], &params).unwrap_err(),
        ClusterMatchError::CorpusSmallerThanFloor { n: 10, d: params.d }
    );

    let (corpus, _) = synthetic_corpus(3);
    let n = corpus.len() / DIM;
    let view = corpus_view(&corpus);
    for bad in [
        vec![1u32, n as u32],                 // does not start at 0
        vec![0u32, (n / 2) as u32],           // does not end at N
        vec![0u32, n as u32, (n / 2) as u32], // decreasing
        vec![0u32],                           // no images
    ] {
        assert_eq!(
            background_floor_clusters(view, &bad, &exact_params(n)).unwrap_err(),
            ClusterMatchError::BadOffsets { n }
        );
    }
}

#[test]
fn deterministic_across_runs() {
    let (corpus, image_starts) = synthetic_corpus(17);
    let n = corpus.len() / DIM;
    let view = corpus_view(&corpus);
    let params = exact_params(n);

    let c1 = background_floor_clusters(view, &image_starts, &params).unwrap();
    let c2 = background_floor_clusters(view, &image_starts, &params).unwrap();
    assert_eq!(c1.cluster_starts, c2.cluster_starts);
    assert_eq!(c1.member_images, c2.member_images);
    assert_eq!(c1.member_features, c2.member_features);

    let p1 = clusters_to_pair_matches(&c1, view, &image_starts);
    let p2 = clusters_to_pair_matches(&c2, view, &image_starts);
    assert_eq!(p1.image_index_pairs, p2.image_index_pairs);
    assert_eq!(p1.match_counts, p2.match_counts);
    assert_eq!(p1.match_feature_indexes, p2.match_feature_indexes);
    assert_eq!(p1.match_descriptor_distances, p2.match_descriptor_distances);
}
