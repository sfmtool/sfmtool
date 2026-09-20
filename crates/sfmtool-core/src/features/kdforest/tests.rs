// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the randomized kd-tree forest.

use super::*;
use crate::features::feature_match::descriptor::descriptor_distance_l2_squared;
use rand::rngs::StdRng;
use rand::RngExt;
use rand::SeedableRng;

#[test]
fn balanced_default_uses_sixteen_feature_leaves() {
    assert_eq!(KdForestParams::balanced().leaf_size, 16);
    assert_eq!(KdForestParams::default().leaf_size, 16);
}

/// Random `u8` point set, flat row-major.
fn random_u8(n: usize, dim: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim)
        .map(|_| rng.random_range(0..=u8::MAX))
        .collect()
}

/// Brute-force exact nearest neighbor (top-1) index for one query.
fn brute_force_nn(points: &[u8], dim: usize, query: &[u8]) -> u32 {
    let n = points.len() / dim;
    let mut best = 0u32;
    let mut best_d = i64::MAX;
    for i in 0..n {
        let d = descriptor_distance_l2_squared(query, &points[i * dim..(i + 1) * dim]);
        if d < best_d {
            best_d = d;
            best = i as u32;
        }
    }
    best
}

/// Exact squared distance to the true nearest neighbor of `query`.
fn brute_force_min_distsq(points: &[u8], dim: usize, query: &[u8]) -> i64 {
    let n = points.len() / dim;
    (0..n)
        .map(|i| descriptor_distance_l2_squared(query, &points[i * dim..(i + 1) * dim]))
        .min()
        .unwrap_or(i64::MAX)
}

/// Top-1 recall of the forest vs exact brute force, distance-based so that ties
/// (a different point at the same distance) count as hits. The forest is
/// approximate (FLANN-style additive bound), so tests assert a high recall
/// threshold rather than exactness — except the single-leaf case, which is
/// exact by construction.
fn recall_at_1(
    forest: &KdForestU8,
    points: &[u8],
    dim: usize,
    queries: &[u8],
    budget: usize,
) -> f64 {
    let nq = queries.len() / dim;
    let hits = queries
        .chunks(dim)
        .filter(|q| {
            let got = forest.search(q, 1, budget, None);
            let exact = brute_force_min_distsq(points, dim, q) as f32;
            !got.is_empty() && (got[0].dist_sq - exact).abs() <= 1.0
        })
        .count();
    hits as f64 / nq as f64
}

#[test]
fn single_leaf_search_is_exact() {
    // The genuine exactness ceiling: with one tree whose root is a single leaf
    // (leaf_size >= n), the search scans every point with no pruning, so it is
    // exact by construction. This validates the bounded result set, distance
    // reporting, and k-selection independent of the approximate tree descent.
    let dim = 16;
    let n = 200;
    let points = random_u8(n, dim, 42);
    let params = KdForestParams {
        num_trees: 1,
        split_dim_candidates: 5,
        leaf_size: n,
        max_leaf_checks: n,
        seed: 7,
    };
    let forest = KdForestU8::build(&points, n, dim, params, &Progress::none())
        .expect("nothing asked it to stop");

    let queries = random_u8(50, dim, 99);
    for q in queries.chunks(dim) {
        let nbrs = forest.search(q, 1, n, None);
        assert_eq!(nbrs.len(), 1);
        let exact = brute_force_min_distsq(&points, dim, q) as f32;
        assert_eq!(nbrs[0].dist_sq, exact);
    }
}

#[test]
fn deep_tree_full_budget_high_recall() {
    // A real (deep) tree with an exhaustive budget recovers almost all true
    // neighbors. The additive priority bound is approximate (it can over-prune
    // when an axis re-splits), so this asserts high recall, not exactness.
    let dim = 16;
    let n = 200;
    let points = random_u8(n, dim, 42);
    let params = KdForestParams {
        num_trees: 1,
        split_dim_candidates: 5,
        leaf_size: 1,
        max_leaf_checks: n,
        seed: 7,
    };
    let forest = KdForestU8::build(&points, n, dim, params, &Progress::none())
        .expect("nothing asked it to stop");
    let queries = random_u8(50, dim, 99);
    assert!(recall_at_1(&forest, &points, dim, &queries, n) >= 0.95);
}

#[test]
fn many_trees_full_budget_high_recall() {
    // Multiple trees with an exhaustive budget: each shared point is scored at
    // most once (a broken cross-tree dedup that dropped points would tank
    // recall well below this threshold).
    let dim = 32;
    let n = 300;
    let points = random_u8(n, dim, 1234);
    let params = KdForestParams {
        num_trees: 6,
        split_dim_candidates: 5,
        leaf_size: 8,
        max_leaf_checks: n,
        seed: 3,
    };
    let forest = KdForestU8::build(&points, n, dim, params, &Progress::none())
        .expect("nothing asked it to stop");
    let queries = random_u8(40, dim, 555);
    assert!(recall_at_1(&forest, &points, dim, &queries, n) >= 0.95);
}

#[test]
fn reported_distance_matches_brute_force() {
    let dim = 24;
    let n = 150;
    let points = random_u8(n, dim, 88);
    // Single leaf => the exact-duplicate query is guaranteed to find itself.
    let params = KdForestParams {
        leaf_size: n,
        ..KdForestParams::balanced()
    };
    let forest = KdForestU8::build(&points, n, dim, params, &Progress::none())
        .expect("nothing asked it to stop");
    let q = &points[10 * dim..11 * dim]; // a query equal to an indexed point
    let nbrs = forest.search(q, 1, n, None);
    assert_eq!(nbrs[0].index, 10);
    assert_eq!(nbrs[0].dist_sq, 0.0);

    let q2 = random_u8(1, dim, 17);
    let nbrs2 = forest.search(&q2, 1, n, None);
    let expected = descriptor_distance_l2_squared(
        &q2,
        &points[nbrs2[0].index as usize * dim..(nbrs2[0].index as usize + 1) * dim],
    ) as f32;
    assert_eq!(nbrs2[0].dist_sq, expected);
}

#[test]
fn determinism_same_seed() {
    let dim = 20;
    let n = 250;
    let points = random_u8(n, dim, 2024);
    let queries = random_u8(60, dim, 4040);

    let f1 = KdForestU8::build(
        &points,
        n,
        dim,
        KdForestParams::balanced(),
        &Progress::none(),
    )
    .expect("nothing asked it to stop");
    let f2 = KdForestU8::build(
        &points,
        n,
        dim,
        KdForestParams::balanced(),
        &Progress::none(),
    )
    .expect("nothing asked it to stop");

    let r1 = f1.search_batch(&queries, 60, 3, 100, None);
    let r2 = f2.search_batch(&queries, 60, 3, 100, None);
    assert_eq!(r1, r2, "same seed must give identical results");
}

#[test]
fn precision_monotone_in_budget() {
    // Measured top-1 precision must not decrease as the budget grows.
    let dim = 32;
    let n = 500;
    let points = random_u8(n, dim, 314);
    let forest = KdForestU8::build(
        &points,
        n,
        dim,
        KdForestParams::balanced(),
        &Progress::none(),
    )
    .expect("nothing asked it to stop");

    let nq = 120;
    let queries = random_u8(nq, dim, 271);
    let exact: Vec<u32> = queries
        .chunks(dim)
        .map(|q| brute_force_nn(&points, dim, q))
        .collect();

    let precision_at = |budget: usize| -> f64 {
        let got = forest.search_batch(&queries, nq, 1, budget, None);
        let hits = (0..nq).filter(|&i| got[i] == exact[i]).count();
        hits as f64 / nq as f64
    };

    let budgets = [4usize, 16, 64, 256, n];
    let mut prev = 0.0;
    for &b in &budgets {
        let p = precision_at(b);
        assert!(
            p + 1e-9 >= prev,
            "precision dropped from {prev} to {p} at budget {b}"
        );
        prev = p;
    }
    // An exhaustive budget recovers (almost) all true neighbors. The bound is
    // approximate, so this is a high-recall floor rather than exactly 1.0.
    assert!(precision_at(n) >= 0.98);
}

#[test]
fn max_dist_cutoff_respected() {
    let dim = 8;
    // Two clusters: near origin and far away.
    let mut points: Vec<u8> = Vec::new();
    for _ in 0..10 {
        points.extend(std::iter::repeat_n(1u8, dim));
    }
    for _ in 0..10 {
        points.extend(std::iter::repeat_n(200u8, dim));
    }
    let n = points.len() / dim;
    let forest = KdForestU8::build(
        &points,
        n,
        dim,
        KdForestParams::accurate(),
        &Progress::none(),
    )
    .expect("nothing asked it to stop");

    let query = vec![0u8; dim];
    // Radius that only reaches the near cluster (dist to near ~ sqrt(8)).
    let nbrs = forest.search(&query, 20, n, Some(10.0));
    assert!(!nbrs.is_empty());
    for nb in &nbrs {
        assert!(nb.index < 10, "cutoff admitted a far-cluster point");
        assert!(nb.dist_sq <= 100.0 + 1e-3);
    }
}

#[test]
fn fewer_than_k_padding() {
    let dim = 4;
    let points: Vec<u8> = vec![0, 0, 0, 0, 5, 5, 5, 5];
    let forest = KdForestU8::build(
        &points,
        2,
        dim,
        KdForestParams::balanced(),
        &Progress::none(),
    )
    .expect("nothing asked it to stop");
    // Ask for more neighbors than exist.
    let nbrs = forest.search(&[0, 0, 0, 0], 5, 100, None);
    assert_eq!(nbrs.len(), 2);
    // Batch form pads with u32::MAX.
    let batch = forest.search_batch(&[0, 0, 0, 0], 1, 5, 100, None);
    assert_eq!(batch.len(), 5);
    assert_eq!(batch[2], u32::MAX);
    assert_eq!(batch[3], u32::MAX);
}

#[test]
fn duplicate_coordinates_build_and_query() {
    // Many points sharing coordinates on several axes (mimics SIFT edges):
    // construction must terminate and queries must stay exact under a full
    // budget despite the duplicate-heavy median splits.
    let dim = 12;
    let n = 400;
    let mut rng = StdRng::seed_from_u64(9);
    let mut points = Vec::with_capacity(n * dim);
    for _ in 0..n {
        for d in 0..dim {
            // Half the axes are constant across all points.
            points.push(if d % 2 == 0 {
                50
            } else {
                rng.random_range(0..=u8::MAX)
            });
        }
    }
    let forest = KdForestU8::build(
        &points,
        n,
        dim,
        KdForestParams::balanced(),
        &Progress::none(),
    )
    .expect("nothing asked it to stop");
    let queries = random_u8(30, dim, 21);
    // Construction must terminate despite the duplicate-heavy median splits, and
    // a full budget still recovers the great majority of true neighbors.
    assert!(recall_at_1(&forest, &points, dim, &queries, n) >= 0.9);
}

#[test]
fn empty_and_zero_k() {
    let dim = 4;
    let forest = KdForestU8::build(&[], 0, dim, KdForestParams::balanced(), &Progress::none())
        .expect("nothing asked it to stop");
    assert!(forest.is_empty());
    assert!(forest.search(&[0, 0, 0, 0], 1, 10, None).is_empty());

    let pts = vec![1u8, 2, 3, 4];
    let f2 = KdForestU8::build(&pts, 1, dim, KdForestParams::balanced(), &Progress::none())
        .expect("nothing asked it to stop");
    assert!(f2.search(&[1, 2, 3, 4], 0, 10, None).is_empty());
}

#[test]
fn calibration_finds_a_budget() {
    let dim = 32;
    let n = 600;
    let points = random_u8(n, dim, 4242);
    let forest = KdForestU8::build(
        &points,
        n,
        dim,
        KdForestParams::balanced(),
        &Progress::none(),
    )
    .expect("nothing asked it to stop");

    let nq = 80;
    let sample = random_u8(nq, dim, 1357);
    let exact: Vec<u32> = sample
        .chunks(dim)
        .map(|q| brute_force_nn(&points, dim, q))
        .collect();

    let budget = forest.calibrate_max_leaf_checks(&sample, &exact, 0.9);
    assert!(budget >= 1 && budget <= n);

    // The fitted budget meets the target on the sample.
    let got = forest.search_batch(&sample, nq, 1, budget, None);
    let hits = (0..nq).filter(|&i| got[i] == exact[i]).count() as f64 / nq as f64;
    assert!(
        hits >= 0.9 - 1e-9,
        "calibrated precision {hits} below target"
    );
}

#[test]
fn f32_forest_basic() {
    let dim = 3;
    let points: Vec<f32> = vec![
        0.0, 0.0, 0.0, //
        10.0, 0.0, 0.0, //
        0.0, 10.0, 0.0, //
        0.5, 0.5, 0.0, //
    ];
    let forest = KdForestF32::build(
        &points,
        4,
        dim,
        KdForestParams::accurate(),
        &Progress::none(),
    )
    .expect("nothing asked it to stop");
    let nbrs = forest.search(&[0.1, 0.1, 0.0], 2, 64, None);
    assert_eq!(nbrs[0].index, 0);
    assert_eq!(nbrs[1].index, 3);
    assert!(nbrs[0].dist_sq < nbrs[1].dist_sq);
}

/// Random `f32` point set in `[0, 1)`, flat row-major.
fn random_f32(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim)
        .map(|_| rng.random_range(0.0f32..1.0))
        .collect()
}

#[test]
fn f32_single_leaf_is_exact() {
    // f32 engine, single leaf => exact. Validates the floating squared-L2 path,
    // OrdF32 ordering, and reported distances against an f32 brute force.
    let dim = 8;
    let n = 120;
    let points = random_f32(n, dim, 2024);
    let params = KdForestParams {
        num_trees: 1,
        leaf_size: n,
        ..KdForestParams::balanced()
    };
    let forest = KdForestF32::build(&points, n, dim, params, &Progress::none())
        .expect("nothing asked it to stop");
    let queries = random_f32(40, dim, 7);
    for q in queries.chunks(dim) {
        let exact = (0..n)
            .map(|i| {
                q.iter()
                    .zip(&points[i * dim..(i + 1) * dim])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
            })
            .fold(f32::INFINITY, f32::min);
        let nbrs = forest.search(q, 1, n, None);
        assert!((nbrs[0].dist_sq - exact).abs() <= 1e-4 * exact.max(1.0));
    }
}

#[test]
fn f32_max_dist_cutoff() {
    // The f32 cutoff path (no integer rounding, unlike u8) must admit only the
    // near cluster and pad the rest.
    let dim = 4;
    let mut points: Vec<f32> = Vec::new();
    for _ in 0..8 {
        points.extend([0.1f32; 4]);
    }
    for _ in 0..8 {
        points.extend([100.0f32; 4]);
    }
    let n = points.len() / dim;
    let forest = KdForestF32::build(
        &points,
        n,
        dim,
        KdForestParams::accurate(),
        &Progress::none(),
    )
    .expect("nothing asked it to stop");
    let nbrs = forest.search(&[0.0, 0.0, 0.0, 0.0], 20, n, Some(1.0));
    assert!(!nbrs.is_empty());
    for nb in &nbrs {
        assert!(nb.index < 8, "cutoff admitted a far-cluster point");
        assert!(nb.dist_sq <= 1.0 + 1e-6);
    }
}

#[test]
fn with_distances_padding() {
    let dim = 4;
    let points: Vec<u8> = vec![0, 0, 0, 0, 9, 9, 9, 9];
    let forest = KdForestU8::build(
        &points,
        2,
        dim,
        KdForestParams::balanced(),
        &Progress::none(),
    )
    .expect("nothing asked it to stop");
    let (idx, dist) = forest.search_batch_with_distances(&[0, 0, 0, 0], 1, 3, 100, None);
    assert_eq!(idx.len(), 3);
    assert_eq!(dist.len(), 3);
    assert_eq!(idx[0], 0);
    assert_eq!(dist[0], 0.0);
    assert_eq!(idx[2], u32::MAX);
    assert_eq!(dist[2], f32::INFINITY);
}

// ── Reporting and cancellation ──────────────────────────────────────────

/// Every fraction a build of `params` reported, in the order it reported them.
fn fractions_of_a_build(
    points: &[u8],
    n: usize,
    dim: usize,
    params: KdForestParams,
) -> (Vec<f32>, KdForestU8) {
    use crate::progress::{Event, Progress};
    use std::sync::Mutex;

    let seen = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        if let Event::Fraction { of_whole } = event {
            seen.lock().unwrap().push(of_whole);
        }
    };
    let forest = KdForestU8::build(points, n, dim, params, &Progress::to(&sink))
        .expect("nothing asked it to stop");
    (seen.into_inner().unwrap(), forest)
}

/// The reporting build says where it has got to, and builds the very forest
/// the silent one does.
#[test]
fn a_reporting_build_climbs_to_the_end_and_builds_the_same_forest() {
    let dim = 16;
    let n = 4_000;
    let points = random_u8(n, dim, 77);
    let (fractions, reported) = fractions_of_a_build(&points, n, dim, KdForestParams::balanced());

    assert!(!fractions.is_empty(), "a build that said nothing");
    assert!(
        fractions.iter().all(|f| (0.0..=1.0).contains(f)),
        "{fractions:?}"
    );
    assert_eq!(
        fractions.last().copied(),
        Some(1.0),
        "a finished build ends at its end"
    );

    // The counter is shared across trees, so the count itself never falls; what
    // can arrive out of order is the *report*, because a `T > 1` build makes
    // them from rayon's workers. One tree is one thread, and there the order
    // the sink sees is the order the leaves were placed in.
    let one_tree = KdForestParams {
        num_trees: 1,
        ..KdForestParams::balanced()
    };
    let (ordered, _) = fractions_of_a_build(&points, n, dim, one_tree);
    assert!(
        ordered.windows(2).all(|w| w[1] >= w[0]),
        "the fractions ran backwards: {ordered:?}"
    );
    assert_eq!(ordered.last().copied(), Some(1.0));

    // Same parameters and the same seed, so the two forests are the same
    // forest: every tree's leaf order and every leaf boundary.
    let silent = KdForestU8::build(
        &points,
        n,
        dim,
        KdForestParams::balanced(),
        &Progress::none(),
    )
    .expect("nothing asked it to stop");
    assert_eq!(reported.num_trees(), silent.num_trees());
    for tree in 0..silent.num_trees() {
        assert_eq!(reported.tree_leaves(tree), silent.tree_leaves(tree));
    }
    let queries = random_u8(40, dim, 78);
    assert_eq!(
        reported.search_batch(&queries, 40, 3, 100, None),
        silent.search_batch(&queries, 40, 3, 100, None),
    );
}

/// A build that is asked to stop stops, and hands back nothing: a forest
/// missing the trees that had not finished is not a forest.
#[test]
fn a_build_that_is_asked_to_stop_hands_back_nothing() {
    use crate::progress::{Cancelled, Progress};
    use std::sync::atomic::{AtomicBool, Ordering};

    let dim = 16;
    let n = 4_000;
    let points = random_u8(n, dim, 79);
    let flag = AtomicBool::new(true);
    let stopped = KdForestU8::build(
        &points,
        n,
        dim,
        KdForestParams::balanced(),
        &Progress::none().cancelled_by(&flag),
    );
    assert_eq!(stopped.err(), Some(Cancelled));

    // And the same flag unset builds the forest, so what stopped it was the
    // flag rather than the arguments.
    flag.store(false, Ordering::Relaxed);
    let forest = KdForestU8::build(
        &points,
        n,
        dim,
        KdForestParams::balanced(),
        &Progress::none().cancelled_by(&flag),
    )
    .expect("nothing is asking it to stop now");
    assert_eq!(forest.len(), n);
}
