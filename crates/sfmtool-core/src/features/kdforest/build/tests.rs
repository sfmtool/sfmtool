// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The tree a build makes is the same whether its subtrees are built in
//! parallel or one after another, and the same as the node-by-node build that
//! drew from its RNG during the recursion, kept here as the reference.

use std::sync::atomic::AtomicBool;

use rand::rngs::StdRng;
use rand::RngExt;
use rand::SeedableRng;

use super::*;
use crate::features::kdforest::KdForest;
use crate::progress::Progress;

/// The build as it was before subtrees were built in parallel: one RNG per
/// tree, drawn from during a depth-first recursion that pushes nodes onto one
/// arena as it reaches them. Kept verbatim apart from the progress counter,
/// which plays no part in the tree.
mod reference {
    use rand::rngs::StdRng;
    use rand::RngExt;
    use rand::SeedableRng;

    use super::super::{Node, Tree, VARIANCE_SAMPLE};
    use crate::features::kdforest::{ForestScalar, KdForestParams};

    struct BuildScratch<S> {
        classes: Vec<std::cmp::Ordering>,
        reordered: Vec<u32>,
        vals: Vec<S>,
        sums: Vec<f64>,
    }

    pub(super) fn build_tree<S: ForestScalar>(
        points: &[S],
        dim: usize,
        mut point_ids: Vec<u32>,
        params: &KdForestParams,
        seed: u64,
    ) -> Tree<S> {
        let mut nodes = Vec::new();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut scratch = BuildScratch {
            classes: Vec::new(),
            reordered: Vec::new(),
            vals: Vec::new(),
            sums: vec![0.0; 2 * dim],
        };
        if !point_ids.is_empty() {
            build_node(
                &mut nodes,
                points,
                dim,
                &mut point_ids,
                0,
                params,
                &mut rng,
                &mut scratch,
            );
        }
        Tree { nodes, point_ids }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_node<S: ForestScalar>(
        nodes: &mut Vec<Node<S>>,
        points: &[S],
        dim: usize,
        ids: &mut [u32],
        offset: u32,
        params: &KdForestParams,
        rng: &mut StdRng,
        scratch: &mut BuildScratch<S>,
    ) -> u32 {
        let n = ids.len();
        if n <= params.leaf_size {
            let idx = nodes.len() as u32;
            nodes.push(Node::Leaf {
                start: offset,
                len: n as u32,
            });
            return idx;
        }

        let variances = estimate_variances(points, dim, ids, rng, &mut scratch.sums);
        let top = top_dims(variances, params.split_dim_candidates);
        let split_dim = top[rng.random_range(0..top.len())];
        let split_val = median_value::<S>(points, dim, split_dim, ids, &mut scratch.vals);
        let left_count = partition::<S>(points, dim, split_dim, split_val, ids, scratch);

        let node_idx = nodes.len() as u32;
        nodes.push(Node::Leaf {
            start: u32::MAX,
            len: u32::MAX,
        });

        let (left_ids, right_ids) = ids.split_at_mut(left_count);
        let left = build_node(nodes, points, dim, left_ids, offset, params, rng, scratch);
        let right = build_node(
            nodes,
            points,
            dim,
            right_ids,
            offset + left_count as u32,
            params,
            rng,
            scratch,
        );

        nodes[node_idx as usize] = Node::Internal {
            split_dim: split_dim as u16,
            split_val,
            left,
            right,
        };
        node_idx
    }

    fn estimate_variances<'a, S: ForestScalar>(
        points: &[S],
        dim: usize,
        ids: &[u32],
        rng: &mut StdRng,
        sums: &'a mut [f64],
    ) -> &'a [f64] {
        let n = ids.len();
        let take = n.min(VARIANCE_SAMPLE);
        let (sum, sum_sq) = sums.split_at_mut(dim);
        sum.fill(0.0);
        sum_sq.fill(0.0);
        for s in 0..take {
            let id = if take == n {
                ids[s]
            } else {
                ids[rng.random_range(0..n)]
            };
            let base = id as usize * dim;
            for d in 0..dim {
                let v = points[base + d].to_f64();
                sum[d] += v;
                sum_sq[d] += v * v;
            }
        }
        let inv = 1.0 / take as f64;
        for d in 0..dim {
            let mean = sum[d] * inv;
            sum[d] = (sum_sq[d] * inv - mean * mean).max(0.0);
        }
        &sums[..dim]
    }

    fn top_dims(variances: &[f64], pool: usize) -> Vec<usize> {
        let pool = pool.max(1).min(variances.len());
        let mut idx: Vec<usize> = (0..variances.len()).collect();
        let cmp = |&a: &usize, &b: &usize| {
            variances[b]
                .partial_cmp(&variances[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        };
        if pool < idx.len() {
            idx.select_nth_unstable_by(pool - 1, cmp);
            idx.truncate(pool);
        }
        idx.sort_unstable_by(cmp);
        idx
    }

    fn median_value<S: ForestScalar>(
        points: &[S],
        dim: usize,
        dim_idx: usize,
        ids: &[u32],
        vals: &mut Vec<S>,
    ) -> S {
        vals.clear();
        vals.extend(ids.iter().map(|&id| points[id as usize * dim + dim_idx]));
        let mid = vals.len() / 2;
        vals.select_nth_unstable_by(mid, |&a, &b| S::coord_cmp(a, b));
        vals[mid]
    }

    fn partition<S: ForestScalar>(
        points: &[S],
        dim: usize,
        split_dim: usize,
        split_val: S,
        ids: &mut [u32],
        scratch: &mut BuildScratch<S>,
    ) -> usize {
        let n = ids.len();
        let target = n / 2;

        scratch.classes.clear();
        let mut n_lt = 0usize;
        let mut n_eq = 0usize;
        for &id in ids.iter() {
            let c = points[id as usize * dim + split_dim];
            let class = S::coord_cmp(c, split_val);
            match class {
                std::cmp::Ordering::Less => n_lt += 1,
                std::cmp::Ordering::Equal => n_eq += 1,
                std::cmp::Ordering::Greater => {}
            }
            scratch.classes.push(class);
        }

        let reordered = &mut scratch.reordered;
        reordered.resize(n, 0);
        let (mut i_lt, mut i_eq, mut i_gt) = (0usize, n_lt, n_lt + n_eq);
        for (&id, &class) in ids.iter().zip(&scratch.classes) {
            match class {
                std::cmp::Ordering::Less => {
                    reordered[i_lt] = id;
                    i_lt += 1;
                }
                std::cmp::Ordering::Equal => {
                    reordered[i_eq] = id;
                    i_eq += 1;
                }
                std::cmp::Ordering::Greater => {
                    reordered[i_gt] = id;
                    i_gt += 1;
                }
            }
        }
        ids.copy_from_slice(&reordered[..n]);

        let eq_to_left = target.saturating_sub(n_lt).min(n_eq);
        let left_count = n_lt + eq_to_left;
        left_count.clamp(1, n - 1)
    }
}

/// Assert two trees are the same node for node: split dimensions, split
/// values (bit for bit), child indices, leaf ranges, and the leaf-ordered
/// point ids.
fn assert_same_tree<S: ForestScalar>(got: &Tree<S>, want: &Tree<S>, what: &str) {
    assert_eq!(got.point_ids, want.point_ids, "{what}: point ids");
    assert_eq!(got.nodes.len(), want.nodes.len(), "{what}: node count");
    for (i, (g, w)) in got.nodes.iter().zip(&want.nodes).enumerate() {
        match (*g, *w) {
            (
                Node::Internal {
                    split_dim: gd,
                    split_val: gv,
                    left: gl,
                    right: gr,
                },
                Node::Internal {
                    split_dim: wd,
                    split_val: wv,
                    left: wl,
                    right: wr,
                },
            ) => {
                assert_eq!(gd, wd, "{what}: node {i} split dimension");
                assert_eq!(
                    gv.to_f64().to_bits(),
                    wv.to_f64().to_bits(),
                    "{what}: node {i} split value"
                );
                assert_eq!((gl, gr), (wl, wr), "{what}: node {i} children");
            }
            (Node::Leaf { start: gs, len: gn }, Node::Leaf { start: ws, len: wn }) => {
                assert_eq!((gs, gn), (ws, wn), "{what}: node {i} leaf range")
            }
            (g, _) => panic!(
                "{what}: node {i} is {}, the reference has the other kind",
                if matches!(g, Node::Leaf { .. }) {
                    "a leaf"
                } else {
                    "internal"
                }
            ),
        }
    }
}

/// Build one tree with subtrees of at least `parallel_min_points` points built
/// as parallel tasks, inside a pool of `threads` threads.
fn build_with<S: ForestScalar>(
    points: &[S],
    dim: usize,
    params: &KdForestParams,
    seed: u64,
    parallel_min_points: usize,
    threads: usize,
) -> Tree<S> {
    let n = points.len() / dim;
    let progress = Progress::none();
    let counter = BuildProgress::new(&progress, 1, n);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(|| {
            build_tree(
                points,
                dim,
                (0..n as u32).collect(),
                params,
                seed,
                &counter,
                parallel_min_points,
            )
        })
        .expect("nothing asked it to stop")
}

/// Check that the tree over `points` is the reference build's, built one node
/// after another, with the production threshold, and with every internal node
/// split across tasks, for each seed.
fn assert_matches_reference<S: ForestScalar>(
    points: &[S],
    dim: usize,
    params: &KdForestParams,
    seeds: &[u64],
    what: &str,
) {
    let n = points.len() / dim;
    for &seed in seeds {
        let want = reference::build_tree(points, dim, (0..n as u32).collect(), params, seed);
        for (parallel_min_points, threads) in
            [(usize::MAX, 1), (PARALLEL_MIN_POINTS, 4), (1, 4), (1, 1)]
        {
            let got = build_with(points, dim, params, seed, parallel_min_points, threads);
            assert_same_tree(
                &got,
                &want,
                &format!(
                    "{what}, n {n}, seed {seed}, parallel from {parallel_min_points}, \
                     {threads} threads"
                ),
            );
        }
    }
}

fn random_u8(n: usize, dim: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim)
        .map(|_| rng.random_range(0..=u8::MAX))
        .collect()
}

/// SIFT descriptors from the checked-in seoul_bull images, flat row-major.
fn seoul_bull_descriptors(images: std::ops::RangeInclusive<usize>) -> Vec<u8> {
    use crate::features::sift::{extract_sift, gray_from_rgb, SiftParams};
    let params = SiftParams::default();
    let mut out = Vec::new();
    for i in images {
        let path = format!(
            "{}/../../test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_{i:02}.jpg",
            env!("CARGO_MANIFEST_DIR")
        );
        let rgb = image::open(&path).expect("load test image").to_rgb8();
        let gray = gray_from_rgb(
            rgb.width(),
            rgb.height(),
            rgb.as_raw(),
            &params.image_to_gray,
        );
        for row in extract_sift(&gray, &params).descriptors.rows() {
            out.extend_from_slice(row);
        }
    }
    out
}

#[test]
fn real_sift_descriptors_build_the_reference_tree() {
    let points = seoul_bull_descriptors(1..=3);
    let n = points.len() / 128;
    assert!(
        n > 2 * PARALLEL_MIN_POINTS,
        "{n} descriptors split at fewer than two levels"
    );
    assert_matches_reference(
        &points,
        128,
        &KdForestParams::balanced(),
        &[0, 1, 2, 3],
        "seoul_bull SIFT",
    );
}

#[test]
fn sizes_around_the_leaf_and_parallel_thresholds_build_the_reference_tree() {
    let params = KdForestParams::balanced();
    let leaf = params.leaf_size;
    for n in [
        0,
        1,
        2,
        leaf,
        leaf + 1,
        2 * leaf + 1,
        VARIANCE_SAMPLE,
        VARIANCE_SAMPLE + 1,
        PARALLEL_MIN_POINTS - 1,
        PARALLEL_MIN_POINTS,
        PARALLEL_MIN_POINTS + 1,
        2 * PARALLEL_MIN_POINTS + 3,
    ] {
        let points = random_u8(n, 16, n as u64);
        assert_matches_reference(&points, 16, &params, &[7], "random u8");
    }
}

#[test]
fn duplicates_and_constant_dimensions_build_the_reference_tree() {
    let params = KdForestParams {
        leaf_size: 4,
        ..KdForestParams::balanced()
    };
    let (n, dim) = (1500, 8);
    // Every point the same.
    assert_matches_reference(&vec![9u8; n * dim], dim, &params, &[0], "identical points");
    // Five distinct rows, repeated; and four of eight dimensions constant.
    let rows = random_u8(5, dim, 1);
    let mut repeated = Vec::with_capacity(n * dim);
    let mut constant = random_u8(n, dim, 2);
    for i in 0..n {
        repeated.extend_from_slice(&rows[(i * 7 % 5) * dim..][..dim]);
        for d in [0, 2, 3, 6] {
            constant[i * dim + d] = 200;
        }
    }
    assert_matches_reference(&repeated, dim, &params, &[0, 1], "five repeated rows");
    assert_matches_reference(&constant, dim, &params, &[0, 1], "constant dimensions");
    // Two values per coordinate: most nodes split inside a long run of equal
    // values.
    let binary: Vec<u8> = random_u8(n, dim, 3).iter().map(|v| v & 1).collect();
    assert_matches_reference(&binary, dim, &params, &[0], "two values per coordinate");
}

#[test]
fn other_parameters_and_f32_points_build_the_reference_tree() {
    // One dimension and more candidates than dimensions; leaves of one point.
    let one_dim = random_u8(1200, 1, 4);
    let params = KdForestParams {
        leaf_size: 1,
        split_dim_candidates: 5,
        ..KdForestParams::balanced()
    };
    assert_matches_reference(&one_dim, 1, &params, &[0], "one dimension");
    // f32, whose order is `total_cmp`: signed zeros and repeats included.
    let mut rng = StdRng::seed_from_u64(5);
    let floats: Vec<f32> = (0..1100 * 6)
        .map(|i| match i % 9 {
            0 => 0.0,
            1 => -0.0,
            2 => 1.5,
            _ => rng.random_range(-10.0f32..10.0),
        })
        .collect();
    assert_matches_reference(&floats, 6, &KdForestParams::balanced(), &[0, 9], "f32");
}

/// The public build, in pools of one and of four threads, gives every tree of
/// the forest the reference build's tree for its `seed + tree_index`.
#[test]
fn forest_build_in_any_pool_matches_the_reference() {
    let (n, dim) = (3000, 32);
    let points = random_u8(n, dim, 6);
    let params = KdForestParams {
        seed: 40,
        ..KdForestParams::balanced()
    };
    for threads in [1, 4] {
        let forest = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| KdForest::build(&points, n, dim, params, &Progress::none()))
            .expect("nothing asked it to stop");
        for (t, tree) in forest.trees.iter().enumerate() {
            let want = reference::build_tree(
                &points,
                dim,
                (0..n as u32).collect(),
                &params,
                40 + t as u64,
            );
            assert_same_tree(tree, &want, &format!("{threads} threads, tree {t}"));
        }
    }
}

/// A build split into parallel tasks still stops when asked, and hands back
/// nothing.
#[test]
fn a_parallel_build_stops_when_cancelled() {
    let (n, dim) = (4 * PARALLEL_MIN_POINTS, 8);
    let points = random_u8(n, dim, 8);
    let flag = AtomicBool::new(true);
    let progress = Progress::none().cancelled_by(&flag);
    let counter = BuildProgress::new(&progress, 1, n);
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| {
            build_tree(
                &points,
                dim,
                (0..n as u32).collect(),
                &KdForestParams::balanced(),
                0,
                &counter,
                PARALLEL_MIN_POINTS,
            )
        });
    assert_eq!(built.err(), Some(Cancelled));
}
