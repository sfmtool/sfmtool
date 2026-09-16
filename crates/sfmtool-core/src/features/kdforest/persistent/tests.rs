// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::features::kdforest::KdForestParams;

fn options() -> LazyKdForestOptions {
    LazyKdForestOptions {
        cache_bytes: 16 << 10,
        max_in_flight_bytes: 16 << 10,
        max_chunk_bytes: 16 << 10,
        max_metadata_bytes: 1 << 20,
        query_workers: 2,
        ..Default::default()
    }
}

#[test]
fn eager_reassembly_rejects_cycles_and_missing_features() {
    let cycle = [
        Node::Internal {
            split_dim: 0,
            split_val: 1u8,
            left: 0,
            right: 1,
        },
        Node::Leaf { start: 0, len: 1 },
    ];
    assert!(validate_loaded_tree(&cycle, &[0], 1).is_err());
    assert!(validate_loaded_tree::<u8>(&[Node::Leaf { start: 0, len: 1 }], &[0], 2).is_err());
    assert!(validate_loaded_tree::<u8>(&[Node::Leaf { start: 0, len: 2 }], &[0, 0], 2).is_err());
}

#[test]
fn shared_reads_preserve_ties_and_reject_invalid_schedules() {
    let points = vec![7u8; 32 * 4];
    let forest = KdForest::build(
        &points,
        32,
        4,
        KdForestParams {
            num_trees: 4,
            leaf_size: 8,
            ..KdForestParams::balanced()
        },
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ties.kdf");
    let order: Vec<u32> = (0..32).rev().collect();
    forest
        .write_kdf_ordered(
            &path,
            None,
            &KdfWriteOptions {
                target_descriptor_block_bytes: 16,
                ..Default::default()
            },
            Some(&order),
        )
        .unwrap();
    let lazy = LazyKdForestU8::open(&path, LazyKdForestOptions::default()).unwrap();
    let queries = vec![7u8; 3 * 4];
    let expected = forest.search_batch_with_distances(&queries, 3, 5, 128, None);
    assert_eq!(
        lazy.search_batch_with_distances(&queries, 3, 5, 128, None)
            .unwrap(),
        expected
    );
    assert_eq!(
        lazy.search_batch_with_distances_ordered(&queries, 3, 5, 128, None, &[2, 0, 1])
            .unwrap(),
        expected
    );
    for order in [&[0, 0, 2][..], &[0, 1, 3][..]] {
        assert!(lazy
            .search_batch_with_distances_ordered(&queries, 3, 5, 128, None, order)
            .is_err());
    }
}

#[test]
fn u8_file_queries_match_eager_results_and_checks() {
    let dim = 7;
    let n = 73;
    let points: Vec<u8> = (0..n * dim)
        .map(|i| ((i * 37 + i / 5) % 251) as u8)
        .collect();
    let forest = KdForest::build(
        &points,
        n,
        dim,
        KdForestParams {
            num_trees: 4,
            leaf_size: 5,
            seed: 77,
            ..KdForestParams::balanced()
        },
    );
    let queries: Vec<u8> = (0..11 * dim).map(|i| ((i * 19 + 3) % 255) as u8).collect();
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("forest.kdf");
        forest
            .write_kdf(
                &path,
                None,
                &KdfWriteOptions {
                    target_descriptor_block_bytes: 19,
                    target_chunk_bytes: 240,
                    compression_level: 1,
                    origin_block_rows: 4,
                },
            )
            .unwrap();
        let lazy = LazyKdForestU8::open(&path, options()).unwrap();
        for (budget, query) in [0usize, 1, 7, 31, 1000]
            .into_iter()
            .flat_map(|b| queries.chunks(dim).map(move |q| (b, q)))
        {
            let expected = forest.search(query, 4, budget, Some(300.0));
            let (got, lazy_stats) = lazy
                .search_with_stats(query, 4, budget, Some(300.0))
                .unwrap();
            assert_eq!(got, expected, "budget={budget}");
            let mut scratch = crate::features::kdforest::search::SearchScratch::new(n);
            let mut eager_stats = crate::features::kdforest::search::QueryStats::default();
            forest.run_query(
                query,
                4,
                budget,
                Some(300.0),
                &mut scratch,
                &mut eager_stats,
            );
            assert_eq!(lazy_stats.checks, eager_stats.checks, "budget={budget}");
        }
        let expected = forest.search_batch_with_distances(&queries, 11, 3, 30, None);
        let got = lazy
            .search_batch_with_distances(&queries, 11, 3, 30, None)
            .unwrap();
        assert_eq!(got, expected);
        let order = [3, 0, 9, 2, 10, 1, 4, 8, 5, 7, 6];
        assert_eq!(
            lazy.search_batch_with_distances_ordered(&queries, 11, 3, 30, None, &order)
                .unwrap(),
            expected
        );
        assert_eq!(
            lazy.search_batch_with_distances(&queries, 11, 0, 30, None)
                .unwrap(),
            (Vec::new(), Vec::new())
        );
        for k in [0, 3, 80] {
            assert_eq!(
                lazy.self_join_with_distances(k, 30, None).unwrap(),
                forest.search_batch_with_distances(&points, n, k, 30, None)
            );
        }
    }
}

/// A forest reloaded from a file answers exactly as the one written did.
///
/// This is what makes the file an index rather than a cache of one: the
/// topology, leaf membership and feature IDs all survive the round trip, so
/// no rebuild is needed and no randomization has to be reproduced. Checked in
/// The descriptor corpus and topology are both restored without rebuilding.
#[test]
fn a_forest_reloaded_from_a_file_answers_identically() {
    let dim = 7;
    let n = 200;
    let points: Vec<u8> = (0..n * dim)
        .map(|i| ((i * 31 + i / 3) % 251) as u8)
        .collect();
    let forest = KdForest::build(
        &points,
        n,
        dim,
        KdForestParams {
            num_trees: 3,
            leaf_size: 8,
            seed: 5,
            ..KdForestParams::balanced()
        },
    );
    let queries: Vec<u8> = (0..9 * dim).map(|i| ((i * 17 + 5) % 255) as u8).collect();

    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("round.kdf");
        forest
            .write_kdf(
                &path,
                None,
                &KdfWriteOptions {
                    target_descriptor_block_bytes: 21,
                    target_chunk_bytes: 300,
                    compression_level: 1,
                    origin_block_rows: 8,
                },
            )
            .unwrap();
        let reloaded = KdForest::<u8>::read_kdf(&path, LazyKdForestOptions::default()).unwrap();

        assert_eq!(reloaded.len(), forest.len());
        assert_eq!(reloaded.dim(), forest.dim());
        assert_eq!(
            reloaded.params().num_trees,
            3,
            "params came from provenance"
        );
        assert_eq!(reloaded.params().leaf_size, 8);
        for (budget, query) in [0usize, 3, 40, 500]
            .into_iter()
            .flat_map(|b| queries.chunks(dim).map(move |q| (b, q)))
        {
            assert_eq!(
                reloaded.search(query, 4, budget, None),
                forest.search(query, 4, budget, None),
                "budget={budget}"
            );
        }
        // The corpus itself must survive, not merely the topology.
        let batch = reloaded.search_batch_with_distances(&points, n, 1, 200, None);
        let want = forest.search_batch_with_distances(&points, n, 1, 200, None);
        assert_eq!(batch, want);
    }
}

#[test]
fn f32_signed_zero_and_cutoff_match_eager() {
    let points = vec![-0.0f32, 0.0, 1.0, 1.0, -1.0, -1.0, 2.0, 2.0];
    let forest = KdForest::build(
        &points,
        4,
        2,
        KdForestParams {
            num_trees: 2,
            leaf_size: 1,
            ..KdForestParams::balanced()
        },
    );
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("float.kdf");
        forest
            .write_kdf(
                &path,
                None,
                &KdfWriteOptions {
                    target_descriptor_block_bytes: 8,
                    target_chunk_bytes: 100,
                    compression_level: 1,
                    origin_block_rows: 4,
                },
            )
            .unwrap();
        let lazy = LazyKdForestF32::open(&path, options()).unwrap();
        for q in [[0.0, 0.0], [-0.0, 0.0], [0.5, 0.5]] {
            assert_eq!(
                lazy.search(&q, 3, 20, Some(f32::INFINITY)).unwrap(),
                forest.search(&q, 3, 20, Some(f32::INFINITY))
            );
        }
        assert!(lazy.search(&[f32::NAN, 0.0], 1, 1, None).is_err());
        assert!(lazy.search(&[0.0, 0.0], 1, 1, Some(-1.0)).is_err());
    }
}

#[test]
fn concurrent_small_cache_eviction_completes_with_parity() {
    let dim = 16;
    let n = 160;
    let points: Vec<u8> = (0..n * dim)
        .map(|i| ((i * 43 + i / 11) % 256) as u8)
        .collect();
    let forest = KdForest::build(
        &points,
        n,
        dim,
        KdForestParams {
            num_trees: 4,
            leaf_size: 4,
            seed: 8,
            ..KdForestParams::balanced()
        },
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("eviction.kdf");
    forest
        .write_kdf(
            &path,
            None,
            &KdfWriteOptions {
                target_descriptor_block_bytes: 64,
                target_chunk_bytes: 300,
                compression_level: 1,
                origin_block_rows: 4,
            },
        )
        .unwrap();
    let lazy = std::sync::Arc::new(
        LazyKdForestU8::open(
            &path,
            LazyKdForestOptions {
                cache_bytes: 700,
                max_in_flight_bytes: 700,
                max_chunk_bytes: 700,
                max_compressed_bytes: 1 << 20,
                max_metadata_bytes: 1 << 20,
                query_workers: 2,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    assert_eq!(
        lazy.io_stats().read_calls,
        0,
        "open must not read descriptor blocks"
    );
    let expected = forest.search(&points[32..48], 3, 40, None);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let lazy = lazy.clone();
            let barrier = barrier.clone();
            let q = points[32..48].to_vec();
            std::thread::spawn(move || {
                barrier.wait();
                lazy.search(&q, 3, 40, None).unwrap()
            })
        })
        .collect();
    for thread in threads {
        assert_eq!(thread.join().unwrap(), expected);
    }
    let stats = lazy.io_stats();
    assert!(stats.evictions > 0);
    assert!(stats.peak_resident_bytes <= 700);
    assert_eq!(stats.in_flight_bytes, 0);
}
