// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks for the randomized kd-tree forest ANN index.
//!
//! Measures build time vs the number of trees, query throughput vs the
//! per-query budget (`L_max`), and end-to-end batched matching against the
//! exact brute-force scanner. Uses synthetic 128-D `u8` descriptors (the SIFT
//! shape) so the bench is self-contained.
//!
//! Run with:
//!   cargo bench -p sfmtool-core --bench kdtree_forest

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sfmtool_core::feature_match::descriptor::descriptor_distance_l2_squared;
use sfmtool_core::kdforest::{KdForestParams, KdForestU8};
use std::hint::black_box;

const DIM: usize = 128;

/// Deterministic xorshift descriptor generator.
fn descriptors(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    let mut out = Vec::with_capacity(n * DIM);
    for _ in 0..n * DIM {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.push((x & 0xFF) as u8);
    }
    out
}

fn bench_build(c: &mut Criterion) {
    let n = 20_000;
    let points = descriptors(n, 1);
    let mut group = c.benchmark_group("kdforest_build");
    for &t in &[1usize, 4, 8, 16] {
        let params = KdForestParams {
            num_trees: t,
            ..KdForestParams::balanced()
        };
        group.bench_with_input(BenchmarkId::from_parameter(t), &t, |b, _| {
            b.iter(|| black_box(KdForestU8::build(&points, n, DIM, params)));
        });
    }
    group.finish();
}

fn bench_query(c: &mut Criterion) {
    let n = 20_000;
    let points = descriptors(n, 1);
    let queries = descriptors(2_000, 2);
    let forest = KdForestU8::build(&points, n, DIM, KdForestParams::balanced());

    let mut group = c.benchmark_group("kdforest_query_batch_2k");
    for &budget in &[32usize, 128, 512, 2048] {
        group.bench_with_input(
            BenchmarkId::from_parameter(budget),
            &budget,
            |b, &budget| {
                b.iter(|| black_box(forest.search_batch(&queries, 2_000, 2, budget, None)));
            },
        );
    }
    group.finish();
}

fn bench_vs_bruteforce(c: &mut Criterion) {
    let n = 5_000;
    let points = descriptors(n, 1);
    let queries = descriptors(500, 2);
    let forest = KdForestU8::build(&points, n, DIM, KdForestParams::balanced());

    // Both find the single nearest neighbor so the comparison is equal work.
    let mut group = c.benchmark_group("kdforest_match_500_queries");
    group.bench_function("forest_L128", |b| {
        b.iter(|| black_box(forest.search_batch(&queries, 500, 1, 128, None)));
    });
    group.bench_function("brute_force", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(500);
            for q in queries.chunks(DIM) {
                let mut best = 0u32;
                let mut best_d = i64::MAX;
                for i in 0..n {
                    let d = descriptor_distance_l2_squared(q, &points[i * DIM..(i + 1) * DIM]);
                    if d < best_d {
                        best_d = d;
                        best = i as u32;
                    }
                }
                out.push(best);
            }
            black_box(out)
        });
    });
    group.finish();
}

criterion_group!(benches, bench_build, bench_query, bench_vs_bruteforce);
criterion_main!(benches);
