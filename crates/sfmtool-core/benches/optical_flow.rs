// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks for the DIS optical flow pipeline.
//!
//! Measures end-to-end flow computation and individual components to identify
//! hotspots for SIMD optimization.
//!
//! Run with:
//!   cargo bench -p sfmtool-core --bench optical_flow
//!
//! The test images used (dino_dog_toy at 2040x1536) are checked into the repo.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sfmtool_core::optical_flow::{self, bench, compute_optical_flow, DisFlowParams, GrayImage};
use std::time::Duration;

fn test_data_dir() -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    std::path::PathBuf::from(manifest)
        .join("../../test-data/images")
        .canonicalize()
        .expect("Could not find test-data/images directory")
}

fn image_subdir(filename: &str) -> &str {
    let stem = filename.rsplit_once('.').map_or(filename, |(s, _)| s);
    stem.rsplit_once('_').map_or(stem, |(prefix, _)| prefix)
}

fn load_gray(filename: &str) -> GrayImage {
    let path = test_data_dir().join(image_subdir(filename)).join(filename);
    let img = image::open(&path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {}", path.display(), e))
        .into_luma8();
    let (w, h) = img.dimensions();
    GrayImage::from_u8(w, h, img.as_raw())
}

// ---------------------------------------------------------------------------
// End-to-end benchmarks
// ---------------------------------------------------------------------------

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("optical_flow/end_to_end");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Large images: dino_dog_toy 2040x1536
    let img_a = load_gray("dino_dog_toy_42.jpg");
    let img_b = load_gray("dino_dog_toy_43.jpg");

    for (name, params) in [
        ("fast", DisFlowParams::fast()),
        ("default", DisFlowParams::default_quality()),
    ] {
        group.bench_function(BenchmarkId::new("dino_dog_toy", name), |b| {
            b.iter(|| compute_optical_flow(black_box(&img_a), black_box(&img_b), &params, None))
        });
    }

    // Small images: seattle_backyard 360x640
    let small_a = load_gray("seattle_backyard_17.jpg");
    let small_b = load_gray("seattle_backyard_18.jpg");

    for (name, params) in [
        ("fast", DisFlowParams::fast()),
        ("default", DisFlowParams::default_quality()),
    ] {
        group.bench_function(BenchmarkId::new("seattle_backyard", name), |b| {
            b.iter(|| compute_optical_flow(black_box(&small_a), black_box(&small_b), &params, None))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Component benchmarks (large images only)
// ---------------------------------------------------------------------------

fn bench_pyramid(c: &mut Criterion) {
    let mut group = c.benchmark_group("optical_flow/pyramid");
    group.sample_size(20);

    let img = load_gray("dino_dog_toy_42.jpg");

    group.bench_function("build_8_levels", |b| {
        b.iter(|| bench::build_pyramid(black_box(&img), 8))
    });

    group.finish();
}

fn bench_dis_refine_single_level(c: &mut Criterion) {
    let mut group = c.benchmark_group("optical_flow/dis_refine");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let img_a = load_gray("dino_dog_toy_42.jpg");
    let img_b = load_gray("dino_dog_toy_43.jpg");

    // Benchmark at full resolution (scale 0) — the most expensive level
    let params_no_var = DisFlowParams {
        variational_refinement: false,
        ..DisFlowParams::default_quality()
    };

    group.bench_function("full_res_no_variational", |b| {
        b.iter(|| {
            let mut flow = optical_flow::FlowField::new(img_a.width(), img_a.height());
            bench::refine_flow_at_level(
                black_box(&img_a),
                black_box(&img_b),
                &mut flow,
                &params_no_var,
                0,
            );
            flow
        })
    });

    group.finish();
}

fn bench_variational(c: &mut Criterion) {
    let mut group = c.benchmark_group("optical_flow/variational");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let img_a = load_gray("dino_dog_toy_42.jpg");
    let img_b = load_gray("dino_dog_toy_43.jpg");
    let params = DisFlowParams::default_quality();

    group.bench_function("full_res_1_outer_iter", |b| {
        b.iter(|| {
            let mut flow = optical_flow::FlowField::new(img_a.width(), img_a.height());
            bench::variational_refine(black_box(&img_a), black_box(&img_b), &mut flow, &params);
            flow
        })
    });

    group.finish();
}

fn bench_bilinear_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("optical_flow/bilinear");
    group.sample_size(50);

    let img = load_gray("dino_dog_toy_42.jpg");
    let w = img.width() as f32;
    let h = img.height() as f32;

    // Sample at 1M points to get a meaningful measurement
    let n = 1_000_000;
    let points: Vec<(f32, f32)> = (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            // Sweep across the image in a diagonal pattern
            (t * (w - 1.0) + 0.5, t * (h - 1.0) + 0.5)
        })
        .collect();

    group.bench_function("1M_samples", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &(x, y) in &points {
                sum += bench::sample_bilinear(black_box(&img), x, y);
            }
            sum
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_end_to_end,
    bench_pyramid,
    bench_dis_refine_single_level,
    bench_variational,
    bench_bilinear_sampling,
);
criterion_main!(benches);
