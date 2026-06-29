// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks for the patch-normal-refinement render primitives.
//!
//! The refinement hot loop is, per candidate normal and per view:
//! `WarpMap::from_patch` → (`compute_svd` →) `remap_bilinear` /
//! `remap_aniso_with_pyramid`. These benches measure each primitive at the
//! small patch resolutions the refinement uses (R = 8..32), and pair each
//! library call with a hand-inlined *sequential* re-implementation to expose
//! how much of the cost is rayon parallel scaffolding (the library versions
//! parallelize over the R rows of a tiny grid, nested inside
//! `refine_patch_cloud_normals`'s per-patch parallelism in real use).
//!
//! Run with:
//!   cargo bench -p sfmtool-core --bench patch_render
//!
//! Compare against a single-threaded pool with:
//!   RAYON_NUM_THREADS=1 cargo bench -p sfmtool-core --bench patch_render

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::{Point3, Vector3};
use sfmtool_core::camera::remap::{
    remap_aniso_with_pyramid, remap_bilinear, sample_bilinear_u8, ImageU8, ImageU8Pyramid,
};
use sfmtool_core::camera::WarpMap;
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::patch::normal_refine::{
    refine_patch_normal, NormalRefineParams, ProjectedImage, Sampler,
};
use sfmtool_core::{CameraIntrinsics, CameraModel};
use std::hint::black_box;

const IMG_W: u32 = 1920;
const IMG_H: u32 = 1080;
const FOCAL: f64 = 1100.0;
const PLANE_Z: f64 = 4.0;
const MAX_ANISOTROPY: u32 = 16;

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: FOCAL,
            focal_length_y: FOCAL,
            principal_point_x: IMG_W as f64 / 2.0,
            principal_point_y: IMG_H as f64 / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
    }
}

/// Procedural 3-channel texture (smooth, multi-frequency).
fn texture(x: f64, y: f64, ch: usize) -> u8 {
    let p = ch as f64 * 0.7;
    let v = 127.5
        + 55.0 * (x * 17.0 + p).sin()
        + 45.0 * (y * 23.0 - p).cos()
        + 25.0 * ((x + y) * 31.0).sin();
    v.clamp(0.0, 255.0) as u8
}

/// The image an axis-aligned pinhole camera at `center` sees of the textured
/// plane `z = PLANE_Z`.
fn render_view(center: [f64; 3]) -> ImageU8 {
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    let mut data = Vec::with_capacity((IMG_W * IMG_H * 3) as usize);
    for row in 0..IMG_H {
        for col in 0..IMG_W {
            let dx = (col as f64 + 0.5 - cx) / FOCAL;
            let dy = (row as f64 + 0.5 - cy) / FOCAL;
            let lambda = PLANE_Z - center[2];
            let x = center[0] + lambda * dx;
            let y = center[1] + lambda * dy;
            for ch in 0..3 {
                data.push(texture(x, y, ch));
            }
        }
    }
    ImageU8::new(IMG_W, IMG_H, 3, data)
}

fn cam_pose(center: [f64; 3]) -> RigidTransform {
    RigidTransform::from_wxyz_translation(
        [1.0, 0.0, 0.0, 0.0],
        [-center[0], -center[1], -center[2]],
    )
}

/// A patch tilted ~35° off the viewing axis so the warp is anisotropic.
fn tilted_patch() -> OrientedPatch {
    let n = Vector3::new(0.6, 0.0, -0.8).normalize();
    OrientedPatch::from_center_normal(
        Point3::new(0.0, 0.0, PLANE_Z),
        n,
        Vector3::new(0.0, 1.0, 0.0),
        [0.05, 0.05],
    )
}

// ---------------------------------------------------------------------------
// Sequential re-implementations (mirror the library logic minus rayon)
// ---------------------------------------------------------------------------

/// `WarpMap::from_patch` with a plain sequential row loop.
fn from_patch_seq(
    patch: &OrientedPatch,
    camera: &CameraIntrinsics,
    cam_from_world: &RigidTransform,
    resolution: u32,
) -> WarpMap {
    let r = resolution.max(1);
    let src_w = camera.width as f64;
    let src_h = camera.height as f64;
    let step = 2.0 / r as f64;
    let mut data = vec![0.0f32; 2 * (r as usize) * (r as usize)];
    for row in 0..r {
        let t = (row as f64 + 0.5) * step - 1.0;
        for col in 0..r {
            let s = (col as f64 + 0.5) * step - 1.0;
            let world = patch.to_world(s, t);
            let p_cam = cam_from_world.transform_point(&world);
            let (sx, sy) = match camera.ray_to_pixel([p_cam.x, p_cam.y, p_cam.z]) {
                Some((px, py)) if px >= 0.0 && py >= 0.0 && px < src_w && py < src_h => (px, py),
                _ => (f64::NAN, f64::NAN),
            };
            let idx = 2 * (row as usize * r as usize + col as usize);
            data[idx] = sx as f32;
            data[idx + 1] = sy as f32;
        }
    }
    WarpMap::new(r, r, data)
}

/// `remap_bilinear` with a plain sequential loop.
fn remap_bilinear_seq(src: &ImageU8, map: &WarpMap) -> ImageU8 {
    let out_w = map.width();
    let out_h = map.height();
    let c = src.channels();
    let mut out = ImageU8::from_channels(out_w, out_h, c);
    let stride = out_w as usize * c as usize;
    for row in 0..out_h {
        for col in 0..out_w {
            let (sx, sy) = map.get(col, row);
            if sx.is_nan() || sy.is_nan() {
                continue;
            }
            let base = row as usize * stride + col as usize * c as usize;
            for ch in 0..c {
                let val = sample_bilinear_u8(src, sx, sy, ch);
                out.data_mut()[base + ch as usize] = (val + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// `remap_aniso_with_pyramid` with a plain sequential loop.
fn remap_aniso_seq(pyramid: &ImageU8Pyramid, map: &WarpMap, max_anisotropy: u32) -> ImageU8 {
    let out_w = map.width();
    let out_h = map.height();
    let c = pyramid.level(0).channels();
    let num_levels = pyramid.num_levels();
    let mut out = ImageU8::from_channels(out_w, out_h, c);
    let stride = out_w as usize * c as usize;
    for row in 0..out_h {
        for col in 0..out_w {
            let (sx, sy) = map.get(col, row);
            if sx.is_nan() || sy.is_nan() {
                continue;
            }
            let (sigma_major, sigma_minor, major_dx, major_dy) = map.get_svd(col, row);
            let base = row as usize * stride + col as usize * c as usize;
            if sigma_major <= 1.0 {
                for ch in 0..c {
                    let val = sample_bilinear_u8(pyramid.level(0), sx, sy, ch);
                    out.data_mut()[base + ch as usize] = (val + 0.5).clamp(0.0, 255.0) as u8;
                }
                continue;
            }
            let level_f = sigma_minor.max(1.0_f32).log2();
            let level_lo = (level_f.floor() as usize).min(num_levels - 1);
            let level_hi = (level_lo + 1).min(num_levels - 1);
            let frac = if level_lo == level_hi {
                0.0
            } else {
                level_f - level_lo as f32
            };
            let ratio = sigma_major / sigma_minor.max(1.0);
            let n = (ratio.ceil() as u32).clamp(1, max_anisotropy);
            let scale_lo = (1u32 << level_lo) as f32;
            let scale_hi = (1u32 << level_hi) as f32;
            for ch in 0..c {
                let mut sum_lo = 0.0f32;
                let mut sum_hi = 0.0f32;
                for i in 0..n {
                    let t = (i as f32 + 0.5) / n as f32 - 0.5;
                    let sample_x = sx + t * sigma_major * major_dx;
                    let sample_y = sy + t * sigma_major * major_dy;
                    sum_lo += sample_bilinear_u8(
                        pyramid.level(level_lo),
                        sample_x / scale_lo,
                        sample_y / scale_lo,
                        ch,
                    );
                    sum_hi += sample_bilinear_u8(
                        pyramid.level(level_hi),
                        sample_x / scale_hi,
                        sample_y / scale_hi,
                        ch,
                    );
                }
                let val = (sum_lo / n as f32) * (1.0 - frac) + (sum_hi / n as f32) * frac;
                out.data_mut()[base + ch as usize] = (val + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Benches
// ---------------------------------------------------------------------------

fn bench_primitives(crit: &mut Criterion) {
    let camera = pinhole();
    let pose = cam_pose([0.3, 0.1, 0.0]);
    let patch = tilted_patch();
    let image = render_view([0.3, 0.1, 0.0]);
    let max_levels = ((IMG_H.min(IMG_W) as f32).log2().floor() as usize).max(1) + 1;
    let pyramid = ImageU8Pyramid::build(&image, max_levels);

    let mut group = crit.benchmark_group("patch_render");
    for &res in &[8u32, 16, 24, 32] {
        // Prebuilt maps for the remap benches.
        let map_plain = WarpMap::from_patch(&patch, &camera, &pose, res);
        let mut map_svd = WarpMap::from_patch(&patch, &camera, &pose, res);
        map_svd.compute_svd();

        group.bench_function(BenchmarkId::new("from_patch", res), |b| {
            b.iter(|| black_box(WarpMap::from_patch(&patch, &camera, &pose, res)))
        });
        group.bench_function(BenchmarkId::new("from_patch_seq", res), |b| {
            b.iter(|| black_box(from_patch_seq(&patch, &camera, &pose, res)))
        });
        group.bench_function(BenchmarkId::new("compute_svd", res), |b| {
            b.iter_batched(
                || WarpMap::from_patch(&patch, &camera, &pose, res),
                |mut m| {
                    m.compute_svd();
                    black_box(m)
                },
                criterion::BatchSize::SmallInput,
            )
        });
        group.bench_function(BenchmarkId::new("remap_bilinear", res), |b| {
            b.iter(|| black_box(remap_bilinear(pyramid.level(0), &map_plain)))
        });
        group.bench_function(BenchmarkId::new("remap_bilinear_seq", res), |b| {
            b.iter(|| black_box(remap_bilinear_seq(pyramid.level(0), &map_plain)))
        });
        group.bench_function(BenchmarkId::new("remap_aniso", res), |b| {
            b.iter(|| black_box(remap_aniso_with_pyramid(&pyramid, &map_svd, MAX_ANISOTROPY)))
        });
        group.bench_function(BenchmarkId::new("remap_aniso_seq", res), |b| {
            b.iter(|| black_box(remap_aniso_seq(&pyramid, &map_svd, MAX_ANISOTROPY)))
        });
    }
    group.finish();
}

fn bench_refine(crit: &mut Criterion) {
    // Six views in a small arc around the patch.
    let centers: Vec<[f64; 3]> = (0..6)
        .map(|i| [-0.75 + 0.3 * i as f64, 0.05 * i as f64, 0.0])
        .collect();
    let camera = pinhole();
    let images: Vec<ImageU8> = centers.iter().map(|&c| render_view(c)).collect();
    let max_levels = ((IMG_H.min(IMG_W) as f32).log2().floor() as usize).max(1) + 1;
    let pyramids: Vec<ImageU8Pyramid> = images
        .iter()
        .map(|im| ImageU8Pyramid::build(im, max_levels))
        .collect();
    let poses: Vec<RigidTransform> = centers.iter().map(|&c| cam_pose(c)).collect();
    let views: Vec<ProjectedImage<'_>> = poses
        .iter()
        .zip(&pyramids)
        .map(|(cam_from_world, pyramid)| ProjectedImage {
            camera: &camera,
            cam_from_world,
            pyramid,
        })
        .collect();
    let patch = tilted_patch();

    let mut group = crit.benchmark_group("refine_patch_normal");
    group.sample_size(20);
    for (name, sampler) in [
        ("anisotropic", Sampler::Anisotropic),
        ("bilinear", Sampler::Bilinear),
    ] {
        let params = NormalRefineParams {
            sampler,
            // Benchmark the full per-patch cost (confidence is off by default).
            compute_confidence: true,
            ..NormalRefineParams::default()
        };
        group.bench_function(BenchmarkId::new(name, 16), |b| {
            b.iter(|| black_box(refine_patch_normal(&patch, &views, 16, &params, None)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_primitives, bench_refine);
criterion_main!(benches);
