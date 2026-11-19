// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Every read through the render entry points equals the whole pyramid's,
//! bit for bit, through copied regions and through photograph tiles of
//! 32 px.

use std::sync::Arc;

use rayon::prelude::*;

use super::*;
use crate::camera::remap::{sample_bilinear_u8, HeldLevel};
use crate::camera::sampler::grid::{affine_regions, grid_cells, grid_regions, sample_grid};
use crate::camera::sampler::test_support::{affine_map, copy, maps, photograph, Lcg, SAMPLERS};
use crate::camera::sampler::{render_tile, render_tile_with_grad_into, TILE_WINDOW_TILES};

const TILE: u32 = 32;

/// The whole pyramid of `image`, and the same photograph cut into tiles of
/// [`TILE`] px.
fn sources(image: &ImageU8) -> (Arc<ImageU8Pyramid>, TiledPhotograph) {
    let levels = ImageU8Pyramid::full_levels(image.width(), image.height());
    let pyramid = Arc::new(ImageU8Pyramid::build(image, levels));
    let source = TiledPhotograph::from_pyramid(pyramid.clone()).with_tile_size(TILE);
    (pyramid, source)
}

/// A pool every read fits in, and one no read fits in.
fn pools() -> [(RegionPool, bool); 2] {
    [
        (RegionPool::with_capacity(u64::MAX), true),
        (RegionPool::with_capacity(0), false),
    ]
}

fn grad_bits(image: &ImageF32WithGrad) -> Vec<u32> {
    [image.value(), image.grad_x(), image.grad_y()]
        .iter()
        .flat_map(|b| b.iter().map(|v| v.to_bits()))
        .collect()
}

#[test]
fn every_value_and_gradient_render_equals_the_whole_pyramid() {
    let image = photograph(200, 150);
    let (pyramid, source) = sources(&image);
    let mut copied = 0;
    let mut tiled = 0;
    for (m, map) in maps(200, 150).into_iter().enumerate() {
        for sampler in SAMPLERS {
            let mut whole_map = copy(&map);
            let whole = render_tile(&pyramid, &mut whole_map, sampler);
            let mut whole_grad = ImageF32WithGrad::empty();
            render_tile_with_grad_into(&pyramid, &mut whole_map, sampler, &mut whole_grad);
            for (pool, fits) in pools() {
                let mut map = copy(&map);
                let (tile, path) = render_source_tile_path(
                    &mut ViewRegions::new(&source, &pool),
                    &mut map,
                    sampler,
                )
                .expect("an in-memory read");
                assert_eq!(tile.data(), whole.data(), "map {m}, {sampler:?}, {path:?}");
                let mut grad = ImageF32WithGrad::empty();
                let grad_path = render_source_tile_with_grad_path(
                    &mut ViewRegions::new(&source, &pool),
                    &mut map,
                    sampler,
                    &mut grad,
                )
                .expect("an in-memory read");
                assert_eq!(
                    grad_bits(&grad),
                    grad_bits(&whole_grad),
                    "map {m}, {sampler:?}"
                );
                for path in [path, grad_path] {
                    match path {
                        ReadPath::Copied { .. } => {
                            assert!(fits);
                            copied += 1;
                        }
                        ReadPath::Tiles(stats) => {
                            assert!(!fits);
                            assert!(stats.max_pinned <= TILE_WINDOW_TILES);
                            tiled += 1;
                        }
                        ReadPath::Reused { .. } => panic!("a fresh view keeps no regions"),
                    }
                }
            }
        }
    }
    assert!(copied > 0 && tiled > 0);
}

#[test]
fn the_tile_path_pins_no_more_than_its_window() {
    let image = photograph(256, 256);
    let (_, source) = sources(&image);
    // A Bilinear render over a large footprint reads level 0 across most of
    // the image's 64 tiles.
    let mut map = affine_map(32, [7.5, 0.5, 4.0, 0.3, 7.5, 6.0], 0);
    let pool = RegionPool::with_capacity(0);
    let (_, path) = render_source_tile_path(
        &mut ViewRegions::new(&source, &pool),
        &mut map,
        Sampler::Bilinear,
    )
    .unwrap();
    let ReadPath::Tiles(stats) = path else {
        panic!("an empty pool takes the tile path");
    };
    assert!(stats.fetched > TILE_WINDOW_TILES as u64, "{stats:?}");
    assert!(stats.max_pinned <= TILE_WINDOW_TILES, "{stats:?}");
}

/// A grid map stretched `stretch` times on a diagonal, from grid indices to
/// level pixels about `(cx, cy)`.
fn grid_map(stretch: f64, cx: f64, cy: f64, size: f64) -> [f64; 6] {
    let (p, q) = (0.5 * (stretch + 1.0), 0.5 * (stretch - 1.0));
    let half = size / 2.0;
    [p, q, cx - (p + q) * half, q, p, cy - (q + p) * half]
}

#[test]
fn the_cluster_grid_samplers_equal_their_whole_level_results() {
    let image = photograph(200, 150);
    let (pyramid, source) = sources(&image);
    let support: Vec<usize> = (0..25 * 25).filter(|p| p % 3 != 1).collect();
    for (level, stretch, cx, cy) in [
        (0, 4.0, 100.0, 75.0),
        (0, 16.0, 30.0, 140.0),
        (1, 8.0, 50.0, 37.0),
        (2, 1.0, 3.0, 3.0),
    ] {
        let a = grid_map(stretch, cx, cy, 25.0);
        let size = source.layout().level_size(level);
        for points in [
            GridPoints::Window {
                first: -3,
                size: 31,
            },
            GridPoints::Pixels {
                pixels: &support,
                resolution: 25,
            },
        ] {
            for edge in [GridEdge::Clamp, GridEdge::Missing, GridEdge::InFrame] {
                let whole = grid_cells(&a, points, edge, size)
                    .map(|cells| sample_grid(&HeldLevel::of(&pyramid, level), &cells));
                for (pool, _) in pools() {
                    let read = sample_source_grid(
                        &mut ViewRegions::new(&source, &pool),
                        &a,
                        level,
                        points,
                        edge,
                    )
                    .unwrap();
                    let bits = |v: &Option<Vec<f32>>| {
                        v.as_ref()
                            .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                    };
                    assert_eq!(bits(&read), bits(&whole), "level {level}, {edge:?}");
                }
            }
        }
    }
}

#[test]
fn the_affine_fast_path_equals_its_whole_level_result() {
    let image = photograph(200, 150);
    let (pyramid, source) = sources(&image);
    let pixels: Vec<usize> = (0..24 * 24).filter(|p| p % 5 != 2).collect();
    for (level, sampler, stretch) in [
        (0, Sampler::Bilinear, 3.0),
        (1, Sampler::BilinearMip, 12.0),
        (0, Sampler::Anisotropic, 16.0),
        (0, Sampler::Anisotropic, 1.5),
    ] {
        let scale = f64::from(1u32 << level);
        let a = grid_map(stretch, 100.0 / scale, 75.0 / scale, 24.0)
            .map(|c| c * if level == 0 { 0.3 } else { 1.0 });
        let a = [a[0], a[1], a[2] + 70.0, a[3], a[4], a[5] + 50.0];
        let mut whole = vec![0f32; 3 * pixels.len()];
        if sampler == Sampler::Anisotropic {
            sample_support_affine_aniso(&*pyramid, &a, &pixels, 24, &mut whole);
        } else {
            sample_support_affine(&HeldLevel::of(&pyramid, level), &a, &pixels, 24, &mut whole);
        }
        for (pool, fits) in pools() {
            let mut out = vec![0f32; 3 * pixels.len()];
            let path = render_source_affine_path(
                &mut ViewRegions::new(&source, &pool),
                &a,
                level,
                &pixels,
                24,
                sampler,
                &mut out,
            )
            .unwrap();
            assert_eq!(matches!(path, ReadPath::Copied { .. }), fits);
            assert_eq!(out, whole, "{sampler:?}, level {level}");
        }
    }
}

/// Positions whose windows straddle tile corners and the image's corners,
/// and some off the image.
fn positions(w: u32, h: u32) -> Vec<[f32; 2]> {
    let mut rng = Lcg(5);
    let (wf, hf) = (w as f32, h as f32);
    let mut positions = vec![
        [0.2, 0.3],
        [wf - 0.1, hf - 0.4],
        [wf - 0.6, 0.1],
        [0.0, hf],
        [-3.0, -2.0],
        [wf + 5.0, hf + 7.0],
        [32.0, 32.0],
        [31.5, 32.5],
        [64.2, 31.9],
        [33.0, 95.6],
    ];
    for _ in 0..40 {
        positions.push([rng.range(-2.0, wf + 2.0), rng.range(-2.0, hf + 2.0)]);
    }
    positions
}

#[test]
fn read_source_samples_gives_the_bilinear_value_of_the_whole_level() {
    let image = photograph(200, 150);
    let (pyramid, source) = sources(&image);
    for level in [0, 1] {
        let img = pyramid.level(level);
        let positions = positions(img.width(), img.height());
        for border in [0, 1] {
            for (pool, fits) in pools() {
                let (samples, path) = read_source_samples_path(
                    &mut ViewRegions::new(&source, &pool),
                    level,
                    &positions,
                    border,
                )
                .unwrap();
                assert_eq!(matches!(path, ReadPath::Copied { .. }), fits);
                let side = samples.side();
                let b = border as usize;
                for (k, &[x, y]) in positions.iter().enumerate() {
                    let window = samples.window(k);
                    let [fx, fy] = samples.fractions(k);
                    for c in 0..3 {
                        let v = |r: usize, q: usize| {
                            f32::from(window[((b + r) * side + b + q) * 3 + c])
                        };
                        let blended = (1.0 - fx) * (1.0 - fy) * v(0, 0)
                            + fx * (1.0 - fy) * v(0, 1)
                            + (1.0 - fx) * fy * v(1, 0)
                            + fx * fy * v(1, 1);
                        assert_eq!(
                            blended.to_bits(),
                            sample_bilinear_u8(img, x, y, c as u32).to_bits(),
                            "level {level}, border {border}, ({x}, {y})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn read_source_rect_returns_the_rectangles_pixels() {
    let image = photograph(200, 150);
    let (pyramid, source) = sources(&image);
    for (level, rect) in [
        (
            0,
            LevelRect {
                x0: 0,
                y0: 0,
                x1: 200,
                y1: 150,
            },
        ),
        (
            0,
            LevelRect {
                x0: 30,
                y0: 31,
                x1: 97,
                y1: 66,
            },
        ),
        (
            0,
            LevelRect {
                x0: 190,
                y0: 140,
                x1: 200,
                y1: 150,
            },
        ),
        (
            1,
            LevelRect {
                x0: 15,
                y0: 0,
                x1: 80,
                y1: 33,
            },
        ),
    ] {
        let img = pyramid.level(level);
        let mut expected = Vec::new();
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                for c in 0..3 {
                    expected.push(img.get_pixel(x, y, c));
                }
            }
        }
        for (pool, fits) in pools() {
            let read = read_source_rect(&source, &pool, level, rect).unwrap();
            assert_eq!(read.image().data(), &expected[..]);
            assert_eq!(pool.held_bytes() > 0, fits);
            drop(read);
            assert_eq!(pool.held_bytes(), 0);
        }
    }
    let pool = RegionPool::with_capacity(0);
    for (level, rect) in [
        (
            0,
            LevelRect {
                x0: 5,
                y0: 5,
                x1: 5,
                y1: 9,
            },
        ),
        (
            0,
            LevelRect {
                x0: 0,
                y0: 0,
                x1: 201,
                y1: 9,
            },
        ),
        (
            9,
            LevelRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
        ),
    ] {
        assert!(matches!(
            read_source_rect(&source, &pool, level, rect),
            Err(PhotographReadError::OutOfRange { .. })
        ));
    }
}

#[test]
fn a_kernel_calls_renders_never_hold_more_than_its_pool() {
    let image = photograph(200, 150);
    let (pyramid, source) = sources(&image);
    let maps = maps(200, 150);
    let typical = {
        let mut map = copy(&maps[0]);
        required_regions(&mut map, Sampler::Anisotropic, source.layout())
            .grown(REGION_REUSE_MARGIN)
            .byte_len(3)
    };
    // Room for about three renders at once.
    let pool = RegionPool::with_capacity(3 * typical);
    let threads = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    let results: Vec<(usize, ReadPath, bool)> = threads.install(|| {
        (0..8)
            .into_par_iter()
            .flat_map(|outer| {
                (0..maps.len())
                    .into_par_iter()
                    .map(|m| {
                        let sampler = SAMPLERS[(outer + m) % 3];
                        let mut map = copy(&maps[m]);
                        let (tile, path) = render_source_tile_path(
                            &mut ViewRegions::new(&source, &pool),
                            &mut map,
                            sampler,
                        )
                        .unwrap();
                        let whole = render_tile(&pyramid, &mut map, sampler);
                        let grown = required_regions(&mut map, sampler, source.layout())
                            .grown(REGION_REUSE_MARGIN)
                            .byte_len(3);
                        let equal = tile.data() == whole.data();
                        let path = match path {
                            ReadPath::Copied { bytes } => {
                                assert_eq!(bytes, grown);
                                path
                            }
                            tiles => tiles,
                        };
                        (m, path, equal)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    });
    assert!(results.iter().all(|&(_, _, equal)| equal));
    let stats = pool.stats();
    assert!(stats.peak_bytes <= pool.capacity(), "{stats:?}");
    assert!(stats.copied_reads > 0 && stats.tiled_reads > 0, "{stats:?}");
    assert_eq!(stats.copied_reads + stats.tiled_reads, results.len() as u64);
    assert_eq!(pool.held_bytes(), 0);
}

#[test]
fn the_grid_affine_and_sample_reads_need_no_more_than_their_exact_regions() {
    // The copied path reads regions grown by the margin, which would hide a
    // region one tap short; these reads take exactly the regions computed.
    let image = photograph(200, 150);
    let (pyramid, source) = sources(&image);
    let layout = source.layout();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

    // The cluster grids.
    let support: Vec<usize> = (0..25 * 25).collect();
    for (level, stretch, cx, cy) in [(0, 6.0, 100.0, 75.0), (1, 3.0, 40.0, 30.0)] {
        let a = grid_map(stretch, cx, cy, 25.0);
        let points = GridPoints::Pixels {
            pixels: &support,
            resolution: 25,
        };
        let cells = grid_cells(&a, points, GridEdge::Missing, layout.level_size(level)).unwrap();
        let mut regions = LevelRegions::new(layout);
        grid_regions(&mut regions, level, &cells);
        let held = copy_regions(&source, &regions).unwrap();
        assert_eq!(
            bits(&sample_grid(&HeldLevel::of(&held, level), &cells)),
            bits(&sample_grid(&HeldLevel::of(&pyramid, level), &cells))
        );
    }

    // The affine fast path, bilinear and anisotropic.
    let pixels: Vec<usize> = (0..24 * 24).collect();
    for (level, aniso, a) in [
        (1, false, [1.7, 0.4, 30.0, -0.3, 1.5, 20.0]),
        (0, true, [6.0, 5.0, 40.0, 5.0, 6.0, 10.0]),
    ] {
        let mut regions = LevelRegions::new(layout);
        affine_regions(&mut regions, &a, level, aniso, &pixels, 24);
        let held = copy_regions(&source, &regions).unwrap();
        let mut whole = vec![0f32; 3 * pixels.len()];
        let mut read = vec![0f32; 3 * pixels.len()];
        if aniso {
            sample_support_affine_aniso(&*pyramid, &a, &pixels, 24, &mut whole);
            sample_support_affine_aniso(&held, &a, &pixels, 24, &mut read);
        } else {
            sample_support_affine(&HeldLevel::of(&pyramid, level), &a, &pixels, 24, &mut whole);
            sample_support_affine(&HeldLevel::of(&held, level), &a, &pixels, 24, &mut read);
        }
        assert_eq!(bits(&read), bits(&whole), "aniso {aniso}");
    }

    // The 4 x 4 windows of positions well inside the image, so every region
    // edge is an inner edge.
    let positions: Vec<[f32; 2]> = (0..20)
        .map(|k| [60.0 + 3.7 * k as f32, 50.0 + 1.9 * k as f32])
        .collect();
    let regions = LevelRegions::for_samples(layout, 0, &positions, 1);
    let held = copy_regions(&source, &regions).unwrap();
    let (held, whole) = (HeldLevel::of(&held, 0), HeldLevel::of(&pyramid, 0));
    for &[x, y] in &positions {
        let (x0, y0, _, _) = bilinear_cell(x, y);
        for r in -1..3 {
            for q in -1..3 {
                let mut a = Vec::new();
                let mut b = Vec::new();
                held.with_pixel(x0 + q, y0 + r, |p| a.extend_from_slice(p));
                whole.with_pixel(x0 + q, y0 + r, |p| b.extend_from_slice(p));
                assert_eq!(a, b, "({x}, {y}) at {q}, {r}");
            }
        }
    }
}

#[test]
fn a_read_past_the_regions_of_a_region_pyramid_is_refused() {
    // Level 0 whole, levels 1 and 2 not held: a read of level 2 is an error
    // on either path, not the black of a level the pyramid does not hold.
    let levels = vec![
        Some(PyramidLevel {
            image: photograph(64, 64),
            origin: [0, 0],
            full_size: [64, 64],
        }),
        None,
        None,
    ];
    let pyramid = Arc::new(ImageU8Pyramid::from_regions(levels).unwrap());
    let source = TiledPhotograph::from_pyramid(pyramid).with_tile_size(TILE);
    for (pool, _) in pools() {
        let mut map = affine_map(16, [4.0, 0.0, 0.0, 0.0, 4.0, 0.0], 0);
        assert!(matches!(
            render_source_tile(
                &mut ViewRegions::new(&source, &pool),
                &mut map,
                Sampler::BilinearMip
            ),
            Err(PhotographReadError::OutOfRange { .. })
        ));
        let mut map = affine_map(16, [1.0, 0.0, 8.0, 0.0, 1.0, 8.0], 0);
        assert!(render_source_tile(
            &mut ViewRegions::new(&source, &pool),
            &mut map,
            Sampler::BilinearMip
        )
        .is_ok());
    }
}
