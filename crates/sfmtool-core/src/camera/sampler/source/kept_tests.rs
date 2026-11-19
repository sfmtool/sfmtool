// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! View-region pyramids assembled from a photograph's tiles of 32 px, the
//! regions a view keeps, grown by the reuse margin, and their reuse and
//! release.

use std::sync::Arc;

use super::*;
use crate::camera::sampler::render_tile;
use crate::camera::sampler::test_support::{affine_map, copy, photograph, SAMPLERS};

const TILE: u32 = 32;

/// The whole pyramid of a 200 x 150 photograph, and the same photograph cut
/// into tiles of [`TILE`] px, so no level is a whole number of tiles.
fn sources() -> (Arc<ImageU8Pyramid>, TiledPhotograph) {
    let image = photograph(200, 150);
    let levels = ImageU8Pyramid::full_levels(200, 150);
    let pyramid = Arc::new(ImageU8Pyramid::build(&image, levels));
    let source = TiledPhotograph::from_pyramid(pyramid.clone()).with_tile_size(TILE);
    (pyramid, source)
}

/// The pixels of `rect` of `image`, row-major.
fn pixels_of(image: &ImageU8, rect: LevelRect) -> Vec<u8> {
    let mut data = Vec::new();
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            for c in 0..image.channels() {
                data.push(image.get_pixel(x, y, c));
            }
        }
    }
    data
}

fn rect(x0: u32, y0: u32, x1: u32, y1: u32) -> LevelRect {
    LevelRect { x0, y0, x1, y1 }
}

#[test]
fn regions_assembled_from_tiles_equal_the_whole_pyramids_pixels() {
    let (pyramid, source) = sources();
    for level in 0..pyramid.num_levels() {
        let (w, h) = source.layout().level_size(level);
        // The whole level; a rectangle crossing tile edges on both axes;
        // one ending on a tile edge; one inside a tile; a single pixel on
        // each side of a tile corner; and the pixels at each image edge.
        let mut rects = vec![
            rect(0, 0, w, h),
            rect(w / 5, h / 7, w - w / 6, h - h / 9),
            rect(0, 0, w.min(TILE), h.min(TILE)),
            rect(1, 1, w.min(TILE - 1), h.min(TILE - 1)),
            rect(0, 0, 1, h),
            rect(w - 1, 0, w, h),
            rect(0, 0, w, 1),
            rect(0, h - 1, w, h),
            rect(w - 1, h - 1, w, h),
        ];
        if w > TILE && h > TILE {
            rects.push(rect(TILE - 1, TILE - 1, TILE + 1, TILE + 1));
            rects.push(rect(TILE - 1, TILE, TILE, TILE + 1));
        }
        for r in rects
            .into_iter()
            .filter(|r| r.width() > 0 && r.height() > 0)
        {
            let read = source.region(level, r).unwrap();
            assert_eq!(
                (read.width(), read.height(), read.channels()),
                (r.width(), r.height(), 3)
            );
            assert_eq!(
                read.data(),
                &pixels_of(pyramid.level(level), r)[..],
                "level {level}, {r:?}"
            );
        }
    }

    // A view-region pyramid of every level's regions, assembled from tiles.
    let mut regions = LevelRegions::new(source.layout());
    for level in 0..pyramid.num_levels() {
        let (w, h) = source.layout().level_size(level);
        regions.set_region(level, rect(w / 3, h / 4, w, h - h / 5));
    }
    let held = copy_regions(&source, &regions).unwrap();
    for level in 0..pyramid.num_levels() {
        let r = regions.region(level).unwrap();
        assert_eq!(held.level_origin(level), [r.x0, r.y0]);
        assert_eq!(
            held.level(level).data(),
            &pixels_of(pyramid.level(level), r)[..],
            "level {level}"
        );
    }
}

/// The region `read_regions` keeps for the required region `required` of
/// level `level`, with the bytes it reserved.
fn kept_for(source: &TiledPhotograph, level: usize, required: LevelRect) -> (LevelRect, u64) {
    let pool = RegionPool::with_capacity(u64::MAX);
    let mut view = ViewRegions::new(source, &pool);
    let mut regions = LevelRegions::new(source.layout());
    regions.set_region(level, required);
    let ((), path) = read_regions(&mut view, &regions, |_| ()).unwrap();
    let kept = view.kept_regions().unwrap();
    for other in (0..kept.levels()).filter(|&l| l != level) {
        assert_eq!(kept.region(other), None);
    }
    assert_eq!(
        path,
        ReadPath::Copied {
            bytes: view.kept_bytes()
        }
    );
    assert_eq!(pool.held_bytes(), view.kept_bytes());
    (kept.region(level).unwrap(), view.kept_bytes())
}

#[test]
fn each_kept_region_is_its_required_region_grown_by_the_margin_and_clipped_to_the_level() {
    let (_, source) = sources();
    // A side of `side` px grows by `grow` px on each side: a quarter of the
    // side, rounded up.
    for (side, grow) in [(1u32, 1u32), (2, 1), (3, 1), (61, 16)] {
        // Inside the image, the region grows on every side.
        let (kept, bytes) = kept_for(&source, 0, rect(80, 50, 80 + side, 50 + side));
        assert_eq!(
            kept,
            rect(80 - grow, 50 - grow, 80 + side + grow, 50 + side + grow)
        );
        assert_eq!(bytes, u64::from((side + 2 * grow).pow(2)) * 3);

        // At each edge and corner of the image, and a pixel inside each
        // edge, it is clipped to the level, 200 x 150.
        for (x0, y0) in [
            (0, 50),
            (1, 50),
            (200 - side, 50),
            (199 - side, 50),
            (80, 0),
            (80, 1),
            (80, 150 - side),
            (80, 149 - side),
            (0, 0),
            (200 - side, 150 - side),
        ] {
            let required = rect(x0, y0, x0 + side, y0 + side);
            let (kept, bytes) = kept_for(&source, 0, required);
            let expected = rect(
                x0.saturating_sub(grow),
                y0.saturating_sub(grow),
                (x0 + side + grow).min(200),
                (y0 + side + grow).min(150),
            );
            assert_eq!(kept, expected, "side {side} at ({x0}, {y0})");
            assert_eq!(
                bytes,
                u64::from(expected.width() * expected.height()) * 3,
                "side {side} at ({x0}, {y0})"
            );
        }
    }
    // A coarser level clips to its own size, 25 x 18 at level 3.
    let (kept, _) = kept_for(&source, 3, rect(20, 13, 25, 18));
    assert_eq!(kept, rect(18, 11, 25, 18));
}

#[test]
fn a_render_inside_the_kept_regions_reuses_them_and_one_outside_reassembles() {
    let (pyramid, source) = sources();
    let pool = RegionPool::with_capacity(u64::MAX);
    for sampler in SAMPLERS {
        let mut view = ViewRegions::new(&source, &pool);
        // A 16 x 16 tile at 1.5 px a sample about (90, 70), the same tile
        // moved 2 px, which stays inside the margin, and moved 40 px, which
        // does not.
        for (dx, dy, reused) in [
            (0.0, 0.0, false),
            (2.0, -2.0, true),
            (-1.5, 1.0, true),
            (40.0, 30.0, false),
            (41.0, 30.5, true),
        ] {
            let map = affine_map(16, [1.5, 0.2, 78.0 + dx, -0.1, 1.5, 58.0 + dy], 0);
            let mut kept_map = copy(&map);
            let (tile, path) = render_source_tile_path(&mut view, &mut kept_map, sampler).unwrap();
            let required = required_regions(&mut copy(&map), sampler, source.layout());
            let kept = view.kept_regions().unwrap().clone();
            match path {
                ReadPath::Reused { bytes } => {
                    assert!(reused, "{sampler:?} at ({dx}, {dy}) reused");
                    assert_eq!(bytes, view.kept_bytes());
                }
                ReadPath::Copied { bytes } => {
                    assert!(!reused, "{sampler:?} at ({dx}, {dy}) reassembled");
                    assert_eq!(kept, required.grown(REGION_REUSE_MARGIN));
                    assert_eq!(bytes, view.kept_bytes());
                }
                ReadPath::Tiles(_) => panic!("the pool holds every read"),
            }
            // The kept regions hold the render's required regions, and its
            // values are the whole pyramid's and a fresh view's, bit for
            // bit.
            assert!(holds(&kept, &required));
            let whole = render_tile(&pyramid, &mut copy(&map), sampler);
            assert_eq!(tile.data(), whole.data(), "{sampler:?} at ({dx}, {dy})");
            let fresh_pool = RegionPool::with_capacity(u64::MAX);
            let fresh = render_source_tile(
                &mut ViewRegions::new(&source, &fresh_pool),
                &mut copy(&map),
                sampler,
            )
            .unwrap();
            assert_eq!(tile.data(), fresh.data());
        }
    }
    let stats = pool.stats();
    assert_eq!(
        (stats.copied_reads, stats.reused_reads),
        (6, 9),
        "{stats:?}"
    );
    assert_eq!(stats.tiled_reads, 0);
}

#[test]
fn the_other_reads_reuse_the_regions_a_render_kept() {
    let (pyramid, source) = sources();
    let pool = RegionPool::with_capacity(u64::MAX);
    let mut view = ViewRegions::new(&source, &pool);
    let mut map = affine_map(24, [1.0, 0.0, 60.0, 0.0, 1.0, 40.0], 0);
    let (_, path) = render_source_tile_path(&mut view, &mut map, Sampler::Bilinear).unwrap();
    assert!(matches!(path, ReadPath::Copied { .. }));

    // Positions inside the render's footprint read the kept level 0.
    let inside: Vec<[f32; 2]> = (0..10)
        .map(|k| [62.0 + 2.0 * k as f32, 42.0 + 1.5 * k as f32])
        .collect();
    let (samples, path) = read_source_samples_path(&mut view, 0, &inside, 1).unwrap();
    assert!(matches!(path, ReadPath::Reused { .. }), "{path:?}");
    let level = pyramid.level(0);
    for (k, &[x, y]) in inside.iter().enumerate() {
        let (x0, y0, _, _) = bilinear_cell(x, y);
        let expected: Vec<u8> = (-1..3)
            .flat_map(|r| (-1..3).map(move |q| (q, r)))
            .flat_map(|(q, r)| (0..3).map(move |c| (q, r, c)))
            .map(|(q, r, c)| level.get_pixel((x0 + q) as u32, (y0 + r) as u32, c))
            .collect();
        assert_eq!(samples.window(k), &expected[..], "({x}, {y})");
    }

    // A level the render did not read, and positions past the kept region,
    // reassemble.
    let (_, path) = read_source_samples_path(&mut view, 1, &inside, 0).unwrap();
    assert!(matches!(path, ReadPath::Copied { .. }), "{path:?}");
    let (_, path) = read_source_samples_path(&mut view, 0, &[[150.0, 120.0]], 0).unwrap();
    assert!(matches!(path, ReadPath::Copied { .. }), "{path:?}");
}

#[test]
fn kept_regions_are_counted_in_the_pool_and_released_when_dropped() {
    let (_, source) = sources();
    let map = || affine_map(16, [1.0, 0.0, 60.0, 0.0, 1.0, 40.0], 0);
    let far = || affine_map(16, [1.0, 0.0, 140.0, 0.0, 1.0, 100.0], 0);
    let one = {
        let pool = RegionPool::with_capacity(u64::MAX);
        let mut view = ViewRegions::new(&source, &pool);
        render_source_tile(&mut view, &mut map(), Sampler::Bilinear).unwrap();
        view.kept_bytes()
    };
    assert!(one > 0);

    // A pool that holds one view's kept regions.
    let pool = RegionPool::with_capacity(one);
    let mut first = ViewRegions::new(&source, &pool);
    let (_, path) = render_source_tile_path(&mut first, &mut map(), Sampler::Bilinear).unwrap();
    assert_eq!(path, ReadPath::Copied { bytes: one });
    assert_eq!(pool.held_bytes(), one);

    // While the first view keeps its regions, a second finds the pool full
    // and takes the tile path, keeping nothing.
    let mut second = ViewRegions::new(&source, &pool);
    let (_, path) = render_source_tile_path(&mut second, &mut map(), Sampler::Bilinear).unwrap();
    assert!(matches!(path, ReadPath::Tiles(_)), "{path:?}");
    assert_eq!(second.kept_bytes(), 0);
    assert_eq!(pool.held_bytes(), one);

    // The first view's next read elsewhere gives its regions back before it
    // reserves the new ones, so the full pool still holds them.
    let (_, path) = render_source_tile_path(&mut first, &mut far(), Sampler::Bilinear).unwrap();
    assert_eq!(path, ReadPath::Copied { bytes: one });
    assert_eq!(pool.held_bytes(), one);

    // Dropping the first view releases them, and the second can copy.
    drop(first);
    assert_eq!(pool.held_bytes(), 0);
    let (_, path) = render_source_tile_path(&mut second, &mut map(), Sampler::Bilinear).unwrap();
    assert_eq!(path, ReadPath::Copied { bytes: one });
    assert_eq!(pool.held_bytes(), one);
    drop(second);
    assert_eq!(pool.held_bytes(), 0);
    assert_eq!(pool.stats().peak_bytes, one);
}
