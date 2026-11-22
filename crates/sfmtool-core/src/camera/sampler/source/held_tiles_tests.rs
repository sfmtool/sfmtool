// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Reads of a photograph made of the photograph tiles it holds
//! (`TiledPhotograph::from_tiles`): each is refused, naming the tiles it
//! needs, until the photograph holds them, and then equals the whole
//! pyramid's read bit for bit, through copied regions and through tiles of
//! 32 px.

use std::sync::Arc;

use super::*;
use crate::camera::sampler::render_tile;
use crate::camera::sampler::test_support::{maps, photograph, SAMPLERS};
use crate::camera::tiled_photograph::{PhotographTile, TileKey};

const TILE: u32 = 32;

/// The whole pyramid of a 200 x 150 photograph, and the same photograph cut
/// into tiles of [`TILE`] px, which the tiles held are cut from.
fn sources() -> (Arc<ImageU8Pyramid>, TiledPhotograph) {
    let image = photograph(200, 150);
    let levels = ImageU8Pyramid::full_levels(200, 150);
    let pyramid = Arc::new(ImageU8Pyramid::build(&image, levels));
    let source = TiledPhotograph::from_pyramid(pyramid.clone()).with_tile_size(TILE);
    (pyramid, source)
}

/// The coarsest level's first tile, which every photograph made of tiles in
/// these tests holds, so it has channels before it holds anything a read
/// needs.
fn first_tile(source: &TiledPhotograph) -> TileKey {
    TileKey {
        level: (source.layout().levels() - 1) as u8,
        col: 0,
        row: 0,
    }
}

/// `source`'s photograph made of its tiles `keys`.
fn held(source: &TiledPhotograph, keys: &[TileKey]) -> TiledPhotograph {
    let tiles = keys.iter().map(|&key| source.tile(key).expect("a tile"));
    TiledPhotograph::from_tiles(source.layout().clone(), tiles).expect("the layout's tiles")
}

/// `read` of a photograph made of the first tile alone, refused with the
/// tiles it needs or answered when it needs none past the first, and then
/// of the photograph made of exactly those tiles, which reads; the
/// photograph made of those tiles less any one of them is refused naming the
/// same tiles. Answers the read's result and the tiles it needed.
fn read_held<T>(
    source: &TiledPhotograph,
    mut read: impl FnMut(&TiledPhotograph) -> Result<T, PhotographReadError>,
) -> (T, Vec<TileKey>) {
    let first = first_tile(source);
    let keys = match read(&held(source, &[first])) {
        Ok(value) => return (value, vec![first]),
        Err(PhotographReadError::NeedsTiles { keys }) => keys,
        Err(e) => panic!("{e}"),
    };
    assert!(!keys.is_empty());
    for left_out in 0..keys.len() {
        let fewer: Vec<TileKey> = (keys.iter().enumerate())
            .filter(|&(k, _)| k != left_out)
            .map(|(_, &key)| key)
            .collect();
        let fewer = if fewer.is_empty() { vec![first] } else { fewer };
        match read(&held(source, &fewer)) {
            Err(PhotographReadError::NeedsTiles { keys: again }) => assert_eq!(again, keys),
            Err(e) => panic!("{e}"),
            Ok(_) => panic!("read without tile {:?}", keys[left_out]),
        }
    }
    let value = read(&held(source, &keys)).expect("the tiles the read needs");
    (value, keys)
}

/// A pool every read fits in, and one no read fits in.
fn pools() -> [RegionPool; 2] {
    [
        RegionPool::with_capacity(u64::MAX),
        RegionPool::with_capacity(0),
    ]
}

#[test]
fn every_render_of_its_needed_tiles_equals_the_whole_pyramid() {
    let (pyramid, source) = sources();
    let levels = source.layout().levels();
    let mut needed_several = false;
    for (m, map) in maps(200, 150).into_iter().enumerate() {
        for sampler in SAMPLERS {
            let mut whole_map = crate::camera::sampler::test_support::copy(&map);
            let whole = render_tile(&pyramid, &mut whole_map, sampler);
            for pool in pools() {
                let (tile, keys) = read_held(&source, |src| {
                    let mut map = crate::camera::sampler::test_support::copy(&map);
                    render_source_tile(&mut ViewRegions::new(src, &pool), &mut map, sampler)
                });
                assert_eq!(tile.data(), whole.data(), "map {m}, {sampler:?}");
                // Every tile named is of a level the photograph has.
                assert!(keys.iter().all(|k| usize::from(k.level) < levels));
                needed_several |= keys.len() > 1;
            }
        }
    }
    assert!(needed_several);
}

#[test]
fn the_grid_and_sample_reads_of_their_needed_tiles_equal_the_whole_pyramid() {
    let (pyramid, source) = sources();
    let a = [3.0, 1.0, 60.0, 1.0, 3.0, 40.0];
    let whole_pool = RegionPool::with_capacity(u64::MAX);
    let whole = TiledPhotograph::from_pyramid(pyramid.clone());
    for pool in pools() {
        let grid = |src: &TiledPhotograph| {
            sample_source_grid(
                &mut ViewRegions::new(src, &pool),
                &a,
                0,
                GridPoints::Window {
                    first: -2,
                    size: 20,
                },
                GridEdge::Clamp,
            )
        };
        let (read, _) = read_held(&source, grid);
        let expected = sample_source_grid(
            &mut ViewRegions::new(&whole, &whole_pool),
            &a,
            0,
            GridPoints::Window {
                first: -2,
                size: 20,
            },
            GridEdge::Clamp,
        )
        .unwrap();
        assert_eq!(read, expected);

        let positions = [[10.5f32, 3.25], [95.0, 70.75], [199.9, 149.9], [-4.0, 31.5]];
        for level in [0, 1] {
            let samples = |src: &TiledPhotograph| {
                read_source_samples(&mut ViewRegions::new(src, &pool), level, &positions, 1)
            };
            let (read, _) = read_held(&source, samples);
            let expected = read_source_samples(
                &mut ViewRegions::new(&whole, &whole_pool),
                level,
                &positions,
                1,
            )
            .unwrap();
            for k in 0..positions.len() {
                assert_eq!(read.window(k), expected.window(k), "level {level}");
                assert_eq!(read.fractions(k), expected.fractions(k));
            }
        }
    }
}

#[test]
fn a_rectangle_of_its_needed_tiles_equals_the_whole_level() {
    let (pyramid, source) = sources();
    for (level, rect) in [
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
            1,
            LevelRect {
                x0: 0,
                y0: 0,
                x1: 100,
                y1: 75,
            },
        ),
    ] {
        for pool in pools() {
            let (read, keys) = read_held(&source, |src| {
                read_source_rect(src, &pool, level, rect).map(|r| r.image().data().to_vec())
            });
            let img = pyramid.level(level);
            let mut expected = Vec::new();
            for y in rect.y0..rect.y1 {
                let row = (y * img.width()) as usize * 3;
                expected.extend_from_slice(
                    &img.data()[row + rect.x0 as usize * 3..row + rect.x1 as usize * 3],
                );
            }
            assert_eq!(read, expected, "level {level}");
            let tiles = |lo: u32, hi: u32| (hi - 1) / TILE - lo / TILE + 1;
            assert_eq!(
                keys.len() as u32,
                tiles(rect.x0, rect.x1) * tiles(rect.y0, rect.y1)
            );
        }
    }
}

#[test]
fn from_tiles_refuses_tiles_the_layout_does_not_have_or_place() {
    let (_, source) = sources();
    let layout = || source.layout().clone();
    let tile = |key| source.tile(key).unwrap();
    let key = TileKey {
        level: 0,
        col: 1,
        row: 2,
    };
    let refused = |tiles: Vec<Arc<PhotographTile>>| {
        matches!(
            TiledPhotograph::from_tiles(layout(), tiles),
            Err(PhotographReadError::OutOfRange { .. })
        )
    };
    assert!(refused(Vec::new()), "no tiles");
    let moved = |f: &dyn Fn(&mut PhotographTile)| {
        let own = tile(key);
        let p = &own.pixels;
        let mut moved = PhotographTile {
            key,
            origin: own.origin,
            size: own.size,
            pixels: ImageU8::new(p.width(), p.height(), p.channels(), p.data().to_vec()),
        };
        f(&mut moved);
        Arc::new(moved)
    };
    assert!(refused(vec![moved(&|t| t.origin[0] += 1)]), "moved");
    assert!(refused(vec![moved(&|t| t.size[1] -= 1)]), "resized");
    assert!(
        refused(vec![moved(
            &|t| t.key.level = source.layout().levels() as u8
        )]),
        "a level past the layout's"
    );
    assert!(
        refused(vec![moved(&|t| t.key.col = 7)]),
        "a column past the level's"
    );
    assert!(
        refused(vec![moved(
            &|t| t.pixels = ImageU8::new(3, 3, 3, vec![0; 27])
        )]),
        "pixels of another size"
    );
    let grey = moved(&|t| {
        let [w, h] = [t.pixels.width(), t.pixels.height()];
        t.pixels = ImageU8::new(w, h, 1, vec![0; (w * h) as usize]);
    });
    assert!(
        refused(vec![tile(key), grey]),
        "tiles of 3 and of 1 channels"
    );
    assert!(!refused(vec![tile(key)]), "the layout's own tile");
}

#[test]
fn the_clipped_share_of_its_needed_tiles_equals_the_whole_pyramid_s() {
    use crate::patch::reference_view::source_clipped_share;
    let (pyramid, source) = sources();
    let whole = TiledPhotograph::from_pyramid(pyramid);
    // An outline across nine of the 32 px tiles, one past the image's right
    // edge, one too thin to hold a pixel centre, which reads the pixels
    // under its corners, and one wholly past the right edge, whose corners'
    // pixels, clamped to the image, lie in tiles its box has none of.
    for outline in [
        vec![[20.0, 15.0], [90.0, 25.0], [80.0, 85.0], [25.0, 70.0]],
        vec![[150.0, 100.0], [230.0, 110.0], [210.0, 149.0]],
        vec![[40.2, 40.2], [100.3, 40.3], [100.3, 40.4]],
        vec![[210.0, 10.0], [230.0, 12.0], [220.0, 70.0]],
    ] {
        let expected = source_clipped_share(&whole, &outline).unwrap();
        let (share, keys) = read_held(&source, |src| source_clipped_share(src, &outline));
        assert_eq!(share, expected, "{outline:?}");
        assert!(keys.len() > 1, "{outline:?} reads several tiles: {keys:?}");
    }
}
