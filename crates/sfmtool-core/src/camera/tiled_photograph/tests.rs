// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A photograph [`TiledPhotograph::from_image`] makes holds every tile of the
//! image's pyramid, and gives every tile, rectangle and level of it bit for
//! bit, cut at 512 px and cut again at a small tile size.

use super::*;

/// A `width` x `height` photograph of three channels of noise.
fn photograph(width: u32, height: u32) -> ImageU8 {
    let mut state = 7u64;
    let data = (0..width * height * 3)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        })
        .collect();
    ImageU8::new(width, height, 3, data)
}

/// The tile `key` of `pyramid` cut at `ts` px, worked out pixel by pixel:
/// its core and a one-pixel border, each address clamped to the level.
fn expected_tile(pyramid: &ImageU8Pyramid, ts: u32, key: TileKey) -> ([u32; 2], [u32; 2], ImageU8) {
    let level = pyramid.level(key.level as usize);
    let (w, h, c) = (level.width(), level.height(), level.channels());
    let origin = [key.col * ts, key.row * ts];
    let size = [ts.min(w - origin[0]), ts.min(h - origin[1])];
    let mut data = Vec::new();
    for j in 0..i64::from(size[1]) + 2 {
        let y = (i64::from(origin[1]) - 1 + j).clamp(0, i64::from(h) - 1) as u32;
        for i in 0..i64::from(size[0]) + 2 {
            let x = (i64::from(origin[0]) - 1 + i).clamp(0, i64::from(w) - 1) as u32;
            data.extend((0..c).map(|ch| level.get_pixel(x, y, ch)));
        }
    }
    (
        origin,
        size,
        ImageU8::new(size[0] + 2, size[1] + 2, c, data),
    )
}

/// `level` of `pyramid`'s pixels `rect`.
fn expected_region(pyramid: &ImageU8Pyramid, level: usize, rect: LevelRect) -> Vec<u8> {
    let image = pyramid.level(level);
    let mut data = Vec::new();
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            data.extend((0..image.channels()).map(|ch| image.get_pixel(x, y, ch)));
        }
    }
    data
}

/// `source`, cut at `ts` px, has `pyramid`'s levels, holds every tile of its
/// layout with `pyramid`'s pixels and its border, and reads each whole level
/// and rectangles across tile edges and at the level's edges as `pyramid`
/// holds them.
fn assert_holds_the_pyramid(source: &TiledPhotograph, pyramid: &ImageU8Pyramid, ts: u32) {
    let layout = source.layout();
    assert_eq!(layout.tile_size(), ts);
    assert_eq!(layout.levels(), pyramid.num_levels());
    assert_eq!(source.channels(), pyramid.level(0).channels());
    assert!(!source.may_lack_tiles());
    let Pixels::Tiles { held, every: true } = &source.pixels else {
        panic!("a photograph from an image holds every tile");
    };
    assert_eq!(held.len(), tile_count(layout));
    for level in 0..layout.levels() {
        let image = pyramid.level(level);
        let (w, h) = layout.level_size(level);
        assert_eq!((w, h), (image.width(), image.height()), "level {level}");
        let (cols, rows) = layout.tiles(level);
        for row in 0..rows {
            for col in 0..cols {
                let key = TileKey {
                    level: level as u8,
                    col,
                    row,
                };
                let tile = source.tile(key).expect("a tile it holds");
                let (origin, size, pixels) = expected_tile(pyramid, ts, key);
                assert_eq!(tile.key, key);
                assert_eq!((tile.origin, tile.size), (origin, size), "{key:?}");
                assert_eq!(tile.pixels.width(), pixels.width(), "{key:?}");
                assert_eq!(tile.pixels.data(), pixels.data(), "{key:?}");
            }
        }
        let whole = LevelRect {
            x0: 0,
            y0: 0,
            x1: w,
            y1: h,
        };
        // Across the first tile edge on each axis where the level has one,
        // and a pixel at the level's far corner.
        let across = LevelRect {
            x0: (ts.min(w) - 1).saturating_sub(2),
            y0: (ts.min(h) - 1).saturating_sub(2),
            x1: (ts + 3).min(w),
            y1: (ts + 3).min(h),
        };
        let corner = LevelRect {
            x0: w - 1,
            y0: h - 1,
            x1: w,
            y1: h,
        };
        for rect in [whole, across, corner] {
            let read = source.region(level, rect).expect("a rectangle inside");
            assert_eq!(
                read.data(),
                &expected_region(pyramid, level, rect)[..],
                "level {level}, {rect:?}"
            );
        }
    }
}

#[test]
fn a_photograph_from_an_image_holds_every_tile_of_its_pyramid() {
    // Three 512 px tiles across and two down at level 0, the last of each
    // partial at the right and bottom edges.
    let image = photograph(1037, 555);
    let pyramid = ImageU8Pyramid::build(&image, ImageU8Pyramid::full_levels(1037, 555));
    let source = TiledPhotograph::from_image(image);
    assert_holds_the_pyramid(&source, &pyramid, PHOTOGRAPH_TILE_SIZE);
    // Cut again at 16 px, from the 512 px tiles, as the tests cut small
    // tiles so that a small image crosses many of them.
    let small = photograph(203, 150);
    let small_pyramid = ImageU8Pyramid::build(&small, ImageU8Pyramid::full_levels(203, 150));
    let cut_again = TiledPhotograph::from_image(small).with_tile_size(16);
    assert_holds_the_pyramid(&cut_again, &small_pyramid, 16);
}

/// `from_image`, which cuts one level at a time and makes the next from it,
/// holds the tiles `cut` holds of the whole pyramid `ImageU8Pyramid::from_image`
/// builds, bit for bit, and the same levels, for sizes whose pyramid stops
/// short of the full depth (a side of 1), sides of 2 and 3, and odd sizes
/// over many tiles.
#[test]
fn from_image_cuts_the_tiles_of_the_whole_pyramid_level_by_level() {
    for (w, h) in [
        (1037, 555),
        (1, 1),
        (1, 9),
        (3, 1000),
        (2, 2),
        (515, 7),
        (33, 1030),
    ] {
        let image = photograph(w, h);
        let pyramid = ImageU8Pyramid::build(&image, ImageU8Pyramid::full_levels(w, h));
        let whole = TiledPhotograph::cut(&pyramid, PHOTOGRAPH_TILE_SIZE);
        let source = TiledPhotograph::from_image(image);
        assert_eq!(source.layout().levels(), pyramid.num_levels(), "{w} x {h}");
        assert_eq!(source.channels(), whole.channels());
        let (Pixels::Tiles { held, every: true }, Pixels::Tiles { held: expected, .. }) =
            (&source.pixels, &whole.pixels)
        else {
            panic!("both hold every tile");
        };
        assert_eq!(held.len(), expected.len(), "{w} x {h}");
        for (key, tile) in expected {
            let got = &held[key];
            assert_eq!((got.origin, got.size), (tile.origin, tile.size), "{key:?}");
            assert_eq!(got.pixels.width(), tile.pixels.width(), "{key:?}");
            assert_eq!(got.pixels.data(), tile.pixels.data(), "{w} x {h} {key:?}");
        }
    }
}
