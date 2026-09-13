// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The scatter, on the CPU, where it can be read back.
//!
//! Worth having beyond the usual reason: placing a tile used to be the driver's
//! job, done by one `write_texture` per tile against an origin, and there was
//! nothing to assert without reading a texture back off the GPU. Now the
//! arithmetic is ours, so a tile landing one row or one cell out is a unit test
//! rather than a smeared atlas somebody notices in the viewport.

use super::Band;

/// The four bytes of the pixel at `(x, y)` of a band `atlas_width` wide.
fn pixel(band: &Band, atlas_width: u32, x: u32, y: u32) -> [u8; 4] {
    let at = (y as usize * atlas_width as usize + x as usize) * 4;
    band.bytes()[at..at + 4].try_into().expect("four bytes")
}

/// A tile whose every pixel carries `mark`, so where it landed is readable.
fn rgba_tile(resolution: usize, mark: u8) -> Vec<u8> {
    (0..resolution * resolution)
        .flat_map(|_| [mark, mark, mark, 255])
        .collect()
}

/// A tile lands in its own cell and nowhere else: at the cell's origin, at its
/// far corner, and not one pixel past either.
#[test]
fn a_tile_lands_in_its_own_cell() {
    let (width, res) = (16, 4);
    let mut band = Band::new(width, res);

    band.place_rgba(2, &rgba_tile(res as usize, 7));

    assert_eq!(pixel(&band, width, 8, 0), [7, 7, 7, 255], "cell origin");
    assert_eq!(pixel(&band, width, 11, 3), [7, 7, 7, 255], "far corner");
    assert_eq!(pixel(&band, width, 7, 0), [0, 0, 0, 0], "spilled left");
    assert_eq!(pixel(&band, width, 12, 0), [0, 0, 0, 0], "spilled right");
}

/// Each row of a tile lands on its own row of the band, which is the whole of
/// what the scatter has to get right: a stride mistake draws the tile sheared.
#[test]
fn a_tiles_rows_stay_in_order() {
    let (width, res) = (8, 4);
    let mut band = Band::new(width, res);
    // Row y of the tile is filled with y, so a sheared placement is visible.
    let tile: Vec<u8> = (0..res)
        .flat_map(|y| (0..res).flat_map(move |_| [y as u8, 0, 0, 255]))
        .collect();

    band.place_rgba(1, &tile);

    for y in 0..res {
        assert_eq!(
            pixel(&band, width, 4, y),
            [y as u8, 0, 0, 255],
            "row {y} of the tile",
        );
    }
}

/// Two tiles in neighbouring cells do not touch each other.
#[test]
fn neighbouring_tiles_do_not_overlap() {
    let (width, res) = (12, 4);
    let mut band = Band::new(width, res);

    band.place_rgba(0, &rgba_tile(res as usize, 1));
    band.place_rgba(1, &rgba_tile(res as usize, 2));
    band.place_rgba(2, &rgba_tile(res as usize, 3));

    for y in 0..res {
        assert_eq!(pixel(&band, width, 3, y)[0], 1, "end of the first cell");
        assert_eq!(pixel(&band, width, 4, y)[0], 2, "start of the second");
        assert_eq!(pixel(&band, width, 7, y)[0], 2, "end of the second");
        assert_eq!(pixel(&band, width, 8, y)[0], 3, "start of the third");
    }
}

/// An RGB tile gains an opaque alpha, and its colour channels keep their order.
#[test]
fn an_rgb_tile_gains_an_opaque_alpha() {
    let (width, res) = (8, 2);
    let mut band = Band::new(width, res);
    let tile: Vec<u8> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];

    band.place_rgb(1, &tile);

    assert_eq!(pixel(&band, width, 2, 0), [10, 20, 30, 255]);
    assert_eq!(pixel(&band, width, 3, 0), [40, 50, 60, 255]);
    assert_eq!(pixel(&band, width, 2, 1), [70, 80, 90, 255]);
    assert_eq!(pixel(&band, width, 3, 1), [100, 110, 120, 255]);
}

/// The band is reused down the atlas, so the cells a short last band does not
/// fill have to stop holding the band before it. This is the one thing a
/// per-tile upload could not get wrong and this one can.
#[test]
fn a_short_band_does_not_keep_the_previous_bands_tiles() {
    let (width, res) = (12, 4);
    let mut band = Band::new(width, res);
    for col in 0..3 {
        band.place_rgba(col, &rgba_tile(res as usize, 9));
    }
    band.blank_from(3);

    // The next band down holds one tile where the last held three.
    band.place_rgba(0, &rgba_tile(res as usize, 4));
    band.blank_from(1);

    for y in 0..res {
        assert_eq!(pixel(&band, width, 0, y)[0], 4, "the tile that was placed");
        assert_eq!(pixel(&band, width, 4, y), [0, 0, 0, 0], "a stale neighbour");
        assert_eq!(pixel(&band, width, 11, y), [0, 0, 0, 0], "a stale end");
    }
}

/// A full band blanks nothing, which is what keeps the memset off every band
/// but the last.
#[test]
fn a_full_band_is_left_alone() {
    let (width, res) = (8, 2);
    let mut band = Band::new(width, res);
    band.place_rgba(0, &rgba_tile(res as usize, 5));
    band.place_rgba(1, &rgba_tile(res as usize, 6));
    band.place_rgba(2, &rgba_tile(res as usize, 7));
    band.place_rgba(3, &rgba_tile(res as usize, 8));

    band.blank_from(4);

    for (col, mark) in [(0, 5), (1, 6), (2, 7), (3, 8)] {
        assert_eq!(pixel(&band, width, col * 2, 0)[0], mark, "cell {col}");
    }
}

/// The pitch is the atlas row, not the tile row: it is what the single
/// `write_texture` reads the band back at, and the two have to agree.
#[test]
fn the_pitch_is_the_atlas_row() {
    let band = Band::new(145 * 24, 24);
    assert_eq!(band.bytes_per_row(), 145 * 24 * 4);
    assert_eq!(band.bytes().len(), 145 * 24 * 4 * 24);
}
