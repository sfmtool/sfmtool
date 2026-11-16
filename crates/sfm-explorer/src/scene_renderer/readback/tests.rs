// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::search_5x5_depth;

/// A staging buffer's 5×5 region, 256 bytes to a row as the copy lays it out,
/// with `cells` set and every other pixel holding no depth.
fn region(cells: &[((usize, usize), f32)]) -> Vec<u8> {
    let mut data = vec![0u8; 5 * 256];
    for &((row, col), value) in cells {
        let offset = row * 256 + col * 4;
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    data
}

#[test]
fn a_surfel_under_the_cursor_gives_its_depth() {
    // A surfel stores its view-space depth negated, so the EDL pass leaves it
    // unshaded; Alt+click still needs the depth.
    assert_eq!(search_5x5_depth(&region(&[((2, 2), -4.0)])), Some(4.0));
}

#[test]
fn the_nearest_depth_in_a_ring_is_chosen_by_magnitude() {
    // A surfel at 2 beside a splat at 6: the surfel is nearer.
    let data = region(&[((1, 1), 6.0), ((3, 3), -2.0)]);
    assert_eq!(search_5x5_depth(&data), Some(2.0));
    // A surfel at 5 beside a splat at 3: the splat is nearer.
    let data = region(&[((1, 1), -5.0), ((3, 3), 3.0)]);
    assert_eq!(search_5x5_depth(&data), Some(3.0));
    // The outer ring, reached only when the inner one holds nothing.
    let data = region(&[((0, 0), -7.0), ((4, 4), 8.0)]);
    assert_eq!(search_5x5_depth(&data), Some(7.0));
}

#[test]
fn a_region_holding_no_depth_gives_none() {
    assert_eq!(search_5x5_depth(&region(&[])), None);
}
