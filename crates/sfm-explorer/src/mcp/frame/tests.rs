// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Unpadding a GPU readback and encoding it, on buffers whose bytes say
//! where they came from.

use super::*;

/// A padded buffer of `width × height` pixels whose bytes say where they
/// came from, so an unpad that reads the wrong row or the wrong column is
/// visible in the output rather than merely wrong in size.
fn padded(width: u32, height: u32, padded_bytes_per_row: u32) -> Vec<u8> {
    let mut data = vec![0xEE; (padded_bytes_per_row * height) as usize];
    for row in 0..height {
        for column in 0..width {
            let at = (row * padded_bytes_per_row + column * BYTES_PER_PIXEL) as usize;
            data[at] = 10 + row as u8; // B, or R
            data[at + 1] = 20 + column as u8; // G
            data[at + 2] = 30 + row as u8; // R, or B
            data[at + 3] = 255;
        }
    }
    data
}

#[test]
fn unpadding_drops_the_row_padding_and_keeps_the_pixels() {
    let (width, height, stride) = (3, 2, 256);
    let out = unpad(&padded(width, height, stride), width, height, stride, false);
    assert_eq!(out.len() as u32, width * height * BYTES_PER_PIXEL);
    // Second row, third pixel: row 1, column 2, unswizzled.
    let at = ((width + 2) * BYTES_PER_PIXEL) as usize;
    assert_eq!(&out[at..at + 4], &[11, 22, 31, 255]);
}

#[test]
fn a_bgra_surface_is_swizzled_to_rgba() {
    let (width, height, stride) = (3, 2, 256);
    let out = unpad(&padded(width, height, stride), width, height, stride, true);
    let at = ((width + 2) * BYTES_PER_PIXEL) as usize;
    // The same pixel, B and R exchanged and G and A left alone.
    assert_eq!(&out[at..at + 4], &[31, 22, 11, 255]);
}
