// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! RGB → RGBA expansion tests.
//!
//! The size cases carry the point of the module: a thumbnail extent other than
//! the format's current 128×128 must produce a correctly-sized `ColorImage`
//! rather than trip the length assertion inside
//! `ColorImage::from_rgba_unmultiplied`. Both the square and the non-square
//! case are covered, because the panel that used to hard-code the size also
//! used its height as a width bound.

use ndarray::{Array3, Array4, Axis};

use super::{rgb_to_color_image, thumbnail_color_image};

#[test]
fn rgb_to_color_image_fills_opaque_alpha() {
    let rgb = [1_u8, 2, 3, 4, 5, 6];
    let image = rgb_to_color_image(&rgb, [2, 1]);

    assert_eq!(image.size, [2, 1]);
    assert_eq!(image.pixels.len(), 2);
    assert_eq!(image.pixels[0].to_array(), [1, 2, 3, 255]);
    assert_eq!(image.pixels[1].to_array(), [4, 5, 6, 255]);
}

#[test]
fn thumbnail_color_image_reads_the_formats_current_128() {
    let thumb = Array3::<u8>::zeros((128, 128, 3));
    let image = thumbnail_color_image(thumb.view());

    assert_eq!(image.size, [128, 128]);
    assert_eq!(image.pixels.len(), 128 * 128);
}

/// The regression: a thumbnail edge other than 128 must not panic. This is the
/// case the `image_browser` copy aborted on before the two were unified.
#[test]
fn thumbnail_color_image_reads_a_non_128_square_extent() {
    for edge in [1_usize, 64, 256] {
        let thumb = Array3::<u8>::zeros((edge, edge, 3));
        let image = thumbnail_color_image(thumb.view());

        assert_eq!(image.size, [edge, edge]);
        assert_eq!(image.pixels.len(), edge * edge);
    }
}

/// Width and height are read independently, so a non-square thumbnail keeps its
/// aspect rather than being squared off by whichever extent was named first.
#[test]
fn thumbnail_color_image_keeps_a_non_square_extent() {
    let thumb = Array3::<u8>::zeros((32, 96, 3));
    let image = thumbnail_color_image(thumb.view());

    assert_eq!(image.size, [96, 32]);
    assert_eq!(image.pixels.len(), 96 * 32);
}

/// The call sites reach a single image through `index_axis` on the stacked
/// `(image, y, x, rgb)` array; that view is contiguous, which is the branch
/// worth pinning since it is the one production takes.
#[test]
fn thumbnail_color_image_handles_an_index_axis_view() {
    let mut stack = Array4::<u8>::zeros((3, 4, 5, 3));
    stack[[1, 2, 3, 0]] = 200;
    stack[[1, 2, 3, 1]] = 201;
    stack[[1, 2, 3, 2]] = 202;

    let view = stack.index_axis(Axis(0), 1);
    assert!(view.as_slice().is_some(), "index_axis view is contiguous");
    let image = thumbnail_color_image(view);

    assert_eq!(image.size, [5, 4]);
    assert_eq!(image.pixels[2 * 5 + 3].to_array(), [200, 201, 202, 255]);
}

/// A non-contiguous view takes the element-wise copy branch and must agree with
/// the contiguous one on both extent and pixel order.
#[test]
fn thumbnail_color_image_copies_a_non_contiguous_view() {
    let mut thumb = Array3::<u8>::zeros((4, 4, 3));
    thumb[[1, 1, 0]] = 42;
    thumb[[1, 1, 1]] = 43;
    thumb[[1, 1, 2]] = 44;

    let cropped = thumb.slice(ndarray::s![1..3, 1..3, ..]);
    assert!(
        cropped.as_slice().is_none(),
        "sliced view is non-contiguous"
    );
    let image = thumbnail_color_image(cropped);

    assert_eq!(image.size, [2, 2]);
    assert_eq!(image.pixels[0].to_array(), [42, 43, 44, 255]);
}
