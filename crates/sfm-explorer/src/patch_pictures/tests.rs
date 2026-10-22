// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests of the patch pictures both Track View and the Track Table draw.

use ndarray::Array3;

use super::stored_patch_image;

/// A patch bitmap of one, three or four channels becomes an opaque picture of
/// its colours, and an all-zero bitmap, which is how no patch is written,
/// becomes none.
#[test]
fn a_stored_patch_image_reads_any_channel_count() {
    let image = |channels: usize, fill: &[u8]| {
        let bitmap = Array3::from_shape_fn((2, 3, channels), |(_, _, c)| fill[c]);
        stored_patch_image(bitmap.view())
    };
    let grey = image(1, &[40]).expect("a grey patch");
    assert_eq!(grey.size, [3, 2]);
    assert_eq!(grey.pixels[0], egui::Color32::from_rgb(40, 40, 40));
    let rgb = image(3, &[10, 20, 30]).expect("an RGB patch");
    assert_eq!(rgb.pixels[5], egui::Color32::from_rgb(10, 20, 30));
    let rgba = image(4, &[10, 20, 30, 7]).expect("an RGBA patch");
    assert_eq!(
        rgba.pixels[0],
        egui::Color32::from_rgb(10, 20, 30),
        "the fourth channel is dropped for an opaque alpha"
    );
    assert!(image(4, &[0, 0, 0, 0]).is_none(), "an empty patch drew");
}
