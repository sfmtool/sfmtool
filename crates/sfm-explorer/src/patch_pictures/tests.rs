// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests of the patch pictures both Track View and the Track Table draw, and
//! the whole-pyramid crop the region reads are held to.

use std::sync::Arc;

use ndarray::Array3;
use sfmtool_core::camera::image::{ImageU8, ImageU8Pyramid};
use sfmtool_core::camera::remap::sample_bilinear_u8;
use sfmtool_core::camera::sampler::{RegionPool, REGION_POOL_BYTES_PER_THREAD};
use sfmtool_core::camera::TiledPhotograph;

use super::crop::{cluster_outline, context, image, region, Region};
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

/// The pixels of the crop `crop`, or with `in_context` of its hover view, as
/// the whole pyramid gives them at `texels`: the picture's part of the
/// photograph sampled at its texels, each texel inside the photograph `sample_bilinear_u8` of the level
/// whose pixel is the largest not wider than the step, at the texel's centre.
/// The crop's read before it read view-region pyramids, which the tests hold
/// the region reads to.
pub(crate) fn whole_pyramid_pixels(
    src: &ImageU8Pyramid,
    crop: Region,
    texels: [usize; 2],
    in_context: bool,
) -> egui::ColorImage {
    let region = if in_context {
        Region {
            min: [crop.min[0] - crop.size[0], crop.min[1] - crop.size[1]],
            size: crop.size.map(|side| side * 3),
        }
    } else {
        crop
    };
    let step = [
        region.size[0] as f64 / texels[0] as f64,
        region.size[1] as f64 / texels[1] as f64,
    ];
    let level = (step[0].min(step[1]).log2().floor().max(0.0) as usize)
        .min(src.num_levels().saturating_sub(1));
    let scale = (1u64 << level) as f64;
    let img = src.level(level);
    let full = src.level(0);
    let (width, height) = (f64::from(full.width()), f64::from(full.height()));
    let channels = img.channels().max(1);
    let mut rgba = Vec::with_capacity(texels[0] * texels[1] * 4);
    for j in 0..texels[1] {
        let y = region.min[1] as f64 + (j as f64 + 0.5) * step[1];
        for i in 0..texels[0] {
            let x = region.min[0] as f64 + (i as f64 + 0.5) * step[0];
            if x < 0.0 || y < 0.0 || x >= width || y >= height {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            let (lx, ly) = ((x / scale) as f32, (y / scale) as f32);
            let value = |c: u32| {
                (sample_bilinear_u8(img, lx, ly, c.min(channels - 1)) + 0.5).clamp(0.0, 255.0) as u8
            };
            rgba.extend_from_slice(&[value(0), value(1), value(2), 255]);
        }
    }
    egui::ColorImage::from_rgba_unmultiplied(texels, &rgba)
}

/// A photograph of 1100 x 1050 px, three by three photograph tiles of 512 px
/// at level 0, with detail at every level: its pixels hash their position.
fn tiled_crop_photograph() -> Arc<ImageU8Pyramid> {
    let (w, h) = (1100u32, 1050u32);
    let data = (0..w * h * 3)
        .map(|i: u32| (i.wrapping_mul(2654435761) >> 13) as u8)
        .collect();
    let image = ImageU8::new(w, h, 3, data);
    Arc::new(ImageU8Pyramid::build(
        &image,
        ImageU8Pyramid::full_levels(w, h),
    ))
}

/// A crop and its hover view read through view-region pyramids equal the
/// crop and hover view the whole pyramid gives, byte for byte, through a
/// region copied from a panel's pool and through photograph tiles: at 1, 3
/// and 40 photograph pixels per texel (levels 0, 1 and 5), over outlines
/// across the tile edges at 512 px, at and off each of the image's corners,
/// and for a crop that covers the whole photograph and more.
#[test]
fn crops_read_through_regions_equal_the_whole_pyramid_crops() {
    let pyramid = tiled_crop_photograph();
    let source = TiledPhotograph::from_pyramid(pyramid.clone());
    let identity = [[1.0, 0.0], [0.0, 1.0]];
    let sheared = [[1.0, 0.35], [-0.2, 0.9]];
    let keypoints = [
        [512.3, 511.6],
        [1024.0, 512.0],
        [2.5, 3.5],
        [1098.7, 1048.2],
        [-6.0, 520.0],
        [550.0, 1056.0],
    ];
    // Half-widths whose crops are about 42, 384 and 5120 px: one texel per
    // pixel, about 3 pixels per texel and about 40, read from levels 0, 1
    // and 5.
    let steps = [(20.0, 1.0), (191.0, 3.0), (2559.0, 40.0)];
    let mut compared = 0;
    for (radius, step) in steps {
        for keypoint in keypoints {
            for shape in [identity, sheared] {
                let outline = cluster_outline(keypoint, shape, radius);
                for panel_pool in [true, false] {
                    let pool = if panel_pool {
                        RegionPool::with_capacity(REGION_POOL_BYTES_PER_THREAD)
                    } else {
                        RegionPool::with_capacity(0)
                    };
                    let crop = image(&outline, &source, &pool)
                        .expect("a photograph in memory reads")
                        .expect("a crop");
                    let context = context(&outline, &source, &pool)
                        .expect("a photograph in memory reads")
                        .expect("a hover view");
                    if shape == identity {
                        let side = crop.crop.size[0] as f64;
                        let texel = side / crop.image.size[0] as f64;
                        assert!(
                            (texel - step).abs() < 0.05 * step,
                            "radius {radius}: a crop of {side} px at {texel} px a texel"
                        );
                    }
                    assert_eq!(Some(crop.crop), region(&outline));
                    assert_eq!(context.crop, crop.crop);
                    assert!(
                        crop.image
                            == whole_pyramid_pixels(&pyramid, crop.crop, crop.image.size, false),
                        "the crop at {keypoint:?}, radius {radius}, shape {shape:?}, \
                         panel pool {panel_pool}"
                    );
                    assert!(
                        context.image
                            == whole_pyramid_pixels(
                                &pyramid,
                                context.crop,
                                context.image.size,
                                true
                            ),
                        "the hover view at {keypoint:?}, radius {radius}, shape {shape:?}, \
                         panel pool {panel_pool}"
                    );
                    let stats = pool.stats();
                    if panel_pool {
                        assert_eq!(stats.tiled_reads, 0, "{stats:?}");
                    } else {
                        assert_eq!((stats.copied_reads, stats.tiled_reads), (0, 2));
                    }
                    assert_eq!(pool.held_bytes(), 0, "a cut picture kept its regions");
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(compared, 3 * 6 * 2 * 2);
}
