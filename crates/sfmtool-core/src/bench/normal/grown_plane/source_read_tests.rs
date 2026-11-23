// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The grown plane's value and Sobel reads from `read_source_samples`'
//! windows, through copied regions and through photograph tiles, equal its
//! whole-level reads.

use std::sync::Arc;

use super::render::{bilinear_from_window, sobel_from_window, sobel_gradient};
use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::remap::sample_bilinear_u8;
use crate::camera::sampler::{read_source_samples, RegionPool, ViewRegions};
use crate::camera::TiledPhotograph;
use crate::progress::Progress;

#[test]
fn sobel_and_value_reads_through_a_source_equal_the_whole_level() {
    let (width, height) = (150u32, 100u32);
    let data = (0..width * height * 3)
        .map(|i: u32| (i.wrapping_mul(2654435761) >> 13) as u8)
        .collect();
    let image = ImageU8::new(width, height, 3, data);
    let levels = ImageU8Pyramid::full_levels(width, height);
    let pyramid = Arc::new(ImageU8Pyramid::build(&image, levels));
    let source = TiledPhotograph::cut(&pyramid, 32);
    // Windows that straddle a tile corner (32, 64 at level 0; 32 at level
    // 1), the image's corners, and positions off the image.
    let positions = [
        [32.0f32, 32.0],
        [31.6, 64.4],
        [64.5, 31.5],
        [0.3, 0.2],
        [149.8, 99.9],
        [0.1, 99.6],
        [149.4, 0.4],
        [-2.5, 40.0],
        [160.0, 120.0],
        [73.3, 48.8],
    ];
    for level in [0, 1] {
        let img = pyramid.level(level);
        for pool in [
            RegionPool::with_capacity(u64::MAX),
            RegionPool::with_capacity(0),
        ] {
            let samples = read_source_samples(
                &mut ViewRegions::new(&source, &pool),
                level,
                &positions,
                1,
                &Progress::none(),
            )
            .unwrap();
            for (k, &[x, y]) in positions.iter().enumerate() {
                let window = samples.window(k);
                let [fx, fy] = samples.fractions(k);
                let pixel = |r: usize, q: usize, c: usize| window[(r * 4 + q) * 3 + c];
                let mut through = [(0.0, 0.0); 3];
                sobel_from_window(pixel, fx, fy, &mut through);
                let mut whole = [(0.0, 0.0); 3];
                sobel_gradient(img, x, y, &mut whole);
                let bits = |g: [(f32, f32); 3]| g.map(|(a, b)| (a.to_bits(), b.to_bits()));
                assert_eq!(bits(through), bits(whole), "level {level}, ({x}, {y})");
                for c in 0..3 {
                    // The value from the window's middle 2 x 2, as the
                    // grown plane blends it.
                    let value = bilinear_from_window(|r, q| pixel(r + 1, q + 1, c), fx, fy);
                    assert_eq!(
                        value.to_bits(),
                        sample_bilinear_u8(img, x, y, c as u32).to_bits()
                    );
                }
            }
            let stats = pool.stats();
            assert_eq!(stats.copied_reads + stats.tiled_reads, 1);
        }
    }
}
