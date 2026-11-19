// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A region pyramid of the required regions gives every sampler the whole
//! pyramid's values, and one a pixel smaller does not.

use super::*;
use crate::camera::image::{ImageU8, ImageU8Pyramid, PyramidLevel};
use crate::camera::sampler::render_tile;
use crate::camera::sampler::test_support::{affine_map, copy, maps, photograph, SAMPLERS};

fn whole(image: &ImageU8) -> ImageU8Pyramid {
    ImageU8Pyramid::build(
        image,
        ImageU8Pyramid::full_levels(image.width(), image.height()),
    )
}

fn layout(pyramid: &ImageU8Pyramid) -> PyramidLayout {
    let [w, h] = pyramid.level_full_size(0);
    PyramidLayout::in_memory(w, h, pyramid.num_levels(), 32)
}

/// A region pyramid of `regions`, copied out of the whole `pyramid`.
fn held(pyramid: &ImageU8Pyramid, regions: &LevelRegions) -> ImageU8Pyramid {
    let levels = (0..regions.levels())
        .map(|level| {
            regions.region(level).map(|r| {
                let img = pyramid.level(level);
                let mut data = Vec::new();
                for y in r.y0..r.y1 {
                    for x in r.x0..r.x1 {
                        for c in 0..img.channels() {
                            data.push(img.get_pixel(x, y, c));
                        }
                    }
                }
                PyramidLevel {
                    image: ImageU8::new(r.width(), r.height(), img.channels(), data),
                    origin: [r.x0, r.y0],
                    full_size: [img.width(), img.height()],
                }
            })
        })
        .collect();
    ImageU8Pyramid::from_regions(levels).expect("regions of the whole pyramid")
}

#[test]
fn a_region_pyramid_of_the_required_regions_equals_the_whole_pyramid() {
    let image = photograph(200, 150);
    let pyramid = whole(&image);
    let layout = layout(&pyramid);
    for map in maps(200, 150) {
        for sampler in SAMPLERS {
            let mut map = copy(&map);
            let regions = required_regions(&mut map, sampler, &layout);
            if regions.is_empty() {
                continue;
            }
            let held = held(&pyramid, &regions);
            assert_eq!(
                render_tile(&held, &mut map, sampler).data(),
                render_tile(&pyramid, &mut map, sampler).data(),
                "{sampler:?}"
            );
        }
    }
}

#[test]
fn a_region_one_pixel_smaller_on_an_inner_edge_changes_the_render() {
    let image = photograph(200, 150);
    let pyramid = whole(&image);
    let layout = layout(&pyramid);
    // A map well inside the image, so every region edge is an inner edge.
    let mut map = affine_map(24, [1.3, 0.2, 60.0, -0.1, 1.2, 40.0], 0);
    for sampler in SAMPLERS {
        let regions = required_regions(&mut map, sampler, &layout);
        let whole = render_tile(&pyramid, &mut map, sampler);
        assert_eq!(
            render_tile(&held(&pyramid, &regions), &mut map, sampler).data(),
            whole.data()
        );
        for side in 0..4 {
            let mut smaller = regions.clone();
            let level = (0..smaller.levels())
                .find(|&l| smaller.region(l).is_some())
                .unwrap();
            let mut rect = smaller.region(level).unwrap();
            match side {
                0 => rect.x0 += 1,
                1 => rect.y0 += 1,
                2 => rect.x1 -= 1,
                _ => rect.y1 -= 1,
            }
            smaller.set_region(level, rect);
            let held = held(&pyramid, &smaller);
            assert_ne!(
                render_tile(&held, &mut map, sampler).data(),
                whole.data(),
                "{sampler:?}, side {side}"
            );
        }
    }
}
