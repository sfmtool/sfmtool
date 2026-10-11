// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The region read's pure half: the rectangle around a capsule, the decode of
//! padded rows, and which points a capsule selects from a decoded region.

use std::collections::HashSet;

use super::*;

const RECON: ReconId = ReconId::from_raw(0);
const OTHER: ReconId = ReconId::from_raw(1);

/// A region over `rect` in which every pixel shows its own point of `recon`,
/// numbered row-major from the region's top-left pixel.
fn every_pixel_a_point(rect: PixelRect, recon: ReconId) -> PickRegion {
    let points = (0..rect.area())
        .map(|i| Some(PointRef::new(recon, i)))
        .collect();
    PickRegion::new(rect, points)
}

/// The point numbers `points_in_capsule` should return, worked out pixel by
/// pixel from the definition rather than from the function's arithmetic.
fn expected(rect: PixelRect, from: [f64; 2], to: [f64; 2], radius: f64) -> HashSet<u32> {
    let mut out = HashSet::new();
    for row in 0..rect.height {
        for col in 0..rect.width {
            let c = [f64::from(rect.x + col) + 0.5, f64::from(rect.y + row) + 0.5];
            // Sample the segment densely and take the nearest sample.
            let nearest = (0..=1000)
                .map(|k| {
                    let t = f64::from(k) / 1000.0;
                    let p = [
                        from[0] + t * (to[0] - from[0]),
                        from[1] + t * (to[1] - from[1]),
                    ];
                    ((c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2)).sqrt()
                })
                .fold(f64::INFINITY, f64::min);
            if nearest <= radius {
                out.insert(row * rect.width + col);
            }
        }
    }
    out
}

#[test]
fn a_capsule_selects_the_pixels_within_its_radius_of_the_segment_and_no_others() {
    let rect = PixelRect {
        x: 10,
        y: 20,
        width: 40,
        height: 30,
    };
    let region = every_pixel_a_point(rect, RECON);
    // Off the pixel grid, diagonal, and long against the radius, so the ends,
    // the sides and the clamp of the projection all decide some pixel.
    let (from, to, radius) = ([17.3_f32, 26.8], [41.6_f32, 39.1], 4.5_f32);
    let found = points_in_capsule(&region, RECON, from, to, radius);
    let want = expected(
        rect,
        [f64::from(from[0]), f64::from(from[1])],
        [f64::from(to[0]), f64::from(to[1])],
        f64::from(radius),
    );
    assert!(!want.is_empty());
    assert_eq!(found, want);
}

#[test]
fn a_capsule_of_zero_length_is_the_circle() {
    let rect = PixelRect {
        x: 0,
        y: 0,
        width: 11,
        height: 11,
    };
    let region = every_pixel_a_point(rect, RECON);
    let found = points_in_capsule(&region, RECON, [5.5, 5.5], [5.5, 5.5], 2.0);
    // The centres at distance 0, 1, sqrt(2), 2 from the middle pixel's: the
    // middle, its four neighbours, the four diagonals and the four two steps
    // out along the axes.
    assert_eq!(found.len(), 13);
    assert!(found.contains(&(5 * 11 + 5)));
    assert!(found.contains(&(5 * 11 + 7)));
    assert!(!found.contains(&(7 * 11 + 7)), "sqrt(8) is outside 2");
}

#[test]
fn a_capsule_skips_the_pixels_of_another_reconstruction_and_empty_ones() {
    let rect = PixelRect {
        x: 0,
        y: 0,
        width: 3,
        height: 1,
    };
    let region = PickRegion::new(
        rect,
        vec![
            Some(PointRef::new(RECON, 7)),
            Some(PointRef::new(OTHER, 8)),
            None,
        ],
    );
    let found = points_in_capsule(&region, RECON, [0.5, 0.5], [2.5, 0.5], 1.0);
    assert_eq!(found, HashSet::from([7]));
    let found = points_in_capsule(&region, OTHER, [0.5, 0.5], [2.5, 0.5], 1.0);
    assert_eq!(found, HashSet::from([8]));
}

#[test]
fn a_point_shown_by_many_pixels_is_selected_once() {
    let rect = PixelRect {
        x: 0,
        y: 0,
        width: 4,
        height: 4,
    };
    let region = PickRegion::new(rect, vec![Some(PointRef::new(RECON, 3)); 16]);
    let found = points_in_capsule(&region, RECON, [2.0, 2.0], [2.0, 2.0], 1.0);
    assert_eq!(found, HashSet::from([3]));
}

#[test]
fn the_rectangle_around_a_capsule_holds_every_pixel_it_can_select() {
    let (from, to, radius) = ([30.2_f32, 12.7], [18.9_f32, 40.4], 6.3_f32);
    let rect = PixelRect::around_capsule(from, to, radius);
    // Read a region well past that rectangle, and check that everything the
    // capsule selects lies inside it.
    let big = PixelRect {
        x: 0,
        y: 0,
        width: 80,
        height: 80,
    };
    let region = every_pixel_a_point(big, RECON);
    let found = points_in_capsule(&region, RECON, from, to, radius);
    assert!(!found.is_empty());
    for index in &found {
        let (x, y) = (index % big.width, index / big.width);
        assert!(
            x >= rect.x && x < rect.x + rect.width && y >= rect.y && y < rect.y + rect.height,
            "pixel ({x}, {y}) is in the capsule but outside {rect:?}"
        );
    }
}

#[test]
fn the_rectangle_around_a_capsule_stops_at_the_top_and_left_edges() {
    let rect = PixelRect::around_capsule([1.0, 2.0], [1.0, 2.0], 5.0);
    assert_eq!((rect.x, rect.y), (0, 0));
    // Columns 0 ..= floor(1 + 5 - 0.5) = 5, rows 0 ..= 6.
    assert_eq!((rect.width, rect.height), (6, 7));
    // Wholly above or left of the texture: an empty rectangle.
    assert_eq!(
        PixelRect::around_capsule([-20.0, 5.0], [-20.0, 5.0], 4.0).area(),
        0
    );
}

#[test]
fn a_rectangle_is_cut_to_the_texture() {
    let rect = PixelRect {
        x: 90,
        y: 40,
        width: 30,
        height: 30,
    };
    assert_eq!(
        rect.clamped((100, 50)),
        Some(PixelRect {
            x: 90,
            y: 40,
            width: 10,
            height: 10,
        })
    );
    assert_eq!(rect.clamped((90, 50)), None);
}

#[test]
fn padded_rows_are_decoded_to_points_and_everything_else_to_none() {
    use super::super::picking::{PICK_TAG_FRUSTUM, PICK_TAG_NONE};
    let rect = PixelRect {
        x: 0,
        y: 0,
        width: 3,
        height: 2,
    };
    let row_bytes = padded_row_bytes(rect.width) as usize;
    assert_eq!(row_bytes, 256);
    let ids = [
        [PICK_TAG_POINT | 5, PICK_TAG_POINT | 5, PICK_TAG_NONE],
        [
            PICK_TAG_FRUSTUM | 5,
            PICK_TAG_POINT | 9,
            PICK_TAG_POINT | 400,
        ],
    ];
    // The padding after each row is filled with point ids, which the decode
    // must not read.
    let mut data = Vec::new();
    for row in ids {
        for id in row {
            data.extend_from_slice(&id.to_le_bytes());
        }
        while data.len() % row_bytes != 0 {
            data.extend_from_slice(&(PICK_TAG_POINT | 1).to_le_bytes());
        }
    }
    // Index 400 is outside every range: a stale id.
    let resolve = |id: u32| {
        let index = id & !PICK_TAG_MASK;
        (index < 100).then(|| PointRef::new(RECON, index as usize))
    };
    let points = decode_rows(&data, rect, row_bytes, resolve);
    assert_eq!(
        points,
        vec![
            Some(PointRef::new(RECON, 5)),
            Some(PointRef::new(RECON, 5)),
            None,
            None,
            Some(PointRef::new(RECON, 9)),
            None,
        ]
    );
    let region = PickRegion::new(rect, points);
    assert_eq!(region.rect(), rect);
    assert_eq!(region.point_at(1, 1), Some(PointRef::new(RECON, 9)));
    assert_eq!(region.point_at(3, 0), None, "outside the region");
}

/// The copy and the read on a headless device: a rectangle running off the
/// texture is cut to it, the copy passes wgpu's validation (row alignment,
/// buffer size, texture bounds), and the read decodes one entry per pixel of
/// the cut rectangle. The `noop` backend renders nothing, so every pixel reads
/// as background.
#[test]
fn a_region_is_copied_cut_to_the_texture_and_read_back() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions::enabled(),
            ..Default::default()
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("noop adapter");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("noop device");
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let (width, height) = (300, 200);
    let mut renderer = SceneRenderer::new();
    renderer.pick_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pick"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: super::super::gpu_types::GBUFFER_PICK_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }));
    renderer.current_size = (width, height);

    for (request, cut) in [
        // Inside the texture, and wider than one 256-byte row.
        (
            PixelRect {
                x: 10,
                y: 20,
                width: 97,
                height: 97,
            },
            PixelRect {
                x: 10,
                y: 20,
                width: 97,
                height: 97,
            },
        ),
        // Off the right and bottom edges.
        (
            PixelRect {
                x: 280,
                y: 150,
                width: 50,
                height: 80,
            },
            PixelRect {
                x: 280,
                y: 150,
                width: 20,
                height: 50,
            },
        ),
    ] {
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer.copy_pick_region(&device, &mut encoder, request);
        assert!(renderer.pick_region_pending());
        queue.submit([encoder.finish()]);
        let region = renderer.read_pick_region(&device).expect("a region");
        assert!(!renderer.pick_region_pending());
        assert_eq!(region.rect(), cut);
        assert_eq!(region.requested(), request);
        assert_eq!(region.points.len(), cut.area());
        assert!(region.points.iter().all(Option::is_none));
    }

    // Wholly outside: nothing is copied and nothing is waiting.
    let mut encoder = device.create_command_encoder(&Default::default());
    renderer.copy_pick_region(
        &device,
        &mut encoder,
        PixelRect {
            x: 400,
            y: 0,
            width: 10,
            height: 10,
        },
    );
    assert!(!renderer.pick_region_pending());
    assert!(renderer.read_pick_region(&device).is_none());

    let error = pollster::block_on(errors.pop());
    assert!(
        error.is_none(),
        "a region copy failed validation: {error:?}"
    );
}
