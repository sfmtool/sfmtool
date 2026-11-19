// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests of the photograph tiles Image Detail draws: the level for a
//! zoom, the zoom limit, the tiles a view asks for, the size of every texture,
//! the coarser fallback, and that a test-data photograph is drawn with the
//! pixels the single texture of the whole photograph drew.
//!
//! The frame tests run whole panel frames through `Context::run_ui` and read
//! what egui was asked to draw: the textures it was handed and the textured
//! meshes painted with them. No GPU is involved, so the comparison with the
//! old drawing samples both on the CPU, the way a GPU's linear filter with
//! clamp-to-edge addressing does.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use sfmtool_core::camera::image::{ImageU8, ImageU8Pyramid};

use super::super::view::zoom_limit;
use super::super::{DetailPhotograph, ImageDetail, Look, ViewGeometry};
use super::{level_for_scale, Layout, PhotographTileKey, MAX_UPLOADS_PER_FRAME, TILE_SIZE};
use crate::scene::SceneNode;
use crate::state::{FeatureDisplaySettings, IntrinsicsDisplaySettings};

#[test]
fn the_level_is_the_one_whose_pixels_are_about_one_screen_pixel() {
    // (physical screen pixels per level-0 pixel, levels, level drawn)
    let cases = [
        (2.0, 10, 0),
        (1.0, 10, 0),
        (0.71, 10, 0),
        (0.70, 10, 1),
        (0.5, 10, 1),
        (0.36, 10, 1),
        (0.35, 10, 2),
        (1.0 / 16.0, 10, 4),
        (1.0 / 1024.0, 5, 4),
        (0.0, 5, 4),
        (f32::NAN, 5, 0),
        (0.25, 1, 0),
    ];
    for (scale, levels, expected) in cases {
        assert_eq!(
            level_for_scale(scale, levels),
            expected,
            "at {scale} screen pixels per pixel with {levels} levels"
        );
    }
}

#[test]
fn the_zoom_limit_lets_one_pixel_reach_eight_points_and_keeps_32_otherwise() {
    // The draft's case: a 16320 px photograph in a 400 point panel fits at
    // 0.0245, and 32 times that is 0.78 points a pixel.
    let fit = 400.0 / 16320.0;
    assert!((zoom_limit(fit) * fit - 8.0).abs() < 1e-4);
    assert!(zoom_limit(fit) > 32.0);
    assert_eq!(zoom_limit(0.1), 80.0);
    // A quarter is where the two meet; anything that fits larger keeps 32.
    assert_eq!(zoom_limit(0.25), 32.0);
    assert_eq!(zoom_limit(0.47), 32.0);
    assert_eq!(zoom_limit(4.0), 32.0);
    assert_eq!(zoom_limit(0.0), 32.0);
    assert_eq!(zoom_limit(f32::NAN), 32.0);
}

#[test]
fn a_view_asks_for_the_tiles_whose_cores_it_meets() {
    let layout = Layout {
        sizes: vec![[2000, 1500], [1000, 750], [500, 375]],
    };
    let all: Vec<(u32, u32)> = (0..3)
        .flat_map(|row| (0..4).map(move |col| (col, row)))
        .collect();
    assert_eq!(layout.tiles_in(0, [0.0, 0.0, 2000.0, 1500.0]), all);
    assert_eq!(
        layout.tiles_in(0, [600.0, 100.0, 1100.0, 600.0]),
        vec![(1, 0), (2, 0), (1, 1), (2, 1)]
    );
    // A core that only touches the rectangle's edge is not asked for.
    assert_eq!(
        layout.tiles_in(0, [512.0, 0.0, 1024.0, 512.0]),
        vec![(1, 0)]
    );
    // Past the photograph, and a rectangle with no area, ask for nothing.
    assert!(layout.tiles_in(0, [2100.0, 0.0, 2500.0, 100.0]).is_empty());
    assert!(layout.tiles_in(0, [10.0, 10.0, 10.0, 50.0]).is_empty());
    // Level 1 halves the photograph, so the whole of it is 2 x 2 tiles; a
    // rectangle overhanging the photograph is cut to it.
    assert_eq!(
        layout.tiles_in(1, [-500.0, -500.0, 5000.0, 5000.0]),
        vec![(0, 0), (1, 0), (0, 1), (1, 1)]
    );
    assert_eq!(layout.core_rect(1, 1, 1), [1024.0, 1024.0, 2000.0, 1500.0]);
    assert_eq!(layout.core_rect(0, 3, 2), [1536.0, 1024.0, 2000.0, 1500.0]);
    assert_eq!(layout.overview_level(), 2);
    let long = Layout {
        sizes: vec![[9000, 16], [4500, 8], [2250, 4], [1125, 2], [562, 1]],
    };
    assert_eq!(
        long.overview_level(),
        4,
        "no level is one tile, so the coarsest"
    );
}

#[test]
fn a_tile_carries_its_neighbours_pixels_as_a_border() {
    let (w, h) = (600u32, 520u32);
    let data: Vec<u8> = (0..h)
        .flat_map(|y| (0..w).flat_map(move |x| [(x % 251) as u8, (y % 241) as u8, 7]))
        .collect();
    let level = ImageU8::new(w, h, 3, data);
    let at = |x: u32, y: u32| egui::Color32::from_rgb((x % 251) as u8, (y % 241) as u8, 7);

    let first = super::cut_tile(&level, 0, 0);
    assert_eq!(first.size, [514, 514]);
    // The core starts one texel in, and the border to the right and below is
    // the next tile's first column and row.
    assert_eq!(first.pixels[514 + 1], at(0, 0));
    assert_eq!(first.pixels[514 * 513 + 513], at(512, 512));
    // At the photograph's own edge the edge pixel is repeated.
    assert_eq!(first.pixels[0], at(0, 0));

    let corner = super::cut_tile(&level, 1, 1);
    assert_eq!(corner.size, [(600 - 512) + 2, (520 - 512) + 2]);
    let cw = corner.size[0];
    assert_eq!(corner.pixels[0], at(511, 511));
    assert_eq!(corner.pixels[cw + 1], at(512, 512));
    assert_eq!(corner.pixels[corner.pixels.len() - 1], at(599, 519));
}

/// What one frame of the panel handed egui.
struct Frame {
    /// Every textured mesh painted with a tile texture: the texture, the
    /// screen rectangle and the texture rectangle.
    meshes: Vec<(egui::TextureId, egui::Rect, egui::Rect)>,
    /// Whether the frame asked for another one at once.
    repaint: bool,
    /// The view the panel published.
    view: Option<ViewGeometry>,
}

/// The textures a run of frames has handed egui, by id, as egui would hold
/// them, and the size of every one made.
#[derive(Default)]
struct Gpu {
    textures: HashMap<egui::TextureId, Arc<egui::ColorImage>>,
    made: Vec<[usize; 2]>,
}

fn node() -> SceneNode {
    SceneNode::from_path(
        std::path::Path::new("/runs/tiles.sfmr"),
        sfmtool_core::SfmrReconstruction::demo(32),
    )
}

/// The photograph's whole pyramid, as the photograph cache builds it.
fn pyramid_of(image: ImageU8) -> ImageU8Pyramid {
    let levels = ImageU8Pyramid::full_levels(image.width(), image.height());
    ImageU8Pyramid::from_image(image, levels)
}

/// A photograph of `width x height` whose pixels say where they are.
fn synthetic(width: u32, height: u32) -> ImageU8 {
    let data = (0..height)
        .flat_map(|y| (0..width).flat_map(move |x| [x as u8, y as u8, (x ^ y) as u8]))
        .collect();
    ImageU8::new(width, height, 3, data)
}

/// Run one frame of the panel at `panel` points showing image 0 of `node`.
fn frame(
    detail: &mut ImageDetail,
    ctx: &egui::Context,
    node: &SceneNode,
    photograph: DetailPhotograph<'_>,
    panel: egui::Vec2,
    look: Option<Look>,
    gpu: &mut Gpu,
) -> Frame {
    let feature_display = FeatureDisplaySettings {
        overlay_mode: crate::state::OverlayMode::None,
        ..FeatureDisplaySettings::default()
    };
    let mut intrinsics_display = IntrinsicsDisplaySettings::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, panel)),
        ..Default::default()
    };
    let mut view = None;
    let mut output = ctx.run_ui(input, |ui| {
        let response = detail.show(
            ui,
            node.edited(),
            node.id,
            node.history.current_version().serial,
            Some(0),
            look,
            None,
            None,
            super::super::BenchMenu::default(),
            &[],
            &crate::platform::ScrollInput::default(),
            None,
            photograph,
            &feature_display,
            &mut intrinsics_display,
            None,
        );
        view = response.view;
    });
    #[allow(clippy::iter_over_hash_type)]
    for (id, deltas) in &output.textures_delta.set {
        for delta in deltas {
            let egui::ImageData::Color(image) = &delta.image;
            {
                if delta.pos.is_none() && id != &egui::TextureId::default() {
                    gpu.made.push(image.size);
                    gpu.textures.insert(*id, Arc::clone(image));
                }
            }
        }
    }
    output.textures_delta.clear();
    let mut meshes = Vec::new();
    for clipped in &output.shapes {
        collect_meshes(&clipped.shape, &mut meshes);
    }
    let repaint = output
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .is_some_and(|v| v.repaint_delay.is_zero());
    Frame {
        meshes,
        repaint,
        view,
    }
}

fn collect_meshes(shape: &egui::Shape, out: &mut Vec<(egui::TextureId, egui::Rect, egui::Rect)>) {
    match shape {
        egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect_meshes(s, out)),
        egui::Shape::Mesh(mesh) if mesh.texture_id != egui::TextureId::default() => {
            let mut rect = egui::Rect::NOTHING;
            let mut uv = egui::Rect::NOTHING;
            for vertex in &mesh.vertices {
                rect.extend_with(vertex.pos);
                uv.extend_with(vertex.uv);
            }
            out.push((mesh.texture_id, rect, uv));
        }
        _ => {}
    }
}

/// A tile by its level, column and row, of the one image these tests show.
type Tile = (usize, u32, u32);

fn tile_of(key: PhotographTileKey) -> Tile {
    (key.level, key.col, key.row)
}

/// The texture of `tile`, if it is uploaded.
fn texture_of(detail: &ImageDetail, tile: Tile) -> Option<egui::TextureId> {
    detail
        .tiles
        .resident()
        .into_iter()
        .find(|(key, _)| tile_of(*key) == tile)
        .map(|(_, id)| id)
}

fn resident_tiles(detail: &ImageDetail) -> HashSet<Tile> {
    detail
        .tiles
        .resident()
        .into_iter()
        .map(|(key, _)| tile_of(key))
        .collect()
}

#[test]
fn a_view_uploads_the_tiles_it_shows_and_the_overview() {
    let node = node();
    let pyramid = pyramid_of(synthetic(2400, 1800));
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();
    let mut gpu = Gpu::default();
    // The rectangle fills a 500 point panel at 1.0 point a pixel, so level 0.
    frame(
        &mut detail,
        &ctx,
        &node,
        DetailPhotograph::Decoded(&pyramid),
        egui::vec2(500.0, 500.0),
        Some(Look::Rect([600.0, 100.0, 1100.0, 600.0])),
        &mut gpu,
    );
    let expected: HashSet<Tile> = [(3, 0, 0), (0, 1, 0), (0, 2, 0), (0, 1, 1), (0, 2, 1)]
        .into_iter()
        .collect();
    assert_eq!(resident_tiles(&detail), expected);
}

#[test]
fn a_long_photograph_never_makes_a_texture_over_514_px() {
    let node = node();
    let pyramid = pyramid_of(synthetic(9000, 16));
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();
    let mut gpu = Gpu::default();
    let panel = egui::vec2(900.0, 700.0);
    // At the fit, then at the closest zoom at three places along it.
    let looks = [
        None,
        Some(Look::Pixel {
            pixel: [10.0, 8.0],
            zoom: Some(1e6),
        }),
        Some(Look::Pixel {
            pixel: [4500.0, 8.0],
            zoom: None,
        }),
        Some(Look::Pixel {
            pixel: [8990.0, 8.0],
            zoom: None,
        }),
    ];
    for look in looks {
        frame(
            &mut detail,
            &ctx,
            &node,
            DetailPhotograph::Decoded(&pyramid),
            panel,
            look,
            &mut gpu,
        );
    }
    assert!(
        resident_tiles(&detail)
            .iter()
            .any(|&(level, ..)| level == 0),
        "the closest zoom draws level 0"
    );
    assert!(gpu.made.len() > 4, "{:?}", gpu.made);
    for size in &gpu.made {
        assert!(size[0] <= 514 && size[1] <= 514, "a texture of {size:?}");
    }
}

#[test]
fn the_zoom_reaches_eight_points_a_pixel_on_a_large_photograph_and_32_on_a_small_one() {
    let node = node();
    let ctx = egui::Context::default();
    let panel = egui::vec2(900.0, 700.0);
    let zoom_in = Some(Look::Pixel {
        pixel: [4.0, 4.0],
        zoom: Some(1e6),
    });

    // 9000 px across 900 points fits at 0.1, so the limit is 80.
    let large = pyramid_of(synthetic(9000, 16));
    let mut detail = ImageDetail::new();
    let mut gpu = Gpu::default();
    let view = frame(
        &mut detail,
        &ctx,
        &node,
        DetailPhotograph::Decoded(&large),
        panel,
        zoom_in,
        &mut gpu,
    )
    .view
    .expect("a drawn frame publishes its view");
    assert!(
        (view.scale() - 8.0).abs() < 1e-3,
        "{} points a pixel at zoom {}",
        view.scale(),
        view.zoom
    );

    let small = pyramid_of(synthetic(640, 480));
    let mut detail = ImageDetail::new();
    let view = frame(
        &mut detail,
        &ctx,
        &node,
        DetailPhotograph::Decoded(&small),
        panel,
        zoom_in,
        &mut gpu,
    )
    .view
    .expect("a drawn frame publishes its view");
    assert_eq!(view.zoom, 32.0);

    // A zoom carried over from the large photograph comes back to the small
    // one's limit, about the panel centre.
    let mut detail = ImageDetail::new();
    frame(
        &mut detail,
        &ctx,
        &node,
        DetailPhotograph::Decoded(&large),
        panel,
        zoom_in,
        &mut gpu,
    );
    let other = ImageU8Pyramid::from_image(synthetic(640, 480), 1);
    let view = frame(
        &mut detail,
        &ctx,
        &node,
        DetailPhotograph::Decoded(&other),
        panel,
        None,
        &mut gpu,
    )
    .view
    .expect("a drawn frame publishes its view");
    assert_eq!(view.zoom, 32.0);
}

/// The level-0 pixels `[x0, y0, x1, y1]` that the texture rectangle `uv` of
/// `tile`, a texture of `size`, reads.
fn level0_of_uv((level, col, row): Tile, size: [usize; 2], uv: egui::Rect) -> [f32; 4] {
    let factor = (1u32 << level) as f32;
    let x = |u: f32| ((u * size[0] as f32) - 1.0 + (col * TILE_SIZE) as f32) * factor;
    let y = |v: f32| ((v * size[1] as f32) - 1.0 + (row * TILE_SIZE) as f32) * factor;
    [x(uv.min.x), y(uv.min.y), x(uv.max.x), y(uv.max.y)]
}

#[test]
fn a_missing_tile_is_drawn_from_the_coarser_level_until_it_is_uploaded() {
    let node = node();
    let pyramid = pyramid_of(synthetic(2400, 1800));
    let ctx = egui::Context::default();
    ctx.set_pixels_per_point(2.0);
    let mut detail = ImageDetail::new();
    let mut gpu = Gpu::default();
    // 1200 x 900 points at 2 pixels a point fits the photograph at one screen
    // pixel a pixel: level 0, all 5 x 4 of its tiles visible, more than one
    // frame uploads.
    let panel = egui::vec2(1200.0, 900.0);
    // egui takes a new scale on the frame after it is set, so one frame with
    // nothing to draw yet settles it.
    frame(
        &mut detail,
        &ctx,
        &node,
        DetailPhotograph::Decoding,
        panel,
        None,
        &mut gpu,
    );
    let first = frame(
        &mut detail,
        &ctx,
        &node,
        DetailPhotograph::Decoded(&pyramid),
        panel,
        None,
        &mut gpu,
    );
    assert_eq!(gpu.made.len(), MAX_UPLOADS_PER_FRAME);
    let overview = texture_of(&detail, (3, 0, 0)).expect("the overview tile first");
    // The last row of level 0 is not uploaded: each of its five cores is
    // drawn from the overview, over exactly the core, with the overview's
    // texels for that place.
    let view = first.view.expect("a drawn frame publishes its view");
    let scale = view.scale();
    let size = gpu.textures[&overview].size;
    let fallback: Vec<_> = first
        .meshes
        .iter()
        .filter(|(id, ..)| *id == overview)
        .collect();
    assert_eq!(fallback.len(), 5);
    for (col, (_, rect, uv)) in fallback.iter().enumerate() {
        assert!(texture_of(&detail, (0, col as u32, 3)).is_none());
        let core = [
            col as f32 * 512.0,
            1536.0,
            (col as f32 * 512.0 + 512.0).min(2400.0),
            1800.0,
        ];
        let drawn = [
            rect.min.x / scale,
            rect.min.y / scale,
            rect.max.x / scale,
            rect.max.y / scale,
        ];
        let read = level0_of_uv((3, 0, 0), size, *uv);
        for i in 0..4 {
            assert!(
                (drawn[i] - core[i]).abs() < 1e-2,
                "drawn over {drawn:?}, not {core:?}"
            );
            assert!(
                (read[i] - core[i]).abs() < 1e-2,
                "reads {read:?}, not {core:?}"
            );
        }
    }

    let second = frame(
        &mut detail,
        &ctx,
        &node,
        DetailPhotograph::Decoded(&pyramid),
        panel,
        None,
        &mut gpu,
    );
    assert_eq!(gpu.made.len(), MAX_UPLOADS_PER_FRAME + 5);
    assert!(!second.meshes.iter().any(|(id, ..)| *id == overview));
    assert_eq!(second.meshes.len(), 20);
    // egui runs one more pass after a pass that asked for a repaint, so the
    // frame after the last upload is the first that asks for none.
    let settled = frame(
        &mut detail,
        &ctx,
        &node,
        DetailPhotograph::Decoded(&pyramid),
        panel,
        None,
        &mut gpu,
    );
    assert!(!settled.repaint, "every visible tile is drawn");

    // Once settled, a frame with tiles still to upload asks for the next.
    for (key, _) in detail.tiles.resident() {
        if key.level == 0 {
            detail.tiles.evict(key);
        }
    }
    let refilling = frame(
        &mut detail,
        &ctx,
        &node,
        DetailPhotograph::Decoded(&pyramid),
        panel,
        None,
        &mut gpu,
    );
    assert!(refilling.repaint, "four tiles are left to upload");
    frame(
        &mut detail,
        &ctx,
        &node,
        DetailPhotograph::Decoded(&pyramid),
        panel,
        None,
        &mut gpu,
    );

    // A tile evicted while the cache no longer holds the pyramid is drawn from
    // the overview, and nothing is uploaded.
    let gone = detail
        .tiles
        .resident()
        .into_iter()
        .find(|(k, _)| tile_of(*k) == (0, 2, 1))
        .map(|(k, _)| k)
        .expect("tile (2, 1) of level 0 is uploaded");
    detail.tiles.evict(gone);
    let made = gpu.made.len();
    let third = frame(
        &mut detail,
        &ctx,
        &node,
        DetailPhotograph::Decoding,
        panel,
        None,
        &mut gpu,
    );
    assert_eq!(
        gpu.made.len(),
        made,
        "nothing is uploaded without a pyramid"
    );
    let from_overview: Vec<_> = third
        .meshes
        .iter()
        .filter(|(id, ..)| *id == overview)
        .collect();
    assert_eq!(from_overview.len(), 1);
    let read = level0_of_uv((3, 0, 0), size, from_overview[0].2);
    let core = [1024.0, 512.0, 1536.0, 1024.0];
    for i in 0..4 {
        assert!((read[i] - core[i]).abs() < 1e-2, "reads {read:?}");
    }
}

/// What a GPU's linear filter with clamp-to-edge addressing reads from
/// `image` at texture coordinate `uv`, per channel.
fn sample(image: &egui::ColorImage, uv: egui::Pos2) -> [f32; 3] {
    let [w, h] = image.size;
    let (tx, ty) = (uv.x * w as f32 - 0.5, uv.y * h as f32 - 0.5);
    let (x0, y0) = (tx.floor(), ty.floor());
    let (fx, fy) = (tx - x0, ty - y0);
    let at = |x: f32, y: f32| {
        let x = (x as i64).clamp(0, w as i64 - 1) as usize;
        let y = (y as i64).clamp(0, h as i64 - 1) as usize;
        image.pixels[y * w + x]
    };
    let mut out = [0.0; 3];
    for (c, value) in out.iter_mut().enumerate() {
        let channel = |p: egui::Color32| f32::from(p.to_array()[c]);
        let top = channel(at(x0, y0)) * (1.0 - fx) + channel(at(x0 + 1.0, y0)) * fx;
        let bottom = channel(at(x0, y0 + 1.0)) * (1.0 - fx) + channel(at(x0 + 1.0, y0 + 1.0)) * fx;
        *value = top * (1.0 - fy) + bottom * fy;
    }
    out
}

/// The largest difference, over every screen pixel of the photograph, between
/// the frame's tiles and one texture of the whole photograph drawn into the
/// same rectangle, as the panel drew it before it drew tiles.
fn largest_difference_from_one_texture(
    frame: &Frame,
    gpu: &Gpu,
    image: &ImageU8,
    panel: egui::Vec2,
) -> f32 {
    let view = frame.view.expect("a drawn frame publishes its view");
    let [dw, dh] = view.display_size();
    let centre = panel / 2.0 + egui::vec2(view.pan[0], view.pan[1]);
    let image_rect = egui::Rect::from_center_size(centre.to_pos2(), egui::vec2(dw, dh));
    let whole = crate::texture::rgb_to_color_image(
        image.data(),
        [image.width() as usize, image.height() as usize],
    );
    let mut largest: f32 = 0.0;
    let mut compared = 0;
    for y in 0..panel.y as usize {
        for x in 0..panel.x as usize {
            let p = egui::pos2(x as f32 + 0.5, y as f32 + 0.5);
            if !image_rect.contains(p) {
                continue;
            }
            let old = sample(
                &whole,
                egui::pos2(
                    (p.x - image_rect.min.x) / image_rect.width(),
                    (p.y - image_rect.min.y) / image_rect.height(),
                ),
            );
            let (id, rect, uv) = frame
                .meshes
                .iter()
                .find(|(_, rect, _)| {
                    p.x >= rect.min.x && p.x < rect.max.x && p.y >= rect.min.y && p.y < rect.max.y
                })
                .unwrap_or_else(|| panic!("no tile drawn at {p:?}"));
            let t = egui::pos2(
                uv.min.x + (p.x - rect.min.x) / rect.width() * uv.width(),
                uv.min.y + (p.y - rect.min.y) / rect.height() * uv.height(),
            );
            let new = sample(&gpu.textures[id], t);
            for c in 0..3 {
                largest = largest.max((new[c] - old[c]).abs());
            }
            compared += 1;
        }
    }
    assert!(compared > 10_000, "only {compared} pixels compared");
    largest
}

/// A test-data photograph, 360 x 640 so level 0 is two tiles with a seam at
/// row 512, is drawn with the pixels the single texture drew, at the fit and
/// zoomed in on the seam. Both views are at level 0; a view at a coarser level
/// is drawn from that level's averaged pixels by design, which the single
/// texture's unfiltered minification was not.
#[test]
fn a_test_data_photograph_looks_as_it_did_with_one_texture() {
    let path = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-data/images/seattle_backyard/seattle_backyard_01.jpg"
    ));
    let image = ImageU8::read_rgb(path).expect("the test-data photograph reads");
    assert_eq!((image.width(), image.height()), (360, 640));
    let pyramid = ImageU8Pyramid::build(
        &image,
        ImageU8Pyramid::full_levels(image.width(), image.height()),
    );
    let node = node();
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();
    let mut gpu = Gpu::default();
    let panel = egui::vec2(400.0, 700.0);
    let looks = [
        None,
        Some(Look::Pixel {
            pixel: [180.0, 512.0],
            zoom: Some(4.0),
        }),
    ];
    for look in looks {
        let drawn = frame(
            &mut detail,
            &ctx,
            &node,
            DetailPhotograph::Decoded(&pyramid),
            panel,
            look,
            &mut gpu,
        );
        assert!(
            drawn.meshes.len() == 2,
            "both tiles of level 0: {:?}",
            drawn.meshes
        );
        let largest = largest_difference_from_one_texture(&drawn, &gpu, &image, panel);
        assert!(
            largest < 0.05,
            "a pixel differs by {largest} of 255 at {look:?}"
        );
    }
}

/// The panel keeps at most 256 tile textures, and an upload past that drops
/// the one drawn longest ago.
#[test]
fn the_panel_keeps_256_tiles_and_drops_the_least_recently_drawn() {
    let ctx = egui::Context::default();
    // 300 tiles side by side, two pixels high, so each costs almost nothing
    // to cut.
    let pyramid = ImageU8Pyramid::from_image(synthetic(300 * TILE_SIZE, 2), 1);
    let image = crate::scene::ImageRef::new(node().id, 0);
    let mut tiles = super::PhotographTiles::new();
    let tile = |col: u32| PhotographTileKey {
        image,
        level: 0,
        col,
        row: 0,
    };
    for col in 0..super::MAX_TILE_TEXTURES as u32 {
        tiles.frame += 1;
        tiles.upload(&ctx, &pyramid, tile(col));
    }
    assert_eq!(tiles.resident.len(), super::MAX_TILE_TEXTURES);
    // Tile 0 is drawn again, so tile 1 is now the one drawn longest ago.
    tiles.frame += 1;
    let painter = egui::Painter::new(
        ctx.clone(),
        egui::LayerId::background(),
        egui::Rect::EVERYTHING,
    );
    tiles.paint(
        &painter,
        tile(0),
        [0.0, 0.0, 1.0, 1.0],
        egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(10.0, 10.0)),
        1.0,
    );
    for col in 256..260 {
        tiles.frame += 1;
        tiles.upload(&ctx, &pyramid, tile(col));
    }
    assert_eq!(tiles.resident.len(), super::MAX_TILE_TEXTURES);
    for col in [0, 5, 255, 256, 259] {
        assert!(tiles.resident.contains_key(&tile(col)), "tile {col} kept");
    }
    for col in 1..5 {
        assert!(
            !tiles.resident.contains_key(&tile(col)),
            "tile {col} dropped"
        );
    }
}
