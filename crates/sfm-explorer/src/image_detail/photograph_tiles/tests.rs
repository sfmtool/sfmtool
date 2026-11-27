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

use sfmtool_core::camera::image::ImageU8;
use sfmtool_core::camera::{
    PhotographCache, PhotographCacheOptions, PhotographTile, PHOTOGRAPH_TILE_SIZE as TILE_SIZE,
};

use super::super::view::zoom_limit;
use super::super::{ImageDetail, Look, ViewGeometry};
use super::{level_for_scale, Layout, PhotographTileKey, MAX_UPLOADS_PER_FRAME};
use crate::scene::SceneNode;
use crate::state::{
    FeatureDisplaySettings, IntrinsicsDisplaySettings, PhotographRequests, PhotographSource,
};
use crate::test_support::HeldPhotograph;

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
    // The case `specs/gui/multi-panel-image-browser.md` § "Image Detail: 2D
    // pan and zoom navigation" gives: a 16320 px photograph in a 400 point
    // panel fits at 0.0245, and 32 times that is 0.78 points a pixel.
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
        tile_size: TILE_SIZE,
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
        tile_size: TILE_SIZE,
    };
    assert_eq!(
        long.overview_level(),
        4,
        "no level is one tile, so the coarsest"
    );
}

/// A tile the cache hands over is uploaded as its pixels, border and all, one
/// colour a pixel, whatever its channels. The cache's tests check that a tile
/// carries its neighbours' pixels as its border.
#[test]
fn a_tile_is_uploaded_as_its_pixels() {
    let rgb = super::texture_image(&ImageU8::new(3, 2, 3, (0..18).collect()));
    assert_eq!(rgb.size, [3, 2]);
    assert_eq!(rgb.pixels[0], egui::Color32::from_rgb(0, 1, 2));
    assert_eq!(rgb.pixels[4], egui::Color32::from_rgb(12, 13, 14));
    let grey = super::texture_image(&ImageU8::new(2, 1, 1, vec![5, 6]));
    assert_eq!(
        grey.pixels,
        vec![egui::Color32::from_gray(5), egui::Color32::from_gray(6)]
    );
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
    /// The level it drew and its tiles still to draw.
    tiles: Option<super::DrawnTiles>,
    /// Every text it painted.
    texts: Vec<String>,
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
    photograph: Option<PhotographSource<'_>>,
    panel: egui::Vec2,
    look: Option<Look>,
    gpu: &mut Gpu,
) -> Frame {
    frame_of(detail, ctx, node, 0, photograph, panel, look, gpu)
}

/// The same, showing image `index` of `node`.
#[allow(clippy::too_many_arguments)]
fn frame_of(
    detail: &mut ImageDetail,
    ctx: &egui::Context,
    node: &SceneNode,
    index: usize,
    photograph: Option<PhotographSource<'_>>,
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
    let (mut view, mut tiles) = (None, None);
    let mut output = ctx.run_ui(input, |ui| {
        let response = detail.show(
            ui,
            node.edited(),
            node.id,
            node.history.current_version().serial,
            Some(index),
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
        tiles = response.tiles;
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
    let texts = crate::test_support::output_texts(&output);
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
        tiles,
        texts,
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
    let pyramid = HeldPhotograph::new(&synthetic(2400, 1800));
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();
    let mut gpu = Gpu::default();
    // The rectangle fills a 500 point panel at 1.0 point a pixel, so level 0.
    frame(
        &mut detail,
        &ctx,
        &node,
        Some(pyramid.source()),
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
    let pyramid = HeldPhotograph::new(&synthetic(9000, 16));
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
            Some(pyramid.source()),
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
    let large = HeldPhotograph::new(&synthetic(9000, 16));
    let mut detail = ImageDetail::new();
    let mut gpu = Gpu::default();
    let view = frame(
        &mut detail,
        &ctx,
        &node,
        Some(large.source()),
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

    let small = HeldPhotograph::new(&synthetic(640, 480));
    let mut detail = ImageDetail::new();
    let view = frame(
        &mut detail,
        &ctx,
        &node,
        Some(small.source()),
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
        Some(large.source()),
        panel,
        zoom_in,
        &mut gpu,
    );
    let other = HeldPhotograph::new(&synthetic(640, 480));
    let view = frame(
        &mut detail,
        &ctx,
        &node,
        Some(other.source()),
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
    let pyramid = HeldPhotograph::new(&synthetic(2400, 1800));
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
    let dropped = HeldPhotograph::dropped();
    frame(
        &mut detail,
        &ctx,
        &node,
        Some(dropped.source()),
        panel,
        None,
        &mut gpu,
    );
    let first = frame(
        &mut detail,
        &ctx,
        &node,
        Some(pyramid.source()),
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
        Some(pyramid.source()),
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
        Some(pyramid.source()),
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
        Some(pyramid.source()),
        panel,
        None,
        &mut gpu,
    );
    assert!(refilling.repaint, "four tiles are left to upload");
    frame(
        &mut detail,
        &ctx,
        &node,
        Some(pyramid.source()),
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
        Some(dropped.source()),
        panel,
        None,
        &mut gpu,
    );
    assert_eq!(
        gpu.made.len(),
        made,
        "nothing is uploaded while the cache does not hold the photograph"
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
    let pyramid = HeldPhotograph::new(&image);
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
            Some(pyramid.source()),
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
    // Tiles of a photograph 300 tiles wide and two pixels high, so each costs
    // almost nothing to make.
    let pixels = |col: u32| PhotographTile {
        key: sfmtool_core::camera::TileKey {
            level: 0,
            col,
            row: 0,
        },
        origin: [col * TILE_SIZE, 0],
        size: [TILE_SIZE, 2],
        pixels: ImageU8::new(
            TILE_SIZE + 2,
            4,
            3,
            vec![7; (TILE_SIZE as usize + 2) * 4 * 3],
        ),
    };
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
        tiles.upload(&ctx, tile(col), &pixels(col));
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
        tiles.upload(&ctx, tile(col), &pixels(col));
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

/// A photograph cache of `budget_bytes`, as the viewer makes its own.
fn cache_of(budget_bytes: u64) -> Arc<PhotographCache> {
    Arc::new(PhotographCache::new(PhotographCacheOptions {
        budget_bytes,
        ..Default::default()
    }))
}

/// A JPEG of `width x height` written in `dir`.
fn jpeg_in(dir: &std::path::Path, name: &str, width: u32, height: u32) -> std::path::PathBuf {
    let path = dir.join(name);
    image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([x as u8, y as u8, (x ^ y) as u8])
    })
    .save(&path)
    .expect("the photograph is written");
    path
}

/// Two panels showing a photograph the cache does not hold, which needs ten
/// tiles of level 0 and the overview's, ask for its layout with one header
/// read and for its eleven tiles with one decode, the second panel starting
/// none while the first's runs, and neither decodes on the thread that draws
/// it: the decode count stays 0 until the queued tasks run on a thread of
/// their own. The frames before its first tiles land draw no image, since the
/// panel had none to keep. With a budget and with a cache that keeps nothing.
#[test]
fn panels_ask_for_a_photographs_tiles_in_one_decode_and_never_decode_themselves() {
    let dir = tempfile::tempdir().unwrap();
    let path = jpeg_in(dir.path(), "wide.jpg", 2560, 1024);
    let node = node();
    // One screen pixel a photograph pixel: level 0, five by two tiles, and
    // the overview at level 3.
    let panel = egui::vec2(2560.0, 1024.0);
    for budget in [1 << 30, 0] {
        let cache = cache_of(budget);
        let (requests, queue) = PhotographRequests::queued();
        let source = Some(PhotographSource {
            cache: &cache,
            requests: &requests,
            path: &path,
        });
        let (ctx_a, ctx_b) = (egui::Context::default(), egui::Context::default());
        let (mut a, mut b) = (ImageDetail::new(), ImageDetail::new());
        let (mut gpu_a, mut gpu_b) = (Gpu::default(), Gpu::default());

        let unknown = frame(&mut a, &ctx_a, &node, source, panel, None, &mut gpu_a);
        assert_eq!(unknown.view, None, "nothing to draw before the layout");
        frame(&mut b, &ctx_b, &node, source, panel, None, &mut gpu_b);
        assert_eq!(queue.len(), 1, "one header read for both panels");
        assert_eq!(queue.run_off_this_thread(), 1);

        let waiting = frame(&mut a, &ctx_a, &node, source, panel, None, &mut gpu_a);
        assert_eq!(waiting.view, None, "nothing drawn before the first tiles");
        frame(&mut b, &ctx_b, &node, source, panel, None, &mut gpu_b);
        frame(&mut a, &ctx_a, &node, source, panel, None, &mut gpu_a);
        assert_eq!(queue.len(), 1, "one decode for both panels");
        assert_eq!(cache.stats().decodes, 0, "no panel decodes as it draws");
        assert!(gpu_a.made.is_empty() && gpu_b.made.is_empty());
        assert_eq!(queue.run_off_this_thread(), 1);
        assert_eq!(cache.stats().decodes, 1);

        let drawn = frame(&mut a, &ctx_a, &node, source, panel, None, &mut gpu_a);
        assert_eq!(gpu_a.made.len(), 11, "budget {budget}");
        assert_eq!(
            drawn.tiles,
            Some(super::DrawnTiles {
                level: 0,
                tiles_pending: 0
            })
        );
        assert_eq!(drawn.meshes.len(), 10);
        if budget > 0 {
            // The cache holds the photograph, so the second panel's tiles are
            // cut from it with no decode.
            frame(&mut b, &ctx_b, &node, source, panel, None, &mut gpu_b);
            assert_eq!(gpu_b.made.len(), 11);
        }
        assert_eq!(queue.len(), 0, "budget {budget}");
        assert_eq!(cache.stats().decodes, 1);
    }
}

/// The tiles asked for ahead of an image switch are the ones the frame that
/// shows the next image draws, at the view the panel carries over to it,
/// zoomed in and panned: that frame starts no read, decodes nothing and
/// leaves no tile pending. Under a cache that keeps nothing, so only what the
/// decode asked for ahead answered is there to draw.
#[test]
fn the_tiles_asked_for_ahead_are_the_ones_the_next_image_draws() {
    let dir = tempfile::tempdir().unwrap();
    // Portrait where the shown image is landscape, so carrying the view over
    // rescales the pan on both axes.
    let next_path = jpeg_in(dir.path(), "next.jpg", 2000, 3000);
    let node = node();
    let panel = egui::vec2(500.0, 500.0);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();
    let mut gpu = Gpu::default();
    let shown = HeldPhotograph::new(&synthetic(2400, 1800));
    let view = frame(
        &mut detail,
        &ctx,
        &node,
        Some(shown.source()),
        panel,
        Some(Look::Pixel {
            pixel: [1700.0, 400.0],
            zoom: Some(4.0),
        }),
        &mut gpu,
    )
    .view
    .expect("a drawn frame publishes its view");

    let cache = cache_of(0);
    let (requests, queue) = PhotographRequests::queued();
    let next = PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &next_path,
    };
    let next_image = crate::scene::ImageRef::new(node.id, 1);
    detail.prefetch(&ctx, &view, next_image, next);
    assert_eq!(queue.run_off_this_thread(), 1, "the header read");
    detail.prefetch(&ctx, &view, next_image, next);
    detail.prefetch(&ctx, &view, next_image, next);
    assert_eq!(queue.len(), 1, "one decode");
    assert_eq!(cache.stats().decodes, 0);
    assert_eq!(queue.run_off_this_thread(), 1);

    let made = gpu.made.len();
    let switched = frame_of(
        &mut detail,
        &ctx,
        &node,
        1,
        Some(next),
        panel,
        None,
        &mut gpu,
    );
    // The view the tiles were asked for at is the one the panel carried over.
    let carried = view.carried_to(next_image, [2000.0, 3000.0]);
    let landed = switched.view.expect("the next image is drawn");
    assert_eq!(landed.zoom, carried.zoom);
    for axis in 0..2 {
        assert!(
            (landed.pan[axis] - carried.pan[axis]).abs() < 1e-3,
            "{landed:?} against {carried:?}"
        );
    }
    let tiles = switched.tiles.expect("the next image is drawn");
    assert_eq!(tiles.tiles_pending, 0, "{tiles:?}");
    assert!(gpu.made.len() > made + 1, "its tiles were uploaded");
    assert_eq!(queue.len(), 0, "nothing was asked for on the switch");
    assert_eq!(cache.stats().decodes, 1);
}

/// Stepping to an image whose tiles are not uploaded keeps drawing the image
/// shown before, its tiles and its view, until the tiles the new image
/// uploads first (its overview) have landed, and then draws the new image,
/// with the look asked for on the switch applied: no frame between draws an
/// empty image area or says it is loading. The decodes are stepped by hand,
/// so the frames that hold are the ones before each lands.
#[test]
fn a_switch_keeps_the_previous_image_until_the_new_ones_overview_lands() {
    let dir = tempfile::tempdir().unwrap();
    let next_path = jpeg_in(dir.path(), "next.jpg", 1200, 900);
    let node = node();
    let panel = egui::vec2(600.0, 450.0);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();
    let mut gpu = Gpu::default();
    let shown = HeldPhotograph::new(&synthetic(800, 600));
    let before = frame(
        &mut detail,
        &ctx,
        &node,
        Some(shown.source()),
        panel,
        None,
        &mut gpu,
    );
    let before_view = before.view.expect("the first image is drawn");
    let before_textures: HashSet<egui::TextureId> =
        before.meshes.iter().map(|(id, ..)| *id).collect();
    assert!(!before_textures.is_empty());

    let cache = cache_of(0);
    let (requests, queue) = PhotographRequests::queued();
    let next = Some(PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &next_path,
    });
    let look = Some(Look::Pixel {
        pixel: [300.0, 200.0],
        zoom: Some(2.0),
    });
    // The header read, then the decode, each held over until it has run.
    for (step, look) in [look, None].into_iter().enumerate() {
        let held = frame_of(&mut detail, &ctx, &node, 1, next, panel, look, &mut gpu);
        assert_eq!(
            held.view,
            Some(before_view),
            "step {step}: the previous view"
        );
        let textures: HashSet<egui::TextureId> = held.meshes.iter().map(|(id, ..)| *id).collect();
        assert_eq!(textures, before_textures, "step {step}: the previous tiles");
        assert!(
            !held.texts.iter().any(|t| t == super::super::DECODING_LABEL),
            "step {step}: {:?}",
            held.texts
        );
        assert_eq!(queue.run_off_this_thread(), 1, "step {step}");
    }

    let switched = frame_of(&mut detail, &ctx, &node, 1, next, panel, None, &mut gpu);
    let view = switched.view.expect("the new image is drawn");
    assert_eq!(view.image, crate::scene::ImageRef::new(node.id, 1));
    assert_eq!(view.image_size, [1200.0, 900.0]);
    assert_eq!(view.zoom, 2.0, "the look asked for on the switch");
    let rect = view.visible_rect();
    let centre = [(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0];
    assert!((centre[0] - 300.0).abs() < 1e-2 && (centre[1] - 200.0).abs() < 1e-2);
    assert!(
        !switched.meshes.is_empty()
            && switched
                .meshes
                .iter()
                .all(|(id, ..)| !before_textures.contains(id)),
        "the new image's own tiles"
    );
}

/// With no image shown before, a photograph whose first tiles have not landed
/// is "Loading image…", and is drawn once they have.
#[test]
fn the_first_image_says_it_is_loading_until_its_tiles_land() {
    let dir = tempfile::tempdir().unwrap();
    let path = jpeg_in(dir.path(), "first.jpg", 1200, 900);
    let node = node();
    let panel = egui::vec2(600.0, 450.0);
    let ctx = egui::Context::default();
    let mut detail = ImageDetail::new();
    let mut gpu = Gpu::default();
    let cache = cache_of(0);
    let (requests, queue) = PhotographRequests::queued();
    let source = Some(PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &path,
    });
    for step in 0..2 {
        let loading = frame(&mut detail, &ctx, &node, source, panel, None, &mut gpu);
        assert_eq!(loading.view, None, "step {step}");
        assert!(
            loading
                .texts
                .iter()
                .any(|t| t == super::super::DECODING_LABEL),
            "step {step}: {:?}",
            loading.texts
        );
        assert_eq!(queue.run_off_this_thread(), 1, "step {step}");
    }
    let drawn = frame(&mut detail, &ctx, &node, source, panel, None, &mut gpu);
    assert!(drawn.view.is_some() && !drawn.meshes.is_empty());
    assert!(!drawn
        .texts
        .iter()
        .any(|t| t == super::super::DECODING_LABEL));
}
