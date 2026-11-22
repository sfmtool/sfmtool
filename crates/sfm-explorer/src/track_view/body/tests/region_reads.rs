// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests that Track View draws the same pixels reading its
//! photographs through view-region pyramids as the whole pyramids give: the
//! tiles, the crops and their hover views of a track at the track stage and
//! at the cluster stage, as whole frames upload them, through the panel's
//! region pool and through photograph tiles.

use std::collections::HashMap;

use sfmtool_core::bench::{EditableTrack, Stage, StageKind};
use sfmtool_core::camera::image::{ImageU8, ImageU8Pyramid};
use sfmtool_core::camera::sampler::{render_tile, Sampler};
use sfmtool_core::camera::WarpMap;
use sfmtool_core::patch::cluster_refine::{sample_member_grid, ClusterRefineParams};

use super::super::patch::{render_patch_frame, PATCH_RES};
use super::super::table::{ColumnLayout, TILE_SIZE};
use super::super::tile::{color_image, tile_sampler};
use super::super::TrackBody;
use super::{input, on_the_bench, opaque, row_y};
use crate::patch_pictures::marks::CONTEXT_FACTOR;
use crate::patch_pictures::tests::whole_pyramid_pixels;
use crate::patch_pictures::PanelRegionPool;
use crate::scene::ReconId;
use crate::state::AppState;

/// The pictures one frame uploaded, by texture.
type Uploaded = HashMap<egui::TextureId, egui::ColorImage>;

/// One frame of the panel with `events`, adding every whole picture it
/// uploaded to `uploaded`.
fn frame(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &AppState,
    events: Vec<egui::Event>,
    uploaded: &mut Uploaded,
) {
    let mut output = ctx.run_ui(input(events), |ui| {
        panel.show(ui, state);
    });
    for (id, deltas) in &output.textures_delta.set {
        for delta in deltas {
            if let (None, egui::ImageData::Color(image)) = (delta.pos, &delta.image) {
                uploaded.insert(*id, (**image).clone());
            }
        }
    }
    output.textures_delta.clear();
}

/// Rest the pointer at `at` until the hover view under it has been drawn.
fn hover(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &AppState,
    at: egui::Pos2,
    uploaded: &mut Uploaded,
) {
    for _ in 0..14 {
        frame(
            panel,
            ctx,
            state,
            vec![egui::Event::PointerMoved(at)],
            uploaded,
        );
    }
}

/// The picture uploaded as `texture`.
fn picture(uploaded: &Uploaded, texture: egui::TextureId) -> &egui::ColorImage {
    uploaded
        .get(&texture)
        .unwrap_or_else(|| panic!("{texture:?} was not uploaded"))
}

/// How a reference picture is read.
#[derive(Clone, Copy, PartialEq)]
enum Reading {
    /// As the panel reads it.
    Whole,
    /// From the pyramid cut to its level 0, so every sample reads level 0:
    /// what a wrong level would draw.
    LevelZero,
    /// At the track stage with `Sampler::Bilinear`: what a wrong sampler
    /// would draw.
    Bilinear,
}

/// The tile and the tile's hover view of `observation` as the whole pyramid
/// gives them, read as `reading` says: `render_tile` of the patch frame
/// re-anchored on the observation, and of it widened [`CONTEXT_FACTOR`]
/// times, at the track stage; `sample_member_grid` at the cluster's radius,
/// and at that many times it, at the cluster stage.
fn whole_pyramid_tiles(
    state: &AppState,
    id: ReconId,
    track: &EditableTrack,
    observation: usize,
    reading: Reading,
) -> (egui::ColorImage, egui::ColorImage) {
    let recon = state.node(id).expect("loaded").recon();
    let row = &track.observations[observation];
    let image = row.image as usize;
    let mut src = state
        .cached_photograph(id, image)
        .expect("a cached photograph");
    if reading == Reading::LevelZero {
        let level = src.level(0);
        let copy = ImageU8::new(
            level.width(),
            level.height(),
            level.channels(),
            level.data().to_vec(),
        );
        src = std::sync::Arc::new(ImageU8Pyramid::from_image(copy, 1));
    }
    let site = crate::bench::observation_site(row).expect("a site");
    let k = CONTEXT_FACTOR;
    match &track.stage {
        Stage::Track(payload) => {
            let patch_frame = payload.patch_frame.as_ref().expect("a patch frame");
            let camera = crate::scene::image_intrinsics(&recon.image_set, image).expect("a camera");
            let pose = crate::scene::cam_from_world(&recon.image_set.images[image]);
            let sampler = match reading {
                Reading::Bilinear => Sampler::Bilinear,
                _ => tile_sampler(recon, track, observation),
            };
            let render = |rendered_frame: &sfmtool_core::patch::cloud::OrientedPatch, side: u32| {
                let mut map = WarpMap::from_patch(rendered_frame, &camera, &pose, side);
                opaque(&render_tile(&src, &mut map, sampler))
            };
            let anchored = render_patch_frame(patch_frame, &camera, &pose, Some(site.pixel));
            let mut wide = patch_frame
                .anchored_at_keypoint(&camera, &pose, site.pixel)
                .unwrap_or_else(|| patch_frame.clone());
            wide.half_extent = wide.half_extent.map(|h| h * f64::from(k));
            (render(&anchored, PATCH_RES), render(&wide, PATCH_RES * k))
        }
        Stage::Cluster(payload) => {
            let mut params = ClusterRefineParams {
                radius: payload.radius,
                ..ClusterRefineParams::default()
            };
            if let Some(template) = payload.template.as_ref() {
                params.resolution = template.samples.shape()[0] as u32;
            }
            params.resolution = params.resolution.max(2);
            let grid = |params: &ClusterRefineParams| {
                let shape = site.shape.expect("a shape");
                let samples =
                    sample_member_grid(&src, site.pixel, shape, params).expect("a member grid");
                let resolution = params.resolution as usize;
                color_image(
                    &samples,
                    resolution,
                    samples.len() / (resolution * resolution),
                )
            };
            let wide = ClusterRefineParams {
                radius: params.radius * f64::from(k),
                resolution: params.resolution * k,
                ..params.clone()
            };
            (grid(&params), grid(&wide))
        }
    }
}

/// Track View's tiles, crops and their hover views of every row of the track
/// on the bench, drawn by whole frames with the panel's own region pool and
/// with a pool that holds nothing, so every read takes the photograph's
/// tiles, equal the pictures the whole pyramid gives, and the pool holds
/// nothing once they are drawn.
fn drawn_pictures_equal_the_whole_pyramids(state: &AppState, id: ReconId, label: &str) {
    let track = state.bench_track(id, label).expect("on the bench").clone();
    let rows = track.observations.len();
    assert!(rows >= 3, "{rows} rows");
    // Where each row is, found on a panel of its own, since finding it draws
    // the tiles that the panels below must upload themselves.
    let ys: Vec<f32> = {
        let mut finder = TrackBody::new();
        let ctx = egui::Context::default();
        let images: Vec<usize> = track
            .observations
            .iter()
            .map(|o| o.image as usize)
            .collect();
        images
            .iter()
            .map(|&image| row_y(&mut finder, &ctx, state, image))
            .collect()
    };
    let layout = ColumnLayout::new();
    for panel_pool in [true, false] {
        let mut panel = TrackBody::new();
        if !panel_pool {
            panel.regions = PanelRegionPool::with_capacity(0);
        }
        let ctx = egui::Context::default();
        ctx.all_styles_mut(|style| {
            style.interaction.tooltip_delay = 0.0;
            style.interaction.tooltip_grace_time = 0.0;
        });
        let mut uploaded = Uploaded::new();
        frame(&mut panel, &ctx, state, Vec::new(), &mut uploaded);
        for (observation, &y) in ys.iter().enumerate() {
            let what = format!("row {observation}, panel pool {panel_pool}");
            let (tile, context) =
                whole_pyramid_tiles(state, id, &track, observation, Reading::Whole);
            let drawn = panel.tiles[&observation].as_ref().expect("a tile").id();
            assert!(*picture(&uploaded, drawn) == tile, "the tile of {what}");

            let at = egui::pos2(layout.tile_x() + TILE_SIZE / 2.0 + 8.0, y + 20.0);
            hover(&mut panel, &ctx, state, at, &mut uploaded);
            let drawn = panel.contexts[&observation]
                .as_ref()
                .expect("a hover view")
                .texture_id();
            assert!(
                *picture(&uploaded, drawn) == context,
                "the tile's hover view of {what}"
            );

            let src = state
                .cached_photograph(id, track.observations[observation].image as usize)
                .expect("a cached photograph");
            let recon = state.node(id).expect("loaded").recon();
            let outline = super::super::crop::outline(recon, &track, observation);
            let crop = crate::patch_pictures::crop::region(&outline.expect("an outline"));
            let drawn = panel.crops[&observation].as_ref().expect("a crop");
            let crop_picture = picture(&uploaded, drawn.texture_id());
            assert!(
                *crop_picture == whole_crop(&src, crop.expect("a crop"), crop_picture, false),
                "the crop of {what}"
            );

            let at = egui::pos2(layout.crop_x() + TILE_SIZE / 2.0 + 8.0, y + 20.0);
            hover(&mut panel, &ctx, state, at, &mut uploaded);
            let drawn = panel.crop_contexts[&observation]
                .as_ref()
                .expect("a crop hover view");
            let context_picture = picture(&uploaded, drawn.texture_id());
            assert!(
                *context_picture == whole_crop(&src, crop.expect("a crop"), context_picture, true),
                "the crop's hover view of {what}"
            );
        }
        let pool = panel.regions.pool();
        assert_eq!(pool.held_bytes(), 0, "a drawn picture kept its regions");
        let stats = pool.stats();
        // Four reads a row: the tile, its hover view, the crop and its hover
        // view.
        let reads = 4 * rows as u64;
        if panel_pool {
            assert_eq!((stats.copied_reads, stats.tiled_reads), (reads, 0));
        } else {
            assert_eq!((stats.copied_reads, stats.tiled_reads), (0, reads));
        }
    }
}

/// The crop, or with `in_context` its hover view, of `crop` as the whole
/// pyramid `src` gives it, at the texels of the picture `drawn`.
fn whole_crop(
    src: &ImageU8Pyramid,
    crop: crate::patch_pictures::crop::Region,
    drawn: &egui::ColorImage,
    in_context: bool,
) -> egui::ColorImage {
    whole_pyramid_pixels(src, crop, drawn.size, in_context)
}

/// At the track stage the tiles, crops and hover views drawn through regions
/// are the whole pyramid's.
#[test]
fn track_stage_pictures_drawn_through_regions_equal_the_whole_pyramid() {
    let (state, id, label, _, _) = on_the_bench();
    cache_hashed_photographs(&state, id);
    let track = state.bench_track(id, &label).expect("on the bench");
    assert!(matches!(track.stage, Stage::Track(_)));
    drawn_pictures_equal_the_whole_pyramids(&state, id, &label);
}

/// At the cluster stage the member grids, crops and hover views drawn through
/// regions are the whole pyramid's.
#[test]
fn cluster_stage_pictures_drawn_through_regions_equal_the_whole_pyramid() {
    let (mut state, id, label, _, _) = on_the_bench();
    cache_hashed_photographs(&state, id);
    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a patch frame downgrades");
    state.finish_background_tasks();
    let track = state.bench_track(id, &label).expect("on the bench");
    assert!(matches!(track.stage, Stage::Cluster(_)));
    drawn_pictures_equal_the_whole_pyramids(&state, id, &label);
}

/// A row whose photograph cannot be read draws the cells' error state: the
/// tile and crop cells say *unreadable* and their hovers say why, while the
/// rows whose photographs read draw their pictures.
#[test]
fn a_photograph_that_cannot_be_read_draws_the_cells_error_state() {
    use sfmtool_core::camera::image::PyramidLevel;
    let (state, id, label, _, _) = on_the_bench();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let image = track.observations[0].image as usize;
    // The cache's pyramid for the first row's photograph holds one pixel of
    // it, so any read past that pixel fails.
    let recon = state.node(id).expect("loaded").recon();
    let camera = &recon.image_set.cameras[0];
    let one_pixel = ImageU8Pyramid::from_regions(vec![Some(PyramidLevel {
        image: ImageU8::new(1, 1, 3, vec![0; 3]),
        origin: [0, 0],
        full_size: [camera.width, camera.height],
    })])
    .expect("a region pyramid");
    let path = crate::state::photograph_path(recon, image).expect("a path");
    state
        .photographs
        .insert(&path, std::sync::Arc::new(one_pixel));

    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let texts = super::painted(&mut panel, &ctx, &state, Vec::new());
    let why = panel
        .read_failure(0)
        .expect("the first row's read failed")
        .to_string();
    assert!(panel.tiles[&0].is_none() && panel.crops[&0].is_none());
    for observation in 1..track.observations.len() {
        assert!(panel.read_failure(observation).is_none());
        assert!(panel.tiles[&observation].is_some());
        assert!(panel.crops[&observation].is_some());
    }
    let unreadable = texts
        .iter()
        .filter(|t| *t == super::super::table::READ_FAILURE_CELL)
        .count();
    assert_eq!(unreadable, 2, "{texts:?}");

    let y = row_y(&mut panel, &ctx, &state, image);
    let layout = ColumnLayout::new();
    let said = super::super::table::read_failure_text(&why);
    for x in [layout.tile_x(), layout.crop_x()] {
        let at = egui::pos2(x + TILE_SIZE / 2.0 + 8.0, y + 20.0);
        let mut texts = Vec::new();
        for _ in 0..14 {
            texts = super::painted(
                &mut panel,
                &ctx,
                &state,
                vec![egui::Event::PointerMoved(at)],
            );
        }
        assert!(texts.contains(&said), "{said:?} not in {texts:?}");
    }
}

/// A photograph of hashed pixels in the cache for every image of `id`, in
/// place of the fixture's pattern, so that each channel, each pixel and each
/// photograph differs from its neighbours and a picture read from the wrong
/// place, level or channel differs from the right one.
fn cache_hashed_photographs(state: &AppState, id: ReconId) {
    let recon = state.node(id).expect("loaded").recon();
    let camera = &recon.image_set.cameras[0];
    let (w, h) = (camera.width, camera.height);
    for image in 0..recon.image_set.images.len() {
        let seed = (image as u32).wrapping_mul(0x9e37_79b9);
        let data = (0..w * h * 3)
            .map(|i: u32| (i.wrapping_add(seed).wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        state.insert_photograph(id, image, ImageU8::new(w, h, 3, data));
    }
}

/// The rows whose tile and tile hover view differ from the pictures level 0
/// alone gives: rows that read coarser levels, where a wrong level or sampler
/// would fail the comparison with the whole pyramid.
fn rows_reading_coarser_levels(state: &AppState, id: ReconId, label: &str) -> usize {
    let track = state.bench_track(id, label).expect("on the bench").clone();
    (0..track.observations.len())
        .filter(|&observation| {
            let read = whole_pyramid_tiles(state, id, &track, observation, Reading::Whole);
            let at_level_zero =
                whole_pyramid_tiles(state, id, &track, observation, Reading::LevelZero);
            read.0 != at_level_zero.0 && read.1 != at_level_zero.1
        })
        .count()
}

/// The track's patch made `factor` times as large, so each texel of a tile
/// spans many photograph pixels.
fn grow_patch(state: &mut AppState, id: ReconId, label: &str, factor: f64) {
    let track = state.bench_track(id, label).expect("on the bench").clone();
    let Stage::Track(payload) = &track.stage else {
        panic!("a track-stage track");
    };
    let half = payload
        .patch_frame
        .as_ref()
        .expect("a patch frame")
        .half_extent[0];
    state
        .edit_bench_patch(
            id,
            label,
            &crate::bench::PatchEdit::Resize {
                half_length: half * factor,
                moved_edge: None,
            },
        )
        .expect("a resize");
    state.finish_background_tasks();
}

/// A track-stage patch seen shrunk, its tiles read from coarser levels: the
/// pictures drawn through regions are the whole pyramid's.
#[test]
fn shrunk_track_stage_pictures_drawn_through_regions_equal_the_whole_pyramid() {
    let (mut state, id, label, _, _) = on_the_bench();
    cache_hashed_photographs(&state, id);
    grow_patch(&mut state, id, &label, 12.0);
    let rows = state
        .bench_track(id, &label)
        .expect("on the bench")
        .observations
        .len();
    assert_eq!(rows_reading_coarser_levels(&state, id, &label), rows);
    drawn_pictures_equal_the_whole_pyramids(&state, id, &label);
}

/// A track-stage patch turned towards edge-on and grown, so its views are
/// stretched and drawn with the anisotropic sampler: the pictures drawn
/// through regions are the whole pyramid's.
#[test]
fn stretched_track_stage_pictures_drawn_through_regions_equal_the_whole_pyramid() {
    let (mut state, id, label, _, _) = on_the_bench();
    cache_hashed_photographs(&state, id);
    grow_patch(&mut state, id, &label, 24.0);
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let Stage::Track(payload) = &track.stage else {
        panic!("a track-stage track");
    };
    let patch_frame = payload.patch_frame.as_ref().expect("a patch frame");
    let (sin, cos) = 75f64.to_radians().sin_cos();
    let normal = patch_frame.normal() * cos + patch_frame.u_axis * sin;
    state
        .edit_bench_patch(
            id,
            &label,
            &crate::bench::PatchEdit::Tilt {
                normal: normal.into(),
            },
        )
        .expect("a tilt");
    state.finish_background_tasks();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let recon = state.node(id).expect("loaded").recon();
    let anisotropic = (0..track.observations.len())
        .filter(|&o| tile_sampler(recon, &track, o) == Sampler::Anisotropic)
        .count();
    // Each such row's pictures differ from the bilinear ones, so a wrong
    // sampler fails the comparison with the whole pyramid.
    for observation in 0..track.observations.len() {
        if tile_sampler(recon, &track, observation) != Sampler::Anisotropic {
            continue;
        }
        let read = whole_pyramid_tiles(&state, id, &track, observation, Reading::Whole);
        let bilinear = whole_pyramid_tiles(&state, id, &track, observation, Reading::Bilinear);
        assert!(
            read.0 != bilinear.0 && read.1 != bilinear.1,
            "row {observation}"
        );
    }
    assert!(
        anisotropic > 0,
        "no row is drawn with the anisotropic sampler"
    );
    drawn_pictures_equal_the_whole_pyramids(&state, id, &label);
}

/// The track moved to the cluster stage and every sighting given the shape
/// `shape` makes of its own.
fn cluster_with_shapes(
    shape: impl Fn([[f64; 2]; 2]) -> [[f64; 2]; 2],
) -> (AppState, ReconId, String) {
    let (mut state, id, label, _, _) = on_the_bench();
    cache_hashed_photographs(&state, id);
    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a patch frame downgrades");
    state.finish_background_tasks();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    for (observation, row) in track.observations.iter().enumerate() {
        let own = crate::bench::observation_site(row)
            .and_then(|site| site.shape)
            .expect("a shape");
        state
            .edit_bench_patch(
                id,
                &label,
                &crate::bench::PatchEdit::Shape {
                    observation,
                    shape: shape(own),
                },
            )
            .expect("a shape");
    }
    state.finish_background_tasks();
    (state, id, label)
}

/// Cluster-stage sightings whose shapes are twelve times as large, so their
/// grids read coarser levels: the pictures drawn through regions are the
/// whole pyramid's.
#[test]
fn shrunk_cluster_stage_pictures_drawn_through_regions_equal_the_whole_pyramid() {
    let (state, id, label) = cluster_with_shapes(|s| s.map(|row| row.map(|v| 12.0 * v)));
    let rows = state
        .bench_track(id, &label)
        .expect("on the bench")
        .observations
        .len();
    assert_eq!(rows_reading_coarser_levels(&state, id, &label), rows);
    drawn_pictures_equal_the_whole_pyramids(&state, id, &label);
}

/// Cluster-stage sightings stretched 24 times along one axis and 6 times
/// along the other: the pictures drawn through regions are the whole
/// pyramid's.
#[test]
fn stretched_cluster_stage_pictures_drawn_through_regions_equal_the_whole_pyramid() {
    let (state, id, label) = cluster_with_shapes(|s| {
        [
            [24.0 * s[0][0], 6.0 * s[0][1]],
            [24.0 * s[1][0], 6.0 * s[1][1]],
        ]
    });
    assert!(rows_reading_coarser_levels(&state, id, &label) > 0);
    drawn_pictures_equal_the_whole_pyramids(&state, id, &label);
}

/// A row whose photograph's tiles are not there yet draws its tile and crop
/// cells empty, their loading state, with no failure, while the other rows
/// draw their pictures; its photograph's layout is asked for once however
/// many frames look, nothing is decoded on the GUI thread, and once the
/// photograph is there the next frame draws the row's pictures with nothing
/// more asked for.
#[test]
fn a_row_whose_tiles_are_not_there_draws_its_loading_state_and_asks_once() {
    let (mut state, id, label, _, _) = on_the_bench();
    let (requests, queue) = crate::state::PhotographRequests::queued();
    state.photograph_requests = requests;
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let image = track.observations[0].image as usize;
    let recon = state.node(id).expect("loaded").recon();
    let path = crate::state::photograph_path(recon, image).expect("a path");
    let pyramid = state.cached_photograph(id, image).expect("cached");
    state.photographs.forget(&path);
    let decodes = state.photographs.stats().decodes;

    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    let mut uploaded = Uploaded::new();
    for _ in 0..3 {
        frame(&mut panel, &ctx, &state, Vec::new(), &mut uploaded);
        assert!(!panel.tiles.contains_key(&0) && !panel.crops.contains_key(&0));
        assert!(panel.read_failure(0).is_none());
        for observation in 1..track.observations.len() {
            assert!(panel.tiles[&observation].is_some());
            assert!(panel.crops[&observation].is_some());
        }
    }
    assert_eq!(queue.len(), 1, "one header read, asked for once");
    assert_eq!(state.photographs.stats().decodes, decodes, "no decode here");

    // The photograph lands, as the decode the next frames would ask for
    // puts it in the cache.
    state.photographs.insert(&path, pyramid);
    frame(&mut panel, &ctx, &state, Vec::new(), &mut uploaded);
    assert!(panel.tiles[&0].is_some() && panel.crops[&0].is_some());
    assert_eq!(queue.len(), 1, "nothing more asked for");
    assert_eq!(state.photographs.stats().decodes, decodes);
}

/// A row whose photograph cannot be read, found once its header read has run
/// off the GUI thread, draws the cells' error state, its hover saying why.
#[test]
fn a_row_whose_photograph_file_cannot_be_read_draws_the_cells_error_state() {
    let (mut state, id, label, _, _) = on_the_bench();
    let (requests, queue) = crate::state::PhotographRequests::queued();
    state.photograph_requests = requests;
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let image = track.observations[0].image as usize;
    let recon = state.node(id).expect("loaded").recon();
    let path = crate::state::photograph_path(recon, image).expect("a path");
    assert!(
        !path.exists(),
        "the fixture's photographs are only in the cache"
    );
    state.photographs.forget(&path);

    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    let mut uploaded = Uploaded::new();
    frame(&mut panel, &ctx, &state, Vec::new(), &mut uploaded);
    assert!(panel.read_failure(0).is_none());
    assert_eq!(queue.run_off_this_thread(), 1, "the header read");
    let texts = super::painted(&mut panel, &ctx, &state, Vec::new());
    assert!(panel.tiles[&0].is_none() && panel.crops[&0].is_none());
    assert_eq!(
        panel.read_failure(0),
        Some(super::super::UNREADABLE_PHOTOGRAPH)
    );
    let unreadable = texts
        .iter()
        .filter(|t| *t == super::super::table::READ_FAILURE_CELL)
        .count();
    assert_eq!(unreadable, 2, "{texts:?}");
}

/// The hover views of a row whose tile and crop are drawn but whose
/// photograph the cache has since dropped say that the photograph is
/// loading, rather than that nothing could be rendered, ask for its layout
/// once, and are drawn once the photograph is back.
#[test]
fn a_hover_view_whose_tiles_are_not_there_says_it_is_loading() {
    let (mut state, id, label, _, _) = on_the_bench();
    let (requests, queue) = crate::state::PhotographRequests::queued();
    state.photograph_requests = requests;
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let image = track.observations[0].image as usize;
    let recon = state.node(id).expect("loaded").recon();
    let path = crate::state::photograph_path(recon, image).expect("a path");
    let pyramid = state.cached_photograph(id, image).expect("cached");

    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let y = row_y(&mut panel, &ctx, &state, image);
    assert!(panel.tiles[&0].is_some() && panel.crops[&0].is_some());
    state.photographs.forget(&path);
    let layout = ColumnLayout::new();
    let loading = super::super::table::HOVER_LOADING.to_string();
    for x in [layout.tile_x(), layout.crop_x()] {
        let at = egui::pos2(x + TILE_SIZE / 2.0 + 8.0, y + 20.0);
        let mut texts = Vec::new();
        for _ in 0..14 {
            texts = super::painted(
                &mut panel,
                &ctx,
                &state,
                vec![egui::Event::PointerMoved(at)],
            );
        }
        assert!(texts.contains(&loading), "{loading:?} not in {texts:?}");
    }
    assert!(!panel.contexts.contains_key(&0) && !panel.crop_contexts.contains_key(&0));
    assert!(panel.read_failure(0).is_none());
    assert_eq!(queue.len(), 1, "one header read, asked for once");

    state.photographs.insert(&path, pyramid);
    let mut uploaded = Uploaded::new();
    for x in [layout.tile_x(), layout.crop_x()] {
        let at = egui::pos2(x + TILE_SIZE / 2.0 + 8.0, y + 20.0);
        hover(&mut panel, &ctx, &state, at, &mut uploaded);
    }
    assert!(panel.contexts[&0].is_some() && panel.crop_contexts[&0].is_some());
    assert_eq!(queue.len(), 1, "nothing more asked for");
}
