// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for Track View's edit mode.
//!
//! egui needs no GPU to lay out a frame, so the whole body runs through
//! `Context::run_ui` here: `show` really does draw the header, the toolbar,
//! the sliders and every row of the table. What the assertions target
//! is what the panel *decides* -- which rows it drew, what each says, what the
//! sliders paint, and what it reports back to the dock -- rather than pixels.

use sfmtool_core::bench::{StageKind, Thresholds, Verdict};
use sfmtool_core::camera::remap::{ImageU8, ImageU8Pyramid};

use super::{TrackEdit, TrackEditResponse};
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::edits::tests::projected_embedded_demo;
use crate::state::AppState;

/// The point the fixtures put on the bench: three observations, in images 0, 1
/// and 2, at their exact projections.
const POINT: u32 = 2;

const VIEWPORT: egui::Vec2 = egui::vec2(1400.0, 900.0);

/// A state holding one `embedded_patches` node with a photograph cached for
/// every image, as the bench's own tests build it.
fn state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(projected_embedded_demo(12)));
    let id = state.selected_recon.expect("a selected reconstruction");
    cache_photographs(&mut state, id);
    (state, id)
}

/// A textured photograph in the node's cache for every one of its images, which
/// is what the tiles are warped out of.
///
/// A pattern rather than a flat field, with short periods that differ between
/// the axes, so two tiles cut at two places are two different pictures.
fn cache_photographs(state: &mut AppState, id: ReconId) {
    let node = crate::scene::node_by_id(&state.scene, id).expect("loaded");
    let camera = &node.recon().image_table.cameras[0];
    let (w, h) = (camera.width, camera.height);
    let images = node.image_count();
    for image in 0..images {
        let data: Vec<u8> = (0..(w * h * 3))
            .map(|i| {
                let p = i / 3;
                ((p % w) % 9 * 14 + (p / w) % 7 * 18) as u8
            })
            .collect();
        state.full_res_cache.insert(
            ImageRef::new(id, image),
            Some(std::sync::Arc::new(ImageU8Pyramid::from_image(
                ImageU8::new(w, h, 3, data),
                crate::state::PYRAMID_LEVELS,
            ))),
        );
    }
}

/// Drive one frame of the panel and hand back what it reported.
fn run_frame(panel: &mut TrackEdit, ctx: &egui::Context, state: &AppState) -> TrackEditResponse {
    run_frame_with(panel, ctx, state, Vec::new())
}

/// The same frame, with `events` delivered to egui.
fn run_frame_with(
    panel: &mut TrackEdit,
    ctx: &egui::Context,
    state: &AppState,
    events: Vec<egui::Event>,
) -> TrackEditResponse {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        events,
        ..Default::default()
    };
    let mut response = None;
    crate::test_support::run_frame_headless(ctx, input, |ui| {
        response = Some(panel.show(ui, state));
    });
    response.expect("the panel ran")
}

/// Park the pointer at `pos` for two frames, clicking there on the second when
/// `click`, and hand back the last frame's response. Two frames are required:
/// egui resolves hover and clicks against the rects the previous pass
/// registered, so a single frame reports no interaction.
fn at_pointer(
    panel: &mut TrackEdit,
    ctx: &egui::Context,
    state: &AppState,
    pos: egui::Pos2,
    click: bool,
) -> TrackEditResponse {
    let mut response = None;
    for frame in 0..2 {
        let mut events = vec![egui::Event::PointerMoved(pos)];
        if click && frame == 1 {
            for pressed in [true, false] {
                events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::default(),
                });
            }
        }
        response = Some(run_frame_with(panel, ctx, state, events));
    }
    response.expect("two frames ran")
}

/// The y at which the row for `image` answers the pointer, found by walking
/// down the panel: what sits above the table is the header, the toolbar and
/// the sliders, and a hard-coded offset would go stale the moment
/// one of them gains a line.
fn row_y(panel: &mut TrackEdit, ctx: &egui::Context, state: &AppState, image: usize) -> f32 {
    for step in 0..(VIEWPORT.y as usize / 8) {
        let y = step as f32 * 8.0;
        let response = at_pointer(panel, ctx, state, egui::pos2(400.0, y), false);
        if response.hovered_image == Some(image) {
            return y;
        }
    }
    panic!("no row of the table answered for image {image}");
}

/// The raw input one headless frame is driven with.
fn input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        events,
        ..Default::default()
    }
}

/// The strings one frame of the panel painted, with `events` delivered.
fn painted(
    panel: &mut TrackEdit,
    ctx: &egui::Context,
    state: &AppState,
    events: Vec<egui::Event>,
) -> Vec<String> {
    crate::test_support::painted_texts(ctx, input(events), |ui| {
        panel.show(ui, state);
    })
}

/// A panel and a context that have laid the table out once, so the pointer has
/// rects to resolve against.
fn settled(state: &AppState) -> (TrackEdit, egui::Context) {
    let mut panel = TrackEdit::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, state);
    (panel, ctx)
}

/// Right-click at `at` and leave the menu it opened laid out.
///
/// Three frames: one to register the rows, one that right-clicks, and one
/// more, because the menu's entries are laid out on a later frame.
fn open_row_menu(panel: &mut TrackEdit, ctx: &egui::Context, state: &AppState, at: egui::Pos2) {
    for frame in 0..3 {
        let mut events = vec![egui::Event::PointerMoved(at)];
        if frame == 1 {
            for pressed in [true, false] {
                events.push(egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Secondary,
                    pressed,
                    modifiers: egui::Modifiers::default(),
                });
            }
        }
        painted(panel, ctx, state, events);
    }
}

/// The strings the first table row's context menu painted.
fn row_menu(panel: &mut TrackEdit, ctx: &egui::Context, state: &AppState) -> Vec<String> {
    let y = row_y(panel, ctx, state, 0);
    let at = egui::pos2(400.0, y);
    open_row_menu(panel, ctx, state, at);
    painted(panel, ctx, state, vec![egui::Event::PointerMoved(at)])
}

/// Where an open menu drew the entry called `text`.
fn menu_entry_pos(
    panel: &mut TrackEdit,
    ctx: &egui::Context,
    state: &AppState,
    text: &str,
) -> egui::Pos2 {
    crate::test_support::painted_text_rects(ctx, input(Vec::new()), |ui| {
        panel.show(ui, state);
    })
    .into_iter()
    .find(|painted| painted.text == text)
    .unwrap_or_else(|| panic!("{text:?} was not painted"))
    .rect
    .center()
}

/// Put [`POINT`] on the bench and draw one frame over it.
fn on_the_bench() -> (AppState, ReconId, String, TrackEdit, egui::Context) {
    let (mut state, id) = state();
    let label = state
        .put_point_on_bench(PointRef::new(id, POINT as usize))
        .expect("a live point");
    let mut panel = TrackEdit::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, &state);
    (state, id, label, panel, ctx)
}

/// With nothing active the body draws nothing: Track View is in view mode
/// then, and the ways in are its empty state's.
#[test]
fn with_nothing_active_the_body_draws_no_rows_and_no_text() {
    let (state, _) = state();
    let mut panel = TrackEdit::new();
    let ctx = egui::Context::default();
    let response = run_frame(&mut panel, &ctx, &state);

    assert!(panel.rows().is_empty());
    assert_eq!(response, TrackEditResponse::default());
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(texts.is_empty(), "{texts:?}");
}

#[test]
fn the_table_has_a_row_per_observation_in_index_order() {
    let (_state, _id, _label, panel, _ctx) = on_the_bench();
    let rows = panel.rows();
    assert_eq!(rows.len(), 3, "the fixture's track has three observations");
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row.observation, index, "rows are not in index order");
        assert_eq!(row.verdict, Verdict::In, "a track from a point is all in");
        assert!(!row.pinned, "a track from a point pins nothing");
    }
    assert_eq!(
        rows.iter().map(|row| row.image).collect::<Vec<_>>(),
        [0, 1, 2]
    );
}

#[test]
fn a_verdict_shows_under_the_same_observation_index() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    run_frame(&mut panel, &ctx, &state);

    let rows = panel.rows();
    assert_eq!(rows.len(), 3, "the refused observation left the table");
    assert_eq!(rows[1].observation, 1);
    assert_eq!(rows[1].verdict, Verdict::Out);
    assert!(rows[1].pinned, "a verdict set by hand is not pinned");
    assert_eq!(rows[0].verdict, Verdict::In);
    assert_eq!(rows[2].verdict, Verdict::In);
}

#[test]
fn the_thresholds_paint_the_rows_and_move_no_pinned_verdict() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // Measure the track, so the rows have numbers for the bars to judge. The
    // fixture's track is at the track stage, where the leave-one-out ZNCC the
    // stored column carries is what a fresh bench track arrives with.
    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    run_frame(&mut panel, &ctx, &state);
    let pinned_before = panel.rows()[1].verdict;

    // A bar nothing can clear: every measured row would be refused.
    panel.thresholds.min_zncc = 1.01;
    run_frame(&mut panel, &ctx, &state);
    let rows = panel.rows();
    assert!(
        rows.iter().any(|row| row.painted == Verdict::Out),
        "an unreachable bar painted nothing out: {rows:?}"
    );
    assert_eq!(
        rows[1].verdict, pinned_before,
        "the painting moved a verdict the person set"
    );
    assert_eq!(
        rows[1].painted, pinned_before,
        "a pinned row is painted as what it is, not as what the bars propose"
    );

    // And the painting is exactly what applying the bars would do.
    state
        .apply_bench_thresholds(id, &label, panel.thresholds())
        .expect("on the bench");
    run_frame(&mut panel, &ctx, &state);
    for row in panel.rows() {
        assert_eq!(
            row.verdict, row.painted,
            "applying the bars produced a verdict the painting did not show"
        );
    }
    assert_eq!(
        panel.rows()[1].verdict,
        pinned_before,
        "applying the bars moved the pinned verdict"
    );
}

/// Every row draws its own rendered tile, at both stages: the patch seen from
/// that observation at the track stage, and the kernel's own grid at the
/// cluster stage.
#[test]
fn every_row_draws_a_tile_at_either_stage() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    assert!(
        panel.rows().iter().all(|row| row.tile),
        "a track from a committed point draws no tile: {:?}",
        panel.rows(),
    );

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    run_frame(&mut panel, &ctx, &state);
    assert!(
        panel.rows().iter().all(|row| row.tile),
        "the cluster stage draws no tile: {:?}",
        panel.rows(),
    );
}

/// A candidate a descriptor search has just added carries no keypoint -- only
/// the seed the index's warp gave it -- and its tile is cut around **that**,
/// which is the whole of what says whether the search found the right surface.
/// Before this, the row drew the patch wherever the bare projection of the
/// point happened to land in a photograph nothing had yet tied it to.
#[test]
fn a_search_candidate_draws_its_tile_where_the_seed_put_it() {
    use sfmtool_core::bench::Stage;

    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, label) = crate::sift_index::tests::searchable(dir.path());
    cache_photographs(&mut state, id);
    state
        .start_bench_descriptor_search(id, &label, 0, None, None)
        .expect("an index is open and the image has keypoints");
    state.finish_background_task();

    let track = state
        .bench_track(id, &label)
        .expect("still on the bench")
        .clone();
    let candidate = track.observations.len() - 1;
    let row = &track.observations[candidate];
    assert!(
        matches!(
            row.provenance,
            sfmtool_core::bench::Provenance::Search { .. }
        ),
        "the search added no candidate: {:?}",
        row.provenance
    );
    assert!(
        row.track.is_none(),
        "the candidate arrived already read, so this proves nothing"
    );
    let seed = row.cluster.as_ref().expect("a searched seed").seed_position;
    let image = ImageRef::new(id, row.image as usize);
    let src = state
        .full_res_cache
        .get(&image)
        .and_then(|slot| slot.clone())
        .expect("a cached photograph");

    let node = crate::scene::node_by_id(&state.scene, id).expect("loaded");
    let recon = node.recon();
    let drawn = super::tile::image(recon, &track, candidate, &src).expect("a tile");

    // The patch cut around where the observation sits, which is the same
    // picture the row draws once a reading has written that pixel as its
    // keypoint: where an observation is, is one question however it is
    // answered.
    let frame = match &track.stage {
        Stage::Track(payload) => payload.placement.clone().expect("a fitted patch"),
        Stage::Cluster(_) => panic!("a track put on from a point is at the track stage"),
    };
    let sfmr_image = &recon.image_table.images[row.image as usize];
    let camera = &recon.image_table.cameras[sfmr_image.camera_index as usize];
    let cam_from_world = crate::scene::cam_from_world(sfmr_image);
    let at_the_seed = crate::track_view::view::patch_color_image(
        &frame,
        camera,
        &cam_from_world,
        Some(seed),
        src.level(0),
    );
    assert_eq!(drawn, at_the_seed, "the tile is not cut around the seed");

    let at_the_projection = crate::track_view::view::patch_color_image(
        &frame,
        camera,
        &cam_from_world,
        None,
        src.level(0),
    );
    assert_ne!(
        drawn, at_the_projection,
        "the tile is the point's own projection rather than the sighting's place"
    );
}

#[test]
fn the_cells_follow_the_stage_the_track_is_in() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // The seven cells are ZNCC, seed shift, projection offset, sigma_pos,
    // reprojection error, ray angle, status. At the track stage the last three
    // and the projection offset have numbers behind them; at the cluster stage
    // there is no geometry behind an observation and they are absent.
    assert_eq!(panel.rows()[0].cells[4], "-", "nothing has measured it yet");

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    state
        .start_bench_evaluate(id, &label, None)
        .expect("a cluster evaluates over its seeds");
    state.finish_background_task();
    run_frame(&mut panel, &ctx, &state);

    let rows = panel.rows();
    assert_ne!(rows[0].cells[0], "-", "the ZNCC column is unmeasured");
    assert_eq!(rows[0].cells[2], "-", "a cluster has no point to project");
    assert_eq!(rows[0].cells[4], "-", "a cluster has no reprojection error");
    assert_eq!(rows[0].cells[5], "-", "a cluster has no ray angle");
}

/// A row the reading refused to widen its window for says so in the Status
/// cell, in the reading's own sentence.
///
/// The seed's distance from the projection is what sizes the search window, and
/// the tile that window renders costs its square, so a sighting placed a long
/// way from the point is left out of the round with a reason rather than
/// allocated for. The cell is where the person meets that decision.
#[test]
fn a_row_seeded_far_from_the_projection_says_so_in_the_status_cell() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state
        .add_bench_observation(
            &label,
            ImageRef::new(id, 5),
            &crate::bench::Seed::Pixel {
                pixel: [24.0, 24.0],
                radius_px: None,
            },
        )
        .expect("a pixel on the sensor");
    state
        .start_bench_evaluate(id, &label, None)
        .expect("a framed track reads");
    state.finish_background_task();
    run_frame(&mut panel, &ctx, &state);

    let rows = panel.rows();
    let status = rows.last().expect("the row just added").cells[6].clone();
    assert!(
        status.contains("beyond the 64 px bound"),
        "the cell names the bound the seed passed: {status}"
    );
    assert!(
        rows.iter().filter(|row| row.cells[0] != "-").count() >= 2,
        "and the rows that could be read still were: {rows:?}"
    );
}

/// Reading and moving are two gestures, so the toolbar offers two buttons, and
/// the search radius they carry is a control of its own beside the threshold
/// sliders -- an input to the next reading rather than a bar the painting
/// judges by.
#[test]
fn the_toolbar_offers_the_reading_and_the_fit_with_a_search_radius() {
    let (state, _, _, mut panel, ctx) = on_the_bench();
    assert_eq!(
        panel.search_px(),
        crate::bench::default_search_px(),
        "the control starts where core's own reading does"
    );

    let texts = crate::test_support::painted_texts(
        &ctx,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
            ..Default::default()
        },
        |ui| {
            panel.show(ui, &state);
        },
    );
    for label in ["Evaluate", "Fit", "search px"] {
        assert!(
            texts.iter().any(|t| t == label),
            "{label} is not in the toolbar: {texts:?}"
        );
    }
}

/// Edit mode draws the active item and nothing else on the bench: no row of
/// item tabs, so the labels of the other items appear nowhere in what the frame
/// painted. The bench as a list is the Scene tree's.
#[test]
fn edit_mode_draws_the_active_item_and_no_item_tabs() {
    let (mut state, id) = state();
    let first = state
        .put_point_on_bench(PointRef::new(id, POINT as usize))
        .expect("a live point");
    let second = state
        .start_bench_cluster(
            ImageRef::new(id, 0),
            &crate::bench::Seed::Pixel {
                pixel: [120.0, 90.0],
                radius_px: Some(6.0),
            },
        )
        .expect("a pixel on the sensor")
        .label;
    let third = state
        .start_bench_cluster(
            ImageRef::new(id, 1),
            &crate::bench::Seed::Pixel {
                pixel: [60.0, 40.0],
                radius_px: Some(6.0),
            },
        )
        .expect("a pixel on the sensor")
        .label;
    state
        .activate_bench_item(id, &second)
        .expect("on the bench");
    let mut panel = TrackEdit::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, &state);

    // The cluster is the active one, and it has the one observation it was
    // started with.
    assert_eq!(panel.rows().len(), 1);

    let texts = crate::test_support::painted_texts(
        &ctx,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
            ..Default::default()
        },
        |ui| {
            panel.show(ui, &state);
        },
    );
    assert!(
        texts.contains(&second),
        "the active item's header: {texts:?}"
    );
    for label in [&first, &third] {
        assert!(
            !texts.iter().any(|t| t.contains(label.as_str())),
            "{label} is not the active item and was painted: {texts:?}"
        );
    }
}

/// A row is an *observation*, so clicking one names both an image and a place
/// in it: the response carries the pixel the Image Detail panel's bench layer
/// draws that observation's mark at, so the two cannot disagree about where the
/// view should land.
#[test]
fn clicking_a_row_selects_its_image_and_reveals_the_observation() {
    let (state, id, label, mut panel, ctx) = on_the_bench();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let expected = crate::bench::observation_pixel(&track.observations[1])
        .expect("a track from a committed point carries its keypoints");

    // The second row, whose observation is in image 1.
    let y = row_y(&mut panel, &ctx, &state, 1);
    let response = at_pointer(&mut panel, &ctx, &state, egui::pos2(400.0, y + 8.0), true);

    assert_eq!(response.select_image, Some(1));
    assert_eq!(response.reveal_feature, Some(expected));
    assert_eq!(
        response.request_camera_view, None,
        "a single click asked for camera view"
    );
}

/// A double-click on a row enters camera view for its image, as a view-mode
/// row's does: the rows of both modes are observations of one track.
#[test]
fn double_clicking_a_row_asks_for_camera_view() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let y = row_y(&mut panel, &ctx, &state, 1);
    let pos = egui::pos2(400.0, y + 8.0);
    run_frame_with(
        &mut panel,
        &ctx,
        &state,
        vec![egui::Event::PointerMoved(pos)],
    );
    let mut events = vec![egui::Event::PointerMoved(pos)];
    for _ in 0..2 {
        for pressed in [true, false] {
            events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            });
        }
    }
    let response = run_frame_with(&mut panel, &ctx, &state, events);
    assert_eq!(response.request_camera_view, Some(1));
    assert_eq!(response.select_image, Some(1));
}

#[test]
fn the_sliders_keep_where_they_were_left() {
    let (state, _) = state();
    let mut panel = TrackEdit::new();
    assert_eq!(panel.thresholds(), &Thresholds::default());
    panel.thresholds.min_zncc = 0.5;
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds().min_zncc, 0.5);
}

/// The sliders show the **active track's** bars, whoever moved them: a step
/// taken over the wire moves the track's, and the panel that paints the rows by
/// them has to be showing the same numbers or it proposes a rule the track does
/// not hold.
#[test]
fn the_sliders_follow_the_active_track_s_own_thresholds() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // Dragged somewhere of the person's own first: what a step on the track
    // replaces is exactly that.
    panel.thresholds.min_zncc = 0.5;
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds().min_zncc, 0.5, "a drag did not survive");

    let bars = Thresholds {
        min_zncc: 0.94,
        min_relative_zncc: 0.62,
        ..Thresholds::default()
    };
    state
        .apply_bench_thresholds(id, &label, &bars)
        .expect("on the bench");
    run_frame(&mut panel, &ctx, &state);

    let track = state.bench_track(id, &label).expect("on the bench");
    assert_eq!(panel.thresholds(), &track.thresholds);
    assert_eq!(panel.thresholds().min_zncc, 0.94);
    // And the painting is the track's rule rather than the slider's old one.
    assert_eq!(
        panel.painted,
        sfmtool_core::bench::apply_thresholds(track)
            .0
            .observations
            .iter()
            .map(|o| o.verdict)
            .collect::<Vec<_>>()
    );
}

/// A second track has its own bars, so making it active moves the sliders.
#[test]
fn a_change_of_active_track_reseats_the_sliders() {
    let (mut state, id, first, mut panel, ctx) = on_the_bench();
    state
        .apply_bench_thresholds(
            id,
            &first,
            &Thresholds {
                min_zncc: 0.94,
                ..Thresholds::default()
            },
        )
        .expect("on the bench");
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds().min_zncc, 0.94);

    let second = state
        .start_bench_cluster(
            ImageRef::new(id, 0),
            &crate::bench::Seed::Pixel {
                pixel: [120.0, 90.0],
                radius_px: Some(6.0),
            },
        )
        .expect("a pixel on the sensor")
        .label;
    assert_ne!(second, first);
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds(), &Thresholds::default());
}

#[test]
fn the_column_headings_are_drawn_outside_the_scrolling_rows() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let painted = crate::test_support::painted_text_rects(
        &ctx,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
            ..Default::default()
        },
        |ui| {
            panel.show(ui, &state);
        },
    );
    let find = |text: &str| {
        painted
            .iter()
            .find(|painted| painted.text == text)
            .unwrap_or_else(|| {
                let all: Vec<&str> = painted.iter().map(|p| p.text.as_str()).collect();
                panic!("{text:?} was not painted: {all:?}")
            })
    };

    // The rows are clipped to the scroll area's viewport; a heading drawn
    // above that viewport is one that cannot scroll out of it. The name cell
    // is the row text furthest from any widget, so it is the one asked.
    let name = find("image_000.jpg");
    for (_, heading) in super::table::ColumnLayout::new().headers() {
        let painted = find(heading);
        assert!(
            painted.clip != name.clip,
            "the {heading:?} heading shares the rows' clip rectangle, so it scrolls with them",
        );
        assert!(
            painted.rect.max.y <= name.clip.min.y,
            "the {heading:?} heading sits inside the scrolling rows ({:?} against {:?})",
            painted.rect,
            name.clip,
        );
    }
}

#[test]
fn a_heading_stands_at_the_x_offset_its_column_is_drawn_at() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let painted = crate::test_support::painted_text_rects(
        &ctx,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
            ..Default::default()
        },
        |ui| {
            panel.show(ui, &state);
        },
    );
    let x_of = |text: &str| {
        painted
            .iter()
            .find(|painted| painted.text == text)
            .map(|painted| painted.rect.min.x)
            .unwrap_or_else(|| panic!("{text:?} was not painted"))
    };
    // Moving the heading out of the scroll area must not have moved it
    // sideways: the scroll bar takes width off the right edge, and every
    // column is measured from the left one.
    assert_eq!(
        x_of("Name"),
        x_of("image_000.jpg"),
        "the Name heading no longer stands over the name column",
    );
}

// ── The row's own menu and the SIFT index behind it ─────────────────────

/// The panel is about one track: nothing above the table names the node's
/// index, which lives on the Scene tree's own row.
#[test]
fn the_panel_draws_no_sift_index_row() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    for absent in ["Descriptor index", "SIFT Index", "Open..."] {
        assert!(
            !texts.iter().any(|t| t == absent),
            "{absent} is still drawn above the table: {texts:?}"
        );
    }
}

/// With no index beside the node, the entry a row offers is the build, because
/// a person who finds the search missing has no reason to look anywhere else.
#[test]
fn a_row_s_menu_offers_the_build_when_the_node_has_no_index() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let texts = row_menu(&mut panel, &ctx, &state);
    assert!(
        texts.iter().any(|t| t == super::BUILD_INDEX_TO_SEARCH),
        "the build entry is not in the row's menu: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t == super::SEARCH_DESCRIPTORS_LABEL),
        "the search is offered with no index to run it against: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t == super::SEARCH_GEOMETRY_LABEL),
        "geometry search wrongly depends on the SIFT index: {texts:?}"
    );
}

/// With a current index the entry is the search itself, live.
#[test]
fn a_row_s_menu_offers_the_search_against_a_current_index() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = crate::sift_index::tests::searchable(dir.path());
    cache_photographs(&mut state, id);
    let (mut panel, ctx) = settled(&state);

    let texts = row_menu(&mut panel, &ctx, &state);
    assert!(
        texts.iter().any(|t| t == super::SEARCH_DESCRIPTORS_LABEL),
        "the search entry is not in the row's menu: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t == super::REBUILD_INDEX_TO_SEARCH),
        "a current index is offered a rebuild: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t == super::SEARCH_GEOMETRY_LABEL),
        "the track-stage geometry entry is absent: {texts:?}"
    );
}

#[test]
fn geometry_search_is_track_stage_only_and_routes_its_own_response() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    let texts = row_menu(&mut panel, &ctx, &state);
    assert!(
        texts.iter().any(|t| t == super::SEARCH_GEOMETRY_LABEL),
        "the geometry entry is absent at the track stage: {texts:?}"
    );

    let entry = menu_entry_pos(&mut panel, &ctx, &state, super::SEARCH_GEOMETRY_LABEL);
    let response = at_pointer(&mut panel, &ctx, &state, entry, true);
    assert_eq!(response.search_geometry, Some(0));
    assert_eq!(
        response.search_descriptors, None,
        "the geometry entry was routed as a SIFT query"
    );

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("the framed track downgrades");
    state.finish_background_task();
    let (mut panel, ctx) = settled(&state);
    let texts = row_menu(&mut panel, &ctx, &state);
    assert!(
        !texts.iter().any(|t| t == super::SEARCH_GEOMETRY_LABEL),
        "a cluster-stage row offers geometry search: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t == super::BUILD_INDEX_TO_SEARCH),
        "stage gating accidentally removed the SIFT/build action: {texts:?}"
    );
}

/// With a stale one the entry is the rebuild, and choosing it starts the build
/// and no search.
#[test]
fn a_row_s_menu_offers_the_rebuild_when_the_index_is_stale_and_starts_it() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = crate::sift_index::tests::searchable(dir.path());
    cache_photographs(&mut state, id);
    // The features of one image, extracted again after the build.
    {
        let recon = state.node(id).expect("loaded").recon();
        crate::sift_index::tests::write_sift(
            &recon.sift_path_for_image(2),
            &recon.image_table.images[2].name,
            &vec![vec![5u8; 128]; 3],
            &[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
        );
    }
    let path = state.sift_index(id).expect("built").path.clone();
    state.open_sift_index(id, Some(path)).expect("it opens");
    assert_eq!(
        state.sift_index_state(id),
        crate::sift_index::SiftIndexState::Stale
    );

    let (mut panel, ctx) = settled(&state);
    let y = row_y(&mut panel, &ctx, &state, 0);
    let at = egui::pos2(400.0, y);
    open_row_menu(&mut panel, &ctx, &state, at);
    let texts = painted(
        &mut panel,
        &ctx,
        &state,
        vec![egui::Event::PointerMoved(at)],
    );
    assert!(
        texts.iter().any(|t| t == super::REBUILD_INDEX_TO_SEARCH),
        "the rebuild entry is not in the row's menu: {texts:?}"
    );

    // Clicking it asks for the build, and asks for no search.
    let entry = menu_entry_pos(&mut panel, &ctx, &state, super::REBUILD_INDEX_TO_SEARCH);
    let response = at_pointer(&mut panel, &ctx, &state, entry, true);
    assert!(response.build_sift_index, "the entry started no build");
    assert_eq!(
        response.search_descriptors, None,
        "the build does not run the search when it finishes, and does not run it now"
    );
}

/// *Duplicate* is the toolbar's own way to a second patch over neighbouring
/// ground: one version, a second item on the bench, and the copy active.
#[test]
fn duplicate_puts_a_second_item_on_the_bench_and_makes_it_active() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    let texts = crate::test_support::painted_texts(
        &ctx,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
            ..Default::default()
        },
        |ui| {
            panel.show(ui, &state);
        },
    );
    assert!(
        texts.iter().any(|t| t == "Duplicate"),
        "the toolbar does not offer it: {texts:?}"
    );

    let before = state.node(id).expect("loaded").history.versions().len();
    let copy = state
        .duplicate_bench_item(id, &label)
        .expect("the label is on the bench");
    assert_eq!(copy, format!("{label} copy"));
    let node = state.node(id).expect("loaded");
    assert_eq!(
        node.history.versions().len(),
        before + 1,
        "one gesture, one version"
    );
    assert_eq!(
        node.history.versions().last().expect("a version").label,
        format!("Duplicated {label} as {copy}")
    );
    let bench = state.bench(id).expect("a loaded node has a bench");
    assert_eq!(bench.len(), 2, "the bench holds the original and the copy");
    assert_eq!(crate::bench::active_track_label(bench), Some(copy.as_str()));
    assert_eq!(
        state.bench_track(id, &copy).and_then(|track| track.origin),
        None,
        "a copy commits as a creation"
    );

    // And the panel shows the copy, and nothing of the original.
    run_frame(&mut panel, &ctx, &state);
    let texts = crate::test_support::painted_texts(
        &ctx,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
            ..Default::default()
        },
        |ui| {
            panel.show(ui, &state);
        },
    );
    assert!(
        !texts.contains(&label),
        "the original was painted beside the copy: {texts:?}"
    );
    // The copy is the one being drawn: its header says it commits as a
    // creation, because it has no origin.
    assert!(texts.contains(&copy), "{texts:?}");
    assert!(
        texts.iter().any(|t| t.contains("new")),
        "the copy's header does not say it creates: {texts:?}"
    );
    assert_eq!(panel.rows().len(), 3, "the copy carries every sighting");
}

/// A track-stage track standing on a bearing: the payload a `w = 0` point
/// arrives on the bench as.
fn bearing_track() -> sfmtool_core::bench::EditableTrack {
    use sfmtool_core::bench::{EditableTrack, Stage, TrackPayload};
    use sfmtool_core::patch::cloud::OrientedPatch;

    let direction = nalgebra::Point3::new(0.0, 0.0, 1.0);
    let mut track = EditableTrack::empty_cluster();
    track.stage = Stage::Track(TrackPayload {
        position: Some(direction),
        // The track's own flag is what the header reads; the frame's `w` agrees
        // with it wherever a frame exists.
        at_infinity: true,
        placement: Some(OrientedPatch::from_infinity_direction(
            direction,
            nalgebra::Vector3::y(),
            [0.03, 0.03],
        )),
        condition_number: Some(68_848.0),
        ..TrackPayload::default()
    });
    track
}

/// The same track standing on a place.
fn position_track() -> sfmtool_core::bench::EditableTrack {
    use sfmtool_core::bench::{Stage, TrackPayload};
    use sfmtool_core::patch::cloud::OrientedPatch;

    let center = nalgebra::Point3::new(1.0, 2.0, 3.0);
    let mut track = bearing_track();
    track.stage = Stage::Track(TrackPayload {
        position: Some(center),
        placement: Some(OrientedPatch::from_center_normal(
            center,
            nalgebra::Vector3::z(),
            nalgebra::Vector3::y(),
            [0.1, 0.1],
        )),
        ..TrackPayload::default()
    });
    track
}

/// Everything the header painted for `track`, joined.
fn header_text(track: &sfmtool_core::bench::EditableTrack) -> String {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        ..Default::default()
    };
    crate::test_support::painted_texts(&ctx, input, |ui| {
        super::show_header(ui, "item", track);
    })
    .join(" | ")
}

/// A bearing and a position are the same three numbers under different rules,
/// so the word in front of them is what tells a reader which they are looking
/// at. Printing a bearing as a position reads as a point one unit from the
/// world origin, which is the one thing it is not.
#[test]
fn the_header_names_a_bearing_a_bearing_and_a_position_a_position() {
    let bearing = header_text(&bearing_track());
    assert!(
        bearing.contains("Bearing (0.000, 0.000, 1.000)"),
        "the header should name the direction a bearing: {bearing}"
    );
    assert!(
        bearing.contains("at infinity"),
        "and say so in the panel's own words: {bearing}"
    );
    assert!(
        !bearing.contains("Position ("),
        "a bearing is not a position: {bearing}"
    );

    let position = header_text(&position_track());
    assert!(
        position.contains("Position (1.000, 2.000, 3.000)"),
        "{position}"
    );
    assert!(
        !position.contains("at infinity") && !position.contains("Bearing ("),
        "{position}"
    );
}

/// A sighting the fit refused to walk did **not** move where the correlation
/// wanted it, which is the one thing about the row a person reading "localized"
/// would get wrong. So the Status cell says it, and says how far.
#[test]
fn a_sighting_kept_at_its_seed_says_so_in_the_status_cell() {
    use sfmtool_core::bench::{Observation, Provenance, TrackMeasurement};

    let walked = Observation {
        image: 0,
        provenance: Provenance::Origin,
        verdict: Verdict::In,
        pinned: false,
        cluster: None,
        track: Some(TrackMeasurement {
            keypoint: Some([10.0, 12.0]),
            zncc: Some(0.41),
            walked_px: Some(19.4),
            ..TrackMeasurement::default()
        }),
    };
    let cells = super::measurements(&walked, StageKind::Track);
    assert_eq!(cells[6], "walked 19 px, kept at seed");

    // The same row without the flag is the ordinary scored row.
    let mut moved = walked.clone();
    moved.track.as_mut().expect("a track slot").walked_px = None;
    assert_eq!(
        super::measurements(&moved, StageKind::Track)[6],
        "localized"
    );
}

/// A track-stage track standing on a bearing with **no patch**: the payload a
/// `w = 0` point of a node with no patch frames arrives on the bench as, which
/// is every row of a `sift_files` value.
fn frameless_bearing_track() -> sfmtool_core::bench::EditableTrack {
    use sfmtool_core::bench::{EditableTrack, Stage, TrackPayload};

    let direction = nalgebra::Point3::new(0.778, -0.611, -0.146);
    let mut track = EditableTrack::empty_cluster();
    track.stage = Stage::Track(TrackPayload {
        position: Some(direction),
        at_infinity: true,
        placement: None,
        ..TrackPayload::default()
    });
    track
}

/// The header reads the **track's** flag and not its patch's `w`, so a bearing
/// that carries no patch still reads as a bearing. Reading the patch printed
/// `Position (0.778, -0.611, -0.146)` for a unit direction, which is a place one
/// unit from the world origin and the one thing that row is not.
#[test]
fn a_bearing_with_no_patch_still_reads_as_a_bearing() {
    let said = header_text(&frameless_bearing_track());
    assert!(
        said.contains("Bearing (0.778, -0.611, -0.146)"),
        "the header should name it a bearing: {said}"
    );
    assert!(said.contains("at infinity"), "{said}");
    assert!(!said.contains("Position ("), "{said}");
}

/// *Evaluate* and the *Stage* toggle are greyed by what the track is missing,
/// the way *Fit* already was: all three ask core's own half of their step's
/// validation, so a button that cannot work is not offered and the sentence a
/// person reads is the one the step would have refused with.
#[test]
fn the_photometric_entries_grey_with_their_own_sentence_on_a_frameless_track() {
    let track = frameless_bearing_track();
    let refusals = super::photometric_refusals(None, &track, StageKind::Cluster);
    for (what, refusal) in [
        ("Evaluate", &refusals.evaluate),
        ("Fit", &refusals.fit),
        ("Stage", &refusals.stage),
    ] {
        let why = refusal
            .as_deref()
            .unwrap_or_else(|| panic!("{what} should be greyed on a track with no patch"));
        assert!(
            why.contains("no patch"),
            "{what} is greyed with {why:?}, which does not name what is missing"
        );
    }

    // And a whole track offers all three.
    let (state, id) = state();
    let mut state = state;
    let label = state
        .put_point_on_bench(PointRef::new(id, POINT as usize))
        .expect("a live point");
    let whole = state.bench_track(id, &label).expect("on the bench");
    let refusals = super::photometric_refusals(None, whole, StageKind::Cluster);
    assert!(refusals.evaluate.is_none(), "{:?}", refusals.evaluate);
    assert!(refusals.fit.is_none(), "{:?}", refusals.fit);
    assert!(refusals.stage.is_none(), "{:?}", refusals.stage);

    // Busy wins over everything, as it did before.
    let refusals = super::photometric_refusals(Some("Busy."), whole, StageKind::Cluster);
    assert_eq!(refusals.evaluate.as_deref(), Some("Busy."));
    assert_eq!(refusals.fit.as_deref(), Some("Busy."));
    assert_eq!(refusals.stage.as_deref(), Some("Busy."));
}
