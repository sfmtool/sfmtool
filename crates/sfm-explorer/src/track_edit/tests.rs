// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the Track Edit panel.
//!
//! egui needs no GPU to lay out a frame, so the whole panel runs through
//! `Context::run_ui` here: `show` really does draw the tabs, the header, the
//! toolbar, the sliders and every row of the table. What the assertions target
//! is what the panel *decides* -- which rows it drew, what each says, what the
//! sliders paint, and what it reports back to the dock -- rather than pixels.

use sfmtool_core::bench::{StageKind, Thresholds, Verdict};
use sfmtool_core::camera::remap::ImageU8;

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
    let camera = &state.scene[0].recon().image_table.cameras[0];
    let (w, h) = (camera.width, camera.height);
    for image in 0..state.scene[0].image_count() {
        let data: Vec<u8> = (0..(w * h * 3))
            .map(|i| {
                let p = i / 3;
                ((p % w) % 9 * 14 + (p / w) % 7 * 18) as u8
            })
            .collect();
        state
            .full_res_cache
            .insert(ImageRef::new(id, image), Some(ImageU8::new(w, h, 3, data)));
    }
    (state, id)
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
/// down the panel: what sits above the table is the tabs, the header, the
/// toolbar and the sliders, and a hard-coded offset would go stale the moment
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

#[test]
fn an_empty_bench_offers_the_ways_in_and_draws_no_rows() {
    let (state, _) = state();
    let mut panel = TrackEdit::new();
    let ctx = egui::Context::default();
    let response = run_frame(&mut panel, &ctx, &state);

    assert!(panel.rows().is_empty());
    assert!(!response.put_selected_point_on_bench);
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
        texts.iter().any(|t| t == "No track on the bench"),
        "{texts:?}"
    );
    assert!(
        texts.iter().any(|t| t == super::PUT_ON_BENCH_LABEL),
        "the way onto the bench is not offered: {texts:?}"
    );
    // The other way in is a pixel, and a pixel is named in the Image Detail
    // panel's context menu rather than here: what this panel carries is the
    // sentence saying so, quoting that entry's own label.
    assert!(
        texts
            .iter()
            .any(|t| t.contains(crate::image_detail::START_CLUSTER_LABEL)),
        "the pixel gesture is not pointed at: {texts:?}"
    );
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

/// Every row draws its own rendered tile, at both stages: the surfel seen from
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

#[test]
fn the_cells_follow_the_stage_the_track_is_in() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // At the track stage the reprojection error and the ray angle have columns;
    // at the cluster stage there is no geometry behind an observation and the
    // two are absent.
    assert_eq!(panel.rows()[0].cells[3], "-", "nothing has measured it yet");

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    state
        .start_bench_evaluate(id, &label)
        .expect("a cluster evaluates over its seeds");
    state.finish_background_task();
    run_frame(&mut panel, &ctx, &state);

    let rows = panel.rows();
    assert_ne!(rows[0].cells[0], "-", "the ZNCC column is unmeasured");
    assert_eq!(rows[0].cells[3], "-", "a cluster has no reprojection error");
    assert_eq!(rows[0].cells[4], "-", "a cluster has no ray angle");
}

#[test]
fn the_tabs_name_every_item_and_the_active_one_is_the_one_shown() {
    let (mut state, id) = state();
    let first = state
        .put_point_on_bench(PointRef::new(id, POINT as usize))
        .expect("a live point");
    let second = state
        .start_bench_cluster(ImageRef::new(id, 0), [120.0, 90.0], 6.0)
        .expect("a pixel on the sensor");
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
    for label in [&first, &second] {
        assert!(
            texts.iter().any(|t| t.starts_with(label.as_str())),
            "{label} is not named anywhere: {texts:?}"
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
