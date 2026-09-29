// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for Track View's body, most of them in Edited mode; the
//! Viewed-mode tests are in `tests/viewed.rs`.
//!
//! egui needs no GPU to lay out a frame, so the whole body runs through
//! `Context::run_ui` here: `show` really does draw the header, the toolbar,
//! the boxes and every row of the table. What the assertions target
//! is what the panel *decides* -- which rows it drew, what each says, what the
//! boxes paint, and what it reports back to the dock -- rather than pixels.

use sfmtool_core::bench::{StageKind, Thresholds, Verdict};
use sfmtool_core::camera::remap::ImageU8;

use super::{TrackBody, TrackBodyResponse, LOCK_LABEL};
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
        state.insert_photograph(id, image, ImageU8::new(w, h, 3, data));
    }
}

/// Drive one frame of the panel and hand back what it reported.
fn run_frame(panel: &mut TrackBody, ctx: &egui::Context, state: &AppState) -> TrackBodyResponse {
    run_frame_with(panel, ctx, state, Vec::new())
}

/// The same frame, with `events` delivered to egui.
fn run_frame_with(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &AppState,
    events: Vec<egui::Event>,
) -> TrackBodyResponse {
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
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &AppState,
    pos: egui::Pos2,
    click: bool,
) -> TrackBodyResponse {
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
/// the boxes, and a hard-coded offset would go stale the moment
/// one of them gains a line.
fn row_y(panel: &mut TrackBody, ctx: &egui::Context, state: &AppState, image: usize) -> f32 {
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
    panel: &mut TrackBody,
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
fn settled(state: &AppState) -> (TrackBody, egui::Context) {
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, state);
    (panel, ctx)
}

/// Right-click at `at` and leave the menu it opened laid out.
///
/// Three frames: one to register the rows, one that right-clicks, and one
/// more, because the menu's entries are laid out on a later frame.
fn open_row_menu(panel: &mut TrackBody, ctx: &egui::Context, state: &AppState, at: egui::Pos2) {
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
fn row_menu(panel: &mut TrackBody, ctx: &egui::Context, state: &AppState) -> Vec<String> {
    let y = row_y(panel, ctx, state, 0);
    let at = egui::pos2(400.0, y);
    open_row_menu(panel, ctx, state, at);
    painted(panel, ctx, state, vec![egui::Event::PointerMoved(at)])
}

/// Where an open menu drew the entry called `text`.
fn menu_entry_pos(
    panel: &mut TrackBody,
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
fn on_the_bench() -> (AppState, ReconId, String, TrackBody, egui::Context) {
    let (mut state, id) = state();
    let label = state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, &state);
    (state, id, label, panel, ctx)
}

/// With nothing focused and no viewed track the body draws nothing: Track View
/// draws its empty state then, and the ways in are that state's.
#[test]
fn with_nothing_active_the_body_draws_no_rows_and_no_text() {
    let (state, _) = state();
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    let response = run_frame(&mut panel, &ctx, &state);

    assert!(panel.rows().is_empty());
    assert_eq!(response, TrackBodyResponse::default());
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
        assert!(row.pinned, "a track from a point arrives pinned");
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

/// The *Keep* cells show what the bars propose, a drag of the boxes moves no
/// pinned verdict, and an unpinned row's proposal is exactly what applying the
/// bars makes it.
#[test]
fn the_thresholds_propose_for_the_rows_and_move_no_pinned_verdict() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // Measure the track, so the rows have numbers for the bars to judge.
    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);
    assert!(
        panel.rows().iter().all(|row| row.proposal.is_some()),
        "every row is measured: {:?}",
        panel.rows()
    );
    let pinned_before = panel.rows()[1].verdict;

    // A bar nothing can clear: every measured row would be refused.
    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = 1.01;
    });
    let rows = panel.rows();
    assert!(
        rows.iter().all(|row| row.proposal == Some(Verdict::Out)),
        "an unreachable bar proposed a row in: {rows:?}"
    );
    assert_eq!(
        rows[1].verdict, pinned_before,
        "the proposal moved a verdict the person set"
    );

    // And an unpinned row's proposal is exactly what applying the bars does.
    state
        .apply_bench_thresholds(id, &label, panel.thresholds())
        .expect("on the bench");
    run_frame(&mut panel, &ctx, &state);
    for row in panel.rows().iter().filter(|row| !row.pinned) {
        assert_eq!(
            Some(row.verdict),
            row.proposal,
            "applying the bars produced a verdict the Keep cell did not show"
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

/// A row drawn before its photograph is decoded gets its tile and its crop on a
/// later frame once the photograph arrives, without the track moving: a
/// photograph not decoded yet is not remembered as a row with neither.
#[test]
fn a_tile_appears_when_its_photograph_arrives_after_the_first_frame() {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(projected_embedded_demo(12)));
    let id = state.selected_recon.expect("a selected reconstruction");
    state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, &state);
    assert!(
        panel.rows().iter().all(|row| !row.tile && !row.crop),
        "a row drew a tile or a crop with no photograph decoded: {:?}",
        panel.rows(),
    );

    cache_photographs(&mut state, id);
    run_frame(&mut panel, &ctx, &state);
    assert!(
        panel.rows().iter().all(|row| row.tile && row.crop),
        "a row drawn before its photograph arrived stayed without a tile or a crop: {:?}",
        panel.rows(),
    );
}

/// A candidate a descriptor search has just added sits where the index's warp
/// put it -- its keypoint and its seed are that pixel, and nothing has read it
/// -- and its tile is cut around **that**, which is the whole of what says
/// whether the search found the right surface, rather than wherever the bare
/// projection of the point happens to land in the photograph.
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
        row.track.as_ref().is_none_or(|m| m.zncc.is_none()),
        "the candidate arrived already read, so this proves nothing"
    );
    let seed = row.cluster.as_ref().expect("a searched seed").seed_position;
    let site = row.site().expect("the candidate sits somewhere");
    assert!(
        (site[0] - seed[0]).abs() < 1e-3 && (site[1] - seed[1]).abs() < 1e-3,
        "the candidate sits at {site:?}, not at the seed {seed:?}"
    );
    let image = ImageRef::new(id, row.image as usize);
    let src = state
        .cached_photograph(image.recon, image.index())
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
    let at_the_seed =
        super::patch::patch_color_image(&frame, camera, &cam_from_world, Some(site), src.level(0));
    assert_eq!(drawn, at_the_seed, "the tile is not cut around the seed");

    let at_the_projection =
        super::patch::patch_color_image(&frame, camera, &cam_from_world, None, src.level(0));
    assert_ne!(
        drawn, at_the_projection,
        "the tile is the point's own projection rather than the sighting's place"
    );
}

#[test]
fn the_cells_follow_the_stage_the_track_is_in() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // The five cells are ZNCC, seed shift, projection error, self-similarity
    // and status. At the track stage the projection error has
    // numbers behind it; at the cluster stage there is no geometry behind an
    // observation and it is absent.
    assert_eq!(panel.rows()[0].cells[2], "-", "nothing has measured it yet");

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);

    let rows = panel.rows();
    assert_ne!(rows[0].cells[0], "-", "the ZNCC column is unmeasured");
    let (whole, middle) = rows[0].cells[0]
        .split_once('\n')
        .expect("the ZNCC cell shows the whole over the middle reading");
    let percent = |line: &str, part: &str| {
        let number = line
            .strip_suffix(part)
            .and_then(|rest| rest.strip_suffix("% "))
            .unwrap_or_else(|| panic!("{line:?} is not a percent named {part}"));
        assert!(number.parse::<f64>().is_ok(), "{line}");
    };
    percent(whole, "whole");
    percent(middle, "mid");
    assert_eq!(rows[0].cells[2], "-", "a cluster has no point to project");
    // Beside the ZNCC cell, the row draws its grid.
    assert!(rows[0].grids.zncc.is_some(), "no ZNCC grid drawn");
    // And beside the self-similarity cell, its grid and its slides.
    assert!(
        rows[0].cells[3].contains(" px whole\n") && rows[0].cells[3].ends_with(" px mid"),
        "{}",
        rows[0].cells[3]
    );
    assert!(
        rows[0].grids.radius.is_some(),
        "no self-similarity grid drawn"
    );
    assert!(
        rows[0].grids.radius_slide.is_some(),
        "no self-similarity slides"
    );
    // And the core's surface plot.
    assert!(
        rows[0].self_similarity_plot,
        "no self-similarity surface plot"
    );
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
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);

    let rows = panel.rows();
    let status = rows.last().expect("the row just added").cells[4].clone();
    assert!(
        status.contains("beyond the 64 px bound"),
        "the cell names the bound the seed passed: {status}"
    );
    assert!(
        rows.iter().filter(|row| row.cells[0] != "-").count() >= 2,
        "and the rows that could be read still were: {rows:?}"
    );
}

/// Evaluation is live, so there is no *Evaluate* button: the toolbar offers
/// *Fit*, which moves the track, and says where the evaluation of the track as
/// it stands is. There is no search radius control: the evaluation looks
/// within the *shift px* bar.
#[test]
fn the_toolbar_offers_the_fit_and_says_where_the_evaluation_stands() {
    let (mut state, _, _, mut panel, ctx) = on_the_bench();
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(
        !texts.iter().any(|t| t == "search px"),
        "the search px box is still drawn: {texts:?}"
    );
    for label in ["Fit", super::MAX_SHIFT_LABEL, super::EVALUATING_LABEL] {
        assert!(
            texts.iter().any(|t| t == label),
            "{label} is not in the toolbar: {texts:?}"
        );
    }
    assert!(
        !texts.iter().any(|t| t == "Evaluate"),
        "the Evaluate button is still drawn: {texts:?}"
    );

    state.settle_bench_evaluation();
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(
        texts.iter().any(|t| t == super::EVALUATED_LABEL),
        "an evaluated track does not say so: {texts:?}"
    );
    assert!(!texts.iter().any(|t| t == super::EVALUATING_LABEL));
}

/// While an evaluation of the track's current inputs is on its way the rows do
/// not present the last evaluation's numbers as the track's: each status cell
/// says the row is being evaluated, until the evaluation lands.
#[test]
fn the_rows_read_evaluating_until_the_evaluation_of_the_current_inputs_lands() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    assert_eq!(
        panel.evaluation(),
        &crate::bench::live::Evaluation::Evaluating
    );
    assert!(
        panel
            .rows()
            .iter()
            .all(|row| row.cells[4] == super::EVALUATING_LABEL),
        "{:?}",
        panel.rows()
    );

    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.evaluation(), &crate::bench::live::Evaluation::Current);
    assert!(
        panel
            .rows()
            .iter()
            .all(|row| row.cells[4] != super::EVALUATING_LABEL),
        "{:?}",
        panel.rows()
    );

    // A step on the track puts every row back to evaluating.
    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    run_frame(&mut panel, &ctx, &state);
    assert!(
        panel
            .rows()
            .iter()
            .all(|row| row.cells[4] == super::EVALUATING_LABEL),
        "{:?}",
        panel.rows()
    );
}

/// A track that cannot be evaluated says why where the *Evaluate* button's
/// refusal used to be, and its rows print no numbers at all.
#[test]
fn a_track_that_cannot_be_evaluated_shows_the_reason_instead_of_values() {
    let (mut state, id) = state();
    let (bench, _) = sfmtool_core::bench::Bench::new().put(
        "bearing",
        sfmtool_core::bench::BenchItem::Track(std::sync::Arc::new(frameless_bearing_track())),
    );
    let index = state.node_index(id).expect("loaded");
    state.scene[index]
        .history
        .push_bench(std::sync::Arc::new(bench), "Put a bearing on the bench");
    state.focus_put_item(id, "bearing");
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();

    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    let why = match panel.evaluation() {
        crate::bench::live::Evaluation::Refused(why) => why.clone(),
        other => panic!("expected a refusal, got {other:?}"),
    };
    assert!(why.starts_with("Cannot evaluate bearing"), "{why}");
    assert!(texts.contains(&why), "the reason is not drawn: {texts:?}");
}

/// Edited mode draws the focused item and nothing else on the bench: no row of
/// item tabs, so the labels of the other items appear nowhere in what the frame
/// painted. The bench as a list is the Scene tree's.
#[test]
fn edit_mode_draws_the_focused_item_and_no_item_tabs() {
    let (mut state, id) = state();
    let first = state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    let second = state
        .start_bench_cluster(
            ImageRef::new(id, 0),
            &crate::bench::Seed::Pixel {
                pixel: [120.0, 90.0],
                radius_px: Some(6.0),
            },
            None,
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
            None,
        )
        .expect("a pixel on the sensor")
        .label;
    state.focus_bench_item(id, &second).expect("on the bench");
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, &state);

    // The cluster is the focused one, and it has the one observation it was
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
        "the focused item's header: {texts:?}"
    );
    for label in [&first, &third] {
        assert!(
            !texts.iter().any(|t| t.contains(label.as_str())),
            "{label} is not the focused item and was painted: {texts:?}"
        );
    }
}

/// A row is an *observation*, so clicking one names both an image and a place
/// in it: the response carries the pixel the Image Detail panel's bench layer
/// draws that observation's mark at, so the two cannot disagree about where the
/// view should land. It also picks the observation out, which the panel reports
/// rather than holds: the selection is the bench's.
#[test]
fn clicking_a_row_selects_its_image_and_reveals_the_observation() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
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
    assert_eq!(response.pick_row, Some((1, false)));

    let (observation, extend) = response.pick_row.expect("a pick");
    state.pick_bench_observation(id, &label, observation, extend);
    assert_eq!(state.selected_bench_observations(id, &label), [1]);
    state.pick_bench_observation(id, &label, 0, true);
    assert_eq!(state.selected_bench_observations(id, &label), [0, 1]);
    state.pick_bench_observation(id, &label, 1, true);
    assert_eq!(state.selected_bench_observations(id, &label), [0]);
}

/// A double-click on a row enters camera view for its image, as a Viewed-mode
/// row's does: the rows of both modes are observations of one track. The
/// row's observation goes with it, for the view to turn toward.
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
    assert!(
        response.reveal_feature.is_some(),
        "the double-click did not say where the observation is"
    );
}

/// *Lock* starts ticked, which is the dot moving the whole patch, and toggling
/// it is a tool setting rather than a step: no version, no gesture reported,
/// and the box holds what it was set to from one frame to the next.
#[test]
fn the_lock_starts_ticked_and_toggling_it_pushes_no_version() {
    let (state, id, _, mut panel, ctx) = on_the_bench();
    assert!(TrackBody::new().lock(), "a new panel starts locked");
    assert!(panel.lock());
    let versions = state.node(id).expect("loaded").history.versions().len();

    let at = menu_entry_pos(&mut panel, &ctx, &state, LOCK_LABEL);
    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    assert!(!panel.lock(), "a click on the box clears it");
    assert_eq!(
        TrackBodyResponse {
            mode: Some(super::BodyMode::Edited),
            has_pointer: response.has_pointer,
            hovered_image: response.hovered_image,
            ..TrackBodyResponse::default()
        },
        response,
        "toggling the lock asked the dock for a step",
    );
    run_frame(&mut panel, &ctx, &state);
    assert!(!panel.lock(), "the box keeps what it was set to");
    assert_eq!(
        state.node(id).expect("loaded").history.versions().len(),
        versions,
        "toggling the lock pushed a version",
    );

    at_pointer(&mut panel, &ctx, &state, at, true);
    assert!(panel.lock(), "a second click ticks it again");
}

/// At the cluster stage every handle is one sighting's already, so the box is
/// drawn but greyed: a click on it changes nothing.
#[test]
fn the_lock_is_greyed_at_the_cluster_stage() {
    let (mut state, id) = state();
    state
        .start_bench_cluster(
            ImageRef::new(id, 0),
            &crate::bench::Seed::Pixel {
                pixel: [120.0, 90.0],
                radius_px: Some(6.0),
            },
            None,
        )
        .expect("a pixel on the sensor");
    let (mut panel, ctx) = settled(&state);
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(
        texts.iter().any(|text| text == LOCK_LABEL),
        "the box is drawn at the cluster stage too: {texts:?}"
    );
    let at = menu_entry_pos(&mut panel, &ctx, &state, LOCK_LABEL);
    at_pointer(&mut panel, &ctx, &state, at, true);
    assert!(panel.lock(), "a greyed box took the click");
}

/// Outside a drag the boxes hold no value of their own: whatever the panel
/// had is replaced by the track's bars on the next frame.
#[test]
fn the_boxes_show_the_focused_item_s_bars_outside_a_drag() {
    let (state, id, label, mut panel, ctx) = on_the_bench();
    panel.thresholds.min_zncc = 0.5;
    run_frame(&mut panel, &ctx, &state);
    let track = state.bench_track(id, &label).expect("on the bench");
    assert_eq!(panel.thresholds(), &track.thresholds);
}

/// The boxes show the **focused item's** bars, whoever moved them: a step
/// taken over the wire moves the track's, and the panel that paints the rows by
/// them has to be showing the same numbers or it proposes a rule the track does
/// not hold. An undo moves them back.
#[test]
fn the_boxes_follow_the_focused_item_s_own_thresholds() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // Handed to the bars, so the painting below is every row's proposal.
    state
        .unpin_bench_verdicts(id, &label, &[0, 1, 2])
        .expect("the rows exist");
    let before = state
        .bench_track(id, &label)
        .expect("on the bench")
        .thresholds
        .clone();

    let bars = Thresholds {
        min_zncc: 0.94,
        min_relative_zncc: 0.62,
        ..Thresholds::default()
    };
    state
        .apply_bench_thresholds(id, &label, &bars)
        .expect("on the bench");
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);

    let track = state.bench_track(id, &label).expect("on the bench");
    assert_eq!(panel.thresholds(), &track.thresholds);
    assert_eq!(panel.thresholds().min_zncc, 0.94);
    // And the proposals are the track's rule rather than the box's old one.
    assert_eq!(
        panel
            .judged
            .iter()
            .map(|judged| judged.map(|j| j.proposal))
            .collect::<Vec<_>>(),
        sfmtool_core::bench::apply_thresholds(track)
            .0
            .observations
            .iter()
            .map(|o| Some(o.verdict))
            .collect::<Vec<_>>()
    );

    state.undo(id).expect("the thresholds step undoes");
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds(), &before, "an undo moves the boxes back");
    state.redo(id).expect("and redoes");
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds().min_zncc, 0.94, "a redo moves them again");
}

/// The pointer events of one drag, as one list per frame: hover, press, two
/// moves and the release. egui resolves a press against the rects the previous
/// frame registered, so each event gets a frame of its own.
fn drag_frames(from: egui::Pos2, to: egui::Pos2) -> Vec<Vec<egui::Event>> {
    let button = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::default(),
    };
    let mid = from + (to - from) * 0.5;
    vec![
        vec![egui::Event::PointerMoved(from)],
        vec![button(from, true)],
        vec![egui::Event::PointerMoved(mid)],
        vec![egui::Event::PointerMoved(to)],
        vec![button(to, false)],
        vec![],
    ]
}

/// Drag the *max shift px* box `by` points to the right (left when negative),
/// applying each frame's response the way the dock does, and hand back every
/// frame's response.
///
/// The box is found from its label: the label comes first and the box one
/// gap to its right.
fn drag_max_shift(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &mut AppState,
    id: ReconId,
    label: &str,
    by: f32,
) -> Vec<TrackBodyResponse> {
    let texts = crate::test_support::painted_text_rects(ctx, input(Vec::new()), |ui| {
        panel.show(ui, state);
    });
    let named = texts
        .iter()
        .find(|t| t.text == super::MAX_SHIFT_LABEL)
        .expect("the max shift box is drawn")
        .rect;
    let spacing = egui::Spacing::default();
    let start = egui::pos2(
        named.right() + spacing.item_spacing.x + 0.5 * spacing.interact_size.x,
        named.center().y,
    );
    let mut responses = Vec::new();
    for events in drag_frames(start, start + egui::vec2(by, 0.0)) {
        let response = run_frame_with(panel, ctx, state, events);
        if let Some(bars) = response.apply_thresholds.as_ref() {
            state
                .apply_bench_thresholds(id, label, bars)
                .expect("on the bench");
        }
        responses.push(response);
    }
    responses
}

fn versions(state: &AppState, id: ReconId) -> usize {
    state.node(id).expect("loaded").history.versions().len()
}

/// A box applies on its release: the drag's frames push nothing, the
/// release pushes exactly one version with one Action Log row, and the
/// track's bar is where the box was let go.
#[test]
fn releasing_a_threshold_box_applies_it_as_one_version() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    let before = versions(&state, id);
    let bar = state
        .bench_track(id, &label)
        .expect("on the bench")
        .thresholds
        .max_shift_px;
    assert_eq!(bar, sfmtool_core::bench::BENCH_MAX_SHIFT_PX);
    state.action_log.clear();

    let responses = drag_max_shift(&mut panel, &ctx, &mut state, id, &label, 100.0);
    let applied: Vec<&Thresholds> = responses
        .iter()
        .filter_map(|r| r.apply_thresholds.as_ref())
        .collect();
    assert_eq!(applied.len(), 1, "one release, one application");
    assert_eq!(versions(&state, id), before + 1, "one version for the drag");
    let track = state.bench_track(id, &label).expect("on the bench");
    assert_ne!(track.thresholds.max_shift_px, bar, "the drag moved the bar");
    assert_eq!(&track.thresholds, applied[0]);
    let rows: Vec<String> = state.action_log.entries().map(|e| e.text.clone()).collect();
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(rows[0].starts_with("Applied the thresholds to"), "{rows:?}");

    // The panel shows the track's bar after the release.
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds(), &track.thresholds);

    // And an undo takes the bar and the box back together.
    state.undo(id).expect("undoes");
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds().max_shift_px, bar);
}

/// There is no button left to forget to press: the boxes are the whole
/// gesture.
#[test]
fn there_is_no_apply_thresholds_button() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(
        !texts.iter().any(|t| t == "Apply thresholds"),
        "the button is gone: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t == super::MAX_SHIFT_LABEL),
        "{texts:?}"
    );
}

/// The bar a box was let go at is the bar the next *Fit* bounds its walk
/// by, and a kept-at-seed row then offers *Accept walk*, which moves the
/// keypoint to the walked pixel as one version.
#[test]
fn a_fit_after_a_release_uses_the_new_bar_and_accept_walk_moves_the_keypoint() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // A bar of zero: any move at all is a walk past it.
    drag_max_shift(&mut panel, &ctx, &mut state, id, &label, -400.0);
    assert_eq!(
        state
            .bench_track(id, &label)
            .expect("on the bench")
            .thresholds
            .max_shift_px,
        0.0
    );

    // Before the fit no row has a walk to accept.
    let menu = row_menu(&mut panel, &ctx, &state);
    assert!(
        !menu.iter().any(|t| t == super::ACCEPT_WALK_LABEL),
        "{menu:?}"
    );

    state
        .start_bench_fit(id, &label)
        .expect("a framed track with three sightings fits");
    state.finish_background_task();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let walked: Vec<usize> = (0..track.observations.len())
        .filter(|&i| {
            track.observations[i]
                .track
                .as_ref()
                .is_some_and(|m| m.walked_to.is_some())
        })
        .collect();
    assert!(
        !walked.is_empty(),
        "a zero bar keeps every moved sighting at its seed"
    );

    // Each kept-at-seed row offers the entry; the others do not.
    for (index, observation) in track.observations.iter().enumerate() {
        let offered = super::table::accepted_walk(observation).is_some();
        assert_eq!(offered, walked.contains(&index), "row {index}");
    }
    // A fresh panel, so the menu opened above is not lying over the rows.
    let (mut panel, ctx) = settled(&state);
    let first = walked[0];
    let image = track.observations[first].image as usize;
    let y = row_y(&mut panel, &ctx, &state, image);
    let at = egui::pos2(400.0, y);
    open_row_menu(&mut panel, &ctx, &state, at);
    let texts = painted(
        &mut panel,
        &ctx,
        &state,
        vec![egui::Event::PointerMoved(at)],
    );
    assert!(
        texts.iter().any(|t| t == super::ACCEPT_WALK_LABEL),
        "{texts:?}"
    );
    let entry = menu_entry_pos(&mut panel, &ctx, &state, super::ACCEPT_WALK_LABEL);
    let response = at_pointer(&mut panel, &ctx, &state, entry, true);
    assert_eq!(response.accept_walk, Some(first));

    let to = track.observations[first]
        .track
        .as_ref()
        .and_then(|m| m.walked_to)
        .expect("a walked pixel");
    let before = versions(&state, id);
    state
        .accept_bench_walk(id, &label, first)
        .expect("a walk to accept");
    assert_eq!(versions(&state, id), before + 1);
    let accepted = state.bench_track(id, &label).expect("on the bench");
    let row = &accepted.observations[first];
    assert!(row.pinned, "an accepted walk is a hand placement");
    let m = row.track.as_ref().expect("a track slot");
    assert_eq!(m.keypoint, Some([to[0] as f32, to[1] as f32]));
    assert_eq!(m.walked_to, None, "the walk is spent");
    assert!(super::table::accepted_walk(row).is_none());
}

/// A second track has its own bars, so focusing it moves the boxes.
#[test]
fn a_change_of_focused_item_reseats_the_boxes() {
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
            None,
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
    for (_, heading, _) in super::table::ColumnLayout::new().headers(super::BodyMode::Edited) {
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

/// Every heading says what its column holds, and the ZNCC heading says what
/// its two numbers are.
#[test]
fn every_heading_carries_hover_text() {
    for (_, heading, tip) in super::table::ColumnLayout::new().headers(super::BodyMode::Edited) {
        assert!(!tip.is_empty(), "the {heading:?} heading has no hover text");
    }
    let tip = super::table::ZNCC_TIP;
    assert!(
        tip.contains("whole patch") && tip.contains("middle"),
        "{tip}"
    );
    assert!(tip.contains("percent"), "{tip}");
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
        texts
            .iter()
            .any(|t| t == crate::index_files::BUILD_INDEX_FILES),
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
        !texts
            .iter()
            .any(|t| t == crate::index_files::REBUILD_INDEX_FILES),
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
        texts
            .iter()
            .any(|t| t == crate::index_files::BUILD_INDEX_FILES),
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
    state.open_sift_index(id, path).expect("it opens");
    assert_eq!(
        state.sift_index_state(id),
        crate::index_files::IndexFileState::Stale
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
        texts
            .iter()
            .any(|t| t == crate::index_files::REBUILD_INDEX_FILES),
        "the rebuild entry is not in the row's menu: {texts:?}"
    );

    // Clicking it asks for the build, and asks for no search.
    let entry = menu_entry_pos(
        &mut panel,
        &ctx,
        &state,
        crate::index_files::REBUILD_INDEX_FILES,
    );
    let response = at_pointer(&mut panel, &ctx, &state, entry, true);
    assert!(response.build_index_files, "the entry started no build");
    assert_eq!(
        response.search_descriptors, None,
        "the build does not run the search when it finishes, and does not run it now"
    );
}

/// *Duplicate* is the toolbar's own way to a second patch over neighbouring
/// ground: one version, a second item on the bench, and the copy focused.
#[test]
fn duplicate_puts_a_second_item_on_the_bench_and_focuses_it() {
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
    assert_eq!(state.focused_item_label(id), Some(copy.as_str()));
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

/// Everything the header and the stage's line under it painted for `track`,
/// joined.
fn header_text(track: &sfmtool_core::bench::EditableTrack) -> String {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        ..Default::default()
    };
    crate::test_support::painted_texts(&ctx, input, |ui| {
        super::show_header(ui, "item", track, None, None);
        super::show_headline(ui, track);
    })
    .join(" | ")
}

/// A track at infinity carries the infinity mark left of its label, and a
/// track on a place does not.
#[test]
fn the_header_marks_a_track_at_infinity() {
    let bearing = header_text(&bearing_track());
    assert!(
        bearing.starts_with(crate::track_view::INFINITY),
        "the mark is not first in the header: {bearing}"
    );
    let position = header_text(&position_track());
    assert!(
        !position.contains(crate::track_view::INFINITY),
        "a track on a place carries the mark: {position}"
    );
}

/// The track's own patch sits left of the toolbar, under the header, and the
/// controls start to its right. Before a fit fuses the observations the
/// demo's track, from a reconstruction that stores no bitmaps, has none and
/// the slot is empty; after it the slot shows the consensus bitmap.
#[test]
fn the_track_s_patch_sits_left_of_the_toolbar() {
    use sfmtool_core::bench::Stage;

    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    let bitmap = |state: &AppState| {
        let track = state.bench_track(id, &label).expect("on the bench");
        let Stage::Track(payload) = &track.stage else {
            panic!("a track put on from a point is at the track stage");
        };
        payload.bitmap.is_some()
    };
    assert!(!bitmap(&state), "the demo stores bitmaps now");
    assert!(
        matches!(panel.track_patch, Some(None)),
        "a track with no bitmap drew one"
    );
    state
        .start_bench_fit(id, &label)
        .expect("a framed track with three sightings fits");
    state.finish_background_task();
    assert!(bitmap(&state), "the fit fused no bitmap");
    let painted = crate::test_support::painted_text_rects(&ctx, input(Vec::new()), |ui| {
        panel.show(ui, &state);
    });
    assert!(
        matches!(panel.track_patch, Some(Some(_))),
        "the track's patch was not uploaded"
    );
    let size = crate::track_view::body::STORED_PATCH_SIZE;
    let x_of = |text: &str| {
        painted
            .iter()
            .find(|p| p.text == text)
            .unwrap_or_else(|| panic!("{text:?} was not painted"))
            .rect
            .min
            .x
    };
    let label_x = x_of(&label);
    assert!(
        x_of("Fit") >= label_x + size,
        "the toolbar overlaps the patch"
    );
    assert!(
        x_of("Crop") < label_x + size,
        "the table's first heading moved right with the toolbar"
    );
}

/// At the cluster stage the slot shows the template once one is cut, and is
/// an empty frame before that.
#[test]
fn the_cluster_stage_shows_its_template_in_the_slot() {
    use sfmtool_core::bench::Stage;

    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    run_frame(&mut panel, &ctx, &state);
    let track = state.bench_track(id, &label).expect("on the bench");
    let Stage::Cluster(payload) = &track.stage else {
        panic!("the track moved to the cluster stage");
    };
    assert_eq!(
        payload.template.is_some(),
        matches!(panel.track_patch, Some(Some(_))),
        "the slot does not follow whether a template is cut"
    );
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);
    let track = state.bench_track(id, &label).expect("on the bench");
    let Stage::Cluster(payload) = &track.stage else {
        panic!("still the cluster stage");
    };
    assert!(payload.template.is_some(), "the evaluation cut no template");
    assert!(
        matches!(panel.track_patch, Some(Some(_))),
        "the template is not shown"
    );
}

/// A patch bitmap of one, three or four channels becomes an opaque picture of
/// its colours, and an all-zero bitmap, which is how no patch is written,
/// becomes none.
#[test]
fn a_stored_patch_image_reads_any_channel_count() {
    use ndarray::Array3;

    let image = |channels: usize, fill: &[u8]| {
        let bitmap = Array3::from_shape_fn((2, 3, channels), |(_, _, c)| fill[c]);
        super::patch::stored_patch_image(bitmap.view())
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
        "the confidence channel is not dropped for an opaque alpha"
    );
    assert!(image(4, &[0, 0, 0, 0]).is_none(), "an empty patch drew");
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
    let current = crate::bench::live::Evaluation::Current;
    let cells = super::measurements(&walked, StageKind::Track, &current);
    assert_eq!(cells[4], "walked 19 grid px, kept at seed");
    // With the ZNCC the fit scored at the walked peak, where it scored one.
    let mut scored = walked.clone();
    let slot = scored.track.as_mut().expect("a track slot");
    slot.walked_zncc = Some(0.873);
    slot.walked_zncc_middle = Some(0.412);
    assert_eq!(
        super::measurements(&scored, StageKind::Track, &current)[4],
        "walked 19 grid px (ZNCC 87% / 41% there), kept at seed"
    );

    // The same row without the flag is the ordinary scored row.
    let mut moved = walked.clone();
    moved.track.as_mut().expect("a track slot").walked_px = None;
    assert_eq!(
        super::measurements(&moved, StageKind::Track, &current)[4],
        "localized"
    );
}

/// The ZNCC cell prints the whole-patch reading, then the middle one, at both
/// stages; a row with no middle reading beside its ZNCC says so with `-`.
#[test]
fn the_zncc_cell_shows_the_whole_and_the_middle_reading() {
    use sfmtool_core::bench::{ClusterMeasurement, Observation, Provenance, TrackMeasurement};

    let current = crate::bench::live::Evaluation::Current;
    let mut cluster = ClusterMeasurement::from_seed([10.0, 12.0], [[1.0, 0.0], [0.0, 1.0]]);
    cluster.zncc = Some(0.923);
    cluster.zncc_middle = Some(0.614);
    let mut row = Observation {
        image: 0,
        provenance: Provenance::Origin,
        verdict: Verdict::In,
        pinned: false,
        cluster: Some(cluster),
        track: Some(TrackMeasurement {
            keypoint: Some([10.0, 12.0]),
            zncc: Some(0.95),
            zncc_middle: Some(0.2),
            ..TrackMeasurement::default()
        }),
    };
    assert_eq!(
        super::measurements(&row, StageKind::Cluster, &current)[0],
        "92% whole\n61% mid"
    );
    assert_eq!(
        super::measurements(&row, StageKind::Track, &current)[0],
        "95% whole\n20% mid"
    );
    // A committed track read back carries the stored ZNCC and no middle.
    row.track.as_mut().expect("a track slot").zncc_middle = None;
    assert_eq!(
        super::measurements(&row, StageKind::Track, &current)[0],
        "95% whole\n- mid"
    );
    row.track.as_mut().expect("a track slot").zncc = None;
    assert_eq!(
        super::measurements(&row, StageKind::Track, &current)[0],
        "-"
    );
}

#[test]
fn a_typed_percent_is_read_as_the_fraction_it_stands_for() {
    use super::parse_percent;
    assert_eq!(parse_percent("70"), Some(0.7));
    assert_eq!(parse_percent(" 85 % "), Some(0.85));
    assert_eq!(parse_percent("abc"), None);
}

#[test]
fn zncc_text_formats_both_readings() {
    use super::zncc_sentence;
    use super::zncc_text;
    assert_eq!(zncc_text(Some(0.918), Some(0.607)), "92% whole\n61% mid");
    assert_eq!(zncc_text(Some(0.5), None), "50% whole\n- mid");
    assert_eq!(zncc_text(Some(f64::NAN), Some(0.3)), "NaN whole\n30% mid");
    assert_eq!(zncc_text(Some(-0.35), Some(1.0)), "-35% whole\n100% mid");
    assert_eq!(zncc_text(None, Some(0.3)), "-");
    assert_eq!(zncc_sentence(Some(0.918), Some(0.607)), "92% / 61%");
    assert_eq!(zncc_sentence(Some(0.918), None), "92% / -");
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

/// *Fit* and the *Stage* toggle are greyed by what the track is missing: both
/// ask core's own half of their step's validation, so a button that cannot work
/// is not offered and the sentence a person reads is the one the step would
/// have refused with.
#[test]
fn the_photometric_entries_grey_with_their_own_sentence_on_a_frameless_track() {
    let track = frameless_bearing_track();
    let refusals = super::photometric_refusals(None, &track, StageKind::Cluster);
    for (what, refusal) in [("Fit", &refusals.fit), ("Stage", &refusals.stage)] {
        let why = refusal
            .as_deref()
            .unwrap_or_else(|| panic!("{what} should be greyed on a track with no patch"));
        assert!(
            why.contains("no patch"),
            "{what} is greyed with {why:?}, which does not name what is missing"
        );
    }

    // And a whole track offers both.
    let (state, id) = state();
    let mut state = state;
    let label = state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    let whole = state.bench_track(id, &label).expect("on the bench");
    let refusals = super::photometric_refusals(None, whole, StageKind::Cluster);
    assert!(refusals.fit.is_none(), "{:?}", refusals.fit);
    assert!(refusals.stage.is_none(), "{:?}", refusals.stage);

    // Busy wins over everything, as it did before.
    let refusals = super::photometric_refusals(Some("Busy."), whole, StageKind::Cluster);
    assert_eq!(refusals.fit.as_deref(), Some("Busy."));
    assert_eq!(refusals.stage.as_deref(), Some("Busy."));
}

/// The ZNCC grid's colours run red, yellow, green, from 50 to 100. A cell
/// with no reading has no colour.
#[test]
fn grid_cells_run_from_red_to_green() {
    use super::zncc_cell_color;
    let red = egui::Color32::from_rgb(220, 0, 40);
    let yellow = egui::Color32::from_rgb(220, 200, 40);
    let green = egui::Color32::from_rgb(0, 200, 40);
    assert_eq!(zncc_cell_color(0.2), Some(red));
    assert_eq!(zncc_cell_color(0.5), Some(red));
    assert_eq!(zncc_cell_color(0.75), Some(yellow));
    assert_eq!(zncc_cell_color(1.0), Some(green));
    assert_eq!(zncc_cell_color(f64::NAN), None);
}

#[test]
fn a_grid_s_hover_text_holds_its_nine_numbers() {
    use super::table::{grid_numbers, GridKind};
    let grid = [[0.92, 0.5, f64::NAN], [1.0, -0.2, 0.33], [0.0, 0.07, 0.8]];
    assert_eq!(
        grid_numbers(&grid, GridKind::Zncc),
        "   92    50     -
  100   -20    33
    0     7    80"
    );
    let radii = [[0.37, 1.0, 1.44], [2.0, 2.26, 3.0], [f64::NAN, 0.04, 3.0]];
    assert_eq!(
        grid_numbers(&radii, GridKind::SelfSimilarity),
        "  0.4   1.0   1.4
  2.0   2.3    3+
    -   0.0    3+"
    );
}

/// The self-similarity cell prints the whole and the middle radius to one
/// decimal, with `3+` for the largest radius the reading searches.
#[test]
fn the_self_similarity_cell_shows_the_whole_and_the_middle_radius() {
    use super::self_similarity_text;
    assert_eq!(
        self_similarity_text(Some(0.37), Some(1.44)),
        "0.4 px whole\n1.4 px mid"
    );
    assert_eq!(
        self_similarity_text(Some(3.0), Some(2.26)),
        "3+ px whole\n2.3 px mid"
    );
    assert_eq!(
        self_similarity_text(Some(1.0), Some(2.96)),
        "1.0 px whole\n3.0 px mid"
    );
    assert_eq!(self_similarity_text(Some(3.0), None), "3+ px whole\n- mid");
    assert_eq!(self_similarity_text(None, Some(1.0)), "-");

    // At both stages, from the fields the measurement carries.
    use sfmtool_core::bench::{ClusterMeasurement, Observation, Provenance, TrackMeasurement};
    let current = crate::bench::live::Evaluation::Current;
    let mut cluster = ClusterMeasurement::from_seed([10.0, 12.0], [[1.0, 0.0], [0.0, 1.0]]);
    cluster.zncc_self_similarity_radius = Some(3.0);
    cluster.zncc_self_similarity_radius_middle = Some(1.04);
    let row = Observation {
        image: 0,
        provenance: Provenance::Origin,
        verdict: Verdict::In,
        pinned: false,
        cluster: Some(cluster),
        track: Some(TrackMeasurement {
            keypoint: Some([10.0, 12.0]),
            zncc: Some(0.9),
            zncc_self_similarity_radius: Some(0.37),
            zncc_self_similarity_radius_middle: Some(1.44),
            ..TrackMeasurement::default()
        }),
    };
    assert_eq!(
        super::measurements(&row, StageKind::Cluster, &current)[3],
        "3+ px whole\n1.0 px mid"
    );
    assert_eq!(
        super::measurements(&row, StageKind::Track, &current)[3],
        "0.4 px whole\n1.4 px mid"
    );
}

/// The self-similarity grid runs green under 1, yellow from 1 to 2, orange
/// from 2 to under the largest radius and red at it; a cell with no reading
/// has no colour.
#[test]
fn self_similarity_cells_run_from_green_to_red() {
    use super::self_similarity_cell_color;
    let green = egui::Color32::from_rgb(0, 200, 40);
    let yellow = egui::Color32::from_rgb(220, 200, 40);
    let orange = egui::Color32::from_rgb(220, 100, 40);
    let red = egui::Color32::from_rgb(220, 0, 40);
    assert_eq!(self_similarity_cell_color(0.0), Some(green));
    assert_eq!(self_similarity_cell_color(0.99), Some(green));
    assert_eq!(self_similarity_cell_color(1.0), Some(yellow));
    assert_eq!(self_similarity_cell_color(1.99), Some(yellow));
    assert_eq!(self_similarity_cell_color(2.0), Some(orange));
    assert_eq!(self_similarity_cell_color(2.99), Some(orange));
    assert_eq!(self_similarity_cell_color(3.0), Some(red));
    assert_eq!(self_similarity_cell_color(f64::NAN), None);
}

/// A self-similarity cell draws a line along its slide where the slide is at
/// least half a unit long, and nothing where it is shorter or absent.
#[test]
fn a_self_similarity_cell_marks_its_slide() {
    use super::table::{slide_mark, CellMark};
    match slide_mark([0.9, 0.0]) {
        CellMark::Line(half) => {
            assert!(half.y.abs() < 1e-6 && half.x.abs() > 3.0, "{half:?}");
        }
        other => panic!("expected a line, got {other:?}"),
    }
    match slide_mark([0.0, -0.5]) {
        CellMark::Line(half) => assert!(half.x.abs() < 1e-6 && half.y.abs() > 3.0),
        other => panic!("expected a line, got {other:?}"),
    }
    assert_eq!(slide_mark([0.3, 0.3]), CellMark::Nothing);
    assert_eq!(slide_mark([0.0, 0.0]), CellMark::Nothing);
    assert_eq!(slide_mark([f64::NAN, 0.0]), CellMark::Nothing);
}

/// The self-similarity heading says what its numbers and grid are, and which
/// bar judges them.
#[test]
fn the_self_similarity_heading_says_what_it_shows() {
    let headings = super::table::ColumnLayout::new().headers(super::BodyMode::Edited);
    assert!(headings
        .iter()
        .any(|&(_, heading, _)| heading == "Self-similarity"));
    let tip = super::table::SELF_SIMILARITY_TIP;
    assert!(tip.contains("middle") && tip.contains("3+"), "{tip}");
    assert!(tip.contains(super::MAX_SELF_SIMILARITY_LABEL), "{tip}");
}

// ---- The self-similarity surface plot ----------------------------------------

/// A bowl `1 - k (dx² + dy²)` over the `7 × 7` square.
fn bowl(k: f64) -> Vec<f64> {
    (0..49)
        .map(|i| {
            let (dx, dy) = ((i % 7) as f64 - 3.0, (i / 7) as f64 - 3.0);
            1.0 - k * (dx * dx + dy * dy)
        })
        .collect()
}

#[test]
fn the_surface_plot_passes_through_the_measured_shifts() {
    use super::surface_plot::SurfacePlot;
    let surface = bowl(0.02);
    let plot = SurfacePlot::new(&surface, 0.05).expect("a textured surface");
    let step = (plot.n - 1) / 6;
    // The picture covers the whole square, corners included, as the radius
    // does.
    assert!(plot.values.iter().all(|v| v.is_finite()));
    for (i, &z) in surface.iter().enumerate() {
        let (x, y) = ((i % 7) * step, (i / 7) * step);
        assert!(
            (plot.values[y * plot.n + x] - z).abs() < 1e-9,
            "shift {i}: {} against {z}",
            plot.values[y * plot.n + x]
        );
    }
    // The shifts inside the contour at 0.95 are those with 0.02 d² <= 0.05,
    // d² <= 2.5: the four axis neighbours and the four diagonals.
    assert_eq!(plot.inside.len(), 8, "{:?}", plot.inside);
}

#[test]
fn a_bowl_s_contour_is_the_circle_at_its_level() {
    use super::surface_plot::SurfacePlot;
    let plot = SurfacePlot::new(&bowl(0.02), 0.05).expect("a textured surface");
    let contour = plot.contour();
    assert!(!contour.is_empty());
    // 1 - 0.02 d² = 0.95 at d = sqrt(2.5) px of shift, in a picture 6 px of
    // shift across.
    let expected = 2.5f32.sqrt() / 6.0;
    for [a, b] in contour {
        for p in [a, b] {
            let d = ((p[0] - 0.5).powi(2) + (p[1] - 0.5).powi(2)).sqrt();
            assert!(
                (d - expected).abs() < 0.02,
                "a contour point {d} from the centre"
            );
        }
    }
}

#[test]
fn the_surface_colours_jump_at_the_contour() {
    use super::surface_plot::surface_color;
    let below = surface_color(0.9499, 0.95);
    let above = surface_color(0.9501, 0.95);
    // Rec.601 luma, 0 to 255.
    let lightness = |c: egui::Color32| {
        0.299 * f64::from(c.r()) + 0.587 * f64::from(c.g()) + 0.114 * f64::from(c.b())
    };
    assert!(
        lightness(above) > lightness(below) + 80.0,
        "{below:?} to {above:?} is no jump"
    );
    // Below the level the ramp is muted and rises towards it.
    assert!(lightness(surface_color(0.5, 0.95)) < lightness(below));
    // Above it the colour lightens towards 1.
    assert!(lightness(surface_color(1.0, 0.95)) > lightness(above));
}

#[test]
fn a_surface_with_nothing_to_draw_has_no_plot() {
    use super::surface_plot::SurfacePlot;
    assert!(SurfacePlot::new(&[f64::NAN; 49], 0.05).is_none());
    assert!(SurfacePlot::new(&bowl(0.02)[..48], 0.05).is_none());
    assert!(SurfacePlot::new(&bowl(0.02), f64::INFINITY).is_none());
}

#[test]
fn a_ridge_s_contour_runs_to_the_edge_of_the_square() {
    use super::surface_plot::SurfacePlot;
    // A ridge along x: the ZNCC falls only across it.
    let surface: Vec<f64> = (0..49)
        .map(|i| {
            let (dx, dy) = ((i % 7) as f64 - 3.0, (i / 7) as f64 - 3.0);
            1.0 - 0.001 * dx * dx - 0.1 * dy * dy
        })
        .collect();
    let plot = SurfacePlot::new(&surface, 0.05).expect("a textured surface");
    assert!(plot.inside.contains(&[3, 0]) && plot.inside.contains(&[-3, 0]));
    assert!(!plot.inside.iter().any(|&[_, dy]| dy != 0));
}

/// The projection error cell prints the error in pixels over the same
/// residual as an angle in degrees, and says `-` for either that is not there.
#[test]
fn the_projection_error_cell_shows_pixels_and_degrees() {
    use super::projection_error_text;
    assert_eq!(
        projection_error_text(Some(0.654), Some(0.081)),
        "0.65 px\n0.08\u{b0}"
    );
    assert_eq!(projection_error_text(Some(1.5), None), "1.50 px\n-");
    assert_eq!(projection_error_text(None, None), "-");

    // Before the track is triangulated there is no point, so the error is
    // measured to the patch's centre.
    use sfmtool_core::bench::{Observation, Provenance, TrackMeasurement};
    let current = crate::bench::live::Evaluation::Current;
    let mut row = Observation {
        image: 0,
        provenance: Provenance::Origin,
        verdict: Verdict::In,
        pinned: false,
        cluster: None,
        track: Some(TrackMeasurement {
            keypoint: Some([10.0, 12.0]),
            projection_offset_px: Some(2.25),
            ..TrackMeasurement::default()
        }),
    };
    assert_eq!(
        super::measurements(&row, StageKind::Track, &current)[2],
        "2.25 px\n-"
    );
    // Once there is a point, the error to it, with its angle.
    let slot = row.track.as_mut().expect("a track slot");
    slot.reprojection_error = Some(0.5);
    slot.ray_angle_deg = Some(0.07);
    assert_eq!(
        super::measurements(&row, StageKind::Track, &current)[2],
        "0.50 px\n0.07\u{b0}"
    );
}

/// A row's *Keep* switch turns a kept row out, and the whole cell takes the
/// click rather than the row behind it.
#[test]
fn clicking_the_keep_switch_turns_a_kept_row_out() {
    let (state, _, _, mut panel, ctx) = on_the_bench();
    assert_eq!(panel.rows()[1].verdict, Verdict::In);
    let y = row_y(&mut panel, &ctx, &state, 1);
    // Left of the switch itself, still in its cell.
    let x = super::table::ColumnLayout::new().keep_x() + 1.0;
    let response = at_pointer(&mut panel, &ctx, &state, egui::pos2(x, y + 8.0), true);
    assert_eq!(response.set_verdict, Some((1, Verdict::Out)));
    assert_eq!(
        response.pick_row, None,
        "the row behind the switch took the click"
    );
}

/// A verdict set by hand is handed back to the thresholds from the switch's
/// own menu and from the row's.
#[test]
fn a_pinned_row_offers_to_unpin_from_the_switch_and_from_the_row() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state
        .set_bench_verdict(id, &label, 0, Verdict::Out)
        .expect("observation 0 exists");
    run_frame(&mut panel, &ctx, &state);
    assert!(panel.rows()[0].pinned);

    let y = row_y(&mut panel, &ctx, &state, 0);
    let switch = super::table::ColumnLayout::new().keep_x() + 12.0;
    for x in [switch, 400.0] {
        let at = egui::pos2(x, y + 8.0);
        open_row_menu(&mut panel, &ctx, &state, at);
        let entry = menu_entry_pos(&mut panel, &ctx, &state, super::table::UNPIN_LABEL);
        let response = at_pointer(&mut panel, &ctx, &state, entry, true);
        assert_eq!(response.unpin_verdicts, Some(vec![0]), "from x = {x}");
    }

    let before = versions(&state, id);
    state
        .unpin_bench_verdicts(id, &label, &[0])
        .expect("observation 0 exists");
    assert_eq!(versions(&state, id), before + 1);
    let track = state.bench_track(id, &label).expect("on the bench");
    assert!(!track.observations[0].pinned);
    state
        .unpin_bench_verdicts(id, &label, &[0])
        .expect("observation 0 exists");
    assert_eq!(
        versions(&state, id),
        before + 1,
        "a second unpin is no effect"
    );
}

/// The header prints the ID of the point the track came from once: as the
/// label when the label is that ID, and beside a renamed label otherwise. A
/// point that is gone from the version is named by its old index instead.
#[test]
fn the_header_names_the_point_the_track_came_from() {
    let mut track = bearing_track();
    track.origin = Some(sfmtool_core::bench::Origin {
        version: 1,
        point: 12,
    });
    let ctx = egui::Context::default();
    let header = |label: &str, id: Option<&str>| {
        crate::test_support::painted_texts(&ctx, input(Vec::new()), |ui| {
            super::show_header(ui, label, &track, id, None);
        })
    };

    let same = header("pt3d_ab_12", Some("pt3d_ab_12"));
    assert_eq!(same.iter().filter(|t| t.contains("pt3d_ab_12")).count(), 1);
    assert!(!same.iter().any(|t| t.contains("from point")), "{same:?}");

    let renamed = header("my track", Some("pt3d_ab_12"));
    assert!(renamed.iter().any(|t| t == "pt3d_ab_12"), "{renamed:?}");

    let gone = header("my track", None);
    assert!(gone.iter().any(|t| t == "\u{b7} from point 12"), "{gone:?}");
}

/// A track put on the bench from a point knows its point's ID, which is what
/// the header's copy button copies.
#[test]
fn a_track_from_a_point_resolves_the_id_the_header_copies() {
    let (state, id, label, _, _) = on_the_bench();
    let node = state.node(id).expect("the node");
    let track = state.bench_track(id, &label).expect("on the bench");
    let index = state
        .resolved_origin(node, track)
        .expect("the point is live");
    assert_eq!(crate::scene::point_id(node, index as usize), label);
}

/// The pin beside the switch pins a verdict the thresholds set, as it stands,
/// and hands a pinned one back to the thresholds.
#[test]
fn the_pin_pins_a_verdict_and_unpins_a_pinned_one() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    // A point's rows arrive pinned; this one is handed to the bars first.
    state
        .unpin_bench_verdicts(id, &label, &[1])
        .expect("observation 1 exists");
    run_frame(&mut panel, &ctx, &state);
    let y = row_y(&mut panel, &ctx, &state, 1);
    let pin = egui::pos2(super::table::ColumnLayout::new().keep_x() + 52.0, y + 8.0);
    let response = at_pointer(&mut panel, &ctx, &state, pin, true);
    assert_eq!(response.set_verdict, Some((1, Verdict::In)));
    assert_eq!(
        response.pick_row, None,
        "the row behind the pin took the click"
    );

    state
        .set_bench_verdict(id, &label, 1, Verdict::In)
        .expect("observation 1 exists");
    let response = at_pointer(&mut panel, &ctx, &state, pin, true);
    assert_eq!(response.unpin_verdicts, Some(vec![1]));
    assert_eq!(response.set_verdict, None);
}

/// The Name column elides a long name in its middle, and hovering it shows
/// the name whole.
#[test]
fn hovering_a_name_shows_it_whole() {
    let (state, id, _, mut panel, ctx) = on_the_bench();
    let node = state.node(id).expect("the node");
    let name = node.recon().image_table.images[1].name.clone();
    let y = row_y(&mut panel, &ctx, &state, 1);
    let at = egui::pos2(190.0, y + 8.0);
    let response = at_pointer(&mut panel, &ctx, &state, at, false);
    assert_eq!(response.hovered_image, Some(1), "the row lost its hover");
    // A tooltip shows after the pointer has rested, so a few more frames.
    let mut texts = Vec::new();
    for _ in 0..4 {
        texts = painted(
            &mut panel,
            &ctx,
            &state,
            vec![egui::Event::PointerMoved(at)],
        );
    }
    assert!(
        texts.iter().any(|t| t == &name),
        "{name:?} not in {texts:?}"
    );
}

/// The hover view of one observation's tile on the focused item, beside the
/// tile itself, both rendered from the node's cached photograph.
fn tile_and_context(
    state: &AppState,
    id: ReconId,
    label: &str,
    observation: usize,
) -> (egui::ColorImage, super::tile::TileContext) {
    let track = state.bench_track(id, label).expect("on the bench").clone();
    let row = &track.observations[observation];
    let src = state
        .cached_photograph(id, row.image as usize)
        .expect("a cached photograph");
    let recon = state.node(id).expect("loaded").recon();
    let tile = super::tile::image(recon, &track, observation, &src).expect("a tile");
    let context = super::tile::context(recon, &track, observation, &src).expect("a context");
    (tile, context)
}

/// The largest difference in any channel between `tile` and the texels of
/// `context`'s picture inside its patch box, which must be the tile's size.
fn largest_difference_in_box(tile: &egui::ColorImage, context: &super::tile::TileContext) -> u8 {
    let [w, h] = tile.size;
    let min = context.patch_box.min;
    assert_eq!(
        (context.patch_box.width(), context.patch_box.height()),
        (w as f32, h as f32),
        "the patch box is not the tile's size in texels"
    );
    let side = context.image.size[0];
    let mut worst = 0u8;
    for y in 0..h {
        for x in 0..w {
            let a = tile.pixels[y * w + x].to_array();
            let (cx, cy) = (min.x as usize + x, min.y as usize + y);
            let b = context.image.pixels[cy * side + cx].to_array();
            for c in 0..3 {
                worst = worst.max(a[c].abs_diff(b[c]));
            }
        }
    }
    worst
}

/// A tile's hover view is the same picture over three times the patch's
/// width, at the tile's own sampling, at either stage: the picture is three
/// tiles across, the patch box is its middle third, the keypoint sits at the
/// box's centre, and the texels inside the box are the tile's.
#[test]
fn a_tile_s_hover_view_holds_the_tile_in_its_middle_at_either_stage() {
    let (mut state, id, label, _, _) = on_the_bench();
    let k = super::tile::CONTEXT_FACTOR as f32;
    let check = |state: &AppState, stage: &str| {
        let observations = state
            .bench_track(id, &label)
            .expect("on the bench")
            .observations
            .len();
        for observation in 0..observations {
            let (tile, context) = tile_and_context(state, id, &label, observation);
            let side = context.image.size[0] as f32;
            assert_eq!(
                context.image.size,
                [tile.size[0] * 3, tile.size[1] * 3],
                "{stage}: the picture is not three tiles across"
            );
            assert_eq!(
                context.patch_box,
                egui::Rect::from_min_size(
                    egui::pos2(side / k, side / k),
                    egui::vec2(side / k, side / k)
                ),
                "{stage}: the patch box is not the middle third"
            );
            let keypoint = context.keypoint.expect("the keypoint meets the patch");
            assert!(
                keypoint.distance(context.patch_box.center()) < 1e-3,
                "{stage}: the keypoint {keypoint:?} is not the box's centre"
            );
            // The same sample positions, computed along a different sum, so a
            // bilinear read may round one level apart.
            let worst = largest_difference_in_box(&tile, &context);
            assert!(
                worst <= 2,
                "{stage}: observation {observation}'s box differs from its tile by {worst}"
            );
        }
    };
    check(&state, "track stage");

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    check(&state, "cluster stage");
}

/// At the track stage the hover view marks where the track's point projects,
/// and it is placed consistently with the row's reprojection error: the
/// distance it states is the row's error, and its distance from the keypoint in
/// the picture, turned into the photograph's pixels by the patch's own width in
/// both, is that error too. A row sitting on its projection has the mark on
/// the keypoint.
#[test]
fn a_hover_view_marks_the_projection_as_far_off_as_the_row_s_error() {
    use sfmtool_core::bench::Stage;

    let (mut state, id, label, _, _) = on_the_bench();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let Stage::Track(payload) = &track.stage else {
        panic!("a track put on from a point is at the track stage");
    };
    let frame = payload.placement.clone().expect("a patch");
    let position = payload.position.expect("a triangulated point");

    // A sighting of the point in a fourth image, placed a few pixels off
    // where the point projects.
    let image = 3;
    let recon = state.node(id).expect("loaded").recon();
    let (camera, pose) =
        crate::bench::geometry::view_of(&recon.image_table, image).expect("a view");
    let projected = crate::bench::geometry::project(&camera, &pose, position.coords, frame.w)
        .expect("the demo cameras see every point");
    let off = [4.0, -3.0];
    state
        .add_bench_observation(
            &label,
            ImageRef::new(id, image),
            &crate::bench::Seed::Pixel {
                pixel: [projected[0] + off[0], projected[1] + off[1]],
                radius_px: None,
            },
        )
        .expect("a pixel on the sensor");
    state.settle_bench_evaluation();

    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let added = track.observations.len() - 1;
    let error = track.observations[added]
        .track
        .as_ref()
        .and_then(|m| m.reprojection_error)
        .expect("the evaluation measured the new row's error");
    assert!(
        (error - 5.0).abs() < 0.05,
        "the row's error is {error}, not 5 px"
    );

    let (_, context) = tile_and_context(&state, id, &label, added);
    assert_eq!(
        context.projection_of,
        Some(super::tile::ProjectionOf::Point)
    );
    let stated = context.projection_px.expect("a projection distance");
    assert!(
        (stated - error).abs() < 1e-6,
        "the hover view states {stated} px, the row {error} px"
    );
    let keypoint = context.keypoint.expect("a keypoint");
    let mark = context.projection.expect("a projection mark");
    // The picture's own map back into the photograph: the widened frame the
    // picture was rendered through, a texel read as `(s, t)` on it, and that
    // place on the plane projected. The patch is seen at a slant here, so a
    // single pixels-per-texel scale would not do; the map is what scales an
    // offset in the picture into the photograph's pixels.
    let keypoint_px = [projected[0] + off[0], projected[1] + off[1]];
    let mut wide = frame
        .anchored_at_keypoint(&camera, &pose, keypoint_px)
        .expect("the keypoint meets the patch");
    wide.half_extent = wide
        .half_extent
        .map(|h| h * f64::from(super::tile::CONTEXT_FACTOR));
    let side = f64::from(context.image.size[0] as u32);
    let to_photograph = |at: egui::Pos2| {
        let s = 2.0 * f64::from(at.x) / side - 1.0;
        let t = 1.0 - 2.0 * f64::from(at.y) / side;
        let (xyz, w) = wide.corner_homogeneous(s, t);
        crate::bench::geometry::project(&camera, &pose, xyz, w).expect("on the plane")
    };
    let distance = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    assert!(
        distance(to_photograph(keypoint), keypoint_px) < 0.01,
        "the keypoint in the picture is not the observation's pixel"
    );
    assert!(
        distance(to_photograph(mark), projected) < 0.01,
        "the mark is not where the point projects"
    );
    let measured = distance(to_photograph(keypoint), to_photograph(mark));
    assert!(
        (measured - error).abs() < 0.01,
        "the mark sits {measured} px from the keypoint in the photograph's pixels, \
         the row says {error}"
    );
    // And it is a visible distance in the picture, not a rounding.
    assert!(
        keypoint.distance(mark) > 1.0,
        "a 5 px error drew the mark {} texels off",
        keypoint.distance(mark)
    );

    // A row the fixture put at its exact projection has the mark on the dot.
    let (_, context) = tile_and_context(&state, id, &label, 0);
    let keypoint = context.keypoint.expect("a keypoint");
    let mark = context.projection.expect("a projection mark");
    assert!(
        keypoint.distance(mark) < 0.05,
        "a row on its projection has the mark {mark:?} off the keypoint {keypoint:?}"
    );
    assert!(context.projection_px.expect("a distance") < 0.01);
}

/// At the cluster stage there is no point, and the table's projection error
/// column is empty, so the hover view draws the picture, the box and the
/// keypoint with no projection mark, and its caption says why.
#[test]
fn a_cluster_stage_hover_view_has_no_projection() {
    let (mut state, id, label, _, _) = on_the_bench();
    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    let (_, context) = tile_and_context(&state, id, &label, 0);
    assert_eq!(context.projection, None);
    assert_eq!(context.projection_px, None);
    assert_eq!(context.projection_of, None);
    let caption = super::tile::context_caption(&context);
    assert!(
        caption.contains("no point to project"),
        "the caption does not say why there is no mark: {caption}"
    );
}

/// Resting the pointer on a row's tile shows the hover view, rendered for that
/// row alone, while the row keeps its hover; and a click on the tile is still
/// the row's click.
#[test]
fn hovering_a_tile_shows_it_in_context_and_keeps_the_row() {
    let (state, _, _, mut panel, ctx) = on_the_bench();
    // A tooltip waits out `tooltip_delay` before it shows, which a headless
    // frame has no wall clock to pass.
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let y = row_y(&mut panel, &ctx, &state, 1);
    // Inside the tile: the tile column's left edge plus half a tile, and far
    // enough below the top of the row to be within the tile's height.
    let x = super::table::ColumnLayout::new().tile_x() + super::table::TILE_SIZE / 2.0;
    let at = egui::pos2(x + 8.0, y + 20.0);
    let response = at_pointer(&mut panel, &ctx, &state, at, false);
    assert_eq!(response.hovered_image, Some(1), "the row lost its hover");
    // egui shows a tooltip once the pointer has come to rest, so a few frames
    // with no movement in them.
    for _ in 0..12 {
        run_frame(&mut panel, &ctx, &state);
    }
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(
        texts.iter().any(|t| t.starts_with("The patch, boxed")),
        "no hover view caption in {texts:?}"
    );
    assert!(
        texts
            .iter()
            .any(|t| t.contains("where the track's point projects")),
        "the caption does not name the projection: {texts:?}"
    );
    let rendered: Vec<usize> = panel.contexts.keys().copied().collect();
    assert_eq!(
        rendered,
        vec![1],
        "hover views rendered for rows not hovered"
    );

    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    assert_eq!(
        response.pick_row,
        Some((1, false)),
        "a click on the tile did not pick the row"
    );
}

// ── The bars' colours and the Keep cell's proposal ──────────────────────

/// Set the boxes the way a drag in progress holds them: from the focused
/// track's bars with `change` applied, and held there for the next frame, which
/// is drawn.
fn with_boxes(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &AppState,
    id: ReconId,
    label: &str,
    change: impl FnOnce(&mut Thresholds),
) {
    let mut bars = state
        .bench_track(id, label)
        .expect("on the bench")
        .thresholds
        .clone();
    change(&mut bars);
    panel.thresholds = bars;
    panel.sliding = true;
    run_frame(panel, ctx, state);
}

/// [`POINT`] on the bench with its first evaluation landed, so every row has
/// readings for the bars to judge.
fn measured_on_the_bench() -> (AppState, ReconId, String, TrackBody, egui::Context) {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state.settle_bench_evaluation();
    run_frame(&mut panel, &ctx, &state);
    (state, id, label, panel, ctx)
}

/// The whole and the middle ZNCC of observation `i` of the focused item.
fn zncc_readings(state: &AppState, id: ReconId, label: &str, i: usize) -> (f64, f64) {
    let track = state.bench_track(id, label).expect("on the bench");
    let m = track.observations[i]
        .track
        .as_ref()
        .expect("a track-stage reading");
    (
        m.zncc.expect("a whole ZNCC"),
        m.zncc_middle.expect("a middle ZNCC"),
    )
}

/// Each line of the ZNCC cell takes the colour of its own bar: the whole
/// reading can pass while the middle fails, and the reverse.
#[test]
fn each_line_of_the_zncc_cell_is_judged_by_its_own_bar() {
    use sfmtool_core::bench::BarCheck;

    let (state, id, label, mut panel, ctx) = measured_on_the_bench();
    let (whole, middle) = zncc_readings(&state, id, &label, 0);
    assert!(middle > 0.02, "the fixture's middle ZNCC is {middle}");

    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = whole - 0.01;
        bars.min_zncc_middle = middle + 0.01;
    });
    assert_eq!(
        panel.rows()[0].checks[0],
        [BarCheck::Pass, BarCheck::Fail],
        "whole over its bar, middle under its own"
    );

    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = whole + 0.01;
        bars.min_zncc_middle = middle - 0.01;
    });
    assert_eq!(
        panel.rows()[0].checks[0],
        [BarCheck::Fail, BarCheck::Pass],
        "whole under its bar, middle over its own"
    );
}

/// The readings no bar judges stay in the plain colour: the projection error,
/// the self-similarity middle radius, the status, the middle ZNCC while its
/// bar is off, a reading that is missing, and every reading of a row nothing
/// has measured or whose evaluation was refused.
#[test]
fn readings_no_bar_judges_are_drawn_plain() {
    use sfmtool_core::bench::{bar_checks, BarCheck, Observation, Provenance, TrackMeasurement};

    let (state, id, label, mut panel, ctx) = measured_on_the_bench();
    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc_middle = 0.0;
    });
    let plain = [BarCheck::NotJudged; 2];
    for row in panel.rows() {
        assert_eq!(row.checks[0][1], BarCheck::NotJudged, "mid ZNCC, bar off");
        assert_eq!(row.checks[2], plain, "Proj. err");
        assert_eq!(row.checks[3][1], BarCheck::NotJudged, "self-similarity mid");
        assert_eq!(row.checks[4], plain, "Status");
        assert_ne!(
            row.checks[0][0],
            BarCheck::NotJudged,
            "ZNCC whole is judged"
        );
    }

    // A reading that is missing (`-`) is not judged, though it clears its bar.
    let row = Observation {
        image: 0,
        provenance: Provenance::Origin,
        verdict: Verdict::In,
        pinned: false,
        cluster: None,
        track: Some(TrackMeasurement {
            keypoint: Some([10.0, 12.0]),
            zncc: Some(0.95),
            ..TrackMeasurement::default()
        }),
    };
    let judged = super::Judgement {
        checks: bar_checks(&row, StageKind::Track, &Thresholds::default()).expect("measured"),
        proposal: Verdict::In,
    };
    let current = crate::bench::live::Evaluation::Current;
    let checks = super::table::cell_checks(Some(&judged), &current);
    assert_eq!(checks[0], [BarCheck::Pass, BarCheck::NotJudged]);
    assert_eq!(checks[1][0], BarCheck::NotJudged, "no shift reading");
    assert_eq!(checks[3][0], BarCheck::NotJudged, "no radius reading");

    // A row nothing has measured, and a row whose cells print no numbers.
    assert_eq!(super::table::cell_checks(None, &current), [plain; 5]);
    let refused = crate::bench::live::Evaluation::Refused("no photographs".to_string());
    assert_eq!(
        super::table::cell_checks(Some(&judged), &refused),
        [plain; 5]
    );
}

/// Dragging a box judges the readings against the box at once, and changes
/// nothing on the track until it is let go.
#[test]
fn dragging_a_box_changes_the_judgement_and_not_the_track() {
    use sfmtool_core::bench::BarCheck;

    let (state, id, label, mut panel, ctx) = measured_on_the_bench();
    let before = state.bench_track(id, &label).expect("on the bench").clone();
    let versions_before = versions(&state, id);

    // Bars every reading clears: each row is proposed in, and says why.
    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = -1.0;
        bars.min_zncc_middle = 0.0;
        bars.max_shift_px = 1.0e6;
        bars.max_zncc_self_similarity_radius = 1.0e6;
    });
    for row in panel.rows() {
        assert_eq!(row.checks[0][0], BarCheck::Pass, "{row:?}");
        assert_eq!(row.proposal, Some(Verdict::In), "{row:?}");
        assert!(
            row.keep_hover.contains("it clears every bar"),
            "{}",
            row.keep_hover
        );
    }

    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = 1.01;
    });
    for row in panel.rows() {
        assert_eq!(row.checks[0][0], BarCheck::Fail, "{row:?}");
        assert_eq!(row.proposal, Some(Verdict::Out), "{row:?}");
        assert_eq!(row.verdict, Verdict::In, "the drag moved a verdict");
    }
    let after = state.bench_track(id, &label).expect("on the bench");
    assert!(
        std::sync::Arc::ptr_eq(&before, after),
        "the drag stepped the track"
    );
    assert_eq!(versions(&state, id), versions_before);
}

/// An unpinned row's *Keep* cell shows what applying the bars makes it, and a
/// pinned row's what unpinning it would: a switch that is on in a red cell is
/// a hand ruling against the bars, and its hover text says which bar.
#[test]
fn a_pinned_row_ruled_against_the_bars_shows_what_unpinning_would_give() {
    let (mut state, id, label, mut panel, ctx) = measured_on_the_bench();
    state
        .set_bench_verdict(id, &label, 0, Verdict::In)
        .expect("observation 0 exists");
    let (whole, _) = zncc_readings(&state, id, &label, 0);
    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = whole + 0.001;
    });

    let row = &panel.rows()[0];
    assert!(row.pinned && row.verdict == Verdict::In, "{row:?}");
    assert_eq!(row.proposal, Some(Verdict::Out), "the bars fail it");
    assert!(
        row.keep_hover.contains("ZNCC whole is under the bar"),
        "{}",
        row.keep_hover
    );
    assert!(
        row.keep_hover.contains("Kept, set by hand."),
        "the switch's own text is kept: {}",
        row.keep_hover
    );

    // And that is what unpinning it gives.
    let bars = panel.thresholds().clone();
    state
        .apply_bench_thresholds(id, &label, &bars)
        .expect("on the bench");
    state
        .unpin_bench_verdicts(id, &label, &[0])
        .expect("observation 0 exists");
    let track = state.bench_track(id, &label).expect("on the bench");
    assert_eq!(track.observations[0].verdict, Verdict::Out);
}

/// Of two sightings in one image that both clear every bar, the one the
/// painting does not take is proposed `out`, and its hover text says another
/// sighting in that image is kept.
#[test]
fn a_row_that_clears_every_bar_but_loses_its_image_says_so() {
    use sfmtool_core::bench::BarCheck;

    let (mut state, id, label, _, _) = on_the_bench();
    let pixel = crate::bench::observation_pixel(
        &state
            .bench_track(id, &label)
            .expect("on the bench")
            .observations[1],
    )
    .expect("a placed observation");
    state
        .add_bench_observation(
            &label,
            ImageRef::new(id, 1),
            &crate::bench::Seed::Pixel {
                pixel: [f64::from(pixel[0]), f64::from(pixel[1])],
                radius_px: None,
            },
        )
        .expect("a pixel in the photograph");
    state.settle_bench_evaluation();
    let (mut panel, ctx) = settled(&state);
    // Bars every measured reading clears.
    with_boxes(&mut panel, &ctx, &state, id, &label, |bars| {
        bars.min_zncc = -1.0;
        bars.min_zncc_middle = 0.0;
        bars.max_shift_px = 1.0e6;
        bars.max_zncc_self_similarity_radius = 1.0e6;
    });

    let rows = panel.rows();
    assert_eq!(rows.len(), 4, "{rows:?}");
    let in_image_1: Vec<_> = rows.iter().filter(|row| row.image == 1).collect();
    assert!(
        in_image_1.iter().all(|row| row.proposal.is_some()
            && row.checks.iter().flatten().all(|&c| c != BarCheck::Fail)),
        "both sightings are measured and clear every bar: {in_image_1:?}"
    );
    let lost: Vec<_> = in_image_1
        .iter()
        .filter(|row| row.proposal == Some(Verdict::Out))
        .collect();
    assert_eq!(
        lost.len(),
        1,
        "one of the two keeps the image: {in_image_1:?}"
    );
    assert!(
        lost[0]
            .keep_hover
            .contains("another sighting in image 1 is kept"),
        "{}",
        lost[0].keep_hover
    );
}

/// The crop stands at the table's left edge under *Crop*, the patch tile
/// after it under *Patch*, and the *Keep* column after that, none of them
/// overlapping.
#[test]
fn the_crop_and_patch_columns_come_first_under_their_headings() {
    let cols = super::table::ColumnLayout::new();
    let size = super::table::TILE_SIZE;
    assert_eq!(cols.crop_x(), 0.0);
    assert!(
        cols.tile_x() >= cols.crop_x() + size,
        "the patch overlaps the crop"
    );
    assert!(
        cols.keep_x() >= cols.tile_x() + size,
        "the Keep column overlaps the patch"
    );
    let headers = cols.headers(super::BodyMode::Edited);
    let firsts: Vec<(f32, &str)> = headers[..3].iter().map(|&(x, h, _)| (x, h)).collect();
    assert_eq!(
        firsts,
        vec![
            (cols.crop_x(), "Crop"),
            (cols.tile_x(), "Patch"),
            (cols.keep_x(), "Keep"),
        ]
    );
}

/// The headings are drawn at the size of the cells under them.
#[test]
fn the_headings_are_as_large_as_the_cells() {
    let (state, _id, _label, mut panel, ctx) = on_the_bench();
    let painted = crate::test_support::painted_text_rects(&ctx, input(Vec::new()), |ui| {
        panel.show(ui, &state);
    });
    let height = |text: &str| {
        painted
            .iter()
            .find(|painted| painted.text == text)
            .map(|painted| painted.rect.height())
            .unwrap_or_else(|| panic!("{text:?} was not painted"))
    };
    assert_eq!(height("Name"), height("image_000.jpg"));
}

/// Where the *Keep* heading's pin is drawn: over the rows' pin column, on the
/// heading's line.
fn heading_pin_pos(panel: &mut TrackBody, ctx: &egui::Context, state: &AppState) -> egui::Pos2 {
    let keep = crate::test_support::painted_text_rects(ctx, input(Vec::new()), |ui| {
        panel.show(ui, state);
    })
    .into_iter()
    .find(|painted| painted.text == "Keep")
    .expect("the Keep heading was painted")
    .rect;
    egui::pos2(
        super::table::ColumnLayout::new().keep_x() + 52.0,
        keep.center().y,
    )
}

/// The pin in the *Keep* heading unpins every pinned row of the track in one
/// step, and once nothing is pinned a click asks to pin every row instead.
#[test]
fn the_keep_heading_pin_unpins_every_pinned_row_then_pins_them_all() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    assert!(
        panel.rows().iter().all(|row| row.pinned),
        "a point's rows arrive pinned"
    );
    let at = heading_pin_pos(&mut panel, &ctx, &state);
    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    assert_eq!(response.unpin_verdicts, Some(vec![0, 1, 2]));
    assert_eq!(response.pick_row, None);

    let before = versions(&state, id);
    state
        .unpin_bench_verdicts(id, &label, &[0, 1, 2])
        .expect("the rows exist");
    assert_eq!(versions(&state, id), before + 1, "one step, one version");
    run_frame(&mut panel, &ctx, &state);
    assert!(panel.rows().iter().all(|row| !row.pinned));

    // Nothing is pinned now: a click asks to pin every row.
    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    assert_eq!(response.unpin_verdicts, None);
    assert_eq!(response.pin_verdicts, Some(vec![0, 1, 2]));
    // And an unpin of rows none of which is pinned pushes no version.
    state
        .unpin_bench_verdicts(id, &label, &[0, 1, 2])
        .expect("the rows exist");
    assert_eq!(versions(&state, id), before + 1);
}

/// With no row pinned, the heading pin pins every row at the verdict it has
/// now, in one step with one Action Log row, and a second click unpins them
/// all again.
#[test]
fn the_keep_heading_pin_pins_every_row_as_it_stands() {
    let (mut state, id, label, mut panel, ctx) = measured_on_the_bench();
    state
        .unpin_bench_verdicts(id, &label, &[0, 1, 2])
        .expect("the rows exist");
    // A whole-ZNCC bar just over the lowest reading turns that row out and
    // leaves the rest to the other bars: a mix of `in` and `out` with nothing
    // pinned.
    let lowest = (0..3)
        .map(|i| zncc_readings(&state, id, &label, i).0)
        .fold(f64::INFINITY, f64::min);
    let mut bars = state
        .bench_track(id, &label)
        .expect("on the bench")
        .thresholds
        .clone();
    bars.min_zncc = lowest + 0.001;
    state
        .apply_bench_thresholds(id, &label, &bars)
        .expect("on the bench");
    run_frame(&mut panel, &ctx, &state);
    let verdicts: Vec<Verdict> = panel.rows().iter().map(|row| row.verdict).collect();
    assert!(verdicts.contains(&Verdict::In) && verdicts.contains(&Verdict::Out));
    assert!(panel.rows().iter().all(|row| !row.pinned));

    let at = heading_pin_pos(&mut panel, &ctx, &state);
    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    let rows = response
        .pin_verdicts
        .expect("the pin asks to pin every row");
    assert_eq!(rows, vec![0, 1, 2]);
    assert_eq!(response.unpin_verdicts, None);

    let before = versions(&state, id);
    state.action_log.clear();
    state
        .pin_bench_verdicts(id, &label, &rows)
        .expect("the rows exist");
    assert_eq!(versions(&state, id), before + 1, "one step, one version");
    let log: Vec<String> = state.action_log.entries().map(|e| e.text.clone()).collect();
    let (kept, out) = (
        verdicts.iter().filter(|&&v| v == Verdict::In).count(),
        verdicts.iter().filter(|&&v| v == Verdict::Out).count(),
    );
    assert_eq!(log.len(), 1, "{log:?}");
    assert!(
        log[0].starts_with(&format!(
            "Pinned 3 verdicts in {label}: {kept} in, {out} out"
        )),
        "{log:?}"
    );
    run_frame(&mut panel, &ctx, &state);
    assert!(panel.rows().iter().all(|row| row.pinned));
    assert_eq!(
        panel
            .rows()
            .iter()
            .map(|row| row.verdict)
            .collect::<Vec<_>>(),
        verdicts,
        "every verdict stands as it was"
    );
    // Pinning rows that are all pinned pushes nothing.
    state
        .pin_bench_verdicts(id, &label, &rows)
        .expect("the rows exist");
    assert_eq!(versions(&state, id), before + 1);

    // One undo takes the pins off again.
    state.undo(id).expect("the pin step undoes");
    let track = state.bench_track(id, &label).expect("on the bench");
    assert!(track.observations.iter().all(|o| !o.pinned));
    state.redo(id).expect("and redoes");
    run_frame(&mut panel, &ctx, &state);

    // Now every row is pinned, so the same click unpins them all.
    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    assert_eq!(response.unpin_verdicts, Some(vec![0, 1, 2]));
    assert_eq!(response.pin_verdicts, None);
}

/// The heading pin's hover text says what a click does, with the count, and
/// why it is greyed. A headless frame shows no tooltip, so the text is asked of
/// the function that writes it.
#[test]
fn the_heading_pin_hover_counts_the_pins() {
    use super::table::{heading_pin_hover, heading_pin_name};
    assert_eq!(
        heading_pin_hover(12, 20, None),
        "Unpin all 12 pinned verdicts and let the bars decide"
    );
    assert_eq!(
        heading_pin_hover(1, 20, None),
        "Unpin the 1 pinned verdict and let the bars decide"
    );
    assert_eq!(
        heading_pin_hover(0, 12, None),
        "Pin all 12 verdicts as they stand"
    );
    assert_eq!(
        heading_pin_hover(0, 1, None),
        "Pin the 1 verdict as it stands"
    );
    assert_eq!(heading_pin_hover(4, 20, Some("busy")), "busy");
    assert_eq!(heading_pin_hover(0, 20, Some("busy")), "busy");
    assert!(heading_pin_hover(0, 0, None).contains("no observations"));
    assert_eq!(heading_pin_name(3), "Unpin all");
    assert_eq!(heading_pin_name(0), "Pin all");
}

/// The header counts the pinned rows beside the kept and the turned out.
#[test]
fn the_header_counts_kept_out_and_pinned() {
    let (mut state, id, label, _, _) = on_the_bench();
    let text = header_text(state.bench_track(id, &label).expect("on the bench"));
    assert!(text.contains("3 kept · 0 out · 3 pinned"), "{text}");
    state
        .unpin_bench_verdicts(id, &label, &[1])
        .expect("observation 1 exists");
    let text = header_text(state.bench_track(id, &label).expect("on the bench"));
    assert!(text.contains(" · 2 pinned"), "{text}");
}

/// On a row that is one of several selected, the row menu's unpin acts on the
/// selection: every pinned verdict among the selected rows in one step, with
/// the count in its label. The *Keep* switch's own menu stays per row.
#[test]
fn the_row_menu_unpins_every_pinned_row_of_a_selection() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    state
        .unpin_bench_verdicts(id, &label, &[1])
        .expect("observation 1 exists");
    for (observation, extend) in [(0, false), (1, true), (2, true)] {
        state.pick_bench_observation(id, &label, observation, extend);
    }
    run_frame(&mut panel, &ctx, &state);

    let text = super::table::unpin_selection_label(2);
    assert_eq!(text, "Unpin 2 verdicts, let the thresholds decide");
    let y = row_y(&mut panel, &ctx, &state, 0);
    open_row_menu(&mut panel, &ctx, &state, egui::pos2(400.0, y + 8.0));
    let entry = menu_entry_pos(&mut panel, &ctx, &state, &text);
    let response = at_pointer(&mut panel, &ctx, &state, entry, true);
    let mut rows = response.unpin_verdicts.expect("the entry was taken");
    rows.sort_unstable();
    assert_eq!(rows, vec![0, 2], "the pinned rows of the selection");

    let switch = egui::pos2(super::table::ColumnLayout::new().keep_x() + 12.0, y + 8.0);
    open_row_menu(&mut panel, &ctx, &state, switch);
    let entry = menu_entry_pos(&mut panel, &ctx, &state, super::table::UNPIN_LABEL);
    let response = at_pointer(&mut panel, &ctx, &state, entry, true);
    assert_eq!(response.unpin_verdicts, Some(vec![0]));

    let before = versions(&state, id);
    state
        .unpin_bench_verdicts(id, &label, &[0, 2])
        .expect("the rows exist");
    assert_eq!(versions(&state, id), before + 1, "one step, one version");
    let track = state.bench_track(id, &label).expect("on the bench");
    assert!(track.observations.iter().all(|o| !o.pinned));
}

// ── The crop column ─────────────────────────────────────────────────────

/// Every row draws the crop of its photograph beside its tile, at the track
/// stage and at the cluster stage.
#[test]
fn every_row_draws_a_crop_at_either_stage() {
    let (mut state, id, label, mut panel, ctx) = on_the_bench();
    assert!(
        panel.rows().iter().all(|row| row.crop),
        "a track-stage row drew no crop: {:?}",
        panel.rows(),
    );

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    run_frame(&mut panel, &ctx, &state);
    assert!(
        panel.rows().iter().all(|row| row.crop),
        "a cluster-stage row drew no crop: {:?}",
        panel.rows(),
    );
}

/// One observation's outline, its crop and the crop's hover view, all from
/// the node's cached photograph.
fn crop_and_context(
    state: &AppState,
    id: ReconId,
    label: &str,
    observation: usize,
) -> (
    super::crop::Outline,
    super::crop::CropPicture,
    super::crop::CropPicture,
) {
    let track = state.bench_track(id, label).expect("on the bench").clone();
    let row = &track.observations[observation];
    let src = state
        .cached_photograph(id, row.image as usize)
        .expect("a cached photograph");
    let recon = state.node(id).expect("loaded").recon();
    let outline = super::crop::outline(recon, &track, observation).expect("an outline");
    let crop = super::crop::image(recon, &track, observation, &src).expect("a crop");
    let context = super::crop::context(recon, &track, observation, &src).expect("a context");
    (outline, crop, context)
}

/// The crop is square, and is the outline's bounding box widened by the margin,
/// rounded out to whole pixels and then widened evenly on its shorter side:
/// every sample of the outline lies at least the margin inside it, on the
/// longer side neither edge reaches a whole pixel further than that, and on
/// both the outline sits in the middle to within the rounding. At one texel per
/// pixel each texel is the photograph's own pixel. Checked at both stages.
#[test]
fn a_crop_holds_the_whole_outline_with_a_pixel_to_spare() {
    let (mut state, id, label, _, _) = on_the_bench();
    let margin = super::crop::CROP_MARGIN_PX;
    let check = |state: &AppState, stage: &str| {
        let track = state.bench_track(id, &label).expect("on the bench").clone();
        for observation in 0..track.observations.len() {
            let (outline, crop, _) = crop_and_context(state, id, &label, observation);
            let region = crop.crop;
            let landed: Vec<[f64; 2]> = outline.samples.iter().flatten().copied().collect();
            assert!(landed.len() >= 4, "{stage}: the outline did not land");
            assert_eq!(
                region.size[0], region.size[1],
                "{stage}: observation {observation}'s crop is not square"
            );
            let gaps: Vec<(f64, f64)> = (0..2)
                .map(|axis| {
                    let lo = landed.iter().map(|p| p[axis]).fold(f64::INFINITY, f64::min);
                    let hi = landed
                        .iter()
                        .map(|p| p[axis])
                        .fold(f64::NEG_INFINITY, f64::max);
                    let start = region.min[axis] as f64;
                    (lo - start, start + region.size[axis] as f64 - hi)
                })
                .collect();
            for (axis, &(before, after)) in gaps.iter().enumerate() {
                assert!(
                    before >= margin && after >= margin,
                    "{stage}: observation {observation}'s outline is {before} and {after} px \
                     from the crop's edges on axis {axis}"
                );
                assert!(
                    (before - after).abs() < 2.0,
                    "{stage}: observation {observation}'s outline is off centre on axis \
                     {axis}: {before} vs {after} px"
                );
            }
            assert!(
                gaps.iter()
                    .any(|&(before, after)| before < margin + 1.0 && after < margin + 1.0),
                "{stage}: observation {observation}'s crop is wider than the outline on \
                 both axes: {gaps:?}"
            );
            // The demo's patches are a few dozen pixels across, so the crop is
            // read one texel per photograph pixel.
            assert_eq!(
                crop.image.size,
                [region.size[0] as usize, region.size[1] as usize],
                "{stage}: the crop is not one texel per pixel"
            );
            let src = state
                .cached_photograph(id, track.observations[observation].image as usize)
                .expect("a cached photograph");
            let (x, y) = (region.min[0] + 1, region.min[1] + 1);
            let texel = crop.image.pixels[crop.image.size[0] + 1].to_array();
            for (c, &value) in texel.iter().take(3).enumerate() {
                assert_eq!(
                    value,
                    src.level(0).get_pixel(x as u32, y as u32, c as u32),
                    "{stage}: a crop texel is not the photograph's pixel"
                );
            }
        }
    };
    check(&state, "track stage");

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    check(&state, "cluster stage");
}

/// A crop's hover view is the crop centred in three times its width and height
/// of the photograph at the crop's own sampling: three crops across and down,
/// the box its middle third, the texels there the crop's, and the outline the
/// crop's moved by one crop width and height.
#[test]
fn a_crop_s_hover_view_holds_the_crop_in_its_middle_third() {
    let (state, id, label, _, _) = on_the_bench();
    let k = super::tile::CONTEXT_FACTOR as usize;
    let observations = state
        .bench_track(id, &label)
        .expect("on the bench")
        .observations
        .len();
    for observation in 0..observations {
        let (_, crop, context) = crop_and_context(&state, id, &label, observation);
        let [w, h] = crop.image.size;
        assert_eq!(context.image.size, [w * k, h * k]);
        assert_eq!(context.crop, crop.crop, "the hover view names another crop");
        assert!(
            (context.keypoint - crop.keypoint - egui::vec2(w as f32, h as f32)).length() < 1e-3,
            "the hover view's dot is not the crop's moved by one crop"
        );
        for y in 0..h {
            for x in 0..w {
                let a = crop.image.pixels[y * w + x];
                let b = context.image.pixels[(y + h) * w * k + x + w];
                assert_eq!(a, b, "observation {observation}: texel ({x}, {y}) differs");
            }
        }
        for (a, b) in crop.outline.iter().zip(&context.outline) {
            let (a, b) = (a.expect("landed"), b.expect("landed"));
            assert!(
                (b - a - egui::vec2(w as f32, h as f32)).length() < 1e-3,
                "the hover view's outline is not the crop's moved by one crop"
            );
        }
    }
}

/// The caption gives the patch's two axes in the photograph's pixels. At the
/// track stage each is the axis's length through the lens, which for the
/// demo's cameras is the distance between the projections of the patch's
/// opposite edge midpoints; at the cluster stage it is the shape's column
/// times the template's width.
#[test]
fn a_crop_s_hover_view_gives_the_axes_in_photograph_pixels() {
    use sfmtool_core::bench::Stage;

    let (mut state, id, label, _, _) = on_the_bench();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let Stage::Track(payload) = &track.stage else {
        panic!("a track put on from a point is at the track stage");
    };
    let patch = payload.placement.clone().expect("a patch");
    let row = &track.observations[0];
    let recon = state.node(id).expect("loaded").recon();
    let (camera, pose) =
        crate::bench::geometry::view_of(&recon.image_table, row.image as usize).expect("a view");
    let frame = crate::bench::geometry::anchored_frame(&patch, &camera, &pose, row);
    let across = |s: f64, t: f64| {
        let (xyz, w) = frame.corner_homogeneous(s, t);
        crate::bench::geometry::project(&camera, &pose, xyz, w).expect("projects")
    };
    let chord = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    let want = [
        chord(across(-1.0, 0.0), across(1.0, 0.0)),
        chord(across(0.0, -1.0), across(0.0, 1.0)),
    ];
    let (_, _, context) = crop_and_context(&state, id, &label, 0);
    for (axis, (got, want)) in context.axes_px.iter().zip(want).enumerate() {
        let got = got.expect("the axis projects");
        assert!(
            (got - want).abs() < 0.01 * want,
            "axis {axis} is {got} px, the edge midpoints {want} px apart"
        );
    }
    let caption = super::crop::context_caption(&context);
    for axis in context.axes_px {
        let text = format!("{:.1} px", axis.expect("projects"));
        assert!(caption.contains(&text), "{text} is not in {caption}");
    }

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let Stage::Cluster(payload) = &track.stage else {
        panic!("the track moved to the cluster stage");
    };
    let shape = track.observations[0].shape().expect("a cluster shape");
    let (_, _, context) = crop_and_context(&state, id, &label, 0);
    for (axis, got) in context.axes_px.iter().enumerate() {
        let want = 2.0 * payload.radius * shape[0][axis].hypot(shape[1][axis]);
        let got = got.expect("a cluster axis");
        assert!(
            (got - want).abs() < 1e-9,
            "cluster axis {axis}: {got} vs {want}"
        );
    }
}

/// Resting the pointer on a row's crop shows its hover view, rendered for
/// that row alone, while the row keeps its hover and its click.
#[test]
fn hovering_a_crop_shows_it_in_context_and_keeps_the_row() {
    let (state, _, _, mut panel, ctx) = on_the_bench();
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    let y = row_y(&mut panel, &ctx, &state, 1);
    let x = super::table::ColumnLayout::new().crop_x() + super::table::TILE_SIZE / 2.0;
    let at = egui::pos2(x + 8.0, y + 20.0);
    let response = at_pointer(&mut panel, &ctx, &state, at, false);
    assert_eq!(response.hovered_image, Some(1), "the row lost its hover");
    for _ in 0..12 {
        run_frame(&mut panel, &ctx, &state);
    }
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("The patch's outline in this photograph")),
        "no crop hover caption in {texts:?}"
    );
    let rendered: Vec<usize> = panel.crop_contexts.keys().copied().collect();
    assert_eq!(
        rendered,
        vec![1],
        "crop hover views rendered for rows not hovered"
    );
    assert!(
        panel.contexts.is_empty(),
        "the tile's hover view was rendered"
    );

    let response = at_pointer(&mut panel, &ctx, &state, at, true);
    assert_eq!(
        response.pick_row,
        Some((1, false)),
        "a click on the crop did not pick the row"
    );
}

/// A crop's hover view marks where the observation sits and, at the track
/// stage, where the track's point projects, each at its own photograph pixel:
/// the picture is the photograph at one texel per pixel, so a mark's texel plus
/// the picture's corner is the pixel it marks. The distance the caption states
/// is the row's reprojection error. At the cluster stage there is a dot and no
/// ring.
#[test]
fn a_crop_s_hover_view_marks_the_keypoint_and_the_projection() {
    use sfmtool_core::bench::Stage;

    let (mut state, id, label, _, _) = on_the_bench();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let Stage::Track(payload) = &track.stage else {
        panic!("a track put on from a point is at the track stage");
    };
    let frame = payload.placement.clone().expect("a patch");
    let position = payload.position.expect("a triangulated point");

    // A sighting of the point in a fourth image, 5 px off its projection.
    let image = 3;
    let recon = state.node(id).expect("loaded").recon();
    let (camera, pose) =
        crate::bench::geometry::view_of(&recon.image_table, image).expect("a view");
    let projected = crate::bench::geometry::project(&camera, &pose, position.coords, frame.w)
        .expect("the demo cameras see every point");
    let keypoint = [projected[0] + 4.0, projected[1] - 3.0];
    state
        .add_bench_observation(
            &label,
            ImageRef::new(id, image),
            &crate::bench::Seed::Pixel {
                pixel: keypoint,
                radius_px: None,
            },
        )
        .expect("a pixel on the sensor");
    state.settle_bench_evaluation();
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let added = track.observations.len() - 1;
    let error = track.observations[added]
        .track
        .as_ref()
        .and_then(|m| m.reprojection_error)
        .expect("the evaluation measured the new row's error");

    let (_, _, context) = crop_and_context(&state, id, &label, added);
    let region = context.crop;
    let k = super::tile::CONTEXT_FACTOR as i64;
    let corner = [
        (region.min[0] - region.size[0] * (k / 2)) as f64,
        (region.min[1] - region.size[1] * (k / 2)) as f64,
    ];
    let to_photograph = |at: egui::Pos2| [corner[0] + f64::from(at.x), corner[1] + f64::from(at.y)];
    let distance = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    assert!(
        distance(to_photograph(context.keypoint), keypoint) < 1e-3,
        "the dot is not on the observation"
    );
    let ring = context.projection.expect("a ring");
    assert!(
        distance(to_photograph(ring), projected) < 1e-3,
        "the ring is not on the point's projection"
    );
    assert_eq!(
        context.projection_of,
        Some(super::tile::ProjectionOf::Point)
    );
    let stated = context.projection_px.expect("a projection distance");
    assert!(
        (stated - error).abs() < 1e-6,
        "the hover view states {stated} px, the row {error} px"
    );
    let caption = super::crop::context_caption(&context);
    assert!(
        caption.contains(&format!("{stated:.2} px away along the dashed line")),
        "the caption does not state the distance: {caption}"
    );

    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    let (_, _, context) = crop_and_context(&state, id, &label, 0);
    assert_eq!(context.projection, None);
    assert_eq!(context.projection_of, None);
    assert!(
        super::crop::context_caption(&context).contains("no point to project"),
        "the cluster caption does not say why there is no ring"
    );
}

// ── The tile's frame, and the name column ───────────────────────────────────

/// A row's tile is rendered through the point's frame re-anchored so that its
/// centre projects onto the observation's own keypoint, even when the keypoint
/// sits pixels off where the geometric frame lands; with no keypoint the
/// geometric frame is used as it is.
#[test]
fn the_tile_frame_is_anchored_on_the_observation_s_own_keypoint() {
    let (state, id) = state();
    let node = state.node(id).expect("loaded");
    let view = node.edited().point(POINT).expect("a live point");
    let frame = view.placement().expect("the point has a frame");
    for (k, obs) in view.observations().iter().enumerate() {
        let (camera, pose) =
            crate::bench::geometry::view_of(&node.recon().image_table, obs.image_index as usize)
                .expect("a posed image");
        let stored = view.keypoint_xy(k).expect("a stored keypoint");
        // Displaced from the projection, so the two anchors differ.
        let keypoint = [f64::from(stored[0]) + 6.0, f64::from(stored[1]) - 4.0];
        let centre = |frame: &sfmtool_core::patch::cloud::OrientedPatch| {
            crate::bench::geometry::project(&camera, &pose, frame.center.coords, frame.w)
                .expect("the centre projects")
        };
        let geometric = centre(&frame);
        let miss = (geometric[0] - keypoint[0]).hypot(geometric[1] - keypoint[1]);
        assert!(miss > 1.0, "the geometric frame missed by only {miss} px");

        let anchored = super::patch::render_frame(&frame, &camera, &pose, Some(keypoint));
        let landed = centre(&anchored);
        assert!(
            (landed[0] - keypoint[0]).abs() < 1e-6 && (landed[1] - keypoint[1]).abs() < 1e-6,
            "the anchored frame is centred at {landed:?}, not {keypoint:?}"
        );

        // With no keypoint to anchor on, the stored frame is drawn as it is.
        let unanchored = super::patch::render_frame(&frame, &camera, &pose, None);
        assert_eq!(unanchored.center, frame.center);
    }
}

/// A keypoint whose ray cannot meet the patch -- here a direction patch
/// pointing behind the camera -- leaves the tile on the stored frame rather
/// than on no frame at all.
#[test]
fn a_keypoint_whose_ray_cannot_meet_the_patch_falls_back_to_the_geometric_frame() {
    use nalgebra::{Point3, Vector3};
    let (state, id) = state();
    let (camera, pose) =
        crate::bench::geometry::view_of(&state.node(id).expect("loaded").recon().image_table, 0)
            .expect("a posed image");
    let behind = sfmtool_core::patch::cloud::OrientedPatch::from_infinity_direction(
        Point3::from(-(pose.to_rotation_matrix().transpose() * Vector3::new(0.0, 0.0, -1.0))),
        Vector3::new(0.0, 1.0, 0.0),
        [0.02, 0.02],
    );
    assert!(behind
        .anchored_at_keypoint(&camera, &pose, [320.0, 240.0])
        .is_none());
    let rendered = super::patch::render_frame(&behind, &camera, &pose, Some([320.0, 240.0]));
    assert_eq!(rendered.center, behind.center);
    assert_eq!(rendered.w, behind.w);
}

/// An image name too long for its column is cut in its middle, so the start of
/// the path and the end of the file name both stay in view.
#[test]
fn a_long_image_name_is_cut_in_its_middle() {
    let mut recon = crate::state::edits::tests::projected_embedded_demo(12);
    for (i, image) in recon.image_table.images.iter_mut().enumerate() {
        image.name =
            format!("images/a_capture_directory_with_a_long_name/fisheye_left/image_{i:03}.jpg");
    }
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(recon));
    let id = state.selected_recon.expect("a selected reconstruction");
    state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    let texts = painted(&mut panel, &ctx, &state, Vec::new());
    let cut: Vec<&String> = texts
        .iter()
        .filter(|t| t.starts_with("images/") && t.contains('\u{2026}'))
        .collect();
    assert_eq!(cut.len(), 3, "not one cut name per row: {texts:?}");
    for (i, name) in cut.iter().enumerate() {
        assert!(name.ends_with(&format!("_{i:03}.jpg")), "{name}");
    }
}

// ── Viewed mode ─────────────────────────────────────────────────────────────

mod viewed;
