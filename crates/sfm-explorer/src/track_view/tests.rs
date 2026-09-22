// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for Track View: the Edit checkbox, the dispatch on it and the
//! selection notice.
//!
//! What each body draws is tested in its own module (`view/tests.rs` and
//! `edit/tests.rs`). What is tested here is what the merge adds: that the box
//! is a reading of the bench and nothing else, that ticking and clearing it
//! are the bench steps they report, and which body a frame draws. The frames
//! run through `Context::run_ui`, which needs neither a GPU nor a window.

use super::{TrackView, TrackViewResponse, EDIT_IT_LABEL, EDIT_LABEL, VIEW_LABEL};
use crate::scene::{ImageRef, PointRef, ReconId};
use crate::state::AppState;

/// The point the bench fixture puts on the bench.
const POINT: usize = 2;

const VIEWPORT: egui::Vec2 = egui::vec2(1400.0, 900.0);

/// The bench module's own fixture: one `embedded_patches` node, a photograph
/// cached for every image.
fn state() -> (AppState, ReconId) {
    crate::bench::tests::state()
}

fn input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        events,
        ..Default::default()
    }
}

/// One frame of the panel, with `events` delivered, and what it reported.
fn run_frame(
    panel: &mut TrackView,
    ctx: &egui::Context,
    state: &AppState,
    events: Vec<egui::Event>,
) -> TrackViewResponse {
    let mut response = None;
    crate::test_support::run_frame_headless(ctx, input(events), |ui| {
        response = Some(panel.show(ui, state, &[], &crate::platform::ScrollInput::default()));
    });
    response.expect("the panel ran")
}

/// The strings one frame painted.
fn painted(panel: &mut TrackView, ctx: &egui::Context, state: &AppState) -> Vec<String> {
    crate::test_support::painted_texts(ctx, input(Vec::new()), |ui| {
        panel.show(ui, state, &[], &crate::platform::ScrollInput::default());
    })
}

/// Where the last frame painted `text`.
fn painted_at(
    panel: &mut TrackView,
    ctx: &egui::Context,
    state: &AppState,
    text: &str,
) -> egui::Pos2 {
    crate::test_support::painted_text_rects(ctx, input(Vec::new()), |ui| {
        panel.show(ui, state, &[], &crate::platform::ScrollInput::default());
    })
    .into_iter()
    .find(|painted| painted.text == text)
    .unwrap_or_else(|| panic!("{text:?} was not painted"))
    .rect
    .center()
}

/// Click at `pos`: hover on one frame, press and release on the next, and hand
/// back the second frame's response. egui resolves a click against the rects
/// the previous pass registered, so one frame cannot click anything.
fn click_at(
    panel: &mut TrackView,
    ctx: &egui::Context,
    state: &AppState,
    pos: egui::Pos2,
) -> TrackViewResponse {
    run_frame(panel, ctx, state, vec![egui::Event::PointerMoved(pos)]);
    let mut events = vec![egui::Event::PointerMoved(pos)];
    for pressed in [true, false] {
        events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        });
    }
    run_frame(panel, ctx, state, events)
}

/// Click the painted text `text`.
fn click_text(
    panel: &mut TrackView,
    ctx: &egui::Context,
    state: &AppState,
    text: &str,
) -> TrackViewResponse {
    let pos = painted_at(panel, ctx, state, text);
    click_at(panel, ctx, state, pos)
}

fn settled(state: &AppState) -> (TrackView, egui::Context) {
    let mut panel = TrackView::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, state, Vec::new());
    run_frame(&mut panel, &ctx, state, Vec::new());
    (panel, ctx)
}

fn versions(state: &AppState, id: ReconId) -> usize {
    state.node(id).expect("loaded").history.versions().len()
}

fn active(state: &AppState, id: ReconId) -> Option<String> {
    crate::bench::active_track_label(state.bench(id).expect("a bench")).map(str::to_string)
}

/// The Point ID view mode's header shows for `point` of `id`.
fn point_id(state: &AppState, id: ReconId, point: usize) -> String {
    crate::scene::point_id(state.node(id).expect("loaded"), point)
}

/// The box is ticked exactly while something is active, and nothing about it
/// is stored in the panel: an undo of a deactivation moves it on the next frame
/// with no panel call in between.
#[test]
fn the_box_reads_the_bench() {
    let (mut state, id) = state();
    let (mut panel, ctx) = settled(&state);
    let editing = |response: &TrackViewResponse| {
        assert_ne!(
            response.edit.is_some(),
            response.view.is_some(),
            "one body per frame"
        );
        response.edit.is_some()
    };
    let response = run_frame(&mut panel, &ctx, &state, Vec::new());
    assert!(!editing(&response), "edit mode with nothing active");

    let label = crate::bench::tests::put_on_bench(&mut state, id);
    let response = run_frame(&mut panel, &ctx, &state, Vec::new());
    assert!(editing(&response), "no edit mode with {label} active");
    assert!(!panel.edit_body().rows().is_empty());

    state.deactivate_bench_item(id).expect("an active item");
    let response = run_frame(&mut panel, &ctx, &state, Vec::new());
    assert!(!editing(&response), "edit mode after a deactivation");

    state.undo(id).expect("a version to undo");
    let response = run_frame(&mut panel, &ctx, &state, Vec::new());
    assert_eq!(active(&state, id).as_deref(), Some(label.as_str()));
    assert!(editing(&response), "the undo did not bring edit mode back");
}

/// Ticking the box over a selected point reports it, and applied it is one
/// version putting the point on the bench. A second tick over the same point
/// after a clear activates the item already there rather than putting a second
/// one on.
#[test]
fn ticking_the_box_puts_the_selected_point_on_the_bench_once() {
    let (mut state, id) = state();
    state.select_point(PointRef::new(id, POINT));
    let (mut panel, ctx) = settled(&state);

    let response = click_text(&mut panel, &ctx, &state, EDIT_LABEL);
    assert_eq!(response.set_edit, Some(true));
    let before = versions(&state, id);
    state.set_editing(id, true).expect("a selected point");
    assert_eq!(versions(&state, id), before + 1, "one tick, one version");
    let label = active(&state, id).expect("the point's track is active");
    assert_eq!(state.bench(id).expect("a bench").len(), 1);

    state.set_editing(id, false).expect("an active item");
    state.set_editing(id, true).expect("a selected point");
    let bench = state.bench(id).expect("a bench");
    assert_eq!(bench.len(), 1, "a second item for one point");
    assert_eq!(active(&state, id).as_deref(), Some(label.as_str()));
}

/// Clearing the box reports it, is one version that leaves the item on the
/// bench, and the next frame draws the selected point's committed track.
#[test]
fn clearing_the_box_deactivates_and_shows_the_selected_point() {
    let (mut state, id) = state();
    state.select_point(PointRef::new(id, POINT));
    let label = crate::bench::tests::put_on_bench(&mut state, id);
    let (mut panel, ctx) = settled(&state);
    assert!(!panel.edit_body().rows().is_empty());

    let response = click_text(&mut panel, &ctx, &state, EDIT_LABEL);
    assert_eq!(response.set_edit, Some(false));
    assert!(response.edit.is_some() && response.view.is_none());
    let before = versions(&state, id);
    state.set_editing(id, false).expect("an active item");
    assert_eq!(versions(&state, id), before + 1, "one clear, one version");
    assert_eq!(active(&state, id), None);
    assert_eq!(
        state
            .bench(id)
            .expect("a bench")
            .labels()
            .collect::<Vec<_>>(),
        [label.as_str()]
    );
    assert_eq!(
        state
            .node(id)
            .expect("loaded")
            .history
            .versions()
            .last()
            .expect("a version")
            .label,
        format!("Stopped editing {label}; it stays on the bench")
    );

    let response = run_frame(&mut panel, &ctx, &state, Vec::new());
    assert!(response.view.is_some() && response.edit.is_none());
    let texts = painted(&mut panel, &ctx, &state);
    let id_text = point_id(&state, id, POINT);
    assert!(
        texts.contains(&id_text),
        "no view-mode header for {id_text}: {texts:?}"
    );
}

/// With no point selected and nothing active the box is greyed, so a click on
/// it reports nothing, and the refusal names the three ways in. The empty state
/// underneath carries the bench line.
#[test]
fn with_nothing_to_edit_the_box_is_greyed_and_names_the_ways_in() {
    let (state, _id) = state();
    let (mut panel, ctx) = settled(&state);
    let response = click_text(&mut panel, &ctx, &state, EDIT_LABEL);
    assert_eq!(response.set_edit, None, "a greyed box reported a tick");

    let why = super::nothing_to_edit();
    assert!(why.contains("select a point"), "{why}");
    assert!(why.contains("Bench groups"), "{why}");
    assert!(
        why.contains(crate::image_detail::START_CLUSTER_LABEL),
        "{why}"
    );

    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.iter().any(|t| t == "No point selected"), "{texts:?}");
    assert!(
        texts
            .iter()
            .any(|t| t.contains(crate::image_detail::START_CLUSTER_LABEL)),
        "the empty bench's line is missing: {texts:?}"
    );
}

/// With items on the bench and none active, the empty state says how many and
/// where to go to edit one.
#[test]
fn the_empty_state_says_what_the_bench_holds() {
    let (mut state, id) = state();
    crate::bench::tests::put_on_bench(&mut state, id);
    state.deactivate_bench_item(id).expect("an active item");
    assert_eq!(state.selected_point, None);
    let (mut panel, ctx) = settled(&state);
    let texts = painted(&mut panel, &ctx, &state);
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("1 item on the bench: double-click it")),
        "{texts:?}"
    );
}

/// A discard of the active item leaves nothing active, so the panel returns to
/// view mode rather than switching to another item on the bench.
#[test]
fn discarding_the_active_item_returns_to_view_mode() {
    let (mut state, id) = state();
    let first = crate::bench::tests::put_on_bench(&mut state, id);
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
    let (mut panel, ctx) = settled(&state);
    state.discard_bench_item(id, &second).expect("on the bench");
    assert_eq!(active(&state, id), None);
    let response = run_frame(&mut panel, &ctx, &state, Vec::new());
    assert!(
        response.view.is_some(),
        "not in view mode after the discard"
    );
    let texts = painted(&mut panel, &ctx, &state);
    assert!(!texts.contains(&first), "switched to {first}: {texts:?}");
}

/// A cluster is edited in the same body at its own stage: the header says the
/// reference observation, which only the cluster stage has.
#[test]
fn edit_mode_on_a_cluster_draws_the_cluster_headline() {
    let (mut state, id) = state();
    let label = state
        .start_bench_cluster(
            ImageRef::new(id, 0),
            &crate::bench::Seed::Pixel {
                pixel: [120.0, 90.0],
                radius_px: Some(6.0),
            },
        )
        .expect("a pixel on the sensor")
        .label;
    let (mut panel, ctx) = settled(&state);
    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.contains(&label), "{texts:?}");
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("Reference: observation 0")),
        "{texts:?}"
    );
    assert!(
        texts.iter().any(|t| t == "· the cluster stage"),
        "{texts:?}"
    );
}

/// The selection notice is drawn exactly when the selected point is not the
/// edited track's origin, and its two buttons report their two gestures.
#[test]
fn the_selection_notice_appears_when_the_selection_parts_from_the_item() {
    let (mut state, id) = state();
    state.select_point(PointRef::new(id, POINT));
    crate::bench::tests::put_on_bench(&mut state, id);
    let (mut panel, ctx) = settled(&state);
    let notice = |texts: &[String]| texts.iter().any(|t| t.starts_with("Selected: "));
    assert!(
        !notice(&painted(&mut panel, &ctx, &state)),
        "a notice while the selection is the item's origin"
    );

    // Editing is sticky: another point selected leaves the panel on the item.
    let other = 0;
    assert!(state.scene[0].edited().point(other as u32).is_some());
    state.select_point(PointRef::new(id, other));
    let versions_before = versions(&state, id);
    run_frame(&mut panel, &ctx, &state, Vec::new());
    let texts = painted(&mut panel, &ctx, &state);
    let expected = format!(
        "Selected: {}, not the track being edited.",
        point_id(&state, id, other)
    );
    assert!(texts.contains(&expected), "{texts:?}");
    assert_eq!(
        versions(&state, id),
        versions_before,
        "a selection pushed a version"
    );
    assert!(!panel.edit_body().rows().is_empty(), "the edit was dropped");

    let response = click_text(&mut panel, &ctx, &state, VIEW_LABEL);
    assert_eq!(response.set_edit, Some(false));
    assert!(!response.edit_selected_point);
    let response = click_text(&mut panel, &ctx, &state, EDIT_IT_LABEL);
    assert!(response.edit_selected_point);
    assert_eq!(response.set_edit, None);

    // *Edit it* applied puts the selected point on the bench, and the notice
    // goes, since the selection is the new item's origin.
    state.set_editing(id, true).expect("a selected point");
    assert_eq!(state.bench(id).expect("a bench").len(), 2);
    run_frame(&mut panel, &ctx, &state, Vec::new());
    assert!(!notice(&painted(&mut panel, &ctx, &state)));
}

/// With no reconstruction there is no box, only the one sentence.
#[test]
fn with_no_reconstruction_there_is_no_box() {
    let state = AppState::new();
    let mut panel = TrackView::new();
    let ctx = egui::Context::default();
    let texts = painted(&mut panel, &ctx, &state);
    assert_eq!(texts, ["No reconstruction loaded"]);
}
