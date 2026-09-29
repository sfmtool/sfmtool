// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for Track View: the Edit checkbox and the dispatch on it.
//!
//! What each body draws is tested in its own module (`view/tests.rs` and
//! `edit/tests.rs`). What is tested here is what the merge adds: that the box
//! is a reading of the bench and nothing else, that ticking and clearing it
//! are the bench steps they report, and which body a frame draws. The frames
//! run through `Context::run_ui`, which needs neither a GPU nor a window.

use super::{TrackView, TrackViewResponse, EDIT_LABEL};
use crate::scene::{ImageRef, PointRef, ReconId};
use crate::state::AppState;

/// The point the bench fixture puts on the bench.
const POINT: usize = 2;

const VIEWPORT: egui::Vec2 = egui::vec2(1400.0, 900.0);

/// The bench module's own fixture: one `embedded_patches` node, a photograph
/// cached for every image.
/// The infinity mark both headers draw is in egui's bundled fonts, so it does
/// not render as a box.
#[test]
fn the_infinity_mark_is_in_the_bundled_fonts() {
    let ctx = egui::Context::default();
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        ui.label("warm the font atlas");
    });
    let font = egui::TextStyle::Body.resolve(&ctx.global_style());
    assert!(
        ctx.fonts_mut(|f| f.has_glyphs(&font, super::INFINITY)),
        "{:?} is not in egui's bundled fonts and would render as a box",
        super::INFINITY
    );
}

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

fn focused(state: &AppState, id: ReconId) -> Option<String> {
    state.focused_item_label(id).map(str::to_string)
}

/// The Point ID view mode's header shows for `point` of `id`.
fn point_id(state: &AppState, id: ReconId, point: usize) -> String {
    crate::scene::point_id(state.node(id).expect("loaded"), point)
}

/// The box is ticked exactly while an item on the selected node's bench is
/// focused, and nothing about it is stored in the panel: an unfocus and a
/// focus made outside it move it on the next frame with no panel call in
/// between.
#[test]
fn the_box_reads_the_focused_item() {
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
    assert!(!editing(&response), "edit mode with nothing focused");

    let label = crate::bench::tests::put_on_bench(&mut state, id);
    let response = run_frame(&mut panel, &ctx, &state, Vec::new());
    assert!(editing(&response), "no edit mode with {label} focused");
    assert!(!panel.edit_body().rows().is_empty());

    state.unfocus_bench_item();
    let response = run_frame(&mut panel, &ctx, &state, Vec::new());
    assert!(!editing(&response), "edit mode after an unfocus");

    state.focus_bench_item(id, &label).expect("on the bench");
    let response = run_frame(&mut panel, &ctx, &state, Vec::new());
    assert_eq!(focused(&state, id).as_deref(), Some(label.as_str()));
    assert!(editing(&response), "the focus did not bring edit mode back");
}

/// Ticking the box over a selected point reports it, and applied it is one
/// version putting the point on the bench. A second tick over the same point
/// after a clear focuses the item already there rather than putting a second
/// one on, and pushes no version.
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
    let label = focused(&state, id).expect("the point's track is focused");
    assert_eq!(state.bench(id).expect("a bench").len(), 1);

    state.set_editing(id, false).expect("a focused item");
    let before = versions(&state, id);
    state.set_editing(id, true).expect("a selected point");
    let bench = state.bench(id).expect("a bench");
    assert_eq!(bench.len(), 1, "a second item for one point");
    assert_eq!(focused(&state, id).as_deref(), Some(label.as_str()));
    assert_eq!(versions(&state, id), before, "focusing pushed a version");
}

/// Clearing the box reports it, unfocuses the item with no version and one
/// `Selection` row, leaves the item on the bench, and the next frame draws the
/// selected point's committed track.
#[test]
fn clearing_the_box_unfocuses_and_shows_the_selected_point() {
    let (mut state, id) = state();
    state.select_point(PointRef::new(id, POINT));
    let label = crate::bench::tests::put_on_bench(&mut state, id);
    let (mut panel, ctx) = settled(&state);
    assert!(!panel.edit_body().rows().is_empty());

    let response = click_text(&mut panel, &ctx, &state, EDIT_LABEL);
    assert_eq!(response.set_edit, Some(false));
    assert!(response.edit.is_some() && response.view.is_none());
    let before = versions(&state, id);
    state.set_editing(id, false).expect("a focused item");
    assert_eq!(versions(&state, id), before, "a clear pushed a version");
    assert_eq!(focused(&state, id), None);
    assert_eq!(
        state
            .bench(id)
            .expect("a bench")
            .labels()
            .collect::<Vec<_>>(),
        [label.as_str()]
    );
    let last = state.action_log.entries().last().expect("a row");
    assert_eq!(last.kind, crate::action_log::Kind::Selection);
    assert_eq!(last.text, format!("Stopped editing {label}"));

    let response = run_frame(&mut panel, &ctx, &state, Vec::new());
    assert!(response.view.is_some() && response.edit.is_none());
    let texts = painted(&mut panel, &ctx, &state);
    let id_text = point_id(&state, id, POINT);
    assert!(
        texts.contains(&id_text),
        "no view-mode header for {id_text}: {texts:?}"
    );
}

/// With no point selected and no item focused this session the box is greyed,
/// so a click on it reports nothing, and the refusal names the three ways in.
/// The empty state underneath carries the bench line.
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

/// With items on the bench, none focused and none focused this session, the
/// empty state says how many and where to go to edit one.
#[test]
fn the_empty_state_says_what_the_bench_holds() {
    let (mut state, id) = state();
    crate::bench::tests::put_on_bench(&mut state, id);
    state.unfocus_bench_item();
    state.deselect_point();
    state.recent_items.clear();
    let (mut panel, ctx) = settled(&state);
    let texts = painted(&mut panel, &ctx, &state);
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("1 item on the bench: double-click it")),
        "{texts:?}"
    );
}

/// A discard of the focused item leaves nothing focused, so the panel returns
/// to view mode rather than switching to another item on the bench.
#[test]
fn discarding_the_focused_item_returns_to_view_mode() {
    let (mut state, id) = state();
    let first = crate::bench::tests::put_on_bench(&mut state, id);
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
    let (mut panel, ctx) = settled(&state);
    state.discard_bench_item(id, &second).expect("on the bench");
    assert_eq!(focused(&state, id), None);
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
            None,
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

/// Every string painted with the pointer resting at `pos`, the tooltip it
/// raises included. The delays go to zero first, since a headless frame has
/// no wall clock to pass.
fn hover_texts_at(
    panel: &mut TrackView,
    ctx: &egui::Context,
    state: &AppState,
    pos: egui::Pos2,
) -> Vec<String> {
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    run_frame(panel, ctx, state, vec![egui::Event::PointerMoved(pos)]);
    painted(panel, ctx, state)
}

/// A point selected while an item is focused leaves edit mode: the next frame
/// draws that point's committed track, and no version was pushed. The
/// selection notice this replaced is gone, since the selection can no longer
/// part from the item.
#[test]
fn selecting_another_point_while_editing_shows_that_point() {
    let (mut state, id) = state();
    state.select_point(PointRef::new(id, POINT));
    crate::bench::tests::put_on_bench(&mut state, id);
    let (mut panel, ctx) = settled(&state);
    assert!(!panel.edit_body().rows().is_empty());

    let other = 0;
    assert!(state.scene[0].edited().point(other as u32).is_some());
    let before = versions(&state, id);
    state.select_point(PointRef::new(id, other));
    assert_eq!(versions(&state, id), before, "a selection pushed a version");
    assert_eq!(focused(&state, id), None);
    let response = run_frame(&mut panel, &ctx, &state, Vec::new());
    assert!(response.view.is_some() && response.edit.is_none());
    let texts = painted(&mut panel, &ctx, &state);
    let id_text = point_id(&state, id, other);
    assert!(texts.contains(&id_text), "{texts:?}");
    assert!(
        !texts.iter().any(|t| t.starts_with("Selected: ")),
        "a selection notice was drawn: {texts:?}"
    );
}

/// With no point selected and an item focused earlier, the box is live, its
/// hover text names the item, the empty state says the same, and the tick
/// applied focuses it and selects its origin.
#[test]
fn ticking_the_box_with_no_point_selected_focuses_the_most_recent_item() {
    let (mut state, id) = state();
    let label = crate::bench::tests::put_on_bench(&mut state, id);
    state.unfocus_bench_item();
    state.deselect_point();
    let (mut panel, ctx) = settled(&state);

    let at = painted_at(&mut panel, &ctx, &state, EDIT_LABEL);
    let texts = hover_texts_at(&mut panel, &ctx, &state, at);
    let hint = super::again_hint(&label);
    assert_eq!(hint, format!("Tick to edit {label} again"));
    assert!(texts.contains(&hint), "{texts:?}");
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with(&format!("Tick {EDIT_LABEL} to edit {label} again"))),
        "the empty state does not name the item: {texts:?}"
    );

    let response = click_text(&mut panel, &ctx, &state, EDIT_LABEL);
    assert_eq!(response.set_edit, Some(true), "the box was greyed");
    let before = versions(&state, id);
    state.set_editing(id, true).expect("an item to go back to");
    assert_eq!(versions(&state, id), before, "focusing pushed a version");
    assert_eq!(focused(&state, id).as_deref(), Some(label.as_str()));
    assert_eq!(state.selected_point, Some(PointRef::new(id, POINT)));
}

/// The tick with no point selected focuses the most recent item on any
/// loaded node, selecting that node, and passes over an item discarded since.
#[test]
fn the_tick_goes_back_to_an_item_on_another_node_and_skips_a_discarded_one() {
    let (mut state, a) = state();
    let on_a = crate::bench::tests::put_on_bench(&mut state, a);
    let b = state.append_node(crate::scene::SceneNode::demo(
        crate::state::edits::tests::projected_embedded_demo(12),
    ));
    let on_b = crate::bench::tests::put_on_bench(&mut state, b);
    // Both were focused, b's last; b's item is then discarded.
    state.discard_bench_item(b, &on_b).expect("on b's bench");
    state.deselect_point();
    assert_eq!(state.selected_recon, Some(b));
    assert_eq!(state.most_recent_item(), Some((a, on_a.clone())));

    let (mut panel, ctx) = settled(&state);
    let response = click_text(&mut panel, &ctx, &state, EDIT_LABEL);
    assert_eq!(response.set_edit, Some(true));
    state.set_editing(b, true).expect("a's item to go back to");
    assert_eq!(state.selected_recon, Some(a), "the tick did not select a");
    assert_eq!(focused(&state, a).as_deref(), Some(on_a.as_str()));
}

/// A task holding the node greys the box only where a tick would put a point
/// on the bench: a clear, and a tick over a point an item already came from,
/// push no version and stay available.
#[test]
fn a_busy_node_greys_the_box_only_for_a_put() {
    let (mut state, id) = state();
    state.select_point(PointRef::new(id, POINT));
    crate::bench::tests::put_on_bench(&mut state, id);
    state
        .start_background_task(
            crate::background::Operation::BENCH_FIT,
            id,
            Box::new(|_| crate::background::Finished::Cancelled),
        )
        .expect("nothing else is running");
    assert!(state.busy_refusal(id).is_some());

    // Editing: the clear is available.
    let (mut panel, ctx) = settled(&state);
    let response = click_text(&mut panel, &ctx, &state, EDIT_LABEL);
    assert_eq!(response.set_edit, Some(false), "the clear was greyed");

    // Over the point the item came from: the tick focuses it.
    state.unfocus_bench_item();
    let (mut panel, ctx) = settled(&state);
    let response = click_text(&mut panel, &ctx, &state, EDIT_LABEL);
    assert_eq!(response.set_edit, Some(true), "the focus was greyed");

    // Over a point not on the bench: the tick is a put, and greyed.
    state.select_point(PointRef::new(id, 0));
    let (mut panel, ctx) = settled(&state);
    let response = click_text(&mut panel, &ctx, &state, EDIT_LABEL);
    assert_eq!(response.set_edit, None, "the put was not greyed");

    // With no point selected: the tick focuses the most recent item.
    state.deselect_point();
    let (mut panel, ctx) = settled(&state);
    let response = click_text(&mut panel, &ctx, &state, EDIT_LABEL);
    assert_eq!(response.set_edit, Some(true), "going back was greyed");

    state.finish_background_task();
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
