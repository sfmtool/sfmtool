// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for Track View: the Edit checkbox, the recent items strip
//! and the choice of mode.
//!
//! What the body draws is tested in its own module (`body/tests.rs`). What is
//! tested here is what the panel adds: that the box is a reading of the bench
//! and nothing else, that ticking and clearing it are the bench steps they
//! report, and which mode, or the empty state, a frame draws. Each frame asks
//! for the viewed track first, as the dock does. The frames run through
//! `Context::run_ui`, which needs neither a GPU nor a window.

use super::{BodyMode, TrackView, TrackViewResponse, EDIT_LABEL};
use crate::scene::{ImageRef, PointRef, ReconId};
use crate::state::AppState;

/// The point the bench fixture puts on the bench.
const POINT: usize = 2;

const VIEWPORT: egui::Vec2 = egui::vec2(1400.0, 900.0);

/// The infinity mark the headers draw is in egui's bundled fonts, so it does
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

/// One frame of the panel, with `events` delivered, and what it reported. The
/// viewed track is asked for first, as the dock asks for it.
fn run_frame(
    panel: &mut TrackView,
    ctx: &egui::Context,
    state: &mut AppState,
    events: Vec<egui::Event>,
) -> TrackViewResponse {
    state.refresh_viewed_track();
    let state = &*state;
    let mut response = None;
    crate::test_support::run_frame_headless(ctx, input(events), |ui| {
        response = Some(panel.show(ui, state));
    });
    response.expect("the panel ran")
}

/// The strings one frame painted.
fn painted(panel: &mut TrackView, ctx: &egui::Context, state: &mut AppState) -> Vec<String> {
    state.refresh_viewed_track();
    let state = &*state;
    crate::test_support::painted_texts(ctx, input(Vec::new()), |ui| {
        panel.show(ui, state);
    })
}

/// Where the last frame painted `text`.
fn painted_at(
    panel: &mut TrackView,
    ctx: &egui::Context,
    state: &mut AppState,
    text: &str,
) -> egui::Pos2 {
    state.refresh_viewed_track();
    let state = &*state;
    crate::test_support::painted_text_rects(ctx, input(Vec::new()), |ui| {
        panel.show(ui, state);
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
    state: &mut AppState,
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
    state: &mut AppState,
    text: &str,
) -> TrackViewResponse {
    let pos = painted_at(panel, ctx, state, text);
    click_at(panel, ctx, state, pos)
}

fn settled(state: &mut AppState) -> (TrackView, egui::Context) {
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

/// The Point ID the Viewed header shows for `point` of `id`.
fn point_id(state: &AppState, id: ReconId, point: usize) -> String {
    crate::scene::point_id(state.node(id).expect("loaded"), point)
}

/// The mode the body drew in, or `None` for the empty state or no body.
fn mode(response: &TrackViewResponse) -> Option<BodyMode> {
    response.body.as_ref().and_then(|body| body.mode)
}

/// The box is ticked exactly while an item on the selected node's bench is
/// focused, and nothing about it is stored in the panel: an unfocus and a
/// focus made outside it move it on the next frame with no panel call in
/// between.
#[test]
fn the_box_reads_the_focused_item() {
    let (mut state, id) = state();
    let (mut panel, ctx) = settled(&mut state);
    let response = run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert_ne!(mode(&response), Some(BodyMode::Edited));

    let label = crate::bench::tests::put_on_bench(&mut state, id);
    let response = run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert_eq!(
        mode(&response),
        Some(BodyMode::Edited),
        "not Edited with {label} focused"
    );
    assert!(!panel.body().rows().is_empty());

    // The unfocus leaves the item's origin selected, so the panel shows it.
    state.unfocus_bench_item();
    let response = run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert_eq!(mode(&response), Some(BodyMode::Viewed));

    state.focus_bench_item(id, &label).expect("on the bench");
    let response = run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert_eq!(focused(&state, id).as_deref(), Some(label.as_str()));
    assert_eq!(mode(&response), Some(BodyMode::Edited));
}

/// Ticking the box over a selected point reports it, and applied it is one
/// version putting the point on the bench. A second tick over the same point
/// after a clear focuses the item already there rather than putting a second
/// one on, and pushes no version.
#[test]
fn ticking_the_box_puts_the_selected_point_on_the_bench_once() {
    let (mut state, id) = state();
    state.select_point(PointRef::new(id, POINT));
    let (mut panel, ctx) = settled(&mut state);

    let response = click_text(&mut panel, &ctx, &mut state, EDIT_LABEL);
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
/// selected point as the viewed track.
#[test]
fn clearing_the_box_unfocuses_and_shows_the_selected_point() {
    let (mut state, id) = state();
    state.select_point(PointRef::new(id, POINT));
    let label = crate::bench::tests::put_on_bench(&mut state, id);
    let (mut panel, ctx) = settled(&mut state);
    assert!(!panel.body().rows().is_empty());

    let response = click_text(&mut panel, &ctx, &mut state, EDIT_LABEL);
    assert_eq!(response.set_edit, Some(false));
    assert_eq!(mode(&response), Some(BodyMode::Edited));
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

    let response = run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert_eq!(mode(&response), Some(BodyMode::Viewed));
    let texts = painted(&mut panel, &ctx, &mut state);
    let id_text = point_id(&state, id, POINT);
    assert!(
        texts.contains(&id_text),
        "no Viewed header for {id_text}: {texts:?}"
    );
}

/// With no point selected and no item focused this session the box is greyed,
/// so a click on it reports nothing, and the refusal names the three ways in.
/// The empty state underneath carries the bench line.
#[test]
fn with_nothing_to_edit_the_box_is_greyed_and_names_the_ways_in() {
    let (mut state, _id) = state();
    let (mut panel, ctx) = settled(&mut state);
    let response = click_text(&mut panel, &ctx, &mut state, EDIT_LABEL);
    assert_eq!(response.set_edit, None, "a greyed box reported a tick");

    let why = super::nothing_to_edit();
    assert!(why.contains("select a point"), "{why}");
    assert!(why.contains("Bench groups"), "{why}");
    assert!(
        why.contains(crate::image_detail::START_CLUSTER_LABEL),
        "{why}"
    );

    let texts = painted(&mut panel, &ctx, &mut state);
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
    let (mut panel, ctx) = settled(&mut state);
    let texts = painted(&mut panel, &ctx, &mut state);
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("1 item on the bench: double-click it")),
        "{texts:?}"
    );
}

/// The empty state's *Go to Point...* asks for the dialog, and a frame nobody
/// clicked in does not: the flag is a click report, not a state read, or the
/// dialog would reopen every frame.
#[test]
fn the_empty_state_offers_a_way_in_by_index_or_id() {
    let (mut state, _id) = state();
    let (mut panel, ctx) = settled(&mut state);
    let idle = run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(!idle.body.expect("the empty state").request_goto_point);

    let response = click_text(&mut panel, &ctx, &mut state, "Go to Point...");
    let body = response.body.expect("the empty state");
    assert_eq!(body.mode, None, "the empty state drew a track");
    assert!(body.request_goto_point, "the button asked for nothing");
}

/// A selected point the version at the cursor no longer holds -- deleted, or
/// an index past the end of the points -- takes the empty state rather than
/// showing a point that is not there.
#[test]
fn a_deleted_or_out_of_range_point_takes_the_empty_state() {
    let (mut state, id) = state();
    state.select_point(PointRef::new(id, POINT));
    state
        .delete_point(PointRef::new(id, POINT))
        .expect("a live point");
    state.selected_point = Some(PointRef::new(id, POINT));
    let (mut panel, ctx) = settled(&mut state);
    let response = run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert_eq!(mode(&response), None, "a deleted point drew a track");
    assert!(panel.body().rows().is_empty());
    let texts = painted(&mut panel, &ctx, &mut state);
    assert!(texts.iter().any(|t| t == "No point selected"), "{texts:?}");

    state.selected_point = Some(PointRef::new(id, 999));
    let response = run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert_eq!(mode(&response), None, "an index past the end drew a track");
    assert!(panel.body().rows().is_empty());
}

/// A discard of the focused item leaves nothing focused, so the panel leaves
/// Edited mode rather than switching to another item on the bench.
#[test]
fn discarding_the_focused_item_leaves_edit_mode() {
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
    let (mut panel, ctx) = settled(&mut state);
    state.discard_bench_item(id, &second).expect("on the bench");
    assert_eq!(focused(&state, id), None);
    let response = run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert_ne!(
        mode(&response),
        Some(BodyMode::Edited),
        "still Edited after the discard"
    );
    let texts = painted(&mut panel, &ctx, &mut state);
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
    let (mut panel, ctx) = settled(&mut state);
    let texts = painted(&mut panel, &ctx, &mut state);
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
    state: &mut AppState,
    pos: egui::Pos2,
) -> Vec<String> {
    ctx.all_styles_mut(|style| {
        style.interaction.tooltip_delay = 0.0;
        style.interaction.tooltip_grace_time = 0.0;
    });
    run_frame(panel, ctx, state, vec![egui::Event::PointerMoved(pos)]);
    painted(panel, ctx, state)
}

/// A point selected while an item is focused leaves Edited mode: the next frame
/// draws that point as the viewed track, and no version was pushed. The
/// selection notice this replaced is gone, since the selection can no longer
/// part from the item.
#[test]
fn selecting_another_point_while_editing_shows_that_point() {
    let (mut state, id) = state();
    state.select_point(PointRef::new(id, POINT));
    crate::bench::tests::put_on_bench(&mut state, id);
    let (mut panel, ctx) = settled(&mut state);
    assert!(!panel.body().rows().is_empty());

    let other = 0;
    assert!(state.scene[0].edited().point(other as u32).is_some());
    let before = versions(&state, id);
    state.select_point(PointRef::new(id, other));
    assert_eq!(versions(&state, id), before, "a selection pushed a version");
    assert_eq!(focused(&state, id), None);
    let response = run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert_eq!(mode(&response), Some(BodyMode::Viewed));
    let texts = painted(&mut panel, &ctx, &mut state);
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
    let (mut panel, ctx) = settled(&mut state);

    let at = painted_at(&mut panel, &ctx, &mut state, EDIT_LABEL);
    let texts = hover_texts_at(&mut panel, &ctx, &mut state, at);
    let hint = super::again_hint(&label);
    assert_eq!(hint, format!("Tick to edit {label} again"));
    assert!(texts.contains(&hint), "{texts:?}");
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with(&format!("Tick {EDIT_LABEL} to edit {label} again"))),
        "the empty state does not name the item: {texts:?}"
    );

    let response = click_text(&mut panel, &ctx, &mut state, EDIT_LABEL);
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

    let (mut panel, ctx) = settled(&mut state);
    let response = click_text(&mut panel, &ctx, &mut state, EDIT_LABEL);
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
    let (mut panel, ctx) = settled(&mut state);
    let response = click_text(&mut panel, &ctx, &mut state, EDIT_LABEL);
    assert_eq!(response.set_edit, Some(false), "the clear was greyed");

    // Over the point the item came from: the tick focuses it.
    state.unfocus_bench_item();
    let (mut panel, ctx) = settled(&mut state);
    let response = click_text(&mut panel, &ctx, &mut state, EDIT_LABEL);
    assert_eq!(response.set_edit, Some(true), "the focus was greyed");

    // Over a point not on the bench: the tick is a put, and greyed.
    state.select_point(PointRef::new(id, 0));
    let (mut panel, ctx) = settled(&mut state);
    let response = click_text(&mut panel, &ctx, &mut state, EDIT_LABEL);
    assert_eq!(response.set_edit, None, "the put was not greyed");

    // With no point selected: the tick focuses the most recent item.
    state.deselect_point();
    let (mut panel, ctx) = settled(&mut state);
    let response = click_text(&mut panel, &ctx, &mut state, EDIT_LABEL);
    assert_eq!(response.set_edit, Some(true), "going back was greyed");

    state.finish_background_task();
}

/// With no reconstruction there is no box, only the one sentence.
#[test]
fn with_no_reconstruction_there_is_no_box() {
    let mut state = AppState::new();
    let mut panel = TrackView::new();
    let ctx = egui::Context::default();
    let texts = painted(&mut panel, &ctx, &mut state);
    assert_eq!(texts, ["No reconstruction loaded"]);
}

// ── The recent items strip ──────────────────────────────────────────────

/// Put point `point` of `id` on the bench, which focuses it, and give back its
/// label.
fn put(state: &mut AppState, id: ReconId, point: usize) -> String {
    state
        .put_point_on_bench(PointRef::new(id, point), None)
        .expect("a live point")
}

/// The labels of the chips the strip drew on the last frame, in order.
fn chips(panel: &TrackView) -> Vec<String> {
    panel
        .recent()
        .drawn()
        .iter()
        .map(|chip| chip.label.clone())
        .collect()
}

/// One frame of the panel in a window `width` points wide.
fn run_frame_at_width(
    panel: &mut TrackView,
    ctx: &egui::Context,
    state: &mut AppState,
    width: f32,
) {
    state.refresh_viewed_track();
    let state = &*state;
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(width, VIEWPORT.y),
        )),
        ..Default::default()
    };
    crate::test_support::run_frame_headless(ctx, raw, |ui| {
        panel.show(ui, state);
    });
}

/// Where the chip for `label` is drawn: the centre of its shown text.
fn chip_at(
    panel: &mut TrackView,
    ctx: &egui::Context,
    state: &mut AppState,
    label: &str,
) -> egui::Pos2 {
    run_frame(panel, ctx, state, Vec::new());
    let shown = panel
        .recent()
        .drawn()
        .iter()
        .find(|chip| chip.label == label)
        .unwrap_or_else(|| panic!("no chip for {label}"))
        .shown
        .clone();
    painted_at(panel, ctx, state, &shown)
}

/// The strip lists the items focused this session, most recent first, with
/// the focused item left out, each chip's label cut in the middle.
#[test]
fn the_strip_lists_recent_items_most_recent_first_without_the_focused_one() {
    let (mut state, id) = state();
    let first = put(&mut state, id, 0);
    let second = put(&mut state, id, 1);
    let third = put(&mut state, id, POINT);
    assert_eq!(focused(&state, id).as_deref(), Some(third.as_str()));
    let (mut panel, ctx) = settled(&mut state);
    assert_eq!(chips(&panel), [second.clone(), first.clone()]);
    let shown = panel.recent().drawn()[0].shown.clone();
    let (head, tail) = shown
        .split_once(crate::elide::ELLIPSIS)
        .unwrap_or_else(|| panic!("{second} was not cut in the middle: {shown}"));
    assert!(
        second.starts_with(head) && second.ends_with(tail),
        "{shown}"
    );

    // With nothing focused, the item just left is the first chip.
    state.unfocus_bench_item();
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert_eq!(chips(&panel), [third, second, first]);
}

/// At most eight chips on a wide panel, the most recent of them on a narrow
/// one, and none when the row has no room left beside the box.
#[test]
fn the_strip_draws_at_most_eight_and_fewer_on_a_narrow_panel() {
    let (mut state, id) = state();
    for point in 0..11 {
        put(&mut state, id, point);
    }
    assert_eq!(state.recent_items.len(), 11);
    let (mut panel, ctx) = settled(&mut state);
    let wide = chips(&panel);
    assert_eq!(wide.len(), super::recent::MAX_CHIPS);

    run_frame_at_width(&mut panel, &ctx, &mut state, 360.0);
    let narrow = chips(&panel);
    assert!(
        (1..super::recent::MAX_CHIPS).contains(&narrow.len()),
        "{} chips at 360 points",
        narrow.len()
    );
    assert_eq!(narrow, wide[..narrow.len()]);

    run_frame_at_width(&mut panel, &ctx, &mut state, 80.0);
    assert!(chips(&panel).is_empty(), "{:?}", chips(&panel));
}

/// A click on a chip reports its item, and applied it focuses the item and
/// selects its origin, with no version and not refused by a busy node.
#[test]
fn clicking_a_chip_focuses_its_item_and_selects_its_origin() {
    let (mut state, id) = state();
    let first = put(&mut state, id, 0);
    put(&mut state, id, POINT);
    state
        .start_background_task(
            crate::background::Operation::BENCH_FIT,
            id,
            Box::new(|_| crate::background::Finished::Cancelled),
        )
        .expect("nothing else is running");
    let (mut panel, ctx) = settled(&mut state);
    let at = chip_at(&mut panel, &ctx, &mut state, &first);
    let response = click_at(&mut panel, &ctx, &mut state, at);
    assert_eq!(response.focus_item, Some((id, first.clone())));

    let before = versions(&state, id);
    state
        .focus_bench_item(id, &first)
        .expect("on the bench, and no step");
    assert_eq!(
        versions(&state, id),
        before,
        "a chip click pushed a version"
    );
    assert_eq!(focused(&state, id).as_deref(), Some(first.as_str()));
    assert_eq!(state.selected_point, Some(PointRef::new(id, 0)));
    state.finish_background_task();
}

/// A rename keeps the chip, under the item's new label.
#[test]
fn a_rename_keeps_the_chip() {
    let (mut state, id) = state();
    let first = put(&mut state, id, 0);
    put(&mut state, id, POINT);
    state
        .rename_bench_item(id, &first, "kerb")
        .expect("a free label");
    let (panel, _ctx) = settled(&mut state);
    assert_eq!(chips(&panel), ["kerb"]);
}

/// A discarded item's chip is hidden, and an undo of the discard brings it
/// back.
#[test]
fn a_discard_hides_the_chip_and_its_undo_brings_it_back() {
    let (mut state, id) = state();
    let first = put(&mut state, id, 0);
    put(&mut state, id, POINT);
    state.discard_bench_item(id, &first).expect("on the bench");
    let (mut panel, ctx) = settled(&mut state);
    assert!(chips(&panel).is_empty(), "{:?}", chips(&panel));

    state.undo(id).expect("the discard to undo");
    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert!(chips(&panel).contains(&first), "{:?}", chips(&panel));
}

/// A chip for an item on another node focuses it, which selects that node.
/// The hover text names the node once more than one is loaded.
#[test]
fn a_chip_for_an_item_on_another_node_selects_that_node() {
    let (mut state, a) = state();
    let on_a = put(&mut state, a, POINT);
    let b = state.append_node(crate::scene::SceneNode::demo(
        crate::state::edits::tests::projected_embedded_demo(12),
    ));
    put(&mut state, b, POINT);
    assert_eq!(state.selected_recon, Some(b));
    let (mut panel, ctx) = settled(&mut state);
    assert_eq!(chips(&panel), std::slice::from_ref(&on_a));

    let at = chip_at(&mut panel, &ctx, &mut state, &on_a);
    let texts = hover_texts_at(&mut panel, &ctx, &mut state, at);
    let node_label = state.node(a).expect("loaded").label.clone();
    assert!(texts.contains(&format!("On {node_label}")), "{texts:?}");

    let response = click_at(&mut panel, &ctx, &mut state, at);
    let (node, label) = response.focus_item.expect("the chip was clicked");
    state.focus_bench_item(node, &label).expect("on a's bench");
    assert_eq!(state.selected_recon, Some(a));
    assert_eq!(focused(&state, a).as_deref(), Some(on_a.as_str()));
    assert_eq!(state.selected_point, Some(PointRef::new(a, POINT)));
}

/// The hover text: the whole label, the stage, the counts, the origin, the
/// position, the evaluation state and what a click does, and no node name
/// with one reconstruction loaded.
#[test]
fn a_chips_hover_text_says_what_the_item_is() {
    let (mut state, id) = state();
    let first = put(&mut state, id, POINT);
    put(&mut state, id, 0);
    let (mut panel, ctx) = settled(&mut state);
    let at = chip_at(&mut panel, &ctx, &mut state, &first);
    let texts = hover_texts_at(&mut panel, &ctx, &mut state, at);

    let track = std::sync::Arc::clone(
        state
            .bench(id)
            .expect("a bench")
            .track(&first)
            .expect("on the bench"),
    );
    let (kept, out) = track.verdict_counts();
    let pinned = track.observations.iter().filter(|o| o.pinned).count();
    let expected = [
        first.clone(),
        "The track stage".to_string(),
        format!("{kept} kept · {out} out · {pinned} pinned"),
        format!("Point {}", point_id(&state, id, POINT)),
        super::recent::CLICK_TO_EDIT.to_string(),
    ];
    for line in &expected {
        assert!(texts.contains(line), "{line:?} missing: {texts:?}");
    }
    assert!(
        texts.iter().any(|t| t.starts_with("Position (")),
        "{texts:?}"
    );
    assert!(
        texts
            .iter()
            .any(|t| t == super::body::EVALUATED_LABEL || t == super::body::EVALUATING_LABEL),
        "no evaluation state: {texts:?}"
    );
    assert!(!texts.iter().any(|t| t.starts_with("On ")), "{texts:?}");
}

/// A chip draws the item's patch, uploaded once and not per frame; an item
/// with neither a bitmap nor a template draws an empty frame, and its hover
/// text says it is new.
#[test]
fn a_chip_draws_the_items_patch_or_an_empty_frame() {
    let (mut state, id) = state();
    let track = put(&mut state, id, POINT);
    // The demo stores no bitmaps; a fit fuses one.
    state
        .start_bench_fit(id, &track)
        .expect("a framed track with three sightings fits");
    state.finish_background_task();
    let cluster = state
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
    put(&mut state, id, 0);
    let (mut panel, ctx) = settled(&mut state);
    let chip = |panel: &TrackView, label: &str| {
        panel
            .recent()
            .drawn()
            .iter()
            .find(|chip| chip.label == label)
            .unwrap_or_else(|| panic!("no chip for {label}"))
            .clone()
    };
    let has_template = matches!(
        &state.bench(id).expect("a bench").track(&cluster).expect("on the bench").stage,
        sfmtool_core::bench::Stage::Cluster(payload) if payload.template.is_some()
    );
    assert!(!has_template, "the fixture's cluster has a template");
    assert_eq!(
        chip(&panel, &cluster).patch,
        None,
        "a cluster with no template"
    );
    let patch = chip(&panel, &track)
        .patch
        .expect("the track's patch bitmap");

    run_frame(&mut panel, &ctx, &mut state, Vec::new());
    assert_eq!(
        chip(&panel, &track).patch,
        Some(patch),
        "the patch was uploaded again"
    );

    let at = chip_at(&mut panel, &ctx, &mut state, &cluster);
    let texts = hover_texts_at(&mut panel, &ctx, &mut state, at);
    assert!(texts.iter().any(|t| t == "The cluster stage"), "{texts:?}");
    assert!(texts.iter().any(|t| t.starts_with("New")), "{texts:?}");
}
