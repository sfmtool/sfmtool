// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for Measurement View: the empty state and its links, the
//! header, the result block, and the endpoint blocks' gestures, on the fixture
//! in [`crate::measurements::fixture`]. The frames run through
//! `Context::run_ui`, which needs neither a GPU nor a window. The fixture's
//! photographs are not on disk, so the crops say so; the pictures are not what
//! is tested here.

use super::*;
use crate::measurements::fixture::measured_state;
use crate::state::AppState;

const VIEWPORT: egui::Vec2 = egui::vec2(1000.0, 900.0);

fn input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), VIEWPORT)),
        events,
        ..Default::default()
    }
}

fn run_frame(
    panel: &mut MeasurementView,
    ctx: &egui::Context,
    state: &AppState,
    events: Vec<egui::Event>,
) -> MeasurementViewResponse {
    let mut response = None;
    crate::test_support::run_frame_headless(ctx, input(events), |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            response = Some(panel.show(ui, state));
        });
    });
    response.expect("the panel ran")
}

fn painted(panel: &mut MeasurementView, ctx: &egui::Context, state: &AppState) -> Vec<String> {
    crate::test_support::painted_texts(ctx, input(Vec::new()), |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            panel.show(ui, state);
        });
    })
}

fn button(pos: egui::Pos2, which: egui::PointerButton, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: which,
        pressed,
        modifiers: egui::Modifiers::default(),
    }
}

/// Hover, press and release `which` at `pos`, `times` times, returning the
/// response of every frame after the first.
fn clicks(
    panel: &mut MeasurementView,
    ctx: &egui::Context,
    state: &AppState,
    pos: egui::Pos2,
    which: egui::PointerButton,
    times: usize,
) -> Vec<MeasurementViewResponse> {
    let mut out = vec![run_frame(
        panel,
        ctx,
        state,
        vec![egui::Event::PointerMoved(pos)],
    )];
    for _ in 0..times {
        out.push(run_frame(panel, ctx, state, vec![button(pos, which, true)]));
        out.push(run_frame(
            panel,
            ctx,
            state,
            vec![button(pos, which, false)],
        ));
    }
    out
}

/// A panel and context that have laid out twice, so a click lands on the
/// rects of the frame before.
fn settled(state: &AppState) -> (MeasurementView, egui::Context) {
    let mut panel = MeasurementView::new();
    let ctx = egui::Context::default();
    for _ in 0..2 {
        run_frame(&mut panel, &ctx, state, Vec::new());
    }
    (panel, ctx)
}

/// With no measurement selected the panel says so and lists the
/// reconstruction's measurements as the Scene tree does, and a click on one
/// selects it.
#[test]
fn the_empty_state_lists_the_measurements_and_a_link_selects_one() {
    let (state, id) = measured_state(Some("m"));
    let (mut panel, ctx) = settled(&state);
    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.iter().any(|t| t == NO_MEASUREMENT), "{texts:?}");
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("Distance #4  paw width  1.8 m")),
        "{texts:?}"
    );
    let links = panel.link_rects().to_vec();
    assert_eq!(links.len(), 5);
    let responses = clicks(
        &mut panel,
        &ctx,
        &state,
        links[3].center(),
        egui::PointerButton::Primary,
        1,
    );
    assert_eq!(
        responses.last().unwrap().select_measurement,
        Some(MeasurementRef::distance(id, 3))
    );
}

/// A selected measurement draws its header with the median line, both
/// endpoint blocks and the result block with the hand-computed numbers.
#[test]
fn a_selected_measurement_draws_its_header_endpoints_and_result() {
    let (mut state, id) = measured_state(Some("m"));
    state.select_measurement(MeasurementRef::distance(id, 0));
    let (mut panel, ctx) = settled(&state);
    let texts = painted(&mut panel, &ctx, &state);
    for expected in [
        "Distance #0",
        "Label: ear to ear",
        "Stated distance: 153 cm",
        "Median over 3: 1 su = 102 cm; this one +0.0%",
        "A  image_000.jpg",
        "B  image_000.jpg",
        "One square holds the pixel.",
        "Reconstructed: 150.0 cm",
        "Error: \u{2212}3.00 cm (\u{2212}1.96%)",
        "Implied scale: 1.0200\u{00d7}",
    ] {
        assert!(
            texts.iter().any(|t| t == expected),
            "{expected:?} not in {texts:?}"
        );
    }
}

/// The result block's other three cases.
#[test]
fn the_result_block_reads_scene_units_infinity_and_a_refusal() {
    let (rows, readings) = {
        let recon = crate::measurements::fixture::measured_reconstruction(None);
        let rows = crate::measurements::distance_rows(&recon.image_set.measurements);
        let edited = sfmtool_core::EditedReconstruction::new(std::sync::Arc::new(recon));
        (rows, crate::measurements::MeasurementReadings::of(&edited))
    };
    let lines = |i: usize| -> Vec<String> {
        result_lines(&rows[i], &readings.distance[i], None)
            .iter()
            .map(|line| line.text().to_string())
            .collect()
    };
    assert_eq!(
        lines(0),
        ["Reconstructed: 1.500 su", "Implied scale: 1 su = 102.0 cm"]
    );
    assert_eq!(lines(2), ["No distance: endpoint B has no 3D position."]);
    assert_eq!(
        lines(3),
        ["No distance: endpoint B lies on a point at infinity."]
    );
}

/// A refused endpoint's block gives the refusal's sentence, which names the
/// remedy.
#[test]
fn a_refused_endpoint_says_why_in_its_block() {
    let (mut state, id) = measured_state(Some("m"));
    state.select_measurement(MeasurementRef::distance(id, 2));
    let (mut panel, ctx) = settled(&state);
    let texts = painted(&mut panel, &ctx, &state);
    assert!(
        texts.iter().any(|t| t
            .starts_with("No position: no observation's square in image 0 holds pixel")
            && t.contains("Create Track Here")),
        "{texts:?}"
    );
}

/// An endpoint block's click selects the endpoint, its double-click asks for
/// camera view through it as well, and hovering it hovers its image. A right
/// click selects nothing.
#[test]
fn an_endpoint_blocks_gestures_are_track_views_row_gestures() {
    let (mut state, id) = measured_state(Some("m"));
    let measurement = MeasurementRef::distance(id, 1);
    state.select_measurement(measurement);
    let (mut panel, ctx) = settled(&state);
    let at = panel
        .name_rect(EndpointSide::B)
        .expect("endpoint B's name was drawn")
        .center();

    let hovered = run_frame(
        &mut panel,
        &ctx,
        &state,
        vec![egui::Event::PointerMoved(at)],
    );
    assert!(hovered.has_pointer);
    assert_eq!(hovered.hovered_image, Some(ImageRef::new(id, 1)));

    let once = clicks(
        &mut panel,
        &ctx,
        &state,
        at,
        egui::PointerButton::Primary,
        1,
    );
    let clicked = once.last().unwrap();
    assert_eq!(
        clicked.select_endpoint,
        Some((measurement, EndpointSide::B))
    );
    assert_eq!(clicked.look_through, None);

    let ctx = egui::Context::default();
    let mut panel = MeasurementView::new();
    for _ in 0..2 {
        run_frame(&mut panel, &ctx, &state, Vec::new());
    }
    let twice = clicks(
        &mut panel,
        &ctx,
        &state,
        at,
        egui::PointerButton::Primary,
        2,
    );
    assert!(
        twice
            .iter()
            .any(|r| r.look_through == Some((measurement, EndpointSide::B))),
        "no double-click was reported"
    );

    let ctx = egui::Context::default();
    let mut panel = MeasurementView::new();
    for _ in 0..2 {
        run_frame(&mut panel, &ctx, &state, Vec::new());
    }
    let right = clicks(
        &mut panel,
        &ctx,
        &state,
        at,
        egui::PointerButton::Secondary,
        1,
    );
    assert!(
        right
            .iter()
            .all(|r| r.select_endpoint.is_none() && r.look_through.is_none()),
        "a right click selected something"
    );
}

// ── The edited mode ──────────────────────────────────────────────────────

/// The Edit box is drawn clear with a stored measurement selected, and a click
/// on it asks for the measurement to be put on the bench.
#[test]
fn ticking_the_edit_box_asks_to_edit_the_selected_measurement() {
    let (mut state, id) = measured_state(Some("m"));
    state.select_measurement(MeasurementRef::distance(id, 0));
    let (mut panel, ctx) = settled(&state);
    assert!(panel.toolbar().is_empty());
    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.iter().any(|t| t == EDIT_LABEL), "{texts:?}");
    // The box is the first widget of the panel, at its top left.
    let at = egui::pos2(14.0, 14.0);
    let responses = clicks(
        &mut panel,
        &ctx,
        &state,
        at,
        egui::PointerButton::Primary,
        1,
    );
    assert!(
        responses.iter().any(|r| r.set_edit == Some((id, true))),
        "the box was not ticked"
    );
}

/// With a measurement item focused the panel shows the item: its toolbar,
/// where it came from, and its live readings, which follow a step on the item
/// while the stored row's stay as they were.
#[test]
fn the_edited_mode_shows_the_item_s_live_readings() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    let (mut panel, ctx) = settled(&state);
    let toolbar: Vec<&str> = panel.toolbar().iter().map(|(t, _)| t.as_str()).collect();
    assert_eq!(toolbar, ["Commit", "Discard", "Rename"]);
    assert!(panel.toolbar().iter().all(|(_, refusal)| refusal.is_none()));
    let texts = painted(&mut panel, &ctx, &state);
    for expected in [
        "distance #0",
        "From Distance #0: Commit replaces it",
        // The label and the stated distance, each in its field.
        "Label",
        "ear to ear",
        "Stated distance",
        "153 cm",
        "Reconstructed: 150.0 cm",
        // The median leaves the origin out, and compares the item with the
        // other two readings.
        "Median over 3: 1 su = 102 cm; this one +0.0%",
    ] {
        assert!(
            texts.iter().any(|t| t == expected),
            "{expected:?} not in {texts:?}"
        );
    }
    let stored = crate::measurements::readings(state.node(id).unwrap());
    let reading = panel.item_reading().expect("read").clone();
    assert_eq!(reading.resolved.as_ref(), stored.distance.first());

    // Move endpoint B onto point 2, as a later step will: the item's reading
    // changes and the stored row's does not.
    let index = state.node_index(id).unwrap();
    let mut moved = (**state.bench_measurement(id, "distance #0").unwrap()).clone();
    let sfmtool_core::bench::EditableMeasurementBody::Distance(d) = &mut moved.body;
    let row4 =
        &crate::measurements::distance_rows(state.node(id).unwrap().edited().measurements())[4];
    d.endpoints[1] = Some(sfmtool_core::bench::MeasurementEndpoint {
        image: row4.images[1],
        pixel: row4.pixels[1],
    });
    let bench = state.scene[index]
        .history
        .current_bench()
        .replace(
            "distance #0",
            sfmtool_core::bench::BenchItem::Measurement(std::sync::Arc::new(moved)),
        )
        .unwrap();
    state.scene[index]
        .history
        .push_bench(std::sync::Arc::new(bench), "moved B");
    run_frame(&mut panel, &ctx, &state, Vec::new());
    let after = panel.item_reading().expect("read").clone();
    assert_ne!(after.resolved, reading.resolved);
    let chosen = after.endpoints[1]
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap()
        .chosen()
        .point;
    assert_eq!(chosen, 2);
    assert_eq!(
        crate::measurements::readings(state.node(id).unwrap()).distance,
        stored.distance
    );
}

/// An endpoint the item lacks is a block saying so, the result says why
/// there is no distance, and Commit is greyed with the refusal.
#[test]
fn an_unplaced_endpoint_greys_commit_and_says_so() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 1))
        .unwrap();
    // Deleting image 1 clears endpoint B, which was in it.
    state.delete_image(ImageRef::new(id, 1)).unwrap();
    let (mut panel, ctx) = settled(&state);
    let texts = painted(&mut panel, &ctx, &state);
    for expected in [
        "Endpoint B is not placed.",
        "No distance: endpoint B is not placed.",
        "From no stored row: Commit adds it as a new row",
    ] {
        assert!(
            texts.iter().any(|t| t == expected),
            "{expected:?} not in {texts:?}"
        );
    }
    let commit = &panel.toolbar()[0];
    assert_eq!(commit.0, "Commit");
    assert_eq!(
        commit.1.as_deref(),
        Some(
            "Cannot commit distance #1: endpoint B is not placed, and a distance needs both \
             endpoints."
        )
    );
}

// ── The header's fields ─────────────────────────────────────────────────

fn key(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }
}

/// A settled panel showing the focused item `distance #0`.
fn editing(state: &mut AppState, id: ReconId) -> (MeasurementView, egui::Context) {
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    settled(state)
}

/// Click into `field`, select what it holds and type `text` over it,
/// returning the response of every frame.
fn type_over(
    panel: &mut MeasurementView,
    ctx: &egui::Context,
    state: &AppState,
    field: HeaderField,
    text: &str,
) -> Vec<MeasurementViewResponse> {
    let at = panel
        .field_rect(field)
        .expect("the field was drawn")
        .center();
    let mut out = clicks(panel, ctx, state, at, egui::PointerButton::Primary, 1);
    out.push(run_frame(
        panel,
        ctx,
        state,
        vec![
            key(egui::Key::A, egui::Modifiers::COMMAND),
            egui::Event::Text(text.to_string()),
        ],
    ));
    out
}

fn any_label(responses: &[MeasurementViewResponse]) -> Option<(ReconId, String, String)> {
    responses.iter().find_map(|r| r.set_label.clone())
}

fn any_distance(responses: &[MeasurementViewResponse]) -> Option<(ReconId, String, String)> {
    responses.iter().find_map(|r| r.set_distance.clone())
}

/// Typing a label and pressing Enter asks for the label step, once, on the
/// item the panel shows.
#[test]
fn the_label_field_commits_on_enter() {
    let (mut state, id) = measured_state(Some("m"));
    let (mut panel, ctx) = editing(&mut state, id);
    let typed = type_over(&mut panel, &ctx, &state, HeaderField::Label, "paw width");
    assert_eq!(any_label(&typed), None, "typing alone committed");
    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.iter().any(|t| t == "paw width"), "{texts:?}");
    let entered = run_frame(
        &mut panel,
        &ctx,
        &state,
        vec![key(egui::Key::Enter, egui::Modifiers::default())],
    );
    assert_eq!(
        entered.set_label,
        Some((id, "distance #0".to_string(), "paw width".to_string()))
    );
    assert_eq!(entered.set_distance, None);
    let after = run_frame(&mut panel, &ctx, &state, Vec::new());
    assert_eq!(after.set_label, None, "the commit was reported twice");
}

/// Typing a distance and clicking elsewhere, which takes the focus from the
/// field, asks for the distance step with the text typed.
#[test]
fn the_distance_field_commits_when_it_loses_focus() {
    let (mut state, id) = measured_state(Some("m"));
    let (mut panel, ctx) = editing(&mut state, id);
    type_over(&mut panel, &ctx, &state, HeaderField::Distance, "4.2 cm");
    // A click on the empty foot of the panel.
    let away = clicks(
        &mut panel,
        &ctx,
        &state,
        egui::pos2(500.0, 890.0),
        egui::PointerButton::Primary,
        1,
    );
    assert_eq!(
        any_distance(&away),
        Some((id, "distance #0".to_string(), "4.2 cm".to_string()))
    );
    assert_eq!(any_label(&away), None);
}

/// Escape puts the field back to what the item holds and asks for nothing.
#[test]
fn escape_puts_a_field_back() {
    let (mut state, id) = measured_state(Some("m"));
    let (mut panel, ctx) = editing(&mut state, id);
    type_over(&mut panel, &ctx, &state, HeaderField::Label, "paw width");
    let escaped = run_frame(
        &mut panel,
        &ctx,
        &state,
        vec![key(egui::Key::Escape, egui::Modifiers::default())],
    );
    assert_eq!(escaped.set_label, None);
    // Nothing is asked for when the focus moves on either.
    let away = clicks(
        &mut panel,
        &ctx,
        &state,
        egui::pos2(500.0, 890.0),
        egui::PointerButton::Primary,
        1,
    );
    assert_eq!(any_label(&away), None);
    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.iter().any(|t| t == "ear to ear"), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "paw width"), "{texts:?}");
}

/// A field left with the text it took focus with asks for no step.
#[test]
fn a_field_left_unchanged_asks_for_nothing() {
    let (mut state, id) = measured_state(Some("m"));
    let (mut panel, ctx) = editing(&mut state, id);
    let typed = type_over(&mut panel, &ctx, &state, HeaderField::Distance, "153 cm");
    let entered = run_frame(
        &mut panel,
        &ctx,
        &state,
        vec![key(egui::Key::Enter, egui::Modifiers::default())],
    );
    assert_eq!(any_distance(&typed), None);
    assert_eq!(entered.set_distance, None);
}

/// While a task holds the node the fields are greyed, with the refusal a step
/// would give (`AppState::bench_edit_refusal`, which a view-only bench answers
/// too): they take no typing and ask for nothing.
#[test]
fn the_fields_are_greyed_while_a_step_would_be_refused() {
    use crate::background::{Finished, Operation};
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    state
        .start_background_task(
            Operation::BENCH_FIT,
            id,
            Box::new(move |_| Finished::Cancelled),
        )
        .expect("nothing else is running");
    let (mut panel, ctx) = settled(&state);
    let mut responses = type_over(&mut panel, &ctx, &state, HeaderField::Label, "typed");
    responses.push(run_frame(
        &mut panel,
        &ctx,
        &state,
        vec![key(egui::Key::Enter, egui::Modifiers::default())],
    ));
    assert_eq!(any_label(&responses), None);
    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.iter().any(|t| t == "ear to ear"), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "typed"), "{texts:?}");
}

/// Typing a distance and clicking straight into the label field commits the
/// distance: each field keeps its own draft, so the field that gains the
/// focus does not take the other's.
#[test]
fn moving_from_the_distance_field_to_the_label_field_commits_the_distance() {
    let (mut state, id) = measured_state(Some("m"));
    let (mut panel, ctx) = editing(&mut state, id);
    type_over(&mut panel, &ctx, &state, HeaderField::Distance, "4.2 cm");
    let label = panel.field_rect(HeaderField::Label).unwrap().center();
    let moved = clicks(
        &mut panel,
        &ctx,
        &state,
        label,
        egui::PointerButton::Primary,
        1,
    );
    assert_eq!(
        any_distance(&moved),
        Some((id, "distance #0".to_string(), "4.2 cm".to_string()))
    );
    assert_eq!(any_label(&moved), None);
}

/// The same the other way: typing a label and clicking straight into the
/// stated distance field commits the label.
#[test]
fn moving_from_the_label_field_to_the_distance_field_commits_the_label() {
    let (mut state, id) = measured_state(Some("m"));
    let (mut panel, ctx) = editing(&mut state, id);
    type_over(&mut panel, &ctx, &state, HeaderField::Label, "paw width");
    let distance = panel.field_rect(HeaderField::Distance).unwrap().center();
    let moved = clicks(
        &mut panel,
        &ctx,
        &state,
        distance,
        egui::PointerButton::Primary,
        1,
    );
    assert_eq!(
        any_label(&moved),
        Some((id, "distance #0".to_string(), "paw width".to_string()))
    );
    assert_eq!(any_distance(&moved), None);
}

// ── The endpoint blocks' layout ─────────────────────────────────────────

/// One frame of the panel in a viewport `width` points wide.
fn run_narrow(panel: &mut MeasurementView, ctx: &egui::Context, state: &AppState, width: f32) {
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(width, 900.0),
        )),
        ..Default::default()
    };
    crate::test_support::run_frame_headless(ctx, raw, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            panel.show(ui, state);
        });
    });
}

/// At the width Measurement View has in the default layout of a 1280 x 720
/// window, about 340 points, the two endpoint blocks are drawn one above the
/// other, each inside the panel, and neither over the other; wide enough for
/// two, they are side by side. In both modes.
#[test]
fn the_endpoint_blocks_never_overlap() {
    let (mut state, id) = measured_state(Some("m"));
    state.select_measurement(MeasurementRef::distance(id, 1));
    for editing in [false, true] {
        if editing {
            state
                .put_measurement_on_bench(MeasurementRef::distance(id, 1))
                .unwrap();
        }
        let mut panel = MeasurementView::new();
        let ctx = egui::Context::default();
        for _ in 0..2 {
            run_narrow(&mut panel, &ctx, &state, 340.0);
        }
        let a = panel.block_rect(EndpointSide::A).expect("A was drawn");
        let b = panel.block_rect(EndpointSide::B).expect("B was drawn");
        assert!(
            !a.intersects(b),
            "editing {editing}: {a:?} and {b:?} overlap"
        );
        assert!(a.max.x <= 340.0 && b.max.x <= 340.0, "{a:?} {b:?}");
        assert!(a.max.y <= b.min.y, "editing {editing}: A is not above B");

        for _ in 0..2 {
            run_narrow(&mut panel, &ctx, &state, 1000.0);
        }
        let a = panel.block_rect(EndpointSide::A).unwrap();
        let b = panel.block_rect(EndpointSide::B).unwrap();
        assert!(a.max.x <= b.min.x, "editing {editing}: not side by side");
        assert_eq!(a.min.y, b.min.y);
    }
}

// ── Switching an endpoint's image ───────────────────────────────────────

/// The stored mode has no *Switch image…*; the edited mode has one in each
/// block, enabled for an endpoint that resolves and greyed, with the
/// refusal as its hover text, for one refused or not placed, and for both
/// while a task holds the node.
#[test]
fn switch_image_is_greyed_where_there_is_nothing_to_offer() {
    let (mut state, id) = measured_state(Some("m"));
    state.select_measurement(MeasurementRef::distance(id, 2));
    let (panel, _ctx) = settled(&state);
    assert!(panel.switch_button(EndpointSide::A).is_none());

    // Distance #2's B is a pixel no square holds.
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 2))
        .unwrap();
    let (panel, _ctx) = settled(&state);
    assert_eq!(panel.switch_button(EndpointSide::A).unwrap().1, None);
    let refused = panel
        .switch_button(EndpointSide::B)
        .unwrap()
        .1
        .clone()
        .unwrap();
    assert!(
        refused.starts_with(
            "Endpoint B has no position, so no other image can be offered: no observation's \
             square in image 0 holds pixel"
        ),
        "{refused}"
    );

    // Deleting image 1 clears distance #1's B.
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 1))
        .unwrap();
    state.delete_image(ImageRef::new(id, 1)).unwrap();
    let (panel, _ctx) = settled(&state);
    assert_eq!(
        panel.switch_button(EndpointSide::B).unwrap().1.as_deref(),
        Some("Endpoint B is not placed, so it has no image to switch from.")
    );

    use crate::background::{Finished, Operation};
    state
        .start_background_task(
            Operation::BENCH_FIT,
            id,
            Box::new(move |_| Finished::Cancelled),
        )
        .expect("nothing else is running");
    let busy = state.busy_refusal(id).unwrap();
    let (panel, _ctx) = settled(&state);
    assert_eq!(panel.switch_button(EndpointSide::A).unwrap().1, Some(busy));
}

/// A right-click on an endpoint block of the item selects nothing and opens
/// its menu, whose *Switch Image…* opens the popup; a click on the popup's
/// row asks for the switch to that image and pixel, once, and closes it.
#[test]
fn the_switch_popup_opens_from_the_menu_and_a_row_asks_for_the_switch() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 1))
        .unwrap();
    let (mut panel, ctx) = settled(&state);
    let at = panel.name_rect(EndpointSide::A).unwrap().center();
    let right = clicks(
        &mut panel,
        &ctx,
        &state,
        at,
        egui::PointerButton::Secondary,
        1,
    );
    assert!(
        right
            .iter()
            .all(|r| r.select_item_endpoint.is_none() && r.look_through_item.is_none()),
        "a right click selected something"
    );
    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.iter().any(|t| t == SWITCH_ENTRY), "{texts:?}");
    assert!(panel.offer_rows().is_empty(), "the popup opened by itself");
    let entry = panel.menu_entry(EndpointSide::A).expect("the menu is open");
    clicks(
        &mut panel,
        &ctx,
        &state,
        entry.center(),
        egui::PointerButton::Primary,
        1,
    );
    run_frame(&mut panel, &ctx, &state, Vec::new());
    let rows = panel.offer_rows().to_vec();
    assert_eq!(rows.len(), 1, "point 1 has one other observation");
    let (row, image, pixel) = rows[0];
    assert_eq!(image, 1);
    let texts = painted(&mut panel, &ctx, &state);
    for expected in ["Switch endpoint A to another image", "image_001.jpg"] {
        assert!(
            texts.iter().any(|t| t == expected),
            "{expected:?} not in {texts:?}"
        );
    }

    let picked = clicks(
        &mut panel,
        &ctx,
        &state,
        row.center(),
        egui::PointerButton::Primary,
        1,
    );
    let asked: Vec<_> = picked
        .iter()
        .filter_map(|r| r.switch_endpoint.clone())
        .collect();
    assert_eq!(
        asked,
        [EndpointSwitch {
            node: id,
            item: "distance #1".to_string(),
            side: EndpointSide::A,
            image,
            pixel,
        }]
    );
    run_frame(&mut panel, &ctx, &state, Vec::new());
    assert!(panel.offer_rows().is_empty(), "the popup stayed open");
}

/// The button opens the popup too, with the point's heading and each row's
/// readings, and no mark where the endpoint lands on the row's own point.
#[test]
fn the_switch_button_opens_the_popup_with_its_readings() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    let (mut panel, ctx) = settled(&state);
    let button = panel.switch_button(EndpointSide::B).unwrap().0;
    clicks(
        &mut panel,
        &ctx,
        &state,
        button.center(),
        egui::PointerButton::Primary,
        1,
    );
    run_frame(&mut panel, &ctx, &state, Vec::new());
    let rows = panel.offer_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].1, 1);
    let texts = painted(&mut panel, &ctx, &state);
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("Point ") && t.ends_with("the point the endpoint resolves to")),
        "{texts:?}"
    );
    // The fixture stores no readings.
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("\u{2113} not read  \u{03b8} ")),
        "{texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.starts_with('\u{26a0}')),
        "{texts:?}"
    );
}

/// The readings line of a row and its mark, as the popup draws them.
#[test]
fn an_offer_reads_its_length_angle_and_mark() {
    use sfmtool_core::reconstruction::measurement_endpoint::EndpointView;
    let (state, id) = measured_state(Some("m"));
    let node = state.node(id).unwrap();
    let mut view = EndpointView {
        point: 1,
        observation: 1,
        image: 1,
        pixel: [10.0, 20.0],
        patch_coords: [0.0, 0.0],
        localization: Some(0.0123),
        viewing_angle_deg: Some(12.34),
        resolves_to: Some(1),
    };
    assert_eq!(
        offer_readings_text(&view, false, Some("m")),
        "\u{2113} 0.0123 m  \u{03b8} 12.3\u{00b0}"
    );
    assert_eq!(
        offer_readings_text(&view, true, Some("m")),
        "\u{2113} 0.0123\u{00b0}  \u{03b8} 12.3\u{00b0}"
    );
    assert_eq!(offer_mark(node, &view), None);
    view.resolves_to = Some(2);
    let other = crate::scene::point_id(node, 2);
    assert_eq!(
        offer_mark(node, &view),
        Some(format!("\u{26a0} resolves to point {other} there"))
    );
    view.resolves_to = None;
    assert_eq!(
        offer_mark(node, &view).as_deref(),
        Some("\u{26a0} no position there")
    );
}

// ── An endpoint's image and pixel fields ────────────────────────────────

/// Click at `at`, select what the field there holds and type `text` over it,
/// returning the response of every frame.
fn type_at(
    panel: &mut MeasurementView,
    ctx: &egui::Context,
    state: &AppState,
    at: egui::Pos2,
    text: &str,
) -> Vec<MeasurementViewResponse> {
    let mut out = clicks(panel, ctx, state, at, egui::PointerButton::Primary, 1);
    out.push(run_frame(
        panel,
        ctx,
        state,
        vec![
            key(egui::Key::A, egui::Modifiers::COMMAND),
            egui::Event::Text(text.to_string()),
        ],
    ));
    out
}

fn pixel_field(panel: &MeasurementView, side: EndpointSide, axis: PixelAxis) -> egui::Pos2 {
    panel
        .pixel_field_rect(side, axis)
        .expect("the field was drawn")
        .center()
}

/// Every endpoint set asked for over `responses`.
fn sets(responses: &[MeasurementViewResponse]) -> Vec<crate::bench::measurement::EndpointSet> {
    responses
        .iter()
        .filter_map(|r| r.set_endpoint.clone())
        .collect()
}

/// The image and pixel of endpoint `side` of the focused `distance #0`.
fn held(state: &AppState, id: ReconId, side: EndpointSide) -> (u32, [f64; 2]) {
    let item = state.bench_measurement(id, "distance #0").unwrap();
    let endpoint = item.as_distance().unwrap().endpoints[side.index()].unwrap();
    (endpoint.image, endpoint.pixel)
}

fn set_of(
    id: ReconId,
    side: EndpointSide,
    image: u32,
    pixel: [f64; 2],
) -> crate::bench::measurement::EndpointSet {
    crate::bench::measurement::EndpointSet {
        node: id,
        item: "distance #0".to_string(),
        side,
        image,
        pixel,
    }
}

/// Typing an x and pressing Enter asks, once, for one step that sets both
/// the image the endpoint holds and the pixel with that x; the fields show
/// the stored pixel as the shortest text that reads back as it.
#[test]
fn a_pixel_field_commits_on_enter_as_one_step_that_sets_both() {
    let (mut state, id) = measured_state(Some("m"));
    let (mut panel, ctx) = editing(&mut state, id);
    let (image, [_, y]) = held(&state, id, EndpointSide::A);
    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.iter().any(|t| *t == pixel_text(y)), "{texts:?}");
    let at = pixel_field(&panel, EndpointSide::A, PixelAxis::X);
    let typed = type_at(&mut panel, &ctx, &state, at, "100.5");
    assert!(sets(&typed).is_empty(), "typing alone committed");
    let entered = run_frame(
        &mut panel,
        &ctx,
        &state,
        vec![key(egui::Key::Enter, egui::Modifiers::default())],
    );
    assert_eq!(
        entered.set_endpoint,
        Some(set_of(id, EndpointSide::A, image, [100.5, y]))
    );
    let after = run_frame(&mut panel, &ctx, &state, Vec::new());
    assert_eq!(after.set_endpoint, None, "the commit was reported twice");
}

/// Typing a y and clicking elsewhere commits it, on endpoint B.
#[test]
fn a_pixel_field_commits_when_it_loses_focus() {
    let (mut state, id) = measured_state(Some("m"));
    let (mut panel, ctx) = editing(&mut state, id);
    let (image, [x, _]) = held(&state, id, EndpointSide::B);
    let at = pixel_field(&panel, EndpointSide::B, PixelAxis::Y);
    type_at(&mut panel, &ctx, &state, at, "12");
    let away = clicks(
        &mut panel,
        &ctx,
        &state,
        egui::pos2(500.0, 890.0),
        egui::PointerButton::Primary,
        1,
    );
    assert_eq!(sets(&away), [set_of(id, EndpointSide::B, image, [x, 12.0])]);
}

/// Escape puts a pixel field back and asks for nothing, then or when the
/// focus moves on.
#[test]
fn escape_puts_a_pixel_field_back() {
    let (mut state, id) = measured_state(Some("m"));
    let (mut panel, ctx) = editing(&mut state, id);
    let at = pixel_field(&panel, EndpointSide::A, PixelAxis::X);
    let mut responses = type_at(&mut panel, &ctx, &state, at, "77");
    responses.push(run_frame(
        &mut panel,
        &ctx,
        &state,
        vec![key(egui::Key::Escape, egui::Modifiers::default())],
    ));
    responses.extend(clicks(
        &mut panel,
        &ctx,
        &state,
        egui::pos2(500.0, 890.0),
        egui::PointerButton::Primary,
        1,
    ));
    assert!(sets(&responses).is_empty(), "{:?}", sets(&responses));
    let texts = painted(&mut panel, &ctx, &state);
    assert!(!texts.iter().any(|t| t == "77"), "{texts:?}");
}

/// Typing an x and clicking straight into the y field commits the x once,
/// and the y field takes no part of it; the y typed then commits on its own.
#[test]
fn moving_from_the_x_field_to_the_y_field_commits_the_x_once() {
    let (mut state, id) = measured_state(Some("m"));
    let (mut panel, ctx) = editing(&mut state, id);
    let (image, [x, y]) = held(&state, id, EndpointSide::A);
    let x_at = pixel_field(&panel, EndpointSide::A, PixelAxis::X);
    let y_at = pixel_field(&panel, EndpointSide::A, PixelAxis::Y);
    let mut responses = type_at(&mut panel, &ctx, &state, x_at, "101");
    responses.extend(type_at(&mut panel, &ctx, &state, y_at, "202"));
    assert_eq!(
        sets(&responses),
        [set_of(id, EndpointSide::A, image, [101.0, y])]
    );
    let entered = run_frame(
        &mut panel,
        &ctx,
        &state,
        vec![key(egui::Key::Enter, egui::Modifiers::default())],
    );
    // No step was made between the two (the dock makes them), so the x set
    // with the y is the one the item still holds.
    assert_eq!(
        entered.set_endpoint,
        Some(set_of(id, EndpointSide::A, image, [x, 202.0]))
    );
}

/// Text that is not a number asks for no step and is reported in a sentence
/// naming the endpoint, the item and the text.
#[test]
fn a_pixel_that_is_not_a_number_is_refused() {
    let (mut state, id) = measured_state(Some("m"));
    let (mut panel, ctx) = editing(&mut state, id);
    let at = pixel_field(&panel, EndpointSide::B, PixelAxis::X);
    let mut responses = type_at(&mut panel, &ctx, &state, at, "abc");
    responses.push(run_frame(
        &mut panel,
        &ctx,
        &state,
        vec![key(egui::Key::Enter, egui::Modifiers::default())],
    ));
    assert!(sets(&responses).is_empty());
    let refused: Vec<String> = responses
        .iter()
        .filter_map(|r| r.endpoint_refused.clone())
        .collect();
    assert_eq!(
        refused,
        ["Cannot move endpoint B of distance #0: \"abc\" is not a number."]
    );
}

/// The image drop-down lists the reconstruction's images; picking another
/// asks for one step that sets that image and keeps the pixel, and picking
/// the one it holds asks for nothing.
#[test]
fn the_image_drop_down_sets_the_image_and_keeps_the_pixel() {
    let (mut state, id) = measured_state(Some("m"));
    let (mut panel, ctx) = editing(&mut state, id);
    let (image, pixel) = held(&state, id, EndpointSide::A);
    assert_eq!(image, 0);
    let open = |panel: &mut MeasurementView| {
        let at = panel
            .image_field_rect(EndpointSide::A)
            .expect("the drop-down was drawn")
            .center();
        clicks(panel, &ctx, &state, at, egui::PointerButton::Primary, 1);
        run_frame(panel, &ctx, &state, Vec::new());
        panel.image_choices(EndpointSide::A).to_vec()
    };
    let choices = open(&mut panel);
    assert_eq!(
        choices.iter().map(|(_, i)| *i).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    let same = clicks(
        &mut panel,
        &ctx,
        &state,
        choices[0].0.center(),
        egui::PointerButton::Primary,
        1,
    );
    assert!(
        sets(&same).is_empty(),
        "the image it holds asked for a step"
    );

    let choices = open(&mut panel);
    let picked = clicks(
        &mut panel,
        &ctx,
        &state,
        choices[2].0.center(),
        egui::PointerButton::Primary,
        1,
    );
    assert_eq!(sets(&picked), [set_of(id, EndpointSide::A, 2, pixel)]);
}

/// While a task holds the node the drop-down opens no list and the pixel
/// fields take no typing, so neither asks for a step.
#[test]
fn the_endpoint_fields_are_greyed_while_a_step_would_be_refused() {
    use crate::background::{Finished, Operation};
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    state
        .start_background_task(
            Operation::BENCH_FIT,
            id,
            Box::new(move |_| Finished::Cancelled),
        )
        .expect("nothing else is running");
    let (mut panel, ctx) = settled(&state);
    let at = pixel_field(&panel, EndpointSide::A, PixelAxis::X);
    let mut responses = type_at(&mut panel, &ctx, &state, at, "100");
    responses.push(run_frame(
        &mut panel,
        &ctx,
        &state,
        vec![key(egui::Key::Enter, egui::Modifiers::default())],
    ));
    let combo = panel.image_field_rect(EndpointSide::A).unwrap().center();
    responses.extend(clicks(
        &mut panel,
        &ctx,
        &state,
        combo,
        egui::PointerButton::Primary,
        1,
    ));
    run_frame(&mut panel, &ctx, &state, Vec::new());
    assert!(sets(&responses).is_empty());
    assert!(panel.image_choices(EndpointSide::A).is_empty());
}

// ── A new measurement ───────────────────────────────────────────────────

/// A measurement started from a pixel shows endpoint B's block with the
/// instructions that place it, both header fields empty with their grey
/// text, the result saying B is missing, and *Commit* greyed with the
/// refusal, until B and the distance are given.
#[test]
fn a_new_measurement_says_how_to_finish_it() {
    let (mut state, id) = measured_state(Some("m"));
    let label = state
        .start_measurement(ImageRef::new(id, 0), [100.0, 50.0])
        .unwrap();
    let (mut panel, ctx) = settled(&state);
    let texts = painted(&mut panel, &ctx, &state);
    let instruction = place_instruction(EndpointSide::B);
    assert_eq!(
        instruction,
        "Right-click a pixel in Image Detail, or a point in the 3D view, and choose Set \
         Endpoint B Here."
    );
    for expected in [
        "Endpoint B is not placed.",
        instruction.as_str(),
        "No label",
        "No stated distance",
        "No distance: endpoint B is not placed.",
        "From no stored row: Commit adds it as a new row",
    ] {
        assert!(
            texts.iter().any(|t| t == expected),
            "{expected:?} not in {texts:?}"
        );
    }
    assert!(panel.field_rect(HeaderField::Distance).is_some());
    let commit = &panel.toolbar()[0];
    assert_eq!(
        commit.1.as_deref(),
        Some(
            format!(
                "Cannot commit {label}: endpoint B is not placed, and a distance needs both \
                 endpoints."
            )
            .as_str()
        )
    );

    // B placed: its block has fields and no instructions, and Commit asks for
    // the distance.
    state.set_endpoint(&crate::bench::measurement::EndpointSet {
        node: id,
        item: label.clone(),
        side: EndpointSide::B,
        image: 1,
        pixel: [300.0, 200.5],
    });
    let texts = painted(&mut panel, &ctx, &state);
    assert!(!texts.contains(&instruction), "{texts:?}");
    run_frame(&mut panel, &ctx, &state, Vec::new());
    assert!(panel
        .pixel_field_rect(EndpointSide::B, PixelAxis::X)
        .is_some());
    assert_eq!(
        panel.toolbar()[0].1.as_deref(),
        Some(format!("Cannot commit {label}: no stated distance is given.").as_str())
    );
    state
        .set_bench_measurement_distance(id, &label, "42 cm")
        .unwrap();
    run_frame(&mut panel, &ctx, &state, Vec::new());
    assert_eq!(panel.toolbar()[0].1, None);
}

/// A pixel field is wide enough for the shortest text that reads back as the
/// `float32` it holds, so a ten-character coordinate is not cut off.
#[test]
fn a_pixel_field_fits_its_coordinate() {
    let (mut state, id) = measured_state(Some("m"));
    let (_, ctx) = editing(&mut state, id);
    let (image, [_, y]) = held(&state, id, EndpointSide::A);
    state
        .set_bench_measurement_endpoint(
            id,
            "distance #0",
            EndpointSide::A,
            image,
            [117.847664, y],
            crate::bench::measurement::EndpointMove::Set,
        )
        .unwrap();
    let value = pixel_text(117.847664);
    assert_eq!(value, "117.847664");
    let (mut panel, ctx) = (MeasurementView::new(), ctx);
    for _ in 0..2 {
        run_frame(&mut panel, &ctx, &state, Vec::new());
    }
    let field = panel
        .pixel_field_rect(EndpointSide::A, PixelAxis::X)
        .expect("the field was drawn");
    let mut text_width = 0.0;
    crate::test_support::run_frame_headless(&ctx, input(Vec::new()), |ui| {
        let font = egui::TextStyle::Body.resolve(ui.style());
        text_width = ui.fonts_mut(|fonts| {
            fonts
                .layout_no_wrap(value.clone(), font, egui::Color32::WHITE)
                .size()
                .x
        });
    });
    // The text, inside the field's margin of 4 points on each side.
    assert!(
        field.width() >= text_width + 8.0,
        "the field is {} wide for {text_width} of text",
        field.width()
    );
    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.contains(&value), "{texts:?}");
}

/// A field takes typing, and commits it on Enter, while the window does not
/// have the operating system's focus, as when an agent types into a viewer
/// behind another window; and an empty field, as a new measurement's stated
/// distance and label are, takes it as one holding text does.
#[test]
fn a_field_takes_typing_while_the_window_lacks_focus() {
    let (mut state, id) = measured_state(Some("m"));
    let label = state
        .start_measurement(ImageRef::new(id, 0), [100.0, 50.0])
        .unwrap();
    let (mut panel, ctx) = settled(&state);
    let unfocused = |panel: &mut MeasurementView, events: Vec<egui::Event>| {
        let mut response = None;
        let raw = egui::RawInput {
            focused: false,
            ..input(events)
        };
        crate::test_support::run_frame_headless(&ctx, raw, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                response = Some(panel.show(ui, &state));
            });
        });
        response.expect("the panel ran")
    };
    for (field, text) in [
        (HeaderField::Distance, "42 cm"),
        (HeaderField::Label, "door"),
    ] {
        let at = panel
            .field_rect(field)
            .expect("the field was drawn")
            .center();
        let mut responses = vec![
            unfocused(&mut panel, vec![egui::Event::PointerMoved(at)]),
            unfocused(
                &mut panel,
                vec![button(at, egui::PointerButton::Primary, true)],
            ),
            unfocused(
                &mut panel,
                vec![button(at, egui::PointerButton::Primary, false)],
            ),
            unfocused(&mut panel, vec![egui::Event::Text(text.to_string())]),
        ];
        assert_eq!(any_distance(&responses), None, "typing alone committed");
        responses.push(unfocused(
            &mut panel,
            vec![key(egui::Key::Enter, egui::Modifiers::default())],
        ));
        let committed = match field {
            HeaderField::Distance => any_distance(&responses),
            HeaderField::Label => any_label(&responses),
        };
        assert_eq!(
            committed,
            Some((id, label.clone(), text.to_string())),
            "{field:?}"
        );
    }
}

/// The galley painted for `text` in one frame of the panel.
fn painted_galley(
    panel: &mut MeasurementView,
    ctx: &egui::Context,
    state: &AppState,
    text: &str,
) -> Option<std::sync::Arc<egui::Galley>> {
    fn find(shape: &egui::Shape, text: &str) -> Option<std::sync::Arc<egui::Galley>> {
        match shape {
            egui::Shape::Text(shape) if shape.galley.text() == text => Some(shape.galley.clone()),
            egui::Shape::Vec(shapes) => shapes.iter().find_map(|s| find(s, text)),
            _ => None,
        }
    }
    let mut output = ctx.run_ui(input(Vec::new()), |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            panel.show(ui, state);
        });
    });
    output.textures_delta.clear();
    output.shapes.iter().find_map(|c| find(&c.shape, text))
}

/// Endpoint B's instruction wraps as a paragraph does where the blocks are side
/// by side: each glyph follows the one before it by that glyph's own advance,
/// with no gap spread between the words to fill the column.
#[test]
fn an_unplaced_endpoint_s_instruction_is_not_justified() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .start_measurement(ImageRef::new(id, 0), [100.0, 50.0])
        .unwrap();
    let (mut panel, ctx) = settled(&state);
    let side_by_side = panel.block_rect(EndpointSide::A).unwrap().top()
        == panel.block_rect(EndpointSide::B).unwrap().top();
    assert!(side_by_side, "the blocks are in two columns at this width");
    let instruction = place_instruction(EndpointSide::B);
    let galley = painted_galley(&mut panel, &ctx, &state, &instruction).expect("drawn");
    assert!(galley.rows.len() > 1, "the instruction wraps in its column");
    for row in &galley.rows {
        for pair in row.glyphs.windows(2) {
            let gap = pair[1].pos.x - (pair[0].pos.x + pair[0].advance_width);
            assert!(
                gap.abs() < 1.5,
                "a gap of {gap} after {:?} in {:?}",
                pair[0].chr,
                instruction
            );
        }
    }
}
