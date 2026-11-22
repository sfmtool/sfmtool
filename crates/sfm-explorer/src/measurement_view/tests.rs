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
        "Label: ear to ear",
        "Stated distance: 153 cm",
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
