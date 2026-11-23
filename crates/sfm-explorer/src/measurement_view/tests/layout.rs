// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests of Measurement View's layout across panel widths: nothing
//! is wider than the panel, and the endpoint blocks stack where two do not
//! fit and sit side by side where they do. The panel is drawn inside a
//! scroll area that scrolls both ways, as egui_dock draws every tab body, so
//! a row wider than the panel widens the content as it does in the viewer.

use super::super::*;
use crate::state::AppState;

/// The widths the panel is drawn at, in points, with whether the endpoint
/// blocks sit side by side there.
const WIDTHS: [(f32, bool); 5] = [
    (200.0, false),
    (272.0, false),
    (428.0, false),
    (500.0, true),
    (900.0, true),
];

/// The measurement fixture with long image names, in a node with a long
/// label, as a file of a real dataset has.
fn long_named_state() -> (AppState, ReconId) {
    let mut recon = crate::measurements::fixture::measured_reconstruction(Some("m"));
    for image in &mut recon.image_set.images {
        image.name = format!(
            "a_long_directory_name/{}_with_a_long_suffix.jpg",
            image.name
        );
    }
    let mut state = AppState::new();
    let node = crate::scene::SceneNode::from_path(
        std::path::Path::new("seoul_bull_sculpture_ground_truth_v02.sfmr"),
        recon,
    );
    let id = state.append_node(node);
    (state, id)
}

/// One frame of the panel in a window `width` points wide, inside a scroll
/// area that scrolls both ways. The part of the content the scroll area
/// shows comes back.
fn dock_frame(
    panel: &mut MeasurementView,
    ctx: &egui::Context,
    state: &AppState,
    width: f32,
) -> egui::Rect {
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(width, 900.0),
        )),
        ..Default::default()
    };
    let mut shown = egui::Rect::NOTHING;
    crate::test_support::run_frame_headless(ctx, raw, |ui| {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                shown = egui::ScrollArea::both()
                    .show(ui, |ui| panel.show(ui, state))
                    .inner_rect;
            });
    });
    shown
}

/// What is wrong with the panel's layout at each width, drawn from `state`:
/// a widget reaching past the panel's right edge, or the blocks side by side
/// where they should stack or stacked where they should sit side by side.
fn layout_faults(state: &AppState) -> Vec<String> {
    let mut faults = Vec::new();
    for (width, side_by_side) in WIDTHS {
        let mut panel = MeasurementView::new();
        let ctx = egui::Context::default();
        let mut shown = egui::Rect::NOTHING;
        for _ in 0..2 {
            shown = dock_frame(&mut panel, &ctx, state, width);
        }
        let widest = ctx.viewport(|viewport| {
            viewport
                .prev_pass
                .widgets
                .layers()
                .flat_map(|(_, rects)| rects.iter().map(|w| w.rect.max.x))
                .fold(f32::NEG_INFINITY, f32::max)
        });
        if widest > shown.max.x + 0.5 {
            faults.push(format!(
                "at {width}: a widget reaches x = {widest}, past the panel's edge at {}",
                shown.max.x
            ));
        }
        let (Some(a), Some(b)) = (
            panel.block_rect(EndpointSide::A),
            panel.block_rect(EndpointSide::B),
        ) else {
            continue;
        };
        let beside = a.max.x <= b.min.x && a.min.y == b.min.y;
        let stacked = a.max.y <= b.min.y;
        if side_by_side && !beside {
            faults.push(format!("at {width}: {a:?} and {b:?} are not side by side"));
        }
        if !side_by_side && !stacked {
            faults.push(format!("at {width}: {a:?} and {b:?} are not stacked"));
        }
    }
    faults
}

/// A stored measurement, read only, and the panel with nothing selected:
/// the header, the blocks and the list of measurements fit at every width.
#[test]
fn the_read_only_panel_fits_every_width() {
    let (mut state, id) = long_named_state();
    let faults = layout_faults(&state);
    assert!(faults.is_empty(), "with nothing selected: {faults:#?}");
    state.select_measurement(MeasurementRef::distance(id, 1));
    let faults = layout_faults(&state);
    assert!(faults.is_empty(), "{faults:#?}");
}

/// An item with both endpoints placed: the toolbar, the header's fields and
/// the blocks' image drop-downs and pixel fields fit at every width.
#[test]
fn the_edited_panel_fits_every_width() {
    let (mut state, id) = long_named_state();
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 1))
        .unwrap();
    let faults = layout_faults(&state);
    assert!(faults.is_empty(), "{faults:#?}");
}

/// A new item with endpoint B not placed, whose bench label names the long
/// image name: B's block with its instruction fits at every width.
#[test]
fn an_unplaced_endpoint_fits_every_width() {
    let (mut state, id) = long_named_state();
    state
        .start_measurement(ImageRef::new(id, 0), [100.0, 50.0])
        .unwrap();
    let faults = layout_faults(&state);
    assert!(faults.is_empty(), "{faults:#?}");
}
