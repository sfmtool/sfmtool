// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the Edit History panel.
//!
//! egui lays a frame out with no GPU behind it, so `show` really does walk the
//! versions and paint every row; the assertions read the strings it painted
//! (`test_support::painted_texts`) and the response it hands the dock.

use jiff::tz::TimeZone;
use sfmtool_core::SfmrReconstruction;

use super::{format_bytes, show, CURSOR_MARK, DISK_MARK};
use crate::action_log::ActionLog;
use crate::scene::{PointRef, ReconId, SceneNode};
use crate::state::AppState;
use crate::test_support::painted_texts;

/// A state holding one demo node, selected, logging in a fixed zone so a row's
/// time is the same string on every machine.
fn state() -> AppState {
    let mut state = AppState::new();
    state.action_log = ActionLog::with_zone(TimeZone::fixed(jiff::tz::offset(-7)));
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(32)));
    state
}

/// The id of the one node.
fn node(state: &AppState) -> ReconId {
    state.selected_recon.expect("a selected reconstruction")
}

/// Every string one frame of the panel painted.
fn texts(state: &AppState) -> Vec<String> {
    let ctx = egui::Context::default();
    painted_texts(&ctx, egui::RawInput::default(), |ui| {
        show(ui, state);
    })
}

/// Whether any painted string contains `needle`.
fn painted(texts: &[String], needle: &str) -> bool {
    texts.iter().any(|text| text.contains(needle))
}

#[test]
fn with_no_node_selected_the_panel_says_so() {
    let state = AppState::new();
    let texts = texts(&state);
    assert!(painted(&texts, "No reconstruction loaded"), "{texts:?}");
}

#[test]
fn a_node_that_has_not_been_edited_lists_its_one_version_and_says_so() {
    let state = state();
    let texts = texts(&state);
    assert!(painted(&texts, "1 version"), "{texts:?}");
    assert!(painted(&texts, "has not been edited"), "{texts:?}");
    assert!(painted(&texts, "Opened demo"), "{texts:?}");
}

#[test]
fn the_rows_are_the_versions_oldest_first_with_the_cursor_and_the_disk_state_marked() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 1))
        .expect("a live point");
    state
        .delete_point(PointRef::new(id, 2))
        .expect("a live point");

    let texts = texts(&state);
    let rows: Vec<&String> = texts
        .iter()
        .filter(|text| text.contains("Opened demo") || text.contains("Deleted point"))
        .collect();
    assert_eq!(rows.len(), 3, "{texts:?}");
    // Oldest first: the load, then the two edits in the order they were made.
    assert!(rows[0].contains("Opened demo"), "{rows:?}");
    assert!(rows[1].contains("Deleted point 1"), "{rows:?}");
    assert!(rows[2].contains("Deleted point 2"), "{rows:?}");
    // The cursor is on the newest; the disk state is still the loaded version.
    assert!(rows[2].starts_with(CURSOR_MARK), "{rows:?}");
    assert!(rows[0].contains(DISK_MARK), "{rows:?}");
    assert!(!rows[2].contains(DISK_MARK), "{rows:?}");
    // Each row carries a size.
    assert!(rows
        .iter()
        .all(|row| row.contains(" B") || row.contains("iB")));
}

#[test]
fn an_undo_moves_the_mark_the_panel_draws() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 1))
        .expect("a live point");
    state.undo(id).expect("one edit to undo");

    let texts = texts(&state);
    let loaded = texts
        .iter()
        .find(|text| text.contains("Opened demo"))
        .expect("the loaded row");
    assert!(loaded.starts_with(CURSOR_MARK), "{texts:?}");
    assert!(loaded.contains(DISK_MARK), "{texts:?}");
}

#[test]
fn a_released_row_still_lists_and_says_it_was_released() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 1))
        .expect("a live point");
    state
        .delete_point(PointRef::new(id, 2))
        .expect("a live point");
    state.scene[0].history.versions_mut_for_test()[1].value = None;

    let texts = texts(&state);
    let row = texts
        .iter()
        .find(|text| text.contains("Deleted point 1"))
        .expect("the released row");
    assert!(row.contains("(released)"), "{texts:?}");
}

#[test]
fn clicking_a_row_asks_for_that_version_and_the_jump_moves_the_cursor() {
    let mut state = state();
    let id = node(&state);
    state
        .delete_point(PointRef::new(id, 1))
        .expect("a live point");
    let loaded = state.scene[0].history.versions()[0].serial;

    // The frame is run twice: the first lays the rows out, the second sends a
    // click at the rect the first gave the loaded version's row.
    let ctx = egui::Context::default();
    let mut rect = None;
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |ui| {
        show(ui, &state);
        rect = Some(ui.min_rect());
    });
    let row = rect.expect("a laid-out panel");
    // The rows run down the panel under the header; the first is the loaded
    // version, so a point a little way into the list's top row is on it.
    let at = egui::pos2(row.left() + 12.0, row.top() + 44.0);
    let input = egui::RawInput {
        events: vec![
            egui::Event::PointerMoved(at),
            egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::default(),
            },
            egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            },
        ],
        ..Default::default()
    };
    let mut response = super::EditHistoryResponse::default();
    let mut output = ctx.run_ui(input, |ui| {
        response = show(ui, &state);
    });
    output.textures_delta.clear();

    let (asked_id, serial) = response.jump.expect("the click reported a row");
    assert_eq!(asked_id, id);
    assert_eq!(serial, loaded);

    state.jump_to_version(id, serial).expect("a live version");
    assert_eq!(state.scene[0].history.cursor(), 0);
    assert!(!state.scene[0].is_point_deleted(1));
    let last = state
        .action_log
        .entries()
        .last()
        .expect("an entry")
        .text
        .clone();
    assert!(last.starts_with("Go to: Opened demo ("), "{last}");
}

#[test]
fn sizes_are_written_for_a_human() {
    assert_eq!(format_bytes(0), "0 B");
    assert_eq!(format_bytes(512), "512 B");
    assert_eq!(format_bytes(2048), "2.00 KiB");
    assert_eq!(format_bytes(20 * 1024), "20.0 KiB");
    assert_eq!(format_bytes(700 * 1024), "700 KiB");
    assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.00 GiB");
}
