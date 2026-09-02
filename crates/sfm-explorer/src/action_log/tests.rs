// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the buffer's rules and the panel that shows them.
//!
//! Every instant here is fixed and every zone is fixed: what a row reads, and
//! whether two entries fold into one, are properties of the log rather than of
//! the machine and the moment the tests happen to run on.

use jiff::tz::{Offset, TimeZone};
use jiff::Timestamp;

use super::{show, ActionLog, Actor, Kind};

/// A log in a fixed zone, seven hours behind UTC, so a formatted row is the
/// same string wherever the tests run.
fn log() -> ActionLog {
    ActionLog::with_zone(TimeZone::fixed(Offset::constant(-7)))
}

/// `2026-09-01 14:04:03 -07:00`, plus `seconds`.
fn at(seconds: f64) -> Timestamp {
    let base: Timestamp = "2026-09-01T21:04:03Z".parse().expect("a valid instant");
    base + jiff::SignedDuration::from_millis((seconds * 1000.0).round() as i64)
}

/// Every entry's text, oldest first.
fn texts(log: &ActionLog) -> Vec<&str> {
    log.entries().map(|entry| entry.text.as_str()).collect()
}

// ── The buffer ──────────────────────────────────────────────────────────

#[test]
fn past_capacity_the_oldest_entry_goes_and_is_counted() {
    let mut log = log();
    // `File` so nothing coalesces and the count is exactly what was recorded.
    for i in 0..ActionLog::CAPACITY + 3 {
        log.record_at(at(i as f64), Kind::File, false, format!("entry {i}"));
    }
    assert_eq!(log.len(), ActionLog::CAPACITY);
    assert_eq!(log.dropped(), 3);
    assert_eq!(
        log.entries().next().expect("a first entry").text,
        "entry 3",
        "the oldest surviving entry is not the fourth"
    );
}

#[test]
fn two_like_entries_inside_the_window_become_one() {
    let mut log = log();
    log.record_at(at(0.0), Kind::Selection, false, "Selected image a");
    log.record_at(at(0.5), Kind::Selection, false, "Selected image b");
    assert_eq!(texts(&log), ["Selected image b"]);
    assert_eq!(
        log.entries().next().expect("an entry").at,
        at(0.5),
        "the surviving entry kept the older timestamp"
    );
}

#[test]
fn two_like_entries_outside_the_window_stay_two() {
    let mut log = log();
    log.record_at(at(0.0), Kind::Selection, false, "Selected image a");
    log.record_at(at(1.5), Kind::Selection, false, "Selected image b");
    assert_eq!(texts(&log), ["Selected image a", "Selected image b"]);
}

/// The window is measured from the entry being *replaced*, which is itself the
/// time of the last replacement — so an unbroken run folds indefinitely even
/// though its ends are further apart than the window.
#[test]
fn an_unbroken_run_coalesces_however_long_it_lasts() {
    let mut log = log();
    for (i, t) in [0.0, 0.8, 1.6].into_iter().enumerate() {
        log.record_at(at(t), Kind::Selection, false, format!("Selected image {i}"));
    }
    assert_eq!(texts(&log), ["Selected image 2"]);
}

#[test]
fn a_discrete_kind_never_coalesces() {
    let mut log = log();
    log.record_at(at(0.0), Kind::File, false, "Opened a");
    log.record_at(at(0.1), Kind::File, false, "Opened b");
    assert_eq!(texts(&log), ["Opened a", "Opened b"]);
}

#[test]
fn a_failure_is_never_coalesced_away_in_either_direction() {
    let mut log = log();
    log.record_at(at(0.0), Kind::Selection, true, "select_point failed: no");
    log.record_at(at(0.1), Kind::Selection, false, "Selected point p");
    log.record_at(at(0.2), Kind::Selection, true, "select_point failed: no");
    assert_eq!(
        texts(&log),
        [
            "select_point failed: no",
            "Selected point p",
            "select_point failed: no"
        ]
    );
}

#[test]
fn different_actors_do_not_coalesce() {
    let mut log = log();
    log.record_at(at(0.0), Kind::Selection, false, "Selected image a");
    log.set_actor(Actor::Mcp);
    log.record_at(at(0.1), Kind::Selection, false, "Selected image b");
    assert_eq!(texts(&log), ["Selected image a", "Selected image b"]);
}

#[test]
fn queries_coalesce_per_tool_and_not_across_tools() {
    let mut log = log();
    log.record_at(at(0.0), Kind::Query("screenshot"), false, "screenshot 8×8");
    log.record_at(at(0.2), Kind::Query("screenshot"), false, "screenshot 9×9");
    assert_eq!(texts(&log), ["screenshot 9×9"]);
    log.record_at(at(0.4), Kind::Query("get_scene"), false, "get_scene");
    assert_eq!(texts(&log), ["screenshot 9×9", "get_scene"]);
}

// ── The status line ─────────────────────────────────────────────────────

#[test]
fn the_status_line_is_empty_until_something_happens() {
    assert_eq!(log().status_line(), None);
}

#[test]
fn the_status_line_skips_queries_and_prefixes_the_agent() {
    let mut log = log();
    log.set_actor(Actor::Mcp);
    log.record_at(at(0.0), Kind::Scene, false, "Soloed beta");
    log.record_at(at(1.0), Kind::Query("get_scene"), false, "get_scene");
    assert_eq!(log.status_line().as_deref(), Some("MCP: Soloed beta"));

    log.set_actor(Actor::User);
    log.record_at(at(2.0), Kind::Scene, false, "Soloed alpha");
    assert_eq!(log.status_line().as_deref(), Some("Soloed alpha"));
}

/// A refusal reaches the status line whichever tool it came from: only a
/// *successful* read is kept off it.
#[test]
fn the_status_line_shows_a_failed_query() {
    let mut log = log();
    log.set_actor(Actor::Mcp);
    log.record_at(at(0.0), Kind::Scene, false, "Soloed beta");
    log.record_at(
        at(1.0),
        Kind::Query("get_camera_image"),
        true,
        "get_camera_image failed: no such image",
    );
    assert_eq!(
        log.status_line().as_deref(),
        Some("MCP: get_camera_image failed: no such image")
    );
}

// ── Muting ──────────────────────────────────────────────────────────────

#[test]
fn mute_nests() {
    let mut log = log();
    log.mute();
    log.mute();
    log.unmute();
    log.record_at(at(0.0), Kind::File, false, "Opened a");
    assert_eq!(log.len(), 0, "an outstanding mute still recorded");
    log.unmute();
    log.record_at(at(1.0), Kind::File, false, "Opened b");
    assert_eq!(texts(&log), ["Opened b"]);
}

// ── The clipboard export ────────────────────────────────────────────────

#[test]
fn the_clipboard_text_carries_the_date_the_actor_and_the_failures() {
    let mut log = log();
    log.set_actor(Actor::Mcp);
    log.record_at(at(0.0), Kind::Scene, false, "text");
    log.record_at(at(0.0), Kind::File, true, "text");
    assert_eq!(
        log.to_clipboard_text(),
        "2026-09-01 14:04:03  MCP     text\n2026-09-01 14:04:03  MCP   ! text\n"
    );
}

// ── The panel ───────────────────────────────────────────────────────────

/// One headless frame of the panel, and everything it painted.
fn painted(log: &mut ActionLog) -> Vec<String> {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(600.0, 400.0),
        )),
        ..Default::default()
    };
    crate::test_support::painted_texts(&ctx, input, |ui| show(ui, log))
}

#[test]
fn the_rows_paint_oldest_first_as_time_actor_and_text() {
    let mut log = log();
    log.record_at(at(0.0), Kind::File, false, "Opened alpha");
    log.set_actor(Actor::Mcp);
    log.record_at(at(61.0), Kind::File, false, "Opened beta");

    let texts = painted(&mut log);
    let index = |needle: &str| {
        texts
            .iter()
            .position(|text| text == needle)
            .unwrap_or_else(|| panic!("{needle:?} was not painted, only {texts:?}"))
    };
    assert!(index("14:04:03") < index("14:05:04"), "{texts:?}");
    assert!(index("Opened alpha") < index("Opened beta"), "{texts:?}");
    assert!(texts.iter().any(|t| t == "User"), "{texts:?}");
    assert!(texts.iter().any(|t| t == "MCP"), "{texts:?}");
    assert!(texts.iter().any(|t| t == "2 entries"), "{texts:?}");
}

/// Every galley the panel painted, with the width it was laid out to.
///
/// The strings alone cannot answer the elision question: an `egui` galley keeps
/// the *whole* job text however few glyphs it drew, so what says a row was
/// truncated is its width, not its text.
fn painted_widths(log: &mut ActionLog, panel: egui::Vec2) -> Vec<(String, f32)> {
    fn walk(shape: &egui::Shape, out: &mut Vec<(String, f32)>) {
        match shape {
            egui::Shape::Text(text) => {
                out.push((text.galley.text().to_owned(), text.galley.rect.width()))
            }
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, out)),
            _ => {}
        }
    }
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, panel)),
        ..Default::default()
    };
    let mut output = ctx.run_ui(input, |ui| show(ui, log));
    output.textures_delta.clear();
    let mut out = Vec::new();
    for clipped in &output.shapes {
        walk(&clipped.shape, &mut out);
    }
    out
}

/// A row never wraps: the list is virtualized on a uniform row height, so a
/// long text is truncated to the width it was given — and truncating it must
/// not panic the frame or spill the row past the panel.
#[test]
fn a_text_wider_than_the_panel_is_elided() {
    let mut log = log();
    let long = "Aligned ".to_string() + &"very-long-label ".repeat(40);
    log.record_at(at(0.0), Kind::Scene, false, long.clone());
    let panel = egui::vec2(600.0, 400.0);
    let (_, width) = painted_widths(&mut log, panel)
        .into_iter()
        .find(|(text, _)| *text == long)
        .expect("the row was not painted at all");
    assert!(
        width <= panel.x,
        "the row was laid out {width}px wide in a {}px panel",
        panel.x,
    );
}

#[test]
fn clear_empties_the_buffer_and_the_next_frame_paints_no_rows() {
    let mut log = log();
    log.record_at(at(0.0), Kind::File, false, "Opened alpha");
    log.clear();
    assert_eq!(log.len(), 0);
    assert_eq!(log.status_line(), None);
    let texts = painted(&mut log);
    assert!(
        !texts.iter().any(|text| text == "Opened alpha"),
        "a cleared entry was still painted: {texts:?}"
    );
}
