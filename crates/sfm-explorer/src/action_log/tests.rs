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

// ── Revisions ───────────────────────────────────────────────────────────

/// The revision of every entry the log holds, oldest first.
fn revisions(log: &ActionLog) -> Vec<u64> {
    log.entries().map(|entry| entry.revision).collect()
}

#[test]
fn every_record_ticks_the_clock_and_stamps_the_entry() {
    let mut log = log();
    assert_eq!(log.revision(), 0, "a fresh log has not written anything");
    for i in 0..3 {
        log.record_at(at(i as f64 * 2.0), Kind::File, false, format!("Opened {i}"));
    }
    assert_eq!(revisions(&log), [1, 2, 3]);
    assert_eq!(log.revision(), 3);
}

/// A fold is a *change* to an entry the agent may already have read, so it
/// takes a new revision — the newest of the log — rather than keeping the one
/// the replaced entry had.
#[test]
fn a_coalescing_replacement_takes_a_fresh_revision() {
    let mut log = log();
    log.record_at(at(0.0), Kind::File, false, "Opened a");
    log.record_at(at(0.1), Kind::Selection, false, "Selected image a");
    log.record_at(at(0.5), Kind::Selection, false, "Selected image b");
    assert_eq!(texts(&log), ["Opened a", "Selected image b"]);
    assert_eq!(
        revisions(&log),
        [1, 3],
        "the fold kept the replaced revision"
    );
    assert_eq!(log.revision(), 3);
}

#[test]
fn since_returns_exactly_the_entries_above_it() {
    let mut log = log();
    for i in 0..4 {
        log.record_at(at(i as f64 * 2.0), Kind::File, false, format!("Opened {i}"));
    }
    let after_two: Vec<&str> = log.since(2).map(|entry| entry.text.as_str()).collect();
    assert_eq!(after_two, ["Opened 2", "Opened 3"], "oldest first");
    assert_eq!(log.since(0).count(), 4, "since 0 is the whole log");
    assert_eq!(
        log.since(log.revision()).count(),
        0,
        "a reader that is up to date is told nothing"
    );
    // A fold brings an entry the reader had already seen back into view.
    log.record_at(at(8.0), Kind::Selection, false, "Selected image a");
    let mark = log.revision();
    log.record_at(at(8.5), Kind::Selection, false, "Selected image b");
    let after_mark: Vec<&str> = log.since(mark).map(|entry| entry.text.as_str()).collect();
    assert_eq!(after_mark, ["Selected image b"]);
}

#[test]
fn oldest_revision_follows_the_entries_that_are_still_held() {
    let mut log = log();
    assert_eq!(
        log.oldest_revision(),
        log.revision(),
        "an empty log is as old as it is new"
    );
    for i in 0..ActionLog::CAPACITY + 2 {
        log.record_at(at(i as f64 * 2.0), Kind::File, false, format!("entry {i}"));
    }
    assert_eq!(
        log.oldest_revision(),
        3,
        "two entries dropped, so the third is the oldest held"
    );
    // Clear empties the buffer and leaves the clock alone, so a reader holding
    // an older revision can still tell that it missed everything.
    let before = log.revision();
    log.clear();
    assert_eq!(log.revision(), before, "Clear rewound the clock");
    assert_eq!(log.oldest_revision(), before);
    log.record_at(at(0.0), Kind::File, false, "Opened again");
    assert_eq!(revisions(&log), [before + 1]);
}

// ── The wire vocabulary ─────────────────────────────────────────────────

/// Every kind has a wire name, and no two of them share one — the compiler
/// guarantees the first through an exhaustive match, and this guarantees the
/// second, which it cannot.
#[test]
fn every_kind_and_actor_has_a_distinct_wire_name() {
    let kinds = [
        Kind::Session,
        Kind::File,
        Kind::Selection,
        Kind::Scene,
        Kind::View,
        Kind::Display,
        Kind::Animation,
        Kind::Layout,
        Kind::Window,
        Kind::Query("get_scene"),
    ];
    let names: std::collections::BTreeSet<&str> =
        kinds.iter().map(|kind| kind.wire_name()).collect();
    assert_eq!(names.len(), kinds.len(), "two kinds share a wire name");
    assert!(names
        .iter()
        .all(|name| name.chars().all(|c| c.is_ascii_lowercase() || c == '_')));
    // Every query is one wire kind, whichever tool it came from: the tool
    // travels beside the kind so that `kind` stays a closed vocabulary.
    assert_eq!(Kind::Query("screenshot").wire_name(), "query");
    assert_eq!(Kind::Query("get_scene").wire_name(), "query");

    let actors: std::collections::BTreeSet<&str> =
        Actor::ALL.iter().map(|actor| actor.wire_name()).collect();
    assert_eq!(actors.len(), Actor::ALL.len());
    for actor in Actor::ALL {
        assert_eq!(Actor::from_wire_name(actor.wire_name()), Some(actor));
    }
    assert_eq!(Actor::from_wire_name("User"), None, "the names are exact");
    assert_eq!(Actor::all_wire_names(), "user, mcp, viewer");
}

/// The wire's timestamp is the panel's, in the panel's zone, so a time an agent
/// reads is the time the human beside it is reading.
#[test]
fn a_wire_timestamp_is_rfc_3339_in_the_logs_own_zone() {
    let log = log();
    assert_eq!(
        log.format_rfc3339(at(0.25)),
        "2026-09-01T14:04:03.250-07:00"
    );
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
