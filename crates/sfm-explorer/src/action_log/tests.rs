// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the buffer's rules and the panel that shows them.
//!
//! Every instant here is fixed and every zone is fixed: what a row reads, and
//! whether two entries fold into one, are properties of the log rather than of
//! the machine and the moment the tests happen to run on.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use jiff::tz::{Offset, TimeZone};
use jiff::Timestamp;

use sfmtool_core::progress::Level;
use sfmtool_core::SfmrReconstruction;

use crate::scene::{PointRef, ReconId};
use crate::state::AppState;

use super::{show, ActionLog, Actor, Entry, Kind, Run, Work};
use crate::progress::{Collector, Detail};
use crate::test_support::{assert_timed_from_the_work, phase_note, phase_rows};

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

/// The runs the tests below record successive values of, spelled as the record
/// sites spell them: two selection slots and two HUD controls.
const IMAGE: Run = Some("image");
const POINT: Run = Some("point");
const POINT_SIZE: Run = Some("Point size");
const GRID: Run = Some("Grid");
const CAMERA: Run = Some("camera");

/// [`ActionLog::query`] at a fixed instant: the tool names both the kind and
/// the run, which is what makes a poll one row.
trait QueryAt {
    fn query_at(&mut self, at: Timestamp, tool: &'static str, text: &str);
}

impl QueryAt for ActionLog {
    fn query_at(&mut self, at: Timestamp, tool: &'static str, text: &str) {
        self.record_at(at, Kind::Query(tool), Some(tool), false, text);
    }
}

// ── The buffer ──────────────────────────────────────────────────────────

#[test]
fn past_capacity_the_oldest_entry_goes_and_is_counted() {
    let mut log = log();
    // `File` so nothing coalesces and the count is exactly what was recorded.
    for i in 0..ActionLog::CAPACITY + 3 {
        log.record_at(at(i as f64), Kind::File, None, false, format!("entry {i}"));
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
    log.record_at(at(0.0), Kind::Selection, IMAGE, false, "Selected image a");
    log.record_at(at(0.5), Kind::Selection, IMAGE, false, "Selected image b");
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
    log.record_at(at(0.0), Kind::Selection, IMAGE, false, "Selected image a");
    log.record_at(at(1.5), Kind::Selection, IMAGE, false, "Selected image b");
    assert_eq!(texts(&log), ["Selected image a", "Selected image b"]);
}

/// The window is measured from the entry being *replaced*, which is itself the
/// time of the last replacement — so an unbroken run folds indefinitely even
/// though its ends are further apart than the window.
#[test]
fn an_unbroken_run_coalesces_however_long_it_lasts() {
    let mut log = log();
    for (i, t) in [0.0, 0.8, 1.6].into_iter().enumerate() {
        log.record_at(
            at(t),
            Kind::Selection,
            IMAGE,
            false,
            format!("Selected image {i}"),
        );
    }
    assert_eq!(texts(&log), ["Selected image 2"]);
}

/// An entry with no run is a discrete act: it neither folds into the line
/// above it nor is folded over by the line below, whatever the kind.
#[test]
fn an_entry_with_no_run_never_coalesces_in_either_direction() {
    let mut log = log();
    log.record_at(at(0.0), Kind::File, None, false, "Opened a");
    log.record_at(at(0.1), Kind::File, None, false, "Opened b");
    assert_eq!(texts(&log), ["Opened a", "Opened b"]);

    // A run, a discrete entry of the same kind and actor inside the window,
    // then the run again: three lines, and the deselection is not folded into
    // the selection it undid.
    log.clear();
    log.record_at(at(0.0), Kind::Selection, IMAGE, false, "Selected image a");
    log.record_at(at(0.1), Kind::Selection, None, false, "Deselected image");
    log.record_at(at(0.2), Kind::Selection, IMAGE, false, "Selected image b");
    assert_eq!(
        texts(&log),
        ["Selected image a", "Deselected image", "Selected image b"]
    );
}

/// **The run, not the kind, decides.** Two entries of one kind fold only when
/// they are successive values of the same control, slot or tool; two different
/// ones are two acts however close together they arrived.
#[test]
fn the_run_and_not_the_kind_decides_a_fold() {
    // Two selection slots that both still hold.
    let mut log = log();
    log.record_at(at(0.0), Kind::Selection, IMAGE, false, "Selected image a");
    log.record_at(at(0.1), Kind::Selection, POINT, false, "Selected point 7");
    assert_eq!(texts(&log), ["Selected image a", "Selected point 7"]);

    // Two HUD controls, and then one of them twice.
    log.clear();
    log.record_at(at(0.0), Kind::Display, POINT_SIZE, false, "Point size 2.0");
    log.record_at(at(0.1), Kind::Display, GRID, false, "Grid off");
    assert_eq!(texts(&log), ["Point size 2.0", "Grid off"]);
    log.record_at(at(0.2), Kind::Display, POINT_SIZE, false, "Point size 3.0");
    log.record_at(at(0.3), Kind::Display, POINT_SIZE, false, "Point size 4.0");
    assert_eq!(
        texts(&log),
        ["Point size 2.0", "Grid off", "Point size 4.0"]
    );

    // A view an agent drives through values, and a deliberate framing.
    log.clear();
    log.record_at(at(0.0), Kind::View, CAMERA, false, "Camera placed");
    log.record_at(at(0.1), Kind::View, CAMERA, false, "Camera placed");
    assert_eq!(texts(&log), ["Camera placed"]);
    log.record_at(at(0.2), Kind::View, None, false, "Framed the scene");
    assert_eq!(texts(&log), ["Camera placed", "Framed the scene"]);
}

#[test]
fn a_failure_is_never_coalesced_away_in_either_direction() {
    let mut log = log();
    log.record_at(
        at(0.0),
        Kind::Selection,
        POINT,
        true,
        "select_point failed: no",
    );
    log.record_at(at(0.1), Kind::Selection, POINT, false, "Selected point p");
    log.record_at(
        at(0.2),
        Kind::Selection,
        POINT,
        true,
        "select_point failed: no",
    );
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
    log.record_at(at(0.0), Kind::Selection, IMAGE, false, "Selected image a");
    log.set_actor(Actor::Mcp);
    log.record_at(at(0.1), Kind::Selection, IMAGE, false, "Selected image b");
    assert_eq!(texts(&log), ["Selected image a", "Selected image b"]);
}

#[test]
fn queries_coalesce_per_tool_and_not_across_tools() {
    let mut log = log();
    for t in [0.0, 0.2] {
        log.query_at(at(t), "get_scene", "get_scene");
    }
    assert_eq!(texts(&log), ["get_scene"]);
    log.query_at(at(0.4), "get_point", "get_point #7");
    assert_eq!(texts(&log), ["get_scene", "get_point #7"]);
}

/// A screenshot is not a value being scrubbed through: it is a picture the
/// agent took and presumably looked at, so the reader is told how many were
/// taken and of what.
#[test]
fn three_screenshots_in_a_second_are_three_rows() {
    let mut log = log();
    for (i, t) in [0.0, 0.2, 0.4].into_iter().enumerate() {
        log.record_at(
            at(t),
            Kind::Query("screenshot"),
            None,
            false,
            format!("screenshot viewer_3d {i}×{i}"),
        );
    }
    assert_eq!(
        texts(&log),
        [
            "screenshot viewer_3d 0×0",
            "screenshot viewer_3d 1×1",
            "screenshot viewer_3d 2×2"
        ]
    );
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
        log.record_at(
            at(i as f64 * 2.0),
            Kind::File,
            None,
            false,
            format!("Opened {i}"),
        );
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
    log.record_at(at(0.0), Kind::File, None, false, "Opened a");
    log.record_at(at(0.1), Kind::Selection, IMAGE, false, "Selected image a");
    log.record_at(at(0.5), Kind::Selection, IMAGE, false, "Selected image b");
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
        log.record_at(
            at(i as f64 * 2.0),
            Kind::File,
            None,
            false,
            format!("Opened {i}"),
        );
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
    log.record_at(at(8.0), Kind::Selection, IMAGE, false, "Selected image a");
    let mark = log.revision();
    log.record_at(at(8.5), Kind::Selection, IMAGE, false, "Selected image b");
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
        log.record_at(
            at(i as f64 * 2.0),
            Kind::File,
            None,
            false,
            format!("entry {i}"),
        );
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
    log.record_at(at(0.0), Kind::File, None, false, "Opened again");
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
    log.record_at(at(0.0), Kind::Scene, None, false, "Soloed beta");
    log.record_at(at(1.0), Kind::Query("get_scene"), None, false, "get_scene");
    assert_eq!(log.status_line().as_deref(), Some("MCP: Soloed beta"));

    log.set_actor(Actor::User);
    log.record_at(at(2.0), Kind::Scene, None, false, "Soloed alpha");
    assert_eq!(log.status_line().as_deref(), Some("Soloed alpha"));
}

/// A refusal reaches the status line whichever tool it came from: only a
/// *successful* read is kept off it.
#[test]
fn the_status_line_shows_a_failed_query() {
    let mut log = log();
    log.set_actor(Actor::Mcp);
    log.record_at(at(0.0), Kind::Scene, None, false, "Soloed beta");
    log.record_at(
        at(1.0),
        Kind::Query("get_camera_image"),
        None,
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
    log.record_at(at(0.0), Kind::File, None, false, "Opened a");
    assert_eq!(log.len(), 0, "an outstanding mute still recorded");
    log.unmute();
    log.record_at(at(1.0), Kind::File, None, false, "Opened b");
    assert_eq!(texts(&log), ["Opened b"]);
}

// ── The clipboard export ────────────────────────────────────────────────

#[test]
fn the_clipboard_text_carries_the_date_the_actor_the_cost_and_the_failures() {
    let mut log = log();
    log.set_actor(Actor::Mcp);
    log.record_at(at(0.0), Kind::Scene, None, false, "text");
    log.record_at(at(0.0), Kind::File, None, true, "text");
    // No frame has been drawn, so the cost column is blank rather than zero:
    // these actions have not finished being waited on.
    assert_eq!(
        log.to_clipboard_text(),
        "2026-09-01 14:04:03  MCP              text\n\
         2026-09-01 14:04:03  MCP   !          text\n"
    );

    // A frame whose upload phase ran after both writes settles both.
    log.settle(std::time::Instant::now(), Vec::new());
    assert_eq!(
        log.to_clipboard_text(),
        "2026-09-01 14:04:03  MCP       <1 ms  text\n\
         2026-09-01 14:04:03  MCP   !   <1 ms  text\n"
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
    log.record_at(at(0.0), Kind::File, None, false, "Opened alpha");
    log.set_actor(Actor::Mcp);
    log.record_at(at(61.0), Kind::File, None, false, "Opened beta");

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
    log.record_at(at(0.0), Kind::Scene, None, false, long.clone());
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
    log.record_at(at(0.0), Kind::File, None, false, "Opened alpha");
    log.clear();
    assert_eq!(log.len(), 0);
    assert_eq!(log.status_line(), None);
    let texts = painted(&mut log);
    assert!(
        !texts.iter().any(|text| text == "Opened alpha"),
        "a cleared entry was still painted: {texts:?}"
    );
}

// ── What an action cost ─────────────────────────────────────────────────

/// The entry at `index`, or a panic naming what was there instead.
fn took_of(log: &ActionLog, index: usize) -> Option<std::time::Duration> {
    log.get(index).expect("an entry at that index").took
}

#[test]
fn a_frame_times_the_entries_its_upload_phase_had_already_seen() {
    let mut log = log();
    log.record_at(at(0.0), Kind::File, None, false, "before the uploads");
    let uploads_began = std::time::Instant::now();
    log.record_at(at(1.0), Kind::View, None, false, "after the uploads");

    log.settle(uploads_began, Vec::new());

    assert!(
        took_of(&log, 0).is_some(),
        "an entry written before the upload phase has been drawn by now"
    );
    assert_eq!(
        took_of(&log, 1),
        None,
        "one written after it has not: its uploads are the next frame's"
    );

    // And the next frame, whose upload phase is later still, picks it up.
    log.settle(std::time::Instant::now(), Vec::new());
    assert!(took_of(&log, 1).is_some(), "the next frame settles it");
}

#[test]
fn a_settled_entry_keeps_the_cost_it_was_given() {
    let mut log = log();
    log.record_at(at(0.0), Kind::File, None, false, "once");
    log.settle(std::time::Instant::now(), Vec::new());
    let first = took_of(&log, 0).expect("settled");

    // Later frames have nothing to settle and must not re-time what is done:
    // the number is the wait that happened, not the age of the row.
    log.settle(std::time::Instant::now(), Vec::new());
    assert_eq!(took_of(&log, 0), Some(first));
}

#[test]
fn a_run_that_folds_carries_the_timing_of_the_row_that_survived() {
    let mut log = log();
    log.record_at(at(0.0), Kind::Display, Some("Scene scale"), false, "0.1");
    log.record_at(at(0.1), Kind::Display, Some("Scene scale"), false, "0.2");
    assert_eq!(log.len(), 1, "the drag is one row");

    log.settle(std::time::Instant::now(), Vec::new());

    assert_eq!(texts(&log), ["0.2"]);
    assert!(
        took_of(&log, 0).is_some(),
        "the surviving row is timed, from the value that replaced the other"
    );
}

#[test]
fn the_cost_column_reads_in_the_unit_the_question_is_asked_in() {
    use std::time::Duration;
    assert_eq!(ActionLog::format_took(Duration::from_micros(400)), "<1 ms");
    assert_eq!(ActionLog::format_took(Duration::from_millis(4)), "4 ms");
    assert_eq!(ActionLog::format_took(Duration::from_millis(990)), "990 ms");
    assert_eq!(
        ActionLog::format_took(Duration::from_millis(1240)),
        "1.24 s"
    );
}

#[test]
fn an_untimed_entry_costs_nothing_to_hold_and_is_bounded() {
    let mut log = log();
    // Nothing calls `settle` here, which is every headless use of the log.
    for i in 0..(ActionLog::CAPACITY.min(3_000)) {
        log.record_at(at(i as f64), Kind::View, None, false, format!("{i}"));
    }
    assert!(
        log.pending_len() <= 1_024,
        "the queue of writes waiting to be timed has to be bounded: {}",
        log.pending_len(),
    );
}

#[test]
fn a_settled_row_paints_what_it_cost_and_an_unsettled_one_paints_nothing() {
    let mut log = log();
    log.record_at(at(0.0), Kind::Edit, None, false, "Deleted point 7");
    log.settle(std::time::Instant::now(), Vec::new());
    log.record_at(at(1.0), Kind::Edit, None, false, "Deleted point 8");

    let texts = painted(&mut log);

    assert!(
        texts.iter().any(|text| text == "<1 ms"),
        "the settled row shows its cost, only {texts:?}",
    );
    assert_eq!(
        texts.iter().filter(|text| text.ends_with("ms")).count(),
        1,
        "the row that has not been drawn yet claims no cost: {texts:?}",
    );
    assert!(
        texts.iter().any(|text| text == "Deleted point 8"),
        "{texts:?}"
    );
}

// -- The detail an operation reports -------------------------------------

/// An entry built by hand, for the arithmetic [`ActionLog::elsewhere`] does.
/// The clock is not involved, so what the sum comes to is a property of the
/// function rather than of the machine.
fn entry_with(took: Duration, detail: Vec<Detail>) -> Entry {
    Entry {
        revision: 1,
        at: at(0.0),
        actor: Actor::User,
        kind: Kind::Edit,
        run: None,
        failed: false,
        text: "Bundle adjusted run_a".to_string(),
        took: Some(took),
        detail,
    }
}

/// A phase row, for the entries built by hand above.
fn phase(name: &'static str, depth: u8, ms: u64, cpu_ms: Option<u64>) -> Detail {
    Detail::Phase {
        name,
        depth,
        took: Duration::from_millis(ms),
        cpu: cpu_ms.map(Duration::from_millis),
        note: None,
        note_last: None,
        runs: 1,
    }
}

/// A message row, for the entries built by hand above.
fn message(level: Level, depth: u8, text: &str) -> Detail {
    Detail::Message {
        level,
        depth,
        text: text.to_string(),
    }
}

/// Every phase row of the newest entry, as `(name, depth, runs)`.
fn detail_phases(log: &ActionLog) -> Vec<(&'static str, u8, u32)> {
    phase_rows(&log.entries().next_back().expect("an entry").detail)
}

#[test]
fn an_operations_collector_becomes_that_entrys_detail_and_no_others() {
    let mut log = log();
    let collector = Collector::new(false);
    drop(collector.phase("materialise"));
    drop(collector.phase("push version"));

    log.record_done(
        Kind::Edit,
        Instant::now(),
        "Bundle adjusted run_a",
        collector.take(),
    );
    log.record_at(at(1.0), Kind::Edit, None, false, "Deleted point 7");

    let entries: Vec<&Entry> = log.entries().collect();
    assert_eq!(
        phase_rows(&entries[0].detail),
        [("materialise", 0, 1), ("push version", 0, 1)],
    );
    assert!(
        entries[1].detail.is_empty(),
        "the next entry inherited the operation's detail: {:?}",
        entries[1].detail,
    );
}

/// `started` is when the work began, not when the row was written, so a long
/// operation reports what it cost rather than what installing its row cost.
#[test]
fn record_done_times_from_when_the_work_began() {
    let mut log = log();
    let started = Instant::now()
        .checked_sub(Duration::from_millis(120))
        .expect("a clock with 120 ms behind it");
    log.record_done(Kind::Edit, started, "Bundle adjusted run_a", Vec::new());
    log.settle(Instant::now(), Vec::new());

    let took = log.entries().next_back().expect("an entry").took;
    assert!(
        took.is_some_and(|took| took >= Duration::from_millis(120)),
        "the row was timed from the write rather than from the work: {took:?}",
    );
}

/// Folding is the collector's and happens as the events arrive, so the cap
/// counts rows kept rather than phases opened.
#[test]
fn folding_happens_before_the_cap() {
    let mut log = log();
    let collector = Collector::new(false);
    {
        let solve = collector.phase("solve");
        for _ in 0..600 {
            drop(solve.phase("linearise"));
        }
    }

    log.record_done(
        Kind::Edit,
        Instant::now(),
        "Bundle adjusted run_a",
        collector.take(),
    );

    assert_eq!(
        detail_phases(&log),
        [("solve", 0, 1), ("linearise", 1, 600)],
        "a phase in a long loop truncated the entry instead of folding",
    );
}

#[test]
fn past_the_cap_the_entry_keeps_the_first_events_and_says_how_many_went() {
    let mut log = log();
    let collector = Collector::new(false);
    let progress = collector.progress();
    for i in 0..ActionLog::DETAIL_EVENTS + 17 {
        // Messages, since two phases of one name under one parent would fold.
        sfmtool_core::progress_info!(progress, "line {i}");
    }

    log.record_done(
        Kind::Edit,
        Instant::now(),
        "Bundle adjusted run_a",
        collector.take(),
    );

    let detail = &log.entries().next_back().expect("an entry").detail;
    assert_eq!(
        detail.len(),
        ActionLog::DETAIL_EVENTS + 1,
        "the cap, plus the line that says what went",
    );
    let Detail::Message { text, .. } = &detail[0] else {
        panic!("the first row is not a message");
    };
    assert_eq!(
        text, "line 0",
        "the entry kept the tail rather than the head"
    );
    let Detail::Message { text, .. } = detail.last().expect("a last row") else {
        panic!("the tally is not a message");
    };
    assert_eq!(text, "17 more events dropped");
}

/// A folded row is the newest value of the run, so it carries what that value
/// reported and not what the value it replaced did.
#[test]
fn a_fold_takes_the_new_values_detail_and_drops_the_replaced_ones() {
    let mut log = log();
    let now = Instant::now();
    log.write(
        at(0.0),
        Kind::Display,
        POINT_SIZE,
        false,
        "Point size 3",
        Work {
            started: now,
            detail: vec![phase("first", 0, 1, None)],
        },
    );
    log.write(
        at(0.5),
        Kind::Display,
        POINT_SIZE,
        false,
        "Point size 4",
        Work {
            started: now,
            detail: vec![phase("second", 0, 1, None)],
        },
    );

    assert_eq!(texts(&log), ["Point size 4"]);
    assert_eq!(detail_phases(&log), [("second", 0, 1)]);
}

// -- The frame's detail, shared by the entries it settled ----------------

/// What a frame reports: one upload phase with a child, and a draw.
///
/// Through a real [`Collector`], since the point is that what the frame's
/// collector hands `settle` is what the entries end up carrying.
fn frame_events() -> Vec<Detail> {
    let frame = Collector::new(false);
    {
        let uploads = frame.phase("uploads");
        drop(uploads.phase("points"));
    }
    drop(frame.phase("scene render"));
    frame.take()
}

/// The message rows of one entry's detail, in order.
/// What [`frame_events`] reads as on an entry: its own row, a level above the
/// stages it gathered.
fn framed() -> Vec<(&'static str, u8, u32)> {
    vec![
        (ActionLog::OVERHEAD, 0, 1),
        ("uploads", 1, 1),
        ("points", 2, 1),
        ("scene render", 1, 1),
    ]
}

/// The note on an entry's `frame` row, which is where it says how many other
/// entries were waiting for the same one.
fn frame_note(detail: &[Detail]) -> Option<&str> {
    detail.iter().find_map(|row| match row {
        Detail::Phase {
            name: ActionLog::OVERHEAD,
            note,
            ..
        } => note.as_deref(),
        _ => None,
    })
}

fn message_texts(detail: &[Detail]) -> Vec<&str> {
    detail
        .iter()
        .filter_map(|row| match row {
            Detail::Message { text, .. } => Some(text.as_str()),
            Detail::Phase { .. } => None,
        })
        .collect()
}

#[test]
fn settle_gives_the_frames_events_to_the_entry_it_stamps() {
    let mut log = log();
    log.record_at(at(0.0), Kind::Edit, None, false, "Deleted point 7");

    log.settle(Instant::now(), frame_events());

    let entry = log.entries().next_back().expect("an entry");
    assert_eq!(
        phase_rows(&entry.detail),
        framed(),
        "the upload and the draw that showed this entry are not on it",
    );
    assert_eq!(
        message_texts(&entry.detail),
        Vec::<&str>::new(),
        "an entry that had the frame to itself claimed it was shared",
    );
}

/// The frame belongs to the action whose effect it uploaded, which is the last
/// one recorded before the uploads began. An earlier entry that merely waited
/// through the same frame is not a second copy of that work: a log of a startup
/// where three rows were pending would otherwise read as though the file had
/// been opened three times.
#[test]
fn one_frame_goes_to_the_entry_that_caused_it_and_not_to_the_ones_that_waited() {
    let mut log = log();
    log.record_at(at(0.0), Kind::Session, None, false, "SfM Explorer started");
    log.record_at(
        at(0.1),
        Kind::File,
        None,
        false,
        "Opened dino_dog_toy-embedded",
    );

    log.settle(Instant::now(), frame_events());

    let entries: Vec<&Entry> = log.entries().collect();
    assert!(
        entries[0].detail.is_empty(),
        "an entry that only waited claimed the frame's work: {:?}",
        entries[0].detail,
    );
    assert!(
        entries[0].took.is_some(),
        "the entry that waited lost its own cost",
    );
    assert_eq!(
        phase_rows(&entries[1].detail),
        framed(),
        "the entry that caused the frame did not get it",
    );
    assert_eq!(
        frame_note(&entries[1].detail),
        Some("also settled 1 earlier entry"),
    );
}

#[test]
fn a_frame_that_stamps_nothing_discards_its_events() {
    let mut log = log();
    // Written after this frame's upload phase, which is every click: it waits
    // for the next frame, and this one has nobody to charge.
    let uploads_began = Instant::now();
    log.record_at(at(0.0), Kind::Edit, None, false, "Deleted point 7");

    log.settle(uploads_began, frame_events());
    log.settle(Instant::now(), Vec::new());

    let entry = log.entries().next_back().expect("an entry");
    assert!(entry.took.is_some(), "the next frame settles it");
    assert!(
        entry.detail.is_empty(),
        "an entry inherited a frame that did not stamp it: {:?}",
        entry.detail,
    );
}

/// An entry reads as a transcript, so what it did itself comes before what the
/// frame that showed it spent.
#[test]
fn an_entry_carries_its_own_detail_and_then_the_frames() {
    let mut log = log();
    let collector = Collector::new(false);
    drop(collector.phase("materialise"));
    drop(collector.phase("push version"));
    log.record_done(
        Kind::Edit,
        Instant::now(),
        "Bundle adjusted run_a",
        collector.take(),
    );

    log.settle(Instant::now(), frame_events());

    assert_eq!(
        detail_phases(&log),
        [
            ("materialise", 0, 1),
            ("push version", 0, 1),
            (ActionLog::OVERHEAD, 0, 1),
            ("uploads", 1, 1),
            ("points", 2, 1),
            ("scene render", 1, 1),
        ],
    );
}

/// The frame's phases are the entry's, so they come out of what had no name.
#[test]
fn a_frame_leaves_less_of_the_entry_elsewhere() {
    let uploads = vec![
        phase("uploads", 0, 120, None),
        // Already inside its parent's 120 ms, and so not counted twice.
        phase("points", 1, 100, None),
    ];
    let alone = elsewhere_after(Vec::new());
    let shared = elsewhere_after(uploads);
    assert!(
        shared < alone,
        "the frame's uploads left `elsewhere` where it was: {shared:?} against {alone:?}",
    );
    // The parent's 120 ms and not its child's 100 ms as well, give or take
    // the clock between two runs of the same entry.
    assert!(
        (alone - shared).abs_diff(Duration::from_millis(120)) < Duration::from_millis(5),
        "a nested phase was counted a second time: {alone:?} against {shared:?}",
    );
}

/// `elsewhere` for one entry that took 400 ms, named 20 ms of it itself, and
/// was settled by a frame reporting `frame`.
fn elsewhere_after(frame: Vec<Detail>) -> Duration {
    let mut log = log();
    let started = Instant::now()
        .checked_sub(Duration::from_millis(400))
        .expect("a clock with 400 ms behind it");
    log.record_done(
        Kind::Edit,
        started,
        "Bundle adjusted run_a",
        vec![phase("push version", 0, 20, None)],
    );
    log.settle(Instant::now(), frame);
    let entry = log.entries().next_back().expect("an entry");
    ActionLog::elsewhere(entry).expect("a settled entry with phases")
}

// -- `elsewhere`, the line that makes the breakdown add up ----------------

#[test]
fn elsewhere_is_took_minus_the_top_level_phases_and_ignores_cpu() {
    let entry = entry_with(
        Duration::from_millis(100),
        vec![
            phase("materialise", 0, 30, Some(900)),
            phase("solve", 0, 50, None),
            // A nested phase's cost is already inside its parent's.
            phase("round", 1, 50, None),
        ],
    );
    assert_eq!(
        ActionLog::elsewhere(&entry),
        Some(Duration::from_millis(20)),
    );
}

#[test]
fn elsewhere_is_none_before_the_entry_settles_and_with_no_phases() {
    let mut unsettled = entry_with(Duration::ZERO, vec![phase("solve", 0, 50, None)]);
    unsettled.took = None;
    assert_eq!(ActionLog::elsewhere(&unsettled), None);

    let unnamed = entry_with(Duration::from_millis(100), Vec::new());
    assert_eq!(ActionLog::elsewhere(&unnamed), None);
}

/// The phases and the settle read the clock at different points, and a folded
/// row sums wall times that may have overlapped, so the sum can exceed `took`.
/// Zero is a reading; a negative duration would be a panic.
#[test]
fn elsewhere_reports_zero_rather_than_a_negative() {
    let entry = entry_with(
        Duration::from_millis(10),
        vec![phase("solve", 0, 400, None)],
    );
    assert_eq!(ActionLog::elsewhere(&entry), Some(Duration::ZERO));
}

// -- The expansion set ---------------------------------------------------

#[test]
fn expansion_toggles_survives_new_entries_and_goes_with_clear() {
    let mut log = log();
    log.record_at(at(0.0), Kind::Edit, None, false, "Bundle adjusted run_a");
    let revision = log.entries().next_back().expect("an entry").revision;

    assert!(!log.is_expanded(revision), "an entry starts collapsed");
    log.toggle_expanded(revision);
    assert!(log.is_expanded(revision));

    log.record_at(at(1.0), Kind::Edit, None, false, "Deleted point 7");
    assert!(
        log.is_expanded(revision),
        "a new entry collapsed the one that was open",
    );

    log.toggle_expanded(revision);
    assert!(
        !log.is_expanded(revision),
        "toggling twice did not collapse"
    );

    log.toggle_expanded(revision);
    log.clear();
    assert!(
        !log.is_expanded(revision),
        "Clear left an expansion whose entry is gone",
    );
}

// -- End to end: one operation that really reports -----------------------

/// A state holding one node a bundle adjustment can run on, its geometry
/// nudged so the solve has something to do and a point deleted so it has
/// something to materialise.
fn adjustable_scene() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(crate::scene_graph::tests::resectable_node(
        "/runs/run_a.sfmr",
    ));
    let id = state.scene[0].id;
    state.scene[0].recon_mut().image_table.images[1].translation_xyz +=
        nalgebra::Vector3::new(0.02, -0.015, 0.01);
    state
        .delete_point(PointRef::new(id, 7))
        .expect("a live point");
    (state, id)
}

/// The wiring test: a bundle adjustment names its own three stages, the kernel
/// names its four underneath them, and the kernel's rounds fold into one row.
#[test]
fn a_headless_bundle_adjustment_records_its_stages_and_the_kernels() {
    let (mut state, id) = adjustable_scene();

    state
        .bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
        .expect("the fixture is well posed");

    let entry = state
        .action_log
        .entries()
        .next_back()
        .expect("the adjustment's entry");
    assert!(!entry.failed, "{}", entry.text);
    let names: Vec<&'static str> = phase_rows(&entry.detail)
        .into_iter()
        .map(|(name, _, _)| name)
        .collect();
    assert_eq!(
        names,
        [
            "materialise",
            "gather arrays",
            "residuals before",
            "solve",
            "round",
            "write back",
            "row map",
            "push version",
        ],
        "{:?}",
        entry.detail,
    );
    let (depth, runs) = phase_rows(&entry.detail)
        .into_iter()
        .find(|(name, _, _)| *name == "round")
        .map(|(_, depth, runs)| (depth, runs))
        .expect("the kernel's rounds");
    assert_eq!(depth, 1, "the rounds did not nest under the solve");
    assert!(runs >= 1, "the rounds folded into nothing");
}

// -- Coverage: every operation names its stages --------------------------

/// A directory of this test's own under the system temp dir, emptied first so
/// a rerun does not read a previous run's file.
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sfm_explorer_progress_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a writable temp dir");
    dir
}

/// A `.sfmr` file in `dir`, and the workspace marker an open resolves against.
///
/// The marker is what makes the file openable at all: a load resolves the
/// workspace directory by searching upward for one, and a temp dir holding a
/// lone `.sfmr` refuses the read rather than guessing.
fn openable_file(dir: &Path) -> PathBuf {
    std::fs::write(dir.join(".sfm-workspace.json"), "{}").expect("a writable temp dir");
    let path = dir.join("recon.sfmr");
    SfmrReconstruction::demo(64)
        .save(&path)
        .expect("a writable temp dir");
    path
}

/// A state holding one node opened from `dir`'s file, with a point deleted, so
/// that it has a file to write over, an overlay to fold and a version to step
/// away from.
fn opened_and_edited(dir: &Path) -> (AppState, ReconId) {
    let mut state = AppState::new();
    let id = state
        .load_file(&openable_file(dir))
        .expect("the fixture file");
    state
        .delete_point(PointRef::new(id, 3))
        .expect("a live point");
    (state, id)
}

/// Overview coverage is a requirement rather than a budget, and this is where
/// it is held: every operation that can outlast a frame names its stages, so
/// that the first question about a surprising row is never unanswerable.
///
/// A table rather than a test apiece, because adding a row here is how the
/// next operation gets covered: one that forgets to name its stages fails
/// this, instead of being found much later by somebody expanding its row and
/// seeing nothing but `elsewhere`.
#[test]
fn every_operation_names_at_least_one_stage() {
    /// Drive one operation once, in a directory of its own, and hand back the
    /// state it left.
    type Drive = fn(&Path) -> AppState;

    let operations: [(&str, Drive); 7] = [
        ("open", |dir| {
            let mut state = AppState::new();
            state
                .load_file(&openable_file(dir))
                .expect("the fixture file");
            state
        }),
        ("save", |dir| {
            let (mut state, id) = opened_and_edited(dir);
            state.save_node(id).expect("a writable path");
            state
        }),
        ("save as", |dir| {
            let (mut state, id) = opened_and_edited(dir);
            state
                .save_node_as(id, &dir.join("other.sfmr"))
                .expect("a writable path");
            state
        }),
        ("undo", |dir| {
            let (mut state, id) = opened_and_edited(dir);
            state.undo(id).expect("a version to step back to");
            state
        }),
        ("redo", |dir| {
            let (mut state, id) = opened_and_edited(dir);
            state.undo(id).expect("a version to step back to");
            state.redo(id).expect("a version to step forward to");
            state
        }),
        ("go to", |dir| {
            let (mut state, id) = opened_and_edited(dir);
            let first = state.scene[0].history.versions()[0].serial;
            state
                .jump_to_version(id, first)
                .expect("a version that still holds its value");
            state
        }),
        ("bundle adjust", |_| {
            let (mut state, id) = adjustable_scene();
            state
                .bundle_adjust(id, &sfmtool_core::BundleAdjustOptions::default())
                .expect("the fixture is well posed");
            state
        }),
    ];

    for (what, drive) in operations {
        let state = drive(&temp_dir(&what.replace(' ', "_")));
        let entry = state
            .action_log
            .entries()
            .next_back()
            .expect("the operation's entry");
        assert!(!entry.failed, "{what} failed: {}", entry.text);
        assert!(
            !phase_rows(&entry.detail).is_empty(),
            "{what} named no stage, so its row expands to nothing but elsewhere",
        );
    }
}

/// What an open is made of: the file becoming a reconstruction, and the
/// reconstruction becoming a node.
#[test]
fn opening_a_file_names_the_read_and_the_append() {
    let dir = temp_dir("open_stages");
    let path = openable_file(&dir);
    let mut state = AppState::new();
    state.load_file(&path).expect("the fixture file");

    let recon = state.scene[0].recon();
    let read = format!(
        "{} points, {} images",
        recon.point_count(),
        recon.image_count()
    );
    let entry = state
        .action_log
        .entries()
        .next_back()
        .expect("the open's entry");
    assert_eq!(
        phase_rows(&entry.detail),
        [("open", 0, 1), ("read", 1, 1), ("append node", 1, 1)],
        "{:?}",
        entry.detail,
    );
    assert_eq!(
        phase_note(&entry.detail, "read"),
        Some(read),
        "the read did not say what it read",
    );
    assert_timed_from_the_work(&mut state.action_log);
}

// -- The panel's expanded rows -------------------------------------------

/// The newest entry's revision, which is what an expansion is keyed on.
fn newest(log: &ActionLog) -> u64 {
    log.entries().next_back().expect("an entry").revision
}

/// A duration in milliseconds, the unit every row below is written in.
fn ms(millis: u64) -> Duration {
    Duration::from_millis(millis)
}

/// A phase row carrying a note and a run count, the two columns [`phase`]
/// above leaves out.
fn folded(name: &'static str, depth: u8, millis: u64, note: Option<&str>, runs: u32) -> Detail {
    Detail::Phase {
        name,
        depth,
        took: ms(millis),
        cpu: None,
        note: note.map(str::to_string),
        note_last: None,
        runs,
    }
}

/// An entry whose cost is set outright rather than measured.
///
/// `settle` would time the row against the wall clock, and what an expanded
/// entry reads has to be arithmetic over fixed numbers rather than a property
/// of how long the test run took.
fn record_settled(log: &mut ActionLog, text: &str, took: Duration, detail: Vec<Detail>) {
    log.record_done(Kind::Edit, Instant::now(), text, detail);
    log.entries.back_mut().expect("the entry just written").took = Some(took);
}

/// The input one headless frame of the panel is given.
fn input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(600.0, 400.0),
        )),
        ..Default::default()
    }
}

/// Every galley one frame painted, with the colour it carried and where it
/// landed.
///
/// The strings alone answer neither of the two questions an expanded entry
/// raises, which are how far a row is indented and which colour its marker is,
/// so the shapes are walked here rather than through
/// `test_support::painted_texts`. The context is the caller's, since a click
/// has to arrive at a frame that already knows where the widgets are.
fn painted_shapes(
    ctx: &egui::Context,
    log: &mut ActionLog,
    input: egui::RawInput,
) -> Vec<(String, egui::Color32, egui::Pos2)> {
    fn walk(shape: &egui::Shape, out: &mut Vec<(String, egui::Color32, egui::Pos2)>) {
        match shape {
            egui::Shape::Text(text) => {
                out.push((text.galley.text().to_owned(), text.fallback_color, text.pos))
            }
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, out)),
            _ => {}
        }
    }
    let mut output = ctx.run_ui(input, |ui| show(ui, log));
    output.textures_delta.clear();
    let mut out = Vec::new();
    for clipped in &output.shapes {
        walk(&clipped.shape, &mut out);
    }
    out
}

/// Where `needle` was painted, and in what colour.
fn painted_at(
    shapes: &[(String, egui::Color32, egui::Pos2)],
    needle: &str,
) -> (egui::Color32, egui::Pos2) {
    shapes
        .iter()
        .find(|(text, _, _)| text == needle)
        .map(|(_, color, pos)| (*color, *pos))
        .unwrap_or_else(|| {
            let painted: Vec<&str> = shapes.iter().map(|(text, _, _)| text.as_str()).collect();
            panic!("{needle:?} was not painted, only {painted:?}")
        })
}

/// One frame carrying a primary click at `pos`.
///
/// The press and the release ride in one frame, which is a click as far as
/// egui is concerned; what matters is that a frame has already run, so that
/// the widget under `pos` is one the context knows about.
fn click(ctx: &egui::Context, log: &mut ActionLog, pos: egui::Pos2) {
    let button = |pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    let mut input = input();
    input.events = vec![egui::Event::PointerMoved(pos), button(true), button(false)];
    painted_shapes(ctx, log, input);
}

#[test]
fn an_entry_with_detail_paints_a_toggle_and_one_without_paints_none() {
    let mut log = log();
    log.record_at(at(0.0), Kind::View, None, false, "Looked at run_a");
    let bare = painted(&mut log);
    assert!(
        !bare.iter().any(|text| text == "+" || text == "-"),
        "an entry with nothing to show offered a toggle: {bare:?}",
    );

    record_settled(
        &mut log,
        "Bundle adjusted run_a",
        ms(100),
        vec![phase("solve", 0, 50, None)],
    );
    let revision = newest(&log);
    let collapsed = painted(&mut log);
    assert!(collapsed.iter().any(|text| text == "+"), "{collapsed:?}");
    assert!(!collapsed.iter().any(|text| text == "-"), "{collapsed:?}");

    log.toggle_expanded(revision);
    let expanded = painted(&mut log);
    assert!(expanded.iter().any(|text| text == "-"), "{expanded:?}");
}

/// Expansion inserts rows rather than making one row tall, which is what keeps
/// the list virtualized on a uniform row height.
#[test]
fn expanding_adds_one_row_per_event_indented_and_in_order() {
    let mut log = log();
    // Left unsettled, so there is no `elsewhere` line and the count is exactly
    // the events.
    log.record_done(
        Kind::Edit,
        Instant::now(),
        "Bundle adjusted run_a",
        vec![
            phase("materialise", 0, 30, None),
            phase("solve", 0, 50, None),
            phase("round", 1, 50, None),
        ],
    );
    let revision = newest(&log);
    let collapsed = super::panel::row_count(&log);

    log.toggle_expanded(revision);
    assert_eq!(
        super::panel::row_count(&log),
        collapsed + 3,
        "expanding did not add one row per event",
    );

    let ctx = egui::Context::default();
    let shapes = painted_shapes(&ctx, &mut log, input());
    let index = |needle: &str| {
        shapes
            .iter()
            .position(|(text, _, _)| text == needle)
            .unwrap_or_else(|| panic!("{needle:?} was not painted"))
    };
    assert!(index("materialise") < index("solve"), "out of order");
    assert!(index("solve") < index("round"), "out of order");
    let (_, top) = painted_at(&shapes, "solve");
    let (_, nested) = painted_at(&shapes, "round");
    let (_, sibling) = painted_at(&shapes, "materialise");
    assert!(nested.x > top.x, "a nested phase was not indented");
    assert_eq!(sibling.x, top.x, "two phases at one depth did not line up");

    log.toggle_expanded(revision);
    assert_eq!(super::panel::row_count(&log), collapsed);
    let again = painted(&mut log);
    assert!(
        !again.iter().any(|text| text == "materialise"),
        "collapsing left the detail behind: {again:?}",
    );
}

/// The column is not decoration: clicking it is what opens the row.
#[test]
fn clicking_the_toggle_expands_the_row_in_place() {
    let mut log = log();
    record_settled(
        &mut log,
        "Bundle adjusted run_a",
        ms(100),
        vec![phase("solve", 0, 50, None)],
    );
    let revision = newest(&log);

    let ctx = egui::Context::default();
    let shapes = painted_shapes(&ctx, &mut log, input());
    let (_, toggle) = painted_at(&shapes, "+");
    click(&ctx, &mut log, toggle + egui::vec2(3.0, 5.0));

    assert!(
        log.is_expanded(revision),
        "the click did not expand the row"
    );
}

#[test]
fn a_warn_message_paints_its_marker_in_the_error_colour() {
    let mut log = log();
    record_settled(
        &mut log,
        "Bundle adjusted run_a",
        ms(100),
        vec![
            message(Level::Warn, 0, "3 points left unsupported"),
            message(Level::Info, 0, "85 images"),
        ],
    );
    log.toggle_expanded(newest(&log));

    let ctx = egui::Context::default();
    let shapes = painted_shapes(&ctx, &mut log, input());
    let error = egui::Visuals::default().error_fg_color;
    assert_eq!(painted_at(&shapes, "!").0, error, "the warning's marker");
    assert_ne!(
        painted_at(&shapes, "\u{2022}").0,
        error,
        "an information marker wears the warning's colour",
    );
}

/// Thread-summed CPU time gets a column of its own and stays out of the
/// wall-clock arithmetic: eight seconds of CPU inside one second of wall is
/// not eight seconds of anybody's wait.
#[test]
fn a_cpu_figure_paints_in_its_own_column_and_is_absent_from_elsewhere() {
    let mut log = log();
    record_settled(
        &mut log,
        "Bundle adjusted run_a",
        ms(100),
        vec![phase("linearise", 0, 30, Some(900))],
    );
    assert_eq!(
        ActionLog::elsewhere(log.entries().next_back().expect("an entry")),
        Some(ms(70)),
        "the CPU figure was folded into the wall-clock total",
    );
    log.toggle_expanded(newest(&log));

    let ctx = egui::Context::default();
    let shapes = painted_shapes(&ctx, &mut log, input());
    let (_, cpu) = painted_at(&shapes, "cpu 900 ms");
    let (_, wall) = painted_at(&shapes, "30 ms");
    assert!(cpu.x < wall.x, "the CPU figure landed in the cost column");
}

#[test]
fn a_folded_phase_paints_its_count_and_one_that_ran_once_paints_none() {
    let mut log = log();
    record_settled(
        &mut log,
        "Bundle adjusted run_a",
        ms(100),
        vec![
            folded("linearise", 1, 30, None, 180),
            folded("thumbnails", 1, 0, Some("reused"), 1),
        ],
    );
    log.toggle_expanded(newest(&log));

    let texts = painted(&mut log);
    assert!(
        texts.iter().any(|text| text == "linearise x180"),
        "{texts:?}",
    );
    assert!(
        texts.iter().any(|text| text == "thumbnails  reused"),
        "a phase that ran once carried a count, or lost its note: {texts:?}",
    );
    // A sub-millisecond stage reads the same here as in the entry column it
    // breaks down, rather than in a spelling of its own.
    assert!(
        texts.iter().any(|text| text == "<1 ms"),
        "a stage under a millisecond was spelled differently to its entry: {texts:?}",
    );
    assert!(
        !texts.iter().any(|text| text == "--"),
        "a stage still renders a dash of its own: {texts:?}",
    );
}

/// The line that makes the breakdown trustworthy: 30 and 50 at the top level,
/// 20 elsewhere, and the entry's own column reading 100.
#[test]
fn the_elsewhere_line_is_last_and_reconciles_with_the_entrys_cost() {
    let mut log = log();
    record_settled(
        &mut log,
        "Bundle adjusted run_a",
        ms(100),
        vec![
            phase("materialise", 0, 30, None),
            phase("solve", 0, 50, None),
            // A nested phase's cost is already inside its parent's.
            phase("round", 1, 50, None),
        ],
    );
    log.toggle_expanded(newest(&log));

    let ctx = egui::Context::default();
    let shapes = painted_shapes(&ctx, &mut log, input());
    let texts: Vec<&str> = shapes.iter().map(|(text, _, _)| text.as_str()).collect();
    let index = |needle: &str| {
        texts
            .iter()
            .position(|text| *text == needle)
            .unwrap_or_else(|| panic!("{needle:?} was not painted, only {texts:?}"))
    };
    assert!(
        index("round") < index("elsewhere"),
        "elsewhere was not last"
    );
    assert!(
        index("100 ms") < index("30 ms"),
        "the entry's own cost is first",
    );
    assert!(
        texts.contains(&"20 ms"),
        "elsewhere did not reconcile: {texts:?}",
    );
    // Never nested, however deep the phases above it went.
    let (_, top) = painted_at(&shapes, "materialise");
    let (_, tail) = painted_at(&shapes, "elsewhere");
    assert_eq!(tail.x, top.x);
}

#[test]
fn copy_carries_an_expanded_entrys_detail_and_not_a_collapsed_ones() {
    let mut log = log();
    record_settled(
        &mut log,
        "Bundle adjusted run_a",
        ms(100),
        vec![
            phase("materialise", 0, 30, None),
            phase("round", 1, 50, None),
        ],
    );
    let revision = newest(&log);
    assert!(
        !log.to_clipboard_text().contains("materialise"),
        "a collapsed entry put its detail on the clipboard",
    );

    log.toggle_expanded(revision);
    let copied = log.to_clipboard_text();
    let lines: Vec<&str> = copied.lines().collect();
    assert_eq!(lines.len(), 4, "{copied}");
    assert!(lines[1].ends_with("  30 ms  materialise"), "{copied}");
    assert!(lines[2].ends_with("  50 ms    round"), "{copied}");
    // 100 less the one top-level phase: `round` is nested, so its cost is
    // already inside `materialise`.
    assert!(lines[3].ends_with("  70 ms  elsewhere"), "{copied}");
}

#[test]
fn the_detailed_timing_checkbox_records_the_change_and_nothing_else() {
    let mut log = log();
    assert!(
        !log.detailed_timing(),
        "detail is off until it is asked for"
    );

    let ctx = egui::Context::default();
    let shapes = painted_shapes(&ctx, &mut log, input());
    let (_, checkbox) = painted_at(&shapes, "Detailed timing");
    click(&ctx, &mut log, checkbox + egui::vec2(8.0, 6.0));

    assert!(log.detailed_timing(), "the click did not reach the level");
    assert_eq!(texts(&log), ["Detailed timing on"]);
    assert_eq!(
        log.entries().next_back().expect("an entry").kind,
        Kind::Display,
    );

    // Handed the value it already has, it records nothing.
    log.set_detailed_timing(true);
    assert_eq!(texts(&log), ["Detailed timing on"]);
}

/// The row lookup is arithmetic over a table of expansions rather than a walk,
/// so two open entries among several are what says the arithmetic is right:
/// each one's detail has to land under it, and the entries after it have to
/// stay themselves.
#[test]
fn two_expanded_entries_keep_every_row_after_them_in_place() {
    let mut log = log();
    for name in ["alpha", "beta", "gamma", "delta"] {
        log.record_done(
            Kind::Edit,
            Instant::now(),
            format!("Adjusted {name}"),
            vec![phase("solve", 0, 50, None)],
        );
    }
    let revisions: Vec<u64> = log.entries().map(|entry| entry.revision).collect();
    log.toggle_expanded(revisions[1]);
    log.toggle_expanded(revisions[3]);
    assert_eq!(super::panel::row_count(&log), 6);

    let painted = painted(&mut log);
    let order: Vec<&str> = painted
        .iter()
        .map(String::as_str)
        .filter(|text| text.starts_with("Adjusted ") || *text == "solve")
        .collect();
    assert_eq!(
        order,
        [
            "Adjusted alpha",
            "Adjusted beta",
            "solve",
            "Adjusted gamma",
            "Adjusted delta",
            "solve",
        ],
    );
}

/// A folded row whose runs said different things paints both ends. One end on
/// its own would be a claim about the row that only one of its runs supports.
#[test]
fn a_folded_note_paints_both_ends_when_the_runs_disagreed() {
    let mut log = log();
    let mut rounds = folded("round", 1, 40, Some("trim 50 px"), 3);
    if let Detail::Phase { note_last, .. } = &mut rounds {
        *note_last = Some("trim 4 px".to_string());
    }
    record_settled(&mut log, "Bundle adjusted run_a", ms(100), vec![rounds]);
    log.toggle_expanded(newest(&log));

    let texts = painted(&mut log);
    assert!(
        texts
            .iter()
            .any(|text| text == "round x3  trim 50 px ... trim 4 px"),
        "{texts:?}",
    );
}

/// Uploading a reconstruction to the GPU is not part of reading it off the
/// disk, so the overhead comes last, under a rule, after `elsewhere` has closed
/// the operation's own account. A reader who takes the upload for the load
/// draws the wrong conclusion about where a slow action went.
#[test]
fn the_overhead_comes_after_the_operation_and_its_elsewhere() {
    let mut log = log();
    let collector = Collector::new(false);
    drop(collector.phase("materialise"));
    log.record_done(
        Kind::Edit,
        Instant::now(),
        "Bundle adjusted run_a",
        collector.take(),
    );
    log.settle(Instant::now(), frame_events());

    let entry = log.entries().next_back().expect("an entry");
    let drawn: Vec<String> = (0..super::panel::detail_rows(entry))
        .map(|row| super::panel::detail_row(entry, row).text)
        .collect();
    assert_eq!(
        drawn,
        [
            "materialise",
            "elsewhere",
            ActionLog::OVERHEAD,
            "uploads",
            "points",
            "scene render",
        ],
        "the overhead did not come last, or elsewhere did not close the account",
    );
    assert!(
        super::panel::detail_row(entry, 2).rules_above,
        "no rule divides the operation from the overhead",
    );
    assert!(
        (0..super::panel::detail_rows(entry))
            .filter(|row| super::panel::detail_row(entry, *row).rules_above)
            .count()
            == 1,
        "more than one rule was drawn",
    );
}
