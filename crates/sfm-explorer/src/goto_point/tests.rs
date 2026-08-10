// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for Go to Point.
//!
//! The parse and the scene lookup are plain functions, so most of this needs no
//! frame at all; the dialog tests that do run through `Context::run_ui`, driving
//! it the way a user would — type, press Enter — and assert on the [`PointRef`]
//! it hands back rather than on pixels.

use sfmtool_core::SfmrReconstruction;

use super::{parse_point_query, resolve_point_query, GotoPointDialog, PointQuery};
use crate::scene::{PointRef, ReconId, SceneNode};
use crate::state::AppState;

// ── Fixtures ────────────────────────────────────────────────────────────

/// A file-backed node holding `points` points, with `hash` as its content hash.
///
/// `SfmrReconstruction::demo` leaves the hash empty, which is the case the
/// `00000000` display fallback exists for — so a test that cares about hashes
/// has to set one.
fn node(path: &str, points: usize, hash: &str) -> SceneNode {
    let mut recon = SfmrReconstruction::demo(points);
    recon.content_hash.content_xxh128 = hash.to_string();
    SceneNode::from_path(std::path::Path::new(path), recon)
}

/// Two loaded reconstructions with distinct hashes, the first selected.
/// Returns the state plus the two ids in load order.
fn two_nodes() -> (AppState, ReconId, ReconId) {
    let mut state = AppState::new();
    let a = state.append_node(node("/runs/a.sfmr", 40, "aaaa1111bbbb2222"));
    let b = state.append_node(node("/runs/b.sfmr", 60, "cccc3333dddd4444"));
    state.select_recon(a);
    (state, a, b)
}

/// Resolve `input` against `state` the way the dialog does: parse, then look up.
fn go_to(state: &AppState, input: &str) -> Result<PointRef, String> {
    let query = parse_point_query(input)?;
    resolve_point_query(&state.scene, state.selected_recon, &query)
}

// ── Parsing ─────────────────────────────────────────────────────────────

#[test]
fn a_bare_number_parses_as_an_index() {
    assert_eq!(parse_point_query("12345"), Ok(PointQuery::Index(12345)));
    assert_eq!(parse_point_query("0"), Ok(PointQuery::Index(0)));
}

#[test]
fn surrounding_whitespace_and_a_leading_hash_are_tolerated() {
    // `#12345` is how the Image Detail tooltip prints a point, so it is a
    // natural thing to retype or copy out of a screenshot.
    for input in ["  12345  ", "#12345", " # 12345 ", "\t12345\n"] {
        assert_eq!(
            parse_point_query(input),
            Ok(PointQuery::Index(12345)),
            "input {input:?}"
        );
    }
}

#[test]
fn a_full_point_id_parses_into_its_hash_and_index() {
    assert_eq!(
        parse_point_query("pt3d_a1b2c3d4_12345"),
        Ok(PointQuery::Qualified {
            hash: "a1b2c3d4".to_string(),
            index: 12345,
        })
    );
}

#[test]
fn a_point_id_is_case_insensitive_and_normalizes_to_lowercase() {
    // Hex is written both ways in the wild, and the hashes we compare against
    // are lowercase — so the case has to be dropped at the parse, once.
    let expected = PointQuery::Qualified {
        hash: "a1b2c3d4".to_string(),
        index: 7,
    };
    assert_eq!(parse_point_query("PT3D_A1B2C3D4_7"), Ok(expected.clone()));
    assert_eq!(parse_point_query("pt3d_A1b2C3d4_7"), Ok(expected));
}

#[test]
fn a_full_32_character_hash_parses_too() {
    // The format spec offers the whole `content_xxh128` for exact
    // disambiguation, so an ID built from one has to be accepted.
    let hash = "0123456789abcdef0123456789abcdef";
    assert_eq!(
        parse_point_query(&format!("pt3d_{hash}_9")),
        Ok(PointQuery::Qualified {
            hash: hash.to_string(),
            index: 9,
        })
    );
}

#[test]
fn unrecognized_text_is_rejected_with_both_accepted_shapes() {
    for input in ["", "banana", "pt3d_", "12.5", "-3", "pt3d_a1b2c3d4"] {
        let error = parse_point_query(input).expect_err("should not parse: {input:?}");
        assert!(
            error.contains("pt3d_a1b2c3d4_12345"),
            "input {input:?} gave {error:?}, which shows no example"
        );
    }
}

#[test]
fn a_malformed_point_id_says_which_half_is_wrong() {
    let bad_hash = parse_point_query("pt3d_zzzz_5").expect_err("hash is not hex");
    assert!(bad_hash.contains("reconstruction hash"), "got {bad_hash:?}");
    let bad_index = parse_point_query("pt3d_a1b2c3d4_five").expect_err("index is not a number");
    assert!(bad_index.contains("point index"), "got {bad_index:?}");
}

// ── Resolving a bare index ──────────────────────────────────────────────

#[test]
fn a_bare_index_lands_in_the_selected_reconstruction() {
    let (mut state, a, b) = two_nodes();

    assert_eq!(go_to(&state, "7"), Ok(PointRef::new(a, 7)));

    // The same text after selecting the other node means the other node's
    // point — that is what "bare" costs and buys.
    state.select_recon(b);
    assert_eq!(go_to(&state, "7"), Ok(PointRef::new(b, 7)));
}

#[test]
fn a_bare_index_past_the_end_reports_the_point_count() {
    let (state, ..) = two_nodes();

    let error = go_to(&state, "40").expect_err("node a holds 40 points, so 40 is one past");
    assert!(error.contains("40 points"), "got {error:?}");
    assert!(
        error.contains('a'),
        "the label should name the node: {error:?}"
    );
}

#[test]
fn a_bare_index_with_nothing_loaded_says_to_open_a_file() {
    let state = AppState::new();

    let error = go_to(&state, "5").expect_err("there is no reconstruction to index into");
    assert!(
        error.contains("No reconstruction is loaded"),
        "got {error:?}"
    );
}

// ── Resolving a full ID ─────────────────────────────────────────────────

#[test]
fn a_full_id_selects_the_reconstruction_it_names() {
    let (state, _a, b) = two_nodes();

    // `b` is loaded but *not* selected: the hash is what puts us there, which
    // is the whole point of pasting a full ID.
    assert_eq!(go_to(&state, "pt3d_cccc3333_11"), Ok(PointRef::new(b, 11)));
}

#[test]
fn a_full_length_hash_resolves_the_same_node_as_its_prefix() {
    let (state, _a, b) = two_nodes();

    assert_eq!(
        go_to(&state, "pt3d_cccc3333dddd4444_11"),
        Ok(PointRef::new(b, 11))
    );
}

#[test]
fn an_unknown_hash_says_which_file_to_open() {
    let (state, ..) = two_nodes();

    let error = go_to(&state, "pt3d_deadbeef_3").expect_err("no node carries that hash");
    assert!(error.contains("deadbeef"), "got {error:?}");
    assert!(error.contains(".sfmr"), "got {error:?}");
}

#[test]
fn a_full_id_past_the_end_is_bounds_checked_against_its_own_node() {
    // Node b holds 60 points and node a only 40 — the check has to use the
    // node the hash named, not the selected one.
    let (state, ..) = two_nodes();

    assert!(go_to(&state, "pt3d_cccc3333_50").is_ok());
    let error = go_to(&state, "pt3d_cccc3333_60").expect_err("b holds 60 points");
    assert!(error.contains("60 points"), "got {error:?}");
}

#[test]
fn an_unhashed_reconstruction_resolves_by_the_zeros_it_displays() {
    // Demo data and any pre-hash file display as `pt3d_00000000_<i>`; copying
    // that ID out of the panel and pasting it back has to work.
    let mut state = AppState::new();
    let id = state.append_node(SceneNode::demo(SfmrReconstruction::demo(20)));

    assert_eq!(go_to(&state, "pt3d_00000000_4"), Ok(PointRef::new(id, 4)));
}

#[test]
fn an_ambiguous_hash_prefers_the_selected_node() {
    // The same file opened from two paths shares a content hash, so both nodes
    // match. Every match holds the same content, so staying where the user is
    // looking is the least surprising answer.
    let mut state = AppState::new();
    let first = state.append_node(node("/runs/one.sfmr", 30, "5555666677778888"));
    let second = state.append_node(node("/copies/one.sfmr", 30, "5555666677778888"));

    state.select_recon(second);
    assert_eq!(
        go_to(&state, "pt3d_55556666_2"),
        Ok(PointRef::new(second, 2))
    );

    state.select_recon(first);
    assert_eq!(
        go_to(&state, "pt3d_55556666_2"),
        Ok(PointRef::new(first, 2))
    );
}

#[test]
fn a_hash_that_matches_no_node_is_not_answered_by_the_selected_one() {
    // The selected-node preference must only break ties among real matches —
    // never turn a miss into a hit.
    let (state, ..) = two_nodes();

    assert!(go_to(&state, "pt3d_ffff9999_1").is_err());
}

// ── The dialog ──────────────────────────────────────────────────────────

/// Drive one frame of the dialog with `events` delivered to egui, returning
/// what it resolved (if anything).
fn run_frame(
    dialog: &mut GotoPointDialog,
    ctx: &egui::Context,
    state: &AppState,
    events: Vec<egui::Event>,
) -> Option<PointRef> {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(1200.0, 800.0),
        )),
        events,
        ..Default::default()
    };
    let mut resolved = None;
    let _ = ctx.run_ui(input, |ui| {
        resolved = dialog.show(ui.ctx(), &state.scene, state.selected_recon);
    });
    resolved
}

/// Type `text` into the focused field and press Enter, over as many frames as
/// egui needs to route the events to the widget.
fn type_and_submit(
    dialog: &mut GotoPointDialog,
    ctx: &egui::Context,
    state: &AppState,
    text: &str,
) -> Option<PointRef> {
    // Frame one draws the field and focuses it; only then can typed text reach
    // it, so the events go in on the second frame.
    run_frame(dialog, ctx, state, Vec::new());
    let events = vec![
        egui::Event::Text(text.to_string()),
        egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        },
    ];
    run_frame(dialog, ctx, state, events)
}

#[test]
fn a_closed_dialog_draws_nothing_and_resolves_nothing() {
    let (state, ..) = two_nodes();
    let mut dialog = GotoPointDialog::default();
    let ctx = egui::Context::default();

    assert!(!dialog.open);
    assert_eq!(run_frame(&mut dialog, &ctx, &state, Vec::new()), None);
}

#[test]
fn submitting_a_valid_index_resolves_it_and_closes_the_dialog() {
    let (state, a, _b) = two_nodes();
    let mut dialog = GotoPointDialog::default();
    let ctx = egui::Context::default();
    dialog.open();

    let resolved = type_and_submit(&mut dialog, &ctx, &state, "12");

    assert_eq!(resolved, Some(PointRef::new(a, 12)));
    assert!(!dialog.open, "a successful jump should close the dialog");
}

#[test]
fn submitting_a_pasted_id_resolves_across_reconstructions() {
    let (state, _a, b) = two_nodes();
    let mut dialog = GotoPointDialog::default();
    let ctx = egui::Context::default();
    dialog.open();

    let resolved = type_and_submit(&mut dialog, &ctx, &state, "pt3d_cccc3333_31");

    assert_eq!(resolved, Some(PointRef::new(b, 31)));
}

#[test]
fn a_bad_query_keeps_the_dialog_open_with_the_reason() {
    let (state, ..) = two_nodes();
    let mut dialog = GotoPointDialog::default();
    let ctx = egui::Context::default();
    dialog.open();

    let resolved = type_and_submit(&mut dialog, &ctx, &state, "banana");

    assert_eq!(resolved, None);
    assert!(dialog.open, "the typo has to stay correctable in place");
    let error = dialog.error.as_deref().expect("a reason was recorded");
    assert!(error.contains("pt3d_a1b2c3d4_12345"), "got {error:?}");
}

#[test]
fn reopening_clears_a_stale_error_but_keeps_the_text() {
    let (state, ..) = two_nodes();
    let mut dialog = GotoPointDialog::default();
    let ctx = egui::Context::default();
    dialog.open();
    type_and_submit(&mut dialog, &ctx, &state, "banana");
    assert!(dialog.error.is_some());

    dialog.open();

    assert!(
        dialog.error.is_none(),
        "the old reason is not about the new attempt"
    );
    assert_eq!(
        dialog.input, "banana",
        "the text stays editable, not retyped"
    );
}

#[test]
fn an_empty_field_submits_nothing() {
    let (state, ..) = two_nodes();
    let mut dialog = GotoPointDialog::default();
    let ctx = egui::Context::default();
    dialog.open();

    // Enter on an empty field: no jump, no error — there is nothing to complain
    // about yet, and the "Go" button is disabled for the same reason.
    let resolved = type_and_submit(&mut dialog, &ctx, &state, "");

    assert_eq!(resolved, None);
    assert!(dialog.open);
    assert!(dialog.error.is_none(), "an empty field is not a mistake");
}

#[test]
fn escape_closes_the_dialog_without_resolving() {
    let (state, ..) = two_nodes();
    let mut dialog = GotoPointDialog::default();
    let ctx = egui::Context::default();
    dialog.open();
    run_frame(&mut dialog, &ctx, &state, Vec::new());

    let resolved = run_frame(
        &mut dialog,
        &ctx,
        &state,
        vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        }],
    );

    assert_eq!(resolved, None);
    assert!(!dialog.open);
}
