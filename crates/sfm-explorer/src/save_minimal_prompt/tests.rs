// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The prompt's prefill, the two keys it answers to, and the save on the other
//! side of it.
//!
//! The native save dialog that names the file cannot be driven from a test, so
//! what is exercised here is everything after it: the field the prompt opens
//! with, the answer it reports, and what `AppState` writes when it is given one.
//! The menu item's presence is the windowed `ui_basic` suite's.

use std::path::{Path, PathBuf};

use sfmtool_core::SfmrReconstruction;

use super::*;
use crate::scene::SceneNode;
use crate::state::AppState;

/// One frame of the prompt over an 800x600 screen, with `events` delivered to
/// it, reporting whatever it answered.
fn frame(prompt: &mut SaveMinimalPrompt, events: Vec<egui::Event>) -> Option<SaveMinimalAnswer> {
    let ctx = egui::Context::default();
    frame_in(prompt, &ctx, events)
}

/// The same, against a context that lives across frames, which is what focus
/// and the field's own state need.
fn frame_in(
    prompt: &mut SaveMinimalPrompt,
    ctx: &egui::Context,
    events: Vec<egui::Event>,
) -> Option<SaveMinimalAnswer> {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(800.0, 600.0),
        )),
        events,
        ..Default::default()
    };
    let mut answer = None;
    crate::test_support::run_frame_headless(ctx, input, |ui| {
        answer = prompt.show(ui.ctx());
    });
    answer
}

/// One key press, as the events of a frame.
fn press(key: egui::Key) -> Vec<egui::Event> {
    vec![egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]
}

/// A directory of this test's own under the system temp dir, emptied first so a
/// rerun does not read a previous run's file.
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sfm_explorer_minimal_prompt_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a writable temp dir");
    dir
}

/// A state holding one node that came from `dir/recon.sfmr`, over a workspace at
/// `dir` itself, so the measurement from anywhere under `dir` means something.
fn state_over(dir: &Path) -> (AppState, crate::scene::ReconId) {
    let mut recon = SfmrReconstruction::demo(32);
    recon.workspace_dir = dir.to_path_buf();
    let mut state = AppState::new();
    let id = state.append_node(SceneNode::from_path(&dir.join("recon.sfmr"), recon));
    (state, id)
}

#[test]
fn a_prompt_that_was_never_asked_draws_nothing_and_answers_nothing() {
    let mut prompt = SaveMinimalPrompt::default();
    assert_eq!(frame(&mut prompt, Vec::new()), None);
    assert!(prompt.pending.is_none());
}

#[test]
fn an_ordinary_frame_answers_nothing_and_leaves_the_prompt_up() {
    let mut prompt = SaveMinimalPrompt::default();
    prompt.ask(
        crate::scene::ReconId::next(),
        "recon".to_string(),
        PathBuf::from("/published/recon.sfmr"),
        "..".to_string(),
    );

    assert_eq!(frame(&mut prompt, Vec::new()), None);
    assert!(prompt.pending.is_some(), "the prompt closed on its own");
}

#[test]
fn asking_twice_keeps_the_path_being_typed() {
    let mut prompt = SaveMinimalPrompt::default();
    let id = crate::scene::ReconId::next();
    prompt.ask(
        id,
        "recon".to_string(),
        PathBuf::from("/published/recon.sfmr"),
        "..".to_string(),
    );
    prompt.pending.as_mut().expect("up").workspace_path = ".".to_string();

    prompt.ask(
        id,
        "recon".to_string(),
        PathBuf::from("/elsewhere/recon.sfmr"),
        "../..".to_string(),
    );

    let pending = prompt.pending.as_ref().expect("still up");
    assert_eq!(pending.workspace_path, ".");
    assert_eq!(pending.path, PathBuf::from("/published/recon.sfmr"));
}

#[test]
fn escape_cancels_and_closes_the_prompt() {
    let mut prompt = SaveMinimalPrompt::default();
    prompt.ask(
        crate::scene::ReconId::next(),
        "recon".to_string(),
        PathBuf::from("/published/recon.sfmr"),
        "..".to_string(),
    );

    assert_eq!(frame(&mut prompt, press(egui::Key::Escape)), None);
    assert!(prompt.pending.is_none(), "Escape left the prompt up");
}

#[test]
fn enter_answers_with_the_field_as_it_stands() {
    let mut prompt = SaveMinimalPrompt::default();
    let id = crate::scene::ReconId::next();
    prompt.ask(
        id,
        "recon".to_string(),
        PathBuf::from("/published/recon.sfmr"),
        "..".to_string(),
    );
    // One frame to draw the field and focus it; only then can a key reach it.
    let ctx = egui::Context::default();
    assert_eq!(frame_in(&mut prompt, &ctx, Vec::new()), None);
    // What the user leaves in the field, whitespace and all.
    prompt.pending.as_mut().expect("up").workspace_path = "  .  ".to_string();

    let answer = frame_in(&mut prompt, &ctx, press(egui::Key::Enter)).expect("the save");

    assert_eq!(
        answer,
        SaveMinimalAnswer {
            recon: id,
            path: PathBuf::from("/published/recon.sfmr"),
            // Trimmed: a path with spaces around it is a typed path, not a
            // workspace whose name has them.
            workspace_path: ".".to_string(),
        }
    );
    assert!(prompt.pending.is_none(), "the answer left the prompt up");
}

#[test]
fn an_empty_field_answers_with_no_path_recorded() {
    let mut prompt = SaveMinimalPrompt::default();
    prompt.ask(
        crate::scene::ReconId::next(),
        "recon".to_string(),
        PathBuf::from("/published/recon.sfmr"),
        String::new(),
    );
    let ctx = egui::Context::default();
    frame_in(&mut prompt, &ctx, Vec::new());

    let answer = frame_in(&mut prompt, &ctx, press(egui::Key::Enter)).expect("the save");
    assert_eq!(answer.workspace_path, "");
}

/// The label and the hint the field carries, so an empty field does not read as
/// a mistake.
#[test]
fn the_prompt_says_what_the_field_is_and_that_empty_is_an_answer() {
    let mut prompt = SaveMinimalPrompt::default();
    prompt.ask(
        crate::scene::ReconId::next(),
        "recon".to_string(),
        PathBuf::from("/published/recon.sfmr"),
        String::new(),
    );
    let ctx = egui::Context::default();
    // A window is an area, and an area sizes itself on its first pass, so the
    // frame that paints its contents is the second one.
    frame_in(&mut prompt, &ctx, Vec::new());
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(800.0, 600.0),
        )),
        ..Default::default()
    };
    let painted = crate::test_support::painted_texts(&ctx, input, |ui| {
        prompt.show(ui.ctx());
    });
    let all = painted.join(" ");
    assert!(all.contains("Workspace path recorded in the copy"), "{all}");
    assert!(all.contains("(none recorded)"), "{all}");
    assert!(all.contains("Leave it empty to record no path"), "{all}");
}

// ── The state on either side of the prompt ──────────────────────────────

#[test]
fn the_field_starts_at_the_path_the_save_would_measure() {
    let dir = temp_dir("prefill");
    let (state, id) = state_over(&dir);
    let out = dir.join("published").join("recon.sfmr");

    // The workspace is one directory up from where the copy is going, and that
    // is what a user accepting the prompt unread gets.
    assert_eq!(state.minimal_copy_workspace_path(id, &out), "..");
    let _ = std::fs::remove_dir_all(&dir);
}

/// With nothing to measure, the path the value already records; and with that
/// empty too, empty, since an empty value is "none recorded".
#[test]
fn the_field_falls_back_to_the_recorded_path_and_then_to_nothing() {
    let dir = temp_dir("fallback");
    let mut recon = SfmrReconstruction::demo(8);
    // No workspace directory to measure to, and the file's own recorded path is
    // all that is left to offer.
    recon.workspace_dir = PathBuf::new();
    recon.metadata.workspace.relative_path = "../ws".to_string();
    let mut state = AppState::new();
    let id = state.append_node(SceneNode::from_path(&dir.join("recon.sfmr"), recon));
    let out = dir.join("copy.sfmr");
    // An empty workspace directory measures nothing against an absolute output.
    assert_eq!(state.minimal_copy_workspace_path(id, &out), "../ws");

    let mut bare = SfmrReconstruction::demo(8);
    bare.workspace_dir = PathBuf::new();
    bare.metadata.workspace.relative_path.clear();
    let bare_id = state.append_node(SceneNode::from_path(&dir.join("bare.sfmr"), bare));
    assert_eq!(state.minimal_copy_workspace_path(bare_id, &out), "");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The copy over the node's own file is refused before the prompt is put up, so
/// nobody fills in a field for a save that cannot happen.
#[test]
fn the_nodes_own_file_is_refused_before_the_prompt() {
    let dir = temp_dir("refusal");
    let (state, id) = state_over(&dir);
    let own = dir.join("recon.sfmr");

    let why = state
        .minimal_copy_refusal(id, &own)
        .expect("its own file is refused");
    assert!(
        why.contains("cannot replace the file it came from"),
        "{why}"
    );
    assert!(state
        .minimal_copy_refusal(id, &dir.join("published").join("recon.sfmr"))
        .is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The whole flow after the file dialog: the prompt opens at the measurement,
/// the user states something else, and the copy records what was stated.
#[test]
fn answering_the_prompt_writes_the_copy_with_the_stated_path() {
    let dir = temp_dir("answer");
    let (mut state, id) = state_over(&dir);
    let out = dir.join("published").join("recon.sfmr");
    std::fs::create_dir_all(out.parent().expect("a parent")).expect("a writable temp dir");
    let label = state.node(id).expect("loaded").label.clone();
    let prefill = state.minimal_copy_workspace_path(id, &out);
    assert_eq!(prefill, "..");

    let mut prompt = SaveMinimalPrompt::default();
    prompt.ask(id, label, out.clone(), prefill);
    let ctx = egui::Context::default();
    frame_in(&mut prompt, &ctx, Vec::new());
    // The copy's home is the workspace root, not the staging directory it is
    // being written to.
    prompt.pending.as_mut().expect("up").workspace_path = ".".to_string();
    let answer = frame_in(&mut prompt, &ctx, press(egui::Key::Enter)).expect("the save");

    state
        .save_minimal_copy(answer.recon, &answer.path, Some(&answer.workspace_path))
        .expect("a writable path");
    let written = sfmtool_sfmr_format::read_sfmr(&out).expect("a readable copy");
    assert_eq!(written.metadata.workspace.relative_path, ".");
    assert!(written.metadata.workspace.absolute_path.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Cancelling writes nothing: the file dialog named a path, and backing out here
/// backs out of the whole save.
#[test]
fn cancelling_the_prompt_writes_no_file() {
    let dir = temp_dir("cancel");
    let (state, id) = state_over(&dir);
    let out = dir.join("published").join("recon.sfmr");
    std::fs::create_dir_all(out.parent().expect("a parent")).expect("a writable temp dir");
    let versions = state.node(id).expect("loaded").history.versions().len();

    let mut prompt = SaveMinimalPrompt::default();
    prompt.ask(id, "recon".to_string(), out.clone(), "..".to_string());
    let ctx = egui::Context::default();
    frame_in(&mut prompt, &ctx, Vec::new());
    assert_eq!(frame_in(&mut prompt, &ctx, press(egui::Key::Escape)), None);

    assert!(!out.exists(), "a cancelled prompt wrote a file");
    let node = state.node(id).expect("loaded");
    assert_eq!(node.history.versions().len(), versions);
    // Nothing was logged either: choosing not to save is not a failure.
    assert!(state.action_log.entries().next().is_none());
    let _ = std::fs::remove_dir_all(&dir);
}
