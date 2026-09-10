// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The dialog's defaults and the keys it answers to. The gate above it is
//! exercised against real values in `state/edits/tests.rs`, where there are
//! nodes to gate.

use crate::scene::ReconId;

use super::*;

/// One frame of the dialog over an 800x600 screen, with `events` delivered to
/// it, reporting whatever it answered.
fn frame(prompt: &mut BundleAdjustPrompt, events: Vec<egui::Event>) -> Option<BundleAdjustAnswer> {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(800.0, 600.0),
        )),
        events,
        ..Default::default()
    };
    let mut answer = None;
    crate::test_support::run_frame_headless(&ctx, input, |ui| {
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

#[test]
fn a_dialog_that_was_never_asked_draws_nothing_and_answers_nothing() {
    let mut prompt = BundleAdjustPrompt::default();
    assert_eq!(frame(&mut prompt, Vec::new()), None);
    assert!(prompt.pending.is_none());
}

#[test]
fn an_ordinary_frame_answers_nothing_and_leaves_the_dialog_up() {
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(ReconId::next(), "bull".to_string(), None);
    assert!(prompt.pending.is_some());

    assert_eq!(frame(&mut prompt, Vec::new()), None);
    assert!(prompt.pending.is_some(), "the dialog closed on its own");
}

#[test]
fn escape_cancels_and_closes_the_dialog() {
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(ReconId::next(), "bull".to_string(), None);

    assert_eq!(
        frame(&mut prompt, press(egui::Key::Escape)),
        None,
        "escape ran the adjustment"
    );
    assert!(prompt.pending.is_none(), "escape left the dialog up");
}

#[test]
fn enter_runs_it_with_the_focal_held_which_is_the_default() {
    let mut prompt = BundleAdjustPrompt::default();
    let id = ReconId::next();
    prompt.ask(id, "bull".to_string(), None);

    assert_eq!(
        frame(&mut prompt, press(egui::Key::Enter)),
        Some(BundleAdjustAnswer {
            recon: id,
            release_focal: false
        })
    );
    assert!(prompt.pending.is_none(), "the answered dialog stayed up");
}

#[test]
fn a_second_ask_does_not_stack_a_second_dialog() {
    let mut prompt = BundleAdjustPrompt::default();
    let first = ReconId::next();
    prompt.ask(first, "bull".to_string(), None);
    prompt.ask(ReconId::next(), "other".to_string(), None);

    let answer = frame(&mut prompt, press(egui::Key::Enter)).expect("one answer");
    assert_eq!(answer.recon, first);
}
