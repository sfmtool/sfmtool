// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What the prompt remembers, and the two ways of leaving it that report no
//! answer. The three buttons are exercised where they are reachable by name, in
//! the windowed `ui_basic` suite.

use crate::scene::ReconId;

use super::*;

/// One frame of the prompt over an 800x600 screen, with `events` delivered to
/// it, reporting whatever it answered.
fn frame(prompt: &mut ClosePrompt, events: Vec<egui::Event>) -> Option<CloseAnswer> {
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
        answer = prompt.show(ui.ctx(), &["demo".to_string(), "walk".to_string()]);
    });
    answer
}

/// An Escape key press, which is the prompt's Cancel.
fn escape() -> Vec<egui::Event> {
    vec![egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]
}

#[test]
fn asking_twice_keeps_the_first_question() {
    // A menu item racing a shortcut must not stack two prompts, and the one
    // that is up is the one that gets answered.
    let mut prompt = ClosePrompt::default();
    prompt.ask(PendingClose::Quit);
    prompt.ask(PendingClose::All);
    assert_eq!(prompt.pending, Some(PendingClose::Quit));
}

#[test]
fn a_prompt_that_was_never_asked_draws_nothing_and_answers_nothing() {
    let mut prompt = ClosePrompt::default();
    assert_eq!(frame(&mut prompt, Vec::new()), None);
    assert_eq!(prompt.pending, None);
}

#[test]
fn a_pending_prompt_stays_up_until_it_is_answered() {
    let mut prompt = ClosePrompt::default();
    prompt.ask(PendingClose::Node(ReconId::from_raw(3)));
    assert_eq!(frame(&mut prompt, Vec::new()), None);
    assert_eq!(
        prompt.pending,
        Some(PendingClose::Node(ReconId::from_raw(3)))
    );
}

#[test]
fn escape_cancels_and_leaves_nothing_pending() {
    // Cancel is the answer that does neither of the other two, so it clears the
    // question without reporting one — which is what stops the close.
    let mut prompt = ClosePrompt::default();
    prompt.ask(PendingClose::All);
    assert_eq!(frame(&mut prompt, escape()), None);
    assert_eq!(prompt.pending, None);
}
