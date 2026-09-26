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
    prompt.ask(
        ReconId::next(),
        "bull".to_string(),
        BundleAdjustGates::default(),
    );
    assert!(prompt.pending.is_some());

    assert_eq!(frame(&mut prompt, Vec::new()), None);
    assert!(prompt.pending.is_some(), "the dialog closed on its own");
}

#[test]
fn escape_cancels_and_closes_the_dialog() {
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(
        ReconId::next(),
        "bull".to_string(),
        BundleAdjustGates::default(),
    );

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
    prompt.ask(id, "bull".to_string(), BundleAdjustGates::default());

    assert_eq!(
        frame(&mut prompt, press(egui::Key::Enter)),
        Some(BundleAdjustAnswer {
            recon: id,
            release_focal: false,
            release_distortion: false,
            spline_coeff_count: None,
        })
    );
    assert!(prompt.pending.is_none(), "the answered dialog stayed up");
}

#[test]
fn a_second_ask_does_not_stack_a_second_dialog() {
    let mut prompt = BundleAdjustPrompt::default();
    let first = ReconId::next();
    prompt.ask(first, "bull".to_string(), BundleAdjustGates::default());
    prompt.ask(
        ReconId::next(),
        "other".to_string(),
        BundleAdjustGates::default(),
    );

    let answer = frame(&mut prompt, press(egui::Key::Enter)).expect("one answer");
    assert_eq!(answer.recon, first);
}

#[test]
fn the_distortion_is_released_only_with_the_focal() {
    let mut prompt = BundleAdjustPrompt::default();
    let id = ReconId::next();
    prompt.ask(id, "kerry".to_string(), BundleAdjustGates::default());
    // Ticked without the focal, as a stale state from an earlier frame could
    // leave it: the answer releases neither.
    prompt.pending.as_mut().unwrap().release_distortion = true;
    let answer = frame(&mut prompt, press(egui::Key::Enter)).expect("one answer");
    assert!(!answer.release_focal);
    assert!(!answer.release_distortion);

    prompt.ask(id, "kerry".to_string(), BundleAdjustGates::default());
    let pending = prompt.pending.as_mut().unwrap();
    pending.release_focal = true;
    pending.release_distortion = true;
    let answer = frame(&mut prompt, press(egui::Key::Enter)).expect("one answer");
    assert!(answer.release_focal && answer.release_distortion);
}

/// Gates for a node whose spline cameras have `counts` coefficients.
fn with_splines(counts: &[usize]) -> BundleAdjustGates {
    BundleAdjustGates {
        spline_coeff_counts: counts.to_vec(),
        ..BundleAdjustGates::default()
    }
}

/// Ask, release the focal and the distortion, and let `set` touch the pending
/// question before Enter answers it.
fn answered(counts: &[usize], set: impl FnOnce(&mut Pending)) -> BundleAdjustAnswer {
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(ReconId::next(), "kerry".to_string(), with_splines(counts));
    let pending = prompt.pending.as_mut().unwrap();
    pending.release_focal = true;
    pending.release_distortion = true;
    set(pending);
    frame(&mut prompt, press(egui::Key::Enter)).expect("one answer")
}

#[test]
fn the_coefficient_count_is_kept_by_default() {
    let answer = answered(&[8], |_| {});
    assert!(answer.release_distortion);
    assert_eq!(answer.spline_coeff_count, None);

    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(ReconId::next(), "kerry".to_string(), with_splines(&[6, 8]));
    let pending = prompt.pending.as_ref().unwrap();
    assert!(pending.keep_coeffs);
    // The control shows the largest count the node holds.
    assert_eq!(pending.coeff_count, 8);
}

#[test]
fn a_changed_count_is_asked_for_only_when_it_changes_something() {
    let answer = answered(&[8], |p| {
        p.keep_coeffs = false;
        p.coeff_count = 12;
    });
    assert_eq!(answer.spline_coeff_count, Some(12));

    // The one count every spline camera already has is no change.
    let answer = answered(&[8], |p| p.keep_coeffs = false);
    assert_eq!(answer.spline_coeff_count, None);

    // With two counts, either of them changes the other camera.
    let answer = answered(&[6, 8], |p| p.keep_coeffs = false);
    assert_eq!(answer.spline_coeff_count, Some(8));
}

#[test]
fn a_count_without_the_distortion_or_a_spline_is_not_asked_for() {
    let answer = answered(&[8], |p| {
        p.release_distortion = false;
        p.keep_coeffs = false;
        p.coeff_count = 12;
    });
    assert_eq!(answer.spline_coeff_count, None);

    let answer = answered(&[], |p| {
        p.keep_coeffs = false;
        p.coeff_count = 12;
    });
    assert_eq!(answer.spline_coeff_count, None);
}
