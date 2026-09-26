// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The dialog's defaults, its camera rows and the keys it answers to. The
//! gates above it are exercised against real values in `state/edits/tests.rs`,
//! where there are nodes to gate.

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

/// One camera row's gate: camera `camera`, `camera_model`, with the refusals
/// its model gives.
fn gate(camera: usize, camera_model: &'static str) -> CameraGate {
    let spline = matches!(camera_model, "SFMTOOL_FISHEYE" | "SFMTOOL_PINHOLE");
    let focal = matches!(
        camera_model,
        "SIMPLE_PINHOLE"
            | "EQUIDISTANT_FISHEYE"
            | "SIMPLE_RADIAL_FISHEYE"
            | "SFMTOOL_FISHEYE"
            | "SFMTOOL_PINHOLE"
    );
    let distortion = spline || camera_model == "SIMPLE_RADIAL_FISHEYE";
    CameraGate {
        camera,
        camera_model,
        images: 3,
        focal_refusal: (!focal).then(|| format!("camera {camera} cannot release its focal")),
        distortion_refusal: (!distortion)
            .then(|| format!("camera {camera} has no distortion to release")),
    }
}

/// A rig: a spline camera 0, an `OPENCV_FISHEYE` camera 1 that can release
/// nothing, and a `SIMPLE_PINHOLE` camera 3 with a focal and no distortion.
/// Camera 2 is in the table and no posed image uses it, so it has no row.
fn rig() -> BundleAdjustGates {
    BundleAdjustGates {
        camera_count: 4,
        cameras: vec![
            gate(0, "SFMTOOL_FISHEYE"),
            gate(1, "OPENCV_FISHEYE"),
            gate(3, "SIMPLE_PINHOLE"),
        ],
    }
}

/// Ask about the rig, let `set` touch the rows, and answer with Enter.
fn rig_answer(set: impl FnOnce(&mut Pending)) -> BundleAdjustAnswer {
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(ReconId::next(), "kerry".to_string(), rig());
    set(prompt.pending.as_mut().unwrap());
    frame(&mut prompt, press(egui::Key::Enter)).expect("one answer")
}

const FOCAL: CameraRelease = CameraRelease::FOCAL;
const BOTH: CameraRelease = CameraRelease::FOCAL_AND_DISTORTION;
const HELD: CameraRelease = CameraRelease::HELD;

#[test]
fn enter_runs_it_with_every_camera_held_which_is_the_default() {
    let mut prompt = BundleAdjustPrompt::default();
    let id = ReconId::next();
    prompt.ask(id, "kerry".to_string(), rig());
    let pending = prompt.pending.as_ref().unwrap();
    assert_eq!(
        pending.rows.len(),
        3,
        "one row per camera the posed images use"
    );
    assert!(pending.rows.iter().all(|r| !r.focal && !r.distortion));

    assert_eq!(
        frame(&mut prompt, press(egui::Key::Enter)),
        Some(BundleAdjustAnswer {
            recon: id,
            releases: vec![HELD; 4],
        })
    );
    assert!(prompt.pending.is_none(), "the answered dialog stayed up");
}

#[test]
fn each_row_releases_its_own_camera_and_the_rest_are_held() {
    let answer = rig_answer(|p| {
        p.rows[0] = RowState {
            focal: true,
            distortion: true,
        };
        p.rows[2].focal = true;
    });
    // Camera 2 has no row and camera 1 was left clear: both held.
    assert_eq!(answer.releases, vec![BOTH, HELD, HELD, FOCAL]);
}

#[test]
fn a_release_the_camera_s_model_cannot_take_is_never_answered() {
    // Ticked on the rows that grey them, as a stale state could leave them:
    // the OPENCV_FISHEYE releases nothing, the pinhole no distortion.
    let answer = rig_answer(|p| {
        p.rows[1] = RowState {
            focal: true,
            distortion: true,
        };
        p.rows[2] = RowState {
            focal: true,
            distortion: true,
        };
    });
    assert_eq!(answer.releases, vec![HELD, HELD, HELD, FOCAL]);
}

#[test]
fn a_row_s_distortion_is_released_only_with_its_own_focal() {
    let answer = rig_answer(|p| {
        p.rows[0].distortion = true;
        p.rows[2].focal = true;
    });
    // The pinhole's focal does not carry the spline camera's distortion.
    assert_eq!(answer.releases, vec![HELD, HELD, HELD, FOCAL]);
}

#[test]
fn a_greyed_checkbox_says_which_camera_and_why() {
    let gates = rig();
    let opencv = &gates.cameras[1];
    assert!(opencv.focal_refusal.is_some() && opencv.distortion_refusal.is_some());
    let pinhole = &gates.cameras[2];
    assert!(pinhole.focal_refusal.is_none());
    assert!(pinhole
        .distortion_refusal
        .as_deref()
        .is_some_and(|why| why.contains("camera 3")));
    let spline = &gates.cameras[0];
    assert!(spline.focal_refusal.is_none() && spline.distortion_refusal.is_none());
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
fn drawing_a_row_clears_its_distortion_when_its_focal_is_clear() {
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(ReconId::next(), "kerry".to_string(), rig());
    prompt.pending.as_mut().unwrap().rows[0].distortion = true;

    assert_eq!(frame(&mut prompt, Vec::new()), None);

    let pending = prompt.pending.as_ref().expect("still up");
    assert!(
        !pending.rows[0].distortion,
        "the drawn row kept its distortion"
    );
}
