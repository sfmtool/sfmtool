// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The `Refit spline…` dialog: its defaults, the answer it builds, the
//! outermost keypoint beside the domain, and the keys it answers to. The gates
//! are read off real values in `state/edits/tests.rs`.

use sfmtool_core::camera::{CameraIntrinsics, CameraModel};

use crate::scene::ReconId;

use super::*;

/// One frame of the dialog over an 800x600 screen, with `events` delivered to
/// it, reporting whatever it answered.
fn frame(prompt: &mut RefitSplinePrompt, events: Vec<egui::Event>) -> Option<RefitSplineAnswer> {
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

fn press(key: egui::Key) -> Vec<egui::Event> {
    vec![egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]
}

/// Gates for camera 1, an eight-coefficient `SFMTOOL_FISHEYE` on a 150°
/// domain, whose outermost keypoints are at the given angles.
fn gates(observed_deg: Option<f64>, detected_deg: Option<f64>) -> RefitSplineGates {
    let reach = |theta_deg: f64| KeypointReach {
        radius_px: 2.0 * theta_deg,
        theta_deg,
        image: 0,
        xy: [0.0, 0.0],
    };
    RefitSplineGates {
        camera: 1,
        camera_model: "SFMTOOL_FISHEYE",
        coeff_count: 8,
        domain_deg: 150.0,
        keypoint_extent: KeypointExtent {
            observed: observed_deg.map(reach),
            detected: detected_deg.map(reach),
        },
    }
}

/// Ask with `gates`, let `set` touch the pending question, and answer it with
/// Enter.
fn answered(gates: RefitSplineGates, set: impl FnOnce(&mut Pending)) -> RefitSplineAnswer {
    let mut prompt = RefitSplinePrompt::default();
    prompt.ask(ReconId::next(), "kerry".to_string(), gates);
    set(prompt.pending.as_mut().unwrap());
    let answer = frame(&mut prompt, press(egui::Key::Enter)).expect("one answer");
    assert!(prompt.pending.is_none(), "the answered dialog stayed up");
    answer
}

#[test]
fn a_dialog_that_was_never_asked_draws_nothing() {
    let mut prompt = RefitSplinePrompt::default();
    assert_eq!(frame(&mut prompt, Vec::new()), None);
}

#[test]
fn the_fields_start_at_what_the_camera_has_and_an_untouched_domain_is_kept() {
    let id = ReconId::next();
    let mut prompt = RefitSplinePrompt::default();
    prompt.ask(id, "kerry".to_string(), gates(None, None));
    let pending = prompt.pending.as_ref().unwrap();
    assert_eq!((pending.coeff_count, pending.domain_deg), (8, 150.0));
    // An ordinary frame leaves it up.
    assert_eq!(frame(&mut prompt, Vec::new()), None);
    assert!(prompt.pending.is_some());

    let answer = frame(&mut prompt, press(egui::Key::Enter)).expect("applied");
    assert_eq!(
        answer,
        RefitSplineAnswer {
            recon: id,
            camera: 1,
            coeff_count: 8,
            spline_domain_deg: None,
        }
    );
}

#[test]
fn the_answer_is_a_switch_of_the_camera_to_its_own_model() {
    let answer = answered(gates(None, None), |p| {
        p.coeff_count = 10;
        p.domain_deg = 108.0;
    });
    assert_eq!(
        (answer.coeff_count, answer.spline_domain_deg),
        (10, Some(108.0))
    );
    let request = answer.request();
    assert_eq!(request.camera, 1);
    assert_eq!(request.camera_model, None, "the camera's own model");
    assert_eq!(request.coeff_count, Some(10));
    assert_eq!(request.spline_domain_deg, Some(108.0));
    assert_eq!(request.theta_fit_deg, None, "a refit over the whole domain");
}

#[test]
fn escape_cancels_and_closes_the_dialog() {
    let mut prompt = RefitSplinePrompt::default();
    prompt.ask(ReconId::next(), "kerry".to_string(), gates(None, None));
    assert_eq!(frame(&mut prompt, press(egui::Key::Escape)), None);
    assert!(prompt.pending.is_none(), "escape left the dialog up");
}

#[test]
fn a_second_ask_does_not_stack_a_second_dialog() {
    let mut prompt = RefitSplinePrompt::default();
    let first = ReconId::next();
    prompt.ask(first, "kerry".to_string(), gates(None, None));
    prompt.ask(ReconId::next(), "other".to_string(), gates(None, None));
    let answer = frame(&mut prompt, press(egui::Key::Enter)).expect("one answer");
    assert_eq!(answer.recon, first);
}

#[test]
fn the_button_takes_the_detected_keypoint_and_else_the_observed_one() {
    let answer = answered(gates(Some(101.2), Some(107.9)), |p| {
        p.use_outermost_keypoint()
    });
    assert_eq!(answer.spline_domain_deg, Some(107.9));
    let answer = answered(gates(Some(101.2), None), |p| p.use_outermost_keypoint());
    assert_eq!(answer.spline_domain_deg, Some(101.2));
    let answer = answered(gates(None, None), |p| p.use_outermost_keypoint());
    assert_eq!(answer.spline_domain_deg, None);
}

#[test]
fn the_outermost_keypoint_is_labelled_by_its_source() {
    let extent = gates(Some(101.2), Some(107.9)).keypoint_extent;
    assert_eq!(
        extent.describe().as_deref(),
        Some("outermost keypoint: 202.4 px, 101.2° observed; 215.8 px, 107.9° detected")
    );
    let observed_only = gates(Some(101.2), None).keypoint_extent;
    assert_eq!(
        observed_only.describe().as_deref(),
        Some("outermost keypoint: 202.4 px, 101.2° observed")
    );
    assert_eq!(KeypointExtent::default().describe(), None);
}

#[test]
fn a_camera_without_a_spline_has_nothing_to_refit() {
    let pinhole = CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    };
    let why = refit_spline_refusal(&pinhole).expect("no spline");
    assert!(
        why.starts_with("A SIMPLE_PINHOLE camera has no spline"),
        "{why}"
    );
    let spline = CameraIntrinsics {
        model: CameraModel::SfmtoolFisheye {
            focal_length: 130.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            bspline_theta_max: 2.0,
            bspline: vec![0.0; 8],
        },
        width: 480,
        height: 480,
    };
    assert_eq!(refit_spline_refusal(&spline), None);
}
