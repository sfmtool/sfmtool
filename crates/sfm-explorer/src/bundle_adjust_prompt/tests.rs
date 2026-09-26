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

/// One camera row's gate: camera `camera`, `model`, with the refusals its
/// model gives.
fn gate(camera: usize, model: &'static str) -> CameraGate {
    let spline = matches!(model, "SFMTOOL_FISHEYE" | "SFMTOOL_PINHOLE");
    let focal = matches!(
        model,
        "SIMPLE_PINHOLE"
            | "EQUIDISTANT_FISHEYE"
            | "SIMPLE_RADIAL_FISHEYE"
            | "SFMTOOL_FISHEYE"
            | "SFMTOOL_PINHOLE"
    );
    let distortion = spline || model == "SIMPLE_RADIAL_FISHEYE";
    CameraGate {
        camera,
        model,
        images: 3,
        focal_refusal: (!focal).then(|| format!("camera {camera} cannot release its focal")),
        distortion_refusal: (!distortion)
            .then(|| format!("camera {camera} has no distortion to release")),
        spline,
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
        spline_coeff_counts: vec![8],
        spline_domains_deg: vec![150.0],
        ..BundleAdjustGates::default()
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
            spline_coeff_count: None,
            spline_domain_deg: None,
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
fn the_spline_rows_need_a_spline_camera_releasing_its_distortion() {
    // The pinhole's focal alone does not open them.
    let answer = rig_answer(|p| {
        p.rows[2].focal = true;
        p.keep_coeffs = false;
        p.coeff_count = 12;
    });
    assert_eq!(answer.spline_coeff_count, None);

    let answer = rig_answer(|p| {
        p.rows[0] = RowState {
            focal: true,
            distortion: true,
        };
        p.keep_coeffs = false;
        p.coeff_count = 12;
    });
    assert_eq!(answer.spline_coeff_count, Some(12));
}

/// Gates for a node with one camera, a spline camera whose coefficient counts
/// are `counts`, or a `SIMPLE_PINHOLE` where `counts` is empty.
fn with_splines(counts: &[usize]) -> BundleAdjustGates {
    let model = if counts.is_empty() {
        "SIMPLE_PINHOLE"
    } else {
        "SFMTOOL_FISHEYE"
    };
    BundleAdjustGates {
        camera_count: 1,
        cameras: vec![gate(0, model)],
        spline_coeff_counts: counts.to_vec(),
        ..BundleAdjustGates::default()
    }
}

/// Ask, release the camera's focal and distortion, and let `set` touch the
/// pending question before Enter answers it.
fn answered(counts: &[usize], set: impl FnOnce(&mut Pending)) -> BundleAdjustAnswer {
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(ReconId::next(), "kerry".to_string(), with_splines(counts));
    let pending = prompt.pending.as_mut().unwrap();
    pending.rows[0] = RowState {
        focal: true,
        distortion: true,
    };
    set(pending);
    frame(&mut prompt, press(egui::Key::Enter)).expect("one answer")
}

#[test]
fn the_coefficient_count_is_kept_by_default() {
    let answer = answered(&[8], |_| {});
    assert_eq!(answer.releases, vec![BOTH]);
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
        p.rows[0].distortion = false;
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

/// Gates for a node whose spline cameras end their domains at `domains` and
/// whose outermost keypoints are the given angles.
fn with_domains(
    domains: &[f64],
    observed_deg: Option<f64>,
    detected_deg: Option<f64>,
) -> BundleAdjustGates {
    let reach = |theta_deg: f64| KeypointReach {
        radius_px: 2.0 * theta_deg,
        theta_deg,
        image: 0,
        xy: [0.0, 0.0],
    };
    BundleAdjustGates {
        camera_count: 1,
        cameras: vec![gate(0, "SFMTOOL_FISHEYE")],
        spline_coeff_counts: vec![8],
        spline_domains_deg: domains.to_vec(),
        keypoint_extent: KeypointExtent {
            observed: observed_deg.map(reach),
            detected: detected_deg.map(reach),
        },
    }
}

/// Ask with `gates`, release the first camera's focal and distortion, let
/// `set` touch the pending question, and answer it with Enter.
fn answered_with(gates: BundleAdjustGates, set: impl FnOnce(&mut Pending)) -> BundleAdjustAnswer {
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(ReconId::next(), "kerry".to_string(), gates);
    let pending = prompt.pending.as_mut().unwrap();
    pending.rows[0] = RowState {
        focal: true,
        distortion: true,
    };
    set(pending);
    frame(&mut prompt, press(egui::Key::Enter)).expect("one answer")
}

#[test]
fn the_domain_is_kept_by_default_and_asked_for_only_when_it_changes() {
    let answer = answered_with(with_domains(&[150.0], None, None), |_| {});
    assert_eq!(answer.spline_domain_deg, None);

    let answer = answered_with(with_domains(&[150.0], None, None), |p| {
        p.keep_domain = false;
        p.domain_deg = 108.0;
    });
    assert_eq!(answer.spline_domain_deg, Some(108.0));

    // The domain every spline camera already has is no change.
    let answer = answered_with(with_domains(&[150.0], None, None), |p| {
        p.keep_domain = false
    });
    assert_eq!(answer.spline_domain_deg, None);

    // Not without the distortion release.
    let answer = answered_with(with_domains(&[150.0], None, None), |p| {
        p.rows[0].distortion = false;
        p.keep_domain = false;
        p.domain_deg = 108.0;
    });
    assert_eq!(answer.spline_domain_deg, None);
}

#[test]
fn the_button_takes_the_detected_keypoint_and_else_the_observed_one() {
    let answer = answered_with(with_domains(&[150.0], Some(101.2), Some(107.9)), |p| {
        p.use_outermost_keypoint()
    });
    assert_eq!(answer.spline_domain_deg, Some(107.9));

    let answer = answered_with(with_domains(&[150.0], Some(101.2), None), |p| {
        p.use_outermost_keypoint()
    });
    assert_eq!(answer.spline_domain_deg, Some(101.2));

    // With no keypoint the button changes nothing.
    let answer = answered_with(with_domains(&[150.0], None, None), |p| {
        p.use_outermost_keypoint()
    });
    assert_eq!(answer.spline_domain_deg, None);
}

#[test]
fn the_outermost_keypoint_is_labelled_by_its_source() {
    let extent = with_domains(&[150.0], Some(101.2), Some(107.9)).keypoint_extent;
    assert_eq!(
        extent.describe().as_deref(),
        Some("outermost keypoint: 202.4 px, 101.2° observed; 215.8 px, 107.9° detected")
    );
    let observed_only = with_domains(&[150.0], Some(101.2), None).keypoint_extent;
    assert_eq!(
        observed_only.describe().as_deref(),
        Some("outermost keypoint: 202.4 px, 101.2° observed")
    );
    assert_eq!(KeypointExtent::default().describe(), None);
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
