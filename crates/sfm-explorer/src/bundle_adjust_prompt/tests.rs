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
    let pinhole = camera_model == "SFMTOOL_PINHOLE";
    CameraGate {
        camera,
        camera_model,
        images: 3,
        focal_refusal: (!focal).then(|| format!("camera {camera} cannot release its focal")),
        distortion_refusal: (!distortion)
            .then(|| format!("camera {camera} has no distortion to release")),
        image_radial_bspline_refusal: (!pinhole)
            .then(|| format!("camera {camera} cannot release a radial correction")),
        image_rolling_shutter_refusal: (!pinhole)
            .then(|| format!("camera {camera} cannot release a rolling shutter")),
        scanlines: None,
    }
}

/// A rig: a spline camera 0, an `OPENCV_FISHEYE` camera 1 that can release
/// nothing, and a `SIMPLE_PINHOLE` camera 3 with a focal and no distortion.
/// Camera 2 is in the image set and no posed image uses it, so it has no row.
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
fn enter_runs_it_with_every_camera_held_and_the_points_crossing_which_is_the_default() {
    let mut prompt = BundleAdjustPrompt::default();
    let id = ReconId::next();
    prompt.ask(id, "kerry".to_string(), rig());
    let pending = prompt.pending.as_ref().unwrap();
    assert_eq!(
        pending.rows.len(),
        3,
        "one row per camera the posed images use"
    );
    assert!(pending
        .rows
        .iter()
        .all(|r| !r.focal && !r.distortion && !r.principal_point));

    assert_eq!(
        frame(&mut prompt, press(egui::Key::Enter)),
        Some(BundleAdjustAnswer {
            recon: id,
            releases: vec![HELD; 4],
            free_points: FreePointPolicy::CROSS,
            schedule: DEFAULT_SCHEDULE.to_vec(),
            rolling_shutter_scanlines: vec![None; 4],
        })
    );
    assert!(prompt.pending.is_none(), "the answered dialog stayed up");
}

#[test]
fn clearing_the_crossing_checkbox_keeps_every_point_as_it_is() {
    let answer = rig_answer(|p| p.free_points_cross = false);
    assert_eq!(answer.free_points, FreePointPolicy::NO_CROSS);
    assert_eq!(answer.releases, vec![HELD; 4]);
}

#[test]
fn each_row_releases_its_own_camera_and_the_rest_are_held() {
    let answer = rig_answer(|p| {
        p.rows[0] = RowState {
            focal: true,
            distortion: true,
            principal_point: false,
            ..RowState::default()
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
            principal_point: false,
            ..RowState::default()
        };
        p.rows[2] = RowState {
            focal: true,
            distortion: true,
            principal_point: false,
            ..RowState::default()
        };
    });
    assert_eq!(answer.releases, vec![HELD, HELD, HELD, FOCAL]);
}

#[test]
fn the_principal_point_is_answered_on_every_model_and_needs_no_focal() {
    // Camera 0 ticks the principal point alone; camera 1, the OPENCV_FISHEYE
    // that can release neither its focal nor its distortion, ticks it too;
    // camera 3 ticks it with its focal.
    let answer = rig_answer(|p| {
        p.rows[0].principal_point = true;
        p.rows[1].principal_point = true;
        p.rows[2] = RowState {
            focal: true,
            distortion: false,
            principal_point: true,
            ..RowState::default()
        };
    });
    let pp = CameraRelease {
        principal_point: true,
        ..HELD
    };
    let focal_and_pp = CameraRelease {
        principal_point: true,
        ..FOCAL
    };
    assert_eq!(answer.releases, vec![pp, pp, HELD, focal_and_pp]);
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

fn round(trim_px: f64, loss_scale: f64) -> BaSchedule {
    BaSchedule {
        trim_px,
        loss_scale,
    }
}

#[test]
fn the_default_schedule_reads_as_the_box_writes_it() {
    assert_eq!(schedule_text(&DEFAULT_SCHEDULE), "(50, 5) (12, 2) (4, 1)");
    assert_eq!(
        parse_schedule("(50, 5) (12, 2) (4, 1)"),
        Ok(DEFAULT_SCHEDULE.to_vec())
    );
    // Every preset survives the round trip through the box.
    for (_, preset) in SCHEDULE_PRESETS {
        assert_eq!(
            parse_schedule(&schedule_text(preset)).as_deref(),
            Ok(preset)
        );
    }
}

#[test]
fn a_schedule_takes_commas_spaces_and_an_unbounded_trim() {
    assert_eq!(
        parse_schedule("  (inf,1000), (8 , 2.5)(4,1) "),
        Ok(vec![
            round(f64::INFINITY, 1000.0),
            round(8.0, 2.5),
            round(4.0, 1.0)
        ])
    );
    assert_eq!(
        schedule_text(&[round(f64::INFINITY, 1000.0)]),
        "(inf, 1000)"
    );
}

#[test]
fn a_schedule_the_box_does_not_state_says_why() {
    for (text, expected) in [
        ("", "no rounds"),
        ("   ", "no rounds"),
        ("50, 5", "Round 1 does not start with '('"),
        ("(50, 5) (12, 2", "Round 2 has no closing ')'"),
        ("(50)", "Round 1, (50), is not two numbers"),
        ("(50, 5, 1)", "is not two numbers"),
        ("(fifty, 5)", "is not two numbers"),
        ("(0, 5)", "Round 1 of the schedule has trim 0"),
        ("(4, 1) (-inf, 1)", "Round 2 of the schedule has trim -inf"),
        ("(NaN, 1)", "has trim NaN"),
        ("(4, 0)", "has loss scale 0"),
        ("(4, inf)", "has loss scale inf"),
        ("(4, nan)", "has loss scale NaN"),
    ] {
        let why = parse_schedule(text).expect_err(text);
        assert!(why.contains(expected), "{text:?}: {why}");
        assert!(why.starts_with(char::is_uppercase), "{why}");
        assert!(!why.contains("  "), "{why}");
    }
}

#[test]
fn enter_runs_the_schedule_the_box_states() {
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(
        ReconId::next(),
        "bull".to_string(),
        BundleAdjustGates::default(),
    );
    assert_eq!(
        prompt.schedule_box(),
        Some("(50, 5) (12, 2) (4, 1)"),
        "the box opens on the default"
    );
    prompt.pending.as_mut().unwrap().schedule_text = "(inf, 1000)".to_string();

    let answer = frame(&mut prompt, press(egui::Key::Enter)).expect("one answer");
    assert_eq!(answer.schedule, vec![round(f64::INFINITY, 1000.0)]);
}

#[test]
fn enter_does_not_run_a_schedule_the_box_does_not_state() {
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(
        ReconId::next(),
        "bull".to_string(),
        BundleAdjustGates::default(),
    );
    prompt.pending.as_mut().unwrap().schedule_text = "(4, 0)".to_string();

    assert_eq!(frame(&mut prompt, press(egui::Key::Enter)), None);
    assert!(
        prompt.pending.is_some(),
        "the dialog closed on a bad schedule"
    );
}

#[test]
fn the_dialog_opens_on_a_node_with_the_schedule_last_run_there() {
    let mut prompt = BundleAdjustPrompt::default();
    let adjusted = ReconId::next();
    let other = ReconId::next();
    prompt.remember_schedule(adjusted, &[round(f64::INFINITY, 1000.0)]);

    prompt.ask(adjusted, "bull".to_string(), BundleAdjustGates::default());
    assert_eq!(prompt.schedule_box(), Some("(inf, 1000)"));
    assert_eq!(frame(&mut prompt, press(egui::Key::Escape)), None);

    prompt.ask(other, "other".to_string(), BundleAdjustGates::default());
    assert_eq!(prompt.schedule_box(), Some("(50, 5) (12, 2) (4, 1)"));
}

#[test]
fn the_per_image_releases_are_answered_without_the_shared_one_of_their_kind() {
    // Camera 0 ticks both per-image boxes beside its shared focal; camera 1,
    // the OPENCV_FISHEYE, ticks both and has its focal refused; camera 3 ticks
    // the per-image principal point beside the shared one.
    let answer = rig_answer(|p| {
        p.rows[0] = RowState {
            focal: true,
            image_focal: true,
            image_principal_point: true,
            ..RowState::default()
        };
        p.rows[1] = RowState {
            image_focal: true,
            image_principal_point: true,
            ..RowState::default()
        };
        p.rows[2] = RowState {
            principal_point: true,
            image_principal_point: true,
            ..RowState::default()
        };
    });
    let both = CameraRelease {
        image_focal: true,
        image_principal_point: true,
        ..HELD
    };
    let image_pp = CameraRelease {
        image_principal_point: true,
        ..HELD
    };
    assert_eq!(answer.releases, vec![both, image_pp, HELD, image_pp]);
}

#[test]
fn a_per_image_focal_carries_the_distortion() {
    let answer = rig_answer(|p| {
        p.rows[0] = RowState {
            distortion: true,
            image_focal: true,
            ..RowState::default()
        };
    });
    assert_eq!(
        answer.releases[0],
        CameraRelease {
            distortion: true,
            image_focal: true,
            ..HELD
        }
    );
}

#[test]
fn drawing_a_row_keeps_its_distortion_beside_a_per_image_focal_and_its_own_boxes() {
    // A drawn row whose per-image focal is ticked keeps its distortion, and
    // drawing the shared boxes ticked and greyed leaves what the user set in
    // them alone, so clearing a per-image box gives the shared one back.
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(ReconId::next(), "kerry".to_string(), rig());
    prompt.pending.as_mut().unwrap().rows[0] = RowState {
        distortion: true,
        image_focal: true,
        image_principal_point: true,
        ..RowState::default()
    };

    assert_eq!(frame(&mut prompt, Vec::new()), None);

    let row = prompt.pending.as_ref().expect("still up").rows[0];
    assert!(row.distortion, "the drawn row lost its distortion");
    assert!(
        !row.focal && !row.principal_point,
        "the shared boxes were set"
    );
}

/// Two `SFMTOOL_PINHOLE` cameras, camera 0 storing no scanlines setting and
/// camera 1 storing columns, and an `OPENCV_FISHEYE` camera 2, which can take
/// neither correction.
fn pinholes() -> BundleAdjustGates {
    BundleAdjustGates {
        camera_count: 3,
        cameras: vec![
            gate(0, "SFMTOOL_PINHOLE"),
            CameraGate {
                scanlines: Some(RollingShutterScanlines::Columns),
                ..gate(1, "SFMTOOL_PINHOLE")
            },
            gate(2, "OPENCV_FISHEYE"),
        ],
    }
}

#[test]
fn the_scanlines_choice_opens_on_the_stored_setting_or_unknown() {
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(ReconId::next(), "dino".to_string(), pinholes());
    let rows = &prompt.pending.as_ref().unwrap().rows;
    // No stored setting is never an assumed rows.
    assert_eq!(rows[0].scanlines, RollingShutterScanlinesChoice::Unknown);
    assert_eq!(rows[1].scanlines, RollingShutterScanlinesChoice::Columns);
    assert!(rows
        .iter()
        .all(|r| !r.image_radial_bspline && !r.image_rolling_shutter));
}

#[test]
fn the_corrections_are_answered_with_the_scanlines_of_each_camera_that_releases_its_rates() {
    let mut prompt = BundleAdjustPrompt::default();
    let id = ReconId::next();
    prompt.ask(id, "dino".to_string(), pinholes());
    {
        let rows = &mut prompt.pending.as_mut().unwrap().rows;
        rows[0].image_radial_bspline = true;
        rows[0].image_rolling_shutter = true;
        rows[1].image_rolling_shutter = true;
        rows[1].scanlines = RollingShutterScanlinesChoice::Rows;
        // Camera 2 is an OPENCV_FISHEYE: what its greyed boxes hold is never
        // answered.
        rows[2].image_radial_bspline = true;
        rows[2].image_rolling_shutter = true;
    }
    let answer = frame(&mut prompt, press(egui::Key::Enter)).expect("one answer");
    assert_eq!(
        answer.releases,
        vec![
            CameraRelease {
                image_radial_bspline: true,
                image_rolling_shutter: true,
                ..HELD
            },
            CameraRelease {
                image_rolling_shutter: true,
                ..HELD
            },
            HELD,
        ]
    );
    assert_eq!(
        answer.rolling_shutter_scanlines,
        vec![
            Some(RollingShutterScanlinesChoice::Unknown),
            Some(RollingShutterScanlinesChoice::Rows),
            None,
        ]
    );

    // The job runs with exactly these, and the core function's defaults for
    // the rest.
    let options = answer.clone().options();
    assert_eq!(options.releases, answer.releases);
    assert_eq!(
        options.rolling_shutter_scanlines,
        answer.rolling_shutter_scanlines
    );
    assert_eq!(options.schedule, answer.schedule);
    assert_eq!(options.free_points, answer.free_points);
    assert_eq!(
        options.image_prior,
        sfmtool_core::geometry::ImagePriorOptions::default()
    );
}

#[test]
fn a_scanlines_choice_is_answered_only_beside_a_released_rolling_shutter() {
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(ReconId::next(), "dino".to_string(), pinholes());
    {
        let rows = &mut prompt.pending.as_mut().unwrap().rows;
        rows[0].image_radial_bspline = true;
        rows[1].scanlines = RollingShutterScanlinesChoice::Rows;
    }
    let answer = frame(&mut prompt, press(egui::Key::Enter)).expect("one answer");
    assert_eq!(
        answer.releases[0],
        CameraRelease {
            image_radial_bspline: true,
            ..HELD
        }
    );
    assert_eq!(answer.rolling_shutter_scanlines, vec![None; 3]);
}

#[test]
fn drawing_a_row_keeps_its_corrections_and_its_scanlines_choice() {
    // Drawing the second line, with the Scanlines choice shown, leaves what
    // the user set alone, and a cleared rolling shutter keeps the choice for
    // when it is ticked again.
    let mut prompt = BundleAdjustPrompt::default();
    prompt.ask(ReconId::next(), "dino".to_string(), pinholes());
    {
        let rows = &mut prompt.pending.as_mut().unwrap().rows;
        rows[0].image_radial_bspline = true;
        rows[0].image_rolling_shutter = true;
        rows[0].scanlines = RollingShutterScanlinesChoice::Columns;
        rows[1].scanlines = RollingShutterScanlinesChoice::Rows;
    }

    assert_eq!(frame(&mut prompt, Vec::new()), None);

    let rows = &prompt.pending.as_ref().expect("still up").rows;
    assert!(rows[0].image_radial_bspline && rows[0].image_rolling_shutter);
    assert_eq!(rows[0].scanlines, RollingShutterScanlinesChoice::Columns);
    assert!(!rows[1].image_rolling_shutter);
    assert_eq!(rows[1].scanlines, RollingShutterScanlinesChoice::Rows);
}
