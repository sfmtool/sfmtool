// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Headless tests for the body's Viewed mode: the selected point drawn as the
//! viewed track, read-only, beside what Edited mode draws of the same track.
//!
//! Each frame asks for the viewed track first, as the dock does, and a box
//! that moves has its bars applied the way the dock applies them.

use sfmtool_core::bench::{BarCheck, Thresholds, Verdict};

use super::super::table::{proposal_tint, ColumnLayout};
use super::super::{
    crop_caption, BodyMode, TrackBody, TrackBodyResponse, EVALUATED_LABEL, EVALUATING_LABEL,
    GEOMETRY_SEARCH_LABEL, MAX_SELF_SIMILARITY_LABEL, MAX_SHIFT_LABEL, MIN_ZNCC_LABEL,
    MIN_ZNCC_MIDDLE_LABEL,
};
use super::{box_point, drag_frames, input, run_frame, run_frame_with, state, versions, POINT};
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::AppState;
use crate::track_view::EDIT_LABEL;

/// Select [`POINT`] with nothing focused and ask for its viewed track.
fn viewing() -> (AppState, ReconId, TrackBody, egui::Context) {
    let (mut state, id) = state();
    view(&mut state, id, POINT);
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, &state);
    (state, id, panel, ctx)
}

/// Select `point` and ask for its viewed track, as the dock does before it
/// draws the panel.
fn view(state: &mut AppState, id: ReconId, point: u32) {
    state.select_point(PointRef::new(id, point as usize));
    state.refresh_viewed_track();
}

/// [`viewing`] with the viewed track's evaluation landed.
fn viewing_measured() -> (AppState, ReconId, TrackBody, egui::Context) {
    let (mut state, id, mut panel, ctx) = viewing();
    state.settle_bench_evaluation();
    state.refresh_viewed_track();
    run_frame(&mut panel, &ctx, &state);
    (state, id, panel, ctx)
}

/// The strings one frame painted.
fn painted(panel: &mut TrackBody, ctx: &egui::Context, state: &AppState) -> Vec<String> {
    crate::test_support::painted_texts(ctx, input(Vec::new()), |ui| {
        panel.show(ui, state);
    })
}

/// The visuals the headless context draws with.
fn visuals(ctx: &egui::Context) -> egui::Visuals {
    ctx.global_style().visuals.clone()
}

/// The labels of every box, and a drag that makes each one stricter.
const BOXES: [(&str, f32); 5] = [
    (MIN_ZNCC_LABEL, 400.0),
    (MIN_ZNCC_MIDDLE_LABEL, 400.0),
    (MAX_SHIFT_LABEL, -400.0),
    (MAX_SELF_SIMILARITY_LABEL, -400.0),
    (GEOMETRY_SEARCH_LABEL, 400.0),
];

/// Drag the box labelled `label` `by` points, applying each frame's
/// read-only bars the way the dock does, and hand back every frame's
/// response.
fn drag_box(
    panel: &mut TrackBody,
    ctx: &egui::Context,
    state: &mut AppState,
    label: &str,
    by: f32,
) -> Vec<TrackBodyResponse> {
    let texts = crate::test_support::painted_text_rects(ctx, input(Vec::new()), |ui| {
        panel.show(ui, state);
    });
    let named = texts
        .iter()
        .find(|t| t.text == label)
        .unwrap_or_else(|| panic!("the {label:?} box is not drawn"))
        .rect;
    let start = box_point(label, named);
    let mut responses = Vec::new();
    for events in drag_frames(start, start + egui::vec2(by, 0.0)) {
        let response = run_frame_with(panel, ctx, state, events);
        if let Some(bars) = response.viewed_thresholds.clone() {
            state.set_viewed_thresholds(bars);
        }
        responses.push(response);
    }
    run_frame(panel, ctx, state);
    responses
}

#[test]
fn viewed_mode_draws_the_point_id_and_no_steps() {
    let (state, id, mut panel, ctx) = viewing();
    let response = run_frame(&mut panel, &ctx, &state);
    assert_eq!(response.mode, Some(BodyMode::Viewed));

    let texts = painted(&mut panel, &ctx, &state);
    let point_id = crate::scene::point_id(state.node(id).expect("loaded"), POINT as usize);
    assert!(texts.contains(&point_id), "no point ID: {texts:?}");
    for step in [
        "Fit",
        "Duplicate",
        "Commit",
        "Discard",
        "Rename",
        super::super::LOCK_LABEL,
    ] {
        assert!(
            !texts.iter().any(|t| t == step),
            "Viewed mode drew {step:?}: {texts:?}"
        );
    }
    assert!(
        !texts.iter().any(|t| t.starts_with("Split off")),
        "Viewed mode drew the split: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.starts_with("Stage:")),
        "Viewed mode drew the stage toggle: {texts:?}"
    );
    assert!(
        texts.contains(&format!("Tick {EDIT_LABEL} to work on this track.")),
        "no line saying how to change the track: {texts:?}"
    );
    // No switch and no pin: every row has a word in its verdict cell, and a
    // click there is only the row's.
    assert!(panel.rows().iter().all(|row| row.verdict_text.is_some()));
    let y = super::row_y(&mut panel, &ctx, &state, 0);
    let x = ColumnLayout::new().keep_x() + 20.0;
    let clicked = super::at_pointer(&mut panel, &ctx, &state, egui::pos2(x, y), true);
    assert_eq!(clicked.set_verdict, None, "a verdict cell took a click");
    assert_eq!(clicked.unpin_verdicts, None, "a verdict cell took a click");
    assert_eq!(clicked.pin_verdicts, None);
    assert_eq!(clicked.select_image, Some(0), "the click was not the row's");
}

/// Viewed mode's headings are Edited mode's at the same offsets, with
/// *Verdict* for *Keep* and no *From*; *Name*, the last column, moves left
/// into the room *From* leaves.
#[test]
fn the_viewed_headings_are_edited_mode_s_with_verdict_for_keep() {
    let cols = ColumnLayout::new();
    let edited: Vec<_> = cols
        .headers(BodyMode::Edited)
        .into_iter()
        .filter(|&(_, heading, _)| heading != "From")
        .map(|(x, heading, _)| {
            (
                if heading == "Name" {
                    cols.name_x(BodyMode::Viewed)
                } else {
                    x
                },
                if heading == "Keep" {
                    "Verdict"
                } else {
                    heading
                },
            )
        })
        .collect();
    let viewed: Vec<_> = cols
        .headers(BodyMode::Viewed)
        .into_iter()
        .map(|(x, heading, _)| (x, heading))
        .collect();
    assert_eq!(viewed, edited);
    assert_eq!(viewed.last().map(|&(_, h)| h), Some("Name"));

    let (state, _id, mut panel, ctx) = viewing();
    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.iter().any(|t| t == "Verdict"), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "Keep"), "{texts:?}");
    assert!(!texts.iter().any(|t| t == "From"), "{texts:?}");
    assert!(
        super::super::table::VERDICT_TIP.contains("nothing here changes that"),
        "the heading does not say the column changes nothing"
    );
}

#[test]
fn each_verdict_cell_reads_what_the_bars_give_it_in_its_colour() {
    let (mut state, _id, mut panel, ctx) = viewing_measured();
    let visuals = visuals(&ctx);
    let check = |panel: &TrackBody, state: &AppState| {
        let verdicts = state.viewed_verdicts().expect("a viewed track");
        for (row, verdict) in panel.rows().iter().zip(verdicts) {
            let word = match verdict {
                Some(Verdict::In) => "in",
                Some(Verdict::Out) => "out",
                None => "-",
            };
            assert_eq!(row.verdict_text.as_deref(), Some(word), "{row:?}");
            assert_eq!(row.proposal, verdict);
            assert_eq!(row.tint, proposal_tint(&visuals, verdict), "{row:?}");
            assert_eq!(row.verdict, Verdict::In, "a viewed row was turned out");
        }
    };
    check(&panel, &state);

    // Bars every reading clears: every row reads in, in a green cell.
    state.set_viewed_thresholds(Thresholds {
        min_zncc: -1.0,
        min_zncc_middle: 0.0,
        max_shift_px: 1.0e6,
        max_zncc_self_similarity_radius: 1.0e6,
        ..Thresholds::default()
    });
    run_frame(&mut panel, &ctx, &state);
    check(&panel, &state);
    assert!(
        panel
            .rows()
            .iter()
            .all(|row| row.verdict_text.as_deref() == Some("in")),
        "a row fails bars every reading clears: {:?}",
        panel.rows()
    );
    assert!(panel.rows()[0].keep_hover.contains("it clears every bar"));

    state.set_viewed_thresholds(Thresholds {
        min_zncc: 1.0,
        ..Thresholds::default()
    });
    run_frame(&mut panel, &ctx, &state);
    check(&panel, &state);
    let row = &panel.rows()[0];
    assert_eq!(row.verdict_text.as_deref(), Some("out"));
    assert!(
        row.keep_hover.contains("ZNCC whole is under the bar"),
        "{}",
        row.keep_hover
    );
}

#[test]
fn an_unmeasured_viewed_row_reads_a_dash_untinted() {
    // Before the evaluation lands the rows carry only what the point stores,
    // and the fixture stores no confidence column: nothing has measured them.
    let (state, _id, panel, _ctx) = viewing();
    let viewed = state.viewed_track().expect("a viewed track");
    assert!(viewed
        .track
        .observations
        .iter()
        .all(|o| o.track.as_ref().is_none_or(|m| m.zncc.is_none())));
    for row in panel.rows() {
        assert_eq!(row.verdict_text.as_deref(), Some("-"), "{row:?}");
        assert_eq!(row.tint, None, "{row:?}");
        assert!(row.keep_hover.contains("measured"), "{}", row.keep_hover);
    }
}

#[test]
fn a_drag_of_each_box_recolours_and_changes_no_verdict() {
    for (label, by) in BOXES {
        let (mut state, id, mut panel, ctx) = viewing_measured();
        let before_versions = versions(&state, id);
        let before_rows = state.action_log.entries().count();
        let before = state.viewed_thresholds.clone();
        let track = std::sync::Arc::clone(&state.viewed_track().expect("viewed").track);

        let responses = drag_box(&mut panel, &ctx, &mut state, label, by);
        assert!(
            responses.iter().any(|r| r.viewed_thresholds.is_some()),
            "the {label:?} box reported no bars"
        );
        assert_ne!(state.viewed_thresholds, before, "{label:?} did not move");
        assert_eq!(panel.thresholds(), &state.viewed_thresholds);
        for response in &responses {
            assert_eq!(response.apply_thresholds, None, "{label:?} applied a step");
        }
        assert_eq!(
            versions(&state, id),
            before_versions,
            "{label:?}: a version"
        );
        assert_eq!(
            state.action_log.entries().count(),
            before_rows,
            "{label:?}: an Action Log row"
        );
        let viewed = state.viewed_track().expect("viewed");
        assert!(
            std::sync::Arc::ptr_eq(&viewed.track, &track),
            "{label:?} changed the viewed track"
        );
        assert!(panel.rows().iter().all(|row| row.verdict == Verdict::In));

        // The ZNCC bars dragged to the top turn every measured row out.
        if label == MIN_ZNCC_LABEL {
            for row in panel.rows() {
                assert_eq!(row.checks[0][0], BarCheck::Fail, "{row:?}");
                assert_eq!(row.verdict_text.as_deref(), Some("out"), "{row:?}");
            }
        }
        if label == MIN_ZNCC_MIDDLE_LABEL {
            for row in panel.rows() {
                assert_eq!(row.checks[0][1], BarCheck::Fail, "{row:?}");
                assert_eq!(row.verdict_text.as_deref(), Some("out"), "{row:?}");
            }
        }
    }
}

#[test]
fn the_read_only_bars_survive_a_selection_change() {
    let (mut state, id, mut panel, ctx) = viewing();
    let strict = Thresholds {
        min_zncc: 0.9,
        ..Thresholds::default()
    };
    state.set_viewed_thresholds(strict.clone());
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds(), &strict);

    let other = (0..state.node(id).expect("loaded").edited().point_count() as u32)
        .find(|&p| p != POINT && state.node(id).expect("loaded").edited().point(p).is_some())
        .expect("another point");
    view(&mut state, id, other);
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(
        panel.thresholds(),
        &strict,
        "the bars reset on another point"
    );
    view(&mut state, id, POINT);
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.thresholds(), &strict);
}

#[test]
fn a_point_with_an_item_on_the_bench_names_it() {
    let (mut state, id) = state();
    let label = state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    state.unfocus_bench_item();
    state.refresh_viewed_track();
    assert!(state.viewed_track().is_some(), "the origin is not viewed");
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    let texts = painted(&mut panel, &ctx, &state);
    assert!(
        texts.contains(&format!(
            "On the bench as {label}. Tick {EDIT_LABEL} to open it."
        )),
        "{texts:?}"
    );
}

#[test]
fn the_viewed_header_carries_the_point_s_summary() {
    let (state, id, mut panel, ctx) = viewing();
    let summary = panel.summary().expect("a summary").clone();
    let node = state.node(id).expect("loaded");
    let stored = node.edited().point(POINT).expect("live").point().clone();
    assert_eq!(summary.point.as_ref(), Some(&stored));
    assert_eq!(summary.error_px, Some(f64::from(stored.error)));
    assert_eq!(summary.track_length, 3);
    assert!(summary.max_angle_deg > 0.0, "{summary:?}");

    let texts = painted(&mut panel, &ctx, &state);
    for expected in [
        format!("error: {:.2}px", stored.error),
        "track: 3 obs".to_string(),
        format!("max pair angle: {:.1}°", summary.max_angle_deg),
    ] {
        assert!(texts.contains(&expected), "no {expected:?}: {texts:?}");
    }
    assert!(
        texts.iter().any(|t| t.starts_with("xyzw: (")),
        "no coordinates: {texts:?}"
    );
}

#[test]
fn a_bench_track_carries_a_summary_and_a_cluster_none() {
    let (mut state, id) = state();
    state
        .put_point_on_bench(PointRef::new(id, POINT as usize), None)
        .expect("a live point");
    state.settle_bench_evaluation();
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, &state);
    let summary = panel.summary().expect("a track-stage summary").clone();
    assert_eq!(summary.point, None, "a bench track has no stored record");
    assert_eq!(summary.track_length, 3);
    assert!(summary.error_px.is_some(), "{summary:?}");
    assert!(summary.max_angle_deg > 0.0, "{summary:?}");
    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.contains(&"track: 3 obs".to_string()), "{texts:?}");

    state
        .start_bench_cluster(
            ImageRef::new(id, 0),
            &crate::bench::Seed::Pixel {
                pixel: [120.0, 90.0],
                radius_px: Some(6.0),
            },
            None,
        )
        .expect("a pixel on the sensor");
    run_frame(&mut panel, &ctx, &state);
    assert_eq!(panel.summary(), None, "a cluster has a summary");
    let texts = painted(&mut panel, &ctx, &state);
    assert!(
        !texts.iter().any(|t| t.starts_with("track: ")),
        "a cluster printed a track length: {texts:?}"
    );
}

#[test]
fn the_crop_caption_gives_the_pixel_and_the_feature_index() {
    let (state, id, panel, _ctx) = viewing_measured();
    let viewed = state.viewed_track().expect("viewed");
    for (k, row) in panel.rows().iter().enumerate() {
        let caption = row.crop_caption.as_deref().expect("a crop and its caption");
        let site = crate::bench::observation_site(&viewed.track.observations[k])
            .expect("placed")
            .pixel;
        assert!(
            caption.contains(&format!("pixel ({:.1}, {:.1})", site[0], site[1])),
            "{caption}"
        );
        // The fixture stores its keypoints, so the index is the observation's
        // place in the point's track.
        assert!(
            caption.contains(&format!("observation {k} of the point")),
            "{caption}"
        );
    }

    // A row named by its `.sift` feature says so.
    let edited = state.node(id).expect("loaded").edited();
    let mut row = viewed.track.observations[0].clone();
    row.provenance = sfmtool_core::bench::Provenance::Descriptor { feature: 847 };
    let caption = crop_caption(edited, None, &row).expect("placed");
    assert!(
        caption.contains("feature 847 of its .sift file"),
        "{caption}"
    );
    row.provenance = sfmtool_core::bench::Provenance::Pixel;
    let caption = crop_caption(edited, None, &row).expect("placed");
    assert!(caption.contains("No feature index"), "{caption}");
}

#[test]
fn a_frameless_point_shows_its_header_and_the_refusal_and_no_readings() {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(sfmtool_core::SfmrReconstruction::demo(12)));
    let id = state.selected_recon.expect("a selected reconstruction");
    view(&mut state, id, 3);
    let viewed = state.viewed_track().expect("a viewed track");
    let crate::bench::live::Evaluation::Refused(why) = viewed.evaluation.clone() else {
        panic!("a frameless point was not refused: {:?}", viewed.evaluation);
    };
    let mut panel = TrackBody::new();
    let ctx = egui::Context::default();
    run_frame(&mut panel, &ctx, &state);
    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.contains(&why), "no refusal sentence: {texts:?}");
    assert!(panel.summary().is_some(), "the header lost the summary");
    assert!(
        texts.iter().any(|t| t.starts_with("xyzw: (")),
        "no coordinates: {texts:?}"
    );
    assert!(!panel.rows().is_empty());
    for row in panel.rows() {
        assert!(!row.crop, "a frameless row drew a crop");
        assert!(!row.tile, "a frameless row drew a tile");
        assert!(row.cells[..4].iter().all(|c| c == "-"), "{row:?}");
        assert_eq!(row.verdict_text.as_deref(), Some("-"));
        assert_eq!(row.tint, None);
    }
}

#[test]
fn the_toolbar_says_evaluating_until_the_viewed_evaluation_lands() {
    let (mut state, _id, mut panel, ctx) = viewing();
    assert_eq!(
        panel.evaluation(),
        &crate::bench::live::Evaluation::Evaluating
    );
    let texts = painted(&mut panel, &ctx, &state);
    assert!(texts.iter().any(|t| t == EVALUATING_LABEL), "{texts:?}");

    state.settle_bench_evaluation();
    state.refresh_viewed_track();
    let texts = painted(&mut panel, &ctx, &state);
    assert_eq!(panel.evaluation(), &crate::bench::live::Evaluation::Current);
    assert!(texts.iter().any(|t| t == EVALUATED_LABEL), "{texts:?}");
    assert!(!texts.iter().any(|t| t == EVALUATING_LABEL), "{texts:?}");
}

#[test]
fn a_viewed_row_click_selects_and_reveals_and_a_double_click_asks_for_the_camera() {
    let (state, _id, mut panel, ctx) = viewing_measured();
    let y = super::row_y(&mut panel, &ctx, &state, 1);
    let at = egui::pos2(400.0, y);
    let clicked = super::at_pointer(&mut panel, &ctx, &state, at, true);
    assert_eq!(clicked.select_image, Some(1));
    assert!(clicked.reveal_feature.is_some(), "no pixel to reveal");
    assert_eq!(
        clicked.pick_row, None,
        "the viewed track has a row selection"
    );

    let button = |pressed| egui::Event::PointerButton {
        pos: at,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::default(),
    };
    let double = run_frame_with(
        &mut panel,
        &ctx,
        &state,
        vec![
            egui::Event::PointerMoved(at),
            button(true),
            button(false),
            button(true),
            button(false),
        ],
    );
    assert_eq!(double.request_camera_view, Some(1));
    assert!(double.reveal_feature.is_some());
}

#[test]
fn the_viewed_header_s_go_to_button_asks_for_the_dialog() {
    let (state, id, mut panel, ctx) = viewing();
    let point_id = crate::scene::point_id(state.node(id).expect("loaded"), POINT as usize);
    let rect = crate::test_support::painted_text_rects(&ctx, input(Vec::new()), |ui| {
        panel.show(ui, &state);
    })
    .into_iter()
    .find(|t| t.text == point_id)
    .expect("the point ID is drawn")
    .rect;
    let opened = (0..20).any(|step| {
        let at = egui::pos2(rect.right() + 2.0 + step as f32 * 3.0, rect.center().y);
        super::at_pointer(&mut panel, &ctx, &state, at, true).request_goto_point
    });
    assert!(opened, "no x beside the ID hit the go-to button");
}

#[test]
fn the_viewed_header_marks_a_point_at_infinity() {
    let header = |w: f64| {
        let summary = super::super::HeaderSummary {
            point: Some(sfmtool_core::Point3D {
                position: nalgebra::Point3::new(0.0, 0.0, 1.0),
                w,
                color: [128, 128, 128],
                error: 0.5,
                normal: nalgebra::Vector3::zeros(),
            }),
            error_px: Some(0.5),
            track_length: 3,
            at_infinity: w == 0.0,
            max_angle_deg: 0.0,
            depth_z: f32::NAN,
            condition: f32::NAN,
        };
        crate::test_support::painted_texts(
            &egui::Context::default(),
            egui::RawInput::default(),
            |ui| {
                super::super::show_viewed_header(ui, "pt3d_deadbeef_2", Some(&summary));
            },
        )
    };
    let at_infinity = header(0.0);
    let mark = at_infinity
        .iter()
        .position(|t| t == crate::track_view::INFINITY)
        .expect("no infinity mark on a point at infinity");
    let id = at_infinity
        .iter()
        .position(|t| t == "pt3d_deadbeef_2")
        .expect("no point ID");
    assert!(mark < id, "the mark is not left of the ID: {at_infinity:?}");
    assert!(at_infinity.iter().any(|t| t == "at infinity"));
    assert!(
        !header(1.0).iter().any(|t| t == crate::track_view::INFINITY),
        "a finite point carries the mark"
    );
}
