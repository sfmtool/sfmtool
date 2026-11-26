// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Scale to Median Measurement dialog and its gate, on the fixture in
//! `crate::measurements::fixture`: five distance measurements, three of which
//! give a factor (1.02 m, 1.02 m and 1.2 m per scene unit), one with an
//! endpoint refused and one with an endpoint at infinity.

use eframe::egui;
use sfmtool_core::{LengthUnit, SfmrReconstruction};

use super::*;
use crate::measurements::fixture::{keeping, measured_reconstruction, measured_state};
use crate::scene::SceneNode;
use crate::state::AppState;

/// One node made from `recon`, alone in a fresh state.
fn state_of(recon: SfmrReconstruction) -> (AppState, crate::scene::ReconId) {
    let mut state = AppState::new();
    let id = state.append_node(SceneNode::from_path(
        std::path::Path::new("measured.sfmr"),
        recon,
    ));
    (state, id)
}

/// Every string one frame of the dialog paints, after a frame to settle its
/// layout.
fn dialog_texts(state: &mut AppState) -> Vec<String> {
    let ctx = egui::Context::default();
    let input = || egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1200.0, 1000.0),
        )),
        ..Default::default()
    };
    let frame = |state: &mut AppState| {
        crate::test_support::painted_texts(&ctx, input(), |ui| {
            let pending = state.measurement_scale_prompt.pending();
            let node = pending.and_then(|p| crate::scene::node_by_id(&state.scene, p.recon));
            let _ = state.measurement_scale_prompt.show(ui.ctx(), node, None);
        })
    };
    frame(state);
    frame(state)
}

// ── The gate ────────────────────────────────────────────────────────────

#[test]
fn a_node_with_usable_measurements_is_not_refused() {
    let (state, id) = measured_state(Some("m"));
    assert_eq!(refusal(state.node(id).unwrap()), None);
}

#[test]
fn a_node_with_no_measurements_is_refused_in_cores_words() {
    let (state, id) = state_of(SfmrReconstruction::demo(10));
    assert_eq!(
        refusal(state.node(id).unwrap()).as_deref(),
        Some(
            "The reconstruction stores no distance measurement, so no stated length gives it a \
             scale."
        )
    );
}

#[test]
fn a_node_none_of_whose_measurements_is_usable_is_refused_with_each_reason() {
    // Measurement 2 has an endpoint refused and 3 one at infinity.
    let (state, id) = state_of(keeping(measured_reconstruction(Some("m")), &[2, 3]));
    let why = refusal(state.node(id).unwrap()).expect("refused");
    assert!(
        why.starts_with("None of the 2 distance measurements gives a scale; measurement 0:"),
        "{why}"
    );
    assert!(
        why.contains("measurement 1: endpoint B lies on a point at infinity"),
        "{why}"
    );
    assert!(why.ends_with('.'), "{why}");
}

// ── The default unit ────────────────────────────────────────────────────

#[test]
fn the_default_unit_is_the_files_own() {
    for (declared, unit) in [("cm", LengthUnit::Centimetre), ("ft", LengthUnit::Foot)] {
        let (state, id) = measured_state(Some(declared));
        assert_eq!(default_unit(state.node(id).unwrap()), unit);
    }
}

#[test]
fn with_no_file_unit_the_default_is_the_one_most_usable_measurements_state() {
    // The usable ones are stated in cm, m and m; the two refused ones are in
    // m too, so a count over every row would give the same answer, and the
    // next test is the one that tells them apart.
    let (state, id) = measured_state(None);
    assert_eq!(default_unit(state.node(id).unwrap()), LengthUnit::Metre);
}

#[test]
fn a_tie_goes_to_the_first_unit_and_unusable_rows_do_not_count() {
    // The three usable rows restated in ft, in and cm, one each: a tie, which
    // goes to cm, the first of the three in mm, cm, m, in, ft. The two
    // unusable rows, still in m, would make m the most common if they counted.
    let mut recon = measured_reconstruction(None);
    {
        let m = recon.image_set.measurements.distance.as_mut().unwrap();
        m.units[0] = LengthUnit::Foot.code();
        m.distances[0] = (1.53 / 0.3048) as f32;
        m.units[1] = LengthUnit::Inch.code();
        m.distances[1] = (3.06 / 0.0254) as f32;
        m.units[4] = LengthUnit::Centimetre.code();
        m.distances[4] = 180.0;
    }
    let (state, id) = state_of(recon);
    assert_eq!(
        default_unit(state.node(id).unwrap()),
        LengthUnit::Centimetre
    );
}

// ── What the dialog shows ───────────────────────────────────────────────

#[test]
fn the_view_of_the_fixture_in_millimetres_has_its_numbers() {
    let (state, id) = measured_state(Some("m"));
    let view = ScaleView::of(state.node(id).unwrap(), LengthUnit::Millimetre);
    let scale = view.scale.as_ref().expect("a scale");
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b.abs().max(1.0);
    assert!(close(scale.factor(), 1020.0), "{}", scale.factor());
    assert!(close(scale.statistics.mean, 1080.0));
    assert_eq!(scale.statistics.count, 3);
    assert!(scale.statistics.disagrees);
    let indexes: Vec<usize> = scale.factors.iter().map(|f| f.index).collect();
    assert_eq!(indexes, [0, 1, 4]);
    assert!(close(
        scale.factors[2].deviation_percent,
        100.0 * (1.2 / 1.02 - 1.0)
    ));
    let excluded: Vec<usize> = scale.excluded.iter().map(|e| e.index).collect();
    assert_eq!(excluded, [2, 3]);

    assert_eq!(
        scale_text(view.unit, scale.factor()),
        "1 scene unit = 1020 mm"
    );
    assert_eq!(
        correction_text(view.file_unit.as_deref(), view.unit, scale.factor()).as_deref(),
        Some("currently m; \u{d7}1.020 correction")
    );
    let lines = view.statistics_lines(id);
    assert_eq!(lines[0], "Used 3 of 5 distance measurements");
    assert_eq!(lines[1], "Median 1020, mean 1080");
    assert_eq!(
        lines.last().unwrap(),
        "Largest deviation +17.65% (Distance #4  paw width)"
    );
    let warning = view.disagreement().expect("a measurement is 17.6% off");
    assert!(
        warning.contains("1 measurement is more than 5% from the median"),
        "{warning}"
    );
}

#[test]
fn the_dialog_draws_the_factor_the_table_and_the_exclusions() {
    let (mut state, id) = measured_state(Some("m"));
    state.open_measurement_scale(id);
    let texts = dialog_texts(&mut state);
    for expected in [
        "Scale measured to Median Measurement",
        "1 scene unit = 1.020 m",
        "currently m; \u{d7}1.020 correction",
        "Used 3 of 5 distance measurements",
        "Distance #4  paw width",
        "153 cm",
        "+17.65%",
        "Left out",
    ] {
        assert!(
            texts.iter().any(|t| t == expected),
            "{expected:?} is not drawn: {texts:?}"
        );
    }
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("Distance #2: Endpoint B has no position")),
        "{texts:?}"
    );
}

#[test]
fn with_no_file_unit_the_dialog_says_so() {
    let (mut state, id) = measured_state(None);
    state.open_measurement_scale(id);
    let texts = dialog_texts(&mut state);
    assert!(
        texts
            .iter()
            .any(|t| t == "The file declares no unit: its coordinates are in scene units."),
        "{texts:?}"
    );
    assert!(
        texts.iter().any(|t| t == "1 scene unit = 1.020 m"),
        "{texts:?}"
    );
}

#[test]
fn a_single_measurement_and_equal_factors_make_one_bin() {
    for keep in [&[0][..], &[0, 1][..]] {
        let (state, id) = state_of(keeping(measured_reconstruction(Some("m")), keep));
        let scale = node_scale(state.node(id).unwrap(), LengthUnit::Metre).expect("a scale");
        let histogram = deviation_histogram(&scale);
        assert_eq!(histogram.counts(), [keep.len()], "{keep:?}");
    }
}

#[test]
fn a_closed_node_closes_the_dialog() {
    let (mut state, id) = measured_state(Some("m"));
    state.open_measurement_scale(id);
    let ctx = egui::Context::default();
    crate::test_support::painted_texts(&ctx, egui::RawInput::default(), |ui| {
        let _ = state.measurement_scale_prompt.show(ui.ctx(), None, None);
    });
    assert_eq!(state.measurement_scale_prompt.pending(), None);
}

#[test]
fn a_dialog_left_open_follows_an_edit_and_its_undo() {
    let (mut state, id) = measured_state(Some("m"));
    state.open_measurement_scale(id);
    let drawn = |state: &mut AppState, expected: &str| {
        let texts = dialog_texts(state);
        assert!(
            texts.iter().any(|t| t == expected),
            "{expected:?} is not drawn: {texts:?}"
        );
    };
    drawn(&mut state, "1 scene unit = 1.020 m");
    state
        .scale_node_to_median_measurement(id, LengthUnit::Metre)
        .expect("a scale");
    // The dialog is still up, on the scaled version, whose measurements now
    // agree with the unit.
    assert!(state.measurement_scale_prompt.pending().is_some());
    drawn(&mut state, "1 scene unit = 1.000 m");
    state.undo(id).expect("an undo");
    drawn(&mut state, "1 scene unit = 1.020 m");
}
