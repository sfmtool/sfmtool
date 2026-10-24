// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The track-quality columns in the Track Table panel: their headings, their
//! first-click order and their cells, and Back-facing obs., which shows
//! itself and its count as Repeat obs. does.

use ndarray::Array2;
use sfmtool_core::analysis::track_table::TrackTableColumn;
use sfmtool_core::camera::sampler::{Sampler, SamplerChoice};
use sfmtool_core::patch::normal_refine::PatchWindow;
use sfmtool_core::reconstruction::{
    observation_reading_options, ObservationReading, ObservationReadings,
};
use sfmtool_core::SfmrReconstruction;

use super::{frame, input, shown, POINTS};
use crate::scene::{ReconId, SceneNode};
use crate::state::AppState;
use crate::table_sort::{SortRule, TableSort};
use crate::test_support::painted_texts;
use crate::track_table_panel::{
    cell_text, column_title, sort_hover, worst_first_ascending, TrackTablePanel,
};

/// A state holding one demo node of [`POINTS`] points with a patch frame on
/// every point, its normal towards the first camera that sees it, or away
/// from it (back-facing in both its views) for the points in `flipped`, and
/// observation readings whose blur-matched score of observation `j` is
/// `1 - j / 100`.
fn state_with_frames(flipped: &[usize]) -> (AppState, ReconId) {
    let mut recon = SfmrReconstruction::demo(POINTS);
    let n = recon.point_set.points.len();
    let mut u = Array2::<f32>::zeros((n, 3));
    let mut v = Array2::<f32>::zeros((n, 3));
    for p in 0..n {
        let first = recon.observations_for_point(p)[0].image_index as usize;
        let to_camera =
            recon.image_set.images[first].camera_center() - recon.point_set.points[p].position;
        let mut normal = to_camera.normalize();
        if flipped.contains(&p) {
            normal = -normal;
        }
        let a = normal.cross(&nalgebra::Vector3::z()).normalize();
        let b = normal.cross(&a);
        for k in 0..3 {
            u[[p, k]] = (a[k] * 0.05) as f32;
            v[[p, k]] = (b[k] * 0.05) as f32;
        }
    }
    recon.point_set.patch_u_halfvec_xyz = Some(u);
    recon.point_set.patch_v_halfvec_xyz = Some(v);
    recon.point_set.reference_observations = Some(vec![0; n]);
    let m = recon.point_set.tracks.len();
    let options = observation_reading_options(
        SamplerChoice::Fixed(Sampler::Bilinear),
        24,
        PatchWindow::Uniform,
    );
    let mut readings = ObservationReadings::not_measured(m, options);
    for (j, row) in readings.rows.iter_mut().enumerate() {
        *row = ObservationReading {
            ellipse_axes: [1.5, 0.5],
            blur_matched_bitmap_zncc: 1.0 - j as f32 / 100.0,
            ..ObservationReading::NOT_MEASURED
        };
    }
    recon.point_set.observation_readings = Some(readings);
    recon.rebuild_derived_fields();
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(recon));
    let id = state.scene[0].id;
    state.select_recon(id);
    (state, id)
}

fn texts(state: &mut AppState, panel: &mut TrackTablePanel, ctx: &egui::Context) -> Vec<String> {
    painted_texts(ctx, input(1200.0), |ui| panel.show(ui, state))
}

/// The headings name the column in a few words and the unit where it has
/// one.
#[test]
fn the_quality_columns_have_short_headings() {
    for (column, title) in [
        (TrackTableColumn::BackFacingObservations, "Back-facing obs."),
        (TrackTableColumn::MaxErrorPx, "Max. error (px)"),
        (
            TrackTableColumn::MinBlurMatchedBitmapZncc,
            "Min. bitmap ZNCC",
        ),
        (
            TrackTableColumn::ReferenceZnccSelfSimilarityRadius,
            "Ref. self-sim. radius",
        ),
        (
            TrackTableColumn::ReferenceZnccSelfSimilarityElongation,
            "Ref. elongation",
        ),
        (
            TrackTableColumn::MaxViewingAngleDeg,
            "Viewing angle (\u{b0})",
        ),
    ] {
        assert_eq!(column_title(column), title);
    }
}

/// The first click puts the bad end first: the lowest score, and the largest
/// count, error, radius, elongation and angle.
#[test]
fn the_first_click_on_a_quality_column_puts_the_bad_end_first() {
    for (column, descending) in [
        (TrackTableColumn::BackFacingObservations, true),
        (TrackTableColumn::MaxErrorPx, true),
        (TrackTableColumn::MinBlurMatchedBitmapZncc, false),
        (TrackTableColumn::ReferenceZnccSelfSimilarityRadius, true),
        (
            TrackTableColumn::ReferenceZnccSelfSimilarityElongation,
            true,
        ),
        (TrackTableColumn::MaxViewingAngleDeg, true),
    ] {
        assert_eq!(
            TableSort::first_click(column),
            TableSort { column, descending },
            "{column:?}"
        );
        assert_eq!(column.worst_first_descending(), descending);
        assert_eq!(worst_first_ascending(column), !descending);
        assert!(
            sort_hover(None, column).ends_with("worst first."),
            "{column:?}"
        );
    }
}

/// Back-facing obs. shows itself while a point has a back-facing
/// observation, the header counts those points, and its cells are whole
/// numbers; the lowest bitmap ZNCC prints two decimals.
#[test]
fn back_facing_observations_show_themselves_and_their_count() {
    let (mut state, id) = state_with_frames(&[5, 9, 30]);
    let (ctx, mut panel) = shown(&mut state);
    // The second run computes the column, which then shows itself.
    state.settle_track_table_columns();
    frame(&ctx, &mut panel, &mut state);
    let painted = texts(&mut state, &mut panel, &ctx);
    assert!(
        painted.iter().any(|t| t == "Back-facing obs."),
        "{painted:?}"
    );
    assert!(
        painted
            .iter()
            .any(|t| t == "\u{b7} 3 with back-facing observations"),
        "{painted:?}"
    );
    assert!(
        !painted.iter().any(|t| t.contains("repeat observations")),
        "{painted:?}"
    );
    assert!(!panel
        .query
        .shown
        .contains(&TrackTableColumn::BackFacingObservations));
    let table = state.track_table(id).unwrap();
    assert_eq!(
        cell_text(table, TrackTableColumn::BackFacingObservations, 5),
        "2"
    );
    assert_eq!(
        cell_text(table, TrackTableColumn::BackFacingObservations, 0),
        "0"
    );
    // Observations 0 and 1 of point 0 score 1 and 0.99.
    assert_eq!(
        cell_text(table, TrackTableColumn::MinBlurMatchedBitmapZncc, 0),
        "0.99"
    );

    // Its first click puts the back-facing points first.
    panel
        .query
        .click_heading(TrackTableColumn::BackFacingObservations);
    frame(&ctx, &mut panel, &mut state);
    assert_eq!(&panel.rows()[..3], [5, 9, 30]);
}

/// With no back-facing observation the column is computed, to find that
/// out, and stays hidden, and the header says nothing of it.
#[test]
fn with_no_back_facing_observation_the_column_stays_hidden() {
    let (mut state, id) = state_with_frames(&[]);
    let (ctx, mut panel) = shown(&mut state);
    state.settle_track_table_columns();
    frame(&ctx, &mut panel, &mut state);
    let painted = texts(&mut state, &mut panel, &ctx);
    assert!(
        state
            .track_table(id)
            .unwrap()
            .has(TrackTableColumn::BackFacingObservations),
        "computed to find out whether to show it"
    );
    assert!(
        !painted.iter().any(|t| t == "Back-facing obs."),
        "{painted:?}"
    );
    assert!(
        !painted
            .iter()
            .any(|t| t.contains("back-facing observations")),
        "{painted:?}"
    );
}

/// The self-similarity cells mark a lower bound with a `+`, as Track View
/// does: `3+` for a radius at the cap, the number and a `+` for one flagged
/// below it, the plain number for an exact one; the elongation takes a `+`
/// where only the long axis is flagged and has no value where the short one
/// is. A point's reading is its first observation's (its reference) and the
/// largest radius reads both.
#[test]
fn a_lower_bound_prints_with_a_plus() {
    let mut recon = SfmrReconstruction::demo(5);
    let n = recon.point_set.points.len();
    recon.point_set.reference_observations = Some(vec![0; n]);
    let options = observation_reading_options(
        SamplerChoice::Fixed(Sampler::Bilinear),
        24,
        PatchWindow::Uniform,
    );
    let mut readings = ObservationReadings::not_measured(recon.point_set.tracks.len(), options);
    let rows: [([f32; 2], [bool; 2]); 5] = [
        // Exact.
        ([1.25, 0.5], [false, false]),
        // An edge at the cap, exact across.
        ([3.0, 0.5], [true, false]),
        // Flagged below the cap, exact across.
        ([2.4142, 0.5], [true, false]),
        // A flat patch: both axes at the cap.
        ([3.0, 3.0], [true, true]),
        // The short axis alone flagged.
        ([2.0, 1.0], [false, true]),
    ];
    for (p, &(axes, is_at_least)) in rows.iter().enumerate() {
        let start = recon.point_set.observation_offsets[p];
        for k in start..start + recon.observations_for_point(p).len() {
            readings.rows[k] = ObservationReading {
                ellipse_axes: axes,
                ellipse_axes_is_at_least: is_at_least,
                ..ObservationReading::NOT_MEASURED
            };
        }
    }
    recon.point_set.observation_readings = Some(readings);
    recon.rebuild_derived_fields();
    let edited = sfmtool_core::EditedReconstruction::new(std::sync::Arc::new(recon));
    let columns = [
        TrackTableColumn::MaxZnccSelfSimilarityRadius,
        TrackTableColumn::ReferenceZnccSelfSimilarityRadius,
        TrackTableColumn::ReferenceZnccSelfSimilarityElongation,
    ];
    let table = sfmtool_core::analysis::track_table::TrackTable::compute(
        &edited,
        &columns,
        &sfmtool_core::progress::Progress::none(),
    )
    .unwrap();
    let text = |column, row| cell_text(&table, column, row);
    for column in &columns[..2] {
        assert_eq!(text(*column, 0), "1.25", "{column:?}");
        assert_eq!(text(*column, 1), "3+", "{column:?}");
        assert_eq!(text(*column, 2), "2.41+", "{column:?}");
        assert_eq!(text(*column, 3), "3+", "{column:?}");
        assert_eq!(text(*column, 4), "2.00", "{column:?}");
    }
    let elongation = TrackTableColumn::ReferenceZnccSelfSimilarityElongation;
    assert_eq!(text(elongation, 0), "2.50");
    assert_eq!(text(elongation, 1), "6.00+");
    assert_eq!(text(elongation, 2), "4.83+");
    assert_eq!(text(elongation, 3), "-");
    assert_eq!(text(elongation, 4), "-");
}
