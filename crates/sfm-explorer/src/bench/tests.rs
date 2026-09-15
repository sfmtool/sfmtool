// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench as a half of the history: what each step does to the version list,
//! what an undo of one restores, and what the Action Log says about it.
//!
//! The steps themselves are `sfmtool-core`'s and are tested there, over a
//! synthetic textured plane whose numbers are known to the pixel
//! (`crates/sfmtool-core/src/bench/tests.rs`). What is under test here is the
//! half core does not have: one version per step, the document half left alone
//! by all but the commit, the dirty question asked of the document half only,
//! and a report that comes home to a bench the item has left.

use std::sync::Arc;

use sfmtool_core::bench::{BenchItem, StageKind, Verdict};
use sfmtool_core::camera::remap::ImageU8;

use crate::action_log::Kind;
use crate::background::{Finished, Operation};
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::edits::tests::projected_embedded_demo;
use crate::state::AppState;

// ── Fixtures ────────────────────────────────────────────────────────────

/// The point every test here puts on the bench: [`projected_embedded_demo`]
/// gives it three observations, in images 0, 1 and 2, at their exact
/// projections, which is what makes a triangulation of it well conditioned.
const POINT: u32 = 2;

/// A state holding one `embedded_patches` node with a photograph cached for
/// every image, so the steps that read pixels find them without a file on disk.
fn state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(projected_embedded_demo(12)));
    let id = state.selected_recon.expect("a selected reconstruction");
    let camera = &state.scene[0].recon().image_table.cameras[0];
    let (w, h) = (camera.width, camera.height);
    for image in 0..state.scene[0].image_count() {
        // A pattern rather than a constant: a flat field gives the correlation
        // kernels nothing to register against, and a refusal would be the
        // fixture's rather than the code's.
        let data: Vec<u8> = (0..(w * h * 3))
            .map(|i| ((i / 3) % 37 * 7 + (i / 3 / w) % 11 * 13) as u8)
            .collect();
        state
            .full_res_cache
            .insert(ImageRef::new(id, image), Some(ImageU8::new(w, h, 3, data)));
    }
    (state, id)
}

/// The node's one bench, at its cursor.
fn bench(state: &AppState, id: ReconId) -> &Arc<sfmtool_core::bench::Bench> {
    state.bench(id).expect("a loaded node has a bench")
}

/// How many versions the node holds.
fn versions(state: &AppState, id: ReconId) -> usize {
    state.node(id).expect("loaded").history.versions().len()
}

/// The `Arc` of the item at `position`, as an address, which is what says
/// whether a step shared it or rebuilt it.
fn item_address(state: &AppState, id: ReconId, position: usize) -> usize {
    match &bench(state, id).entries()[position].item {
        BenchItem::Track(track) => Arc::as_ptr(track) as usize,
    }
}

/// Put [`POINT`] on the bench and give back the label it took.
fn put_on_bench(state: &mut AppState, id: ReconId) -> String {
    state
        .put_point_on_bench(PointRef::new(id, POINT as usize))
        .expect("a live point")
}

/// The Action Log's entries, oldest first, as `(kind, text)`.
fn rows(state: &AppState) -> Vec<(Kind, String)> {
    state
        .action_log
        .entries()
        .map(|entry| (entry.kind, entry.text.clone()))
        .collect()
}

// ── The list, the activation and the discard ────────────────────────────

#[test]
fn putting_a_second_item_on_activating_and_discarding_are_three_versions() {
    let (mut state, id) = state();
    let first = put_on_bench(&mut state, id);
    let before = versions(&state, id);
    let first_address = item_address(&state, id, 0);

    let second = state
        .start_bench_cluster(ImageRef::new(id, 0), [120.0, 90.0], 6.0)
        .expect("a pixel on the sensor");
    assert_eq!(bench(&state, id).len(), 2);
    assert_eq!(
        crate::bench::active_track_label(bench(&state, id)),
        Some(second.as_str()),
        "a new item is the active one"
    );

    state.activate_bench_item(id, &first).expect("on the bench");
    assert_eq!(
        crate::bench::active_track_label(bench(&state, id)),
        Some(first.as_str())
    );

    state.discard_bench_item(id, &second).expect("on the bench");
    assert_eq!(bench(&state, id).len(), 1);
    assert_eq!(
        versions(&state, id) - before,
        3,
        "the three steps are three versions"
    );
    assert_eq!(
        item_address(&state, id, 0),
        first_address,
        "a step on the list rebuilt an item it did not touch"
    );

    state.undo(id).expect("a discard to undo");
    let bench = bench(&state, id);
    assert_eq!(bench.len(), 2, "the undo put the item back");
    assert_eq!(
        crate::bench::active_track_label(bench),
        Some(first.as_str()),
        "the undo restored the activation as it stood"
    );
    assert_eq!(item_address(&state, id, 0), first_address);
}

/// The two gestures the Image Detail context menu carries: a cluster started at
/// a pixel and a candidate added at one, each one version and one `Bench` row.
///
/// The second observation is in the **same image** as the first, which is what
/// the menu entry's rule turns on: a second sighting in one image joins as a
/// candidate like any other, and it is the verdict a track cannot hold twice.
#[test]
fn the_two_pixel_gestures_are_one_version_and_one_bench_row_each() {
    let (mut state, id) = state();
    state.action_log.clear();
    let before = versions(&state, id);

    let label = state
        .start_bench_cluster(ImageRef::new(id, 0), [120.0, 90.0], 6.0)
        .expect("a pixel on the sensor");
    state
        .add_bench_observation(&label, ImageRef::new(id, 0), [124.0, 93.0])
        .expect("a second sighting in one image is a candidate");

    assert_eq!(versions(&state, id) - before, 2);
    let track = state.bench_track(id, &label).expect("on the bench");
    assert_eq!(track.observations.len(), 2);
    assert_eq!(track.observations[1].image, 0, "the same image as the seed");
    assert_eq!(track.observations[1].verdict, Verdict::Candidate);

    let rows = rows(&state);
    assert_eq!(rows.len(), 2, "one row per step: {rows:?}");
    assert!(rows.iter().all(|(kind, _)| *kind == Kind::Bench));
    assert!(rows[0].1.starts_with("Started "), "{}", rows[0].1);
    assert!(rows[1].1.starts_with("Added "), "{}", rows[1].1);

    // And the undo walks them back one at a time, as every other step does.
    state.undo(id).expect("the added observation");
    assert_eq!(
        state
            .bench_track(id, &label)
            .expect("on the bench")
            .observations
            .len(),
        1,
    );
}

#[test]
fn putting_a_point_on_twice_activates_the_track_it_already_made() {
    let (mut state, id) = state();
    let first = put_on_bench(&mut state, id);
    state
        .start_bench_cluster(ImageRef::new(id, 0), [120.0, 90.0], 6.0)
        .expect("a pixel on the sensor");
    let before = versions(&state, id);

    let again = put_on_bench(&mut state, id);
    assert_eq!(again, first, "a second item was put on for one point");
    assert_eq!(bench(&state, id).len(), 2);
    assert_eq!(
        versions(&state, id) - before,
        1,
        "the activation is the one version it pushed"
    );
}

// ── Undo, redo and the document edit between two bench steps ────────────

#[test]
fn a_verdict_a_stage_change_and_an_evaluation_are_three_versions_undo_retraces_them() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let before = versions(&state, id);

    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("a track with a frame downgrades");
    state.finish_background_task();
    state
        .start_bench_evaluate(id, &label)
        .expect("a cluster evaluates over its seeds");
    state.finish_background_task();

    assert_eq!(versions(&state, id) - before, 3);
    let measured = state
        .bench_track(id, &label)
        .expect("still on the bench")
        .observations[0]
        .cluster
        .as_ref()
        .expect("the cluster stage seeds every observation")
        .zncc
        .is_some();
    assert!(measured, "the evaluation measured nothing");
    assert_eq!(
        state
            .bench_track(id, &label)
            .expect("on the bench")
            .stage_kind(),
        StageKind::Cluster
    );

    // Back out, one step at a time, and the bench is what it was at each.
    state.undo(id).expect("the evaluation");
    assert!(
        state
            .bench_track(id, &label)
            .expect("on the bench")
            .observations[0]
            .cluster
            .as_ref()
            .is_some_and(|m| m.zncc.is_none()),
        "the undo kept the evaluation's measurements"
    );
    state.undo(id).expect("the stage change");
    assert_eq!(
        state
            .bench_track(id, &label)
            .expect("on the bench")
            .stage_kind(),
        StageKind::Track
    );
    state.undo(id).expect("the verdict");
    assert_eq!(
        state
            .bench_track(id, &label)
            .expect("on the bench")
            .observations[1]
            .verdict,
        Verdict::In
    );

    // And forward again, which replays them in the order they were made.
    state.redo(id).expect("the verdict");
    assert_eq!(
        state
            .bench_track(id, &label)
            .expect("on the bench")
            .observations[1]
            .verdict,
        Verdict::Out
    );
    state.redo(id).expect("the stage change");
    state.redo(id).expect("the evaluation");
    assert_eq!(
        state
            .bench_track(id, &label)
            .expect("on the bench")
            .stage_kind(),
        StageKind::Cluster
    );
}

#[test]
fn a_document_edit_between_two_bench_steps_is_a_version_whose_undo_leaves_the_bench_alone() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    let before = versions(&state, id);
    let points = state.scene[0].point_count();

    // A point the bench knows nothing about, deleted between two bench steps.
    state
        .delete_point(PointRef::new(id, 5))
        .expect("a live point");
    state
        .set_bench_verdict(id, &label, 2, Verdict::Out)
        .expect("observation 2 exists");
    assert_eq!(versions(&state, id) - before, 2);

    state.undo(id).expect("the second verdict");
    state.undo(id).expect("the deletion");
    assert_eq!(state.scene[0].point_count(), points, "the point is back");
    let track = state.bench_track(id, &label).expect("on the bench");
    assert_eq!(
        track.observations[1].verdict,
        Verdict::Out,
        "undoing a document edit moved the bench"
    );
    assert_eq!(track.observations[2].verdict, Verdict::In);
}

// ── The commit ──────────────────────────────────────────────────────────

#[test]
fn a_commit_replaces_the_origin_point_and_an_undo_restores_the_pair() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state.selected_point = Some(PointRef::new(id, POINT as usize));
    let points = state.scene[0].point_count();
    let before = versions(&state, id);

    state.commit_bench_track(id, &label).expect("a track stage");

    assert_eq!(versions(&state, id) - before, 1, "a commit is one version");
    assert_eq!(
        state.scene[0].point_count(),
        points,
        "a replacement changed the point count"
    );
    let selected = state.selected_point.expect("the selection followed");
    assert_ne!(selected.point, POINT, "a replaced point takes a new index");
    assert!(
        state.scene[0].edited().point(selected.point).is_some(),
        "the selection followed to no point"
    );

    state.undo(id).expect("the commit");
    assert_eq!(state.scene[0].point_count(), points);
    assert!(
        state.scene[0].edited().point(POINT).is_some(),
        "the undo did not restore the point the commit replaced"
    );
    let track = state.bench_track(id, &label).expect("still on the bench");
    assert_eq!(
        track.origin.map(|origin| origin.point),
        Some(POINT),
        "the undo left the track seated on what the commit wrote"
    );
}

#[test]
fn a_commit_is_an_edit_row_and_every_other_bench_step_is_a_bench_row() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state.action_log.clear();

    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    state
        .apply_bench_thresholds(id, &label, &sfmtool_core::bench::Thresholds::default())
        .expect("on the bench");
    state
        .rename_bench_item(id, &label, "bull-nose")
        .expect("on the bench");
    state
        .commit_bench_track(id, "bull-nose")
        .expect("a track stage");

    let rows = rows(&state);
    assert_eq!(rows.len(), 4, "one row per step: {rows:?}");
    assert_eq!(
        rows.iter().map(|(kind, _)| *kind).collect::<Vec<_>>(),
        [Kind::Bench, Kind::Bench, Kind::Bench, Kind::Edit],
        "only the commit's row is an Edit"
    );
    assert!(rows[0].1.starts_with("Turned "), "{}", rows[0].1);
    assert!(rows[3].1.starts_with("Committed track:"), "{}", rows[3].1);
}

#[test]
fn a_step_that_changes_nothing_pushes_no_version() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let before = versions(&state, id);

    state
        .set_bench_verdict(id, &label, 0, Verdict::In)
        .expect("observation 0 is already in");
    state
        .start_bench_stage(id, &label, StageKind::Track)
        .expect("already at the track stage");
    state.activate_bench_item(id, &label).expect("on the bench");

    assert_eq!(versions(&state, id), before);
}

// ── Dirty ───────────────────────────────────────────────────────────────

#[test]
fn a_run_of_bench_steps_over_a_clean_value_is_clean_and_a_commit_is_dirty() {
    let (mut state, id) = state();
    // A node that came from no file is clean until something is done to it,
    // which is the state every other dirty test starts from.
    assert!(!state.is_dirty(id));

    let label = put_on_bench(&mut state, id);
    state
        .set_bench_verdict(id, &label, 1, Verdict::Out)
        .expect("observation 1 exists");
    assert!(
        !state.is_dirty(id),
        "a bench step made the reconstruction unsaved work"
    );

    state.commit_bench_track(id, &label).expect("a track stage");
    assert!(state.is_dirty(id), "a commit left the node clean");

    state.undo(id).expect("the commit");
    assert!(!state.is_dirty(id), "undoing the commit left it dirty");
}

// ── A report that comes home late ───────────────────────────────────────

#[test]
fn a_report_lands_on_the_item_it_measured() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let before = versions(&state, id);
    let mut measured = (**state.bench_track(id, &label).expect("on the bench")).clone();
    measured.observations[0]
        .track
        .as_mut()
        .expect("a slot")
        .zncc = Some(0.5);

    let item = label.clone();
    state
        .start_background_task(
            Operation::BENCH_EVALUATE,
            id,
            Box::new(move |_| Finished::BenchTrack {
                label: item,
                track: Box::new(measured),
                version_label: "Evaluated a track".to_string(),
                text: "Evaluated a track".to_string(),
            }),
        )
        .expect("nothing else is running");
    state.finish_background_task();

    assert_eq!(versions(&state, id) - before, 1);
    assert_eq!(
        state
            .bench_track(id, &label)
            .expect("on the bench")
            .observations[0]
            .track
            .as_ref()
            .and_then(|m| m.zncc),
        Some(0.5)
    );
}

#[test]
fn a_report_for_an_item_that_is_gone_is_discarded_with_one_row() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let measured = (**state.bench_track(id, &label).expect("on the bench")).clone();
    let before = versions(&state, id);
    state.action_log.clear();

    state
        .start_background_task(
            Operation::BENCH_EVALUATE,
            id,
            Box::new(move |_| Finished::BenchTrack {
                label: "nothing-is-called-this".to_string(),
                track: Box::new(measured),
                version_label: "Evaluated a track".to_string(),
                text: "Evaluated a track".to_string(),
            }),
        )
        .expect("nothing else is running");
    state.finish_background_task();

    assert_eq!(
        versions(&state, id),
        before,
        "a report with nothing to describe pushed a version"
    );
    let rows = rows(&state);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(
        rows[0].1.contains("no longer on"),
        "the row did not say why: {}",
        rows[0].1
    );
    assert_eq!(rows[0].0, Kind::Bench);
}
