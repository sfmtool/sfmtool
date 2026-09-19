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
use sfmtool_core::camera::remap::{ImageU8, ImageU8Pyramid};

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
///
/// Crate-visible: the background tests build their bench jobs over this, so the
/// test that holds each operation's cancellable declaration to its claim runs
/// the real work.
pub(crate) fn state() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(projected_embedded_demo(12)));
    let id = state.selected_recon.expect("a selected reconstruction");
    let camera = &state.scene[0].recon().image_table.cameras[0];
    let (w, h) = (camera.width, camera.height);
    for image in 0..state.scene[0].image_count() {
        // A pattern rather than a constant: a flat field gives the correlation
        // kernels nothing to register against, and a refusal would be the
        // fixture's rather than the code's. Its periods are a few pixels and
        // they differ between the axes, because the patch of the demo's frames
        // is only a few pixels across: a pattern coarser than the template
        // hands the localizability gate a one-dimensional tile, and every
        // member is refused before anything fits it.
        let data: Vec<u8> = (0..(w * h * 3))
            .map(|i| {
                let p = i / 3;
                ((p % w) % 9 * 14 + (p / w) % 7 * 18) as u8
            })
            .collect();
        state.full_res_cache.insert(
            ImageRef::new(id, image),
            Some(Arc::new(ImageU8Pyramid::from_image(
                ImageU8::new(w, h, 3, data),
                crate::state::PYRAMID_LEVELS,
            ))),
        );
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

/// The seed the Image Detail context menu makes: a pixel, with the node's own
/// default patch radius where the gesture starts a cluster and none where it
/// adds to one.
fn pixel_seed(pixel: [f64; 2], radius_px: Option<f64>) -> crate::bench::Seed {
    crate::bench::Seed::Pixel { pixel, radius_px }
}

/// Put [`POINT`] on the bench and give back the label it took.
pub(crate) fn put_on_bench(state: &mut AppState, id: ReconId) -> String {
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
        .start_bench_cluster(ImageRef::new(id, 0), &pixel_seed([120.0, 90.0], Some(6.0)))
        .expect("a pixel on the sensor")
        .label;
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
        .start_bench_cluster(ImageRef::new(id, 0), &pixel_seed([120.0, 90.0], Some(6.0)))
        .expect("a pixel on the sensor")
        .label;
    state
        .add_bench_observation(
            &label,
            ImageRef::new(id, 0),
            &pixel_seed([124.0, 93.0], None),
        )
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
        .start_bench_cluster(ImageRef::new(id, 0), &pixel_seed([120.0, 90.0], Some(6.0)))
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
        .start_bench_evaluate(id, &label, None)
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
    assert_eq!(
        state.selected_point.map(|point| point.point),
        Some(POINT),
        "the undo left the selection on a point the commit deleted"
    );
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

/// A commit selects the row it wrote, wherever the selection was standing.
///
/// The map carries a selection that was already on the origin, and says nothing
/// about one that was elsewhere -- and a commit is a gesture about one point
/// whose landing index is the one thing the person who asked for it cannot work
/// out.
#[test]
fn a_commit_selects_the_point_it_wrote_from_a_selection_elsewhere() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let elsewhere = PointRef::new(id, 0);
    state.select_point(elsewhere);

    let written = state.commit_bench_track(id, &label).expect("a track stage");

    assert_eq!(written.replaced, Some(POINT), "this one replaces");
    assert_eq!(
        state.selected_point,
        Some(PointRef::new(id, written.point as usize)),
        "the commit left the selection where it found it"
    );
}

/// The creating commit selects what it created, which no map could have done:
/// `PointMap::Created` is the identity forward, so a selection left to follow it
/// stays wherever it was.
#[test]
fn a_commit_that_creates_a_point_selects_the_point_it_created() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    // An origin that resolves to nothing leaves the commit creating a point
    // rather than replacing one.
    state
        .delete_point(PointRef::new(id, POINT as usize))
        .expect("a live point");
    state.deselect_point();
    let points = state.scene[0].point_count();

    let written = state.commit_bench_track(id, &label).expect("a track stage");

    assert_eq!(written.replaced, None, "this one creates");
    assert_eq!(
        state.scene[0].point_count(),
        points + 1,
        "a creation did not grow the point count"
    );
    assert_eq!(
        state.selected_point,
        Some(PointRef::new(id, written.point as usize)),
        "nothing was selected, and nothing is"
    );

    // The version that created the point is the one that holds it, so stepping
    // off it drops the selection rather than carrying it to an index that held
    // nothing -- which is what an undo across any creation does.
    state.undo(id).expect("the commit");
    assert_eq!(state.selected_point, None);
    state.redo(id).expect("the commit");
    assert_eq!(
        state.selected_point, None,
        "a redo invents no selection the undo dropped"
    );
}

/// Pressing *Commit* again on a track the point already holds writes nothing:
/// no version, no new index, and the no-effect row every bench step answers a
/// nothing-to-do with.
#[test]
fn committing_a_track_the_point_already_holds_pushes_no_version() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let first = state.commit_bench_track(id, &label).expect("a track stage");
    let after_first = versions(&state, id);
    let points = state.scene[0].point_count();
    let bench_address = item_address(&state, id, 0);
    state.action_log.clear();

    for _ in 0..4 {
        let written = state.commit_bench_track(id, &label).expect("a track stage");
        assert!(!written.changed, "a repeated commit wrote something");
        assert_eq!(written.point, first.point, "it named another point");
        assert_eq!(written.replaced, None, "nothing was replaced");
    }

    assert_eq!(
        versions(&state, id),
        after_first,
        "a version per press of the button"
    );
    assert_eq!(
        state.scene[0].point_count(),
        points,
        "a point per press of the button"
    );
    assert_eq!(
        item_address(&state, id, 0),
        bench_address,
        "the bench item was rebuilt for nothing"
    );
    assert!(
        state.scene[0].edited().point(first.point).is_some(),
        "the point the track is seated on stopped resolving"
    );
    assert_eq!(
        state.selected_point,
        Some(PointRef::new(id, first.point as usize)),
        "the point it named is not the selection"
    );
    let rows = rows(&state);
    assert_eq!(rows.len(), 4, "one row per press: {rows:?}");
    for (kind, text) in &rows {
        assert_eq!(*kind, Kind::Bench, "a no-effect row is a Bench row");
        assert_eq!(
            text,
            &format!(
                "Committed {label}: no effect, point {} already holds this track",
                first.point
            )
        );
    }
}

/// And the press after something moved writes again: the no-effect reading is
/// about what the point holds now, not about having committed once already.
#[test]
fn a_commit_after_a_sighting_is_turned_out_writes_again() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let first = state.commit_bench_track(id, &label).expect("a track stage");
    let before = versions(&state, id);

    state
        .set_bench_verdict(id, &label, 2, Verdict::Out)
        .expect("observation 2 exists");
    let written = state.commit_bench_track(id, &label).expect("a track stage");

    assert!(written.changed, "the track lost a sighting");
    assert_eq!(written.replaced, Some(first.point));
    assert_eq!(
        versions(&state, id) - before,
        2,
        "the verdict and the commit are a version each"
    );

    // And an undo of the commit takes the point back, so the next press writes
    // it again rather than reading the value as already holding the track.
    state.undo(id).expect("the commit");
    let again = state.commit_bench_track(id, &label).expect("a track stage");
    assert!(again.changed, "the point the commit wrote is gone");
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
    // One row per step, and the commit's own move of the selection onto the
    // point it wrote after it.
    assert_eq!(rows.len(), 5, "one row per step: {rows:?}");
    assert_eq!(
        rows.iter().map(|(kind, _)| *kind).collect::<Vec<_>>(),
        [
            Kind::Bench,
            Kind::Bench,
            Kind::Bench,
            Kind::Edit,
            Kind::Selection
        ],
        "only the commit's row is an Edit"
    );
    assert!(rows[0].1.starts_with("Turned "), "{}", rows[0].1);
    assert!(rows[3].1.starts_with("Committed track:"), "{}", rows[3].1);
    assert!(rows[4].1.starts_with("Selected point "), "{}", rows[4].1);
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

/// The photographs a photometric step reads are decoded **on the worker**.
///
/// Asserted by the one thing a headless test can see of it: a node with
/// nothing decoded and no readable photographs starts the task all the same,
/// and the refusal comes home through it. Decoding here would have refused the
/// gesture instead -- and would have spent the seconds of the read on the GUI
/// thread before the task it defers to had begun.
#[test]
fn a_photometric_step_decodes_on_the_worker() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state.full_res_cache.clear();
    state.action_log.clear();
    let before = versions(&state, id);

    state
        .start_bench_evaluate(id, &label, None)
        .expect("the task started");
    assert!(
        state.background_task().is_some(),
        "the step read the photographs here instead of deferring"
    );
    state.finish_background_task();

    assert_eq!(
        versions(&state, id),
        before,
        "a refused evaluation pushed a version"
    );
    let rows = rows(&state);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(rows[0].1.contains("Cannot read"), "{rows:?}");
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

/// What the track alone decides is decided **before** the worker, so a step
/// that was never going to happen costs no decode and refuses in the caller's
/// own hand.
///
/// The mirror of [`a_photometric_step_decodes_on_the_worker`]: that one says a
/// question about the photographs belongs to the task, and this one says a
/// question about the track does not.
#[test]
fn a_photometric_step_refuses_what_the_track_alone_decides() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    // Down to the cluster stage, then all but one sighting turned out: the
    // upgrade back has nothing to triangulate from.
    state
        .start_bench_stage(id, &label, StageKind::Cluster)
        .expect("the task started");
    state.finish_background_task();
    for observation in 1..3 {
        state
            .set_bench_verdict(id, &label, observation, Verdict::Out)
            .expect("a live observation");
    }
    state.action_log.clear();
    let before = versions(&state, id);

    let refusal = state
        .start_bench_stage(id, &label, StageKind::Track)
        .expect_err("one sighting triangulates nothing");
    assert_eq!(
        refusal,
        format!(
            "Cannot set the stage of {label}: 1 observations are in, and the track stage needs \
             two or more"
        )
    );
    assert!(
        state.background_task().is_none(),
        "a refusal started a task"
    );
    assert_eq!(versions(&state, id), before, "a refusal pushed a version");
    // One row, in the refusal's own words, which is what the panel's status
    // line shows.
    assert_eq!(rows(&state), vec![(Kind::Bench, refusal)]);
}

// ── The descriptor search ───────────────────────────────────────────────

/// The search is a bench step like the others: one version, one `Bench` row,
/// and the candidates it added are in the track under the search's own
/// provenance, at the place and the size the index's warp says.
///
/// The fixture is [`crate::descriptor_index::tests::searchable`]: a workspace of
/// real `.sift` files with one patch planted in the query image and carried
/// into two others under warps the test states, indexed into a real `.kdf`.
#[test]
fn a_descriptor_search_seeds_a_candidate_at_the_warped_pixel_and_shape() {
    use crate::descriptor_index::tests as fixture;

    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, label) = fixture::searchable(dir.path());
    state.action_log.clear();
    let before = versions(&state, id);
    let center = {
        let track = state.bench_track(id, &label).expect("on the bench");
        let keypoint = track.observations[0]
            .track
            .as_ref()
            .and_then(|m| m.keypoint)
            .expect("a committed track carries its keypoints");
        [f64::from(keypoint[0]), f64::from(keypoint[1])]
    };

    state
        .start_bench_descriptor_search(id, &label, 0, None, None)
        .expect("an index is open and the image has keypoints");
    assert!(state.background_task().is_some(), "the search ran inline");
    state.finish_background_task();

    assert_eq!(versions(&state, id), before + 1, "one version for the step");
    let track = state.bench_track(id, &label).expect("still on the bench");
    assert_eq!(
        track.observations.len(),
        4,
        "the three the point had, plus the one image the search found that it did not hold"
    );
    let added = track.observations.last().expect("the search added one");
    assert_eq!(added.image, fixture::FOUND_IMAGE);
    assert_eq!(added.verdict, sfmtool_core::bench::Verdict::Candidate);
    assert_eq!(
        added.provenance,
        sfmtool_core::bench::Provenance::Search {
            inliers: fixture::PLANTED as u32
        }
    );
    // The image the track already holds was found and left exactly as it was.
    assert!(
        track
            .observations
            .iter()
            .filter(|o| o.image == fixture::HELD_IMAGE)
            .count()
            == 1,
        "a search proposed a second sighting of an image the track already names"
    );

    let seed = added
        .cluster
        .as_ref()
        .expect("a searched seed is a cluster seed");
    let want = fixture::warp_to_found(center);
    assert!(
        (seed.seed_position[0] - want[0]).abs() < 1.0
            && (seed.seed_position[1] - want[1]).abs() < 1.0,
        "{:?} against {want:?}",
        seed.seed_position
    );
    // The shape is the observation's own under the warp's linear part. The
    // track came from a committed point and has no cluster seed of its own, so
    // what is warped is the identity `add_observation` falls back to.
    let linear = fixture::linear_of(fixture::warp_to_found);
    for (row, want) in seed.seed_shape.iter().zip(linear.iter()) {
        for (got, want) in row.iter().zip(want.iter()) {
            assert!((got - want).abs() < 1e-6, "{got} against {want}");
        }
    }

    // One row, of kind `Bench`, carrying the report's own sentence under the
    // item's name.
    let rows = rows(&state);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0].0, Kind::Bench);
    assert!(
        rows[0].1.starts_with(&format!(
            "{label}: Searched from observation 0 of 3: 2 images matched, 1 candidates added, \
             1 already in the track"
        )),
        "{}",
        rows[0].1
    );
}

/// With no index open the gesture is refused in the caller's own hand, naming
/// the row that would give it one, and starts no task.
#[test]
fn a_descriptor_search_with_no_index_is_refused_before_the_worker() {
    use crate::descriptor_index::tests as fixture;

    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = fixture::state_in(dir.path());
    let label = put_on_bench(&mut state, id);
    state.action_log.clear();
    let before = versions(&state, id);

    let why = state
        .bench_search_refusal(id, &label, 0)
        .expect("nothing to search");
    assert!(why.contains("No descriptor index is open"), "{why}");
    let refusal = state
        .start_bench_descriptor_search(id, &label, 0, None, None)
        .expect_err("the step asks the same question the menu does");
    assert_eq!(refusal, why);
    assert!(
        state.background_task().is_none(),
        "a refusal started a task"
    );
    assert_eq!(versions(&state, id), before, "a refusal pushed a version");
    assert_eq!(rows(&state), vec![(Kind::Bench, refusal)]);
}

/// A fit's **version label** carries the classification, not just the item.
///
/// Finite or at infinity is the fit's real outcome on a distant track, and a
/// history row reading only "Fitted X" hides the one thing a person scrolling it
/// is looking for -- which fit crossed the boundary, and on what evidence. The
/// Action Log keeps the whole report, counts and all.
#[test]
fn a_fit_s_version_label_carries_the_classification_and_the_log_the_whole_report() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    state.action_log.clear();
    state
        .start_bench_fit(id, &label, None)
        .expect("a framed track with three sightings fits");
    state.finish_background_task();

    let version = state
        .node(id)
        .expect("loaded")
        .history
        .versions()
        .last()
        .expect("the fit pushed a version")
        .label
        .clone();
    assert!(
        version.starts_with(&format!("Fitted {label}: ")),
        "the label should name the item and then say something: {version}"
    );
    assert!(
        version.contains("finite at (") || version.contains("at infinity along ("),
        "and what it says is which representation the rays earned: {version}"
    );
    assert!(
        version.contains("rms"),
        "with the residuals it was judged on: {version}"
    );

    let rows = rows(&state);
    let row = rows
        .iter()
        .find(|(kind, text)| *kind == Kind::Bench && text.starts_with("Fitted "))
        .map(|(_, text)| text.clone())
        .unwrap_or_else(|| panic!("no fit row in {rows:?}"));
    assert!(
        row.contains("placed") && row.contains("measured"),
        "the log row is the whole report: {row}"
    );
}

/// A drag that leaves and comes back to where it started is not a move, and the
/// exact comparison that decided it was is defeated by the pixel round trip: a
/// pointer's pixel becomes a ray, meets the patch's plane and is projected back,
/// and the number that comes home is the one it left to within the arithmetic's
/// last bits. The history filled with rows saying "by 0.000 units".
#[test]
fn a_drag_out_and_back_to_where_it_started_pushes_nothing() {
    let (mut state, id) = state();
    let label = put_on_bench(&mut state, id);
    let track = state.bench_track(id, &label).expect("on the bench").clone();
    let site = track.observations[0].site().expect("a sighting");

    let before = versions(&state, id);
    let out = crate::bench::PatchEdit::Translate {
        observation: 0,
        pixel: [site[0] + 4.0, site[1] - 3.0],
    };
    let edited = state
        .edit_bench_patch(id, &label, &out)
        .expect("a pixel on the sensor");
    assert!(edited.changed, "the drag out moved the patch");
    assert_eq!(versions(&state, id), before + 1);

    let back = crate::bench::PatchEdit::Translate {
        observation: 0,
        pixel: site,
    };
    let edited = state
        .edit_bench_patch(id, &label, &back)
        .expect("a pixel on the sensor");
    assert!(edited.changed, "the drag back moved it again");
    assert_eq!(versions(&state, id), before + 2);

    // And now the release on the place it already sits pushes nothing, with a
    // row that says why rather than silence.
    state.action_log.clear();
    let edited = state
        .edit_bench_patch(id, &label, &back)
        .expect("a pixel on the sensor");
    assert!(!edited.changed, "the patch already sits there");
    assert_eq!(versions(&state, id), before + 2, "a no-effect step pushed");
    let rows = rows(&state);
    assert_eq!(rows.len(), 1, "one row, saying nothing happened: {rows:?}");
    assert_eq!(rows[0].0, Kind::Bench);
    assert!(rows[0].1.contains("no effect"), "{}", rows[0].1);
    assert!(!rows[0].1.contains("0.000 units"), "{}", rows[0].1);
}
