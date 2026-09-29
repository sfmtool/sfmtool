// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! *Find Nearby Tracks* as one version: what a find that lands existing points
//! leaves on the bench, what one that builds and commits new tracks leaves in
//! the reconstruction, how an undo and a redo take the whole of it back and
//! forth, what a find that finds nothing leaves (a row and no version), and
//! when the gesture is refused.
//!
//! The fixtures are *Create Track Here*'s plane capture
//! ([`crate::bench::track_at_pixel::tests`]): the near plane, whose held-out
//! grid point's pixel has eight existing points around it, and the same
//! capture of a plane so far out that only the far-field sweep finds anything.

use std::sync::Arc;

use super::{FoundNearby, Landing};
use crate::action_log::Kind;
use crate::bench::track_at_pixel::tests::{
    far_plane_state, lonely_pixel, plane_state, textured_pixel,
};
use crate::bench::track_at_pixel::{NOT_EMBEDDED_PATCHES, NOT_POSED};
use crate::scene::{ImageRef, ReconId, SceneNode};
use crate::state::AppState;

fn versions(state: &AppState, id: ReconId) -> usize {
    state.node(id).expect("loaded").history.versions().len()
}

fn live_points(state: &AppState, id: ReconId) -> usize {
    state.node(id).expect("loaded").edited().point_count()
}

fn bench_labels(state: &AppState, id: ReconId) -> Vec<String> {
    state
        .bench(id)
        .expect("loaded")
        .labels()
        .map(str::to_string)
        .collect()
}

/// The Action Log's rows written since `since` entries, as `(kind, failed,
/// text)`.
fn rows_after(state: &AppState, since: usize) -> Vec<(Kind, bool, String)> {
    state
        .action_log
        .entries()
        .skip(since)
        .map(|entry| (entry.kind, entry.failed, entry.text.clone()))
        .collect()
}

/// Run a find at `pixel` of image 0 to its end, and what it left behind.
fn find(
    state: &mut AppState,
    id: ReconId,
    pixel: [f64; 2],
    commit: bool,
    label: Option<&str>,
) -> FoundNearby {
    state
        .start_find_nearby_tracks(ImageRef::new(id, 0), pixel, commit, label)
        .expect("a posed image of an embedded_patches node");
    state.finish_background_task();
    let last = state
        .last_background_task
        .as_ref()
        .expect("the run finished");
    assert_eq!(last.operation.name, "Find nearby tracks");
    match last.found_nearby.as_deref() {
        Some(found) => found.clone(),
        None => panic!("the run found nothing to report: {:?}", last.outcome),
    }
}

// ── Existing points ────────────────────────────────────────────────────

/// At the held-out grid point's pixel the eight points around it come back as
/// existing points on one layer: each goes on the bench as its own track,
/// seated on its point and labelled with it, nothing is committed, and the
/// whole is one bench version. `1a` is focused and its point selected.
#[test]
fn existing_points_go_on_the_bench_as_their_own_tracks_in_one_version() {
    let (mut state, id) = plane_state();
    let versions_before = versions(&state, id);
    let points_before = live_points(&state, id);
    let rows_before = state.action_log.len();

    let found = find(&mut state, id, textured_pixel(), true, None);

    assert!(found.changed);
    assert_eq!(found.landed.len(), 8, "{:?}", found.landed);
    assert_eq!(versions(&state, id) - versions_before, 1, "one version");
    assert_eq!(live_points(&state, id), points_before, "nothing committed");

    let p = textured_pixel();
    let group = format!("image_0@{},{}", p[0].round() as i64, p[1].round() as i64);
    assert_eq!(found.found.group_label, group);
    let node = state.node(id).expect("loaded");
    let bench = node.history.current_bench();
    for (k, landing) in &found.landed {
        let Landing::Existing { item, point, .. } = landing else {
            panic!("an existing point, got {landing:?}");
        };
        let track = &found.found.tracks[*k];
        assert_eq!(track.point, Some(*point));
        assert_eq!(Some(item), track.label.as_ref());
        assert!(
            item.starts_with(&format!("{group} 1")) && item.ends_with(&format!(" pt {point}")),
            "{item}"
        );
        let on_bench = bench.track(item).expect("on the bench");
        assert_eq!(state.resolved_origin(node, on_bench), Some(*point));
    }
    let (first_item, first_point) = match &found.landed[0].1 {
        Landing::Existing { item, point, .. } => (item.clone(), *point),
        other => panic!("{other:?}"),
    };
    assert!(first_item.starts_with(&format!("{group} 1a pt ")));
    assert_eq!(state.focused_item_label(id), Some(first_item.as_str()));
    assert_eq!(state.selected_point.map(|p| p.point), Some(first_point));

    // One bench row, then the selection's.
    let rows = rows_after(&state, rows_before);
    assert_eq!(rows[0].0, Kind::Bench, "{rows:?}");
    assert!(
        !rows[0].1
            && rows[0]
                .2
                .starts_with("Found 8 nearby tracks in 1 layer at (")
            && rows[0].2.contains("8 existing points put on the bench")
            && rows[0].2.contains(&format!("editing {first_item}")),
        "{rows:?}"
    );
    assert_eq!(
        node.history.current_version().label,
        format!("Found 8 nearby tracks at {group}")
    );

    // A second find at the same pixel finds the same points on the bench
    // already, puts no second copy on and, with `1a` still focused, pushes no
    // version.
    let labels = bench_labels(&state, id);
    let versions_before = versions(&state, id);
    let rows_before = state.action_log.len();
    let again = find(&mut state, id, textured_pixel(), true, None);
    assert!(!again.changed);
    assert_eq!(versions(&state, id), versions_before);
    assert_eq!(bench_labels(&state, id), labels);
    let items: Vec<&str> = again.landed.iter().filter_map(|(_, l)| l.item()).collect();
    let before: Vec<&str> = found.landed.iter().filter_map(|(_, l)| l.item()).collect();
    assert_eq!(items, before);
    assert!(again
        .landed
        .iter()
        .all(|(_, l)| matches!(l, Landing::Existing { already: true, .. })));
    let rows = rows_after(&state, rows_before);
    assert!(
        rows[0].2.contains("8 existing points on the bench already")
            && rows[0]
                .2
                .ends_with("no effect, the bench holds them already"),
        "{rows:?}"
    );
}

// ── New tracks, committed ──────────────────────────────────────────────

/// On the far plane the far-field sweep's reading is the one track: it is
/// built, put on the bench under its label and committed as a new point, all
/// in one version whose row is an `Edit`. One undo takes back the point and
/// the bench item together; a redo puts both back.
#[test]
fn new_tracks_are_committed_in_the_same_version_and_one_undo_takes_them_back() {
    let (mut state, id, pixel) = far_plane_state();
    let versions_before = versions(&state, id);
    let points_before = live_points(&state, id);
    let labels_before = bench_labels(&state, id);
    let rows_before = state.action_log.len();

    let found = find(&mut state, id, pixel, true, None);

    let committed: Vec<(String, u32)> = found
        .landed
        .iter()
        .filter_map(|(_, l)| match l {
            Landing::Committed { item, point } => Some((item.clone(), *point)),
            _ => None,
        })
        .collect();
    assert!(!committed.is_empty(), "{:?}", found.landed);
    assert_eq!(versions(&state, id) - versions_before, 1, "one version");
    assert_eq!(live_points(&state, id), points_before + committed.len());
    let node = state.node(id).expect("loaded");
    for (item, point) in &committed {
        let track = node
            .history
            .current_bench()
            .track(item)
            .expect("on the bench");
        assert_eq!(state.resolved_origin(node, track), Some(*point), "{item}");
        assert!(item.starts_with("image_0@64,64 1"), "{item}");
    }
    let rows = rows_after(&state, rows_before);
    assert_eq!(rows[0].0, Kind::Edit, "{rows:?}");
    assert!(rows[0].2.contains("1 committed as a new point"), "{rows:?}");

    // The committed point has an id minted from the edit that created it.
    let id_text = crate::scene::point_id(node, committed[0].1 as usize);
    assert!(!id_text.is_empty());

    let after_points = live_points(&state, id);
    let after_labels = bench_labels(&state, id);
    state.undo(id).expect("an undo");
    assert_eq!(live_points(&state, id), points_before);
    assert_eq!(bench_labels(&state, id), labels_before);
    state.redo(id).expect("a redo");
    assert_eq!(live_points(&state, id), after_points);
    assert_eq!(bench_labels(&state, id), after_labels);
}

/// Told not to commit, the find puts the built tracks on the bench and writes
/// nothing to the reconstruction; its row is a `Bench` row. A second find at
/// the same pixel puts its tracks beside the first's, their labels taking the
/// `" (2)"` suffix.
#[test]
fn without_commit_the_tracks_go_on_the_bench_and_nothing_is_written() {
    let (mut state, id, pixel) = far_plane_state();
    let points_before = live_points(&state, id);
    let rows_before = state.action_log.len();

    let found = find(&mut state, id, pixel, false, Some("sky"));

    assert_eq!(live_points(&state, id), points_before);
    assert_eq!(found.found.group_label, "sky");
    let items: Vec<String> = found
        .landed
        .iter()
        .map(|(_, l)| match l {
            Landing::OnBench { item, why: None } => item.clone(),
            other => panic!("on the bench uncommitted, got {other:?}"),
        })
        .collect();
    assert_eq!(items[0], "sky 1a");
    assert_eq!(rows_after(&state, rows_before)[0].0, Kind::Bench);

    let again = find(&mut state, id, pixel, false, Some("sky"));
    assert_eq!(again.landed[0].1.item(), Some("sky 1a (2)"));
}

// ── Nothing found ──────────────────────────────────────────────────────

/// A pixel far from every point on the near plane: no source finds anything
/// and the far-field sweep reads nothing far, so the run writes one row that
/// says so, pushes no version and leaves the bench alone.
#[test]
fn a_find_with_nothing_usable_pushes_no_version() {
    let (mut state, id) = plane_state();
    let versions_before = versions(&state, id);
    let bench_before = Arc::clone(state.bench(id).expect("loaded"));
    let rows_before = state.action_log.len();

    let found = find(&mut state, id, lonely_pixel(), true, None);

    assert!(!found.changed);
    assert!(found.landed.is_empty());
    assert_eq!(versions(&state, id), versions_before);
    assert!(Arc::ptr_eq(&bench_before, state.bench(id).expect("loaded")));
    let rows = rows_after(&state, rows_before);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(
        !rows[0].1
            && rows[0]
                .2
                .starts_with("Found no nearby tracks at (2.0, 125.0) in image_0.jpg"),
        "{rows:?}"
    );
}

// ── Refusals ───────────────────────────────────────────────────────────

/// Refused in front of the worker with *Create Track Here*'s reasons: an image
/// with no pose, a pixel off the photograph, a busy node, and a node whose
/// observations are `.sift` features -- which only a find that commits is
/// refused on.
#[test]
fn find_nearby_tracks_is_refused_where_create_track_here_is() {
    let (mut state, id) = plane_state();
    let image = ImageRef::new(id, 0);
    assert_eq!(state.find_nearby_tracks_refusal(image, true), None);

    state.scene[0].recon_mut().image_table.images[1]
        .translation_xyz
        .x = f64::NAN;
    assert_eq!(
        state
            .find_nearby_tracks_refusal(ImageRef::new(id, 1), true)
            .as_deref(),
        Some(NOT_POSED)
    );

    let rows_before = state.action_log.len();
    assert!(state
        .start_find_nearby_tracks(image, [-5.0, 10.0], true, None)
        .is_err());
    assert!(state.background_task().is_none());
    let rows = rows_after(&state, rows_before);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(
        rows[0].1 && rows[0].2.contains("not on the 128x128 photograph"),
        "{rows:?}"
    );

    state
        .start_find_nearby_tracks(image, textured_pixel(), true, None)
        .expect("starts");
    let why = state.find_nearby_tracks_refusal(image, true).expect("busy");
    assert!(why.contains("is busy"), "{why}");
    state.cancel_background_task();
    state.finish_background_task();

    let mut sift = AppState::new();
    sift.append_node(SceneNode::demo(sfmtool_core::SfmrReconstruction::demo(4)));
    let sift_id = sift.selected_recon.expect("selected");
    let image = ImageRef::new(sift_id, 0);
    assert_eq!(
        sift.find_nearby_tracks_refusal(image, true).as_deref(),
        Some(NOT_EMBEDDED_PATCHES)
    );
    assert_eq!(
        sift.find_nearby_tracks_refusal(image, false).as_deref(),
        sift.posed_image_refusal(image).as_deref()
    );
}
