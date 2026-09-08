// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The history's own semantics: the cursor, truncation, the maps that outlive
//! a truncation, and the budget that releases values without releasing maps.

use std::sync::Arc;

use sfmtool_core::{EditedReconstruction, RowMap, SfmrReconstruction};

use super::*;

/// A history over a small demo reconstruction.
fn history() -> History {
    History::new(SfmrReconstruction::demo(32), "Opened demo")
}

/// The value at `history`'s cursor with point `index` also deleted.
fn with_deleted(history: &History, index: u32) -> EditedReconstruction {
    let mut next = history.current().clone();
    next.delete_point(index).expect("a live point");
    next
}

/// Push a point edit deleting `index`.
fn push_delete(history: &mut History, index: u32) -> VersionSerial {
    let next = with_deleted(history, index);
    history.push(
        next,
        PointMap::Removed(vec![index]),
        format!("Deleted {index}"),
    )
}

#[test]
fn a_new_history_holds_one_version_and_no_maps() {
    let history = history();
    assert_eq!(history.versions().len(), 1);
    assert_eq!(history.cursor(), 0);
    assert_eq!(history.map_count(), 0);
    assert!(!history.can_undo());
    assert!(!history.can_redo());
}

#[test]
fn undo_and_redo_walk_the_cursor_without_moving_the_versions() {
    let mut history = history();
    push_delete(&mut history, 1);
    push_delete(&mut history, 2);
    let serials: Vec<VersionSerial> = history.versions().iter().map(|v| v.serial).collect();

    assert_eq!(history.cursor(), 2);
    assert_eq!(history.current().deleted_points.len(), 2);

    let (undone, now) = history.undo().expect("two edits to undo");
    assert_eq!((undone, now), (serials[2], serials[1]));
    assert_eq!(history.current().deleted_points.len(), 1);
    history.undo().expect("one more");
    assert_eq!(history.cursor(), 0);
    assert!(history.current().deleted_points.is_empty());
    assert!(!history.can_undo());

    let (from, redone) = history.redo().expect("a redo tail");
    assert_eq!((from, redone), (serials[0], serials[1]));
    assert_eq!(history.cursor(), 1);
    // The versions themselves never moved: walking the cursor is not an edit.
    let after: Vec<VersionSerial> = history.versions().iter().map(|v| v.serial).collect();
    assert_eq!(after, serials);
}

#[test]
fn an_edit_after_an_undo_truncates_the_redo_tail_and_keeps_its_maps() {
    let mut history = history();
    let first = push_delete(&mut history, 1);
    let discarded = push_delete(&mut history, 2);
    history.undo().expect("one to undo");

    let maps_before = history.map_count();
    let fresh = push_delete(&mut history, 3);

    // The discarded version is gone as a value...
    assert!(history.versions().iter().all(|v| v.serial != discarded));
    assert_eq!(history.versions().len(), 3);
    assert_eq!(history.current().deleted_points.len(), 2);
    assert!(!history.can_redo());
    // ...and its map is still there, alongside the new one.
    assert_eq!(history.map_count(), maps_before + 1);
    assert!(history.map_between(first, discarded).is_some());
    assert!(history.map_between(first, fresh).is_some());
    // Serials are never reused, so the new version is not the discarded one.
    assert_ne!(fresh, discarded);
}

#[test]
fn the_budget_releases_the_oldest_values_and_never_a_map() {
    let mut history = history();
    for index in 1..5u32 {
        push_delete(&mut history, index);
    }
    let maps = history.map_count();
    let versions = history.versions().len();

    // Charge every version far past the budget, then push one more so the
    // budget is enforced.
    for version in history.versions_mut_for_test() {
        version.unshared_bytes = HISTORY_BUDGET_BYTES;
    }
    push_delete(&mut history, 5);

    assert_eq!(history.versions().len(), versions + 1);
    assert_eq!(history.map_count(), maps + 1);
    assert!(
        history.versions().iter().any(|v| v.value.is_none()),
        "nothing was released under a budget every version alone exceeds"
    );
    // The cursor's version is never released, and undo refuses rather than
    // stepping onto one that was.
    assert!(history.current().deleted_points.contains(&5));
    let released_before_cursor = history.versions()[..history.cursor()]
        .iter()
        .filter(|v| v.value.is_none())
        .count();
    if released_before_cursor > 0 && history.versions()[history.cursor() - 1].value.is_none() {
        assert!(!history.can_undo());
    }
}

#[test]
fn a_point_edit_keeps_the_base_and_the_timestamps_do_not_go_backwards() {
    let mut history = history();
    let base = Arc::clone(&history.current().base);
    push_delete(&mut history, 1);
    push_delete(&mut history, 2);
    assert!(Arc::ptr_eq(&base, &history.current().base));
    let times: Vec<_> = history.versions().iter().map(|v| v.at).collect();
    assert!(times.windows(2).all(|w| w[0] <= w[1]));
    // A point edit's unshared cost is the overlay's, not the base's.
    assert!(history.versions()[1].unshared_bytes < 1024);
}

// ── The maps ────────────────────────────────────────────────────────────

#[test]
fn a_removal_map_keeps_every_surviving_index_where_it_was() {
    let map = PointMap::Removed(vec![2, 5]);
    assert_eq!(map.forward(1), Some(1));
    assert_eq!(map.forward(2), None);
    assert_eq!(map.forward(6), Some(6));
    assert_eq!(map.inverse(6), Some(6));
    assert_eq!(map.inverse(5), None);
}

/// A point-mask filter over `recon`, and the map the history stores for it:
/// the one `RowMap::by_scan` reads off the filter's input and output, which is
/// how a bulk edit's map is made.
fn dropping(recon: &SfmrReconstruction, keep: &[bool]) -> (SfmrReconstruction, PointMap) {
    let after = recon.filter_points_by_mask(keep);
    let map = RowMap::by_scan(recon, &after, None).expect("a filter keeps point order");
    (after, PointMap::Rows(map))
}

#[test]
fn a_row_map_closes_up_behind_what_it_removed_and_inverts() {
    let removed = [0u32, 2, 3];
    let before = SfmrReconstruction::demo(6);
    let keep: Vec<bool> = (0..6).map(|i| !removed.contains(&i)).collect();
    let (_, map) = dropping(&before, &keep);
    // Survivors 1, 4, 5 become 0, 1, 2.
    assert_eq!(map.forward(1), Some(0));
    assert_eq!(map.forward(4), Some(1));
    assert_eq!(map.forward(5), Some(2));
    assert_eq!(map.forward(0), None);
    assert_eq!(map.forward(3), None);
    // And back, landing on a live slot every time.
    for (new, old) in [(0u32, 1u32), (1, 4), (2, 5)] {
        assert_eq!(map.inverse(new), Some(old), "inverse of {new}");
        assert!(!removed.contains(&old));
    }
}

#[test]
fn a_chain_applies_its_steps_in_order_and_inverts_in_reverse() {
    // Remove 1 of 5, then remove what was 3 (2 after the first step).
    let before = SfmrReconstruction::demo(5);
    let (middle, first) = dropping(&before, &[true, false, true, true, true]);
    let (_, second) = dropping(&middle, &[true, true, false, true]);
    let chain = PointMap::Chain(vec![first, second]);
    assert_eq!(chain.forward(0), Some(0));
    assert_eq!(chain.forward(1), None);
    assert_eq!(chain.forward(2), Some(1));
    assert_eq!(chain.forward(3), None);
    assert_eq!(chain.forward(4), Some(2));
    for (new, old) in [(0u32, 0u32), (1, 2), (2, 4)] {
        assert_eq!(chain.inverse(new), Some(old));
    }
}
