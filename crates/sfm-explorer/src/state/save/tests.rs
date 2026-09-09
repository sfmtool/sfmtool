// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Save and Save As end to end: what reaches the disk, what the history holds
//! afterwards, and what the dirty marker says on either side of it.

use std::path::PathBuf;

use sfmtool_core::{LineageMap, SfmrReconstruction};

use crate::scene::{PointRef, ReconId, SceneNode};
use crate::state::AppState;

/// A directory of this test's own under the system temp dir, emptied first so a
/// rerun does not read a previous run's file.
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sfm_explorer_save_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a writable temp dir");
    dir
}

/// A state holding one demo node that came from `dir/recon.sfmr`, selected.
///
/// Built with [`SceneNode::from_path`] rather than by reading a file back:
/// loading resolves a workspace directory, which a temp dir holding one `.sfmr`
/// and nothing else has none of, and none of these tests is about that. What
/// matters here is the state a loaded node is in -- a path, and a cursor at the
/// version the disk serial names, which is to say clean.
fn state_from_file(dir: &std::path::Path) -> (AppState, ReconId, PathBuf) {
    let path = dir.join("recon.sfmr");
    let mut state = AppState::new();
    let id = state.append_node(SceneNode::from_path(&path, SfmrReconstruction::demo(64)));
    (state, id, path)
}

/// The Action Log's texts, oldest first.
fn texts(state: &AppState) -> Vec<String> {
    state
        .action_log
        .entries()
        .map(|entry| entry.text.clone())
        .collect()
}

#[test]
fn a_loaded_node_starts_clean_and_an_edit_makes_it_dirty() {
    let dir = temp_dir("dirty");
    let (mut state, id, _) = state_from_file(&dir);
    assert!(!state.is_dirty(id));
    assert!(!state.any_dirty());

    state
        .delete_point(PointRef::new(id, 3))
        .expect("a live point");

    assert!(state.is_dirty(id));
    assert_eq!(state.dirty_labels(), vec!["recon".to_string()]);
}

#[test]
fn a_node_that_came_from_no_file_has_only_save_as() {
    // Untouched demo data is not unsaved work, so it carries no marker; what it
    // has no way of doing is writing over a file it never had.
    let mut state = AppState::new();
    let id = state.append_node(SceneNode::demo(SfmrReconstruction::demo(16)));

    assert!(!state.is_dirty(id));
    let error = state.save_node(id).expect_err("no file to write over");
    assert!(error.contains("came from no file"), "{error}");
}

#[test]
fn a_save_with_an_overlay_materialises_into_a_version_the_cursor_sits_on() {
    let dir = temp_dir("materialise");
    let (mut state, id, path) = state_from_file(&dir);
    let versions_before = state.scene[0].history.versions().len();

    state
        .delete_point(PointRef::new(id, 3))
        .expect("a live point");
    state.save_node(id).expect("a writable path");

    let node = &state.scene[0];
    // The deletion, then the materialisation: two more versions.
    assert_eq!(node.history.versions().len(), versions_before + 2);
    let current = node.history.current();
    assert!(current.deleted_points.is_empty());
    assert!(current.added.points.is_empty());
    // The disk mark is on the version the cursor is on, so nothing is dirty.
    assert_eq!(
        node.history.disk_serial(),
        node.history.current_version().serial
    );
    assert!(!node.is_dirty());

    // And the hash the session holds for that base is the file's own.
    let stored = sfmr_format::read_sfmr_content_hash(&path).expect("a written file");
    assert_eq!(
        current
            .base_content_hash()
            .expect("hashable")
            .content_xxh128,
        stored.content_xxh128
    );
}

#[test]
fn a_save_stamps_the_provenance_it_hashed() {
    let dir = temp_dir("provenance");
    let (mut state, id, path) = state_from_file(&dir);

    state
        .delete_point(PointRef::new(id, 3))
        .expect("a live point");
    state.save_node(id).expect("a writable path");

    let metadata = sfmr_format::read_sfmr_metadata(&path).expect("a written file");
    assert_eq!(metadata.operation, "edit");
    assert_eq!(metadata.tool, "sfm-explorer");
    assert_eq!(metadata.version, sfmr_format::SFMR_FORMAT_VERSION);
}

#[test]
fn a_save_records_the_lineage_of_the_base_it_came_from() {
    let dir = temp_dir("lineage");
    let (mut state, id, path) = state_from_file(&dir);
    let ancestor = state.scene[0]
        .edited()
        .base_content_hash()
        .expect("hashable")
        .content_xxh128
        .clone();

    state
        .delete_point(PointRef::new(id, 3))
        .expect("a live point");
    state.save_node(id).expect("a writable path");

    let metadata = sfmr_format::read_sfmr_metadata(&path).expect("a written file");
    let entry = metadata
        .lineage
        .iter()
        .find(|e| e.hash == ancestor)
        .expect("the base the session started from");
    assert_eq!(entry.kind, sfmr_format::LINEAGE_KIND_BASE);
    // A deletion preserves order, so the map is the small encoding, and it says
    // exactly which row went.
    match &entry.map {
        LineageMap::Monotone {
            source_rows,
            deleted,
            created,
        } => {
            assert_eq!(*source_rows, 64);
            assert_eq!(deleted, &vec![3]);
            assert!(created.is_empty());
        }
        other => panic!("a deletion should compress: {other:?}"),
    }
    // Rows either side of the deletion land where the map says they do.
    assert_eq!(entry.map.forward(2), Some(2));
    assert_eq!(entry.map.forward(3), None);
    assert_eq!(entry.map.forward(4), Some(3));
}

#[test]
fn an_ancestors_lineage_carries_forward_every_row_it_still_has() {
    // The node's base already records where an *earlier* content's rows went:
    // a monotone map whose only mentioned row is row 0, over 65 source rows. The
    // save has to compose all 65 of them into the file it writes, not just the
    // ones the map happens to name -- the rows a monotone map says nothing about
    // are exactly the ones that came through unchanged, and they are the bulk of
    // any real map. `source_rows` is what says where the domain ends; a
    // composition that guessed it from the highest mentioned row would enumerate
    // row 0 alone, find it deleted, and drop the whole ancestor.
    let dir = temp_dir("compose");
    let (mut state, id, path) = state_from_file(&dir);
    let grandparent = "aaaabbbbccccddddeeeeffff00001111";
    state.scene[0].recon_mut().metadata.lineage = vec![sfmr_format::LineageEntry {
        hash: grandparent.to_string(),
        kind: sfmr_format::LINEAGE_KIND_BASE.to_string(),
        map: LineageMap::Monotone {
            source_rows: 65,
            deleted: vec![0],
            created: vec![],
        },
    }];

    // An addition as well as a deletion, so the map the composition goes
    // through both loses a row and gains one rather than being a pure shift.
    let record = state.scene[0]
        .edited()
        .point(5)
        .expect("a live point")
        .to_record();
    state.scene[0]
        .history
        .current_mut()
        .add_point(record)
        .expect("a well-formed record");
    state
        .delete_point(PointRef::new(id, 3))
        .expect("a live point");
    state.save_node(id).expect("a writable path");

    let metadata = sfmr_format::read_sfmr_metadata(&path).expect("a written file");
    let entry = metadata
        .lineage
        .iter()
        .find(|e| e.hash == grandparent)
        .expect("the ancestor its own base recorded");
    assert_eq!(entry.map.source_rows(), 65);

    // The far survivor: the ancestor's last row was row 63 of the node's base
    // (row 0 having gone), and the save's deletion of row 3 moved it down one
    // more. It is past every row the ancestor's own map mentions, which is the
    // point of the test.
    assert_eq!(entry.map.forward(64), Some(62));
    // And what the ancestor map already said was gone is still gone.
    assert_eq!(entry.map.forward(0), None);
}

#[test]
fn an_id_from_before_a_save_still_resolves_after_it() {
    // What the lineage is for. The id a point is *shown* under moves onto the
    // file just written, since that is the file a reader now has; the id taken
    // before the save keeps landing on the same point, because the save recorded
    // where the earlier content's rows went.
    let dir = temp_dir("ids");
    let (mut state, id, _) = state_from_file(&dir);
    let before = crate::scene::point_id(&state.scene[0], 40);

    state
        .delete_point(PointRef::new(id, 3))
        .expect("a live point");
    state.save_node(id).expect("a writable path");

    let node = &state.scene[0];
    // Row 40 shifted down to 39 in the written file, and the id follows it there.
    let after = crate::scene::point_id(node, 39);
    assert_ne!(after, before);
    assert!(after.ends_with("_39"), "{after}");
    let hash = before.split('_').nth(1).expect("pt3d_<hash>_<index>");
    assert_eq!(crate::point_ids::resolve(node, hash, 40), Ok(39));
}

#[test]
fn a_save_with_no_overlay_writes_the_value_and_mints_nothing() {
    let dir = temp_dir("clean");
    let (mut state, id, _) = state_from_file(&dir);
    let versions_before = state.scene[0].history.versions().len();
    let hash_before = state.scene[0]
        .edited()
        .base_content_hash()
        .expect("hashable")
        .content_xxh128
        .clone();

    state.save_node(id).expect("a writable path");

    let node = &state.scene[0];
    assert_eq!(node.history.versions().len(), versions_before);
    assert_eq!(
        node.edited()
            .base_content_hash()
            .expect("hashable")
            .content_xxh128,
        hash_before
    );
    assert!(!node.is_dirty());
}

#[test]
fn save_as_writes_elsewhere_and_re_points_the_node() {
    let dir = temp_dir("save_as");
    let mut state = AppState::new();
    let id = state.append_node(SceneNode::demo(SfmrReconstruction::demo(16)));
    let path = dir.join("chosen.sfmr");

    state.save_node_as(id, &path).expect("a writable path");

    let node = &state.scene[0];
    assert_eq!(node.path.as_deref(), Some(path.as_path()));
    assert_eq!(node.label, "chosen");
    assert!(!node.is_dirty());
    assert!(path.exists());
}

#[test]
fn a_save_writes_one_log_entry_naming_the_path_and_the_version() {
    let dir = temp_dir("log");
    let (mut state, id, path) = state_from_file(&dir);

    state.save_node(id).expect("a writable path");

    let saved: Vec<String> = texts(&state)
        .into_iter()
        .filter(|text| text.starts_with("Saved "))
        .collect();
    assert_eq!(saved.len(), 1, "{saved:?}");
    assert!(saved[0].contains(&path.display().to_string()), "{saved:?}");
    assert!(
        saved[0].contains(&state.scene[0].history.current_version().serial.to_string()),
        "{saved:?}"
    );
}

#[test]
fn a_write_that_fails_is_refused_with_a_reason_and_changes_nothing() {
    let dir = temp_dir("refused");
    let (mut state, id, _) = state_from_file(&dir);
    // A directory is not a path a file can be written to.
    let blocked = dir.join("a-directory.sfmr");
    std::fs::create_dir_all(&blocked).expect("a writable temp dir");

    let error = state.save_node_as(id, &blocked).expect_err("a directory");

    assert!(error.contains("Cannot write"), "{error}");
    assert_eq!(
        state.scene[0].path.as_deref(),
        Some(dir.join("recon.sfmr").as_path())
    );
}

#[test]
fn the_window_title_marks_the_first_node_while_it_is_dirty() {
    let dir = temp_dir("title");
    let (mut state, id, _) = state_from_file(&dir);
    assert_eq!(state.window_title(), "SfM Explorer - recon.sfmr");

    state
        .delete_point(PointRef::new(id, 3))
        .expect("a live point");
    assert_eq!(state.window_title(), "SfM Explorer - *recon.sfmr");

    state.save_node(id).expect("a writable path");
    assert_eq!(state.window_title(), "SfM Explorer - recon.sfmr");
}
