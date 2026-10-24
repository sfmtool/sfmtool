// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The `.sift` build: what the entry is called, when it is refused, where it
//! writes, and the switch of the workspace definition it makes first on a node
//! whose definition names another feature tool.
//!
//! The fixture is an `embedded_patches` node saved as `demo.sfmr` in a
//! temporary directory that is also its workspace, with one of seoul_bull's
//! photographs at the path of each of its eight images and no `.sift` file.

use std::path::Path;

use sfmtool_core::features::sift::{
    BuildSiftFilesOptions, SFMTOOL_FEATURE_TOOL, WORKSPACE_CONFIG_FILE,
};

use super::{BUILD_SIFT_FILES, SWITCH_AND_BUILD_SIFT_FILES};
use crate::action_log::Kind;
use crate::scene::{ReconId, SceneNode};
use crate::state::edits::tests::projected_embedded_demo;
use crate::state::AppState;

const SEOUL_BULL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test-data/images/seoul_bull_sculpture"
);

/// The prefix directory the fixture's `colmap` definition names.
const COLMAP_PREFIX: &str = "features/sift-colmap-test";

/// The prefix directory the fixture's own `sfmtool` definition names, which is
/// not the default one, so a test can tell the node's options from the defaults.
const OWN_PREFIX: &str = "features/sift-sfmtool-own";

/// A state holding the fixture node, whose workspace definition names
/// `feature_tool`, with a photograph at each image's path in `dir`.
fn state_in(dir: &Path, feature_tool: &str) -> (AppState, ReconId) {
    let mut recon = projected_embedded_demo(12);
    recon.workspace_dir = dir.to_path_buf();
    recon.metadata.workspace.absolute_path = dir.display().to_string();
    recon.metadata.workspace.relative_path = ".".into();
    let contents = &mut recon.metadata.workspace.contents;
    contents.feature_tool = feature_tool.into();
    if feature_tool == SFMTOOL_FEATURE_TOOL {
        contents.feature_type = "sift-sfmtool".into();
        contents.feature_options = BuildSiftFilesOptions::sfmtool_defaults().feature_options;
        contents.feature_prefix_dir = OWN_PREFIX.into();
    } else {
        contents.feature_type = "sift-colmap".into();
        contents.feature_options = serde_json::json!({ "max_num_features": 8192 });
        contents.feature_prefix_dir = COLMAP_PREFIX.into();
    }
    for (i, image) in recon.image_set.images.iter().enumerate() {
        let from = Path::new(SEOUL_BULL).join(format!("seoul_bull_sculpture_{:02}.jpg", i + 1));
        std::fs::copy(from, dir.join(&image.name)).expect("the photograph is copied");
    }
    let mut node = SceneNode::demo(recon);
    node.path = Some(dir.join("demo.sfmr"));
    let mut state = AppState::new();
    state.append_node(node);
    let id = state.selected_recon.expect("a selected reconstruction");
    (state, id)
}

/// The fixture as the background tests start it: a `colmap` definition, so the
/// job would switch it and then extract every photograph.
pub(crate) fn state_with_photographs(dir: &Path) -> (AppState, ReconId) {
    state_in(dir, "colmap")
}

/// Whether every image of `id` has a `.sift` file where the node looks for it.
fn every_sift_file_is_there(state: &AppState, id: ReconId) -> bool {
    let recon = state.node(id).expect("loaded").recon();
    (0..recon.image_set.images.len()).all(|image| recon.sift_path_for_image(image).is_file())
}

/// The entry says it will switch the tool when the definition names another,
/// and is a plain build when it names `sfmtool`.
#[test]
fn the_label_says_whether_the_build_switches_the_feature_tool() {
    let colmap = tempfile::tempdir().unwrap();
    let (state, id) = state_in(colmap.path(), "colmap");
    assert_eq!(
        state.sift_files_build_label(id),
        SWITCH_AND_BUILD_SIFT_FILES
    );
    assert_eq!(
        state.sift_extraction_switch_from(id).as_deref(),
        Some("colmap")
    );

    let sfmtool = tempfile::tempdir().unwrap();
    let (state, id) = state_in(sfmtool.path(), "sfmtool");
    assert_eq!(state.sift_files_build_label(id), BUILD_SIFT_FILES);
    assert_eq!(state.sift_extraction_switch_from(id), None);
    assert_eq!(state.build_sift_files_refusal(id), None);
}

/// A node whose workspace directory is not there has no photographs to read,
/// and one whose observations are feature indexes cannot switch tools, since
/// the new files would hold other features. Each is refused in its own words,
/// and the starter logs the refusal.
#[test]
fn the_build_is_refused_without_a_workspace_or_when_a_switch_would_break_feature_indexes() {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(projected_embedded_demo(12)));
    let id = state.selected_recon.expect("a selected reconstruction");
    let refusal = state.build_sift_files_refusal(id).expect("refused");
    assert!(refusal.contains("workspace directory"), "{refusal}");
    assert!(state.start_build_sift_files(id).is_err());
    let entry = state.action_log.entries().next_back().expect("an entry");
    assert!(entry.failed && entry.text == refusal, "{}", entry.text);

    let dir = tempfile::tempdir().unwrap();
    let mut recon = sfmtool_core::SfmrReconstruction::demo(12);
    recon.workspace_dir = dir.path().to_path_buf();
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(recon));
    let id = state.selected_recon.expect("a selected reconstruction");
    let refusal = state.build_sift_files_refusal(id).expect("refused");
    assert!(refusal.contains("feature indexes"), "{refusal}");
    assert!(refusal.contains("'none'"), "{refusal}");
}

/// On a node that already extracts with `sfmtool`, the build writes every
/// image's file where the node looks for it, with the node's own options and
/// prefix directory, and pushes no version. A second build finds them all up
/// to date. Afterwards *Build Index Files* has `.sift` files to read.
#[test]
fn a_build_writes_every_file_where_the_node_looks_and_pushes_no_version() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path(), "sfmtool");
    let refusal = state.sift_sources_refusal(id).expect("no .sift file yet");
    assert!(refusal.contains("No .sift file"), "{refusal}");
    let versions = state.node(id).unwrap().history.versions().len();

    state.start_build_sift_files(id).expect("it starts");
    let polled = state.finish_background_tasks();
    let entry = state.action_log.entries().next_back().expect("an entry");
    assert!(!entry.failed, "{}", entry.text);
    assert_eq!(entry.kind, Kind::Bench);
    assert!(
        entry.text.contains("8 written and 0 already up to date"),
        "{}",
        entry.text
    );
    assert_eq!(state.node(id).unwrap().history.versions().len(), versions);
    // The frame drops the panels' caches of the node, which read the files.
    assert_eq!(polled.installed, vec![id]);

    assert!(every_sift_file_is_there(&state, id));
    let recon = state.node(id).unwrap().recon();
    assert!(recon
        .sift_path_for_image(0)
        .starts_with(dir.path().join(OWN_PREFIX)));
    assert_eq!(state.sift_sources_refusal(id), None);
    assert_eq!(state.build_index_files_refusal(id), None);

    state.start_build_sift_files(id).expect("it starts again");
    state.finish_background_tasks();
    let entry = state.action_log.entries().next_back().expect("an entry");
    assert!(
        entry.text.contains("0 written and 8 already up to date"),
        "{}",
        entry.text
    );
}

/// On a node whose definition names `colmap`, the build switches the
/// definition the node holds to `sfmtool` with its default options and prefix
/// directory, as one version once the files are built, and leaves the
/// workspace's `.sfm-workspace.json` alone. Undo puts the old definition back.
#[test]
fn a_switch_changes_the_node_s_definition_only_and_is_undone_as_one_version() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path(), "colmap");
    let on_disk =
        r#"{"feature_tool": "colmap", "feature_prefix_dir": "features/sift-colmap-test"}"#;
    std::fs::write(dir.path().join(WORKSPACE_CONFIG_FILE), on_disk).unwrap();
    let versions = state.node(id).unwrap().history.versions().len();

    state.start_build_sift_files(id).expect("it starts");
    state.finish_background_tasks();
    let entry = state.action_log.entries().next_back().expect("an entry");
    assert!(!entry.failed, "{}", entry.text);
    assert_eq!(entry.kind, Kind::Edit);
    assert!(entry.text.contains("from 'colmap'"), "{}", entry.text);

    let node = state.node(id).unwrap();
    assert_eq!(node.history.versions().len(), versions + 1);
    let defaults = BuildSiftFilesOptions::sfmtool_defaults();
    let contents = &node.recon().metadata.workspace.contents;
    assert_eq!(contents.feature_tool, "sfmtool");
    assert_eq!(contents.feature_type, "sift-sfmtool");
    assert_eq!(contents.feature_options, defaults.feature_options);
    assert_eq!(
        Path::new(&contents.feature_prefix_dir),
        defaults.feature_prefix_dir
    );
    assert!(every_sift_file_is_there(&state, id));
    assert!(!dir.path().join(COLMAP_PREFIX).exists());
    assert_eq!(
        std::fs::read_to_string(dir.path().join(WORKSPACE_CONFIG_FILE)).unwrap(),
        on_disk
    );
    assert_eq!(state.sift_files_build_label(id), BUILD_SIFT_FILES);

    state.undo(id).expect("the switch is undone");
    let contents = &state.node(id).unwrap().recon().metadata.workspace.contents;
    assert_eq!(contents.feature_tool, "colmap");
    assert_eq!(contents.feature_prefix_dir, COLMAP_PREFIX);
}

/// A build cancelled before it starts an image pushes no switch and leaves the
/// definition naming its old tool.
#[test]
fn a_cancelled_switch_pushes_no_version() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path(), "colmap");
    let job = state.build_sift_files_job(id).expect("it can build");
    let cancel = std::sync::atomic::AtomicBool::new(true);
    let progress = sfmtool_core::progress::Progress::none().cancelled_by(&cancel);
    assert!(matches!(
        job(&progress),
        crate::background::Finished::Cancelled
    ));
    let contents = &state.node(id).unwrap().recon().metadata.workspace.contents;
    assert_eq!(contents.feature_tool, "colmap");
    // Started for real, a build of a busy node is refused.
    state.start_build_sift_files(id).expect("it starts");
    let busy = state.build_sift_files_refusal(id).expect("busy");
    assert!(busy.contains("is busy"), "{busy}");
    state.cancel_background_tasks();
    state.finish_background_tasks();
}

/// A SIFT index open beside the node is judged again when a build rewrites a
/// `.sift` file under it, without being opened again: here a photograph
/// replaced by another makes its `.sift` file new, so the index goes stale.
#[test]
fn an_open_index_is_judged_again_after_a_build() {
    use crate::index_files::IndexFileState;
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path(), "sfmtool");
    state.start_build_sift_files(id).expect("it starts");
    state.finish_background_tasks();
    state.start_build_index_files(id).expect("it can index");
    state.finish_background_tasks();
    assert_eq!(state.sift_index_state(id), IndexFileState::Current);

    let image = state.node(id).unwrap().recon().image_set.images[0]
        .name
        .clone();
    let other = Path::new(SEOUL_BULL).join("seoul_bull_sculpture_17.jpg");
    // Written rather than copied, since a copy keeps the source's modification
    // time, and after a pause, so the new time is past the `.sift` file's.
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(dir.path().join(image), std::fs::read(other).unwrap()).unwrap();
    state.start_build_sift_files(id).expect("it starts");
    state.finish_background_tasks();
    let entry = state.action_log.entries().next_back().expect("an entry");
    assert!(
        entry.text.contains("1 written and 7 already up to date"),
        "{}",
        entry.text
    );
    assert_eq!(state.sift_index_state(id), IndexFileState::Stale);
}

/// Undoing the switch moves where the node's `.sift` files are, so an open
/// index built over the files the switch pointed at is judged again on the
/// next look: stale after the undo, where no file is, and current after the
/// redo.
#[test]
fn undoing_the_switch_judges_an_open_index_again() {
    use crate::index_files::IndexFileState;
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path(), "colmap");
    state.start_build_sift_files(id).expect("it starts");
    state.finish_background_tasks();
    state.start_build_index_files(id).expect("it can index");
    state.finish_background_tasks();
    assert_eq!(state.sift_index_state(id), IndexFileState::Current);

    state.undo(id).expect("the switch is undone");
    state.refresh_index_files(id);
    assert_eq!(state.sift_index_state(id), IndexFileState::Stale);
    assert!(state.sift_index_search_refusal(id).is_some());

    state.redo(id).expect("the switch is redone");
    state.refresh_index_files(id);
    assert_eq!(state.sift_index_state(id), IndexFileState::Current);
}
