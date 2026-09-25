// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The index-files build as one operation: what it builds when, what it
//! reports, and what a stop or a failure in its second half leaves.
//!
//! Over the SIFT index's workspace fixture ([`crate::sift_index::tests`]).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use sfmtool_core::progress::{Event, Progress};

use super::{IndexFileState, BUILD_INDEX_FILES, REBUILD_INDEX_FILES};
use crate::background::{Finished, IndexFilesEnd};
use crate::sift_index::tests::{index_of, searchable, state_in, with_sift_files};

/// A node with `.sift` files and photographs and neither index file yet.
fn unbuilt(dir: &std::path::Path) -> (crate::state::AppState, crate::scene::ReconId) {
    let (state, id) = state_in(dir);
    with_sift_files(&state, id, [900.0, 500.0]);
    (state, id)
}

/// The entry reads *Build* on a node with neither file and *Rebuild* once
/// either is there.
#[test]
fn the_build_entry_reads_build_then_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = unbuilt(dir.path());
    state.refresh_index_files(id);
    assert_eq!(state.index_files_build_label(id), BUILD_INDEX_FILES);
    state.start_build_index_files(id).expect("it starts");
    state.finish_background_task();
    assert_eq!(state.index_files_build_label(id), REBUILD_INDEX_FILES);
}

/// One build makes both files, opens both, and both read current.
#[test]
fn one_build_makes_both_files_and_both_read_current() {
    let dir = tempfile::tempdir().unwrap();
    let (state, id, _) = searchable(dir.path());
    assert_eq!(state.sift_index_state(id), IndexFileState::Current);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::Current);
    assert!(index_of(dir.path()).is_file());
    assert!(dir.path().join("demo-cluster-patches.matches").is_file());
    assert!(
        state
            .action_log
            .entries()
            .any(|entry| entry.text.starts_with("Built the index files of demo")),
        "the build writes one row naming both files"
    );
}

/// With a current index and no cluster patches, the build keeps the index and
/// makes only the cluster patches, from the index that is open.
#[test]
fn a_build_with_a_current_index_keeps_it_and_makes_the_cluster_patches() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = searchable(dir.path());
    let patches = state.cluster_patches(id).expect("built").path.clone();
    let before = std::fs::read(index_of(dir.path())).unwrap();
    std::fs::remove_file(&patches).unwrap();
    state.close_index_files(id).expect("both are open");
    state
        .open_index_files(id, None, None)
        .expect("the index is there");
    assert_eq!(state.cluster_patches_state(id), IndexFileState::None);

    let job = state.build_index_files_job(id).expect("it can build");
    let finished = job(&Progress::none());
    let Finished::IndexFiles {
        index,
        cluster_patches,
        end: IndexFilesEnd::Built(text),
    } = finished
    else {
        panic!("the build did not finish");
    };
    assert!(index.is_none(), "a current index was built again");
    assert_eq!(cluster_patches.as_deref(), Some(patches.as_path()));
    assert!(text.contains("already current"), "{text}");
    assert_eq!(std::fs::read(index_of(dir.path())).unwrap(), before);
    state.install_index_files(id, index, cluster_patches);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::Current);
}

/// With a stale index the build writes both files again, so the cluster
/// patches are made from the index it has just written.
#[test]
fn a_build_with_a_stale_index_rebuilds_both() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id, _) = searchable(dir.path());
    {
        let recon = state.node(id).expect("loaded").recon();
        crate::sift_index::tests::write_sift(
            &recon.sift_path_for_image(2),
            &recon.image_table.images[2].name,
            &vec![vec![5u8; 128]; 30],
            &(0..30)
                .map(|i| [100.0 + 10.0 * f64::from(i), 300.0])
                .collect::<Vec<_>>(),
        );
    }
    state
        .open_index_files(id, None, None)
        .expect("both files open again");
    assert_eq!(state.sift_index_state(id), IndexFileState::Stale);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::Stale);

    state.start_build_index_files(id).expect("it starts");
    state.finish_background_task();
    assert_eq!(state.sift_index_state(id), IndexFileState::Current);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::Current);
}

/// The bar moves through both halves: fractions inside the index's eighth and
/// inside the cluster patches' seven eighths, under the phases of each, and
/// the build ends at its end.
#[test]
fn the_build_reports_through_both_halves() {
    let dir = tempfile::tempdir().unwrap();
    let (state, id) = unbuilt(dir.path());
    let job = state.build_index_files_job(id).expect("it can build");
    let fractions = Mutex::new(Vec::new());
    let phases = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| match event {
        Event::Fraction { of_whole } => fractions.lock().unwrap().push(of_whole),
        Event::Enter { phase, .. } => phases.lock().unwrap().push(phase),
        _ => {}
    };
    let finished = job(&Progress::to(&sink));
    assert!(
        matches!(
            finished,
            Finished::IndexFiles {
                end: IndexFilesEnd::Built(_),
                ..
            }
        ),
        "the build did not finish"
    );
    let fractions = fractions.into_inner().unwrap();
    let phases = phases.into_inner().unwrap();
    assert!(
        fractions.iter().any(|f| *f > 0.0 && *f < 0.125),
        "the index reported nothing: {fractions:?}"
    );
    assert!(
        fractions.iter().any(|f| *f > 0.125 && *f < 1.0),
        "the cluster patches reported nothing: {fractions:?}"
    );
    assert_eq!(fractions.last().copied(), Some(1.0), "{fractions:?}");
    for phase in [
        "read descriptors",
        "build forest",
        "write index",
        "count features",
        "cluster features",
        "read photographs",
        "refine patches",
        "write cluster patches",
    ] {
        assert!(
            phases.contains(&phase),
            "{phase} was not reported: {phases:?}"
        );
    }
}

/// A cancel in the cluster half stops it, writes no cluster patches, and hands
/// back the index the first half wrote, so the node opens the file now on
/// disk.
#[test]
fn a_cancel_in_the_cluster_half_keeps_the_index_it_wrote() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = unbuilt(dir.path());
    let job = state.build_index_files_job(id).expect("it can build");
    let flag = AtomicBool::new(false);
    let sink = |event: Event<'_>| {
        if let Event::Enter {
            phase: "cluster features",
            ..
        } = event
        {
            flag.store(true, Ordering::Relaxed);
        }
    };
    let finished = job(&Progress::to(&sink).cancelled_by(&flag));
    let Finished::IndexFiles {
        index,
        cluster_patches,
        end: IndexFilesEnd::Cancelled,
    } = finished
    else {
        panic!("the build did not stop in its cluster half");
    };
    assert!(index.is_some(), "the index it wrote was not handed back");
    assert!(cluster_patches.is_none());
    assert!(index_of(dir.path()).is_file());
    assert!(!dir.path().join("demo-cluster-patches.matches").exists());
    state.install_index_files(id, index, cluster_patches);
    assert_eq!(state.sift_index_state(id), IndexFileState::Current);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::None);
}

/// A photograph that cannot be read fails the cluster half naming it, and the
/// index the build wrote is still handed back.
#[test]
fn a_missing_photograph_fails_the_cluster_half_naming_it() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = unbuilt(dir.path());
    let missing = {
        let recon = state.node(id).expect("loaded").recon();
        recon.workspace_dir.join(&recon.image_table.images[4].name)
    };
    std::fs::remove_file(&missing).unwrap();
    state.start_build_index_files(id).expect("it starts");
    state.finish_background_task();
    assert_eq!(state.sift_index_state(id), IndexFileState::Current);
    assert_eq!(state.cluster_patches_state(id), IndexFileState::None);
    let failed = state
        .action_log
        .entries()
        .find(|entry| entry.text.contains("could not build the cluster patches"))
        .expect("the failure is a row");
    assert!(
        failed.text.contains(&missing.display().to_string()),
        "{}",
        failed.text
    );
}

/// Opening names a file that has to open, and closing has to have something
/// to close.
#[test]
fn open_and_close_refuse_when_there_is_nothing_to_do() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, id) = state_in(dir.path());
    let why = state
        .open_index_files(id, None, None)
        .expect_err("neither file is there");
    assert!(why.contains("No index file of demo"), "{why}");
    let why = state
        .open_index_files(id, Some(dir.path().join("missing.kdf")), None)
        .expect_err("a named file has to open");
    assert!(why.starts_with("Cannot open"), "{why}");
    let why = state.close_index_files(id).expect_err("nothing is open");
    assert!(why.contains("No index file is open"), "{why}");
}
