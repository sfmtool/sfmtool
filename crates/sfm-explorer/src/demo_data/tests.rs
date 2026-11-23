// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The demo workspaces' lock files, the sweep at start and the deletion at
//! exit, each in a temporary directory of the test's own, never the system
//! one; and the dialog's number of points, mapped to the scene and loaded.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use sfmtool_core::reconstruction::DemoSceneOptions;

use super::*;
use crate::state::AppState;

/// A directory `name` in `root` with one file in it, as a load leaves it.
fn workspace(root: &Path, name: &str) -> PathBuf {
    let dir = root.join(name);
    std::fs::create_dir_all(dir.join("images")).unwrap();
    std::fs::write(dir.join("images").join("demo_0.jpg"), b"jpeg").unwrap();
    dir
}

/// The lock file of `name` in `root`, created and not locked.
fn lock_file(root: &Path, name: &str) -> PathBuf {
    let path = root.join(format!("{name}.lock"));
    File::create(&path).unwrap();
    path
}

/// A handle on `path` holding its lock, as a running viewer holds its own.
fn held(path: &Path) -> File {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    file.try_lock().expect("nothing else holds the lock");
    file
}

/// Whether a fresh handle could take `path`'s lock now.
fn lockable(path: &Path) -> bool {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    file.try_lock().is_ok()
}

/// The names in `root`, sorted.
fn names(root: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// A time well after every file the test made, as a sweep that started later.
fn later() -> SystemTime {
    SystemTime::now() + Duration::from_secs(60)
}

// -- The sweep at start ---------------------------------------------------

/// A directory whose lock file no process holds belongs to a viewer that has
/// ended, so the sweep deletes it and then its lock file.
#[test]
fn the_sweep_deletes_a_directory_whose_lock_no_process_holds() {
    let root = tempfile::tempdir().unwrap();
    workspace(root.path(), "sfmtool-demo-41-0");
    lock_file(root.path(), "sfmtool-demo-41-0");

    let swept = sweep(root.path(), later());

    assert_eq!(names(root.path()), Vec::<String>::new());
    assert_eq!(
        swept,
        Swept {
            dirs: 1,
            lone_lock_files: 0
        }
    );
}

/// A directory whose lock a live process holds, this test's, is kept with its
/// lock file, and the lock is still held afterwards.
#[test]
fn the_sweep_keeps_a_directory_whose_lock_a_live_process_holds() {
    let root = tempfile::tempdir().unwrap();
    workspace(root.path(), "sfmtool-demo-41-0");
    let lock_path = lock_file(root.path(), "sfmtool-demo-41-0");
    let lock = held(&lock_path);

    let swept = sweep(root.path(), later());

    assert_eq!(
        names(root.path()),
        ["sfmtool-demo-41-0", "sfmtool-demo-41-0.lock"]
    );
    assert!(root
        .path()
        .join("sfmtool-demo-41-0/images/demo_0.jpg")
        .is_file());
    assert_eq!(swept, Swept::default());
    assert!(
        !lockable(&lock_path),
        "the sweep let go of a lock it never had"
    );
    drop(lock);
}

/// A directory with no lock file beside it has no owner, since an owner
/// creates its lock file before the directory and deletes it after, so the
/// sweep deletes it.
#[test]
fn the_sweep_deletes_a_directory_with_no_lock_file() {
    let root = tempfile::tempdir().unwrap();
    workspace(root.path(), "sfmtool-demo-41-3");

    let swept = sweep(root.path(), later());

    assert_eq!(names(root.path()), Vec::<String>::new());
    assert_eq!(swept.dirs, 1);
}

/// A lock file with no directory may be one a viewer has created and not yet
/// locked: the sweep leaves it when it was modified after the sweep started,
/// and deletes it when it was modified before.
#[test]
fn the_sweep_deletes_a_lone_lock_file_only_once_it_is_older_than_the_sweep() {
    let root = tempfile::tempdir().unwrap();
    let path = lock_file(root.path(), "sfmtool-demo-41-0");
    let modified = SystemTime::now() - Duration::from_secs(3600);
    File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(modified)
        .unwrap();

    // A sweep that started before the file was last modified.
    let swept = sweep(root.path(), modified - Duration::from_secs(1));
    assert_eq!(names(root.path()), ["sfmtool-demo-41-0.lock"]);
    assert_eq!(swept, Swept::default());

    // A later one.
    let swept = sweep(root.path(), modified + Duration::from_secs(1));
    assert_eq!(names(root.path()), Vec::<String>::new());
    assert_eq!(swept.lone_lock_files, 1);
}

/// A lone lock file whose lock is held is kept however old it is: its viewer
/// has claimed the name and not yet written the directory.
#[test]
fn the_sweep_keeps_a_lone_lock_file_whose_lock_is_held() {
    let root = tempfile::tempdir().unwrap();
    let path = lock_file(root.path(), "sfmtool-demo-41-0");
    let lock = held(&path);

    let swept = sweep(root.path(), later());

    assert_eq!(names(root.path()), ["sfmtool-demo-41-0.lock"]);
    assert_eq!(swept, Swept::default());
    drop(lock);
}

/// Only `sfmtool-demo-<process id>-<n>` and its `.lock` are the sweep's: a
/// name that only starts the same way is not touched.
#[test]
fn the_sweep_touches_only_names_of_its_own_form() {
    let root = tempfile::tempdir().unwrap();
    for name in [
        "sfmtool-demo-notes",
        "sfmtool-demo-41",
        "sfmtool-demo-41-x",
        "sfmtool-demo--0",
        "sfmtool-demo-41-0-copy",
    ] {
        workspace(root.path(), name);
    }
    std::fs::write(root.path().join("sfmtool-demo-41-0.txt"), b"").unwrap();
    let before = names(root.path());

    let swept = sweep(root.path(), later());

    assert_eq!(names(root.path()), before);
    assert_eq!(swept, Swept::default());
}

// -- Claiming a directory -------------------------------------------------

/// The name this process's load `n` takes.
fn own(n: u64) -> String {
    format!("sfmtool-demo-{}-{n}", std::process::id())
}

/// A claim creates and locks the lock file and names the directory beside
/// it, which the load then writes; the next claim takes the next `n`.
#[test]
fn a_claim_locks_its_lock_file_and_the_next_claim_takes_the_next_name() {
    let root = tempfile::tempdir().unwrap();
    let mut workspaces = DemoWorkspaces::in_dir(root.path().to_path_buf());

    let first = workspaces.claim().unwrap();
    let second = workspaces.claim().unwrap();

    assert_eq!(first, root.path().join(own(0)));
    assert_eq!(second, root.path().join(own(1)));
    assert!(
        !first.exists(),
        "the load writes the directory, not the claim"
    );
    for n in [0, 1] {
        let lock_path = root.path().join(format!("{}.lock", own(n)));
        assert!(
            !lockable(&lock_path),
            "{} is not locked",
            lock_path.display()
        );
    }
}

/// A claim whose `try_lock` fails, since something else holds that name's
/// lock, moves on to the next `n`, and leaves the lock it could not take.
#[test]
fn a_claim_that_cannot_take_its_lock_moves_on_to_the_next_name() {
    let root = tempfile::tempdir().unwrap();
    let taken = lock_file(root.path(), &own(0));
    let lock = held(&taken);
    let mut workspaces = DemoWorkspaces::in_dir(root.path().to_path_buf());

    let dir = workspaces.claim().unwrap();

    assert_eq!(dir, root.path().join(own(1)));
    assert!(taken.is_file());
    drop(lock);
    assert!(
        lockable(&taken),
        "the claim kept a handle on the name it skipped"
    );
}

/// A directory left under a name that is now this viewer's, by an ended viewer
/// that had the same process id, is deleted by the claim that takes the name,
/// so the load writes into an empty directory.
#[test]
fn a_claim_deletes_what_an_ended_viewer_left_under_its_name() {
    let root = tempfile::tempdir().unwrap();
    workspace(root.path(), &own(0));
    lock_file(root.path(), &own(0));
    let mut workspaces = DemoWorkspaces::in_dir(root.path().to_path_buf());

    let dir = workspaces.claim().unwrap();

    assert_eq!(dir, root.path().join(own(0)));
    assert!(!dir.exists());
}

/// With no root set, as in a test that never sets one, a claim is refused and
/// writes nothing anywhere.
#[test]
fn a_claim_with_no_root_is_refused() {
    let mut workspaces = DemoWorkspaces::default();
    let refusal = workspaces.claim().unwrap_err();
    assert!(refusal.contains("no temporary directory"), "{refusal}");
}

// -- Deletion at exit -----------------------------------------------------

/// Dropping the workspaces, which the viewer does at exit, deletes every
/// directory claimed and then every lock file, and leaves what is not its own.
#[test]
fn exit_deletes_each_directory_and_then_its_lock_file() {
    let root = tempfile::tempdir().unwrap();
    let other = workspace(root.path(), "sfmtool-demo-41-0");
    let mut workspaces = DemoWorkspaces::in_dir(root.path().to_path_buf());
    for _ in 0..2 {
        let dir = workspaces.claim().unwrap();
        workspace(root.path(), &dir.file_name().unwrap().to_string_lossy());
    }
    assert_eq!(names(root.path()).len(), 5);

    drop(workspaces);

    assert_eq!(names(root.path()), ["sfmtool-demo-41-0"]);
    assert!(other.is_dir());
}

// -- The dialog's number of points ----------------------------------------

/// A number in the box's range is the scene `for_point_count` gives for it,
/// and a number past either end is held to that end, as the box holds it.
#[test]
fn the_number_of_points_maps_through_for_point_count_within_the_range() {
    for n in [300, DemoSceneOptions::DEFAULT_POINT_COUNT, 2000, 10_000] {
        assert_eq!(
            demo_scene_options(n),
            DemoSceneOptions::for_point_count(n).unwrap(),
            "{n}"
        );
    }
    assert_eq!(DEFAULT_POINT_COUNT, DemoSceneOptions::DEFAULT_POINT_COUNT);
    assert_eq!(
        demo_scene_options(1),
        DemoSceneOptions::for_point_count(300).unwrap()
    );
    assert_eq!(
        demo_scene_options(100_000),
        DemoSceneOptions::for_point_count(10_000).unwrap()
    );
}

/// A load writes the scene into its claimed directory and opens it: the node
/// is labelled `demo`, has the written `demo.sfmr` as its path, holds about the
/// number of points asked for, and its row names the number kept.
#[test]
fn a_load_opens_the_written_scene_from_its_temporary_workspace() {
    let root = tempfile::tempdir().unwrap();
    let mut state = AppState::new();
    state.demo_workspaces = DemoWorkspaces::in_dir(root.path().to_path_buf());

    let task = state.start_load_demo(300).unwrap();
    state.finish_background_task(task);

    let finished = state.finished_task(task).unwrap();
    assert!(finished.outcome.is_ok(), "{:?}", finished.outcome);
    let id = finished.opened.expect("the load made a node");
    let node = state.node(id).unwrap();
    let dir = root.path().join(own(0));
    assert_eq!(node.label, "demo");
    assert_eq!(node.path.as_deref(), Some(dir.join("demo.sfmr").as_path()));
    assert!(dir.join("images").join("demo_4.jpg").is_file());
    assert!(dir.join(".sfm-workspace.json").is_file());
    let kept = node.recon().point_count();
    assert!((285..=315).contains(&kept), "{kept} points for 300");
    assert_eq!(state.selected_recon, Some(id));
    let entry = state.action_log.entries().last().unwrap();
    assert_eq!(
        entry.text,
        format!(
            "Loaded demo data from {}: {kept} points, for 300 asked for",
            dir.join("demo.sfmr").display()
        )
    );

    // The state's drop is the viewer's exit: the workspace goes with it.
    drop(state);
    assert_eq!(names(root.path()), Vec::<String>::new());
}
