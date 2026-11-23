// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! *File > Load Demo Data* and `--demo`: the demo scene written as a
//! workspace in a temporary directory of its own and opened from there.
//!
//! See `specs/gui/background-tasks.md` section "Loading demo data". Each
//! load is one background task, [`Operation::LOAD_DEMO_DATA`]: it renders
//! the scene
//! ([`DemoScene::render`]) at the options the dialog's number of points maps
//! to ([`demo_scene_options`]), writes it with [`DemoScene::write_workspace`]
//! into a directory [`DemoWorkspaces::claim`] gave it, and opens the written
//! `demo.sfmr` as File > Open does. The node has a path like any opened file's.
//!
//! The directories live in the system temporary directory, one per load,
//! named `sfmtool-demo-<process id>-<n>`, and each has a lock file beside it,
//! `sfmtool-demo-<process id>-<n>.lock`, that the viewer holds locked while it
//! runs. At exit the viewer deletes its directories and then its lock files
//! ([`DemoWorkspaces`]'s `Drop`); at start it deletes every directory whose
//! owner has ended ([`sweep`]), which is how a crashed viewer's directories go.

use std::fs::{File, OpenOptions, TryLockError};
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use sfmtool_core::camera::PhotographCache;
use sfmtool_core::reconstruction::{DemoScene, DemoSceneError, DemoSceneOptions};

use crate::action_log::Kind;
use crate::background::{Finished, Job, Operation};
use crate::scene::ReconId;
use crate::state::open::{load_for_display, OpenedFile, Stop};
use crate::state::AppState;

#[cfg(test)]
mod tests;

/// The numbers of points the dialog's *Number of points* box takes.
///
/// Below about 250 the face spacing passes twice the patch half-extent and
/// the measurement's cube corners are no longer inside a patch, and 10,000
/// points keep the patch bitmap render to about a second.
pub(crate) const POINT_COUNT_RANGE: RangeInclusive<u32> = 300..=10_000;

/// The number the box starts at: the default scene's point count.
pub(crate) const DEFAULT_POINT_COUNT: u32 = DemoSceneOptions::DEFAULT_POINT_COUNT;

/// The prefix of every directory and lock file a load writes.
const PREFIX: &str = "sfmtool-demo-";

/// The extension of a lock file.
const LOCK_EXTENSION: &str = "lock";

/// How many names [`DemoWorkspaces::claim`] tries before it gives up.
const CLAIM_ATTEMPTS: u32 = 1000;

/// The scene options a load of `point_count` points renders: the default
/// scene with its point spacings scaled by
/// [`DemoSceneOptions::for_point_count`], after `point_count` is held to
/// [`POINT_COUNT_RANGE`], as the box holds what is typed into it.
pub(crate) fn demo_scene_options(point_count: u32) -> DemoSceneOptions {
    let held = point_count.clamp(*POINT_COUNT_RANGE.start(), *POINT_COUNT_RANGE.end());
    // `for_point_count` refuses only 0, which the clamp has ruled out.
    DemoSceneOptions::for_point_count(held).unwrap_or_default()
}

/// The temporary workspaces this viewer's loads wrote, and the lock on each.
///
/// The directory a load writes is claimed on the GUI thread before the task
/// starts, so the task is handed a name no other load and no other viewer
/// uses. Dropping the value deletes every directory it claimed and then each
/// one's lock file, which is the deletion at exit: [`AppState`] holds it, and
/// the viewer drops its state when the window closes.
#[derive(Default)]
pub(crate) struct DemoWorkspaces {
    /// The directory the workspaces are written in: the system temporary
    /// directory in the viewer, and a test's own temporary directory in a
    /// test. `None` until the viewer sets it, so a test that never sets it
    /// writes nothing anywhere.
    root: Option<PathBuf>,
    /// The `n` the next claim tries first.
    next: u64,
    /// Every workspace claimed, in the order claimed.
    held: Vec<Held>,
}

/// One claimed workspace and the open, locked handle of its lock file.
struct Held {
    dir: PathBuf,
    lock_path: PathBuf,
    /// Held for its lock, which the operating system releases when the handle
    /// closes or the process ends.
    lock: File,
}

impl DemoWorkspaces {
    /// Workspaces written in `root`.
    pub(crate) fn in_dir(root: PathBuf) -> Self {
        Self {
            root: Some(root),
            next: 0,
            held: Vec::new(),
        }
    }

    /// Claim a directory for one load: create and lock its lock file, and
    /// answer with the directory's path, which does not exist yet.
    ///
    /// The lock file is created before the directory and locked with
    /// [`File::try_lock`]. A name whose lock is held elsewhere, by a sweep in
    /// another viewer that took it in the moment between this viewer creating
    /// the file and locking it, is skipped for the next `n`, and so is a name
    /// whose lock file a sweep deleted before the lock was taken, since a lock
    /// on a deleted file guards nothing. A directory left behind under a name
    /// that is now this viewer's, by an ended viewer that had the same process
    /// id, is deleted.
    pub(crate) fn claim(&mut self) -> Result<PathBuf, String> {
        let Some(root) = self.root.clone() else {
            return Err(
                "Failed to load demo data: there is no temporary directory to write it to"
                    .to_string(),
            );
        };
        let pid = std::process::id();
        for _ in 0..CLAIM_ATTEMPTS {
            let n = self.next;
            self.next += 1;
            let name = format!("{PREFIX}{pid}-{n}");
            let dir = root.join(&name);
            let lock_path = root.join(format!("{name}.{LOCK_EXTENSION}"));
            let lock = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&lock_path)
                .map_err(|e| {
                    format!(
                        "Failed to load demo data: could not create {}: {e}",
                        lock_path.display()
                    )
                })?;
            match lock.try_lock() {
                Ok(()) => {}
                Err(TryLockError::WouldBlock) => continue,
                Err(TryLockError::Error(e)) => {
                    return Err(format!(
                        "Failed to load demo data: could not lock {}: {e}",
                        lock_path.display()
                    ))
                }
            }
            if !lock_path.is_file() {
                continue;
            }
            if dir.exists() && std::fs::remove_dir_all(&dir).is_err() {
                continue;
            }
            self.held.push(Held {
                dir: dir.clone(),
                lock_path,
                lock,
            });
            return Ok(dir);
        }
        Err(format!(
            "Failed to load demo data: no free name for its directory in {} after {CLAIM_ATTEMPTS} tries",
            root.display()
        ))
    }
}

impl Drop for DemoWorkspaces {
    /// Delete every claimed directory, then each lock file, the lock still
    /// held while its file is deleted. A directory that cannot be deleted,
    /// such as one a running task is still writing, is left for the next
    /// viewer's sweep, which finds it with no lock file beside it.
    fn drop(&mut self) {
        for held in &self.held {
            if let Err(e) = std::fs::remove_dir_all(&held.dir) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    log::warn!(
                        "Could not delete the demo workspace {}: {e}",
                        held.dir.display()
                    );
                }
            }
        }
        for held in self.held.drain(..) {
            if let Err(e) = std::fs::remove_file(&held.lock_path) {
                log::warn!(
                    "Could not delete the demo lock file {}: {e}",
                    held.lock_path.display()
                );
            }
            drop(held.lock);
        }
    }
}

/// What a [`sweep`] deleted.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Swept {
    /// The directories deleted, with their lock files where they had one.
    pub(crate) dirs: usize,
    /// The lock files deleted that had no directory beside them.
    pub(crate) lone_lock_files: usize,
}

/// Delete the demo workspaces in `root` whose owner has ended, as the viewer
/// does at start with `root` the system temporary directory and `started` the
/// time the sweep began.
///
/// Only names of the form `sfmtool-demo-<process id>-<n>` and
/// `sfmtool-demo-<process id>-<n>.lock` are looked at. For each lock file
/// whose lock the sweep can take, the owner has ended, since the operating
/// system releases a lock when its process ends: the sweep deletes the
/// directory of the same name and then the lock file, while it holds the
/// lock. A lock file with no directory beside it may be one a viewer has just
/// created and not yet locked, so it is deleted only when it was last
/// modified before `started`, and is otherwise left to a later sweep. A
/// directory with no lock file beside it has no owner, since an owner creates
/// its lock file first and deletes it last, and is deleted. A lock held by a
/// running viewer keeps both.
///
/// Nothing here is an error: a file that cannot be opened, locked or deleted
/// is left where it is for a later sweep.
pub(crate) fn sweep(root: &Path, started: SystemTime) -> Swept {
    let mut swept = Swept::default();
    let Ok(entries) = std::fs::read_dir(root) else {
        return swept;
    };
    let mut lock_files = Vec::new();
    let mut dirs = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        match name.strip_suffix(&format!(".{LOCK_EXTENSION}")) {
            Some(stem) if is_workspace_name(stem) => lock_files.push(stem.to_string()),
            None if is_workspace_name(name) => dirs.push(name.to_string()),
            _ => {}
        }
    }
    for stem in &lock_files {
        let lock_path = root.join(format!("{stem}.{LOCK_EXTENSION}"));
        let dir = root.join(stem);
        let Ok(lock) = OpenOptions::new().read(true).write(true).open(&lock_path) else {
            continue;
        };
        if lock.try_lock().is_err() {
            continue;
        }
        if dir.is_dir() {
            if std::fs::remove_dir_all(&dir).is_ok() && std::fs::remove_file(&lock_path).is_ok() {
                swept.dirs += 1;
            }
        } else if !dir.exists() {
            let old = lock
                .metadata()
                .and_then(|meta| meta.modified())
                .is_ok_and(|modified| modified < started);
            if old && std::fs::remove_file(&lock_path).is_ok() {
                swept.lone_lock_files += 1;
            }
        }
    }
    for stem in &dirs {
        let dir = root.join(stem);
        let lock_path = root.join(format!("{stem}.{LOCK_EXTENSION}"));
        if dir.is_dir() && !lock_path.exists() && std::fs::remove_dir_all(&dir).is_ok() {
            swept.dirs += 1;
        }
    }
    swept
}

/// Whether `name` is `sfmtool-demo-<digits>-<digits>`.
fn is_workspace_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix(PREFIX) else {
        return false;
    };
    let Some((pid, n)) = rest.split_once('-') else {
        return false;
    };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    digits(pid) && digits(n)
}

impl AppState {
    /// Start loading the demo scene at `point_count` points as a background
    /// task, and answer with its id.
    ///
    /// The directory is claimed here, on the GUI thread, so a refusal is known
    /// before the task starts. **Nothing is logged here**, as for
    /// [`AppState::start_open`]: the task's own row is written when it lands.
    pub(crate) fn start_load_demo(&mut self, point_count: u32) -> Result<u64, String> {
        let options = demo_scene_options(point_count);
        let dir = self.demo_workspaces.claim()?;
        self.start_task(
            Operation::LOAD_DEMO_DATA,
            None,
            "demo".to_string(),
            load_demo_job(options, point_count, dir, Arc::clone(&self.photographs)),
        )
    }

    /// Load the demo scene for the dialog's Load button and `--demo`, writing
    /// a failed row when the load cannot start.
    pub(crate) fn load_demo(&mut self, point_count: u32) {
        if let Err(message) = self.start_load_demo(point_count) {
            self.action_log.fail(Kind::File, message);
        }
    }
}

/// The work of one load: render the scene, write it into `dir`, and open the
/// written `demo.sfmr` as File > Open does.
///
/// The bar splits 1 : 2 : 1 between the render (`render` and `encode`), the
/// write (`points`, `patch bitmaps` and `write`) and the open (the phase
/// `open` with the read's own stages). The written file carries its
/// thumbnails and patch bitmaps, so the open fills in neither.
pub(crate) fn load_demo_job(
    options: DemoSceneOptions,
    asked: u32,
    dir: PathBuf,
    photographs: Arc<PhotographCache>,
) -> Job {
    Box::new(move |progress| {
        let [render, write, open] = progress.split([1.0, 2.0, 1.0]);
        let written = DemoScene::render(&options, &render)
            .and_then(|scene| scene.write_workspace(&dir, &write));
        let path = match written {
            Ok(path) => path,
            Err(DemoSceneError::Cancelled) => return Finished::Cancelled,
            Err(e) => return Finished::Failed(format!("Failed to load demo data: {e}")),
        };
        let phase = open.phase("open");
        let outcome = match load_for_display(&path, &photographs, &phase) {
            Ok(loaded) => Ok(loaded),
            Err(Stop::Cancelled) => return Finished::Cancelled,
            Err(Stop::Failed(message)) => Err(message),
        };
        progress.set_fraction(1.0);
        Finished::DemoLoaded {
            file: OpenedFile { path, outcome },
            asked,
        }
    })
}

/// The Action Log sentence of a load that made `node` for `asked` points: the
/// number of points the scene kept, which is near the number asked for and not
/// equal to it, and then the file it opened. The counts come first and the
/// path last, as other rows that name a file put it, since the temporary
/// workspace's path is long enough to push anything after it out of view.
pub(crate) fn loaded_sentence(state: &AppState, node: ReconId, asked: u32) -> String {
    let Some(node) = state.node(node) else {
        return "Loaded demo data".to_string();
    };
    let path = node
        .path
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    format!(
        "Loaded demo data: {} points, for {asked} asked for, from {path}",
        node.recon().point_count()
    )
}

/// Point a viewer's state at the system temporary directory, after deleting
/// what ended viewers left there. Called once at start.
pub(crate) fn start_in_temp_dir(state: &mut AppState) {
    let root = std::env::temp_dir();
    let swept = sweep(&root, SystemTime::now());
    if swept != Swept::default() {
        log::info!(
            "Deleted {} demo workspaces and {} lone lock files that ended viewers left in {}",
            swept.dirs,
            swept.lone_lock_files,
            root.display()
        );
    }
    state.demo_workspaces = DemoWorkspaces::in_dir(root);
}
