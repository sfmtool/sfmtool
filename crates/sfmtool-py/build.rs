// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Records which commit of the source tree the extension is built from.
//!
//! The build sets the compile-time variable `SFMTOOL_BUILD_GIT_DESCRIBE` to
//! the output of `git describe --tags --long --always` run in the repository
//! root, two directories above this crate, with `-dirty` added when
//! `git status --porcelain --untracked-files=no` lists a change there, and
//! `build_git_describe` in `src/lib.rs` returns it. `git describe --dirty`
//! would say the same, but it rewrites the index when the index's file times
//! are out of date, and the index is one of the paths below whose change runs
//! this script again; `git status` with `GIT_OPTIONAL_LOCKS=0` leaves it. The variable is the empty string when there is
//! no `.git` there (a build from the sdist), when git is not installed, and
//! when git fails or takes longer than `GIT_TIMEOUT`, so none of those stops
//! the build. `specs/cli/other/version-command.md` says how `sfm version` uses it.
//!
//! The script asks Cargo to run it again when the commit or the tag it
//! describes changes (the `HEAD` file, the branch's ref, `packed-refs`, the tag
//! refs), when the index changes, and when a file the extension is compiled
//! from changes (each crate's `Cargo.toml`, `build.rs` and `src/`, and the
//! workspace's `Cargo.toml` and `Cargo.lock`), so that `-dirty` follows edits
//! to the Rust sources. An edit elsewhere in the tree, such as to a Python
//! file, does not run it again, so the recorded `-dirty` reflects such an edit
//! only once one of those has changed. A missing path is never listed, since
//! Cargo runs a script again on every build while a path it watches is
//! missing.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// How long one git command may run before it is killed and treated as failed.
const GIT_TIMEOUT: Duration = Duration::from_secs(10);

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let describe = repository_root()
        .and_then(|root| {
            watch_sources(&root);
            describe(&root)
        })
        .unwrap_or_default();
    println!("cargo:rustc-env=SFMTOOL_BUILD_GIT_DESCRIBE={describe}");
}

/// `git describe --tags --long --always` of the checkout at `root`, with
/// `-dirty` added when a tracked file differs from the commit, or `None` when
/// either git command fails.
fn describe(root: &Path) -> Option<String> {
    let describe = git(root, &["describe", "--tags", "--long", "--always"])?;
    let changes = git(root, &["status", "--porcelain", "--untracked-files=no"])?;
    if describe.is_empty() {
        None
    } else if changes.is_empty() {
        Some(describe)
    } else {
        Some(format!("{describe}-dirty"))
    }
}

/// The repository root this crate sits in (`<root>/crates/sfmtool-py`), when
/// it holds a `.git` directory or, in a linked worktree, a `.git` file.
fn repository_root() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR")?);
    let root = manifest_dir.parent()?.parent()?.to_path_buf();
    root.join(".git").exists().then_some(root)
}

/// Prints a `cargo:rerun-if-changed` line for each existing path whose change
/// can change the describe output.
fn watch_sources(root: &Path) {
    let mut paths = vec![root.join("Cargo.toml"), root.join("Cargo.lock")];
    if let Ok(entries) = std::fs::read_dir(root.join("crates")) {
        for entry in entries.flatten() {
            let crate_dir = entry.path();
            for name in ["Cargo.toml", "build.rs", "src"] {
                paths.push(crate_dir.join(name));
            }
        }
    }
    // `--path-format=absolute` gives both directories as absolute paths; the
    // common directory holds the refs, and differs from the git directory in a
    // linked worktree.
    let dirs = git(
        root,
        &[
            "rev-parse",
            "--path-format=absolute",
            "--git-dir",
            "--git-common-dir",
        ],
    );
    if let Some(dirs) = dirs {
        let mut lines = dirs.lines();
        if let (Some(git_dir), Some(common_dir)) = (lines.next(), lines.next()) {
            let git_dir = PathBuf::from(git_dir);
            let common_dir = PathBuf::from(common_dir);
            let head = git_dir.join("HEAD");
            if let Ok(text) = std::fs::read_to_string(&head) {
                if let Some(branch_ref) = text.trim().strip_prefix("ref: ") {
                    paths.push(common_dir.join(branch_ref));
                }
            }
            paths.push(head);
            paths.push(git_dir.join("index"));
            paths.push(common_dir.join("packed-refs"));
            paths.push(common_dir.join("refs").join("tags"));
        }
    }
    for path in paths.iter().filter(|path| path.exists()) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

/// The trimmed standard output of `git -C <root> <args>`, or `None` when git
/// cannot be started, exits with an error, or runs longer than `GIT_TIMEOUT`.
fn git(root: &Path, args: &[&str]) -> Option<String> {
    let mut child = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        // Read-only queries: `git status` then leaves the index as it is,
        // which would otherwise make Cargo run this script again.
        .env("GIT_OPTIONAL_LOCKS", "0")
        // `-C` picks the repository; a GIT_DIR or GIT_WORK_TREE inherited
        // from a git hook would override it.
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if start.elapsed() < GIT_TIMEOUT => {
                std::thread::sleep(Duration::from_millis(5));
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    if !status.success() {
        return None;
    }
    let mut output = String::new();
    child.stdout.take()?.read_to_string(&mut output).ok()?;
    Some(output.trim().to_string())
}
