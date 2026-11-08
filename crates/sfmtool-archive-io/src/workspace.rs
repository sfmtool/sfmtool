// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! How a reader finds the workspace a file's `workspace` metadata records.
//!
//! `.sfmr`, `.matches` and `.kdf` record the same two paths and resolve them by
//! the same rule, so the rule is here once rather than in each format crate.

use std::path::{Path, PathBuf};

/// The file a workspace directory holds, and what makes a directory one.
pub const WORKSPACE_MARKER: &str = ".sfm-workspace.json";

/// The workspace directory that the file at `file` records, or `None` when
/// none of the three lookups finds one.
///
/// A candidate is the workspace only when it holds [`WORKSPACE_MARKER`]; a
/// directory that exists without one is skipped, and the next lookup is tried.
/// In order:
///
/// 1. `relative_path` joined to `file`'s directory, both resolved to their
///    real locations;
/// 2. `absolute_path` as written;
/// 3. the nearest directory at or above `file`'s directory that holds the
///    marker.
///
/// An empty `relative_path` or `absolute_path` means none was recorded, and its
/// lookup is skipped. The paths are separate arguments rather than one metadata
/// struct because `.kdf` records them in a type of its own.
///
/// # Example
///
/// ```no_run
/// use sfmtool_archive_io::resolve_workspace_dir;
/// let file = std::path::Path::new("/projects/ws/sfmr/recon.sfmr");
/// let workspace = resolve_workspace_dir(file, "..", "/projects/ws");
/// ```
pub fn resolve_workspace_dir(
    file: &Path,
    relative_path: &str,
    absolute_path: &str,
) -> Option<PathBuf> {
    let parent = file.parent().unwrap_or_else(|| Path::new("."));
    let file_dir = dunce::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf());

    if !relative_path.is_empty() {
        if let Ok(candidate) = dunce::canonicalize(file_dir.join(relative_path)) {
            if candidate.join(WORKSPACE_MARKER).exists() {
                return Some(candidate);
            }
        }
    }

    if !absolute_path.is_empty() {
        let candidate = PathBuf::from(absolute_path);
        if candidate.join(WORKSPACE_MARKER).exists() {
            return Some(candidate);
        }
    }

    file_dir
        .ancestors()
        .find(|dir| dir.join(WORKSPACE_MARKER).exists())
        .map(Path::to_path_buf)
}
