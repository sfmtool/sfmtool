// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! How a reader finds the workspace a file's `workspace` metadata records.
//!
//! `.sfmr`, `.matches` and `.kdf` record the same two paths and resolve them by
//! the same rule, so the rule is here once rather than in each format crate.
//! The rule's last step, the search upward from a path
//! ([`find_workspace_for_path`]), is also how a command finds the workspace of
//! an input path that records none.

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
///    marker ([`find_workspace_for_path`]).
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

    find_workspace_for_path(file)
}

/// The nearest directory at or above `path` that holds [`WORKSPACE_MARKER`],
/// or `None` when no directory up to the filesystem root holds one.
///
/// `path` may be a directory, a file, or a path that does not exist yet (an
/// output about to be written, say): the search starts at `path` itself, and
/// since a file or a missing path holds no marker, it finds the same directory
/// a search from `path`'s directory does. A relative `path` is taken against
/// the current directory, so the search goes on above it.
///
/// The search walks up `path` as given, made absolute but with no link
/// resolved and no `..` collapsed: a directory reached through a link belongs
/// to the workspace around the link, not to one around its target. That keeps a workspace
/// whose image folder is a link to storage outside it a workspace for those
/// images, and keeps a path measured against the result in the form the
/// caller gave. The result is spelled the same way, as an ancestor of the
/// absolute `path`.
///
/// # Example
///
/// ```no_run
/// use sfmtool_archive_io::find_workspace_for_path;
/// let image = std::path::Path::new("/projects/ws/images/a.jpg");
/// let workspace = find_workspace_for_path(image);
/// ```
pub fn find_workspace_for_path(path: &Path) -> Option<PathBuf> {
    find_workspace_from(path, || std::env::current_dir().ok())
}

/// [`find_workspace_for_path`] with the directory a relative `path` is taken
/// against given, so a test can name one without changing the process's
/// current directory. `current_dir` is called only for a relative `path`, so an
/// absolute one is searched even where the current directory cannot be read.
fn find_workspace_from(
    path: &Path,
    current_dir: impl FnOnce() -> Option<PathBuf>,
) -> Option<PathBuf> {
    // A relative `path` goes under the current directory, and the empty path,
    // which is a bare file name's parent, names the current directory itself.
    // A Windows path with a drive but no root (`D:x`) is still relative after
    // the join, and only `std::path::absolute` knows that drive's directory.
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else if path.as_os_str().is_empty() {
        current_dir()?
    } else {
        current_dir()?.join(path)
    };
    let absolute = if joined.is_absolute() {
        joined
    } else {
        std::path::absolute(joined).ok()?
    };
    // Collecting the components drops `.` and repeated separators, as a Python
    // `Path` does, and keeps `..` as written: the walk goes up the path's text
    // (the parent of `ws/colmap/..` is `ws/colmap`), so the directory returned
    // is spelled as the caller gave it and a path measured against it with
    // `relative_to` keeps its form. `std::path::absolute` is not used on an
    // absolute path because on Windows it resolves `..` as text.
    let absolute: PathBuf = absolute.components().collect();
    absolute
        .ancestors()
        .find(|dir| dir.join(WORKSPACE_MARKER).exists())
        .map(Path::to_path_buf)
}

#[cfg(test)]
mod tests;
