// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The metadata a save stamps, and the minimal file.
//!
//! A save by a tool records who wrote the file and where the workspace is from
//! the file's own location ([`SfmrReconstruction::stamp_save`]), or from the
//! path the caller states ([`SaveStamp::workspace_path`]). A **minimal**
//! file is the smallest file that still holds the whole reconstruction: no
//! thumbnails, no patch bitmaps, no `lineage`, no recorded
//! `workspace.absolute_path`, and `tool_options` holding only the options of the
//! operation that wrote it. `sfm xform --minimal` and the viewer's
//! `File > Save As Minimal...` both write it through this module, so there is one
//! definition of what a minimal file leaves out
//! (`specs/cli/reconstruction/xform/xform-command.md` section "`--minimal`").
//!
//! The two halves are separate methods because `xform` applies them at
//! different times: its column half is a step at its own position in the chain,
//! which a later `--add-thumbnails` can undo, while its metadata half belongs to
//! the save. [`SfmrReconstruction::to_minimal`] is both at once, for a caller
//! that writes a minimal copy of a value in one go.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::SfmrReconstruction;

/// Who a save records as having written the file, and where it records the
/// workspace as being.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaveStamp<'a> {
    /// The operation that produced the file's content, such as `"xform"`.
    pub operation: &'a str,
    /// The tool that performed it, such as `"sfmtool"`.
    pub tool: &'a str,
    /// That tool's version.
    pub tool_version: &'a str,
    /// The `workspace.relative_path` to record, stated rather than measured.
    ///
    /// `None` measures it from the output's directory to the workspace, which
    /// is what an ordinary save wants. `Some(p)` records `p` as it stands, for
    /// a caller that knows where the file will sit relative to its workspace
    /// better than the directory it is being written from says: a ground-truth
    /// file written for a repository inside its own workspace states `"."`
    /// however it was produced. `Some("")` records an empty value, which the
    /// format reads as no path recorded.
    pub workspace_path: Option<&'a str>,
}

impl SfmrReconstruction {
    /// Stamp the metadata a save of this value to `path` by `stamp` records.
    ///
    /// Sets `operation`, `tool` and `tool_version`, refreshes the four counts
    /// from the arrays, and recomputes both workspace paths: `relative_path`
    /// as [`Self::measured_workspace_path`] measures it from `path`, and
    /// `absolute_path` as the workspace directory itself. `relative_path` is
    /// left as it was when there is no path to measure between the two.
    ///
    /// A `stamp` carrying a [`SaveStamp::workspace_path`] states
    /// `relative_path` instead: the stated value is recorded through
    /// [`Self::set_workspace_relative_path`] and no measurement happens, so
    /// neither directory is resolved and `path` is not consulted for it.
    /// `absolute_path` is stamped either way.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sfmtool_core::progress::Progress;
    /// use sfmtool_core::reconstruction::minimal::SaveStamp;
    /// use sfmtool_core::SfmrReconstruction;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut recon = SfmrReconstruction::load("in.sfmr".as_ref(), &Progress::none())?;
    /// let out = std::path::Path::new("out/moved.sfmr");
    /// recon.stamp_save(
    ///     out,
    ///     &SaveStamp {
    ///         operation: "xform",
    ///         tool: "sfmtool",
    ///         tool_version: "0.2.0",
    ///         workspace_path: None,
    ///     },
    /// );
    /// recon.save(out)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn stamp_save(&mut self, path: &Path, stamp: &SaveStamp<'_>) {
        let meta = &mut self.metadata;
        meta.operation = stamp.operation.to_string();
        meta.tool = stamp.tool.to_string();
        meta.tool_version = stamp.tool_version.to_string();
        meta.image_count = self.image_table.images.len() as u32;
        meta.point_count = self.point_set.points.len() as u32;
        meta.observation_count = self.point_set.tracks.len() as u32;
        meta.camera_count = self.image_table.cameras.len() as u32;

        match stamp.workspace_path {
            Some(stated) => self.set_workspace_relative_path(stated),
            None => {
                if let Some(measured) = self.measured_workspace_path(path) {
                    self.metadata.workspace.relative_path = measured;
                }
            }
        }
        self.metadata.workspace.absolute_path = self.workspace_dir.to_string_lossy().to_string();
    }

    /// The `workspace.relative_path` a save of this value to `output` measures:
    /// the path from `output`'s directory to [`Self::workspace_dir`], in POSIX
    /// form.
    ///
    /// The measurement [`Self::stamp_save`] performs when the caller states no
    /// path, offered on its own so that a caller about to state one can show what
    /// it would have been. `None` where there is no such path to measure, which
    /// is a root `output` with no directory to stand in, or two paths with
    /// nothing in common to walk between (different drives on Windows, say); the
    /// stamp then leaves the recorded path as it was. A relative `output` is
    /// taken against the current directory, and both directories are resolved to
    /// their real locations first where they exist (`comparable`), so a
    /// symlinked or aliased output gives the step or two between the two rather
    /// than a walk from the filesystem root.
    pub fn measured_workspace_path(&self, output: &Path) -> Option<String> {
        let output = std::path::absolute(output).unwrap_or_else(|_| output.to_path_buf());
        let parent = output.parent()?;
        let (workspace, parent) = comparable(&self.workspace_dir, parent);
        let relative = pathdiff::diff_paths(workspace, parent)?;
        Some(relative.to_string_lossy().replace('\\', "/"))
    }

    /// Record `stated` as `workspace.relative_path`, as it stands.
    ///
    /// The one place a stated path is put into the metadata, so every caller
    /// records it in the same form the measured path is in: POSIX, with any
    /// `\` turned into `/`. Nothing else about it is interpreted, so `"."` is a
    /// file beside its workspace marker and `""` is no path recorded.
    pub fn set_workspace_relative_path(&mut self, stated: &str) {
        self.metadata.workspace.relative_path = stated.replace('\\', "/");
    }

    /// Clear the metadata a minimal file does not carry.
    ///
    /// Empties `workspace.absolute_path` (an empty value means none was
    /// recorded), drops every `lineage` entry, and empties `tool_options`, which
    /// the caller then fills with the options of the operation writing the file.
    /// Everything else stays: `workspace.relative_path` and
    /// `workspace.contents` are how a reader finds the workspace and its
    /// features, and the rest is the reconstruction or a format fact.
    ///
    /// Run after [`Self::stamp_save`], which records the absolute path this
    /// clears.
    pub fn clear_minimal_metadata(&mut self) {
        let meta = &mut self.metadata;
        meta.workspace.absolute_path.clear();
        meta.lineage.clear();
        meta.tool_options.clear();
    }

    /// The minimal copy of this value, to be written at `path` by `stamp`.
    ///
    /// Both heavy columns dropped (the thumbnails and the patch bitmaps, whether
    /// the bitmaps are the reconstruction's or were rendered for display; the
    /// patch frames and normals stay), then [`Self::stamp_save`] for `path`,
    /// then [`Self::clear_minimal_metadata`], then `tool_options` set to the
    /// given map. The copy carries no content hash, since it is not the content
    /// this value was read as; the save that writes it computes one.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::collections::BTreeMap;
    /// use sfmtool_core::progress::Progress;
    /// use sfmtool_core::reconstruction::minimal::SaveStamp;
    /// use sfmtool_core::SfmrReconstruction;
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let recon = SfmrReconstruction::load("in.sfmr".as_ref(), &Progress::none())?;
    /// let out = std::path::Path::new("published/in.sfmr");
    /// let stamp = SaveStamp {
    ///     operation: "minimal",
    ///     tool: "sfm-explorer",
    ///     tool_version: "0.2.0",
    ///     workspace_path: None,
    /// };
    /// recon.to_minimal(out, &stamp, BTreeMap::new()).save(out)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_minimal(
        &self,
        path: &Path,
        stamp: &SaveStamp<'_>,
        tool_options: BTreeMap<String, serde_json::Value>,
    ) -> SfmrReconstruction {
        let mut minimal = self.clone_for_edit();
        minimal.image_table.thumbnails_y_x_rgb = None;
        minimal.point_set.patch_bitmaps_y_x_rgba = None;
        minimal.point_set.patch_bitmaps_for_display = false;
        minimal.stamp_save(path, stamp);
        minimal.clear_minimal_metadata();
        minimal.metadata.tool_options = tool_options;
        minimal
    }
}

/// The two paths a save diffs, in one form, so the diff between them means
/// something.
///
/// A reader resolves the workspace from a canonicalized `.sfmr` directory, so
/// both sides are canonicalized here where they can be. With one side resolved
/// and the other not, the two share only the filesystem root on a host whose
/// temporary directory is a symlink (`/var` to `/private/var`) or a short 8.3
/// alias (`RUNNER~1` for `runneradmin`), and the relative path comes out as a
/// walk from that root rather than the step or two it is. Canonicalizing needs
/// the directory to exist, and the output's does not have to yet, so a pair
/// where either side is missing stays lexical: the two agreeing matters more
/// than either being resolved. Only the relative result is kept, so a verbatim
/// `\\?\` prefix on Windows never reaches the metadata.
fn comparable(to: &Path, from: &Path) -> (PathBuf, PathBuf) {
    match (std::fs::canonicalize(to), std::fs::canonicalize(from)) {
        (Ok(to), Ok(from)) => (to, from),
        _ => (to.to_path_buf(), from.to_path_buf()),
    }
}

#[cfg(test)]
mod tests;
