// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The metadata a save stamps, and the minimal file.
//!
//! A save by a tool records who wrote the file and where the workspace is from
//! the file's own location ([`SfmrReconstruction::stamp_save`]). A **minimal**
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
use std::path::Path;

use super::SfmrReconstruction;

/// Who a save records as having written the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaveStamp<'a> {
    /// The operation that produced the file's content, such as `"xform"`.
    pub operation: &'a str,
    /// The tool that performed it, such as `"sfmtool"`.
    pub tool: &'a str,
    /// That tool's version.
    pub tool_version: &'a str,
}

impl SfmrReconstruction {
    /// Stamp the metadata a save of this value to `path` by `stamp` records.
    ///
    /// Sets `operation`, `tool` and `tool_version`, refreshes the four counts
    /// from the arrays, and recomputes both workspace paths: `relative_path`
    /// from `path`'s directory to [`Self::workspace_dir`], in POSIX form, and
    /// `absolute_path` as the workspace directory itself. A relative `path` is
    /// taken against the current directory. `relative_path` is left as it was
    /// when no relative path exists between the two (different drives, say).
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
    ///     &SaveStamp { operation: "xform", tool: "sfmtool", tool_version: "0.2.0" },
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

        let output = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
        if let Some(parent) = output.parent() {
            if let Some(relative) = pathdiff::diff_paths(&self.workspace_dir, parent) {
                meta.workspace.relative_path = relative.to_string_lossy().replace('\\', "/");
            }
        }
        meta.workspace.absolute_path = self.workspace_dir.to_string_lossy().to_string();
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
    /// let stamp = SaveStamp { operation: "minimal", tool: "sfm-explorer", tool_version: "0.2.0" };
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

#[cfg(test)]
mod tests;
