// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Where a photograph's mip level files are, found from its path alone.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use super::MIPMAP_DIR_NAME;

/// The contents of the `CACHEDIR.TAG` written in a new mipmap directory: the
/// standard signature line, which backup tools honour by skipping the
/// directory, and a comment.
pub const CACHEDIR_TAG: &str = "Signature: 8a477f597d28d172789f06886806bc55\n\
# This file is a cache directory tag created by sfmtool.\n\
# For information about cache directory tags, see https://bford.info/cachedir/\n";

/// Where a photograph's mip level files are: its workspace root, its path
/// relative to that root, and the directory under the root's mipmap
/// directory that mirrors the photograph's own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MipmapLocation {
    root: PathBuf,
    relative: String,
    dir: PathBuf,
    file_name: OsString,
}

impl MipmapLocation {
    /// The workspace root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The photograph's path relative to the workspace root, with `/`
    /// separators, as a record's `source` holds it.
    pub fn relative(&self) -> &str {
        &self.relative
    }

    /// The directory holding the photograph's mip level files,
    /// `<root>/.sfm-mipmaps/<parent of relative path>`.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The path of the mip level file of `level` at `quality`:
    /// `<dir>/<file name>-mip<level>-q<quality>.jpg`.
    pub fn level_path(&self, level: usize, quality: u8) -> PathBuf {
        self.dir
            .join(mip_level_file_name(&self.file_name, level, quality))
    }
}

/// The name of a mip level file: the original's whole file name, in its own
/// case and extension included, then `-mip<level>-q<quality>.jpg`.
pub fn mip_level_file_name(original: &std::ffi::OsStr, level: usize, quality: u8) -> OsString {
    let mut name = original.to_os_string();
    name.push(format!("-mip{level}-q{quality}.jpg"));
    name
}

/// Where the mip level files of `photograph` are: `None` when the photograph
/// is in no workspace (no `.sfm-workspace.json` in any ancestor), or its
/// path has no file name or leads outside the workspace (a `..`).
///
/// The workspace is found by [`find_workspace_for_path`], which walks up the
/// path as given without resolving links, so a photograph reached through a
/// linked image folder belongs to the workspace around the link.
///
/// [`find_workspace_for_path`]: crate::workspace::find_workspace_for_path
pub fn mipmap_location(photograph: &Path) -> Option<MipmapLocation> {
    let root = crate::workspace::find_workspace_for_path(photograph)?;
    let absolute = std::path::absolute(photograph).ok()?;
    let relative_path = absolute.strip_prefix(&root).ok()?;
    let mut parts = Vec::new();
    for component in relative_path.components() {
        match component {
            Component::Normal(part) => parts.push(part),
            Component::CurDir => {}
            _ => return None,
        }
    }
    let (file_name, parents) = parts.split_last()?;
    let mut dir = root.join(MIPMAP_DIR_NAME);
    for part in parents {
        dir.push(part);
    }
    let relative = parts
        .iter()
        .map(|p| p.to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    Some(MipmapLocation {
        root,
        relative,
        dir,
        file_name: file_name.to_os_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn location(ws: &Path, relative: &str) -> MipmapLocation {
        mipmap_location(&ws.join(relative)).unwrap()
    }

    #[test]
    fn names_keep_the_original_s_whole_name_in_its_case() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path();
        std::fs::write(ws.join(crate::workspace::WORKSPACE_FILE_NAME), "{}").unwrap();
        let root = crate::workspace::find_workspace_for_path(&ws.join("x")).unwrap();
        let mipmaps = root.join(MIPMAP_DIR_NAME);

        let img = location(ws, "images/IMG_0001.JPG");
        assert_eq!(img.relative(), "images/IMG_0001.JPG");
        assert_eq!(
            img.level_path(1, 95),
            mipmaps.join("images").join("IMG_0001.JPG-mip1-q95.jpg")
        );
        assert_eq!(
            img.level_path(0, 98),
            mipmaps.join("images").join("IMG_0001.JPG-mip0-q98.jpg")
        );
        assert_eq!(
            location(ws, "a/b/a.b.png").level_path(2, 90),
            mipmaps.join("a").join("b").join("a.b.png-mip2-q90.jpg")
        );
        assert_eq!(
            location(ws, "README").level_path(1, 5),
            mipmaps.join("README-mip1-q5.jpg")
        );
        // Two photographs whose names differ only in extension.
        assert_ne!(
            location(ws, "x.jpg").level_path(1, 95),
            location(ws, "x.png").level_path(1, 95)
        );
        assert_eq!(location(ws, "./images/./c.jpg").relative(), "images/c.jpg");
    }

    #[test]
    fn a_photograph_outside_every_workspace_has_no_location() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(mipmap_location(&dir.path().join("images/a.jpg")), None);
    }
}
