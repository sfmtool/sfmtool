// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A photograph's mipmap: its reduced levels, and when its original cannot be
//! read by region its full-resolution copy, stored as JPEG mip level files
//! under the workspace's mipmap directory, `.sfm-mipmaps/`.
//!
//! A photograph of a workspace whose `.sfm-workspace.json` has a
//! `mipmap_options` object ([`MipmapOptions`], read by
//! [`crate::workspace::read_mipmap_options`]) is out of core when a side is
//! over the object's `out_of_core_long_side`; [`PyramidLayout`] says so and
//! gives every level's size and how many levels are stored.
//! [`photograph_layout`] reads both from a photograph's path. A photograph's
//! mip level files are named by [`mipmap_location`]; [`build_mipmaps`] makes
//! them from one decode of the original, [`check_mipmap`] says whether those
//! on disk are current, and [`MipLevelFile`] reads one, whole or a tile at a
//! time. [`check_mipmaps`] checks the mipmaps of many photographs, writing
//! nothing, and [`prune_mipmaps`] deletes the orphan files of a mipmap
//! directory, as `sfm ws mipmaps --check` and `--prune` do.
//! [`region_layout`] is the region test, which decides whether an original
//! needs a `-mip0`, and [`OriginalRegions`] reads a rectangle of an
//! original that passes it. The design is in the spec
//! `specs/core/camera/mipmaps.md`.

use std::fmt;
use std::path::PathBuf;

use crate::progress::Cancelled;
use crate::workspace::WorkspaceConfigError;

mod build;
mod check;
mod file;
mod layout;
mod location;
mod prune;
mod record;
mod region;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
mod tile;

pub use build::{build_mipmaps, check_mipmap, photograph_layout, BuiltMipmap};
pub use check::{check_mipmaps, MipmapCheckReport};
pub use file::{ExpectedMipLevel, MipLevelFile};
pub use layout::PyramidLayout;
pub use location::{mip_level_file_name, mipmap_location, MipmapLocation, CACHEDIR_TAG};
pub use prune::{prune_mipmaps, MipmapPruneReport, PruneMipmapsOptions, TEMPORARY_FILE_MAX_AGE};
pub use record::{MipLevelRecord, MipmapSource, MIP_LEVEL_RECORD_VERSION};
pub use region::{region_layout, NotRegionReadable, RegionLayout};
pub use tile::OriginalRegions;

/// The directory at a workspace root that holds the workspace's mip level
/// files. Every search for a workspace's images skips a directory of this
/// name, since the mip level files are JPEGs named after the photographs.
pub const MIPMAP_DIR_NAME: &str = ".sfm-mipmaps";

/// The `out_of_core_long_side` that `sfm ws init` writes, in
/// `NEW_WORKSPACE_MIPMAP_OPTIONS`: a photograph with a side over this is out
/// of core.
pub const DEFAULT_OUT_OF_CORE_LONG_SIDE: u32 = 4096;

/// The largest side of a photograph any pyramid reader accepts, the largest
/// side a JPEG frame can hold. The photograph cache refuses a photograph with
/// a larger side from its header, in every workspace, so every file the
/// mipmap may write fits in a JPEG.
pub const MAX_JPEG_SIDE: u32 = 65535;

/// The side of a photograph tile, the block a level is read in; the coarsest
/// stored mip level is the first whose longer side fits in one.
pub const PHOTOGRAPH_TILE_SIZE: u32 = 512;

/// The restart interval of every mip level file, in MCUs: 128 px of one MCU
/// row at 4:4:4, the unit a tile is decoded in.
pub const MIPMAP_RESTART_INTERVAL_MCUS: u16 = 16;

/// The widest restart interval, in pixels, of an original that can be read by
/// region: two tiles. An original with wider intervals gets a `-mip0`.
pub const MAX_REGION_INTERVAL_WIDTH: u32 = 1024;

/// A workspace's `mipmap_options` object. Every field is required in the
/// file; there are no defaults inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MipmapOptions {
    /// A photograph whose width or height is greater than this is out of
    /// core. At least 1.
    pub out_of_core_long_side: u32,
    /// The JPEG quality, 1 to 100, of the reduced mip level files, `-mip1`
    /// and coarser.
    pub jpeg_quality: u8,
    /// The JPEG quality, 1 to 100, of `-mip0`, the full-resolution copy of an
    /// original that cannot be read by region.
    pub full_resolution_jpeg_quality: u8,
}

impl MipmapOptions {
    /// The quality the mip level file of `level` is encoded at:
    /// `full_resolution_jpeg_quality` for level 0, `jpeg_quality` otherwise.
    pub fn quality_for_level(&self, level: usize) -> u8 {
        if level == 0 {
            self.full_resolution_jpeg_quality
        } else {
            self.jpeg_quality
        }
    }
}

/// Why a mipmap could not be made, read or found current.
#[derive(Debug)]
pub enum MipmapError {
    /// A file could not be read or written.
    Io {
        path: PathBuf,
        error: std::io::Error,
    },
    /// A mip level file is missing, damaged, or not current for its
    /// photograph; `why` names the first test it fails. The path is empty
    /// for bytes that were not read from a file.
    Stale { path: PathBuf, why: String },
    /// The original could not be decoded, or its header read.
    Source {
        path: PathBuf,
        error: ::image::ImageError,
    },
    /// The workspace's `mipmap_options` is invalid.
    Workspace(WorkspaceConfigError),
    /// A side is over [`MAX_JPEG_SIDE`], so no JPEG can hold the level.
    TooLarge {
        path: PathBuf,
        width: u32,
        height: u32,
    },
    /// The photograph has no mipmap: it is in no workspace, its workspace
    /// has no `mipmap_options`, or it is in core under them.
    NotOutOfCore { path: PathBuf },
    /// A rectangle asked of a mip level file is empty or not inside its
    /// level; `why` says which. The path is empty for bytes that were not
    /// read from a file.
    OutOfRange { path: PathBuf, why: String },
    /// The JPEG encoder refused a level.
    Encode(String),
    /// The caller's progress asked the build to stop.
    Cancelled,
}

impl fmt::Display for MipmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Stale { path, why } => write!(f, "{} is not current: {why}", path.display()),
            Self::Source { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Workspace(e) => write!(f, "{e}"),
            Self::TooLarge {
                path,
                width,
                height,
            } => write!(
                f,
                "{} is {width} x {height}, and a photograph with a side over \
                 {MAX_JPEG_SIDE} px cannot be read",
                path.display()
            ),
            Self::NotOutOfCore { path } => {
                write!(
                    f,
                    "{} is not out of core, so it has no mipmap",
                    path.display()
                )
            }
            Self::OutOfRange { path, why } => write!(f, "{}: {why}", path.display()),
            Self::Encode(why) => write!(f, "a mip level could not be encoded: {why}"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for MipmapError {}

impl From<Cancelled> for MipmapError {
    fn from(_: Cancelled) -> Self {
        Self::Cancelled
    }
}
