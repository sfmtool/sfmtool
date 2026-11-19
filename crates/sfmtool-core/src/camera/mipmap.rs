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
//! gives every level's size and how many levels are stored. The design is in
//! the spec `specs/core/camera/mipmaps.md`.

mod layout;
mod region;
#[cfg(test)]
mod test_support;

pub use layout::PyramidLayout;
pub use region::{region_layout, NotRegionReadable, RegionLayout};

/// The directory at a workspace root that holds the workspace's mip level
/// files. Every search for a workspace's images skips a directory of this
/// name, since the mip level files are JPEGs named after the photographs.
pub const MIPMAP_DIR_NAME: &str = ".sfm-mipmaps";

/// The `out_of_core_long_side` that `sfm ws init` is to write: a photograph
/// with a side over this is out of core.
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
