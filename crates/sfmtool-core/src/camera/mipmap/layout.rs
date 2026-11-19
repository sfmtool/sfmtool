// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Every level's size for a photograph's size, its tier and how many of its
//! levels its mipmap stores.

use super::{MipmapOptions, MAX_JPEG_SIDE};
use crate::camera::image::ImageU8Pyramid;

/// Every level's size for a photograph's size, whether it is out of core, and
/// how many levels its mipmap stores.
///
/// Level `l` is `floor(width / 2^l)` by `floor(height / 2^l)`, as repeated
/// [`ImageU8::downsample_2x`](crate::camera::image::ImageU8::downsample_2x)
/// makes it, and there are [`ImageU8Pyramid::full_levels`] levels. An
/// out-of-core photograph's mipmap stores levels 1 to `K`, where `K` is the
/// first level whose longer side is at most one tile, or the last level whose
/// sides are both at least 1 px if that comes first; the levels below `K` are
/// box-filtered from it in memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyramidLayout {
    width: u32,
    height: u32,
    levels: usize,
    stored_levels: usize,
    tile_size: u32,
    options: Option<MipmapOptions>,
}

impl PyramidLayout {
    /// The layout of a `width` x `height` photograph under `options`, the
    /// workspace's `mipmap_options` (`None` for a workspace without the object
    /// or no workspace, where every photograph is in core), with tiles of
    /// `tile_size` px (a size of 0 is taken as 1).
    ///
    /// `None` when a side is over [`MAX_JPEG_SIDE`]: such a photograph has no
    /// layout, since every pyramid reader refuses it.
    pub fn for_size(
        width: u32,
        height: u32,
        options: Option<&MipmapOptions>,
        tile_size: u32,
    ) -> Option<Self> {
        if width > MAX_JPEG_SIDE || height > MAX_JPEG_SIDE {
            return None;
        }
        let tile_size = tile_size.max(1);
        let levels = ImageU8Pyramid::full_levels(width, height);
        let out_of_core = options.is_some_and(|o| width.max(height) > o.out_of_core_long_side);
        let stored_levels = if out_of_core {
            // The last level with both sides at least 1 px.
            let last = (0..levels)
                .take_while(|&l| (width >> l) >= 1 && (height >> l) >= 1)
                .last()
                .unwrap_or(0);
            (1..=last)
                .find(|&l| (width >> l).max(height >> l) <= tile_size)
                .unwrap_or(last)
        } else {
            0
        };
        Some(Self {
            width,
            height,
            levels,
            stored_levels,
            tile_size,
            options: options.copied(),
        })
    }

    /// The layout of a pyramid held in memory, with `levels` levels of a
    /// `width` x `height` level 0 and tiles of `tile_size` px (a size of 0 is
    /// taken as 1): in core, so no level is stored.
    pub(crate) fn in_memory(width: u32, height: u32, levels: usize, tile_size: u32) -> Self {
        Self {
            width,
            height,
            levels,
            stored_levels: 0,
            tile_size: tile_size.max(1),
            options: None,
        }
    }

    /// The number of levels, [`ImageU8Pyramid::full_levels`] of the size.
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// The size of level `level`, level 0's halved `level` times; `(0, 0)`
    /// past the last level.
    pub fn level_size(&self, level: usize) -> (u32, u32) {
        if level >= self.levels {
            return (0, 0);
        }
        // `levels` is at most 17 for a side of at most 65535, so the shift
        // cannot overflow.
        (self.width >> level, self.height >> level)
    }

    /// `K`, the coarsest level the mipmap stores: 0 for an in-core
    /// photograph, which has no mip level files, and for an out-of-core one
    /// too thin to have a level 1.
    pub fn stored_levels(&self) -> usize {
        self.stored_levels
    }

    /// Whether the photograph is out of core: its workspace has
    /// `mipmap_options` and a side is over its `out_of_core_long_side`.
    pub fn is_out_of_core(&self) -> bool {
        self.options
            .is_some_and(|o| self.width.max(self.height) > o.out_of_core_long_side)
    }

    /// The workspace's `mipmap_options` the layout was made under.
    pub fn options(&self) -> Option<&MipmapOptions> {
        self.options.as_ref()
    }

    /// The photograph tile size the layout was made with.
    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    /// The number of tile columns and rows of level `level`; `(0, 0)` past
    /// the last level.
    pub fn tiles(&self, level: usize) -> (u32, u32) {
        let (w, h) = self.level_size(level);
        (w.div_ceil(self.tile_size), h.div_ceil(self.tile_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(threshold: u32) -> MipmapOptions {
        MipmapOptions {
            out_of_core_long_side: threshold,
            jpeg_quality: 95,
            full_resolution_jpeg_quality: 95,
        }
    }

    fn layout(width: u32, height: u32) -> PyramidLayout {
        PyramidLayout::for_size(width, height, Some(&options(4096)), 512).unwrap()
    }

    #[test]
    fn stored_levels_follow_the_size() {
        // The rows of the draft's table, and the edge where level 3 of a
        // 4097 px side is exactly one tile.
        for (w, h, levels, k, coarsest) in [
            (16320, 12240, 14, 5, (510, 382)),
            (6000, 4000, 12, 4, (375, 250)),
            (4097, 2304, 12, 3, (512, 288)),
            (30000, 1000, 10, 6, (468, 15)),
        ] {
            let l = layout(w, h);
            assert!(l.is_out_of_core());
            assert_eq!(l.levels(), levels, "{w} x {h}");
            assert_eq!(l.stored_levels(), k, "{w} x {h}");
            assert_eq!(l.level_size(k), coarsest, "{w} x {h}");
        }
    }

    #[test]
    fn the_threshold_and_the_object_decide_the_tier() {
        let at = PyramidLayout::for_size(4096, 3000, Some(&options(4096)), 512).unwrap();
        assert!(!at.is_out_of_core());
        assert_eq!(at.stored_levels(), 0);
        assert!(layout(4097, 10).is_out_of_core());
        let none = PyramidLayout::for_size(16320, 12240, None, 512).unwrap();
        assert!(!none.is_out_of_core());
        assert_eq!(none.stored_levels(), 0);
        assert_eq!(none.levels(), 14);
    }

    #[test]
    fn the_test_photograph_has_several_levels_and_many_tiles() {
        // The 300 x 200 synthetic photograph of the tests, with a threshold of
        // 64 and tiles of 32: level 3 is 37 x 25, so level 4 (18 x 12) is
        // the first whose longer side is at most 32 px, and is K.
        let l = PyramidLayout::for_size(300, 200, Some(&options(64)), 32).unwrap();
        assert!(l.is_out_of_core());
        assert_eq!(l.levels(), 8);
        assert_eq!(l.stored_levels(), 4);
        assert_eq!(l.level_size(4), (18, 12));
        assert_eq!(l.tiles(0), (10, 7));
        assert_eq!(l.tiles(4), (1, 1));
        assert_eq!(l.level_size(8), (0, 0));
        assert_eq!(l.tiles(8), (0, 0));
    }

    #[test]
    fn a_one_pixel_high_photograph_stores_no_reduced_level() {
        // Level 1 of a 70 x 1 photograph is 0 px high, so its coarsest level
        // with both sides at least 1 px is level 0.
        let l = PyramidLayout::for_size(70, 1, Some(&options(64)), 32).unwrap();
        assert!(l.is_out_of_core());
        assert_eq!(l.stored_levels(), 0);
    }

    #[test]
    fn a_side_over_the_jpeg_limit_has_no_layout() {
        let l = PyramidLayout::for_size(65535, 100, Some(&options(64)), 32).unwrap();
        assert!(l.is_out_of_core());
        assert!(PyramidLayout::for_size(65536, 100, Some(&options(64)), 32).is_none());
        assert!(PyramidLayout::for_size(100, 65536, None, 32).is_none());
    }
}
