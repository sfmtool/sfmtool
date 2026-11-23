// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The tile-addressed path: a render that reads each tap from the photograph
//! tile that holds it, through a window of pinned tiles.
//!
//! A bilinear cell's taps `x0` and `x0 + 1`, clamped to the level, lie in the
//! core of the tile that holds `x0` (clamped) and its one-pixel border, which
//! holds the neighbour's pixel, or the edge pixel at the image's edge, as
//! clamping reads it. So a cell is read from one tile, and every tap is the
//! pixel the whole level gives.

use std::cell::RefCell;
use std::sync::Arc;

use crate::camera::remap::{LevelTexels, PyramidTexels};
use crate::camera::tiled_photograph::{
    PhotographReadError, PhotographTile, TileKey, TiledPhotograph,
};
use crate::progress::Progress;

/// The most photograph tiles one render on the tile path keeps pinned; the
/// least recently used goes out first.
pub const TILE_WINDOW_TILES: usize = 16;

/// The tiles one render on the tile path holds, and what it has done.
///
/// A render runs on one thread, its rows in order, so the window has one
/// owner. A tile read that fails is remembered, its taps read as 0, and the
/// render's result is discarded for the error ([`Self::finish`]).
pub(crate) struct TileWindow<'s> {
    source: &'s TiledPhotograph,
    /// The caller's `Progress`, which a tile read polls for cancellation.
    progress: Progress<'s>,
    state: RefCell<WindowState>,
}

struct WindowState {
    tiles: Vec<Pinned>,
    /// The tile the last tap came from.
    current: usize,
    clock: u64,
    /// The tiles fetched from the source.
    fetched: u64,
    /// The most tiles pinned at once.
    max_pinned: usize,
    error: Option<PhotographReadError>,
    /// What a tap reads after a failed tile read.
    zeros: Vec<u8>,
}

struct Pinned {
    tile: Arc<PhotographTile>,
    used: u64,
}

/// What a render on the tile path did, for the tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WindowStats {
    pub(crate) fetched: u64,
    pub(crate) max_pinned: usize,
}

impl<'s> TileWindow<'s> {
    pub(crate) fn new(source: &'s TiledPhotograph, progress: &Progress<'s>) -> Self {
        Self {
            source,
            progress: *progress,
            state: RefCell::new(WindowState {
                tiles: Vec::with_capacity(TILE_WINDOW_TILES),
                current: 0,
                clock: 0,
                fetched: 0,
                max_pinned: 0,
                error: None,
                // Two cells' worth, so any base index in `0..4` and any
                // channel can be read.
                zeros: vec![0; 4 * source.channels() as usize],
            }),
        }
    }

    /// The window's counts, and the first failed tile read, if any.
    pub(crate) fn finish(self) -> Result<WindowStats, PhotographReadError> {
        let state = self.state.into_inner();
        match state.error {
            Some(error) => Err(error),
            None => Ok(WindowStats {
                fetched: state.fetched,
                max_pinned: state.max_pinned,
            }),
        }
    }

    /// Calls `f` with the tile of `level` whose core holds `(x, y)` (level
    /// pixels inside the level), or with `None` after a failed read.
    fn with_tile<R>(
        &self,
        level: usize,
        x: u32,
        y: u32,
        f: impl FnOnce(Option<&PhotographTile>, &[u8]) -> R,
    ) -> R {
        let ts = self.source.layout().tile_size();
        let key = TileKey {
            level: level as u8,
            col: x / ts,
            row: y / ts,
        };
        let mut state = self.state.borrow_mut();
        let state = &mut *state;
        state.clock += 1;
        let found = match state.tiles.get(state.current) {
            Some(pinned) if pinned.tile.key == key => Some(state.current),
            _ => state.tiles.iter().position(|p| p.tile.key == key),
        };
        let index = match found {
            Some(index) => index,
            None => match self.source.tile(key, &self.progress) {
                Ok(tile) => {
                    state.fetched += 1;
                    let pinned = Pinned {
                        tile,
                        used: state.clock,
                    };
                    if state.tiles.len() < TILE_WINDOW_TILES {
                        state.tiles.push(pinned);
                        state.max_pinned = state.max_pinned.max(state.tiles.len());
                        state.tiles.len() - 1
                    } else {
                        let oldest = (0..state.tiles.len())
                            .min_by_key(|&i| state.tiles[i].used)
                            .expect("the window is full");
                        state.tiles[oldest] = pinned;
                        oldest
                    }
                }
                Err(error) => {
                    state.error.get_or_insert(error);
                    return f(None, &state.zeros);
                }
            },
        };
        state.current = index;
        state.tiles[index].used = state.clock;
        f(Some(&state.tiles[index].tile), &state.zeros)
    }
}

/// One level of a [`TileWindow`]'s photograph, read through its tiles.
pub(crate) struct TileLevel<'w, 's> {
    window: &'w TileWindow<'s>,
    level: usize,
    width: i32,
    height: i32,
    channels: usize,
}

impl LevelTexels for TileLevel<'_, '_> {
    #[inline]
    fn channels(&self) -> usize {
        self.channels
    }

    fn with_cell<R>(&self, x0: i32, y0: i32, f: impl FnOnce(&[u8], [usize; 4]) -> R) -> R {
        // A level of no pixels clamps to 0, whose tile the source refuses.
        let (w, h, c) = (
            (self.width - 1).max(0),
            (self.height - 1).max(0),
            self.channels,
        );
        let cx0 = x0.clamp(0, w);
        let cx1 = x0.saturating_add(1).clamp(0, w);
        let cy0 = y0.clamp(0, h);
        let cy1 = y0.saturating_add(1).clamp(0, h);
        self.window
            .with_tile(self.level, cx0 as u32, cy0 as u32, |tile, zeros| {
                let Some(tile) = tile else {
                    return f(zeros, [0; 4]);
                };
                // The tile's pixels start one pixel above and left of its
                // core.
                let lx0 = (cx0 as u32 - tile.origin[0] + 1) as usize;
                let ly0 = (cy0 as u32 - tile.origin[1] + 1) as usize;
                let lx1 = lx0 + (cx1 - cx0) as usize;
                let ly1 = ly0 + (cy1 - cy0) as usize;
                let stride = tile.pixels.width() as usize * c;
                f(
                    tile.pixels.data(),
                    [
                        ly0 * stride + lx0 * c,
                        ly0 * stride + lx1 * c,
                        ly1 * stride + lx0 * c,
                        ly1 * stride + lx1 * c,
                    ],
                )
            })
    }

    fn with_pixel<R>(&self, x: i32, y: i32, f: impl FnOnce(&[u8]) -> R) -> R {
        let c = self.channels;
        let x = x.clamp(0, (self.width - 1).max(0)) as u32;
        let y = y.clamp(0, (self.height - 1).max(0)) as u32;
        self.window.with_tile(self.level, x, y, |tile, zeros| {
            let Some(tile) = tile else {
                return f(&zeros[..c]);
            };
            let lx = (x - tile.origin[0] + 1) as usize;
            let ly = (y - tile.origin[1] + 1) as usize;
            let at = (ly * tile.pixels.width() as usize + lx) * c;
            f(&tile.pixels.data()[at..at + c])
        })
    }
}

impl<'s> PyramidTexels for TileWindow<'s> {
    type Level<'a>
        = TileLevel<'a, 's>
    where
        Self: 'a;

    fn texels(&self, level: usize) -> TileLevel<'_, 's> {
        let (w, h) = self.source.layout().level_size(level);
        TileLevel {
            window: self,
            level,
            width: w as i32,
            height: h as i32,
            channels: self.source.channels() as usize,
        }
    }

    fn num_levels(&self) -> usize {
        self.source.layout().levels()
    }

    fn channels(&self) -> usize {
        self.source.channels() as usize
    }
}
