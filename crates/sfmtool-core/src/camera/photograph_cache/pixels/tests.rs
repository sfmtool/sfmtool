// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `read_layout` and the reads of levels, tiles and regions, on 300 x 200
//! synthetic photographs in a temporary workspace whose `mipmap_options` sets
//! a threshold of 64, read with tiles of 32.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use crate::camera::mipmap::test_support::{encode_jpeg, synthetic_rgb, JpegOptions};
use crate::camera::sampler::{render_source_tile, test_regions, Sampler};
use crate::camera::{TiledPhotograph, WarpMap};
use crate::progress::Event;

use super::super::mipmap_tests::{cache_of, readable_jpeg, restartless_jpeg, Workspace, TILE};
use super::super::GIB;
use super::*;

/// The three photographs of a workspace: one in core, one out of core and
/// region-readable, and one out of core with no restart markers, which gets a
/// `-mip0`.
fn three(ws: &Workspace) -> [(PathBuf, Option<bool>); 3] {
    let small = encode_jpeg(&synthetic_rgb(60, 40), JpegOptions::default());
    [
        (ws.photograph("small.jpg", &small), None),
        (ws.photograph("readable.jpg", &readable_jpeg()), Some(true)),
        (
            ws.photograph("restartless.jpg", &restartless_jpeg()),
            Some(false),
        ),
    ]
}

/// `read_layout` answers the layout `get` answers the photograph under, and
/// out of core the region answer, before the mipmap is made, from the
/// region test, making no mipmap, and after, from the files' records; a
/// photograph outside every workspace is in core.
#[test]
fn read_layout_answers_the_layout_get_reads_under_and_makes_no_mipmap() {
    let ws = Workspace::new();
    let cache = cache_of(GIB);
    for (path, region_readable) in three(&ws) {
        let files = ws.mipmap_files();
        let before = cache.read_layout(&path).unwrap();
        assert_eq!(ws.mipmap_files(), files, "{path:?}: no mipmap made");
        assert_eq!(before.region_readable, region_readable, "{path:?}");
        assert_eq!(before.layout.is_out_of_core(), region_readable.is_some());
        let photograph = cache.get(&path, &Progress::none()).unwrap();
        assert_eq!(*before.layout, *photograph.layout(), "{path:?}");
        assert_eq!(cache.read_layout(&path).unwrap(), before, "{path:?}");
    }
    let outside = tempfile::tempdir().unwrap();
    let path = outside.path().join("big.jpg");
    std::fs::write(&path, readable_jpeg()).unwrap();
    let layout = cache.read_layout(&path).unwrap();
    assert_eq!(layout.region_readable, None);
    assert!(!layout.layout.is_out_of_core());
    assert_eq!(layout.layout.level_size(0), (300, 200));
}

/// A level `l` rendered by a kernel, a map whose texel `(c, r)` samples the
/// centre of level `l`'s pixel `(c, r)`, so `Bilinear` at level 0 and
/// `BilinearMip` pinned to level `l` by its scale of `2^l` read each pixel
/// back. The map has a column and a row past the level, cropped from the
/// render, so a level one pixel wide still has a scale along each axis.
fn rendered(photograph: &TiledPhotograph, level: usize) -> ImageU8 {
    let (w, h) = photograph.layout().level_size(level);
    let scale = (1u32 << level) as f32;
    let mut data = Vec::with_capacity(2 * ((w + 1) * (h + 1)) as usize);
    for r in 0..=h {
        for c in 0..=w {
            data.extend([scale * (c as f32 + 0.5), scale * (r as f32 + 0.5)]);
        }
    }
    let mut map = WarpMap::new(w + 1, h + 1, data);
    let sampler = if level == 0 {
        Sampler::Bilinear
    } else {
        Sampler::BilinearMip
    };
    let render = render_source_tile(
        &mut test_regions(photograph),
        &mut map,
        sampler,
        &Progress::none(),
    )
    .unwrap();
    let rect = LevelRect {
        x0: 0,
        y0: 0,
        x1: w,
        y1: h,
    };
    ImageU8::new(w, h, render.channels(), crop(&render, rect))
}

/// The pixels `rect` of `image`.
fn crop(image: &ImageU8, rect: LevelRect) -> Vec<u8> {
    let c = image.channels() as usize;
    let stride = image.width() as usize * c;
    (rect.y0..rect.y1)
        .flat_map(|y| {
            let from = y as usize * stride + rect.x0 as usize * c;
            image.data()[from..from + rect.width() as usize * c].to_vec()
        })
        .collect()
}

/// Every level read whole, each tile and a region read through the cache, are
/// the pixels a kernel samples from the same file through another cache: a
/// `Bilinear` render of level 0 and a `BilinearMip` render pinned to each
/// level read them back. In core, out of core from a region-readable
/// original, and out of core from a `-mip0`; with a budget and with one of
/// 0. In core level 0 is the file's own decode.
#[test]
fn a_level_a_tile_and_a_region_are_what_a_kernel_samples() {
    let ws = Workspace::new();
    for (path, _) in three(&ws) {
        // The kernel's cache makes the mipmap of an out-of-core photograph.
        let kernel = cache_of(GIB).get(&path, &Progress::none()).unwrap();
        let layout = kernel.layout().clone();
        for budget in [GIB, 0] {
            let cache = cache_of(budget);
            for level in 0..layout.levels() {
                let read = cache.read_level(&path, level, &Progress::none()).unwrap();
                let render = rendered(&kernel, level);
                assert_eq!(
                    (read.width(), read.height(), read.channels()),
                    (render.width(), render.height(), 3)
                );
                assert!(read.data() == render.data(), "{path:?} level {level}");
                let (cols, rows) = layout.tiles(level);
                for (col, row) in [(0, 0), (cols - 1, rows - 1)] {
                    let key = TileKey {
                        level: level as u8,
                        col,
                        row,
                    };
                    let tile = cache.read_tile(&path, key, &Progress::none()).unwrap();
                    let (w, h) = layout.level_size(level);
                    let rect = LevelRect {
                        x0: col * TILE,
                        y0: row * TILE,
                        x1: ((col + 1) * TILE).min(w),
                        y1: ((row + 1) * TILE).min(h),
                    };
                    assert_eq!((tile.width(), tile.height()), (rect.width(), rect.height()));
                    assert!(tile.data() == crop(&read, rect), "{path:?} {key:?}");
                }
            }
            // A rectangle across four tiles of level 0.
            let rect = LevelRect {
                x0: 20,
                y0: 7,
                x1: 45,
                y1: 39,
            };
            let region = cache
                .read_region(&path, 0, rect, &Progress::none())
                .unwrap();
            assert!(
                region.data() == crop(&rendered(&kernel, 0), rect),
                "{path:?}"
            );
        }
        if layout.is_out_of_core() {
            continue;
        }
        let decoded = ImageU8::read_rgb(&path).unwrap();
        let level0 = cache_of(0).read_level(&path, 0, &Progress::none()).unwrap();
        assert!(level0.data() == decoded.data(), "{path:?}");
    }
}

/// A read of an out-of-core photograph whose mipmap is missing makes it,
/// writing its files, and reports `mipmaps_made` once; a read of the current
/// files makes nothing and reports nothing.
#[test]
fn a_read_makes_a_missing_mipmap_and_reports_it() {
    let ws = Workspace::new();
    let path = ws.photograph("a.jpg", &readable_jpeg());
    let made = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        if let Event::Value {
            name: MIPMAPS_MADE,
            value,
        } = event
        {
            made.lock().unwrap().push(value);
        }
    };
    let cache = cache_of(0);
    cache.read_level(&path, 1, &Progress::to(&sink)).unwrap();
    assert!(!ws.mipmap_files().is_empty(), "the mipmap was written");
    assert_eq!(*made.lock().unwrap(), [1.0]);
    let files = ws.mipmap_files();
    cache_of(GIB)
        .read_level(&path, 2, &Progress::to(&sink))
        .unwrap();
    assert_eq!(*made.lock().unwrap(), [1.0], "nothing made again");
    assert_eq!(ws.mipmap_files(), files);
}

/// A level, tile or rectangle the photograph does not have, and an empty
/// rectangle, are `OutOfRange` naming the photograph; a missing file is
/// `NotFound`, to the reads and to `read_layout`; an invalid
/// `mipmap_options` is `Workspace`; a cancelled read is `Cancelled`.
#[test]
fn reads_refuse_what_the_photograph_does_not_have() {
    let ws = Workspace::new();
    let path = ws.photograph("a.jpg", &readable_jpeg());
    let cache = cache_of(GIB);
    let none = &Progress::none();
    let rect = |x0, y0, x1, y1| LevelRect { x0, y0, x1, y1 };
    let levels = cache.read_layout(&path).unwrap().layout.levels();
    let out_of_range = [
        cache.read_level(&path, levels, none),
        cache.read_region(&path, 0, rect(0, 0, 301, 10), none),
        cache.read_region(&path, 1, rect(0, 0, 10, 101), none),
        cache.read_region(&path, 0, rect(5, 5, 5, 10), none),
        cache.read_region(&path, levels, rect(0, 0, 1, 1), none),
        cache.read_tile(
            &path,
            TileKey {
                level: 0,
                col: 10,
                row: 0,
            },
            none,
        ),
        cache.read_tile(
            &path,
            TileKey {
                level: levels as u8,
                col: 0,
                row: 0,
            },
            none,
        ),
    ];
    for (i, result) in out_of_range.into_iter().enumerate() {
        match result {
            Err(TileError::OutOfRange { path: named, .. }) => assert_eq!(named, path, "{i}"),
            other => panic!("{i}: {:?}", other.map(|_| ())),
        }
    }

    let gone = ws.dir.path().join("images").join("gone.jpg");
    match cache.read_level(&gone, 0, none) {
        Err(TileError::Unreadable {
            why: GetError::NotFound(named),
            ..
        }) => assert_eq!(named, gone),
        other => panic!("{:?}", other.map(|_| ())),
    }
    assert_eq!(
        cache.read_layout(&gone),
        Err(GetError::NotFound(gone.clone()))
    );

    let cancel = AtomicBool::new(true);
    let cancelled = Progress::none().cancelled_by(&cancel);
    assert_eq!(
        cache_of(0).read_level(&path, 0, &cancelled).map(|_| ()),
        Err(TileError::Cancelled)
    );

    let broken = tempfile::tempdir().unwrap();
    std::fs::write(
        broken.path().join(crate::workspace::WORKSPACE_FILE_NAME),
        r#"{"version": 2, "mipmap_options": {"out_of_core_long_side": "big"}}"#,
    )
    .unwrap();
    let path = broken.path().join("a.jpg");
    std::fs::write(&path, readable_jpeg()).unwrap();
    assert!(matches!(
        cache.read_layout(&path),
        Err(GetError::Workspace(_))
    ));
    assert!(matches!(
        cache.read_level(&path, 0, none),
        Err(TileError::Unreadable {
            why: GetError::Workspace(_),
            ..
        })
    ));
}
