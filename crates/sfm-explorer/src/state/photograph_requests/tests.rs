// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use sfmtool_core::camera::image::{ImageU8, ImageU8Pyramid};
use sfmtool_core::camera::sampler::{read_source_rect, LevelRect, RegionPool};
use sfmtool_core::progress::Progress;
use sfmtool_core::SfmrReconstruction;

use super::super::{photograph_path, TaskQueue};

use super::*;

/// A demo reconstruction whose workspace is `dir`, with a PNG written for
/// image 0 and none for image 1.
fn demo_in(dir: &Path) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(8);
    assert!(recon.image_count() > 1, "the demo has two images");
    recon.workspace_dir = dir.to_path_buf();
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.name = format!("photo_{i:02}.png");
    }
    image::RgbImage::from_fn(6, 4, |x, y| image::Rgb([x as u8, y as u8, 3]))
        .save(dir.join("photo_00.png"))
        .unwrap();
    recon
}

// ── Image Detail's layout and tile requests ─────────────────────────────

/// A cache of `budget_bytes` whose whole decodes may reserve
/// `build_budget_bytes`.
fn tile_cache(budget_bytes: u64, build_budget_bytes: u64) -> Arc<PhotographCache> {
    Arc::new(PhotographCache::new(
        sfmtool_core::camera::PhotographCacheOptions {
            budget_bytes,
            build_budget_bytes,
            ..Default::default()
        },
    ))
}

/// Image 0 of [`demo_in`]'s reconstruction, 6 x 4: one tile a level. Its path
/// is the one the panels read, `photograph_path`'s.
fn photo_in(dir: &Path) -> PathBuf {
    photograph_path(&demo_in(dir), 0).unwrap()
}

/// Every tile of every level of a photograph whose levels are each one tile.
fn one_tile_a_level(layout: &PyramidLayout) -> Vec<TileKey> {
    (0..layout.levels())
        .map(|level| TileKey {
            level: level as u8,
            col: 0,
            row: 0,
        })
        .collect()
}

fn is_ready(tiles: &[DisplayTile]) -> bool {
    tiles.iter().all(|t| matches!(t, DisplayTile::Ready(_)))
}

fn is_loading(tiles: &[DisplayTile]) -> bool {
    tiles.iter().all(|t| matches!(t, DisplayTile::Loading))
}

/// The layout of `source`'s photograph once its header read has run.
fn known_layout(source: PhotographSource<'_>, queue: &TaskQueue) -> Arc<PyramidLayout> {
    let ctx = egui::Context::default();
    assert!(matches!(
        display_layout(source, &ctx),
        DisplayLayout::Loading
    ));
    assert_eq!(queue.run_off_this_thread(), 1, "one header read");
    match display_layout(source, &ctx) {
        DisplayLayout::Known(layout) => layout,
        other => panic!("{other:?}"),
    }
}

/// The layout and the tiles are asked for once each however many times they
/// are looked at while the read runs, are read off the calling thread, with a
/// budget, with one smaller than the photograph's level 0 and with a cache
/// that keeps nothing, and the tiles a decode answers are handed over on a
/// later look, with nothing asked for again.
#[test]
fn the_layout_and_the_tiles_are_read_once_off_the_calling_thread() {
    // Level 0 of the 6 x 4 photograph is 72 bytes.
    for budget in [1 << 30, 36, 0] {
        let dir = tempfile::tempdir().unwrap();
        let path = photo_in(dir.path());
        let cache = tile_cache(budget, 1 << 30);
        let (requests, queue) = PhotographRequests::queued();
        let source = PhotographSource {
            cache: &cache,
            requests: &requests,
            path: &path,
        };
        let ctx = egui::Context::default();
        for _ in 0..2 {
            assert!(matches!(
                display_layout(source, &ctx),
                DisplayLayout::Loading
            ));
        }
        assert_eq!(queue.len(), 1, "one header read however often it is asked");
        assert_eq!(queue.run_off_this_thread(), 1);
        let DisplayLayout::Known(layout) = display_layout(source, &ctx) else {
            panic!("the header was read");
        };
        assert_eq!(layout.level_size(0), (6, 4));
        let keys = one_tile_a_level(&layout);
        for _ in 0..3 {
            assert!(is_loading(&display_tiles(source, &keys, &ctx)));
        }
        assert_eq!(queue.len(), 1, "one decode however often it is asked");
        assert_eq!(cache.stats().decodes, 0, "nothing decoded on this thread");
        assert_eq!(queue.run_off_this_thread(), 1);
        assert_eq!(cache.stats().decodes, 1);
        let tiles = display_tiles(source, &keys, &ctx);
        assert!(is_ready(&tiles), "budget {budget}");
        assert_eq!(queue.len(), 0, "nothing asked for again");
        assert_eq!(cache.stats().decodes, 1);
    }
}

/// A decode whose reservation the build budget refuses, the budget used up by
/// another caller, decodes nothing and ends at once, without waiting for the
/// reservation: the task returns within the time limit while the other
/// reservation is still held, which nothing gives back while it runs. The
/// next look asks again, and once the reservation is given back the decode
/// runs. With a budget and with a cache that keeps nothing.
#[test]
fn a_refused_decode_is_asked_again_and_decodes_once_the_budget_has_room() {
    for budget in [1 << 30, 0] {
        let dir = tempfile::tempdir().unwrap();
        let path = photo_in(dir.path());
        let build_budget = 1000;
        let cache = tile_cache(budget, build_budget);
        let (requests, queue) = PhotographRequests::queued();
        let source = PhotographSource {
            cache: &cache,
            requests: &requests,
            path: &path,
        };
        let ctx = egui::Context::default();
        let keys = one_tile_a_level(&known_layout(source, &queue));
        let other = cache.reserve_builds(build_budget, 1);
        for _ in 0..2 {
            assert!(is_loading(&display_tiles(source, &keys, &ctx)));
            assert_eq!(queue.len(), 1, "asked again on a later look");
            assert_eq!(queue.run_off_this_thread(), 1);
            assert_eq!(cache.stats().decodes, 0, "budget {budget}");
            assert_eq!(cache.stats().build_reserved_bytes, build_budget);
        }
        drop(other);
        assert!(is_loading(&display_tiles(source, &keys, &ctx)));
        assert_eq!(queue.run_off_this_thread(), 1);
        assert_eq!(cache.stats().decodes, 1);
        assert_eq!(cache.stats().build_reserved_bytes, 0);
        assert!(is_ready(&display_tiles(source, &keys, &ctx)));
    }
}

/// A photograph whose header reads but whose pixels do not is answered as
/// unreadable once its decode has failed, layout and tiles alike, and is not
/// decoded again while its file stays the same, under a cache that keeps
/// nothing and so remembers no failure itself; a look starts only a check of
/// the file, off the calling thread. Once the file is fixed, the check drops
/// the failure and the photograph is decoded.
#[test]
fn a_failed_decode_is_answered_as_unreadable_until_the_file_changes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cut_short.png");
    image::RgbImage::from_fn(64, 64, |x, y| image::Rgb([x as u8, y as u8, 3]))
        .save(&path)
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
    let cache = tile_cache(0, 1 << 30);
    let (requests, queue) = PhotographRequests::queued();
    let source = PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &path,
    };
    let ctx = egui::Context::default();
    let keys = one_tile_a_level(&known_layout(source, &queue));
    assert!(is_loading(&display_tiles(source, &keys, &ctx)));
    assert_eq!(queue.run_off_this_thread(), 1);
    assert!(matches!(
        display_layout(source, &ctx),
        DisplayLayout::Unreadable
    ));
    let tiles = display_tiles(source, &keys, &ctx);
    assert!(tiles.iter().all(|t| matches!(t, DisplayTile::Unreadable)));
    prefetch_tiles(source, &keys, &ctx);
    assert_eq!(queue.len(), 1, "one check of the file, however often asked");
    let misses = cache.stats().misses;
    assert_eq!(queue.run_off_this_thread(), 1);
    assert!(matches!(
        display_layout(source, &ctx),
        DisplayLayout::Unreadable
    ));
    assert_eq!(queue.run_off_this_thread(), 1, "the file is checked again");
    assert_eq!(cache.stats().misses, misses, "not decoded again");

    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(
        display_layout(source, &ctx),
        DisplayLayout::Unreadable
    ));
    assert_eq!(queue.run_off_this_thread(), 1, "the check finds it changed");
    assert!(matches!(
        display_layout(source, &ctx),
        DisplayLayout::Known(_)
    ));
    assert!(is_loading(&display_tiles(source, &keys, &ctx)));
    assert_eq!(queue.run_off_this_thread(), 1);
    assert!(is_ready(&display_tiles(source, &keys, &ctx)));
}

/// Panels that show more photographs whose decode fails than the answers
/// kept for the panels, here five under a cache that keeps nothing and so
/// remembers no failure itself, and ask for each on every frame, do not
/// decode them in turn: once each has failed, a look answers it as
/// unreadable and starts only a check of its file, which finds it unchanged
/// and asks for no repaint, so the frames end.
#[test]
fn more_failed_photographs_than_the_kept_answers_are_not_decoded_again() {
    let dir = tempfile::tempdir().unwrap();
    let paths: Vec<PathBuf> = (0..MAX_ANSWERED_PHOTOGRAPHS + 1)
        .map(|i| {
            let path = dir.path().join(format!("cut_short_{i}.png"));
            image::RgbImage::from_fn(64, 64, |x, y| image::Rgb([x as u8, y as u8, i as u8]))
                .save(&path)
                .unwrap();
            let bytes = std::fs::read(&path).unwrap();
            std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
            path
        })
        .collect();
    let cache = tile_cache(0, 1 << 30);
    let (requests, queue) = PhotographRequests::queued();
    let ctx = egui::Context::default();
    let repaints = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = Arc::clone(&repaints);
    ctx.set_request_repaint_callback(move |_| {
        counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
    let pool = RegionPool::with_capacity(u64::MAX);
    let mut needs = vec![Vec::new(); paths.len()];
    let mut look = || {
        paths
            .iter()
            .zip(&mut needs)
            .map(|(path, needs)| {
                let source = PhotographSource {
                    cache: &cache,
                    requests: &requests,
                    path,
                };
                matches!(
                    read_peeked(source, needs, &ctx, |src| read_rect(src, &pool, LEFT)),
                    PeekedRead::Unreadable
                )
            })
            .collect::<Vec<bool>>()
    };
    // The headers, then the first tiles, whose decodes fail.
    for tasks in ["the header reads", "the decodes"] {
        assert!(look().iter().all(|unreadable| !unreadable), "{tasks}");
        assert_eq!(queue.run_off_this_thread(), paths.len(), "{tasks}");
    }
    assert_eq!(cache.stats().misses, paths.len() as u64);
    let repainted = repaints.load(std::sync::atomic::Ordering::Relaxed);
    for round in 0..10 {
        assert!(look().iter().all(|&unreadable| unreadable), "round {round}");
        assert_eq!(
            queue.run_off_this_thread(),
            paths.len(),
            "a check of each file, round {round}"
        );
        assert_eq!(
            cache.stats().misses,
            paths.len() as u64,
            "decoded again, round {round}"
        );
    }
    assert_eq!(
        repaints.load(std::sync::atomic::Ordering::Relaxed),
        repainted,
        "a repaint asked for with nothing changed"
    );
}

/// Tiles a decode answered for a view that has moved on are not kept: each
/// ask for the photograph drops its kept tiles it does not ask for, so a
/// panel that pans while its decodes run holds at most one decode's answer,
/// under a cache that keeps nothing, where nothing else bounds them.
#[test]
fn tiles_answered_for_a_view_that_moved_on_are_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("long.png");
    image::RgbImage::from_fn(1600, 8, |x, y| image::Rgb([x as u8, y as u8, 3]))
        .save(&path)
        .unwrap();
    let cache = tile_cache(0, 1 << 30);
    let (requests, queue) = PhotographRequests::queued();
    let source = PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &path,
    };
    let ctx = egui::Context::default();
    known_layout(source, &queue);
    let tile = |col| TileKey {
        level: 0,
        col,
        row: 0,
    };
    for col in 0..4 {
        // The view has moved on to the next tile before the last decode's
        // answer was asked for.
        assert!(is_loading(&display_tiles(source, &[tile(col)], &ctx)));
        assert_eq!(queue.run_off_this_thread(), 1);
        assert_eq!(requests.kept_tiles(), 1, "column {col}");
    }
    assert!(is_ready(&display_tiles(source, &[tile(3)], &ctx)));
    assert_eq!(requests.kept_tiles(), 0);
    // Asking ahead drops what it no longer asks for too.
    prefetch_tiles(source, &[tile(0)], &ctx);
    assert_eq!(queue.run_off_this_thread(), 1);
    prefetch_tiles(source, &[tile(1)], &ctx);
    assert_eq!(requests.kept_tiles(), 0);
}

/// A decode refused because it asked for a tile the photograph does not have,
/// which only a layout read before the file changed size can name, drops the
/// layout, so the next look reads the header again rather than ask for the
/// same tile on every frame.
#[test]
fn a_tile_the_photograph_does_not_have_reads_the_layout_again() {
    let dir = tempfile::tempdir().unwrap();
    let path = photo_in(dir.path());
    let cache = tile_cache(0, 1 << 30);
    let (requests, queue) = PhotographRequests::queued();
    let source = PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &path,
    };
    let ctx = egui::Context::default();
    known_layout(source, &queue);
    let past = TileKey {
        level: 0,
        col: 5,
        row: 0,
    };
    assert!(is_loading(&display_tiles(source, &[past], &ctx)));
    assert_eq!(queue.run_off_this_thread(), 1);
    assert!(matches!(
        display_layout(source, &ctx),
        DisplayLayout::Loading
    ));
    assert_eq!(queue.len(), 1, "the header read again");
}

/// Tiles asked for ahead of time are decoded off the calling thread, without
/// a peek, and handed over by the look that draws them with no second decode;
/// once they are answered, asking ahead again starts nothing. Under a cache
/// that keeps nothing, where only the answer holds them.
#[test]
fn tiles_asked_for_ahead_are_handed_over_when_drawn() {
    let dir = tempfile::tempdir().unwrap();
    let path = photo_in(dir.path());
    let cache = tile_cache(0, 1 << 30);
    let (requests, queue) = PhotographRequests::queued();
    let source = PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &path,
    };
    let ctx = egui::Context::default();
    let keys = one_tile_a_level(&known_layout(source, &queue));
    prefetch_tiles(source, &keys, &ctx);
    prefetch_tiles(source, &keys, &ctx);
    assert_eq!(queue.len(), 1);
    assert_eq!(cache.stats().decodes, 0);
    assert_eq!(queue.run_off_this_thread(), 1);
    prefetch_tiles(source, &keys, &ctx);
    assert_eq!(queue.len(), 0, "every tile is answered");
    assert!(is_ready(&display_tiles(source, &keys, &ctx)));
    assert_eq!(cache.stats().decodes, 1);
    assert_eq!(queue.len(), 0);
}

// ── The 3D background's level ───────────────────────────────────────────

/// A 37 x 23 photograph at `dir/name`, whose levels cross several tiles of
/// [`small_tile_cache`]'s 8 px, with a partial tile at the right and bottom
/// edges.
fn level_photo(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    image::RgbImage::from_fn(37, 23, |x, y| {
        image::Rgb([(x * 7) as u8, (y * 11) as u8, (x * y) as u8])
    })
    .save(&path)
    .unwrap();
    path
}

/// A cache of `budget_bytes` that cuts tiles of 8 px.
fn small_tile_cache(budget_bytes: u64, build_budget_bytes: u64) -> Arc<PhotographCache> {
    Arc::new(PhotographCache::new(
        sfmtool_core::camera::PhotographCacheOptions {
            budget_bytes,
            build_budget_bytes,
            tile_size: 8,
        },
    ))
}

fn level_of(answer: DisplayLevel) -> Arc<ImageU8> {
    match answer {
        DisplayLevel::Ready(pixels) => pixels,
        DisplayLevel::Loading => panic!("still loading"),
        DisplayLevel::Unreadable => panic!("unreadable"),
    }
}

/// Every level of the photograph is read off the calling thread by one task
/// however often it is asked for while the task runs, built from the tiles
/// `try_decode_tiles` answers with a budget and with a cache that keeps
/// nothing, and is the level of the whole pyramid the background drew
/// before step 13c3. The task makes one call to the
/// cache, `try_decode_tiles`: with a budget, a level read of a photograph the
/// cache holds counts one hit, which a second read of the cache, such as a
/// `get_level` that could wait on the slot, would make two.
#[test]
fn the_level_is_read_once_off_the_calling_thread_and_is_the_pyramid_s_level() {
    let dir = tempfile::tempdir().unwrap();
    let path = level_photo(dir.path(), "level.png");
    // The pyramid the cache's decode builds.
    let before = {
        let image = ImageU8::read_rgb(&path).unwrap();
        let levels = ImageU8Pyramid::full_levels(image.width(), image.height());
        ImageU8Pyramid::from_image(image, levels)
    };
    for budget in [1 << 30, 0] {
        let cache = small_tile_cache(budget, 1 << 30);
        let (requests, queue) = PhotographRequests::queued();
        let source = PhotographSource {
            cache: &cache,
            requests: &requests,
            path: &path,
        };
        let ctx = egui::Context::default();
        let layout = known_layout(source, &queue);
        assert_eq!(layout.levels(), before.num_levels());
        assert_eq!(layout.tiles(0), (5, 3), "level 0 crosses several tiles");
        for level in 0..layout.levels() {
            for _ in 0..3 {
                assert!(matches!(
                    display_level(source, &layout, level, &ctx),
                    DisplayLevel::Loading
                ));
            }
            assert_eq!(queue.len(), 1, "one task however often it is asked");
            let (decodes, hits) = (cache.stats().decodes, cache.stats().hits);
            assert_eq!(queue.run_off_this_thread(), 1);
            assert!(requests.keeps_a_level());
            let pixels = level_of(display_level(source, &layout, level, &ctx));
            assert!(!requests.keeps_a_level(), "handed over once");
            let expected = before.level(level);
            assert_eq!(
                (pixels.width(), pixels.height(), pixels.channels()),
                (expected.width(), expected.height(), expected.channels()),
            );
            assert!(
                pixels.data() == expected.data(),
                "level {level}, budget {budget}"
            );
            // Under a budget the photograph is decoded once and then held;
            // under a budget of 0 each level's task decodes it. None is
            // decoded on this thread.
            let expected_decodes = if budget == 0 || level == 0 {
                decodes + 1
            } else {
                decodes
            };
            assert_eq!(cache.stats().decodes, expected_decodes);
            let expected_hits = if budget == 0 || level == 0 {
                hits
            } else {
                hits + 1
            };
            assert_eq!(
                cache.stats().hits,
                expected_hits,
                "one read of the cache, level {level}, budget {budget}"
            );
        }
    }
}

/// A level read whose decode does not fit under the build budget, which
/// another caller's reservation holds, answers `Busy`: it decodes nothing and
/// ends within the time limit while the reservation is still held, and the
/// next frame asks again. Once the reservation is given back the level is
/// read. With a budget and with a cache that keeps nothing.
#[test]
fn a_busy_level_read_is_asked_again_on_a_later_frame() {
    for budget in [1 << 30, 0] {
        let dir = tempfile::tempdir().unwrap();
        let path = level_photo(dir.path(), "busy.png");
        let build_budget = 1000;
        let cache = small_tile_cache(budget, build_budget);
        let (requests, queue) = PhotographRequests::queued();
        let source = PhotographSource {
            cache: &cache,
            requests: &requests,
            path: &path,
        };
        let ctx = egui::Context::default();
        let layout = known_layout(source, &queue);
        let other = cache.reserve_builds(build_budget, 1);
        for _ in 0..2 {
            assert!(matches!(
                display_level(source, &layout, 1, &ctx),
                DisplayLevel::Loading
            ));
            assert_eq!(queue.len(), 1, "asked again on a later frame");
            assert_eq!(queue.run_off_this_thread(), 1);
            assert_eq!(cache.stats().decodes, 0, "budget {budget}");
            assert!(!requests.keeps_a_level());
        }
        drop(other);
        assert!(matches!(
            display_level(source, &layout, 1, &ctx),
            DisplayLevel::Loading
        ));
        assert_eq!(queue.run_off_this_thread(), 1);
        assert_eq!(cache.stats().decodes, 1);
        assert_eq!(
            level_of(display_level(source, &layout, 1, &ctx)).width(),
            18
        );
    }
}

/// The background shares the tile decodes' one task per photograph: while
/// Image Detail's decode of the photograph runs, a level read of it starts
/// nothing, and once that decode has ended the level read starts. A level
/// answered and not taken is dropped when another level is asked for.
#[test]
fn a_level_read_waits_its_turn_behind_a_tile_decode_of_the_photograph() {
    let dir = tempfile::tempdir().unwrap();
    let path = level_photo(dir.path(), "shared.png");
    let cache = small_tile_cache(0, 1 << 30);
    let (requests, queue) = PhotographRequests::queued();
    let source = PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &path,
    };
    let ctx = egui::Context::default();
    let layout = known_layout(source, &queue);
    let tile = TileKey {
        level: 0,
        col: 0,
        row: 0,
    };
    assert!(is_loading(&display_tiles(source, &[tile], &ctx)));
    assert!(matches!(
        display_level(source, &layout, 2, &ctx),
        DisplayLevel::Loading
    ));
    assert_eq!(queue.len(), 1, "one task for the photograph");
    assert_eq!(queue.run_off_this_thread(), 1);
    assert!(!requests.keeps_a_level(), "the task was the tile decode");
    assert!(matches!(
        display_level(source, &layout, 2, &ctx),
        DisplayLevel::Loading
    ));
    assert_eq!(queue.run_off_this_thread(), 1, "the level read, after it");
    assert!(requests.keeps_a_level());
    assert!(matches!(
        display_level(source, &layout, 1, &ctx),
        DisplayLevel::Loading
    ));
    assert!(!requests.keeps_a_level(), "level 2 dropped");
    assert_eq!(queue.run_off_this_thread(), 1);
    assert_eq!(
        level_of(display_level(source, &layout, 1, &ctx)).width(),
        18
    );
}

/// A level read of a photograph whose file has changed size since its layout
/// was read, wider by two pixels, so a tile no longer fits the level, or
/// narrower by one, so the tiles fit but would leave the last column unset,
/// or of a level the layout does not have, reads the layout again rather than
/// ask for the same level on every frame, with a budget and with a cache that
/// keeps nothing.
#[test]
fn a_level_read_against_a_stale_layout_reads_the_layout_again() {
    for budget in [1 << 30, 0] {
        for (width, level) in [(Some(39), 0), (Some(36), 0), (None, 9)] {
            let dir = tempfile::tempdir().unwrap();
            let path = level_photo(dir.path(), "changed.png");
            let cache = small_tile_cache(budget, 1 << 30);
            let (requests, queue) = PhotographRequests::queued();
            let source = PhotographSource {
                cache: &cache,
                requests: &requests,
                path: &path,
            };
            let ctx = egui::Context::default();
            let layout = known_layout(source, &queue);
            if let Some(width) = width {
                image::RgbImage::from_fn(width, 23, |x, _| image::Rgb([x as u8, 0, 0]))
                    .save(&path)
                    .unwrap();
                assert_eq!(layout.tiles(0), (5, 3), "the same tiles asked for");
            }
            assert!(matches!(
                display_level(source, &layout, level, &ctx),
                DisplayLevel::Loading
            ));
            assert_eq!(queue.run_off_this_thread(), 1);
            assert!(
                !requests.keeps_a_level(),
                "budget {budget}, width {width:?}, level {level}"
            );
            assert!(matches!(
                display_layout(source, &ctx),
                DisplayLayout::Loading
            ));
            assert_eq!(queue.len(), 1, "the header read again");
        }
    }
}

/// A level read of a photograph whose header reads but whose pixels do not
/// is answered as unreadable, layout and level alike, and is not decoded
/// again while its file stays the same, under a cache that keeps nothing and
/// so remembers no failure itself.
#[test]
fn a_failed_level_read_is_answered_as_unreadable_and_not_decoded_again() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cut_short.png");
    image::RgbImage::from_fn(64, 64, |x, y| image::Rgb([x as u8, y as u8, 3]))
        .save(&path)
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
    let cache = tile_cache(0, 1 << 30);
    let (requests, queue) = PhotographRequests::queued();
    let source = PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &path,
    };
    let ctx = egui::Context::default();
    let layout = known_layout(source, &queue);
    assert!(matches!(
        display_level(source, &layout, 0, &ctx),
        DisplayLevel::Loading
    ));
    assert_eq!(queue.run_off_this_thread(), 1);
    assert_eq!(cache.stats().misses, 1);
    for _ in 0..2 {
        assert!(matches!(
            display_level(source, &layout, 0, &ctx),
            DisplayLevel::Unreadable
        ));
        assert!(matches!(
            display_layout(source, &ctx),
            DisplayLayout::Unreadable
        ));
        // Only a check of the file runs, and it finds the file unchanged.
        assert_eq!(queue.run_off_this_thread(), 1);
    }
    assert_eq!(cache.stats().misses, 1, "not decoded again");
}

// ── Track View's renders from peeked tiles ──────────────────────────────

/// `rect` of level 0 of the photograph `src`, read through `pool`.
fn read_rect(
    src: &TiledPhotograph,
    pool: &RegionPool,
    rect: LevelRect,
) -> Result<Vec<u8>, PhotographReadError> {
    read_source_rect(src, pool, 0, rect, &Progress::none()).map(|read| read.image().data().to_vec())
}

/// The pixels of `rect` of the photograph at `path`, decoded here.
fn decoded_rect(path: &Path, rect: LevelRect) -> Vec<u8> {
    let image = ImageU8::read_rgb(path).unwrap();
    let mut data = Vec::new();
    for y in rect.y0..rect.y1 {
        let row = (y * image.width()) as usize * 3;
        data.extend_from_slice(
            &image.data()[row + rect.x0 as usize * 3..row + rect.x1 as usize * 3],
        );
    }
    data
}

/// A rectangle of level 0 of [`level_photo`] that covers 12 of its 8 px
/// tiles.
const LEFT: LevelRect = LevelRect {
    x0: 3,
    y0: 2,
    x1: 30,
    y1: 20,
};

/// A render through `read_peeked` waits, drawing nothing, while the layout
/// and then the tiles it needs are read off the calling thread, each asked
/// for once however often it is looked at, and runs once every tile is
/// there, on a photograph made of them, giving the photograph's own pixels.
/// Nothing is decoded on the calling thread, with a budget and with a cache
/// that keeps nothing.
#[test]
fn a_render_from_peeked_tiles_runs_once_they_are_all_there() {
    for budget in [1 << 30, 0] {
        let dir = tempfile::tempdir().unwrap();
        let path = level_photo(dir.path(), "photo.png");
        let cache = small_tile_cache(budget, 1 << 30);
        let (requests, queue) = PhotographRequests::queued();
        let source = PhotographSource {
            cache: &cache,
            requests: &requests,
            path: &path,
        };
        let ctx = egui::Context::default();
        let pool = RegionPool::with_capacity(u64::MAX);
        let mut needs = Vec::new();
        let renders = std::cell::Cell::new(0);
        let look = |needs: &mut Vec<TileKey>| {
            read_peeked(source, needs, &ctx, |src| {
                renders.set(renders.get() + 1);
                read_rect(src, &pool, LEFT)
            })
        };
        // The header, then the coarsest level's first tile.
        for tasks in ["a header read", "a decode of the first tile"] {
            for _ in 0..3 {
                assert!(matches!(look(&mut needs), PeekedRead::Loading));
            }
            assert_eq!(queue.len(), 1, "{tasks}, once, budget {budget}");
            assert_eq!(cache.stats().decodes, 0, "nothing decoded on this thread");
            assert_eq!(queue.run_off_this_thread(), 1);
        }
        assert_eq!(renders.get(), 0, "nothing rendered before a tile is there");
        assert_eq!(cache.stats().decodes, 1);
        let read = if budget > 0 {
            // The cache holds the photograph: the frame that learns the
            // tiles has them.
            look(&mut needs)
        } else {
            // Only the decode's answer holds them: the frame that learns
            // them asks for them, once.
            for _ in 0..3 {
                assert!(matches!(look(&mut needs), PeekedRead::Loading));
            }
            assert_eq!(needs.len(), 12, "the twelve tiles under the rectangle");
            assert_eq!(renders.get(), 1, "the render that learned them");
            assert_eq!(queue.len(), 1, "one decode of the tiles");
            assert_eq!(cache.stats().decodes, 1, "nothing decoded on this thread");
            assert_eq!(queue.run_off_this_thread(), 1);
            look(&mut needs)
        };
        let PeekedRead::Read(Ok(pixels)) = read else {
            panic!("the render ran, budget {budget}");
        };
        assert_eq!(pixels, decoded_rect(&path, LEFT));
        assert_eq!(needs.len(), 12);
        // One render on the first tile alone learned the tiles; the next ran.
        assert_eq!(renders.get(), 2, "budget {budget}");
        assert_eq!(queue.len(), 0);
        assert_eq!(cache.stats().decodes, if budget > 0 { 1 } else { 2 });
    }
}

/// A photograph that cannot be read is answered as unreadable, its render
/// never run.
#[test]
fn a_render_of_a_photograph_that_cannot_be_read_is_answered_as_unreadable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing.png");
    let cache = small_tile_cache(1 << 30, 1 << 30);
    let (requests, queue) = PhotographRequests::queued();
    let source = PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &path,
    };
    let ctx = egui::Context::default();
    let mut needs = Vec::new();
    let render = |_: &TiledPhotograph| -> Result<(), PhotographReadError> {
        panic!("rendered a photograph that cannot be read")
    };
    assert!(matches!(
        read_peeked(source, &mut needs, &ctx, render),
        PeekedRead::Loading
    ));
    assert_eq!(queue.run_off_this_thread(), 1);
    assert!(matches!(
        read_peeked(source, &mut needs, &ctx, render),
        PeekedRead::Unreadable
    ));
}

/// A render through `read_peeked_without_asking` starts nothing. On a
/// photograph nothing has asked for it waits, with no header read or decode
/// started however often it looks; once `read_peeked` has had the
/// photograph read into the cache it runs on what is there, giving the
/// photograph's own pixels. A photograph whose decode failed under a cache
/// that keeps nothing is answered as unreadable, with no check of its file
/// started.
#[test]
fn a_render_that_asks_for_nothing_runs_only_on_what_is_there() {
    let dir = tempfile::tempdir().unwrap();
    let path = level_photo(dir.path(), "photo.png");
    let cache = small_tile_cache(1 << 30, 1 << 30);
    let (requests, queue) = PhotographRequests::queued();
    let source = PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &path,
    };
    let ctx = egui::Context::default();
    let pool = RegionPool::with_capacity(u64::MAX);
    let render = |src: &TiledPhotograph| read_rect(src, &pool, LEFT);
    let mut needs = Vec::new();
    for _ in 0..3 {
        assert!(matches!(
            read_peeked_without_asking(source, &mut needs, render),
            PeekedRead::Loading
        ));
    }
    assert_eq!(queue.len(), 0, "nothing asked for");
    // The header, then the first tile, whose decode keeps the photograph.
    let mut asked = Vec::new();
    for _ in 0..2 {
        assert!(matches!(
            read_peeked(source, &mut asked, &ctx, render),
            PeekedRead::Loading
        ));
        assert_eq!(queue.run_off_this_thread(), 1);
    }
    let PeekedRead::Read(Ok(pixels)) = read_peeked_without_asking(source, &mut needs, render)
    else {
        panic!("the photograph is in the cache");
    };
    assert_eq!(pixels, decoded_rect(&path, LEFT));
    assert_eq!(needs.len(), 12, "the twelve tiles under the rectangle");
    assert_eq!(queue.len(), 0);

    // A file that goes once its header is read.
    let cache = small_tile_cache(0, 1 << 30);
    let gone = level_photo(dir.path(), "gone.png");
    let source = PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &gone,
    };
    let mut needs = Vec::new();
    assert!(matches!(
        read_peeked(source, &mut needs, &ctx, render),
        PeekedRead::Loading
    ));
    assert_eq!(queue.run_off_this_thread(), 1, "the header read");
    std::fs::remove_file(&gone).unwrap();
    assert!(matches!(
        read_peeked(source, &mut needs, &ctx, render),
        PeekedRead::Loading
    ));
    assert_eq!(queue.run_off_this_thread(), 1, "the decode, which fails");
    assert!(matches!(
        read_peeked_without_asking(source, &mut needs, render),
        PeekedRead::Unreadable
    ));
    assert_eq!(queue.len(), 0, "no check of the file started");
}

/// Two renders of one photograph that need different tiles, and Image
/// Detail asking for others of its tiles until it has them, drawn before
/// both on every frame, all finish under a cache that keeps nothing, where
/// only the decodes' answers hold the tiles: no ask drops tiles another
/// render is waiting for so that it never gets them.
#[test]
fn two_renders_and_image_detail_on_one_photograph_all_get_their_tiles() {
    let dir = tempfile::tempdir().unwrap();
    let path = level_photo(dir.path(), "photo.png");
    let cache = small_tile_cache(0, 1 << 30);
    let (requests, queue) = PhotographRequests::queued();
    let source = PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &path,
    };
    let ctx = egui::Context::default();
    let pool = RegionPool::with_capacity(u64::MAX);
    let right = LevelRect {
        x0: 20,
        y0: 9,
        x1: 37,
        y1: 23,
    };
    let detail_keys: Vec<TileKey> = (0..3)
        .map(|col| TileKey {
            level: 0,
            col,
            row: 2,
        })
        .collect();
    let (mut left_needs, mut right_needs) = (Vec::new(), Vec::new());
    let (mut left_read, mut right_read, mut detail) = (None, None, false);
    for _ in 0..12 {
        if !detail {
            detail = is_ready(&display_tiles(source, &detail_keys, &ctx));
        }
        for (needs, rect, done) in [
            (&mut left_needs, LEFT, &mut left_read),
            (&mut right_needs, right, &mut right_read),
        ] {
            if done.is_none() {
                if let PeekedRead::Read(read) =
                    read_peeked(source, needs, &ctx, |src| read_rect(src, &pool, rect))
                {
                    *done = Some(read.expect("the tiles it needs"));
                }
            }
        }
        if detail && left_read.is_some() && right_read.is_some() {
            break;
        }
        queue.run_off_this_thread();
    }
    assert!(detail, "Image Detail got its tiles");
    assert_eq!(left_read, Some(decoded_rect(&path, LEFT)));
    assert_eq!(right_read, Some(decoded_rect(&path, right)));
    assert_eq!(requests.kept_tiles(), 0, "every answer was taken");
}

/// A photograph's answered tiles are bounded: an answer that would take
/// them past `MAX_KEPT_TILES` replaces them, and one that would not is
/// added to them.
#[test]
fn a_photograph_keeps_at_most_max_kept_tiles_answered_tiles() {
    let path = PathBuf::from("/kept/photograph.png");
    let key = |col| TileKey {
        level: 0,
        col,
        row: 0,
    };
    let answer = |cols: std::ops::Range<u32>| {
        cols.map(|col| {
            let tile = PhotographTile {
                key: key(col),
                origin: [col, 0],
                size: [1, 1],
                pixels: ImageU8::new(3, 3, 3, vec![0; 27]),
            };
            (key(col), Arc::new(tile))
        })
        .collect::<Answered>()
    };
    let mut tiles = TileRequests::default();
    tiles.keep(path.clone(), answer(0..20));
    tiles.keep(path.clone(), answer(20..30));
    assert_eq!(tiles.kept_tiles(), 30, "added");
    tiles.keep(path.clone(), answer(30..40));
    assert_eq!(tiles.kept_tiles(), 10, "replaced");
    assert!(tiles.holds(&path, key(35)));
    assert!(!tiles.holds(&path, key(5)));
    // One answer past the bound is kept whole.
    tiles.keep(path.clone(), answer(40..80));
    assert_eq!(tiles.kept_tiles(), 40);
    assert!((30..40).contains(&MAX_KEPT_TILES));
}

/// A render's tiles are handed over together or not at all: under a cache
/// that keeps nothing, a set whose tiles are partly answered takes none of
/// them, so the answered ones stay kept while the rest are decoded, and the
/// look after that decode hands over the whole set with no third decode.
#[test]
fn a_tile_set_takes_nothing_until_every_tile_is_there() {
    let dir = tempfile::tempdir().unwrap();
    let path = level_photo(dir.path(), "photo.png");
    let cache = small_tile_cache(0, 1 << 30);
    let (requests, queue) = PhotographRequests::queued();
    let source = PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &path,
    };
    let ctx = egui::Context::default();
    let tile = |col| TileKey {
        level: 0,
        col,
        row: 0,
    };
    // Another render's answer holds the first tile.
    assert!(matches!(
        display_tile_set(source, &[tile(0)], Some(&ctx)),
        DisplayTileSet::Loading
    ));
    assert_eq!(queue.run_off_this_thread(), 1);
    assert_eq!(requests.kept_tiles(), 1);
    // This render needs it and a second.
    let set = [tile(0), tile(1)];
    assert!(matches!(
        display_tile_set(source, &set, Some(&ctx)),
        DisplayTileSet::Loading
    ));
    assert_eq!(requests.kept_tiles(), 1, "the first tile is left kept");
    assert_eq!(queue.run_off_this_thread(), 1, "a decode of the second");
    let DisplayTileSet::Ready(tiles) = display_tile_set(source, &set, Some(&ctx)) else {
        panic!("both tiles are answered");
    };
    assert_eq!(tiles.iter().map(|t| t.key).collect::<Vec<_>>(), set);
    assert_eq!(requests.kept_tiles(), 0, "taken together");
    assert_eq!(cache.stats().decodes, 2);
}

// ── An out-of-core photograph without a current mipmap ──────────────────

/// The `mipmap_options` of [`out_of_core_photo`]'s workspace: a photograph
/// with a side over 64 px is out of core.
fn mipmap_options() -> sfmtool_core::camera::MipmapOptions {
    sfmtool_core::camera::MipmapOptions {
        out_of_core_long_side: 64,
        jpeg_quality: 95,
        full_resolution_jpeg_quality: 95,
    }
}

/// A 300 x 200 PNG in a workspace at `dir` whose `mipmap_options` puts it
/// out of core. A PNG is not region-readable, so its mipmap has a `-mip0`.
fn out_of_core_photo(dir: &Path) -> PathBuf {
    let o = mipmap_options();
    std::fs::write(
        dir.join(sfmtool_core::workspace::WORKSPACE_FILE_NAME),
        format!(
            r#"{{"version": 2, "mipmap_options": {{"out_of_core_long_side": {},
            "jpeg_quality": {}, "full_resolution_jpeg_quality": {}}}}}"#,
            o.out_of_core_long_side, o.jpeg_quality, o.full_resolution_jpeg_quality
        ),
    )
    .unwrap();
    let path = dir.join("large.png");
    image::RgbImage::from_fn(300, 200, |x, y| {
        image::Rgb([(x * 7) as u8, (y * 5) as u8, (x ^ y) as u8])
    })
    .save(&path)
    .unwrap();
    path
}

/// Make the mipmap of `path` with tiles of 32 px, the cache's.
fn build_mipmap(path: &Path) {
    sfmtool_core::camera::build_mipmaps(
        path,
        None,
        &mipmap_options(),
        32,
        &Progress::none(),
        &mut |_| {},
    )
    .unwrap();
}

/// Remove the mip level file of level 1 of `path`'s mipmap, which a read of
/// level 1 then finds missing.
fn remove_level_1(path: &Path) {
    let location = sfmtool_core::camera::mipmap::mipmap_location(path).unwrap();
    std::fs::remove_file(location.level_path(1, mipmap_options().jpeg_quality)).unwrap();
}

/// End a frame of `ctx`, so egui, which asks for one repaint however many
/// are requested before the next frame, tells the next request again.
fn end_frame(ctx: &egui::Context) {
    ctx.run_ui(egui::RawInput::default(), |_| {})
        .textures_delta
        .clear();
}

/// How many times each panel looks at the photograph, every look running
/// the tasks the looks started: more than its reads need, so a panel that
/// asked again on every look would start more tasks than the bound.
const LOOKS: usize = 10;

/// One look of Image Detail, Track View and the 3D background, in turn, at
/// an out-of-core photograph, each asking only for what it has not drawn:
/// Image Detail the tiles `detail_keys` once its layout is known, Track View
/// one render of a rectangle of the right of level 0, and the background
/// level 1 whole once its layout is known. Panics on anything answered as
/// unreadable.
#[allow(clippy::too_many_arguments)]
fn look_at(
    source: PhotographSource<'_>,
    ctx: &egui::Context,
    detail_keys: &[TileKey],
    drawn: &mut HashMap<TileKey, Arc<PhotographTile>>,
    needs: &mut Vec<TileKey>,
    rendered: &mut Option<Vec<u8>>,
    level: &mut Option<Arc<ImageU8>>,
    what: &str,
) {
    let right = LevelRect {
        x0: 200,
        y0: 100,
        x1: 260,
        y1: 150,
    };
    let pool = RegionPool::with_capacity(u64::MAX);
    match display_layout(source, ctx) {
        DisplayLayout::Known(layout) => {
            let missing: Vec<TileKey> = detail_keys
                .iter()
                .copied()
                .filter(|key| !drawn.contains_key(key))
                .collect();
            // Once drawn it asks for nothing, as its textures keep the tiles.
            let tiles = if missing.is_empty() {
                Vec::new()
            } else {
                display_tiles(source, &missing, ctx)
            };
            for (key, tile) in missing.iter().zip(tiles) {
                match tile {
                    DisplayTile::Ready(tile) => {
                        drawn.insert(*key, tile);
                    }
                    DisplayTile::Loading => {}
                    DisplayTile::Unreadable => panic!("{what}: tile unreadable"),
                }
            }
            if level.is_none() {
                match display_level(source, &layout, 1, ctx) {
                    DisplayLevel::Ready(pixels) => *level = Some(pixels),
                    DisplayLevel::Loading => {}
                    DisplayLevel::Unreadable => panic!("{what}: level unreadable"),
                }
            }
        }
        DisplayLayout::Loading | DisplayLayout::Building(_) => {}
        DisplayLayout::Unreadable | DisplayLayout::WorkspaceFileError(_) => {
            panic!("{what}: layout unreadable")
        }
    }
    if rendered.is_none() {
        match read_peeked(source, needs, ctx, |src| read_rect(src, &pool, right)) {
            PeekedRead::Read(read) => *rendered = Some(read.unwrap()),
            PeekedRead::Loading | PeekedRead::Building => {}
            PeekedRead::Unreadable => panic!("{what}: render unreadable"),
        }
    }
}

/// Image Detail, Track View and the 3D background ask, over many looks, for
/// an out-of-core photograph whose mipmap is missing, and for one whose
/// level-1 file is removed after Image Detail's tiles were answered, which
/// the background's read of level 1 finds. Under a budget of 0 and under one
/// smaller than the photograph and than any of its files: while the
/// photograph is `Building` no panel asks for a tile of it, and it is queued
/// once on *Build mipmaps*' list; no task a panel starts makes a mipmap; one
/// run of the task (here [`run_chunk`] on this thread) makes it, after which
/// every panel draws; none is answered as unreadable; and the looks start no
/// more tasks and repaints than the reads need.
#[test]
fn panels_wait_on_build_mipmaps_for_an_out_of_core_photograph_without_a_current_mipmap() {
    use super::super::build_mipmaps::run_chunk;
    let level0 = |col, row| TileKey { level: 0, col, row };
    let detail_keys = [level0(0, 0), level0(1, 0)];
    for case in ["missing", "level 1 removed"] {
        for budget in [0, 20_000] {
            let dir = tempfile::tempdir().unwrap();
            let path = out_of_core_photo(dir.path());
            if case != "missing" {
                build_mipmap(&path);
            }
            let cache = Arc::new(PhotographCache::new(
                sfmtool_core::camera::PhotographCacheOptions {
                    budget_bytes: budget,
                    build_budget_bytes: 1 << 30,
                    tile_size: 32,
                },
            ));
            let (requests, queue) = PhotographRequests::queued();
            let source = PhotographSource {
                cache: &cache,
                requests: &requests,
                path: &path,
            };
            let ctx = egui::Context::default();
            // The repaints egui asks for itself on its first frames are not
            // counted.
            for _ in 0..3 {
                end_frame(&ctx);
            }
            let repaints = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let counted = Arc::clone(&repaints);
            ctx.set_request_repaint_callback(move |_| {
                counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            });
            let what = format!("{case}, budget {budget}");
            let mut drawn = HashMap::new();
            let (mut needs, mut rendered, mut level) = (Vec::new(), None, None);
            let (mut tasks, mut runs, mut queued_at_most) = (0, 0, 0);
            let mut first_tasks = 0;
            if case != "missing" {
                // Image Detail's tiles first, from the files.
                for _ in 0..LOOKS {
                    let known = matches!(display_layout(source, &ctx), DisplayLayout::Known(_));
                    if known && drawn.len() < detail_keys.len() {
                        let tiles = display_tiles(source, &detail_keys, &ctx);
                        for (key, tile) in detail_keys.iter().zip(tiles) {
                            if let DisplayTile::Ready(tile) = tile {
                                drawn.insert(*key, tile);
                            }
                        }
                    }
                    first_tasks += queue.run_off_this_thread();
                    end_frame(&ctx);
                }
                assert_eq!(drawn.len(), detail_keys.len(), "{what}: drew first");
                remove_level_1(&path);
            }
            for _ in 0..LOOKS {
                look_at(
                    source,
                    &ctx,
                    &detail_keys,
                    &mut drawn,
                    &mut needs,
                    &mut rendered,
                    &mut level,
                    &what,
                );
                if matches!(cache.layout(&path), PeekedLayout::Building { .. }) {
                    assert_eq!(queue.len(), 0, "{what}: a tile asked for while Building");
                }
                let made = cache.stats().mipmaps_made;
                tasks += queue.run_off_this_thread();
                assert_eq!(cache.stats().mipmaps_made, made, "{what}: a panel made one");
                let waiting = requests.mipmaps_waiting();
                queued_at_most = queued_at_most.max(waiting.len());
                if !waiting.is_empty() {
                    assert_eq!(waiting, std::slice::from_ref(&path), "{what}");
                    runs += 1;
                    run_chunk(&cache, &requests, &Progress::none(), None);
                }
                end_frame(&ctx);
            }
            assert_eq!(drawn.len(), detail_keys.len(), "{what}: Image Detail drew");
            assert!(rendered.is_some(), "{what}: Track View drew");
            assert_eq!(
                level.map(|l| (l.width(), l.height())),
                Some((150, 100)),
                "{what}: the background drew"
            );
            assert_eq!(runs, 1, "{what}: runs of Build mipmaps");
            assert_eq!(queued_at_most, 1, "{what}: queued once");
            assert_eq!(cache.stats().mipmaps_made, 1, "{what}");
            // Before the file is removed, a header read and Image Detail's
            // decode. Then a header read for a missing mipmap, Image Detail's
            // decode, two for Track View's first tile and the tiles it needs
            // and the background's level, and for a removed file the read that
            // found it missing and the reads it cut short asked again: far
            // fewer than the thirty asks of the looks.
            assert!(first_tasks <= 2, "{what}: {first_tasks} tasks first");
            assert!(tasks <= 8, "{what}: {tasks} tasks");
            let repainted = repaints.load(std::sync::atomic::Ordering::Relaxed);
            assert!(
                repainted <= 2 * (first_tasks + tasks),
                "{what}: {repainted} repaints"
            );
        }
    }
}

/// Twenty panels, each showing its own out-of-core photograph with no
/// mipmap and asking for it on every look, Image Detail's way and Track
/// View's, add the twenty photographs to the one list *Build mipmaps*
/// drains, each once, each at the front as it is asked for, so the last
/// asked for is first; once the headers are read the looks start no task,
/// decode nothing and make no mipmap. With a budget and with one of 0.
#[test]
fn twenty_panels_add_twenty_photographs_to_the_one_list_each_once_at_the_front() {
    use super::super::build_mipmaps::tests::{cache_for, mipmap_file_count, out_of_core_workspace};
    for budget in [1 << 30, 0] {
        let (dir, paths) = out_of_core_workspace(20);
        let cache = cache_for(budget, 1);
        let (requests, queue) = PhotographRequests::queued();
        let ctx = egui::Context::default();
        let mut tasks = Vec::new();
        for look in 0..3 {
            for path in &paths {
                let source = PhotographSource {
                    cache: &cache,
                    requests: &requests,
                    path,
                };
                let _ = display_layout(source, &ctx);
                let read = read_peeked(source, &mut Vec::new(), &ctx, |_| Ok(()));
                // Loading until the header read has run, then Building.
                if look == 0 {
                    assert!(matches!(read, PeekedRead::Loading));
                } else {
                    assert!(matches!(read, PeekedRead::Building));
                }
            }
            tasks.push(queue.run_off_this_thread());
        }
        assert_eq!(tasks, [20, 0, 0], "budget {budget}: the header reads alone");
        let mut first = paths.clone();
        first.reverse();
        assert_eq!(requests.mipmaps_waiting(), first, "budget {budget}");
        let stats = cache.stats();
        assert_eq!((stats.decodes, stats.mipmaps_made), (0, 0), "{budget}");
        assert_eq!(mipmap_file_count(dir.path()), 0);
    }
}

/// An out-of-core photograph whose mipmap *Build mipmaps* cannot make (its
/// header reads, its pixels do not) is answered as unreadable once the run
/// has failed, and is not queued again while its file stays the same, under
/// a cache that keeps nothing and so remembers no failure itself; a look
/// starts only a check of the file. Once the file is fixed it is queued
/// again, and the next run makes it.
#[test]
fn a_photograph_build_mipmaps_cannot_make_is_unreadable_until_its_file_changes() {
    use super::super::build_mipmaps::run_chunk;
    let dir = tempfile::tempdir().unwrap();
    let path = out_of_core_photo(dir.path());
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
    let cache = tile_cache(0, 1 << 30);
    let (requests, queue) = PhotographRequests::queued();
    let source = PhotographSource {
        cache: &cache,
        requests: &requests,
        path: &path,
    };
    let ctx = egui::Context::default();
    assert!(matches!(
        display_layout(source, &ctx),
        DisplayLayout::Loading
    ));
    assert_eq!(queue.run_off_this_thread(), 1, "the header read");
    assert!(matches!(
        display_layout(source, &ctx),
        DisplayLayout::Building(None)
    ));
    run_chunk(&cache, &requests, &Progress::none(), None);
    for look in 0..3 {
        assert!(
            matches!(display_layout(source, &ctx), DisplayLayout::Unreadable),
            "look {look}"
        );
        assert_eq!(
            queue.run_off_this_thread(),
            1,
            "look {look}: a check of the file"
        );
        assert!(
            requests.mipmaps_waiting().is_empty(),
            "look {look}: queued again"
        );
    }
    assert_eq!(cache.stats().mipmaps_made, 0);

    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(
        display_layout(source, &ctx),
        DisplayLayout::Unreadable
    ));
    assert_eq!(queue.run_off_this_thread(), 1, "the check finds it changed");
    assert!(matches!(
        display_layout(source, &ctx),
        DisplayLayout::Building(None)
    ));
    assert_eq!(requests.mipmaps_waiting(), std::slice::from_ref(&path));
    run_chunk(&cache, &requests, &Progress::none(), None);
    assert_eq!(cache.stats().mipmaps_made, 1);
    assert!(matches!(
        display_layout(source, &ctx),
        DisplayLayout::Known(_)
    ));
}

/// A render of an out-of-core photograph whose mipmap is missing is
/// `Building` once its header is read, with a budget smaller than the
/// photograph and with one of 0: it asks for no tile, and is queued for
/// *Build mipmaps* by a read that asks, once, and not by one that only
/// looks, as the Track Table's rows do while they scroll. Once the run has
/// made the mipmap the render runs.
#[test]
fn a_render_of_a_photograph_without_a_mipmap_is_building_and_asks_for_no_tile() {
    use super::super::build_mipmaps::run_chunk;
    for budget in [20_000, 0] {
        let dir = tempfile::tempdir().unwrap();
        let path = out_of_core_photo(dir.path());
        let cache = tile_cache(budget, 1 << 30);
        let (requests, queue) = PhotographRequests::queued();
        let source = PhotographSource {
            cache: &cache,
            requests: &requests,
            path: &path,
        };
        let ctx = egui::Context::default();
        let mut needs = Vec::new();
        let read = |needs: &mut Vec<TileKey>, ask: bool| {
            let render = |_: &TiledPhotograph| Ok(());
            if ask {
                read_peeked(source, needs, &ctx, render)
            } else {
                read_peeked_without_asking(source, needs, render)
            }
        };
        assert!(matches!(read(&mut needs, true), PeekedRead::Loading));
        assert_eq!(queue.run_off_this_thread(), 1, "{budget}: the header read");

        for look in 0..3 {
            assert!(
                matches!(read(&mut needs, false), PeekedRead::Building),
                "{budget}, look {look}"
            );
        }
        assert!(requests.mipmaps_waiting().is_empty(), "{budget}: queued");
        for look in 0..3 {
            assert!(
                matches!(read(&mut needs, true), PeekedRead::Building),
                "{budget}, look {look}"
            );
            assert_eq!(queue.len(), 0, "{budget}, look {look}: a tile asked for");
        }
        assert_eq!(requests.mipmaps_waiting(), std::slice::from_ref(&path));

        run_chunk(&cache, &requests, &Progress::none(), None);
        let mut rendered = false;
        for _ in 0..4 {
            match read(&mut needs, true) {
                PeekedRead::Read(answer) => {
                    answer.unwrap();
                    rendered = true;
                    break;
                }
                PeekedRead::Loading => {
                    queue.run_off_this_thread();
                }
                PeekedRead::Building | PeekedRead::Unreadable => panic!("{budget}"),
            }
        }
        assert!(rendered, "{budget}: rendered once the mipmap was made");
    }
}

/// A `mipmap_options` whose `jpeg_quality` is out of range.
const INVALID_MIPMAP_OPTIONS: &str = r#"{"version": 2, "mipmap_options": {
    "out_of_core_long_side": 64, "jpeg_quality": 0, "full_resolution_jpeg_quality": 95}}"#;

/// Whether `e`, a workspace file's error, names the key at fault.
fn names_jpeg_quality(e: &WorkspaceConfigError) -> bool {
    e.to_string().contains("mipmap_options.jpeg_quality")
}

/// A photograph of a workspace whose `mipmap_options` is invalid is a
/// workspace file error, with the file's error naming the key, whether the
/// header read finds it or a decode of a photograph whose layout was read
/// before the file broke does, with a budget smaller than the photograph
/// and with one of 0. A render of it is unreadable, and once it is found
/// the looks start only a check of the workspace file, decode nothing and
/// ask for no repaint. Once the workspace file is fixed the check finds it
/// changed and the photograph is read again.
#[test]
fn a_photograph_its_workspace_file_refuses_is_a_workspace_file_error() {
    for budget in [20_000, 0] {
        for found_by in ["the header read", "a decode"] {
            let what = format!("{found_by}, budget {budget}");
            let dir = tempfile::tempdir().unwrap();
            let path = out_of_core_photo(dir.path());
            let workspace = dir
                .path()
                .join(sfmtool_core::workspace::WORKSPACE_FILE_NAME);
            if found_by == "the header read" {
                std::fs::write(&workspace, INVALID_MIPMAP_OPTIONS).unwrap();
            } else {
                // In core, so its tiles are decoded from the photograph.
                std::fs::write(&workspace, r#"{"version": 2}"#).unwrap();
            }
            let cache = tile_cache(budget, 1 << 30);
            let (requests, queue) = PhotographRequests::queued();
            let source = PhotographSource {
                cache: &cache,
                requests: &requests,
                path: &path,
            };
            let ctx = egui::Context::default();
            assert!(matches!(
                display_layout(source, &ctx),
                DisplayLayout::Loading
            ));
            assert_eq!(queue.run_off_this_thread(), 1, "{what}: the header read");
            if found_by == "a decode" {
                let DisplayLayout::Known(layout) = display_layout(source, &ctx) else {
                    panic!("{what}: the layout is read");
                };
                std::fs::write(&workspace, INVALID_MIPMAP_OPTIONS).unwrap();
                let keys = one_tile_a_level(&layout);
                let _ = display_tiles(source, &keys[keys.len() - 1..], &ctx);
                assert_eq!(queue.run_off_this_thread(), 1, "{what}: the decode");
            }
            for _ in 0..3 {
                end_frame(&ctx);
            }
            let repaints = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let counted = Arc::clone(&repaints);
            ctx.set_request_repaint_callback(move |_| {
                counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            });
            let decodes = cache.stats().decodes;
            for look in 0..5 {
                match display_layout(source, &ctx) {
                    DisplayLayout::WorkspaceFileError(e) => {
                        assert!(names_jpeg_quality(&e), "{what}, look {look}: {e}");
                    }
                    other => panic!("{what}, look {look}: {other:?}"),
                }
                let read = read_peeked(source, &mut Vec::new(), &ctx, |_| Ok(()));
                assert!(
                    matches!(read, PeekedRead::Unreadable),
                    "{what}, look {look}"
                );
                // A check of the workspace file, which finds it unchanged.
                assert!(queue.run_off_this_thread() <= 1, "{what}, look {look}");
                end_frame(&ctx);
            }
            assert_eq!(cache.stats().decodes, decodes, "{what}: decoded again");
            assert_eq!(
                repaints.load(std::sync::atomic::Ordering::Relaxed),
                0,
                "{what}: a repaint asked for"
            );
            assert!(requests.mipmaps_waiting().is_empty(), "{what}: queued");

            // Fixed, with a threshold that leaves the photograph in core: the
            // next look's check finds the file changed and asks for a
            // repaint, and the photograph's layout is read again.
            std::fs::write(
                &workspace,
                r#"{"version": 2, "mipmap_options": {"out_of_core_long_side": 4096,
                "jpeg_quality": 95, "full_resolution_jpeg_quality": 95}}"#,
            )
            .unwrap();
            let _ = display_layout(source, &ctx);
            assert_eq!(queue.run_off_this_thread(), 1, "{what}: the check");
            assert_eq!(
                repaints.load(std::sync::atomic::Ordering::Relaxed),
                1,
                "{what}: the check's repaint"
            );
            assert!(matches!(
                display_layout(source, &ctx),
                DisplayLayout::Loading
            ));
            assert_eq!(queue.run_off_this_thread(), 1, "{what}: the header read");
            assert!(
                matches!(display_layout(source, &ctx), DisplayLayout::Known(_)),
                "{what}: read once the file is fixed"
            );
        }
    }
}

/// A photograph queued for *Build mipmaps* whose workspace file breaks
/// before its build runs: the build is refused, and the panels then answer
/// it as a workspace file error naming the key, as when the header read
/// finds the refusal, not as an unreadable photograph; once the file is
/// fixed the next look's check finds it changed and the photograph is read
/// again and queued again. With a budget smaller than the photograph and
/// with one of 0, which keeps no record of the refusal.
#[test]
fn a_workspace_file_broken_before_a_queued_build_is_a_workspace_file_error() {
    use super::super::build_mipmaps::run_chunk;
    for budget in [20_000, 0] {
        let dir = tempfile::tempdir().unwrap();
        let path = out_of_core_photo(dir.path());
        let workspace = dir
            .path()
            .join(sfmtool_core::workspace::WORKSPACE_FILE_NAME);
        let valid = std::fs::read(&workspace).unwrap();
        let cache = tile_cache(budget, 1 << 30);
        let (requests, queue) = PhotographRequests::queued();
        let source = PhotographSource {
            cache: &cache,
            requests: &requests,
            path: &path,
        };
        let ctx = egui::Context::default();
        let _ = display_layout(source, &ctx);
        assert_eq!(queue.run_off_this_thread(), 1, "{budget}: the header read");
        assert!(matches!(
            display_layout(source, &ctx),
            DisplayLayout::Building(_)
        ));
        assert_eq!(requests.mipmaps_waiting(), std::slice::from_ref(&path));

        std::fs::write(&workspace, INVALID_MIPMAP_OPTIONS).unwrap();
        run_chunk(&cache, &requests, &Progress::none(), None);
        assert_eq!(cache.stats().mipmaps_made, 0, "{budget}");
        for look in 0..3 {
            match display_layout(source, &ctx) {
                DisplayLayout::WorkspaceFileError(e) => {
                    assert!(names_jpeg_quality(&e), "{budget}, look {look}: {e}");
                }
                other => panic!("{budget}, look {look}: {other:?}"),
            }
            // A check of the workspace file, which finds it unchanged.
            assert!(queue.run_off_this_thread() <= 1, "{budget}, look {look}");
        }
        assert!(requests.mipmaps_waiting().is_empty(), "{budget}: queued");

        std::fs::write(&workspace, &valid).unwrap();
        let _ = display_layout(source, &ctx);
        assert_eq!(queue.run_off_this_thread(), 1, "{budget}: the check");
        assert!(matches!(
            display_layout(source, &ctx),
            DisplayLayout::Loading
        ));
        assert_eq!(queue.run_off_this_thread(), 1, "{budget}: the header read");
        assert!(
            matches!(display_layout(source, &ctx), DisplayLayout::Building(_)),
            "{budget}: read again once the file is fixed"
        );
        assert_eq!(requests.mipmaps_waiting(), std::slice::from_ref(&path));
        run_chunk(&cache, &requests, &Progress::none(), None);
        assert_eq!(cache.stats().mipmaps_made, 1, "{budget}");
    }
}
