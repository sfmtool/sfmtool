// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use sfmtool_core::camera::image::ImageU8;
use sfmtool_core::camera::sampler::{read_source_rect, LevelRect, RegionPool};

use super::super::TaskQueue;

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

/// Wait for the decode of `path` that `requests` started to end.
fn wait_for(requests: &PhotographRequests, path: &Path) {
    let start = Instant::now();
    while requests.is_in_flight(path) {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the decode never ended"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_miss_is_decoded_off_the_calling_thread_and_found_on_a_later_look() {
    let dir = tempfile::tempdir().unwrap();
    let recon = demo_in(dir.path());
    let cache = Arc::new(PhotographCache::new(
        sfmtool_core::camera::PhotographCacheOptions {
            budget_bytes: 1 << 30,
            ..Default::default()
        },
    ));
    let requests = PhotographRequests::default();
    let ctx = egui::Context::default();
    let path = photograph_path(&recon, 0).unwrap();

    // Asked twice, perhaps before it lands: one decode either way.
    for _ in 0..2 {
        assert!(
            !matches!(
                display_photograph(&cache, &requests, &recon, 0, &ctx),
                DisplayPhotograph::Unreadable
            ),
            "the photograph is on disk"
        );
    }
    wait_for(&requests, &path);
    match display_photograph(&cache, &requests, &recon, 0, &ctx) {
        DisplayPhotograph::Decoded(pyramid) => assert_eq!(pyramid.level(0).width(), 6),
        _ => panic!("the decode landed in the cache"),
    }
    assert_eq!(cache.stats().misses, 1, "one decode for every look");
}

#[test]
fn a_file_that_cannot_be_read_ends_as_unreadable_rather_than_decoding_forever() {
    let dir = tempfile::tempdir().unwrap();
    let recon = demo_in(dir.path());
    let cache = Arc::new(PhotographCache::new(
        sfmtool_core::camera::PhotographCacheOptions {
            budget_bytes: 1 << 30,
            ..Default::default()
        },
    ));
    let requests = PhotographRequests::default();
    let ctx = egui::Context::default();
    let path = photograph_path(&recon, 1).unwrap();

    assert!(matches!(
        display_photograph(&cache, &requests, &recon, 1, &ctx),
        DisplayPhotograph::Decoding | DisplayPhotograph::Unreadable
    ));
    wait_for(&requests, &path);
    assert!(matches!(
        display_photograph(&cache, &requests, &recon, 1, &ctx),
        DisplayPhotograph::Unreadable
    ));
    assert!(!requests.is_in_flight(&path), "no second request");
    assert_eq!(cache.stats().misses, 1);

    // An index past the image set has no file to ask for.
    assert!(matches!(
        display_photograph(&cache, &requests, &recon, recon.image_count(), &ctx),
        DisplayPhotograph::Unreadable
    ));
}

#[test]
fn a_cached_photograph_is_answered_without_a_request() {
    let dir = tempfile::tempdir().unwrap();
    let recon = demo_in(dir.path());
    let cache = Arc::new(PhotographCache::new(
        sfmtool_core::camera::PhotographCacheOptions {
            budget_bytes: 1 << 30,
            ..Default::default()
        },
    ));
    let path = photograph_path(&recon, 1).unwrap();
    let pyramid = Arc::new(ImageU8Pyramid::from_image(
        ImageU8::new(2, 2, 3, vec![9; 12]),
        ImageU8Pyramid::full_levels(2, 2),
    ));
    cache.insert(&path, Arc::clone(&pyramid));
    let requests = PhotographRequests::default();
    match display_photograph(&cache, &requests, &recon, 1, &egui::Context::default()) {
        DisplayPhotograph::Decoded(found) => assert!(Arc::ptr_eq(&found, &pyramid)),
        _ => panic!("the inserted pyramid"),
    }
    assert!(!requests.is_in_flight(&path));
}

/// A cache that keeps nothing would never show a background decode to the
/// next look, so the photograph is decoded where it is asked for.
#[test]
fn a_cache_that_keeps_nothing_decodes_on_the_calling_thread() {
    let dir = tempfile::tempdir().unwrap();
    let recon = demo_in(dir.path());
    let cache = Arc::new(PhotographCache::new(
        sfmtool_core::camera::PhotographCacheOptions {
            budget_bytes: 0,
            ..Default::default()
        },
    ));
    let requests = PhotographRequests::default();
    let ctx = egui::Context::default();
    assert!(matches!(
        display_photograph(&cache, &requests, &recon, 0, &ctx),
        DisplayPhotograph::Decoded(_)
    ));
    assert!(matches!(
        display_photograph(&cache, &requests, &recon, 1, &ctx),
        DisplayPhotograph::Unreadable
    ));
    assert!(!requests.is_in_flight(&photograph_path(&recon, 0).unwrap()));
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
/// budget and with a cache that keeps nothing, and the tiles a decode answers
/// are handed over on a later look.
#[test]
fn the_layout_and_the_tiles_are_read_once_off_the_calling_thread() {
    for budget in [1 << 30, 0] {
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
/// nothing, and is the level of the pyramid the background drew before,
/// which `display_photograph` handed it. The task makes one call to the
/// cache, `try_decode_tiles`: with a budget, a level read of a photograph the
/// cache holds counts one hit, which a second read of the cache, such as a
/// `get_level` that could wait on the slot, would make two.
#[test]
fn the_level_is_read_once_off_the_calling_thread_and_is_the_pyramid_s_level() {
    let dir = tempfile::tempdir().unwrap();
    let path = level_photo(dir.path(), "level.png");
    let before = PhotographCache::new(sfmtool_core::camera::PhotographCacheOptions {
        budget_bytes: 0,
        ..Default::default()
    })
    .get(&path, &Progress::none())
    .unwrap();
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
    read_source_rect(src, pool, 0, rect).map(|read| read.image().data().to_vec())
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
        Answered::Tiles(
            cols.map(|col| {
                let tile = PhotographTile {
                    key: key(col),
                    origin: [col, 0],
                    size: [1, 1],
                    pixels: ImageU8::new(3, 3, 3, vec![0; 27]),
                };
                (key(col), Arc::new(tile))
            })
            .collect(),
        )
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
        display_tile_set(source, &[tile(0)], &ctx),
        DisplayTileSet::Loading
    ));
    assert_eq!(queue.run_off_this_thread(), 1);
    assert_eq!(requests.kept_tiles(), 1);
    // This render needs it and a second.
    let set = [tile(0), tile(1)];
    assert!(matches!(
        display_tile_set(source, &set, &ctx),
        DisplayTileSet::Loading
    ));
    assert_eq!(requests.kept_tiles(), 1, "the first tile is left kept");
    assert_eq!(queue.run_off_this_thread(), 1, "a decode of the second");
    let DisplayTileSet::Ready(tiles) = display_tile_set(source, &set, &ctx) else {
        panic!("both tiles are answered");
    };
    assert_eq!(tiles.iter().map(|t| t.key).collect::<Vec<_>>(), set);
    assert_eq!(requests.kept_tiles(), 0, "taken together");
    assert_eq!(cache.stats().decodes, 2);
}
