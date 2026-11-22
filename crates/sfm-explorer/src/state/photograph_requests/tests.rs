// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use sfmtool_core::camera::image::ImageU8;

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
/// asked for again, under a cache that keeps nothing and so remembers no
/// failure itself.
#[test]
fn a_failed_decode_is_answered_as_unreadable_and_not_asked_again() {
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
    assert_eq!(queue.len(), 0, "a failure is not asked for again");
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
