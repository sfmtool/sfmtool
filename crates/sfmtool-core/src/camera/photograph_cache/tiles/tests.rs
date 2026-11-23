// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
use std::sync::Mutex;

use super::super::tests::{
    cache_holding_its_first_decode, cache_watching_its_count, on_a_thread, small_photographs,
    unseen, write_png, HANG, SMALL,
};
use super::super::{PeekedPhotograph, PhotographCacheOptions, SlotEvent};
use super::*;

const GIB: u64 = 1 << 30;

/// A cache of `budget_bytes` cutting tiles of `tile_size` px.
fn cache_of(budget_bytes: u64, tile_size: u32) -> PhotographCache {
    PhotographCache::new(PhotographCacheOptions {
        budget_bytes,
        tile_size,
        ..Default::default()
    })
}

fn key(level: u8, col: u32, row: u32) -> TileKey {
    TileKey { level, col, row }
}

/// Every tile of every level of `layout`, level by level, row by row.
fn every_key(layout: &PyramidLayout) -> Vec<TileKey> {
    (0..layout.levels())
        .flat_map(|level| {
            let (cols, rows) = layout.tiles(level);
            (0..rows).flat_map(move |row| (0..cols).map(move |col| key(level as u8, col, row)))
        })
        .collect()
}

/// Assert that `tile` is the tile `key` of `pyramid` as Image Detail cut it
/// before the cache cut tiles: the core of `tile_size` px a side, clipped to
/// the level, with a one-pixel border of the neighbours' pixels, or the edge
/// pixel repeated at the level's edge.
fn assert_tile(pyramid: &ImageU8Pyramid, tile_size: u32, key: TileKey, tile: &PhotographTile) {
    let level = pyramid.level(usize::from(key.level));
    let (w, h) = (level.width(), level.height());
    let origin = [key.col * tile_size, key.row * tile_size];
    let size = [tile_size.min(w - origin[0]), tile_size.min(h - origin[1])];
    assert_eq!(tile.key, key);
    assert_eq!((tile.origin, tile.size), (origin, size), "{key:?}");
    let pixels = &tile.pixels;
    assert_eq!(
        (pixels.width(), pixels.height(), pixels.channels()),
        (size[0] + 2, size[1] + 2, 3),
        "{key:?}"
    );
    let clamp = |v: i64, n: u32| v.clamp(0, i64::from(n) - 1) as u32;
    for ty in 0..size[1] + 2 {
        let sy = clamp(i64::from(origin[1]) + i64::from(ty) - 1, h);
        for tx in 0..size[0] + 2 {
            let sx = clamp(i64::from(origin[0]) + i64::from(tx) - 1, w);
            for c in 0..3 {
                assert_eq!(
                    pixels.get_pixel(tx, ty, c),
                    level.get_pixel(sx, sy, c),
                    "{key:?} at {tx}, {ty}"
                );
            }
        }
    }
}

/// A photograph's layout is read from its header, and every tile, a region
/// and every level the cache hands out are the pixels of the pyramid it holds,
/// each tile with its border as Image Detail cut it: through a panel's
/// decode, a peek and a kernel's read alike. The photograph `peek` and `get`
/// answer is the one the tiles are cut from, with the same layout. A second
/// ask decodes nothing.
#[test]
fn tiles_regions_and_levels_are_the_held_pyramids_pixels() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_png(dir.path(), "p.png", 70, 50);
    let cache = cache_of(GIB, 16);
    assert_eq!(cache.layout(&path), PeekedLayout::NotKnown);
    assert!(matches!(
        cache.peek_tile(&path, key(0, 0, 0)),
        PeekedTile::NotReady
    ));

    cache.request_layout(&path);
    let PeekedLayout::Known(layout) = cache.layout(&path) else {
        panic!("the header was read");
    };
    assert_eq!(layout.levels(), 6);
    assert_eq!(layout.level_size(1), (35, 25));
    assert_eq!(layout.tiles(0), (5, 4));
    assert_eq!(layout.tile_size(), 16);
    assert!(!layout.is_out_of_core());
    assert_eq!(cache.stats().misses, 0, "a header read decodes nothing");

    let keys = every_key(&layout);
    let TryDecode::Decoded(tiles) = cache
        .try_decode_tiles(&path, &keys, &Progress::none())
        .unwrap()
    else {
        panic!("nothing else holds the slot or the build count");
    };
    assert_eq!(tiles.len(), keys.len());
    let stats = cache.stats();
    assert_eq!((stats.misses, stats.decodes, stats.hits), (1, 1, 0));
    assert_eq!(
        stats.build_reserved_bytes, 0,
        "the decode gave its share back"
    );
    let photograph = cache.peek(&path).unwrap();
    assert_eq!(photograph.layout(), &*layout);
    let pyramid = photograph.held_pyramid();
    assert_eq!((stats.entries, stats.bytes), (1, pyramid.byte_len() as u64));
    assert_eq!(pyramid.num_levels(), layout.levels());
    for (&key, tile) in keys.iter().zip(&tiles) {
        assert_tile(pyramid, 16, key, tile);
        let PeekedTile::Ready(peeked) = cache.peek_tile(&path, key) else {
            panic!("{key:?} is held");
        };
        assert_tile(pyramid, 16, key, &peeked);
        assert_tile(
            pyramid,
            16,
            key,
            &cache.get_tile(&path, key, &Progress::none()).unwrap(),
        );
    }
    for level in 0..layout.levels() {
        let whole = cache.get_level(&path, level, &Progress::none()).unwrap();
        let expected = pyramid.level(level);
        assert_eq!(
            (whole.width(), whole.height(), whole.channels()),
            (expected.width(), expected.height(), 3)
        );
        assert!(whole.data() == expected.data(), "level {level}");
    }
    let mut regions = LevelRegions::new(&layout);
    let rect = LevelRect {
        x0: 3,
        y0: 4,
        x1: 20,
        y1: 9,
    };
    regions.set_region(1, rect);
    let region = cache
        .view_region_pyramid(&path, &regions, &Progress::none())
        .unwrap();
    assert_eq!(region.level_origin(1), [3, 4]);
    assert_eq!(region.level_full_size(1), [35, 25]);
    let held = region.level(1);
    assert_eq!((held.width(), held.height()), (17, 5));
    for y in 0..5 {
        for x in 0..17 {
            for c in 0..3 {
                assert_eq!(
                    held.get_pixel(x, y, c),
                    pyramid.level(1).get_pixel(x + 3, y + 4, c)
                );
            }
        }
    }

    // A second panel's ask is answered from the pyramid held, as a hit, and
    // so is a kernel's `get`, with the photograph `peek` answered.
    let hits = cache.stats().hits;
    assert!(Arc::ptr_eq(
        &cache.get(&path, &Progress::none()).unwrap(),
        &photograph
    ));
    assert!(matches!(
        cache.try_decode_tiles(&path, &keys[..1], &Progress::none()),
        Ok(TryDecode::Decoded(_))
    ));
    let stats = cache.stats();
    assert_eq!((stats.misses, stats.decodes, stats.hits), (1, 1, hits + 2));
}

/// A level, tile or rectangle the photograph does not have is refused with
/// an error, never a panic; a panel's decode refuses such a key before it
/// reserves or decodes anything.
#[test]
fn what_the_photograph_does_not_have_is_out_of_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_png(dir.path(), "p.png", 70, 50);
    let cache = cache_of(GIB, 16);
    let out_of_range = |result: Result<TryDecode, TileError>| matches!(result, Err(TileError::OutOfRange { path: p, .. }) if p == path);
    for bad in [key(0, 5, 0), key(0, 0, 4), key(6, 0, 0), key(255, 0, 0)] {
        assert!(
            out_of_range(cache.try_decode_tiles(&path, &[bad], &Progress::none())),
            "{bad:?}"
        );
    }
    assert_eq!(cache.stats().decodes, 0);
    assert_eq!(cache.stats().build_reserved_bytes, 0);
    assert!(matches!(
        cache.try_decode_tiles(&path, &[key(0, 4, 3)], &Progress::none()),
        Ok(TryDecode::Decoded(_))
    ));

    let tile_err = |e: TileError| matches!(e, TileError::OutOfRange { .. });
    assert!(tile_err(
        cache
            .get_tile(&path, key(1, 3, 0), &Progress::none())
            .err()
            .unwrap()
    ));
    assert!(tile_err(
        cache.get_level(&path, 6, &Progress::none()).err().unwrap()
    ));
    assert!(matches!(
        cache.peek_tile(&path, key(2, 0, 1)),
        PeekedTile::Unreadable
    ));
    let PeekedLayout::Known(layout) = cache.layout(&path) else {
        panic!("the photograph is held");
    };
    let empty = LevelRegions::new(&layout);
    assert!(tile_err(
        cache
            .view_region_pyramid(&path, &empty, &Progress::none())
            .err()
            .unwrap()
    ));
    let mut past = LevelRegions::new(&layout);
    past.set_region(
        2,
        LevelRect {
            x0: 0,
            y0: 0,
            x1: 18,
            y1: 12,
        },
    );
    let e = cache
        .view_region_pyramid(&path, &past, &Progress::none())
        .err()
        .unwrap();
    assert!(tile_err(e.clone()), "{e}");
    assert!(e.to_string().contains("p.png"), "{e}");
}

/// A photograph that cannot be read is answered as unreadable by every
/// request, naming the file, and a panel's decode remembers the failure in
/// its slot, so later peeks see it without asking again. `forget` and
/// `clear` drop a layout as they drop an entry.
#[test]
fn an_unreadable_photograph_is_answered_and_remembered() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing.png");
    let cache = cache_of(GIB, 16);
    cache.request_layout(&missing);
    match cache.layout(&missing) {
        PeekedLayout::Unreadable(why) => assert!(why.contains("missing.png"), "{why}"),
        other => panic!("{other:?}"),
    }
    let unreadable =
        |e: TileError| matches!(e, TileError::Unreadable { path, .. } if path == missing);
    assert!(unreadable(
        cache
            .try_decode_tiles(&missing, &[key(0, 0, 0)], &Progress::none())
            .err()
            .unwrap()
    ));
    assert!(matches!(
        cache.peek_tile(&missing, key(0, 0, 0)),
        PeekedTile::Unreadable
    ));
    let stats = cache.stats();
    assert_eq!((stats.misses, stats.decodes), (1, 0));
    // Answered from the remembered failure, with no second read.
    assert!(unreadable(
        cache
            .try_decode_tiles(&missing, &[key(0, 0, 0)], &Progress::none())
            .err()
            .unwrap()
    ));
    assert_eq!(cache.stats().misses, 1);
    assert!(unreadable(
        cache
            .get_tile(&missing, key(0, 0, 0), &Progress::none())
            .err()
            .unwrap()
    ));
    assert!(unreadable(
        cache
            .get_level(&missing, 0, &Progress::none())
            .err()
            .unwrap()
    ));

    // With a budget of 0 nothing is remembered, and the answer is the same.
    let none = cache_of(0, 16);
    assert!(unreadable(
        none.try_decode_tiles(&missing, &[key(0, 0, 0)], &Progress::none())
            .err()
            .unwrap()
    ));
    assert!(matches!(
        none.peek_tile(&missing, key(0, 0, 0)),
        PeekedTile::NotReady
    ));

    let path = write_png(dir.path(), "p.png", 20, 10);
    cache.request_layout(&path);
    assert!(matches!(cache.layout(&path), PeekedLayout::Known(_)));
    cache.forget(&path);
    assert_eq!(cache.layout(&path), PeekedLayout::NotKnown);
    cache.request_layout(&path);
    cache.clear();
    assert_eq!(cache.layout(&path), PeekedLayout::NotKnown);
}

/// The layout of a photograph the cache holds is the held pyramid's, so a
/// file replaced by one of another size after its header was read is laid
/// out as the pixels drawn from it once it is decoded again. A cache of
/// budget 0 keeps the layouts it reads.
#[test]
fn the_layout_follows_the_photograph_held() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_png(dir.path(), "p.png", 20, 10);
    let cache = cache_of(GIB, 16);
    cache.request_layout(&path);
    let width = |cache: &PhotographCache| match cache.layout(&path) {
        PeekedLayout::Known(layout) => layout.level_size(0).0,
        other => panic!("{other:?}"),
    };
    assert_eq!(width(&cache), 20);
    std::fs::remove_file(&path).unwrap();
    write_png(dir.path(), "p.png", 40, 10);
    assert_eq!(width(&cache), 20, "the header read before");
    cache.get(&path, &Progress::none()).unwrap();
    assert_eq!(width(&cache), 40);

    let none = cache_of(0, 16);
    none.request_layout(&path);
    assert_eq!(width(&none), 40);
}

/// A held photograph's layout is answered as the same `Arc` on every ask
/// while the held pyramid's size stays the same, so a panel asking every
/// frame allocates nothing, and as a new one once a pyramid of another size
/// is held at the path.
#[test]
fn a_held_photographs_layout_is_kept_and_answered_again() {
    let path = std::path::PathBuf::from("/nowhere/held.png");
    let cache = cache_of(GIB, 16);
    let pyramid = |w: u32| {
        let image = ImageU8::new(w, 10, 3, vec![9; (w * 10 * 3) as usize]);
        Arc::new(ImageU8Pyramid::from_image(
            image,
            ImageU8Pyramid::full_levels(w, 10),
        ))
    };
    let known = |cache: &PhotographCache| match cache.layout(&path) {
        PeekedLayout::Known(layout) => layout,
        other => panic!("{other:?}"),
    };
    cache.insert(&path, pyramid(20));
    let first = known(&cache);
    assert!(Arc::ptr_eq(&first, &known(&cache)), "the same layout again");
    cache.insert(&path, pyramid(40));
    let second = known(&cache);
    assert_eq!(second.level_size(0), (40, 10));
    assert!(Arc::ptr_eq(&second, &known(&cache)));
}

/// A panel's decode whose `try_reserve_build` is refused, the build budget
/// used up by another caller's reservation, decodes nothing and reserves
/// nothing; asked again once that reservation is given back, it decodes.
/// With a budget and with a cache that keeps nothing.
#[test]
fn a_refused_panel_decode_decodes_nothing_until_the_budget_has_room() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_png(dir.path(), "p.png", 16, 16);
    for budget_bytes in [GIB, 0] {
        let cache = PhotographCache::new(PhotographCacheOptions {
            budget_bytes,
            build_budget_bytes: SMALL,
            tile_size: 16,
        });
        let other = cache.reserve_builds(1, 1);
        for _ in 0..3 {
            assert!(matches!(
                cache.try_decode_tiles(&path, &[key(0, 0, 0)], &Progress::none()),
                Ok(TryDecode::Busy)
            ));
        }
        let stats = cache.stats();
        assert_eq!((stats.misses, stats.decodes), (0, 0), "{budget_bytes}");
        assert_eq!(stats.build_reserved_bytes, 1);
        drop(other);
        assert!(matches!(
            cache.try_decode_tiles(&path, &[key(0, 0, 0)], &Progress::none()),
            Ok(TryDecode::Decoded(tiles)) if tiles.len() == 1
        ));
        let stats = cache.stats();
        assert_eq!((stats.misses, stats.decodes), (1, 1), "{budget_bytes}");
        assert_eq!(stats.build_reserved_bytes, 0);
    }
    // A build budget of 0 refuses a panel decode while another decode is
    // reserved, and grants it alone once the count is idle.
    let one_at_a_time = PhotographCache::new(PhotographCacheOptions {
        build_budget_bytes: 0,
        ..PhotographCacheOptions::default()
    });
    let other = one_at_a_time.reserve_builds(1, 1);
    assert!(matches!(
        one_at_a_time.try_decode_tiles(&path, &[], &Progress::none()),
        Ok(TryDecode::Busy)
    ));
    assert_eq!(one_at_a_time.stats().decodes, 0);
    drop(other);
    assert!(matches!(
        one_at_a_time.try_decode_tiles(&path, &[], &Progress::none()),
        Ok(TryDecode::Decoded(_))
    ));
    assert_eq!(one_at_a_time.stats().decodes, 1);
}

/// A panel's decode of a photograph another caller is decoding answers
/// `Busy` at once, reserving nothing and never waiting on the slot, while
/// that decode is held in progress; once it ends, the panel's next ask is
/// answered from it.
#[test]
fn a_panel_decode_never_waits_on_a_slot_being_made() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_png(dir.path(), "p.png", 16, 16);
    let held = cache_holding_its_first_decode();
    let maker = {
        let (cache, path) = (Arc::clone(&held.cache), path.clone());
        on_a_thread(move || cache.get(&path, &Progress::none()).map(|_| ()))
    };
    let reserved = held.expect(SlotEvent::DecodeStarted);
    let panel = {
        let (cache, path) = (Arc::clone(&held.cache), path.clone());
        on_a_thread(move || {
            let busy = matches!(
                cache.try_decode_tiles(&path, &[key(0, 0, 0)], &Progress::none()),
                Ok(TryDecode::Busy)
            );
            (busy, cache.stats().build_reserved_bytes)
        })
    };
    let (busy, during) = panel.recv_timeout(HANG).expect("the panel's decode waited");
    assert!(busy);
    assert_eq!(during, reserved, "the panel reserved nothing");
    assert!(
        held.events.try_recv().is_err(),
        "the panel's decode started no wait and no decode"
    );
    held.release.send(()).unwrap();
    maker.recv_timeout(HANG).unwrap().unwrap();
    assert!(matches!(
        held.cache
            .try_decode_tiles(&path, &[key(0, 0, 0)], &Progress::none()),
        Ok(TryDecode::Decoded(_))
    ));
    assert_eq!(held.decodes.load(Ordering::SeqCst), 1);
}

/// A panel's decode that has its reservation and then finds the slot taken
/// by another caller gives the reservation back and answers `Busy`; one that
/// finds the photograph made by then answers its tiles. Neither decodes.
#[test]
fn a_panel_decode_that_loses_the_slot_gives_its_reservation_back() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_png(dir.path(), "p.png", 16, 16);
    let made = Arc::new(TiledPhotograph::from_image(ImageU8::new(
        16,
        16,
        3,
        vec![9; 16 * 16 * 3],
    )));
    for (taken, expect_busy) in [
        (SlotState::Making, true),
        (SlotState::Made(Ok(made)), false),
    ] {
        let reserved_then = Arc::new(AtomicU64::new(0));
        let taken = Mutex::new(Some(taken));
        let mut cache = cache_of(GIB, 16);
        let seen = Arc::clone(&reserved_then);
        let watched = path.clone();
        cache.on_slot_event = Some(Box::new(move |cache, event| {
            if event == SlotEvent::PanelReserved {
                seen.store(
                    cache.build_reserved.load(Ordering::SeqCst),
                    Ordering::SeqCst,
                );
                let slot = cache.lock().map.get(&watched).map(Arc::clone).unwrap();
                *slot.lock() = taken.lock().unwrap().take().unwrap();
            }
        }));
        let answer = cache.try_decode_tiles(&path, &[key(0, 0, 0)], &Progress::none());
        assert_eq!(
            reserved_then.load(Ordering::SeqCst),
            16 * 16 * 5,
            "reserved before the claim"
        );
        assert_eq!(matches!(answer, Ok(TryDecode::Busy)), expect_busy);
        assert_eq!(matches!(answer, Ok(TryDecode::Decoded(_))), !expect_busy);
        let stats = cache.stats();
        assert_eq!((stats.decodes, stats.build_reserved_bytes), (0, 0));
        assert_eq!(
            stats.hits,
            u64::from(!expect_busy),
            "an answer from the slot is a hit"
        );
    }
}

/// Panel decodes from eight threads at once over 20 photographs never take
/// the build count past the build budget, set to three decodes, whatever
/// order the threads run in; with a budget and with a cache that keeps
/// nothing.
#[test]
fn panel_decodes_never_take_the_count_past_the_build_budget() {
    let dir = tempfile::tempdir().unwrap();
    let photographs = Arc::new(small_photographs(dir.path(), 20));
    for budget_bytes in [GIB, 0] {
        let seen = unseen();
        let cache = Arc::new(cache_watching_its_count(
            PhotographCacheOptions {
                budget_bytes,
                build_budget_bytes: 3 * SMALL,
                tile_size: 16,
            },
            &seen,
        ));
        let decoded = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (cache, photographs, decoded) = (
                    Arc::clone(&cache),
                    Arc::clone(&photographs),
                    Arc::clone(&decoded),
                );
                on_a_thread(move || {
                    for path in photographs.iter() {
                        // Asked again on every refusal, as a panel asks on
                        // later frames.
                        while let Ok(TryDecode::Busy) =
                            cache.try_decode_tiles(path, &[key(0, 0, 0)], &Progress::none())
                        {
                            std::thread::yield_now();
                        }
                        decoded.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread
                .recv_timeout(HANG)
                .expect("a panel decode never ended");
        }
        assert_eq!(decoded.load(Ordering::SeqCst), 8 * 20);
        let stats = cache.stats();
        assert!(stats.decodes >= 20, "{stats:?}");
        assert!(
            seen.1.load(Ordering::SeqCst) <= 3 * SMALL,
            "{} reserved, past the build budget",
            seen.1.load(Ordering::SeqCst)
        );
        assert!(
            seen.0.load(Ordering::SeqCst) >= SMALL,
            "every decode reserved"
        );
        assert_eq!(stats.build_reserved_bytes, 0);
    }
}

/// A cache of `budget_bytes` whose every decode, once started, sets `flag`,
/// so a call whose `Progress` is cancelled by it is cancelled mid-decode.
fn cache_cancelling(budget_bytes: u64, flag: &Arc<AtomicBool>) -> PhotographCache {
    let mut cache = cache_of(budget_bytes, 16);
    let flag = Arc::clone(flag);
    cache.on_slot_event = Some(Box::new(move |_, event| {
        if event == SlotEvent::DecodeStarted {
            flag.store(true, Ordering::SeqCst);
        }
    }));
    cache
}

/// One of the calls that decode a whole photograph on a miss.
type DecodingCall = fn(&PhotographCache, &Path, &Progress<'_>) -> Result<(), TileError>;

/// Each call that decodes a whole photograph on a miss, cancelled during its
/// decode, answers `Cancelled`, gives its reservation back and remembers
/// nothing, so the next call decodes; with a budget and with a cache that
/// keeps nothing.
#[test]
fn a_cancelled_decode_answers_cancelled_and_leaves_nothing_behind() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_png(dir.path(), "p.png", 16, 16);
    let calls: [(&str, DecodingCall); 4] = [
        ("try_decode_tiles", |cache, path, progress| {
            cache
                .try_decode_tiles(path, &[key(0, 0, 0)], progress)
                .map(|_| ())
        }),
        ("get_tile", |cache, path, progress| {
            cache.get_tile(path, key(0, 0, 0), progress).map(|_| ())
        }),
        ("get_level", |cache, path, progress| {
            cache.get_level(path, 1, progress).map(|_| ())
        }),
        ("view_region_pyramid", |cache, path, progress| {
            let layout = PyramidLayout::in_memory(16, 16, 5, 16);
            let mut regions = LevelRegions::new(&layout);
            regions.set_region(
                0,
                LevelRect {
                    x0: 0,
                    y0: 0,
                    x1: 4,
                    y1: 4,
                },
            );
            cache
                .view_region_pyramid(path, &regions, progress)
                .map(|_| ())
        }),
    ];
    for budget_bytes in [GIB, 0] {
        for (name, call) in calls {
            let flag = Arc::new(AtomicBool::new(false));
            let cache = cache_cancelling(budget_bytes, &flag);
            let cancelled = Progress::none().cancelled_by(&flag);
            assert_eq!(
                call(&cache, &path, &cancelled),
                Err(TileError::Cancelled),
                "{name}, budget {budget_bytes}"
            );
            let stats = cache.stats();
            assert_eq!(stats.decodes, 1, "{name}");
            assert_eq!(stats.build_reserved_bytes, 0, "{name} gave its share back");
            assert!(
                matches!(cache.peek_state(&path), PeekedPhotograph::NotDecoded),
                "{name} remembered nothing"
            );
            call(&cache, &path, &Progress::none()).unwrap();
            assert_eq!(cache.stats().decodes, 2, "{name} decoded again");
            assert_eq!(cache.stats().build_reserved_bytes, 0);
        }
    }
    assert_eq!(TileError::Cancelled.to_string(), Cancelled.to_string());
}
