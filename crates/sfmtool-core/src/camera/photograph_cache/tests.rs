// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::AtomicBool;
use std::sync::Barrier;

use crate::progress::Event;

use super::*;

/// Bytes of a one-level RGB pyramid of `w x h`.
fn bytes_of(w: u32, h: u32) -> u64 {
    u64::from(w) * u64::from(h) * 3
}

/// Write a `w x h` PNG at `dir/name` and return its path.
fn write_png(dir: &Path, name: &str, w: u32, h: u32) -> PathBuf {
    let path = dir.join(name);
    let image = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([x as u8, y as u8, 7]));
    image.save(&path).unwrap();
    path
}

fn width(pyramid: &Option<Arc<ImageU8Pyramid>>) -> u32 {
    pyramid.as_ref().unwrap().level(0).width()
}

#[test]
fn byte_len_sums_every_level() {
    let pyramid = ImageU8Pyramid::from_image(ImageU8::from_channels(8, 4, 3), 3);
    assert_eq!(pyramid.byte_len(), 8 * 4 * 3 + 4 * 2 * 3 + 2 * 3);
}

#[test]
fn budget_from_physical_memory_and_env() {
    // A quarter of physical memory, clamped to [1 GiB, 16 GiB].
    assert_eq!(budget_from(None, Some(64 * GIB)), 16 * GIB);
    assert_eq!(budget_from(None, Some(128 * GIB)), 16 * GIB);
    assert_eq!(budget_from(None, Some(16 * GIB)), 4 * GIB);
    assert_eq!(budget_from(None, Some(2 * GIB)), GIB);
    // Unknown physical memory.
    assert_eq!(budget_from(None, None), 4 * GIB);
    // The environment variable wins, in megabytes; a bad value is ignored.
    assert_eq!(budget_from(Some("512"), Some(64 * GIB)), 512_000_000);
    assert_eq!(budget_from(Some(" 0 "), None), 0);
    assert_eq!(budget_from(Some("lots"), None), 4 * GIB);
}

#[test]
fn default_budget_is_in_range_or_overridden() {
    let budget = default_budget_bytes();
    if std::env::var(BUDGET_ENV_VAR).is_err() {
        assert!((GIB..=16 * GIB).contains(&budget), "{budget}");
    }
}

/// Every pyramid is built with the cache's depth, which `levels` reports and
/// `Debug` shows beside the stats.
#[test]
fn levels_and_debug_report_the_depth_and_the_stats() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_png(dir.path(), "a.png", 64, 48);
    let cache = PhotographCache::new(GIB, 3);
    assert_eq!(cache.levels(), 3);
    assert_eq!(cache.get(&path).unwrap().num_levels(), 3);
    let shown = format!("{cache:?}");
    assert!(
        shown.contains("levels: 3") && shown.contains("entries: 1"),
        "{shown}"
    );
}

#[test]
fn one_decode_per_path_under_contention() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_png(dir.path(), "a.png", 64, 48);
    let cache = PhotographCache::new(GIB, 3);
    let threads = 16;
    let barrier = Barrier::new(threads);
    let results: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    cache.get(&path).unwrap()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for pyramid in &results {
        assert!(Arc::ptr_eq(pyramid, &results[0]));
    }
    let stats = cache.stats();
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.hits, threads as u64 - 1);
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.bytes, results[0].byte_len() as u64);
}

#[test]
fn evicts_least_recently_used_within_budget() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 8);
    let b = write_png(dir.path(), "b.png", 8, 8);
    let c = write_png(dir.path(), "c.png", 8, 8);
    let one = bytes_of(8, 8);
    let cache = PhotographCache::new(2 * one + one / 2, 1);

    cache.get(&a).unwrap();
    cache.get(&b).unwrap();
    // Using `a` again makes `b` the least recently used.
    cache.get(&a).unwrap();
    cache.get(&c).unwrap();

    assert!(cache.peek(&a).is_some());
    assert!(cache.peek(&b).is_none());
    assert!(cache.peek(&c).is_some());
    let stats = cache.stats();
    assert_eq!(stats.entries, 2);
    assert_eq!(stats.bytes, 2 * one);
    assert!(stats.bytes <= stats.budget_bytes);
    assert_eq!(stats.misses, 3);
    assert_eq!(stats.hits, 1);
}

#[test]
fn keeps_the_entry_just_decoded_even_over_budget() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 8);
    let b = write_png(dir.path(), "b.png", 8, 8);
    let cache = PhotographCache::new(bytes_of(8, 8) / 2, 1);
    cache.get(&a).unwrap();
    assert!(cache.peek(&a).is_some());
    // The next decode pushes the older one out.
    cache.get(&b).unwrap();
    assert!(cache.peek(&a).is_none());
    assert!(cache.peek(&b).is_some());
    assert_eq!(cache.stats().bytes, bytes_of(8, 8));
}

#[test]
fn a_held_pyramid_survives_eviction() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 8);
    let b = write_png(dir.path(), "b.png", 8, 8);
    let cache = PhotographCache::new(bytes_of(8, 8), 1);
    let held = cache.get(&a).unwrap();
    cache.get(&b).unwrap();
    assert!(cache.peek(&a).is_none());
    assert_eq!(held.level(0).width(), 8);
    assert_eq!(held.level(0).get_pixel(3, 5, 0), 3);
    assert_eq!(held.level(0).get_pixel(3, 5, 1), 5);
}

#[test]
fn remembers_a_failure_until_forgotten() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing.png");
    let corrupt = dir.path().join("corrupt.png");
    std::fs::write(&corrupt, b"not a png").unwrap();
    let cache = PhotographCache::new(GIB, 1);

    assert!(cache.get(&missing).is_none());
    assert!(cache.get(&corrupt).is_none());
    assert_eq!(cache.stats().misses, 2);
    assert!(cache.get(&missing).is_none());
    assert!(cache.get(&corrupt).is_none());
    let stats = cache.stats();
    assert_eq!((stats.misses, stats.hits), (2, 2));
    assert_eq!((stats.entries, stats.bytes), (0, 0));

    cache.forget(&corrupt);
    assert!(cache.get(&corrupt).is_none());
    assert_eq!(cache.stats().misses, 3);
    cache.clear();
    assert!(cache.get(&missing).is_none());
    assert_eq!(cache.stats().misses, 4);
}

#[test]
fn a_file_that_appears_is_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("late.png");
    let cache = PhotographCache::new(GIB, 1);
    assert!(cache.get(&path).is_none());
    write_png(dir.path(), "late.png", 8, 8);
    assert_eq!(width(&cache.get(&path)), 8);
}

#[test]
fn a_rewritten_file_is_decoded_again() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_png(dir.path(), "a.png", 8, 8);
    let cache = PhotographCache::new(GIB, 1);
    let first = cache.get(&path);
    assert_eq!(width(&first), 8);
    write_png(dir.path(), "a.png", 16, 8);
    // `peek` does not look at the file, so it still has the old pixels.
    assert_eq!(width(&cache.peek(&path)), 8);
    assert_eq!(width(&cache.get(&path)), 16);
    let stats = cache.stats();
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.bytes, bytes_of(16, 8));
}

#[test]
fn forget_and_clear_release_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 8);
    let b = write_png(dir.path(), "b.png", 4, 4);
    let cache = PhotographCache::new(GIB, 1);
    cache.get(&a).unwrap();
    cache.get(&b).unwrap();
    assert_eq!(cache.stats().bytes, bytes_of(8, 8) + bytes_of(4, 4));
    cache.forget(&a);
    assert!(cache.peek(&a).is_none());
    assert_eq!(cache.stats().bytes, bytes_of(4, 4));
    cache.clear();
    let stats = cache.stats();
    assert_eq!((stats.entries, stats.bytes), (0, 0));
}

#[test]
fn get_many_keeps_order_and_reports_the_tally() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let b = write_png(dir.path(), "b.png", 12, 4);
    let c = write_png(dir.path(), "c.png", 16, 4);
    let missing = dir.path().join("missing.png");
    let cache = PhotographCache::new(GIB, 2);
    cache.get(&b).unwrap();

    let counts = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        if let Event::Count { done, total, unit } = event {
            counts.lock().unwrap().push((done, total, unit));
        }
    };
    let paths: Vec<&Path> = vec![&a, &missing, &b, &c];
    let (pyramids, tally) = cache.get_many(&paths, &Progress::to(&sink)).unwrap();

    assert_eq!(pyramids.len(), 4);
    assert_eq!(width(&pyramids[0]), 8);
    assert!(pyramids[1].is_none());
    assert_eq!(width(&pyramids[2]), 12);
    assert_eq!(width(&pyramids[3]), 16);
    assert!(Arc::ptr_eq(
        pyramids[2].as_ref().unwrap(),
        &cache.peek(&b).unwrap()
    ));
    assert_eq!(
        tally,
        GetManyTally {
            read: 2,
            reused: 1,
            unreadable: 1
        }
    );

    let mut counts = counts.into_inner().unwrap();
    counts.sort();
    assert_eq!(
        counts,
        (1..=4).map(|n| (n, Some(4), "images")).collect::<Vec<_>>()
    );

    // A second call reads nothing.
    let (_, tally) = cache.get_many(&paths, &Progress::none()).unwrap();
    assert_eq!(
        tally,
        GetManyTally {
            read: 0,
            reused: 3,
            unreadable: 1
        }
    );
}

#[test]
fn get_many_of_one_path_twice_decodes_it_once() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let cache = PhotographCache::new(GIB, 1);
    let paths: Vec<&Path> = vec![&a, &a];
    let (pyramids, tally) = cache.get_many(&paths, &Progress::none()).unwrap();
    assert!(Arc::ptr_eq(
        pyramids[0].as_ref().unwrap(),
        pyramids[1].as_ref().unwrap()
    ));
    assert_eq!((tally.read, tally.reused), (1, 1));
    assert_eq!(cache.stats().misses, 1);
}

#[test]
fn get_many_returns_cancelled() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let cache = PhotographCache::new(GIB, 1);
    let flag = AtomicBool::new(true);
    let paths: Vec<&Path> = vec![&a, &a, &a];
    let result = cache.get_many(&paths, &Progress::none().cancelled_by(&flag));
    assert!(matches!(result, Err(Cancelled)));
    assert_eq!(cache.stats().misses, 0);
}

#[test]
fn budget_zero_keeps_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let cache = PhotographCache::new(0, 1);
    let first = cache.get(&a).unwrap();
    let second = cache.get(&a).unwrap();
    assert!(!Arc::ptr_eq(&first, &second));
    assert!(cache.peek(&a).is_none());
    let (_, tally) = cache.get_many(&[a.as_path()], &Progress::none()).unwrap();
    assert_eq!(tally.read, 1);
    let stats = cache.stats();
    assert_eq!((stats.entries, stats.bytes, stats.misses), (0, 0, 3));
}

#[test]
fn peek_state_tells_a_failure_from_no_answer_yet() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let missing = dir.path().join("missing.png");
    let cache = PhotographCache::new(GIB, 1);
    assert!(matches!(cache.peek_state(&a), PeekedPhotograph::NotDecoded));
    assert!(matches!(
        cache.peek_state(&missing),
        PeekedPhotograph::NotDecoded
    ));

    let got = cache.get(&a).unwrap();
    assert!(cache.get(&missing).is_none());
    match cache.peek_state(&a) {
        PeekedPhotograph::Decoded(pyramid) => assert!(Arc::ptr_eq(&pyramid, &got)),
        _ => panic!("expected the decoded pyramid"),
    }
    assert!(matches!(
        cache.peek_state(&missing),
        PeekedPhotograph::Unreadable
    ));
    // Neither look is a hit or a miss, and neither re-reads the file.
    let stats = cache.stats();
    assert_eq!((stats.hits, stats.misses), (0, 2));

    cache.forget(&missing);
    assert!(matches!(
        cache.peek_state(&missing),
        PeekedPhotograph::NotDecoded
    ));
    // A cache that keeps nothing never has an answer to peek at.
    let none = PhotographCache::new(0, 1);
    assert_eq!(none.budget_bytes(), 0);
    none.get(&a).unwrap();
    assert!(matches!(none.peek_state(&a), PeekedPhotograph::NotDecoded));
}

#[test]
fn peek_state_does_not_wait_on_a_decode_in_flight() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let cache = PhotographCache::new(GIB, 1);
    // A slot whose decode has not finished, as `get_found` leaves it while the
    // decoding thread runs.
    cache.lock().map.insert(
        a.clone(),
        Arc::new(Slot {
            value: OnceLock::new(),
            stamp: file_stamp(&a),
            last_used: AtomicU64::new(0),
            charged: AtomicU64::new(0),
        }),
    );
    assert!(matches!(cache.peek_state(&a), PeekedPhotograph::NotDecoded));
}

#[test]
fn peek_does_not_decode() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let cache = PhotographCache::new(GIB, 1);
    assert!(cache.peek(&a).is_none());
    let stats = cache.stats();
    assert_eq!((stats.entries, stats.hits, stats.misses), (0, 0, 0));
    let got = cache.get(&a).unwrap();
    assert!(Arc::ptr_eq(&cache.peek(&a).unwrap(), &got));
    assert_eq!(cache.stats().hits, 0);
}

#[test]
fn decode_into_a_forgotten_slot_still_returns_and_charges_nothing() {
    // A slot forgotten while its decode runs: the decoding caller gets the
    // pyramid, and the byte count stays at what the map holds.
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 8);
    let cache = PhotographCache::new(GIB, 1);
    let slot = Arc::new(Slot {
        value: OnceLock::new(),
        stamp: file_stamp(&a),
        last_used: AtomicU64::new(0),
        charged: AtomicU64::new(0),
    });
    // Not in the map, as after a `forget` during the decode.
    cache.charge(&a, &slot, 192);
    assert_eq!(cache.stats().bytes, 0);
    // In the map, it is charged once.
    cache.lock().map.insert(a.clone(), Arc::clone(&slot));
    cache.charge(&a, &slot, 192);
    assert_eq!(cache.stats().bytes, 192);
    cache.forget(&a);
    assert_eq!(cache.stats().bytes, 0);
}

#[test]
fn an_inserted_pyramid_is_what_get_answers() {
    let dir = tempfile::tempdir().unwrap();
    // No file at this path: the entry stands in for one.
    let absent = dir.path().join("absent.png");
    let cache = PhotographCache::new(GIB, 1);
    let pyramid = Arc::new(ImageU8Pyramid::from_image(
        ImageU8::from_channels(8, 4, 3),
        1,
    ));
    cache.insert(&absent, Arc::clone(&pyramid));
    assert!(Arc::ptr_eq(&cache.get(&absent).unwrap(), &pyramid));
    assert!(Arc::ptr_eq(&cache.peek(&absent).unwrap(), &pyramid));
    let stats = cache.stats();
    assert_eq!((stats.entries, stats.bytes), (1, pyramid.byte_len() as u64));
    assert_eq!((stats.hits, stats.misses), (1, 0));

    // A second insert replaces the first and is charged once.
    let other = Arc::new(ImageU8Pyramid::from_image(
        ImageU8::from_channels(4, 4, 3),
        1,
    ));
    cache.insert(&absent, Arc::clone(&other));
    assert!(Arc::ptr_eq(&cache.peek(&absent).unwrap(), &other));
    assert_eq!(cache.stats().bytes, other.byte_len() as u64);

    // Over the budget, the older entry goes, as with a decode.
    let small = PhotographCache::new(bytes_of(8, 4), 1);
    let b = dir.path().join("b.png");
    small.insert(&absent, Arc::clone(&pyramid));
    small.insert(&b, Arc::clone(&pyramid));
    assert!(small.peek(&absent).is_none());
    assert!(small.peek(&b).is_some());

    // A budget of 0 keeps nothing.
    let none = PhotographCache::new(0, 1);
    none.insert(&absent, pyramid);
    assert!(none.peek(&absent).is_none());
}
