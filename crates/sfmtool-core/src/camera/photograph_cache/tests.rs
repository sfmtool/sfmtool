// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::AtomicBool;
use std::sync::Barrier;

use crate::progress::Event;

use super::*;

/// Bytes of the RGB pyramid the cache builds for a `w x h` photograph: every
/// level of `full_levels` depth, each half the size of the one before.
fn bytes_of(w: u32, h: u32) -> u64 {
    let (mut w, mut h) = (u64::from(w), u64::from(h));
    let mut bytes = w * h * 3;
    for _ in 1..ImageU8Pyramid::full_levels(w as u32, h as u32) {
        if w < 2 || h < 2 {
            break;
        }
        (w, h) = (w / 2, h / 2);
        bytes += w * h * 3;
    }
    bytes
}

/// Write a `w x h` PNG at `dir/name` and return its path.
fn write_png(dir: &Path, name: &str, w: u32, h: u32) -> PathBuf {
    let path = dir.join(name);
    let image = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([x as u8, y as u8, 7]));
    image.save(&path).unwrap();
    path
}

/// Options with the default build budget and `budget_bytes`.
fn with_budget(budget_bytes: u64) -> PhotographCacheOptions {
    PhotographCacheOptions {
        budget_bytes,
        ..Default::default()
    }
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
fn build_budget_from_physical_memory() {
    assert_eq!(build_budget_from(Some(64 * GIB)), 16 * GIB);
    assert_eq!(build_budget_from(Some(16 * GIB)), 4 * GIB);
    assert_eq!(build_budget_from(None), 4 * GIB);
    // A header claiming any size gives an estimate rather than an overflow.
    assert_eq!(whole_decode_bytes(4, 2), 40);
    assert_eq!(whole_decode_bytes(u32::MAX, u32::MAX), u64::MAX);
    let options = PhotographCacheOptions::default();
    assert_eq!(options.build_budget_bytes, default_build_budget_bytes());
    assert_eq!(options.tile_size, PHOTOGRAPH_TILE_SIZE);
}

#[test]
fn default_budget_is_in_range_or_overridden() {
    let budget = default_budget_bytes();
    if std::env::var(BUDGET_ENV_VAR).is_err() {
        assert!((GIB..=16 * GIB).contains(&budget), "{budget}");
    }
}

/// Each pyramid's depth is `full_levels` of its own photograph's size, down to
/// one or two pixels on the short side, so two photographs in one cache can
/// have different depths. `Debug` shows the stats.
#[test]
fn each_entry_has_the_depth_of_its_own_size() {
    let dir = tempfile::tempdir().unwrap();
    let small = write_png(dir.path(), "small.png", 64, 48);
    let tall = write_png(dir.path(), "tall.png", 270, 480);
    let cache = PhotographCache::new(with_budget(GIB));
    let small = cache.get(&small, &Progress::none()).unwrap();
    let tall = cache.get(&tall, &Progress::none()).unwrap();
    assert_eq!(small.num_levels(), 6);
    assert_eq!(tall.num_levels(), 9);
    // The top level of each is one or two pixels on the short side.
    assert_eq!(small.level(5).width(), 2);
    assert_eq!(small.level(5).height(), 1);
    assert_eq!(tall.level(8).width(), 1);
    assert_eq!(tall.level(8).height(), 1);
    let shown = format!("{cache:?}");
    assert!(shown.contains("entries: 2"), "{shown}");
}

/// A view that shrinks a photograph 64 times asks the kernels for level 6.
/// Both `BilinearMip` and `Anisotropic` read it from the cache's pyramid, which
/// reaches it; a pyramid of 6 levels would clamp them to level 5.
///
/// The photograph is a checkerboard of 32 px squares. Level 5 resolves it, one
/// square to a pixel, so it reads 0 or 255; level 6 averages two squares of
/// each colour into every pixel, so it reads 127 or 128. Each output pixel
/// samples the centre of a level-5 pixel, 64 photograph pixels from the next.
#[test]
fn a_view_shrunk_64_times_reads_level_6() {
    use crate::camera::remap::{remap_aniso_with_pyramid, remap_bilinear_mip};
    use crate::camera::WarpMap;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("checker.png");
    image::GrayImage::from_fn(512, 512, |x, y| {
        image::Luma([if (x / 32 + y / 32) % 2 == 0 { 0 } else { 255 }])
    })
    .save(&path)
    .unwrap();
    let pyramid = PhotographCache::new(with_budget(0))
        .get(&path, &Progress::none())
        .unwrap();
    assert!(pyramid.num_levels() > 6);

    let side = 8u32;
    let mut data = Vec::with_capacity(2 * (side * side) as usize);
    for row in 0..side {
        for col in 0..side {
            data.push(64.0 * col as f32 + 16.0);
            data.push(64.0 * row as f32 + 16.0);
        }
    }
    let mut map = WarpMap::new(side, side, data);
    map.compute_svd();

    for (name, out) in [
        ("BilinearMip", remap_bilinear_mip(&pyramid, &map)),
        ("Anisotropic", remap_aniso_with_pyramid(&pyramid, &map, 16)),
    ] {
        for row in 0..side {
            for col in 0..side {
                let value = out.get_pixel(col, row, 0);
                assert!(
                    (127..=128).contains(&value),
                    "{name} ({col}, {row}) read {value}, not level 6's grey"
                );
            }
        }
    }
}

#[test]
fn one_decode_per_path_under_contention() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_png(dir.path(), "a.png", 64, 48);
    let cache = PhotographCache::new(with_budget(GIB));
    let threads = 16;
    let barrier = Barrier::new(threads);
    let results: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    cache.get(&path, &Progress::none()).unwrap()
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
    let cache = PhotographCache::new(with_budget(2 * one + one / 2));

    cache.get(&a, &Progress::none()).unwrap();
    cache.get(&b, &Progress::none()).unwrap();
    // Using `a` again makes `b` the least recently used.
    cache.get(&a, &Progress::none()).unwrap();
    cache.get(&c, &Progress::none()).unwrap();

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
    let cache = PhotographCache::new(with_budget(bytes_of(8, 8) / 2));
    cache.get(&a, &Progress::none()).unwrap();
    assert!(cache.peek(&a).is_some());
    // The next decode pushes the older one out.
    cache.get(&b, &Progress::none()).unwrap();
    assert!(cache.peek(&a).is_none());
    assert!(cache.peek(&b).is_some());
    assert_eq!(cache.stats().bytes, bytes_of(8, 8));
}

#[test]
fn a_held_pyramid_survives_eviction() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 8);
    let b = write_png(dir.path(), "b.png", 8, 8);
    let cache = PhotographCache::new(with_budget(bytes_of(8, 8)));
    let held = cache.get(&a, &Progress::none()).unwrap();
    cache.get(&b, &Progress::none()).unwrap();
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
    let cache = PhotographCache::new(with_budget(GIB));

    assert_eq!(
        cache.get(&missing, &Progress::none()).err().unwrap(),
        GetError::NotFound(missing.clone())
    );
    assert!(matches!(
        cache.get(&corrupt, &Progress::none()).err().unwrap(),
        GetError::Unreadable(why) if why.contains("corrupt.png")
    ));
    let stats = cache.stats();
    // Neither header can be read, so neither is decoded.
    assert_eq!((stats.misses, stats.decodes), (2, 0));
    assert!(cache.get(&missing, &Progress::none()).is_err());
    assert!(cache.get(&corrupt, &Progress::none()).is_err());
    let stats = cache.stats();
    assert_eq!((stats.misses, stats.hits), (2, 2));
    assert_eq!((stats.entries, stats.bytes), (0, 0));

    cache.forget(&corrupt);
    assert!(cache.get(&corrupt, &Progress::none()).is_err());
    assert_eq!(cache.stats().misses, 3);
    cache.clear();
    assert!(cache.get(&missing, &Progress::none()).is_err());
    assert_eq!(cache.stats().misses, 4);
}

#[test]
fn a_file_that_appears_is_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("late.png");
    let cache = PhotographCache::new(with_budget(GIB));
    assert!(cache.get(&path, &Progress::none()).is_err());
    write_png(dir.path(), "late.png", 8, 8);
    assert_eq!(width(&cache.get(&path, &Progress::none()).ok()), 8);
}

#[test]
fn a_rewritten_file_is_decoded_again() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_png(dir.path(), "a.png", 8, 8);
    let cache = PhotographCache::new(with_budget(GIB));
    let first = cache.get(&path, &Progress::none()).ok();
    assert_eq!(width(&first), 8);
    write_png(dir.path(), "a.png", 16, 8);
    // `peek` does not look at the file, so it still has the old pixels.
    assert_eq!(width(&cache.peek(&path)), 8);
    assert_eq!(width(&cache.get(&path, &Progress::none()).ok()), 16);
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
    let cache = PhotographCache::new(with_budget(GIB));
    cache.get(&a, &Progress::none()).unwrap();
    cache.get(&b, &Progress::none()).unwrap();
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
    let cache = PhotographCache::new(with_budget(GIB));
    cache.get(&b, &Progress::none()).unwrap();

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
    let cache = PhotographCache::new(with_budget(GIB));
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
    let cache = PhotographCache::new(with_budget(GIB));
    let flag = AtomicBool::new(true);
    let paths: Vec<&Path> = vec![&a, &a, &a];
    let result = cache.get_many(&paths, &Progress::none().cancelled_by(&flag));
    assert!(matches!(result, Err(Cancelled)));
    assert_eq!(cache.stats().misses, 0);
}

/// How long a test waits for a thread it started before it fails: far longer
/// than any call here takes, so only a caller that never returns reaches it.
const HANG: Duration = Duration::from_secs(20);

/// Run `call` on a thread of its own and hand back the receiver of its
/// answer, so the test can wait for it with [`HANG`] as its limit rather
/// than hang with a caller that never returns.
fn on_a_thread<T: Send + 'static>(
    call: impl FnOnce() -> T + Send + 'static,
) -> std::sync::mpsc::Receiver<T> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(call());
    });
    rx
}

/// A cache whose first decode, once started, waits until the test sends on
/// `release`. `events` receives every [`SlotEvent`] as it happens, with the
/// bytes reserved on the build count at that moment, and `decodes` counts the
/// decodes started.
struct HeldDecode {
    cache: Arc<PhotographCache>,
    events: std::sync::mpsc::Receiver<(SlotEvent, u64)>,
    release: std::sync::mpsc::Sender<()>,
    decodes: Arc<AtomicUsize>,
}

fn cache_holding_its_first_decode() -> HeldDecode {
    let (event_tx, events) = std::sync::mpsc::channel();
    let (release, release_rx) = std::sync::mpsc::channel::<()>();
    let release_rx = Mutex::new(release_rx);
    let decodes = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&decodes);
    let mut cache = PhotographCache::new(with_budget(GIB));
    cache.on_slot_event = Some(Box::new(move |cache, event| {
        let first =
            event == SlotEvent::DecodeStarted && counted.fetch_add(1, Ordering::SeqCst) == 0;
        let _ = event_tx.send((event, cache.build_reserved.load(Ordering::SeqCst)));
        if first {
            let _ = release_rx.lock().unwrap().recv_timeout(HANG);
        }
    }));
    HeldDecode {
        cache: Arc::new(cache),
        events,
        release,
        decodes,
    }
}

impl HeldDecode {
    /// Wait for the next event, which must be `expected`, and answer the
    /// bytes reserved on the build count when it happened.
    fn expect(&self, expected: SlotEvent) -> u64 {
        let (event, reserved) = self.events.recv_timeout(HANG).expect("no slot event came");
        assert_eq!(event, expected);
        reserved
    }
}

/// A caller cancelled during its decode stops before building the pyramid,
/// remembers nothing and leaves the slot empty, and the caller waiting on the
/// slot decodes in its place rather than wait for ever.
#[test]
fn a_cancelled_maker_empties_its_slot_and_a_waiter_decodes_in_its_place() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let held = cache_holding_its_first_decode();
    let cancel = Arc::new(AtomicBool::new(false));

    let maker = on_a_thread({
        let (cache, a, cancel) = (Arc::clone(&held.cache), a.clone(), Arc::clone(&cancel));
        move || cache.get(&a, &Progress::none().cancelled_by(&cancel))
    });
    held.expect(SlotEvent::DecodeStarted);
    let waiter = on_a_thread({
        let (cache, a) = (Arc::clone(&held.cache), a.clone());
        move || cache.get(&a, &Progress::none())
    });
    held.expect(SlotEvent::WaitStarted);

    cancel.store(true, Ordering::SeqCst);
    held.release.send(()).unwrap();
    assert_eq!(
        maker
            .recv_timeout(HANG)
            .expect("the maker never returned")
            .err(),
        Some(GetError::Cancelled)
    );
    let pyramid = waiter
        .recv_timeout(HANG)
        .expect("the waiter never took the decode over")
        .unwrap();
    assert_eq!(pyramid.level(0).get_pixel(3, 2, 0), 3);
    assert_eq!(held.decodes.load(Ordering::SeqCst), 2);
    let stats = held.cache.stats();
    assert_eq!((stats.misses, stats.hits, stats.entries), (2, 0, 1));
}

/// A caller waiting on another caller's decode, through `get` or `get_many`,
/// stops when it is cancelled, while the decode it waited for is still held,
/// and the decode goes on for its maker.
#[test]
fn a_cancelled_waiter_stops_waiting_and_leaves_the_decode_to_its_maker() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let held = cache_holding_its_first_decode();
    let cancel = Arc::new(AtomicBool::new(false));

    let maker = on_a_thread({
        let (cache, a) = (Arc::clone(&held.cache), a.clone());
        move || cache.get(&a, &Progress::none())
    });
    held.expect(SlotEvent::DecodeStarted);
    let waiter = on_a_thread({
        let (cache, a, cancel) = (Arc::clone(&held.cache), a.clone(), Arc::clone(&cancel));
        move || cache.get(&a, &Progress::none().cancelled_by(&cancel))
    });
    held.expect(SlotEvent::WaitStarted);
    let many = on_a_thread({
        let (cache, a, cancel) = (Arc::clone(&held.cache), a.clone(), Arc::clone(&cancel));
        move || {
            cache
                .get_many(&[a.as_path()], &Progress::none().cancelled_by(&cancel))
                .err()
        }
    });
    held.expect(SlotEvent::WaitStarted);

    // Both are waiting on the slot, which the held decode keeps as being
    // made, so only their own cancellation can end their wait.
    cancel.store(true, Ordering::SeqCst);
    assert_eq!(
        waiter
            .recv_timeout(HANG)
            .expect("the waiter never stopped")
            .err(),
        Some(GetError::Cancelled)
    );
    assert_eq!(
        many.recv_timeout(HANG).expect("get_many never stopped"),
        Some(Cancelled)
    );
    held.release.send(()).unwrap();
    let pyramid = maker.recv_timeout(HANG).expect("the maker never returned");
    assert_eq!(pyramid.unwrap().level(0).width(), 8);
    assert_eq!(held.decodes.load(Ordering::SeqCst), 1);
    let stats = held.cache.stats();
    assert_eq!((stats.misses, stats.hits, stats.entries), (1, 0, 1));
}

/// A decoding caller that unwinds leaves the slot empty, so the next caller
/// decodes rather than wait for ever on a slot marked as being made.
#[test]
fn a_maker_that_panics_empties_its_slot() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let decodes = Arc::new(AtomicUsize::new(0));
    let mut cache = PhotographCache::new(with_budget(GIB));
    cache.on_slot_event = Some(Box::new({
        let decodes = Arc::clone(&decodes);
        move |_, event| {
            if event == SlotEvent::DecodeStarted && decodes.fetch_add(1, Ordering::SeqCst) == 0 {
                panic!("a decode that unwinds, as the test asks");
            }
        }
    }));
    let cache = Arc::new(cache);

    let unwound = std::thread::spawn({
        let (cache, a) = (Arc::clone(&cache), a.clone());
        move || cache.get(&a, &Progress::none())
    })
    .join();
    assert!(unwound.is_err());
    // The decode was reserved before it unwound, and the unwinding gave its
    // share back.
    assert_eq!(cache.stats().build_reserved_bytes, 0);
    let next = on_a_thread({
        let (cache, a) = (Arc::clone(&cache), a.clone());
        move || cache.get(&a, &Progress::none())
    });
    let pyramid = next
        .recv_timeout(HANG)
        .expect("the slot was left as being made");
    assert_eq!(pyramid.unwrap().level(0).width(), 8);
    assert_eq!(decodes.load(Ordering::SeqCst), 2);
}

/// A decode whose caller was cancelled is not remembered, with a budget or
/// without one: the next caller decodes the photograph.
#[test]
fn a_cancelled_decode_is_not_remembered() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let cancel = AtomicBool::new(true);
    let cancelled = Progress::none().cancelled_by(&cancel);
    for budget in [GIB, 0] {
        let cache = Arc::new(PhotographCache::new(with_budget(budget)));
        assert_eq!(cache.get(&a, &cancelled).err(), Some(GetError::Cancelled));
        assert_eq!(cache.stats().build_reserved_bytes, 0);
        assert!(matches!(cache.peek_state(&a), PeekedPhotograph::NotDecoded));
        assert_eq!(cache.stats().entries, 0);
        // On a thread of its own, since a slot left as being made would make
        // it wait for ever.
        let next = on_a_thread({
            let (cache, a) = (Arc::clone(&cache), a.clone());
            move || cache.get(&a, &Progress::none())
        });
        let pyramid = next
            .recv_timeout(HANG)
            .expect("the slot was left as being made");
        assert_eq!(pyramid.unwrap().level(0).width(), 8);
        assert_eq!(cache.stats().misses, 2);
    }
}

#[test]
fn budget_zero_keeps_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let cache = PhotographCache::new(with_budget(0));
    let first = cache.get(&a, &Progress::none()).unwrap();
    let second = cache.get(&a, &Progress::none()).unwrap();
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
    let cache = PhotographCache::new(with_budget(GIB));
    assert!(matches!(cache.peek_state(&a), PeekedPhotograph::NotDecoded));
    assert!(matches!(
        cache.peek_state(&missing),
        PeekedPhotograph::NotDecoded
    ));

    let got = cache.get(&a, &Progress::none()).unwrap();
    assert!(cache.get(&missing, &Progress::none()).is_err());
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
    let none = PhotographCache::new(with_budget(0));
    assert_eq!(none.budget_bytes(), 0);
    none.get(&a, &Progress::none()).unwrap();
    assert!(matches!(none.peek_state(&a), PeekedPhotograph::NotDecoded));
}

#[test]
fn peek_state_does_not_wait_on_a_decode_in_flight() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let cache = PhotographCache::new(with_budget(GIB));
    // A slot whose decode has not finished, as `get_found` leaves it while the
    // decoding thread runs.
    cache.lock().map.insert(
        a.clone(),
        Arc::new(Slot::new(SlotState::Making, file_stamp(&a))),
    );
    assert!(matches!(cache.peek_state(&a), PeekedPhotograph::NotDecoded));
}

#[test]
fn peek_does_not_decode() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let cache = PhotographCache::new(with_budget(GIB));
    assert!(cache.peek(&a).is_none());
    let stats = cache.stats();
    assert_eq!((stats.entries, stats.hits, stats.misses), (0, 0, 0));
    let got = cache.get(&a, &Progress::none()).unwrap();
    assert!(Arc::ptr_eq(&cache.peek(&a).unwrap(), &got));
    assert_eq!(cache.stats().hits, 0);
}

#[test]
fn decode_into_a_forgotten_slot_still_returns_and_charges_nothing() {
    // A slot forgotten while its decode runs: the decoding caller gets the
    // pyramid, and the byte count stays at what the map holds.
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 8);
    let cache = PhotographCache::new(with_budget(GIB));
    let slot = Arc::new(Slot::new(SlotState::Making, file_stamp(&a)));
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
    let cache = PhotographCache::new(with_budget(GIB));
    let pyramid = Arc::new(ImageU8Pyramid::from_image(
        ImageU8::from_channels(8, 4, 3),
        1,
    ));
    cache.insert(&absent, Arc::clone(&pyramid));
    assert!(Arc::ptr_eq(
        &cache.get(&absent, &Progress::none()).unwrap(),
        &pyramid
    ));
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
    let small = PhotographCache::new(with_budget(bytes_of(8, 4)));
    let b = dir.path().join("b.png");
    small.insert(&absent, Arc::clone(&pyramid));
    small.insert(&b, Arc::clone(&pyramid));
    assert!(small.peek(&absent).is_none());
    assert!(small.peek(&b).is_some());

    // A budget of 0 keeps nothing.
    let none = PhotographCache::new(with_budget(0));
    none.insert(&absent, pyramid);
    assert!(none.peek(&absent).is_none());
}

/// The bytes a 16 x 16 photograph's decode reserves.
const SMALL: u64 = 16 * 16 * WHOLE_DECODE_BYTES_PER_PIXEL;

/// Write `n` 16 x 16 PNGs into `dir`.
fn small_photographs(dir: &Path, n: usize) -> Vec<PathBuf> {
    (0..n)
        .map(|i| write_png(dir, &format!("p{i:02}.png"), 16, 16))
        .collect()
}

/// The least bytes reserved on a build count as any decode started, and the
/// most as any decode started or any caller started to wait.
type Seen = Arc<(AtomicU64, AtomicU64)>;

fn unseen() -> Seen {
    Arc::new((AtomicU64::new(u64::MAX), AtomicU64::new(0)))
}

/// A cache made with `options` whose hook keeps in `seen` what its build count
/// held at each decode's start and each wait's.
fn cache_watching_its_count(options: PhotographCacheOptions, seen: &Seen) -> PhotographCache {
    let mut cache = PhotographCache::new(options);
    let seen = Arc::clone(seen);
    cache.on_slot_event = Some(Box::new(move |cache, event| {
        let reserved = cache.build_reserved.load(Ordering::SeqCst);
        if event == SlotEvent::DecodeStarted {
            seen.0.fetch_min(reserved, Ordering::SeqCst);
        }
        seen.1.fetch_max(reserved, Ordering::SeqCst);
    }));
    cache
}

/// `reserve_builds` grants as many decodes as fit beside what is reserved,
/// and one when none fit, a decode larger than the whole budget included;
/// each is given back as it ends and the rest when the reservation drops.
/// No size or budget, however absurd, panics.
#[test]
fn reserve_builds_grants_what_fits_and_one_when_none_does() {
    let cache = PhotographCache::new(PhotographCacheOptions {
        build_budget_bytes: 10_000,
        ..with_budget(0)
    });
    let reserved = || cache.stats().build_reserved_bytes;
    let three = cache.reserve_builds(3_000, 5);
    assert_eq!((three.granted(), reserved()), (3, 9_000));
    // None fits, so one is granted past the budget.
    let past = cache.reserve_builds(3_000, 5);
    assert_eq!((past.granted(), reserved()), (1, 12_000));
    three.release_one();
    assert_eq!(reserved(), 9_000);
    drop(three);
    assert_eq!(reserved(), 3_000);
    drop(past);
    assert_eq!(reserved(), 0);
    // Each decode goes back once, however often it is released.
    let two = cache.reserve_builds(1_000, 2);
    for _ in 0..4 {
        two.release_one();
    }
    assert_eq!(reserved(), 0);
    drop(two);
    assert_eq!(reserved(), 0);
    // One decode larger than the whole budget is let through alone.
    let large = cache.reserve_builds(50_000, 2);
    assert_eq!((large.granted(), reserved()), (1, 50_000));
    drop(large);
    assert_eq!(cache.reserve_builds(1_000, 0).granted(), 0);
    assert_eq!(cache.reserve_builds(0, 7).granted(), 7);
    let absurd = cache.reserve_builds(u64::MAX, usize::MAX);
    assert_eq!(absurd.granted(), 1);
    drop(absurd);
    assert_eq!(reserved(), 0);
}

/// `try_reserve_build` is refused whenever the decode would take the count
/// past the build budget, from one thread and from eight at once.
#[test]
fn try_reserve_build_never_takes_the_count_past_the_build_budget() {
    let cache = PhotographCache::new(PhotographCacheOptions {
        build_budget_bytes: 10_000,
        ..with_budget(0)
    });
    let reserved = || cache.stats().build_reserved_bytes;
    let held = cache.reserve_builds(6_000, 1);
    assert!(cache.try_reserve_build(4_001).is_none());
    let fits = cache.try_reserve_build(4_000).unwrap();
    assert_eq!(reserved(), 10_000);
    assert!(cache.try_reserve_build(1).is_none());
    assert!(cache.try_reserve_build(u64::MAX).is_none());
    drop((held, fits));
    assert_eq!(reserved(), 0);
    assert!(cache.try_reserve_build(10_001).is_none());

    let threads = 8;
    let barrier = Barrier::new(threads);
    let most = AtomicU64::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                barrier.wait();
                for _ in 0..200 {
                    if let Some(granted) = cache.try_reserve_build(3_000) {
                        most.fetch_max(reserved(), Ordering::SeqCst);
                        drop(granted);
                    }
                }
            });
        }
    });
    assert!(most.load(Ordering::SeqCst) <= 10_000);
    assert_eq!(reserved(), 0);

    // A build budget of 0 refuses every decode that has a size.
    let none = PhotographCache::new(PhotographCacheOptions {
        build_budget_bytes: 0,
        ..with_budget(0)
    });
    assert!(none.try_reserve_build(1).is_none());
}

/// A `get_many` over 20 photographs decodes them in groups whose reservations
/// never pass the build budget, set to three decodes, and every decode is
/// reserved when it starts; with a budget and with a cache that keeps
/// nothing.
#[test]
fn get_many_decodes_in_groups_within_the_build_budget() {
    let dir = tempfile::tempdir().unwrap();
    let photographs = small_photographs(dir.path(), 20);
    let paths: Vec<&Path> = photographs.iter().map(PathBuf::as_path).collect();
    for budget in [GIB, 0] {
        let seen = unseen();
        let cache = cache_watching_its_count(
            PhotographCacheOptions {
                build_budget_bytes: 3 * SMALL,
                ..with_budget(budget)
            },
            &seen,
        );
        let (found, tally) = cache.get_many(&paths, &Progress::none()).unwrap();
        assert!(found.iter().all(Option::is_some));
        assert_eq!(tally.read, 20);
        let (least, most) = (seen.0.load(Ordering::SeqCst), seen.1.load(Ordering::SeqCst));
        assert!(
            least >= SMALL,
            "a decode started with {least} bytes reserved"
        );
        assert!(
            most <= 3 * SMALL,
            "{most} bytes reserved, past the build budget"
        );
        let stats = cache.stats();
        assert_eq!((stats.decodes, stats.build_reserved_bytes), (20, 0));
        assert_eq!(stats.build_budget_bytes, 3 * SMALL);
    }
}

/// Four `get_many` callers on one cache, started together on threads of their
/// own over the same 20 photographs, never reserve more than the build budget
/// plus one decode each, whatever order their reservations and decodes run in;
/// each photograph is decoded once, and nothing stays reserved.
#[test]
fn get_many_callers_together_reserve_at_most_the_budget_plus_one_decode_each() {
    let dir = tempfile::tempdir().unwrap();
    let photographs = Arc::new(small_photographs(dir.path(), 20));
    let seen = unseen();
    let cache = Arc::new(cache_watching_its_count(
        PhotographCacheOptions {
            build_budget_bytes: 2 * SMALL,
            ..with_budget(GIB)
        },
        &seen,
    ));
    let callers = 4;
    let barrier = Arc::new(Barrier::new(callers));
    let answers: Vec<_> = (0..callers)
        .map(|_| {
            let (cache, photographs, barrier) = (
                Arc::clone(&cache),
                Arc::clone(&photographs),
                Arc::clone(&barrier),
            );
            on_a_thread(move || {
                let paths: Vec<&Path> = photographs.iter().map(PathBuf::as_path).collect();
                barrier.wait();
                let (found, _) = cache.get_many(&paths, &Progress::none()).unwrap();
                found.iter().all(Option::is_some)
            })
        })
        .collect();
    for answer in answers {
        assert!(answer
            .recv_timeout(HANG)
            .expect("a get_many never returned"));
    }
    let most = seen.1.load(Ordering::SeqCst);
    let bound = 2 * SMALL + callers as u64 * SMALL;
    assert!(most <= bound, "{most} bytes reserved, past {bound}");
    assert!(seen.0.load(Ordering::SeqCst) >= SMALL);
    let stats = cache.stats();
    assert_eq!((stats.misses, stats.decodes), (20, 20));
    assert_eq!(stats.build_reserved_bytes, 0);
}

/// Two caches in one process count separately: one whose build budget is
/// used up does not refuse the other's decodes. A decode adds only its own
/// reservation to its cache's count, and gives it back when it ends.
#[test]
fn two_caches_keep_separate_counts() {
    let dir = tempfile::tempdir().unwrap();
    let photographs = small_photographs(dir.path(), 6);
    let paths: Vec<&Path> = photographs.iter().map(PathBuf::as_path).collect();
    let options = PhotographCacheOptions {
        build_budget_bytes: 4 * SMALL,
        ..with_budget(GIB)
    };
    let (seen_a, seen_b) = (unseen(), unseen());
    let a = cache_watching_its_count(options, &seen_a);
    let b = cache_watching_its_count(options, &seen_b);

    let full = a.reserve_builds(SMALL, 4);
    assert_eq!(full.granted(), 4);
    assert!(a.try_reserve_build(SMALL).is_none());
    assert!(b.try_reserve_build(SMALL).is_some());
    // A's decode is granted past its full budget and counts only itself.
    a.get(paths[0], &Progress::none()).unwrap();
    assert_eq!(seen_a.0.load(Ordering::SeqCst), 5 * SMALL);
    assert_eq!(seen_a.1.load(Ordering::SeqCst), 5 * SMALL);
    assert_eq!(a.stats().build_reserved_bytes, 4 * SMALL);
    // B decodes within its own budget, unaffected by A's.
    b.get_many(&paths, &Progress::none()).unwrap();
    assert!(seen_b.0.load(Ordering::SeqCst) >= SMALL);
    assert!(seen_b.1.load(Ordering::SeqCst) <= 4 * SMALL);
    assert_eq!(b.stats().build_reserved_bytes, 0);
    assert_eq!(a.stats().build_reserved_bytes, 4 * SMALL);
    drop(full);
    assert_eq!(a.stats().build_reserved_bytes, 0);
}

/// A maker cancelled during its decode gives its share back before its slot
/// empties, and the waiter that decodes in its place reserves: a `get` when
/// it takes the decode over, holding nothing while it waits, and a `get_many`
/// member with the share it kept while it waited.
#[test]
fn a_cancelled_maker_gives_its_share_back_and_its_successor_reserves() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let one = 8 * 4 * WHOLE_DECODE_BYTES_PER_PIXEL;
    for many in [false, true] {
        let held = cache_holding_its_first_decode();
        let cancel = Arc::new(AtomicBool::new(false));
        let maker = on_a_thread({
            let (cache, a, cancel) = (Arc::clone(&held.cache), a.clone(), Arc::clone(&cancel));
            move || cache.get(&a, &Progress::none().cancelled_by(&cancel))
        });
        assert_eq!(held.expect(SlotEvent::DecodeStarted), one);
        let waiter = on_a_thread({
            let (cache, a) = (Arc::clone(&held.cache), a.clone());
            move || {
                if many {
                    let (found, _) = cache.get_many(&[a.as_path()], &Progress::none()).unwrap();
                    found[0].clone().ok_or(GetError::NotFound(a))
                } else {
                    cache.get(&a, &Progress::none())
                }
            }
        });
        let waiting = held.expect(SlotEvent::WaitStarted);
        assert_eq!(waiting, if many { 2 * one } else { one }, "many: {many}");

        cancel.store(true, Ordering::SeqCst);
        held.release.send(()).unwrap();
        // The successor's decode, with the maker's share given back.
        assert_eq!(held.expect(SlotEvent::DecodeStarted), one, "many: {many}");
        assert_eq!(
            maker
                .recv_timeout(HANG)
                .expect("the maker never returned")
                .err(),
            Some(GetError::Cancelled)
        );
        let pyramid = waiter
            .recv_timeout(HANG)
            .expect("the waiter never took the decode over")
            .unwrap();
        assert_eq!(pyramid.level(0).width(), 8);
        let stats = held.cache.stats();
        assert_eq!((stats.decodes, stats.build_reserved_bytes), (2, 0));
    }
}
