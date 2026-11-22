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

/// Each pyramid's depth is `full_levels` of its own photograph's size, down to
/// one or two pixels on the short side, so two photographs in one cache can
/// have different depths. `Debug` shows the stats.
#[test]
fn each_entry_has_the_depth_of_its_own_size() {
    let dir = tempfile::tempdir().unwrap();
    let small = write_png(dir.path(), "small.png", 64, 48);
    let tall = write_png(dir.path(), "tall.png", 270, 480);
    let cache = PhotographCache::new(GIB);
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
    let pyramid = PhotographCache::new(0)
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
    let cache = PhotographCache::new(GIB);
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
    let cache = PhotographCache::new(2 * one + one / 2);

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
    let cache = PhotographCache::new(bytes_of(8, 8) / 2);
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
    let cache = PhotographCache::new(bytes_of(8, 8));
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
    let cache = PhotographCache::new(GIB);

    assert_eq!(
        cache.get(&missing, &Progress::none()).err().unwrap(),
        GetError::NotFound(missing.clone())
    );
    assert!(matches!(
        cache.get(&corrupt, &Progress::none()).err().unwrap(),
        GetError::Unreadable(why) if why.contains("corrupt.png")
    ));
    assert_eq!(cache.stats().misses, 2);
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
    let cache = PhotographCache::new(GIB);
    assert!(cache.get(&path, &Progress::none()).is_err());
    write_png(dir.path(), "late.png", 8, 8);
    assert_eq!(width(&cache.get(&path, &Progress::none()).ok()), 8);
}

#[test]
fn a_rewritten_file_is_decoded_again() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_png(dir.path(), "a.png", 8, 8);
    let cache = PhotographCache::new(GIB);
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
    let cache = PhotographCache::new(GIB);
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
    let cache = PhotographCache::new(GIB);
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
    let cache = PhotographCache::new(GIB);
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
    let cache = PhotographCache::new(GIB);
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
/// `release`. `events` receives every [`SlotEvent`] as it happens, and
/// `decodes` counts the decodes started.
struct HeldDecode {
    cache: Arc<PhotographCache>,
    events: std::sync::mpsc::Receiver<SlotEvent>,
    release: std::sync::mpsc::Sender<()>,
    decodes: Arc<AtomicUsize>,
}

fn cache_holding_its_first_decode() -> HeldDecode {
    let (event_tx, events) = std::sync::mpsc::channel();
    let (release, release_rx) = std::sync::mpsc::channel::<()>();
    let release_rx = Mutex::new(release_rx);
    let decodes = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&decodes);
    let mut cache = PhotographCache::new(GIB);
    cache.on_slot_event = Some(Box::new(move |event| {
        let first =
            event == SlotEvent::DecodeStarted && counted.fetch_add(1, Ordering::SeqCst) == 0;
        let _ = event_tx.send(event);
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
    /// Wait for the next event, which must be `expected`.
    fn expect(&self, expected: SlotEvent) {
        let event = self.events.recv_timeout(HANG).expect("no slot event came");
        assert_eq!(event, expected);
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
    let mut cache = PhotographCache::new(GIB);
    cache.on_slot_event = Some(Box::new({
        let decodes = Arc::clone(&decodes);
        move |event| {
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
        let cache = Arc::new(PhotographCache::new(budget));
        assert_eq!(cache.get(&a, &cancelled).err(), Some(GetError::Cancelled));
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
    let cache = PhotographCache::new(0);
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
    let cache = PhotographCache::new(GIB);
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
    let none = PhotographCache::new(0);
    assert_eq!(none.budget_bytes(), 0);
    none.get(&a, &Progress::none()).unwrap();
    assert!(matches!(none.peek_state(&a), PeekedPhotograph::NotDecoded));
}

#[test]
fn peek_state_does_not_wait_on_a_decode_in_flight() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_png(dir.path(), "a.png", 8, 4);
    let cache = PhotographCache::new(GIB);
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
    let cache = PhotographCache::new(GIB);
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
    let cache = PhotographCache::new(GIB);
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
    let cache = PhotographCache::new(GIB);
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
    let small = PhotographCache::new(bytes_of(8, 4));
    let b = dir.path().join("b.png");
    small.insert(&absent, Arc::clone(&pyramid));
    small.insert(&b, Arc::clone(&pyramid));
    assert!(small.peek(&absent).is_none());
    assert!(small.peek(&b).is_some());

    // A budget of 0 keeps nothing.
    let none = PhotographCache::new(0);
    none.insert(&absent, pyramid);
    assert!(none.peek(&absent).is_none());
}
