// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! An out-of-core photograph's mipmap made in its slot by `get` and
//! `get_many`: 300 x 200 synthetic photographs in a temporary workspace whose
//! `mipmap_options` sets a threshold of 64, read with tiles of 32, so each is
//! out of core with levels 1 to 4 stored, as in `mipmap_tests.rs`.

use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, SystemTime};

use crate::camera::mipmap::test_support::{encode_jpeg, synthetic_rgb, JpegOptions};
use crate::camera::mipmap::MIPMAP_DIR_NAME;
use crate::camera::sampler::{LevelRect, LevelRegions};
use crate::progress::{Event, Level, Progress};

use super::mipmap_tests::{
    cache_of, files_levels, level_file, out_of_core_layout, read_level, readable_jpeg,
    restartless_jpeg, same_tile, tiles_of, Workspace, TILE,
};

use super::tests::{on_a_thread, HANG};
use super::*;

/// The levels of `photograph` as a photograph `get` answered reads them.
fn levels_read(photograph: &TiledPhotograph) -> Vec<Vec<u8>> {
    (0..photograph.layout().levels())
        .map(|level| read_level(photograph, level).unwrap())
        .collect()
}

/// The levels `files_levels` reads from the files on disk.
fn files_level_bytes(path: &Path) -> Vec<Vec<u8>> {
    files_levels(path)
        .into_iter()
        .map(|level| level.data().to_vec())
        .collect()
}

/// The mip level files under the workspace's mipmap directory, by level.
fn levels_on_disk(ws: &Workspace) -> Vec<u32> {
    let mut levels: Vec<u32> = ws
        .mipmap_files()
        .iter()
        .filter_map(|(path, _)| {
            let name = path.file_name()?.to_str()?;
            let at = name.rfind("-mip")?;
            let rest = &name[at + 4..];
            rest[..rest.find('-')?].parse().ok()
        })
        .collect();
    levels.sort_unstable();
    levels
}

/// A kernel's reads (`get_tile`, `view_region_pyramid`, `get_level`) of an
/// out-of-core photograph whose mipmap is missing answer `Changed` naming
/// it, decoding and making nothing; a `get` makes the mipmap, writing its
/// files, and answers the photograph read from them with nothing decoded
/// whole, after which the kernel's reads read those files. A photograph whose
/// original changes has its stale mipmap made again by the next `get`. A
/// `get_many` after a panel decoded the photograph in core makes the mipmap
/// rather than answer the in-core tiles, reporting the build's phases through
/// nothing. With a budget and with one of 0, for a region-readable original
/// and one that gets a `-mip0`.
#[test]
fn a_get_makes_a_missing_or_stale_mipmap_and_reads_it() {
    let key = TileKey {
        level: 1,
        col: 0,
        row: 0,
    };
    let mut regions = LevelRegions::new(&out_of_core_layout());
    let rect = LevelRect {
        x0: 0,
        y0: 0,
        x1: 8,
        y1: 8,
    };
    regions.set_region(1, rect);
    // The synthetic photograph and its negative, so a remade mipmap holds
    // other pixels.
    let negative = {
        let image = synthetic_rgb(300, 200);
        let data = image.data().iter().map(|v| 255 - v).collect();
        ImageU8::new(300, 200, 3, data)
    };
    let readable = JpegOptions {
        sampling: jpeg_encoder::SamplingFactor::F_2_2,
        restart_interval: 19,
        ..JpegOptions::default()
    };
    for (name, jpeg, stored) in [
        ("readable.jpg", readable, vec![1, 2, 3, 4]),
        (
            "restartless.jpg",
            JpegOptions::default(),
            vec![0, 1, 2, 3, 4],
        ),
    ] {
        for budget in [GIB, 0] {
            let ws = Workspace::new();
            let path = ws.photograph(name, &encode_jpeg(&synthetic_rgb(300, 200), jpeg));
            let cache = cache_of(budget);
            let changed = |e: Option<TileError>| {
                assert_eq!(e, Some(TileError::Changed { path: path.clone() }))
            };
            changed(cache.get_tile(&path, key, &Progress::none()).err());
            changed(cache.get_level(&path, 1, &Progress::none()).err());
            changed(
                cache
                    .view_region_pyramid(&path, &regions, &Progress::none())
                    .err(),
            );
            assert!(
                ws.mipmap_files().is_empty(),
                "{name}: no kernel read made it"
            );

            let photograph = cache.get(&path, &Progress::none()).unwrap();
            assert_eq!(levels_on_disk(&ws), stored, "{name}, {budget}");
            assert_eq!(*photograph.layout(), out_of_core_layout());
            assert_eq!(levels_read(&photograph), files_level_bytes(&path));
            let stats = cache.stats();
            assert_eq!((stats.decodes, stats.mipmaps_made), (0, 1), "{name}");
            let tile = cache.get_tile(&path, key, &Progress::none()).unwrap();
            let expected = tiles_of(&out_of_core_layout(), &files_levels(&path));
            assert!(same_tile(&tile, &expected[&key]));

            // Another original at the same path: its mipmap is stale.
            let before = files_level_bytes(&path);
            std::fs::write(&path, encode_jpeg(&negative, jpeg)).unwrap();
            let photograph = cache.get(&path, &Progress::none()).unwrap();
            assert_eq!(cache.stats().mipmaps_made, 2, "{name}: made again");
            let after = files_level_bytes(&path);
            assert_ne!(after, before);
            assert_eq!(levels_read(&photograph), after);
        }
    }

    // A panel's decode in core, then a kernel's `get_many`.
    let ws = Workspace::new();
    let path = ws.photograph("a.jpg", &readable_jpeg());
    let cache = cache_of(GIB);
    let Ok(TryDecode::Decoded(_)) = cache.try_decode_tiles(&path, &[key], &Progress::none()) else {
        panic!("decoded in core");
    };
    let phases = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        if let Event::Enter { phase, .. } = event {
            phases.lock().unwrap().push(phase);
        }
    };
    let (found, _) = cache
        .get_many(&[path.as_path()], &Progress::to(&sink))
        .unwrap();
    let photograph = found[0].as_ref().unwrap();
    assert_eq!(*photograph.layout(), out_of_core_layout());
    assert_eq!(levels_read(photograph), files_level_bytes(&path));
    let stats = cache.stats();
    assert_eq!((stats.decodes, stats.mipmaps_made), (1, 1));
    assert!(phases.lock().unwrap().is_empty(), "{:?}", phases.lock());
}

/// A cache whose decodes and builds each wait, once started, until the test
/// sends on `release`, reporting each slot event on `events`, and counting
/// the builds started in `started`.
struct HeldBuilds {
    cache: Arc<PhotographCache>,
    events: Receiver<(SlotEvent, u64)>,
    release: Sender<()>,
}

fn cache_holding_every_build() -> HeldBuilds {
    let (event_tx, events) = channel();
    let (release, release_rx) = channel::<()>();
    let release_rx = Mutex::new(release_rx);
    let mut cache = PhotographCache::new(PhotographCacheOptions {
        budget_bytes: GIB,
        build_budget_bytes: GIB,
        tile_size: TILE,
    });
    cache.set_slot_hook(Box::new(move |cache, event, _| {
        // Read without `stats`, which takes the slot lock a waiter holds.
        let reserved = cache.shared.build_reserved.load(Ordering::SeqCst);
        let _ = event_tx.send((event, reserved));
        if event == SlotEvent::DecodeStarted {
            let _ = release_rx.lock().unwrap().recv_timeout(HANG);
        }
    }));
    HeldBuilds {
        cache: Arc::new(cache),
        events,
        release,
    }
}

impl HeldBuilds {
    /// Wait for the next event, which must be `expected`, and answer the
    /// bytes reserved on the build count when it happened.
    fn expect(&self, expected: SlotEvent) -> u64 {
        let (event, reserved) = self.events.recv_timeout(HANG).expect("no slot event came");
        assert_eq!(event, expected);
        reserved
    }
}

/// Two threads asking for one path with `get` make one mipmap: the second
/// waits on the slot while the first makes it, holding no reservation, and
/// reads what it made.
#[test]
fn two_threads_asking_for_one_path_make_one_mipmap() {
    let ws = Workspace::new();
    let path = ws.photograph("a.jpg", &readable_jpeg());
    let held = cache_holding_every_build();
    let get = || {
        let (cache, path) = (Arc::clone(&held.cache), path.clone());
        on_a_thread(move || cache.get(&path, &Progress::none()))
    };
    let first = get();
    let build_bytes = whole_decode_bytes(300, 200);
    assert_eq!(held.expect(SlotEvent::DecodeStarted), build_bytes);
    let second = get();
    assert_eq!(held.expect(SlotEvent::WaitStarted), build_bytes);
    held.release.send(()).unwrap();
    for answer in [first, second] {
        let photograph = answer.recv_timeout(HANG).expect("never answered").unwrap();
        assert_eq!(*photograph.layout(), out_of_core_layout());
    }
    let stats = held.cache.stats();
    assert_eq!((stats.mipmaps_made, stats.misses), (1, 1));
    assert_eq!(stats.build_reserved_bytes, 0);
    assert!(held.events.try_recv().is_err(), "one build started");
}

/// A maker cancelled during its build's encodes stops before the next
/// level, having written the files of the levels it finished, but not the
/// coarsest, which is written last; it leaves the slot to the caller waiting
/// on it, which makes only the levels whose files are not current and answers
/// the photograph. The cancelled build is not counted as made.
#[test]
fn a_cancelled_build_stops_within_a_stage_and_the_taker_makes_only_the_rest() {
    let ws = Workspace::new();
    let path = ws.photograph("a.jpg", &readable_jpeg());
    let held = cache_holding_every_build();
    let cancel = Arc::new(AtomicBool::new(false));
    let encoded = Arc::new(Mutex::new(Vec::new()));
    let maker = on_a_thread({
        let (cache, path) = (Arc::clone(&held.cache), path.clone());
        let (cancel, encoded) = (Arc::clone(&cancel), Arc::clone(&encoded));
        move || {
            let phase = Mutex::new("");
            let sink = |event: Event<'_>| match event {
                Event::Enter { phase: name, .. } => *phase.lock().unwrap() = name,
                Event::Count { done, .. } if *phase.lock().unwrap() == "encode" => {
                    encoded.lock().unwrap().push(done);
                    // Cancelled once two levels are encoded.
                    if done == 2 {
                        cancel.store(true, Ordering::SeqCst);
                    }
                }
                _ => {}
            };
            let progress = Progress::to(&sink).cancelled_by(&cancel);
            cache.get(&path, &progress).err()
        }
    });
    held.expect(SlotEvent::DecodeStarted);
    let waiter = on_a_thread({
        let (cache, path) = (Arc::clone(&held.cache), path.clone());
        move || cache.get(&path, &Progress::none())
    });
    held.expect(SlotEvent::WaitStarted);
    held.release.send(()).unwrap();
    let stopped = maker.recv_timeout(HANG).expect("the maker never stopped");
    assert_eq!(stopped, Some(GetError::Cancelled));
    assert_eq!(*encoded.lock().unwrap(), [1, 2], "stopped within one level");

    // The waiter has taken the build over and is held as it starts.
    held.expect(SlotEvent::DecodeStarted);
    assert_eq!(levels_on_disk(&ws), [3], "level 4 is encoded, not written");
    let old = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000);
    let set_old = std::fs::File::options()
        .write(true)
        .open(level_file(&path, 3))
        .unwrap();
    set_old.set_modified(old).unwrap();
    drop(set_old);
    held.release.send(()).unwrap();
    let photograph = waiter
        .recv_timeout(HANG)
        .expect("the waiter never took the build over")
        .unwrap();
    assert_eq!(levels_on_disk(&ws), [1, 2, 3, 4]);
    let modified = std::fs::metadata(level_file(&path, 3))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(modified, old, "level 3 was not made again");
    assert_eq!(levels_read(&photograph), files_level_bytes(&path));
    let stats = held.cache.stats();
    assert_eq!(stats.mipmaps_made, 1);
    assert!(held.events.try_recv().is_err(), "two builds started");
    assert_eq!(stats.build_reserved_bytes, 0);
}

/// A `get` that makes a mipmap reports the phases `decode`, `reduce` and
/// `encode` through its `Progress`, counting the levels it encodes:
/// `-mip0` and the four reduced levels for an original that is not
/// region-readable.
#[test]
fn a_build_reports_its_phases_through_the_caller_s_progress() {
    let ws = Workspace::new();
    let path = ws.photograph("a.jpg", &restartless_jpeg());
    let events = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| match event {
        Event::Enter { phase, .. } => events.lock().unwrap().push(phase.to_string()),
        Event::Count {
            done,
            total,
            unit: "levels",
        } => events
            .lock()
            .unwrap()
            .push(format!("{done} of {}", total.unwrap_or(0))),
        _ => {}
    };
    cache_of(GIB).get(&path, &Progress::to(&sink)).unwrap();
    let events = events.into_inner().unwrap();
    let encode = events.iter().position(|e| e == "encode").unwrap();
    assert_eq!(events[..2], ["decode", "reduce"]);
    assert_eq!(
        events[encode + 1..],
        ["1 of 5", "2 of 5", "3 of 5", "4 of 5", "5 of 5"]
    );
}

/// A build whose mipmap directory is blocked, so no file can be written, is
/// read from the files it made in memory, with a warning through the
/// caller's `Progress`: the same pixels as the files a build in an
/// unblocked workspace writes, their bytes counted against the budget and
/// kept under a budget too small to hold them, until the path is forgotten.
#[test]
fn a_build_behind_a_blocked_mipmap_directory_is_read_from_memory() {
    let jpeg = restartless_jpeg();
    let written = Workspace::new();
    let on_disk = written.photograph("a.jpg", &jpeg);
    cache_of(GIB).get(&on_disk, &Progress::none()).unwrap();
    let want = files_level_bytes(&on_disk);
    let held_bytes: u64 = written
        .mipmap_files()
        .iter()
        .filter(|(path, _)| path.extension().is_some_and(|e| e == "jpg"))
        .map(|(_, bytes)| bytes.len() as u64)
        .sum();

    let blocked = Workspace::new();
    let path = blocked.photograph("a.jpg", &jpeg);
    std::fs::write(blocked.dir.path().join(MIPMAP_DIR_NAME), b"not a directory").unwrap();
    let warnings = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        if let Event::Message {
            level: Level::Warn,
            text,
            ..
        } = event
        {
            warnings.lock().unwrap().push(text.to_string());
        }
    };
    let cache = cache_of(1);
    let photograph = cache.get(&path, &Progress::to(&sink)).unwrap();
    let warnings = warnings.into_inner().unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("held in memory"), "{warnings:?}");
    assert_eq!(levels_read(&photograph), want);
    let stats = cache.stats();
    assert_eq!((stats.files, stats.mipmaps_made, stats.decodes), (0, 1, 0));
    assert!(stats.bytes >= held_bytes, "{stats:?} against {held_bytes}");
    cache.forget(&path);
    assert_eq!(cache.stats().bytes, 0);
}
