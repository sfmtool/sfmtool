// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! An out-of-core photograph's mipmap made in its slot by `get` and
//! `get_many`: 300 x 200 synthetic photographs in a temporary workspace whose
//! `mipmap_options` sets a threshold of 64, read with tiles of 32, so each is
//! out of core with levels 1 to 4 stored, as in `mipmap_tests.rs`.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, SystemTime};

use crate::camera::mipmap::test_support::{encode_jpeg, synthetic_rgb, JpegOptions};
use crate::camera::mipmap::MIPMAP_DIR_NAME;
use crate::camera::sampler::{read_source_rect, test_pool, LevelRect, LevelRegions};
use crate::progress::{Event, Level, Progress};

use super::mipmap_tests::{
    build, cache_of, files_levels, level_file, out_of_core_layout, read_level, readable_jpeg,
    restartless_jpeg, same_tile, tiles_of, Workspace, TILE,
};

use super::tests::{assert_within_the_build_budget, cache_watching_its_builds, on_a_thread, HANG};
use super::*;

/// The levels of `photograph` as a photograph `get` answered reads them.
fn levels_read(photograph: &TiledPhotograph) -> Vec<Vec<u8>> {
    (0..photograph.layout().levels())
        .map(|level| read_level(photograph, level).unwrap())
        .collect()
}

/// [`levels_read`], each level read through a render entry point with
/// `progress`, which hears what a build made during the read says.
fn levels_read_with(photograph: &TiledPhotograph, progress: &Progress<'_>) -> Vec<Vec<u8>> {
    (0..photograph.layout().levels())
        .map(|level| {
            let (x1, y1) = photograph.layout().level_size(level);
            let rect = LevelRect {
                x0: 0,
                y0: 0,
                x1,
                y1,
            };
            let read = read_source_rect(photograph, test_pool(), level, rect, progress).unwrap();
            read.image().data().to_vec()
        })
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
/// panel's decode of a photograph with no mipmap answers `Changed` and
/// leaves it `Building`, decoding nothing, and a `get_many` then makes the
/// mipmap, reporting the build's phases through nothing, after which the
/// photograph's layout is `Known`. With a budget and with one of 0, for a region-readable original
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

    // A panel's decode, which reads nothing in core and makes the
    // photograph `Building`, then a kernel's `get_many`, which makes it
    // `Known`.
    let ws = Workspace::new();
    let path = ws.photograph("a.jpg", &readable_jpeg());
    let cache = cache_of(GIB);
    assert_eq!(
        cache
            .try_decode_tiles(&path, &[key], &Progress::none())
            .err(),
        Some(TileError::Changed { path: path.clone() })
    );
    assert_eq!(cache.layout(&path), PeekedLayout::Building);
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
    assert_eq!((stats.decodes, stats.mipmaps_made), (0, 1));
    assert!(phases.lock().unwrap().is_empty(), "{:?}", phases.lock());
    assert_eq!(
        cache.layout(&path),
        PeekedLayout::Known(Arc::new(out_of_core_layout()))
    );
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

/// A temporary workspace whose mipmap directory is blocked by a file, so no
/// mip level file can be written in it, with `n` copies of the restartless
/// photograph; and the levels its files hold, read from an unblocked
/// workspace, with the bytes one cache keeps for them as file entries.
fn blocked_workspace(n: usize) -> (Workspace, Vec<PathBuf>, Vec<Vec<u8>>, u64) {
    let jpeg = restartless_jpeg();
    let written = Workspace::new();
    let on_disk = written.photograph("a.jpg", &jpeg);
    let cache = cache_of(GIB);
    cache.get(&on_disk, &Progress::none()).unwrap();
    let files_bytes = cache.stats().bytes;
    let blocked = Workspace::new();
    let paths = (0..n)
        .map(|i| blocked.photograph(&format!("{i}.jpg"), &jpeg))
        .collect();
    std::fs::write(blocked.dir.path().join(MIPMAP_DIR_NAME), b"not a directory").unwrap();
    (blocked, paths, files_level_bytes(&on_disk), files_bytes)
}

/// The warnings a `Progress` made with [`Progress::to`] of this sink is told.
struct Warnings(Mutex<Vec<String>>);

impl Warnings {
    fn sink(&self, event: Event<'_>) {
        if let Event::Message {
            level: Level::Warn,
            text,
            ..
        } = event
        {
            self.0.lock().unwrap().push(text.to_string());
        }
    }
}

/// Builds behind a blocked mipmap directory are read from the files they
/// made in memory, kept as file entries counted against the budget and
/// evicted when other photographs need the room: over five photographs under
/// a budget of three photographs' files, the bytes kept after each `get` stay
/// within the budget and one photograph's files, less than keeping them all,
/// and reading each photograph's every level makes no build again. A panel's
/// read of a level evicted since is answered from a decode in core and makes
/// nothing; a kernel's read of it makes the build again, once, reading the
/// same pixels.
/// Each photograph is warned about once, though six builds are made.
#[test]
fn builds_behind_a_blocked_mipmap_directory_stay_within_the_budget() {
    let (_ws, paths, want, files_bytes) = blocked_workspace(5);
    let budget = 3 * files_bytes;
    let cache = cache_of(budget);
    let warnings = Warnings(Mutex::new(Vec::new()));
    let sink = |event: Event<'_>| warnings.sink(event);
    let progress = Progress::to(&sink);
    let mut photographs = Vec::new();
    for path in &paths {
        let photograph = cache.get(path, &progress).unwrap();
        let stats = cache.stats();
        assert!(stats.bytes <= budget + files_bytes, "{stats:?}");
        assert!(stats.files > 0, "kept as file entries");
        assert_eq!(levels_read(&photograph), want);
        photographs.push(photograph);
    }
    assert_eq!(cache.stats().mipmaps_made, 5, "no build made again");

    // The first photograph's levels have been evicted for the others'.
    let key = TileKey {
        level: 1,
        col: 1,
        row: 1,
    };
    let in_core = cut_image(ImageU8::read_rgb(&paths[0]).unwrap(), TILE).tiles;
    let Ok(TryDecode::Decoded(tiles)) =
        cache.try_decode_tiles(&paths[0], &[key], &Progress::none())
    else {
        panic!("a panel's decode");
    };
    assert!(same_tile(&tiles[0], &in_core[&key]), "decoded in core");
    assert_eq!(cache.stats().mipmaps_made, 5, "a panel makes nothing");

    assert_eq!(levels_read_with(&photographs[0], &progress), want);
    let stats = cache.stats();
    assert_eq!(stats.mipmaps_made, 6, "made again once");
    assert!(stats.bytes <= budget + files_bytes, "{stats:?}");
    let warnings = warnings.0.into_inner().unwrap();
    assert_eq!(warnings.len(), 5, "{warnings:?}");
    assert!(warnings.iter().all(|w| w.contains("held in memory")));
    cache.clear();
    assert_eq!(cache.stats().bytes, 0);
}

/// A photograph `get` answered from a build held in memory reads every level
/// after its slot has left the map, which drops the slot's entries: the first
/// read makes the build again, once, and the photograph holds what it made,
/// so its later reads make nothing; every level is the same pixels as the
/// files an unblocked build writes.
#[test]
fn an_evicted_in_memory_level_is_made_again_and_reads_the_same_pixels() {
    let (_ws, paths, want, _) = blocked_workspace(1);
    let cache = cache_of(GIB);
    let photograph = cache.get(&paths[0], &Progress::none()).unwrap();
    cache.forget(&paths[0]);
    assert_eq!(cache.stats().bytes, 0);
    assert_eq!(levels_read(&photograph), want);
    assert_eq!(cache.stats().mipmaps_made, 2);
    assert_eq!(levels_read(&photograph), want);
    let stats = cache.stats();
    assert_eq!((stats.mipmaps_made, stats.bytes), (2, 0), "held, not kept");
}

/// Under a budget of 0, which keeps no build, a photograph whose mipmap a
/// `get_many` made in memory, its files not written, is answered to a panel
/// from a decode in core while its original stays the same: `request_layout`
/// then answers its in-core layout, and a panel's decode answers the
/// in-core tiles each of the three times it is asked, with no build after
/// the first. Before that build `request_layout` answered `Building`, and a
/// panel's decode `Changed`.
#[test]
fn under_a_budget_of_0_a_panel_reads_a_build_held_in_memory_in_core() {
    let (_ws, paths, _, _) = blocked_workspace(1);
    let path = &paths[0];
    let key = TileKey {
        level: 1,
        col: 1,
        row: 0,
    };
    let cache = cache_of(0);
    cache.request_layout(path);
    assert_eq!(cache.layout(path), PeekedLayout::Building);
    assert_eq!(
        cache
            .try_decode_tiles(path, &[key], &Progress::none())
            .err(),
        Some(TileError::Changed { path: path.clone() })
    );

    let (found, tally) = cache
        .get_many(&[path.as_path()], &Progress::none())
        .unwrap();
    assert!(found[0].is_some());
    assert_eq!(tally.mipmaps_made, 1);
    cache.request_layout(path);
    match cache.layout(path) {
        PeekedLayout::Known(layout) => assert!(!layout.is_out_of_core()),
        other => panic!("{other:?}"),
    }
    let in_core = cut_image(ImageU8::read_rgb(path).unwrap(), TILE).tiles;
    for ask in 0..3 {
        let Ok(TryDecode::Decoded(tiles)) = cache.try_decode_tiles(path, &[key], &Progress::none())
        else {
            panic!("ask {ask}: a panel's decode");
        };
        assert!(same_tile(&tiles[0], &in_core[&key]), "ask {ask}");
    }
    let stats = cache.stats();
    assert_eq!((stats.decodes, stats.mipmaps_made), (3, 1));
}

/// A file a read found missing is made again by the next build, rather than
/// taken as current on the coarsest file's word, after which the photograph
/// reads; deleted again under the same original, it is a first find again,
/// and the next build makes it again, so a mipmap directory deleted twice is
/// made twice and the photograph stays readable. Under a budget too small to
/// keep a file, and under a budget of 0.
#[test]
fn a_file_a_read_found_missing_is_made_again_each_time() {
    for budget in [1, 0] {
        let ws = Workspace::new();
        let path = ws.photograph("a.jpg", &readable_jpeg());
        build(&path);
        let cache = cache_of(budget);
        cache.request_layout(&path);
        let level_1 = level_file(&path, 1);
        for round in 0..2u32 {
            let key = TileKey {
                level: 1,
                col: round,
                row: 0,
            };
            std::fs::remove_file(&level_1).unwrap();
            assert_eq!(
                cache
                    .try_decode_tiles(&path, &[key], &Progress::none())
                    .err(),
                Some(TileError::Changed {
                    path: level_1.clone()
                }),
                "{budget}, round {round}"
            );
            assert_eq!(cache.layout(&path), PeekedLayout::Building, "{budget}");

            let (found, tally) = cache
                .get_many(&[path.as_path()], &Progress::none())
                .unwrap();
            assert!(found[0].is_some(), "{budget}, round {round}");
            assert_eq!(tally.mipmaps_made, 1, "{budget}, round {round}: made again");
            assert!(level_1.exists(), "{budget}, round {round}");
            assert!(matches!(cache.layout(&path), PeekedLayout::Known(_)));
            assert!(
                matches!(
                    cache.try_decode_tiles(&path, &[key], &Progress::none()),
                    Ok(TryDecode::Decoded(_))
                ),
                "{budget}, round {round}: readable"
            );
        }
        assert_eq!(cache.stats().mipmaps_made, 2, "{budget}");
    }
}

/// Overwrite the entropy-coded bytes of the JPEG at `path` with 0xFE, in
/// place, keeping its markers, its byte stuffing and its record: the file
/// opens, and its record is current, but its scan does not decode.
fn spoil_scan(path: &Path) {
    let mut bytes = std::fs::read(path).unwrap();
    let sos = bytes.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
    let header = usize::from(u16::from_be_bytes([bytes[sos + 2], bytes[sos + 3]]));
    let end = bytes.len() - 2;
    let mut i = sos + 2 + header;
    while i < end {
        if bytes[i] == 0xFF {
            i += 2;
        } else {
            bytes[i] = 0xFE;
            i += 1;
        }
    }
    std::fs::write(path, &bytes).unwrap();
}

/// A file that is there with a current record but whose scan does not
/// decode is found not current by a read; the next build leaves it as it is,
/// since its record is current, and a read that finds it so again, unchanged,
/// makes the photograph unreadable, naming the build, rather than `Building`
/// again, so nothing makes its mipmap again and again. Under a budget too
/// small to keep a file, and under a budget of 0.
#[test]
fn a_file_no_build_mends_makes_the_photograph_unreadable() {
    let key = |col| TileKey {
        level: 1,
        col,
        row: 0,
    };
    for budget in [1, 0] {
        let ws = Workspace::new();
        let path = ws.photograph("a.jpg", &readable_jpeg());
        build(&path);
        let level_1 = level_file(&path, 1);
        spoil_scan(&level_1);
        let spoiled = std::fs::read(&level_1).unwrap();
        let cache = cache_of(budget);
        cache.request_layout(&path);
        assert_eq!(
            cache
                .try_decode_tiles(&path, &[key(0)], &Progress::none())
                .err(),
            Some(TileError::Changed {
                path: level_1.clone()
            }),
            "{budget}"
        );

        let (found, tally) = cache
            .get_many(&[path.as_path()], &Progress::none())
            .unwrap();
        assert!(found[0].is_some(), "{budget}");
        assert_eq!(tally.mipmaps_made, 1, "{budget}");
        assert_eq!(
            std::fs::read(&level_1).unwrap(),
            spoiled,
            "{budget}: left as it was"
        );

        match cache
            .try_decode_tiles(&path, &[key(1)], &Progress::none())
            .err()
        {
            Some(TileError::Unreadable { why, .. }) => {
                assert!(why.contains("made again"), "{budget}: {why}")
            }
            other => panic!("{budget}: {other:?}"),
        }
        assert!(matches!(cache.layout(&path), PeekedLayout::Unreadable(_)));
        let made = cache.stats().mipmaps_made;
        let _ = cache
            .get_many(&[path.as_path()], &Progress::none())
            .unwrap();
        assert_eq!(cache.stats().mipmaps_made, made, "{budget}: not made again");

        // Forgotten, the photograph is a first find again.
        cache.forget(&path);
        cache.request_layout(&path);
        assert_eq!(
            cache
                .try_decode_tiles(&path, &[key(2)], &Progress::none())
                .err(),
            Some(TileError::Changed {
                path: level_1.clone()
            }),
            "{budget}: forgotten"
        );
    }
}

/// The side of the photographs the `get_many` tests make many mipmaps of:
/// over the threshold of 64, so each is out of core, and small, so each
/// build is quick.
const MANY_SIZE: (u32, u32) = (96, 64);

/// `n` out-of-core photographs of [`MANY_SIZE`] in `ws`, none with a mipmap.
fn many_out_of_core(ws: &Workspace, n: usize) -> Vec<PathBuf> {
    let jpeg = encode_jpeg(
        &synthetic_rgb(MANY_SIZE.0, MANY_SIZE.1),
        JpegOptions::default(),
    );
    (0..n)
        .map(|i| ws.photograph(&format!("p{i:02}.jpg"), &jpeg))
        .collect()
}

/// The values `mipmaps_made` a `Progress` made with [`Progress::to`] of
/// this sink is told, in order.
struct MipmapsMade(Mutex<Vec<f64>>);

impl MipmapsMade {
    fn sink(&self, event: Event<'_>) {
        if let Event::Value {
            name: "mipmaps_made",
            value,
        } = event
        {
            self.0.lock().unwrap().push(value);
        }
    }
}

/// A `get_many` over 20 out-of-core photographs with no mipmap never has more
/// builds in progress than the build budget allows, set to three builds:
/// as each build starts, the build count holds a reservation for every build
/// in progress (those started less those made) and never more than three.
/// Its tally counts the 20 mipmaps made, and its `Progress` is told
/// `mipmaps_made` once a build, up to 20. With a budget and with a cache that
/// keeps nothing.
#[test]
fn get_many_makes_its_mipmaps_in_groups_within_the_build_budget() {
    let build_bytes = whole_decode_bytes(MANY_SIZE.0, MANY_SIZE.1);
    for budget in [GIB, 0] {
        let ws = Workspace::new();
        let photographs = many_out_of_core(&ws, 20);
        let paths: Vec<&Path> = photographs.iter().map(PathBuf::as_path).collect();
        let (cache, starts) = cache_watching_its_builds(
            PhotographCacheOptions {
                budget_bytes: budget,
                build_budget_bytes: 3 * build_bytes,
                tile_size: TILE,
            },
            |cache| cache.shared.mipmaps_made.load(Ordering::SeqCst),
        );
        let made = MipmapsMade(Mutex::new(Vec::new()));
        let sink = |event: Event<'_>| made.sink(event);
        let (found, tally) = cache.get_many(&paths, &Progress::to(&sink)).unwrap();
        assert!(found.iter().all(Option::is_some));
        assert_eq!(
            tally,
            GetManyTally {
                mipmaps_made: 20,
                ..GetManyTally::default()
            },
            "budget {budget}"
        );
        assert_eq!(starts.lock().unwrap().len(), 20);
        assert_within_the_build_budget(&starts, build_bytes, 3);
        let stats = cache.stats();
        assert_eq!((stats.mipmaps_made, stats.decodes), (20, 0));
        assert_eq!(stats.build_reserved_bytes, 0);
        let mut made = made.0.into_inner().unwrap();
        made.sort_by(f64::total_cmp);
        assert_eq!(made, (1..=20).map(f64::from).collect::<Vec<_>>());
    }
}

/// A `get_many` resolves each photograph's layout before it reserves: an
/// out-of-core photograph whose mipmap is current is answered from its files
/// with nothing reserved on the build count, and counted apart from those it
/// decoded in core, made or reused. Over one such photograph, one with no
/// mipmap, an in-core photograph and a missing file, the first call reads
/// one from its files, makes one mipmap and decodes one; a second call
/// reuses all three with a budget, and with one of 0, which keeps nothing,
/// reads the two out-of-core ones from their files and decodes the in-core
/// one again.
#[test]
fn get_many_reads_a_current_mipmap_without_reserving_and_counts_what_it_did() {
    for budget in [GIB, 0] {
        let ws = Workspace::new();
        let current = ws.photograph("current.jpg", &readable_jpeg());
        super::mipmap_tests::build(&current);
        let no_mipmap = ws.photograph("no_mipmap.jpg", &readable_jpeg());
        let in_core = ws.photograph(
            "in_core.jpg",
            &encode_jpeg(&synthetic_rgb(48, 32), JpegOptions::default()),
        );
        let gone = ws.dir.path().join("images").join("gone.jpg");
        let paths: Vec<&Path> = vec![&current, &no_mipmap, &in_core, &gone];
        let cache = cache_of(budget);
        let made = MipmapsMade(Mutex::new(Vec::new()));
        let sink = |event: Event<'_>| made.sink(event);
        let (found, tally) = cache.get_many(&paths, &Progress::to(&sink)).unwrap();
        assert_eq!(
            found.iter().map(Option::is_some).collect::<Vec<_>>(),
            [true, true, true, false]
        );
        assert_eq!(
            tally,
            GetManyTally {
                read: 1,
                from_mipmaps: 1,
                mipmaps_made: 1,
                reused: 0,
                unreadable: 1
            },
            "budget {budget}"
        );
        assert_eq!(*made.0.lock().unwrap(), [1.0]);
        let stats = cache.stats();
        assert_eq!((stats.decodes, stats.mipmaps_made), (1, 1));

        let (_, tally) = cache.get_many(&paths, &Progress::none()).unwrap();
        let again = if budget == 0 {
            GetManyTally {
                read: 1,
                from_mipmaps: 2,
                unreadable: 1,
                ..GetManyTally::default()
            }
        } else {
            GetManyTally {
                reused: 3,
                unreadable: 1,
                ..GetManyTally::default()
            }
        };
        assert_eq!(tally, again, "budget {budget}");
        assert_eq!(cache.stats().mipmaps_made, 1);
        assert_eq!(
            cache.shared.reserved_reading_files.load(Ordering::SeqCst),
            0,
            "a photograph read from its files reserved a build"
        );
    }
}

/// A `get_many` that finds a photograph's mipmap being made by another
/// caller waits on the slot rather than start a second build, holding its
/// share while it waits, and reads what the other caller made, counted as
/// reused rather than made.
#[test]
fn get_many_waits_on_a_build_another_caller_is_making() {
    let ws = Workspace::new();
    let path = ws.photograph("a.jpg", &readable_jpeg());
    let held = cache_holding_every_build();
    let maker = {
        let (cache, path) = (Arc::clone(&held.cache), path.clone());
        on_a_thread(move || cache.get(&path, &Progress::none()))
    };
    let build_bytes = whole_decode_bytes(300, 200);
    assert_eq!(held.expect(SlotEvent::DecodeStarted), build_bytes);
    let many = {
        let (cache, path) = (Arc::clone(&held.cache), path.clone());
        on_a_thread(move || {
            let made = MipmapsMade(Mutex::new(Vec::new()));
            let sink = |event: Event<'_>| made.sink(event);
            let (found, tally) = cache
                .get_many(&[path.as_path()], &Progress::to(&sink))
                .unwrap();
            (found[0].is_some(), tally, made.0.into_inner().unwrap())
        })
    };
    assert_eq!(
        held.expect(SlotEvent::WaitStarted),
        2 * build_bytes,
        "the waiting member keeps its share"
    );
    held.release.send(()).unwrap();
    maker.recv_timeout(HANG).expect("never made").unwrap();
    let (answered, tally, made) = many.recv_timeout(HANG).expect("never answered");
    assert!(answered);
    assert_eq!(
        tally,
        GetManyTally {
            reused: 1,
            ..GetManyTally::default()
        }
    );
    assert!(made.is_empty(), "{made:?}");
    let stats = held.cache.stats();
    assert_eq!((stats.mipmaps_made, stats.build_reserved_bytes), (1, 0));
    assert!(held.events.try_recv().is_err(), "one build started");
}
