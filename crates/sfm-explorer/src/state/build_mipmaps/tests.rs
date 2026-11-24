// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! *Build mipmaps* in the queue: one chunk a run, queued again while its
//! list is not empty, giving way to a task someone asks for, and running
//! while a slow task holds a node. The photographs are 300 x 200 PNGs in a
//! temporary workspace whose `mipmap_options` puts them out of core, read
//! with tiles of 32 px.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use sfmtool_core::camera::{MipmapOptions, PeekedLayout, PhotographCache, PhotographCacheOptions};
use sfmtool_core::progress::Progress;
use sfmtool_core::SfmrReconstruction;

use super::{lock, run_chunk};
use crate::background::{Finished, Job, Operation, Queue};
use crate::scene::{ReconId, SceneNode};
use crate::state::{AppState, PhotographRequests};

/// The `mipmap_options` of [`out_of_core_workspace`]: a photograph with a
/// side over 64 px is out of core.
pub(crate) fn mipmap_options() -> MipmapOptions {
    MipmapOptions {
        out_of_core_long_side: 64,
        jpeg_quality: 95,
        full_resolution_jpeg_quality: 95,
    }
}

/// A temporary workspace whose `mipmap_options` puts a photograph with a
/// side over 64 px out of core, holding `n` 300 x 200 PNGs, `0.png` on,
/// none of which has a mipmap. A PNG is not region-readable, so a mipmap of
/// one has a `-mip0`.
pub(crate) fn out_of_core_workspace(n: usize) -> (tempfile::TempDir, Vec<PathBuf>) {
    let dir = tempfile::tempdir().unwrap();
    let o = mipmap_options();
    std::fs::write(
        dir.path()
            .join(sfmtool_core::workspace::WORKSPACE_FILE_NAME),
        format!(
            r#"{{"version": 2, "mipmap_options": {{"out_of_core_long_side": {},
            "jpeg_quality": {}, "full_resolution_jpeg_quality": {}}}}}"#,
            o.out_of_core_long_side, o.jpeg_quality, o.full_resolution_jpeg_quality
        ),
    )
    .unwrap();
    let paths = (0..n)
        .map(|i| {
            let path = dir.path().join(format!("{i}.png"));
            image::RgbImage::from_fn(300, 200, |x, y| {
                image::Rgb([(x * 7) as u8, (y * 5 + i as u32) as u8, (x ^ y) as u8])
            })
            .save(&path)
            .unwrap();
            path
        })
        .collect();
    (dir, paths)
}

/// The estimated bytes of a whole decode of one of
/// [`out_of_core_workspace`]'s photographs, which a build reserves: 5 bytes
/// a pixel.
const BUILD_BYTES: u64 = 300 * 200 * 5;

/// A cache reading tiles of 32 px, keeping `budget_bytes`, whose build budget
/// fits `builds` builds of one of [`out_of_core_workspace`]'s photographs at
/// once.
pub(crate) fn cache_for(budget_bytes: u64, builds: u64) -> Arc<PhotographCache> {
    Arc::new(PhotographCache::new(PhotographCacheOptions {
        budget_bytes,
        build_budget_bytes: builds * BUILD_BYTES,
        tile_size: 32,
    }))
}

/// How many files are under `dir`'s mipmap directory.
pub(crate) fn mipmap_file_count(dir: &Path) -> usize {
    let mut count = 0;
    let mut stack = vec![dir.join(sfmtool_core::camera::mipmap::MIPMAP_DIR_NAME)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                count += 1;
            }
        }
    }
    count
}

/// Queue `paths` on `requests`' list in order, as panels asking for them one
/// after another do: the last first.
fn queue_all(requests: &PhotographRequests, paths: &[PathBuf]) {
    let mut list = lock(&requests.mipmaps);
    for path in paths {
        list.queue(path);
    }
}

/// A state with one demo node, and that node's id.
fn state_with_a_node() -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.append_node(SceneNode::demo(SfmrReconstruction::demo(32)));
    let id = state.scene[0].id;
    (state, id)
}

/// A job that waits until `gate` opens, or its sender is dropped, and
/// produces nothing.
fn held_job(gate: mpsc::Receiver<()>) -> Job {
    Box::new(move |_progress| {
        let _ = gate.recv();
        Finished::Failed("the fake worker produced nothing".to_string())
    })
}

/// Each run of *Build mipmaps* runs in the fast queue, named with how many
/// photographs it has answered of how many are queued, and takes one chunk
/// from the front of the list: as many photographs as the build budget fits
/// (two here) of the three queued, so two runs, queued again while the list
/// is not empty. It runs while a slow task holds the node on screen, and
/// pushes no version of it. Every photograph's mipmap is made and its layout
/// is then known.
#[test]
fn the_task_makes_one_chunk_a_run_and_is_queued_again_while_its_list_is_not_empty() {
    let (dir, paths) = out_of_core_workspace(3);
    let (mut state, id) = state_with_a_node();
    let versions = state.scene[0].history.versions().len();
    let (slow_gate, held) = mpsc::channel();
    let slow = state
        .start_background_task(Operation::BUNDLE_ADJUST, id, held_job(held))
        .expect("an idle slow queue");
    assert!(state.busy_refusal(id).is_some(), "the node is locked");

    let cache = cache_for(1 << 30, 2);
    state.photographs = Arc::clone(&cache);
    let chunks = Arc::new(Mutex::new(Vec::new()));
    let told = Arc::clone(&chunks);
    state
        .photograph_requests
        .set_chunk_hook(Arc::new(move |chunk: &[PathBuf]| {
            told.lock().unwrap().push(chunk.to_vec());
        }));
    queue_all(&state.photograph_requests, &paths);

    let mut labels = Vec::new();
    for _ in 0..10 {
        state.drive_build_mipmaps();
        if !state.build_mipmaps_queued() {
            break;
        }
        let run = state
            .quiet_running_in(Queue::Fast)
            .expect("a run starts in the idle fast queue, on a locked node");
        assert_eq!(run.operation, Operation::BUILD_MIPMAPS);
        labels.push(run.label.clone());
        state.land_quiet_tasks();
    }
    assert_eq!(
        labels,
        ["Building mipmaps: 0 of 3", "Building mipmaps: 2 of 3"]
    );
    let chunks = chunks.lock().unwrap().clone();
    assert_eq!(
        chunks,
        [
            vec![paths[2].clone(), paths[1].clone()],
            vec![paths[0].clone()]
        ],
        "the front of the list, as many as the build budget fits"
    );
    assert_eq!(cache.stats().mipmaps_made, 3);
    for path in &paths {
        assert!(matches!(cache.layout(path), PeekedLayout::Known(_)));
    }
    assert!(mipmap_file_count(dir.path()) > 0);
    assert!(state.photograph_requests.mipmaps_waiting().is_empty());

    assert_eq!(state.scene[0].history.versions().len(), versions);
    assert!(state.running_task(slow).is_some(), "the slow task held on");
    drop(slow_gate);
    state.finish_background_task(slow);
}

/// A run that a task someone asks for in the fast queue displaces puts the
/// photographs of its chunk back at the front of the list, in their order,
/// having made none of them and holding nothing on the build count; the
/// run queued again waits behind the displacing task and, once that has
/// landed, takes the same chunk first and makes every photograph.
#[test]
fn a_run_gives_way_to_an_asked_for_task_and_puts_its_chunk_back_at_the_front() {
    let (_dir, paths) = out_of_core_workspace(3);
    let (mut state, id) = state_with_a_node();
    let cache = cache_for(1 << 30, 2);
    state.photographs = Arc::clone(&cache);
    // The first run is held once it has taken its chunk, until the test
    // has asked for a task.
    let (taken_tx, taken) = mpsc::channel::<Vec<PathBuf>>();
    let (release, released) = mpsc::channel::<()>();
    let released = Mutex::new(Some(released));
    state
        .photograph_requests
        .set_chunk_hook(Arc::new(move |chunk: &[PathBuf]| {
            let _ = taken_tx.send(chunk.to_vec());
            if let Some(released) = released.lock().unwrap().take() {
                let _ = released.recv();
            }
        }));
    queue_all(&state.photograph_requests, &paths);

    state.drive_build_mipmaps();
    let first = taken.recv().expect("the run took a chunk");
    assert_eq!(first, [paths[2].clone(), paths[1].clone()]);
    let (fit_gate, held) = mpsc::channel();
    let fit = state
        .start_background_task(Operation::BENCH_FIT, id, held_job(held))
        .expect("a busy fast queue queues the task");
    release.send(()).unwrap();
    state.land_quiet_tasks();

    assert_eq!(
        state.photograph_requests.mipmaps_waiting(),
        [paths[2].clone(), paths[1].clone(), paths[0].clone()],
        "the chunk is back at the front, in its order"
    );
    let stats = cache.stats();
    assert_eq!(stats.mipmaps_made, 0);
    assert_eq!(
        stats.build_reserved_bytes, 0,
        "nothing holds the build count"
    );
    assert!(
        state.running_task(fit).is_some(),
        "the asked-for task started"
    );

    // Queued again, behind the task that displaced it.
    state.drive_build_mipmaps();
    assert!(state.build_mipmaps_queued());
    assert!(state.quiet_running_in(Queue::Fast).is_none(), "it waits");
    drop(fit_gate);
    state.finish_background_task(fit);
    assert_eq!(
        taken.recv().expect("the run queued again started"),
        [paths[2].clone(), paths[1].clone()]
    );
    for _ in 0..10 {
        state.land_quiet_tasks();
        state.drive_build_mipmaps();
        if !state.build_mipmaps_queued() {
            break;
        }
    }
    assert_eq!(cache.stats().mipmaps_made, 3);
    for path in &paths {
        assert!(matches!(cache.layout(path), PeekedLayout::Known(_)));
    }
}

/// A kernel's `get_many` on its own thread, over photographs a run of
/// *Build mipmaps* is making at the same time, makes each mipmap at most
/// once between them: whichever finds a photograph's slot being made waits
/// on it and reads what the other made, rather than make a second.
#[test]
fn a_kernel_and_the_task_make_each_mipmap_once_between_them() {
    let (dir, paths) = out_of_core_workspace(4);
    let cache = cache_for(1 << 30, 4);
    let (requests, _) = PhotographRequests::queued();
    queue_all(&requests, &paths);
    let start = Arc::new(std::sync::Barrier::new(2));
    let kernel = {
        let (cache, start, paths) = (Arc::clone(&cache), Arc::clone(&start), paths.clone());
        std::thread::spawn(move || {
            let paths: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
            start.wait();
            let (found, _) = cache.get_many(&paths, &Progress::none()).unwrap();
            found.iter().all(Option::is_some)
        })
    };
    start.wait();
    // One chunk is as many as the rayon pool runs at once; the rest follow.
    while !requests.mipmaps_waiting().is_empty() {
        run_chunk(&cache, &requests, &Progress::none());
    }
    assert!(kernel.join().unwrap(), "the kernel read every photograph");
    assert_eq!(cache.stats().mipmaps_made, 4);
    assert!(mipmap_file_count(dir.path()) > 0);
    assert!(requests.mipmaps_waiting().is_empty());
}

/// While a run of *Build mipmaps* runs, the Background Tasks panel's quiet
/// line under the fast queue says what it does and how far it has got, as
/// *Building mipmaps: {done} of {total}*, right after the word `quiet`, with
/// no operation name or node, since it is queued on none.
#[test]
fn the_quiet_line_says_how_many_mipmaps_are_built_of_how_many() {
    let (_dir, paths) = out_of_core_workspace(2);
    let mut state = AppState::new();
    state.photographs = cache_for(1 << 30, 1);
    let (release, released) = mpsc::channel::<()>();
    let released = Mutex::new(released);
    state
        .photograph_requests
        .set_chunk_hook(Arc::new(move |_: &[PathBuf]| {
            let _ = released.lock().unwrap().recv();
        }));
    queue_all(&state.photograph_requests, &paths);
    state.drive_build_mipmaps();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(260.0, 400.0),
        )),
        ..Default::default()
    };
    let ctx = egui::Context::default();
    let texts = crate::test_support::painted_texts(&ctx, input, |ui| {
        crate::background::panel::show(ui, &mut state)
    });
    let at = texts
        .iter()
        .position(|t| t == "quiet")
        .unwrap_or_else(|| panic!("no quiet line: {texts:?}"));
    assert_eq!(texts[at + 1], "Building mipmaps: 0 of 2", "{texts:?}");
    drop(release);
    state.land_quiet_tasks();
}
