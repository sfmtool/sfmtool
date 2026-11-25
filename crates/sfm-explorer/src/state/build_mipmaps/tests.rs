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
    write_out_of_core_workspace_file(dir.path());
    let paths = (0..n)
        .map(|i| write_out_of_core_photo(&dir.path().join(format!("{i}.png")), i))
        .collect();
    (dir, paths)
}

/// Write the workspace file of a workspace at `dir` whose `mipmap_options`
/// is [`mipmap_options`].
fn write_out_of_core_workspace_file(dir: &Path) {
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
}

/// Write the `i`th of [`out_of_core_workspace`]'s 300 x 200 PNGs at `path`.
fn write_out_of_core_photo(path: &Path, i: usize) -> PathBuf {
    image::RgbImage::from_fn(300, 200, |x, y| {
        image::Rgb([(x * 7) as u8, (y * 5 + i as u32) as u8, (x ^ y) as u8])
    })
    .save(path)
    .unwrap();
    path.to_path_buf()
}

/// Make `recon`'s workspace `dir`, a workspace whose `mipmap_options` puts
/// a photograph with a side over 64 px out of core, its images named
/// `image_000.png` on, and write the photographs of `images` there as
/// [`out_of_core_workspace`]'s 300 x 200 PNGs, none with a mipmap; answer
/// their paths. The photographs are smaller than the cameras say, so a
/// picture read from one once its mipmap is made may land off it, which
/// the panels draw as they draw any picture with nothing to cut.
pub(crate) fn out_of_core_photographs_of(
    recon: &mut SfmrReconstruction,
    dir: &Path,
    images: &[usize],
) -> Vec<PathBuf> {
    recon.workspace_dir = dir.to_path_buf();
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.name = format!("image_{i:03}.png");
    }
    write_out_of_core_workspace_file(dir);
    images
        .iter()
        .map(|&i| write_out_of_core_photo(&dir.join(format!("image_{i:03}.png")), i))
        .collect()
}

/// Run *Build mipmaps* on this thread, a chunk a run, until its list is
/// empty, answering how many runs it took.
pub(crate) fn run_until_empty(cache: &PhotographCache, requests: &PhotographRequests) -> usize {
    let mut runs = 0;
    while !requests.mipmaps_waiting().is_empty() {
        run_chunk(cache, requests, &Progress::none(), None);
        runs += 1;
        assert!(runs <= 64, "the list never emptied");
    }
    runs
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
        run_chunk(&cache, &requests, &Progress::none(), None);
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

/// How long a held run of *Build mipmaps* waits to be released before it
/// goes on by itself, so a test that fails before releasing it ends.
const HANG: std::time::Duration = std::time::Duration::from_secs(60);

/// A run of *Build mipmaps* on a thread of its own, held once a build of its
/// chunk has published its photograph's coarsest level, until the test sends
/// on `release`.
pub(crate) struct HeldRun {
    /// Told when the coarsest level is published, before the run is held.
    pub(crate) published: mpsc::Receiver<()>,
    pub(crate) release: mpsc::Sender<()>,
    /// Told when the run has ended.
    pub(crate) ended: mpsc::Receiver<()>,
}

impl HeldRun {
    /// Wait until the coarsest level is published.
    pub(crate) fn wait_published(&self) {
        self.published
            .recv_timeout(HANG)
            .expect("a build published its coarsest level");
    }

    /// Let the run go on, and wait until it has ended.
    pub(crate) fn finish(self) {
        let _ = self.release.send(());
        self.ended.recv_timeout(HANG).expect("the run ended");
    }
}

/// Start a run of *Build mipmaps* over `requests`' list, held as
/// [`HeldRun`] says.
pub(crate) fn run_held_once_published(
    cache: &Arc<PhotographCache>,
    requests: &PhotographRequests,
) -> HeldRun {
    let (published_tx, published) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let (ended_tx, ended) = mpsc::channel();
    let (cache, requests) = (Arc::clone(cache), requests.clone());
    std::thread::spawn(move || {
        let released = Mutex::new(released);
        let wake = move || {
            let _ = published_tx.send(());
            let _ = released.lock().unwrap().recv_timeout(HANG);
        };
        run_chunk(&cache, &requests, &Progress::none(), Some(&wake));
        let _ = ended_tx.send(());
    });
    HeldRun {
        published,
        release,
        ended,
    }
}

/// A run of *Build mipmaps* wakes the viewer once for each photograph of its
/// chunk whose coarsest level a build published, so the panels waiting on it
/// draw that level before the run ends: three photographs, three wakes, one
/// each, with a budget smaller than a photograph and with one of 0; and a
/// run over photographs whose mipmaps are current builds nothing and wakes
/// no one.
#[test]
fn a_run_wakes_the_viewer_once_for_each_coarsest_level_it_publishes() {
    for budget in [20_000, 0] {
        let (_dir, paths) = out_of_core_workspace(3);
        let cache = cache_for(budget, 3);
        let (requests, _) = PhotographRequests::queued();
        let woken = std::sync::atomic::AtomicUsize::new(0);
        let wake = || {
            woken.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        };
        for round in ["made", "current"] {
            woken.store(0, std::sync::atomic::Ordering::Relaxed);
            queue_all(&requests, &paths);
            while !requests.mipmaps_waiting().is_empty() {
                run_chunk(&cache, &requests, &Progress::none(), Some(&wake));
            }
            let expected = if round == "made" { 3 } else { 0 };
            assert_eq!(
                woken.load(std::sync::atomic::Ordering::Relaxed),
                expected,
                "budget {budget}, {round}"
            );
        }
        assert_eq!(cache.stats().mipmaps_made, 3, "budget {budget}");
    }
}

/// A reconstruction of eight images saved as `recon.sfmr` in a temporary
/// workspace whose `mipmap_options` puts a photograph with a side over 64 px
/// out of core: images 0 to 4 are [`out_of_core_workspace`]'s 300 x 200
/// PNGs with no mipmap, image 5 is in core (40 x 30), image 6 is out of
/// core with a current mipmap, and image 7 has no file. Answers the
/// workspace, the file and the paths of images 0 to 4.
fn saved_out_of_core_recon() -> (tempfile::TempDir, PathBuf, Vec<PathBuf>) {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = SfmrReconstruction::demo(32);
    assert_eq!(recon.image_set.images.len(), 8);
    let mut paths = out_of_core_photographs_of(&mut recon, dir.path(), &[0, 1, 2, 3, 4, 6]);
    image::RgbImage::from_pixel(40, 30, image::Rgb([90, 120, 150]))
        .save(dir.path().join("image_005.png"))
        .unwrap();
    let current = paths.pop().unwrap();
    let (made, _) = cache_for(1 << 30, 1)
        .get_many(&[current.as_path()], &Progress::none())
        .unwrap();
    assert!(made[0].is_some(), "image 6's mipmap is made");
    let path = dir.path().join("recon.sfmr");
    recon.save(&path, &Progress::none()).unwrap();
    (dir, path, paths)
}

/// A state whose photograph cache makes one build at a time, so each run of
/// *Build mipmaps* takes one photograph, with `saved` open.
fn opened(saved: &Path) -> (AppState, ReconId) {
    let mut state = AppState::new();
    state.photographs = cache_for(1 << 30, 1);
    let id = state.open_now(saved).expect("the file opens");
    (state, id)
}

/// Record the file names of each chunk *Build mipmaps* takes in `events`,
/// holding the first run once it has taken its chunk until the answered
/// sender is sent on or dropped.
fn record_chunks(state: &mut AppState, events: &Arc<Mutex<Vec<String>>>) -> mpsc::Sender<()> {
    let (release, released) = mpsc::channel::<()>();
    let released = Mutex::new(Some(released));
    let events = Arc::clone(events);
    state
        .photograph_requests
        .set_chunk_hook(Arc::new(move |chunk: &[PathBuf]| {
            let names: Vec<String> = chunk
                .iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
                .collect();
            events.lock().unwrap().push(names.join(","));
            if let Some(released) = released.lock().unwrap().take() {
                let _ = released.recv_timeout(HANG);
            }
        }));
    release
}

/// An open in a workspace with `mipmap_options` puts the reconstruction's
/// out-of-core photographs whose mipmaps are not current on *Build
/// mipmaps*' list, in image order, and not one that is in core, has a
/// current mipmap or has no file; its `mipmaps` phase says how many of the
/// photographs it read need their mipmaps. The open has read their
/// layouts, so the panel showing an image selected before the task's first
/// run starts (as when the fast queue is busy when the open lands) finds it
/// building at once and puts it at the front, and the first run, named with
/// the whole list, makes that image's mipmap.
#[test]
fn an_open_fills_the_list_and_the_first_run_makes_the_selected_images_mipmap() {
    let (_dir, saved, paths) = saved_out_of_core_recon();
    let (mut state, id) = opened(&saved);
    assert_eq!(state.photograph_requests.mipmaps_waiting(), paths);
    let entry = state.action_log.entries().next_back().unwrap();
    assert_eq!(
        crate::test_support::phase_note(&entry.detail, "mipmaps").as_deref(),
        Some("5 of 8 photographs need their mipmaps made")
    );

    state.select_image(Some(crate::scene::ImageRef {
        recon: id,
        image: 3,
    }));
    let ctx = egui::Context::default();
    let layout = crate::state::display_layout(
        crate::state::PhotographSource {
            cache: &state.photographs,
            requests: &state.photograph_requests,
            path: &paths[3],
        },
        &ctx,
    );
    assert!(matches!(layout, crate::state::DisplayLayout::Building(_)));
    let events = Arc::new(Mutex::new(Vec::new()));
    drop(record_chunks(&mut state, &events));

    state.drive_build_mipmaps();
    let run = state
        .quiet_running_in(Queue::Fast)
        .expect("the run started");
    assert_eq!(run.label, "Building mipmaps: 0 of 5");
    state.land_quiet_tasks();
    assert_eq!(*events.lock().unwrap(), ["image_003.png"]);
    assert!(matches!(
        state.photographs.layout(&paths[3]),
        PeekedLayout::Known(_)
    ));
    assert_eq!(state.photographs.stats().mipmaps_made, 1);
}

/// A photograph a panel queues while a run is making another goes to the
/// front of the list, ahead of the open's, and the next run takes it; the
/// running one is not asked to stop and makes its own.
#[test]
fn a_photograph_a_panel_queues_is_taken_by_the_next_run_and_the_running_one_goes_on() {
    let (_dir, saved, paths) = saved_out_of_core_recon();
    let (mut state, _) = opened(&saved);
    let events = Arc::new(Mutex::new(Vec::new()));
    let release = record_chunks(&mut state, &events);
    state.drive_build_mipmaps();
    let run = state
        .quiet_running_in(Queue::Fast)
        .expect("the run started")
        .id;
    let waited = std::time::Instant::now();
    while events.lock().unwrap().is_empty() {
        assert!(waited.elapsed() < HANG, "the run took no chunk");
        std::thread::yield_now();
    }

    lock(&state.photograph_requests.mipmaps).queue(&paths[4]);
    assert_eq!(state.photograph_requests.mipmaps_waiting()[0], paths[4]);
    assert!(!state.quiet_task_cancelled(run));
    release.send(()).unwrap();
    state.land_quiet_tasks();
    assert!(matches!(
        state.photographs.layout(&paths[0]),
        PeekedLayout::Known(_)
    ));
    state.drive_build_mipmaps();
    state.land_quiet_tasks();
    assert_eq!(*events.lock().unwrap(), ["image_000.png", "image_004.png"]);
    assert_eq!(state.photographs.stats().mipmaps_made, 2);
}

/// A camera model proposal's job, queued while a run of *Build mipmaps*
/// makes the first of a long list, waits for that run rather than stop it,
/// and starts before the next run: the task ends after each chunk.
#[test]
fn a_waiting_proposal_job_starts_between_two_chunks_of_a_long_list() {
    let (_dir, saved, paths) = saved_out_of_core_recon();
    let (mut state, id) = opened(&saved);
    let events = Arc::new(Mutex::new(Vec::new()));
    let release = record_chunks(&mut state, &events);
    state.drive_build_mipmaps();
    let run = state
        .quiet_running_in(Queue::Fast)
        .expect("the run started")
        .id;

    let told = Arc::clone(&events);
    state
        .start_quiet_task(
            Operation::CAMERA_MODEL_PROPOSAL,
            crate::background::QuietSource::CameraModelProposal,
            crate::background::QuietRank::CameraModelProposal,
            Some(id),
            "proposal".to_string(),
            crate::background::QuietJob::new(
                move |_: &Progress<'_>| told.lock().unwrap().push("proposal".to_string()),
                || {},
            ),
        )
        .expect("the job is queued");
    assert!(!state.quiet_task_cancelled(run), "the run goes on");
    release.send(()).unwrap();
    state.land_quiet_tasks();
    state.drive_build_mipmaps();
    state.land_quiet_tasks();
    assert_eq!(
        *events.lock().unwrap(),
        ["image_000.png", "proposal", "image_001.png"]
    );
    assert_eq!(
        state.photograph_requests.mipmaps_waiting(),
        paths[2..],
        "the list is still long"
    );
}

/// Closing a node takes the photographs only it has off the list and keeps
/// those another node in the scene has: the same file opened twice keeps
/// them until both are closed, and Close All empties the list.
#[test]
fn closing_a_node_takes_off_the_list_the_photographs_no_other_node_has() {
    let (_dir, saved, paths) = saved_out_of_core_recon();
    let (mut state, first) = opened(&saved);
    let second = state.open_now(&saved).expect("the file opens again");
    assert_eq!(state.photograph_requests.mipmaps_waiting(), paths);
    state.close_node(first).unwrap();
    assert_eq!(state.photograph_requests.mipmaps_waiting(), paths);
    state.close_node(second).unwrap();
    assert!(state.photograph_requests.mipmaps_waiting().is_empty());

    state.open_now(&saved).expect("the file opens");
    assert_eq!(state.photograph_requests.mipmaps_waiting(), paths);
    state.close_all().unwrap();
    assert!(state.photograph_requests.mipmaps_waiting().is_empty());
    state.drive_build_mipmaps();
    assert!(!state.build_mipmaps_queued(), "nothing is made");
}

/// An open in a workspace whose file has no `mipmap_options` queues nothing
/// and reads no photograph's layout, though its photographs are over the
/// threshold the other tests' workspaces set, and has no `mipmaps` phase.
#[test]
fn an_open_in_a_workspace_without_mipmap_options_queues_nothing() {
    let (dir, saved, paths) = saved_out_of_core_recon();
    std::fs::write(
        dir.path()
            .join(sfmtool_core::workspace::WORKSPACE_FILE_NAME),
        r#"{"version": 2}"#,
    )
    .unwrap();
    let (state, _) = opened(&saved);
    assert!(state.photograph_requests.mipmaps_waiting().is_empty());
    for path in &paths {
        assert_eq!(state.photographs.layout(path), PeekedLayout::NotKnown);
    }
    let entry = state.action_log.entries().next_back().unwrap();
    assert!(crate::test_support::phase_rows(&entry.detail)
        .iter()
        .all(|(name, ..)| *name != "mipmaps"));
}

/// The list's order: what panels queued first, the last first, then each
/// open's photographs in their order, the newest open's first; a run's
/// cancelled chunk goes back ahead of every open's.
#[test]
fn an_opens_photographs_wait_behind_the_panels_and_ahead_of_an_earlier_opens() {
    let p = |name: &str| PathBuf::from(name);
    let mut list = super::MipmapList::default();
    list.queue_opened(vec![p("b"), p("c")]);
    list.queue(&p("a"));
    list.queue_opened(vec![p("d"), p("a")]);
    assert_eq!(list.waiting(), [p("a"), p("d"), p("b"), p("c")]);
    let taken = list.take(2);
    list.put_back(taken);
    list.queue_opened(vec![p("e")]);
    assert_eq!(list.waiting(), [p("a"), p("d"), p("e"), p("b"), p("c")]);
}

/// A photograph a running chunk took, dropped from the list by a close and
/// queued again before the run ends, by an open or by a panel, is never on
/// the list twice: a run that makes it leaves no copy waiting, and one that
/// is cancelled puts it back once. One no one queues again is not put back,
/// and the quiet line counts it neither as done nor in the total.
#[test]
fn a_photograph_queued_again_while_its_chunk_runs_is_on_the_list_once() {
    let x = || PathBuf::from("x");
    type Requeue = fn(&mut super::MipmapList);
    let requeues: [(&str, Requeue); 2] = [
        ("open", |list| list.queue_opened(vec![PathBuf::from("x")])),
        ("panel", |list| list.queue(Path::new("x"))),
    ];
    for (by, requeue) in requeues {
        let mut list = super::MipmapList::default();
        list.queue_opened(vec![x()]);
        list.take(1);
        list.keep_only(|_| false);
        requeue(&mut list);
        list.answered(&x());
        assert!(list.waiting().is_empty(), "made, queued again by {by}");
        list.queue(&x());
        assert_eq!(list.waiting(), [x()], "made, then a panel, {by}");

        let mut list = super::MipmapList::default();
        list.queue_opened(vec![x()]);
        let taken = list.take(1);
        list.keep_only(|_| false);
        requeue(&mut list);
        list.put_back(taken);
        assert_eq!(list.waiting(), [x()], "put back, queued again by {by}");
        list.queue(&x());
        assert_eq!(list.waiting(), [x()], "put back, then a panel, {by}");
    }

    let mut list = super::MipmapList::default();
    list.queue_opened(vec![x()]);
    let taken = list.take(1);
    list.keep_only(|_| false);
    list.put_back(taken);
    assert!(list.waiting().is_empty(), "dropped and not queued again");
    list.queue(&x());
    assert_eq!(list.waiting(), [x()], "a panel queues it afresh");

    let mut list = super::MipmapList::default();
    list.queue_opened(vec![PathBuf::from("a"), x(), PathBuf::from("y")]);
    list.take(1);
    list.answered(Path::new("a"));
    list.take(1);
    list.keep_only(|path| path != Path::new("x"));
    assert_eq!(list.progress_label(), "Building mipmaps: 1 of 2");
    list.answered(&x());
    assert_eq!(list.progress_label(), "Building mipmaps: 1 of 2");
}
