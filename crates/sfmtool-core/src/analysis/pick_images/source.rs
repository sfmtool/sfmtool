// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Where [`super::pick_images`] gets each frame's `.sift` files: the
//! [`FrameSource`] it asks, the [`ExtractingSource`] that extracts missing
//! files on a pool of workers ahead of the walk, and the [`ReadOnlySource`]
//! that only checks that the files are there. See
//! `specs/core/analysis/pick-images.md` § "Sources".

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::capture::{CaptureFrame, CaptureImage, FrameId};
use crate::features::sift::{
    build_sift_file, default_concurrent_images, sift_file_up_to_date, BuildSiftFilesError,
    BuildSiftFilesOptions, SiftParams,
};
use crate::progress::Progress;

/// How often a [`FrameSource::wait`] that is still waiting reads the cancel
/// flag and reports, so that a cancellation stops it within this time.
const WAIT_POLL: Duration = Duration::from_millis(50);

/// Makes the `.sift` files of a capture's frames exist for
/// [`super::pick_images`].
///
/// The walk tells the source the frames it expects to read with
/// [`prefetch`](Self::prefetch), which starts the work and returns at once,
/// and blocks in [`wait`](Self::wait) for the frames it is about to read. It
/// decides only from frames it waited for, so what a source does ahead of the
/// walk changes how long a run takes and never what it chooses.
pub trait FrameSource {
    /// How many frames the source works on at once, which sizes how far the
    /// walk predicts ahead; 0 for a source that does no work ahead, whose
    /// [`prefetch`](Self::prefetch) does nothing.
    fn concurrency(&self) -> usize;

    /// Start making the `.sift` files of `frames` exist, in this order and
    /// after the frames a [`wait`](Self::wait) is blocked on, and return at
    /// once. The frames of an earlier call that are not in `frames` and not
    /// yet started are dropped: each call replaces the last prediction.
    fn prefetch(&mut self, frames: &[(FrameId, &CaptureFrame)]);

    /// Block until the `.sift` file of every image of every frame in
    /// `frames` exists, starting those not yet started ahead of every
    /// prefetched frame.
    ///
    /// # Errors
    ///
    /// The error of the first frame of `frames`, in their order, that the
    /// source could not make exist, and [`FrameSourceError::Cancelled`] when
    /// `progress` is cancelled while it waits.
    fn wait(
        &mut self,
        frames: &[(FrameId, &CaptureFrame)],
        progress: &Progress,
    ) -> Result<(), FrameSourceError>;

    /// Drop the work no [`wait`](Self::wait) asked for, let the work in
    /// flight stop or finish, and return every frame whose `.sift` files the
    /// source wrote since it was made or last finished, sorted.
    fn finish(&mut self) -> Vec<FrameId>;
}

/// Why a [`FrameSource`] could not make a frame's `.sift` files exist.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameSourceError {
    /// A [`ReadOnlySource`] found no `.sift` file for an image of the frame.
    Missing { frame: FrameId, path: PathBuf },
    /// An [`ExtractingSource`] was given an image whose `.sift` path is not
    /// the one its options put the image's file at.
    SiftPath {
        frame: FrameId,
        sift_path: PathBuf,
        expected: PathBuf,
    },
    /// Extracting an image of the frame failed: its image could not be read
    /// or decoded ([`BuildSiftFilesError::Image`]) or its `.sift` file
    /// could not be written ([`BuildSiftFilesError::Write`]).
    Extract {
        frame: FrameId,
        error: BuildSiftFilesError,
    },
    /// The work on the frame panicked, which is a bug; the message is the
    /// panic's.
    Panicked { frame: FrameId, message: String },
    /// The progress's cancel flag was set while the source waited.
    Cancelled,
}

impl fmt::Display for FrameSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { path, .. } => write!(
                f,
                "{} does not exist, and the frame's .sift files are read, not extracted",
                path.display()
            ),
            Self::SiftPath {
                sift_path,
                expected,
                ..
            } => write!(
                f,
                "the .sift path {} is not {}, where the extraction writes it",
                sift_path.display(),
                expected.display()
            ),
            Self::Extract { error, .. } => write!(f, "{error}"),
            Self::Panicked { message, .. } => {
                write!(f, "extracting a frame panicked: {message}")
            }
            Self::Cancelled => f.write_str("picking images was cancelled"),
        }
    }
}

impl std::error::Error for FrameSourceError {}

/// A [`FrameSource`] that extracts each missing `.sift` file with the
/// `sfmtool` feature tool, as [`crate::features::sift::build_sift_files`]
/// does: the same file, at the same path, skipping an image whose `.sift`
/// file is at least as new as it, and kept on disk.
///
/// The work runs on a pool of workers, each extracting one frame at a time
/// the way `build_sift_files` runs one image: the decode on the worker's own
/// thread, the extraction and the write on the rayon pool. The workers start
/// at the first frame asked for and stop at [`FrameSource::finish`] or when
/// the source is dropped.
pub struct ExtractingSource {
    pool: WorkerPool,
}

impl ExtractingSource {
    /// A source that extracts with `options`: the workspace's
    /// `feature_options` and `feature_prefix_dir`, and
    /// `concurrent_images` workers, or when `None` the count
    /// `build_sift_files` takes by default (the `SFMTOOL_SIFT_EXTRACT_WORKERS`
    /// environment variable, or the core count capped so that the images in
    /// flight fit in half of physical memory at the first image's size).
    ///
    /// # Errors
    ///
    /// [`BuildSiftFilesError::FeatureOptions`] for options that describe no
    /// valid extraction, and [`BuildSiftFilesError::Version1Settings`] for
    /// options with no `image_to_gray`, from which no `.sift` file is
    /// written.
    pub fn new(options: BuildSiftFilesOptions) -> Result<Self, BuildSiftFilesError> {
        let params = SiftParams::from_feature_options(&options.feature_options)?;
        if !crate::workspace::has_image_to_gray(&options.feature_options) {
            return Err(BuildSiftFilesError::Version1Settings);
        }
        let threads = options.concurrent_images.map(|n| n.max(1));
        let job: Arc<Job> = Arc::new(move |frame, images, progress| {
            extract_frame(frame, images, &options, &params, progress)
        });
        Ok(Self {
            pool: WorkerPool::new(job, threads.map_or(Threads::Default, Threads::Fixed)),
        })
    }
}

/// Extract the `.sift` file of each image of one frame that is not up to
/// date, returning whether any was written.
fn extract_frame(
    frame: FrameId,
    images: &[CaptureImage],
    options: &BuildSiftFilesOptions,
    params: &SiftParams,
    progress: &Progress,
) -> Result<bool, FrameSourceError> {
    let extract = |error| match error {
        BuildSiftFilesError::Cancelled => FrameSourceError::Cancelled,
        error => FrameSourceError::Extract { frame, error },
    };
    let mut wrote = false;
    for image in images {
        let expected = options.sift_path_for_image(&image.image_path);
        if image.sift_path != expected {
            return Err(FrameSourceError::SiftPath {
                frame,
                sift_path: image.sift_path.clone(),
                expected,
            });
        }
        if sift_file_up_to_date(&image.image_path, &image.sift_path).map_err(extract)? {
            continue;
        }
        if let Some(folder) = image.sift_path.parent() {
            std::fs::create_dir_all(folder).map_err(|e| {
                extract(BuildSiftFilesError::Write {
                    path: folder.to_path_buf(),
                    problem: format!("could not create the folder: {e}"),
                })
            })?;
        }
        build_sift_file(
            &image.image_path,
            &image.sift_path,
            options,
            params,
            progress,
        )
        .map_err(extract)?;
        wrote = true;
    }
    Ok(wrote)
}

impl FrameSource for ExtractingSource {
    fn concurrency(&self) -> usize {
        self.pool.concurrency()
    }

    fn prefetch(&mut self, frames: &[(FrameId, &CaptureFrame)]) {
        self.pool.prefetch(frames);
    }

    fn wait(
        &mut self,
        frames: &[(FrameId, &CaptureFrame)],
        progress: &Progress,
    ) -> Result<(), FrameSourceError> {
        self.pool.wait(frames, progress)
    }

    fn finish(&mut self) -> Vec<FrameId> {
        self.pool.finish()
    }
}

/// A [`FrameSource`] for `.sift` files that already exist: it extracts
/// nothing, and a frame whose file is missing is an error naming it.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReadOnlySource;

impl FrameSource for ReadOnlySource {
    fn concurrency(&self) -> usize {
        0
    }

    fn prefetch(&mut self, _frames: &[(FrameId, &CaptureFrame)]) {}

    fn wait(
        &mut self,
        frames: &[(FrameId, &CaptureFrame)],
        progress: &Progress,
    ) -> Result<(), FrameSourceError> {
        if progress.is_cancelled() {
            return Err(FrameSourceError::Cancelled);
        }
        for &(id, frame) in frames {
            for image in &frame.images {
                if !image.sift_path.is_file() {
                    return Err(FrameSourceError::Missing {
                        frame: id,
                        path: image.sift_path.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Vec<FrameId> {
        Vec::new()
    }
}

/// The work a [`WorkerPool`] does for one frame: make its images' `.sift`
/// files exist, returning whether it wrote any. The progress carries only the
/// pool's stop flag.
pub(super) type Job =
    dyn Fn(FrameId, &[CaptureImage], &Progress) -> Result<bool, FrameSourceError> + Send + Sync;

/// How many workers a [`WorkerPool`] runs.
#[derive(Clone, Copy, Debug)]
pub(super) enum Threads {
    Fixed(usize),
    /// [`default_concurrent_images`] at the first frame's first image.
    Default,
}

/// Workers that run a [`Job`] for each frame asked for, the frames a wait is
/// blocked on first and then the prefetched ones in their order.
///
/// What the pool holds per frame is its state, a few bytes, and for a frame
/// queued its image paths; the files the workers write are on disk, and no
/// worker holds more than the one frame it works on.
pub(super) struct WorkerPool {
    shared: Arc<Shared>,
    job: Arc<Job>,
    threads: Threads,
    /// The workers, started at the first frame asked for.
    handles: Vec<JoinHandle<()>>,
}

struct Shared {
    state: Mutex<State>,
    /// Signalled when a task is queued or the pool stops.
    work: Condvar,
    /// Signalled when a task ends.
    done: Condvar,
    /// Set to stop the workers; it is the cancel flag of their progress, so
    /// a job stops at its next check.
    stop: AtomicBool,
}

#[derive(Default)]
struct State {
    /// The frames a wait is blocked on, in its order.
    urgent: VecDeque<Task>,
    /// The frames of the last prefetch not yet started, in its order.
    ahead: VecDeque<Task>,
    status: HashMap<FrameId, Status>,
    /// Every frame a job wrote files for.
    written: Vec<FrameId>,
    /// How many tasks have ended.
    ended: u64,
}

struct Task {
    id: FrameId,
    images: Vec<CaptureImage>,
}

enum Status {
    Queued { urgent: bool },
    Running,
    Done,
    Failed(FrameSourceError),
}

fn lock(shared: &Shared) -> MutexGuard<'_, State> {
    shared.state.lock().unwrap_or_else(|p| p.into_inner())
}

impl WorkerPool {
    pub(super) fn new(job: Arc<Job>, threads: Threads) -> Self {
        Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State::default()),
                work: Condvar::new(),
                done: Condvar::new(),
                stop: AtomicBool::new(false),
            }),
            job,
            threads,
            handles: Vec::new(),
        }
    }

    pub(super) fn concurrency(&self) -> usize {
        if !self.handles.is_empty() {
            return self.handles.len();
        }
        match self.threads {
            Threads::Fixed(n) => n,
            Threads::Default => std::thread::available_parallelism().map_or(1, |n| n.get()),
        }
    }

    /// Start the workers, sized from `first` when the count is the default.
    fn start(&mut self, first: Option<&CaptureImage>) {
        if !self.handles.is_empty() {
            return;
        }
        let count = match (self.threads, first) {
            (Threads::Fixed(n), _) => n,
            (Threads::Default, Some(image)) => {
                default_concurrent_images(&image.image_path, &Progress::none())
            }
            (Threads::Default, None) => self.concurrency(),
        }
        .max(1);
        for _ in 0..count {
            let shared = Arc::clone(&self.shared);
            let job = Arc::clone(&self.job);
            self.handles
                .push(std::thread::spawn(move || work(&shared, &*job)));
        }
    }

    pub(super) fn prefetch(&mut self, frames: &[(FrameId, &CaptureFrame)]) {
        if frames.is_empty() {
            return;
        }
        let first = frames.first().and_then(|(_, f)| f.images.first());
        {
            let mut state = lock(&self.shared);
            let state = &mut *state;
            let wanted: HashSet<FrameId> = frames.iter().map(|&(id, _)| id).collect();
            let mut kept: HashMap<FrameId, Task> = HashMap::new();
            for task in state.ahead.drain(..) {
                if wanted.contains(&task.id) {
                    kept.insert(task.id, task);
                } else {
                    state.status.remove(&task.id);
                }
            }
            for &(id, frame) in frames {
                let task = match state.status.get(&id) {
                    None => Task {
                        id,
                        images: frame.images.clone(),
                    },
                    Some(Status::Queued { urgent: false }) => match kept.remove(&id) {
                        Some(task) => task,
                        // Listed twice in `frames`: already queued again.
                        None => continue,
                    },
                    Some(_) => continue,
                };
                state.status.insert(id, Status::Queued { urgent: false });
                state.ahead.push_back(task);
            }
        }
        self.start(first);
        self.shared.work.notify_all();
    }

    pub(super) fn wait(
        &mut self,
        frames: &[(FrameId, &CaptureFrame)],
        progress: &Progress,
    ) -> Result<(), FrameSourceError> {
        if progress.is_cancelled() {
            return Err(self.cancel());
        }
        if frames.is_empty() {
            return Ok(());
        }
        {
            let mut state = lock(&self.shared);
            let state = &mut *state;
            for &(id, frame) in frames {
                let task = match state.status.get(&id) {
                    None => Task {
                        id,
                        images: frame.images.clone(),
                    },
                    Some(Status::Queued { urgent: false }) => {
                        let at = state.ahead.iter().position(|t| t.id == id);
                        match at.and_then(|i| state.ahead.remove(i)) {
                            Some(task) => task,
                            None => Task {
                                id,
                                images: frame.images.clone(),
                            },
                        }
                    }
                    Some(_) => continue,
                };
                state.status.insert(id, Status::Queued { urgent: true });
                state.urgent.push_back(task);
            }
        }
        self.start(frames.first().and_then(|(_, f)| f.images.first()));
        self.shared.work.notify_all();

        let mut reported = Instant::now();
        for &(id, frame) in frames {
            loop {
                let mut state = lock(&self.shared);
                match state.status.get(&id) {
                    Some(Status::Done) => break,
                    Some(Status::Failed(error)) => return Err(error.clone()),
                    Some(_) => {}
                    // Never left without a status while a wait is blocked
                    // on it; queue it again rather than wait for nothing.
                    None => {
                        state.status.insert(id, Status::Queued { urgent: true });
                        state.urgent.push_back(Task {
                            id,
                            images: frame.images.clone(),
                        });
                        self.shared.work.notify_all();
                    }
                }
                let (state, _) = self
                    .shared
                    .done
                    .wait_timeout(state, WAIT_POLL)
                    .unwrap_or_else(|p| p.into_inner());
                let ended = state.ended;
                drop(state);
                if progress.is_cancelled() {
                    return Err(self.cancel());
                }
                if reported.elapsed() >= WAIT_POLL {
                    // An event while waiting, so a caller whose sink checks
                    // for a signal, as the binding's does, sees it here.
                    progress.value("source_frames_ended", ended as f64);
                    reported = Instant::now();
                }
            }
        }
        Ok(())
    }

    /// Stop every worker and drop the queued work, keeping what was written.
    fn cancel(&mut self) -> FrameSourceError {
        self.stop_workers();
        FrameSourceError::Cancelled
    }

    fn stop_workers(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.work.notify_all();
        for handle in self.handles.drain(..) {
            // A worker catches a job's panic, so joining it does not fail.
            let _ = handle.join();
        }
        let mut state = lock(&self.shared);
        state.urgent.clear();
        state.ahead.clear();
        state
            .status
            .retain(|_, s| matches!(s, Status::Done | Status::Failed(_)));
        drop(state);
        self.shared.stop.store(false, Ordering::Relaxed);
    }

    pub(super) fn finish(&mut self) -> Vec<FrameId> {
        self.stop_workers();
        let mut state = lock(&self.shared);
        let mut written = std::mem::take(&mut state.written);
        *state = State::default();
        written.sort();
        written
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.stop_workers();
    }
}

/// One worker: run the next task until the pool stops.
fn work(shared: &Shared, job: &Job) {
    let progress = Progress::none().cancelled_by(&shared.stop);
    loop {
        let task = {
            let mut state = lock(shared);
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(task) = state.urgent.pop_front().or_else(|| state.ahead.pop_front()) {
                    state.status.insert(task.id, Status::Running);
                    break task;
                }
                state = shared.work.wait(state).unwrap_or_else(|p| p.into_inner());
            }
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            job(task.id, &task.images, &progress)
        }))
        .unwrap_or_else(|panic| {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            Err(FrameSourceError::Panicked {
                frame: task.id,
                message,
            })
        });
        let mut state = lock(shared);
        match result {
            Ok(wrote) => {
                if wrote {
                    state.written.push(task.id);
                }
                state.status.insert(task.id, Status::Done);
            }
            // Stopped part way: nothing was written, and the frame is as if
            // never asked for.
            Err(FrameSourceError::Cancelled) => {
                state.status.remove(&task.id);
            }
            Err(error) => {
                state.status.insert(task.id, Status::Failed(error));
            }
        }
        state.ended += 1;
        drop(state);
        shared.done.notify_all();
    }
}

#[cfg(test)]
mod tests;
