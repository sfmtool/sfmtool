// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! *Build mipmaps*, the quiet task that makes the mipmaps the panels wait on.
//!
//! A panel never makes a mipmap. One that needs an out-of-core photograph
//! whose mipmap is not current ([`PeekedLayout::Building`]) adds it to the
//! front of the shared list ([`MipmapList`]) through its
//! [`super::PhotographRequests`], once per path, and draws its waiting
//! state. This task drains the list: each run takes one chunk from the front,
//! as many photographs as the photograph cache's build budget lets it make
//! at once ([`PhotographCache::builds_that_fit`]), calls
//! [`PhotographCache::get_many`] on them on its job thread, which makes their
//! mipmaps, and ends; [`AppState::drive_build_mipmaps`] queues it again while
//! the list is not empty. Ending after each chunk is what lets the quiet
//! tasks ahead of it in `QuietRank` start between chunks.
//!
//! An open fills the list too: the reconstruction's out-of-core photographs
//! whose mipmaps are not current, found on the open's worker
//! ([`not_current`]) and put behind every photograph a panel has queued
//! ([`AppState::queue_opened_mipmaps`]). A panel that shows one of them
//! moves it to the front. Closing a node takes every photograph no node in
//! the scene has off the list ([`AppState::forget_closed_mipmaps`]).
//!
//! It runs in the fast queue and gives way to every task someone asks for
//! there; a run that is cancelled puts the photographs of its chunk it had
//! not finished back at the front of the list, so the run queued again
//! behind the displacing task makes them first and no photograph a panel
//! queued is lost. It pushes no version and does not wait for a node: it
//! writes mip level files and fills the photograph cache, so it runs while a
//! slow task holds the node the panels show.
//!
//! A photograph the task answers, made or found current, leaves the list's
//! memory, so one that goes back to `Building` (a file of its mipmap found
//! changed) is queued again. One it could not make is kept as a failure of
//! the panels' reads, with its file's stamp, as a decode that failed is, so
//! a cache that remembers no failure (a budget of 0) is not asked to make it
//! again on every frame. See `specs/gui/background-tasks.md` § "Quiet tasks"
//! and `specs/drafts/out-of-core-photograph-pyramid.md` § "Who makes the
//! mipmaps, and when".
//!
//! [`PeekedLayout::Building`]: sfmtool_core::camera::PeekedLayout::Building

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard};

use rayon::prelude::*;
use sfmtool_core::camera::{PeekedLayout, PhotographCache, COARSEST_PUBLISHED};
use sfmtool_core::progress::{Cancelled, Event, Progress};
use sfmtool_core::workspace::{find_workspace_for_path, read_mipmap_options};
use sfmtool_core::SfmrReconstruction;

use super::{AppState, PhotographRequests};
use crate::background::{Operation, QuietJob, QuietRank, QuietSource};

#[cfg(test)]
pub(crate) mod tests;

/// The photographs *Build mipmaps* is to make, shared by the panels that
/// queue them, the open that fills it and the task's job that takes them.
///
/// The paths a panel queued come first, the last queued first, then those
/// an open queued, the newest open's first and each open's in image order.
#[derive(Debug, Default)]
pub(crate) struct MipmapList {
    /// Waiting to be taken, the next first.
    waiting: VecDeque<PathBuf>,
    /// How many of `waiting`, from the front, a panel queued or a run put
    /// back; an open queued the rest.
    asked: usize,
    /// The waiting paths an open queued that no panel has queued since.
    opened: HashSet<PathBuf>,
    /// Every path queued and not yet answered: waiting, or in the chunk a
    /// run has taken. A panel queues a path once while it is here, so no
    /// path is in `waiting` twice.
    queued: HashSet<PathBuf>,
    /// The paths of `queued` in the chunk a run has taken, until the run
    /// answers them or puts them back.
    taken: HashSet<PathBuf>,
    /// The paths of `taken` that [`Self::keep_only`] refused and nothing has
    /// queued since: the run is left to finish them, but they are not put
    /// back or counted.
    dropped: HashSet<PathBuf>,
    /// The paths answered since the list was last empty, what the quiet line
    /// counts as done.
    done: usize,
    /// Every path a run has answered, made or not, since
    /// [`Self::take_answered`] last took them: what fills in the display
    /// thumbnail rows awaiting a mipmap ([`crate::state::thumbnail_fills`]).
    answered_paths: Vec<PathBuf>,
}

impl MipmapList {
    /// Add `path` to the front of the list, unless a panel has queued it
    /// already or a run has taken it. One an open queued moves to the front.
    pub(crate) fn queue(&mut self, path: &Path) {
        if self.queued.insert(path.to_path_buf()) {
            self.waiting.push_front(path.to_path_buf());
            self.asked += 1;
        } else if self.dropped.remove(path) {
            // Taken by a running chunk, which makes it: asked for again.
        } else if self.opened.remove(path) {
            if let Some(at) = self.waiting.iter().position(|p| p == path) {
                self.waiting.remove(at);
            }
            self.waiting.push_front(path.to_path_buf());
            self.asked += 1;
        }
    }

    /// Add `paths`, the photographs of a reconstruction that has just opened
    /// whose mipmaps are not current, in their order, behind every path a
    /// panel queued and ahead of those an earlier open queued. A path
    /// already queued stays where it is.
    pub(crate) fn queue_opened(&mut self, paths: Vec<PathBuf>) {
        let earlier = self.waiting.split_off(self.asked);
        for path in paths {
            if self.queued.insert(path.clone()) {
                self.opened.insert(path.clone());
                self.waiting.push_back(path);
            } else {
                self.dropped.remove(&path);
            }
        }
        self.waiting.extend(earlier);
    }

    /// Forget every path, waiting or taken, that `keep` refuses: no run
    /// takes it, and a cancelled run that had taken it does not put it back.
    /// A taken one stays queued, marked dropped, until its run ends, so
    /// queueing it again meanwhile only clears the mark and adds no copy.
    pub(crate) fn keep_only(&mut self, keep: impl Fn(&Path) -> bool) {
        let mut asked = 0;
        let mut waiting = VecDeque::with_capacity(self.waiting.len());
        for (i, path) in self.waiting.drain(..).enumerate() {
            if keep(&path) {
                if i < self.asked {
                    asked += 1;
                }
                waiting.push_back(path);
            }
        }
        self.waiting = waiting;
        self.asked = asked;
        self.opened.retain(|path| keep(path));
        for path in &self.taken {
            if !keep(path) {
                self.dropped.insert(path.clone());
            }
        }
        let taken = &self.taken;
        self.queued
            .retain(|path| keep(path) || taken.contains(path));
        if self.queued.len() == self.dropped.len() {
            self.done = 0;
        }
    }

    /// The paths waiting, the next first.
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> Vec<PathBuf> {
        self.waiting.iter().cloned().collect()
    }

    /// Whether no path is waiting.
    fn is_empty(&self) -> bool {
        self.waiting.is_empty()
    }

    /// Take up to `n` paths from the front, at least one when any waits.
    fn take(&mut self, n: usize) -> Vec<PathBuf> {
        let n = n.max(1).min(self.waiting.len());
        self.asked = self.asked.saturating_sub(n);
        let taken: Vec<PathBuf> = self.waiting.drain(..n).collect();
        for path in &taken {
            self.opened.remove(path);
            self.taken.insert(path.clone());
        }
        taken
    }

    /// `path` was answered: it leaves the list's memory, counted as done
    /// unless it was dropped.
    fn answered(&mut self, path: &Path) {
        self.answered_paths.push(path.to_path_buf());
        self.taken.remove(path);
        let dropped = self.dropped.remove(path);
        if self.queued.remove(path) && !dropped {
            self.done += 1;
        }
        if self.queued.len() == self.dropped.len() {
            self.done = 0;
        }
    }

    /// Put `paths`, taken by a run that stopped before answering them, back
    /// at the front, in their order, but for those dropped since, which
    /// leave the list.
    fn put_back(&mut self, paths: Vec<PathBuf>) {
        for path in paths.into_iter().rev() {
            self.taken.remove(&path);
            if self.dropped.remove(&path) {
                self.queued.remove(&path);
            } else if self.queued.contains(&path) {
                self.waiting.push_front(path);
                self.asked += 1;
            }
        }
        if self.queued.len() == self.dropped.len() {
            self.done = 0;
        }
    }

    /// Take every path a run has answered since the last call, in the order
    /// answered.
    pub(crate) fn take_answered(&mut self) -> Vec<PathBuf> {
        std::mem::take(&mut self.answered_paths)
    }

    /// The task's label, which its quiet line shows: how many photographs
    /// are answered of how many queued since the list was last empty.
    fn progress_label(&self) -> String {
        let total = self.done + self.queued.len() - self.dropped.len();
        format!("Building mipmaps: {} of {total}", self.done)
    }
}

/// The photographs of `recon`, in image order, whose workspace file has
/// `mipmap_options`: the only ones that can be out of core. Reads no
/// photograph: the stats that find each image folder's workspace, and each
/// workspace file once. A workspace file that cannot be read or holds an
/// invalid object counts as having none here; the panels report it.
pub(crate) fn in_mipmap_workspaces(recon: &SfmrReconstruction) -> Vec<PathBuf> {
    let mut roots: HashMap<PathBuf, Option<PathBuf>> = HashMap::new();
    let mut has_options: HashMap<PathBuf, bool> = HashMap::new();
    recon
        .image_set
        .images
        .iter()
        .map(|image| recon.workspace_dir.join(&image.name))
        .filter(|path| {
            let folder = path.parent().unwrap_or(Path::new("")).to_path_buf();
            let root = roots
                .entry(folder)
                .or_insert_with(|| find_workspace_for_path(path));
            root.as_ref().is_some_and(|root| {
                *has_options
                    .entry(root.clone())
                    .or_insert_with(|| matches!(read_mipmap_options(root), Ok(Some(_))))
            })
        })
        .collect()
}

/// Of `paths`, the out-of-core photographs whose mipmaps are not current, in
/// their order, as `cache` finds them ([`PhotographCache::request_layout`]: a
/// header read, the region test and the coarsest mip level file's record
/// each), on the rayon pool. `cache` keeps each layout it reads, so a panel
/// that shows one of the photographs reads none of them again. Counts the
/// photographs read through `progress` and stops when it is cancelled.
pub(crate) fn not_current(
    paths: &[PathBuf],
    cache: &PhotographCache,
    progress: &Progress<'_>,
) -> Result<Vec<PathBuf>, Cancelled> {
    let read = AtomicUsize::new(0);
    let total = Some(paths.len() as u64);
    let building: Vec<bool> = paths
        .par_iter()
        .map(|path| {
            if progress.is_cancelled() {
                return false;
            }
            cache.request_layout(path);
            let done = read.fetch_add(1, Ordering::Relaxed) + 1;
            progress.count(done as u64, total, "photographs");
            matches!(cache.layout(path), PeekedLayout::Building { .. })
        })
        .collect();
    progress.check_cancel()?;
    Ok(paths
        .iter()
        .zip(building)
        .filter(|(_, building)| *building)
        .map(|(path, _)| path.clone())
        .collect())
}

/// Lock `list`, ignoring a poison: nothing panics while it is held.
pub(super) fn lock(list: &Mutex<MipmapList>) -> MutexGuard<'_, MipmapList> {
    list.lock().unwrap_or_else(|e| e.into_inner())
}

/// One run of *Build mipmaps*: take a chunk from the front of `requests`'
/// list and make its photographs' mipmaps with `cache.get_many`, on this
/// thread and the rayon workers `get_many` uses, polling `progress`.
///
/// Each photograph answered leaves the list's memory; one that could not be
/// made is kept as a failure of the panels' reads. A run that is cancelled
/// puts the photographs of its chunk whose layout is not yet known back at
/// the front, in their order.
///
/// Calls `wake` each time a build of the chunk has published its
/// photograph's coarsest level ([`COARSEST_PUBLISHED`]), so the panels
/// waiting on it draw it before the build ends: once a build, so at most
/// one repaint a photograph besides the one the run's end gives.
pub(crate) fn run_chunk(
    cache: &PhotographCache,
    requests: &PhotographRequests,
    progress: &Progress<'_>,
    wake: Option<&(dyn Fn() + Sync)>,
) {
    let list = &requests.mipmaps;
    let chunk = {
        let list = lock(list);
        let Some(front) = list.waiting.front().cloned() else {
            return;
        };
        drop(list);
        // The header read runs with no lock held.
        let n = cache.builds_that_fit(&front);
        lock(&requests.mipmaps).take(n)
    };
    if chunk.is_empty() {
        return;
    }
    let paths: Vec<&Path> = chunk.iter().map(PathBuf::as_path).collect();
    #[cfg(test)]
    requests.chunk_started(&chunk);
    let published = |event: Event<'_>| {
        if let (
            Event::Value {
                name: COARSEST_PUBLISHED,
                ..
            },
            Some(wake),
        ) = (&event, wake)
        {
            wake();
        }
        progress.forward(event);
    };
    match cache.get_many(&paths, &progress.tapped(&published)) {
        Ok((answers, _)) => {
            for (path, answer) in chunk.iter().zip(answers) {
                if answer.is_none() {
                    requests.failed_to_build(cache, path);
                }
                lock(list).answered(path);
            }
        }
        Err(_) => {
            let (done, rest): (Vec<PathBuf>, Vec<PathBuf>) = chunk
                .into_iter()
                .partition(|path| matches!(cache.layout(path), PeekedLayout::Known(_)));
            let mut list = lock(list);
            for path in &done {
                list.answered(path);
            }
            list.put_back(rest);
        }
    }
}

/// What the viewer keeps of *Build mipmaps*: the run queued last, until it
/// has ended or was dropped, as the end of a channel its job holds the other
/// end of, which is disconnected when the job ends or the queue drops it
/// unrun.
#[derive(Default)]
pub(crate) struct BuildMipmaps {
    queued: Option<Receiver<()>>,
}

impl AppState {
    /// Queue a run of *Build mipmaps* when the panels' list has photographs
    /// waiting and no run is queued or running, and forget the run queued
    /// last once it has ended or was dropped.
    ///
    /// Called once per frame. Returns whether anything changed that the next
    /// frame has to draw: a run ended, so the photographs it made are there
    /// to draw, or a run was queued.
    pub(crate) fn drive_build_mipmaps(&mut self) -> bool {
        let mut changed = false;
        if let Some(ended) = &self.build_mipmaps.queued {
            match ended.try_recv() {
                Ok(()) | Err(TryRecvError::Disconnected) => {
                    self.build_mipmaps.queued = None;
                    changed = true;
                }
                Err(TryRecvError::Empty) => return false,
            }
        }
        let label = {
            let list = lock(&self.photograph_requests.mipmaps);
            if list.is_empty() {
                return changed;
            }
            list.progress_label()
        };
        let cache = Arc::clone(&self.photographs);
        let requests = self.photograph_requests.clone();
        let wake = self.wake.clone();
        let (sender, ended) = std::sync::mpsc::channel::<()>();
        let job = QuietJob::new(
            move |progress: &Progress<'_>| {
                let wake = wake.as_deref().map(|wake| wake as &(dyn Fn() + Sync));
                run_chunk(&cache, &requests, progress, wake);
                drop(sender);
            },
            // Nothing was taken from the list: the next frame queues again.
            || {},
        );
        let started = self.start_quiet_task(
            Operation::BUILD_MIPMAPS,
            QuietSource::BuildMipmaps,
            QuietRank::BuildMipmaps,
            None,
            label,
            job,
        );
        match started {
            Ok(_) => {
                self.build_mipmaps.queued = Some(ended);
                true
            }
            // A worker that could not be spawned: the next frame tries again.
            Err(_) => changed,
        }
    }

    /// Put `paths`, the photographs of a reconstruction that has just opened
    /// whose mipmaps are not current, on *Build mipmaps*' list, in image
    /// order, behind every photograph a panel has queued: the next frame's
    /// [`Self::drive_build_mipmaps`] queues the task.
    ///
    /// No image of the node is selected yet, and the Image Browser strip
    /// shows a node that has just opened from its first image, so image
    /// order puts the strip's images first. An image selected later is put
    /// at the front by the panel that shows it.
    pub(crate) fn queue_opened_mipmaps(&self, paths: Vec<PathBuf>) {
        if !paths.is_empty() {
            lock(&self.photograph_requests.mipmaps).queue_opened(paths);
        }
    }

    /// Take every photograph that no node in the scene has off *Build
    /// mipmaps*' list, after a node is closed: no panel waits on it any
    /// more, so no later run makes its mipmap.
    pub(crate) fn forget_closed_mipmaps(&self) {
        let mut list = lock(&self.photograph_requests.mipmaps);
        if list.queued.is_empty() {
            return;
        }
        let shown: HashSet<PathBuf> = self
            .scene
            .iter()
            .flat_map(|node| {
                let recon = node.recon();
                recon
                    .image_set
                    .images
                    .iter()
                    .map(|image| recon.workspace_dir.join(&image.name))
            })
            .collect();
        list.keep_only(|path| shown.contains(path));
    }

    /// Whether a run of *Build mipmaps* is queued or running.
    #[cfg(test)]
    pub(crate) fn build_mipmaps_queued(&self) -> bool {
        self.build_mipmaps.queued.is_some()
    }
}

/// Whether a run of *Build mipmaps* whose `Progress` is already cancelled
/// stops rather than make its chunk: it makes no mipmap, writes no file and
/// puts its chunk back on the list.
#[cfg(test)]
pub(crate) fn job_stops_when_cancelled() -> bool {
    let (dir, paths) = tests::out_of_core_workspace(1);
    let cache = tests::cache_for(1 << 30, 1);
    let (requests, _) = PhotographRequests::queued();
    lock(&requests.mipmaps).queue(&paths[0]);
    let cancel = std::sync::atomic::AtomicBool::new(true);
    run_chunk(
        &cache,
        &requests,
        &Progress::none().cancelled_by(&cancel),
        None,
    );
    cache.stats().mipmaps_made == 0
        && tests::mipmap_file_count(dir.path()) == 0
        && requests.mipmaps_waiting() == paths
}
