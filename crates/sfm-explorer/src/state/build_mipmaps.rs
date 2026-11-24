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

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard};

use sfmtool_core::camera::{PeekedLayout, PhotographCache, COARSEST_PUBLISHED};
use sfmtool_core::progress::{Event, Progress};

use super::{AppState, PhotographRequests};
use crate::background::{Operation, QuietJob, QuietRank, QuietSource};

#[cfg(test)]
pub(crate) mod tests;

/// The photographs *Build mipmaps* is to make, shared by the panels that
/// queue them and the task's job that takes them.
#[derive(Debug, Default)]
pub(crate) struct MipmapList {
    /// Waiting to be taken, the next first.
    waiting: VecDeque<PathBuf>,
    /// Every path queued and not yet answered: waiting, or in the chunk a
    /// run has taken. A panel queues a path once while it is here.
    queued: HashSet<PathBuf>,
    /// The paths answered since the list was last empty, what the quiet line
    /// counts as done.
    done: usize,
}

impl MipmapList {
    /// Add `path` to the front of the list, unless it is queued already.
    pub(crate) fn queue(&mut self, path: &Path) {
        if self.queued.insert(path.to_path_buf()) {
            self.waiting.push_front(path.to_path_buf());
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
        self.waiting.drain(..n).collect()
    }

    /// `path` was answered: it leaves the list's memory.
    fn answered(&mut self, path: &Path) {
        if self.queued.remove(path) {
            self.done += 1;
        }
        if self.queued.is_empty() {
            self.done = 0;
        }
    }

    /// Put `paths`, taken by a run that stopped before answering them, back
    /// at the front, in their order.
    fn put_back(&mut self, paths: Vec<PathBuf>) {
        for path in paths.into_iter().rev() {
            if self.queued.contains(&path) {
                self.waiting.push_front(path);
            }
        }
    }

    /// The task's label, which its quiet line shows: how many photographs
    /// are answered of how many queued since the list was last empty.
    fn progress_label(&self) -> String {
        let total = self.done + self.queued.len();
        format!("Building mipmaps: {} of {total}", self.done)
    }
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
                    requests.failed_to_build(path);
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
